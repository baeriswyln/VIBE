# ─────────────────────────────────────────────────────────────────────
# Windows fix — MUST run before importing anything that drags in pywin32
# (pyttsx3 does, via the SAPI5 driver). pywin32 sets the COM threading
# model to STA, which breaks bleak's WinRT scanner ("Thread is configured
# for Windows GUI but callbacks are not working"). Forcing coinit_flags=0
# keeps the process in MTA so bleak is happy.
# See https://bleak.readthedocs.io/en/latest/troubleshooting.html
# ─────────────────────────────────────────────────────────────────────
import sys as _sys
if _sys.platform == "win32":
    _sys.coinit_flags = 0  # 0 == MTA

"""
myo_pinch_ml.py — ML-based Myo pinch calibration and runtime detection.

Usage:
    python myo_pinch_ml.py calibrate                 # guided calibration, up to 30 events
    python myo_pinch_ml.py calibrate --n-trials 20   # change number of pinches
    python myo_pinch_ml.py detect                    # stream pinch markers to LSL
    python myo_pinch_ml.py detect --dry-run          # detect without LSL

Requires: dl-myo, pylsl, numpy, scikit-learn, sounddevice, pyttsx3
    pip install dl-myo pylsl numpy scikit-learn sounddevice pyttsx3

Design notes
------------
* No vibration is used as a cue. We use audio beeps (low pitch = prepare,
  high pitch = pinch NOW) and offline TTS for spoken instructions.
* Each calibration trial captures a fixed-length window of EMG starting
  just before the high beep — discrete pinches, cleaner labels for ML.
* Rest samples are gathered from the inter-trial gaps.
* We train Shrinkage-LDA, L2 Logistic Regression and a small Random Forest,
  cross-validate them on the captured trials, report all three and save
  the best one.
* Runtime uses the same windowing + feature extraction, with consecutive-
  sample debouncing on top of the classifier output.
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
import threading
import time
from collections import deque

import numpy as np

from myo import MyoClient, EMGMode, SleepMode, ClassifierMode, IMUMode
from myo.types import *

CONFIG_FILE = "myo_pinch_ml_config.json"
MODEL_FILE  = "myo_pinch_ml_model.pkl"

# ─────────────────────────────────────────────────────────────────────
# Acquisition / window config
# ─────────────────────────────────────────────────────────────────────
EMG_RATE_HZ        = 50      # filtered EMG comes in around 50 Hz via aggregated stream
WINDOW_MS          = 300     # length of the analysis window
WINDOW_SAMPLES     = max(8, int(EMG_RATE_HZ * WINDOW_MS / 1000))
PRE_CUE_MS         = 80      # capture this much before the "pinch NOW" cue
PRE_CUE_SAMPLES    = int(EMG_RATE_HZ * PRE_CUE_MS / 1000)

# Calibration timing (seconds)
PREP_DURATION       = 1.5    # low beep → user gets ready
HOLD_DURATION       = 0.6    # how long after high beep we record into the window
INTER_TRIAL_REST    = 2.0    # gap between trials, source of "rest" samples
REST_WINDOW_OFFSET  = 0.7    # take rest window this long into the gap (avoid release transient)

# ─────────────────────────────────────────────────────────────────────
# Runtime detection knobs
# ─────────────────────────────────────────────────────────────────────
STEP_MS         = 60         # how often we re-classify (sliding window stride)
STEP_SAMPLES    = max(1, int(EMG_RATE_HZ * STEP_MS / 1000))
ON_CONSEC       = 2          # consecutive "pinch" predictions to fire pinch_start
OFF_CONSEC      = 3          # consecutive "rest" predictions to fire pinch_end
PROB_THRESHOLD  = 0.9       # min P(pinch) to count a sample as positive
DEBOUNCE_S      = 0.10
MAX_PINCH_S     = 8.0
HEARTBEAT_S     = 5.0


# ─────────────────────────────────────────────────────────────────────
# Audio cues — generated sine tones via sounddevice, TTS via pyttsx3.
# Both are optional and degrade gracefully if missing.
# ─────────────────────────────────────────────────────────────────────
class Cue:
    """Audio cues: low/high beeps + spoken instructions, no Myo vibration."""

    SR = 44100

    def __init__(self):
        self._sd = None
        self._tts = None
        try:
            import sounddevice as sd
            self._sd = sd
        except Exception as e:
            print(f"[cue] sounddevice unavailable ({e}); beeps will be silent.")

        try:
            import pyttsx3
            self._tts = pyttsx3.init()
            self._tts.setProperty("rate", 175)
        except Exception as e:
            print(f"[cue] pyttsx3 unavailable ({e}); will print instead of speak.")

        # Pre-render tones so beeps fire with low jitter.
        self._low  = self._tone(freq=440.0,  ms=180)   # A4 — "prepare"
        self._high = self._tone(freq=1175.0, ms=120)   # D6 — "pinch NOW"
        self._done = self._tone(freq=880.0,  ms=250)   # A5 — "done"

    def _tone(self, freq, ms, volume=0.35):
        t = np.linspace(0, ms / 1000.0, int(self.SR * ms / 1000.0), endpoint=False)
        wave = np.sin(2 * np.pi * freq * t).astype(np.float32) * volume
        # short fade in/out to avoid clicks
        fade = int(0.005 * self.SR)
        if len(wave) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            wave[:fade]  *= ramp
            wave[-fade:] *= ramp[::-1]
        return wave

    def _play(self, wave):
        if self._sd is None:
            return
        try:
            self._sd.play(wave, self.SR, blocking=False)
        except Exception as e:
            print(f"[cue] play failed: {e}")

    # public API ──────────────────────────────────────────────────
    def prep(self):
        self._play(self._low)
        print("ready")

    def go(self):
        self._play(self._high)
        print("go")

    def done(self):
        self._play(self._done)
        print("done")

    def say(self, text):
        print(f"[say] {text}")
        if self._tts is None:
            return
        # pyttsx3 blocks; do it in a thread so we can keep the audio loop alive.
        def _run():
            try:
                self._tts.say(text)
                self._tts.runAndWait()
            except Exception as e:
                print(f"[cue] tts failed: {e}")
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=8.0)  # don't hang forever if a driver misbehaves


# ─────────────────────────────────────────────────────────────────────
# Myo client. Identical wiring to the reference; we just route samples
# into a ring buffer instead of doing inline logic.
# ─────────────────────────────────────────────────────────────────────
class MyMyo(MyoClient):
    buffer = None  # deque of (timestamp_monotonic, emg_tuple)
    on_sample = None

    def attach_buffer(self, buf, on_sample=None):
        self.buffer = buf
        self.on_sample = on_sample

    async def on_classifier_event(self, ce: ClassifierEvent):
        pass

    async def on_aggregated_data(self, ad: AggregatedData):
        emg = tuple(ad.fvd.fv) + (ad.fvd.mask,)
        if self.buffer is not None:
            self.buffer.append((time.monotonic(), emg))
        if self.on_sample is not None:
            self.on_sample(emg)

    async def on_emg_data(self, emg: EMGData):           pass
    async def on_fv_data(self, fvd: FVData):             pass
    async def on_imu_data(self, imu: IMUData):           pass
    async def on_motion_event(self, me: MotionEvent):    pass
    async def on_emg_data_aggregated(self, emg: EMGData): pass


async def connect_myo():
    print("Connecting to Myo…")
    m = await MyMyo.with_device(aggregate_all=True)
    await m.setup()
    await m.set_sleep_mode(SleepMode.NEVER_SLEEP)
    print("Connected.")
    return m


# ─────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────
def extract_features(window):
    """
    Window: (T, C) float array of EMG samples (the 8 filtered channels;
    the 9th 'mask' element is dropped here).

    Returns a 1D feature vector with 5 standard EMG features per channel:
      MAV  — mean absolute value
      RMS  — root mean square
      WL   — waveform length (sum of |x[t]-x[t-1]|)
      VAR  — variance
      ZC   — zero crossings around the channel mean (robust to DC offset)
    """
    w = np.asarray(window, dtype=np.float32)
    if w.ndim == 1:
        w = w[None, :]
    # drop mask channel if present (last column was the FVData mask)
    if w.shape[1] == 9:
        w = w[:, :8]

    mav = np.mean(np.abs(w), axis=0)
    rms = np.sqrt(np.mean(w * w, axis=0))
    diff = np.diff(w, axis=0)
    wl  = np.sum(np.abs(diff), axis=0) if diff.size else np.zeros(w.shape[1], dtype=np.float32)
    var = np.var(w, axis=0)
    centered = w - w.mean(axis=0, keepdims=True)
    zc = np.sum(np.diff(np.sign(centered), axis=0) != 0, axis=0).astype(np.float32) \
        if w.shape[0] > 1 else np.zeros(w.shape[1], dtype=np.float32)

    return np.concatenate([mav, rms, wl, var, zc]).astype(np.float32)


FEATURE_NAMES = (
    [f"mav_c{i}" for i in range(8)] +
    [f"rms_c{i}" for i in range(8)] +
    [f"wl_c{i}"  for i in range(8)] +
    [f"var_c{i}" for i in range(8)] +
    [f"zc_c{i}"  for i in range(8)]
)


# ─────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────
def _extract_window_around(buffer_snapshot, target_time, length_samples,
                           pre_samples=0):
    """
    From a list of (timestamp, emg) samples, take a window centered such
    that `pre_samples` come before `target_time` and the rest comes after.
    Returns None if not enough samples are available.
    """
    if not buffer_snapshot:
        return None
    times = np.array([t for t, _ in buffer_snapshot])
    # index of the first sample at or after target_time
    idx = int(np.searchsorted(times, target_time))
    start = idx - pre_samples
    end = start + length_samples
    if start < 0 or end > len(buffer_snapshot):
        return None
    return np.array([emg for _, emg in buffer_snapshot[start:end]],
                    dtype=np.float32)


async def run_calibration(m, n_trials, cue):
    buf = deque(maxlen=EMG_RATE_HZ * 60 * 3)  # 3 minutes is plenty
    m.attach_buffer(buf)

    await m.set_mode(
        classifier_mode=ClassifierMode.ENABLED,
        emg_mode=EMGMode.SEND_FILT,
        imu_mode=IMUMode.SEND_ALL,
    )
    await m.start()

    cue.say("Calibration starting. Keep your arm relaxed between cues. "
            "When you hear the high beep, pinch your thumb and index finger "
            "briefly, then relax.")
    await asyncio.sleep(1.0)

    pinch_windows = []
    rest_windows  = []

    print(f"\nWill record up to {n_trials} pinches. "
          f"Press Ctrl-C any time after the first ~10 to stop early.\n")

    try:
        for trial in range(1, n_trials + 1):
            print(f"--- trial {trial}/{n_trials} ---")

            # 1) prep beep → countdown silence → go beep
            cue.prep()
            await asyncio.sleep(PREP_DURATION/2)
            cue.prep()
            await asyncio.sleep(PREP_DURATION/2)

            go_time = time.monotonic()
            cue.go()
            # let the pinch develop into the window
            await asyncio.sleep(HOLD_DURATION + 0.05)

            # 2) cut a window starting slightly before the cue
            snapshot = list(buf)
            target = go_time  # window aligned so a few samples are pre-cue
            window = _extract_window_around(
                snapshot,
                target_time=target,
                length_samples=WINDOW_SAMPLES,
                pre_samples=PRE_CUE_SAMPLES,
            )
            if window is None:
                print("    ⚠ not enough buffered samples — skipping trial")
            else:
                pinch_windows.append(window)
                print(f"    captured pinch window ({window.shape})")

            # 3) inter-trial rest gap, also grab a rest window from it
            await asyncio.sleep(REST_WINDOW_OFFSET)
            rest_target = time.monotonic()
            await asyncio.sleep(HOLD_DURATION + 0.05)
            snapshot = list(buf)

            rest_win = _extract_window_around(
                snapshot,
                target_time=rest_target,
                length_samples=WINDOW_SAMPLES,
                pre_samples=WINDOW_SAMPLES // 2,
            )
            if rest_win is not None:
                rest_windows.append(rest_win)

            await asyncio.sleep(max(0.0, INTER_TRIAL_REST - REST_WINDOW_OFFSET))

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nCalibration stopped early by user.")

    await m.stop()

    if len(pinch_windows) < 5:
        raise RuntimeError(
            f"Only captured {len(pinch_windows)} pinch trials — need at least ~5. "
            "Retry calibration."
        )

    cue.done()
    cue.say(f"Captured {len(pinch_windows)} pinches. Training models now.")

    return pinch_windows, rest_windows


# ─────────────────────────────────────────────────────────────────────
# Model training: try a few classifiers, pick the best by CV.
# ─────────────────────────────────────────────────────────────────────
def train_and_select(pinch_windows, rest_windows):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X_pinch = np.stack([extract_features(w) for w in pinch_windows])
    X_rest  = np.stack([extract_features(w) for w in rest_windows])

    # If too few rest windows (rare), synthesize a few extra by jittering
    # the calibration rest. This still respects label separation.
    if len(X_rest) < len(X_pinch):
        deficit = len(X_pinch) - len(X_rest)
        rng = np.random.default_rng(0)
        extras_idx = rng.integers(0, len(X_rest), size=deficit)
        X_rest = np.vstack([X_rest, X_rest[extras_idx]])

    X = np.vstack([X_rest, X_pinch]).astype(np.float32)
    y = np.array([0] * len(X_rest) + [1] * len(X_pinch))

    print(f"\nTraining set: {X.shape[0]} windows, {X.shape[1]} features "
          f"({(y==1).sum()} pinch, {(y==0).sum()} rest)")

    candidates = {
        "shrinkage_lda": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "logreg_l2": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=0, n_jobs=1,
        ),
    }

    # Use stratified K-fold; clamp k to what's feasible with small N.
    k = min(5, len(X_pinch), len(X_rest))
    if k < 2:
        raise RuntimeError("Not enough trials per class to cross-validate.")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

    results = {}
    for name, model in candidates.items():
        try:
            acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            f1  = cross_val_score(model, X, y, cv=cv, scoring="f1")
            results[name] = {
                "acc_mean": float(acc.mean()),
                "acc_std":  float(acc.std()),
                "f1_mean":  float(f1.mean()),
                "f1_std":   float(f1.std()),
            }
            print(f"  {name:>14s}  acc = {acc.mean():.3f} ± {acc.std():.3f}   "
                  f"f1 = {f1.mean():.3f} ± {f1.std():.3f}")
        except Exception as e:
            print(f"  {name:>14s}  failed: {e}")
            results[name] = {"error": str(e)}

    # Pick best by F1, tie-break by accuracy.
    scored = [(n, r) for n, r in results.items() if "f1_mean" in r]
    if not scored:
        raise RuntimeError("All candidate models failed during CV.")
    scored.sort(key=lambda kv: (kv[1]["f1_mean"], kv[1]["acc_mean"]), reverse=True)
    best_name = scored[0][0]
    print(f"\n→ best model: {best_name}")

    best_model = candidates[best_name]
    best_model.fit(X, y)

    return best_name, best_model, results, X, y


async def cmd_calibrate(args):
    # Connect to the Myo FIRST. We deliberately build the audio Cue object
    # (which imports pyttsx3 → SAPI5 → pywin32) only after the BLE scan
    # succeeds, so even on systems where the coinit_flags=0 trick at the
    # top of this file doesn't fully isolate things, bleak's WinRT init
    # has already happened in MTA mode by the time SAPI5 loads.
    m = await connect_myo()
    cue = Cue()
    try:
        pinch_w, rest_w = await run_calibration(m, args.n_trials, cue)
    finally:
        await m.disconnect()


    best_name, best_model, results, X, y = train_and_select(pinch_w, rest_w)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": best_model, "name": best_name}, f)

    cfg = {
        "version":           2,
        "best_model":        best_name,
        "cv_results":        results,
        "n_pinch_trials":    len(pinch_w),
        "n_rest_trials":     len(rest_w),
        "window_samples":    WINDOW_SAMPLES,
        "window_ms":         WINDOW_MS,
        "emg_rate_hz":       EMG_RATE_HZ,
        "feature_names":     FEATURE_NAMES,
        "calibrated_at":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nWrote {MODEL_FILE} and {CONFIG_FILE}")
    cue.say("Calibration complete.")


# ─────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────
class MLPinchDetector:
    """
    Sliding-window pinch detector that runs the trained classifier and
    debounces its output to produce pinch_start / pinch_end events.
    """

    def __init__(self, model, window_samples, emit_fn):
        self.model = model
        self.window_samples = window_samples
        self.emit = emit_fn

        self.ring = deque(maxlen=window_samples + STEP_SAMPLES * 4)
        self.samples_since_step = 0

        self.state = "idle"
        self.above_count = 0
        self.below_count = 0
        self.last_event_t = 0.0
        self.pinch_start_t = 0.0
        self.n_start = self.n_end = self.n_autorelease = 0

        self.recent_p = deque(maxlen=200)
        self.last_heartbeat = time.monotonic()

    def _predict_p_pinch(self):
        if len(self.ring) < self.window_samples:
            return None
        window = np.array(list(self.ring)[-self.window_samples:], dtype=np.float32)
        feats = extract_features(window).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(feats)[0, 1])
        # decision_function fallback → squash to (0,1)
        d = float(self.model.decision_function(feats)[0])
        return 1.0 / (1.0 + np.exp(-d))

    def feed(self, emg_sample):
        self.ring.append(emg_sample)
        self.samples_since_step += 1
        if self.samples_since_step < STEP_SAMPLES:
            return
        self.samples_since_step = 0

        p = self._predict_p_pinch()
        if p is None:
            return
        self.recent_p.append(p)
        now = time.monotonic()
        is_pinch = p >= PROB_THRESHOLD

        if self.state == "idle":
            if is_pinch:
                self.above_count += 1
                if (self.above_count >= ON_CONSEC
                        and now - self.last_event_t >= DEBOUNCE_S):
                    self.state = "active"
                    self.pinch_start_t = now
                    self.above_count = self.below_count = 0
                    self.last_event_t = now
                    self.n_start += 1
                    self.emit("pinch_start")
                    print(f"[{time.strftime('%H:%M:%S')}] pinch_start  p={p:.2f}")
            else:
                self.above_count = 0
        else:
            if not is_pinch:
                self.below_count += 1
                if (self.below_count >= OFF_CONSEC
                        and now - self.last_event_t >= DEBOUNCE_S):
                    self.state = "idle"
                    self.above_count = self.below_count = 0
                    self.last_event_t = now
                    held = now - self.pinch_start_t
                    self.n_end += 1
                    self.emit("pinch_end")
                    print(f"[{time.strftime('%H:%M:%S')}] pinch_end    "
                          f"p={p:.2f} held={held*1000:.0f}ms")
            else:
                self.below_count = 0

            if now - self.pinch_start_t > MAX_PINCH_S:
                self.state = "idle"
                self.above_count = self.below_count = 0
                self.last_event_t = now
                self.n_autorelease += 1
                self.emit("pinch_end")
                print(f"[{time.strftime('%H:%M:%S')}] pinch_end    "
                      f"(AUTO-RELEASE after {MAX_PINCH_S:.0f}s)")

        if now - self.last_heartbeat >= HEARTBEAT_S:
            if self.recent_p:
                avg = sum(self.recent_p) / len(self.recent_p)
                peak = max(self.recent_p)
            else:
                avg = peak = 0.0
            print(f"  · heartbeat: state={self.state:6s} "
                  f"avg_p={avg:.2f} peak={peak:.2f}  "
                  f"starts={self.n_start} ends={self.n_end} auto={self.n_autorelease}")
            self.last_heartbeat = now


def make_emitter(dry_run):
    if dry_run:
        print("DRY RUN — no LSL outlet, events print only.")
        return (lambda label: None), (lambda: None)

    from pylsl import StreamInfo, StreamOutlet, local_clock

    info = StreamInfo(
        name="Experiment.VIBE.MyoPinch", type="Markers", channel_count=1,
        nominal_srate=0, channel_format="string", source_id="myo-pinch-ml-01",
    )
    outlet = StreamOutlet(info)
    print("LSL outlet 'Experiment.VIBE.MyoPinch' is live. Waiting for consumers…")
    return (lambda label: outlet.push_sample([label], timestamp=local_clock())), \
           (lambda: None)


async def cmd_detect(args):
    if not os.path.exists(CONFIG_FILE) or not os.path.exists(MODEL_FILE):
        print(f"Missing {CONFIG_FILE} or {MODEL_FILE}. "
              f"Run: python {sys.argv[0]} calibrate")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    with open(MODEL_FILE, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    print(f"Loaded model: {bundle['name']} "
          f"(trained {cfg.get('calibrated_at', '?')}, "
          f"{cfg.get('n_pinch_trials', '?')} pinch trials)")

    emit, cleanup = make_emitter(args.dry_run)
    detector = MLPinchDetector(model, cfg["window_samples"], emit)

    m = await connect_myo()
    m.attach_buffer(deque(maxlen=16), on_sample=detector.feed)

    await m.set_mode(
        classifier_mode=ClassifierMode.ENABLED,
        emg_mode=EMGMode.SEND_FILT,
        imu_mode=IMUMode.SEND_ALL,
    )
    await m.start()

    print("Streaming. Ctrl-C to stop.\n")
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nShutting down…")
    finally:
        await m.stop()
        if detector.state == "active":
            emit("pinch_end")
        cleanup()
        await m.disconnect()
        print(f"Final counts: starts={detector.n_start}  "
              f"ends={detector.n_end}  auto={detector.n_autorelease}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="ML-based Myo pinch calibration and detection.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate", help="Guided calibration; trains and saves a model.")
    p_cal.add_argument("--n-trials", type=int, default=30,
                       help="Number of pinch trials to record (max 30; default 30).")

    p_det = sub.add_parser("detect", help="Stream pinch markers via LSL.")
    p_det.add_argument("--dry-run", action="store_true",
                       help="Print events only; do not create an LSL outlet.")

    args = ap.parse_args()
    if args.cmd == "calibrate":
        args.n_trials = max(1, min(30, args.n_trials))

    handler = {"calibrate": cmd_calibrate, "detect": cmd_detect}[args.cmd]
    asyncio.run(handler(args))


if __name__ == "__main__":
    main()