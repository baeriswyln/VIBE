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
myo_pinch_ml.py — Multi-class ML pinch calibration and runtime detection.

v3 changes (false-positive reduction)
-------------------------------------
* MULTI-CLASS CALIBRATION. We now also capture confusable non-pinch
  movements: fist, finger-spread, wrist-in, wrist-out, arm-movement.
  The classifier learns to discriminate pinch from those specifically,
  rather than just "rest vs anything muscular".
* STEADY-STATE WINDOWS. Calibration windows are now sampled during the
  *hold* phase of the gesture (250 ms into the cue, 400 ms long), not
  across the onset transient where every gesture looks similar.
* STRICTER INFERENCE. We require both an absolute pinch-probability
  threshold AND a margin over the next-best class before firing. Cuts
  fist/spread confusions sharply.
* RICHER HEARTBEAT. Heartbeat now prints the average probability per
  class so you can see which gesture is being confused with pinch and
  re-balance calibration accordingly.

Usage
-----
    python myo_pinch_ml.py calibrate           # full guided calibration
    python myo_pinch_ml.py calibrate --quick   # only pinch + fist + rest
    python myo_pinch_ml.py detect              # stream pinch markers to LSL
    python myo_pinch_ml.py detect --dry-run    # print events only

Requires: dl-myo, pylsl, numpy, scikit-learn, sounddevice, pyttsx3
    pip install dl-myo pylsl numpy scikit-learn sounddevice pyttsx3
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
EMG_RATE_HZ      = 50      # filtered EMG aggregated stream rate
WINDOW_MS        = 400     # length of analysis window (longer → more steady-state info)
WINDOW_SAMPLES   = max(8, int(EMG_RATE_HZ * WINDOW_MS / 1000))
HOLD_OFFSET_MS   = 250     # capture window starting this long AFTER the cue
HOLD_OFFSET_SAMP = int(EMG_RATE_HZ * HOLD_OFFSET_MS / 1000)

# Calibration timing (seconds)
PREP_DURATION      = 1.5   # low beep → user gets ready
HOLD_DURATION      = 0.9   # how long the user holds the gesture after the high beep
                           # (must be ≥ HOLD_OFFSET_MS/1000 + WINDOW_MS/1000)
INTER_TRIAL_REST   = 2.5   # gap between trials
REST_WINDOW_OFFSET = 1.2   # take rest window this long into the gap

# ─────────────────────────────────────────────────────────────────────
# Class definitions and calibration plan
# ─────────────────────────────────────────────────────────────────────
# "label" is what the classifier sees; "is_pinch" tells inference which
# one means "fire pinch_start". All other labels are negatives.
#
# At runtime we collapse classes into: pinch (positive) vs everything else.
# But we keep their identities during training so the model can carve out
# tight decision boundaries between pinch and the specific confusers.
CLASSES = [
    # label,      is_pinch, prompt (TTS)
    ("pinch",     True,  "Pinch thumb and index finger and hold."),
    ("fist",      False, "Make a tight fist and hold."),
    ("spread",    False, "Spread your fingers wide and hold."),
    ("wave_in",   False, "Bend your wrist inward, toward your forearm, and hold."),
    ("wave_out",  False, "Bend your wrist outward, away from your forearm, and hold."),
    ("arm_move",  False, "Move your whole arm around gently — no finger movement."),
    ("rest",      False, "Relax your arm completely."),
]

# Default calibration plan: how many trials of each class.
# Total: 14 + 5+5+3+3+5+5 (rest also pulled from gaps) ≈ 30 events.
DEFAULT_PLAN = {
    "pinch":    14,
    "fist":     5,
    "spread":   5,
    "wave_in":  3,
    "wave_out": 3,
    "arm_move": 5,
    "rest":     5,    # explicit rest events in addition to inter-trial rest
}

# "Quick" plan if you only care about the most common confusion (fist).
QUICK_PLAN = {
    "pinch":    12,
    "fist":     8,
    "rest":     5,
}

# ─────────────────────────────────────────────────────────────────────
# Runtime detection knobs
# ─────────────────────────────────────────────────────────────────────
STEP_MS         = 60         # how often we re-classify (sliding window stride)
STEP_SAMPLES    = max(1, int(EMG_RATE_HZ * STEP_MS / 1000))

# Inference gate (tuned for low false-positive rate):
#   P(pinch) must clear PROB_THRESHOLD,
#   AND P(pinch) must beat the next-best class by MARGIN.
PROB_THRESHOLD  = 0.60
MARGIN          = 0.15       # pinch P - next-best P

ON_CONSEC       = 3          # ~180 ms of "pinch-like" classifier output before firing
OFF_CONSEC      = 3          # ~180 ms of "not-pinch" before releasing
DEBOUNCE_S      = 0.10
MAX_PINCH_S     = 8.0
HEARTBEAT_S     = 5.0


# ─────────────────────────────────────────────────────────────────────
# Audio cues — generated sine tones via sounddevice, TTS via pyttsx3.
# Both are optional and degrade gracefully if missing.
# ─────────────────────────────────────────────────────────────────────
class Cue:
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

        self._low  = self._tone(freq=440.0,  ms=180)   # A4 — "prepare"
        self._high = self._tone(freq=1175.0, ms=120)   # D6 — "go NOW"
        self._done = self._tone(freq=880.0,  ms=250)   # A5 — "done"

    def _tone(self, freq, ms, volume=0.35):
        t = np.linspace(0, ms / 1000.0, int(self.SR * ms / 1000.0), endpoint=False)
        wave = np.sin(2 * np.pi * freq * t).astype(np.float32) * volume
        fade = int(0.005 * self.SR)
        if len(wave) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            wave[:fade]  *= ramp
            wave[-fade:] *= ramp[::-1]
        return wave

    def _play(self, wave):
        if self._sd is None: return
        try: self._sd.play(wave, self.SR, blocking=False)
        except Exception as e: print(f"[cue] play failed: {e}")

    def prep(self): self._play(self._low)
    def go(self):   self._play(self._high)
    def done(self): self._play(self._done)

    def say(self, text):
        print(f"[say] {text}")
        if self._tts is None: return
        def _run():
            try:
                self._tts.say(text)
                self._tts.runAndWait()
            except Exception as e:
                print(f"[cue] tts failed: {e}")
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=8.0)


# ─────────────────────────────────────────────────────────────────────
# Myo client
# ─────────────────────────────────────────────────────────────────────
class MyMyo(MyoClient):
    buffer = None
    on_sample = None

    def attach_buffer(self, buf, on_sample=None):
        self.buffer = buf
        self.on_sample = on_sample

    async def on_classifier_event(self, ce: ClassifierEvent): pass

    async def on_aggregated_data(self, ad: AggregatedData):
        emg = tuple(ad.fvd.fv) + (ad.fvd.mask,)
        if self.buffer is not None:
            self.buffer.append((time.monotonic(), emg))
        if self.on_sample is not None:
            self.on_sample(emg)

    async def on_emg_data(self, emg: EMGData):            pass
    async def on_fv_data(self, fvd: FVData):              pass
    async def on_imu_data(self, imu: IMUData):            pass
    async def on_motion_event(self, me: MotionEvent):     pass
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
    Window: (T, C) float array of EMG samples (8 filtered channels; the
    9th 'mask' element is dropped here).

    Features per channel (5):
      MAV, RMS, WL (waveform length), VAR, ZC (zero crossings around mean)
    """
    w = np.asarray(window, dtype=np.float32)
    if w.ndim == 1:
        w = w[None, :]
    if w.shape[1] == 9:
        w = w[:, :8]

    mav = np.mean(np.abs(w), axis=0)
    rms = np.sqrt(np.mean(w * w, axis=0))
    diff = np.diff(w, axis=0)
    wl  = (np.sum(np.abs(diff), axis=0)
           if diff.size else np.zeros(w.shape[1], dtype=np.float32))
    var = np.var(w, axis=0)
    centered = w - w.mean(axis=0, keepdims=True)
    zc = (np.sum(np.diff(np.sign(centered), axis=0) != 0, axis=0).astype(np.float32)
          if w.shape[0] > 1 else np.zeros(w.shape[1], dtype=np.float32))

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
def _extract_window_at(buffer_snapshot, start_time, length_samples):
    """
    Take a window starting at start_time. Returns None if the buffer
    doesn't yet have enough samples after start_time.
    """
    if not buffer_snapshot:
        return None
    times = np.array([t for t, _ in buffer_snapshot])
    start = int(np.searchsorted(times, start_time))
    end = start + length_samples
    if start < 0 or end > len(buffer_snapshot):
        return None
    return np.array([emg for _, emg in buffer_snapshot[start:end]],
                    dtype=np.float32)


def _interleave_trials(plan):
    """
    Build a randomised but balanced trial order.

    We don't run all 14 pinches in a row then all the fists — that
    encourages the band to drift or the user to fatigue in class-correlated
    ways. Shuffle so each class is spread evenly through the session.
    """
    trials = []
    for label, _, _ in CLASSES:
        trials += [label] * plan.get(label, 0)
    rng = np.random.default_rng(42)
    rng.shuffle(trials)
    return trials


async def run_calibration(m, plan, cue):
    buf = deque(maxlen=EMG_RATE_HZ * 60 * 6)  # 6 minutes is plenty
    m.attach_buffer(buf)

    await m.set_mode(
        classifier_mode=ClassifierMode.ENABLED,
        emg_mode=EMGMode.SEND_FILT,
        imu_mode=IMUMode.SEND_ALL,
    )
    await m.start()

    label_to_prompt = {lbl: prompt for (lbl, _, prompt) in CLASSES}
    trials = _interleave_trials(plan)
    total = len(trials)

    cue.say(
        "Calibration starting. You will be asked to perform several different "
        "gestures. Each time you hear the low beep, get ready for the gesture "
        "I just named. When you hear the high beep, perform the gesture and "
        "hold it until you hear the next instruction."
    )
    await asyncio.sleep(1.0)

    print(f"\nWill record {total} trials total:")
    for lbl, _, _ in CLASSES:
        n = plan.get(lbl, 0)
        if n: print(f"   {lbl:>9s}: {n}")
    print("Press Ctrl-C anytime after a reasonable subset to stop early.\n")

    windows_by_label = {lbl: [] for lbl, _, _ in CLASSES}

    try:
        for i, label in enumerate(trials, 1):
            print(f"--- trial {i}/{total}: {label} ---")
            # 1) speak the instruction first, so the user knows what's coming
            cue.say(label_to_prompt[label])
            # short settling time after TTS
            await asyncio.sleep(0.4)

            # 2) prep beep → silence → go beep
            cue.prep()
            await asyncio.sleep(PREP_DURATION)

            go_time = time.monotonic()
            cue.go()

            # 3) wait until well into the steady-state hold, then cut a window
            await asyncio.sleep(HOLD_OFFSET_MS / 1000 + WINDOW_MS / 1000 + 0.05)
            window_start = go_time + HOLD_OFFSET_MS / 1000
            snapshot = list(buf)
            window = _extract_window_at(snapshot, window_start, WINDOW_SAMPLES)
            if window is None:
                print("    ⚠ not enough buffered samples — skipping trial")
            else:
                windows_by_label[label].append(window)
                act = float(np.linalg.norm(np.mean(np.abs(window[:, :8]), axis=0)))
                print(f"    captured ({window.shape})  activity={act:.1f}")

            # 4) finish out the trial: instruction to relax, plus extra rest
            #    window pulled from the gap.
            await asyncio.sleep(0.15)
            cue.say("Relax.")
            await asyncio.sleep(REST_WINDOW_OFFSET)
            rest_start = time.monotonic() - WINDOW_MS / 1000
            snapshot = list(buf)
            rest_win = _extract_window_at(snapshot, rest_start, WINDOW_SAMPLES)
            if rest_win is not None:
                windows_by_label["rest"].append(rest_win)

            await asyncio.sleep(max(0.0, INTER_TRIAL_REST - REST_WINDOW_OFFSET))

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nCalibration stopped early by user.")

    await m.stop()

    # Sanity: we need at least some pinches.
    if len(windows_by_label["pinch"]) < 5:
        raise RuntimeError(
            f"Only captured {len(windows_by_label['pinch'])} pinch trials — "
            "need at least 5. Retry calibration."
        )

    cue.done()
    summary = ", ".join(f"{lbl}:{len(ws)}" for lbl, ws in windows_by_label.items()
                        if ws)
    cue.say(f"Captured {sum(len(ws) for ws in windows_by_label.values())} "
            "trials. Training models now.")
    print(f"\nCapture summary: {summary}")

    return windows_by_label


# ─────────────────────────────────────────────────────────────────────
# Multi-class training: try a few classifiers, pick the best by CV.
# We score by pinch-class F1 (not overall accuracy), since the cost we
# care about is "did pinch get confused for/with something else".
# ─────────────────────────────────────────────────────────────────────
def train_and_select(windows_by_label):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import make_scorer, f1_score

    # Drop classes with <2 examples (StratifiedKFold needs ≥2 per class).
    labels_with_data = [lbl for lbl in windows_by_label
                        if len(windows_by_label[lbl]) >= 2]
    if "pinch" not in labels_with_data:
        raise RuntimeError("Need at least 2 pinch trials to train.")

    X_list, y_list, class_names = [], [], []
    for cls_id, lbl in enumerate(sorted(labels_with_data)):
        for w in windows_by_label[lbl]:
            X_list.append(extract_features(w))
            y_list.append(cls_id)
        class_names.append(lbl)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    pinch_id = class_names.index("pinch")

    print(f"\nTraining set: {X.shape[0]} windows, {X.shape[1]} features, "
          f"{len(class_names)} classes ({', '.join(class_names)})")

    candidates = {
        "shrinkage_lda": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "logreg_l2": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=1,
            class_weight="balanced",
        ),
    }

    # Score by macro-averaged accuracy AND by pinch-class F1 specifically.
    pinch_f1 = make_scorer(f1_score, labels=[pinch_id], average="macro",
                           zero_division=0)

    # CV folds: clamped to the smallest class size.
    min_class_n = min((y == c).sum() for c in range(len(class_names)))
    k = min(5, int(min_class_n))
    if k < 2:
        raise RuntimeError("At least one class has <2 trials; cannot CV.")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

    results = {}
    for name, model in candidates.items():
        try:
            acc_macro = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            f1_p      = cross_val_score(model, X, y, cv=cv, scoring=pinch_f1)
            results[name] = {
                "acc_mean":      float(acc_macro.mean()),
                "acc_std":       float(acc_macro.std()),
                "pinch_f1_mean": float(f1_p.mean()),
                "pinch_f1_std":  float(f1_p.std()),
            }
            print(f"  {name:>14s}  acc = {acc_macro.mean():.3f} ± "
                  f"{acc_macro.std():.3f}   pinch_f1 = {f1_p.mean():.3f} ± "
                  f"{f1_p.std():.3f}")
        except Exception as e:
            print(f"  {name:>14s}  failed: {e}")
            results[name] = {"error": str(e)}

    scored = [(n, r) for n, r in results.items() if "pinch_f1_mean" in r]
    if not scored:
        raise RuntimeError("All candidate models failed during CV.")
    # Pick best by pinch_f1, tie-break by overall accuracy.
    scored.sort(key=lambda kv: (kv[1]["pinch_f1_mean"], kv[1]["acc_mean"]),
                reverse=True)
    best_name = scored[0][0]
    print(f"\n→ best model: {best_name}")

    best_model = candidates[best_name]
    best_model.fit(X, y)

    # Print a self-CV confusion matrix so user can see where the residual
    # confusions are — i.e. which class to add more examples of if needed.
    from sklearn.model_selection import cross_val_predict
    try:
        y_pred = cross_val_predict(best_model, X, y, cv=cv)
        print("\nConfusion matrix (rows=true, cols=pred):")
        cm = confusion_matrix(y, y_pred)
        header = "          " + " ".join(f"{c:>8s}" for c in class_names)
        print(header)
        for i, c in enumerate(class_names):
            row = " ".join(f"{cm[i, j]:>8d}" for j in range(len(class_names)))
            print(f"  {c:>8s} {row}")
        print(classification_report(y, y_pred, target_names=class_names,
                                    zero_division=0))
    except Exception as e:
        print(f"(confusion matrix unavailable: {e})")

    return best_name, best_model, results, class_names, pinch_id


async def cmd_calibrate(args):
    m = await connect_myo()
    cue = Cue()

    if args.quick:
        plan = QUICK_PLAN
        print("Using QUICK calibration plan (pinch + fist + rest only).")
    else:
        plan = DEFAULT_PLAN
        print("Using DEFAULT calibration plan (all classes).")

    try:
        windows_by_label = await run_calibration(m, plan, cue)
    finally:
        await m.disconnect()

    best_name, best_model, results, class_names, pinch_id = \
        train_and_select(windows_by_label)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump({
            "model": best_model,
            "name": best_name,
            "class_names": class_names,
            "pinch_id": pinch_id,
        }, f)

    cfg = {
        "version":           3,
        "best_model":        best_name,
        "cv_results":        results,
        "class_names":       class_names,
        "pinch_id":          pinch_id,
        "n_trials_per_class": {lbl: len(ws)
                               for lbl, ws in windows_by_label.items()},
        "window_samples":    WINDOW_SAMPLES,
        "window_ms":         WINDOW_MS,
        "emg_rate_hz":       EMG_RATE_HZ,
        "feature_names":     FEATURE_NAMES,
        "prob_threshold":    PROB_THRESHOLD,
        "margin":            MARGIN,
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
    Multi-class sliding-window pinch detector.

    A sample is counted as "pinch-like" only when:
      1) P(pinch) >= PROB_THRESHOLD, AND
      2) P(pinch) - max(other class P) >= MARGIN.

    Consecutive-sample debouncing on top of that gives the final event.
    """

    def __init__(self, model, class_names, pinch_id, window_samples, emit_fn):
        self.model = model
        self.class_names = class_names
        self.pinch_id = pinch_id
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

        self.recent_probs = deque(maxlen=200)
        self.last_heartbeat = time.monotonic()

    def _predict_probs(self):
        if len(self.ring) < self.window_samples:
            return None
        window = np.array(list(self.ring)[-self.window_samples:],
                          dtype=np.float32)
        feats = extract_features(window).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(feats)[0]
        # For LDA with no proba: convert decision_function via softmax.
        d = self.model.decision_function(feats)[0]
        if np.ndim(d) == 0:
            d = np.array([-d, d])
        e = np.exp(d - d.max())
        return e / e.sum()

    def _is_pinch_like(self, probs):
        p_pinch = probs[self.pinch_id]
        others = np.delete(probs, self.pinch_id)
        next_best = float(others.max()) if len(others) else 0.0
        is_pinch = (p_pinch >= PROB_THRESHOLD) and (p_pinch - next_best >= MARGIN)
        return is_pinch, float(p_pinch), next_best

    def feed(self, emg_sample):
        self.ring.append(emg_sample)
        self.samples_since_step += 1
        if self.samples_since_step < STEP_SAMPLES:
            return
        self.samples_since_step = 0

        probs = self._predict_probs()
        if probs is None:
            return
        self.recent_probs.append(probs)
        now = time.monotonic()
        is_pinch, p_pinch, next_best = self._is_pinch_like(probs)

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
                    print(f"[{time.strftime('%H:%M:%S')}] pinch_start  "
                          f"p={p_pinch:.2f} next={next_best:.2f}")
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
                          f"p={p_pinch:.2f} next={next_best:.2f} "
                          f"held={held*1000:.0f}ms")
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
            if self.recent_probs:
                stacked = np.stack(self.recent_probs)
                avg = stacked.mean(axis=0)
                peak = stacked.max(axis=0)
                msg = "  · heartbeat: state={:6s}  ".format(self.state)
                msg += " ".join(
                    f"{name}={avg[i]:.2f}/{peak[i]:.2f}"
                    for i, name in enumerate(self.class_names)
                )
                msg += (f"  starts={self.n_start} ends={self.n_end} "
                        f"auto={self.n_autorelease}")
                print(msg)
            self.last_heartbeat = now


def make_emitter(dry_run):
    if dry_run:
        print("DRY RUN — no LSL outlet, events print only.")
        return (lambda label: None), (lambda: None)

    from pylsl import StreamInfo, StreamOutlet, local_clock

    info = StreamInfo(
        name="MyoPinch", type="Markers", channel_count=1,
        nominal_srate=0, channel_format="string", source_id="myo-pinch-ml-01",
    )
    outlet = StreamOutlet(info)
    print("LSL outlet 'MyoPinch' is live. Waiting for consumers…")
    return ((lambda label: outlet.push_sample([label], timestamp=local_clock())),
            (lambda: None))


async def cmd_detect(args):
    if not os.path.exists(CONFIG_FILE) or not os.path.exists(MODEL_FILE):
        print(f"Missing {CONFIG_FILE} or {MODEL_FILE}. "
              f"Run: python {sys.argv[0]} calibrate")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    with open(MODEL_FILE, "rb") as f:
        bundle = pickle.load(f)

    if cfg.get("version", 1) < 3:
        print(f"{CONFIG_FILE} is from an older version. Please re-run calibrate.")
        sys.exit(1)

    model       = bundle["model"]
    class_names = bundle["class_names"]
    pinch_id    = bundle["pinch_id"]

    print(f"Loaded model: {bundle['name']} | classes: {class_names} "
          f"| trained {cfg.get('calibrated_at', '?')}")
    print(f"Inference gate: P(pinch) ≥ {PROB_THRESHOLD}, "
          f"margin ≥ {MARGIN}, ON_CONSEC={ON_CONSEC}")

    emit, cleanup = make_emitter(args.dry_run)
    detector = MLPinchDetector(
        model, class_names, pinch_id, cfg["window_samples"], emit
    )

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
        description="Multi-class ML Myo pinch calibration and detection.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate", help="Guided calibration; trains and saves a model.")
    p_cal.add_argument("--quick", action="store_true",
                       help="Use the QUICK plan (only pinch + fist + rest).")

    p_det = sub.add_parser("detect", help="Stream pinch markers via LSL.")
    p_det.add_argument("--dry-run", action="store_true",
                       help="Print events only; do not create an LSL outlet.")

    args = ap.parse_args()
    handler = {"calibrate": cmd_calibrate, "detect": cmd_detect}[args.cmd]
    asyncio.run(handler(args))


if __name__ == "__main__":
    main()