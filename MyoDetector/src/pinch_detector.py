"""
myo_pinch.py — Unified Myo pinch calibration and runtime detection.

Usage:
    python myo_pinch.py calibrate          # run 50s guided calibration
    python myo_pinch.py detect             # stream pinch markers to LSL
    python myo_pinch.py detect --dry-run   # detect without LSL (debug only)

Requires: dl-myo, pylsl, numpy
Run calibrate first; it writes myo_pinch_config.json that detect loads.
"""

import argparse
import asyncio
import sys
import time
from collections import defaultdict, deque

import numpy as np
from myo import MyoClient, EMGMode, SleepMode, ClassifierMode, IMUMode, VibrationType
from myo.types import *

CONFIG_FILE = "myo_pinch_config.json"

# ─────────────────────────────────────────────────────────────────────
# Calibration config
# ─────────────────────────────────────────────────────────────────────
PHASES = [
    ("rest_1", "Relax your arm completely.",  10.0),
    ("pinch",  "Pinch thumb + index, HOLD.",  10.0),
    ("rest_2", "Relax again.",                10.0),
    ("fist",   "Make a tight FIST, hold.",    10.0),
    ("rest_3", "Relax. Almost done.",         10.0),
]
COUNTDOWN = 3

# ─────────────────────────────────────────────────────────────────────
# Detection config
# ─────────────────────────────────────────────────────────────────────
ON_CONSEC    = 3       # consecutive matching samples to fire pinch_start (~60 ms @ 50 Hz)
OFF_CONSEC   = 5       # consecutive non-matching samples to fire pinch_end
DEBOUNCE_S   = 0.08    # min gap between any two events
MAX_PINCH_S  = 8.0     # auto-release after this long stuck active
HEARTBEAT_S  = 5.0

# Multi-channel detection knobs
MARGIN       = 1.20    # pinch_score must beat next-best by at least this factor
MIN_ACTIVITY = 1.30    # current sample's L2 norm must exceed rest baseline by this factor


class MyMyo(MyoClient):
    samples_by_phase = None
    current_phase = None
    detector = None
    is_calibration_mode = False

    def prepare_for_calibration(self, current_phase, samples_by_phase):
        self.is_calibration_mode = True
        self.current_phase = current_phase
        self.samples_by_phase = samples_by_phase

    def prepare_for_detection(self, detector):
        self.is_calibration_mode = False
        self.detector = detector

    async def on_classifier_event(self, ce: ClassifierEvent):
        pass

    async def on_aggregated_data(self, ad: AggregatedData):
        emg = ad.fvd.fv + (ad.fvd.mask,)

        if self.is_calibration_mode:
            phase = self.current_phase["name"]
            if phase is not None:
                self.samples_by_phase[phase].append(list(emg))
        else:
            self.detector.feed(emg)

    async def on_emg_data(self, emg: EMGData):
        pass

    async def on_fv_data(self, fvd: FVData):
        pass

    async def on_imu_data(self, imu: IMUData):
        pass

    async def on_motion_event(self, me: MotionEvent):
        pass

    async def on_emg_data_aggregated(self, emg: EMGData):
        pass

# ─────────────────────────────────────────────────────────────────────
# Shared: connect to Myo
# ─────────────────────────────────────────────────────────────────────
async def connect_myo():
    print("Connecting to Myo…")
    m = await MyMyo.with_device(aggregate_all=True)
    await m.setup()
    await m.set_sleep_mode(SleepMode.NEVER_SLEEP)
    print("Connected.")
    return m


# ─────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────
async def run_calibration(m):
    samples_by_phase = defaultdict(list)
    current_phase = {"name": None}

    m.prepare_for_calibration(current_phase, samples_by_phase)

    await m.set_mode(
        classifier_mode=ClassifierMode.ENABLED,
        emg_mode=EMGMode.SEND_FILT,
        imu_mode=IMUMode.SEND_ALL,
    )
    await m.vibrate(VibrationType.MEDIUM)
    await m.start()

    for name, instruction, duration in PHASES:
        print(f"\n>>> NEXT: {instruction}")
        for k in range(COUNTDOWN, 0, -1):
            print(f"    starting in {k}…")
            await asyncio.sleep(1)

        await m.vibrate(VibrationType.MEDIUM)
        current_phase["name"] = name
        print(f"    RECORDING [{name}] for {duration:.0f}s")
        await asyncio.sleep(duration)
        current_phase["name"] = None
        await m.vibrate(VibrationType.MEDIUM)

        n = len(samples_by_phase[name])
        print(f"    captured {n} samples (~{n/duration:.0f} Hz)")

    await m.stop()
    await m.vibrate(VibrationType.SHORT)
    await m.vibrate(VibrationType.MEDIUM)
    return samples_by_phase


def _normalize(v, eps=1e-6):
    """L2-normalize a vector. Returns zero vector if input is ~zero."""
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


def analyze(samples_by_phase):
    arrays = {}
    for name, samples in samples_by_phase.items():
        a = np.asarray(samples, dtype=np.float32)
        if len(a) < 50:
            raise RuntimeError(f"Phase '{name}' captured too few samples: {len(a)}")
        skip = max(1, len(a) // 10)  # drop first ~10% (transition)
        arrays[name] = a[skip:]

    rest  = np.vstack([arrays["rest_1"], arrays["rest_2"], arrays["rest_3"]])
    pinch = arrays["pinch"]
    fist  = arrays["fist"]

    n_ch = rest.shape[1]            # usually 8, but we keep it generic
    rest_mean  = rest.mean(axis=0)
    pinch_mean = pinch.mean(axis=0)
    fist_mean  = fist.mean(axis=0)

    # ── Templates: subtract rest baseline, then unit-normalize.
    # The "shape" of activation matters, not its overall magnitude — this
    # makes the detector tolerant of subject-to-subject amplitude differences
    # and of someone pinching harder or softer than during calibration.
    pinch_template = _normalize(pinch_mean - rest_mean)
    fist_template  = _normalize(fist_mean  - rest_mean)

    # Activity gate: typical pinch magnitude over rest.
    rest_norm  = float(np.linalg.norm(rest_mean))
    pinch_norm = float(np.linalg.norm(pinch_mean))

    if pinch_norm <= rest_norm * 1.2:
        raise RuntimeError(
            f"Pinch activation ({pinch_norm:.1f}) not clearly above rest "
            f"({rest_norm:.1f}). Re-seat the band and retry."
        )

    # ── Report separability so user can see why it might fail.
    cos_pinch_fist = float(np.dot(pinch_template, fist_template))
    print("\nPer-channel means:")
    print(f"  {'ch':>3}  {'rest':>8}  {'pinch':>8}  {'fist':>8}  "
          f"{'p-tmpl':>8}  {'f-tmpl':>8}")
    for ch in range(n_ch):
        print(f"  {ch:>3}  {rest_mean[ch]:>8.1f}  {pinch_mean[ch]:>8.1f}  "
              f"{fist_mean[ch]:>8.1f}  {pinch_template[ch]:>+8.3f}  "
              f"{fist_template[ch]:>+8.3f}")
    print(f"\nPinch–fist template cosine similarity: {cos_pinch_fist:+.3f}")
    if cos_pinch_fist > 0.85:
        print("  ⚠  Pinch and fist patterns are very similar — false positives "
              "from fist are likely. Try re-seating the band rotated by one "
              "pod-width and re-running calibration.")

    return {
        "n_channels":     n_ch,
        "rest_mean":      rest_mean.tolist(),
        "pinch_template": pinch_template.tolist(),
        "fist_template":  fist_template.tolist(),
        "rest_norm":      rest_norm,
        "pinch_norm":     pinch_norm,
        "margin":         MARGIN,
        "min_activity":   MIN_ACTIVITY,
        "diagnostics": {
            "pinch_mean":      pinch_mean.tolist(),
            "fist_mean":       fist_mean.tolist(),
            "cos_pinch_fist":  cos_pinch_fist,
            "calibrated_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


async def cmd_calibrate(args):
    m = await connect_myo()
    try:
        samples = await run_calibration(m)
    finally:
        await m.disconnect()

    config = analyze(samples)

    print("\n" + "=" * 50)
    print(f"Saved templates for {config['n_channels']} channels")
    print(f"Rest baseline norm: {config['rest_norm']:.1f}")
    print(f"Pinch activation norm: {config['pinch_norm']:.1f}")
    print("=" * 50)

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nWrote {CONFIG_FILE}")


# ─────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────
class PinchDetector:
    """
    Multi-channel template-matching pinch detector.

    For each incoming EMG sample, we:
      1) subtract the rest baseline,
      2) compute cosine similarity to the pinch template and fist template,
      3) declare "pinch-like" only if the sample is energetic enough AND
         pinch similarity beats fist similarity by MARGIN.

    This rejects wrist/finger motion that lights up a different channel
    pattern than pinch, even when it happens to drive the previously-chosen
    single channel above threshold.
    """

    def __init__(self, cfg, emit_fn):
        self.rest_mean      = np.asarray(cfg["rest_mean"], dtype=np.float32)
        self.pinch_template = np.asarray(cfg["pinch_template"], dtype=np.float32)
        self.fist_template  = np.asarray(cfg["fist_template"],  dtype=np.float32)
        self.rest_norm      = float(cfg["rest_norm"])
        self.margin         = float(cfg.get("margin",       MARGIN))
        self.min_activity   = float(cfg.get("min_activity", MIN_ACTIVITY))
        self.emit = emit_fn

        self.state = "idle"
        self.above_count = 0
        self.below_count = 0
        self.last_event_t = 0.0
        self.pinch_start_t = 0.0

        self.recent_scores = deque(maxlen=100)
        self.recent_act    = deque(maxlen=100)
        self.last_heartbeat = time.monotonic()
        self.n_start = self.n_end = self.n_autorelease = 0

    def _score(self, emg_sample):
        """
        Return (is_pinch_like, pinch_sim, fist_sim, activity_ratio).

        - pinch_sim / fist_sim are cosine similarities in [-1, 1].
        - activity_ratio is current activation norm / rest baseline norm.
        - is_pinch_like requires both an activity gate and a margin over fist.
        """
        x = np.asarray(emg_sample, dtype=np.float32) - self.rest_mean
        x_norm = float(np.linalg.norm(x))

        # Activity gate: ignore samples that aren't above baseline at all.
        activity_ratio = (np.linalg.norm(emg_sample) /
                          (self.rest_norm + 1e-6))
        if x_norm < 1e-3 or activity_ratio < self.min_activity:
            return False, 0.0, 0.0, activity_ratio

        x_unit = x / x_norm
        pinch_sim = float(np.dot(x_unit, self.pinch_template))
        fist_sim  = float(np.dot(x_unit, self.fist_template))

        # Pinch wins, AND wins by the configured margin. The +0.05 floor
        # prevents tiny-similarity ties from passing the ratio check.
        is_pinch_like = (
            pinch_sim > 0.5 and
            pinch_sim > self.margin * max(fist_sim, 0.05)
        )
        return is_pinch_like, pinch_sim, fist_sim, activity_ratio

    def feed(self, emg_sample):
        is_pinch, p_sim, f_sim, act = self._score(emg_sample)
        self.recent_scores.append(p_sim)
        self.recent_act.append(act)
        now = time.monotonic()

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
                          f"p={p_sim:+.2f} f={f_sim:+.2f} act={act:.2f}")
            else:
                self.above_count = 0

        else:  # active
            if not is_pinch:
                self.below_count += 1
                if (self.below_count >= OFF_CONSEC
                        and now - self.last_event_t >= DEBOUNCE_S):
                    self.state = "idle"
                    self.above_count = self.below_count = 0
                    self.last_event_t = now
                    self.n_end += 1
                    held = now - self.pinch_start_t
                    self.emit("pinch_end")
                    print(f"[{time.strftime('%H:%M:%S')}] pinch_end    "
                          f"p={p_sim:+.2f} f={f_sim:+.2f} "
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
            if self.recent_scores:
                avg_p   = sum(self.recent_scores) / len(self.recent_scores)
                peak_p  = max(self.recent_scores)
                avg_act = sum(self.recent_act)    / len(self.recent_act)
            else:
                avg_p = peak_p = avg_act = 0.0
            print(f"  · heartbeat: state={self.state:6s} "
                  f"avg_p_sim={avg_p:+.2f} peak={peak_p:+.2f} "
                  f"act={avg_act:.2f}  starts={self.n_start} "
                  f"ends={self.n_end} auto={self.n_autorelease}")
            self.last_heartbeat = now


def make_emitter(dry_run):
    """Return (emit_fn, cleanup_fn). Builds the LSL outlet unless dry-run."""
    if dry_run:
        print("DRY RUN — no LSL outlet, events print only.")
        return (lambda label: None), (lambda: None)

    from pylsl import StreamInfo, StreamOutlet, local_clock

    info = StreamInfo(
        name="Experiment.VIBE.MyoPinch", type="Markers", channel_count=1,
        nominal_srate=0, channel_format="string", source_id="myo-pinch-01",
    )
    outlet = StreamOutlet(info)
    print("LSL outlet 'Experiment.VIBE.MyoPinch' is live. Waiting for consumers…")

    def emit(label):
        outlet.push_sample([label], timestamp=local_clock())

    def cleanup():
        pass

    return emit, cleanup


async def cmd_detect(args):
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        print(f"No {CONFIG_FILE} found. Run: python {sys.argv[0]} calibrate")
        sys.exit(1)

    # Backward-compat guard: refuse old single-channel configs.
    if "pinch_template" not in cfg:
        print(f"{CONFIG_FILE} is from an older single-channel version. "
              "Please re-run: python {sys.argv[0]} calibrate")
        sys.exit(1)

    print(f"Loaded multi-channel config "
          f"(margin={cfg.get('margin', MARGIN)}, "
          f"min_activity={cfg.get('min_activity', MIN_ACTIVITY)})")

    emit, cleanup = make_emitter(args.dry_run)
    detector = PinchDetector(cfg, emit)

    m = await connect_myo()
    m.prepare_for_detection(detector)

    await m.setup(
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
    ap = argparse.ArgumentParser(description="Myo pinch calibration and detection.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("calibrate", help="Run guided calibration and write config.")

    p_detect = sub.add_parser("detect", help="Stream pinch markers via LSL.")
    p_detect.add_argument("--dry-run", action="store_true",
                          help="Print events only; do not create an LSL outlet.")

    args = ap.parse_args()

    handler = {"calibrate": cmd_calibrate, "detect": cmd_detect}[args.cmd]
    asyncio.run(handler(args))


if __name__ == "__main__":
    main()