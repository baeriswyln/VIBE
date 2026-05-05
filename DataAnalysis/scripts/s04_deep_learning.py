"""
04_sliding_window_inference.py — Realistic deployment evaluation.

Training on epochs centered on events tells you how separable event vs.
no-event windows are. But at deployment, you don't get nicely centered
windows — you get a continuous EEG stream and must find events in it.

This script:
  1. Trains a detector on epoched data (positive + negative epochs).
  2. Slides a window across a held-out continuous recording with a fixed
     step size, scoring each window.
  3. Post-processes scores: smoothing + non-maximum suppression to merge
     overlapping high-score regions into discrete event predictions.
  4. Compares predictions to ground-truth event times with a tolerance
     window, computing precision, recall, and false alarms per minute —
     the metrics you'd actually report.

Key parameters:
  --window-sec     Match the training window length.
  --step-sec       How far to advance between scores. Smaller = finer
                   localization but more compute. 0.1–0.25 s is typical.
  --tolerance-sec  How close a predicted event must be to a true event to
                   count as a hit. Set based on your application's needs.

Usage:
  uv run python 04_sliding_window_inference.py \\
      --train-epochs data/epochs/sub-01-epo.fif \\
      --test-raw data/clean/sub-02-clean.fif \\
      --test-events data/events/sub-02-events.csv \\
      --window-sec 1.0 --step-sec 0.25 --tolerance-sec 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_detector(epochs_path: Path):
    epochs = mne.read_epochs(epochs_path, preload=True)
    X = epochs.get_data(copy=False)
    y = (epochs.events[:, 2] == 1).astype(int)
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        clf = Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts",  TangentSpace()),
            ("sc",  StandardScaler()),
            ("lr",  LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
    except ImportError:
        # Fallback: log-variance per channel
        from sklearn.preprocessing import FunctionTransformer
        def logvar(x): return np.log(x.var(axis=-1) + 1e-12)
        clf = Pipeline([
            ("logvar", FunctionTransformer(logvar)),
            ("sc",     StandardScaler()),
            ("lr",     LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
    clf.fit(X, y)
    return clf, epochs.info


def slide_and_score(raw, clf, window_sec: float, step_sec: float):
    """Score every window in the recording. Returns (centers_sec, scores)."""
    sfreq = raw.info["sfreq"]
    win = int(round(window_sec * sfreq))
    step = int(round(step_sec * sfreq))
    data = raw.get_data(picks="eeg")  # (n_channels, n_times)
    n_total = data.shape[1]

    starts = np.arange(0, n_total - win, step)
    centers = (starts + win / 2) / sfreq

    # Build a (n_windows, n_channels, win) array. For long recordings this
    # may not fit in memory — chunk if needed.
    X = np.stack([data[:, s:s + win] for s in starts])
    scores = clf.predict_proba(X)[:, 1]
    return centers, scores


def detect_events(centers: np.ndarray, scores: np.ndarray,
                  threshold: float, smoothing_windows: int = 5,
                  refractory_sec: float = 0.5) -> np.ndarray:
    """Smooth scores, threshold, NMS within a refractory period."""
    # Smooth: average across `smoothing_windows` adjacent scores. Reduces
    # spurious single-window peaks.
    smooth = uniform_filter1d(scores, size=smoothing_windows, mode="nearest")
    above = smooth >= threshold
    # NMS: in each connected above-threshold run, take the argmax.
    detections = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            run_scores = smooth[i:j]
            peak = i + int(np.argmax(run_scores))
            detections.append(centers[peak])
            i = j
        else:
            i += 1
    # Apply refractory: drop detections closer than `refractory_sec` to a
    # higher-scoring earlier detection.
    detections = np.array(detections)
    if len(detections) <= 1:
        return detections
    keep = np.ones(len(detections), dtype=bool)
    for i in range(len(detections)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(detections)):
            if detections[j] - detections[i] < refractory_sec:
                keep[j] = False
    return detections[keep]


def score_detections(true_times: np.ndarray, pred_times: np.ndarray,
                     tolerance_sec: float, total_minutes: float):
    """Match predictions to truth within tolerance. Each true event matches
    at most one prediction; greedy nearest-first matching."""
    matched_true = np.zeros(len(true_times), dtype=bool)
    matched_pred = np.zeros(len(pred_times), dtype=bool)
    # For each prediction in score order (here we just take given order),
    # match to nearest unmatched true event within tolerance.
    for pi, pt in enumerate(pred_times):
        diffs = np.abs(true_times - pt)
        diffs[matched_true] = np.inf
        if len(diffs) == 0:
            break
        ti = int(np.argmin(diffs))
        if diffs[ti] <= tolerance_sec:
            matched_true[ti] = True
            matched_pred[pi] = True

    tp = int(matched_true.sum())
    fn = len(true_times) - tp
    fp = int((~matched_pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fa_per_min = fp / total_minutes if total_minutes > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=f1, false_alarms_per_min=fa_per_min)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-epochs", type=Path, required=True)
    p.add_argument("--test-raw", type=Path, required=True)
    p.add_argument("--test-events", type=Path, required=True)
    p.add_argument("--window-sec", type=float, default=1.0)
    p.add_argument("--step-sec", type=float, default=0.25)
    p.add_argument("--tolerance-sec", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold. Sweep in production to choose.")
    args = p.parse_args()

    print("Training detector...")
    clf, _info = train_detector(args.train_epochs)

    print(f"Loading test recording {args.test_raw.name}...")
    raw = mne.io.read_raw_fif(args.test_raw, preload=True)
    total_min = raw.times[-1] / 60

    print("Sliding-window scoring...")
    centers, scores = slide_and_score(raw, clf, args.window_sec, args.step_sec)
    print(f"  Scored {len(scores)} windows")

    # Sweep thresholds for a sense of the operating curve
    print("\nThreshold sweep:")
    true_times = pd.read_csv(args.test_events)["timestamp_seconds"].to_numpy()
    print(f"  (Test set has {len(true_times)} true events in {total_min:.1f} min)")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds = detect_events(centers, scores, threshold=thr)
        m = score_detections(true_times, preds, args.tolerance_sec, total_min)
        print(f"  thr={thr:.2f}: prec={m['precision']:.2f} "
              f"rec={m['recall']:.2f} F1={m['f1']:.2f} "
              f"FA/min={m['false_alarms_per_min']:.2f}")


if __name__ == "__main__":
    main()