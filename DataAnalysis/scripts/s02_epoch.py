"""
02_epoch_binary.py — Cut positive epochs at events AND sample negatives.

For event detection, you need both classes:
  - Positives: windows centered on (or covering) your event timestamps
  - Negatives: windows from periods where no event occurred

Negative sampling rules implemented here:
  1. Same recording as positives (matches subject/session conditions).
  2. Minimum gap from any positive event (default: 2 s buffer on both sides).
  3. Same window length as positives.
  4. Configurable positive:negative ratio.
  5. Random seed for reproducibility.

Window choice for detection:
  - If your events are punctate (a button press, a stimulus): center the
    window on the event. Use --tmin/--tmax like before.
  - If your events have duration (a seizure, a sleep spindle): use windows
    that overlap the event. Set --positive-mode 'covers' below — a window
    counts as positive if it overlaps any event.
  - If you want to *predict* events before they happen: use --tmin -2.0
    --tmax 0.0 (window ends at event onset).

Usage:
  uv run python 02_epoch.py \\
      --raw data/clean/sub-01-clean.fif \\
      --events data/events/sub-01-events.csv \\
      --output data/epochs/sub-01-epo.fif \\
      --tmin -0.5 --tmax 1.5 \\
      --neg-ratio 3 --neg-buffer 2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd


def sample_negatives(
    raw: mne.io.Raw,
    pos_sample_indices: np.ndarray,
    window_len_samples: int,
    n_negatives: int,
    buffer_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample window-start indices that don't overlap any positive epoch."""
    sfreq = raw.info["sfreq"]
    n_samples_total = raw.n_times

    # Build a boolean mask: True wherever a window is forbidden.
    # A negative window starting at sample s spans [s, s + window_len).
    # It must avoid any positive event ± buffer.
    forbidden = np.zeros(n_samples_total, dtype=bool)
    for pos_s in pos_sample_indices:
        lo = max(0, pos_s - window_len_samples - buffer_samples)
        hi = min(n_samples_total, pos_s + window_len_samples + buffer_samples)
        forbidden[lo:hi] = True

    # Valid start indices: window fits in recording AND start sample
    # is not in the forbidden zone (we also need the whole window to be clear)
    max_start = n_samples_total - window_len_samples
    valid_starts = []
    # Use a simple rejection sample. For very dense events this could be slow;
    # in that case build the valid set explicitly.
    attempts = 0
    while len(valid_starts) < n_negatives and attempts < 100 * n_negatives:
        candidate = rng.integers(0, max_start)
        # Check the whole candidate window is clear
        if not forbidden[candidate:candidate + window_len_samples].any():
            valid_starts.append(candidate)
            # Mark this window as used so we don't sample overlapping negatives
            forbidden[candidate:candidate + window_len_samples] = True
        attempts += 1

    if len(valid_starts) < n_negatives:
        print(f"WARNING: only found {len(valid_starts)} valid negative "
              f"windows (asked for {n_negatives}). Recording may be too "
              f"dense with events, or buffer/window too large.")
    return np.array(valid_starts, dtype=int)

def build_binary_epochs_from_file(
    raw_path: Path,
    events_path: Path,
    output_path: Path,
    tmin: float = -0.2,
    tmax: float = 0.8,
    neg_ratio: float = 1.0,
    neg_buffer: float = 2.0,
    reject_uv: float = 150.0,
    seed: int = 42,
) -> mne.Epochs:
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    sfreq = raw.info["sfreq"]

    # --- Positives ---------------------------------------------------------
    df = pd.read_csv(events_path)
    pos_samples = (df["timestamp_seconds"].to_numpy() * sfreq).astype(int)

    build_binary_epochs(raw, df["timestamp_seconds"], output_path, tmin, tmax, neg_ratio, neg_buffer, reject_uv, seed)


def build_binary_epochs(
    raw: mne.Raw,
    marker_ts: pd.Series,
    output_path: Path,
    tmin: float = -0.2,
    tmax: float = 0.8,
    neg_ratio: float = 1.0,
    neg_buffer: float = 2.0,
    reject_uv: float = 150.0,
    seed: int = 42,
) -> mne.Epochs:
    sfreq = raw.info["sfreq"]
    rng = np.random.default_rng(seed)
    pos_samples = (marker_ts.to_numpy() * sfreq).astype(int)

    # Window length in samples (number of samples in [tmin, tmax])
    window_len = int(round((tmax - tmin) * sfreq)) + 1
    buffer_samples = int(round(neg_buffer * sfreq))

    # --- Negatives ---------------------------------------------------------
    n_neg = int(round(len(pos_samples) * neg_ratio))
    # The positive event sits at -tmin into its window, so the window starts
    # at sample (event - tmin*sfreq). We pass event samples directly to the
    # forbidden-zone builder; it already adds window_len padding.
    neg_starts = sample_negatives(
        raw, pos_samples, window_len, n_neg, buffer_samples, rng,
    )
    # Convert negative window starts to "event samples" so MNE epoching with
    # the same tmin/tmax produces the right windows.
    neg_event_samples = neg_starts - int(tmin * sfreq)

    # --- Build event array (MNE format: [sample, prev, id]) ---------------
    pos_events = np.column_stack([pos_samples,
                                  np.zeros_like(pos_samples),
                                  np.ones_like(pos_samples)])     # id=1
    neg_events = np.column_stack([neg_event_samples,
                                  np.zeros_like(neg_event_samples),
                                  np.zeros_like(neg_event_samples) + 2])  # id=2
    events = np.vstack([pos_events, neg_events])
    events = events[events[:, 0].argsort()]

    # Drop events that don't fit in the recording.
    valid = ((events[:, 0] + int(tmin * sfreq) >= 0) &
             (events[:, 0] + int(tmax * sfreq) < raw.n_times))
    events = events[valid]

    epochs = mne.Epochs(
        raw, events,
        event_id={"event": 1, "no_event": 2},
        tmin=tmin, tmax=tmax,
        baseline=(None, 0.0) if tmax > 0 else None,
        reject=dict(eeg=reject_uv * 1e-6),
        preload=True,
    )

    n_pos = (epochs.events[:, 2] == 1).sum()
    n_neg = (epochs.events[:, 2] == 2).sum()
    print(f"Built {len(epochs)} epochs: {n_pos} event, {n_neg} no_event")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(output_path, overwrite=True)
    return epochs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tmin", type=float, default=-0.2)
    p.add_argument("--tmax", type=float, default=0.8)
    p.add_argument("--neg-ratio", type=float, default=1.0,
                   help="negatives per positive (1.0 = balanced)")
    p.add_argument("--neg-buffer", type=float, default=2.0,
                   help="seconds of buffer around positives where negatives "
                        "may not be drawn")
    p.add_argument("--reject-uv", type=float, default=150.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_binary_epochs_from_file(args.raw, args.events, args.output,
                                  args.tmin, args.tmax, args.neg_ratio,
                                  args.neg_buffer, args.reject_uv, args.seed)


if __name__ == "__main__":
    main()