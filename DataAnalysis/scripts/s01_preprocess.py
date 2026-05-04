"""
s01_preprocess.py — Clean raw EEG.

Steps:
  1. Load raw recording (EDF/BDF/FIF/etc.)
  2. Set channel montage (standard 10-20 here; change to match your cap)
  3. Bandpass filter (1–40 Hz default) + notch at line frequency
  4. Detect bad channels and interpolate
  5. Re-reference to average
  6. Run ICA, auto-label components, drop eye/heart/muscle artifacts
  7. Save cleaned recording

Notes on choices:
  - 1 Hz highpass is aggressive enough to remove drift but may attenuate slow
    cognitive components (CNV, P300 baseline). Drop to 0.1 Hz if you care about
    slow ERPs; keep at 1 Hz if you're doing ICA (ICA needs ~1 Hz highpass).
  - 40 Hz lowpass cuts most muscle artifact. Raise to 80–100 Hz if you need
    gamma-band features.
  - Line noise: 50 Hz in Europe, 60 Hz in North America. Adjust below.

Usage:
  uv run python s01_preprocess.py --input data/raw/sub-01.edf \\
      --output data/clean/sub-01-clean.fif --line-freq 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
from mne.preprocessing import ICA


def preprocess_from_file(
    input_path: Path,
    output_path: Path,
    line_freq: float = 50.0,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    montage: str = "standard_1020",
) -> mne.io.Raw:
    raw = mne.io.read_raw(input_path, preload=True)
    return preprocess(raw, output_path, line_freq, l_freq, h_freq, montage)

def preprocess(
    raw: mne.io.Raw,
    output_path: Path,
    line_freq: float = 50.0,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    montage: str = "standard_1020",
) -> mne.io.Raw:
    # --- 1. Montage --------------------------------------------------------
    # Channel positions are needed for spherical interpolation, average
    # reference, and any topographic plotting. If your channel names don't
    # match the montage, you'll need to rename or use a custom montage.
    try:
        raw.set_montage(montage, on_missing="warn")
    except ValueError as e:
        print(f"Montage warning: {e}")

    # --- 2. Filter ---------------------------------------------------------
    # Bandpass first to remove drift and high-frequency noise.
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin")
    # Notch removes power-line interference and its harmonic.
    raw.notch_filter(freqs=[line_freq, 2 * line_freq])

    # --- 3. Bad channels ---------------------------------------------------
    # Simple heuristic: flag channels with extreme variance. For production,
    # use the PREP pipeline or pyprep, which has better detection.
    import numpy as np
    data = raw.get_data(picks="eeg")
    channel_std = data.std(axis=1)
    median_std = np.median(channel_std)
    bad_mask = (channel_std > 5 * median_std) | (channel_std < 0.1 * median_std)
    bads = [raw.ch_names[i] for i, b in enumerate(bad_mask) if b]
    if bads:
        print(f"Marking bad channels: {bads}")
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True)

    # --- 4. Re-reference ---------------------------------------------------
    # Average reference is the standard choice when you don't have a strong
    # reason to prefer another. It requires good coverage of the scalp.
    raw.set_eeg_reference("average", projection=False)

    # --- 5. ICA ------------------------------------------------------------
    # ICA assumes the brain produces signals that are mixed at the scalp by
    # volume conduction. It separates them back out. Eye blinks usually appear
    # as a frontal component with characteristic spikes; heartbeat as a slow
    # rhythmic component. We use ICLabel to auto-label and drop non-brain
    # components.
    n_components = min(20, len(raw.ch_names) - len(raw.info["bads"]) - 1)
    ica = ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),  # required by ICLabel
        random_state=42,
        max_iter="auto",
    )
    # ICA is sensitive to slow drift; fit on a 1 Hz highpassed copy.
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None)
    ica.fit(raw_for_ica)

    try:
        from mne_icalabel import label_components
        labels = label_components(raw_for_ica, ica, method="iclabel")
        # Drop anything that's clearly not brain or "other" (mixed).
        exclude = [
            i for i, lab in enumerate(labels["labels"])
            if lab not in ("brain", "other") and labels["y_pred_proba"][i] > 0.7
        ]
        print(f"ICA dropping {len(exclude)} components: "
              f"{[labels['labels'][i] for i in exclude]}")
        ica.exclude = exclude
    except ImportError:
        print("mne-icalabel not installed; skipping auto-labeling.")
        # Fallback: use EOG correlation if you have an EOG channel.
        # eog_inds, _ = ica.find_bads_eog(raw)
        # ica.exclude = eog_inds

    ica.apply(raw)

    # --- 6. Save -----------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output_path, overwrite=True)
    print(f"Saved cleaned recording to {output_path}")
    return raw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--line-freq", type=float, default=50.0,
                   help="50 Hz in Europe, 60 Hz in North America")
    p.add_argument("--l-freq", type=float, default=1.0)
    p.add_argument("--h-freq", type=float, default=40.0)
    p.add_argument("--montage", default="standard_1020")
    args = p.parse_args()
    preprocess_from_file(args.input, args.output, args.line_freq,
               args.l_freq, args.h_freq, args.montage)


if __name__ == "__main__":
    main()
