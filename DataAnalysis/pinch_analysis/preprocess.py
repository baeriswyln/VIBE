"""
preprocess.py — Clean the VIBE DSI-24 EEG for motor decoding.

Design choices, and why:

* **Channel selection.** A1/A2 are mastoid references, X1-X3 are flat analog aux
  (~0.2 uV std), TRG is an all-zero trigger. None are scalp EEG. That leaves 18
  *recorded* scalp electrodes for analysis; A1/A2 build a linked-mastoid reference.

* **Recording reference = Pz.** The data were recorded referenced to Pz, which is
  therefore absent from the file (a reference electrode carries no signal against
  itself). This needs NO correction here, because re-referencing to linked mastoids
  cancels the reference: each sample is ``V_ch - V_Pz`` and subtracting
  ``(V_A1+V_A2)/2 - V_Pz`` leaves ``V_ch - (V_A1+V_A2)/2`` — the Pz term vanishes,
  so the recorded channels are unaffected by the choice of online reference.
  Optionally (``restore_reference=True``) we reconstitute Pz as an explicit
  all-zero channel *before* re-referencing; re-referencing then gives Pz a real
  signal, recovering it as a usable 19th electrode (and making an average
  reference unbiased). This does not change any of the 18 recorded channels — see
  ``validate_pz_reference.py``.

* **Bad-channel detection.** The raw signals contain flat channels (Fz ~0.4 uV
  std) and saturating channels (O1/O2/T5 railing near +/-10 mV). We flag both
  extremes robustly (MAD around the log-std) and interpolate them. The
  reconstituted Pz is excluded from this check (it is legitimately ~0 until
  re-referencing).

* **Reference.** Linked mastoids ((A1+A2)/2) is the conventional, robust choice
  for this cap and keeps a clean physical reference; we fall back to a robust
  average reference if the mastoids are themselves bad. (Average reference is
  fragile with ~18 electrodes and heavy artifact.)

* **Filtering.** 1-40 Hz band-pass (captures delta..beta, the motor mu/beta
  bands live at 8-30 Hz; a pinch's readiness potential is slow <5 Hz) plus a
  50 Hz notch (Swiss mains). 1 Hz high-pass also kills the electrode drift.

The function returns a cleaned copy; it never writes over the raw input.
"""
from __future__ import annotations

import mne
import numpy as np

from .dataset import MASTOIDS, NON_SCALP

# 18 recorded scalp electrodes after legacy renaming (T3->T7 etc.).
# Pz is absent because it was the online recording reference; it can be
# reconstituted via preprocess(restore_reference=True).
SCALP_CHANNELS = ["P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz", "Fp1", "Fp2",
                  "T7", "P7", "O1", "O2", "F7", "F8", "P8", "T8"]
REFERENCE_CHANNEL = "Pz"


def detect_bad_channels(raw: mne.io.BaseRaw, picks: list[str],
                        flat_uv: float = 1.0, z_thresh: float = 3.5,
                        sat_uv: float = 5000.0, sat_frac: float = 0.02
                        ) -> dict[str, list[str]]:
    """Flag flat, saturating, and high-variance channels.

    Returns dict {'flat', 'saturating', 'noisy', 'all'} of channel names.
    """
    data = raw.get_data(picks=picks) * 1e6  # V -> uV
    std = data.std(axis=1)
    flat = [picks[i] for i in range(len(picks)) if std[i] < flat_uv]
    sat_fraction = (np.abs(data) > sat_uv).mean(axis=1)
    saturating = [picks[i] for i in range(len(picks))
                  if sat_fraction[i] > sat_frac]
    # robust z on log-std over the non-flat channels
    ok = std > flat_uv
    logstd = np.log(std[ok] + 1e-9)
    med = np.median(logstd) # alternative median less sensitive to extreme values
    mad = np.median(np.abs(logstd - med)) + 1e-9 # robust counterpart to std - measure spread without being biased by extreme values
    z = 0.6745 * (np.log(std + 1e-9) - med) / mad # z-score
    noisy = [picks[i] for i in range(len(picks))
             if ok[i] and z[i] > z_thresh]
    allbad = sorted(set(flat) | set(saturating) | set(noisy))
    return {"flat": flat, "saturating": saturating, "noisy": noisy,
            "all": allbad}


def preprocess(raw: mne.io.BaseRaw, l_freq: float = 1.0, h_freq: float = 40.0,
               line_freq: float = 50.0, reference: str = "linked_mastoid",
               drop_bads: bool = True, restore_reference: bool = False,
               verbose: bool = False) -> tuple[mne.io.BaseRaw, dict]:
    """Return a cleaned copy of ``raw`` plus a small report dict.

    ``restore_reference``: reconstitute the online reference (Pz) as an all-zero
    channel before re-referencing, recovering it as a usable 19th electrode. This
    leaves the 18 recorded channels unchanged under linked-mastoid referencing.
    """
    raw = raw.copy()
    report: dict = {}

    recorded_scalp = [c for c in SCALP_CHANNELS if c in raw.ch_names]
    mast = [c for c in MASTOIDS if c in raw.ch_names]

    # --- reconstitute the recording reference (Pz) as an explicit zero channel ---
    if restore_reference and REFERENCE_CHANNEL not in raw.ch_names:
        raw = mne.add_reference_channels(raw, REFERENCE_CHANNEL)
        # give the reconstituted electrode its 10-20 position (others keep theirs)
        raw.set_montage("standard_1020", on_missing="ignore", match_case=False)
    restored = restore_reference and REFERENCE_CHANNEL in raw.ch_names
    scalp = recorded_scalp + ([REFERENCE_CHANNEL] if restored else [])
    report["restored_reference"] = REFERENCE_CHANNEL if restored else None

    # --- bad channels (on the *recorded* scalp only; a reconstituted Pz is
    #     legitimately ~0 until re-referencing and must not be flagged flat) ---
    bads = detect_bad_channels(raw, recorded_scalp)
    report["bad_channels"] = bads
    raw.info["bads"] = bads["all"]

    # --- filter (before referencing/interpolation) ---
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks="all",
               fir_design="firwin", verbose="ERROR")
    if line_freq and h_freq > line_freq:
        raw.notch_filter(freqs=[line_freq], picks="all", verbose="ERROR")

    # --- reference ---
    used_ref = reference
    if reference == "linked_mastoid" and len(mast) >= 1:
        good_mast = [m for m in mast if m not in raw.info["bads"]]
        if good_mast:
            raw.set_eeg_reference(ref_channels=good_mast, verbose="ERROR")
        else:
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")
            used_ref = "average(fallback)"
    elif reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    report["reference"] = used_ref

    # --- interpolate bad scalp channels (needs montage positions) ---
    if drop_bads and raw.info["bads"]:
        interpable = [b for b in raw.info["bads"] if b in scalp]
        raw.info["bads"] = interpable
        try:
            raw.interpolate_bads(reset_bads=True, verbose="ERROR")
            report["interpolated"] = interpable
        except Exception as e:  # noqa: BLE001
            report["interpolate_error"] = str(e)

    report["scalp_channels"] = scalp
    if verbose:
        print("Preprocess report:", report)
    return raw, report


def pick_scalp(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Return a copy with only scalp EEG channels (drops refs/aux/stim)."""
    keep = [c for c in raw.ch_names if c not in NON_SCALP]
    return raw.copy().pick(keep)


if __name__ == "__main__":
    from .dataset import find_sessions, load_eeg, DATA_ROOT
    sess = find_sessions(DATA_ROOT / "subject_8")[0]
    raw, ts = load_eeg(sess)
    clean, rep = preprocess(raw, verbose=True)
    print("Clean raw:", clean)
