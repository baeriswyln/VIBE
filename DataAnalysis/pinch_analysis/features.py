"""
features.py — Feature extraction for pinch detection/prediction.

Two complementary views of an EEG window (n_channels x n_times):

* **Band power + Hjorth** (interpretable): log power in delta/theta/alpha/mu/
  beta1/beta2 per channel, plus Hjorth mobility & complexity. Captures the
  mu/beta desynchronisation that precedes a pinch.
* **Riemannian** (state of the art for motor): channel covariance matrix of the
  8-30 Hz signal, projected to the tangent space. Handled in the pipeline via
  pyriemann; here we just expose the band-limited epoch data.

Optionally we append **gaze features** (angular speed, pupil, openness summary
over the window) which the EDA showed to be strongly pinch-locked.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 12),
    "mu": (8, 13), "beta1": (13, 20), "beta2": (20, 30),
}


def hjorth(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hjorth mobility and complexity along the last axis."""
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var0 = x.var(axis=-1) + 1e-12
    var1 = dx.var(axis=-1) + 1e-12
    var2 = ddx.var(axis=-1) + 1e-12
    mob = np.sqrt(var1 / var0)
    comp = np.sqrt(var2 / var1) / (mob + 1e-12)
    return mob, comp


def bandpower_features(data: np.ndarray, sfreq: float) -> np.ndarray:
    """(n_epochs, n_ch, n_times) -> (n_epochs, n_ch*(n_bands+2)) features."""
    n_ep, n_ch, n_t = data.shape
    nper = min(n_t, int(sfreq))  # ~1 s segments
    freqs, psd = welch(data, fs=sfreq, nperseg=nper, axis=-1)
    feats = []
    for lo, hi in BANDS.values():
        m = (freqs >= lo) & (freqs < hi)
        bp = psd[:, :, m].mean(axis=-1)  # (n_ep, n_ch)
        feats.append(np.log(bp + 1e-12))
    mob, comp = hjorth(data)
    feats.append(mob)
    feats.append(comp)
    return np.concatenate(feats, axis=1)  # (n_ep, n_ch*8)


def bandpower_feature_names(ch_names: list[str]) -> list[str]:
    names = []
    for b in BANDS:
        names += [f"{b}:{c}" for c in ch_names]
    names += [f"hjorth_mob:{c}" for c in ch_names]
    names += [f"hjorth_comp:{c}" for c in ch_names]
    return names
