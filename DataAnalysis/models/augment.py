"""
EEG data augmentation for short (e.g. 2s), per-trial z-scored epochs.

Fixes vs. the original version:
- Time shift no longer wraps the signal (torch.roll -> pad-and-crop),
  which avoided injecting a fake broadband discontinuity at the wrap point.
- Each transform fires independently with its own probability `p`, so the
  model still sees mildly-perturbed / near-original trials sometimes,
  rather than always stacking noise + shift + scale together.
- Scale is applied before noise so the noise magnitude stays close to the
  requested `sigma` regardless of the amplitude draw.
- Added channel dropout and a bandstop filter, both well supported in the
  EEG augmentation literature (Rommel et al. 2022; Mohsenvand et al. 2020)
  as attacking a different failure mode (over-reliance on one channel /
  one frequency band) than noise+shift+scale.

All defaults assume `x` is already z-scored (roughly unit variance per
channel), so noise sigma and scale range are expressed in those units.
"""
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_SIGMA = 0.03            # ~3% of one std, since data is z-scored
DEFAULT_MAX_SHIFT = 25          # samples, e.g. ~0.1s at 250 Hz — keep small vs. epoch length
DEFAULT_SCALE_RANGE = (0.9, 1.1)
DEFAULT_P = 0.5                 # per-transform application probability


def _safe_time_shift(x: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Shift along the last (time) dim without wraparound. Signal shifted
    out of the window is dropped; the vacated edge is zero-padded, avoiding
    the discontinuity that torch.roll introduces on short epochs."""
    if max_shift <= 0:
        return x
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    if shift == 0:
        return x
    T = x.shape[-1]
    if shift > 0:
        # pad at the start, drop from the end
        x = F.pad(x, (shift, 0))[..., :T]
    else:
        x = F.pad(x, (0, -shift))[..., -T:]
    return x


def _channel_dropout(x: torch.Tensor, p: float = DEFAULT_P, max_frac: float = 0.2) -> torch.Tensor:
    """Zero out a random subset of channels (simulates bad/loose electrodes).
    Expects channel dim at -2 (..., channels, time)."""
    if torch.rand(1).item() >= p:
        return x
    n_channels = x.shape[-2]
    n_drop = torch.randint(1, max(2, int(n_channels * max_frac) + 1), (1,)).item()
    drop_idx = torch.randperm(n_channels)[:n_drop]
    x = x.clone()
    x[..., drop_idx, :] = 0.0
    return x


def _bandstop_filter(x: torch.Tensor, fs: float, p: float = DEFAULT_P,
                      band_width_hz: float = 2.0, low_hz: float = 1.0, high_hz: float = None) -> torch.Tensor:
    """Notch out a random narrow frequency band across all channels, forcing
    the model to not over-rely on any single spectral region. Requires `fs`
    (sampling rate) since the epoch itself doesn't carry that info."""
    if torch.rand(1).item() >= p:
        return x
    high_hz = high_hz if high_hz is not None else fs / 2 - band_width_hz
    center = torch.empty(1).uniform_(low_hz + band_width_hz / 2, high_hz).item()
    low = max(center - band_width_hz / 2, 0.5)
    high = center + band_width_hz / 2

    X = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs).to(x.device)
    mask = ~((freqs >= low) & (freqs <= high))
    X = X * mask
    return torch.fft.irfft(X, n=x.shape[-1], dim=-1)


def augment_eeg(
    x: torch.Tensor,
    fs: float= 300,
    sigma: float = DEFAULT_SIGMA,
    max_shift: int = DEFAULT_MAX_SHIFT,
    scale_range: Tuple[float, float] = DEFAULT_SCALE_RANGE,
    p: float = DEFAULT_P,
) -> torch.Tensor:
    """Randomly perturb one z-scored trial `x` (..., channels, time).

    fs: sampling rate in Hz (needed for the bandstop filter).
    p: per-transform probability of being applied; each transform is
       independently sampled, so trials are perturbed to varying degrees
       rather than always receiving the full stack.

    Redraws all randomness on every call, so using this per-sample (e.g.
    in Dataset.__getitem__) gives every trial its own independent
    augmentation each time it's seen.
    """
    if torch.rand(1).item() < p:
        scale = torch.empty(1).uniform_(*scale_range).item()
        x = x * scale

    if torch.rand(1).item() < p:
        x = x + torch.randn_like(x) * sigma

    if torch.rand(1).item() < p:
        x = _safe_time_shift(x, max_shift)

    x = _channel_dropout(x, p=p)
    x = _bandstop_filter(x, fs=fs, p=p)

    return x


# ---------------------------------------------------------------------------
# NumPy / offline equivalents (batched over n_trials, per-trial independent draws)
# ---------------------------------------------------------------------------

def _safe_time_shift_trial(x: np.ndarray, shift: int) -> np.ndarray:
    """Same non-wrapping pad-and-crop shift as the torch version, for one
    (channels, time) trial."""
    if shift == 0:
        return x
    T = x.shape[-1]
    pad_width = [(0, 0)] * (x.ndim - 1) + [(shift, 0) if shift > 0 else (0, -shift)]
    x = np.pad(x, pad_width)
    return x[..., :T] if shift > 0 else x[..., -T:]


def _channel_dropout_trial(x: np.ndarray, rng: np.random.Generator, max_frac: float = 0.2,
                           fill: str = "zero") -> np.ndarray:
    """fill="zero" matches the torch version. fill="noise" instead replaces each dropped
    channel with Gaussian noise of that channel's own std: the channel's information is
    still destroyed, but it isn't flat -- a flat channel has zero power/variance, which
    makes log-power, dB-ratio and Hjorth features (see EEGForest) -inf / NaN."""
    n_channels = x.shape[-2]
    n_drop = rng.integers(1, max(2, int(n_channels * max_frac) + 1))
    drop_idx = rng.permutation(n_channels)[:n_drop]
    x = x.copy()
    if fill == "noise":
        std = x[..., drop_idx, :].std(-1, keepdims=True)
        x[..., drop_idx, :] = rng.normal(0.0, 1.0, size=x[..., drop_idx, :].shape) * std
    else:
        x[..., drop_idx, :] = 0.0
    return x


def _bandstop_filter_trial(x: np.ndarray, fs: float, rng: np.random.Generator,
                            band_width_hz: float = 2.0, low_hz: float = 1.0,
                            high_hz: Optional[float] = None) -> np.ndarray:
    high_hz = high_hz if high_hz is not None else fs / 2 - band_width_hz
    center = rng.uniform(low_hz + band_width_hz / 2, high_hz)
    low, high = center - band_width_hz / 2, center + band_width_hz / 2

    X = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
    mask = ~((freqs >= low) & (freqs <= high))
    X = X * mask
    return np.fft.irfft(X, n=x.shape[-1], axis=-1).astype(x.dtype)


def augment_array(
    X: np.ndarray,
    fs: float,
    sigma: float = DEFAULT_SIGMA,
    max_shift: int = DEFAULT_MAX_SHIFT,
    scale_range: Tuple[float, float] = DEFAULT_SCALE_RANGE,
    p: float = DEFAULT_P,
    rng: Optional[np.random.Generator] = None,
    dropout_fill: str = "zero",
) -> np.ndarray:
    """Vectorized equivalent of augment_eeg for a whole (n_trials, channels, time)
    array. Each trial gets independent draws for every transform, and each
    transform is independently gated by `p` per trial (so trials end up with
    varying degrees of perturbation, rather than always receiving the full
    stack — matches the torch version's behaviour).

    fs: sampling rate in Hz (needed for the bandstop filter).
    dropout_fill: "zero" (torch-equivalent) or "noise", see _channel_dropout_trial.
    """
    rng = rng if rng is not None else np.random.default_rng()
    X = X.copy()
    n = len(X)

    # --- scale (applied before noise, so noise magnitude stays ~sigma) ---
    scale_mask = rng.random(n) < p
    scale = np.where(scale_mask, rng.uniform(*scale_range, size=n), 1.0).astype(X.dtype)
    X *= scale[:, None, None]

    # --- additive noise ---
    noise_mask = (rng.random(n) < p).astype(X.dtype)
    noise = rng.normal(0.0, sigma, size=X.shape).astype(X.dtype)
    X += noise * noise_mask[:, None, None]

    # --- safe (non-wrapping) time shift ---
    shift_mask = rng.random(n) < p
    shifts = rng.integers(-max_shift, max_shift + 1, size=n)
    X = np.stack([
        _safe_time_shift_trial(x, int(s)) if m else x
        for x, s, m in zip(X, shifts, shift_mask)
    ])

    # --- channel dropout ---
    drop_mask = rng.random(n) < p
    X = np.stack([
        _channel_dropout_trial(x, rng, fill=dropout_fill) if m else x
        for x, m in zip(X, drop_mask)
    ])

    # --- bandstop filter ---
    bs_mask = rng.random(n) < p
    X = np.stack([
        _bandstop_filter_trial(x, fs, rng) if m else x
        for x, m in zip(X, bs_mask)
    ])

    return X


def augment_epochs(
    epochs,
    n_copies: int = 1,
    sigma: float = DEFAULT_SIGMA,
    max_shift: int = DEFAULT_MAX_SHIFT,
    scale_range: Tuple[float, float] = DEFAULT_SCALE_RANGE,
    p: float = DEFAULT_P,
    seed: Optional[int] = None,
) -> list:
    """Offline augmentation for estimators that fit a static feature matrix
    rather than iterating over epochs (e.g. a random forest over epoch
    features): returns `n_copies` extra, perturbed copies of `epochs` as new
    Epochs objects, ready to be appended to the original training epochs with
    mne.concatenate_epochs (labels are unchanged by these perturbations).

    Dropped channels are filled with noise rather than zeros here, since feature
    extractors that take logs/ratios of per-channel power break on flat channels.
    """
    import mne  # local import so this module doesn't hard-require mne for the torch/np paths

    rng = np.random.default_rng(seed)
    data = epochs.get_data()  # (n_trials, n_channels, n_times)
    fs = epochs.info["sfreq"]
    meta = epochs.metadata.copy() if epochs.metadata is not None else None
    n = len(data)
    # epochs.events[:, 0] (absolute sample numbers) needn't be sorted here -- X_fit is
    # typically a shuffled subset -- but EpochsArray warns if the *events it's given*
    # aren't chronological, so each copy gets its own synthetic, strictly increasing
    # sample column (event ids/codes are kept) instead of reusing the original one.
    copies = []
    for i in range(n_copies):
        events = epochs.events.copy()
        events[:, 0] = np.arange(n) + (i + 1) * n
        aug_data = augment_array(data, fs, sigma, max_shift, scale_range, p, rng, dropout_fill="noise")
        aug = mne.EpochsArray(aug_data, epochs.info, events=events, tmin=epochs.tmin,
                              event_id=epochs.event_id, raw_sfreq=epochs._raw_sfreq,
                              metadata=meta, baseline=None, verbose=False)
        # mne.concatenate_epochs requires matching baselines. Passing baseline=epochs.baseline
        # to EpochsArray would re-apply the correction to already-corrected (and z-scored)
        # data, so only copy the label over -- the data is already in the same space.
        aug.baseline = epochs.baseline
        copies.append(aug)
    return copies