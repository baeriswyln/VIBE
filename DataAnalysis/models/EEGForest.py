"""Handcrafted features + label-dependent spatial-filter features -> random forest,
for DSI-24 dry-EEG epochs (MI, movement, SPN, ErrP, pinch). EEG channels only; the
AUX inputs X1-X3 are ignored even if present. All windows are in seconds relative to
each epoch's own reference (time-lock) event.

Usage, mirroring models/EEGNet.py:

    from models.EEGForest import EEGForest, train_eegforest
    result = train_eegforest(X_train, y_train, X_test, y_test, run_name="eegforest-general")
    result["model"], result["test_accuracy"], result["confusion_matrix"], ...

    model = EEGForest.load("eegforest.joblib")
    model.predict(new_epochs)

Handcrafted features alone can also be computed directly, e.g. for exploration:

    X = pd.concat([bandpower_features(epochs), mrcp_features(epochs)], axis=1)

Many splits of the same data (e.g. leave-one-subject-out): compute every label-free
feature once into a FeatureStore, then train on any subset of it -- only the supervised
spatial filters and the forest are refit per split. The parent dataset needs a `row_id`
metadata column (0..n-1) that its subsets carry along:

    parent.metadata = parent.metadata.assign(row_id=np.arange(len(parent)))
    store = build_feature_store(parent)
    for train, test in splits_of(subset_of(parent)):
        train_eegforest(train, None, test, None, store=store)

Cross-validate on a single set of epochs (label-free features computed once,
supervised spatial filters refit inside every fold):

    from models.EEGForest import cross_validate_eegforest
    res = cross_validate_eegforest(epochs)
    res["group_importance"]   # which feature families matter

Event-position handling
------------------------
Two families of Epochs objects are used against this module:

  - "event-locked-at-t=0" epochs (e.g. data/epoch_dataset_filtered): every epoch's own
    time-lock event sits at exactly t=0, tmin is negative (the classic MNE convention).
    `epochs.metadata` has no `t_offset` column.
  - "robust"/jittered epochs (data/epoch_dataset_robust): every epoch spans a fixed
    [0, epoch_duration] time axis, but the actual event sits at a per-epoch, per-row
    offset given by `epochs.metadata['t_offset']` (NaN for epochs with no real event,
    e.g. quiet-baseline/finger-movement negatives).

Every window default in this module is therefore resolved *per epoch*, anchored at
that epoch's own reference time (`_ref_times`), rather than assuming a single,
batch-wide t=0. For event-locked-at-t=0 epochs this reference time is 0.0 for every
row, which makes every window computation here reduce identically to the original,
pre-jitter behaviour (see the module's test coverage / EEGForest.py git history for
the explicit old-vs-new equivalence check on data/epoch_dataset_filtered).
"""
import warnings
from contextlib import contextmanager
from functools import partial

import joblib
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import eigh
from scipy.signal import welch
from threadpoolctl import threadpool_limits
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight

from typing import Optional, Tuple

from models.augment import augment_epochs, DEFAULT_MAX_SHIFT, DEFAULT_SCALE_RANGE, DEFAULT_SIGMA
from models.wandb import *

BANDS = {"theta": (4, 8), "mu": (8, 13), "beta_low": (13, 20), "beta_high": (20, 30)}
MOTOR = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]
ERRP = ["Fz", "F3", "F4", "Cz", "C3", "C4", "Pz"]
SPN = ["F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8"]
SPN_PAIRS = [("F4", "F3"), ("F8", "F7"), ("C4", "C3"), ("T8", "T7"), ("P4", "P3"), ("P8", "P7")]
# Large Laplacian on the 10-20 grid (McFarland et al. 1997); reference-free, so it also
# removes the influence of the DSI-24's Pz reference on C3/C4.
LAPLACIAN = {"C3": ["F3", "T7", "P3", "Cz"], "C4": ["F4", "T8", "P4", "Cz"], "Cz": ["Fz", "C3", "C4", "Pz"]}


AUX = ("X1", "X2", "X3")   # DSI-24 auxiliary inputs - not used

CHUNK_SIZE = 1000
"""Epochs processed at once by the per-epoch work (handcrafted features, filtering +
cropping, supervised-pipeline transforms). Every one of those steps is independent per
epoch, so chunking gives bit-identical results while bounding the float64 temporaries
(filtered copies, Morlet TFR power, ...) to CHUNK_SIZE epochs instead of the whole batch --
which is what made the ~27k-epoch robust dataset run out of memory. Lower it if memory is
still tight; raise it for slightly less per-chunk overhead."""


# ----------------------------------------------------------------------------- helpers
def _chunks(n, size=None):
    size = size or CHUNK_SIZE
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def _epoch_chunks(epochs):
    """(slice, epochs-subset) pairs covering `epochs` in CHUNK_SIZE pieces (no copy if it fits in one)."""
    n = len(epochs)
    if n <= CHUNK_SIZE:
        return [(slice(0, n), epochs)]
    return [(sl, epochs[sl]) for sl in _chunks(n)]


def eeg_channels(epochs):
    return [epochs.ch_names[i] for i in mne.pick_types(epochs.info, eeg=True) if epochs.ch_names[i] not in AUX]


def eeg_input(epochs):
    """(n_epochs, n_channels, n_times) EEG in µV, input for the supervised pipelines."""
    X = epochs.get_data(picks=eeg_channels(epochs))
    X *= 1e6   # in place: get_data(picks=...) already returned a fresh copy
    return X


def _data(epochs, chs=None, lap=False):
    """EEG in µV for the available channels in `chs` (+ Laplacian derivations)."""
    avail = eeg_channels(epochs)
    names = [c for c in (chs or avail) if c in avail]
    laps = {c: nb for c, nb in LAPLACIAN.items() if lap and c in avail and all(n in avail for n in nb)}
    picks = list(dict.fromkeys(names + [n for c, nb in laps.items() for n in (c, *nb)]))
    X = epochs.get_data(picks=picks)
    X *= 1e6
    idx = {c: i for i, c in enumerate(picks)}
    if picks == names:
        return X, names
    out = np.empty((X.shape[0], len(names) + len(laps), X.shape[2]), dtype=X.dtype)
    out[:, :len(names)] = X[:, [idx[c] for c in names]]
    for j, (c, nb) in enumerate(laps.items(), len(names)):
        out[:, j] = X[:, idx[c]] - np.mean(X[:, [idx[n] for n in nb]], axis=1)
    return out, names + [f"{c}lap" for c in laps]


# ----------------------------------------------------------------------- per-epoch windowing
def _ref_times(epochs) -> np.ndarray:
    """Per-epoch reference (time-lock event) time, in seconds, on that epoch's own time
    axis -- the anchor every window below is defined relative to.

    - If `epochs.metadata` has a `t_offset` column: that row's `t_offset` (the real
      event's position within its epoch), or -- for rows with no real event (NaN
      `t_offset`, e.g. quiet-baseline/finger-movement negatives) -- the epoch's own
      midpoint, which gives the most generous, symmetric margin available on either
      side.
    - Otherwise (no `t_offset` column at all -- classic event-locked-at-t=0 epochs):
      0.0 for every row, exactly reproducing the pre-jitter behaviour.
    """
    n = len(epochs)
    md = epochs.metadata
    if md is not None and "t_offset" in md.columns:
        t_off = md["t_offset"].to_numpy(dtype=float)
        mid = (epochs.times[0] + epochs.times[-1]) / 2.0
        return np.where(np.isnan(t_off), mid, t_off)
    return np.zeros(n, dtype=float)


def _is_legacy(epochs) -> bool:
    """True for event-locked-at-t=0 epochs (no `t_offset` metadata column) -- i.e.
    every row's reference time is 0.0 and `epochs.times` itself is already directly
    ref-relative, which is what lets `_resolve_window` reproduce the old boolean-mask
    behaviour bit-for-bit (see its docstring)."""
    return epochs.metadata is None or "t_offset" not in epochs.metadata.columns


MIN_MARGIN_S = 0.5
"""Floor on the batch-level pre-/post-reference margin `_margins` returns, in seconds.

A handful of epochs (chiefly Step-3 "dense pool" windows on the robust dataset, which --
unlike Step-1's jittered pinch epochs -- has no guaranteed minimum distance between a
contained pinch and its window's own edge) can have a genuinely tiny true margin. Without
a floor, a single such outlier epoch in a batch would force *every* window default in
that batch down to a pathological, sub-10-sample duration -- which doesn't crash inside
this module, but is numerically unstable for the covariance-based supervised pipelines
(singular/non-positive-definite matrices in pyriemann). Epochs whose own true margin is
below the floor simply get their window clamped to the nearest in-bounds position instead
of shrunk (see `_resolve_window`), which can mean that specific window ends up sitting
closer to (or briefly past) that one epoch's own event than its nominal definition
intends -- an accepted trade-off for a handful of outlier trials, not a crash. This never
affects event-locked-at-t=0 epochs (old dataset margins are ~1.7s/~0.3s, both well above
the floor)."""


def _margins(epochs):
    """(pre, post): the smallest pre-reference and post-reference margin, in seconds,
    across every epoch currently in `epochs` -- the tightest constraint in this
    particular batch, which every window default below sizes itself to fit inside
    (floored at MIN_MARGIN_S on non-legacy data -- see its docstring).

    For event-locked-at-t=0 (`legacy`) epochs (reference time 0.0 for every row) this
    is exactly `(-epochs.times[0], epochs.times[-1])` for every batch -- *not* floored
    -- matching the old, dataset-wide `(t0, t1) = _span(epochs)` convention this
    replaces (`pre == -t0`, `post == t1`) exactly, so every old `t0`-based window
    default below is written as `-pre`, and every old `t1`-based default as `post`,
    which is an exact (not approximate) substitution. (The old dataset's own margins,
    ~1.7s/~0.3s, are comfortably above MIN_MARGIN_S regardless -- the floor is skipped
    for legacy data on principle, not because it would happen not to bite.)
    """
    if _FROZEN_MARGINS is not None:
        return _FROZEN_MARGINS
    ref = _ref_times(epochs)
    pre = float((ref - epochs.times[0]).min())
    post = float((epochs.times[-1] - ref).min())
    if _is_legacy(epochs):
        return pre, post
    return max(pre, MIN_MARGIN_S), max(post, MIN_MARGIN_S)


_FROZEN_MARGINS = None


@contextmanager
def _frozen_margins(margins):
    """Make `_margins` return `margins` for every call inside the block. Used to compute
    features chunk by chunk with the window defaults of the *whole* batch -- otherwise
    each chunk would size its windows to its own tightest epoch, and chunked features
    would differ from unchunked ones."""
    global _FROZEN_MARGINS
    prev, _FROZEN_MARGINS = _FROZEN_MARGINS, margins
    try:
        yield
    finally:
        _FROZEN_MARGINS = prev


def _resolve_window(times, ref, win, legacy):
    """(start_idx, n_samples) for a window `win = (rel_start, rel_end)` relative to
    each row's reference time `ref` (array).

    `start` is always found via `searchsorted` against the *actual* `times` grid
    (the first index with `times >= ref[i] + win[0]`) -- a real absolute time value
    in a real absolute time array, regardless of `ref`.

    `n_samples` needs care: on `legacy` (event-locked-at-t=0) data, `ref` is 0 for
    every row, so `times` itself *is* already win-relative, and counting
    `(times >= win[0]) & (times <= win[1])` directly against it reproduces the old
    boolean-mask sample count exactly -- including its small grid-fuzz-driven +-1
    quirks near a boundary, which an independent `round(duration * sfreq)`
    reconstruction can disagree with. On non-legacy (jittered) data this same
    comparison is meaningless (`times` is an absolute [0, epoch_duration] axis, `win`
    is ref-relative and can be negative, e.g. a pre-event baseline) -- e.g. it can
    silently collapse to ~0 samples -- so `n_samples` there is instead the ordinary
    `round(duration * sfreq) + 1` sample count."""
    n_times = len(times)
    if legacy:
        n_samples = max(1, int(((times >= win[0]) & (times <= win[1])).sum()))
    else:
        sfreq_est = (n_times - 1) / (times[-1] - times[0]) if n_times > 1 and times[-1] != times[0] else 300.0
        n_samples = max(1, min(n_times, int(round((win[1] - win[0]) * sfreq_est)) + 1))
    start = np.searchsorted(times, ref + win[0], side="left")
    start = np.clip(start, 0, n_times - n_samples)
    return start, n_samples


def _win(epochs, win):
    """Per-epoch `(start_idx, n_samples)` for a window `win = (rel_start, rel_end)`
    given in seconds relative to each epoch's own reference time (`_ref_times`).
    `n_samples` is the same for every epoch (fixed window duration -- required so
    handcrafted features stack into a normal 2-D matrix and CSP/Riemann/xDAWN see a
    consistent trial length); `start_idx` varies per epoch, clipped to stay within
    that epoch's own valid sample range (never raises -- a window that doesn't fit
    degrades to the closest in-range window rather than crashing)."""
    return _resolve_window(epochs.times, _ref_times(epochs), win, _is_legacy(epochs))


def _take(X, start_idx, n_samples):
    """Gather the fixed-length, per-epoch-positioned window described by
    `(start_idx, n_samples)` (see `_win`) out of `X` (n_epochs, n_channels, n_times).
    Returns (n_epochs, n_channels, n_samples)."""
    idx = np.clip(start_idx[:, None] + np.arange(n_samples)[None, :], 0, X.shape[-1] - 1)
    return np.take_along_axis(X, idx[:, None, :], axis=-1)


def _filt(X, sfreq, lo, hi):
    return mne.filter.filter_data(X, sfreq, lo, hi, method="iir", verbose=False)


def _power(X, sfreq, bands):
    n = min(X.shape[-1], int(sfreq))                      # 1 s Welch segments
    f, P = welch(X, sfreq, nperseg=n, noverlap=n // 2, axis=-1)
    return {b: P[..., (f >= lo) & (f < hi)].mean(-1) for b, (lo, hi) in bands.items()}


def _df(arr, prefix, names):
    return pd.DataFrame(arr, columns=[f"{prefix}_{c}" for c in names])


def _win_means(epochs, X, names, windows, prefix):
    return pd.concat([_df(_take(X, *_win(epochs, w)).mean(-1), f"{prefix}_{w[0]:+.2f}_{w[1]:+.2f}", names)
                      for w in windows], axis=1)


def _slope(epochs, X, win):
    """Least-squares slope, µV/s, of `X` over `win` (per-epoch-anchored, see `_win`).
    Translation-invariant (the fit always uses a zero-mean time vector), so this needs
    no special-casing for where the window actually sits within each epoch."""
    start_idx, n_samples = _win(epochs, win)
    Xw = _take(X, start_idx, n_samples)
    t = np.arange(n_samples) / epochs.info["sfreq"]
    t = t - t.mean()
    return (Xw * t).sum(-1) / (t ** 2).sum()


def _slow(epochs, chs, baseline, lo, hi, lap=False):
    X, names = _data(epochs, chs, lap)
    X = _filt(X, epochs.info["sfreq"], lo, hi)
    base = _take(X, *_win(epochs, baseline)).mean(-1, keepdims=True)
    return X - base, names


# ----------------------------------------------------------------------------- features
def _near_event_window(pre, post, dur, gap=0.1):
    """A `dur`-second window anchored to the reference time (0): placed just after it
    when there's enough post-reference span, otherwise just before it -- e.g. epochs
    that end shortly after the event (like the old pinch data), which only leaves
    margin before the event. `dur` is clamped to whatever's actually available on the
    chosen side, so this degrades gracefully instead of returning an inverted/empty
    window on tight-margin batches."""
    if post - gap >= dur:
        return gap, gap + dur
    avail = max(0.0, min(dur, pre - gap))
    return -gap - avail, -gap


def bandpower_features(epochs, active=None, baseline=None, bands=BANDS, chs=MOTOR, pmbr=None):
    """MI / movement / pinch: log power, ERD/ERS in dB (Pfurtscheller & Lopes da Silva 1999),
    C3/C4 lateralisation and optional post-movement beta rebound (Pfurtscheller et al. 1996)."""
    pre, post = _margins(epochs)
    baseline = baseline or (-pre, -pre + min(0.3, pre))
    active = active or _near_event_window(pre, post, dur=0.5)
    X, names = _data(epochs, chs, lap=True)
    sf = epochs.info["sfreq"]
    act = _power(_take(X, *_win(epochs, active)), sf, bands)
    base = _power(_take(X, *_win(epochs, baseline)), sf, bands)
    dfs = []
    for b in bands:
        dfs += [_df(np.log10(act[b]), f"logpow_{b}", names),
                _df(10 * np.log10(act[b] / base[b]), f"erd_db_{b}", names)]
        for l, r in [("C3lap", "C4lap"), ("C3", "C4")]:       # >0: stronger left-hemisphere ERD
            if l in names and r in names:
                pl, pr = act[b][:, names.index(l)], act[b][:, names.index(r)]
                dfs.append(pd.DataFrame({f"LI_{b}_{l}_{r}": (pr - pl) / (pr + pl)}))
    if pmbr is not None:
        beta = {"beta": (13, 30)}
        reb = _power(_take(X, *_win(epochs, pmbr)), sf, beta)["beta"]
        base_beta = _power(_take(X, *_win(epochs, baseline)), sf, beta)["beta"]
        dfs.append(_df(10 * np.log10(reb / base_beta), "pmbr_db", names))
    return pd.concat(dfs, axis=1)


def hjorth_features(epochs, window=None, chs=MOTOR):
    """Hjorth activity/mobility/complexity of 8-30 Hz activity (Vidaurre et al. 2009)."""
    window = window or _near_event_window(*_margins(epochs), dur=0.5)
    X, names = _data(epochs, chs, lap=True)
    X = _filt(X, epochs.info["sfreq"], 8, 30)
    X = _take(X, *_win(epochs, window))
    v0, v1, v2 = X.var(-1), np.diff(X).var(-1), np.diff(X, 2).var(-1)
    mob = np.sqrt(v1 / v0)
    return pd.concat([_df(np.log10(v0), "hjorth_act", names), _df(mob, "hjorth_mob", names),
                      _df(np.sqrt(v2 / v1) / mob, "hjorth_comp", names)], axis=1)


def mrcp_features(epochs, baseline=None, lowpass=3.0, chs=MOTOR, windows=None, slopes=None, peak=None):
    """Movement-related cortical potentials, epochs locked to movement onset
    (Shibasaki & Hallett 2006; Niazi et al. 2011; Jochumsen et al. 2013, 2016)."""
    pre, post = _margins(epochs)
    baseline = baseline or (-pre, -pre + min(0.3, pre))
    if windows is None:
        # quarter the post-baseline, pre-event span into 4 windows, halve the post-event
        # span into 2 -- 6 windows total, tracing early/late CNV components up to the event.
        # (On a tight-margin batch this quartering can collapse toward zero width -- an
        # expected consequence of a much smaller guaranteed pre-event budget than the old
        # dataset's ~1.7s, not a bug; it degrades to fewer informative windows rather than
        # crashing.)
        pre_start = baseline[1]
        pre_q = -pre_start / 4
        bounds = [pre_start + i * pre_q for i in range(5)] + [post / 2, post]
        windows = list(zip(bounds[:-1], bounds[1:]))
    if slopes is None:
        slopes = ((windows[0][0], windows[1][1]), (windows[2][0], windows[3][1]))
    if peak is None:
        peak = (windows[3][0], post)
    X, names = _slow(epochs, chs, baseline, None, lowpass, lap=True)
    start_idx, n_samples = _win(epochs, peak)
    Xpeak = _take(X, start_idx, n_samples)
    sfreq = epochs.info["sfreq"]
    argmin = Xpeak.argmin(-1)
    dfs = [_win_means(epochs, X, names, windows, "mrcp_mean")]
    dfs += [_df(_slope(epochs, X, w), f"mrcp_slope_{w[0]:+.2f}_{w[1]:+.2f}", names) for w in slopes]
    dfs += [_df(Xpeak.min(-1), "mrcp_peak_amp", names),
            _df(peak[0] + argmin / sfreq, "mrcp_peak_lat", names)]
    return pd.concat(dfs, axis=1)


def lowfreq_features(epochs, window=None, band=(0.3, 3.0), rate=10.0):
    """Decimated 0.3-3 Hz amplitudes of all channels for grasp/pinch decoding
    (Ofner et al. 2017; Schwarz et al. 2018)."""
    pre, post = _margins(epochs)
    if window is None:
        skip = min(0.7, pre)  # skip the baseline/filter run-in, keep the rest of the trial
        window = (-pre + skip, post)
    X, names = _data(epochs)
    X = _filt(X, epochs.info["sfreq"], *band)
    start_idx, n_samples = _win(epochs, window)
    step = max(1, int(round(epochs.info["sfreq"] / rate)))
    offsets = np.arange(0, n_samples, step)
    rel_times = window[0] + offsets / epochs.info["sfreq"]   # relative to each epoch's own reference time
    cols = [f"lf_{c}_{rt:+.2f}" for c in names for rt in rel_times]
    idx = np.clip(start_idx[:, None] + offsets[None, :], 0, X.shape[-1] - 1)
    Xd = np.take_along_axis(X, idx[:, None, :], axis=-1)
    return pd.DataFrame(Xd.reshape(len(X), -1), columns=cols)


def spn_features(epochs, baseline=None, lowpass=5.0, chs=SPN, windows=None):
    """Stimulus-preceding negativity, t=0 = anticipated stimulus: amplitude, build-up slope and
    right-left asymmetry (Brunia & van Boxtel 2001; Böcker et al. 1994)."""
    pre, post = _margins(epochs)
    baseline = baseline or (-pre, -pre + min(0.2, pre))
    if windows is None:
        if pre >= 1.0:
            # unchanged literal default -- reproduces the original behaviour exactly
            # whenever the batch's pre-event budget actually supports it.
            windows = ((-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.0))
        else:
            # tight-margin fallback: evenly split whatever pre-event budget remains
            # after the baseline into 3 windows, same principle as mrcp's quartering.
            pre_start = baseline[1]
            third = -pre_start / 3
            bounds = [pre_start + i * third for i in range(4)]
            windows = list(zip(bounds[:-1], bounds[1:]))
    X, names = _slow(epochs, chs, baseline, None, lowpass)
    late = _take(X, *_win(epochs, windows[-1])).mean(-1)
    asym = {f"spn_asym_{r}-{l}": late[:, names.index(r)] - late[:, names.index(l)]
            for r, l in SPN_PAIRS if r in names and l in names}
    return pd.concat([_win_means(epochs, X, names, windows, "spn_mean"),
                      _df(_slope(epochs, X, (windows[0][0], windows[-1][1])), "spn_slope", names),
                      pd.DataFrame(asym)], axis=1)


def errp_features(epochs, baseline=None, band=(1.0, 10.0), chs=ERRP, start=0.05, stop=None,
                  step=0.05, theta_win=None, theta_base=(-0.3, -0.05)):
    """ErrP, t=0 = feedback: 50 ms windowed means (Chavarriaga et al. 2014), N2/P3 peaks at Fz/Cz
    (Ferrez & Millán 2008) and frontal-midline theta (Trujillo & Allen 2007; Cavanagh & Frank 2014)."""
    pre, post = _margins(epochs)
    baseline = baseline or (max(-pre, -0.2), 0.0)
    stop = post if stop is None else stop
    theta_win = theta_win or (start, post)
    X, names = _slow(epochs, chs, baseline, *band)
    windows = [(s, s + step) for s in np.arange(start, stop - 1e-9, step)]
    dfs = [_win_means(epochs, X, names, windows, "errp_mean")]
    peaks = {}
    # N2/P3 latencies are fixed literals (post-event); clamp their end to the batch's
    # own available post-margin so a short post-reference budget (old dataset: 0.3s)
    # truncates them exactly like the old `_mask` naturally did, rather than "fitting"
    # the full nominal duration by sliding the window earlier (which would shift it off
    # its intended, literature-anchored latency).
    n2_win = (0.18, min(0.32, post))
    p3_win = (0.25, min(0.45, post))
    for c in ("Fz", "Cz"):
        if c in names:
            n2 = _take(X[:, [names.index(c)]], *_win(epochs, n2_win)).min(-1)[:, 0]
            p3 = _take(X[:, [names.index(c)]], *_win(epochs, p3_win)).max(-1)[:, 0]
            peaks.update({f"errp_N2_{c}": n2, f"errp_P3_{c}": p3, f"errp_P3-N2_{c}": p3 - n2})
    freqs = np.arange(4.0, 9.0)
    Xt, tn = _data(epochs, ["Fz", "Cz", "F3", "F4"])
    P = mne.time_frequency.tfr_array_morlet(Xt, epochs.info["sfreq"], freqs, n_cycles=freqs / 2,
                                            output="power", verbose=False).mean(2)
    theta_act = _take(P, *_win(epochs, theta_win)).mean(-1)
    theta_bas = _take(P, *_win(epochs, theta_base)).mean(-1)
    theta = 10 * np.log10(theta_act / theta_bas)
    return pd.concat(dfs + [pd.DataFrame(peaks), _df(theta, "errp_theta_db", tn)], axis=1)




# ----------------------------------------------------------------------------- supervised
# FBCSP (MI/movement; Ang et al. 2008, shrinkage as in Lotte & Guan 2011), Riemannian
# tangent space (MI; Barachant et al. 2012) and xDAWN + tangent space (ErrP/SPN). These
# learn from labels, so they're only ever fit on training epochs. Everything label-free
# they need (band-pass filtering, cropping, per-epoch covariances) is computed up front
# into a FeatureStore; only the cheap label-dependent part is fit per training set.
FBCSP_BANDS = tuple((f, f + 4) for f in range(4, 40, 4))
RIEMANN_BAND = (8, 30)
XDAWN_BAND = (1, 10)
SUPERVISED = ("fbcsp", "riemann", "xdawn")

FEATURE_STORE_VERSION = 1
"""Bump whenever anything that ends up in a FeatureStore changes (feature functions,
bands, windows, covariance estimators, ...), so on-disk stores keyed on it are rebuilt."""


def _epoch_context(epochs):
    """(ref_times, times, legacy) describing a batch of epochs: what `_resolve_window`
    needs to crop a plain (n_epochs, n_channels, n_times) array, which on its own carries
    no per-row timing. `times` is `epochs.times` itself (not just its endpoints), so
    cropping uses the exact same `searchsorted`-against-the-real-grid logic as every
    handcrafted feature above -- see `_resolve_window`'s docstring for why that (and
    `legacy`) matter for exact old-data parity."""
    return _ref_times(epochs), epochs.times, _is_legacy(epochs)


def _bandpass_crop(X, sfreq, band, win, ref_times, times, legacy):
    """Band-pass filter, then crop to the fixed-length window `win` (seconds relative
    to each epoch's own reference time). Filtering happens before cropping, on the full
    epoch, so the window doesn't include the filter's edge transient."""
    start, n_samples = _resolve_window(times, ref_times, win, legacy)
    return _take(_filt(X, sfreq, *band), start, n_samples)


def _scm(X):
    """Per-epoch non-centred sample covariance X Xᵀ / T, (n, C, T) -> (n, C, C). With it,
    CSP's log-variance feature log(mean((W x)²)) is exactly log(diag(W C Wᵀ)), so CSP can
    be fit and applied on cached covariances instead of the filtered time series."""
    return np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]


def _band_scm(X, sfreq, band, win, ref_times, times, legacy):
    return _scm(_bandpass_crop(X, sfreq, band, win, ref_times, times, legacy))


def supervised_windows(epochs):
    """(osc_win, erp_win) default crop windows for the supervised features, sized to the
    batch's tightest pre-/post-reference margin (see `_margins`), since CSP/covariances/
    xDAWN need a consistent trial length across the batch.

    - osc_win (FBCSP, Riemann): the pre-event span, trimmed a little on both ends (0.2s on
      the old dataset's ~1.7s budget; clamped so it can't eat the whole window on a
      tight-margin batch).
    - erp_win (xDAWN): up to 0.3s pre-reference through the epoch's post-reference end."""
    pre, post = _margins(epochs)
    osc_trim = min(0.2, pre * 0.3)
    return (-(pre - osc_trim), -osc_trim), (-min(0.3, pre), post)


class FeatureStore:
    """Everything label-free EEGForest needs from a set of epochs, computed once:

    hand       handcrafted feature DataFrame (+ `groups`, the feature family of each column)
    fbcsp      (n, n_bands, C, C) per-epoch non-centred covariance per FBCSP band, osc window
    riemann    (n, C, C) per-epoch OAS covariance, RIEMANN_BAND, osc window
    xdawn      (n, C, T) float32 XDAWN_BAND signal, erp window (xDAWN needs the time series
               itself to estimate its class-evoked prototypes)

    Rows can be sliced out with `take` (positional) or `for_epochs` (by the `key` metadata
    column, e.g. a `row_id` assigned to the parent dataset before it was subset), so one
    store built on a parent dataset serves every train/validation/test split of any subset
    of it -- e.g. all folds of all leave-one-subject-out variants.

    Exactness: every feature here is computed per epoch, but the window defaults depend on
    the batch's tightest margin (`_margins`). A store is therefore only equivalent to
    featurizing a subset directly if that subset has the same margins as the parent -- true
    whenever every epoch shares its reference time (event-locked-at-t=0 data, or robust data
    with `t_offset` blanked). `for_epochs` checks this and raises otherwise.
    """

    def __init__(self, hand, groups, ref_times, fbcsp, riemann, xdawn, context, key=None):
        self.hand, self.groups, self.ref_times = hand, list(groups), ref_times
        self.fbcsp, self.riemann, self.xdawn = fbcsp, riemann, xdawn
        self.context, self.key = context, key

    def __len__(self):
        return len(self.hand)

    @property
    def supervised(self):
        return tuple(k for k in SUPERVISED if getattr(self, k) is not None)

    def take(self, idx) -> "FeatureStore":
        idx = np.asarray(idx)
        return FeatureStore(
            self.hand.iloc[idx].reset_index(drop=True), self.groups, self.ref_times[idx],
            *(None if a is None else a[idx] for a in (self.fbcsp, self.riemann, self.xdawn)),
            context=self.context, key=self.key)

    def for_epochs(self, epochs) -> "FeatureStore":
        """The rows of this store matching `epochs` (a subset of the dataset it was built
        from), looked up via the `key` metadata column."""
        if self.key is None or epochs.metadata is None or self.key not in epochs.metadata.columns:
            raise ValueError(f"epochs.metadata needs the store's key column {self.key!r} to look up rows")
        idx = epochs.metadata[self.key].to_numpy()
        ctx = self.context
        if idx.min() < 0 or idx.max() >= len(self):
            raise ValueError(f"{self.key!r} out of range for a store of {len(self)} rows")
        if eeg_channels(epochs) != ctx["ch_names"] or not np.array_equal(epochs.times, ctx["times"]):
            raise ValueError("epochs' channels/time axis differ from the dataset the store was built from")
        if not np.allclose(_ref_times(epochs), self.ref_times[idx]):
            raise ValueError("epochs' reference times differ from the store's (e.g. t_offset was blanked "
                             "when the store was built but isn't now, or vice versa)")
        if _is_legacy(epochs) != ctx["legacy"] or not np.allclose(_margins(epochs), ctx["margins"]):
            raise ValueError(f"epochs' margins {_margins(epochs)} differ from the store's {ctx['margins']}, "
                             "so their window defaults would differ -- build a store from these epochs instead")
        return self.take(idx)

    @staticmethod
    def concat(stores) -> "FeatureStore":
        first = stores[0]
        cat = lambda k: None if getattr(first, k) is None else np.concatenate([getattr(s, k) for s in stores])
        return FeatureStore(pd.concat([s.hand for s in stores], ignore_index=True), first.groups,
                            cat("ref_times"), cat("fbcsp"), cat("riemann"), cat("xdawn"),
                            context=first.context, key=first.key)


def build_feature_store(epochs, supervised=SUPERVISED, feature_kwargs=None, *, osc_win=None,
                        erp_win=None, bands=FBCSP_BANDS, like=None, key="row_id", n_jobs=-1,
                        verbose=False) -> FeatureStore:
    """Compute a FeatureStore for `epochs` (see there), chunk by chunk (CHUNK_SIZE).

    Window defaults come from the margins of `epochs` as a whole -- or, with `like=store`,
    from that store (its margins, windows, bands, parts and feature_kwargs), so e.g. test
    or augmented epochs are featurized exactly like the store a model was trained on.
    `key` names the metadata column `FeatureStore.for_epochs` looks rows up by (None if the
    store is only ever sliced positionally)."""
    if like is not None:
        ctx = like.context
        supervised, feature_kwargs, bands = like.supervised, ctx["feature_kwargs"], ctx["bands"]
        margins, osc_win, erp_win = ctx["margins"], ctx["osc_win"], ctx["erp_win"]
    else:
        margins = _margins(epochs)
    feature_kwargs = feature_kwargs or {}
    unknown = set(supervised) - set(SUPERVISED)
    if unknown:
        raise ValueError(f"unknown supervised feature sets {sorted(unknown)}; choose from {SUPERVISED}")

    with _frozen_margins(margins):
        hand, groups = all_features(epochs, **feature_kwargs)
        d_osc, d_erp = supervised_windows(epochs)
    osc_win, erp_win = tuple(osc_win or d_osc), tuple(erp_win or d_erp)
    sf = epochs.info["sfreq"]
    from pyriemann.estimation import Covariances

    parts = {k: [] for k in supervised}
    chunks = _epoch_chunks(epochs)
    for i, (_, ep) in enumerate(chunks, 1):
        raw = eeg_input(ep)   # shared by all parts -- none of them modify it in place
        ctx = _epoch_context(ep)
        if "fbcsp" in parts:
            # per-band work is independent and releases the GIL -> threads. BLAS is limited to
            # one thread each: letting every band thread call into OpenBLAS with its own
            # full-size pool (9 x 16 threads, across numpy's and scipy's separate OpenBLAS
            # copies) corrupted the heap and crashed with an access violation on the large
            # robust dataset. The thread count doesn't change results.
            with threadpool_limits(limits=1, user_api="blas"):
                per_band = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_band_scm)(raw, sf, b, osc_win, *ctx) for b in bands)
            parts["fbcsp"].append(np.stack(per_band, axis=1))
        if "riemann" in parts:
            parts["riemann"].append(Covariances("oas").transform(
                _bandpass_crop(raw, sf, RIEMANN_BAND, osc_win, *ctx)))
        if "xdawn" in parts:
            parts["xdawn"].append(_bandpass_crop(raw, sf, XDAWN_BAND, erp_win, *ctx).astype(np.float32))
        del raw
        if verbose:
            print(f"\rfeature store: chunk {i}/{len(chunks)}", end="", flush=True)
    if verbose:
        print()

    context = dict(margins=margins, legacy=_is_legacy(epochs), osc_win=osc_win, erp_win=erp_win,
                   bands=tuple(bands), feature_kwargs=feature_kwargs, ch_names=eeg_channels(epochs),
                   times=epochs.times.copy(), sfreq=sf,
                   n_osc_samples=_resolve_window(epochs.times, np.zeros(1), osc_win, _is_legacy(epochs))[1],
                   version=FEATURE_STORE_VERSION)
    arrays = {k: np.concatenate(parts[k]) if k in parts else None for k in SUPERVISED}
    return FeatureStore(hand, groups, _ref_times(epochs), arrays["fbcsp"], arrays["riemann"],
                        arrays["xdawn"], context, key=key)


def _oas(S, n_samples):
    """Oracle-approximating shrinkage (Chen et al. 2010) of sample covariance `S` estimated
    from `n_samples` samples -- same formula as sklearn.covariance.oas, which needs the raw
    data rather than just its covariance."""
    p = S.shape[0]
    mu = np.trace(S) / p
    alpha = np.mean(S ** 2)
    den = (n_samples + 1.0) * (alpha - mu ** 2 / p)
    shrinkage = 1.0 if den == 0 else min((alpha + mu ** 2) / den, 1.0)
    return (1.0 - shrinkage) * S + shrinkage * mu * np.eye(p)


def _fit_csp(covs, y, n_samples, n_filters=4):
    """CSP spatial filters (n_filters, C) from per-epoch covariances. Two classes: generalized
    eigendecomposition of the OAS-shrunk class covariances (the mean of per-epoch covariances
    is the covariance of the class's concatenated epochs), keeping the most discriminative
    eigenvectors (|λ - 0.5| largest). More classes: pyriemann's approximate joint
    diagonalization CSP."""
    classes = np.unique(y)
    if len(classes) == 2:
        C = [_oas(covs[y == c].mean(0), n_samples * np.sum(y == c)) for c in classes]
        evals, evecs = eigh(C[1], C[0] + C[1])
        return evecs[:, np.argsort(np.abs(evals - 0.5))[::-1][:n_filters]].T
    from pyriemann.spatialfilters import CSP
    return CSP(nfilter=n_filters, metric="euclid").fit(covs, y).filters_


def _csp_transform(W, covs):
    """log-variance of the CSP components: log(diag(W C Wᵀ)) per epoch."""
    return np.log(np.einsum("fc,ncd,fd->nf", W, covs, W))


class SupervisedFeatures:
    """The label-dependent features, fit on a FeatureStore's training rows:

    fbcsp    per-band CSP log-variance (4 filters x 9 bands) -> top-k by mutual information
    riemann  tangent-space projection of the 8-30 Hz covariances
    xdawn    xDAWN covariances (2 filters/class) -> tangent space
    """

    def __init__(self, supervised=SUPERVISED, k_fbcsp=8):
        self.supervised, self.k_fbcsp = tuple(supervised), k_fbcsp

    def _fbcsp_all(self, store):
        return np.hstack([_csp_transform(W, store.fbcsp[:, b]) for b, W in enumerate(self.csp_)])

    def fit(self, store, y):
        from pyriemann.estimation import XdawnCovariances
        from pyriemann.tangentspace import TangentSpace
        missing = set(self.supervised) - set(store.supervised)
        if missing:
            raise ValueError(f"feature store lacks {sorted(missing)}")
        if "fbcsp" in self.supervised:
            n = store.context["n_osc_samples"]
            self.csp_ = [_fit_csp(store.fbcsp[:, b], y, n) for b in range(store.fbcsp.shape[1])]
            F = self._fbcsp_all(store)
            # seeded: mutual_info_classif's kNN estimator adds random jitter, which otherwise makes
            # the selected components (and so the whole model) vary from run to run
            self.fbcsp_select_ = SelectKBest(partial(mutual_info_classif, random_state=0),
                                             k=min(self.k_fbcsp, F.shape[1])).fit(F, y)
        if "riemann" in self.supervised:
            self.riemann_ = TangentSpace().fit(store.riemann)
        if "xdawn" in self.supervised:
            self.xdawn_ = make_pipeline(XdawnCovariances(2, estimator="oas", xdawn_estimator="lwf"),
                                        TangentSpace()).fit(store.xdawn.astype(np.float64), y)
        return self

    def transform(self, store) -> dict:
        out = {}
        if "fbcsp" in self.supervised:
            out["fbcsp"] = self.fbcsp_select_.transform(self._fbcsp_all(store))
        if "riemann" in self.supervised:
            out["riemann"] = self.riemann_.transform(store.riemann)
        if "xdawn" in self.supervised:
            # chunked: XdawnCovariances builds a (n, 2*C_proto+C, ...) float64 temporary per epoch
            out["xdawn"] = np.concatenate([self.xdawn_.transform(store.xdawn[sl].astype(np.float64))
                                           for sl in _chunks(len(store))])
        return out


# ----------------------------------------------------------------------------- legacy
# Pre-FeatureStore implementations, kept only so result pickles from before it (which hold
# EEGForest models whose `pipes_` contain these objects) can still be unpickled. Not used by
# anything in this module anymore, and such old models' predict() no longer works.
class BandpassCrop(BaseEstimator, TransformerMixin):
    """Band-pass filter, then crop to a fixed-length window anchored at each epoch's
    own reference time. `win = (rel_start, rel_end)` is in seconds relative to that
    reference time (`win=(None, None)` skips cropping entirely). `ref_times`/`times`/
    `legacy` describe the batch currently being processed and are *not* constructor
    parameters -- they're refreshed externally (see `_set_epoch_context`) right
    before every fit()/transform() call, since they can differ from one call to the
    next (train vs. val vs. test each have their own epochs)."""

    def __init__(self, sfreq=300.0, band=(8, 30), win=(None, None)):
        self.sfreq, self.band, self.win = sfreq, band, win
        self.ref_times = None
        self.times = None
        self.legacy = True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.win[0] is None:
            return _filt(X, self.sfreq, *self.band)
        n_times = X.shape[-1]
        ref = self.ref_times if self.ref_times is not None else np.zeros(X.shape[0])
        times = self.times if self.times is not None else np.arange(n_times) / self.sfreq
        start, n_samples = _resolve_window(times, ref, self.win, self.legacy)
        # filter + crop chunk by chunk, so only the (usually much shorter) cropped window
        # is ever held for the whole batch, never a full-length filtered copy
        out = np.empty(X.shape[:2] + (n_samples,), dtype=np.float64)
        for sl in _chunks(len(X)):
            out[sl] = _take(_filt(X[sl], self.sfreq, *self.band), start[sl], n_samples)
        return out


def _fit_band_csp(X, y, sfreq, band, win, ref_times, times, legacy):
    bc = BandpassCrop(sfreq, band, win)
    bc.ref_times, bc.times, bc.legacy = ref_times, times, legacy
    return mne.decoding.CSP(4, reg="ledoit_wolf", log=True).fit(bc.transform(X), y)


def _transform_band_csp(csp, X, sfreq, band, win, ref_times, times, legacy):
    bc = BandpassCrop(sfreq, band, win)
    bc.ref_times, bc.times, bc.legacy = ref_times, times, legacy
    return csp.transform(bc.transform(X))


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Filter-bank CSP (Ang et al. 2008) with shrinkage (Lotte & Guan 2011). The per-band
    filter+CSP work is independent across bands, so it's parallelized over threads (the
    actual work -- IIR filtering, covariance/eigendecomposition -- is numpy/BLAS/mne C code
    that releases the GIL, so threads parallelize it without the pickling cost of processes).

    While the band threads run, BLAS is limited to one thread each: letting every band thread
    call into OpenBLAS with its own full-size thread pool (9 x 16 threads, across numpy's and
    scipy's separate OpenBLAS copies) corrupted the heap and crashed the process with an
    access violation on the large robust dataset. The thread count doesn't change results.

    `win` and the `ref_times`/`times`/`legacy` context work exactly as in BandpassCrop -- see there."""

    def __init__(self, sfreq=300.0, win=(None, None), bands=tuple((f, f + 4) for f in range(4, 40, 4)),
                 n_jobs=-1):
        self.sfreq, self.win, self.bands, self.n_jobs = sfreq, win, bands, n_jobs
        self.ref_times = None
        self.times = None
        self.legacy = True

    def fit(self, X, y):
        with threadpool_limits(limits=1, user_api="blas"):
            self.csps_ = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_fit_band_csp)(X, y, self.sfreq, b, self.win, self.ref_times, self.times, self.legacy)
                for b in self.bands)
        return self

    def transform(self, X):
        with threadpool_limits(limits=1, user_api="blas"):
            return np.hstack(Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_transform_band_csp)(c, X, self.sfreq, b, self.win, self.ref_times, self.times, self.legacy)
                for c, b in zip(self.csps_, self.bands)))


FEATURES = [bandpower_features, hjorth_features, mrcp_features, lowfreq_features,
            spn_features, errp_features]


def all_features(epochs, **feature_kwargs):
    """All handcrafted features + the feature-group name of every column.

    Computed in CHUNK_SIZE-epoch chunks (see there) with the window defaults of the whole
    batch frozen via `_frozen_margins`, so the result is identical to one unchunked pass."""
    parts = []
    with _frozen_margins(_margins(epochs)):
        for _, ep in _epoch_chunks(epochs):
            dfs = [f(ep, **feature_kwargs.get(f.__name__, {})) for f in FEATURES]
            parts.append(pd.concat(dfs, axis=1))
    groups = [f.__name__.removesuffix("_features") for f, d in zip(FEATURES, dfs) for _ in d.columns]
    return pd.concat(parts, axis=0, ignore_index=True), groups


# ----------------------------------------------------------------------------- labels / splits
def _labels(y, epochs) -> np.ndarray:
    """Accept an explicit label array/Series, an Epochs object to take events from,
    or None to fall back to `epochs.events[:, 2]`."""
    if y is None:
        if isinstance(epochs, FeatureStore):
            raise ValueError("labels are required when passing a FeatureStore")
        return np.asarray(epochs.events[:, 2])
    if hasattr(y, "events"):            # someone passed Epochs as the label source
        y = y.events[:, 2]
    return np.asarray(y).ravel()


def _stratified_split(y: np.ndarray, frac: float, rng: np.random.Generator):
    """Return (train_idx, val_idx) keeping class proportions."""
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = max(1, int(round(frac * len(idx)))) if len(idx) > 1 else 0
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


def _subset(X, idx):
    """Rows `idx` of an Epochs object or a FeatureStore."""
    return X.take(idx) if isinstance(X, FeatureStore) else X[idx]


# ----------------------------------------------------------------------------- model
class EEGForest:
    """Handcrafted spectral/temporal EEG features plus label-dependent spatial-filter
    features (FBCSP, Riemannian tangent space, xDAWN tangent space) feeding a random
    forest. The supervised features are fit only on the data passed to fit().

    fit/predict/predict_proba accept either MNE Epochs or a FeatureStore (precomputed with
    build_feature_store -- much faster when the same epochs are reused across many splits).
    Epochs passed after fit() are featurized with the training data's windows/margins.

    Once fitted (directly, or via train_eegforest()), use like a normal estimator:
        model.predict(new_epochs)
        model.predict_proba(new_epochs)
        model.importances()
        model.save(path) / EEGForest.load(path)
    """

    def __init__(self, supervised=SUPERVISED, feature_kwargs=None,
                 pipeline_kwargs=None, **rf_params):
        self.supervised, self.feature_kwargs = tuple(supervised), feature_kwargs or {}
        self.pipeline_kwargs = pipeline_kwargs or {}   # osc_win / erp_win overrides
        self.rf_params = {"max_features": "sqrt", "min_samples_leaf": 2,
                          "class_weight": "balanced", "n_jobs": -1, "random_state": 0, **rf_params}

    def _store(self, X):
        """FeatureStore for `X`: as is if it already is one, else computed from Epochs --
        like the training store once fitted, from `X`'s own margins otherwise."""
        if isinstance(X, FeatureStore):
            return X
        if hasattr(self, "store_like_"):
            return build_feature_store(X, like=self.store_like_, key=None)
        return build_feature_store(X, self.supervised, self.feature_kwargs, key=None, **self.pipeline_kwargs)

    def _stack(self, store):
        """Feature matrix for the random forest, as float32 (what sklearn's trees cast to
        internally anyway, so predictions are unchanged)."""
        blocks = self.sup_.transform(store)
        names, grp = list(store.hand.columns), list(store.groups)
        Z = [store.hand.to_numpy(dtype=np.float32)]
        for k in self.supervised:
            Z.append(blocks[k].astype(np.float32))
            names += [f"{k}_{i}" for i in range(blocks[k].shape[1])]
            grp += [k] * blocks[k].shape[1]
        return np.hstack(Z), names, grp

    def fit(self, X, y=None, *, validation_data=None,
            n_estimators: int = 500, stage_size: int = 25, verbose: bool = True, wb_run=None):
        """Fit the supervised features + random forest on `X` (Epochs or FeatureStore).

        With `validation_data=(X_val, y_val)` the forest is grown in stages of
        `stage_size` trees, and the stage with the lowest validation loss is kept
        (mirrors train_eegnet's best-checkpoint selection); `self.history_` then
        holds per-stage loss/acc/val_loss/val_acc. Without it, all `n_estimators`
        trees are fit directly (fast path, used by cross-validation) and
        `self.history_` is left empty.
        """
        y = _labels(y, X)
        store = self._store(X)
        # an empty-row stand-in carrying just the store's context, to featurize later Epochs alike
        self.store_like_ = store.take(np.arange(0))
        self.sup_ = SupervisedFeatures(self.supervised).fit(store, y)
        Ztr, self.feature_names_, self.groups_ = self._stack(store)
        del store

        if validation_data is None:
            self.rf_ = RandomForestClassifier(**{**self.rf_params, "n_estimators": n_estimators}).fit(Ztr, y)
            self.history_ = {}
            return self

        X_val, y_val = validation_data
        y_val = _labels(y_val, X_val)
        Zva, _, _ = self._stack(self._store(X_val))

        # tr_loss/tr_acc below are only for the printed/logged training curve -- model
        # selection uses va_loss only -- so a fixed random subsample of the (usually much
        # larger) training set is enough, instead of paying for predict_proba on all of Ztr
        # at every one of the ~n_estimators/stage_size stages.
        metric_rng = np.random.default_rng(0)
        tr_metric_idx = metric_rng.choice(len(Ztr), size=min(2000, len(Ztr)), replace=False)
        Ztr_metric, y_metric = Ztr[tr_metric_idx], y[tr_metric_idx]

        stages = list(range(stage_size, n_estimators, stage_size)) + [n_estimators]
        rf_params = {**self.rf_params, "warm_start": True, "n_estimators": stages[0]}
        # class_weight="balanced"/"balanced_subsample" is unreliable under warm_start (sklearn
        # recomputes it per .fit() call). Every stage here refits on the same (Ztr, y) though, so
        # the class frequencies never change -- compute the weights once and pass them as
        # sample_weight instead, which sidesteps the warning and behaves identically.
        class_weight = rf_params.pop("class_weight", None)
        sample_weight = compute_sample_weight("balanced", y) if class_weight in (
            "balanced", "balanced_subsample") else None
        if class_weight is not None and sample_weight is None:
            rf_params["class_weight"] = class_weight
        rf = RandomForestClassifier(**rf_params)
        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        best_val, best_estimators, best_n = np.inf, None, None

        for i, n in enumerate(stages, 1):
            rf.n_estimators = n
            rf.fit(Ztr, y, sample_weight=sample_weight)
            labels = rf.classes_
            tr_proba, va_proba = rf.predict_proba(Ztr_metric), rf.predict_proba(Zva)
            tr_loss = log_loss(y_metric, tr_proba, labels=labels)
            va_loss = log_loss(y_val, va_proba, labels=labels)
            tr_acc = (labels[tr_proba.argmax(1)] == y_metric).mean()
            va_acc = (labels[va_proba.argmax(1)] == y_val).mean()
            history["loss"].append(tr_loss)
            history["acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)

            if wb_run is not None:
                wb_run.log({"train/loss": tr_loss, "train/acc": tr_acc,
                            "val/loss": va_loss, "val/acc": va_acc}, step=i)
            if verbose:
                print(f"trees {n:4d}/{n_estimators}  loss {tr_loss:.4f}  acc {tr_acc:.3f}"
                      f"  val_loss {va_loss:.4f}  val_acc {va_acc:.3f}")
            if va_loss < best_val:
                # warm_start only ever appends new trees -- earlier trees in estimators_ are
                # never refit -- so a shallow list snapshot is enough to restore this exact
                # checkpoint later; unlike copy.deepcopy(rf), it doesn't copy every tree's
                # internal node arrays on every stage that improves.
                best_val, best_estimators, best_n = va_loss, list(rf.estimators_), rf.n_estimators

        if best_estimators is not None:
            rf.estimators_, rf.n_estimators = best_estimators, best_n
        self.rf_ = rf
        self.history_ = history
        return self

    def predict(self, X):
        return self.rf_.predict(self._stack(self._store(X))[0])

    def predict_proba(self, X):
        return self.rf_.predict_proba(self._stack(self._store(X))[0])

    def importances(self):
        return pd.DataFrame({"importance": self.rf_.feature_importances_, "group": self.groups_},
                            index=self.feature_names_)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "EEGForest":
        return joblib.load(path)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def train_eegforest(X_train, y_train, X_test, y_test, *,
    store: Optional[FeatureStore] = None,
    supervised: tuple = SUPERVISED,
    feature_kwargs: Optional[dict] = None, pipeline_kwargs: Optional[dict] = None,
    n_estimators: int = 400, stage_size: int = 25,
    max_features="sqrt", min_samples_leaf: int = 2, class_weight="balanced", n_jobs: int = -1,
    validation_split: float = 0.2, seed: int = 42, verbose: bool = True,
    augment: bool = False, augment_copies: int = 1, augment_sigma: float = DEFAULT_SIGMA,
    augment_max_shift: int = DEFAULT_MAX_SHIFT, augment_scale_range: Tuple[float, float] = DEFAULT_SCALE_RANGE,
    run_name: Optional[str] = None, wandb_group: Optional[str] = None,
    wandb_tags: Optional[list[str]] = None, log_wandb: bool = True,
    save_as: Optional[str] = None,
):
    """
    Train and test EEGForest (handcrafted features + FBCSP/Riemann/xDAWN + random
    forest) on epoched EEG data. Used the same way as train_eegnet.

    X_train / X_test : MNE Epochs. The handcrafted features need each trial's own
                       timing (windows relative to each epoch's own reference event,
                       see the module docstring), so unlike train_eegnet, plain
                       (n_trials, C, T) arrays are not accepted.
    y_train / y_test : (n_trials,) labels (ints or strings), or None to fall back to
                       X_train.events[:, 2] / X_test.events[:, 2].
    store            : optional FeatureStore built (build_feature_store) on a dataset that
                       X_train and X_test are both subsets of; their features are then
                       looked up (via its key metadata column) instead of recomputed, which
                       makes repeated splits of the same data -- e.g. LOSO folds -- much
                       cheaper. `supervised`/`feature_kwargs`/`pipeline_kwargs` are then
                       taken from the store (`supervised` may pick a subset of its parts).
    augment          : if True, `augment_copies` extra, perturbed copies of the training
                       split (Gaussian noise + time shift + amplitude scale; see
                       models/augment.py) are appended before features/pipelines are fit.
                       Unlike train_eegnet/train_gat_gru_transformer this is offline (the
                       random forest fits a static feature matrix, not epoch-by-epoch), so
                       it multiplies the effective training set size instead of resampling
                       it fresh every epoch. Validation/test data is never augmented.
                       Defaults assume roughly unit-scale (e.g. z-scored) input. The copies
                       are always featurized live (with a store: like the store).

    Returns a dict with the trained model, per-stage history, test accuracy,
    predictions, probabilities and the confusion matrix.
    """
    rng = np.random.default_rng(seed)

    for name, X in (("X_train", X_train), ("X_test", X_test)):
        if not hasattr(X, "get_data") or not hasattr(X, "times"):
            raise ValueError(f"{name} must be an MNE Epochs object (EEGForest features need per-trial timing)")

    ytr_raw, yte_raw = _labels(y_train, X_train), _labels(y_test, X_test)
    if len(X_train) != len(ytr_raw) or len(X_test) != len(yte_raw):
        raise ValueError("Number of epochs and number of labels do not match")

    classes = np.unique(ytr_raw)
    lut = {c: i for i, c in enumerate(classes)}
    if not set(np.unique(yte_raw)).issubset(lut):
        raise ValueError("Test set contains labels not present in the training set")
    ytr = np.array([lut[c] for c in ytr_raw], dtype=np.int64)
    yte = np.array([lut[c] for c in yte_raw], dtype=np.int64)

    # everything below works on either Epochs or (with a store) their FeatureStore rows
    src_train = store.for_epochs(X_train) if store is not None else X_train
    src_test = store.for_epochs(X_test) if store is not None else X_test

    if validation_split and validation_split > 0:
        fit_idx, val_idx = _stratified_split(ytr, validation_split, rng)
        validation_data = (_subset(src_train, val_idx), ytr[val_idx])
    else:
        fit_idx, validation_data = np.arange(len(X_train)), None
    X_fit, y_fit = _subset(src_train, fit_idx), ytr[fit_idx]
    n_fit = len(fit_idx)
    if augment and augment_copies > 0:
        copies = augment_epochs(X_train[fit_idx], n_copies=augment_copies, sigma=augment_sigma,
                                max_shift=augment_max_shift, scale_range=augment_scale_range, seed=seed)
        with warnings.catch_warnings():
            # X_train[fit_idx] (a stratified-split subset) is already sample-shuffled, so its
            # own event order isn't chronological -- harmless here since these are synthetic
            # sample numbers anyway, not real recording offsets.
            warnings.filterwarnings("ignore", message=".*not chronologically ordered.*")
            aug = mne.concatenate_epochs(copies, verbose=False) if len(copies) > 1 else copies[0]
            if store is not None:
                X_fit = FeatureStore.concat([X_fit, build_feature_store(aug, like=store, key=None)])
            else:
                X_fit = mne.concatenate_epochs([X_fit, aug], verbose=False)
        y_fit = np.concatenate([y_fit] * (1 + augment_copies))

    if store is not None:
        missing = set(supervised) - set(store.supervised)
        if missing:
            raise ValueError(f"store was built without {sorted(missing)}")
        feature_kwargs, pipeline_kwargs = store.context["feature_kwargs"], None
    model = EEGForest(
        supervised=supervised, feature_kwargs=feature_kwargs, pipeline_kwargs=pipeline_kwargs,
        max_features=max_features, min_samples_leaf=min_samples_leaf,
        class_weight=class_weight, n_jobs=n_jobs, random_state=seed,
    )

    if verbose:
        n_val = len(X_train) - n_fit
        print(f"EEGForest | supervised={list(supervised)} | {len(classes)} classes | "
              f"{len(y_fit)} train / {n_val} val / {len(X_test)} test epochs | n_estimators={n_estimators}"
              f"{' | cached features' if store is not None else ''}")

    wb_run = wandb_start_run(
        "EEGForest", run_name,
        config=dict(
            supervised=list(supervised), n_estimators=n_estimators, stage_size=stage_size,
            max_features=str(max_features), min_samples_leaf=min_samples_leaf,
            class_weight=str(class_weight), validation_split=validation_split, seed=seed,
            n_classes=len(classes), n_train=len(y_fit),
            n_val=len(X_train) - n_fit, n_test=len(X_test),
            augment=augment, augment_copies=augment_copies if augment else 0,
            augment_sigma=augment_sigma if augment else None,
            augment_max_shift=augment_max_shift if augment else None,
            augment_scale_range=augment_scale_range if augment else None,
        ),
        group=wandb_group, tags=wandb_tags, enabled=log_wandb,
    )

    try:
        model.fit(X_fit, y_fit, validation_data=validation_data, n_estimators=n_estimators,
                  stage_size=stage_size, verbose=verbose, wb_run=wb_run)
        history = model.history_

        proba = model.predict_proba(src_test)
        labels = model.rf_.classes_
        y_pred_i = labels[proba.argmax(1)]
        test_acc = (y_pred_i == yte).mean()
        test_loss = log_loss(yte, proba, labels=labels)
        cm = confusion_matrix(yte, y_pred_i, labels=labels)

        if verbose:
            print(f"\nTest loss {test_loss:.4f} | test accuracy {test_acc:.4f}")
            print("Confusion matrix (rows = true, cols = predicted):")
            print(cm)

        if save_as is not None:
            model.save(save_as)

        wandb_log_test_results(
            wb_run, test_accuracy=test_acc, test_loss=test_loss,
            y_true=yte, y_pred=y_pred_i, classes=classes,
        )
    finally:
        wandb_finish_run(wb_run)

    return {
        "model": model,
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "y_pred": classes[y_pred_i],
        "y_proba": proba,
        "classes": classes,
        "confusion_matrix": cm,
    }


def cross_validate_eegforest(epochs, y=None, n_splits=5, **model_kwargs):
    """Stratified k-fold CV. All label-free work (handcrafted features, filtering,
    covariances) is done once up front; the supervised spatial filters and the forest
    are refit inside every fold.

        res = cross_validate_eegforest(epochs)
        res["group_importance"]   # which feature families matter
    """
    y = _labels(y, epochs)
    store = build_feature_store(epochs, model_kwargs.get("supervised", SUPERVISED),
                                model_kwargs.get("feature_kwargs"), key=None,
                                **(model_kwargs.get("pipeline_kwargs") or {}))
    n_hand = store.hand.shape[1]
    scores, cm, imps = [], 0, []
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=0).split(store.hand, y):
        model = EEGForest(**model_kwargs).fit(store.take(tr), y[tr])
        pred = model.predict(store.take(te))
        scores.append(balanced_accuracy_score(y[te], pred))
        cm = cm + confusion_matrix(y[te], pred, labels=np.unique(y))
        imps.append(model.importances())
    imp = pd.concat(imps).groupby(level=0, sort=False).agg({"importance": "mean", "group": "first"})
    print(f"balanced accuracy {np.mean(scores):.3f} ± {np.std(scores):.3f} "
          f"(chance {1 / len(np.unique(y)):.2f}, {n_hand} handcrafted + {len(imp) - n_hand} supervised features)")
    return {"scores": np.array(scores), "confusion": cm,
            "importances": imp.sort_values("importance", ascending=False),
            "group_importance": imp.groupby("group")["importance"].sum().sort_values(ascending=False)}
