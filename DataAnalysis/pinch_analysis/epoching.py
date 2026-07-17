"""
epoching.py — Turn a session into pinch-centred epochs and matched negatives.

Shared by the EDA, the classifier and the regressor so they all see identical,
consistently preprocessed data.

Key conventions
---------------
* Epochs are built on a copy of the *scalp-only* cleaned Raw.
* The event sample index comes from the real LSL timestamp of each pinch
  (``dataset.times_to_samples``), not from EEG sample counting — robust to the
  small clock jitter.
* Positive events are pinch gesture onsets (PinchLog). Negative windows are
  sampled from inside experiment blocks, at least ``buffer`` seconds from any
  pinch, so a "no pinch here" window never clips the pre-movement build-up.
"""
from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd

from . import dataset as ds
from .preprocess import preprocess, pick_scalp


@dataclass
class PreparedSession:
    session: ds.Session
    raw: mne.io.BaseRaw          # cleaned, scalp-only
    eeg_ts: np.ndarray
    pinch: pd.DataFrame          # merged pinch+select, sorted by time
    blocks: pd.DataFrame
    markers: dict
    report: dict


def prepare_session(session: ds.Session, **preprocess_kwargs) -> PreparedSession:
    raw_full, ts = ds.load_eeg(session)
    clean, report = preprocess(raw_full, **preprocess_kwargs)
    scalp = pick_scalp(clean)
    markers = ds.parse_markers(session)
    pinch = ds.merge_pinch_select(markers["pinch"], markers["select"])
    blocks = ds.experiment_blocks(markers, ts)
    return PreparedSession(session, scalp, ts, pinch, blocks, markers, report)


def _events_from_times(times: np.ndarray, eeg_ts: np.ndarray,
                       codes: np.ndarray | int = 1) -> np.ndarray:
    samp = ds.times_to_samples(times, eeg_ts)
    if np.isscalar(codes):
        codes = np.full(len(samp), int(codes))
    ev = np.column_stack([samp, np.zeros(len(samp), int), np.asarray(codes, int)])
    # MNE needs strictly increasing, unique sample indices
    order = np.argsort(ev[:, 0])
    ev = ev[order]
    keep = np.concatenate([[True], np.diff(ev[:, 0]) > 0])
    return ev[keep]


def pinch_epochs(prep: PreparedSession, tmin: float = -2.0, tmax: float = 1.0,
                 experiments: list[str] | None = None,
                 baseline=None, reject_uv: float | None = 300.0
                 ) -> mne.Epochs:
    """Epochs centred on pinch onsets, carrying rich metadata."""
    pinch = prep.pinch
    if experiments is not None:
        pinch = pinch[pinch["experiment"].isin(experiments)]
    pinch = pinch.reset_index(drop=True)
    ev = _events_from_times(pinch["timestamp"].to_numpy(), prep.eeg_ts, 1)
    # metadata must line up with the (deduplicated, sorted) events
    samp_all = ds.times_to_samples(pinch["timestamp"].to_numpy(), prep.eeg_ts)
    order = np.argsort(samp_all)
    samp_sorted = samp_all[order]
    keep = np.concatenate([[True], np.diff(samp_sorted) > 0])
    meta = pinch.iloc[order[keep]].reset_index(drop=True)

    reject = {"eeg": reject_uv * 1e-6} if reject_uv else None
    epochs = mne.Epochs(prep.raw, ev, event_id={"pinch": 1}, tmin=tmin, tmax=tmax,
                        baseline=baseline, metadata=meta, reject=reject,
                        preload=True, verbose="ERROR", on_missing="ignore")
    return epochs


def sample_negative_times(prep: PreparedSession, n: int, window: float,
                          buffer: float = 2.0, rng=None,
                          experiments: list[str] | None = None) -> np.ndarray:
    """Sample negative-window *center* times inside experiment blocks.

    A candidate center t is accepted if [t-window/2, t+window/2] lies inside a
    block and is >= ``buffer`` s from every pinch onset.
    """
    rng = np.random.default_rng(rng)
    pinch_t = prep.pinch["timestamp"].to_numpy()
    blocks = prep.blocks
    if experiments is not None:
        blocks = blocks[blocks["experiment"].isin(experiments)]
    # exclude tutorial/end blocks that carry no real trials (keep unlabeled
    # None blocks — subject_5 session 1 has no ExperimentName but real pinches)
    blocks = blocks[~blocks["experiment"].isin(["Tutorial", "End", ""])]
    spans = [(r.t_start + window, r.t_end - window)
             for r in blocks.itertuples() if r.t_end - r.t_start > 3 * window]
    if not spans:
        return np.array([])
    weights = np.array([hi - lo for lo, hi in spans], dtype=float)
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()
    out = []
    tries = 0
    half = window / 2
    while len(out) < n and tries < n * 200:
        tries += 1
        k = rng.choice(len(spans), p=weights)
        lo, hi = spans[k]
        t = rng.uniform(lo, hi)
        if np.all(np.abs(pinch_t - t) >= (buffer + half)):
            out.append(t)
    return np.array(out)


def negative_epochs(prep: PreparedSession, neg_times: np.ndarray,
                    tmin: float, tmax: float,
                    reject_uv: float | None = 300.0) -> mne.Epochs | None:
    if len(neg_times) == 0:
        return None
    ev = _events_from_times(neg_times, prep.eeg_ts, 0)
    reject = {"eeg": reject_uv * 1e-6} if reject_uv else None
    return mne.Epochs(prep.raw, ev, event_id={"nopinch": 0}, tmin=tmin, tmax=tmax,
                      baseline=None, reject=reject, preload=True,
                      verbose="ERROR", on_missing="ignore")
