"""
run_hardneg.py — Does the detector really use EEG, or just "eyes on a target"?

A detector trained with *random* negatives can cheat: the pre-pinch window has a
saccade-into-target + fixation, and almost any window far from a pinch does not,
so gaze alone separates them. But in real use the participant looks at many
stimuli they do NOT pinch — every one is a potential false positive.

The experiment surfaces this. We add **hard negatives**: GAZE_ENTER events on
*highlighted-negative* (distractor) stimuli, placed one fixation-dwell later so
the window ends at the same phase as a positive ends at the pinch. These windows
have the identical gaze pattern but no motor pinch.

We train each detector with both random and hard negatives and report, per
leave-one-subject-out fold:
  * AUC(pinch vs random negative)   — the easy, possibly gaze-inflated number
  * AUC(pinch vs hard  negative)    — the honest number (look-vs-look-and-pinch)
  * false-alarm rate on distractor looks at 90 % sensitivity

Feature sets include EEG-only variants (the requested gaze-free run):
  gaze | eeg-bandpower | eeg-riemann | eeg+gaze.

Run:  uv run python -m pinch_analysis.run_hardneg
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from . import dataset as ds
from .epoching import (prepare_session, pinch_epochs, negative_epochs,
                       sample_negative_times)
from .features import bandpower_features
from .run_classify import gaze_window_features, impute

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = ds._ROOT / "out"
FIGS = OUT / "figs"

W = 1.0
NEG_RATIO = 2.0


def build(preps, W):
    """Positives + random negatives + hard (distractor) negatives at window W."""
    cov, bp, gz, y, ntype, grp, tim = [], [], [], [], [], [], []

    def add(prep, ep, times, lab, typ):
        d = ep.get_data(copy=False)
        band = ep.copy().filter(8, 30, verbose="ERROR").get_data(copy=False)
        cov.append(band)
        bp.append(bandpower_features(d * 1e6, ep.info["sfreq"]))
        gz.append(gaze_window_features(prep, times, W))
        y.append(np.full(len(ep), lab))
        ntype.append(np.array([typ] * len(ep), dtype=object))
        grp.append([prep.session.subject] * len(ep))
        tim.append(np.asarray(times, dtype=float))

    for prep in preps:
        pos = pinch_epochs(prep, tmin=-W, tmax=0.0)
        if len(pos) == 0:
            continue
        add(prep, pos, pos.metadata["timestamp"].to_numpy(), 1, "pos")

        neg_t = sample_negative_times(prep, n=int(len(pos) * NEG_RATIO),
                                      window=2 * W, buffer=1.0, rng=7)
        neg = negative_epochs(prep, neg_t, tmin=-W, tmax=0.0)
        if neg is not None and len(neg):
            nt = prep.eeg_ts[np.clip(neg.events[:, 0], 0, len(prep.eeg_ts) - 1)]
            add(prep, neg, nt, 0, "rand")

        hard_t = ds.hard_negative_times(prep.markers, prep.pinch)
        hard = negative_epochs(prep, hard_t, tmin=-W, tmax=0.0)
        if hard is not None and len(hard):
            ht = prep.eeg_ts[np.clip(hard.events[:, 0], 0, len(prep.eeg_ts) - 1)]
            add(prep, hard, ht, 0, "hard")

    return (np.concatenate(cov), np.concatenate(bp), np.concatenate(gz),
            np.concatenate(y), np.concatenate(ntype), np.concatenate(grp),
            np.concatenate(tim))


def fa_at_sens(pos_scores, neg_scores, sens=0.90):
    """False-alarm rate among negatives at a threshold giving `sens` on positives."""
    thr = np.quantile(pos_scores, 1 - sens)
    return float(np.mean(neg_scores >= thr))


def _pipes():
    def flat_pipe():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000))
    def cov_pipe():
        return make_pipeline(Covariances("oas"), TangentSpace(),
                             StandardScaler(), LogisticRegression(max_iter=2000))
    return flat_pipe, cov_pipe


def _feat_specs(cov, bp, gz):
    return {
        "gaze": ("flat", gz, None),
        "eeg-bandpow": ("flat", bp, None),
        "eeg-riemann": ("cov", None, cov),
        "eeg+gaze": ("flat", None, None),  # special: hstack(bp,gz)
    }


def _make(name, kind, idx, cov, bp, gz):
    if name == "eeg+gaze":
        return impute(np.hstack([bp[idx], gz[idx]]))
    if kind == "cov":
        return cov[idx]
    src = gz if name == "gaze" else bp
    return impute(src[idx])


def _auc_by_type(oof, y, ntype):
    pos = (y == 1); rand = (ntype == "rand"); hard = (ntype == "hard")
    ok = ~np.isnan(oof)
    def auc(mneg):
        m = ok & (pos | mneg)
        yy = pos[m].astype(int)
        return roc_auc_score(yy, oof[m]) if len(np.unique(yy)) == 2 else np.nan
    return auc(rand), auc(hard), fa_at_sens(oof[pos & ok], oof[rand & ok]), \
        fa_at_sens(oof[pos & ok], oof[hard & ok])


def evaluate_loso(cov, bp, gz, y, ntype, grp):
    flat_pipe, cov_pipe = _pipes()
    logo = LeaveOneGroupOut()
    rows, oof_store = [], {}
    for name, (kind, _, _) in _feat_specs(cov, bp, gz).items():
        oof = np.full(len(y), np.nan)
        for tr, te in logo.split(np.zeros(len(y)), y, grp):
            pipe = cov_pipe() if kind == "cov" else flat_pipe()
            pipe.fit(_make(name, kind, tr, cov, bp, gz), y[tr])
            oof[te] = pipe.predict_proba(_make(name, kind, te, cov, bp, gz))[:, 1]
        oof_store[name] = oof
        ar, ah, fr, fh = _auc_by_type(oof, y, ntype)
        rows.append({"eval": "cross-subject", "feature_set": name,
                     "auc_vs_random": ar, "auc_vs_hard": ah,
                     "fa_random@90sens": fr, "fa_hard@90sens": fh})
    return pd.DataFrame(rows), oof_store


def evaluate_within(cov, bp, gz, y, ntype, grp, times, n_folds=5):
    """Per-subject temporal folds; pool per-subject AUCs (calibrated BCI)."""
    flat_pipe, cov_pipe = _pipes()
    rows = []
    for name, (kind, _, _) in _feat_specs(cov, bp, gz).items():
        ar_l, ah_l, fr_l, fh_l = [], [], [], []
        for subj in sorted(set(grp)):
            sel = np.where(grp == subj)[0]
            order = sel[np.argsort(times[sel])]
            ys = y[order]
            if ys.sum() < n_folds or (ys == 0).sum() < n_folds:
                continue
            oof = np.full(len(order), np.nan)
            folds = np.array_split(np.arange(len(order)), n_folds)
            for f in range(n_folds):
                te = folds[f]
                tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
                if len(np.unique(ys[tr])) < 2:
                    continue
                pipe = cov_pipe() if kind == "cov" else flat_pipe()
                pipe.fit(_make(name, kind, order[tr], cov, bp, gz), ys[tr])
                oof[te] = pipe.predict_proba(
                    _make(name, kind, order[te], cov, bp, gz))[:, 1]
            ar, ah, fr, fh = _auc_by_type(oof, ys, ntype[order])
            ar_l.append(ar); ah_l.append(ah); fr_l.append(fr); fh_l.append(fh)
        rows.append({"eval": "within-subject", "feature_set": name,
                     "auc_vs_random": np.nanmean(ar_l),
                     "auc_vs_hard": np.nanmean(ah_l),
                     "fa_random@90sens": np.nanmean(fr_l),
                     "fa_hard@90sens": np.nanmean(fh_l)})
    return pd.DataFrame(rows)


def main():
    print("Preparing sessions...")
    preps = [prepare_session(s) for s in ds.all_sessions()]

    all_res = []
    stores = {}
    for w in (0.5, 1.0):
        cov, bp, gz, y, ntype, grp, tim = build(preps, w)
        print(f"\nW={w}s  pos={int((y==1).sum())} "
              f"random-neg={(ntype=='rand').sum()} hard-neg={(ntype=='hard').sum()}")
        r1, oof = evaluate_loso(cov, bp, gz, y, ntype, grp)
        r2 = evaluate_within(cov, bp, gz, y, ntype, grp, tim)
        for r in (r1, r2):
            r.insert(0, "window_s", w)
            all_res.append(r)
        stores[w] = (oof, y, ntype)
        print(pd.concat([r1, r2]).to_string(index=False))

    res = pd.concat(all_res, ignore_index=True)
    res.to_csv(OUT / "hardneg_results.csv", index=False)

    # ---- figure: within-subject at W=0.5 (most ERD-sensitive) ----
    sub = res[res["window_s"] == 0.5]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ev, title in [
        (axes[0], "cross-subject", "Cross-subject (leave-one-subject-out)"),
        (axes[1], "within-subject", "Within-subject (calibrated)")]:
        d = sub[sub["eval"] == ev]
        names = d["feature_set"].tolist()
        x = np.arange(len(names))
        ax.bar(x - 0.2, d["auc_vs_random"], 0.4, label="vs random neg",
               color="#56B4E9")
        bh = ax.bar(x + 0.2, d["auc_vs_hard"], 0.4,
                    label="vs hard neg (distractor look)", color="#D55E00")
        ax.bar_label(bh, fmt="%.2f", fontsize=7, padding=1)
        ax.axhline(0.5, color="#999", ls="--")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.3, 1.0)
        ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle("Distractor-look hard negatives (W=0.5s): gaze can't separate "
                 "look from look+pinch — EEG must carry it")
    fig.tight_layout(); fig.savefig(FIGS / "08_hardneg.png"); plt.close(fig)
    print("\nSaved", OUT / "hardneg_results.csv", "and", FIGS / "08_hardneg.png")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
