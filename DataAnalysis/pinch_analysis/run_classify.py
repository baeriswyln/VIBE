"""
run_classify.py — Detect a pinch from a preceding EEG window.

Framing (causal detection):
  * Positive window: [t0 - W, t0] where t0 is a pinch onset  ->  "pinch at t+0".
  * Negative window: [t - W, t] where t is >= (W + buffer) s from every pinch.
  The window ends exactly at the reference time, so a positive contains the
  build-up *up to* the pinch and a negative contains matched no-pinch data. This
  is the online-decodable framing (you only ever use past samples).

We sweep window length W in {0.5, 1.0, 1.5, 2.0} s and compare feature sets:
  * riemann  : tangent-space projection of 8-30 Hz channel covariances + LogReg
  * bandpow  : band power (delta..beta2) + Hjorth per channel + LogReg
  * bandpow+gaze : the above plus gaze/pupil summary of the same window

Evaluation is **leave-one-subject-out** (group = subject) so scores reflect
generalisation to unseen people, not memorised trials. We report ROC-AUC and
PR-AUC (chance PR-AUC = positive prevalence), and sensitivity at 5 % false-alarm.

Run:  uv run python -m pinch_analysis.run_classify
"""
from __future__ import annotations

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from . import dataset as ds
from .epoching import (prepare_session, pinch_epochs, negative_epochs,
                       sample_negative_times, _events_from_times)
from .features import bandpower_features

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = ds._ROOT / "out"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

WINDOWS = [0.5, 1.0, 1.5, 2.0]
NEG_RATIO = 2.0            # negatives per positive
GAZE_GRID_HZ = 60.0


def gaze_window_features(prep, center_times, W) -> np.ndarray:
    """Summary gaze features over [t-W, t] for each center time.

    Returns (n, 6): mean/last pupil, mean eye-openness, mean/last gaze speed,
    fraction low-speed (fixation). NaN rows if the session has no gaze.
    """
    n = len(center_times)
    if not ds.has_gaze(prep.session):
        return np.full((n, 6), np.nan)
    gaze = ds.load_gaze(prep.session)
    grid = np.arange(prep.eeg_ts[0], prep.eeg_ts[-1], 1 / GAZE_GRID_HZ)
    gon = ds.gaze_on_grid(gaze, grid)
    gt = gon.index.to_numpy()
    pupil = gon["left_pupil_diameter"].to_numpy()
    openness = gon["left_eye_openness"].to_numpy()
    dx = np.gradient(gon["left_gaze_direction_x"].to_numpy())
    dy = np.gradient(gon["left_gaze_direction_y"].to_numpy())
    speed = np.sqrt(dx ** 2 + dy ** 2) * GAZE_GRID_HZ
    out = np.zeros((n, 6))
    for i, t in enumerate(center_times):
        m = (gt >= t - W) & (gt <= t)
        if m.sum() < 2:
            out[i] = np.nan
            continue
        sp = speed[m]
        out[i] = [np.nanmean(pupil[m]), pupil[m][-1], np.nanmean(openness[m]),
                  np.nanmean(sp), sp[-1], np.mean(sp < np.nanmedian(speed))]
    return out


def build_dataset(preps, W):
    """Assemble epochs, labels, groups and gaze features for window W."""
    Xcov, Xbp, Xgaze, y, groups, exps, times = [], [], [], [], [], [], []
    for prep in preps:
        # positives: window ends at pinch
        pos = pinch_epochs(prep, tmin=-W, tmax=0.0)
        if len(pos) == 0:
            continue
        pos_t = pos.metadata["timestamp"].to_numpy()
        pos_exp = pos.metadata["experiment"].to_numpy()
        # negatives: >= (W + 1.0) s from any pinch
        neg_t = sample_negative_times(prep, n=int(len(pos) * NEG_RATIO),
                                      window=2 * W, buffer=1.0, rng=7)
        neg = negative_epochs(prep, neg_t, tmin=-W, tmax=0.0)
        if neg is None or len(neg) == 0:
            continue
        # recover the actual accepted negative times (epochs may be rejected)
        neg_samp = neg.events[:, 0]
        neg_t_used = prep.eeg_ts[np.clip(neg_samp, 0, len(prep.eeg_ts) - 1)]

        for ep, t, lab, ex in [(pos, pos_t, 1, pos_exp),
                               (neg, neg_t_used, 0, None)]:
            d = ep.get_data(copy=False)  # (n, ch, times) in Volts
            # band-limit copy for covariance (8-30 Hz motor band)
            band = ep.copy().filter(8, 30, verbose="ERROR").get_data(copy=False)
            Xcov.append(band)
            Xbp.append(bandpower_features(d * 1e6, ep.info["sfreq"]))
            Xgaze.append(gaze_window_features(prep, t, W))
            y.append(np.full(len(ep), lab))
            groups.append([prep.session.subject] * len(ep))
            exps.append(ex if ex is not None else np.array([None] * len(ep)))
            times.append(t)
    Xcov = np.concatenate(Xcov)
    Xbp = np.concatenate(Xbp)
    Xgaze = np.concatenate(Xgaze)
    y = np.concatenate(y)
    groups = np.concatenate(groups)
    exps = np.concatenate([np.asarray(e, dtype=object) for e in exps])
    times = np.concatenate(times)
    return Xcov, Xbp, Xgaze, y, groups, exps, times


def sens_at_fpr(y, s, fpr_target=0.05):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def loso_eval(name, make_X, Xcov, Xbp, Xgaze, y, groups, pipe_factory):
    """Leave-one-subject-out; returns dict of pooled metrics + per-fold."""
    logo = LeaveOneGroupOut()
    oof = np.full(len(y), np.nan)
    per_fold = []
    for tr, te in logo.split(np.zeros(len(y)), y, groups):
        Xtr = make_X(Xcov[tr], Xbp[tr], Xgaze[tr])
        Xte = make_X(Xcov[te], Xbp[te], Xgaze[te])
        if isinstance(Xtr, tuple):  # riemann uses raw cov array
            pipe = pipe_factory()
            pipe.fit(Xtr[0], y[tr])
            s = pipe.predict_proba(Xte[0])[:, 1]
        else:
            pipe = pipe_factory()
            pipe.fit(Xtr, y[tr])
            s = pipe.predict_proba(Xte)[:, 1]
        oof[te] = s
        if len(np.unique(y[te])) == 2:
            per_fold.append({
                "subject": groups[te][0],
                "auc": roc_auc_score(y[te], s),
                "ap": average_precision_score(y[te], s),
                "n": int(len(te)),
            })
    ok = ~np.isnan(oof)
    return {
        "feature_set": name,
        "auc": float(roc_auc_score(y[ok], oof[ok])),
        "pr_auc": float(average_precision_score(y[ok], oof[ok])),
        "sens@fpr5": sens_at_fpr(y[ok], oof[ok]),
        "prevalence": float(y.mean()),
        "per_fold_auc_mean": float(np.mean([f["auc"] for f in per_fold])),
        "per_fold_auc_std": float(np.std([f["auc"] for f in per_fold])),
        "n": int(ok.sum()),
        "_oof": oof, "_y": y,
    }


def within_subject_eval(name, make_X, Xcov, Xbp, Xgaze, y, groups, times,
                        pipe_factory, n_folds=5):
    """Per-subject temporal K-fold (calibrated-BCI ceiling).

    For each subject the epochs are ordered in time and split into contiguous
    folds, so train and test never share adjacent windows. Returns the mean and
    std of the per-subject pooled ROC-AUC.
    """
    aucs = []
    for subj in sorted(set(groups)):
        sel = np.where(groups == subj)[0]
        order = sel[np.argsort(times[sel])]
        ys = y[order]
        if len(np.unique(ys)) < 2 or len(order) < 3 * n_folds:
            continue
        folds = np.array_split(np.arange(len(order)), n_folds)
        oof = np.full(len(order), np.nan)
        for f in range(n_folds):
            te = folds[f]
            tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
            if len(np.unique(ys[tr])) < 2:
                continue
            Xtr = make_X(Xcov[order][tr], Xbp[order][tr], Xgaze[order][tr])
            Xte = make_X(Xcov[order][te], Xbp[order][te], Xgaze[order][te])
            pipe = pipe_factory()
            if isinstance(Xtr, tuple):
                pipe.fit(Xtr[0], ys[tr]); s = pipe.predict_proba(Xte[0])[:, 1]
            else:
                pipe.fit(Xtr, ys[tr]); s = pipe.predict_proba(Xte)[:, 1]
            oof[te] = s
        ok = ~np.isnan(oof)
        if len(np.unique(ys[ok])) == 2:
            aucs.append(roc_auc_score(ys[ok], oof[ok]))
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs)


def impute(X):
    X = X.copy()
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    idx = np.where(~np.isfinite(X))
    X[idx] = np.take(col_med, idx[1])
    return X


def main():
    print("Preparing sessions...")
    preps = [prepare_session(s) for s in ds.all_sessions()]

    # pipeline factories
    def riemann_pipe():
        return make_pipeline(Covariances("oas"), TangentSpace(),
                             StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))

    def bp_pipe():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))

    results = []
    roc_store = {}
    for W in WINDOWS:
        print(f"\n=== Window {W}s ===")
        Xcov, Xbp, Xgaze, y, groups, exps, times = build_dataset(preps, W)
        print(f"  n={len(y)}  pos={int(y.sum())}  neg={int((1-y).sum())}  "
              f"subjects={sorted(set(groups))}")

        # feature adapters
        make_cov = lambda c, b, g: (c,)                       # noqa: E731
        make_bp = lambda c, b, g: impute(b)                   # noqa: E731
        make_bpg = lambda c, b, g: impute(np.hstack([b, g]))  # noqa: E731

        for name, mk, fac in [
            ("riemann", make_cov, riemann_pipe),
            ("bandpow", make_bp, bp_pipe),
            ("bandpow+gaze", make_bpg, bp_pipe),
        ]:
            r = loso_eval(name, mk, Xcov, Xbp, Xgaze, y, groups, fac)
            r["window_s"] = W
            wm, wsd, wn = within_subject_eval(name, mk, Xcov, Xbp, Xgaze, y,
                                              groups, times, fac)
            r["within_auc_mean"] = wm
            r["within_auc_std"] = wsd
            roc_store[(W, name)] = (r.pop("_y"), r.pop("_oof"))
            results.append(r)
            print(f"  {name:14s} LOSO-AUC={r['auc']:.3f} PR={r['pr_auc']:.3f} "
                  f"sens@5%FPR={r['sens@fpr5']:.2f} | within-subj AUC "
                  f"{wm:.3f}±{wsd:.3f}")

    res = pd.DataFrame(results)
    res.to_csv(OUT / "classify_results.csv", index=False)

    # --- figure: AUC vs window, and ROC for best config ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for name in ["riemann", "bandpow", "bandpow+gaze"]:
        sub = res[res.feature_set == name].sort_values("window_s")
        ax.plot(sub.window_s, sub.auc, "o-", label=name)
    ax.axhline(0.5, color="#999", ls="--")
    ax.set_xlabel("window length (s)"); ax.set_ylabel("LOSO ROC-AUC")
    ax.set_title("Pinch detection AUC vs window size"); ax.legend()

    ax = axes[1]
    best = res.loc[res.auc.idxmax()]
    yb, ob = roc_store[(best.window_s, best.feature_set)]
    ok = ~np.isnan(ob)
    fpr, tpr, _ = roc_curve(yb[ok], ob[ok])
    ax.plot(fpr, tpr, color="#0072B2",
            label=f"{best.feature_set} @ {best.window_s}s (AUC={best.auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999")
    ax.set_xlabel("false-alarm rate"); ax.set_ylabel("sensitivity")
    ax.set_title("Best detector ROC (leave-subject-out)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "06_classify.png"); plt.close(fig)

    print("\nSaved", OUT / "classify_results.csv", "and", FIGS / "06_classify.png")
    print(res[["window_s", "feature_set", "auc", "pr_auc", "sens@fpr5",
               "within_auc_mean", "within_auc_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
