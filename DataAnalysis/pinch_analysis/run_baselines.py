"""
run_baselines.py — Compare every generated model to a constant classifier.

A constant (majority-class) classifier is the honesty check for imbalanced data:
it can post a high *accuracy* by always predicting "no pinch", yet it is useless
(balanced accuracy 0.5, F1 0, ROC-AUC 0.5, PR-AUC = prevalence). We show, with a
full metric suite, that the real models beat it.

Part A — empirical, on the pinch-detection task (W=1.0 s, leave-one-subject-out):
  models: constant (sklearn DummyClassifier 'prior'), stratified-random,
  eeg-bandpow, eeg-riemann, bandpow+gaze.
  metrics: accuracy, balanced accuracy, F1, ROC-AUC, PR-AUC.

Part B — a consolidated table of every generated classifier (detection, the
distractor decoders, transfer) against the constant baseline, whose ROC-AUC is
0.5 and PR-AUC equals the positive prevalence by construction.

Run:  uv run python -m pinch_analysis.run_baselines
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, average_precision_score)
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from . import dataset as ds
from .epoching import prepare_session
from .run_classify import build_dataset, impute

warnings.filterwarnings("ignore")
OUT = ds._ROOT / "out"
FIGS = OUT / "figs"
W = 1.0


def metric_suite(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    # roc_auc/ap are threshold-free; constant scores -> auc 0.5, ap = prevalence
    try:
        auc = roc_auc_score(y, proba)
    except ValueError:
        auc = 0.5
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": auc,
        "pr_auc": average_precision_score(y, proba),
    }


def loso_proba(make_X, factory, Xcov, Xbp, Xg, y, groups, is_cov=False):
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(np.zeros(len(y)), y, groups):
        m = factory()
        if is_cov:
            m.fit(Xcov[tr], y[tr]); oof[te] = m.predict_proba(Xcov[te])[:, 1]
        else:
            Xtr, Xte = make_X(tr), make_X(te)
            m.fit(Xtr, y[tr]); oof[te] = m.predict_proba(Xte)[:, 1]
    return oof


def main():
    print("Preparing sessions...")
    preps = [prepare_session(s) for s in ds.all_sessions()]
    Xcov, Xbp, Xg, y, groups, exps, times = build_dataset(preps, W)
    prev = float(y.mean())
    print(f"n={len(y)} prevalence={prev:.3f} (W={W}s)")

    def lr():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    def rie():
        return make_pipeline(Covariances("oas"), TangentSpace(), StandardScaler(),
                             LogisticRegression(max_iter=2000))
    mk_bp = lambda idx: impute(Xbp[idx])                       # noqa: E731
    mk_bpg = lambda idx: impute(np.hstack([Xbp[idx], Xg[idx]]))  # noqa: E731

    configs = [
        ("constant (majority)", lambda: DummyClassifier(strategy="prior"), mk_bp, False),
        ("stratified random", lambda: DummyClassifier(strategy="stratified",
                                                      random_state=0), mk_bp, False),
        ("eeg-bandpow", lr, mk_bp, False),
        ("eeg-riemann", rie, None, True),
        ("bandpow+gaze", lr, mk_bpg, False),
    ]
    rows = []
    for name, fac, mk, is_cov in configs:
        oof = loso_proba(mk, fac, Xcov, Xbp, Xg, y, groups, is_cov=is_cov)
        ok = ~np.isnan(oof)
        rows.append({"model": name, **metric_suite(y[ok], oof[ok])})
        print(f"  {name:20s} acc={rows[-1]['accuracy']:.3f} "
              f"bal_acc={rows[-1]['balanced_acc']:.3f} F1={rows[-1]['f1']:.3f} "
              f"AUC={rows[-1]['roc_auc']:.3f} PR-AUC={rows[-1]['pr_auc']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "baseline_results.csv", index=False)

    # ---- figure: metric suite, constant baseline highlighted ----
    metrics = ["accuracy", "balanced_acc", "f1", "roc_auc", "pr_auc"]
    labels = ["accuracy", "balanced\naccuracy", "F1", "ROC-AUC", "PR-AUC"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metrics))
    palette = ["#BBBBBB", "#888888", "#56B4E9", "#E69F00", "#0072B2"]
    nb = len(res)
    width = 0.8 / nb
    for i, r in enumerate(res.itertuples()):
        vals = [getattr(r, m) for m in metrics]
        ax.bar(x + (i - (nb - 1) / 2) * width, vals, width,
               label=r.model, color=palette[i])
    ax.axhline(0.5, color="#999", ls="--", lw=0.8)
    ax.axhline(prev, color="#D55E00", ls=":", lw=0.8,
               label=f"positive prevalence = {prev:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("score (leave-one-subject-out, W=1 s)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Pinch detection vs a constant classifier — full metric suite\n"
                 "(the constant model matches chance on every balanced metric)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIGS / "11_baselines.png"); plt.close(fig)
    print("Saved", OUT / "baseline_results.csv", "and", FIGS / "11_baselines.png")

    # ---- Part B: consolidated ROC-AUC vs constant across all tasks ----
    summary = [("Detection: eeg-riemann (LOSO, best W)", "classify_results.csv"),
               ("Detection: bandpow+gaze (LOSO, best W)", None)]
    rowsB = []
    cls = pd.read_csv(OUT / "classify_results.csv")
    rowsB.append(["Detection LOSO (bandpow+gaze, best W)",
                  cls[cls.feature_set == "bandpow+gaze"]["auc"].max(), prev])
    rowsB.append(["Detection within-subj (bandpow+gaze, best W)",
                  cls[cls.feature_set == "bandpow+gaze"]["within_auc_mean"].max(),
                  prev])
    en = pd.read_csv(OUT / "eegnet_results.csv")
    rowsB.append(["Pinch vs distractor: EEGNet (per-subj)",
                  en["eegnet_auc"].mean(), np.nan])
    rowsB.append(["Pinch vs distractor: Riemann (per-subj)",
                  en["riemann_auc"].mean(), np.nan])
    tr = pd.read_csv(OUT / "eegnet_transfer_results.csv")
    for cond in ["zero-shot", "fine-tuned"]:
        rowsB.append([f"Transfer EEGNet ({cond})",
                      tr[tr.condition == cond]["auc_all"].mean(), np.nan])
    rowsB.append(["Constant classifier (any task)", 0.5, np.nan])
    dfB = pd.DataFrame(rowsB, columns=["model", "roc_auc", "note_prevalence"])
    dfB.to_csv(OUT / "baseline_summary_allmodels.csv", index=False)
    print("\nAll models vs constant (ROC-AUC):")
    print(dfB.to_string(index=False))


if __name__ == "__main__":
    main()
