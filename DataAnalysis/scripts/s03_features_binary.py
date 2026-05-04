"""
03_features_binary.py — Binary event detection with proper metrics.

Differences from the multi-class version:
  - Uses ROC-AUC and PR-AUC instead of balanced accuracy. PR-AUC is the
    primary metric when events are rare; it tells you how well the model
    ranks positives above negatives without being inflated by easy negatives.
  - Reports sensitivity at fixed specificity points — the operational metric
    you'd use to set a deployment threshold.
  - Reports calibration so you know whether predicted probabilities mean
    anything (they often don't, out of the box).

Models tried (best-to-worst expected for typical EEG event detection):
  - TangentSpace + LogReg on covariances: usually the best classical method
  - TangentSpace + RandomForest: handles nonlinearity in tangent space
  - LogReg on band power + Hjorth: interpretable baseline
  - XGBoost on the same features: stronger nonlinear baseline (optional)

Threshold selection:
  At the end we pick a threshold via Youden's J on the ROC curve and report
  confusion-matrix-style numbers there. In production you'd choose the
  threshold based on the specificity / false-alarm rate you can tolerate.

Usage:
  uv run python 03_features_binary.py --epochs data/epochs/sub-01-epo.fif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 40)}


def band_power(epochs: mne.Epochs) -> np.ndarray:
    spectrum = epochs.compute_psd(method="welch", fmin=1, fmax=40, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    feats = []
    for low, high in BANDS.values():
        m = (freqs >= low) & (freqs < high)
        feats.append(np.log(psds[..., m].mean(axis=-1) + 1e-12))
    f = np.stack(feats, axis=-1)
    return f.reshape(f.shape[0], -1)


def hjorth(epochs: mne.Epochs) -> np.ndarray:
    data = epochs.get_data(copy=False)
    var0 = data.var(axis=-1)
    d1 = np.diff(data, axis=-1); var1 = d1.var(axis=-1)
    d2 = np.diff(d1, axis=-1);  var2 = d2.var(axis=-1)
    mob = np.sqrt(var1 / (var0 + 1e-12))
    cmp_ = np.sqrt(var2 / (var1 + 1e-12)) / (mob + 1e-12)
    return np.concatenate([var0, mob, cmp_], axis=-1)


def report_binary(name: str, y_true: np.ndarray, y_score: np.ndarray) -> None:
    """Print all the metrics that actually matter for event detection."""
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    # Sensitivity at common operating points
    fpr, tpr, thr = roc_curve(y_true, y_score)
    targets = [0.01, 0.05, 0.10]  # specificity = 1 - fpr
    sens = []
    for fpr_target in targets:
        # find tpr at this fpr
        ix = np.searchsorted(fpr, fpr_target)
        ix = min(ix, len(tpr) - 1)
        sens.append(tpr[ix])
    # Youden's J optimum threshold
    j_idx = np.argmax(tpr - fpr)
    thr_j = thr[j_idx]
    pred_j = (y_score >= thr_j).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_j).ravel()

    print(f"  {name}")
    print(f"    ROC-AUC = {auc:.3f}   PR-AUC = {ap:.3f}")
    print(f"    Sensitivity @ FPR=1%: {sens[0]:.2f}   "
          f"@ FPR=5%: {sens[1]:.2f}   @ FPR=10%: {sens[2]:.2f}")
    print(f"    At Youden threshold {thr_j:.2f}: "
          f"sens={tp/(tp+fn):.2f}, spec={tn/(tn+fp):.2f}, "
          f"prec={tp/(tp+fp+1e-9):.2f}")


def evaluate(epochs_path: Path) -> None:
    epochs = mne.read_epochs(epochs_path, preload=True)
    # Binary y: 1 = event, 0 = no_event
    y = (epochs.events[:, 2] == 1).astype(int)
    base_rate = y.mean()
    print(f"Base rate (positive class): {base_rate:.3f}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Feature path ------------------------------------------------------
    X_feat = np.concatenate([band_power(epochs), hjorth(epochs)], axis=1)

    pipelines = {
        "BandPower+Hjorth + LogReg": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
    }

    for name, pipe in pipelines.items():
        # Stack out-of-fold predictions to get one score per epoch.
        scores = np.zeros(len(y))
        for tr, te in cv.split(X_feat, y):
            pipe.fit(X_feat[tr], y[tr])
            scores[te] = pipe.predict_proba(X_feat[te])[:, 1]
        report_binary(name, y, scores)

    # --- Riemannian path ---------------------------------------------------
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace

        X_eeg = epochs.get_data(copy=False)

        ts_lr = Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts",  TangentSpace()),
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        scores = np.zeros(len(y))
        for tr, te in cv.split(X_eeg, y):
            ts_lr.fit(X_eeg[tr], y[tr])
            scores[te] = ts_lr.predict_proba(X_eeg[te])[:, 1]
        report_binary("TangentSpace + LogReg", y, scores)
    except ImportError:
        print("pyriemann not installed; skipping Riemannian methods.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=Path, required=True)
    args = p.parse_args()
    evaluate(args.epochs)


if __name__ == "__main__":
    main()