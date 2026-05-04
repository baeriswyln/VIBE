"""
s05_evaluate.py — Realistic evaluation strategies.

The single biggest mistake in EEG ML is using random splits with multi-subject
data. Brain signals are highly subject-specific; a random split lets the model
memorize each subject and inflates accuracy.

Three evaluation regimes:

  1. Within-subject: train and test on the same subject. Useful for personal
     BCIs that calibrate per user. Use stratified k-fold over the subject's
     epochs.

  2. Cross-subject (leave-one-subject-out): train on N-1 subjects, test on the
     held-out subject. Reports how well your model generalizes to a new user
     without calibration. This is the realistic deployment number.

  3. Subject-stratified (leave-N-subjects-out k-fold): faster than LOSO when
     you have many subjects, same logic.

Also covers temporal leakage:
  Even within one subject, if you predict from pre-event windows and events
  are close together, your test windows can overlap your training windows.
  Use time-based splits (train on first 80% of recording, test on last 20%)
  if there's any chance of this.

Usage example:
  uv run python s05_evaluate.py --epochs-dir data/epochs/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline


def load_subjects(epochs_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all *-epo.fif files and stack with subject IDs."""
    Xs, ys, gs = [], [], []
    for fp in sorted(epochs_dir.glob("*-epo.fif")):
        ep = mne.read_epochs(fp, preload=True, verbose=False)
        # Subject id from filename (e.g. sub-01-epo.fif -> sub-01)
        subj = fp.stem.replace("-epo", "")
        Xs.append(ep.get_data(copy=False))
        ys.append(ep.events[:, 2])
        gs.append(np.full(len(ep), subj))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs))


def featurize(X: np.ndarray) -> np.ndarray:
    """Quick covariance-based featurization for the example."""
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        cov = Covariances(estimator="oas").fit_transform(X)
        return TangentSpace().fit_transform(cov)
    except ImportError:
        # Fallback: log-variance per channel.
        return np.log(X.var(axis=-1) + 1e-12)


def leave_one_subject_out(epochs_dir: Path) -> None:
    X, y, groups = load_subjects(epochs_dir)
    subjects = np.unique(groups)
    print(f"Loaded {len(X)} epochs across {len(subjects)} subjects")

    feats = featurize(X)

    logo = LeaveOneGroupOut()
    accs = []
    for tr, te in logo.split(feats, y, groups):
        held_out = np.unique(groups[te])[0]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(feats[tr], y[tr])
        pred = clf.predict(feats[te])
        acc = balanced_accuracy_score(y[te], pred)
        accs.append(acc)
        print(f"  Held-out {held_out}: balanced_acc = {acc:.3f}")
    print(f"\nLOSO mean: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print("(The standard deviation tells you how much subjects vary in "
          "predictability. Large std = your model needs per-subject tuning.)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs-dir", type=Path, required=True)
    args = p.parse_args()
    leave_one_subject_out(args.epochs_dir)


if __name__ == "__main__":
    main()
