"""
Task 4 for the thesis: 12-fold LOSO robustness check for EEGNet, excluding
subject_22 and subject_26 (flagged for data-quality reasons in
12_data_quality_analysis.ipynb, cell 24) entirely from both training and test
data, then comparing the new 12-fold mean/std accuracy against the existing
14-fold baseline (data/results/eegnet_leave_one_out_run.pkl: mean 0.6449, std 0.0573).

This exactly replicates 14_classification.ipynb's data-loading (cell 2) and its
EEGNet LOSO loop (classify_leave_one_out / classify_cross_subject, cell 10):
  - same filtering query: ((EventType=='PinchLog') & (PinchType=='PINCH')) | (EventType=='NoPinchLog')
  - same per-channel z-scoring (epochs_scaled)
  - same train_eegnet call signature used for the LOSO loop: n_epochs=150, log_wandb=False,
    augment=False (default -- the base run does not pass augment=True)
  - the only difference is read_raw_dataset(..., subjects_to_drop=['subject_22','subject_26'])
    at load time, so those two subjects never appear in ANY fold's train or test split.

Does NOT touch data/results/*.pkl (no caching there) and does NOT modify any existing
notebook/module. Trained models are not persisted (only fold accuracies/predictions),
to keep this a lightweight, disk-safe robustness check.

Writes:
  out_thesis/subject_exclusion_robustness.csv
  out_thesis/subject_exclusion_robustness.md
"""
import pathlib
import sys
import time

import mne
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.EEGNet import train_eegnet          # noqa: E402
from scripts.dataset_manager import read_raw_dataset  # noqa: E402

EPOCH_FILTERED_DATASET_NAME = ROOT / "data" / "epoch_dataset_filtered"
OUT_DIR = ROOT / "out_thesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_SUBJECTS = ["subject_22", "subject_26"]

# ---- baseline numbers from the existing 14-fold EEGNet LOSO run (computed in
# thesis_analysis/01_model_metrics_significance.py from data/results/eegnet_leave_one_out_run.pkl) ----
BASELINE_14FOLD_MEAN = 0.6449
BASELINE_14FOLD_STD = 0.0573


def merge_epochs_dict(epochs_dict):
    """Copied verbatim from 14_classification.ipynb cell 2 (read-only reuse, not an edit)."""
    all_epochs = []
    for subj, sessions in epochs_dict.items():
        for sess, ep in sessions.items():
            ep = ep.copy()
            n = len(ep)
            extra = pd.DataFrame(
                {"subject": [subj] * n, "session": [sess] * n},
                index=ep.metadata.index if ep.metadata is not None else range(n),
            )
            if ep.metadata is None:
                ep.metadata = extra
            else:
                meta = ep.metadata.copy()
                meta["subject"] = subj
                meta["session"] = sess
                ep.metadata = meta
            all_epochs.append(ep)
    merged = mne.concatenate_epochs(all_epochs)
    return merged


def query_epochs(epochs, query):
    ep = epochs.copy()
    keep_idx = ep.metadata.query(query).index
    mask = ep.metadata.index.isin(keep_idx)
    return ep[mask]


def main():
    t_start = time.time()

    print(f"Loading epoch_dataset_filtered, excluding {EXCLUDED_SUBJECTS} at load time...")
    epoch_dict = read_raw_dataset(EPOCH_FILTERED_DATASET_NAME, subjects_to_drop=EXCLUDED_SUBJECTS)
    print(f"Subjects loaded: {sorted(epoch_dict.keys())} (n={len(epoch_dict)})")
    assert not (set(EXCLUDED_SUBJECTS) & set(epoch_dict.keys())), "excluded subjects leaked into load"
    assert len(epoch_dict) == 12, f"expected 12 subjects, got {len(epoch_dict)}"

    epochs = merge_epochs_dict(epoch_dict)
    epochs_scaled = epochs.copy()
    epochs_scaled.load_data().apply_function(
        lambda x: (x - x.mean()) / (x.std() + 1e-12), channel_wise=True
    )

    filtered_epochs_scaled = query_epochs(
        epochs_scaled,
        "((EventType == 'PinchLog') & (PinchType == 'PINCH')) | (EventType == 'NoPinchLog')",
    ).copy()
    print(f"filtered_epochs_scaled: {len(filtered_epochs_scaled)} epochs, "
          f"{filtered_epochs_scaled.metadata['subject'].nunique()} subjects")

    subjects = filtered_epochs_scaled.metadata["subject"].unique().tolist()
    print(f"12-fold LOSO over: {subjects}")

    fold_rows = []
    for i, subject in enumerate(subjects, 1):
        t_fold = time.time()
        train_subjects = [s for s in subjects if s != subject]
        test_subjects = [subject]

        train_epochs = query_epochs(filtered_epochs_scaled, f"subject in {train_subjects}")
        test_epochs = query_epochs(filtered_epochs_scaled, f"subject in {test_subjects}")

        print(f"\n[{i}/12] held-out={subject} | train={len(train_epochs)} epochs "
              f"({len(train_subjects)} subj) | test={len(test_epochs)} epochs")

        result = train_eegnet(
            train_epochs, train_epochs.metadata.EventType,
            test_epochs, test_epochs.metadata.EventType,
            log_wandb=False, n_epochs=150,  # exact call signature used in 14_classification.ipynb's LOSO loop
        )

        y_true = test_epochs.metadata["EventType"].to_numpy()
        y_pred = np.asarray(result["y_pred"])
        acc = float((y_true == y_pred).mean())
        assert abs(acc - result["test_accuracy"]) < 1e-6, "accuracy mismatch vs train_eegnet's own report"

        elapsed = time.time() - t_fold
        print(f"[{i}/12] held-out={subject} -> test_accuracy={acc:.4f}  ({elapsed:.1f}s)")

        fold_rows.append(dict(
            subject=subject, n_train=len(train_epochs), n_test=len(test_epochs),
            test_accuracy=acc, fold_seconds=elapsed,
        ))

    df = pd.DataFrame(fold_rows)
    mean_acc = df["test_accuracy"].mean()
    std_acc = df["test_accuracy"].std(ddof=0)  # population std, matches how the 14-fold baseline was computed

    total_elapsed = time.time() - t_start
    print(f"\n12-fold (excl. subject_22, subject_26) EEGNet LOSO: "
          f"mean={mean_acc:.4f}  std={std_acc:.4f}  (total wall time {total_elapsed/60:.1f} min)")
    print(f"14-fold baseline (all subjects, from data/results/eegnet_leave_one_out_run.pkl): "
          f"mean={BASELINE_14FOLD_MEAN:.4f}  std={BASELINE_14FOLD_STD:.4f}")
    print(f"Delta (12-fold minus 14-fold baseline): mean {mean_acc - BASELINE_14FOLD_MEAN:+.4f}, "
          f"std {std_acc - BASELINE_14FOLD_STD:+.4f}")

    df.to_csv(OUT_DIR / "subject_exclusion_robustness_per_fold.csv", index=False)

    summary = pd.DataFrame([
        dict(variant="14-fold (all subjects, baseline)", n_folds=14,
             accuracy_mean=BASELINE_14FOLD_MEAN, accuracy_std=BASELINE_14FOLD_STD),
        dict(variant="12-fold (excl. subject_22, subject_26)", n_folds=12,
             accuracy_mean=mean_acc, accuracy_std=std_acc),
    ])
    summary.to_csv(OUT_DIR / "subject_exclusion_robustness.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'subject_exclusion_robustness.csv'}")
    print(f"Wrote {OUT_DIR / 'subject_exclusion_robustness_per_fold.csv'}")

    md_lines = [
        "# Subject-exclusion robustness check (EEGNet, LOSO, excl. subject_22 & subject_26)\n",
        f"Total wall time: {total_elapsed/60:.1f} min for {len(subjects)} folds.\n",
        "| variant | n_folds | accuracy_mean | accuracy_std |",
        "|---|---|---|---|",
        f"| 14-fold (all subjects, baseline) | 14 | {BASELINE_14FOLD_MEAN:.4f} | {BASELINE_14FOLD_STD:.4f} |",
        f"| 12-fold (excl. subject_22, subject_26) | 12 | {mean_acc:.4f} | {std_acc:.4f} |",
        "",
        f"Delta (12-fold - 14-fold): mean {mean_acc - BASELINE_14FOLD_MEAN:+.4f}, "
        f"std {std_acc - BASELINE_14FOLD_STD:+.4f}",
        "",
        "## Per-fold accuracy (12-fold run)",
        "",
        "| subject | n_train | n_test | test_accuracy |",
        "|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        md_lines.append(f"| {r['subject']} | {int(r['n_train'])} | {int(r['n_test'])} | {r['test_accuracy']:.4f} |")
    (OUT_DIR / "subject_exclusion_robustness.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'subject_exclusion_robustness.md'}")


if __name__ == "__main__":
    main()
