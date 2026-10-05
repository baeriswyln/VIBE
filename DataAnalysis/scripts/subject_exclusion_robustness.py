"""
Task 4 for the thesis: LOSO robustness check for EEGNet, excluding
subject_22 and subject_26 (flagged for data-quality reasons in
12_data_quality_analysis.ipynb, high epoch-rejection rate) entirely from both training and test
data, then comparing its mean/std accuracy against the all-subject EEGNet LOSO
baseline read from out/results/eegnet_leave_one_out_run.pkl (same dataset version).

This exactly replicates 15_classification.ipynb's data-loading (cell 2) and its
EEGNet LOSO loop (classify_leave_one_out / classify_cross_subject, cell 10):
  - same filtering query: ((EventType=='PinchLog') & (PinchType=='PINCH')) | (EventType=='NoPinchLog')
  - same per-channel z-scoring (epochs_scaled)
  - same train_eegnet call signature used for the LOSO loop: n_epochs=150, log_wandb=False,
    augment=False (default -- the base run does not pass augment=True)
  - the only difference is read_raw_dataset(..., subjects_to_drop=['subject_22','subject_26'])
    at load time, so those two subjects never appear in ANY fold's train or test split.

Does NOT touch out/results/*.pkl (no caching there) and does NOT modify any existing
notebook/module. Trained models are not persisted (only fold accuracies/predictions),
to keep this a lightweight, disk-safe robustness check.

Writes:
  out/subject_exclusion_robustness.csv
  out/subject_exclusion_robustness.md
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
OUT_DIR = ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_SUBJECTS = ["subject_22", "subject_26"]



def load_baseline():
    """Per-fold accuracy of the all-subject EEGNet LOSO run (out/results/eegnet_leave_one_out_run.pkl)."""
    import pickle
    with open(OUT_DIR / "results" / "eegnet_leave_one_out_run.pkl", "rb") as fh:
        result = pickle.load(fh)["result"]
    accs = []
    for fold in result.values():
        y_true = fold["test_metadata"]["EventType"].to_numpy()
        accs.append(float((y_true == np.asarray(fold["result"]["y_pred"])).mean()))
    accs = np.asarray(accs)
    return len(accs), accs.mean(), accs.std(ddof=0)  # population std, as in notebook 17


def merge_epochs_dict(epochs_dict):
    """Copied verbatim from 15_classification.ipynb cell 2 (read-only reuse, not an edit)."""
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
    base_n, base_mean, base_std = load_baseline()
    assert len(epoch_dict) == base_n - len(EXCLUDED_SUBJECTS), \
        f"expected {base_n - len(EXCLUDED_SUBJECTS)} subjects, got {len(epoch_dict)}"

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
    n = len(subjects)
    print(f"{n}-fold LOSO over: {subjects}")

    fold_rows = []
    for i, subject in enumerate(subjects, 1):
        t_fold = time.time()
        train_subjects = [s for s in subjects if s != subject]
        test_subjects = [subject]

        train_epochs = query_epochs(filtered_epochs_scaled, f"subject in {train_subjects}")
        test_epochs = query_epochs(filtered_epochs_scaled, f"subject in {test_subjects}")

        print(f"\n[{i}/{n}] held-out={subject} | train={len(train_epochs)} epochs "
              f"({len(train_subjects)} subj) | test={len(test_epochs)} epochs")

        result = train_eegnet(
            train_epochs, train_epochs.metadata.EventType,
            test_epochs, test_epochs.metadata.EventType,
            log_wandb=False, n_epochs=150,  # exact call signature used in 15_classification.ipynb's LOSO loop
        )

        y_true = test_epochs.metadata["EventType"].to_numpy()
        y_pred = np.asarray(result["y_pred"])
        acc = float((y_true == y_pred).mean())
        assert abs(acc - result["test_accuracy"]) < 1e-6, "accuracy mismatch vs train_eegnet's own report"

        elapsed = time.time() - t_fold
        print(f"[{i}/{n}] held-out={subject} -> test_accuracy={acc:.4f}  ({elapsed:.1f}s)")

        fold_rows.append(dict(
            subject=subject, n_train=len(train_epochs), n_test=len(test_epochs),
            test_accuracy=acc, fold_seconds=elapsed,
        ))

    df = pd.DataFrame(fold_rows)
    mean_acc = df["test_accuracy"].mean()
    std_acc = df["test_accuracy"].std(ddof=0)  # population std, same as the baseline

    excl_label = f"{n}-fold (excl. {', '.join(EXCLUDED_SUBJECTS)})"
    base_label = f"{base_n}-fold (all subjects, baseline)"
    total_elapsed = time.time() - t_start
    print(f"\n{excl_label} EEGNet LOSO: "
          f"mean={mean_acc:.4f}  std={std_acc:.4f}  (total wall time {total_elapsed/60:.1f} min)")
    print(f"{base_label}, from out/results/eegnet_leave_one_out_run.pkl: "
          f"mean={base_mean:.4f}  std={base_std:.4f}")
    print(f"Delta (excluded minus baseline): mean {mean_acc - base_mean:+.4f}, "
          f"std {std_acc - base_std:+.4f}")

    df.to_csv(OUT_DIR / "subject_exclusion_robustness_per_fold.csv", index=False)

    summary = pd.DataFrame([
        dict(variant=base_label, n_folds=base_n, accuracy_mean=base_mean, accuracy_std=base_std),
        dict(variant=excl_label, n_folds=n, accuracy_mean=mean_acc, accuracy_std=std_acc),
    ])
    summary.to_csv(OUT_DIR / "subject_exclusion_robustness.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'subject_exclusion_robustness.csv'}")
    print(f"Wrote {OUT_DIR / 'subject_exclusion_robustness_per_fold.csv'}")

    md_lines = [
        "# Subject-exclusion robustness check (EEGNet, LOSO, excl. subject_22 & subject_26)\n",
        f"Total wall time: {total_elapsed/60:.1f} min for {n} folds.\n",
        "| variant | n_folds | accuracy_mean | accuracy_std |",
        "|---|---|---|---|",
        f"| {base_label} | {base_n} | {base_mean:.4f} | {base_std:.4f} |",
        f"| {excl_label} | {n} | {mean_acc:.4f} | {std_acc:.4f} |",
        "",
        f"Delta (excluded - baseline): mean {mean_acc - base_mean:+.4f}, "
        f"std {std_acc - base_std:+.4f}",
        "",
        f"## Per-fold accuracy ({n}-fold run)",
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
