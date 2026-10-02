"""
Task 1 + 2 for the thesis: pooled precision/recall/F1/FPR + per-fold accuracy
mean/std for the base 14-fold LOSO runs of RF (EEGForest), EEGNet and the
GAT-GRU-Transformer, plus paired Wilcoxon signed-rank significance tests
across the 14 held-out subjects.

Reads data/results/{rf,eegnet,transformer}_leave_one_out_run.pkl (read-only).
Writes:
  out_thesis/model_metrics_loso.csv
  out_thesis/model_significance_loso.csv
  out_thesis/per_fold_accuracy_loso.csv   (raw per-fold numbers, for appendix/debugging)

Structure of each pkl (confirmed by direct inspection, see thesis_analysis/_inspect.py):
    pickle.load(fh) -> {"data_version": str, "result": result}
    result: dict keyed by subject id (e.g. "subject_21"), 14 keys
        result[subj] = {"result": <train_fn output dict>, "test_metadata": <pd.DataFrame>}
    The train_fn output dict always has "y_pred" (array of label strings) and
    "confusion_matrix"; test_metadata has an "EventType" column ("PinchLog"/"NoPinchLog")
    in the SAME row order as y_pred (verified: none of train_eegnet/train_eegforest/
    train_gat_gru_transformer shuffle their test DataLoader/array, so y_true is taken
    directly from test_metadata["EventType"] rather than trusting a possibly-absent
    "y_true" key -- this keeps the three models' extraction code identical).
"""
import gc
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def print_markdown_table(df: pd.DataFrame):
    """Minimal markdown table printer (avoids a 'tabulate' dependency)."""
    cols = list(df.columns)
    widths = [max(len(str(c)), *(len(str(v)) for v in df[c])) for c in cols]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    print(header)
    print(sep)
    for _, row in df.iterrows():
        print("| " + " | ".join(str(row[c]).ljust(w) for c, w in zip(cols, widths)) + " |")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "data" / "results"
OUT_DIR = ROOT / "out_thesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POS_LABEL = "PinchLog"
NEG_LABEL = "NoPinchLog"

RUN_FILES = {
    "RF": "rf_leave_one_out_run.pkl",
    "EEGNet": "eegnet_leave_one_out_run.pkl",
    "Transformer": "transformer_leave_one_out_run.pkl",
}


def load_run(name):
    path = RESULTS_DIR / name
    with open(path, "rb") as fh:
        cached = pickle.load(fh)
    return cached["result"]


def extract_fold_arrays(fold_dict):
    """Given result[subject] = {"result": ..., "test_metadata": ...}, return
    (subject_order_of_rows_is_preserved) y_true, y_pred as numpy arrays of strings."""
    meta = fold_dict["test_metadata"]
    y_true = meta["EventType"].to_numpy()
    y_pred = np.asarray(fold_dict["result"]["y_pred"])
    assert len(y_true) == len(y_pred), "misaligned fold: y_true/y_pred length mismatch"
    return y_true, y_pred


def fold_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == POS_LABEL) & (y_pred == POS_LABEL)))
    fn = int(np.sum((y_true == POS_LABEL) & (y_pred == NEG_LABEL)))
    fp = int(np.sum((y_true == NEG_LABEL) & (y_pred == POS_LABEL)))
    tn = int(np.sum((y_true == NEG_LABEL) & (y_pred == NEG_LABEL)))
    n = tp + fn + fp + tn
    acc = (tp + tn) / n
    majority = max(np.sum(y_true == POS_LABEL), np.sum(y_true == NEG_LABEL)) / n
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, n=n, acc=acc, chance=majority)


def main():
    per_model_fold_rows = []  # subject, model, acc, chance, tp, fn, fp, tn, n
    per_model_pooled = {}

    for model_name, fname in RUN_FILES.items():
        print(f"Loading {fname} ...")
        result = load_run(fname)
        subjects = list(result.keys())

        pooled = dict(tp=0, fn=0, fp=0, tn=0, n=0)
        for subj in subjects:
            y_true, y_pred = extract_fold_arrays(result[subj])
            m = fold_metrics(y_true, y_pred)
            per_model_fold_rows.append(dict(model=model_name, subject=subj, **m))
            for k in ("tp", "fn", "fp", "tn", "n"):
                pooled[k] += m[k]

        per_model_pooled[model_name] = pooled

        del result
        gc.collect()
        print(f"  done, {len(subjects)} folds, pooled n={pooled['n']}")

    fold_df = pd.DataFrame(per_model_fold_rows)
    fold_df.to_csv(OUT_DIR / "per_fold_accuracy_loso.csv", index=False)
    print(f"Wrote {OUT_DIR / 'per_fold_accuracy_loso.csv'}")

    # ---------------------------------------------------------------- #
    # Task 1: pooled precision/recall/F1/FPR + per-fold acc mean/std
    # ---------------------------------------------------------------- #
    rows = []
    for model_name in RUN_FILES:
        sub = fold_df[fold_df.model == model_name]
        acc_mean = sub["acc"].mean()
        # ddof=0 (population std) to match the reference numbers quoted in the task brief
        # (EEGNet base LOSO: mean 0.6449, std 0.0573) which were computed with np.std default ddof=0
        acc_std = sub["acc"].std(ddof=0)
        chance_mean = sub["chance"].mean()

        p = per_model_pooled[model_name]
        tp, fn, fp, tn, n = p["tp"], p["fn"], p["fp"], p["tn"], p["n"]
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        pooled_acc = (tp + tn) / n

        rows.append(dict(
            model=model_name,
            accuracy_mean=acc_mean, accuracy_std=acc_std,
            pooled_accuracy=pooled_acc,
            precision=precision, recall=recall, f1=f1, fpr=fpr,
            chance_mean=chance_mean,
            n_pooled=n, tp=tp, fn=fn, fp=fp, tn=tn,
        ))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_DIR / "model_metrics_loso.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'model_metrics_loso.csv'}\n")

    print("### Model metrics (pooled across 14 LOSO folds, positive class = PinchLog)\n")
    md_cols = ["model", "accuracy_mean", "accuracy_std", "pooled_accuracy",
               "precision", "recall", "f1", "fpr", "chance_mean"]
    md = metrics_df[md_cols].copy()
    for c in md_cols[1:]:
        md[c] = md[c].map(lambda x: f"{x:.4f}")
    print_markdown_table(md)

    # ---------------------------------------------------------------- #
    # Task 2: paired Wilcoxon signed-rank tests across the 14 folds
    # ---------------------------------------------------------------- #
    pivot = fold_df.pivot(index="subject", columns="model", values="acc")
    chance_pivot = fold_df.pivot(index="subject", columns="model", values="chance")
    # sanity: same subject order everywhere, same n=14
    assert pivot.shape[0] == 14, f"expected 14 subjects, got {pivot.shape[0]}"

    sig_rows = []
    model_names = list(RUN_FILES.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a, b = model_names[i], model_names[j]
            xa, xb = pivot[a].values, pivot[b].values
            diff = xa - xb
            if np.allclose(diff, 0):
                stat, p = float("nan"), 1.0
            else:
                stat, p = wilcoxon(xa, xb, zero_method="wilcox")
            sig_rows.append(dict(
                comparison=f"{a} vs {b}", statistic=stat, p_value=p,
                mean_diff=diff.mean(), median_diff=np.median(diff),
                n_folds=len(diff),
            ))

    # each model vs its own per-fold chance baseline (paired, same held-out subject)
    for m in model_names:
        xa = pivot[m].values
        xc = chance_pivot[m].values
        diff = xa - xc
        if np.allclose(diff, 0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(xa, xc, zero_method="wilcox")
        sig_rows.append(dict(
            comparison=f"{m} vs chance", statistic=stat, p_value=p,
            mean_diff=diff.mean(), median_diff=np.median(diff),
            n_folds=len(diff),
        ))

    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(OUT_DIR / "model_significance_loso.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'model_significance_loso.csv'}\n")

    print("### Paired Wilcoxon signed-rank tests (n=14 held-out subjects)\n")
    sig_md = sig_df.copy()
    for c in ("statistic", "p_value", "mean_diff", "median_diff"):
        sig_md[c] = sig_md[c].map(lambda x: f"{x:.4g}")
    print_markdown_table(sig_md)


if __name__ == "__main__":
    main()
