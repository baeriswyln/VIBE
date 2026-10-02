"""
Task 3 for the thesis: feedback-condition breakdown of EEGNet's out-of-fold
LOSO detection performance, restricted to true PinchLog (pinch-positive) epochs.

Framing (documented explicitly per the task brief): the per-epoch feedback/task
metadata columns (hasVisualFeedback, hasAuditoryFeedback, hasVirtualHand,
feedbackIsDelayed, pinchIsHinted, isChoiceStimuli, numberOfStimuli) are only
populated (non-null) for PinchLog epochs -- NoPinchLog epochs have all of
them set to None/NaN in this dataset (verified directly, see thesis_analysis/_inspect.py
and the ad-hoc check run during development). So a negative-class (NoPinchLog)
false-positive-rate breakdown by these same flags is NOT possible -- there is no
per-condition metadata to group NoPinchLog epochs by. We therefore report, for
each condition flag, the pinch-DETECTION RATE (= recall for the PinchLog class,
which is also the raw accuracy on this PinchLog-only subset since only two
outcomes are possible: predicted PinchLog [correct] or predicted NoPinchLog
[missed]) for flag=True vs flag=False, i.e. "given a real pinch happened under
condition X, how often did EEGNet correctly detect it".

LOSO out-of-fold aggregation: each of the 14 EEGNet LOSO folds' test_metadata
(PinchLog subset) + y_pred is concatenated. Since LOSO test sets partition the
14 subjects exactly once, every PinchLog epoch in the dataset appears exactly
once in this pooled table, predicted by the model that did NOT see its subject
during training -- the standard LOSO "out-of-fold prediction" definition.

Reads data/results/eegnet_leave_one_out_run.pkl (read-only).
Writes:
  out_thesis/feedback_condition_breakdown.csv
  out_thesis/feedback_condition_breakdown.md
  out_thesis/figs/feedback_condition_accuracy.pdf
"""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "data" / "results"
OUT_DIR = ROOT / "out_thesis"
FIG_DIR = OUT_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

POS_LABEL = "PinchLog"

# boolean/tri-state condition flags to break down by (all object-dtype columns
# with True/False/None in the metadata)
BOOL_FLAGS = [
    "hasVisualFeedback",
    "hasAuditoryFeedback",
    "hasVirtualHand",
    "feedbackIsDelayed",
    "pinchIsHinted",
    "isChoiceStimuli",
]


def load_oof_pinch_table():
    with open(RESULTS_DIR / "eegnet_leave_one_out_run.pkl", "rb") as fh:
        cached = pickle.load(fh)
    result = cached["result"]

    rows = []
    for subj, fold in result.items():
        meta = fold["test_metadata"]
        y_pred = np.asarray(fold["result"]["y_pred"])
        assert len(meta) == len(y_pred)
        sub = meta.copy()
        sub["y_pred"] = y_pred
        sub["subject_fold"] = subj
        rows.append(sub)
    full = pd.concat(rows, axis=0, ignore_index=True)

    pinch = full[full["EventType"] == POS_LABEL].copy()
    pinch["detected"] = pinch["y_pred"] == POS_LABEL
    return pinch, full


def main():
    pinch, full = load_oof_pinch_table()
    n_total_pinch = len(pinch)
    n_subjects = pinch["subject_fold"].nunique()
    overall_recall = pinch["detected"].mean()
    print(f"Out-of-fold PinchLog epochs (all 14 folds pooled): {n_total_pinch} "
          f"from {n_subjects} subjects; overall detection rate (recall) = {overall_recall:.4f}")
    rows = []
    for flag in BOOL_FLAGS:
        sub = pinch[pinch[flag].notna()].copy()
        sub[flag] = sub[flag].astype(bool)
        if sub[flag].nunique() < 2:
            print(f"[skip] {flag}: fewer than 2 distinct non-null values in this subset")
            continue

        grp = sub.groupby(flag)["detected"].agg(["sum", "count"])
        # grp indexed by False/True
        n_true = int(grp.loc[True, "count"]) if True in grp.index else 0
        n_false = int(grp.loc[False, "count"]) if False in grp.index else 0
        det_true = int(grp.loc[True, "sum"]) if True in grp.index else 0
        det_false = int(grp.loc[False, "sum"]) if False in grp.index else 0
        rate_true = det_true / n_true if n_true else float("nan")
        rate_false = det_false / n_false if n_false else float("nan")

        # 2x2 contingency: rows = flag True/False, cols = detected/missed
        table = [[det_true, n_true - det_true], [det_false, n_false - det_false]]
        odds_ratio, p_value = fisher_exact(table)

        rows.append(dict(
            condition=flag,
            n_true=n_true, detected_true=det_true, detection_rate_true=rate_true,
            n_false=n_false, detected_false=det_false, detection_rate_false=rate_false,
            diff_true_minus_false=rate_true - rate_false,
            fisher_odds_ratio=odds_ratio, fisher_p_value=p_value,
        ))

    # numberOfStimuli: not boolean -- report mean detection rate by its distinct values instead,
    # plus a Mann-Whitney comparing numberOfStimuli between detected vs missed pinches.
    from scipy.stats import mannwhitneyu
    ns = pinch[pinch["numberOfStimuli"].notna()].copy()
    if ns["numberOfStimuli"].nunique() > 1:
        detected_vals = ns.loc[ns["detected"], "numberOfStimuli"]
        missed_vals = ns.loc[~ns["detected"], "numberOfStimuli"]
        u_stat, u_p = mannwhitneyu(detected_vals, missed_vals, alternative="two-sided")
        print(f"\nnumberOfStimuli: detected mean={detected_vals.mean():.3f} (n={len(detected_vals)}), "
              f"missed mean={missed_vals.mean():.3f} (n={len(missed_vals)}), "
              f"Mann-Whitney U={u_stat:.1f}, p={u_p:.4g}")
        by_val = ns.groupby("numberOfStimuli")["detected"].agg(["mean", "count"])
        print(by_val)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "feedback_condition_breakdown.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'feedback_condition_breakdown.csv'}\n")

    # ---- markdown summary ----
    lines = []
    lines.append("# Feedback-condition breakdown (EEGNet, base LOSO, out-of-fold, PinchLog epochs only)\n")
    lines.append(f"Pooled out-of-fold PinchLog epochs: {n_total_pinch} across {n_subjects} held-out-subject folds. "
                  f"Overall detection rate (recall): {overall_recall:.4f}.\n")
    lines.append("Framing: NoPinchLog epochs carry no feedback/task metadata (all null) in this dataset, "
                  "so this breakdown is restricted to PinchLog epochs and reports the detection rate "
                  "(fraction correctly classified as PinchLog) for flag=True vs flag=False, with a "
                  "Fisher's exact test on the underlying 2x2 (flag x detected/missed) contingency table.\n")
    lines.append("| condition | rate (True) | n (True) | rate (False) | n (False) | diff (T-F) | odds ratio | p-value |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['condition']} | {r['detection_rate_true']:.4f} | {r['n_true']} | "
                      f"{r['detection_rate_false']:.4f} | {r['n_false']} | {r['diff_true_minus_false']:+.4f} | "
                      f"{r['fisher_odds_ratio']:.3f} | {r['fisher_p_value']:.4g} |")
    md_text = "\n".join(lines)
    (OUT_DIR / "feedback_condition_breakdown.md").write_text(md_text, encoding="utf-8")
    print(md_text)

    # ---------------------------------------------------------------- #
    # Figure: single-column bar chart, detection rate by condition (True vs False)
    # ---------------------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import paper_style  # paper style shared with the notebooks (sizes/font in paper_style.py)

    labels_map = {
        "hasVisualFeedback": "Visual FB",
        "hasAuditoryFeedback": "Auditory FB",
        "hasVirtualHand": "Virtual hand",
        "feedbackIsDelayed": "FB delayed",
        "pinchIsHinted": "Pinch hinted",
        "isChoiceStimuli": "Choice trial",
    }
    plot_df = df.copy()
    plot_df["label"] = plot_df["condition"].map(labels_map)

    fig, ax = plt.subplots(figsize=(paper_style.COL_W, 2.4))
    x = np.arange(len(plot_df))
    width = 0.35
    bars_f = ax.bar(x - width / 2, plot_df["detection_rate_false"], width, label="False", color="#9aa5b1")
    bars_t = ax.bar(x + width / 2, plot_df["detection_rate_true"], width, label="True", color="#2f6fed")

    # significance stars
    for i, (_, r) in enumerate(plot_df.iterrows()):
        if r["fisher_p_value"] < 0.05:
            y = max(r["detection_rate_true"], r["detection_rate_false"]) + 0.03
            star = "**" if r["fisher_p_value"] < 0.01 else "*"
            ax.text(i, y, star, ha="center", va="bottom", fontsize=8)

    ax.axhline(overall_recall, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_ylabel("Pinch detection rate")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=35, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(title=None, frameon=False, loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig_path = FIG_DIR / "feedback_condition_accuracy.pdf"
    fig.savefig(fig_path)
    print(f"\nWrote {fig_path}")


if __name__ == "__main__":
    main()
