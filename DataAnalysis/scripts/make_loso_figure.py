"""Headline LOSO model-comparison bar chart for the thesis (single-column figure).
Reads out/model_metrics_loso.csv (produced by model_metrics_significance.py)
so the numbers plotted are always the same ones reported in the text."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import paper_style  # noqa: E402  paper style shared with the notebooks

root = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(root / "out" / "model_metrics_loso.csv")))
order = ["RF", "EEGNet", "Transformer"]
rows = sorted(rows, key=lambda r: order.index(r["model"]))
labels = {"RF": "EEGForest\n(RF)", "EEGNet": "EEGNet", "Transformer": "GAT-GRU-\nTransformer"}

means = [float(r["accuracy_mean"]) for r in rows]
stds = [float(r["accuracy_std"]) for r in rows]
chance = float(rows[0]["chance_mean"])

fig, ax = plt.subplots(figsize=(paper_style.COL_W, 2.3))
x = range(len(rows))
bars = ax.bar(x, means, yerr=stds, capsize=3, width=0.6,
              color=["#8899aa", "#1f77b4", "#2ca02c"], edgecolor="black", linewidth=0.5)
ax.axhline(chance, color="firebrick", linestyle="--", linewidth=1, label=f"chance ({chance:.2f})")
ax.set_xticks(list(x))
ax.set_xticklabels([labels[r["model"]] for r in rows])
ax.set_ylabel("LOSO accuracy")
ax.set_ylim(0.4, 0.8)
for xi, m, s in zip(x, means, stds):  # above the error bar, not on it
    ax.text(xi, m + s + 0.01, f"{m:.3f}", ha="center")
ax.legend(loc="upper left", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout(pad=0.3)

out = root / "out" / "figs" / "loso_model_comparison.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print("wrote", out)
