"""Per-subject x model LOSO accuracy bar chart (appendix, full-width figure).
Reads out/per_fold_accuracy_loso.csv (produced by model_metrics_significance.py)."""
import csv
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import paper_style  # noqa: E402  paper style shared with the notebooks
import numpy as np

root = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(root / "out" / "per_fold_accuracy_loso.csv")))

by_subject = defaultdict(dict)
chance = {}
for r in rows:
    by_subject[r["subject"]][r["model"]] = float(r["acc"])
    chance[r["subject"]] = float(r["chance"])

subjects = sorted(by_subject.keys(), key=lambda s: int(s.split("_")[1]))
models = ["RF", "EEGNet", "Transformer"]
colors = {"RF": "#8899aa", "EEGNet": "#1f77b4", "Transformer": "#2ca02c"}

x = np.arange(len(subjects))
width = 0.25

fig, ax = plt.subplots(figsize=(paper_style.TEXT_W, 2.6))
for i, m in enumerate(models):
    vals = [by_subject[s][m] for s in subjects]
    ax.bar(x + (i - 1) * width, vals, width, label=m, color=colors[m], edgecolor="black", linewidth=0.3)

chance_vals = [chance[s] for s in subjects]
ax.plot(x, chance_vals, "D", color="firebrick", markersize=3, label="chance (per-fold)")

ax.set_xticks(list(x))
ax.set_xticklabels([s.replace("subject_", "") for s in subjects])
ax.set_xlabel("Participant ID")
ax.set_ylabel("LOSO accuracy")
ax.set_ylim(0.4, 0.85)
ax.legend(loc="upper right", frameon=False, ncol=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout(pad=0.3)

out = root / "out" / "figs" / "persubject_accuracy.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print("wrote", out)
