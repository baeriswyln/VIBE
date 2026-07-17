"""
run_eegnet_transfer.py — Cross-subject transfer EEGNet, richer negatives.

Two changes over ``run_eegnet.py``:

1. **More distractors / negatives.** The negative class now combines
   * **random** inter-pinch windows (task-block windows >= 2.5 s from any pinch), and
   * **hard** highlighted-negative distractor looks (GAZE_ENTER + dwell).
   Metrics are reported overall *and* split by negative subtype, so the honest
   "pinch vs distractor look" number stays visible.

2. **Transfer learning.** For each held-out subject we compare, on the *same*
   temporally-held-out test split of that subject:
   * **scratch**    — EEGNet trained only on the subject's calibration split
   * **zero-shot**  — EEGNet pretrained on the *other* subjects, applied as-is
   * **fine-tuned** — that pretrained net, fine-tuned on the calibration split
   This shows whether pretraining across people plus a little per-user calibration
   beats training from scratch — the practical way to cut calibration time.

Windows are peri-onset [-0.5, +0.5] s (the verification decoder). EEG is
z-scored per subject (held-out subject uses calibration-split stats only, so no
leakage). Single seed per condition to keep runtime bounded.

Run:  uv run python -m pinch_analysis.run_eegnet_transfer
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from . import dataset as ds
from .epoching import (prepare_session, pinch_epochs, negative_epochs,
                       sample_negative_times)
from .eegnet import EEGNet

warnings.filterwarnings("ignore", category=RuntimeWarning)
torch.set_num_threads(10)
OUT = ds._ROOT / "out"
FIGS = OUT / "figs"

TMIN, TMAX = -0.5, 0.5
RANDOM_RATIO = 1.0     # random negatives per positive
CALIB_FRAC = 0.5       # first half of a held-out subject = calibration
EPOCHS_PRE, EPOCHS_FT, EPOCHS_SC = 60, 40, 80
LR_PRE, LR_FT, LR_SC = 1e-3, 5e-4, 1e-3
BATCH = 32


def subject_data(preps):
    """Pool a subject's sessions -> X(uV), y, ntype, times (peri-onset windows)."""
    Xs, ys, nt, ts = [], [], [], []
    for prep in preps:
        pos = pinch_epochs(prep, tmin=TMIN, tmax=TMAX)
        if len(pos):
            Xs.append(pos.get_data(copy=False) * 1e6)
            ys.append(np.ones(len(pos), int))
            nt.append(np.array(["pos"] * len(pos), object))
            ts.append(pos.metadata["timestamp"].to_numpy())
        # hard negatives: distractor looks
        hard_t = ds.hard_negative_times(prep.markers, prep.pinch)
        hard = negative_epochs(prep, hard_t, tmin=TMIN, tmax=TMAX)
        if hard is not None and len(hard):
            ht = prep.eeg_ts[np.clip(hard.events[:, 0], 0, len(prep.eeg_ts) - 1)]
            Xs.append(hard.get_data(copy=False) * 1e6)
            ys.append(np.zeros(len(hard), int))
            nt.append(np.array(["hard"] * len(hard), object)); ts.append(ht)
        # random negatives: inter-pinch buffered windows
        n_rand = int(len(pos) * RANDOM_RATIO)
        rand_t = sample_negative_times(prep, n=n_rand, window=1.0, buffer=2.0,
                                       rng=11)
        rand = negative_epochs(prep, rand_t, tmin=TMIN, tmax=TMAX)
        if rand is not None and len(rand):
            rt = prep.eeg_ts[np.clip(rand.events[:, 0], 0, len(prep.eeg_ts) - 1)]
            Xs.append(rand.get_data(copy=False) * 1e6)
            ys.append(np.zeros(len(rand), int))
            nt.append(np.array(["rand"] * len(rand), object)); ts.append(rt)
    if not Xs:
        return None
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(nt), np.concatenate(ts))


def train(model, X, y, epochs, lr, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    cw = torch.tensor([len(y) / (2 * max(1, (y == c).sum())) for c in (0, 1)],
                      dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss(weight=cw)
    Xt = torch.tensor(X[:, None], dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    model.train()
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(
            model(torch.tensor(X[:, None], dtype=torch.float32)), 1)[:, 1].numpy()


def zscore_fit(X):
    return X.mean((0, 2), keepdims=True), X.std((0, 2), keepdims=True) + 1e-7


def auc_split(y, nt, scores):
    """AUC overall, pos-vs-hard, pos-vs-random."""
    def a(mask_neg):
        m = (y == 1) | mask_neg
        yy = y[m]
        return roc_auc_score(yy, scores[m]) if len(np.unique(yy)) == 2 else np.nan
    return a(y == 0), a(nt == "hard"), a(nt == "rand")


def main():
    print("Preparing sessions...")
    per_subj = {}
    for sess in ds.all_sessions():
        per_subj.setdefault(sess.subject, []).append(prepare_session(sess))
    data = {s: subject_data(p) for s, p in per_subj.items()}
    data = {s: d for s, d in data.items() if d is not None}
    n_ch, n_t = next(iter(data.values()))[0].shape[1:]
    for s, (X, y, nt, ts) in data.items():
        print(f"  {s}: pos={int((y==1).sum())} hard={(nt=='hard').sum()} "
              f"rand={(nt=='rand').sum()}")

    rows = []
    for held in data:
        # --- pretrain on the other subjects (each z-scored by its own stats) ---
        Xtr, ytr = [], []
        for s, (X, y, nt, ts) in data.items():
            if s == held:
                continue
            mu, sd = zscore_fit(X)
            Xtr.append((X - mu) / sd); ytr.append(y)
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
        pre = train(EEGNet(n_ch, n_t), Xtr, ytr, EPOCHS_PRE, LR_PRE)

        # --- held-out subject: temporal calib/test split ---
        X, y, nt, ts = data[held]
        order = np.argsort(ts)
        X, y, nt = X[order], y[order], nt[order]
        k = int(len(y) * CALIB_FRAC)
        # ensure both classes present in calib
        while k < len(y) and len(np.unique(y[:k])) < 2:
            k += 1
        Xc, yc, ntc = X[:k], y[:k], nt[:k]
        Xe, ye, nte = X[k:], y[k:], nt[k:]
        mu, sd = zscore_fit(Xc)  # calib stats only
        Xc_z, Xe_z = (Xc - mu) / sd, (Xe - mu) / sd
        if len(np.unique(ye)) < 2:
            continue

        # zero-shot (pretrained applied to held-out test)
        s_zero = predict(pre, Xe_z)
        # fine-tune the pretrained net on calibration
        ft = EEGNet(n_ch, n_t); ft.load_state_dict(pre.state_dict())
        ft = train(ft, Xc_z, yc, EPOCHS_FT, LR_FT)
        s_ft = predict(ft, Xe_z)
        # scratch (calibration only)
        sc = train(EEGNet(n_ch, n_t), Xc_z, yc, EPOCHS_SC, LR_SC)
        s_sc = predict(sc, Xe_z)

        for cond, sco in [("scratch", s_sc), ("zero-shot", s_zero),
                          ("fine-tuned", s_ft)]:
            ao, ah, ar = auc_split(ye, nte, sco)
            rows.append({"subject": held, "condition": cond,
                         "auc_all": ao, "auc_vs_hard": ah, "auc_vs_rand": ar,
                         "n_test": int(len(ye)),
                         "n_test_hard": int((nte == "hard").sum())})
        print(f"  {held}: scratch={rows[-3]['auc_all']:.3f} "
              f"zero-shot={rows[-2]['auc_all']:.3f} "
              f"fine-tuned={rows[-1]['auc_all']:.3f} "
              f"(hard: {rows[-3]['auc_vs_hard']:.2f}/{rows[-2]['auc_vs_hard']:.2f}/"
              f"{rows[-1]['auc_vs_hard']:.2f})")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "eegnet_transfer_results.csv", index=False)
    summ = res.groupby("condition")[["auc_all", "auc_vs_hard", "auc_vs_rand"]].mean()
    summ = summ.reindex(["scratch", "zero-shot", "fine-tuned"])
    print("\nMean over held-out subjects:")
    print(summ.to_string())

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    conds = ["scratch", "zero-shot", "fine-tuned"]
    colors = {"scratch": "#999999", "zero-shot": "#E69F00", "fine-tuned": "#0072B2"}
    ax = axes[0]
    subs = sorted(res["subject"].unique())
    x = np.arange(len(subs))
    for i, cond in enumerate(conds):
        vals = [res[(res.subject == s) & (res.condition == cond)]["auc_all"].mean()
                for s in subs]
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=cond, color=colors[cond])
    ax.axhline(0.5, color="#999", ls="--")
    ax.set_xticks(x); ax.set_xticklabels(subs, rotation=20, ha="right")
    ax.set_ylabel("held-out test ROC-AUC (all negatives)"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Transfer EEGNet per subject"); ax.legend(fontsize=8)

    ax = axes[1]
    metrics = ["auc_all", "auc_vs_rand", "auc_vs_hard"]
    labels = ["all neg", "vs random", "vs hard (distractor)"]
    xm = np.arange(len(metrics))
    for i, cond in enumerate(conds):
        vals = [summ.loc[cond, m] for m in metrics]
        ax.bar(xm + (i - 1) * 0.27, vals, 0.27, label=cond, color=colors[cond])
    ax.axhline(0.5, color="#999", ls="--")
    ax.set_xticks(xm); ax.set_xticklabels(labels)
    ax.set_ylabel("mean held-out ROC-AUC"); ax.set_ylim(0.4, 1.0)
    ax.set_title("By negative type (mean over subjects)"); ax.legend(fontsize=8)
    fig.suptitle("Cross-subject transfer: pretrain on others + fine-tune per user")
    fig.tight_layout(); fig.savefig(FIGS / "10_eegnet_transfer.png"); plt.close(fig)
    print("\nSaved", OUT / "eegnet_transfer_results.csv", "and",
          FIGS / "10_eegnet_transfer.png")


if __name__ == "__main__":
    main()
