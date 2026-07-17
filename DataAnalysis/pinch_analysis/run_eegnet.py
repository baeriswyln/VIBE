"""
run_eegnet.py — Per-subject EEGNet for pinch vs distractor hard negatives.

This is the motor-specific decoder suggested by the hard-negative analysis: the
one contrast where gaze is useless and only genuine movement‑related EEG can
separate the classes. We train **one EEGNet per subject** (pooling that subject's
sessions) to classify:

  * positive : a real pinch  (peri-onset window [-0.5, +0.5] s)
  * negative : a look at a *highlighted-negative* distractor stimulus
               (GAZE_ENTER + fixation dwell, same window)

Evaluation is within-subject stratified 5-fold, ensembled over 3 seeds to tame
the small-sample variance. For a fair head-to-head we train a Riemannian
tangent-space + logistic-regression model on the *identical* folds.

Data are scarce (≈35–70 distractor negatives per subject), so treat absolute
numbers cautiously; the question is whether a learned spatio-temporal filter
beats the classical baseline and chance on this hard contrast.

Run:  uv run python -m pinch_analysis.run_eegnet
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from . import dataset as ds
from .epoching import prepare_session, pinch_epochs, negative_epochs
from .eegnet import EEGNet

warnings.filterwarnings("ignore", category=RuntimeWarning)
torch.set_num_threads(8)
OUT = ds._ROOT / "out"
FIGS = OUT / "figs"

TMIN, TMAX = -0.5, 0.5
N_FOLDS = 5
N_SEEDS = 3
EPOCHS = 80
BATCH = 32


def subject_epochs(subject_sessions):
    """Pool a subject's sessions into (X, y) with peri-event windows."""
    Xs, ys = [], []
    for prep in subject_sessions:
        pos = pinch_epochs(prep, tmin=TMIN, tmax=TMAX)
        hard_t = ds.hard_negative_times(prep.markers, prep.pinch)
        neg = negative_epochs(prep, hard_t, tmin=TMIN, tmax=TMAX)
        if len(pos):
            Xs.append(pos.get_data(copy=False) * 1e6)  # uV
            ys.append(np.ones(len(pos), int))
        if neg is not None and len(neg):
            Xs.append(neg.get_data(copy=False) * 1e6)
            ys.append(np.zeros(len(neg), int))
    if not Xs:
        return None, None
    return np.concatenate(Xs), np.concatenate(ys)


def zscore(train, *others):
    """Per-channel z-score using train statistics."""
    m = train.mean(axis=(0, 2), keepdims=True)
    s = train.std(axis=(0, 2), keepdims=True) + 1e-7
    return tuple((a - m) / s for a in (train, *others))


def train_eegnet(Xtr, ytr, Xte, n_ch, n_t, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cpu"
    model = EEGNet(n_ch, n_t).to(dev)
    # class weights for imbalance
    cw = torch.tensor([len(ytr) / (2 * (ytr == c).sum()) for c in (0, 1)],
                      dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss(weight=cw)
    Xt = torch.tensor(Xtr[:, None], dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    model.train()
    n = len(Xt)
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = lossf(out, yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(Xte[:, None], dtype=torch.float32)),
                          dim=1)[:, 1].numpy()
    return p


def riemann_fold(Xtr, ytr, Xte):
    pipe = make_pipeline(Covariances("oas"), TangentSpace(), StandardScaler(),
                         LogisticRegression(max_iter=2000))
    # covariance on the broadband (1-40 Hz) window: captures both the slow
    # movement-related potential and the mu/beta ERD of a genuine pinch.
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1]


def evaluate_subject(X, y):
    n_ch, n_t = X.shape[1], X.shape[2]
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=0)
    oof_net = np.full(len(y), np.nan)
    oof_rie = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr_z, Xte_z = zscore(X[tr], X[te])
        # ensemble EEGNet over seeds
        preds = [train_eegnet(Xtr_z, y[tr], Xte_z, n_ch, n_t, s)
                 for s in range(N_SEEDS)]
        oof_net[te] = np.mean(preds, axis=0)
        oof_rie[te] = riemann_fold(X[tr], y[tr], X[te])
    def auc(oof):
        ok = ~np.isnan(oof)
        return roc_auc_score(y[ok], oof[ok]) if len(np.unique(y[ok])) == 2 else np.nan
    return auc(oof_net), auc(oof_rie)


def main():
    print("Preparing sessions (grouped by subject)...")
    by_subject: dict[str, list] = {}
    for sess in ds.all_sessions():
        prep = prepare_session(sess)
        by_subject.setdefault(sess.subject, []).append(prep)

    rows = []
    for subj, preps in by_subject.items():
        X, y = subject_epochs(preps)
        if X is None or (y == 0).sum() < N_FOLDS or (y == 1).sum() < N_FOLDS:
            print(f"  {subj}: too few epochs (pos={int((y==1).sum()) if y is not None else 0},"
                  f" neg={int((y==0).sum()) if y is not None else 0}) — skipped")
            continue
        auc_net, auc_rie = evaluate_subject(X, y)
        rows.append({"subject": subj, "n_pos": int((y == 1).sum()),
                     "n_hardneg": int((y == 0).sum()),
                     "eegnet_auc": auc_net, "riemann_auc": auc_rie})
        print(f"  {subj}: pos={int((y==1).sum())} hardneg={int((y==0).sum())} "
              f"EEGNet AUC={auc_net:.3f}  Riemann AUC={auc_rie:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "eegnet_results.csv", index=False)
    mean_net = res["eegnet_auc"].mean()
    mean_rie = res["riemann_auc"].mean()
    print(f"\nMean over subjects: EEGNet {mean_net:.3f}  Riemann {mean_rie:.3f}")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(res))
    ax.bar(x - 0.2, res["eegnet_auc"], 0.4, label="EEGNet", color="#0072B2")
    ax.bar(x + 0.2, res["riemann_auc"], 0.4, label="Riemann+LR", color="#E69F00")
    ax.axhline(0.5, color="#999", ls="--", label="chance")
    ax.axhline(mean_net, color="#0072B2", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.subject}\n(+{r.n_pos}/-{r.n_hardneg})"
                        for r in res.itertuples()], fontsize=8)
    ax.set_ylabel("within-subject ROC-AUC"); ax.set_ylim(0.3, 1.0)
    ax.set_title("Per-subject pinch vs distractor hard negatives\n"
                 f"EEGNet {mean_net:.2f} vs Riemann {mean_rie:.2f} (mean)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "09_eegnet.png"); plt.close(fig)
    print("Saved", OUT / "eegnet_results.csv", "and", FIGS / "09_eegnet.png")


if __name__ == "__main__":
    main()
