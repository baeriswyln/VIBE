"""
run_regress.py — Predict *future* pinch events from EEG (+ gaze).

Instead of asking "is there a pinch now?", we slide a causal window of length W
across the task blocks and regress the **time-to-next-pinch (TTP)**:

    target(t) = (onset of next pinch at or after t) - t,   clipped to [0, HORIZON]

A window whose next pinch is further than HORIZON away is a "far" window
(target = HORIZON). The model therefore learns a continuous countdown that
should ramp down to 0 as a pinch approaches — i.e. it anticipates the pinch.

Features are extracted directly from the cleaned raw arrays (fast, thousands of
windows): band power + Hjorth per channel, optionally plus a gaze summary of the
same window. Evaluation is leave-one-subject-out. We report MAE, R² and Spearman
correlation, and a lead-time analysis: how early can an upcoming pinch be flagged?

Run:  uv run python -m pinch_analysis.run_regress
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from . import dataset as ds
from .epoching import prepare_session
from .features import bandpower_features

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = ds._ROOT / "out"
FIGS = OUT / "figs"

W = 1.5            # window length (s)
STEP = 0.1        # slide step (s)
HORIZON = 2.0      # cap on time-to-next-pinch (s)
GAZE_GRID_HZ = 60.0


def _gaze_arrays(prep):
    if not ds.has_gaze(prep.session):
        return None
    gaze = ds.load_gaze(prep.session)
    grid = np.arange(prep.eeg_ts[0], prep.eeg_ts[-1], 1 / GAZE_GRID_HZ)
    gon = ds.gaze_on_grid(gaze, grid)
    gt = gon.index.to_numpy()
    dx = np.gradient(gon["left_gaze_direction_x"].to_numpy())
    dy = np.gradient(gon["left_gaze_direction_y"].to_numpy())
    speed = np.sqrt(dx ** 2 + dy ** 2) * GAZE_GRID_HZ
    return {"gt": gt, "pupil": gon["left_pupil_diameter"].to_numpy(),
            "open": gon["left_eye_openness"].to_numpy(), "speed": speed,
            "speed_med": np.nanmedian(speed)}


def build_session(prep):
    """Return (Xbp, Xgaze, ttp, end_times) over sliding windows in task blocks."""
    sf = prep.raw.info["sfreq"]
    data = prep.raw.get_data() * 1e6  # (ch, n) µV
    ts = prep.eeg_ts
    wsamp = int(round(W * sf))
    pinch_t = np.sort(prep.pinch["timestamp"].to_numpy())
    ga = _gaze_arrays(prep)

    blocks = prep.blocks[~ds.is_excluded_block(prep.blocks["experiment"])]
    end_times = []
    for r in blocks.itertuples():
        t = r.t_start + W
        while t <= r.t_end:
            end_times.append(t)
            t += STEP
    end_times = np.array(end_times)
    if len(end_times) == 0:
        return None
    end_idx = ds.times_to_samples(end_times, ts)

    # windows -> band power features (batched)
    keep = end_idx >= wsamp
    end_times, end_idx = end_times[keep], end_idx[keep]
    wins = np.stack([data[:, i - wsamp:i] for i in end_idx])  # (n, ch, wsamp)
    Xbp = bandpower_features(wins, sf)

    # time-to-next-pinch
    nxt = np.searchsorted(pinch_t, end_times, side="left")
    ttp = np.full(len(end_times), HORIZON)
    valid = nxt < len(pinch_t)
    ttp[valid] = np.clip(pinch_t[nxt[valid]] - end_times[valid], 0, HORIZON)

    # gaze features over window
    Xg = np.full((len(end_times), 6), np.nan)
    if ga is not None:
        for k, t in enumerate(end_times):
            m = (ga["gt"] >= t - W) & (ga["gt"] <= t)
            if m.sum() >= 2:
                sp = ga["speed"][m]
                Xg[k] = [np.nanmean(ga["pupil"][m]), ga["pupil"][m][-1],
                         np.nanmean(ga["open"][m]), np.nanmean(sp), sp[-1],
                         np.mean(sp < ga["speed_med"])]
    return Xbp, Xg, ttp, end_times


def impute(X):
    X = X.copy()
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    idx = np.where(~np.isfinite(X))
    X[idx] = np.take(med, idx[1])
    return X


def loso(X, y, groups, model_factory):
    logo = LeaveOneGroupOut()
    oof = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def main():
    print("Preparing sessions...")
    preps = [prepare_session(s) for s in ds.all_sessions()]

    Xbp_all, Xg_all, y_all, grp_all, t_all, sidx_all = [], [], [], [], [], []
    pinch_by_session = []
    for si, prep in enumerate(preps):
        out = build_session(prep)
        if out is None:
            continue
        Xbp, Xg, ttp, endt = out
        Xbp_all.append(Xbp); Xg_all.append(Xg); y_all.append(ttp)
        grp_all.append([prep.session.subject] * len(ttp))
        t_all.append(endt)
        sidx_all.append(np.full(len(ttp), si))
        pinch_by_session.append((si, np.sort(prep.pinch["timestamp"].to_numpy())))
        print(f"  {prep.session.subject}: {len(ttp)} windows, "
              f"gaze={'yes' if ds.has_gaze(prep.session) else 'no'}")
    Xbp = np.concatenate(Xbp_all); Xg = np.concatenate(Xg_all)
    y = np.concatenate(y_all)
    groups = np.concatenate(grp_all)
    end_t = np.concatenate(t_all)
    sess_idx = np.concatenate(sidx_all)
    print(f"Total windows: {len(y)}  (mean TTP {y.mean():.2f}s)")

    feats = {
        "eeg": impute(Xbp),
        "eeg+gaze": impute(np.hstack([Xbp, Xg])),
        "gaze": impute(Xg),
    }
    models = {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "hgb": lambda: HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, random_state=0),
    }

    rows = []
    preds = {}
    for fname, X in feats.items():
        for mname, fac in models.items():
            oof = loso(X, y, groups, fac)
            ok = ~np.isnan(oof)
            mae = mean_absolute_error(y[ok], oof[ok])
            r2 = r2_score(y[ok], oof[ok])
            rho = spearmanr(y[ok], oof[ok]).statistic
            rows.append({"features": fname, "model": mname, "mae_s": mae,
                         "r2": r2, "spearman": rho, "n": int(ok.sum())})
            preds[(fname, mname)] = oof
            print(f"  {fname:10s} {mname:6s}  MAE={mae:.3f}s  R²={r2:.3f}  ρ={rho:.3f}")

    # baseline: predict constant mean TTP
    base_mae = mean_absolute_error(y, np.full_like(y, y.mean()))
    print(f"  baseline (mean) MAE={base_mae:.3f}s")

    # imminence detection: can -pred rank windows with a pinch in <=0.5 s?
    from sklearn.metrics import roc_auc_score
    print("  imminent-pinch (TTP<=0.5s) detection AUC from -predicted_TTP:")
    imminence = {}
    for (fname, mname), oof in preds.items():
        ok = ~np.isnan(oof)
        lab = (y[ok] <= 0.5).astype(int)
        if lab.sum() > 0:
            auc = roc_auc_score(lab, -oof[ok])
            imminence[(fname, mname)] = auc
            print(f"    {fname:10s} {mname:6s}  AUC={auc:.3f}")
    res = pd.DataFrame(rows)
    res["imminent_auc"] = [imminence.get((r.features, r.model), np.nan)
                           for r in res.itertuples()]
    res.to_csv(OUT / "regress_results.csv", index=False)

    # ---- figures ----
    best = res.loc[res.r2.idxmax()]
    oof = preds[(best.features, best.model)]
    ok = ~np.isnan(oof)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # (a) MAE bars
    ax = axes[0]
    order = res.sort_values("mae_s")
    labels = [f"{r.features}\n{r.model}" for r in order.itertuples()]
    ax.bar(range(len(order)), order.mae_s, color="#0072B2")
    ax.axhline(base_mae, color="#D55E00", ls="--", label=f"mean baseline {base_mae:.2f}s")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("LOSO MAE (s)"); ax.set_title("Time-to-pinch error"); ax.legend()

    # (b) predicted countdown vs true TTP (binned mean)
    ax = axes[1]
    bins = np.linspace(0, HORIZON, 17)
    bc = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(y[ok], bins) - 1
    mean_pred = [oof[ok][idx == b].mean() if np.any(idx == b) else np.nan
                 for b in range(len(bc))]
    ax.plot(bc, mean_pred, "o-", color="#009E73", label="mean prediction")
    ax.plot([0, HORIZON], [0, HORIZON], "--", color="#999", label="ideal")
    ax.set_xlabel("true time-to-pinch (s)"); ax.set_ylabel("predicted TTP (s)")
    ax.set_title(f"Calibration ({best.features}/{best.model})"); ax.legend()

    # (c) event-locked anticipation: mean predicted TTP vs time before a pinch
    ax = axes[2]
    off_bins = np.arange(-3.0, 0.55, 0.15)
    obc = 0.5 * (off_bins[:-1] + off_bins[1:])
    for fname, color in [("eeg", "#0072B2"), ("gaze", "#009E73"),
                         ("eeg+gaze", "#D55E00")]:
        oofm = preds[(fname, "hgb")]
        acc = [[] for _ in obc]
        for si, ptimes in pinch_by_session:
            m = sess_idx == si
            et = end_t[m]; pr = oofm[m]
            for pt in ptimes:
                off = et - pt
                sel = (off >= -3.0) & (off <= 0.5)
                for o, p in zip(off[sel], pr[sel]):
                    b = int((o - (-3.0)) / 0.15)
                    if 0 <= b < len(obc):
                        acc[b].append(p)
        mean_curve = [np.mean(a) if a else np.nan for a in acc]
        ax.plot(obc, mean_curve, "-", color=color, label=fname, lw=1.6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("time relative to pinch (s)")
    ax.set_ylabel("predicted TTP (s)")
    ax.set_title("Anticipation: predicted countdown\nlocked to real pinch (hgb)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "07_regress.png"); plt.close(fig)

    print("\nSaved", OUT / "regress_results.csv", "and", FIGS / "07_regress.png")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
