"""
run_eda.py — Exploratory analysis of the VIBE pinch EEG.

Produces figures in out/figs/ and a written report at out/eda_report.md:

  1. Data inventory across subjects/sessions.
  2. Signal quality (per-channel amplitude, bad channels, saturation).
  3. Power spectra (line noise, alpha peak, 1/f).
  4. Pinch-locked motor signature: ERP + time-frequency mu/beta ERD at C3/Cz/C4,
     plus a topography of the movement-related beta desynchronisation.
  5. Gaze / pupil / blink dynamics around the pinch.

Run:  uv run python -m pinch_analysis.run_eda
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from . import dataset as ds
from .epoching import prepare_session, pinch_epochs
from .preprocess import detect_bad_channels, SCALP_CHANNELS

def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


OUT = ds._ROOT / "out"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# colourblind-safe trio for C3 / Cz / C4
CCOL = {"C3": "#0072B2", "Cz": "#009E73", "C4": "#D55E00"}
plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})


def inventory() -> pd.DataFrame:
    rows = []
    for sess in ds.all_sessions():
        eeg_ts = pd.read_csv(sess.eeg_csv, usecols=[0]).iloc[:, 0].to_numpy()
        mk = ds.parse_markers(sess)
        pinch = ds.merge_pinch_select(mk["pinch"], mk["select"])
        by_exp = pinch.groupby("experiment").size().to_dict()
        rows.append({
            "subject": sess.subject, "stamp": sess.stamp,
            "eeg_dur_s": round(eeg_ts[-1] - eeg_ts[0], 1),
            "n_eeg": len(eeg_ts),
            "fs_eff": round((len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0]), 2),
            "n_pinch": len(pinch),
            "n_select": int(pinch["produced_select"].sum()),
            "n_myo": int(pinch["is_myo"].sum()),
            **{f"exp_{k}": v for k, v in by_exp.items()},
        })
    return pd.DataFrame(rows)


def fig_signal_quality(prep, report_lines):
    """Per-channel amplitude on the raw (pre-clean) EEG, example subject."""
    raw_full, ts = ds.load_eeg(prep.session)
    scalp = [c for c in SCALP_CHANNELS if c in raw_full.ch_names]
    data = raw_full.get_data(picks=scalp) * 1e6
    std = data.std(axis=1)
    bads = detect_bad_channels(raw_full, scalp)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    colors = ["#D55E00" if c in bads["all"] else "#0072B2" for c in scalp]
    ax.bar(range(len(scalp)), std, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(scalp)))
    ax.set_xticklabels(scalp, rotation=90)
    ax.set_ylabel("per-channel std (µV, log)")
    ax.set_title(f"{prep.session.subject}: raw amplitude "
                 f"(orange = flagged bad: {bads['all']})")

    # a 6-second slice of cleaned motor channels
    ax = axes[1]
    seg = prep.raw.copy().pick(["C3", "Cz", "C4"]).get_data(
        start=int(300 * 300), stop=int(306 * 300)) * 1e6
    t = np.arange(seg.shape[1]) / 300.0
    for i, ch in enumerate(["C3", "Cz", "C4"]):
        ax.plot(t, seg[i] + i * 40, color=CCOL[ch], lw=0.7, label=ch)
    ax.set_xlabel("time (s)"); ax.set_ylabel("µV (offset)")
    ax.set_title("cleaned motor channels, 6 s slice")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "01_signal_quality.png"); plt.close(fig)

    report_lines.append(
        f"- **{prep.session.subject}** raw channel std ranged "
        f"{std.min():.1f}–{std.max():.0f} µV; flagged bad "
        f"(flat/saturating/noisy): {bads}.")


def fig_psd(preps, report_lines):
    """Welch PSD of cleaned scalp EEG, averaged over sessions."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    all_psd = []
    freqs = None
    for prep in preps:
        spec = prep.raw.compute_psd(method="welch", fmin=1, fmax=45,
                                    n_fft=2048, verbose="ERROR")
        psd, freqs = spec.get_data(return_freqs=True)
        all_psd.append(psd.mean(axis=0) * 1e12)  # V²/Hz -> µV²/Hz
    m = np.mean(all_psd, axis=0)
    sd = np.std(all_psd, axis=0)
    db = 10 * np.log10(m)
    ax.plot(freqs, db, color="#0072B2", lw=1.5, label="mean over sessions")
    lo = np.clip(m - sd, 1e-6 * m.max(), None)
    ax.fill_between(freqs, 10 * np.log10(lo), 10 * np.log10(m + sd),
                    color="#0072B2", alpha=0.15)
    for f, lab in [(10, "α (~10 Hz)"), (50, "mains")]:
        if f <= 45:
            ax.axvline(f, color="#999", ls="--", lw=0.8)
            ax.text(f, ax.get_ylim()[1], lab, fontsize=8, va="top")
    # find alpha peak
    aband = (freqs >= 7) & (freqs <= 13)
    apk = freqs[aband][np.argmax(m[aband])]
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("PSD (dB µV²/Hz)")
    ax.set_title(f"Cleaned scalp PSD (mean of {len(preps)} sessions); "
                 f"alpha peak ≈ {apk:.1f} Hz")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "02_psd.png"); plt.close(fig)
    report_lines.append(f"- Grand-mean alpha peak ≈ **{apk:.1f} Hz**; "
                        f"1 Hz–40 Hz band, 50 Hz notch applied.")


def fig_motor(all_epochs, report_lines):
    """ERP + time-frequency ERD at C3/Cz/C4, and a beta-ERD topography."""
    ep = all_epochs
    tmin = ep.tmin
    picks = [c for c in ["C3", "Cz", "C4"] if c in ep.ch_names]

    # ---- time-domain average (movement-related potential) ----
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
    axerp = fig.add_subplot(gs[0, 0])
    ev = ep.copy().apply_baseline((tmin, tmin + 0.5))  # baseline at window start
    avg = ev.average()
    times = avg.times
    for ch in picks:
        y = avg.get_data(picks=ch)[0] * 1e6
        axerp.plot(times, y, color=CCOL[ch], label=ch, lw=1.3)
    axerp.axvline(0, color="k", lw=0.8)
    axerp.set_title("Pinch-locked average (baseline = window start)")
    axerp.set_xlabel("time from pinch (s)"); axerp.set_ylabel("µV")
    axerp.legend(fontsize=8)

    # ---- time-frequency (Morlet) mu/beta ERD ----
    freqs = np.arange(4, 31, 1.0)
    n_cycles = freqs / 2.0
    tfr = ep.compute_tfr("morlet", freqs=freqs, n_cycles=n_cycles,
                         return_itc=False, average=True, decim=3,
                         verbose="ERROR")
    tfr.apply_baseline((tmin, tmin + 0.5), mode="logratio")

    for j, ch in enumerate(picks):
        axtf = fig.add_subplot(gs[1, j])
        d = tfr.get_data(picks=ch)[0]  # (freq, time)
        im = axtf.pcolormesh(tfr.times, freqs, d, cmap="RdBu_r",
                             vmin=-0.5, vmax=0.5, shading="auto")
        axtf.axvline(0, color="k", lw=0.8)
        axtf.axhline(13, color="k", lw=0.4, ls=":")
        axtf.set_title(f"{ch}: mu/beta ERD (log-ratio)")
        axtf.set_xlabel("time from pinch (s)")
        if j == 0:
            axtf.set_ylabel("frequency (Hz)")
        fig.colorbar(im, ax=axtf, fraction=0.046, pad=0.04)

    # ---- beta-band ERD trace + topography ----
    axb = fig.add_subplot(gs[0, 1])
    beta = (freqs >= 13) & (freqs <= 30)
    for ch in picks:
        d = tfr.get_data(picks=ch)[0][beta].mean(axis=0)
        axb.plot(tfr.times, d, color=CCOL[ch], label=ch, lw=1.3)
    axb.axvline(0, color="k", lw=0.8); axb.axhline(0, color="#999", lw=0.6)
    axb.set_title("Beta (13–30 Hz) power change")
    axb.set_xlabel("time from pinch (s)"); axb.set_ylabel("log-ratio")
    axb.legend(fontsize=8)

    # topomap of beta ERD in [-0.5, 0] s (just before pinch)
    axtopo = fig.add_subplot(gs[0, 2])
    tmask = (tfr.times >= -0.5) & (tfr.times <= 0.0)
    topo = tfr.get_data()[:, beta][:, :, tmask].mean(axis=(1, 2))
    mne.viz.plot_topomap(topo, tfr.info, axes=axtopo, show=False,
                         cmap="RdBu_r", vlim=(-0.3, 0.3), contours=4)
    axtopo.set_title("Beta ERD topo, [-0.5,0] s")

    fig.suptitle(f"Motor signature around pinch onset "
                 f"(N={len(ep)} epochs, {len(picks)} channels)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "03_pinch_motor.png"); plt.close(fig)

    # numeric summary: min beta log-ratio near movement at C3/C4
    for ch in picks:
        d = tfr.get_data(picks=ch)[0][beta].mean(axis=0)
        near = (tfr.times >= -0.5) & (tfr.times <= 0.5)
        report_lines.append(
            f"- {ch}: beta power change min = **{d[near].min():+.2f}** "
            f"log-ratio near movement (negative = desynchronisation/ERD).")


def fig_gaze(preps, report_lines):
    """Pupil, eye openness, and gaze-direction speed around pinch onset."""
    tmin, tmax = -2.0, 1.0
    grid = np.arange(tmin, tmax, 1 / 60.0)
    acc = {k: [] for k in ["left_pupil_diameter", "left_eye_openness"]}
    speed = []
    n_used = 0
    for prep in preps:
        if not ds.has_gaze(prep.session):
            continue
        n_used += 1
        gaze = ds.load_gaze(prep.session)
        pinch_t = prep.pinch["timestamp"].to_numpy()
        gon = ds.gaze_on_grid(gaze, np.linspace(
            prep.eeg_ts[0], prep.eeg_ts[-1], 0) if False else
            np.arange(prep.eeg_ts[0], prep.eeg_ts[-1], 1 / 60.0))
        gt = gon.index.to_numpy()
        for ch in acc:
            y = gon[ch].to_numpy()
            for pt in pinch_t:
                sl = np.interp(pt + grid, gt, y)
                acc[ch].append(sl)
        # gaze angular speed
        dx = np.gradient(gon["left_gaze_direction_x"].to_numpy())
        dy = np.gradient(gon["left_gaze_direction_y"].to_numpy())
        sp = np.sqrt(dx ** 2 + dy ** 2) * 60.0
        for pt in pinch_t:
            speed.append(np.interp(pt + grid, gt, sp))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    def plot_mean(ax, arr, title, ylab, color):
        arr = np.array(arr)
        m = np.nanmean(arr, axis=0); se = np.nanstd(arr, axis=0) / np.sqrt(len(arr))
        ax.plot(grid, m, color=color, lw=1.5)
        ax.fill_between(grid, m - se, m + se, color=color, alpha=0.2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(title); ax.set_xlabel("time from pinch (s)"); ax.set_ylabel(ylab)
    plot_mean(axes[0], acc["left_pupil_diameter"], "Pupil diameter", "mm", "#0072B2")
    plot_mean(axes[1], acc["left_eye_openness"], "Eye openness", "norm", "#009E73")
    plot_mean(axes[2], speed, "Gaze angular speed", "a.u./s", "#D55E00")
    fig.suptitle(f"Gaze / pupil dynamics around pinch "
                 f"(mean ± SE, {n_used} sessions with gaze)")
    fig.tight_layout(); fig.savefig(FIGS / "04_gaze.png"); plt.close(fig)

    sp = np.array(speed)
    pre = np.nanmean(sp[:, (grid >= -1.0) & (grid < -0.3)])
    at = np.nanmean(sp[:, (grid >= -0.15) & (grid < 0.15)])
    report_lines.append(
        f"- Gaze angular speed drops from {pre:.3f} (pre) to {at:.3f} (at pinch), "
        f"consistent with fixation before selection.")


def fig_timeline(prep, report_lines):
    fig, ax = plt.subplots(figsize=(12, 3))
    blocks = prep.blocks
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(blocks.itertuples()):
        ax.barh(0, r.t_end - r.t_start, left=r.t_start, height=0.6,
                color=cmap(i % 20), alpha=0.7)
        ax.text((r.t_start + r.t_end) / 2, 0.35, str(r.experiment),
                rotation=90, ha="center", va="bottom", fontsize=7)
    pt = prep.pinch["timestamp"].to_numpy()
    ax.plot(pt, np.zeros_like(pt), "|", color="k", markersize=8)
    ax.set_yticks([]); ax.set_xlabel("LSL time (s)")
    ax.set_title(f"{prep.session.subject}: experiment blocks & pinch onsets (ticks)")
    fig.tight_layout(); fig.savefig(FIGS / "05_timeline.png"); plt.close(fig)


def main():
    report = ["# EEG Exploratory Data Analysis — VIBE pinch experiment\n"]

    print("== Inventory ==")
    inv = inventory()
    inv.to_csv(OUT / "inventory.csv", index=False)
    print(inv.to_string())
    report.append("## 1. Data inventory\n")
    report.append(df_to_md(inv))
    report.append("")

    # Prepare all sessions once (cleaned, scalp-only) for pooled analyses.
    print("\n== Preparing sessions ==")
    preps = []
    for sess in ds.all_sessions():
        print("  ", sess)
        preps.append(prepare_session(sess))

    report.append("\n## 2. Signal quality & spectra\n")
    example = next(i for i, p in enumerate(preps)
                   if p.session.subject == "subject_8")
    fig_signal_quality(preps[example], report)   # subject_8 as example
    fig_timeline(preps[example], report)
    fig_psd(preps, report)

    # Pooled pinch epochs for the motor analysis.
    print("== Building pooled pinch epochs ==")
    ep_list = []
    for prep in preps:
        e = pinch_epochs(prep, tmin=-2.0, tmax=1.0)
        if len(e):
            ep_list.append(e)
    pooled = mne.concatenate_epochs(ep_list, verbose="ERROR")
    print("pooled epochs:", len(pooled))
    report.append("\n## 3. Motor signature around the pinch\n")
    report.append(f"Pooled pinch epochs: **{len(pooled)}** "
                  f"(from {len(ep_list)} sessions).")
    fig_motor(pooled, report)

    report.append("\n## 4. Gaze / pupil around the pinch\n")
    fig_gaze(preps, report)

    (OUT / "eda_report.md").write_text("\n".join(str(x) for x in report))
    print("\nWrote", OUT / "eda_report.md", "and figures to", FIGS)


if __name__ == "__main__":
    main()
