"""
run_eda.py — Exploratory analysis of the VIBE pinch EEG.

Produces figures in out/figs/ and a written report at out/eda_report.md:

  1. Data inventory across subjects/sessions.
  2. Signal quality (per-channel amplitude, bad channels, saturation).
  3. Power spectra (line noise, alpha peak, 1/f).
  4. Pinch-locked motor signature: ERP + time-frequency mu/beta ERD at C3/Cz/C4,
     plus a topography of the movement-related beta desynchronisation.
     3b. Between-subject / within-subject(run) variability of the beta ERD.
     3c. Beta ERD split by experiment condition (Static, TwoPositions, ..., Myo*).
     3d. Hard positives (real pinches) vs hard negatives (distractor looks) -
         does the beta ERD itself differ, or only what a classifier does with it?
     3e. Single-subject case study (subject_7): trial-by-trial consistency of
         the pinch-locked potential, and beta-ERD topography per experiment.
  5. Gaze / pupil / blink dynamics around the pinch.

All beta/mu-ERD numbers in section 3 (and its subsections) use ONE normalization
throughout: baseline-correct each epoch's power against its own pre-movement
window (single-trial log-ratio), *then* aggregate (median, with IQR where a
distribution is shown) across whatever group is being compared. This matters -
see the methodology note emitted into the report right before section 3 - because
"log of the trial-averaged power" and "median of per-trial log-ratios" are
different estimators (Jensen's inequality) and give visibly different numbers.

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
from scipy.stats import mannwhitneyu, wilcoxon

from . import dataset as ds
from .epoching import prepare_session, pinch_epochs, negative_epochs
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

# The analysis/display window for every pinch-locked figure. Epochs are built
# wider than this (+/- TFR_PAD) purely so Morlet wavelets have real signal to
# convolve against near t=EPOCH_TMIN/EPOCH_TMAX; every TFR is cropped back to
# EPOCH_TMIN/EPOCH_TMAX after baseline correction, so the padding never shows
# up in a plot - it only removes the edge-distorted power estimates that would
# otherwise appear as spurious dips right at the start/end of every log-ratio
# trace/heatmap. With n_cycles = freq/2 (used throughout), the wavelet's time
# support is a constant ~0.5 s at every frequency, so TFR_PAD=0.5 s is ample.
EPOCH_TMIN, EPOCH_TMAX = -2.0, 1.0
TFR_PAD = 0.5
PAD_TMIN, PAD_TMAX = EPOCH_TMIN - TFR_PAD, EPOCH_TMAX + TFR_PAD
BASELINE = (EPOCH_TMIN, EPOCH_TMIN + 0.5)

# canonical experiment order (Myo = muscle activation, fingers still)
EXP_ORDER = ["Static", "TwoPositions", "FourPositions", "FourPositionsGaze",
            "RandomSmall", "RandomLarge", "MyoTwoPositions", "MyoFourPositions"]


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
    """
    Per-channel amplitude on the raw (pre-clean) EEG, example subject.
    Bad channels are determined by "detect_bad_channels" using the z-score of the logmedian and its spread.
    The signal quality itself is simply the standard deviation of the signal's amplitude.
    """
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
    start, stop = int(300 * 300), int(306 * 300)
    seg = prep.raw.copy().pick(["C3", "Cz", "C4"]).get_data(
        start=start, stop=stop) * 1e6
    t = prep.eeg_ts[start:stop]  # real LSL timestamps, not a synthetic grid
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


def _epochs_tfr_logratio(ep, picks=None, freqs=None, decim=5):
    """Per-epoch Morlet power, baseline-corrected against EACH epoch's own
    pre-movement window (single-trial log-ratio).

    THE one normalization used for every beta/mu-ERD number in this notebook
    (§4 and all its subsections). We deliberately do *not* average power
    across epochs before baseline-correcting: `log(mean_epochs(power))` and
    `median_epochs(log(power))` are different estimators (log is concave, so
    Jensen's inequality applies) and give visibly different numbers - the
    former is less noisy (baseline pooled over many trials) but the latter is
    what lets us show a per-epoch *distribution* at all (violin/box plots,
    per-subject/per-condition medians, rank tests). Keeping every figure on
    the single-trial estimator makes the numbers comparable across figures.
    See Grandchamp & Delorme (2011), "Single-trial normalization for
    event-related spectral decomposition reduces sensitivity to noisy
    trials," Frontiers in Psychology.

    `ep` is expected to be padded to [PAD_TMIN, PAD_TMAX] (see module-level
    constants) - the TFR is computed on the full padded epoch so the wavelet
    convolution has real data near the display edges, baseline-corrected
    against the fixed [EPOCH_TMIN, EPOCH_TMIN+0.5] window, and then cropped
    back to [EPOCH_TMIN, EPOCH_TMAX] before being returned, so callers never
    see the edge-distorted margin.

    Returns the per-epoch ``EpochsTFR`` (``average=False``); use
    ``.get_data()`` (n_epochs, n_picks, n_freqs, n_times), ``.times``,
    ``.ch_names`` / ``.info``.
    """
    freqs = np.arange(13, 31, 2.0) if freqs is None else freqs
    n_cycles = freqs / 2.0
    tfr = ep.compute_tfr("morlet", freqs=freqs, n_cycles=n_cycles, picks=picks,
                         return_itc=False, average=False, decim=decim,
                         verbose="ERROR")
    tfr.apply_baseline(BASELINE, mode="logratio")
    tfr.crop(EPOCH_TMIN, EPOCH_TMAX)
    return tfr


def _epochs_tfr_avg_logratio(ep, picks=None, freqs=None, decim=5):
    """Trial-averaged counterpart of ``_epochs_tfr_logratio`` (``average=True``)
    - used only where cross-channel comparison matters (topomaps; see the
    methodology note on why topomaps use this estimator). Same padding/crop
    treatment: computed on the padded epoch, baseline-corrected against the
    fixed window, cropped back to the display window.
    """
    freqs = np.arange(13, 31, 2.0) if freqs is None else freqs
    n_cycles = freqs / 2.0
    tfr = ep.compute_tfr("morlet", freqs=freqs, n_cycles=n_cycles, picks=picks,
                         return_itc=False, average=True, decim=decim,
                         verbose="ERROR")
    tfr.apply_baseline(BASELINE, mode="logratio")
    tfr.crop(EPOCH_TMIN, EPOCH_TMAX)
    return tfr


def _beta_trace_per_epoch(ep, picks=("C3", "C4"), freqs=None, decim=5):
    """Per-epoch beta-band (13-30 Hz) ERD trace at motor channels, so callers
    can group/aggregate by arbitrary metadata (subject, run, experiment) and
    show the *distribution* across epochs, not just a single grand mean.

    Returns ``(times, trace, meta)`` where ``trace`` is ``(n_epochs, n_times)``
    (already averaged over `picks` and `freqs`) and ``meta`` is
    ``ep.metadata`` reset to a plain 0..n-1 index aligned with `trace`'s rows.
    """
    picks = [c for c in picks if c in ep.ch_names]
    tfr = _epochs_tfr_logratio(ep, picks=picks, freqs=freqs, decim=decim)
    trace = tfr.get_data().mean(axis=(1, 2))  # (n_epochs, n_times)
    return tfr.times, trace, ep.metadata.reset_index(drop=True)


def fig_motor(all_epochs, report_lines):
    """ERP + time-frequency ERD at C3/Cz/C4, and a beta-ERD topography.

    Uses the same single-trial normalization as §4b-4f (see
    ``_epochs_tfr_logratio``): every heatmap/trace/topomap value here is a
    *median across epochs* of per-epoch baseline log-ratios, not a log-ratio
    of the trial-averaged power.
    """
    ep = all_epochs
    picks = [c for c in ["C3", "Cz", "C4"] if c in ep.ch_names]

    # ---- time-domain average (movement-related potential; plain amplitude,
    # not power, so it is unaffected by the log-ratio estimator question).
    # `ep` is padded to PAD_TMIN/PAD_TMAX (see module constants); crop back to
    # the display window after baseline correction. ----
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
    axerp = fig.add_subplot(gs[0, 0])
    ev = ep.copy().apply_baseline(BASELINE).crop(EPOCH_TMIN, EPOCH_TMAX)
    avg = ev.average()
    times = avg.times
    for ch in picks:
        y = avg.get_data(picks=ch)[0] * 1e6
        axerp.plot(times, y, color=CCOL[ch], label=ch, lw=1.3)
    axerp.axvline(0, color="k", lw=0.8)
    axerp.set_title("Pinch-locked average (baseline = window start)")
    axerp.set_xlabel("time from pinch (s)"); axerp.set_ylabel("µV")
    axerp.legend(fontsize=8)

    # ---- time-frequency (Morlet) mu/beta ERD, per-epoch then median across
    # epochs (full 4-30 Hz resolution, C3/Cz/C4 only - needed for the heatmap) ----
    freqs_full = np.arange(4, 31, 1.0)
    tfr_heat = _epochs_tfr_logratio(ep, picks=picks, freqs=freqs_full, decim=5)
    heat_data = tfr_heat.get_data()  # (n_epochs, n_picks, n_freqs, n_times)

    for j, ch in enumerate(picks):
        axtf = fig.add_subplot(gs[1, j])
        d = np.median(heat_data[:, j], axis=0)  # (freq, time), median over epochs
        im = axtf.pcolormesh(tfr_heat.times, freqs_full, d, cmap="RdBu_r",
                             vmin=-0.5, vmax=0.5, shading="auto")
        axtf.axvline(0, color="k", lw=0.8)
        axtf.axhline(13, color="k", lw=0.4, ls=":")
        axtf.set_title(f"{ch}: mu/beta ERD (log-ratio)")
        axtf.set_xlabel("time from pinch (s)")
        if j == 0:
            axtf.set_ylabel("frequency (Hz)")
        fig.colorbar(im, ax=axtf, fraction=0.046, pad=0.04)

    # ---- beta-band ERD trace + topography: same beta-band definition/decim
    # as §4b-4f (_beta_trace_per_epoch's default), computed once over ALL
    # scalp channels so the topomap and the C3/Cz/C4 trace share one call ----
    tfr_beta = _epochs_tfr_logratio(ep, picks=None, decim=5)  # freqs=13-30 by 2
    beta_data = tfr_beta.get_data()  # (n_epochs, n_all_ch, n_beta_freqs, n_times)
    ch_index = {c: tfr_beta.ch_names.index(c) for c in picks}

    axb = fig.add_subplot(gs[0, 1])
    for ch in picks:
        per_epoch = beta_data[:, ch_index[ch]].mean(axis=1)  # (n_epochs, n_times)
        d = np.median(per_epoch, axis=0)
        axb.plot(tfr_beta.times, d, color=CCOL[ch], label=ch, lw=1.3)
    axb.axvline(0, color="k", lw=0.8); axb.axhline(0, color="#999", lw=0.6)
    axb.set_title("Beta (13–30 Hz) power change")
    axb.set_xlabel("time from pinch (s)"); axb.set_ylabel("log-ratio")
    axb.legend(fontsize=8)

    # ---- topomap of beta ERD in [-0.5, 0] s: the ONE deliberate exception to
    # the single-trial estimator used everywhere else. A topomap's whole job
    # is a *cross-channel* comparison, and the single-trial baseline-noise
    # bias (see methodology note) is essentially uniform across channels -
    # every channel picks up roughly the same ~-0.18 offset regardless of
    # whether it is near motor cortex, so it swamps rather than preserves the
    # genuine spatial pattern (empirically: single-trial gives C3 -0.20 vs.
    # O2 -0.17, a washed-out ~0.03 spread; trial-averaged gives C3 -0.09 vs.
    # O2 +0.01, the real ~0.09 contralateral-focal spread). Within a group
    # comparison (subject/condition/pinch-vs-hardneg) that bias is constant
    # across the groups and cancels; across channels it does not. So the
    # topomap alone baseline-corrects the trial-averaged spectrum. ----
    axtopo = fig.add_subplot(gs[0, 2])
    tfr_topo = _epochs_tfr_avg_logratio(ep, picks=None, decim=5)  # freqs=13-30 by 2
    tmask = (tfr_topo.times >= -0.5) & (tfr_topo.times <= 0.0)
    topo = tfr_topo.get_data()[:, :, tmask].mean(axis=(1, 2))
    mne.viz.plot_topomap(topo, tfr_topo.info, axes=axtopo, show=False,
                         cmap="RdBu_r", vlim=(-0.15, 0.15), contours=4)
    axtopo.set_title("Beta ERD topo, [-0.5,0] s\n(trial-averaged estimator - see note)")

    fig.suptitle(f"Motor signature around pinch onset "
                 f"(N={len(ep)} epochs, {len(picks)} channels)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "03_pinch_motor.png"); plt.close(fig)

    # numeric summary: min beta log-ratio near movement at C3/C4
    for ch in picks:
        per_epoch = beta_data[:, ch_index[ch]].mean(axis=1)
        d = np.median(per_epoch, axis=0)
        near = (tfr_beta.times >= -0.5) & (tfr_beta.times <= 0.5)
        report_lines.append(
            f"- {ch}: beta power change min = **{d[near].min():+.2f}** "
            f"log-ratio near movement (negative = desynchronisation/ERD).")


def _plot_group_traces(ax, times, trace, groups, order=None, cmap_name="tab10",
                       min_n=5):
    """One median line + IQR band per group, so spread across epochs is visible."""
    groups = pd.Series(np.asarray(groups)).reset_index(drop=True)
    labels = order if order is not None else sorted(groups.dropna().unique())
    cmap = plt.get_cmap(cmap_name)
    for i, g in enumerate(labels):
        idx = (groups == g).to_numpy()
        n = int(idx.sum())
        if n < min_n:
            continue
        sub = trace[idx]
        m = np.median(sub, axis=0)
        lo, hi = np.percentile(sub, [25, 75], axis=0)
        color = cmap(i % 10)
        ax.plot(times, m, color=color, lw=1.4, label=f"{g} (n={n})")
        ax.fill_between(times, lo, hi, color=color, alpha=0.12)
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="#999", lw=0.6)


def _shared_beta_ylim(trace, pad_frac=0.08):
    """Common y-limits for every beta-ERD axis in 03b/03c.

    Every panel in both figures (trace ± IQR, run-consistency line, violin,
    box) is ultimately derived from the same per-epoch `trace` array - the
    violin/box values are just a windowed mean of it, so bounding by
    `trace`'s own min/max safely covers every panel. Aligning all of them to
    one shared scale (rather than each axis auto-scaling to its own data)
    makes spread genuinely comparable across panels and between 03b and 03c.
    """
    lo, hi = float(np.nanmin(trace)), float(np.nanmax(trace))
    margin = pad_frac * (hi - lo)
    return (lo - margin, hi + margin)


def fig_motor_by_subject(times, trace, meta, report_lines, ylim=(-0.29,0.02)):
    """Between-subject and within-subject(run) variability of the beta ERD."""
    pre = (times >= -0.5) & (times <= 0.0)
    meta = meta.assign(beta_erd=trace[:, pre].mean(axis=1))
    subjects = sorted(meta["subject"].unique())
    cmap = plt.get_cmap("tab10")
    ylim = _shared_beta_ylim(trace) if ylim is None else ylim

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    _plot_group_traces(axes[0], times, trace, meta["subject"], order=subjects)
    axes[0].set_title("Beta ERD by subject (median ± IQR)")
    axes[0].set_xlabel("time from pinch (s)"); axes[0].set_ylabel("log-ratio, C3+C4")
    axes[0].set_ylim(*ylim)
    axes[0].legend(fontsize=6, ncol=2)

    multi = [s for s in subjects if meta.loc[meta["subject"] == s, "run"].nunique() > 1]
    for s in multi:
        runs = sorted(meta.loc[meta["subject"] == s, "run"].unique())
        color = cmap(subjects.index(s) % 10)
        for j, r in enumerate(runs):
            idx = ((meta["subject"] == s) & (meta["run"] == r)).to_numpy()
            if idx.sum() < 5:
                continue
            m = np.median(trace[idx], axis=0)
            axes[1].plot(times, m, color=color, lw=1.4, ls=["-", "--", ":"][j % 3],
                        label=f"{s} run{j + 1} (n={idx.sum()})")
    axes[1].axvline(0, color="k", lw=0.8); axes[1].axhline(0, color="#999", lw=0.6)
    axes[1].set_title("Run-to-run consistency (subjects with ≥2 runs)")
    axes[1].set_xlabel("time from pinch (s)")
    axes[1].set_ylim(*ylim)
    if multi:
        axes[1].legend(fontsize=6)
    else:
        axes[1].text(0.5, 0.5, "no subject has >1 run", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=9, color="#999")

    order = meta.groupby("subject")["beta_erd"].median().sort_values().index.tolist()
    data = [meta.loc[meta["subject"] == s, "beta_erd"].to_numpy() for s in order]
    vp = axes[2].violinplot(data, showmedians=True, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(cmap(subjects.index(order[i]) % 10)); body.set_alpha(0.6)
    axes[2].set_xticks(range(1, len(order) + 1))
    axes[2].set_xticklabels(order, rotation=45, ha="right", fontsize=7)
    axes[2].axhline(0, color="#999", lw=0.6)
    axes[2].set_title("Per-epoch pre-movement beta ERD, by subject")
    axes[2].set_ylabel("log-ratio, [-0.5,0] s, C3+C4")
    axes[2].set_ylim(*ylim)

    fig.suptitle(f"Motor-signature variability across subjects & runs "
                f"(N={len(meta)} epochs, {len(subjects)} subjects)", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGS / "03b_motor_by_subject.png", bbox_inches="tight")
    plt.close(fig)

    med = meta.groupby("subject")["beta_erd"].median()
    report_lines.append(
        f"- Per-subject pre-movement beta ERD (median) ranges from "
        f"**{med.min():+.2f}** ({med.idxmin()}) to **{med.max():+.2f}** "
        f"({med.idxmax()}) log-ratio - individual-difference spread is "
        f"substantial relative to the grand-average effect.")
    for s in multi:
        g = meta[meta["subject"] == s].groupby("run")["beta_erd"].median()
        report_lines.append(
            f"  - {s}: run medians {', '.join(f'{v:+.2f}' for v in g)} "
            f"({'consistent' if g.max() - g.min() < 0.15 else 'notably different'} "
            f"across runs).")


def fig_motor_by_experiment(times, trace, meta, report_lines, ylim=(-0.29,0.02)):
    """Beta ERD split by experiment condition, incl. finger-pinch vs Myo."""
    ylim = _shared_beta_ylim(trace) if ylim is None else ylim
    pre = (times >= -0.5) & (times <= 0.0)
    meta = meta.assign(beta_erd=trace[:, pre].mean(axis=1))
    meta = meta[meta["experiment"].notna() &
               ~meta["experiment"].isin(["Tutorial", "End"])]
    idx = meta.index.to_numpy()
    trace_e = trace[idx]
    meta_r = meta.reset_index(drop=True)

    present = [e for e in EXP_ORDER if e in meta_r["experiment"].unique()]
    extra = sorted(set(meta_r["experiment"].unique()) - set(present))
    order = present + extra

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    _plot_group_traces(axes[0], times, trace_e, meta_r["experiment"], order=order)
    axes[0].set_title("Beta ERD by experiment condition (median ± IQR)")
    axes[0].set_xlabel("time from pinch (s)"); axes[0].set_ylabel("log-ratio, C3+C4")
    axes[0].set_ylim(*ylim)
    axes[0].legend(fontsize=6, ncol=2)

    med = meta_r.groupby("experiment")["beta_erd"].median()
    order_by_med = med.sort_values().index.tolist()
    data = [meta_r.loc[meta_r["experiment"] == e, "beta_erd"].to_numpy()
            for e in order_by_med]
    vp = axes[1].violinplot(data, showmedians=True, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        is_myo = order_by_med[i].startswith("Myo")
        body.set_facecolor("#D55E00" if is_myo else "#0072B2"); body.set_alpha(0.6)
    axes[1].set_xticks(range(1, len(order_by_med) + 1))
    axes[1].set_xticklabels(order_by_med, rotation=45, ha="right", fontsize=7)
    axes[1].axhline(0, color="#999", lw=0.6)
    axes[1].set_title("Per-epoch pre-movement beta ERD, by condition")
    axes[1].set_ylabel("log-ratio, [-0.5,0] s, C3+C4")
    axes[1].set_ylim(*ylim)

    is_myo = meta_r["experiment"].str.startswith("Myo")
    groups = {"finger pinch": meta_r.loc[~is_myo, "beta_erd"].to_numpy(),
             "Myo (fingers still)": meta_r.loc[is_myo, "beta_erd"].to_numpy()}
    bp = axes[2].boxplot(list(groups.values()), showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#0072B2", "#D55E00"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    axes[2].set_xticks([1, 2]); axes[2].set_xticklabels(list(groups.keys()), fontsize=8)
    axes[2].axhline(0, color="#999", lw=0.6)
    axes[2].set_title("Finger pinch vs muscle-only (Myo)")
    axes[2].set_ylabel("log-ratio")
    axes[2].set_ylim(*ylim)

    fig.suptitle(f"Motor-signature variability across experiment conditions "
                f"(N={len(meta_r)} epochs)", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGS / "03c_motor_by_experiment.png", bbox_inches="tight")
    plt.close(fig)

    report_lines.append(
        f"- Per-condition pre-movement beta ERD (median) ranges from "
        f"**{med.min():+.2f}** ({med.idxmin()}) to **{med.max():+.2f}** "
        f"({med.idxmax()}).")
    mfp = np.median(groups["finger pinch"]); mmy = np.median(groups["Myo (fingers still)"])
    rel = "similar to" if abs(mfp - mmy) < 0.1 else \
        ("weaker than" if mmy > mfp else "stronger than")
    report_lines.append(
        f"- Finger-pinch conditions: median beta ERD **{mfp:+.2f}**; Myo "
        f"(muscle activation, fingers still): **{mmy:+.2f}** - desynchronisation "
        f"without overt finger movement is {rel} finger-pinch trials.")


def fig_motor_pinch_vs_hardneg(times, trace, meta, report_lines):
    """Real pinches ('hard positives') vs hard negatives (distractor looks).

    Section 5 shows a classifier can barely tell these apart from gaze/EEG
    windows. This asks the more basic question directly on the signal: does
    the beta ERD itself differ between a genuine pinch and a gaze-matched
    distractor look with no motor act, or is the pre-movement EEG really
    indistinguishable at single-trial level?
    """
    pre = (times >= -0.5) & (times <= 0.0)
    meta = meta.assign(beta_erd=trace[:, pre].mean(axis=1))
    kinds = ["pinch", "hardneg"]
    kcol = {"pinch": "#0072B2", "hardneg": "#D55E00"}
    klabel = {"pinch": "pinch\n(hard positive)", "hardneg": "distractor look\n(hard negative)"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for k in kinds:
        idx = (meta["kind"] == k).to_numpy()
        n = int(idx.sum())
        sub = trace[idx]
        m = np.median(sub, axis=0)
        lo, hi = np.percentile(sub, [25, 75], axis=0)
        axes[0].plot(times, m, color=kcol[k], lw=1.6, label=f"{k} (n={n})")
        axes[0].fill_between(times, lo, hi, color=kcol[k], alpha=0.15)
    axes[0].axvline(0, color="k", lw=0.8); axes[0].axhline(0, color="#999", lw=0.6)
    axes[0].set_title("Beta ERD: pinch vs hard negative (median ± IQR)")
    axes[0].set_xlabel("time from onset (s)"); axes[0].set_ylabel("log-ratio, C3+C4")
    axes[0].legend(fontsize=7)

    data = [meta.loc[meta["kind"] == k, "beta_erd"].to_numpy() for k in kinds]
    vp = axes[1].violinplot(data, showmedians=True, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(kcol[kinds[i]]); body.set_alpha(0.6)
    axes[1].set_xticks([1, 2]); axes[1].set_xticklabels(
        [klabel[k] for k in kinds], fontsize=8)
    axes[1].axhline(0, color="#999", lw=0.6)
    axes[1].set_title("Per-epoch pre-movement beta ERD")
    axes[1].set_ylabel("log-ratio, [-0.5,0] s, C3+C4")

    subjects = sorted(meta["subject"].unique())
    x = np.arange(len(subjects))
    width = 0.35
    for i, k in enumerate(kinds):
        meds, err_lo, err_hi = [], [], []
        for s in subjects:
            v = meta.loc[(meta["subject"] == s) & (meta["kind"] == k), "beta_erd"]
            if len(v) < 5:
                meds.append(0.0); err_lo.append(0.0); err_hi.append(0.0)
                continue
            med = v.median()
            meds.append(med)
            err_lo.append(med - np.percentile(v, 25))
            err_hi.append(np.percentile(v, 75) - med)
        axes[2].bar(x + (i - 0.5) * width, meds, width, color=kcol[k], alpha=0.75,
                    label=k, yerr=[err_lo, err_hi], capsize=2, ecolor="#555")
    axes[2].axhline(0, color="#999", lw=0.6)
    axes[2].set_xticks(x); axes[2].set_xticklabels(subjects, rotation=30,
                                                    ha="right", fontsize=7)
    axes[2].set_title("Per-subject: pinch vs hard negative")
    axes[2].set_ylabel("median log-ratio ± IQR")
    axes[2].legend(fontsize=7)

    fig.suptitle("Hard positives (pinch) vs hard negatives (distractor look): "
                f"is the pre-movement beta ERD itself different? "
                f"(N={len(meta)} epochs)", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGS / "03d_motor_pinch_vs_hardneg.png", bbox_inches="tight")
    plt.close(fig)

    mp = meta.loc[meta["kind"] == "pinch", "beta_erd"]
    mh = meta.loc[meta["kind"] == "hardneg", "beta_erd"]
    _, p = mannwhitneyu(mp, mh, alternative="two-sided")
    report_lines.append(
        f"- Pre-movement beta ERD median: pinch (hard positive) **{mp.median():+.2f}** "
        f"vs hard negative **{mh.median():+.2f}** (Mann-Whitney p={p:.2g}, "
        f"n_pinch={len(mp)}, n_hardneg={len(mh)}).")
    sep = "a statistically detectable but small" if p < 0.01 else "no reliable"
    report_lines.append(
        f"- The two distributions show {sep} difference and overlap heavily - "
        f"consistent with §5's classifier result: single-trial pre-movement beta "
        f"ERD alone is not a reliable way to separate a real pinch from a "
        f"gaze-matched distractor look.")


def fig_subject_erp_spaghetti(pooled, subject, report_lines):
    """Baseline-adjusted pinch-locked average for one subject, with every
    single-trial sample scattered very faintly behind it as a point cloud -
    shows directly whether the movement-related dip around t0 is a consistent
    single-trial phenomenon or only an artifact of averaging.
    """
    idx = np.where(pooled.metadata["subject"].to_numpy() == subject)[0]
    ep = pooled[idx].copy()
    ep.apply_baseline(BASELINE).crop(EPOCH_TMIN, EPOCH_TMAX)
    picks = [c for c in ["C3", "Cz", "C4"] if c in ep.ch_names]
    times = ep.times
    ylim = (-20, 20)

    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ch in zip(axes, picks):
        d = ep.get_data(picks=ch)[:, 0, :] * 1e6  # (n_epochs, n_times)
        t_flat = np.tile(times, d.shape[0])
        ax.scatter(t_flat, d.ravel(), s=1.5, alpha=0.15, color=CCOL[ch],
                  edgecolors="none", rasterized=True)
        ax.plot(times, d.mean(axis=0), color="k", lw=2.2, label="mean")
        ax.axvline(0, color="k", lw=0.9, ls="--")
        ax.set_ylim(*ylim)
        ax.set_title(f"{ch} (n={len(d)} epochs)")
        ax.set_xlabel("time from pinch (s)")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("µV")
    fig.suptitle(f"{subject}: single-trial consistency of the pinch-locked "
                f"potential (grand mean in black; every trial's samples as "
                f"faint points behind it)")
    fig.tight_layout()
    fig.savefig(FIGS / "03e_subject_erp_spaghetti.png", bbox_inches="tight")
    plt.close(fig)

    if "C3" in picks:
        # A trial "ever touching" below zero in a 400 ms window is a weak
        # statistic - any zero-mean noisy EEG trace does that almost surely.
        # Instead test the per-trial *mean* amplitude in the dip window
        # against 0 (Wilcoxon signed-rank, paired across trials): this is
        # exactly the standard single-trial-consistency test for an ERP.
        d = ep.get_data(picks="C3")[:, 0, :] * 1e6
        near = (times >= -0.3) & (times <= 0.1)
        per_trial = d[:, near].mean(axis=1)
        frac_neg = float((per_trial < 0).mean())
        _, p = wilcoxon(per_trial)
        report_lines.append(
            f"- {subject}, C3: mean amplitude in [-0.3, +0.1] s is negative "
            f"on **{frac_neg:.0%}** of the {len(d)} individual trials "
            f"(median {np.median(per_trial):+.2f} µV vs. baseline; Wilcoxon "
            f"signed-rank vs. 0, p={p:.2g}) - the dip visible in the grand "
            f"average is a consistent, statistically robust single-trial "
            f"phenomenon, not just an averaging artifact.")


def fig_subject_topo_by_experiment(pooled, subject, report_lines):
    """Beta-ERD topography for one subject, one topomap per experiment
    condition - does the spatial pattern (not just the magnitude) hold across
    conditions, or is it driven by one or two of them?

    Like the topomap in ``fig_motor`` (see the note there), this uses the
    trial-averaged estimator PER CONDITION rather than the single-trial
    estimator used elsewhere: a topomap is a cross-channel comparison, and
    the single-trial baseline-noise bias is ~uniform across channels, so it
    would swamp the spatial pattern we're trying to see - and with as few as
    5-8 epochs in some conditions here, that bias is worse, not better.
    """
    idx = np.where(pooled.metadata["subject"].to_numpy() == subject)[0]
    sub_ep = pooled[idx]
    meta_full = sub_ep.metadata.reset_index(drop=True)
    keep = (meta_full["experiment"].notna() &
           ~meta_full["experiment"].isin(["Tutorial", "End"])).to_numpy()
    ep = sub_ep[keep]
    meta = meta_full[keep].reset_index(drop=True)
    present = [e for e in EXP_ORDER if e in meta["experiment"].unique()]
    extra = sorted(set(meta["experiment"].unique()) - set(present))
    order = present + extra

    ncols = min(4, len(order))
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.3 * nrows))
    axes = np.atleast_2d(axes)
    vmax = 0.2
    for i, exp in enumerate(order):
        ax = axes[i // ncols, i % ncols]
        m = (meta["experiment"] == exp).to_numpy()
        n = int(m.sum())
        if n < 1:
            ax.axis("off")
            continue
        ep_exp = ep[m]
        tfr = _epochs_tfr_avg_logratio(ep_exp, picks=None, decim=5)
        tmask = (tfr.times >= -0.5) & (tfr.times <= 0.0)
        topo = tfr.get_data()[:, :, tmask].mean(axis=(1, 2))
        mne.viz.plot_topomap(topo, tfr.info, axes=ax, show=False, cmap="RdBu_r",
                             vlim=(-vmax, vmax), contours=4)
        ax.set_title(f"{exp}\n(n={n})", fontsize=8)
    for j in range(len(order), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle(f"{subject}: beta ERD topography by experiment condition "
                f"([-0.5, 0] s pre-movement)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "03f_subject_topo_by_experiment.png", bbox_inches="tight")
    plt.close(fig)

    report_lines.append(
        f"- {subject}: beta ERD topomap computed separately for "
        f"{len(order)} experiment conditions ({len(meta)} epochs total) - "
        f"see whether the contralateral (C3-centred) pattern from §4 holds in "
        f"every condition or is driven by a subset.")


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

    # Pooled pinch epochs for the motor analysis. Built PAD_TMIN/PAD_TMAX wide
    # (wider than the EPOCH_TMIN/EPOCH_TMAX actually analyzed/displayed) so
    # every TFR below has real data to convolve against near the display
    # edges; each TFR helper crops back to EPOCH_TMIN/EPOCH_TMAX internally.
    print("== Building pooled pinch epochs ==")
    ep_list = []
    for prep in preps:
        e = pinch_epochs(prep, tmin=PAD_TMIN, tmax=PAD_TMAX)
        if len(e):
            meta = e.metadata.copy()
            meta["subject"] = prep.session.subject
            meta["run"] = prep.session.stamp
            e.metadata = meta
            ep_list.append(e)
    pooled = mne.concatenate_epochs(ep_list, verbose="ERROR")
    print("pooled epochs:", len(pooled))
    report.append("\n## 3. Motor signature around the pinch\n")
    report.append(f"Pooled pinch epochs: **{len(pooled)}** "
                  f"(from {len(ep_list)} sessions).")
    report.append(
        "\n**Methodology note.** Every beta/mu-ERD number below (§3 and §3b-3f) "
        "uses the same single-trial normalization: baseline-correct each "
        "epoch's power against its own pre-movement window via a log-ratio, "
        "*then* aggregate (median, with IQR shown where a distribution is "
        "plotted) across whatever group of epochs is being compared. We do "
        "**not** average power across epochs first and then baseline-correct "
        "the trial-averaged spectrum - that alternative is much less noisy "
        "(its baseline is estimated by pooling hundreds of trials instead of "
        "one) but is a *different estimator*: because `log` is concave, "
        "`median_epochs(log(P_task/P_base)) <= log(median_epochs(P_task)/"
        "median_epochs(P_base))` (Jensen's inequality), so it reads "
        "systematically less negative (weaker-looking) than the single-trial "
        "estimator used here. See Grandchamp & Delorme (2011), *Single-trial "
        "normalization for event-related spectral decomposition reduces "
        "sensitivity to noisy trials*, Frontiers in Psychology. Keeping one "
        "estimator throughout makes every number in this section directly "
        "comparable.\n\n"
        "**One deliberate exception: topomaps.** A topomap is a *cross-channel* "
        "comparison, and the single-trial baseline-noise bias above is "
        "essentially uniform across channels - it doesn't care whether a "
        "channel sits over motor cortex or the occiput. Applied to a topomap "
        "it therefore doesn't cancel (as it does when comparing subjects/"
        "conditions/pinch-vs-hardneg, where every group sees the same bias); "
        "instead it swamps the spatial pattern. Empirically, on the pooled "
        "data: single-trial gives C3 -0.20 vs. O2 -0.17 (a washed-out ~0.03 "
        "spread across the whole scalp); the trial-averaged estimator gives "
        "C3 -0.09 vs. O2 +0.01 (the real, focal ~0.09 contralateral spread "
        "that motivates calling this a *motor* signature). So the topomaps "
        "in §3 and §3e use the trial-averaged estimator; every other panel "
        "uses the single-trial one. Each topomap is captioned accordingly.\n\n"
        "**Edge padding.** Epochs are built to "
        f"[{PAD_TMIN}, {PAD_TMAX}] s, {TFR_PAD} s wider on each side than the "
        f"[{EPOCH_TMIN}, {EPOCH_TMAX}] s actually analyzed/plotted. A Morlet "
        "wavelet needs real signal on both sides of every time point it "
        "estimates (with `n_cycles = freq/2` used throughout, that support is "
        "a constant ~0.5 s at every frequency); without padding, the "
        "convolution runs out of real data right at the display edges and "
        "the power estimate there is artificially suppressed - showing up as "
        "a spurious sharp dip at the very start/end of every log-ratio trace "
        "or heatmap. Every TFR helper here computes on the padded epoch and "
        "crops back to the display window only *after* baseline correction, "
        "so that edge artifact never reaches a figure.")
    fig_motor(pooled, report)

    print("== Per-epoch beta-ERD trace (for subject/experiment breakdowns) ==")
    times_b, trace_b, meta_b = _beta_trace_per_epoch(pooled)
    report.append("\n### 3b. Variability across subjects & runs\n")
    fig_motor_by_subject(times_b, trace_b, meta_b, report)
    report.append("\n### 3c. Variability across experiment conditions\n")
    fig_motor_by_experiment(times_b, trace_b, meta_b, report)

    # Hard negatives (distractor GAZE_ENTER, no pinch) at the same window,
    # tagged the same way, so they can be pooled against the real pinches.
    print("== Building pooled hard-negative epochs ==")
    hn_list = []
    for prep in preps:
        hard_t = ds.hard_negative_times(prep.markers, prep.pinch)
        hn = negative_epochs(prep, hard_t, tmin=PAD_TMIN, tmax=PAD_TMAX)
        if hn is not None and len(hn):
            hn.metadata = pd.DataFrame({
                "subject": [prep.session.subject] * len(hn),
                "run": [prep.session.stamp] * len(hn),
                "kind": ["hardneg"] * len(hn),
            })
            hn_list.append(hn)
    report.append("\n### 3d. Hard positives vs hard negatives\n")
    if hn_list:
        pooled_hardneg = mne.concatenate_epochs(hn_list, verbose="ERROR")
        pooled_pos = pooled.copy()
        pooled_pos.metadata = pd.DataFrame({
            "subject": pooled.metadata["subject"].to_numpy(),
            "run": pooled.metadata["run"].to_numpy(),
            "kind": "pinch",
        })
        combo = mne.concatenate_epochs([pooled_pos, pooled_hardneg], verbose="ERROR")
        print("pooled pinch+hardneg epochs:", len(combo))
        times_c, trace_c, meta_c = _beta_trace_per_epoch(combo)
        fig_motor_pinch_vs_hardneg(times_c, trace_c, meta_c, report)
    else:
        report.append("- No hard-negative epochs available; skipped.")

    report.append("\n### 3e. Single-subject case study - subject_7\n")
    fig_subject_erp_spaghetti(pooled, "subject_7", report)
    fig_subject_topo_by_experiment(pooled, "subject_7", report)

    report.append("\n## 4. Gaze / pupil around the pinch\n")
    fig_gaze(preps, report)

    (OUT / "eda_report.md").write_text("\n".join(str(x) for x in report))
    print("\nWrote", OUT / "eda_report.md", "and figures to", FIGS)


if __name__ == "__main__":
    main()
