"""
dataset.py — Load and align the VIBE pinch experiment recordings.

Each subject directory holds up to three synchronised streams that share one
LSL clock (the ``timestamp`` column, in seconds):

  * ``eeg_*.csv``      — 24 ch WearableSensing DSI-24 @ ~300 Hz
  * ``gaze_*.csv``     — 19 ch Vive Pro Eye gaze @ nominal 120 Hz (effective ~38 Hz)
  * ``markers_*.jsonl`` — one JSON object per event marker, one line per event.
    Each line is the recorder's merge of the raw LSL marker payload (as JSON,
    published by the Unity ``ExperimentLogger``) with two fields it adds itself:
    ``timestamp`` (LSL clock, seconds) and ``stream`` (the source LSL stream
    name, e.g. ``Experiment.VIBE.GameEventStream``). The payload carries
    ``EventType``/``EventName``/``ExperimentName``/``ExperimentConfig``/
    ``Payload``/``HeadTracking`` straight from Unity's ``EventLog`` JSON.

Because the streams share the LSL clock we can align everything by timestamp.
The EEG ``TRG`` (trigger) channel is all-zero in these recordings, so marker
timestamps are the *only* way to locate events in the EEG.

Ground-truth "pinch" events
---------------------------
The task defines a pinch as the ``ExperimentLog`` message
``action=select,trial=N,target=Stimuli(Clone)``. Empirically each such select
is preceded ~19 ms by a ``PinchLog`` event (the raw gesture detection), which is
the better-aligned motor onset. We therefore expose *both*:

  * ``PinchLog`` events (``PINCH_STIMULUS`` = pinch while gazing a stimulus,
    ``PINCH_NONE`` = pinch into empty space) — the actual gesture onsets.
  * the ``action=select`` logs — carry trial number and reaction time (rt).

We merge them: every PinchLog gesture is annotated with whether it produced a
select, its trial and rt. This module makes no modelling decisions; it just
returns tidy, aligned objects.

This module reuses ``scripts.s00_mne_info_builder`` for the MNE Info.
"""
from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# Reuse the researcher's info builder (do not modify it).
import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.s00_mne_info_builder import build_info, LEGACY_TO_MODERN  # noqa: E402

# Subject recordings live under data/raw/subject_*; fall back to data/ itself
# so a flat data/subject_* layout (no raw/preprocessed split) still works.
DATA_ROOT = _ROOT / "data" / "raw" if (_ROOT / "data" / "raw").exists() \
    else _ROOT / "data"

# Channels that are auxiliary / not scalp EEG on this DSI-24 export.
#   A1, A2 : ear/mastoid references
#   X1,X2,X3: analog aux (flat in these recordings ~0.2 uV std)
#   TRG    : trigger (all zero here)
NON_SCALP = {"A1", "A2", "X1", "X2", "X3", "TRG"}
MASTOIDS = ["A1", "A2"]
# Motor channels most relevant to a hand/finger pinch (contralateral C3 for a
# right-hand pinch, plus C4 and the midline Cz).
MOTOR_CHANNELS = ["C3", "Cz", "C4"]

EEG_SFREQ = 300.0
GAZE_CHANNELS = [
    "left_gaze_direction_x", "left_gaze_direction_y", "left_gaze_direction_z",
    "right_gaze_direction_x", "right_gaze_direction_y", "right_gaze_direction_z",
    "left_pupil_diameter", "right_pupil_diameter",
    "left_eye_openness", "right_eye_openness",
]

MYO_EXPERIMENTS = {"MyoTwoPositions", "MyoFourPositions"}

# Blocks that carry no real experimental trials and should be excluded from
# condition-level stats / negative-window sampling. Matches both the legacy
# short names ("Tutorial", "End") and the current naming, where a practice
# block's ExperimentName is e.g. "TutorialSession1 (Experiments.TutorialSession1)".
_EXCLUDE_BLOCK_NAMES = {"", "End"}
_TUTORIAL_RE = re.compile(r"Tutorial", re.IGNORECASE)


def is_excluded_block(experiment: pd.Series) -> pd.Series:
    """True for practice/tutorial or unlabeled 'end of session' block names."""
    exp = experiment.fillna("")
    return exp.isin(_EXCLUDE_BLOCK_NAMES) | exp.str.contains(_TUTORIAL_RE)


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    """One recording session (a matched eeg/gaze/markers triple)."""
    subject: str
    stamp: str                # the YYYYMMDD_HHMMSS token
    eeg_csv: Path
    eeg_meta: Path
    gaze_csv: Path
    gaze_meta: Path
    markers_jsonl: Path

    def __repr__(self) -> str:
        return f"Session(subject={self.subject!r}, stamp={self.stamp!r})"


_STAMP_RE = re.compile(r"(\d{8}_\d{6})")


def _stamp(p: Path) -> str:
    m = _STAMP_RE.search(p.name)
    return m.group(1) if m else ""


def find_sessions(subject_dir: str | Path) -> list[Session]:
    """Return all sessions in a subject directory (subjects 5/6 have two)."""
    subject_dir = Path(subject_dir)
    subject = subject_dir.name
    eeg_csvs = sorted(p for p in subject_dir.glob("eeg_*.csv")
                      if "metadata" not in p.name)
    sessions = []
    for eeg in eeg_csvs:
        st = _stamp(eeg)
        def one(pat: str) -> Path:
            hits = [p for p in subject_dir.glob(pat) if _stamp(p) == st]
            if not hits:
                raise FileNotFoundError(f"No {pat} for stamp {st} in {subject_dir}")
            return hits[0]
        sessions.append(Session(
            subject=subject, stamp=st,
            eeg_csv=eeg,
            eeg_meta=one("eeg_*_metadata.txt"),
            gaze_csv=one("gaze_*.csv"),
            gaze_meta=one("gaze_*_metadata.txt"),
            markers_jsonl=one("markers_*.jsonl"),
        ))
    return sessions


def all_sessions(data_root: str | Path = DATA_ROOT) -> list[Session]:
    out: list[Session] = []
    for d in sorted(Path(data_root).glob("subject_*")):
        out.extend(find_sessions(d))
    return out


# --------------------------------------------------------------------------- #
# EEG
# --------------------------------------------------------------------------- #
def load_eeg(session: Session) -> tuple[mne.io.RawArray, np.ndarray]:
    """Load EEG into an MNE Raw plus the original per-sample timestamp vector.

    The Raw is built on a uniform ``EEG_SFREQ`` grid (MNE requires uniform
    sampling). The returned timestamp vector is the *real* LSL clock per sample
    and must be used to map marker times to sample indices (``times_to_samples``),
    which stays correct despite the small clock jitter.
    """
    df = pd.read_csv(session.eeg_csv)
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    ch_names = [c for c in df.columns if c != "timestamp"]
    data = df[ch_names].to_numpy(dtype=np.float64).T  # (n_ch, n_samples), uV

    info, rename_map = build_info(
        session.eeg_meta, subject_id=session.subject, stream_name="EEG",
    )
    # build_info renamed legacy channels in `info`; apply the same to our data
    # order. `info.ch_names` is already in renamed order matching df columns.
    data_v = data * 1e-6  # uV -> V for MNE
    raw = mne.io.RawArray(data_v, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations([], [], []))
    return raw, ts


def times_to_samples(times, eeg_ts: np.ndarray) -> np.ndarray:
    """Map LSL timestamps to nearest EEG sample indices (robust to jitter)."""
    times = np.atleast_1d(np.asarray(times, dtype=np.float64))
    idx = np.searchsorted(eeg_ts, times)
    idx = np.clip(idx, 1, len(eeg_ts) - 1)
    left = eeg_ts[idx - 1]
    right = eeg_ts[idx]
    idx = np.where(np.abs(times - left) <= np.abs(right - times), idx - 1, idx)
    return idx.astype(int)


# --------------------------------------------------------------------------- #
# Gaze
# --------------------------------------------------------------------------- #
def load_gaze(session: Session) -> pd.DataFrame:
    """Load gaze samples (irregularly sampled, ~38 Hz effective).

    Returns an empty DataFrame if the gaze file is empty (subject_6 session 2
    has a 0-byte gaze export).
    """
    if session.gaze_csv.stat().st_size == 0:
        return pd.DataFrame(columns=["timestamp"] + GAZE_CHANNELS)
    return pd.read_csv(session.gaze_csv)


def has_gaze(session: Session) -> bool:
    return session.gaze_csv.stat().st_size > 0


def gaze_on_grid(gaze: pd.DataFrame, grid_ts: np.ndarray,
                 channels: list[str] | None = None) -> pd.DataFrame:
    """Resample gaze onto an arbitrary timestamp grid via linear interpolation.

    Invalid samples (any_data_valid == 0) are set to NaN before interpolation
    so blinks/dropouts don't leak. Returns a DataFrame indexed by grid_ts.
    """
    channels = channels or GAZE_CHANNELS
    g = gaze.sort_values("timestamp")
    valid = g["any_data_valid"].to_numpy() > 0.5
    src_t = g["timestamp"].to_numpy()
    out = {}
    for ch in channels:
        y = g[ch].to_numpy(dtype=np.float64).copy()
        y[~valid] = np.nan
        # linear interp over valid points; edges held constant
        good = np.isfinite(y)
        if good.sum() < 2:
            out[ch] = np.full(len(grid_ts), np.nan)
            continue
        out[ch] = np.interp(grid_ts, src_t[good], y[good])
    return pd.DataFrame(out, index=grid_ts)


# --------------------------------------------------------------------------- #
# Markers
# --------------------------------------------------------------------------- #
def _read_markers_jsonl(path: Path) -> pd.DataFrame:
    """Read a markers_*.jsonl file into a flat DataFrame, one row per line.

    Each line is already a single JSON object (see module docstring), so this
    is a straight `json.loads` per line - no nested string-encoded payloads
    to unwrap. Malformed lines are dropped.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _select_message(payload: dict) -> dict | None:
    """Parse an ExperimentLog Payload's `Message` field.

    Most ExperimentLog messages carry a JSON-encoded dict (e.g.
    `{"Action":"select","Trial":0,"Target":"Stimuli(Clone)","ResponseTime":...}`);
    a few are plain text (e.g. "Finished: 3 events.") and are not selects.
    """
    msg = payload.get("Message") if isinstance(payload, dict) else None
    if not isinstance(msg, str):
        return None
    try:
        obj = json.loads(msg)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_markers(session: Session) -> dict[str, pd.DataFrame]:
    """Parse the markers jsonl file into tidy per-kind DataFrames.

    Returns dict with keys:
      'game'  : all GameEventStream events (etype, ename, exp, payload cols)
      'pinch' : PinchLog gesture onsets (the motor pinch events)
      'select': ExperimentLog action=select rows (trial, rt, target)
      'myo'   : MyoPinch pinch_start/pinch_end onsets
      'blink' : BlinkLog events (eye, state)
      'gaze_evt': GazeLog events (gaze enter/exit on stimulus)
    """
    mk = _read_markers_jsonl(session.markers_jsonl)
    empty_cols = ["timestamp", "experiment", "ename"]
    if "stream" not in mk.columns:
        game = pd.DataFrame(columns=empty_cols + ["etype", "payload"])
    else:
        game = mk[mk["stream"] == "Experiment.VIBE.GameEventStream"].copy()
    game["etype"] = game.get("EventType")
    game["ename"] = game.get("EventName")
    game["exp"] = game.get("ExperimentName")
    game["payload"] = (game["Payload"] if "Payload" in game.columns
                       else pd.Series([{}] * len(game), index=game.index)) \
        .apply(lambda p: p if isinstance(p, dict) else {})

    # --- Pinch gesture onsets (PinchLog) ---
    # Stimulus info sits under Payload["Interactable"] (empty dict for
    # PINCH_NONE, a pinch into empty space); Payload["PinchType"] is a new
    # field distinguishing a real motor pinch ("PINCH") from a gaze-only
    # selection logged as a PinchLog for bookkeeping ("GAZE"), a hinted
    # Wizard-of-Oz pinch ("PINCH_WIZARD_OF_OZ"), etc.
    pl = game[game["etype"] == "PinchLog"].copy()
    interactable = pl["payload"].apply(
        lambda p: p.get("Interactable")
        if isinstance(p.get("Interactable"), dict) else {})
    def pv(key, default=np.nan):
        return [i.get(key, default) for i in interactable]
    pinch = pd.DataFrame({
        "timestamp": pl["timestamp"].to_numpy(),
        "experiment": pl["exp"].to_numpy(),
        "ename": pl["ename"].to_numpy(),
        "pinch_type": [p.get("PinchType") for p in pl["payload"]],
        "stimulus_id": pv("StimulusId", None),
        "is_selected": pv("IsSelected", False),
        "is_highlighted": pv("IsHighlighted", False),
        "is_highlighted_neg": pv("IsHighlightedNegative", False),
        "is_gazed": pv("IsGazed", False),
    })
    # position (only present for PINCH_STIMULUS)
    pos = [i.get("Position", {}) if isinstance(i.get("Position"), dict) else {}
           for i in interactable]
    pinch["pos_x"] = [q.get("x", np.nan) for q in pos]
    pinch["pos_y"] = [q.get("y", np.nan) for q in pos]
    pinch["pos_z"] = [q.get("z", np.nan) for q in pos]
    pinch["is_myo"] = pinch["experiment"].isin(MYO_EXPERIMENTS)
    pinch = pinch.sort_values("timestamp").reset_index(drop=True)

    # --- select logs (trial, rt) ---
    el = game[game["etype"] == "ExperimentLog"].copy()
    el["msg_obj"] = el["payload"].apply(_select_message)
    is_select = el["msg_obj"].apply(
        lambda d: isinstance(d, dict) and d.get("Action") == "select")
    sel = el[is_select].copy()
    select = pd.DataFrame({
        "timestamp": sel["timestamp"].to_numpy(),
        "experiment": sel["exp"].to_numpy(),
        "trial": [d.get("Trial") for d in sel["msg_obj"]],
        "target": [d.get("Target") for d in sel["msg_obj"]],
        "rt": [d.get("ResponseTime") for d in sel["msg_obj"]],
    })
    if len(select):
        select["trial"] = pd.to_numeric(select["trial"], errors="coerce")
        select["rt"] = pd.to_numeric(select["rt"], errors="coerce")

    # --- blink / gaze logs ---
    bl = game[game["etype"] == "BlinkLog"].copy()
    blink = pd.DataFrame({
        "timestamp": bl["timestamp"].to_numpy(),
        "experiment": bl["exp"].to_numpy(),
        "eye": [p.get("Eye") for p in bl["payload"]],
        "state": [p.get("State") for p in bl["payload"]],
    })
    gl = game[game["etype"] == "GazeLog"].copy()
    def gv(key, default=None):
        return [p.get(key, default) if isinstance(p, dict) else default
                for p in gl["payload"]]
    gaze_evt = pd.DataFrame({
        "timestamp": gl["timestamp"].to_numpy(),
        "experiment": gl["exp"].to_numpy(),
        "ename": gl["ename"].to_numpy(),
        "stimulus_id": gv("StimulusId"),
        "is_highlighted": gv("IsHighlighted", False),
        "is_highlighted_neg": gv("IsHighlightedNegative", False),
        "is_selected": gv("IsSelected", False),
    })

    # --- myo ---
    # The Myo bridge publishes plain-string samples ("pinch_start"/"pinch_end",
    # not JSON) on a separate LSL stream; the recorder's `json.loads` on that
    # payload fails and drops those samples, so no Experiment.VIBE.MyoPinch
    # rows currently reach this file. Handled defensively in case that's fixed
    # upstream and the stream starts publishing JSON objects.
    if "stream" in mk.columns:
        myo_raw = mk[mk["stream"] == "Experiment.VIBE.MyoPinch"].copy()
    else:
        myo_raw = mk.iloc[0:0]
    def myo_state(row) -> str | None:
        payload = row.get("Payload")
        if isinstance(payload, dict):
            for key in ("state", "State"):
                if key in payload:
                    return payload[key]
        for key in ("state", "State"):
            val = row.get(key)
            if isinstance(val, str):
                return val
        return None
    myo = pd.DataFrame({
        "timestamp": myo_raw["timestamp"].to_numpy() if len(myo_raw) else
                     np.array([], dtype=np.float64),
        "state": [myo_state(r) for _, r in myo_raw.iterrows()],
    })

    return {"game": game, "pinch": pinch, "select": select,
            "myo": myo, "blink": blink, "gaze_evt": gaze_evt}


def merge_pinch_select(pinch: pd.DataFrame, select: pd.DataFrame,
                       tol: float = 0.25) -> pd.DataFrame:
    """Annotate each PinchLog gesture with the trial/rt of the nearest select.

    A pinch is matched to a select if a select falls within +/- ``tol`` seconds
    (empirically select follows the pinch by ~19 ms).
    """
    pinch = pinch.copy()
    pinch["trial"] = np.nan
    pinch["rt"] = np.nan
    pinch["produced_select"] = False
    if len(select) == 0:
        return pinch
    st = select["timestamp"].to_numpy()
    for i, t in enumerate(pinch["timestamp"].to_numpy()):
        d = st - t
        j = np.argmin(np.abs(d))
        if abs(d[j]) <= tol:
            pinch.at[pinch.index[i], "trial"] = select["trial"].iloc[j]
            pinch.at[pinch.index[i], "rt"] = select["rt"].iloc[j]
            pinch.at[pinch.index[i], "produced_select"] = True
    return pinch


# --------------------------------------------------------------------------- #
# Experiment segmentation
# --------------------------------------------------------------------------- #
def gaze_enter_to_pinch_dwell(markers: dict[str, pd.DataFrame],
                              pinch: pd.DataFrame, max_dwell: float = 5.0
                              ) -> float:
    """Median time from a gaze-enter to the pinch that followed it.

    This is how long, on a real trial, the eyes rest on the target before the
    pinch fires (~0.3-0.45 s here). Used to place hard negatives at a comparable
    phase of a *distractor* fixation.
    """
    ge = markers["gaze_evt"]
    enters = np.sort(ge.loc[ge["ename"] == "GAZE_ENTER", "timestamp"].to_numpy())
    dwell = []
    for pt in pinch["timestamp"].to_numpy():
        prev = enters[enters <= pt]
        if len(prev) and (pt - prev[-1]) < max_dwell:
            dwell.append(pt - prev[-1])
    return float(np.median(dwell)) if dwell else 0.35


def hard_negative_times(markers: dict[str, pd.DataFrame], pinch: pd.DataFrame,
                        dwell: float | None = None) -> np.ndarray:
    """Reference times for 'look-but-do-not-pinch' distractor stimuli.

    A highlighted-*negative* stimulus is one the participant should look at but
    must NOT pinch. Its GAZE_ENTER events therefore reproduce the exact gaze
    pattern (saccade in + fixation) of a real target without any motor pinch —
    ideal hard negatives that stop the classifier passing on gaze alone. We place
    the reference ``dwell`` seconds after gaze-enter so the window ends at the
    same fixation phase as a positive window ends at the pinch.
    """
    if dwell is None:
        dwell = gaze_enter_to_pinch_dwell(markers, pinch)
    ge = markers["gaze_evt"]
    neg = ge[(ge["ename"] == "GAZE_ENTER") & (ge["is_highlighted_neg"] == True)]  # noqa: E712
    return np.sort(neg["timestamp"].to_numpy()) + dwell


def experiment_blocks(markers: dict[str, pd.DataFrame],
                      eeg_ts: np.ndarray) -> pd.DataFrame:
    """Infer per-experiment time blocks from the span of that experiment's events.

    Returns a DataFrame: experiment, t_start, t_end, n_pinch. Blocks are clipped
    to the EEG coverage window.
    """
    game = markers["game"]
    rows = []
    lo, hi = eeg_ts[0], eeg_ts[-1]
    g2 = game.copy()
    g2["exp"] = g2["exp"].where(g2["exp"].notna(), "None")
    for exp, g in g2.groupby("exp", dropna=False):
        t0, t1 = g["timestamp"].min(), g["timestamp"].max()
        rows.append({"experiment": exp,
                     "t_start": max(t0, lo), "t_end": min(t1, hi)})
    blocks = pd.DataFrame(rows, columns=["experiment", "t_start", "t_end"])
    return blocks.sort_values("t_start").reset_index(drop=True)


if __name__ == "__main__":
    sess = all_sessions()[0]
    print("Session:", sess)
    raw, ts = load_eeg(sess)
    print("EEG raw:", raw)
    print("  duration %.1fs, %d ch @ %g Hz" % (raw.times[-1], raw.info["nchan"],
                                               raw.info["sfreq"]))
    mk = parse_markers(sess)
    for k, v in mk.items():
        print(f"  markers[{k}]: {len(v)} rows")
    pinch = merge_pinch_select(mk["pinch"], mk["select"])
    print("Pinch onsets:", len(pinch),
          "| produced select:", int(pinch["produced_select"].sum()))
    print(pinch.head())
    samp = times_to_samples(pinch["timestamp"], ts)
    print("First pinch sample idx:", samp[:5], "of", len(ts))
