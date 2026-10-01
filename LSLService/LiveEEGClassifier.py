#!/usr/bin/env python3
"""
LiveEEGClassifier.py - Classify EEG signals in real time. The last 2 seconds are filtered
and given to a torch model for classification. The results are published on a
separate LSL stream.

The classifier takes two models and follows a 3 phase run, driven by phase-end
tokens ("end_phase_1", "end_phase_2", "end_phase_3") on a backend stream:
    Phase 1: idle (LSLLiveViewLogger records the data used for fine tuning).
             At its end, the phase 1 data of the most recent recording is epoched
             (same preprocessing and epoching as DataAnalysis/11_preprocessing.ipynb)
             and both models are fine tuned on it.
    Phase 2: live inference with the fine tuned first model.
    Phase 3: live inference with the fine tuned second model.
Readiness for each phase ("classifier_ready_phase_N") is published on the marker
output stream; phases 2 and 3 count as ready once inference is running.
"""

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pylsl import StreamInlet, StreamInfo, StreamOutlet, resolve_byprop, proc_clocksync, proc_dejitter
from scipy import signal

LSL_STREAM_NAME_IN  = "Experiment.VIBE.EEG"
LSL_STREAM_NAME_OUT = "Experiment.VIBE.Pinch"
LSL_MARKER_NAME_IN  = "Experiment.VIBE.MarkerReplay"
CLASSIFY_WINDOW_SECONDS = 2

# Backend stream publishing the phase ends.
BACKEND_STREAM_NAME_PHASES = "Experiment.VIBE.Backend.Game"
MSG_END_PHASE = re.compile(r"^end_phase_(\d+)$", re.IGNORECASE)
MSG_READY_PHASE = "classifier_ready_phase_{}"
READY_REPUBLISH_SECONDS = 2.0   # re-publish the ready state so late subscribers see it
N_PHASES = 3

# Channels of the DSI stream that are not part of the model input (dropped in
# 11_preprocessing.ipynb as well).
NON_EEG_CHANNELS = {"X1", "X2", "X3", "TRG", "timestamp"}

# dsi2lsl records against Pz (--reference=Pz), so the stream has no Pz channel. Models
# trained with the reference restored expect it as an extra, last channel: all zeros
# before the average re-referencing (mne.add_reference_channels).
REFERENCE_CHANNEL = "Pz"

# Epoching, as in 11_preprocessing.ipynb.
EPOCH_T_MIN = -1.7
EPOCH_T_MAX = 0.3
MIN_DIST_BETWEEN_EVENTS = 1.1
LABEL_POSITIVE = "PinchLog"
LABEL_NEGATIVE = "NoPinchLog"
LABELS = [LABEL_NEGATIVE, LABEL_POSITIVE]   # model output order

# Fine tuning.
FINETUNE_EPOCHS = 30
FINETUNE_LR = 1e-4
FINETUNE_BATCH_SIZE = 32
FINETUNE_MIN_EPOCHS_PER_CLASS = 10   # keep the original models if fewer epochs per class are available

ANSI_GREEN_BG = "\033[42m"
ANSI_RED_BG = "\033[41m"
ANSI_RESET = "\033[0m"


def status_cell(is_positive):
    bg = ANSI_GREEN_BG if is_positive else ANSI_RED_BG
    return f"{bg}  {ANSI_RESET}"


def render_classification_buffer(buffer, next_index):
    """buffer is a ring buffer; next_index is where the next write will land
    (i.e. the oldest entry). Renders oldest -> newest, so the most recent
    classification is always the rightmost cell."""
    ordered = buffer[next_index:] + buffer[:next_index]
    return "".join(status_cell(v) for v in ordered)



# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Real-time EEG filtering + torch classification over LSL.")
    p.add_argument("model", help="Path to the pre-trained model (torch.export .pt2) used in phase 2")
    p.add_argument("model2", help="Path to the pre-trained model (torch.export .pt2) used in phase 3")
    p.add_argument("--hop", type=float, default=0.25,
                   help="Seconds between classifications (default 0.5)")
    p.add_argument("--replay", action="store_true",
                   help=f"Also subscribe to the '{LSL_MARKER_NAME_IN}' stream from EEGReplay.py, forward "
                        f"each of its pinch markers as PINCH on '{LSL_STREAM_NAME_OUT}' and report whether "
                        f"a ground-truth marker falls inside the classified window")
    p.add_argument("--consecutive", type=int, default=2,
                   help="Number of consecutively epochs classified as positive to trigger a marker.")
    p.add_argument("--positive-threshold", type=float, default=0.7,
                   help="Classification threshold for positive event.")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data",
                   help="Directory LSLLiveViewLogger writes to; the most recent recording is used for fine tuning")
    p.add_argument("--subject", default=None,
                   help="Only consider recordings of this subject id when looking for the most recent one")
    return p.parse_args()


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def build_filter(fs):
    nyq = fs / 2.0
    if 45.0 >= nyq:
        sys.exit(f"Sampling rate {fs} Hz is too low for a 45 Hz low-pass.")
    hp = signal.butter(4, 0.3, btype="highpass", fs=fs, output="sos")
    lp = signal.butter(4, 45.0, btype="lowpass", fs=fs, output="sos")
    b, a = signal.iirnotch(50.0, Q=30.0, fs=fs)
    notch = signal.tf2sos(b, a)
    return np.vstack([hp, lp, notch])


class StreamingFilter:
    def __init__(self, sos, n_ch):
        self.sos = sos
        self.n_ch = n_ch
        self.zi = None

    def __call__(self, x):
        """x: (n_ch, n_samples) -> filtered array, same shape."""
        if self.zi is None:
            # Initialise state as if the signal had been sitting at its first
            # value forever; avoids a huge start-up transient from DC offsets.
            zi0 = signal.sosfilt_zi(self.sos)                 # (sections, 2)
            self.zi = zi0[:, None, :] * x[:, 0][None, :, None]  # (sections, n_ch, 2)
        y, self.zi = signal.sosfilt(self.sos, x, axis=-1, zi=self.zi)
        return y


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def load_model(path, device):
    """-> (model, number of channels of its (1, 1, n_ch, n_samples) input)."""
    ep = torch.export.load(path)
    model = ep.module()
    n_ch = ep.example_inputs[0][0].shape[-2]
    print(f"Loaded model from {path} ({n_ch} input channels)")
    return model, n_ch


def make_input(window, device):
    """window: (1, n_ch, n_samples) float array -> channel-wise z-scored (1, 1, n_ch, n_samples)
    torch tensor for the model."""
    w = (window - window.mean(axis=-1, keepdims=True)) / (window.std(axis=-1, keepdims=True) + 1e-8)
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).unsqueeze(0)  # batch dim
    return t.to(device)


@torch.no_grad()
def classify(model, window):
    x = make_input(window, "cuda" if torch.cuda.is_available() else "cpu")
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    out = out.reshape(out.shape[0], -1)
    if out.shape[1] == 1:                       # binary model with a single logit
        p1 = torch.sigmoid(out)
        probs = torch.cat([1 - p1, p1], dim=1)
    else:                                       # assume logits
        probs = torch.softmax(out, dim=1)
    probs = probs[0].cpu().numpy()
    return int(np.argmax(probs)), probs


# --------------------------------------------------------------------------
# Replay markers
# --------------------------------------------------------------------------
class MarkerTracker:
    """Collects ground-truth markers from the replay stream, forwards each of them
    as a PINCH on the output stream and answers 'was there one inside the window
    that just got classified?'."""

    def __init__(self, inlet, window, outlet):
        self.inlet = inlet
        self.window = window
        self.outlet = outlet
        self.stamps = []   # marker timestamps, oldest -> newest

    def flush(self):
        """Drop the markers that queued up while nobody polled."""
        self.inlet.flush()

    def poll(self):
        samples, stamps = self.inlet.pull_chunk(timeout=0.0)
        for t in stamps:
            self.outlet.push_sample(["PINCH"], timestamp=t)
        self.stamps.extend(stamps)

    def seen_within(self, t_end):
        """True if a marker lies in (t_end - window, t_end]."""
        t_start = t_end - self.window
        # markers older than any future window are of no further use
        self.stamps = [t for t in self.stamps if t > t_start]
        return any(t <= t_end for t in self.stamps)


# --------------------------------------------------------------------------
# EEG input
# --------------------------------------------------------------------------
def channel_labels(info):
    """Channel labels from a full StreamInfo (empty list if the stream has none)."""
    labels = []
    ch = info.desc().child("channels").child("channel")
    while not ch.empty():
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return labels


class EEGInput:
    """The EEG inlet plus the channel selection that matches the model input:
    auxiliary channels are dropped, the recording reference is added back if the
    model expects it, and the result is re-referenced to the average, as in
    11_preprocessing.ipynb."""

    def __init__(self, inlet):
        self.inlet = inlet
        info = inlet.info()
        self.fs = info.nominal_srate()
        labels = channel_labels(info)
        if len(labels) == info.channel_count():
            self.picks = [i for i, l in enumerate(labels) if l not in NON_EEG_CHANNELS]
            self.ch_names = [labels[i] for i in self.picks]
        else:
            self.picks = list(range(info.channel_count()))
            self.ch_names = []
        self.restore_reference = False

    @property
    def n_ch(self):
        return len(self.picks) + self.restore_reference

    def match_model(self, n_model_ch):
        """Adapt the channel selection to the number of channels the model expects."""
        missing = n_model_ch - len(self.picks)
        if missing == 1 and REFERENCE_CHANNEL not in self.ch_names:
            self.restore_reference = True
            if self.ch_names:
                self.ch_names.append(REFERENCE_CHANNEL)
        elif missing != 0:
            sys.exit(f"Model expects {n_model_ch} channels, the stream provides {len(self.picks)} "
                     f"{self.ch_names or ''} (+1 with the {REFERENCE_CHANNEL} reference restored).")

    def select(self, samples):
        """samples: list of (n_ch_stream,) rows -> (n_ch, n) average-referenced array."""
        x = np.asarray(samples, dtype=np.float64).T[self.picks]
        if self.restore_reference:
            x = np.vstack([x, np.zeros((1, x.shape[1]))])
        return x - x.mean(axis=0, keepdims=True)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def run_inference(eeg, model, outlet, markers, args, should_stop, on_running=None):
    """Classify the EEG stream until should_stop() returns True. on_running is
    called once, right after the first classification."""
    fs = eeg.fs
    win_len = int(round(CLASSIFY_WINDOW_SECONDS * fs))+1
    hop_len = max(1, int(round(args.hop * fs)))
    settle_len = int(round(CLASSIFY_WINDOW_SECONDS * fs))+1

    # ---- processing state ----
    eeg.inlet.flush()    # drop whatever queued up while we were not classifying
    if markers is not None:
        markers.flush()
    filt = StreamingFilter(build_filter(fs), eeg.n_ch)
    buf = np.zeros((1, eeg.n_ch, win_len), dtype=np.float32)   # rolling window, oldest -> newest
    total = 0            # total samples processed
    since_last = 0       # samples since last classification
    min_samples = win_len + settle_len

    print(f"Window {CLASSIFY_WINDOW_SECONDS}s ({win_len} samples), hop {args.hop}s. "
          f"Discarding first {CLASSIFY_WINDOW_SECONDS}s while filters settle. Ctrl+C to stop.")

    last_was_positive = False # a debouncer for pinch event triggers
    class_buffer = [False] * args.consecutive # the last N predictions. True -> predicted PinchLog
    class_buffer_i = 0
    while not should_stop():
        samples, stamps = eeg.inlet.pull_chunk(timeout=0.1, max_samples=int(fs))
        if markers is not None:
            markers.poll()
        if not stamps:
            continue

        x = eeg.select(samples)                              # (n_ch, n)
        y = filt(x).astype(np.float32)
        n = y.shape[1]

        # update rolling buffer
        if n >= win_len:
            buf[0] = y[:, -win_len:]
        else:
            buf[0] = np.concatenate([buf[0, :, n:], y], axis=1)

        total += n
        since_last += n

        if total >= min_samples and since_last >= hop_len:
            since_last = 0
            t_start = time.perf_counter()
            cls, probs = classify(model, buf)
            latency_ms = (time.perf_counter() - t_start) * 1000

            if on_running is not None:
                on_running()
                on_running = None

            class_buffer[class_buffer_i] = (LABELS[cls] == LABEL_POSITIVE and probs[cls] > args.positive_threshold)
            class_buffer_i = (class_buffer_i + 1) % args.consecutive

            if all(class_buffer) and not last_was_positive:
                outlet.push_sample(["PINCH"], timestamp=stamps[-1])
                last_was_positive = True
            else:
                last_was_positive = False

            buf_vis = render_classification_buffer(class_buffer, class_buffer_i)

            gt_vis = ""
            if markers is not None:
                gt_vis = "  " + status_cell(markers.seen_within(stamps[-1]))

            print(f"\r{LABELS[cls]:>12s}  p={probs[cls]:.2f}  [{buf_vis}]{gt_vis}  ({latency_ms:.1f} ms)   ",
                  end="", flush=True)
    print()


# --------------------------------------------------------------------------
# Phases: backend streams
# --------------------------------------------------------------------------
def parse_phase_end(message):
    """Phase number of an end-of-phase message, or None. Accepts plain tokens
    ('end_phase_1') as well as game event JSON ({"EventName": "END_PHASE_1", ...})."""
    token = message.strip()
    try:
        obj = json.loads(token)
        if isinstance(obj, dict):
            token = str(obj.get("EventName") or "")
    except json.JSONDecodeError:
        pass
    m = MSG_END_PHASE.match(token)
    return int(m.group(1)) if m else None


class PhaseListener:
    """Collects the phase ends (and their LSL timestamps) from the backend stream."""

    def __init__(self, inlet):
        self.inlet = inlet
        self.end_stamps = {}   # phase -> LSL timestamp of its end

    def poll(self, timeout=0.0):
        while True:
            sample, stamp = self.inlet.pull_sample(timeout=timeout)
            if sample is None:
                return
            timeout = 0.0
            phase = parse_phase_end(sample[0])
            if phase is not None and phase not in self.end_stamps:
                self.end_stamps[phase] = stamp
                print(f"\n[phases] phase {phase} ended")

    def has_ended(self, phase):
        self.poll()
        return phase in self.end_stamps


class ReadyPublisher:
    """Publishes the classifier's ready state on the marker outlet and repeats it
    periodically, so a subscriber that connects late still sees it."""

    def __init__(self, outlet):
        self.outlet = outlet
        self.message = None
        self.last_push = 0.0

    def set(self, message):
        print(f"\n[phases] {message}")
        self.message = message
        self.last_push = 0.0
        self.tick()

    def clear(self):
        self.message = None

    def tick(self):
        if self.message is not None and time.monotonic() - self.last_push >= READY_REPUBLISH_SECONDS:
            self.outlet.push_sample([self.message])
            self.last_push = time.monotonic()


# --------------------------------------------------------------------------
# Fine tuning data (see DataAnalysis/11_preprocessing.ipynb)
# --------------------------------------------------------------------------
def find_latest_recording(data_dir, subject=None):
    """Most recent EEG csv written by LSLLiveViewLogger."""
    pattern = f"subject_{subject}/eeg_*.csv" if subject else "*/eeg_*.csv"
    ts_pat = re.compile(r"_(\d{8}_\d{6})\.csv$")
    candidates = [(m.group(1), p) for p in Path(data_dir).glob(pattern) if (m := ts_pat.search(p.name))]
    if not candidates:
        raise FileNotFoundError(f"No recording matching '{pattern}' in {data_dir}")
    return max(candidates)[1]


def read_markers(path):
    """Marker jsonl -> flat DataFrame (nested fields become 'Payload.PinchType' etc.).
    The logger is still writing the file, so an incomplete last line is skipped."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    markers = pd.json_normalize(records)
    return markers.sort_values("timestamp").reset_index(drop=True)


def experiment_intervals(markers):
    """(start, end) of every experiment; an experiment without an end runs to infinity."""
    events = markers[(markers.EventType == "ExperimentManagerLog")
                     & markers.EventName.isin(["EXPERIMENT_START", "EXPERIMENT_END"])]
    intervals, start = [], None
    for name, t in zip(events.EventName, events.timestamp):
        if name == "EXPERIMENT_START":
            start = t
        elif start is not None:
            intervals.append((start, t))
            start = None
    if start is not None:
        intervals.append((start, np.inf))
    return intervals


def ground_truth_pinches(pinches, end_phase_1):
    """Real pinches: in phase 1 the pinch source is the hand tracker (PINCH), afterwards the
    classifier triggers PINCH and the hand tracker's pinches are logged as IGNORED_GROUND_TRUTH."""
    ptype = pinches["Payload.PinchType"]
    truth = (ptype == "IGNORED_GROUND_TRUTH") | ((ptype == "PINCH") & (pinches.timestamp <= end_phase_1))
    pos = pinches[truth].copy()

    # drop pinch double activations: group events closer than .3s and keep the first one
    new_group = (pos.timestamp.diff() > 0.3) | pos.timestamp.diff().isna()
    pos = pos[new_group.to_numpy()]

    # the pinch log of delayed feedback is written .5s after the real pinch
    delayed = pos["ExperimentConfig.feedbackIsDelayed"].fillna(False).astype(bool) & (pos["Payload.PinchType"] == "PINCH")
    pos.loc[delayed, "timestamp"] -= 0.5
    return pos.timestamp.to_numpy()


def empty_marker_timestamps(pinches, segment):
    """One NoPinch timestamp in every gap between two pinches of the same segment that
    leaves MIN_DIST_BETWEEN_EVENTS on both sides."""
    rng = np.random.default_rng()
    out = []
    for _, ts in pinches.timestamp.groupby(segment):
        ts = np.sort(ts.to_numpy())
        for t1, t2 in zip(ts[:-1], ts[1:]):
            low, high = t1 + MIN_DIST_BETWEEN_EVENTS, t2 - MIN_DIST_BETWEEN_EVENTS
            if low < high:
                out.append(rng.uniform(low, high))
    return np.asarray(out)


def build_finetune_dataset(csv_path, backend_end_stamps, restore_reference=False):
    """Phase 1 epochs of the recording as (X, is_positive), X: (n_epochs, n_ch, n_times),
    channel-wise z-scored as in 14_classification.ipynb. The channels match
    EEGInput: with restore_reference the recording reference is the last one."""
    import mne
    mne.set_log_level("WARNING")

    t = re.search(r"_(\d{8}_\d{6})\.csv$", csv_path.name).group(1)
    metadata = csv_path.with_name(f"{csv_path.stem}_metadata.txt").read_text()
    match = re.search(r"Info:\s*(\{.*\})", metadata, re.DOTALL)
    if not match:
        raise ValueError("Info block not found in EEG metadata")
    sfreq = json.loads(match.group(1))["sfreq"]

    eeg_csv = pd.read_csv(csv_path).dropna()   # the last row may be incomplete
    markers = read_markers(csv_path.with_name(f"markers_{t}.jsonl"))

    # phase ends: prefer the game's own log (same file, same clock), else the backend stream
    phase_ends = []
    for phase in range(1, N_PHASES):
        logged = markers[markers.EventName == f"END_PHASE_{phase}"].timestamp
        phase_ends.append(logged.min() if len(logged) else backend_end_stamps.get(phase, np.inf))
    end_phase_1 = phase_ends[0]

    # leave out preparation and tutorial measurements
    starts = markers[markers.EventName == "EXPERIMENT_START"]
    start_ts = starts[starts.ExperimentIndex == 1].timestamp.min()
    if np.isnan(start_ts):
        start_ts = starts.timestamp.min()
    if np.isnan(start_ts):
        raise ValueError("No EXPERIMENT_START marker in the recording")
    intervals = experiment_intervals(markers)
    markers = markers[markers.timestamp > start_ts]

    # only phase 1 is used; experiment indices restart in every phase, so a segment is (phase, experiment)
    pinches = markers[(markers.EventType == "PinchLog") & (markers.timestamp <= end_phase_1)]
    segment = [np.searchsorted(phase_ends, pinches.timestamp.to_numpy()), pinches.ExperimentIndex.to_numpy()]
    pos_ts = ground_truth_pinches(pinches, end_phase_1)
    neg_ts = empty_marker_timestamps(pinches, segment)

    # ---- raw EEG, cropped 10s before the first experiment to let the filters settle ----
    # a stream channel named like the logger's timestamp column is read as 'timestamp.1'
    ch_names = [c for c in eeg_csv.columns if c.split(".")[0] not in NON_EEG_CHANNELS]
    timestamps = eeg_csv.timestamp.to_numpy()
    t_idx = np.searchsorted(timestamps, start_ts - 10)
    t0 = timestamps[t_idx]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg_csv[ch_names].to_numpy()[t_idx:].T * 1e-6, info)
    raw.rename_channels({k: v for k, v in {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}.items() if k in ch_names})
    if restore_reference:
        raw = mne.add_reference_channels(raw, REFERENCE_CHANNEL)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    raw.set_eeg_reference(ref_channels="average")

    # everything outside an experiment is a break
    t_end = t0 + raw.times[-1]
    bad_onsets, bad_durations, last_end = [], [], t0
    for s, e in intervals:
        if s > last_end:
            bad_onsets.append(last_end - t0)
            bad_durations.append(s - last_end)
        last_end = max(last_end, e)
    if last_end < t_end:
        bad_onsets.append(last_end - t0)
        bad_durations.append(t_end - last_end)
    raw.set_annotations(mne.Annotations(bad_onsets, bad_durations, ["BAD_BREAK"] * len(bad_onsets)))

    # 'eeg_light' preprocessing: frequency filters only
    raw.filter(l_freq=0.3, h_freq=None)
    raw.notch_filter(freqs=[50])
    raw.filter(l_freq=None, h_freq=45.)

    # ---- epochs ----
    event_id = {LABEL_NEGATIVE: 1, LABEL_POSITIVE: 2}
    ts = np.concatenate([neg_ts, pos_ts])
    ids = np.concatenate([np.full(len(neg_ts), 1), np.full(len(pos_ts), 2)])
    samples = np.round((ts - t0) * sfreq).astype(int)
    events = np.column_stack([samples, np.zeros_like(samples), ids])
    events = events[np.argsort(events[:, 0])]
    events = events[np.unique(events[:, 0], return_index=True)[1]]
    events = events[(events[:, 0] >= 0) & (events[:, 0] < raw.n_times)]

    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=EPOCH_T_MIN, tmax=EPOCH_T_MAX, preload=True)
    X = epochs.get_data().astype(np.float32)
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-12)
    is_positive = epochs.events[:, 2] == event_id[LABEL_POSITIVE]
    print(f"[finetune] {csv_path.name}: {is_positive.sum()} {LABEL_POSITIVE} / "
          f"{(~is_positive).sum()} {LABEL_NEGATIVE} epochs of shape {X.shape[1:]}")
    return X, is_positive


# --------------------------------------------------------------------------
# Fine tuning
# --------------------------------------------------------------------------
def finetune(model, X, y, device, validation_split=0.2, seed=42):
    """Fine tunes model in place on X: (n, n_ch, n_times), y: (n,) class indices and
    keeps the weights with the lowest validation loss. The exported model cannot be
    put in train mode, so dropout stays off and batch norm statistics stay frozen."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    tr_idx, va_idx = [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        n_val = int(round(validation_split * len(idx)))
        va_idx.append(idx[:n_val])
        tr_idx.append(idx[n_val:])
    tr_idx, va_idx = np.concatenate(tr_idx), np.concatenate(va_idx)

    Xt = torch.from_numpy(X).unsqueeze(1)     # (n, 1, n_ch, n_times)
    yt = torch.from_numpy(y.astype(np.int64))
    n_classes = int(y.max()) + 1
    counts = np.bincount(y[tr_idx], minlength=n_classes).astype(np.float64)
    weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)

    # The model is exported with a batch size of 1 and its constrained layers modify their
    # weights in place during forward, so every sample gets its own forward + backward; the
    # gradients accumulate to those of the class-weighted mean loss over the batch.
    def val_loss():
        with torch.no_grad():
            logits = torch.cat([model(Xt[i:i + 1].to(device)) for i in va_idx])
            return criterion(logits, yt[va_idx].to(device)).item()

    def train_step(batch):
        optimizer.zero_grad()
        batch_weight = weights[yt[batch].to(device)].sum()
        batch_loss = 0.0
        for i in batch:
            yi = yt[i:i + 1].to(device)
            loss = criterion(model(Xt[i:i + 1].to(device)), yi) * weights[yi].sum() / batch_weight
            loss.backward()
            batch_loss += loss.item()
        optimizer.step()
        return batch_loss

    best_loss, best_state = val_loss(), copy.deepcopy(model.state_dict())
    print(f"[finetune] {len(tr_idx)} train / {len(va_idx)} validation epochs, val_loss before {best_loss:.4f}")
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        run_loss = 0.0
        for batch in np.array_split(rng.permutation(tr_idx), max(1, len(tr_idx) // FINETUNE_BATCH_SIZE)):
            run_loss += train_step(batch) * len(batch)
        v_loss = val_loss()
        if v_loss < best_loss:
            best_loss, best_state = v_loss, copy.deepcopy(model.state_dict())
        print(f"[finetune] epoch {epoch:3d}/{FINETUNE_EPOCHS}  loss {run_loss / len(tr_idx):.4f}  "
              f"val_loss {v_loss:.4f}")
    model.load_state_dict(best_state)
    print(f"[finetune] done, best val_loss {best_loss:.4f}")


def load_finetune_data(eeg, win_len, backend_end_stamps, args):
    """-> (csv_path, X, y): the phase 1 epochs of the most recent recording and their class indices."""
    csv_path = find_latest_recording(args.data_dir, args.subject)
    X, is_positive = build_finetune_dataset(csv_path, backend_end_stamps, eeg.restore_reference)
    n_pos, n_neg = int(is_positive.sum()), int((~is_positive).sum())
    if min(n_pos, n_neg) < FINETUNE_MIN_EPOCHS_PER_CLASS:
        raise ValueError(f"not enough epochs ({n_pos} {LABEL_POSITIVE}, {n_neg} {LABEL_NEGATIVE})")
    if X.shape[1:] != (eeg.n_ch, win_len):
        raise ValueError(f"epochs have {X.shape[1]} channels x {X.shape[2]} samples, "
                         f"model expects {eeg.n_ch} x {win_len}")
    y = np.where(is_positive, LABELS.index(LABEL_POSITIVE), LABELS.index(LABEL_NEGATIVE))
    return csv_path, X, y


def finetune_and_save(model, name, X, y, out_dir, device):
    """Fine tunes model on (X, y) and saves it to out_dir. Returns True on success; on
    failure the original weights are restored."""
    print(f"[finetune] fine tuning '{name}'")
    original_state = copy.deepcopy(model.state_dict())
    try:
        finetune(model, X, y, device)
    except Exception as e:
        model.load_state_dict(original_state)
        print(f"[finetune] FAILED, continuing with the original '{name}': {e}")
        return False

    out_path = Path(out_dir) / f"finetuned_{name}_{time.strftime('%Y%m%d_%H%M%S')}.pt2"
    try:
        ep = torch.export.export(model, (torch.from_numpy(X[:1]).unsqueeze(1).to(device),))
        torch.export.save(ep, out_path)
        print(f"[finetune] saved fine tuned '{name}' to {out_path}")
    except Exception as e:
        print(f"[finetune] could not save the fine tuned '{name}': {e}")
    return True


def run_phases(eeg, models, outlet, markers, args, device, win_len):
    """models: [(name, model) for phase 2, (name, model) for phase 3]."""
    ready = ReadyPublisher(outlet)
    print(f"Publishing ready state on '{LSL_STREAM_NAME_OUT}'.")

    streams = resolve_byprop("name", BACKEND_STREAM_NAME_PHASES, timeout=60)
    if not streams:
        sys.exit(f"No '{BACKEND_STREAM_NAME_PHASES}' stream found.")
    phases = PhaseListener(StreamInlet(streams[0], max_buflen=360, processing_flags=proc_clocksync))
    print(f"Listening for phase ends on '{BACKEND_STREAM_NAME_PHASES}'.")

    def phase_over(phase):
        ready.tick()
        return phases.has_ended(phase)

    # ---- phase 1: idle, the logger records the data ----
    ready.set(MSG_READY_PHASE.format(1))
    while not phases.has_ended(1):
        ready.tick()
        if markers is not None:
            markers.poll()
        phases.poll(timeout=0.2)
    ready.clear()

    # ---- fine tune both models on the phase 1 data ----
    print("[phases] fine tuning both models on the phase 1 recording")
    try:
        csv_path, X, y = load_finetune_data(eeg, win_len, phases.end_stamps, args)
    except Exception as e:
        print(f"[finetune] FAILED to build the fine tuning data, continuing with the original models: {e}")
    else:
        for name, model in models:
            finetune_and_save(model, name, X, y, csv_path.parent, device)

    # ---- phases 2 and 3: inference with the first / second fine tuned model ----
    for phase, (name, model) in zip((2, 3), models):
        print(f"[phases] phase {phase}: inference with the fine tuned '{name}'")
        run_inference(eeg, model, outlet, markers, args, should_stop=lambda p=phase: phase_over(p),
                      on_running=lambda p=phase: ready.set(MSG_READY_PHASE.format(p)))
        ready.clear()
    print("[phases] all phases done.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = parse_args()

    # ---- connect to input stream ----
    streams = resolve_byprop("name", LSL_STREAM_NAME_IN, timeout=30)
    if not streams:
        sys.exit("No matching LSL stream found.")
    info_in = streams[0]
    fs = info_in.nominal_srate()
    if fs <= 0:
        sys.exit("Input stream has an irregular sampling rate; this script needs a regular one.")

    inlet = StreamInlet(info_in, max_buflen=30, processing_flags=proc_clocksync | proc_dejitter)
    eeg = EEGInput(inlet)
    print(f"Connected to '{info_in.name()}': {info_in.channel_count()} channels @ {fs} Hz")

    win_len = int(round(CLASSIFY_WINDOW_SECONDS * fs))+1

    # ---- models + dry run to check their input and output ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = [args.model, args.model2]
    loaded = [load_model(path, device) for path in paths]
    if len({n for _, n in loaded}) != 1:
        sys.exit(f"The models expect different numbers of channels: {[n for _, n in loaded]}")
    eeg.match_model(loaded[0][1])
    n_ch = eeg.n_ch
    print(f"Model input: {n_ch} channels {eeg.ch_names or ''}")

    models = []
    for path, (model, _) in zip(paths, loaded):
        model.to(device)
        try:
            _, probs = classify(model, np.zeros((1, n_ch, win_len), dtype=np.float32))
        except Exception as e:
            sys.exit(f"Model '{path}' rejected a dummy input of {n_ch} channels x {win_len} samples: {e}")
        if len(probs) != len(LABELS):
            sys.exit(f"Model '{path}' outputs {len(probs)} classes, expected {len(LABELS)}: {LABELS}")
        models.append((Path(path).stem, model))

    # ---- output stream (irregular rate, string) ----
    info_out = StreamInfo(LSL_STREAM_NAME_OUT, "Classification", 1, 0.0, "string",
                          f"{info_in.source_id() or 'eeg'}_classifier")
    chans = info_out.desc().append_child("channels")
    for name in ["predicted_class"] + [f"p_{l}" for l in LABELS]:
        chans.append_child("channel").append_child_value("label", name)
    outlet = StreamOutlet(info_out)
    print(f"Publishing on '{LSL_STREAM_NAME_OUT}' (type='Classification').")

    # ---- optional replay-marker stream ----
    markers = None
    if args.replay:
        mstreams = resolve_byprop("name", LSL_MARKER_NAME_IN, timeout=10)
        if not mstreams:
            sys.exit(f"--replay given but no '{LSL_MARKER_NAME_IN}' stream found.")
        m_inlet = StreamInlet(mstreams[0], max_buflen=30, processing_flags=proc_clocksync)
        markers = MarkerTracker(m_inlet, CLASSIFY_WINDOW_SECONDS, outlet)
        print(f"Connected to '{LSL_MARKER_NAME_IN}'; its pinch markers are forwarded as PINCH.")

    try:
        run_phases(eeg, models, outlet, markers, args, device, win_len)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
