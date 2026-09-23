#!/usr/bin/env python3
"""
eeg_classify.py - Classify EEG signals in real time. The last 2 seconds are filtered
and given to a torch model for classification. The results are published on a
separate LSL stream.
"""

import argparse
import sys
import time

import numpy as np
import torch
from pylsl import StreamInlet, StreamInfo, StreamOutlet, resolve_byprop, proc_clocksync, proc_dejitter
from scipy import signal

LSL_STREAM_NAME_IN  = "Experiment.VIBE.EEG"
LSL_STREAM_NAME_OUT = "Experiment.VIBE.Pinch"
LSL_MARKER_NAME_IN  = "Experiment.VIBE.MarkerReplay"
CLASSIFY_WINDOW_SECONDS = 2

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
    p.add_argument("model", help="Path to the pre-trained model (TorchScript, or a fully pickled nn.Module)")
    p.add_argument("--hop", type=float, default=0.5,
                   help="Seconds between classifications (default 0.5)")
    p.add_argument("--zscore", action="store_true",
                   help="Z-score each channel within the window (use only if you did so in training)")
    p.add_argument("--labels", default=None, help="Comma-separated class labels (metadata only)")
    p.add_argument("--replay", action="store_true",
                   help=f"Also subscribe to the '{LSL_MARKER_NAME_IN}' stream from EEGReplay.py and "
                        f"report whether a ground-truth marker falls inside the classified window")
    p.add_argument("--consecutive", type=int, default=2,
                   help="Number of consecutively epochs classified as positive to trigger a marker.")
    p.add_argument("--finetune", action="store_true",
                   help="Create a fine tuned model based on the input model.")
    p.add_argument("--positive-threshold", type=float, default=0.8,
                   help="Classification threshold for positive event.")
    return p.parse_args()


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def build_filter(fs):
    nyq = fs / 2.0
    if 45.0 >= nyq:
        sys.exit(f"Sampling rate {fs} Hz is too low for a 45 Hz low-pass.")
    hp = signal.butter(4, 1.0, btype="highpass", fs=fs, output="sos")
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
    ep = torch.export.load(path)
    model = ep.module()
    print(f"Loaded model from {path}")
    return model


def make_input(window, args, device):
    """window: (n_ch, n_samples) float array -> torch tensor for the model."""
    w = window
    if args.zscore:
        w = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-8)

    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).unsqueeze(0)  # batch dim

    return t.to(device)


@torch.no_grad()
def classify(model, window, args):
    x = make_input(window, args, "cuda" if torch.cuda.is_available() else "cpu")
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
    """Collects ground-truth markers from the replay stream and answers
    'was there one inside the window that just got classified?'."""

    def __init__(self, inlet, window):
        self.inlet = inlet
        self.window = window
        self.stamps = []   # marker timestamps, oldest -> newest

    def poll(self):
        samples, stamps = self.inlet.pull_chunk(timeout=0.0)
        if stamps:
            self.stamps.extend(stamps)

    def seen_within(self, t_end):
        """True if a marker lies in (t_end - window, t_end]."""
        t_start = t_end - self.window
        # markers older than any future window are of no further use
        self.stamps = [t for t in self.stamps if t > t_start]
        return any(t <= t_end for t in self.stamps)


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
    n_ch = info_in.channel_count()
    print(f"Connected to '{info_in.name()}': {n_ch} channels @ {fs} Hz")
    if fs <= 0:
        sys.exit("Input stream has an irregular sampling rate; this script needs a regular one.")

    inlet = StreamInlet(info_in, max_buflen=30, processing_flags=proc_clocksync | proc_dejitter)

    # ---- optional replay-marker stream ----
    markers = None
    if args.replay:
        mstreams = resolve_byprop("name", LSL_MARKER_NAME_IN, timeout=10)
        if not mstreams:
            sys.exit(f"--replay given but no '{LSL_MARKER_NAME_IN}' stream found.")
        m_inlet = StreamInlet(mstreams[0], max_buflen=30, processing_flags=proc_clocksync)
        markers = MarkerTracker(m_inlet, CLASSIFY_WINDOW_SECONDS)
        print(f"Connected to '{LSL_MARKER_NAME_IN}' for ground-truth markers.")

    win_len = int(round(CLASSIFY_WINDOW_SECONDS * fs))+1
    hop_len = max(1, int(round(args.hop * fs)))
    settle_len = int(round(CLASSIFY_WINDOW_SECONDS * fs))+1

    # ---- model + dry run to find number of classes ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    model.to(device)
    try:
        _, probs = classify(model, np.zeros((1, n_ch, win_len), dtype=np.float32), args)
    except Exception as e:
        sys.exit(f"Model rejected a dummy input of {n_ch} channels x {win_len} samples "
                 f": {e}")
    n_classes = len(probs)
    labels = args.labels.split(",") if args.labels else [f"class_{i}" for i in range(n_classes)]
    if len(labels) != n_classes:
        print(f"Warning: {len(labels)} labels given but model outputs {n_classes} classes; ignoring labels.")
        labels = [f"class_{i}" for i in range(n_classes)]
    print(f"Model outputs {n_classes} classes: {labels}")

    # ---- output stream (irregular rate, float32) ----
    info_out = StreamInfo(LSL_STREAM_NAME_OUT, "Classification", 1, 0.0, "string",
                          f"{info_in.source_id() or 'eeg'}_classifier")
    chans = info_out.desc().append_child("channels")
    for name in ["predicted_class"] + [f"p_{l}" for l in labels]:
        chans.append_child("channel").append_child_value("label", name)
    outlet = StreamOutlet(info_out)
    print(f"Publishing on '{LSL_STREAM_NAME_OUT}' (type='Classification').")

    # ---- processing state ----
    filt = StreamingFilter(build_filter(fs), n_ch)
    buf = np.zeros((1, n_ch, win_len), dtype=np.float32)   # rolling window, oldest -> newest
    total = 0            # total samples processed
    since_last = 0       # samples since last classification
    min_samples = win_len + settle_len

    print(f"Window {CLASSIFY_WINDOW_SECONDS}s ({win_len} samples), hop {args.hop}s. "
          f"Discarding first {CLASSIFY_WINDOW_SECONDS}s while filters settle. Ctrl+C to stop.")

    last_was_positive = False # a debouncer for pinch event triggers
    class_buffer = [False] * args.consecutive # the last N predictions. True -> predicted PinchLog
    class_buffer_i = 0
    try:
        while True:
            samples, stamps = inlet.pull_chunk(timeout=0.1, max_samples=int(fs))
            if markers is not None:
                markers.poll()
            if not stamps:
                continue

            x = np.asarray(samples, dtype=np.float64).T          # (n_ch, n)
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
                cls, probs = classify(model, buf, args)
                latency_ms = (time.perf_counter() - t_start) * 1000

                class_buffer[class_buffer_i] = (labels[cls] == 'PinchLog' and probs[cls] > args.positive_threshold)
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

                print(f"\r{labels[cls]:>12s}  p={probs[cls]:.2f}  [{buf_vis}]{gt_vis}  ({latency_ms:.1f} ms)   ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()