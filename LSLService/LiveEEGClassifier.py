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

LSL_STREAM_NAME_IN  = ""
LSL_STREAM_NAME_OUT = ""
CLASSIFY_WINDOW_SECONDS = 2



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
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
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
    x = make_input(window, args, "cuda")
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

    win_len = int(round(args.window * fs))
    hop_len = max(1, int(round(args.hop * fs)))
    settle_len = int(round(CLASSIFY_WINDOW_SECONDS * fs))

    # ---- model + dry run to find number of classes ----
    model = load_model(args.model, "cuda")
    model.to("cuda")
    try:
        _, probs = classify(model, np.zeros((n_ch, win_len), dtype=np.float32), args)
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
    buf = np.zeros((n_ch, win_len), dtype=np.float32)   # rolling window, oldest -> newest
    total = 0            # total samples processed
    since_last = 0       # samples since last classification
    min_samples = win_len + settle_len

    print(f"Window {CLASSIFY_WINDOW_SECONDS}s ({win_len} samples), hop {args.hop}s. "
          f"Discarding first {CLASSIFY_WINDOW_SECONDS}s while filters settle. Ctrl+C to stop.")

    try:
        while True:
            samples, stamps = inlet.pull_chunk(timeout=0.1, max_samples=int(fs))
            if not stamps:
                continue

            x = np.asarray(samples, dtype=np.float64).T          # (n_ch, n)
            y = filt(x).astype(np.float32)
            n = y.shape[1]

            # update rolling buffer
            if n >= win_len:
                buf = y[:, -win_len:]
            else:
                buf = np.concatenate([buf[:, n:], y], axis=1)

            total += n
            since_last += n

            if total >= min_samples and since_last >= hop_len:
                since_last = 0
                t_start = time.perf_counter()
                cls, probs = classify(model, buf, args)
                latency_ms = (time.perf_counter() - t_start) * 1000
                outlet.push_sample([float(cls)] + probs.astype(float).tolist(), timestamp=stamps[-1])
                print(f"\r{labels[cls]:>12s}  p={probs[cls]:.2f}  ({latency_ms:.1f} ms)   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()