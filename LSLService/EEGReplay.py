#!/usr/bin/env python3
"""
eeg_replay.py - Replay a CSV file of EEG samples onto an LSL stream.
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
from pylsl import StreamInfo, StreamOutlet, local_clock

LSL_STREAM_NAME_OUT = ""
CHUNK = 10

def parse_args():
    p = argparse.ArgumentParser(description="Replay a CSV of EEG data to an LSL stream.")
    p.add_argument("csv", help="Path to CSV file (rows = samples, columns = channels)")
    p.add_argument("--srate", type=float, default=300.0, help="Sampling rate of the file in Hz (default 300)")
    p.add_argument("--drop-cols", default="",
                   help="Comma-separated column names to exclude (e.g. time,label). "
                        "Non-numeric columns are always dropped.")
    p.add_argument("--loop", action="store_true", help="Loop the file forever")
    return p.parse_args()


def load_csv(path, drop_cols):
    df = pd.read_csv(path)
    drop = [c.strip() for c in drop_cols.split(",") if c.strip()]
    missing = [c for c in drop if c not in df.columns]
    if missing:
        sys.exit(f"Columns not found in CSV: {missing}. Available: {list(df.columns)}")
    df = df.drop(columns=drop)
    # keep only numeric columns
    numeric = df.select_dtypes(include=[np.number])
    skipped = [c for c in df.columns if c not in numeric.columns]
    if skipped:
        print(f"Skipping non-numeric columns: {skipped}")
    if numeric.shape[1] == 0:
        sys.exit("No numeric channel columns left in CSV.")
    return numeric


def main():
    args = parse_args()
    df = load_csv(args.csv, args.drop_cols)
    data = df.to_numpy(dtype=np.float32)
    n_samples, n_ch = data.shape
    labels = [str(c) for c in df.columns]
    print(f"Loaded {n_samples} samples x {n_ch} channels "
          f"({n_samples / args.srate:.1f} s at {args.srate} Hz)")

    # --- LSL outlet --------------------------------------------------------
    info = StreamInfo(LSL_STREAM_NAME_OUT, "EEG", n_ch, args.srate, "float32", "eeg_source_1")
    chans = info.desc().append_child("channels")
    for lab in labels:
        ch = chans.append_child("channel")
        ch.append_child_value("label", lab)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    outlet = StreamOutlet(info, chunk_size=CHUNK, max_buffered=360)
    print(f"Streaming '{LSL_STREAM_NAME_OUT}'. Ctrl+C to stop.")

    # --- Playback loop -----------------------------------------------------
    # Wall-clock scheduling: sample k is due at t0 + k / (srate * speed).
    # This avoids cumulative drift from sleep() inaccuracies.
    eff_rate = args.srate
    dt = 1.0 / eff_rate
    t0 = local_clock()
    sent = 0  # total samples sent since t0
    pos = 0   # position within the file

    try:
        while True:
            end = min(pos + CHUNK, n_samples)
            chunk = data[pos:end]
            n = len(chunk)

            # Wait until the last sample of this chunk is "due"
            due = t0 + (sent + n) * dt
            delay = due - local_clock()
            if delay > 0:
                time.sleep(delay)

            # Timestamp refers to the LAST sample in the chunk; pylsl back-dates
            # earlier samples using the nominal sampling rate.
            outlet.push_chunk(chunk.tolist(), timestamp=due)

            sent += n
            pos = end
            if pos >= n_samples:
                if args.loop:
                    pos = 0
                else:
                    print("End of file reached.")
                    break
    except KeyboardInterrupt:
        print("\nStopped.")

    # give consumers a moment to drain the buffer
    time.sleep(0.5)


if __name__ == "__main__":
    main()