#!/usr/bin/env python3
"""
eeg_replay.py - Replay a CSV file of EEG samples onto an LSL stream.

If a sibling markers_<stamp>.jsonl file is found next to the CSV, its PinchLog
events are replayed in sync on a second LSL stream.
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from pylsl import StreamInfo, StreamOutlet, local_clock

LSL_STREAM_NAME_OUT = "Experiment.VIBE.EEG"
LSL_MARKER_NAME_OUT = "Experiment.VIBE.MarkerReplay"
MARKER_PINCH_TYPES = ("PINCH", "PINCH_IGNORED")
TIME_COL = "timestamp"
CHUNK = 10

def parse_args():
    p = argparse.ArgumentParser(description="Replay a CSV of EEG data to an LSL stream.")
    p.add_argument("csv", help="Path to CSV file (rows = samples, columns = channels)")
    p.add_argument("--srate", type=float, default=300.0, help="Sampling rate of the file in Hz (default 300)")
    p.add_argument("--drop-cols", default="",
                   help="Comma-separated column names to exclude (e.g. time,label). "
                        "Non-numeric columns are always dropped.")
    p.add_argument("--loop", action="store_true", help="Loop the file forever")
    p.add_argument("--markers", default=None,
                   help="Path to the markers .jsonl file (default: markers_<stamp>.jsonl next to the CSV)")
    p.add_argument("--no-markers", action="store_true", help="Do not replay markers")
    return p.parse_args()


def load_csv(path, drop_cols):
    df = pd.read_csv(path)
    # Keep the recording clock before it is (usually) dropped: markers are
    # timestamped in the same time base and need it as an anchor.
    t0_file = float(df[TIME_COL].iloc[0]) if TIME_COL in df.columns else None
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
    return numeric, t0_file


def find_marker_file(csv_path):
    """markers_<YYYYmmdd_HHMMSS>.jsonl sitting next to the CSV, same stamp."""
    d = os.path.dirname(os.path.abspath(csv_path))
    m = re.search(r"(\d{8}_\d{6})", os.path.basename(csv_path))
    if not m:
        return None
    cand = os.path.join(d, f"markers_{m.group(1)}.jsonl")
    return cand if os.path.exists(cand) else None


def load_markers(path):
    """-> list of (timestamp, pinch_type), sorted by timestamp."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("EventType") != "PinchLog":
                continue
            payload = ev.get("Payload") or {}
            ptype = payload.get("PinchType")
            if ptype not in MARKER_PINCH_TYPES:
                continue
            ts = ev.get(TIME_COL)
            if ts is None:
                continue
            out.append((float(ts), ptype))
    out.sort(key=lambda e: e[0])
    return out


def main():
    args = parse_args()
    df, t0_file = load_csv(args.csv, args.drop_cols)
    data = df.to_numpy(dtype=np.float32)
    n_samples, n_ch = data.shape
    labels = [str(c) for c in df.columns]
    print(f"Loaded {n_samples} samples x {n_ch} channels "
          f"({n_samples / args.srate:.1f} s at {args.srate} Hz)")

    # --- markers ------------------------------------------------------------
    markers = []
    marker_outlet = None
    if not args.no_markers:
        mpath = find_marker_file(args.csv)
        markers = load_markers(mpath)
        print(f"Loaded {len(markers)} {'/'.join(MARKER_PINCH_TYPES)} markers from {os.path.basename(mpath)}")

    # --- LSL outlets --------------------------------------------------------
    info = StreamInfo(LSL_STREAM_NAME_OUT, "EEG", n_ch, args.srate, "float32", "eeg_source_1")
    chans = info.desc().append_child("channels")
    for lab in labels:
        ch = chans.append_child("channel")
        ch.append_child_value("label", lab)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    outlet = StreamOutlet(info, chunk_size=CHUNK, max_buffered=360)
    print(f"Streaming '{LSL_STREAM_NAME_OUT}'. Ctrl+C to stop.")

    if markers:
        minfo = StreamInfo(LSL_MARKER_NAME_OUT, "Markers", 1, 0.0, "string", "eeg_source_1_markers")
        mchans = minfo.desc().append_child("channels")
        mchans.append_child("channel").append_child_value("label", "PinchType")
        marker_outlet = StreamOutlet(minfo)
        print(f"Streaming '{LSL_MARKER_NAME_OUT}' (type='Markers').")

    # --- Playback loop -----------------------------------------------------
    # Wall-clock scheduling: sample k is due at t0 + k / (srate * speed).
    # This avoids cumulative drift from sleep() inaccuracies.
    eff_rate = args.srate
    dt = 1.0 / eff_rate
    t0 = local_clock()
    sent = 0  # total samples sent since t0
    pos = 0   # position within the file
    m_idx = 0        # next marker to emit
    m_epoch = t0     # local_clock time matching t0_file for the current pass

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

            # Emit every marker whose recording time has now elapsed, keeping
            # its own timestamp so consumers see it where it really happened.
            if marker_outlet is not None:
                while m_idx < len(markers):
                    m_ts, m_type = markers[m_idx]
                    m_due = m_epoch + (m_ts - t0_file)
                    if m_due > due:
                        break
                    marker_outlet.push_sample([m_type], timestamp=m_due)
                    m_idx += 1

            sent += n
            pos = end
            if pos >= n_samples:
                if args.loop:
                    pos = 0
                    m_idx = 0
                    m_epoch = t0 + sent * dt
                else:
                    print("End of file reached.")
                    break
    except KeyboardInterrupt:
        print("\nStopped.")

    # give consumers a moment to drain the buffer
    time.sleep(0.5)


if __name__ == "__main__":
    main()
