# live_plot.py
import sys
import time
import csv
import datetime
import os
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mne_lsl.lsl import resolve_streams
from mne_lsl.stream import StreamLSL
from mne_lsl.lsl import StreamInlet, resolve_streams

if len(sys.argv) < 2:
    print("Usage: python LSLLiveViewLogger.py <subject_id> [use_eeg_sim]")
    sys.exit(1)
SUBJECT_ID = sys.argv[1]
USE_EEG_SIM = len(sys.argv) > 2 and sys.argv[2] == "use_eeg_sim"

MARKER_COLORS = ["#ff6b6b", "#6bffb8", "#6bb8ff", "#ffb86b", "#d96bff"]

# --- Output directory (created early so gaze/marker/eeg setup can all use it) ---
_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_data_dir = os.path.join("data", f"subject_{SUBJECT_ID}")
os.makedirs(_data_dir, exist_ok=True)

# --- Optionally start EEG simulator ---
_sim_proc = None
if USE_EEG_SIM:
    _sim_script = os.path.join(os.path.dirname(__file__), "EEGSim.py")
    _sim_proc = subprocess.Popen([sys.executable, _sim_script])
    print("Started EEGSim.py, waiting for stream to appear...")
    time.sleep(2.0)

# --- Connect to streams ---
print("Connecting to EEG stream...")
eeg_streams = resolve_streams(timeout=5.0)
eeg_stream_info = next((s for s in eeg_streams if s.stype == "EEG"), None)
if eeg_stream_info is None:
    print("No EEG stream found.")
    if _sim_proc:
        _sim_proc.terminate()
    sys.exit(1)
print(f"Found EEG stream: {eeg_stream_info.name}")
eeg = StreamLSL(name=eeg_stream_info.name, bufsize=5.0)
eeg.connect(acquisition_delay=0.05)
eeg_correction = eeg._inlet.time_correction(timeout=5.0)

print("Discovering marker and gaze streams...")
all_streams = resolve_streams(timeout=2.0)
marker_streams_info = [s for s in all_streams if s.stype == "Markers"]
gaze_streams_info   = [s for s in all_streams if s.stype == "Gaze"]

marker_inlets = []
marker_corrections = []
for s in marker_streams_info:
    inlet = StreamInlet(s)
    inlet.open_stream()
    correction = inlet.time_correction(timeout=5.0)
    marker_inlets.append(inlet)
    marker_corrections.append(correction)
    print(f"Connected to marker stream: {s.name}")
if not marker_inlets:
    print("No marker streams found, continuing without markers.")

def get_channel_names_log_metadata(info, file_prefix):
    try:
        csv_path = os.path.join(_data_dir, f"{file_prefix}_{info.name}_{_ts}_metadata.txt")
        f = open(csv_path, "w", newline="\n")
        # w = csv.writer(f)
        channels = info.get_channel_names()
        info_meta = info.get_channel_info()
        f.write("Channels: " + str(channels) + "\n")
        f.write("Types: " + str(info.get_channel_types()) + "\n")
        f.write("Units: " + str(info.get_channel_units()) + "\n")
        f.write("Info: "  + json.dumps(info_meta.to_json_dict()) + "\n")

        f.close()

        return channels

    except Exception as e:
        print(e)
        exit()
        pass

    return [f"ch_{i}" for i in range(info.n_channels)]

gaze_inlets = []
gaze_corrections = []
gaze_writers = []
gaze_csvs = []
for s in gaze_streams_info:
    inlet = StreamInlet(s)
    inlet.open_stream()
    correction = inlet.time_correction(timeout=5.0)
    ch_names = get_channel_names_log_metadata(inlet.get_sinfo(), "gaze")
    csv_path = os.path.join(_data_dir, f"gaze_{s.name}_{_ts}.csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["timestamp"] + ch_names)
    gaze_inlets.append(inlet)
    gaze_corrections.append(correction)
    gaze_writers.append(w)
    gaze_csvs.append(f)
    print(f"Connected to gaze stream: {s.name} ({len(ch_names)} ch)")
if not gaze_inlets:
    print("No gaze streams found, continuing without gaze logging.")

eeg_ch_names   = get_channel_names_log_metadata(eeg._inlet.get_sinfo(), "eeg")
eeg_n_channels = len(eeg_ch_names)

# --- CSV logging setup ---
_eeg_csv = open(os.path.join(_data_dir, f"eeg_{eeg._inlet.get_sinfo().name}_{_ts}.csv"), "w", newline="")
_marker_csv = open(os.path.join(_data_dir, f"markers_{_ts}.csv"), "w", newline="")
_eeg_writer = csv.writer(_eeg_csv)
_marker_writer = csv.writer(_marker_csv)
_eeg_writer.writerow(["timestamp"] + eeg_ch_names)
_marker_writer.writerow(["timestamp", "stream", "value"])
_last_eeg_ts = -1.0
WINDOW_SEC = 5.0          # how many seconds to show
SPACING    = 40.0         # vertical spacing between channels (µV)
REFRESH    = 0.05         # plot refresh interval (s)

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")

ax.set_title("Live EEG + Markers", color="white", fontsize=14, pad=10)
ax.set_xlabel("Time (s)", color="white")
ax.set_ylabel("Channel", color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444466")

# Y-axis: one tick per channel
ax.set_yticks([i * SPACING for i in range(eeg_n_channels)])
ax.set_yticklabels(eeg_ch_names, color="white", fontsize=9)

# Pre-create line objects (one per channel) and marker lines list
colors = plt.cm.cool(np.linspace(0, 1, eeg_n_channels))
lines  = [ax.plot([], [], lw=0.8, color=colors[i])[0] for i in range(eeg_n_channels)]
marker_lines = []   # vertical lines added dynamically
marker_texts = []

plt.ion()
plt.tight_layout()

# Track which marker timestamps we've already drawn
drawn_marker_ts = set()

print("Plotting — close the window to stop.")

all_marker_ts = []

while plt.fignum_exists(fig.number):
    # --- Fetch EEG ---
    eeg_data, eeg_ts = eeg.get_data(winsize=WINDOW_SEC)
    eeg_ts = eeg_ts + eeg_correction

    if eeg_ts.size == 0:
        plt.pause(REFRESH)
        continue

    t_start = eeg_ts[0]
    t_rel   = eeg_ts - t_start      # relative time axis starting at 0

    # --- Log new EEG samples to CSV ---
    new_mask = eeg_ts > _last_eeg_ts
    if new_mask.any():
        for j in np.where(new_mask)[0]:
            _eeg_writer.writerow([eeg_ts[j]] + eeg_data[:, j].tolist())
        _last_eeg_ts = float(eeg_ts[new_mask][-1])

    # Update EEG lines
    for i, line in enumerate(lines):
        offset = i * SPACING
        line.set_data(t_rel, eeg_data[i] + offset)

    ax.set_xlim(0, WINDOW_SEC)
    ax.set_ylim(-SPACING, eeg_n_channels * SPACING)

    # --- Log gaze samples ---
    for inlet, correction, writer in zip(gaze_inlets, gaze_corrections, gaze_writers):
        gaze_samples, gaze_ts = inlet.pull_chunk(timeout=0.0)
        if gaze_ts.size > 0:
            corrected = np.array(gaze_ts) + correction
            for t, row in zip(corrected.tolist(), gaze_samples):
                writer.writerow([t] + list(row))

    # --- Fetch markers ---
    # 1. Collect new markers into the absolute timestamp list
    for i, (inlet, correction) in enumerate(zip(marker_inlets, marker_corrections)):
        samples, ts = inlet.pull_chunk(timeout=0.0)
        for t, s in zip(ts, samples):
            print(f"{t} [{inlet.name}]: {s}")
        if ts.size > 0:
            corrected = np.array(ts) + correction
            for t, s in zip(corrected.tolist(), samples):
                _marker_writer.writerow([t, inlet.name, s])
            all_marker_ts.extend([(t,i) for t in corrected.tolist()])

    # 2. Clear old marker lines (they get redrawn below)
    for ml in marker_lines:
        ml.remove()
    marker_lines.clear()

    for mt in marker_texts:
        mt.remove()
    marker_texts.clear()

    # 3. Redraw all markers that fall within the current visible window
    for ts, stream_idx in all_marker_ts:
        t_rel_mk = ts - t_start
        color = MARKER_COLORS[stream_idx % len(MARKER_COLORS)]
        if 0 <= t_rel_mk <= WINDOW_SEC:
            vline = ax.axvline(x=t_rel_mk, color=color, lw=1.5, linestyle="--", alpha=0.85)
            txt = ax.text(t_rel_mk + 0.05, (eeg_n_channels - 0.7) * SPACING,
                    "stim", color="#ff6b6b", fontsize=8, alpha=0.9)
            marker_lines.append(vline)
            marker_texts.append(txt)

    # 4. Prune markers older than the visible window to avoid unbounded list growth
    all_marker_ts = [(ts, i) for ts, i in all_marker_ts if ts >= eeg_ts[0]]

    # Refresh legend only once
    if len(drawn_marker_ts) == 1:
        ax.legend(loc="upper right", facecolor="#2a2a4a", labelcolor="white", fontsize=9)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(REFRESH)

eeg.disconnect()
for inlet in marker_inlets:
    inlet.close_stream()
for inlet, f in zip(gaze_inlets, gaze_csvs):
    inlet.close_stream()
    f.close()
_eeg_csv.close()
_marker_csv.close()
if _sim_proc:
    _sim_proc.terminate()
    _sim_proc.wait()
    print("EEGSim stopped.")
print("Disconnected.")