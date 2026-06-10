import sys
import time
import csv
import datetime
import os
import json
import signal
import subprocess
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # explicit backend so we can control window behaviour
import matplotlib.pyplot as plt
from mne_lsl.lsl import StreamInlet, resolve_streams
from mne_lsl.lsl import StreamInfo as _LSLStreamInfo, StreamOutlet as _LSLStreamOutlet

# --- handle SIGTERM for graceful exit ---
def sigterm_handler(_signo, _stack_frame):
    sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)

if len(sys.argv) < 2:
    print("Usage: python LSLLiveViewLogger.py <subject_id> [use_eeg_sim]")
    sys.exit(1)
SUBJECT_ID = sys.argv[1]
USE_EEG_SIM = len(sys.argv) > 2 and sys.argv[2] == "use_eeg_sim"

MARKER_COLORS = ["#c0392b", "#27ae60", "#2980b9", "#e67e22", "#8e44ad"]

# Channels to hide from the visualization
HIDDEN_CHANNELS = {"X1", "X2", "X3", "TRG"}

# Display order — channels are shown top-to-bottom in this sequence.
# Any channel present in the stream but absent from this list is appended at the end.
CHANNEL_ORDER = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8",
    "Cz", "C3", "C4", "T3", "T4", "T5", "T6", "P3", "P4",
    "O1", "O2", "A1", "A2",
]

# Amplitude scale applied only in the visualisation.
VIS_AMPLITUDE_SCALE = 0.25

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
eeg_inlet = StreamInlet(eeg_stream_info)
eeg_inlet.open_stream()
eeg_correction = eeg_inlet.time_correction(timeout=5.0)

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

eeg_ch_names   = get_channel_names_log_metadata(eeg_inlet.get_sinfo(), "eeg")
eeg_n_channels = len(eeg_ch_names)

# --- CSV logging setup ---
_eeg_csv = open(os.path.join(_data_dir, f"eeg_{eeg_inlet.get_sinfo().name}_{_ts}.csv"), "w", newline="")
_marker_csv = open(os.path.join(_data_dir, f"markers_{_ts}.csv"), "w", newline="")
_eeg_writer = csv.writer(_eeg_csv)
_marker_writer = csv.writer(_marker_csv)
_eeg_writer.writerow(["timestamp"] + eeg_ch_names)
_marker_writer.writerow(["timestamp", "stream", "value"])
_last_eeg_ts = -1.0
WINDOW_SEC = 5.0          # how many seconds to show
SPACING    = 40.0         # vertical spacing between channels (µV)
REFRESH    = 0.10         # plot refresh interval (s)

# --- Visible channel indices in DSI display order ---
_ch_index = {name: i for i, name in enumerate(eeg_ch_names)}
_ordered   = [name for name in CHANNEL_ORDER
              if name in _ch_index and name not in HIDDEN_CHANNELS]
_remainder = [name for name in eeg_ch_names
              if name not in set(CHANNEL_ORDER) and name not in HIDDEN_CHANNELS]
vis_names   = _ordered + _remainder
vis_indices = [_ch_index[name] for name in vis_names]
n_vis       = len(vis_indices)

# DC offset per channel to center signals (estimated on a window)
DC_ALPHA     = 0.001   # EMA coefficient (smaller = slower drift correction)
dc_offset    = np.zeros(eeg_n_channels)

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_facecolor("#111111")
fig.patch.set_facecolor("#111111")

# Prevent the window from stealing focus on every draw
fig.canvas.manager.window.wm_attributes("-topmost", False)
try:
    # Works on most Tk builds — open without grabbing focus
    fig.canvas.manager.window.attributes("-alpha", 1.0)
    fig.canvas.manager.window.lower()   # push behind other windows initially
except Exception:
    pass

ax.set_title("Live EEG", color="#aaaaaa", fontsize=12, pad=10)
ax.set_xlabel("Time (s)", color="#888888")
ax.set_ylabel("Channel", color="#888888")
ax.tick_params(colors="#888888")
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")

# Y-axis: one tick per visible channel
ax.set_yticks([i * SPACING for i in range(n_vis)])
ax.set_yticklabels(vis_names, color="#aaaaaa", fontsize=9)

# Pre-create line objects
LINE_COLOR = "#4caf7d"
lines  = [ax.plot([], [], lw=0.7, color=LINE_COLOR, alpha=0.85)[0] for _ in range(n_vis)]
marker_lines = []
marker_texts = []

plt.ion()
plt.tight_layout()

drawn_marker_ts = set()

# check that all required streams are available
REQUIRED_MARKER_STREAMS = [
    "Experiment.VIBE.ViveOpenXR_Events",
    "Experiment.VIBE.MyoPinch",
    "Experiment.VIBE.GameEventStream",
]
BACKEND_STREAM_NAME = "Experiment.VIBE.Backend.Logger"
MSG_READY = "logger_ready"
STREAM_CHECK_TIMEOUT = 10.0

print("Verifying required marker streams are available...")
_check_deadline = time.time() + STREAM_CHECK_TIMEOUT
_found_names = set()
while time.time() < _check_deadline:
    _present = {s.name for s in resolve_streams(timeout=2.0)}
    _found_names = {n for n in REQUIRED_MARKER_STREAMS if n in _present}
    if len(_found_names) == len(REQUIRED_MARKER_STREAMS):
        break
if len(_found_names) != len(REQUIRED_MARKER_STREAMS):
    _missing = [n for n in REQUIRED_MARKER_STREAMS if n not in _found_names]
    print(f"Required marker stream(s) not found: {_missing}")
    sys.exit(1)
print("All required marker streams are available.")

# Publish readiness on the backend stream so the orchestrator can proceed.
_backend_info = _LSLStreamInfo(
    name=BACKEND_STREAM_NAME,
    stype="Markers",
    n_channels=1,
    sfreq=0.0,
    dtype="string",
    source_id="lsl_live_view_logger",
)
_backend_outlet = _LSLStreamOutlet(_backend_info)
time.sleep(.5) # wait for outlet to be ready
_backend_outlet.push_sample([MSG_READY])

print("Plotting — close the window to stop.")

all_marker_ts = []

EEG_BUFFER_SAMPLES = int(eeg_inlet.get_sinfo().sfreq * WINDOW_SEC)
eeg_buf_data = np.zeros((eeg_n_channels, EEG_BUFFER_SAMPLES))
eeg_buf_ts   = np.zeros(EEG_BUFFER_SAMPLES)

# gracefully quit the loop on external signal
try:
    while plt.fignum_exists(fig.number):
        # --- Fetch EEG ---
        eeg_chunk, eeg_ts = eeg_inlet.pull_chunk(timeout=0.0)
        if eeg_ts.size == 0:
            plt.pause(REFRESH)
            continue

        eeg_ts    = np.array(eeg_ts) + eeg_correction
        eeg_chunk = np.array(eeg_chunk)   # shape: (n_samples, n_channels)

        n_new = len(eeg_ts)
        sfreq = eeg_inlet.get_sinfo().sfreq

        # Reconstruct evenly-spaced timestamps anchored to the last sample
        t_last           = eeg_ts[-1]
        reconstructed_ts = t_last - np.arange(n_new - 1, -1, -1) / sfreq

        # Update DC offset estimate (exponential moving average per channel)
        for j in range(n_new):
            dc_offset += DC_ALPHA * (eeg_chunk[j] - dc_offset)

        # Ring buffer update
        eeg_buf_data = np.roll(eeg_buf_data, -n_new, axis=1)
        eeg_buf_ts   = np.roll(eeg_buf_ts,   -n_new)
        eeg_buf_data[:, -n_new:] = eeg_chunk.T
        eeg_buf_ts[  -n_new:]    = reconstructed_ts

        # CSV logging (raw, uncorrected data)
        for j in range(n_new):
            _eeg_writer.writerow([reconstructed_ts[j]] + eeg_chunk[j].tolist())
        _last_eeg_ts = float(reconstructed_ts[-1])

        # --- Plot visible channels with DC offset removed and amplitude scaled ---
        t_start = eeg_buf_ts[0]
        t_rel   = eeg_buf_ts - t_start
        for plot_i, ch_i in enumerate(vis_indices):
            centred = (eeg_buf_data[ch_i] - dc_offset[ch_i]) * VIS_AMPLITUDE_SCALE
            lines[plot_i].set_data(t_rel, centred + plot_i * SPACING)

        ax.set_xlim(0, WINDOW_SEC)
        ax.set_ylim(-SPACING, n_vis * SPACING)

        # --- Log gaze samples ---
        for inlet, correction, writer in zip(gaze_inlets, gaze_corrections, gaze_writers):
            gaze_samples, gaze_ts = inlet.pull_chunk(timeout=0.0)
            if gaze_ts.size > 0:
                corrected = np.array(gaze_ts) + correction
                for t, row in zip(corrected.tolist(), gaze_samples):
                    writer.writerow([t] + list(row))

        # --- Fetch markers ---
        for idx, (inlet, correction) in enumerate(zip(marker_inlets, marker_corrections)):
            samples, ts = inlet.pull_chunk(timeout=0.0)
            for t, s in zip(ts, samples):
                print(f"{t} [{inlet.name}]: {s}")
            if ts.size > 0:
                corrected = np.array(ts) + correction
                for t, s in zip(corrected.tolist(), samples):
                    _marker_writer.writerow([t, inlet.name, s])
                all_marker_ts.extend([(t, idx) for t in corrected.tolist()])

        # Clear old marker lines
        for ml in marker_lines:
            ml.remove()
        marker_lines.clear()
        for mt in marker_texts:
            mt.remove()
        marker_texts.clear()

        # Redraw markers within the visible window
        for mts, stream_idx in all_marker_ts:
            t_rel_mk = mts - t_start
            color = MARKER_COLORS[stream_idx % len(MARKER_COLORS)]
            if 0 <= t_rel_mk <= WINDOW_SEC:
                vline = ax.axvline(x=t_rel_mk, color=color, lw=1.0, linestyle="--", alpha=0.6)
                txt   = ax.text(t_rel_mk + 0.05, (n_vis - 0.7) * SPACING,
                                "stim", color="#aaaaaa", fontsize=8, alpha=0.7)
                marker_lines.append(vline)
                marker_texts.append(txt)

        # Prune old markers
        all_marker_ts = [(mts, i) for mts, i in all_marker_ts if mts >= eeg_buf_ts[0]]

        if len(drawn_marker_ts) == 1:
            ax.legend(loc="upper right", facecolor="#222222", labelcolor="#aaaaaa", fontsize=9)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(REFRESH)
        _marker_csv.flush()
finally:
    eeg_inlet.close_stream()
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