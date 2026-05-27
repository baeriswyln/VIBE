# simulate_streams.py
import time
import numpy as np
from mne_lsl.lsl import StreamInfo, StreamOutlet

# --- EEG Stream ---
eeg_info = StreamInfo(
    name="Experiment.VIBE.EEG",
    stype="EEG",
    n_channels=8,
    sfreq=256.0,
    dtype="float32",
    source_id="sim_eeg_001"
)

# Name the channels
channels = eeg_info.desc.append_child("channels")
for ch in ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]:
    chan = channels.append_child("channel")
    chan.append_child_value("label", ch)
    chan.append_child_value("unit", "microvolts")

eeg_outlet = StreamOutlet(eeg_info)


print("Streaming EEG + Markers... Ctrl+C to stop.")

t = 0.0
samples_per_chunk = 16          # push in chunks of 16 samples
dt = samples_per_chunk / 256.0  # time per chunk
next_marker_time = 2.0          # send first marker after 2s

while True:
    # Simulate 8-channel EEG: alpha wave + noise
    t_samples = t + np.arange(samples_per_chunk) / 256.0  # shape: (16,)

    chunk = (
            10.0 * np.sin(2 * np.pi * 10.0 * t_samples[:, np.newaxis])  # alpha wave
            + np.random.randn(samples_per_chunk, 8) * 3.0  # noise
    ).astype(np.float32)

    eeg_outlet.push_chunk(chunk)

    t += dt
    time.sleep(dt)