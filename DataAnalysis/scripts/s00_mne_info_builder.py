"""
mne_info_builder.py — Construct an mne.Info object from the metadata file format.

The metadata gives us channel names, types, units, sampling rate, and some
header info. It does NOT give us channel positions (all NaN), so we apply a
standard montage. It also doesn't distinguish auxiliary channels from EEG, so
we re-type them based on naming conventions.

Channel naming notes:
  - T3/T4/T5/T6 are the legacy 10-20 names. Modern 10-10 (and MNE's default
    standard_1020 montage) uses T7/T8/P7/P8 for the same positions. We rename
    so the montage applies cleanly.
  - X1/X2/X3 are typically auxiliary (EOG, EMG, ECG). Without channel-by-channel
    documentation we conservatively mark them as 'misc'. If you know the
    mapping (e.g. X1=EOG, X2=ECG), set it explicitly via `aux_channel_types`.
  - A2 is a mastoid reference channel.
  - TRG is the trigger / stim channel.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import mne
import numpy as np


# Legacy → modern 10-10 names. Same scalp positions, just renamed.
LEGACY_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Path filename pattern:
#   eeg_[stream_name]_[YYYYMMDD]_[HHMMSS]_metadata.txt
# stream_name may itself contain underscores, so we anchor on the date+time
# block at the end and capture everything between 'eeg_' and that block.
_FILENAME_RE = re.compile(
    r"^eeg_(?P<stream>.+)_(?P<date>\d{8})_(?P<time>\d{6})_metadata"
    r"\.[^.]+$"
)
# Subject directory: 'subject_<id>' anywhere in the path.
_SUBJECT_RE = re.compile(r"subject_(?P<sid>[^/\\]+)")


def parse_path(path: Path | str) -> dict:
    """Extract subject, stream, and recording datetime from a metadata path.

    Expected layout:
      .../subject_<id>/eeg_<stream>_<YYYYMMDD>_<HHMMSS>_metadata.<ext>

    Returns
    -------
    dict with keys:
      subject_id : str
      stream     : str
      datetime   : naive datetime.datetime  (no timezone — caller localizes)

    Raises ValueError if the path doesn't match the expected pattern.
    """
    path = Path(path)
    parts = path.as_posix()  # forward slashes regardless of OS

    sm = _SUBJECT_RE.search(parts)
    if not sm:
        raise ValueError(
            f"Could not find 'subject_<id>' in path: {path}"
        )
    fm = _FILENAME_RE.match(path.name)
    if not fm:
        raise ValueError(
            f"Filename doesn't match expected pattern "
            f"'eeg_<stream>_<YYYYMMDD>_<HHMMSS>_metadata.<ext>': {path.name}"
        )

    dt = _dt.datetime.strptime(fm["date"] + fm["time"], "%Y%m%d%H%M%S")
    return {
        "subject_id": sm["sid"],
        "stream": fm["stream"],
        "datetime": dt,
    }


def build_info(
    metadata_path: Path | str | dict,
    aux_channel_types: dict[str, str] | None = None,
    montage_name: str = "standard_1020",
    rename_legacy: bool = True,
    timezone: str = "Europe/Zurich",
    subject_id: str | None = None,
    meas_datetime: _dt.datetime | None = None,
    stream_name: str | None = None,
) -> tuple[mne.Info, dict[str, str]]:
    """Build an mne.Info from the metadata format.

    When `metadata_path` is a path, the filename and parent directory are
    parsed to extract subject_id, stream, and recording datetime, which are
    stored in info['subject_info'], info['meas_date'], and info['description']
    respectively. You can override any of these via the explicit kwargs.

    Parameters
    ----------
    metadata_path
        Path to the metadata file (parsed for subject + datetime), or a dict
        already loaded (in which case path metadata is unavailable and you
        should pass subject_id / meas_datetime explicitly if needed).
    aux_channel_types
        Map of {channel_name: type} for non-EEG channels. Defaults assume
        X1-X3 are 'misc'. Override with e.g. {'X1': 'eog', 'X2': 'ecg'} if
        you know better.
    montage_name
        MNE built-in montage to apply.
    rename_legacy
        If True, rename T3/T4/T5/T6 → T7/T8/P7/P8 so the montage applies.
    timezone
        IANA timezone the recording datetime is in. Default Europe/Zurich.
        MNE stores meas_date in UTC, so the local time gets converted.
    subject_id, meas_datetime, stream_name
        Override values parsed from the path. Useful when passing a dict.

    Returns
    -------
    info, rename_map
    """
    # --- Load metadata ----------------------------------------------------
    path_meta: dict | None = None
    if isinstance(metadata_path, (str, Path)):
        p = Path(metadata_path)
        text = p.read_text()
        try:
            meta = json.loads(text)
        except json.JSONDecodeError:
            meta = _parse_loose_metadata(text)
        # Parse the path. We don't fail hard if it doesn't match — the
        # caller may have a non-standard layout — but we surface a warning.
        try:
            path_meta = parse_path(p)
        except ValueError as e:
            print(f"Path parsing skipped: {e}")
    else:
        meta = metadata_path

    # Resolve final subject / datetime / stream values (explicit kwargs win)
    final_subject = subject_id or (path_meta["subject_id"] if path_meta else None)
    final_dt_naive = meas_datetime or (path_meta["datetime"] if path_meta else None)
    final_stream = stream_name or (path_meta["stream"] if path_meta else None)

    ch_names: list[str] = list(meta["Channels"])
    ch_types_raw: list[str] = list(meta["Types"])
    info_dict: dict = meta["Info"]
    sfreq: float = float(info_dict["sfreq"])

    # --- Decide channel types ---------------------------------------------
    # The metadata marks everything 'EEG'. Override based on channel name:
    #   TRG          → 'stim'  (trigger channel)
    #   X1, X2, X3   → user-specified or 'misc'
    #   A2           → 'eeg'   (mastoid; treated as a regular EEG until we
    #                          re-reference, at which point we may drop it)
    if aux_channel_types is None:
        aux_channel_types = {}

    ch_types: list[str] = []
    for name, t_raw in zip(ch_names, ch_types_raw):
        if name == "TRG":
            ch_types.append("stim")
        elif name in aux_channel_types:
            ch_types.append(aux_channel_types[name])
        elif name in {"X1", "X2", "X3"}:
            ch_types.append("misc")  # safe default
        else:
            ch_types.append("eeg")

    # --- Rename legacy channel names so the montage applies ---------------
    rename_map: dict[str, str] = {}
    if rename_legacy:
        rename_map = {old: new for old, new in LEGACY_TO_MODERN.items()
                      if old in ch_names}
        ch_names = [rename_map.get(n, n) for n in ch_names]

    # --- Build the Info object --------------------------------------------
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Apply line frequency if we know it (Switzerland / Europe = 50 Hz; the
    # metadata had it as null). This is metadata only; it doesn't filter.
    with info._unlock():
        info["line_freq"] = 50.0
        # Carry over the lowpass info for completeness (informational only).
        if info_dict.get("lowpass"):
            info["lowpass"] = float(info_dict["lowpass"])
        if info_dict.get("highpass"):
            info["highpass"] = float(info_dict["highpass"])

        # Subject info — `his_id` is the canonical key for the participant
        # identifier in MNE / FIF format.
        if final_subject is not None:
            info["subject_info"] = {"his_id": str(final_subject)}

        # Measurement date — must be timezone-aware UTC.
        if final_dt_naive is not None:
            local = final_dt_naive.replace(tzinfo=ZoneInfo(timezone))
            info["meas_date"] = local.astimezone(_dt.timezone.utc)

        # Description — useful place for the stream name.
        if final_stream is not None:
            info["description"] = f"stream={final_stream}"

    # --- Apply the montage ------------------------------------------------
    # `on_missing='ignore'` so that the auxiliary channels (X1-X3, TRG)
    # without 10-20 positions don't raise. Only the EEG channels need
    # positions for downstream processing.
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage, on_missing="ignore", match_case=False)

    return info, rename_map


def _parse_loose_metadata(text: str) -> dict:
    """Parse the 'Channels: [...]\\nTypes: [...]\\n...Info: {json}' format."""
    out: dict = {}
    # Each top-level field is on one line of the form 'Name: <python literal>'.
    # The Info field is JSON. Channels/Types/Units are Python list literals
    # that happen to also be valid JSON if they only contain strings — which
    # they do here.
    import ast
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "Info":
            out[key] = json.loads(value)
        else:
            try:
                out[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                out[key] = value
    return out


def raw_from_data_and_metadata(
    data: np.ndarray,
    metadata_path: Path | str | dict,
    data_units: str = "microvolts",
    **kwargs,
) -> mne.io.RawArray:
    """Convenience: build a Raw from a (n_channels, n_samples) array.

    MNE expects volts internally. If your data is in microvolts (as the
    metadata indicates), pass data_units='microvolts' and we'll scale.
    """
    info, _ = build_info(metadata_path, **kwargs)
    if data.shape[0] != info["nchan"]:
        raise ValueError(
            f"data has {data.shape[0]} channels, info has {info['nchan']}"
        )
    if data_units == "microvolts":
        data = data * 1e-6
    elif data_units != "volts":
        raise ValueError(f"Unknown data_units: {data_units}")
    return mne.io.RawArray(data, info)


if __name__ == "__main__":
    # Self-test 1: path parsing
    test_path = Path("some/path/data/subject_P042/"
                     "eeg_main_20240315_143022_metadata.txt")
    parsed = parse_path(test_path)
    print(f"Parsed path: {parsed}")
    assert parsed["subject_id"] == "P042"
    assert parsed["stream"] == "main"
    assert parsed["datetime"] == _dt.datetime(2024, 3, 15, 14, 30, 22)

    # Stream name with underscores
    p2 = Path("data/subject_S001/eeg_left_motor_cortex_20231101_090000_metadata.txt")
    parsed2 = parse_path(p2)
    print(f"Parsed (multi-word stream): {parsed2}")
    assert parsed2["stream"] == "left_motor_cortex"

    # Self-test 2: build_info with explicit metadata dict + path-style kwargs
    example = {
        "Channels": ["P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz", "Pz",
                     "Fp1", "Fp2", "T3", "T5", "O1", "O2", "X3", "X2",
                     "F7", "F8", "X1", "A2", "T6", "T4", "TRG"],
        "Types": ["EEG"] * 24,
        "Units": ["microvolts"] * 24,
        "Info": {"sfreq": 300.0, "lowpass": 150.0, "highpass": 0.0},
    }
    info, renames = build_info(
        example,
        aux_channel_types={"X1": "eog", "X2": "ecg"},
        subject_id="P042",
        meas_datetime=_dt.datetime(2024, 3, 15, 14, 30, 22),
        stream_name="main",
    )
    print(f"\nInfo summary:")
    print(f"  channels:    {info['nchan']} @ {info['sfreq']} Hz")
    print(f"  subject:     {info['subject_info']}")
    print(f"  meas_date:   {info['meas_date']} (UTC)")
    print(f"  description: {info['description']}")
    print(f"  line_freq:   {info['line_freq']}")
    print(f"  renamed:     {renames}")
    eeg_picks = mne.pick_types(info, eeg=True)
    has_pos = sum(1 for i in eeg_picks
                  if not np.isnan(info["chs"][i]["loc"][0]))
    print(f"  EEG positions resolved: {has_pos}/{len(eeg_picks)}")