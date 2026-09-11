import json
import pathlib
import pickle
import shutil

import mne
import numpy as np
import pandas as pd


def _save_leaf(value, out_dir: pathlib.Path, key: str, manifest: dict):
    if isinstance(value, mne.io.BaseRaw):
        value.load_data()
        value._data = value._data.astype('float32', copy=False)
        fname = f"{key}_raw.fif"
        value.save(out_dir / fname, overwrite=True)
        manifest[key] = {'type': 'raw', 'file': fname}

    elif isinstance(value, mne.BaseEpochs):
        value.load_data()
        # MNE enforces this suffix convention for epochs files
        fname = f"{key}-epo.fif"
        value.save(out_dir / fname, overwrite=True)
        manifest[key] = {'type': 'epochs', 'file': fname}

    elif isinstance(value, pd.DataFrame):
        fname = f"{key}.parquet"
        value.to_parquet(out_dir / fname, compression='zstd')
        manifest[key] = {'type': 'dataframe', 'file': fname}

    elif isinstance(value, np.ndarray):
        fname = f"{key}.npy"
        np.save(out_dir / fname, value)
        manifest[key] = {'type': 'ndarray', 'file': fname}

    elif isinstance(value, str):
        manifest[key] = {'type': 'str', 'value': value}

    else:
        # covers eeg_metadata (dict) and anything else unforeseen
        fname = f"{key}.pkl"
        with open(out_dir / fname, 'wb') as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        manifest[key] = {'type': 'pickle', 'file': fname}


def _load_leaf(entry: dict, in_dir: pathlib.Path):
    kind = entry['type']
    if kind == 'raw':
        return mne.io.read_raw_fif(in_dir / entry['file'], preload=True)
    if kind == 'epochs':
        return mne.read_epochs(in_dir / entry['file'], preload=True)
    if kind == 'dataframe':
        return pd.read_parquet(in_dir / entry['file'])
    if kind == 'ndarray':
        return np.load(in_dir / entry['file'])
    if kind == 'str':
        return entry['value']
    if kind == 'pickle':
        with open(in_dir / entry['file'], 'rb') as f:
            return pickle.load(f)
    raise ValueError(f"Unknown leaf type '{kind}' in manifest")


def write_raw_dataset(raw_dataset: dict, path: pathlib.Path):
    """Persist raw_dataset as a plain directory tree at `path` (typically somewhere under
    this project's data/ folder), instead of a zip archive. Any notebook/kernel pointed at
    the same `path` can then read the underlying .fif/.parquet files directly off disk --
    no per-kernel unzip into a throwaway temp dir, and no risk of that temp dir disappearing
    out from under an already-loaded (but still file-backed) mne object.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # stage the write in a sibling directory, then swap it in, so a crash mid-write can't
    # leave `path` half-written
    staging = path.with_name(f'.{path.name}.tmp')
    if staging.exists():
        shutil.rmtree(staging)

    try:
        manifest = {}
        for subject, sessions in raw_dataset.items():
            manifest[subject] = {}
            for session, session_value in sessions.items():
                session_dir = staging / subject / session
                session_dir.mkdir(parents=True, exist_ok=True)

                if isinstance(session_value, dict):
                    # a session holding multiple named leaves, e.g. raw_dataset[subject][session] == {'eeg': ..., 'markers': ...}
                    manifest[subject][session] = {}
                    for key, value in session_value.items():
                        _save_leaf(value, session_dir, key, manifest[subject][session])
                else:
                    # a session that is itself a single leaf, e.g. epoch_dataset[subject][session] == <Epochs>
                    leaf_manifest = {}
                    _save_leaf(session_value, session_dir, 'session', leaf_manifest)
                    manifest[subject][session] = {'__leaf__': leaf_manifest['session']}

        with open(staging / 'manifest.json', 'w') as f:
            json.dump(manifest, f)

        if path.exists():
            shutil.rmtree(path)
        staging.rename(path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_raw_dataset(path: pathlib.Path, subjects_to_drop=[]) -> dict:
    """Read a dataset directory written by write_raw_dataset. Reads straight from `path` on
    disk -- shared by every notebook/kernel pointed at the same data/ folder, with nothing
    to extract or copy first.
    """
    path = pathlib.Path(path)
    with open(path / 'manifest.json') as f:
        manifest = json.load(f)

    data = {}
    for subject, sessions in manifest.items():
        if subject in subjects_to_drop:
            continue
        data[subject] = {}
        for session, leaves in sessions.items():
            session_dir = path / subject / session
            if '__leaf__' in leaves:
                data[subject][session] = _load_leaf(leaves['__leaf__'], session_dir)
            else:
                data[subject][session] = {
                    key: _load_leaf(entry, session_dir)
                    for key, entry in leaves.items()
                }
    return data
