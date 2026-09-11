import json
import pathlib
import pickle
import shutil
import tempfile
import zipfile

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
        value._data = value._data.astype('float32', copy=False)
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
        # preload=True: data must be fully read before the temp dir is cleaned up
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
    path = pathlib.Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        manifest = {}

        for subject, sessions in raw_dataset.items():
            manifest[subject] = {}
            for session, session_dict in sessions.items():
                session_dir = tmp / subject / session
                session_dir.mkdir(parents=True, exist_ok=True)
                manifest[subject][session] = {}
                for key, value in session_dict.items():
                    _save_leaf(value, session_dir, key, manifest[subject][session])

        with open(tmp / 'manifest.json', 'w') as f:
            json.dump(manifest, f)

        archive = shutil.make_archive(str(tmp / '_archive'), 'zip', root_dir=tmp)
        shutil.move(archive, path)


def read_raw_dataset(path: pathlib.Path) -> dict:
    path = pathlib.Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(tmp)

        with open(tmp / 'manifest.json') as f:
            manifest = json.load(f)

        data = {}
        for subject, sessions in manifest.items():
            data[subject] = {}
            for session, leaves in sessions.items():
                session_dir = tmp / subject / session
                data[subject][session] = {
                    key: _load_leaf(entry, session_dir)
                    for key, entry in leaves.items()
                }
        return data