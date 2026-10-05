"""Offline evaluation of the live inference in LSLService/LiveEEGClassifier.py.

A recording is split into a training part (first half of the experiments, or phase 1 of a live
run) and a test part (the rest). The test part is replayed through the live pipeline (average
reference, causal filter, a 2s window every `hop` seconds, per-window z-scoring) and the pinch
detector (n consecutive windows with p(pinch) > threshold) is scored against the ground-truth
pinches, using the classifier's own functions wherever possible.
"""

import contextlib
import copy
import io
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

LSL_SERVICE_DIR = Path(__file__).resolve().parents[2] / 'LSLService'


def _import_live_classifier():
    """Imports LiveEEGClassifier. pylsl is only needed for the live streams, so it is stubbed
    if it is not installed."""
    try:
        import pylsl  # noqa: F401
    except ImportError:
        stub = types.ModuleType('pylsl')
        for name in ('StreamInlet', 'StreamInfo', 'StreamOutlet', 'resolve_byprop', 'proc_clocksync', 'proc_dejitter'):
            setattr(stub, name, None)
        sys.modules['pylsl'] = stub
    if str(LSL_SERVICE_DIR) not in sys.path:
        sys.path.insert(0, str(LSL_SERVICE_DIR))
    import LiveEEGClassifier
    return LiveEEGClassifier


lec = _import_live_classifier()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SECONDS = lec.CLASSIFY_WINDOW_SECONDS
PINCH_TYPES_EXCLUDED = {'GAZE'}   # gaze selections are not a hand movement
POS = lec.LABELS.index(lec.LABEL_POSITIVE)

# the current fine tuning settings of LiveEEGClassifier
# data: 'mne' -> epochs as in LiveEEGClassifier.build_finetune_dataset (zero-phase filters,
#                z-scored over the whole dataset)
#       'live' -> the windows the live pipeline classifies (causal filters, per-window z-score)
DEFAULT_FINETUNE = dict(lr=lec.FINETUNE_LR, epochs=lec.FINETUNE_EPOCHS,
                        batch_size=lec.FINETUNE_BATCH_SIZE, data='mne')
DEFAULT_DETECTOR = (lec.DETECTOR_CONSECUTIVE, lec.DETECTOR_THRESHOLD)


# --------------------------------------------------------------------------
# Recordings
# --------------------------------------------------------------------------
def _recording_files(directory):
    ts_pat = re.compile(r'_(\d{8}_\d{6})\.csv$')
    return sorted((m.group(1), p) for p in Path(directory).glob('eeg_*.csv') if (m := ts_pat.search(p.name)))


def _read_eeg(csv_path):
    metadata = csv_path.with_name(f'{csv_path.stem}_metadata.txt').read_text()
    sfreq = json.loads(re.search(r'Info:\s*(\{.*\})', metadata, re.DOTALL).group(1))['sfreq']
    eeg_csv = pd.read_csv(csv_path).dropna()
    ch_names = [c for c in eeg_csv.columns if c.split('.')[0] not in lec.NON_EEG_CHANNELS]
    return sfreq, eeg_csv.timestamp.to_numpy(), eeg_csv[ch_names].to_numpy(), ch_names


def _dedupe_and_shift(pinches):
    """Drops pinch double activations (< .3s apart) and moves delayed-feedback pinches back to the
    real pinch, as in 11_preprocessing.ipynb -> sorted pinch timestamps."""
    pinches = pinches.sort_values('timestamp')
    gap = pinches.timestamp.diff()
    pinches = pinches[((gap > 0.3) | gap.isna()).to_numpy()]
    ts = pinches.timestamp.to_numpy().copy()
    # only stimulus selections are delayed, pinches without a target (PINCH_NONE) are logged at once
    delayed = (pinches['ExperimentConfig.feedbackIsDelayed'].fillna(False).astype(bool)
               & (pinches.EventName == 'PINCH_STIMULUS')).to_numpy()
    ts[delayed] -= lec.FEEDBACK_DELAY_SECONDS
    return np.sort(ts)


def _negative_timestamps(pos_ts, intervals, seed):
    """One NoPinch timestamp in every gap between two pinches of the same experiment that leaves
    MIN_DIST_BETWEEN_EVENTS on both sides (LiveEEGClassifier.empty_marker_timestamps, seeded)."""
    rng = np.random.default_rng(seed)
    out = []
    for s, e in intervals:
        ts = pos_ts[(pos_ts >= s) & (pos_ts <= e)]
        for t1, t2 in zip(ts[:-1], ts[1:]):
            low, high = t1 + lec.MIN_DIST_BETWEEN_EVENTS, t2 - lec.MIN_DIST_BETWEEN_EVENTS
            if low < high:
                out.append(rng.uniform(low, high))
    return np.asarray(out)


def in_intervals(ts, intervals):
    inside = np.zeros(len(ts), dtype=bool)
    for s, e in intervals:
        inside |= (ts >= s) & (ts <= e)
    return inside


def _make_recording(name, csv_path, markers, start_ts, train_intervals, test_intervals, pos_ts, seed):
    sfreq, timestamps, data, ch_names = _read_eeg(csv_path)
    intervals = train_intervals + test_intervals
    pos_ts = pos_ts[in_intervals(pos_ts, intervals)]
    neg_ts = _negative_timestamps(pos_ts, intervals, seed)
    return SimpleNamespace(name=name, subject=name.split('/')[0], csv_path=csv_path, sfreq=sfreq, timestamps=timestamps, data=data,
                           ch_names=ch_names, start_ts=start_ts, intervals=intervals,
                           train_intervals=train_intervals, test_intervals=test_intervals,
                           pos_ts=pos_ts, neg_ts=neg_ts)


def load_study_session(subject_dir, session=2, seed=0, skip_hinted=False):
    """A session of the study (data/raw/subject_XX): the first half of its experiments is the
    training part, the second half the test part. Every PinchLog except gaze selections is a
    ground-truth pinch (all of them are hand movements, see 19_induced_error_errp.ipynb).
    skip_hinted drops the experiments with hinted pinches entirely (before the split); the EEG
    keeps running through them, but none of their windows or pinches are used."""
    subject_dir = Path(subject_dir)
    _, csv_path = _recording_files(subject_dir)[session - 1]
    t = re.search(r'_(\d{8}_\d{6})\.csv$', csv_path.name).group(1)
    markers = lec.read_markers(csv_path.with_name(f'markers_{t}.jsonl'))

    exp = markers[(markers.EventType == 'ExperimentManagerLog') & (markers.ExperimentIndex >= 1)]
    starts = exp[exp.EventName == 'EXPERIMENT_START'].groupby('ExperimentIndex').timestamp.min()
    ends = exp[exp.EventName == 'EXPERIMENT_END'].groupby('ExperimentIndex').timestamp.min()
    hinted = exp[exp.EventName == 'EXPERIMENT_START'].groupby('ExperimentIndex')['ExperimentConfig.pinchIsHinted']         .first().fillna(False).astype(bool)
    intervals = [(starts[i], ends.get(i, np.inf)) for i in starts.index if not (skip_hinted and hinted.get(i, False))]
    n_train = len(intervals) // 2

    pinches = markers[(markers.EventType == 'PinchLog') & (markers.ExperimentIndex >= 1)
                      & ~markers['Payload.PinchType'].isin(PINCH_TYPES_EXCLUDED)]
    return _make_recording(f'{subject_dir.name}/session_{session}', csv_path, markers, starts.min(),
                           intervals[:n_train], intervals[n_train:], _dedupe_and_shift(pinches), seed)


def load_live_recording(csv_path, seed=0):
    """A recording of a live run (LSLLiveViewLogger, with END_PHASE_N markers): phase 1 is the
    training part, phases 2 and 3 the test part. Ground truth as in
    LiveEEGClassifier.ground_truth_pinches: hand tracker pinches in phase 1, IGNORED_GROUND_TRUTH
    afterwards. Recordings without IGNORED_GROUND_TRUTH (the hand tracker still drove the game)
    use the PINCH events of all phases."""
    csv_path = Path(csv_path)
    t = re.search(r'_(\d{8}_\d{6})\.csv$', csv_path.name).group(1)
    markers = lec.read_markers(csv_path.with_name(f'markers_{t}.jsonl'))

    end_phase_1 = markers[markers.EventName == 'END_PHASE_1'].timestamp.min()
    starts = markers[markers.EventName == 'EXPERIMENT_START']
    start_ts = starts[starts.ExperimentIndex == 1].timestamp.min()
    intervals = [(s, e) for s, e in lec.experiment_intervals(markers) if s >= start_ts]
    end_ts = markers.timestamp.max()
    intervals = [(s, min(e, end_ts)) for s, e in intervals]
    train = [(s, min(e, end_phase_1)) for s, e in intervals if s < end_phase_1]
    test = [(max(s, end_phase_1), e) for s, e in intervals if e > end_phase_1]

    pinches = markers[(markers.EventType == 'PinchLog') & (markers.timestamp > start_ts)]
    if (pinches['Payload.PinchType'] == 'IGNORED_GROUND_TRUTH').any():
        pos_ts = np.sort(lec.ground_truth_pinches(pinches, end_phase_1))
    else:
        print(f'{csv_path.name}: no IGNORED_GROUND_TRUTH markers, using the PINCH events of all phases as ground truth')
        pos_ts = _dedupe_and_shift(pinches[pinches['Payload.PinchType'] == 'PINCH'])
    return _make_recording(csv_path.parent.name, csv_path, markers, start_ts, train, test, pos_ts, seed)


def load_recordings(directory, session=2, seed=0):
    """All recordings in directory: a live run if it has END_PHASE_1 markers, else a study
    session (`session` selects which one)."""
    out = []
    for t, csv_path in _recording_files(directory):
        markers_path = csv_path.with_name(f'markers_{t}.jsonl')
        if markers_path.exists() and 'END_PHASE_1' in markers_path.read_text(encoding='utf-8'):
            out.append(load_live_recording(csv_path, seed))
    return out or [load_study_session(directory, session, seed)]


# --------------------------------------------------------------------------
# Live pipeline replay
# --------------------------------------------------------------------------
def live_signal(rec):
    """The recording as the live pipeline sees it: recording reference restored, average
    reference, causal filters. Starts 10s before the first experiment for the filters to settle.
    -> (timestamps, (n_ch, n) float32 array)."""
    i0 = np.searchsorted(rec.timestamps, rec.start_ts - 10)
    x = rec.data[i0:].T.astype(np.float64)
    x = np.vstack([x, np.zeros((1, x.shape[1]))])      # Pz, the recording reference
    x = x - x.mean(axis=0, keepdims=True)
    y = lec.StreamingFilter(lec.build_filter(rec.sfreq), x.shape[0])(x).astype(np.float32)
    return rec.timestamps[i0:], y


def window_len(rec):
    return int(round(WINDOW_SECONDS * rec.sfreq)) + 1


def replay(model, rec, signal, hop):
    """Classifies a window every hop seconds, as LiveEEGClassifier.run_inference does
    -> (window end timestamps, p(pinch) per window)."""
    ts, y = signal
    win_len = window_len(rec)
    ends = np.arange(2 * win_len, y.shape[1] + 1, max(1, int(round(hop * rec.sfreq))))
    p = np.array([lec.classify(model, y[None, :, e - win_len:e])[1][POS] for e in ends])
    return ts[ends - 1], p


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------
def f_beta(tp, fp, fn, beta=1.0):
    b2 = beta ** 2
    return (1 + b2) * tp / max(1e-12, (1 + b2) * tp + b2 * fn + fp)


def detector_counts(t_end, p, pinch_ts, intervals, consecutive, threshold):
    """(hits, false alarms, misses) of the detector, counting only detections and pinches inside
    intervals (as LiveEEGClassifier.optimize_detector)."""
    counted = in_intervals(t_end, intervals)
    pinch_ts = pinch_ts[in_intervals(pinch_ts, intervals)]
    det = lec.trigger_indices(p > threshold, consecutive)
    det = det[counted[det]]
    tp = lec.count_hits(t_end[det], pinch_ts, WINDOW_SECONDS)
    return tp, len(det) - tp, len(pinch_ts) - tp


def optimize_detector(t_end, p, pinch_ts, intervals, beta=1.0):
    """-> (consecutive, threshold) with the best F-beta over the grids of LiveEEGClassifier."""
    best = None
    for consecutive in lec.OPTIMIZE_CONSECUTIVE:
        for threshold in lec.OPTIMIZE_THRESHOLDS:
            score = f_beta(*detector_counts(t_end, p, pinch_ts, intervals, consecutive, threshold), beta)
            if best is None or score > best[0]:
                best = (score, consecutive, float(threshold))
    return best[1], best[2]


def detector_metrics(t_end, p, pinch_ts, intervals, consecutive, threshold):
    tp, fp, fn = detector_counts(t_end, p, pinch_ts, intervals, consecutive, threshold)
    return dict(consecutive=consecutive, threshold=threshold, hits=tp, false_alarms=fp, misses=fn,
                precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
                accuracy=tp / max(1, tp + fp + fn),   # no true negatives in pinch detection
                f1=f_beta(tp, fp, fn, 1.0), f05=f_beta(tp, fp, fn, 0.5))


def detection_times(t_end, p, intervals, consecutive, threshold):
    """Timestamps at which the detector publishes a PINCH (end of the window completing the run),
    restricted to intervals as in detector_counts."""
    det = lec.trigger_indices(p > threshold, consecutive)
    return t_end[det[in_intervals(t_end[det], intervals)]]


def match_latencies(detection_ts, pinch_ts, window=WINDOW_SECONDS):
    """The one-to-one matching of LiveEEGClassifier.count_hits -> latency (detection - pinch) of every
    detection, NaN for false alarms."""
    matched = np.zeros(len(pinch_ts), dtype=bool)
    latency = np.full(len(detection_ts), np.nan)
    for i, t in enumerate(detection_ts):
        candidates = np.flatnonzero(~matched & (pinch_ts > t - window) & (pinch_ts <= t))
        if len(candidates):
            matched[candidates[0]] = True
            latency[i] = t - pinch_ts[candidates[0]]
    return latency


def event_locked(t_end, p, events, lags):
    """p(pinch) of the window ending at event + lag (linear interpolation) -> (n_events, n_lags)."""
    return np.interp(np.asarray(events)[:, None] + np.asarray(lags)[None, :], t_end, p, left=np.nan, right=np.nan)


def event_auc(t_end, p, rec, intervals):
    """Window-level ROC AUC: p(pinch) of the window aligned like a training epoch (ending
    EPOCH_T_MAX after the event) for every pinch / no pinch event in intervals."""
    pos = rec.pos_ts[in_intervals(rec.pos_ts, intervals)]
    neg = rec.neg_ts[in_intervals(rec.neg_ts, intervals)]
    if not len(pos) or not len(neg):
        return np.nan
    events = np.concatenate([pos, neg]) + lec.EPOCH_T_MAX
    idx = np.clip(np.searchsorted(t_end, events), 0, len(t_end) - 1)
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], p[idx])


# --------------------------------------------------------------------------
# Fine tuning
# --------------------------------------------------------------------------
def train_events(rec):
    """Pinch / no pinch events of the training part whose epoch lies inside an experiment."""
    def keep(ts):
        inside = np.zeros(len(ts), dtype=bool)
        for s, e in rec.train_intervals:
            inside |= (ts + lec.EPOCH_T_MIN >= s) & (ts + lec.EPOCH_T_MAX <= e)
        return ts[inside]
    return keep(rec.pos_ts), keep(rec.neg_ts)


def finetune_dataset(rec, signal, data='mne'):
    """-> (X, y): training-part epochs (n, n_ch, n_times) and their class indices."""
    pos_ts, neg_ts = train_events(rec)
    if data == 'mne':
        part = SimpleNamespace(**{**vars(rec), 'intervals': rec.train_intervals, 'pos_ts': pos_ts, 'neg_ts': neg_ts})
        with contextlib.redirect_stdout(io.StringIO()):
            X, is_positive = lec.build_finetune_dataset(part, rec.name, restore_reference=True)
    elif data == 'live':
        ts, y = signal
        win_len = window_len(rec)
        events = np.concatenate([neg_ts, pos_ts])
        ends = np.searchsorted(ts, events + lec.EPOCH_T_MAX) + 1
        ok = (ends >= win_len) & (ends <= y.shape[1])
        X = np.stack([y[:, e - win_len:e] for e in ends[ok]])
        X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)
        is_positive = np.r_[np.zeros(len(neg_ts), bool), np.ones(len(pos_ts), bool)][ok]
    else:
        raise ValueError(f'unknown fine tuning data {data!r}')
    return X.astype(np.float32), np.where(is_positive, POS, 1 - POS)


@contextlib.contextmanager
def _finetune_settings(cfg):
    names = dict(lr='FINETUNE_LR', epochs='FINETUNE_EPOCHS', batch_size='FINETUNE_BATCH_SIZE')
    old = {k: getattr(lec, v) for k, v in names.items()}
    for k, v in names.items():
        setattr(lec, v, cfg[k])
    try:
        yield
    finally:
        for k, v in names.items():
            setattr(lec, v, old[k])


def load_model(path):
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = lec.load_model(path, DEVICE)
    return model.to(DEVICE)


def finetuned_model(model_path, X, y, cfg, verbose=False):
    """A fresh copy of the model at model_path, fine tuned with LiveEEGClassifier.finetune."""
    model = load_model(model_path)
    cfg = {**DEFAULT_FINETUNE, **cfg}
    out = sys.stdout if verbose else io.StringIO()
    with _finetune_settings(cfg), contextlib.redirect_stdout(out):
        lec.finetune(model, X, y, DEVICE)
    return model


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(rec, model_path, finetune=None, hop=0.25, betas=(1.0, 0.5), signal=None):
    """Runs one model over a recording, as LiveEEGClassifier would.

    finetune: None for the original model, else a dict of fine tuning settings (see DEFAULT_FINETUNE).
    For every detector setting the test part is scored:
      'default'       DETECTOR_CONSECUTIVE / DETECTOR_THRESHOLD of LiveEEGClassifier
      'opt_F{beta}'   optimized on the training part (what LiveEEGClassifier does after phase 1)
      'oracle_F{beta}' optimized on the test part itself (upper bound)
    -> dict with the per-window trace and a list of result rows."""
    signal = signal if signal is not None else live_signal(rec)
    if finetune is None:
        model = load_model(model_path)
        n_train = None
    else:
        X, y = finetune_dataset(rec, signal, finetune.get('data', DEFAULT_FINETUNE['data']))
        model = finetuned_model(model_path, X, y, finetune)
        n_train = (int((y == POS).sum()), int((y != POS).sum()))
    t_end, p = replay(model, rec, signal, hop)
    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    detectors = {'default': DEFAULT_DETECTOR}
    for beta in betas:
        detectors[f'opt_F{beta:g}'] = optimize_detector(t_end, p, rec.pos_ts, rec.train_intervals, beta)
        detectors[f'oracle_F{beta:g}'] = optimize_detector(t_end, p, rec.pos_ts, rec.test_intervals, beta)

    base = dict(recording=rec.name, model=Path(model_path).stem, finetuned=finetune is not None,
                finetune=json.dumps(finetune, sort_keys=True) if finetune else '',
                auc_train=event_auc(t_end, p, rec, rec.train_intervals),
                auc_test=event_auc(t_end, p, rec, rec.test_intervals),
                n_finetune_pos=n_train[0] if n_train else 0, n_finetune_neg=n_train[1] if n_train else 0)
    rows = [{**base, 'detector': name, **detector_metrics(t_end, p, rec.pos_ts, rec.test_intervals, *params)}
            for name, params in detectors.items()]
    return dict(rows=rows, t_end=t_end, p=p)


def pooled(df, by):
    """Pools hits / false alarms / misses over recordings (micro average) and recomputes the metrics."""
    g = df.groupby(by, sort=False)[['hits', 'false_alarms', 'misses']].sum()
    tp, fp, fn = g.hits, g.false_alarms, g.misses
    g['precision'] = tp / (tp + fp).clip(lower=1)
    g['recall'] = tp / (tp + fn).clip(lower=1)
    g['accuracy'] = tp / (tp + fp + fn).clip(lower=1)
    g['f1'] = [f_beta(*c, 1.0) for c in zip(tp, fp, fn)]
    g['f05'] = [f_beta(*c, 0.5) for c in zip(tp, fp, fn)]
    g['auc_test'] = df.groupby(by, sort=False).auc_test.mean()
    return g.reset_index()


def run_grid(recs, models, finetunes, hop=0.25, betas=(1.0, 0.5), cache=None, recompute=False, keep_traces=True):
    """Evaluates every model with every fine tuning setting on every recording.

    models: {label: path, or a function recording -> path (e.g. a leave-one-subject-out model per subject)},
    finetunes: {run label: None (original model) or fine tuning dict}.
    Results are cached in `cache` (a .pkl path) unless recompute is set.
    -> (DataFrame of result rows with 'model' and 'run' columns, {(recording, model, run): (t_end, p)})."""
    from tqdm.auto import tqdm

    cache = Path(cache) if cache else None
    if cache is not None and cache.exists() and not recompute:
        out = pd.read_pickle(cache)
        return out['rows'], out['traces']

    rows, traces = [], {}
    jobs = [(m, r) for m in models for r in finetunes]
    for rec in tqdm(recs, desc='recordings'):
        signal = live_signal(rec)
        for model, run in jobs:
            path = models[model](rec) if callable(models[model]) else models[model]
            res = evaluate(rec, path, finetunes[run], hop=hop, betas=betas, signal=signal)
            rows += [{**row, 'model': model, 'run': run} for row in res['rows']]
            if keep_traces:
                traces[(rec.name, model, run)] = (res['t_end'], res['p'])
    rows = pd.DataFrame(rows)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({'rows': rows, 'traces': traces}, cache)
    return rows, traces
