"""
Multi-Branch GAT-GRU-Transformer for EEG motor imagery.

Reimplementation of Wang, Z. & Wang, Y. (2025), "Multi-branch GAT-GRU-transformer
for explainable EEG-based finger motor imagery classification", Front. Hum.
Neurosci. 19:1599960.

    result = train_gat_gru_transformer(epochs_train, y_train, epochs_test, y_test)

X may be an mne.Epochs object or a (n_trials, n_channels, n_times) array. With
Epochs, the sampling rate, channel names and electrode positions are read from
`info`, so the distance-based graph works without a hand-maintained montage table.

Architecture (paper sections 2.3-2.4):

    x : (N, C, T)
      |
    +-+--------------------+---------------------+
  spatial               temporal              frequency
  hierarchical GAT      GRU -> Transformer    FIR bank -> one
  over channels         + temporal attention  1-D CNN per band
      |                     |                     |
      +--- concat tokens ---+---------------------+
                    |
        1-D CNN head -> global avg pool -> FC(n_classes)
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from models.wandb import *
from models.EEGNet import _to_array, check_decim
from models.augment import augment_eeg, DEFAULT_MAX_SHIFT, DEFAULT_SCALE_RANGE, DEFAULT_SIGMA

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, firwin, hilbert
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# Physiological bands of paper section 2.3.3.
BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


# =============================================================================
# Input handling: mne.Epochs or ndarray
# =============================================================================
def _unpack(X, name: str = "X", decim: int = 1) -> Tuple[np.ndarray, Any]:
    """
    Returns (array (N, C, T), mne.Info or None), keeping every `decim`-th sample.

    Epochs go through the notebook's `_to_array` when it exists, so whatever
    scaling or channel picking it does is applied consistently -- including the
    decimation and its low-pass check (see models.EEGNet.check_decim). The returned
    info still describes the full-rate data.
    """
    if hasattr(X, "get_data") and hasattr(X, "info"):          # mne.Epochs
        data = _to_array(X, decim)
        return np.asarray(data, dtype=np.float32), X.info

    x = np.asarray(X.detach().cpu() if isinstance(X, torch.Tensor) else X,
                   dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"{name} must be (n_trials, n_channels, n_times); got {x.shape}")
    if x.shape[1] > x.shape[2]:
        warnings.warn(f"{name} looks like (n_trials, n_times, n_channels); transposing")
        x = np.transpose(x, (0, 2, 1))
    check_decim(1.0, decim)
    return (np.ascontiguousarray(x[..., ::decim]) if decim > 1 else x), None


def _model_decim(model) -> int:
    """Decimation a trained model expects its input to get (1 for models from before
    `decim` existed)."""
    return getattr(model, "decim", 1)


def _montage(info, n_channels: int) -> Tuple[List[str], Optional[np.ndarray]]:
    """Channel names and 3-D electrode positions from an mne.Info."""
    if info is None:
        return [f"ch{i}" for i in range(n_channels)], None

    names = list(info["ch_names"])
    pos = np.array([ch["loc"][:3] for ch in info["chs"]], dtype=float)
    if pos.shape[0] != n_channels or not np.isfinite(pos).all() or np.allclose(pos, 0):
        montage = info.get_montage() if hasattr(info, "get_montage") else None
        ch_pos = montage.get_positions().get("ch_pos", {}) if montage else {}
        pos = np.array([ch_pos.get(n, [np.nan] * 3) for n in names], dtype=float)
        if not np.isfinite(pos).all():
            warnings.warn("no usable electrode positions in info; "
                          "distance-based graphs unavailable")
            pos = None
    return names, pos


def _encode_labels(y_train, y_test):
    """Map arbitrary labels onto 0..K-1; returns (y_train, y_test, classes)."""
    to_np = lambda y: np.asarray(
        y.detach().cpu() if isinstance(y, torch.Tensor) else y).ravel()
    y_train, y_test = to_np(y_train), to_np(y_test)
    classes = np.unique(np.concatenate([y_train, y_test]))
    lut = {c: i for i, c in enumerate(classes)}
    enc = lambda y: np.array([lut[v] for v in y], dtype=np.int64)
    return enc(y_train), enc(y_test), classes


def _bands_for(sfreq: float) -> Dict[str, Tuple[float, float]]:
    """
    Bands clipped to what the sampling rate can represent.

    The paper band-passes to 100 Hz then resamples to 128 Hz, which puts Nyquist at
    64 Hz — its own gamma band is unobservable in the data the model sees. At 300 Hz
    there is no such problem; the clipping is here so lower rates degrade instead of
    throwing a filter-design error.
    """
    nyq = sfreq / 2.0
    return {name: (lo, min(hi, 0.9 * nyq)) for name, (lo, hi) in BANDS.items()
            if min(hi, 0.9 * nyq) > lo}


# =============================================================================
# PLV and the GAT adjacency (sections 2.3.1 / 2.5.2)
# =============================================================================

def _bandpass(x: np.ndarray, lo: float, hi: float, sfreq: float, order: int = 4):
    nyq = sfreq / 2.0
    b, a = butter(order, [max(lo, 1e-3) / nyq, min(hi, 0.99 * nyq) / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1,
                    padlen=min(3 * max(len(a), len(b)), x.shape[-1] - 1))


def compute_plv(epochs: np.ndarray, sfreq: float, band: Tuple[float, float],
                max_trials: int = 500) -> np.ndarray:
    """
    Equation 8: PLV_ij = | (1/T) sum_t exp(i (phi_i - phi_j)) |, averaged over
    trials, with phases from the analytic (Hilbert) signal. (N, C, T) -> (C, C).
    """
    if len(epochs) > max_trials:
        idx = np.random.default_rng(0).choice(len(epochs), max_trials, replace=False)
        epochs = epochs[idx]
    filtered = _bandpass(epochs.astype(np.float64), band[0], band[1], sfreq)
    z = np.exp(1j * np.angle(hilbert(filtered, axis=-1)))
    return np.abs(np.einsum("nct,ndt->ncd", z, np.conj(z)) / z.shape[-1]).mean(0).real


def _topk_graph(weights: np.ndarray, degree: int) -> np.ndarray:
    """Keep the `degree` strongest edges per node, symmetrise, add self-loops."""
    C = weights.shape[0]
    w = weights.astype(float).copy()
    np.fill_diagonal(w, -np.inf)
    idx = np.argsort(-w, axis=1)[:, : int(np.clip(degree, 1, C - 1))]
    adj = np.zeros((C, C), dtype=bool)
    np.put_along_axis(adj, idx, True, axis=1)
    adj |= adj.T
    np.fill_diagonal(adj, True)
    return adj


def build_adjacency(train_epochs: np.ndarray, sfreq: float, kind: str = "plv",
                    degree: int = 6, positions: Optional[np.ndarray] = None
                    ) -> np.ndarray:
    """
    The GAT graph (section 2.3.1).

    "plv"      — PLV averaged across bands, the paper's biologically informed prior.
    "distance" — the paper's alternative, connecting nearby electrodes. Built as a
                 k-nearest-neighbour graph over the MNE positions rather than a fixed
                 distance d, so it does not depend on the montage's units or scale.
    "dense"    — every channel connected to every other.
    """
    C = train_epochs.shape[1]
    if kind == "dense":
        return np.ones((C, C), dtype=bool)
    if kind == "distance":
        if positions is None:
            warnings.warn("no electrode positions available; using a PLV graph")
        else:
            d = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
            return _topk_graph(-d, degree)
    plv = np.mean([compute_plv(train_epochs, sfreq, band)
                   for band in _bands_for(sfreq).values()], axis=0)
    return _topk_graph(plv, degree)


# =============================================================================
# Blocks
# =============================================================================

class GATLayer(nn.Module):
    """
    Velickovic et al. (2018) graph attention, dense masked form:

        e_ij = LeakyReLU(a^T [W h_i || W h_j]),  alpha_i = softmax_j(e_ij),
        h'_i = sum_j alpha_ij W h_j

    Heads are concatenated when `concat`, else averaged. Attention is cached for
    the interpretability helpers.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int, concat: bool = True,
                 dropout: float = 0.4):
        super().__init__()
        self.concat, self.dropout = concat, dropout
        self.W = nn.Parameter(torch.empty(heads, in_dim, out_dim))
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.last_attention: Optional[torch.Tensor] = None
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0), gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0), gain=1.414)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """h: (B, C, in_dim), adj: (C, C) bool -> (B, C, heads*out | out)"""
        Wh = torch.einsum("bci,hio->bhco", h, self.W)
        src = (Wh * self.a_src[None, :, None]).sum(-1)
        dst = (Wh * self.a_dst[None, :, None]).sum(-1)
        e = F.leaky_relu(src.unsqueeze(-1) + dst.unsqueeze(-2), 0.2)
        e = e.masked_fill(~adj.bool()[None, None], torch.finfo(e.dtype).min)

        alpha = torch.softmax(e, dim=-1)
        self.last_attention = alpha.detach()
        out = torch.einsum("bhij,bhjo->bhio",
                           F.dropout(alpha, self.dropout, self.training), Wh)
        return out.permute(0, 2, 1, 3).flatten(2) if self.concat else out.mean(1)


class SpatialBranch(nn.Module):
    """
    Section 2.3.1, hierarchical GAT: a low-level layer (heads concatenated) for local
    inter-channel interactions, a high-level layer (heads averaged) for global
    cross-region patterns.

    The paper does not say how a channel's time course becomes a node feature, so a
    weight-shared 1-D CNN does it. Its equations 1 and 2 also give the low-level
    output as (C x D) but the high-level one as (C x T x D), which cannot both hold
    for stacked GAT layers; both are (C x D) here and the time axis lives in the
    temporal branch.
    """

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.BatchNorm1d(16), nn.ELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
            nn.Linear(128, 64), nn.ELU(), nn.Dropout(dropout),
        )
        low_dim = max(8, d_model // heads)
        self.low = GATLayer(64, low_dim, heads, concat=True, dropout=dropout)
        self.high = GATLayer(low_dim * heads, d_model, max(1, heads // 2),
                             concat=False, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C, d_model): one token per electrode."""
        B, C, T = x.shape
        h = self.encoder(x.reshape(B * C, 1, T)).reshape(B, C, -1)
        h = F.elu(self.low(h, adj))
        return self.norm(F.elu(self.high(h, adj)))


class TemporalBranch(nn.Module):
    """
    Section 2.3.2: a unidirectional GRU for short-range dynamics, then a Transformer
    encoder for long-range ones. The "learnable temporal embedding vectors [that]
    dynamically weight the importance of different time points" are an additive
    positional embedding plus a learned query giving one softmax weight per step.
    `use_transformer=False` is the Table 5 GRU-only ablation.
    """

    def __init__(self, n_channels: int, n_times: int, d_model: int, heads: int,
                 dropout: float, use_transformer: bool = True, n_tokens: int = 16):
        super().__init__()
        stride = max(1, math.ceil(n_times / 256))     # keep the sequence manageable
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, d_model, 7, stride=stride, padding=3),
            nn.BatchNorm1d(d_model), nn.ELU())
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

        self.time_embed = nn.Parameter(torch.zeros(1, n_times // stride + 1, d_model))
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        self.query = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.last_weights: Optional[torch.Tensor] = None

        self.encoder = None
        if use_transformer:
            layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout,
                                               activation="gelu", batch_first=True,
                                               norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, n_tokens + 1, d_model)"""
        h, _ = self.gru(self.stem(x).transpose(1, 2))
        if self.encoder is not None:
            h = self.encoder(h + self.time_embed[:, : h.size(1)])
        h = self.drop(self.norm(h))

        w = torch.softmax((h @ self.query) / math.sqrt(h.size(-1)), dim=-1)
        self.last_weights = w.detach()
        pooled = torch.einsum("bl,bld->bd", w, h)
        tokens = self.pool((h * w.unsqueeze(-1) * h.size(1)).transpose(1, 2)).transpose(1, 2)
        return torch.cat([tokens, pooled.unsqueeze(1)], dim=1)


class FrequencyBranch(nn.Module):
    """
    Section 2.3.3: the signal is split into physiological bands and each gets its own
    1-D CNN (kernels 5/3/3, stride 1, ReLU, max pool); outputs are concatenated
    (equation 5). The filter bank is a frozen FIR layer inside the graph, so gradient
    and SHAP attributions run end to end from the raw epoch. Filter length is capped
    at half the epoch, which bounds how sharply delta can be separated.
    """

    def __init__(self, n_channels: int, n_times: int, sfreq: float, d_model: int,
                 dropout: float, n_tokens: int = 4):
        super().__init__()
        bands = _bands_for(sfreq)
        self.band_names = list(bands)
        numtaps = int(min(129, max(15, n_times // 2))) | 1
        nyq = sfreq / 2.0
        taps = [firwin(numtaps, [lo / nyq, min(hi, 0.99 * nyq) / nyq],
                       pass_zero=False, window="hamming") for lo, hi in bands.values()]
        self.register_buffer("taps", torch.tensor(np.stack(taps), dtype=torch.float32))
        self.pad = numtaps // 2

        def cnn():
            layers, in_ch = [], n_channels
            for out_ch, k in zip((d_model, d_model, d_model), (5, 3, 3)):
                layers += [nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
                           nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
                           nn.MaxPool1d(4), nn.Dropout(dropout)]
                in_ch = out_ch
            return nn.Sequential(*layers, nn.AdaptiveMaxPool1d(n_tokens))

        self.cnns = nn.ModuleList([cnn() for _ in self.band_names])
        self.band_embed = nn.Parameter(torch.zeros(len(self.band_names), d_model))
        nn.init.trunc_normal_(self.band_embed, std=0.02)

    def filter_bank(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, n_bands, C, T)"""
        B, C, T = x.shape
        y = F.conv1d(x.reshape(B * C, 1, T), self.taps.unsqueeze(1), padding=self.pad)
        return y.reshape(B, C, len(self.band_names), -1).permute(0, 2, 1, 3)[..., :T]

    def forward(self, x: torch.Tensor,
                x_bands: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_bands is None:
            x_bands = self.filter_bank(x)
        return torch.cat([cnn(x_bands[:, i]).transpose(1, 2) + self.band_embed[i]
                          for i, cnn in enumerate(self.cnns)], dim=1)


# =============================================================================
# Model
# =============================================================================

class MultiBranchGATGRUTransformer(nn.Module):
    """
    Spatial (GAT) + temporal (GRU -> Transformer) + frequency (multi-band 1-D CNN)
    branches. Each emits `d_model`-wide tokens; these are concatenated, passed
    through a 1-D CNN and reduced by global average pooling (equations 6-7).
    """

    def __init__(self, n_channels: int, n_times: int, n_classes: int, sfreq: float,
                 adjacency: Optional[np.ndarray] = None, d_model: int = 32,
                 gat_heads: int = 4, dropout: float = 0.4, use_gat: bool = True,
                 use_transformer: bool = True, use_freq: bool = True):
        super().__init__()
        if not (use_gat or use_transformer or use_freq):
            raise ValueError("at least one branch must be enabled")
        self.n_channels, self.n_times, self.sfreq = n_channels, n_times, sfreq
        self.n_classes = n_classes

        adj = (np.ones((n_channels,) * 2, bool) if adjacency is None
               else np.asarray(adjacency, bool))
        if adj.shape != (n_channels, n_channels):
            raise ValueError(f"adjacency {adj.shape} does not match {n_channels} channels")
        self.register_buffer("adj", torch.tensor(adj))

        self.spatial = SpatialBranch(d_model, gat_heads, dropout) if use_gat else None
        self.temporal = TemporalBranch(n_channels, n_times, d_model, gat_heads,
                                       dropout, use_transformer)
        self.frequency = (FrequencyBranch(n_channels, n_times, sfreq, d_model, dropout)
                          if use_freq else None)

        self.head = nn.Sequential(nn.Conv1d(d_model, 2 * d_model, 3, padding=1),
                                  nn.BatchNorm1d(2 * d_model), nn.ELU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(nn.Dropout(0.6),
                                        nn.Linear(2 * d_model, n_classes))

    def forward(self, x: torch.Tensor,
                x_bands: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, C, T) -> logits (B, n_classes)"""
        tokens = []
        if self.spatial is not None:
            tokens.append(self.spatial(x, self.adj))
        tokens.append(self.temporal(x))
        if self.frequency is not None:
            tokens.append(self.frequency(x, x_bands))
        return self.classifier(self.head(torch.cat(tokens, 1).transpose(1, 2)).mean(-1))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Training
# =============================================================================

class _Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, augment=False, **augment_kwargs):
        self.x = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
        self.y = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.long)
        self.augment, self.augment_kwargs = augment, augment_kwargs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.augment:
            x = augment_eeg(x, **self.augment_kwargs)
        return x, self.y[i]


@torch.no_grad()
def _evaluate(model, loader, device, criterion=None) -> Dict[str, Any]:
    model.eval()
    logits_all, y_all, loss_sum, n = [], [], 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if criterion is not None:
            loss_sum += criterion(logits, yb).item() * len(yb)
        n += len(yb)
        logits_all.append(logits.float().cpu())
        y_all.append(yb.cpu())

    logits, y = torch.cat(logits_all), torch.cat(y_all).numpy()
    proba = torch.softmax(logits, 1).numpy()
    pred = proba.argmax(1)
    return {"loss": loss_sum / max(n, 1), "acc": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "y_true": y, "y_pred": pred, "proba": proba}

def train_gat_gru_transformer(
    X_train, y_train, X_test, y_test, *,
    sfreq: float = 300.0,
    run_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_tags: Optional[list] = None,
    log_wandb: bool = True,
    # architecture
    d_model: int = 32,
    gat_heads: int = 4,
    dropout: float = 0.4,
    adjacency: Union[str, np.ndarray] = "plv",
    graph_degree: int = 6,
    use_gat: bool = True,
    use_transformer: bool = True,
    use_freq: bool = True,
    # optimisation
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    val_size: float = 0.1,
    seed: int = 42,
    device: Optional[str] = None,
    verbose: int = 1,
    augment: bool = False,
    augment_sigma: float = DEFAULT_SIGMA,
    augment_max_shift: int = DEFAULT_MAX_SHIFT,
    augment_scale_range: Tuple[float, float] = DEFAULT_SCALE_RANGE,
    decim: int = 1,
) -> Dict[str, Any]:
    """
    Train the Multi-Branch GAT-GRU-Transformer and evaluate it on the test set.

    X_train / X_test may be mne.Epochs or (n_trials, n_channels, n_times) arrays;
    with Epochs the sampling rate, channel names and electrode positions come from
    `info` and the `sfreq` argument is ignored. Labels may be of any type and come
    back in their original values.

    Knobs: `d_model` (width of every branch), `gat_heads`, `dropout`, `adjacency`
    ("plv", "distance", "dense", or a boolean matrix), `graph_degree` (edges kept
    per node), and `use_gat` / `use_transformer` / `use_freq` for the Table 5
    ablations.

    augment : if True, every training sample gets a freshly-drawn Gaussian noise + time
             shift + amplitude scale each epoch (see models/augment.py); validation/test
             data is never augmented. augment_sigma/augment_max_shift/augment_scale_range
             control it; defaults assume roughly unit-scale (e.g. z-scored) input.

    decim  : keep every decim-th sample (e.g. 2: 300 Hz -> 150 Hz, close to the paper's
             128 Hz), roughly 1.6x faster. Only allowed if the data's low-pass is <= new
             sfreq / 3 (see models.EEGNet.check_decim). Bands, filter bank and PLV graph
             follow the new rate. The model remembers its decim, so `predict` and the
             interpretability helpers below still take full-rate epochs.

    Returns a dict with the trained model, per-epoch history, test accuracy,
    predictions, probabilities and the confusion matrix (keys: model, history,
    test_accuracy, y_pred, y_proba, confusion_matrix; plus y_true, classes,
    ch_names, positions, adjacency, test_loss, test_macro_f1, best_epoch,
    n_parameters, training_time, run).
    """
    t0 = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data ---------------------------------------------------------------
    x_tr, info = _unpack(X_train, "X_train", decim)
    x_te, _ = _unpack(X_test, "X_test", decim)
    if x_tr.shape[1:] != x_te.shape[1:]:
        raise ValueError(f"train/test shapes differ: {x_tr.shape[1:]} vs {x_te.shape[1:]}")
    if info is not None:
        sfreq = float(info["sfreq"])
    sfreq = check_decim(sfreq, decim)   # effective rate the model sees

    y_tr, y_te, classes = _encode_labels(y_train, y_test)
    n_channels, n_times = x_tr.shape[1], x_tr.shape[2]
    ch_names, positions = _montage(info, n_channels)

    if val_size > 0:
        strat = y_tr if np.bincount(y_tr).min() >= 2 else None
        x_fit, x_va, y_fit, y_va = train_test_split(x_tr, y_tr, test_size=val_size,
                                                    stratify=strat, random_state=seed)
    else:
        x_fit, y_fit, x_va, y_va = x_tr, y_tr, None, None

    # Built on the training split only, so the graph leaks nothing.
    adj = (build_adjacency(x_fit, sfreq, adjacency, graph_degree, positions)
           if isinstance(adjacency, str) else np.asarray(adjacency, bool))

    # ---- model --------------------------------------------------------------
    model = MultiBranchGATGRUTransformer(
        n_channels, n_times, len(classes), sfreq, adj, d_model=d_model,
        gat_heads=gat_heads, dropout=dropout, use_gat=use_gat,
        use_transformer=use_transformer, use_freq=use_freq).to(device)
    model.decim = decim   # input this model expects = full-rate epochs decimated by this factor
    if verbose:
        print(f"[model] {model.n_parameters():,} parameters on {device} | "
              f"{n_channels} ch x {n_times} samples @ {sfreq:g} Hz | "
              f"{len(classes)} classes | graph density {adj.mean():.2f}")

    # ---- wandb --------------------------------------------------------------
    run = wandb_start_run("gat_gru_transformer", run_name=run_name, group=wandb_group,
                tags=wandb_tags, enabled=log_wandb,
                config=dict(sfreq=sfreq, decim=decim, n_channels=n_channels, n_times=n_times,
                            n_classes=len(classes), d_model=d_model,
                            gat_heads=gat_heads, dropout=dropout,
                            adjacency=str(adjacency)[:32],
                            graph_degree=graph_degree, use_gat=use_gat,
                            use_transformer=use_transformer, use_freq=use_freq,
                            epochs=epochs, batch_size=batch_size, lr=lr,
                            weight_decay=weight_decay, seed=seed,
                            n_train=len(x_fit), n_test=len(x_te),
                            n_parameters=model.n_parameters(),
                            augment=augment, augment_sigma=augment_sigma if augment else None,
                            augment_max_shift=augment_max_shift if augment else None,
                            augment_scale_range=augment_scale_range if augment else None))

    try:
        mk = lambda x, y, sh, aug=False: torch.utils.data.DataLoader(
            _Dataset(x, y, augment=aug, sigma=augment_sigma, max_shift=augment_max_shift,
                    scale_range=augment_scale_range),
            batch_size=batch_size, shuffle=sh, drop_last=sh and len(y) > batch_size)
        tr_loader = mk(x_fit, y_fit, True, aug=augment)
        va_loader = mk(x_va, y_va, False) if y_va is not None else None
        te_loader = mk(x_te, y_te, False)

        counts = np.bincount(y_fit, minlength=len(classes)).astype(np.float32)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            counts.sum() / (len(classes) * np.maximum(counts, 1)), device=device),
            label_smoothing=0.1)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [],
                    "lr": [], "epoch_time": [], "val_macro_f1": []}
        best_score, best_state, best_epoch = -np.inf, None, 0
        for epoch in range(1, epochs + 1):
            model.train()
            t_epoch, run_loss, correct, n = time.time(), 0.0, 0, 0
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                run_loss += loss.item() * len(yb)
                correct += int((logits.argmax(1) == yb).sum())
                n += len(yb)
            sched.step()

            history["loss"].append(run_loss / max(n, 1))
            history["acc"].append(correct / max(n, 1))
            history["lr"].append(opt.param_groups[0]["lr"])
            history["epoch_time"].append(time.time() - t_epoch)

            if va_loader is not None:
                va = _evaluate(model, va_loader, device, criterion)
                history["val_loss"].append(va["loss"])
                history["val_acc"].append(va["acc"])
                history["val_macro_f1"].append(va["macro_f1"])

            score = history["val_acc"][-1] if va_loader is not None else history["acc"][-1]
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                msg = (f"epoch {epoch:4d}/{epochs} | "
                       f"train {history['loss'][-1]:.4f}/{history['acc'][-1]:.4f}")
                if va_loader is not None:
                    msg += f" | val {history['val_loss'][-1]:.4f}/{history['val_acc'][-1]:.4f}"
                print(msg + f" | {history['epoch_time'][-1]:.1f}s")

        if best_state is not None:
            model.load_state_dict(best_state)

        # ---- test ------------------------------------------------------------
        test = _evaluate(model, te_loader, device, criterion)
        labels, names = np.arange(len(classes)), [str(c) for c in classes]
        cm = confusion_matrix(test["y_true"], test["y_pred"], labels=labels)
        if verbose:
            print(f"\ntest accuracy {test['acc']:.4f} | macro-F1 {test['macro_f1']:.4f}"
                  f" | best epoch {best_epoch}")
            print(classification_report(test["y_true"], test["y_pred"], labels=labels,
                                        target_names=names, digits=4, zero_division=0))

        wandb_log_test_results(run, test_accuracy=test["acc"], y_true=classes[test["y_true"]],
                 y_pred=classes[test["y_pred"]], classes=names,
                 test_loss=test["loss"],
                 extra={"test_macro_f1": test["macro_f1"], "best_epoch": best_epoch,
                        "n_parameters": model.n_parameters()})

        return {"model": model, "history": history, "test_accuracy": test["acc"],
                "y_pred": classes[test["y_pred"]], "y_proba": test["proba"],
                "confusion_matrix": cm, "y_true": classes[test["y_true"]],
                "classes": classes, "ch_names": ch_names, "positions": positions,
                "adjacency": adj, "test_loss": test["loss"],
                "test_macro_f1": test["macro_f1"], "best_epoch": best_epoch,
                "n_parameters": model.n_parameters(),
                "training_time": time.time() - t0, "run": run,
                "sfreq": sfreq, "decim": decim}
    finally:
        wandb_finish_run(run)


@torch.no_grad()
def predict(model: MultiBranchGATGRUTransformer, X, classes=None,
            batch_size: int = 64):
    """Run a trained model on new epochs. Returns (labels, probabilities)."""
    x, _ = _unpack(X, decim=_model_decim(model))

    device = next(model.parameters()).device
    model.eval()
    proba = np.concatenate([
        torch.softmax(model(torch.as_tensor(x[i:i + batch_size], device=device)), 1)
        .float().cpu().numpy() for i in range(0, len(x), batch_size)])
    idx = proba.argmax(1)
    return (classes[idx] if classes is not None else idx), proba


# =============================================================================
# Interpretability (paper section 2.5)
# =============================================================================

def channel_importance(model, X, y, ch_names: Sequence[str], top_k: int = 10,
                       batch_size: int = 32) -> List[Tuple[str, float]]:
    """
    Gradient x input attribution averaged per channel — the Figure 6A ranking. For
    true SHAP values, `shap.GradientExplainer(model, background)` works directly on
    this model; DeepExplainer does not, having no rule for the GRU and Transformer
    ops the paper's own model also contains.
    """
    x, _ = _unpack(X, decim=_model_decim(model))
    device = next(model.parameters()).device
    model.eval()
    scores = np.zeros(x.shape[1])
    # eval-mode forward + backward through a cuDNN GRU raises "cudnn RNN backward
    # can only be called in training mode"; the non-cuDNN path has no such
    # restriction and costs nothing at this (attribution-only) scale.
    with torch.backends.cudnn.flags(enabled=False):
        for i in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[i:i + batch_size], device=device).requires_grad_(True)
            yb = torch.as_tensor(np.asarray(y[i:i + batch_size]), dtype=torch.long,
                                 device=device)
            model(xb).gather(1, yb[:, None]).sum().backward()
            scores += (xb.grad * xb).abs().mean(-1).sum(0).detach().cpu().numpy()
    scores /= max(len(x), 1)
    return [(ch_names[j], float(scores[j])) for j in np.argsort(-scores)[:top_k]]


def band_importance(model, X, y, batch_size: int = 32) -> Dict[str, float]:
    """
    Per-band importance (Figure 6B): gradient x input on the filter-bank output
    rather than on the raw signal. Needs the frequency branch.
    """
    if model.frequency is None:
        raise ValueError("model was built with use_freq=False")
    x, _ = _unpack(X, decim=_model_decim(model))
    device = next(model.parameters()).device
    model.eval()
    totals = np.zeros(len(model.frequency.band_names))
    with torch.backends.cudnn.flags(enabled=False):
        for i in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[i:i + batch_size], device=device)
            yb = torch.as_tensor(np.asarray(y[i:i + batch_size]), dtype=torch.long,
                                 device=device)
            bands = model.frequency.filter_bank(xb).detach().requires_grad_(True)
            model(xb, x_bands=bands).gather(1, yb[:, None]).sum().backward()
            totals += ((bands.grad * bands).abs().mean((0, 2, 3)).detach().cpu().numpy()
                       * len(yb))
    return dict(zip(model.frequency.band_names, (totals / max(len(x), 1)).tolist()))


def gat_attention(model, X, ch_names: Sequence[str], top_k: int = 10,
                  batch_size: int = 64):
    """Mean incoming GAT attention per electrode — the learned spatial graph."""
    if model.spatial is None:
        raise ValueError("model was built with use_gat=False")
    x, _ = _unpack(X, decim=_model_decim(model))
    device = next(model.parameters()).device
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            model(torch.as_tensor(x[i:i + batch_size], device=device))
            maps = [model.spatial.low.last_attention, model.spatial.high.last_attention]
            total = total + torch.stack([m.mean(1).mean(0) for m in maps]).mean(0)
            n += 1
    att = (total / max(n, 1)).cpu().numpy()
    incoming = att.mean(0)
    return [(ch_names[j], float(incoming[j])) for j in np.argsort(-incoming)[:top_k]], att


def plv_report(X, sfreq: float = 300.0, ch_names: Optional[Sequence[str]] = None,
               bands: Sequence[str] = ("alpha", "beta", "gamma"), top_k: int = 10):
    """Section 4.2 / Figure 7: strongest phase-synchronised channel pairs per band."""
    x, info = _unpack(X)
    if info is not None:
        sfreq, ch_names = float(info["sfreq"]), list(info["ch_names"])
    ch_names = list(ch_names or [f"ch{i}" for i in range(x.shape[1])])

    out = {}
    for name, band in _bands_for(sfreq).items():
        if name not in bands:
            continue
        plv = compute_plv(x, sfreq, band)
        iu = np.triu_indices_from(plv, k=1)
        vals = plv[iu]
        out[name] = [(ch_names[iu[0][j]], ch_names[iu[1][j]], float(vals[j]))
                     for j in np.argsort(-vals)[:top_k]]
    return out