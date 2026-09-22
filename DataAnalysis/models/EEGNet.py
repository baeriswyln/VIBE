import copy
import torch

from typing import Optional

import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.wandb import *

# --------------------------------------------------------------------------- #
# Layers with Keras-style max-norm weight constraints
# --------------------------------------------------------------------------- #
class ConstrainedConv2d(nn.Conv2d):
    """Conv2d whose per-filter L2 weight norm is clipped, like Keras max_norm."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm))
        return super().forward(x)


class ConstrainedLinear(nn.Linear):
    """Linear layer with max-norm constraint on each output unit's weights."""

    def __init__(self, *args, max_norm: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm))
        return super().forward(x)


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
class EEGNet(nn.Module):
    """
    Parameters
    ----------
    n_classes   : number of output classes
    n_chans     : number of EEG channels C
    n_samples   : number of time samples T per epoch
    kern_length : length of the temporal filters. The paper uses half the
                  sampling rate (64 samples @ 128 Hz) so the filters cover 500 ms.
    F1, D, F2   : temporal filters, spatial filters per temporal filter,
                  pointwise filters. The paper's default is F1=8, D=2, F2=F1*D=16.
    dropout     : 0.25 for within-subject (few trials), 0.5 for cross-subject.
    dropout_type: 'Dropout' or 'SpatialDropout2D' (nn.Dropout2d).
    """

    def __init__(self, n_classes: int, n_chans: int, n_samples: int, kern_length: int = 64, F1: int = 8, D: int = 2, F2: Optional[int] = None, dropout: float = 0.25, dropout_type: str = "Dropout"):
        super().__init__()
        F2 = F1 * D if F2 is None else F2

        if dropout_type == "SpatialDropout2D":
            drop = lambda: nn.Dropout2d(dropout)  # noqa: E731
        elif dropout_type == "Dropout":
            drop = lambda: nn.Dropout(dropout)  # noqa: E731
        else:
            raise ValueError("dropout_type must be 'Dropout' or 'SpatialDropout2D'")

        # ---- Block 1: temporal convolution + depthwise spatial filtering ----
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            ConstrainedConv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False, max_norm=1.0),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            drop(),
        )

        # ---- Block 2: separable convolution (depthwise then pointwise) ----
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            drop(),
        )

        # ---- Classifier ----
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_samples)
            n_features = self.block2(self.block1(dummy)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            ConstrainedLinear(n_features, n_classes, max_norm=0.25),
        )

    def forward(self, x):
        # x: (batch, 1, n_chans, n_samples) -> logits (batch, n_classes)
        return self.classifier(self.block2(self.block1(x)))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _to_array(X) -> np.ndarray:
    """Accept an MNE Epochs object or an array of shape (n_trials, C, T)."""
    if hasattr(X, "get_data"):          # mne.Epochs / EpochsArray
        X = X.get_data()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected epochs of shape (n_trials, n_chans, n_samples), got {X.shape}")
    return X


def _to_labels(y) -> np.ndarray:
    if hasattr(y, "events"):            # someone passed Epochs as the label source
        y = y.events[:, 2]
    return np.asarray(y).ravel()


def _stratified_split(y: np.ndarray, frac: float, rng: np.random.Generator):
    """Return (train_idx, val_idx) keeping class proportions."""
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = max(1, int(round(frac * len(idx)))) if len(idx) > 1 else 0
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, n, logits_all = 0.0, 0, []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * len(xb)
        n += len(xb)
        logits_all.append(logits.cpu())
    logits_all = torch.cat(logits_all)
    y_true = torch.cat([yb for _, yb in loader])
    acc = (logits_all.argmax(1) == y_true).float().mean().item()
    return loss_sum / n, acc, logits_all


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def train_eegnet(X_train, y_train, X_test, y_test, *, sfreq: float = 300.0, kern_length: Optional[int] = None,
    F1: int = 8, D: int = 2, F2: Optional[int] = None,
    dropout: float = 0.25, dropout_type: str = "Dropout",
    n_epochs: int = 100, batch_size: int = 256, lr: float = 3e-3,
    validation_split: float = 0.2, class_weights: bool = True,
    scale: str | float = "auto", device: Optional[str] = None, seed: int = 42, verbose: bool = True,
    run_name: Optional[str] = None, wandb_group: Optional[str] = None,
    wandb_tags: Optional[list[str]] = None, log_wandb: bool = True,
    save_as: Optional[str] = None,
):
    """
    Train and test EEGNet on epoched EEG data.

    X_train / X_test : (n_trials, n_chans, n_samples) arrays or MNE Epochs.
    y_train / y_test : (n_trials,) labels (ints or strings).

    sfreq            : sampling rate, used to set kern_length = sfreq // 2 as in the paper.
    validation_split : fraction of the training epochs held out to pick the best
                       weights (the paper selects the checkpoint with lowest val loss).
                       Set to 0 to train on everything and keep the last epoch.
    scale            : 'auto' converts volts to microvolts when the data looks like
                       volts; pass a float to multiply by it, or 1.0 to leave as is.
    run_name / wandb_group / wandb_tags : forwarded to wandb_start_run (see above);
                       every call logs to Weights & Biases unless log_wandb=False.

    Returns a dict with the trained model, per-epoch history, test accuracy,
    predictions, probabilities and the confusion matrix.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True

    # ---- data preparation -------------------------------------------------
    Xtr, Xte = _to_array(X_train), _to_array(X_test)
    ytr, yte = _to_labels(y_train), _to_labels(y_test)
    if Xtr.shape[1:] != Xte.shape[1:]:
        raise ValueError(f"Train/test epochs differ in shape: {Xtr.shape[1:]} vs {Xte.shape[1:]}")
    if len(Xtr) != len(ytr) or len(Xte) != len(yte):
        raise ValueError("Number of epochs and number of labels do not match")

    if scale == "auto":
        factor = 1e6 if np.abs(Xtr).max() < 1e-2 else 1.0  # volts -> microvolts
    else:
        factor = float(scale)
    Xtr, Xte = Xtr * factor, Xte * factor

    classes = np.unique(ytr)
    lut = {c: i for i, c in enumerate(classes)}
    if not set(np.unique(yte)).issubset(lut):
        raise ValueError("Test set contains labels not present in the training set")
    ytr_i = np.array([lut[c] for c in ytr], dtype=np.int64)
    yte_i = np.array([lut[c] for c in yte], dtype=np.int64)

    n_trials, n_chans, n_samples = Xtr.shape
    kern_length = int(sfreq // 2) if kern_length is None else int(kern_length)

    # add the singleton "image channel": (N, 1, C, T)
    Xtr_t = torch.from_numpy(Xtr).float().unsqueeze(1)
    Xte_t = torch.from_numpy(Xte).float().unsqueeze(1)
    ytr_t = torch.from_numpy(ytr_i)
    yte_t = torch.from_numpy(yte_i)

    if validation_split and validation_split > 0:
        tr_idx, va_idx = _stratified_split(ytr_i, validation_split, rng)
    else:
        tr_idx, va_idx = np.arange(n_trials), np.array([], dtype=int)

    train_loader = DataLoader(
        TensorDataset(Xtr_t[tr_idx], ytr_t[tr_idx]), batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = (
        DataLoader(TensorDataset(Xtr_t[va_idx], ytr_t[va_idx]), batch_size=256, pin_memory=True)
        if len(va_idx) else None
    )
    test_loader = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=256, pin_memory=True)

    # ---- model ------------------------------------------------------------
    model = EEGNet( n_classes=len(classes), n_chans=n_chans, n_samples=n_samples, kern_length=kern_length, F1=F1, D=D, F2=F2, dropout=dropout, dropout_type=dropout_type,).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if class_weights:
        counts = np.bincount(ytr_i[tr_idx], minlength=len(classes)).astype(np.float64)
        w = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=w.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose:
        print(f"EEGNet-{F1},{D} | {n_chans} ch x {n_samples} samples | "
              f"{len(classes)} classes | {n_par} trainable params | device={device}")
        if factor != 1.0:
            print(f"Scaled inputs by {factor:g} (volts -> microvolts)")

    wb_run = wandb_start_run(
        "EEGNet", run_name,
        config=dict(
            F1=F1, D=D, F2=F2 or F1 * D, kern_length=kern_length,
            dropout=dropout, dropout_type=dropout_type,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            validation_split=validation_split, class_weights=class_weights,
            scale=factor, sfreq=sfreq, seed=seed,
            n_chans=n_chans, n_samples=n_samples, n_classes=len(classes),
            n_train=len(tr_idx), n_val=len(va_idx), n_test=len(Xte),
            n_params=n_par, device=str(device),
        ),
        group=wandb_group, tags=wandb_tags, enabled=log_wandb,
    )

    try:
        # ---- training loop --------------------------------------------------
        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        best_val, best_state = np.inf, copy.deepcopy(model.state_dict())

        for epoch in range(1, n_epochs + 1):
            model.train()
            run_loss, correct, seen = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                run_loss += loss.item() * len(xb)
                correct += (logits.argmax(1) == yb).sum().item()
                seen += len(xb)

            history["loss"].append(run_loss / seen)
            history["acc"].append(correct / seen)
            epoch_metrics = {"train/loss": history["loss"][-1], "train/acc": history["acc"][-1]}

            if val_loader is not None:
                v_loss, v_acc, _ = _evaluate(model, val_loader, criterion, device)
                history["val_loss"].append(v_loss)
                history["val_acc"].append(v_acc)
                epoch_metrics["val/loss"] = v_loss
                epoch_metrics["val/acc"] = v_acc
                if v_loss < best_val:
                    best_val = v_loss
                    best_state = copy.deepcopy(model.state_dict())
            if wb_run is not None:
                wb_run.log(epoch_metrics, step=epoch)
            if verbose and (epoch % 10 == 0 or epoch == 1):
                msg = f"epoch {epoch:4d}/{n_epochs}  loss {history['loss'][-1]:.4f}  acc {history['acc'][-1]:.3f}"
                if val_loader is not None:
                    msg += f"  val_loss {history['val_loss'][-1]:.4f}  val_acc {history['val_acc'][-1]:.3f}"
                print(msg)

        if val_loader is not None:
            model.load_state_dict(best_state)  # restore best-validation weights

        # ---- test -----------------------------------------------------------
        test_loss, test_acc, logits = _evaluate(model, test_loader, criterion, device)
        proba = torch.softmax(logits, dim=1).numpy()
        y_pred_i = proba.argmax(1)

        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(yte_i, y_pred_i):
            cm[t, p] += 1

        if verbose:
            print(f"\nTest loss {test_loss:.4f} | test accuracy {test_acc:.4f}")
            print("Confusion matrix (rows = true, cols = predicted):")
            print(cm)

        if save_as is not None:
            n_channels = len(X_train.ch_names)
            n_times = len(X_train.times)
            ep = torch.export.export(model, (torch.randn(1,n_channels, n_times, dtype=torch.float32).unsqueeze(1),))
            torch.export.save(ep, save_as)

        wandb_log_test_results(
            wb_run, test_accuracy=test_acc, test_loss=test_loss,
            y_true=yte_i, y_pred=y_pred_i, classes=classes,
        )
    finally:
        wandb_finish_run(wb_run)

    return {
        "model": model,
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "y_pred": classes[y_pred_i],
        "y_proba": proba,
        "classes": classes,
        "confusion_matrix": cm,
    }