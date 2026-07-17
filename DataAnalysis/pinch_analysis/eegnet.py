"""
eegnet.py — Compact EEGNet (Lawhern et al. 2018) in PyTorch.

A small, heavily-regularised CNN suited to few‑trial EEG. Input is
(batch, 1, n_channels, n_times). Two blocks:

  1. temporal conv (learns frequency filters) -> depthwise spatial conv over
     channels (learns per‑filter spatial patterns, e.g. a C3‑focused motor map)
  2. separable conv (temporal summary) -> pool -> classifier

Kept deliberately tiny (F1=8, D=2, F2=16, dropout 0.5) because we train per
subject on only a few hundred epochs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16, kern_length: int = 64,
                 dropout: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
        )
        # depthwise spatial conv across all channels
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        # separable conv = depthwise temporal + pointwise
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same",
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        feat = F2 * (n_times // 32)
        self.classify = nn.Linear(feat, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = torch.flatten(x, 1)
        return self.classify(x)
