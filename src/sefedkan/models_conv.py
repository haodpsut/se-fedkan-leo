"""Conv front-end models for raw I/Q automatic modulation classification.

The flat featurization (downsampled I/Q + spectral moments -> 26 dims) underfits
RML2016 badly (~0.22 avg acc) because it throws away the temporal I/Q structure
that AMC needs. Here a small 1D-CNN front-end consumes the raw (2, L) frame and
produces a compact, bounded feature vector that feeds the KAN (or MLP) head.

Self-evolution (grid-extension) applies ONLY to the KAN head; the conv front-end
is a fixed architecture that is FedAvg-aggregated normally. The whole model's
parameter vector stays consistently ordered across nodes once all heads are
aligned to the same grid, so the existing sparse-FedAvg aggregator works
unchanged (front params first, head params after).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .kan import KANClassifier, MLPClassifier


class ConvFrontend(nn.Module):
    """Small 1D-CNN: (B, in_ch, L) -> (B, feat_dim), tanh-bounded for the KAN grid.

    Uses an adaptive pool to a fixed temporal length then flattens, so the
    temporal/phase structure that modulation classification needs is preserved
    (a global average pool over time destroys it), and the feature size is fixed
    regardless of L (128 for RML2016, 1024 for RML2018).
    """

    def __init__(self, in_ch=2, channels=(16, 32), kernel=5, feat_dim=24, pool_len=16):
        super().__init__()
        layers, c_prev = [], in_ch
        for c in channels:
            layers += [nn.Conv1d(c_prev, c, kernel, padding=kernel // 2),
                       nn.ReLU(), nn.MaxPool1d(2)]
            c_prev = c
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(pool_len)
        self.proj = nn.Linear(c_prev * pool_len, feat_dim)
        self.feat_dim = feat_dim

    def forward(self, x):
        h = self.conv(x)                  # (B, C, L')
        h = self.pool(h)                  # (B, C, pool_len)
        h = h.flatten(1)                  # (B, C*pool_len)
        return torch.tanh(self.proj(h))   # bounded into the KAN grid range


class ConvKANClassifier(nn.Module):
    """Conv front-end + KAN head. evolve() extends only the head's grid."""

    def __init__(self, in_ch, L, n_classes, kan_hidden=(16,), grid_size=5,
                 spline_order=3, channels=(16, 32), kernel=5, feat_dim=24,
                 grid_range=(-2.0, 2.0)):
        super().__init__()
        self.front = ConvFrontend(in_ch, channels, kernel, feat_dim)
        self.head = KANClassifier(feat_dim, kan_hidden, n_classes,
                                  grid_size, spline_order, grid_range)
        self.grid_size = grid_size

    def forward(self, x):
        return self.head(self.front(x))

    @torch.no_grad()
    def evolve(self, new_grid_size):
        self.head.evolve(new_grid_size)
        self.grid_size = self.head.grid_size

    def regularization(self, l1=1.0):
        return self.head.regularization(l1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class ConvMLPClassifier(nn.Module):
    """Conv front-end + MLP head (param-matched baseline; no grid / no evolve)."""

    def __init__(self, in_ch, L, n_classes, mlp_hidden=(140,), channels=(16, 32),
                 kernel=5, feat_dim=24):
        super().__init__()
        self.front = ConvFrontend(in_ch, channels, kernel, feat_dim)
        self.head = MLPClassifier(feat_dim, mlp_hidden, n_classes)

    def forward(self, x):
        return self.head(self.front(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def conv_mlp_matching(in_ch, L, n_classes, kan_model: ConvKANClassifier, tol=0.1, **kw):
    """Pick an MLP-head width so the Conv-MLP param count matches the Conv-KAN."""
    target = kan_model.num_params()
    best, best_gap = None, float("inf")
    for h in range(8, 1024, 4):
        m = ConvMLPClassifier(in_ch, L, n_classes, (h,), **kw)
        gap = abs(m.num_params() - target) / target
        if gap < best_gap:
            best, best_gap = m, gap
        if gap <= tol and m.num_params() >= target:
            return m, gap
    return best, best_gap
