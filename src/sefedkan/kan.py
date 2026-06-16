"""KAN building blocks.

A compact efficient-KAN style B-spline layer (after Blealtan/efficient-kan, MIT)
in pure PyTorch, with an explicit ``extend_grid`` operator that adds spline knots
while preserving the currently learned function. That operator is the
*self-evolution* mechanism in the paper: on detected drift a node grows its grid
to absorb a new data regime without overwriting old knots (low forgetting),
something an MLP cannot do natively.

Also provides a parameter-matched MLP so baselines isolate the *evolution*
mechanism rather than the KAN vs MLP capacity difference (empirical-verification
playbook: param-matched, ablated baselines).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """One KAN layer: y = base(SiLU(x)) + spline(x), splines over a B-spline grid."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range=(-2.0, 2.0),
        base_activation=nn.SiLU,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = tuple(grid_range)
        self.base_activation = base_activation()

        # Extended knot vector: grid_size + 1 interior knots, padded by spline_order
        # on each side -> length grid_size + 2*spline_order + 1.
        grid = self._build_grid(grid_size, self.grid_range, spline_order, in_features)
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.reset_parameters()

    @staticmethod
    def _build_grid(grid_size, grid_range, spline_order, in_features):
        h = (grid_range[1] - grid_range[0]) / grid_size
        steps = torch.arange(-spline_order, grid_size + spline_order + 1)
        grid = (steps * h + grid_range[0])  # (grid_size + 2*spline_order + 1,)
        return grid.expand(in_features, -1).contiguous()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            nn.init.normal_(self.spline_weight, std=0.1)

    @property
    def n_basis(self) -> int:
        return self.grid_size + self.spline_order

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Cox-de Boor basis. x: (B, in) -> (B, in, n_basis)."""
        grid = self.grid  # (in, n_knots)
        x = x.unsqueeze(-1)  # (B, in, 1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()  # (B, in, n_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(self.base_activation(x), self.base_weight)
        bs = self.b_splines(x)  # (B, in, n_basis)
        spline = torch.einsum("bik,oik->bo", bs, self.spline_weight)
        return base + spline

    # ---- self-evolution operator -------------------------------------------
    @torch.no_grad()
    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Least-squares fit new spline coeffs so basis(x) @ coeff ~= y.

        x: (S, in); y: (S, in, out) target spline outputs.
        returns coeff: (out, in, n_basis).
        """
        A = self.b_splines(x).transpose(0, 1)            # (in, S, n_basis)
        B = y.transpose(0, 1)                            # (in, S, out)
        sol = torch.linalg.lstsq(A, B).solution          # (in, n_basis, out)
        return sol.permute(2, 0, 1).contiguous()         # (out, in, n_basis)

    @torch.no_grad()
    def extend_grid(self, new_grid_size: int, n_samples: int = 256):
        """Grow the grid to ``new_grid_size`` keeping the current function.

        Samples the current spline densely, rebuilds a finer uniform grid, and
        refits coefficients. This is the paper's evolve operator.
        """
        if new_grid_size <= self.grid_size:
            return
        device = self.spline_weight.device
        xs = torch.linspace(self.grid_range[0], self.grid_range[1], n_samples, device=device)
        x = xs.unsqueeze(1).expand(n_samples, self.in_features)        # (S, in)
        bs = self.b_splines(x)                                         # (S, in, n_basis)
        y = torch.einsum("bik,oik->bio", bs, self.spline_weight)      # (S, in, out)

        new_grid = self._build_grid(new_grid_size, self.grid_range, self.spline_order,
                                    self.in_features).to(device)
        self.grid = new_grid
        self.grid_size = new_grid_size
        new_coeff = self._curve2coeff(x, y)                           # (out, in, n_basis')
        self.spline_weight = nn.Parameter(new_coeff)

    def regularization(self, l1: float = 1.0):
        """L1 on spline magnitude (sparsity / interpretability)."""
        return l1 * self.spline_weight.abs().mean()

    def num_params(self) -> int:
        return self.base_weight.numel() + self.spline_weight.numel()


class KANClassifier(nn.Module):
    """Small stacked-KAN classifier with a global grid-extension hook."""

    def __init__(self, in_dim, hidden, n_classes, grid_size=5, spline_order=3,
                 grid_range=(-2.0, 2.0)):
        super().__init__()
        dims = [in_dim] + list(hidden) + [n_classes]
        self.layers = nn.ModuleList(
            KANLinear(dims[i], dims[i + 1], grid_size, spline_order, grid_range)
            for i in range(len(dims) - 1)
        )
        self.grid_size = grid_size

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)  # keep activations inside grid_range
        return x

    @torch.no_grad()
    def evolve(self, new_grid_size: int):
        """Extend every layer's grid (the self-evolution action)."""
        for layer in self.layers:
            layer.extend_grid(new_grid_size)
        self.grid_size = new_grid_size

    def regularization(self, l1=1.0):
        return sum(layer.regularization(l1) for layer in self.layers)

    def num_params(self) -> int:
        return sum(l.num_params() for l in self.layers)


class MLPClassifier(nn.Module):
    """Plain MLP baseline; hidden sizes chosen to param-match a KAN config."""

    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        dims = [in_dim] + list(hidden) + [n_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def mlp_matching_kan(in_dim, n_classes, kan: KANClassifier, tol=0.1):
    """Build an MLP whose param count is within ``tol`` of ``kan``."""
    target = kan.num_params()
    best, best_gap = None, float("inf")
    for h in range(8, 1024, 4):
        m = MLPClassifier(in_dim, [h], n_classes)
        gap = abs(m.num_params() - target) / target
        if gap < best_gap:
            best, best_gap = m, gap
        if gap <= tol and m.num_params() >= target:
            return m, gap
    return best, best_gap
