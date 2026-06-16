"""Evolve / compress / participate controllers.

Each slot, for a participating node, the controller picks an action
    a = (evolve_step, sparsify_rate)
given a context (drift flag/severity, current SNR, current grid size, budget
price, recent accuracy). The reward returned next slot is
    r = accuracy_gain - lambda * comm_cost
so the controller learns the comm/accuracy trade-off that the static Lagrangian
dual computes by hand. This is the RL half of the supervised(KAN)+RL union the
JSTSP call asks for.

Three controllers for ablation:
  FixedController        - never evolve, constant sparsity (lower bound).
  DualThresholdController- evolve on drift flag, sparsity from a budget heuristic.
  BanditController       - LinUCB contextual bandit over discrete actions (ours).
"""
from __future__ import annotations
import numpy as np

# Discrete action grid: (evolve_step in grid intervals, keep-fraction of update).
ACTIONS = [(0, 0.25), (0, 0.5), (0, 1.0),
           (2, 0.25), (2, 0.5), (2, 1.0)]


class FixedController:
    name = "fixed"

    def __init__(self, keep=1.0, **kw):
        self.keep = keep

    def act(self, ctx):
        return (0, self.keep)

    def update(self, ctx, action, reward):
        pass


class DualThresholdController:
    """Evolve when drift is flagged; set keep-fraction from a budget price.

    Mirrors the Step-1 Lagrangian intuition: when the per-slot budget price is
    high (budget tight), compress harder (smaller keep)."""
    name = "dual_threshold"

    def __init__(self, evolve_step=2, **kw):
        self.evolve_step = evolve_step

    def act(self, ctx):
        evolve = self.evolve_step if ctx.get("drift", 0) > 0 else 0
        price = float(np.clip(ctx.get("budget_price", 0.0), 0.0, 1.0))
        keep = float(np.clip(1.0 - 0.75 * price, 0.25, 1.0))
        return (evolve, keep)

    def update(self, ctx, action, reward):
        pass


class BanditController:
    """LinUCB over the discrete action set (Li et al. 2010)."""
    name = "bandit"

    def __init__(self, ctx_dim=5, alpha=0.6, actions=None, seed=0, **kw):
        self.actions = actions or ACTIONS
        self.alpha = alpha
        d = ctx_dim
        self.A = [np.eye(d) for _ in self.actions]
        self.b = [np.zeros(d) for _ in self.actions]
        self.rng = np.random.default_rng(seed)
        self.ctx_dim = d

    @staticmethod
    def _vec(ctx):
        return np.array([
            ctx.get("drift", 0.0),
            ctx.get("snr_norm", 0.0),
            ctx.get("grid_norm", 0.0),
            ctx.get("budget_price", 0.0),
            ctx.get("acc", 0.0),
        ], dtype=float)

    def act(self, ctx):
        x = self._vec(ctx)
        best, best_p = 0, -np.inf
        for i in range(len(self.actions)):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            p = float(theta @ x + self.alpha * np.sqrt(x @ A_inv @ x))
            if p > best_p:
                best_p, best = p, i
        self._last = (x, best)
        return self.actions[best]

    def update(self, ctx, action, reward):
        # action may be the returned tuple; map back to index via _last
        x, i = self._last
        self.A[i] += np.outer(x, x)
        self.b[i] += reward * x


def make_controller(name, **kw):
    name = (name or "bandit").lower()
    if name == "fixed":
        return FixedController(**kw)
    if name in ("dual", "dual_threshold", "threshold"):
        return DualThresholdController(**kw)
    if name in ("bandit", "linucb", "rl"):
        return BanditController(**kw)
    raise ValueError(name)
