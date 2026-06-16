"""Federated mechanics: local training, top-k sparsification with comm-bit
accounting, and FedAvg/FedProx aggregation of (possibly grid-extended) KANs.

Heterogeneous grids: when a node self-evolves, its parameter vector grows. We
aggregate in a *unified* grid space for the round: the server grid is raised to
the largest participant grid, every model is grid-extended up to it (extend_grid
preserves the function, so this is lossless), and deltas are FedAvg-ed there.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def clone_model(model):
    return copy.deepcopy(model)


def align_to_grid(model, grid_size):
    """Grid-extend a KAN classifier up to grid_size (no-op if already >=)."""
    if hasattr(model, "evolve") and model.grid_size < grid_size:
        model.evolve(grid_size)
    return model


def local_train(model, X, y, epochs=3, lr=0.01, mu=0.0, global_vec=None,
                l1=0.0, weight_decay=1e-4, batch=128, device="cpu"):
    """Train a client model on its current-slot data. FedProx term if mu>0."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = model(X[idx])
            loss = F.cross_entropy(out, y[idx])
            if l1 > 0 and hasattr(model, "regularization"):
                loss = loss + model.regularization(l1)
            if mu > 0 and global_vec is not None:
                cur = parameters_to_vector(model.parameters())
                loss = loss + 0.5 * mu * ((cur - global_vec) ** 2).sum()
            loss.backward()
            opt.step()
    return float(loss.detach().cpu())


def sparsify(delta: torch.Tensor, keep_frac: float):
    """Top-k by magnitude. Returns (masked_delta, n_nonzero)."""
    keep_frac = float(np.clip(keep_frac, 1e-3, 1.0))
    k = max(1, int(keep_frac * delta.numel()))
    if k >= delta.numel():
        return delta, delta.numel()
    thresh = torch.topk(delta.abs(), k, largest=True).values.min()
    mask = delta.abs() >= thresh
    return delta * mask, int(mask.sum().item())


def fed_aggregate(global_model, participants, bits_per_param=32, mu=0.0):
    """One round of sparse FedAvg in unified grid space.

    participants: list of dicts with keys
        model   -> trained client KAN/MLP
        weight  -> aggregation weight (e.g. n_samples)
        keep    -> sparsify keep-fraction chosen by the controller
    Returns total uplink bits this round and the per-node nonzero counts.
    """
    is_kan = hasattr(global_model, "evolve")
    round_grid = global_model.grid_size if is_kan else None
    if is_kan:
        round_grid = max([round_grid] + [p["model"].grid_size for p in participants])
        align_to_grid(global_model, round_grid)
    ref = parameters_to_vector(global_model.parameters()).detach().clone()

    wsum = sum(p["weight"] for p in participants) or 1.0
    agg = torch.zeros_like(ref)
    total_bits, nonzeros = 0, []
    for p in participants:
        m = p["model"]
        if is_kan:
            align_to_grid(m, round_grid)
        vec = parameters_to_vector(m.parameters()).detach()
        delta = vec - ref
        sdelta, nz = sparsify(delta, p.get("keep", 1.0))
        agg += (p["weight"] / wsum) * sdelta
        total_bits += nz * bits_per_param
        nonzeros.append(nz)

    new_vec = ref + agg
    vector_to_parameters(new_vec, global_model.parameters())
    return total_bits, nonzeros


@torch.no_grad()
def evaluate(model, X, y, device="cpu"):
    model.to(device).eval()
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    pred = model(X).argmax(1)
    return float((pred == y).float().mean().cpu())


@torch.no_grad()
def pseudo_label(model, X, conf_thresh=0.9, device="cpu"):
    """Self-generated labels for unlabeled samples above a confidence threshold."""
    model.to(device).eval()
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    prob = F.softmax(model(X), dim=1)
    conf, lab = prob.max(1)
    keep = conf >= conf_thresh
    return lab.cpu().numpy(), keep.cpu().numpy()
