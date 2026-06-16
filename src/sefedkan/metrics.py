"""Evaluation metrics for non-stationary federated learning."""
from __future__ import annotations
import numpy as np


def average_accuracy(acc_series):
    """Time-averaged accuracy under drift (the headline metric)."""
    a = np.asarray([x for x in acc_series if x is not None], dtype=float)
    return float(a.mean()) if len(a) else 0.0


def backward_transfer(regime_acc_log):
    """Mean (final - first) accuracy across regimes seen more than once.

    regime_acc_log: dict regime -> list of (slot, acc) observed for that regime.
    Positive = the model improved on old regimes (good); negative = forgetting.
    """
    vals = []
    for _, seq in regime_acc_log.items():
        if len(seq) >= 2:
            vals.append(seq[-1][1] - seq[0][1])
    return float(np.mean(vals)) if vals else 0.0


def forgetting(regime_acc_log):
    """Mean (max-so-far - final) across regimes; lower is better."""
    vals = []
    for _, seq in regime_acc_log.items():
        if len(seq) >= 2:
            accs = [a for _, a in seq]
            vals.append(max(accs) - accs[-1])
    return float(np.mean(vals)) if vals else 0.0


def detection_delay(drift_truth, drift_flags):
    """Mean slots between a true regime change and the next raised flag."""
    delays = []
    flags = np.where(np.asarray(drift_flags))[0]
    for t in np.where(np.asarray(drift_truth))[0]:
        later = flags[flags >= t]
        if len(later):
            delays.append(int(later[0] - t))
    return float(np.mean(delays)) if delays else float("nan")
