"""Online drift detectors on a per-node loss/error stream.

A detector consumes a scalar quality signal each slot (e.g. local error rate) and
flags a change point. A flag is the trigger for the self-evolution + pseudo-label
refresh in the controller. Two standard detectors so the paper can ablate the
detector choice rather than hard-code one.
"""
from __future__ import annotations
from collections import deque
import numpy as np


class PageHinkley:
    """Page-Hinkley test for an increase in the mean of the stream."""

    def __init__(self, delta=0.005, lambda_=0.05, alpha=0.999):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.mean = 0.0
        self.n = 0
        self.m_t = 0.0
        self.M_t = 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.m_t = self.alpha * self.m_t + (x - self.mean - self.delta)
        self.M_t = min(self.M_t, self.m_t)
        flagged = (self.m_t - self.M_t) > self.lambda_
        if flagged:
            self.reset()
        return flagged


class ADWINLite:
    """Lightweight windowed change detector: split a sliding window in two and
    flag when the sub-window means differ beyond a Hoeffding-style bound."""

    def __init__(self, max_window=40, delta=0.05, min_sub=5):
        self.w = deque(maxlen=max_window)
        self.delta = delta
        self.min_sub = min_sub

    def reset(self):
        self.w.clear()

    def update(self, x: float) -> bool:
        self.w.append(float(x))
        n = len(self.w)
        if n < 2 * self.min_sub:
            return False
        arr = np.array(self.w)
        for cut in range(self.min_sub, n - self.min_sub + 1):
            a, b = arr[:cut], arr[cut:]
            m = 1.0 / (1.0 / len(a) + 1.0 / len(b))
            eps = np.sqrt(0.5 / m * np.log(4.0 / self.delta))
            if abs(a.mean() - b.mean()) > eps:
                self.w.clear()
                return True
        return False


def make_detector(name: str, **kw):
    name = (name or "page_hinkley").lower()
    if name in ("ph", "page_hinkley"):
        return PageHinkley(**kw)
    if name in ("adwin", "adwin_lite"):
        return ADWINLite(**kw)
    raise ValueError(name)
