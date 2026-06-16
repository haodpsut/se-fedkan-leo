"""Data layer.

Three sources, one interface:
  - synthetic_amc(): dependency-free I/Q frame generator for several digital
    modulations at a target SNR. Used for the local CPU smoke test.
  - load_radioml2016(path): RML2016.10a pickle ((mod,snr) -> frames).
  - load_radioml2018(path): RML2018.01A HDF5 (X, Y one-hot, Z snr).

All return an AMCBank with frames of shape (N, 2, L) (I/Q), integer labels, and
per-frame SNR (dB), plus a featurizer that produces the model input:
  - "iq":   downsampled raw I/Q (the adaptive-SP raw view)
  - "stat": higher-order statistics + spectral moments (the engineered view)
  - "multimodal": concatenation of both (the SI multimodal hook)
The drift stream then samples, per node and per slot, frames whose SNR matches the
current LEO-elevation SNR, yielding a non-stationary local distribution.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

DEFAULT_MODS = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "PAM4", "GFSK", "CPFSK"]


def _constellation(mod: str, rng):
    if mod == "BPSK":
        pts = np.array([1, -1], dtype=complex)
    elif mod == "QPSK":
        pts = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
    elif mod == "8PSK":
        pts = np.exp(1j * np.arange(8) * np.pi / 4)
    elif mod == "PAM4":
        pts = np.array([-3, -1, 1, 3], dtype=complex)
    elif mod == "QAM16":
        g = np.array([-3, -1, 1, 3]); pts = (g[:, None] + 1j * g[None, :]).ravel()
    elif mod == "QAM64":
        g = np.array([-7, -5, -3, -1, 1, 3, 5, 7]); pts = (g[:, None] + 1j * g[None, :]).ravel()
    elif mod in ("GFSK", "CPFSK"):
        return None  # handled as FSK below
    else:
        raise ValueError(mod)
    return pts / np.sqrt((np.abs(pts) ** 2).mean())


def _gen_frame(mod, snr_db, L, rng):
    if mod in ("GFSK", "CPFSK"):
        bits = rng.integers(0, 2, L) * 2 - 1
        h = 0.5
        phase = np.cumsum(bits) * np.pi * h
        sig = np.exp(1j * phase)
    else:
        pts = _constellation(mod, rng)
        idx = rng.integers(0, len(pts), L)
        sig = pts[idx]
    p_sig = (np.abs(sig) ** 2).mean()
    snr = 10 ** (snr_db / 10.0)
    n_std = np.sqrt(p_sig / (2 * snr))
    noise = n_std * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
    x = sig + noise
    return np.stack([x.real, x.imag]).astype(np.float32)  # (2, L)


@dataclass
class AMCBank:
    frames: np.ndarray   # (N, 2, L)
    labels: np.ndarray   # (N,)
    snr: np.ndarray      # (N,)
    mods: list

    @property
    def n_classes(self):
        return len(self.mods)

    def snr_levels(self):
        return np.unique(np.round(self.snr).astype(int))

    def index_by_snr(self):
        """dict: snr_bin -> array of frame indices."""
        out = {}
        for s in self.snr_levels():
            out[int(s)] = np.where(np.round(self.snr).astype(int) == s)[0]
        return out


def synthetic_amc(mods=None, snr_list=None, per_combo=200, L=128, seed=0) -> AMCBank:
    mods = mods or DEFAULT_MODS
    snr_list = snr_list if snr_list is not None else list(range(-10, 21, 2))
    rng = np.random.default_rng(seed)
    frames, labels, snrs = [], [], []
    for ci, mod in enumerate(mods):
        for s in snr_list:
            for _ in range(per_combo):
                frames.append(_gen_frame(mod, s, L, rng))
                labels.append(ci)
                snrs.append(s)
    return AMCBank(np.asarray(frames), np.asarray(labels), np.asarray(snrs, float), list(mods))


# ---- featurization ---------------------------------------------------------
def _spectral_moments(frame):
    x = frame[0] + 1j * frame[1]
    X = np.fft.fftshift(np.fft.fft(x))
    psd = (np.abs(X) ** 2); psd = psd / (psd.sum() + 1e-9)
    f = np.linspace(-0.5, 0.5, len(psd))
    m1 = (f * psd).sum()
    m2 = ((f - m1) ** 2 * psd).sum()
    m3 = ((f - m1) ** 3 * psd).sum()
    m4 = ((f - m1) ** 4 * psd).sum()
    # amplitude / phase higher-order stats (classic AMC cumulant proxies)
    a = np.abs(x); ph = np.angle(x)
    feats = [m1, m2, m3, m4, a.mean(), a.std(), a.max(),
             np.std(ph), np.mean(a ** 2), np.mean(a ** 4)]
    return np.array(feats, dtype=np.float32)


def featurize(bank: AMCBank, mode="multimodal", iq_ds=8):
    """Return (X, view_dims). view_dims documents the multimodal split."""
    if mode in ("iq", "multimodal"):
        # downsample raw I/Q to keep KAN input modest
        L = bank.frames.shape[-1]
        step = max(1, L // iq_ds)
        iq = bank.frames[:, :, ::step][:, :, :iq_ds].reshape(len(bank.frames), -1)
    if mode in ("stat", "multimodal"):
        stat = np.stack([_spectral_moments(f) for f in bank.frames])
    if mode == "iq":
        X, dims = iq, {"iq": iq.shape[1]}
    elif mode == "stat":
        X, dims = stat, {"stat": stat.shape[1]}
    else:
        X = np.concatenate([iq, stat], axis=1)
        dims = {"iq": iq.shape[1], "stat": stat.shape[1]}
    # per-feature standardization (fit on whole bank; deterministic)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    return X, dims


# ---- real RadioML loaders (server) ----------------------------------------
def load_radioml2016(path) -> AMCBank:
    import pickle
    with open(path, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    mods = sorted({k[0] for k in d.keys()})
    mod_idx = {m: i for i, m in enumerate(mods)}
    frames, labels, snrs = [], [], []
    for (mod, snr), arr in d.items():
        frames.append(arr.astype(np.float32))           # (n, 2, 128)
        labels.append(np.full(len(arr), mod_idx[mod]))
        snrs.append(np.full(len(arr), snr))
    return AMCBank(np.concatenate(frames), np.concatenate(labels),
                   np.concatenate(snrs).astype(float), mods)


def load_radioml2018(path, max_per_combo=None) -> AMCBank:
    import h5py
    with h5py.File(path, "r") as f:
        X = f["X"]; Y = f["Y"]; Z = f["Z"]
        labels_oh = np.asarray(Y)
        snr = np.asarray(Z).ravel()
        y = labels_oh.argmax(1)
        frames = np.asarray(X).transpose(0, 2, 1).astype(np.float32)  # (N,2,1024)
    mods = [f"mod{i}" for i in range(labels_oh.shape[1])]
    bank = AMCBank(frames, y, snr.astype(float), mods)
    if max_per_combo:
        bank = _subsample(bank, max_per_combo)
    return bank


def _subsample(bank: AMCBank, max_per_combo, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    snr_r = np.round(bank.snr).astype(int)
    for c in range(bank.n_classes):
        for s in np.unique(snr_r):
            idx = np.where((bank.labels == c) & (snr_r == s))[0]
            if len(idx) > max_per_combo:
                idx = rng.choice(idx, max_per_combo, replace=False)
            keep.append(idx)
    keep = np.concatenate(keep)
    return AMCBank(bank.frames[keep], bank.labels[keep], bank.snr[keep], bank.mods)
