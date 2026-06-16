"""Lightweight analytic LEO geometry: per-node elevation, receive SNR, and
visibility over one or more passes. This is the non-stationarity driver: as a
satellite rises and sets, the slant range and hence the receive SNR sweep, so the
effective signal regime at each edge node drifts in time.

A full SGP4/TLE path (Walker constellation) lives in the reused
``new-paper-colab/code/src/satqkd/orbit_tle.py`` and can be swapped in on the
server; this analytic version keeps the smoke test dependency-free.
"""
from __future__ import annotations
import numpy as np

RE_KM = 6371.0  # Earth radius


def slant_range_km(elev_deg: np.ndarray, alt_km: float = 550.0) -> np.ndarray:
    """Slant range from ground station to LEO sat at given elevation."""
    el = np.radians(np.asarray(elev_deg, dtype=float))
    re, h = RE_KM, alt_km
    return np.sqrt((re * np.sin(el)) ** 2 + 2 * re * h + h * h) - re * np.sin(el)


def pass_elevation(n_slots: int, max_elev_deg: float, theta_min_deg: float = 10.0) -> np.ndarray:
    """Triangular-ish elevation profile of a single pass over n_slots.

    Rises from horizon to ``max_elev``, then sets. Slots below ``theta_min`` are
    invisible (returned as elevation < theta_min so the visibility gate drops them).
    """
    half = n_slots / 2.0
    t = np.arange(n_slots)
    # symmetric peak at the middle
    el = max_elev_deg * (1.0 - np.abs(t - half) / half)
    return np.clip(el, 0.0, max_elev_deg)


def snr_from_elevation(elev_deg, snr_zenith_db=20.0, alt_km=550.0, ref_elev_deg=90.0):
    """Map elevation -> receive SNR via free-space slant-range path loss.

    SNR(el) = snr_zenith - 20*log10(d(el)/d(ref)). At the zenith d is minimal so
    SNR is maximal; near the horizon d grows and SNR drops, producing the drift.
    """
    elev_deg = np.asarray(elev_deg, dtype=float)
    d = slant_range_km(np.maximum(elev_deg, 1e-3), alt_km)
    d_ref = slant_range_km(np.array([ref_elev_deg]), alt_km)[0]
    return snr_zenith_db - 20.0 * np.log10(d / d_ref)


def node_snr_stream(n_slots, max_elev_deg, theta_min_deg=10.0, snr_zenith_db=20.0,
                    n_passes=1, gap_slots=0, rng=None):
    """Concatenate ``n_passes`` passes (optionally separated by invisible gaps).

    Returns (snr_db, visible) each of length ~ n_passes*(n_slots+gap_slots).
    Invisible slots carry snr=-inf and visible=False.
    """
    rng = rng or np.random.default_rng(0)
    snr_all, vis_all = [], []
    for _ in range(n_passes):
        el = pass_elevation(n_slots, max_elev_deg, theta_min_deg)
        vis = el >= theta_min_deg
        snr = snr_from_elevation(el, snr_zenith_db)
        snr = np.where(vis, snr, -np.inf)
        snr_all.append(snr)
        vis_all.append(vis)
        if gap_slots > 0:
            snr_all.append(np.full(gap_slots, -np.inf))
            vis_all.append(np.zeros(gap_slots, dtype=bool))
    return np.concatenate(snr_all), np.concatenate(vis_all)
