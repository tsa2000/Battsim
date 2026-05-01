"""
utils.py — BattSim Shared Utilities
====================================
دوال مساعدة مشتركة بين:
  • machine1_dfn.py   (DFN simulation)
  • machine2_ekf.py   (EKF observer)
  • app.py            (Streamlit UI)
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


# ─────────────────────────────────────────────────────────────────────────────
# 1. Time & Array Utilities
# ─────────────────────────────────────────────────────────────────────────────

def safe_array(x, dtype: type = float) -> np.ndarray:
    """Convert any sequence to a 1-D numpy array safely."""
    return np.asarray(x, dtype=dtype).ravel()


def downsample(arr: np.ndarray, max_points: int = 2000) -> np.ndarray:
    """
    Uniform downsample to at most `max_points` elements.
    Used to reduce plot render time in Streamlit without losing trend shape.
    """
    n = len(arr)
    if n <= max_points:
        return arr
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return arr[idx]


def time_to_hours(t_sec: np.ndarray) -> np.ndarray:
    """Convert time array from seconds → hours."""
    return t_sec / 3600.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. SOC & Capacity Utilities
# ─────────────────────────────────────────────────────────────────────────────

def soc_to_percent(soc: np.ndarray) -> np.ndarray:
    """Convert SOC in [0, 1] → percentage [0, 100]."""
    return np.clip(soc, 0.0, 1.0) * 100.0


def rmse(actual: np.ndarray, estimated: np.ndarray) -> float:
    """Root-Mean-Square Error between two arrays."""
    a = safe_array(actual)
    e = safe_array(estimated)
    return float(np.sqrt(np.mean((a - e) ** 2)))


def mae(actual: np.ndarray, estimated: np.ndarray) -> float:
    """Mean-Absolute Error between two arrays."""
    a = safe_array(actual)
    e = safe_array(estimated)
    return float(np.mean(np.abs(a - e)))


def max_error(actual: np.ndarray, estimated: np.ndarray) -> float:
    """Maximum absolute error between two arrays."""
    a = safe_array(actual)
    e = safe_array(estimated)
    return float(np.max(np.abs(a - e)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-Cycle Analysis
# ─────────────────────────────────────────────────────────────────────────────

def detect_cycles(soc: np.ndarray, threshold: float = 0.8) -> list[tuple[int, int]]:
    """
    Detect charge/discharge cycle boundaries from SOC trace.

    Returns a list of (start_idx, end_idx) tuples.
    A new cycle begins each time SOC crosses `threshold` going upward
    after having been below it.

    Parameters
    ----------
    soc       : np.ndarray  SOC in [0, 1]
    threshold : float       SOC crossing level (default 0.8)
    """
    soc = safe_array(soc)
    cycles: list[tuple[int, int]] = []
    in_cycle = False
    start = 0

    for i in range(1, len(soc)):
        if not in_cycle and soc[i] >= threshold and soc[i - 1] < threshold:
            start = i
            in_cycle = True
        elif in_cycle and soc[i] < threshold and soc[i - 1] >= threshold:
            cycles.append((start, i))
            in_cycle = False

    if in_cycle:
        cycles.append((start, len(soc) - 1))

    return cycles


def per_cycle_stats(log: dict, n_cycles: int | None = None) -> list[dict]:
    """
    Compute per-cycle summary statistics from a run_cosim() log dict.

    Returns a list of dicts, one per detected cycle:
        {
          "cycle"      : int,          # cycle number (1-indexed)
          "rmse_soc"   : float,        # RMSE(soc_true, soc_est)  [% points]
          "mae_soc"    : float,        # MAE(soc_true, soc_est)   [% points]
          "max_err_soc": float,        # max |error| SOC          [% points]
          "mean_sigma" : float,        # mean sigma_soc            [% points]
          "max_sigma"  : float,        # max  sigma_soc            [% points]
          "mean_nis"   : float,        # mean NIS (target ≈ 1.0)
          "rmse_v"     : float,        # RMSE(V_true, V_est)      [mV]
        }
    """
    soc_true  = safe_array(log["soc_true"])
    soc_est   = safe_array(log["soc_est"])
    sigma_soc = safe_array(log.get("sigma_soc", np.zeros_like(soc_true)))
    nis       = safe_array(log.get("NIS", np.ones_like(soc_true)))
    v_true    = safe_array(log["V_true"])
    v_est     = safe_array(log.get("V_est", v_true))

    cycles = detect_cycles(soc_true)

    # If no cycles detected, treat entire log as one cycle
    if not cycles:
        cycles = [(0, len(soc_true) - 1)]

    # Optionally limit to n_cycles
    if n_cycles is not None:
        cycles = cycles[:n_cycles]

    stats = []
    for c_num, (s, e) in enumerate(cycles, start=1):
        sl = slice(s, e + 1)
        st_sl = soc_to_percent(soc_true[sl])
        se_sl = soc_to_percent(soc_est[sl])
        sg_sl = soc_to_percent(sigma_soc[sl])

        stats.append({
            "cycle"      : c_num,
            "rmse_soc"   : rmse(st_sl, se_sl),
            "mae_soc"    : mae(st_sl, se_sl),
            "max_err_soc": max_error(st_sl, se_sl),
            "mean_sigma" : float(np.mean(sg_sl)),
            "max_sigma"  : float(np.max(sg_sl)),
            "mean_nis"   : float(np.mean(nis[sl])),
            "rmse_v"     : rmse(v_true[sl] * 1000, v_est[sl] * 1000),  # → mV
        })

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 4. NIS Calibration Check
# ─────────────────────────────────────────────────────────────────────────────

def nis_calibration(nis: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Check EKF calibration using Normalised Innovation Squared (NIS).

    For a well-calibrated filter with scalar measurement:
        NIS ~ χ²(1)  →  E[NIS] ≈ 1.0

    Chi-squared 95% confidence interval for χ²(1): [0.004, 5.024]

    Returns
    -------
    dict:
        mean_nis    : float   — should be ≈ 1.0
        pct_in_band : float   — % of NIS values inside [χ²_lo, χ²_hi]
        calibrated  : bool    — True if mean ≈ 1.0 ± 0.3 and pct_in_band ≥ 0.90
        verdict     : str     — human-readable assessment
    """
    nis = safe_array(nis)
    chi2_lo, chi2_hi = 0.004, 5.024  # χ²(1) 95% bounds
    mean_nis    = float(np.mean(nis))
    pct_in_band = float(np.mean((nis >= chi2_lo) & (nis <= chi2_hi)))
    calibrated  = (abs(mean_nis - 1.0) < 0.3) and (pct_in_band >= 0.90)

    if mean_nis < 0.7:
        verdict = "Over-confident: Q/R too small — increase process noise"
    elif mean_nis > 1.3:
        verdict = "Under-confident: Q/R too large — decrease process noise"
    else:
        verdict = "Well-calibrated ✅" if calibrated else "Borderline — monitor NIS"

    return {
        "mean_nis"    : mean_nis,
        "pct_in_band" : pct_in_band * 100,
        "calibrated"  : calibrated,
        "verdict"     : verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Formatting Helpers (for Streamlit UI)
# ─────────────────────────────────────────────────────────────────────────────

def fmt_soc(soc: float, decimals: int = 1) -> str:
    """Format SOC as percentage string: 0.853 → '85.3 %'"""
    return f"{soc * 100:.{decimals}f} %"


def fmt_rmse(rmse_val: float, unit: str = "%") -> str:
    """Format RMSE value with unit: 1.23, '%' → '1.23 %'"""
    return f"{rmse_val:.2f} {unit}"


def fmt_sigma(sigma: float) -> str:
    """Format ±1σ uncertainty: 0.012 → '±1.2 %'"""
    return f"±{sigma * 100:.2f} %"


def summary_dict(log: dict) -> dict:
    """
    Compute a flat summary dict from a run_cosim() log for display in Streamlit metrics.

    Returns
    -------
    dict with keys:
        rmse_soc_pct, mae_soc_pct, max_err_soc_pct,
        mean_sigma_pct, max_sigma_pct,
        rmse_v_mv,
        nis_mean, nis_calibrated, nis_verdict
    """
    soc_true  = soc_to_percent(safe_array(log["soc_true"]))
    soc_est   = soc_to_percent(safe_array(log["soc_est"]))
    sigma_soc = soc_to_percent(safe_array(log.get("sigma_soc", np.zeros_like(soc_true))))
    v_true    = safe_array(log["V_true"]) * 1000  # → mV
    v_est     = safe_array(log.get("V_est", log["V_true"])) * 1000

    _cs   = per_cycle_stats(log)
    _s    = slice(len(log["soc_true"]) // max(len(_cs), 1), None)
    
    nis_stats = nis_calibration(safe_array(log.get("NIS", np.ones(1))))


    return {
        "rmse_soc_pct"   : rmse(soc_true[_s], soc_est[_s]),
        "mae_soc_pct"    : mae(soc_true[_s], soc_est[_s]),
        "max_err_soc_pct": max_error(soc_true[_s], soc_est[_s]),
        "mean_sigma_pct" : float(np.mean(sigma_soc)),
        "max_sigma_pct"  : float(np.max(sigma_soc)),
        "rmse_v_mv"      : rmse(v_true[_s], v_est[_s]),
        "nis_mean"       : nis_stats["mean_nis"],
        "nis_calibrated" : nis_stats["calibrated"],
        "nis_verdict"    : nis_stats["verdict"],
    }

