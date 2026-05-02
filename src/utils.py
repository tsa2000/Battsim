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
# 3. Per-Cycle Analysis (Uncertainty Propagation)
# ─────────────────────────────────────────────────────────────────────────────

def detect_cycles(soc: np.ndarray, threshold: float = 0.8) -> list[tuple[int, int]]:
    """
    Detect charge/discharge cycle boundaries from SOC trace.

    A new cycle begins each time SOC crosses `threshold` going upward
    after having been below it — consistent with CC and CC-CV protocols.

    Parameters
    ----------
    soc       : np.ndarray  SOC in [0, 1]
    threshold : float       SOC crossing level (default 0.8)

    Returns
    -------
    list of (start_idx, end_idx) tuples
    """
    soc = safe_array(soc)
    cycles: list[tuple[int, int]] = []
    in_cycle = False
    start = 0

    for i in range(1, len(soc)):
        if not in_cycle and soc[i] >= threshold and soc[i - 1] < threshold:
            start    = i
            in_cycle = True
        elif in_cycle and soc[i] < threshold and soc[i - 1] >= threshold:
            cycles.append((start, i))
            in_cycle = False

    if in_cycle:
        cycles.append((start, len(soc) - 1))

    return cycles


def per_cycle_stats(log: dict, n_cycles: int | None = None) -> list[dict]:
    """
    Compute per-cycle uncertainty propagation metrics from a run_cosim() log.

    This directly addresses the research objective:
    "quantify uncertainty propagation as the battery charges and discharges
    over multiple cycles." (Supervisor task brief)

    For each detected cycle the following metrics are computed:
      - rmse_soc    : RMS SOC estimation error [% points]
      - mae_soc     : Mean absolute SOC error [% points]
      - max_err_soc : Peak absolute SOC error [% points]
      - mean_sigma  : Mean 1σ uncertainty from EKF P[0,0] [% points]
      - max_sigma   : Maximum 1σ in cycle [% points]
      - mean_nis    : Mean NIS (expected ≈ 1.0 for a consistent filter)
      - rmse_v      : Voltage estimation RMSE [mV]

    References
    ----------
    Bar-Shalom et al. 2001, Estimation with Applications to Tracking
    and Navigation, Wiley — Chapters 5 & 10 (consistency & UQ metrics)
    Plett 2004, J. Power Sources 134, 252–261 — NIS interpretation
    """
    soc_true  = safe_array(log["soc_true"])
    soc_est   = safe_array(log["soc_est"])
    sigma_soc = safe_array(log.get("sigma_soc", np.zeros_like(soc_true)))
    nis       = safe_array(log.get("NIS",        np.ones_like(soc_true)))
    v_true    = safe_array(log["V_true"])
    v_est     = safe_array(log.get("V_est", v_true))

    cycles = detect_cycles(soc_true)

    # If no cycles detected, treat entire log as one cycle
    if not cycles:
        cycles = [(0, len(soc_true) - 1)]

    if n_cycles is not None:
        cycles = cycles[:n_cycles]

    stats = []
    for c_num, (s, e) in enumerate(cycles, start=1):
        sl    = slice(s, e + 1)
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
            "rmse_v"     : rmse(v_true[sl] * 1000.0, v_est[sl] * 1000.0),  # → mV
        })

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 4. NIS Calibration Check
# ─────────────────────────────────────────────────────────────────────────────

def nis_calibration(nis: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Check EKF filter calibration using Normalised Innovation Squared (NIS).

    For a well-calibrated filter with a scalar measurement:
        NIS ~ χ²(1)  →  E[NIS] ≈ 1.0

    The 95% confidence interval for a single χ²(1) sample is [0.004, 5.024].
    These bounds come from:
        χ²(1, 0.025) = 0.004   (lower 2.5% tail)
        χ²(1, 0.975) = 5.024   (upper 2.5% tail)

    References
    ----------
    Bar-Shalom et al. 2001, Estimation with Applications, Ch. 5 — NIS test
    Julier & Uhlmann 1997, SPIE — consistency of sigma-point filters
    kalman-filter.com/normalized-innovation-squared — derivation

    Returns
    -------
    dict:
        mean_nis    : float  — should be ≈ 1.0
        pct_in_band : float  — % of NIS inside [0.004, 5.024]
        calibrated  : bool   — True if |mean − 1| < 0.3 and pct_in_band ≥ 90%
        verdict     : str    — human-readable calibration assessment
    """
    nis = safe_array(nis)

    # χ²(1) 95% confidence bounds
    # Source: Bar-Shalom 2001 Table B.1 | scipy.stats.chi2.ppf([0.025, 0.975], df=1)
    chi2_lo, chi2_hi = 0.004, 5.024

    mean_nis    = float(np.mean(nis))
    pct_in_band = float(np.mean((nis >= chi2_lo) & (nis <= chi2_hi)))
    calibrated  = (abs(mean_nis - 1.0) < 0.3) and (pct_in_band >= 0.90)

    if mean_nis < 0.7:
        verdict = "Over-confident: Q too small — increase process noise (q_scale)"
    elif mean_nis > 1.3:
        verdict = "Under-confident: Q too large — decrease process noise (q_scale)"
    else:
        verdict = "Well-calibrated ✅" if calibrated else "Borderline — monitor NIS trend"

    return {
        "mean_nis"    : mean_nis,
        "pct_in_band" : pct_in_band * 100.0,
        "calibrated"  : calibrated,
        "verdict"     : verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Formatting Helpers (Streamlit UI)
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
    Compute a flat summary dict from a run_cosim() log for Streamlit metrics.

    Skips the first 10% of the simulation (warm-up / convergence period)
    before computing RMSE and MAE — consistent with Plett 2004 evaluation
    practice where the first few steps are excluded from performance metrics.

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
    v_true    = safe_array(log["V_true"])  * 1000.0   # → mV
    v_est     = safe_array(log.get("V_est", log["V_true"])) * 1000.0

    # Skip warm-up: first 10% of timesteps
    _skip = max(1, len(soc_true) // 10)
    _s    = slice(_skip, None)

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
