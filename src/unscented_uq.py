from __future__ import annotations
import numpy as np


def uncertainty_per_cycle(log: dict, dt: float = 10.0) -> list[dict]:
    """
    Compute uncertainty propagation metrics per charge/discharge cycle.

    This function answers the core research question:
    "How does uncertainty propagate as the battery charges and discharges
    over multiple cycles?"

    For each detected cycle, the following metrics are computed:
      - RMSE        : RMS error between estimated and true SOC
      - MAE         : Mean absolute SOC error
      - mean_sigma  : Mean 1σ SOC uncertainty from EKF covariance P[0,0]
      - mean_NIS    : Mean Normalised Innovation Squared (filter consistency)
      - ci_width    : Mean 95% confidence interval width (= 4σ)
      - max_error   : Peak absolute SOC error in the cycle

    Cycle detection
    ---------------
    A new cycle begins each time SOC crosses above 0.95 (fully charged).
    This is consistent with CC-CV and CC charging protocols where
    each cycle starts at near-full charge.

    References
    ----------
    Bar-Shalom et al. 2001, Estimation with Applications to Tracking
    and Navigation, Wiley — Chapters 5 & 10 (consistency metrics)
    Plett 2004, J. Power Sources 134, 252–261 — NIS interpretation
    """
    t         = np.asarray(log["t"])
    soc_true  = np.asarray(log["soc_true"])
    soc_est   = np.asarray(log["soc_est"])
    sigma_soc = np.asarray(log["sigma_soc"])
    NIS       = np.asarray(log["NIS"])
    ci_upper  = np.asarray(log["ci_upper"])
    ci_lower  = np.asarray(log["ci_lower"])
    N         = len(t)

    # ── Cycle boundary detection ──────────────────────────────────────────
    # A cycle starts when SOC_true crosses above 0.95 (charge complete).
    # First cycle always starts at index 0.
    starts = [0]
    for i in range(1, N):
        if soc_true[i] >= 0.95 and soc_true[i - 1] < 0.95:
            starts.append(i)
    starts.append(N)  # sentinel

    results = []
    for k in range(len(starts) - 1):
        i0 = starts[k]
        i1 = starts[k + 1]
        if i1 - i0 < 5:
            continue  # skip degenerate segments

        mask = slice(i0, i1)
        err  = soc_est[mask] - soc_true[mask]

        results.append({
            "cycle":      k + 1,
            "t_start":    float(t[i0]),
            "t_end":      float(t[i1 - 1]),
            "rmse":       float(np.sqrt(np.mean(err ** 2))),
            "mae":        float(np.mean(np.abs(err))),
            "max_error":  float(np.max(np.abs(err))),
            "mean_sigma": float(np.mean(sigma_soc[mask])),
            "mean_NIS":   float(np.mean(NIS[mask])),
            "ci_width":   float(np.mean(ci_upper[mask] - ci_lower[mask])),
        })

    return results


def per_cycle_arrays(per_cycle: list[dict]):
    """
    Extract aligned numpy arrays from per_cycle list for plotting.

    Returns
    -------
    cycles     : np.ndarray  cycle indices [1, 2, …, K]
    rmse       : np.ndarray  RMSE per cycle [dimensionless]
    mae        : np.ndarray  MAE per cycle [dimensionless]
    max_error  : np.ndarray  peak |error| per cycle [dimensionless]
    mean_sigma : np.ndarray  mean 1σ uncertainty per cycle
    mean_NIS   : np.ndarray  mean NIS per cycle (expected ≈ 1)
    ci_width   : np.ndarray  mean 95% CI width per cycle (≈ 4σ)
    """
    if not per_cycle:
        empty = np.array([])
        return empty, empty, empty, empty, empty, empty, empty

    cycles     = np.array([d["cycle"]      for d in per_cycle])
    rmse       = np.array([d["rmse"]       for d in per_cycle])
    mae        = np.array([d["mae"]        for d in per_cycle])
    max_error  = np.array([d["max_error"]  for d in per_cycle])
    mean_sigma = np.array([d["mean_sigma"] for d in per_cycle])
    mean_NIS   = np.array([d["mean_NIS"]   for d in per_cycle])
    ci_width   = np.array([d["ci_width"]   for d in per_cycle])

    return cycles, rmse, mae, max_error, mean_sigma, mean_NIS, ci_width
