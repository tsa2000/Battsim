"""
mc_uq.py — Monte Carlo Uncertainty Quantification
===================================================
يُطبّق انتشار عدم اليقين عبر:
  1. MC-EKF  : تشغيل N نسخة من EKF بـ seeds مختلفة → توزيع SOC
  2. ANEES   : Average Normalised Estimation Error Squared عبر الدورات
  3. PCRLB   : Posterior Cramér-Rao Lower Bound (حدّ نظري أدنى لـ P_SOC)

References
----------
[1] Bar-Shalom et al. 2001 — Estimation with Applications to Tracking (ch.10)
[2] Tichavsky et al. 1998 — IEEE Trans. Signal Process. 46(5) — PCRLB recursion
[3] Xiong et al. 2013 — J. Power Sources 243 — AEKF for Li-ion batteries
[4] Plett 2004 — J. Power Sources 134 — EKF for SOC estimation
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .machine2_ekf import run_cosim
from .utils import safe_array, soc_to_percent, rmse, mae


# ─────────────────────────────────────────────────────────────────────────────
# Data Containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MCResult:
    """
    Results of a Monte Carlo UQ run.

    Attributes
    ----------
    soc_matrix  : (N_runs × T) — all SOC trajectories
    soc_mean    : (T,)         — ensemble mean
    soc_std     : (T,)         — ensemble std  (empirical σ)
    ci_95_upper : (T,)         — 97.5th percentile
    ci_95_lower : (T,)         — 2.5th  percentile
    ci_68_upper : (T,)         — 84.1st percentile (±1σ band)
    ci_68_lower : (T,)         — 15.9th percentile
    rmse_ensemble : (N_runs,)  — per-run RMSE vs DFN truth [%]
    nis_matrix  : (N_runs × T) — NIS per run per step
    anees       : float        — Average NIS over all runs and steps
    converge_steps : int       — time steps until |mean - true| < 2 %
    t           : (T,)         — time axis [s]
    soc_true    : (T,)         — DFN ground truth SOC
    seeds       : list[int]    — random seeds used
    """
    soc_matrix    : np.ndarray
    soc_mean      : np.ndarray
    soc_std       : np.ndarray
    ci_95_upper   : np.ndarray
    ci_95_lower   : np.ndarray
    ci_68_upper   : np.ndarray
    ci_68_lower   : np.ndarray
    rmse_ensemble : np.ndarray
    nis_matrix    : np.ndarray
    anees         : float
    converge_steps: int
    t             : np.ndarray
    soc_true      : np.ndarray
    seeds         : list


@dataclass
class PCRLBResult:
    """
    Posterior Cramér-Rao Lower Bound on SOC estimation.

    J_k   : Fisher Information Matrix (3×3) at each step
    pcrlb : lower bound on Var(SOC_k) at each step
    t     : time axis [s]

    Reference: Tichavsky 1998, eq. (3)–(5)
    """
    J_k    : np.ndarray   # (T × 3 × 3)
    pcrlb  : np.ndarray   # (T,) — PCRLB on SOC variance
    t      : np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# 1. Monte Carlo EKF Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_mc_ekf(
    dfn_log   : dict,
    Q_nom     : float,
    chem      : dict,
    noise_std : float,
    n_runs    : int = 200,
    p0_scale  : float = 1e-4,
    q_scale   : float = 1.0,
    r_scale   : float = 1.0,
    max_workers: int = 4,
) -> MCResult:
    """
    Run EKF N times with independent noise realisations (different seeds).

    Each run uses a fresh EKF instance → independent noise trajectory →
    the ensemble spread gives the **empirical** SOC uncertainty distribution,
    which is compared to the **analytical** ±2σ from the filter's own P matrix.

    If empirical std ≈ filter σ → filter is well-calibrated (consistent).
    If empirical std > filter σ → filter is over-confident.

    Parameters
    ----------
    dfn_log    : dict   output of run_cosim() or containing t,V,I,soc,T arrays
    Q_nom      : float  nominal capacity [Ah]
    chem       : dict   chemistry dict from build_chem()
    noise_std  : float  voltage sensor noise std [V]
    n_runs     : int    number of MC runs (200 default, 500 for publication)
    max_workers: int    parallel threads (set to 1 for deterministic debug)

    Returns
    -------
    MCResult
    """
    t        = safe_array(dfn_log["t"])
    V_true   = safe_array(dfn_log["V_true"])
    I_true   = safe_array(dfn_log["I_true"])
    soc_true = safe_array(dfn_log["soc_true"])
    T_true   = safe_array(dfn_log["T_true"])
    N        = len(t)

    seeds = list(range(n_runs))

    soc_matrix  = np.empty((n_runs, N))
    nis_matrix  = np.empty((n_runs, N))
    rmse_arr    = np.empty(n_runs)

    def _single_run(seed: int) -> tuple[int, np.ndarray, np.ndarray, float]:
        log = run_cosim(
            t=t, V_true=V_true, I_true=I_true,
            soc_true=soc_true, T_true=T_true,
            Q_nom=Q_nom, chem=chem,
            noise_std=noise_std,
            p0_scale=p0_scale, q_scale=q_scale, r_scale=r_scale,
            seed=seed,
        )
        soc_e = safe_array(log["soc_est"])
        nis_e = safe_array(log["NIS"])
        r_val = rmse(
            soc_to_percent(soc_true),
            soc_to_percent(soc_e),
        )
        return seed, soc_e, nis_e, r_val

    # Parallel execution — each run is independent (no shared state)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_single_run, s): s for s in seeds}
        for fut in as_completed(futures):
            seed, soc_e, nis_e, r_val = fut.result()
            idx = seeds.index(seed)
            soc_matrix[idx]  = soc_e
            nis_matrix[idx]  = nis_e
            rmse_arr[idx]    = r_val

    # Ensemble statistics
    soc_mean    = np.mean(soc_matrix, axis=0)
    soc_std     = np.std(soc_matrix,  axis=0)
    ci_95_upper = np.percentile(soc_matrix, 97.5, axis=0)
    ci_95_lower = np.percentile(soc_matrix,  2.5, axis=0)
    ci_68_upper = np.percentile(soc_matrix, 84.1, axis=0)
    ci_68_lower = np.percentile(soc_matrix, 15.9, axis=0)

    # ANEES (Bar-Shalom 2001, eq. 5.4.2-4)
    # ANEES = (1/N_runs) * mean_time(NIS)
    # Expected: ANEES ≈ 1.0 for consistent filter
    anees = float(np.mean(nis_matrix))

    # Convergence: first step where |ensemble_mean - true| < 2%
    err_pct = np.abs(soc_to_percent(soc_mean) - soc_to_percent(soc_true))
    conv_mask = np.where(err_pct < 2.0)[0]
    converge_steps = int(conv_mask[0]) if len(conv_mask) > 0 else N

    return MCResult(
        soc_matrix    = soc_matrix,
        soc_mean      = soc_mean,
        soc_std       = soc_std,
        ci_95_upper   = ci_95_upper,
        ci_95_lower   = ci_95_lower,
        ci_68_upper   = ci_68_upper,
        ci_68_lower   = ci_68_lower,
        rmse_ensemble = rmse_arr,
        nis_matrix    = nis_matrix,
        anees         = anees,
        converge_steps= converge_steps,
        t             = t,
        soc_true      = soc_true,
        seeds         = seeds,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Posterior Cramér-Rao Lower Bound (PCRLB)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pcrlb(
    log         : dict,
    chem        : dict,
    Q_nom       : float,
    noise_var   : float,
    ocv_fn      : Callable,
) -> PCRLBResult:
    """
    Compute PCRLB on SOC estimation via Fisher Information Matrix recursion.

    Recursion (Tichavsky 1998, eq. 3):
        J_{k+1} = (Q_d + A_k · J_k⁻¹ · A_kᵀ)⁻¹ + H_kᵀ · R⁻¹ · H_k

    Where:
        A_k  : state transition Jacobian (3×3)
        H_k  : observation Jacobian (1×3) = [dOCV/dSOC, -1, -1]
        Q_d  : discrete-time process noise (3×3)
        R    : measurement noise variance (scalar)

    PCRLB on SOC: var(SOC_k) ≥ [J_k⁻¹]_{0,0}

    Parameters
    ----------
    log       : dict   output from run_cosim()
    chem      : dict   chemistry dict (contains ECM params)
    Q_nom     : float  [Ah]
    noise_var : float  σ² of voltage sensor [V²]
    ocv_fn    : callable  OCV(SOC) interpolant

    Returns
    -------
    PCRLBResult
    """
    from .chemistry import docv_dsoc

    t   = safe_array(log["t"])
    soc = safe_array(log["soc_true"])
    I   = safe_array(log["I_true"])
    N   = len(t)
    dt  = float(np.median(np.diff(t)))

    R0 = chem["R0"]; R1 = chem["R1"]; C1 = chem["C1"]
    R2 = chem["R2"]; C2 = chem["C2"]

    # ZOH constants
    e1 = np.exp(-dt / (R1 * C1 + 1e-12))
    e2 = np.exp(-dt / (R2 * C2 + 1e-12))

    # State transition A (same as EKF)
    A = np.diag([1.0, e1, e2])

    # Process noise Q (match EKF defaults)
    Qd = np.diag([1e-5, 1e-6, 1e-6])

    # Initial J (large → little prior info)
    J = np.diag([1.0 / 0.01, 1.0 / 1e-4, 1.0 / 1e-4])

    J_all   = np.empty((N, 3, 3))
    pcrlb   = np.empty(N)

    for k in range(N):
        s_k = float(np.clip(soc[k], 0.01, 0.99))
        dOCV = docv_dsoc(ocv_fn, s_k)
        Hk   = np.array([[dOCV, -1.0, -1.0]])

        # Measurement contribution: H^T R^{-1} H
        meas_info = (1.0 / noise_var) * (Hk.T @ Hk)

        # Prediction step: invert (Q_d + A J^{-1} A^T)
        try:
            J_inv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            J_inv = np.eye(3) * 1e-6

        pred_cov = Qd + A @ J_inv @ A.T
        try:
            J_pred = np.linalg.inv(pred_cov)
        except np.linalg.LinAlgError:
            J_pred = np.eye(3) * 1.0

        # Update: J_{k+1} = J_pred + H^T R^{-1} H
        J = J_pred + meas_info

        J_all[k] = J

        # PCRLB: [J^{-1}]_{0,0}
        try:
            pcrlb[k] = np.linalg.inv(J)[0, 0]
        except np.linalg.LinAlgError:
            pcrlb[k] = pcrlb[k - 1] if k > 0 else 1e-4

    return PCRLBResult(J_k=J_all, pcrlb=pcrlb, t=t)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANEES per-cycle (filter health over ageing)
# ─────────────────────────────────────────────────────────────────────────────

def anees_per_cycle(mc_result: MCResult, n_cycles: int) -> list[dict]:
    """
    Compute ANEES per detected cycle segment from MC runs.

    A well-calibrated filter: ANEES ≈ 1.0
    Over-confident: ANEES > 1.0 (uncertainty under-reported)
    Under-confident: ANEES < 1.0 (uncertainty over-reported)

    Returns list of dicts: {cycle, anees, mean_std_soc_pct}
    """
    from .utils import detect_cycles

    soc_true = mc_result.soc_true
    cycles   = detect_cycles(soc_true)
    if not cycles:
        cycles = [(0, len(soc_true) - 1)]
    cycles = cycles[:n_cycles]

    results = []
    for c_num, (s, e) in enumerate(cycles, start=1):
        sl   = slice(s, e + 1)
        nis_seg = mc_result.nis_matrix[:, sl]
        std_seg = mc_result.soc_std[sl]
        results.append({
            "cycle"         : c_num,
            "anees"         : float(np.mean(nis_seg)),
            "mean_std_soc"  : float(np.mean(soc_to_percent(std_seg))),
        })

    return results
