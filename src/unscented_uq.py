from __future__ import annotations

"""
src/unscented_uq.py — Unscented Transform Uncertainty Quantification
=====================================================================
Unscented Transform (UT/UKF) على نموذج ECM 2-RC لحساب:
  • SOC المُقدَّر عبر UT (مستقل عن AEKF)
  • σ_SOC و 95% CI (±2σ)
  • P[0,0] variance

State: x = [SOC, V_RC1, V_RC2]  →  7 نقاط sigma

Ref: Julier & Uhlmann 1997, SPIE Proc. 3068
     Wan & van der Merwe 2000, ASSPCC
"""

import numpy as np
from src.chemistry import make_ocv
from src.utils     import safe_array


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sigma-point weights  (scaled UT: α=1e-3, β=2, κ=0)
# ─────────────────────────────────────────────────────────────────────────────

def _ut_weights(n: int, alpha: float = 1e-3,
                beta: float = 2.0, kappa: float = 0.0):
    """
    Compute UT mean/covariance weights for an n-dim state.

    Returns
    -------
    Wm  : (2n+1,) mean weights
    Wc  : (2n+1,) covariance weights
    lam : float   scaling parameter  λ = α²(n+κ) − n
    """
    lam   = alpha**2 * (n + kappa) - n
    Wm    = np.full(2*n + 1, 1.0 / (2*(n + lam)))
    Wc    = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)
    return Wm, Wc, lam


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sigma-point matrix
# ─────────────────────────────────────────────────────────────────────────────

def _sigma_points(x: np.ndarray, P: np.ndarray, lam: float) -> np.ndarray:
    """
    Build 2n+1 sigma points around mean x with covariance P.

    Falls back to regularised P if Cholesky fails (non-positive-definite).
    """
    n = len(x)
    try:
        S = np.linalg.cholesky((n + lam) * P)
    except np.linalg.LinAlgError:
        S = np.linalg.cholesky((n + lam) * P + 1e-8 * np.eye(n))

    sp      = np.zeros((2*n + 1, n))
    sp[0]   = x
    for i in range(n):
        sp[i + 1]     = x + S[:, i]
        sp[n + i + 1] = x - S[:, i]
    return sp


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main UT-UQ runner
# ─────────────────────────────────────────────────────────────────────────────

def unscented_uq(
    chem:       dict,
    Q_nom:      float,
    log:        dict,
    noise_std:  float,
    alpha:      float = 1e-3,
    beta:       float = 2.0,
    kappa:      float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a full UKF pass (predict + update) over the simulation log
    using the Unscented Transform — independent of the AEKF in machine2_ekf.

    The UT σ_SOC is compared against the EKF σ_SOC in Tab 2 of app.py to
    validate the Jacobian linearisation assumption of the AEKF.

    Parameters
    ----------
    chem      : dict   chemistry dict  (R0, R1, C1, R2, C2, ocv_lut, soc_lut)
    Q_nom     : float  nominal capacity [Ah]
    log       : dict   run_cosim() output  (keys: I_true, V_true, t, …)
    noise_std : float  measurement noise σ [V]
    alpha, beta, kappa : UT tuning parameters

    Returns
    -------
    soc_ut   : (N,)  UT mean SOC  [0, 1]
    sigma_ut : (N,)  UT σ_SOC     [0, 1]
    ci_upper : (N,)  SOC + 2σ clipped to [0, 1]
    ci_lower : (N,)  SOC − 2σ clipped to [0, 1]
    p_soc_ut : (N,)  SOC variance P[0,0]
    """
    n            = 3
    Wm, Wc, lam  = _ut_weights(n, alpha, beta, kappa)

    ocv_fn = make_ocv(chem)
    R0  = chem["R0"];  R1 = chem["R1"];  C1 = chem["C1"]
    R2  = chem["R2"];  C2 = chem["C2"]

    # Fixed timestep (same assumption as machine2_ekf.py)
    dt  = 10.0
    e1  = np.exp(-dt / (R1 * C1 + 1e-12))
    e2  = np.exp(-dt / (R2 * C2 + 1e-12))

    I_arr = safe_array(log["I_true"])
    V_arr = safe_array(log["V_true"])
    N     = len(I_arr)

    # Initial state and noise matrices
    x  = np.array([1.0, 0.0, 0.0])
    P  = np.diag([0.01, 1e-6, 1e-6])
    Qd = np.diag([1e-5, 1e-6, 1e-6])   # process noise
    R  = noise_std**2                   # measurement noise (adaptive)

    soc_ut   = np.empty(N)
    sigma_ut = np.empty(N)
    ci_upper = np.empty(N)
    ci_lower = np.empty(N)
    p_soc_ut = np.empty(N)

    for k in range(N):
        I_k = float(I_arr[k])
        V_k = float(V_arr[k])
        eta = 1.0 if I_k >= 0.0 else 0.99   # coulombic efficiency

        # ── Predict ──────────────────────────────────────────────────────
        sp      = _sigma_points(x, P, lam)
        sp_pred = np.array([
            [np.clip(s - eta * I_k * dt / (Q_nom * 3600), 0.0, 1.0),
             v1 * e1 + I_k * R1 * (1 - e1),
             v2 * e2 + I_k * R2 * (1 - e2)]
            for s, v1, v2 in sp
        ])
        x_p = np.einsum("i,ij->j", Wm, sp_pred)
        P_p = Qd + sum(
            Wc[i] * np.outer(sp_pred[i] - x_p, sp_pred[i] - x_p)
            for i in range(2*n + 1)
        )

        # ── Update ───────────────────────────────────────────────────────
        y_sigma = np.array([
            float(ocv_fn(float(np.clip(sp_pred[i][0], 0.01, 0.99))))
            - sp_pred[i][1] - sp_pred[i][2] - I_k * R0
            for i in range(2*n + 1)
        ])
        y_p  = float(np.einsum("i,i->", Wm, y_sigma))
        Pyy  = R + float(np.einsum("i,i->", Wc, (y_sigma - y_p)**2))
        Pxy  = np.einsum("i,ij->j", Wc,
                         (sp_pred - x_p) * (y_sigma - y_p)[:, None])
        K    = Pxy / (Pyy + 1e-12)
        nu   = V_k - y_p
        x    = x_p + K * nu
        P    = P_p - np.outer(K, K) * Pyy
        P    = 0.5 * (P + P.T)               # symmetry enforcement

        # Adaptive measurement noise update (innovation-based)
        R = float(np.clip(0.95 * R + 0.05 * nu**2, 1e-8, 1e-1))

        # ── Store ────────────────────────────────────────────────────────
        s_k         = float(np.clip(x[0], 0.0, 1.0))
        sig_k       = float(np.sqrt(max(P[0, 0], 0.0)))
        soc_ut[k]   = s_k
        sigma_ut[k] = sig_k
        p_soc_ut[k] = float(P[0, 0])
        ci_upper[k] = float(np.clip(s_k + 2*sig_k, 0.0, 1.0))
        ci_lower[k] = float(np.clip(s_k - 2*sig_k, 0.0, 1.0))

    return soc_ut, sigma_ut, ci_upper, ci_lower, p_soc_ut
