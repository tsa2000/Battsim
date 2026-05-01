from __future__ import annotations

"""
unscented_uq.py — Unscented Transform UQ
==========================================
يحسب انتشار عدم اليقين على SOC باستخدام Unscented Transform.

بدل N run من Monte Carlo → 7 نقاط sigma فقط (n=3 states).
السرعة: أسرع ~100x من Monte Carlo.

References
----------
[1] Julier & Uhlmann 1997 — A new extension of the Kalman filter
[2] Wan & van der Merwe 2000 — The Unscented Kalman Filter for NL estimation
[3] Plett 2004 — J. Power Sources 134 — EKF for Li-ion SOC
"""

import numpy as np
from src.chemistry import make_ocv, docv_dsoc
from src.utils     import safe_array, soc_to_percent


def compute_sigma_points(x: np.ndarray, P: np.ndarray, lam: float):
    """
    حساب نقاط Sigma (2n+1 نقطة).
    Ref: Wan & van der Merwe 2000, eq. (15)
    """
    n = len(x)
    try:
        S = np.linalg.cholesky((n + lam) * P)
    except np.linalg.LinAlgError:
        P  = P + 1e-8 * np.eye(n)
        S  = np.linalg.cholesky((n + lam) * P)

    sigmas      = np.zeros((2 * n + 1, n))
    sigmas[0]   = x
    for i in range(n):
        sigmas[i + 1]     = x + S[:, i]
        sigmas[n + i + 1] = x - S[:, i]
    return sigmas


def ut_weights(n: int, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
    """
    أوزان UT للمتوسط (Wm) والتباين (Wc).
    Ref: Julier & Uhlmann 1997
    """
    lam = alpha ** 2 * (n + kappa) - n
    Wm  = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    Wc  = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - alpha ** 2 + beta)
    return Wm, Wc, lam


def unscented_uq(
    chem      : dict,
    Q_nom     : float,
    log       : dict,
    noise_std : float,
    alpha     : float = 1e-3,
    beta      : float = 2.0,
    kappa     : float = 0.0,
):
    """
    Unscented Transform لتقدير انتشار عدم اليقين على SOC.

    State vector : x = [SOC, V_RC1, V_RC2]
    نقاط Sigma  : 2*3+1 = 7 نقاط فقط

    Returns
    -------
    soc_ut    : (N,)  SOC مُقدَّر من UT
    sigma_ut  : (N,)  ±1σ SOC (من P[0,0])
    ci_upper  : (N,)  SOC + 2σ
    ci_lower  : (N,)  SOC - 2σ
    p_soc_ut  : (N,)  P[0,0] — تباين SOC
    """
    n   = 3
    Wm, Wc, lam = ut_weights(n, alpha, beta, kappa)

    ocv_fn = make_ocv(chem)
    R0 = chem["R0"]; R1 = chem["R1"]; C1 = chem["C1"]
    R2 = chem["R2"]; C2 = chem["C2"]
    dt = 10.0
    e1 = np.exp(-dt / (R1 * C1 + 1e-12))
    e2 = np.exp(-dt / (R2 * C2 + 1e-12))

    t_arr  = safe_array(log["t"])
    I_arr  = safe_array(log["I_true"])
    V_arr  = safe_array(log["V_true"])
    T_arr  = safe_array(log.get("T_true", np.full_like(t_arr, 25.0)))
    N      = len(t_arr)

    # ── حالة أولية ───────────────────────────────────────────────────────────
    x  = np.array([1.0, 0.0, 0.0])
    P  = np.diag([0.01, 1e-6, 1e-6])
    Qd = np.diag([1e-5, 1e-6, 1e-6])
    R  = noise_std ** 2

    soc_ut   = np.empty(N)
    sigma_ut = np.empty(N)
    ci_upper = np.empty(N)
    ci_lower = np.empty(N)
    p_soc_ut = np.empty(N)

    for k in range(N):
        I_k = float(I_arr[k])
        V_k = float(V_arr[k])
        eta = 1.0 if I_k >= 0.0 else 0.99

        # ── نقاط Sigma ───────────────────────────────────────────────────────
        sigmas = compute_sigma_points(x, P, lam)

        # ── Propagate عبر معادلة الحالة ──────────────────────────────────────
        sigmas_pred = np.empty_like(sigmas)
        for i, sp in enumerate(sigmas):
            s, v1, v2 = sp
            sigmas_pred[i] = [
                np.clip(s - eta * I_k * dt / (Q_nom * 3600.0), 0.0, 1.0),
                v1 * e1 + I_k * R1 * (1.0 - e1),
                v2 * e2 + I_k * R2 * (1.0 - e2),
            ]

        # ── Predicted mean و covariance ───────────────────────────────────────
        x_pred = np.einsum("i,ij->j", Wm, sigmas_pred)
        P_pred = Qd.copy()
        for i in range(2 * n + 1):
            d       = sigmas_pred[i] - x_pred
            P_pred += Wc[i] * np.outer(d, d)

        # ── Predicted measurement ─────────────────────────────────────────────
        s_c    = float(np.clip(x_pred[0], 0.01, 0.99))
        y_pred = float(ocv_fn(s_c)) - x_pred[1] - x_pred[2] - I_k * R0

        # ── Innovation covariance Pyy + H·P·H (cross-covariance) ─────────────
        Pyy = R
        Pxy = np.zeros(n)
        for i in range(2 * n + 1):
            sp    = sigmas_pred[i]
            s_sp  = float(np.clip(sp[0], 0.01, 0.99))
            y_sp  = float(ocv_fn(s_sp)) - sp[1] - sp[2] - I_k * R0
            dy    = y_sp - y_pred
            Pyy  += Wc[i] * dy ** 2
            Pxy  += Wc[i] * (sp - x_pred) * dy

        # ── Kalman gain ───────────────────────────────────────────────────────
        K  = Pxy / (Pyy + 1e-12)

        # ── Measurement noise — adaptive R (simple innovation-based) ─────────
        nu = V_k - y_pred
        R  = 0.95 * R + 0.05 * nu ** 2
        R  = float(np.clip(R, 1e-8, 1e-1))

        # ── Update ────────────────────────────────────────────────────────────
        x = x_pred + K * nu
        P = P_pred - np.outer(K, K) * Pyy

        # Symmetrise و regularise P
        P = 0.5 * (P + P.T)
        P = np.clip(P, -1e-3, 1.0)

        # ── Store ─────────────────────────────────────────────────────────────
        soc_k       = float(np.clip(x[0], 0.0, 1.0))
        sig_k       = float(np.sqrt(max(P[0, 0], 0.0)))
        soc_ut[k]   = soc_k
        sigma_ut[k] = sig_k
        p_soc_ut[k] = float(P[0, 0])
        ci_upper[k] = float(np.clip(soc_k + 2.0 * sig_k, 0.0, 1.0))
        ci_lower[k] = float(np.clip(soc_k - 2.0 * sig_k, 0.0, 1.0))

    return soc_ut, sigma_ut, ci_upper, ci_lower, p_soc_ut
