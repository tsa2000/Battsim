import numpy as np
from .chemistry import make_ocv, docv_dsoc


class EKF:
    """
    2-RC Thevenin ECM + Adaptive Extended Kalman Filter (AEKF).

    State vector : x = [SOC, V_RC1, V_RC2]ᵀ
    Observation  : y = V_terminal (noisy)

    Discrete-time state equations (ZOH discretisation):
        SOC_{k+1} = SOC_k − η · I_k · dt / (Q_nom · 3600)
        V1_{k+1}  = e1 · V1_k + R1 · (1 − e1) · I_k
        V2_{k+1}  = e2 · V2_k + R2 · (1 − e2) · I_k
        y_k       = OCV(SOC_k) − V1_k − V2_k − R0 · I_k + w_k

    Standards / References
    ----------------------
    [1] Plett (2004) J. Power Sources 134, 252-261   — EKF for Li-ion SOC
    [2] Simon (2006) Optimal State Estimation, Wiley  — Joseph form (ch.5)
    [3] Mehra (1972) IEEE Trans. Autom. Control 17    — Adaptive Q (covariance matching)
    [4] Sage & Husa (1969) Proc. 7th IEEE Symp.      — Adaptive Q/R (original)
    [5] Yue et al. (2026) Mech. Syst. Signal Process. — Adaptive R (eIAEKF)
    [6] Xiong et al. (2013) J. Power Sources 243     — AEKF for multiple Li chemistries
    [7] Prada et al. (2013) J. Electrochem. Soc. 160 — Coulombic efficiency η
    [8] Huria et al. (2012) IEEE IEVC                — 2-RC ECM parameter identification
    """

    def __init__(
        self,
        Q_nom:     float,
        chem:      dict,
        noise_var: float,
        p0_scale:  float = 1e-4,
        q_scale:   float = 1.0,
        r_scale:   float = 1.0,
    ):
        self.dt    = 10.0
        self.ocv   = make_ocv(chem)
        self.R0    = chem["R0"]
        self.R1    = chem["R1"]
        self.C1    = chem["C1"]
        self.R2    = chem["R2"]
        self.C2    = chem["C2"]
        self.Q_nom = Q_nom
        self.I3    = np.eye(3)

        # ── Initial state: SOC = 1.0, RC voltages = 0  ───────────────────────
        # Plett 2004: start at known SOC if possible; 1.0 = fully charged
        self.x = np.array([[1.0], [0.0], [0.0]])

        # ── Initial covariance P ──────────────────────────────────────────────
        # P_SOC: large initial uncertainty → allows fast convergence (Plett 2004)
        # P_RC:  small (RC voltages start near 0 with high confidence)
        self.P = np.diag([
            max(p0_scale, 1e-2),   # SOC variance [dimensionless²]
            p0_scale * 0.01,       # V_RC1 variance [V²]
            p0_scale * 0.01,       # V_RC2 variance [V²]
        ])

        # ── Process noise Q ───────────────────────────────────────────────────
        # Initial values follow Plett 2004 Table I recommendation.
        # Q is adapted online via Mehra (1972) covariance matching.
        self.Q = np.diag([
            q_scale * 1e-5,   # SOC process noise
            q_scale * 1e-6,   # V_RC1 process noise
            q_scale * 1e-6,   # V_RC2 process noise
        ])

        # ── Measurement noise R ───────────────────────────────────────────────
        # Initialised from sensor noise_var; adapted online (eIAEKF, Yue 2026)
        self._R_base  = r_scale * noise_var
        self._R_adapt = self._R_base      # running adaptive estimate
        self._beta    = 0.95              # forgetting factor for R adaptation
        self.R        = np.array([[self._R_base]])

        # ── Adaptive Q (Mehra 1972) window ────────────────────────────────────
        self._alpha   = 0.97    # exponential forgetting factor
        self._nu_win  = []      # innovation window
        self._Hk_win  = []      # H_k window (needed for full Mehra eq.)
        self._Pp_win  = []      # P_predict window
        self._WIN     = 50      # window length

        # ── NIS history ───────────────────────────────────────────────────────
        self.NIS_hist = []

    # ─────────────────────────────────────────────────────────────────────────
    # Adaptive Q — Mehra (1972) covariance matching, full equation
    #   C_hat = (1/N)·Σ νₖνₖᵀ − Hₖ·P⁻ₖ·Hₖᵀ
    #   Q_hat = Kₖ·C_hat·Kₖᵀ
    # Reference: Mehra 1972, IEEE Trans. Autom. Control 17(5), 693-702
    # ─────────────────────────────────────────────────────────────────────────
    def _adapt_Q(self, Kk: np.ndarray, nu: float,
                  Hk: np.ndarray, Pp: np.ndarray) -> None:
        self._nu_win.append(nu)
        self._Hk_win.append(Hk.copy())
        self._Pp_win.append(Pp.copy())
        if len(self._nu_win) > self._WIN:
            self._nu_win.pop(0)
            self._Hk_win.pop(0)
            self._Pp_win.pop(0)

        N = len(self._nu_win)
        if N >= 10:
            # Innovation autocorrelation estimate
            nu_arr  = np.array(self._nu_win)
            C_nu    = float(np.mean(nu_arr ** 2))   # scalar: E[νν]

            # Subtract measurement noise contribution: H·P⁻·Hᵀ (avg over window)
            HPH_avg = float(np.mean([
                float(self._Hk_win[i] @ self._Pp_win[i] @ self._Hk_win[i].T)
                for i in range(N)
            ]))
            C_hat = max(C_nu - HPH_avg, 1e-12)   # scalar innovation covariance

            # Q update: Q = α·Q + (1−α)·K·C_hat·Kᵀ
            dQ = (1.0 - self._alpha) * C_hat * (Kk @ Kk.T)
            self.Q = self._alpha * self.Q + dQ
            self.Q = np.clip(self.Q, 1e-14 * self.I3, 1e-3 * self.I3)

    # ─────────────────────────────────────────────────────────────────────────
    # Adaptive R — eIAEKF (Yue et al. 2026)
    #   R_{k+1} = β·R_k + (1−β)·νₖ²
    # Applied first, then temperature-scaling applied as a multiplier.
    # Reference: Yue et al. 2026, Mech. Syst. Signal Process. (eIAEKF)
    # ─────────────────────────────────────────────────────────────────────────
    def _adapt_R(self, nu: float, T_celsius: float = 25.0) -> None:
        # Step 1: exponential forgetting adaptive R
        self._R_adapt = (
            self._beta * self._R_adapt
            + (1.0 - self._beta) * nu ** 2
        )
        self._R_adapt = float(np.clip(self._R_adapt, 1e-8, 1e-1))

        # Step 2: temperature scaling as multiplicative factor
        # Sensor noise increases at low temperature (CW-AEKF 2024)
        T_factor = 1.0 + 0.015 * max(0.0, 25.0 - T_celsius)

        self.R = np.array([[self._R_adapt * T_factor]])

    # ═════════════════════════════════════════════════════════════════════════
    def step(
        self,
        v_meas:    float,
        current:   float,
        T_celsius: float = 25.0,
    ):
        """
        One EKF correction step (predict → update).

        Parameters
        ----------
        v_meas    : float   noisy terminal voltage [V]
        current   : float   current [A]  (+ = discharge, − = charge)
        T_celsius : float   cell temperature [°C]

        Returns
        -------
        v_est  : float   estimated terminal voltage [V]
        soc_e  : float   estimated SOC [0–1]
        tr_P   : float   trace(P)  — total state uncertainty proxy
        P_soc  : float   P[0,0]   — SOC variance [dimensionless²]
        nu     : float   innovation ν = y − ŷ [V]
        NIS    : float   Normalised Innovation Squared (χ²(1) ≈ 1 if calibrated)
        sigma_soc : float  ±1σ SOC uncertainty [dimensionless]
        """
        dt = self.dt

        # ZOH RC time constants (Huria 2012)
        e1 = np.exp(-dt / (self.R1 * self.C1 + 1e-12))
        e2 = np.exp(-dt / (self.R2 * self.C2 + 1e-12))
        s, v1, v2 = self.x[:, 0]

        # ── PREDICT ──────────────────────────────────────────────────────────

        # Coulombic efficiency η (Prada 2013, Table II):
        #   η = 1.0 for discharge (positive current)
        #   η = 0.99 for charge (negative current) — accounts for side reactions
        eta = 1.0 if current >= 0.0 else 0.99

        s_p  = s  - eta * current * dt / (self.Q_nom * 3600.0)
        s_p  = float(np.clip(s_p, 0.0, 1.0))    # physical constraint on state
        v1_p = v1 * e1 + current * self.R1 * (1.0 - e1)
        v2_p = v2 * e2 + current * self.R2 * (1.0 - e2)
        x_p  = np.array([[s_p], [v1_p], [v2_p]])

        # State transition Jacobian A (Plett 2004, eq. 13)
        A = np.diag([1.0, e1, e2])

        # Predicted covariance (Plett 2004, eq. 14)
        P_p = A @ self.P @ A.T + self.Q

        # ── UPDATE ───────────────────────────────────────────────────────────

        s_c  = float(np.clip(s_p, 0.01, 0.99))
        dOCV = docv_dsoc(self.ocv, s_c)

        # Observation Jacobian H_k (Plett 2004, eq. 17)
        Hk = np.array([[dOCV, -1.0, -1.0]])

        # Predicted voltage
        v_hat = (
            float(self.ocv(s_c))
            - v1_p - v2_p
            - current * self.R0
        )
        nu = v_meas - v_hat   # innovation

        # Adaptive R (order matters: adapt first, then use in S)
        self._adapt_R(nu, T_celsius)

        # Innovation covariance S
        S  = float((Hk @ P_p @ Hk.T + self.R)[0, 0])

        # Kalman gain K_k (Plett 2004, eq. 18)
        Kk = P_p @ Hk.T / S

        # State update (Plett 2004, eq. 19)
        # NOTE: SOC clipping is applied to x_p (predict), NOT to post-update x.
        # Clipping the updated state would inject non-linearity into P and
        # corrupt the Joseph form. (Plett 2004, Section 4.3)
        self.x = x_p + Kk * nu

        # Joseph form P update — numerically stable (Simon 2006, eq. 5.55)
        IKH    = self.I3 - Kk @ Hk
        self.P = IKH @ P_p @ IKH.T + Kk @ self.R @ Kk.T

        # Adaptive Q — full Mehra (1972) with H·P⁻·Hᵀ correction
        self._adapt_Q(Kk, float(nu), Hk, P_p)

        # NIS — χ²(1) distributed if filter is consistent
        # Expected value: E[NIS] ≈ 1.0  (95% CI: [0.05, 5.02] for scalar obs)
        NIS = float(nu ** 2 / S)
        self.NIS_hist.append(NIS)

        # ── Outputs ──────────────────────────────────────────────────────────
        # Clip SOC only for output — internal state x[0] unconstrained
        soc_e     = float(np.clip(self.x[0, 0], 0.0, 1.0))
        sigma_soc = float(np.sqrt(max(self.P[0, 0], 0.0)))

        v_est = (
            float(self.ocv(np.clip(soc_e, 0.01, 0.99)))
            - float(self.x[1, 0])
            - float(self.x[2, 0])
            - current * self.R0
        )

        return (
            v_est,
            soc_e,
            float(np.trace(self.P)),
            float(self.P[0, 0]),
            float(nu),
            NIS,
            sigma_soc,
        )


# ══════════════════════════════════════════════════════════════════════════════

def run_cosim(
    t:         np.ndarray,
    V_true:    np.ndarray,
    I_true:    np.ndarray,
    soc_true:  np.ndarray,
    T_true:    np.ndarray,
    Q_nom:     float,
    chem:      dict,
    noise_std: float,
    p0_scale:  float = 1e-4,
    q_scale:   float = 1.0,
    r_scale:   float = 1.0,
    seed:      int   = 42,
):
    """
    Co-simulation: DFN ground truth (Machine 1) → noisy channel → EKF (Machine 2).

    Uncertainty quantification outputs per timestep:
      P_soc     — SOC error variance P[0,0] [dimensionless²]
      sigma_soc — 1σ SOC uncertainty = √P[0,0]
      ci_upper  — SOC + 2σ (95% confidence upper bound)
      ci_lower  — SOC − 2σ (95% confidence lower bound)
      NIS       — Normalised Innovation Squared (filter consistency check)
      P_tr      — trace(P) total state uncertainty

    Reference for UQ metrics:
      Bar-Shalom et al. 2001, Estimation with Applications to Tracking
      and Navigation — Chapters 5, 10.
    """
    N   = len(t)
    rng = np.random.default_rng(seed)

    ekf = EKF(
        Q_nom=Q_nom, chem=chem, noise_var=noise_std ** 2,
        p0_scale=p0_scale, q_scale=q_scale, r_scale=r_scale,
    )

    log = {
        "t":        t,
        "V_true":   V_true,
        "I_true":   I_true,
        "soc_true": soc_true,
        "T_true":   T_true,
        "V_meas":   np.empty(N),
        "V_est":    np.empty(N),
        "soc_est":  np.empty(N),
        "P_tr":     np.empty(N),
        "P_soc":    np.empty(N),
        "sigma_soc": np.empty(N),
        "ci_upper": np.empty(N),
        "ci_lower": np.empty(N),
        "innov":    np.empty(N),
        "NIS":      np.empty(N),
    }

    for k in range(N):
        # Additive white Gaussian measurement noise (AWGN)
        vm  = V_true[k] + rng.normal(0.0, noise_std)
        out = ekf.step(vm, I_true[k], T_true[k])

        v_est, soc_e, tr_P, p_soc, nu, nis, sigma = out

        log["V_meas"][k]   = vm
        log["V_est"][k]    = v_est
        log["soc_est"][k]  = soc_e
        log["P_tr"][k]     = tr_P
        log["P_soc"][k]    = p_soc
        log["sigma_soc"][k] = sigma
        log["ci_upper"][k] = np.clip(soc_e + 2.0 * sigma, 0.0, 1.0)
        log["ci_lower"][k] = np.clip(soc_e - 2.0 * sigma, 0.0, 1.0)
        log["innov"][k]    = nu
        log["NIS"][k]      = nis

    return log