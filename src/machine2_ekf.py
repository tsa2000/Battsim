import numpy as np
from .chemistry import make_ocv, docv_dsoc


class EKF:
    """
    2-RC Thevenin ECM + Adaptive Extended Kalman Filter

    State vector : x = [SOC, V_RC1, V_RC2]
    Observation  : V_terminal (noisy)

    Standards
    ---------
    Plett (2004)       J. Power Sources 134        — EKF for SOC
    Joseph Form        IEEE Trans. AES (1964)       — P stability
    Mehra (1972)       IEEE Trans. AC               — Adaptive Q
    Yue et al. (2026)  Mech. Sys. Signal Process.  — Adaptive R (eIAEKF)
    CW-AEKF (2024)     J. Energy Storage            — Temperature R
    Prada (2013)       J. Electrochem. Soc.        — Coulombic η
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

        # State  [SOC=1.0, V_RC1=0, V_RC2=0]
        # SOC0 = 1.0  — realistic start (no DFN cheating)
        self.x = np.array([[0.9], [0.0], [0.0]])

        # Covariance P
        self.P = np.diag([
            p0_scale,
            p0_scale * 0.1,
            p0_scale * 0.1,
        ])

        # Process noise Q — Adaptive (Mehra 1972)
        self.Q = np.diag([
            q_scale * 1e-6,
            q_scale * 1e-5,
            q_scale * 1e-5,
        ])
        self._alpha   = 0.97
        self._nu_win  = []
        self._WIN     = 50

        # Measurement noise R — Adaptive (eIAEKF 2026)
        self._R_base  = r_scale * noise_var
        self._beta    = 0.95
        self.R        = np.array([[self._R_base]])

        # NIS history
        self.NIS_hist = []

    # ── Adaptive Q  (Mehra 1972) ──────────────────────────────
    def _adapt_Q(self, Kk: np.ndarray, nu: float) -> None:
        self._nu_win.append(nu)
        if len(self._nu_win) > self._WIN:
            self._nu_win.pop(0)
        if len(self._nu_win) >= 10:
            var   = float(np.var(self._nu_win))
            dQ    = (1.0 - self._alpha) * var * (Kk @ Kk.T)
            self.Q = self._alpha * self.Q + dQ
            self.Q = np.clip(self.Q,
                             1e-12 * self.I3,
                             1e-3  * self.I3)

    # ── Adaptive R  (eIAEKF — Yue et al. 2026) ───────────────
    def _adapt_R(self, nu: float) -> None:
        self.R = np.array([[
            S_k = float(self.R[0, 0])
            self.R = np.array([[
                self._beta * float(self.R[0, 0])
                + (1.0 - self._beta) * max(nu**2 - S_k, 1e-8)
            ]])
        self.R = np.clip(self.R, 1e-8, 1e-2)

    # ── Temperature-aware R baseline (CW-AEKF 2024) ──────────
    def _temperature_R(self, T_c: float) -> None:
        factor         = 1.0 + 0.015 * max(0.0, 25.0 - T_c)
        self._R_cur    = self._R_base * factor

    # ═════════════════════════════════════════════════════════
    def step(
        self,
        v_meas:    float,
        current:   float,
        T_celsius: float = 25.0,
    ):
        """
        One EKF step.

        Parameters
        ----------
        v_meas    : float   noisy terminal voltage [V]
        current   : float   measured current [A]  (+ = discharge)
        T_celsius : float   cell temperature [°C]

        Returns
        -------
        v_est  : float   estimated terminal voltage [V]
        soc_e  : float   estimated SOC [0-1]
        tr_P   : float   trace(P) — state uncertainty
        P_soc  : float   P[0,0]   — SOC variance
        nu     : float   innovation [V]
        NIS    : float   Normalized Innovation Squared
        """
        dt = self.dt
        e1 = np.exp(-dt / (self.R1 * self.C1 + 1e-12))
        e2 = np.exp(-dt / (self.R2 * self.C2 + 1e-12))
        s, v1, v2 = self.x[:, 0]

        # ══ PREDICT ══════════════════════════════════════════

        # Coulombic efficiency η (Prada 2013)
        eta = 0.99 if current < 0.0 else 1.0

        s_p  = s  - eta * current * dt / (self.Q_nom * 3600.0)
        v1_p = v1 * e1 + current * self.R1 * (1.0 - e1)
        v2_p = v2 * e2 + current * self.R2 * (1.0 - e2)
        x_p  = np.array([[s_p], [v1_p], [v2_p]])

        A   = np.diag([1.0, e1, e2])
        P_p = A @ self.P @ A.T + self.Q

        # ══ UPDATE ═══════════════════════════════════════════

        s_c  = float(np.clip(s_p, 0.01, 0.99))
        dOCV = docv_dsoc(self.ocv, s_c)
        Ck   = np.array([[dOCV, -1.0, -1.0]])

        v_hat = (
            float(self.ocv(s_c))
            - v1_p - v2_p
            - current * self.R0
        )
        nu = v_meas - v_hat

        # Temperature-aware R then adaptive R
        self._temperature_R(T_celsius)
        self.R = np.array([[self._R_cur]])
        self._adapt_R(nu)

        S  = Ck @ P_p @ Ck.T + self.R
        Kk = P_p @ Ck.T / float(S[0, 0])

        # Joseph Form  — numerically stable P update
        IKC    = self.I3 - Kk @ Ck
        self.P = IKC @ P_p @ IKC.T + Kk @ self.R @ Kk.T

        # State update + SOC clamp
        self.x       = x_p + Kk * nu
        self.x[0, 0] = float(np.clip(self.x[0, 0], 0.0, 1.0))

        # Adaptive Q
        self._adapt_Q(Kk, float(nu))

        # NIS
        NIS = float(nu ** 2 / float(S[0, 0]))
        self.NIS_hist.append(NIS)

        # Output voltage estimate
        soc_e = float(self.x[0, 0])
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
        )


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
    Co-simulation: DFN ground truth → noisy channel → EKF observer.

    Parameters
    ----------
    t, V_true, I_true, soc_true, T_true : np.ndarray
        DFN outputs from machine1_dfn.run_dfn()
    Q_nom     : float   nominal capacity [Ah]
    chem      : dict    chemistry entry from chemistry.build_chem()
    noise_std : float   sensor noise std [V]
    p0_scale  : float   initial P diagonal scale
    q_scale   : float   Q matrix scale
    r_scale   : float   R matrix scale
    seed      : int     random seed for reproducibility

    Returns
    -------
    log : dict  — full simulation log
    """
    N   = len(t)
    rng = np.random.default_rng(seed)

    ekf = EKF(
        Q_nom     = Q_nom,
        chem      = chem,
        noise_var = noise_std ** 2,
        p0_scale  = p0_scale,
        q_scale   = q_scale,
        r_scale   = r_scale,
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
        "innov":    np.empty(N),
        "NIS":      np.empty(N),
    }

    for k in range(N):
        vm = V_true[k] + rng.normal(0.0, noise_std)
        out = ekf.step(vm, I_true[k], T_true[k])

        log["V_meas"][k]  = vm
        log["V_est"][k]   = out[0]
        log["soc_est"][k] = out[1]
        log["P_tr"][k]    = out[2]
        log["P_soc"][k]   = out[3]
        log["innov"][k]   = out[4]
        log["NIS"][k]     = out[5]

    return log

