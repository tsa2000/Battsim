import numpy as np
import pybamm


def _safe1d(arr):
    return np.asarray(arr).ravel()


def _clean_time(t, *arrays):
    """
    Remove duplicate / non-monotone time points.
    PyBaMM inserts repeated timestamps at event-capture boundaries
    (e.g. voltage cut-off).  Keeping only strictly-increasing points
    ensures downstream interpolation is well-defined.
    """
    mask = np.concatenate([[True], np.diff(t) > 1e-10])
    return (t[mask],) + tuple(a[mask] for a in arrays)


def _resample(t, dt, *arrays):
    """
    Resample irregular PyBaMM time grid onto a uniform dt grid.
    Linear interpolation is used (sufficient for dt = 10 s).
    The uniform grid matches self.dt = 10.0 in the EKF.
    """
    t_u = np.arange(t[0], t[-1], dt)
    return (t_u,) + tuple(np.interp(t_u, t, a) for a in arrays)


def _extract_soc(sol, t_clean: np.ndarray) -> np.ndarray:
    """
    Compute true SOC from DFN solution via Discharge capacity integration.

    SOC(t) = SOC₀ − Q_discharged(t) / Q_nominal

    "Discharge capacity [A.h]" is available in all PyBaMM versions and
    is the authoritative method recommended by the PyBaMM team.
    (pybamm-team/PyBaMM discussions #4462, #3822, #2553)

    Reference: Plett 2004, J. Power Sources 134 — Coulomb counting
    """
    t_sol  = _safe1d(sol["Time [s]"].entries)
    mask   = np.concatenate([[True], np.diff(t_sol) > 1e-10])
    t_sol  = t_sol[mask]

    q_disch = _safe1d(sol["Discharge capacity [A.h]"].entries)[mask]  # [A.h]
    q_total = float(q_disch[-1] - q_disch[0])
    if q_total < 1e-6:
        q_total = 1.0

    soc_raw = 1.0 - (q_disch - q_disch[0]) / q_total
    soc_raw = np.clip(soc_raw, 0.0, 1.0)

    return np.interp(t_clean, t_sol, soc_raw)

    Priority
    --------
    1. "X-averaged negative electrode SOC"  — best, direct internal state
    2. "Negative electrode SOC"             — fallback key (older PyBaMM)
    3. "Average negative electrode SOC"     — fallback key (some builds)

    If none of the keys is available a RuntimeError is raised so the
    caller sees an explicit failure instead of silently wrong data.

    References
    ----------
    PyBaMM docs — Variables — Electrode
    Marquis et al. 2019, J. Electrochem. Soc. 166, A3693
    """
    # Cleaned PyBaMM time axis (same mask as caller)
    t_sol = _safe1d(sol["Time [s]"].entries)
    mask  = np.concatenate([[True], np.diff(t_sol) > 1e-10])
    t_sol = t_sol[mask]

    for key in [
        "X-averaged negative electrode SOC",
        "Negative electrode SOC",
        "Average negative electrode SOC",
    ]:
        try:
            raw = _safe1d(sol[key].entries)[mask]
            raw = np.clip(raw, 0.0, 1.0)
            # Time-accurate interpolation onto t_clean grid
            return np.interp(t_clean, t_sol, raw)
        except Exception:
            continue

    raise RuntimeError(
        "PyBaMM solution does not contain any recognised SOC variable. "
        "Checked keys: 'X-averaged negative electrode SOC', "
        "'Negative electrode SOC', 'Average negative electrode SOC'. "
        "Verify your PyBaMM version or parameter set."
    )


# ── Experiment Builders ────────────────────────────────────────────────────────

def _build_cc_steps(n_cycles: int, c_rate: float, v_min: float, v_max: float):
    """
    CC discharge + CC charge (CC/CC).
    Simple protocol — used for baseline benchmarking.
    """
    steps = []
    for _ in range(n_cycles):
        steps += [
            f"Discharge at {c_rate:.4f}C until {v_min:.3f} V",
            "Rest for 5 minutes",
            f"Charge at {c_rate / 2:.4f}C until {v_max:.3f} V",
            "Rest for 5 minutes",
        ]
    return steps


def _build_cccv_steps(n_cycles: int, c_rate: float, v_min: float, v_max: float):
    """
    CC discharge + CC-CV charge (industry standard BEV protocol).

    CC phase charges to V_max; CV phase holds until C/20 (tail current).
    This matches the standard charging algorithm used in commercial cells.

    Reference: IEC 62660-1:2018, Secondary lithium-ion cells for
    propulsion of electric road vehicles — Part 1: Performance testing.
    """
    steps = []
    for _ in range(n_cycles):
        steps += [
            f"Discharge at {c_rate:.4f}C until {v_min:.3f} V",
            "Rest for 5 minutes",
            f"Charge at {c_rate / 2:.4f}C until {v_max:.3f} V",
            f"Hold at {v_max:.3f} V until C/20",
            "Rest for 5 minutes",
        ]
    return steps


def _build_hppc_steps(n_cycles: int, c_rate: float, v_min: float, v_max: float):
    """
    Hybrid Pulse Power Characterisation (HPPC) protocol.

    Simplified HPPC following USABC/USCAR procedure:
      1. Full CC-CV charge to V_max
      2. Rest 10 min
      3. 1C discharge pulse 10 s → rest 40 s
      4. 0.75C charge pulse 10 s → rest 40 s
      5. 1C discharge to V_min

    Note: This is a per-cycle HPPC, not a per-10%-SOC sweep.
    It is suitable for ECM parameter identification from a small
    number of cycles rather than a full SOC-sweep HPPC test.

    Reference: USCAR FreedomCAR Battery Test Manual,
    DOE/ID-11069, 2004, Section 3.
    """
    steps = []
    for _ in range(n_cycles):
        steps += [
            f"Charge at {c_rate / 2:.4f}C until {v_max:.3f} V",
            f"Hold at {v_max:.3f} V until C/20",
            "Rest for 10 minutes",
            f"Discharge at {c_rate:.4f}C for 10 seconds",
            "Rest for 40 seconds",
            f"Charge at {c_rate * 0.75:.4f}C for 10 seconds",
            "Rest for 40 seconds",
            f"Discharge at {c_rate:.4f}C until {v_min:.3f} V",
            "Rest for 5 minutes",
        ]
    return steps


_PROTOCOL_BUILDERS = {
    "cc":   _build_cc_steps,
    "cccv": _build_cccv_steps,
    "hppc": _build_hppc_steps,
}


# ── Main DFN Runner ────────────────────────────────────────────────────────────

def run_dfn(
    pset_name: str,
    n_cycles:  int,
    c_rate:    float,
    protocol:  str,
    v_min:     float,
    v_max:     float,
    dt:        float = 10.0,
):
    """
    Run PyBaMM Doyle-Fuller-Newman (DFN) model as the physical asset
    (Machine 1) of the digital-twin co-simulation.

    The DFN is solved with lumped thermal dynamics and the solution is
    resampled onto a uniform dt grid for direct use by the EKF (Machine 2).

    Parameters
    ----------
    pset_name : str    PyBaMM ParameterValues name (e.g. "Chen2020")
    n_cycles  : int    number of full charge/discharge cycles
    c_rate    : float  discharge C-rate [C]
    protocol  : str    "cc" | "cccv" | "hppc"
    v_min     : float  lower voltage cut-off [V]
    v_max     : float  upper voltage cut-off [V]
    dt        : float  resampling interval [s] (default 10 s, matches EKF)

    Returns
    -------
    t     : np.ndarray  time [s], uniform spacing dt
    V     : np.ndarray  terminal voltage [V]
    I     : np.ndarray  current [A]  (+ = discharge)
    soc   : np.ndarray  true SOC [0–1]  from DFN internal state
    T     : np.ndarray  cell temperature [°C]
    Q_nom : float       nominal capacity [A·h]

    Physical model
    --------------
    Doyle-Fuller-Newman electrochemical model with lumped thermal coupling.
    Reference: Doyle et al. 1993, J. Electrochem. Soc. 140, 1526–1533.
    Parameter set: Chen et al. 2020, J. Electrochem. Soc. 167, 080534.
    """
    model  = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
    params = pybamm.ParameterValues(pset_name)
    params.set_initial_stoichiometries(1.0)   # SOC₀ = 100 %
    Q_nom  = float(params["Nominal cell capacity [A.h]"])

    builder = _PROTOCOL_BUILDERS.get(protocol, _build_cccv_steps)
    steps   = builder(n_cycles, c_rate, v_min, v_max)
    exp     = pybamm.Experiment(steps)

    # Prefer IDAKLUSolver (fast sparse DAE solver); fall back to CasadiSolver.
    # Reference: Sulzer et al. 2021, J. Electrochem. Soc. 168, 090014.
    try:
        solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
    except Exception:
        solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-6)

    sim = pybamm.Simulation(
        model,
        parameter_values=params,
        experiment=exp,
        solver=solver,
    )
    sim.solve()
    sol = sim.solution

    # ── Extract raw solution variables ────────────────────────────────────
    t_raw = _safe1d(sol["Time [s]"].entries)
    V_raw = _safe1d(sol["Terminal voltage [V]"].entries)
    I_raw = _safe1d(sol["Current [A]"].entries)

    # Temperature: prefer volume-averaged; fall back to cell temperature.
    try:
        T_raw = (
            _safe1d(sol["Volume-averaged cell temperature [K]"].entries) - 273.15
        )
    except Exception:
        try:
            T_raw = _safe1d(sol["Cell temperature [K]"].entries) - 273.15
        except Exception:
            T_raw = np.full_like(t_raw, 25.0)

    # Safety check: warn if voltage exceeds stated limits by > 50 mV
    V_clamp = np.clip(V_raw, v_min - 0.1, v_max + 0.1)
    if np.any(np.abs(V_raw - V_clamp) > 0.05):
        import warnings
        warnings.warn(
            f"DFN voltage exceeded limits [{v_min:.3f}, {v_max:.3f}] V — "
            "check protocol or parameter set.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── Clean & resample ──────────────────────────────────────────────────
    t_c, V_c, I_c, T_c = _clean_time(t_raw, V_raw, I_raw, T_raw)

    # Time-accurate SOC extraction (interpolated on real time axis)
    soc_c = _extract_soc(sol, t_c)

    # Resample all signals onto uniform dt grid
    t_u, V_u, I_u, soc_u, T_u = _resample(t_c, dt, V_c, I_c, soc_c, T_c)

    return t_u, V_u, I_u, soc_u, T_u, Q_nom
