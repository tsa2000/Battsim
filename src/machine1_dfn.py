import numpy as np
import pybamm


def _safe1d(arr):
    return np.asarray(arr).ravel()


def _clean_time(t, *arrays):
    """Remove duplicate time points (PyBaMM event-capture artefact)."""
    mask = np.concatenate([[True], np.diff(t) > 1e-10])
    return (t[mask],) + tuple(a[mask] for a in arrays)


def _resample(t, dt, *arrays):
    """Uniform resampling on regular dt grid via linear interpolation."""
    t_u = np.arange(t[0], t[-1], dt)
    return (t_u,) + tuple(np.interp(t_u, t, a) for a in arrays)


def _extract_soc(sol, t_clean: np.ndarray, Q_nom: float) -> np.ndarray:
    """
    Extract true SOC from PyBaMM solution.

    Priority order (most to least accurate):
    1. X-averaged negative electrode SOC  — PyBaMM internal state (best)
    2. Negative electrode SOC             — alternative key
    3. Coulomb counting fallback          — last resort, drifts over cycles

    Reference: PyBaMM docs — Variables — Electrode
    """
    # Priority 1: internal PyBaMM SOC state (most accurate)
    for key in [
        "X-averaged negative electrode SOC",
        "Negative electrode SOC",
        "Average negative electrode SOC",
    ]:
        try:
            raw = _safe1d(sol[key].entries)
            raw = np.clip(raw, 0.0, 1.0)
            if len(raw) != len(t_clean):
                raw = np.interp(
                    np.linspace(0, 1, len(t_clean)),
                    np.linspace(0, 1, len(raw)),
                    raw,
                )
            return raw
        except Exception:
            continue

    # Priority 2: discharge-capacity based SOC
    try:
        disc  = _safe1d(sol["Discharge capacity [A.h]"].entries)
        total = float(
            sol["Total lithium capacity in particles [A.h]"].entries.ravel()[0]
        )
        if total > 0:
            soc = np.clip(1.0 - disc / total, 0.0, 1.0)
            if len(soc) != len(t_clean):
                soc = np.interp(
                    np.linspace(0, 1, len(t_clean)),
                    np.linspace(0, 1, len(soc)),
                    soc,
                )
            return soc
    except Exception:
        pass

    # Priority 3: Coulomb counting (accumulates error — acceptable for 1-3 cycles)
    try:
        t_raw = _safe1d(sol["Time [s]"].entries)
        I_raw = _safe1d(sol["Current [A]"].entries)
        mask  = np.concatenate([[True], np.diff(t_raw) > 1e-10])
        t_c, I_c = t_raw[mask], I_raw[mask]
        dt_arr = np.diff(t_c, prepend=t_c[0])
        soc    = np.clip(1.0 - np.cumsum(I_c * dt_arr) / (3600.0 * Q_nom), 0.0, 1.0)
        return np.interp(t_clean, t_c, soc)
    except Exception:
        return np.linspace(1.0, 0.0, len(t_clean))


# ── Experiment Builders ────────────────────────────────────────────────────────

def _build_cc_steps(n_cycles: int, c_rate: float, v_min: float, v_max: float):
    """CC discharge + CC charge (CC/CC)."""
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
    CC phase charges to V_max, CV phase holds until C/20 (tail current).
    Reference: IEC 62660-1:2018, Battery Test Procedures for EVs.
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

    Follows USABC/USCAR HPPC procedure:
      - Full charge to V_max (CC-CV)
      - At each SOC level (every 10% decrement):
          * 1C discharge pulse 10 s → rest 40 s → 3/4C charge pulse 10 s → rest 40 s
          * 1C discharge for 10% SOC decrement
    Reference: USCAR FreedomCAR Battery Test Manual, DOE/ID-11069, 2004.

    Note: this is a simplified HPPC (pulses every full cycle, not per 10% SOC
    step), suitable for ECM parameter identification rather than full SOC sweep.
    """
    steps = []
    for _ in range(n_cycles):
        steps += [
            # Full charge first
            f"Charge at {c_rate / 2:.4f}C until {v_max:.3f} V",
            f"Hold at {v_max:.3f} V until C/20",
            "Rest for 10 minutes",
            # Discharge pulse: 1C for 10 s
            f"Discharge at {c_rate:.4f}C for 10 seconds",
            "Rest for 40 seconds",
            # Charge pulse: 0.75C for 10 s
            f"Charge at {c_rate * 0.75:.4f}C for 10 seconds",
            "Rest for 40 seconds",
            # Discharge to v_min
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
    Run PyBaMM DFN model as the physical asset (Machine 1).

    Parameters
    ----------
    pset_name : str    — PyBaMM ParameterValues name (e.g. "Chen2020")
    n_cycles  : int    — number of full charge/discharge cycles
    c_rate    : float  — discharge C-rate
    protocol  : str    — "cc" | "cccv" | "hppc"
    v_min     : float  — lower voltage cut-off [V]
    v_max     : float  — upper voltage cut-off [V]
    dt        : float  — resampling interval [s] (default 10 s)

    Returns
    -------
    t, V, I, soc, T : np.ndarray
    Q_nom           : float [Ah]

    Physical model: Doyle-Fuller-Newman (DFN) with lumped thermal.
    Reference: Doyle et al. 1993, J. Electrochem. Soc. 140, 1526.
    """
    model  = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
    params = pybamm.ParameterValues(pset_name)
    params.set_initial_stoichiometries(1.0)   # SOC₀ = 100%
    Q_nom  = float(params["Nominal cell capacity [A.h]"])

    builder = _PROTOCOL_BUILDERS.get(protocol, _build_cccv_steps)
    steps   = builder(n_cycles, c_rate, v_min, v_max)
    exp     = pybamm.Experiment(steps)

    # Solver: prefer IDAKLUSolver (fast, stiff); fall back to CasadiSolver
    try:
        solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
    except Exception:
        solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-6)

    sim = pybamm.Simulation(model, parameter_values=params,
                             experiment=exp, solver=solver)
    sim.solve()
    sol = sim.solution

    t_raw = _safe1d(sol["Time [s]"].entries)
    V_raw = _safe1d(sol["Terminal voltage [V]"].entries)
    I_raw = _safe1d(sol["Current [A]"].entries)

    try:
        T_raw = _safe1d(sol["Volume-averaged cell temperature [K]"].entries) - 273.15
    except Exception:
        try:
            T_raw = _safe1d(sol["Cell temperature [K]"].entries) - 273.15
        except Exception:
            T_raw = np.full_like(t_raw, 25.0)

    # Clamp V to stated limits (safety check — warns on violation)
    V_raw_clipped = np.clip(V_raw, v_min - 0.1, v_max + 0.1)
    if np.any(np.abs(V_raw - V_raw_clipped) > 0.05):
        import warnings
        warnings.warn(
            f"DFN voltage exceeded limits [{v_min}, {v_max}] V — "
            "check protocol or parameter set.", RuntimeWarning)

    t_c, V_c, I_c, T_c = _clean_time(t_raw, V_raw, I_raw, T_raw)
    soc_c = _extract_soc(sol, t_c, Q_nom)

    t_u, V_u, I_u, soc_u, T_u = _resample(t_c, dt, V_c, I_c, soc_c, T_c)

    return t_u, V_u, I_u, soc_u, T_u, Q_nom
