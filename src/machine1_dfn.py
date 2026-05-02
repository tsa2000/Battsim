import numpy as np
import pybamm


def _safe1d(arr):
    return np.asarray(arr).ravel()


def _clean_time(t, *arrays):
    mask = np.concatenate([[True], np.diff(t) > 1e-10])
    return (t[mask],) + tuple(a[mask] for a in arrays)


def _resample(t, dt, *arrays):
    t_u = np.arange(t[0], t[-1], dt)
    return (t_u,) + tuple(np.interp(t_u, t, a) for a in arrays)


def _extract_soc(sol, t_clean: np.ndarray) -> np.ndarray:
    """
    Extract SOC robustly using Discharge Capacity.
    Ensures SOC is always 1.0 at beginning of discharge/simulation.
    """
    t_sol = _safe1d(sol["Time [s]"].entries)
    mask  = np.concatenate([[True], np.diff(t_sol) > 1e-10])
    t_sol = t_sol[mask]

    # Q_discharged starts at 0 at t=0 if initial_soc=1.0 is set
    q_disch = _safe1d(sol["Discharge capacity [A.h]"].entries)[mask]
    q_total = float(sol["Discharge capacity [A.h]"].entries[-1])

    if q_total < 1e-6: q_total = 1.0

    # SOC = 1.0 - (Q_discharged / Q_total)
    soc_raw = 1.0 - (q_disch / q_total)
    soc_raw = np.clip(soc_raw, 0.0, 1.0)

    return np.interp(t_clean, t_sol, soc_raw)


# ── Experiment Builders ────────────────────────────────────────────────────────

def _build_cc_steps(n_cycles, c_rate, v_min, v_max):
    steps = []
    for _ in range(n_cycles):
        steps += [
            f"Discharge at {c_rate:.4f}C until {v_min:.3f} V",
            "Rest for 5 minutes",
            f"Charge at {c_rate / 2:.4f}C until {v_max:.3f} V",
            "Rest for 5 minutes",
        ]
    return steps


def _build_cccv_steps(n_cycles, c_rate, v_min, v_max):
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


def _build_hppc_steps(n_cycles, c_rate, v_min, v_max):
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

def run_dfn(pset_name, n_cycles, c_rate, protocol, v_min, v_max, dt=10.0):
    model  = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
    params = pybamm.ParameterValues(pset_name)
    Q_nom  = float(params["Nominal cell capacity [A.h]"])

    builder = _PROTOCOL_BUILDERS.get(protocol, _build_cccv_steps)
    steps   = builder(n_cycles, c_rate, v_min, v_max)
    exp     = pybamm.Experiment(steps)

    sim = pybamm.Simulation(model, parameter_values=params, experiment=exp)

    # Use initial_soc=1.0 for consistent DFN start
    sim.solve(initial_soc=1.0)
    sol = sim.solution

    t_raw = _safe1d(sol["Time [s]"].entries)
    V_raw = _safe1d(sol["Terminal voltage [V]"].entries)
    I_raw = _safe1d(sol["Current [A]"].entries)

    try:
        T_raw = (_safe1d(sol["Volume-averaged cell temperature [K]"].entries) - 273.15)
    except Exception:
        T_raw = np.full_like(t_raw, 25.0)

    t_c, V_c, I_c, T_c = _clean_time(t_raw, V_raw, I_raw, T_raw)
    soc_c = _extract_soc(sol, t_c)
    t_u, V_u, I_u, soc_u, T_u = _resample(t_c, dt, V_c, I_c, soc_c, T_c)

    return t_u, V_u, I_u, soc_u, T_u, Q_nom
