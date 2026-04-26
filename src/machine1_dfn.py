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


def _extract_soc(sol, t_clean, Q_nom):
    try:
        disc = _safe1d(sol["Discharge capacity [A.h]"].entries)
        total = float(
            sol["Total lithium capacity in particles [A.h]"].entries.ravel()[0]
        )
        soc = np.clip(1.0 - disc / (total + 1e-12), 0.0, 1.0)
        if len(soc) != len(t_clean):
            soc = np.interp(
                np.linspace(0, 1, len(t_clean)),
                np.linspace(0, 1, len(soc)),
                soc,
            )
        return soc
    except Exception:
        pass

    try:
        t_raw = _safe1d(sol["Time [s]"].entries)
        I_raw = _safe1d(sol["Current [A]"].entries)
        mask  = np.concatenate([[True], np.diff(t_raw) > 1e-10])
        t_c   = t_raw[mask]
        I_c   = I_raw[mask]
        dt    = np.diff(t_c, prepend=t_c[0])
        soc   = np.clip(
            1.0 - np.cumsum(I_c * dt) / (3600.0 * Q_nom),
            0.0, 1.0,
        )
        return np.interp(t_clean, t_c, soc)
    except Exception:
        return np.linspace(1.0, 0.0, len(t_clean))


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
            f"Discharge at {c_rate:.4f}C for 10 seconds",
            "Rest for 40 seconds",
            f"Discharge at {c_rate * 2:.4f}C for 10 seconds",
            "Rest for 40 seconds",
            f"Charge at {c_rate:.4f}C for 10 seconds",
            "Rest for 40 seconds",
            f"Discharge at {c_rate:.4f}C until {v_min:.3f} V",
            "Rest for 5 minutes",
            f"Charge at {c_rate / 2:.4f}C until {v_max:.3f} V",
            "Rest for 5 minutes",
        ]
    return steps


_PROTOCOL_BUILDERS = {
    "cc":   _build_cc_steps,
    "cccv": _build_cccv_steps,
    "hppc": _build_hppc_steps,
}


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
    Run PyBaMM DFN Physical Asset.

    Parameters
    ----------
    pset_name : str
        PyBaMM ParameterValues name  (e.g. "Chen2020").
    n_cycles  : int
        Number of full charge/discharge cycles.
    c_rate    : float
        Discharge C-rate  (e.g. 1.0 for 1C).
    protocol  : str
        "cc" | "cccv" | "hppc"
    v_min     : float   [V]
        Lower voltage cut-off.
    v_max     : float   [V]
        Upper voltage cut-off.
    dt        : float   [s]
        Output resampling interval (default 10 s).

    Returns
    -------
    t     : np.ndarray  [s]
    V     : np.ndarray  [V]   terminal voltage (ground truth)
    I     : np.ndarray  [A]   current (+ discharge)
    soc   : np.ndarray  [0-1] state of charge (ground truth)
    T     : np.ndarray  [°C]  cell temperature
    Q_nom : float       [Ah]  nominal capacity
    """

    model = pybamm.lithium_ion.DFN(
        options={"thermal": "lumped"}
    )

    params = pybamm.ParameterValues(pset_name)
    params.set_initial_stoichiometries(1.0)

    Q_nom = float(params["Nominal cell capacity [A.h]"])

    builder = _PROTOCOL_BUILDERS.get(protocol, _build_cc_steps)
    steps   = builder(n_cycles, c_rate, v_min, v_max)

    exp = pybamm.Experiment(steps)

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

    t_raw = _safe1d(sol["Time [s]"].entries)
    V_raw = _safe1d(sol["Terminal voltage [V]"].entries)
    I_raw = _safe1d(sol["Current [A]"].entries)

    try:
        T_raw = _safe1d(
            sol["Volume-averaged cell temperature [K]"].entries
        ) - 273.15
    except Exception:
        try:
            T_raw = _safe1d(
                sol["Cell temperature [K]"].entries
            ) - 273.15
        except Exception:
            T_raw = np.full_like(t_raw, 25.0)

    t_c, V_c, I_c, T_c = _clean_time(t_raw, V_raw, I_raw, T_raw)

    soc_c = _extract_soc(sol, t_c, Q_nom)

    t_u, V_u, I_u, soc_u, T_u = _resample(
        t_c, dt, V_c, I_c, soc_c, T_c
    )

    return t_u, V_u, I_u, soc_u, T_u, Q_nom

