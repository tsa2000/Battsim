import numpy as np
import pybamm
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


_OCV_CACHE = {}
_ECM_CACHE = {}

_PSETS = {
    "NMC — Chen2020 (LG M50 21700)": {
        "pset": "Chen2020",
        "color": "#00b4d8",
        "desc": "NMC811 — LG M50 21700 — experimentally validated (Chen et al. 2020)",
    },
    "LFP — Prada2013 (A123 26650)": {
        "pset": "Prada2013",
        "color": "#2dc653",
        "desc": "LFP — A123 26650 — experimentally validated (Prada et al. 2013)",
    },
    "NMA — OKane2022 (LG M50 proxy)": {
        "pset": "OKane2022",
        "color": "#f77f00",
        "desc": "NMA proxy — OKane2022 (LG M50 base) — closest available in PyBaMM",
    },
}

_FALLBACK_OCV = {
    "Chen2020": (
        [0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,
         .55,.6,.65,.7,.75,.8,.85,.9,.95,1.0],
        [3.0,3.3,3.42,3.5,3.54,3.57,3.62,3.65,3.68,3.71,
         3.74,3.77,3.8,3.84,3.88,3.92,3.96,4.01,4.06,4.13,4.2],
    ),
    "Prada2013": (
        [0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,
         .55,.6,.65,.7,.75,.8,.85,.9,.95,1.0],
        [3.0,3.1,3.2,3.25,3.28,3.30,3.31,3.32,3.325,3.33,
         3.335,3.34,3.345,3.35,3.36,3.37,3.38,3.39,3.40,3.42,3.6],
    ),
    "OKane2022": (
        [0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,
         .55,.6,.65,.7,.75,.8,.85,.9,.95,1.0],
        [3.0,3.35,3.48,3.55,3.60,3.64,3.68,3.72,3.75,3.78,
         3.82,3.85,3.88,3.92,3.96,4.00,4.05,4.10,4.16,4.22,4.3],
    ),
}

_FALLBACK_ECM = {
    "Chen2020":  dict(R0=0.010, R1=0.015, C1=3000.0, R2=0.008, C2=8000.0),
    "Prada2013": dict(R0=0.020, R1=0.018, C1=2500.0, R2=0.010, C2=6000.0),
    "OKane2022": dict(R0=0.012, R1=0.016, C1=2800.0, R2=0.009, C2=7500.0),
}

_FALLBACK_VLIM = {
    "Chen2020":  (2.5, 4.2),
    "Prada2013": (2.0, 3.6),
    "OKane2022": (2.5, 4.3),
}


# Stoichiometry limits for each parameter set
# Source: PyBaMM parameter sets (Chen2020, Prada2013, OKane2022)
# x = anode stoichiometry,  y = cathode stoichiometry
# at SOC=0%: x=x_0, y=y_0  |  at SOC=100%: x=x_100, y=y_100
_STOICH = {
    "Chen2020":  dict(x_0=0.0015, x_100=0.7522, y_0=0.9084, y_100=0.4379),
    "Prada2013": dict(x_0=0.0000, x_100=0.8300, y_0=0.9800, y_100=0.3000),
    "OKane2022": dict(x_0=0.0015, x_100=0.7522, y_0=0.9084, y_100=0.4379),
}


def _extract_ocv(pset: str, n_points: int = 41):
    """
    Extract OCV via quasi-static C/20 DFN discharge.
    At C/20, V_terminal ≈ OCV(SOC) — near-equilibrium condition.
    Reference: Newman & Thomas-Alyea (2004), Ecker et al. (2015)
    """
    if pset in _OCV_CACHE:
        return _OCV_CACHE[pset]

    try:
        params   = pybamm.ParameterValues(pset)
        v_min, v_max = _FALLBACK_VLIM.get(pset, (2.5, 4.2))

        model = pybamm.lithium_ion.DFN(
            options={"thermal": "isothermal"})
        exp = pybamm.Experiment([
            f"Discharge at C/20 until {v_min}V",
        ])
        sim = pybamm.Simulation(
            model, parameter_values=params, experiment=exp)
        sim.solve(calc_esoh=False)
        sol = sim.solution

        soc_raw = sol["Discharge capacity [A.h]"].entries.ravel()
        Q_nom   = float(pybamm.ParameterValues(pset)["Nominal cell capacity [A.h]"])
        soc_raw = 1.0 - soc_raw / (Q_nom + 1e-8)
        V_raw   = sol["Terminal voltage [V]"].entries.ravel()
        t_raw   = sol["Time [s]"].entries.ravel()

        mask    = np.concatenate([[True], np.diff(t_raw) > 1e-10])
        soc_raw = soc_raw[mask]
        V_raw   = V_raw[mask]

        idx   = np.argsort(soc_raw)
        soc_s = soc_raw[idx]
        V_s   = V_raw[idx]

        ocv_fn_tmp = interp1d(soc_s, V_s, kind="cubic",
                               bounds_error=False,
                               fill_value=(float(V_s[0]), float(V_s[-1])))
        soc_pts = np.linspace(0.0, 1.0, n_points)
        ocv_pts = [float(ocv_fn_tmp(s)) for s in soc_pts]

        result = (soc_pts.tolist(), ocv_pts)
        _OCV_CACHE[pset] = result
        return result

    except Exception as _e:
        try:
            import streamlit as st
            st.warning(f"⚠️ OCV C/20 failed for {pset}: {_e} — using fallback")
        except Exception:
            pass
        return _FALLBACK_OCV.get(pset)

def _extract_ecm(pset: str):
    if pset in _ECM_CACHE:
        return _ECM_CACHE[pset]

    try:
        params = pybamm.ParameterValues(pset)
        Q_nom  = float(params["Nominal cell capacity [A.h]"])
        v_min, v_max = _FALLBACK_VLIM.get(pset, (2.5, 4.2))

        model = pybamm.lithium_ion.DFN(
            options={"thermal": "lumped"})
        exp = pybamm.Experiment([
            f"Discharge at 1C for 10 seconds",
            "Rest for 40 seconds",
            f"Charge at 1C for 10 seconds",
            "Rest for 40 seconds",
        ])
        sim = pybamm.Simulation(
            model, parameter_values=params, experiment=exp)
        sim.solve()
        sol = sim.solution

        t = sol["Time [s]"].entries.ravel()
        V = sol["Terminal voltage [V]"].entries.ravel()
        I = sol["Current [A]"].entries.ravel()

        mask = np.concatenate([[True], np.diff(t) > 1e-10])
        t, V, I = t[mask], V[mask], I[mask]

        idx_jump = np.where(np.abs(np.diff(I)) > 0.01)[0]
        if len(idx_jump) > 0:
            dV = abs(float(V[idx_jump[0]+1]) - float(V[idx_jump[0]]))
            dI = abs(float(I[idx_jump[0]+1]) - float(I[idx_jump[0]])) + 1e-8
            R0 = float(np.clip(dV / dI, 1e-4, 0.5))
        else:
            R0 = _FALLBACK_ECM[pset]["R0"]

        rest_mask = (t > 10.5) & (t < 49.5)
        t_r = t[rest_mask] - t[rest_mask][0]
        V_r = V[rest_mask]

        def two_rc(t_, V0, R1_, tau1, R2_, tau2):
            return (V0
                    + R1_ * (1 - np.exp(-t_ / (tau1 + 1e-8)))
                    + R2_ * (1 - np.exp(-t_ / (tau2 + 1e-8))))

        p0     = [float(V_r[0]), 0.010, 45.0, 0.005, 400.0]
        bounds = (
            [float(V_r.min()) - 0.1, 1e-4,   1.0,  1e-4,   10.0],
            [float(V_r.max()) + 0.1, 0.200, 200.0,  0.200, 5000.0],
        )
        popt, _ = curve_fit(
            two_rc, t_r, V_r, p0=p0,
            bounds=bounds, maxfev=8000)

        _, R1, tau1, R2, tau2 = popt
        C1 = float(tau1) / (float(R1) + 1e-8)
        C2 = float(tau2) / (float(R2) + 1e-8)

        result = dict(
            R0=round(float(R0), 5),
            R1=round(float(R1), 5), C1=round(C1, 1),
            R2=round(float(R2), 5), C2=round(C2, 1),
        )
        _ECM_CACHE[pset] = result
        return result

    except Exception:
        return _FALLBACK_ECM.get(pset)


def _vlim_from_pybamm(pset: str):
    try:
        params = pybamm.ParameterValues(pset)
        v_min  = float(params.get(
            "Lower voltage cut-off [V]",
            _FALLBACK_VLIM[pset][0]))
        v_max  = float(params.get(
            "Upper voltage cut-off [V]",
            _FALLBACK_VLIM[pset][1]))
        return v_min, v_max
    except Exception:
        return _FALLBACK_VLIM.get(pset, (2.5, 4.2))


def _qnom_from_pybamm(pset: str):
    try:
        params = pybamm.ParameterValues(pset)
        return float(params["Nominal cell capacity [A.h]"])
    except Exception:
        return {"Chen2020": 5.0, "Prada2013": 1.1,
                "OKane2022": 5.0}.get(pset, 3.0)


def build_chem():
    chem = {}
    for label, meta in _PSETS.items():
        pset = meta["pset"]

        soc_lut, ocv_lut = _extract_ocv(pset)
        ecm               = _extract_ecm(pset)
        v_min, v_max      = _vlim_from_pybamm(pset)
        Q                 = _qnom_from_pybamm(pset)

        chem[label] = dict(
            pybamm  = pset,
            color   = meta["color"],
            desc    = meta["desc"],
            v_min   = v_min,
            v_max   = v_max,
            Q       = Q,
            soc_lut = soc_lut,
            ocv_lut = ocv_lut,
            **ecm,
        )
    return chem


def make_ocv(chem: dict):
    return interp1d(
        chem["soc_lut"], chem["ocv_lut"],
        kind="cubic", bounds_error=False,
        fill_value=(chem["ocv_lut"][0], chem["ocv_lut"][-1]),
    )


def docv_dsoc(ocv_fn, soc: float, eps: float = 1e-4) -> float:
    s_hi = np.clip(soc + eps, 0.0, 1.0)
    s_lo = np.clip(soc - eps, 0.0, 1.0)
    return float(ocv_fn(s_hi) - ocv_fn(s_lo)) / (s_hi - s_lo + 1e-12)
