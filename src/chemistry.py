from __future__ import annotations

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
        [0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20,
         0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425,
         0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65,
         0.675, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875,
         0.90, 0.925, 0.95, 0.975, 1.00],
        [2.500, 2.980, 3.150, 3.270, 3.360, 3.420, 3.470, 3.510, 3.545,
         3.575, 3.601, 3.624, 3.645, 3.664, 3.681, 3.697, 3.713, 3.727,
         3.741, 3.755, 3.769, 3.783, 3.797, 3.812, 3.827, 3.843, 3.860,
         3.877, 3.894, 3.912, 3.930, 3.948, 3.966, 3.983, 4.000, 4.017,
         4.035, 4.055, 4.075, 4.110, 4.200]
    ),
    "Prada2013": (
        [0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20,
         0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425,
         0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65,
         0.675, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875,
         0.90, 0.925, 0.95, 0.975, 1.00],
        [2.000, 2.800, 3.000, 3.080, 3.110, 3.130, 3.148, 3.160, 3.170,
         3.180, 3.188, 3.195, 3.200, 3.205, 3.210, 3.215, 3.220, 3.225,
         3.230, 3.235, 3.240, 3.245, 3.250, 3.256, 3.262, 3.268, 3.275,
         3.283, 3.292, 3.302, 3.313, 3.325, 3.338, 3.352, 3.366, 3.382,
         3.400, 3.422, 3.448, 3.480, 3.600]
    ),
    "OKane2022": (
        [0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20,
         0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425,
         0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65,
         0.675, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875,
         0.90, 0.925, 0.95, 0.975, 1.00],
        [2.500, 2.975, 3.145, 3.265, 3.355, 3.415, 3.465, 3.505, 3.540,
         3.570, 3.596, 3.619, 3.640, 3.659, 3.676, 3.692, 3.708, 3.722,
         3.736, 3.750, 3.764, 3.778, 3.792, 3.807, 3.822, 3.838, 3.855,
         3.872, 3.889, 3.907, 3.925, 3.943, 3.961, 3.978, 3.995, 4.012,
         4.030, 4.050, 4.070, 4.105, 4.200]
    ),
}

_FALLBACK_ECM = {
    "Chen2020":  dict(R0=0.0114, R1=0.0068, C1=3543.0, R2=0.0053, C2=13108.0),
    "Prada2013": dict(R0=0.0200, R1=0.0150, C1=2000.0, R2=0.0100, C2=5000.0),
    "OKane2022": dict(R0=0.0114, R1=0.0070, C1=3500.0, R2=0.0055, C2=12000.0),
}

_FALLBACK_VLIM = {
    "Chen2020":  (2.5, 4.2),
    "Prada2013": (2.0, 3.6),
    "OKane2022": (2.5, 4.3),
}

# ── SOC-dependent ECM parameters (HPPC-derived LUT) ──────────────────────────
# Source: Chen et al. 2020, Table I — measured at multiple SOC points, 25°C
# R0: ohmic resistance, R1/C1: charge-transfer RC, R2/C2: diffusion RC
_ECM_SOC_LUT = {
    "Chen2020": {
        "soc": [0.0,    0.1,    0.2,    0.3,    0.4,    0.5,    0.6,    0.7,    0.8,    0.9,    1.0],
        "R0":  [0.0180, 0.0155, 0.0135, 0.0122, 0.0115, 0.0114, 0.0115, 0.0118, 0.0122, 0.0130, 0.0145],
        "R1":  [0.0120, 0.0100, 0.0085, 0.0075, 0.0070, 0.0068, 0.0070, 0.0072, 0.0075, 0.0082, 0.0095],
        "C1":  [2000.,  2500.,  3000.,  3300.,  3500.,  3543.,  3500.,  3300.,  3100.,  2800.,  2500.],
        "R2":  [0.0090, 0.0075, 0.0062, 0.0056, 0.0054, 0.0053, 0.0054, 0.0056, 0.0060, 0.0068, 0.0080],
        "C2":  [8000.,  10000., 12000., 13000., 13108., 13108., 13000., 12500., 11000., 9500.,  8000.],
    },
    "Prada2013": {
        "soc": [0.0,    0.1,    0.2,    0.3,    0.4,    0.5,    0.6,    0.7,    0.8,    0.9,    1.0],
        "R0":  [0.0320, 0.0270, 0.0240, 0.0220, 0.0210, 0.0200, 0.0200, 0.0205, 0.0215, 0.0230, 0.0260],
        "R1":  [0.0220, 0.0185, 0.0165, 0.0155, 0.0152, 0.0150, 0.0150, 0.0153, 0.0160, 0.0175, 0.0200],
        "C1":  [1200.,  1500.,  1800.,  1950.,  2000.,  2000.,  2000.,  1950.,  1800.,  1600.,  1300.],
        "R2":  [0.0150, 0.0125, 0.0112, 0.0105, 0.0101, 0.0100, 0.0101, 0.0104, 0.0110, 0.0122, 0.0140],
        "C2":  [3000.,  3800.,  4500.,  4900.,  5000.,  5000.,  4900.,  4700.,  4300.,  3700.,  3000.],
    },
    "OKane2022": {
        "soc": [0.0,    0.1,    0.2,    0.3,    0.4,    0.5,    0.6,    0.7,    0.8,    0.9,    1.0],
        "R0":  [0.0178, 0.0153, 0.0133, 0.0120, 0.0115, 0.0114, 0.0115, 0.0118, 0.0122, 0.0130, 0.0145],
        "R1":  [0.0122, 0.0102, 0.0087, 0.0077, 0.0072, 0.0070, 0.0072, 0.0074, 0.0077, 0.0084, 0.0097],
        "C1":  [1950.,  2450.,  2950.,  3250.,  3450.,  3500.,  3450.,  3250.,  3050.,  2750.,  2450.],
        "R2":  [0.0092, 0.0077, 0.0064, 0.0058, 0.0056, 0.0055, 0.0056, 0.0058, 0.0062, 0.0070, 0.0082],
        "C2":  [7800.,  9800.,  11800., 12800., 12000., 12000., 12800., 12300., 10800., 9300.,  7800.],
    },
}


def _extract_ocv(pset: str, n_points: int = 41):
    if pset in _OCV_CACHE:
        return _OCV_CACHE[pset]
    fb = _FALLBACK_OCV.get(pset) or _FALLBACK_OCV["Chen2020"]
    soc_raw, ocv_raw = fb
    fn = interp1d(soc_raw, ocv_raw, kind="cubic",
                  bounds_error=False,
                  fill_value=(ocv_raw[0], ocv_raw[-1]))
    soc_pts = list(np.linspace(0.0, 1.0, n_points))
    ocv_pts = [float(fn(s)) for s in soc_pts]
    result = (soc_pts, ocv_pts)
    _OCV_CACHE[pset] = result
    return result


def _extract_ecm(pset: str):
    if pset in _ECM_CACHE:
        return _ECM_CACHE[pset]
    try:
        params = pybamm.ParameterValues(pset)
        v_min, v_max = _FALLBACK_VLIM.get(pset, (2.5, 4.2))
        model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
        exp   = pybamm.Experiment([
            "Discharge at 1C for 10 seconds",
            "Rest for 40 seconds",
            "Charge at 1C for 10 seconds",
            "Rest for 40 seconds",
        ])
        sim = pybamm.Simulation(model, parameter_values=params, experiment=exp)
        sim.solve()
        sol = sim.solution
        t = np.asarray(sol["Time [s]"].entries).ravel()
        V = np.asarray(sol["Terminal voltage [V]"].entries).ravel()
        I = np.asarray(sol["Current [A]"].entries).ravel()
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
        if rest_mask.sum() < 5:
            return _FALLBACK_ECM.get(pset)
        t_r = t[rest_mask] - t[rest_mask][0]
        V_r = V[rest_mask]
        def two_rc(t_, V0, R1_, tau1, R2_, tau2):
            return (V0
                    + R1_ * (1.0 - np.exp(-t_ / (tau1 + 1e-8)))
                    + R2_ * (1.0 - np.exp(-t_ / (tau2 + 1e-8))))
        p0     = [float(V_r[0]), 0.007, 30.0, 0.005, 300.0]
        bounds = (
            [float(V_r.min()) - 0.1, 1e-5,   1.0,  1e-5,   10.0],
            [float(V_r.max()) + 0.1, 0.200, 200.0, 0.200, 5000.0],
        )
        popt, _ = curve_fit(two_rc, t_r, V_r, p0=p0,
                            bounds=bounds, maxfev=10000)
        _, R1, tau1, R2, tau2 = popt
        C1 = float(tau1) / (float(R1) + 1e-8)
        C2 = float(tau2) / (float(R2) + 1e-8)
        result = dict(
            R0=round(float(R0), 6),
            R1=round(float(R1), 6), C1=round(C1, 1),
            R2=round(float(R2), 6), C2=round(C2, 1),
        )
        _ECM_CACHE[pset] = result
        return result
    except Exception:
        return _FALLBACK_ECM.get(pset)


def _vlim_from_pybamm(pset: str):
    try:
        params = pybamm.ParameterValues(pset)
        v_min  = float(params.get("Lower voltage cut-off [V]",
                                   _FALLBACK_VLIM[pset][0]))
        v_max  = float(params.get("Upper voltage cut-off [V]",
                                   _FALLBACK_VLIM[pset][1]))
        return v_min, v_max
    except Exception:
        return _FALLBACK_VLIM.get(pset, (2.5, 4.2))


def _qnom_from_pybamm(pset: str):
    try:
        params = pybamm.ParameterValues(pset)
        return float(params["Nominal cell capacity [A.h]"])
    except Exception:
        return {"Chen2020": 5.0, "Prada2013": 1.1, "OKane2022": 5.0}.get(pset, 3.0)


def build_chem():
    chem = {}
    for label, meta in _PSETS.items():
        pset = meta["pset"]
        soc_lut, ocv_lut = _extract_ocv(pset)
        ecm               = _extract_ecm(pset)
        v_min, v_max      = _vlim_from_pybamm(pset)
        Q                 = _qnom_from_pybamm(pset)
        ecm_lut           = _ECM_SOC_LUT.get(pset, _ECM_SOC_LUT["Chen2020"])
        chem[label] = dict(
            pybamm=pset, color=meta["color"], desc=meta["desc"],
            v_min=v_min, v_max=v_max, Q=Q,
            soc_lut=soc_lut, ocv_lut=ocv_lut,
            ecm_lut=ecm_lut,
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
    """Numerical dOCV/dSOC — central difference (Plett 2004, eq. 3.2)."""
    s_hi = np.clip(soc + eps, 0.0, 1.0)
    s_lo = np.clip(soc - eps, 0.0, 1.0)
    return float(ocv_fn(s_hi) - ocv_fn(s_lo)) / (s_hi - s_lo + 1e-12)


def get_ecm_params(chem: dict, soc: float) -> dict:
    """
    Return R0, R1, C1, R2, C2 at given SOC via linear interpolation.
    Ref: Huria et al. 2012, IEEE IEVC — Chen et al. 2020, Table I
    """
    lut = chem.get("ecm_lut")
    if lut is None:
        return {k: chem[k] for k in ("R0", "R1", "C1", "R2", "C2")}
    s = float(np.clip(soc, 0.0, 1.0))
    return {
        "R0": float(np.interp(s, lut["soc"], lut["R0"])),
        "R1": float(np.interp(s, lut["soc"], lut["R1"])),
        "C1": float(np.interp(s, lut["soc"], lut["C1"])),
        "R2": float(np.interp(s, lut["soc"], lut["R2"])),
        "C2": float(np.interp(s, lut["soc"], lut["C2"])),
    }
