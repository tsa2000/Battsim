"""
BattSim v5.0 — Digital Twin Co-Simulation Package
===================================================

Architecture
------------
Machine 1  (Physical Asset):
    DFN-surrogate simulator — 2-RC ECM with PyBaMM-validated parameters.
    Produces ground-truth: V(t), SOC(t), I(t), T(t), Q_nom.

Machine 2  (Digital Observer):
    2-RC Thevenin ECM + Extended Kalman Filter (EKF).
    Observes noisy V from Machine 1, estimates SOC + full uncertainty.

Chemistry  (Parameter Sets):
    NMC  — Chen2020   (LG M50 21700,  Q=5.0 Ah, 2.5–4.2 V)
    LFP  — Prada2013  (ANR26650,      Q=2.3 Ah, 2.5–3.65 V)
    NMA  — OKane2022  (Kokam SLPB,    Q=3.5 Ah, 2.7–4.3 V)

Uncertainty Quantification (UQ)
--------------------------------
Per timestep:
    sigma_soc  = √P[0,0]          ← ±1σ SOC uncertainty
    ci_upper   = SOC_est + 2·σ    ← 95% confidence upper
    ci_lower   = SOC_est − 2·σ    ← 95% confidence lower
    P_tr       = trace(P)          ← total state uncertainty
    NIS        = ν²/S              ← filter consistency (target ≈ 1.0)

References
----------
    Plett G.L. (2004) J. Power Sources 134, 252–261
    Chen et al. (2020) J. Electrochem. Soc. 167, 080534
    Coman et al. (2022) J. Power Sources

Author
------
    Eng. Thaer Abushawar | Thaer199@gmail.com

Public API
----------
    from src import build_chem, make_ocv_fn, docv_dsoc_num
    from src import run_machine1, run_cosim, EKF
    from src import (
        safe_array, downsample, time_to_hours, soc_to_percent,
        rmse, mae, max_error,
        detect_cycles, per_cycle_stats, cycle_stats,
        nis_calibration, summary_dict, summary_metrics,
        fmt_soc, fmt_sigma, fmt_rmse,
        CHEMISTRY_DB,
    )

Typical Usage
-------------
    import src

    # 1. Pick chemistry
    chem = src.CHEMISTRY_DB["NMC — Chen2020 (LG M50 21700)"]

    # 2. Machine 1 — DFN physical asset
    t, V, I, soc, T, Q = src.run_machine1(
        chem, n_cycles=5, c_rate=1.0, protocol="cccv"
    )

    # 3. Machine 2 — EKF digital observer
    log = src.run_cosim(
        t, V, I, soc, T, Q, chem,
        noise_std=0.010,      # 10 mV
        P0_scale=1e-3,
        Q_scale=1.0,
        R_scale=1.0,
    )

    # 4. Per-cycle statistics
    df = src.cycle_stats(log)            # pandas DataFrame
    sm = src.summary_metrics(log)        # flat dict for Streamlit metrics

    # 5. NIS calibration check
    cal = src.nis_calibration(log["NIS"])
    print(cal["verdict"])

    # 6. Uncertainty outputs in log:
    #   log["sigma_soc"]  — ±1σ SOC per timestep
    #   log["ci_upper"]   — SOC + 2σ
    #   log["ci_lower"]   — SOC − 2σ
    #   log["NIS"]        — filter consistency
    #   log["P_tr"]       — state covariance trace
"""

# ── Version & metadata ────────────────────────────────────────────
__version__ = "5.0.0"
__author__  = "Eng. Thaer Abushawar"
__email__   = "Thaer199@gmail.com"
__all__ = [
    # ── Chemistry ──────────────────────────────────────────────
    "CHEMISTRY_DB",
    "build_chem",
    "make_ocv_fn",
    "docv_dsoc_num",
    # ── Machine 1 ──────────────────────────────────────────────
    "run_machine1",
    # ── Machine 2 ──────────────────────────────────────────────
    "EKF",
    "run_cosim",
    # ── Analytics ──────────────────────────────────────────────
    "detect_cycles",
    "cycle_stats",
    "per_cycle_stats",
    "summary_metrics",
    "summary_dict",
    "nis_calibration",
    # ── Utilities ──────────────────────────────────────────────
    "safe_array",
    "downsample",
    "time_to_hours",
    "soc_to_percent",
    "rmse",
    "mae",
    "max_error",
    "fmt_soc",
    "fmt_sigma",
    "fmt_rmse",
]

# ── Imports ───────────────────────────────────────────────────────
# Chemistry & ECM parameters
from .chemistry import (
    CHEMISTRY_DB,
    build_chem,
    make_ocv_fn,
    docv_dsoc_num,
)

# Machine 1 — DFN physical asset simulator
from .machine1_dfn import run_machine1

# Machine 2 — EKF digital observer
from .machine2_ekf import EKF, run_cosim

# Analytics & stats (from utils)
from .utils import (
    # array helpers
    safe_array,
    downsample,
    time_to_hours,
    soc_to_percent,
    # error metrics
    rmse,
    mae,
    max_error,
    # cycle analysis
    detect_cycles,
    per_cycle_stats,
    nis_calibration,
    # summary
    summary_dict,
    # formatting
    fmt_soc,
    fmt_sigma,
    fmt_rmse,
)

# App-level analytics (app.py implements these directly;
# expose here for notebook/script usage)
try:
    from .app import cycle_stats, summary_metrics  # optional — only when used as package
except ImportError:
    # When app.py is run standalone via `streamlit run app.py`,
    # cycle_stats and summary_metrics are defined inside app.py itself.
    # No action needed — they are not required at package import time.
    pass
