"""
src/__init__.py — BattSim Package Initialiser
==============================================
يُعرِّف حزمة src ويُصدِّر الواجهة العامة للمشروع.

Structure
---------
src/
  __init__.py          ← هذا الملف
  chemistry.py         ← OCV/ECM parameter tables + build_chem()
  machine1_dfn.py      ← DFN simulation via PyBaMM (Machine 1)
  machine2_ekf.py      ← 2-RC ECM + AEKF observer  (Machine 2)
  utils.py             ← shared utilities: metrics, per-cycle UQ, NIS
  pdf_report.py        ← ReportLab PDF export
"""

from .chemistry      import build_chem, make_ocv, docv_dsoc
from .machine1_dfn   import run_dfn
from .machine2_ekf   import run_cosim
from .utils          import (
    safe_array,
    downsample,
    time_to_hours,
    soc_to_percent,
    rmse,
    mae,
    max_error,
    detect_cycles,
    per_cycle_stats,
    nis_calibration,
    fmt_soc,
    fmt_rmse,
    fmt_sigma,
    summary_dict,
)
from .pdf_report     import build_pdf_report

__all__ = [
    # chemistry
    "build_chem",
    "make_ocv",
    "docv_dsoc",
    # simulation
    "run_dfn",
    "run_cosim",
    # utilities
    "safe_array",
    "downsample",
    "time_to_hours",
    "soc_to_percent",
    "rmse",
    "mae",
    "max_error",
    "detect_cycles",
    "per_cycle_stats",
    "nis_calibration",
    "fmt_soc",
    "fmt_rmse",
    "fmt_sigma",
    "summary_dict",
    # report
    "build_pdf_report",
]
