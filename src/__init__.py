from __future__ import annotations

"""
src/__init__.py — BattSim Package Initialiser
"""

from .chemistry import build_chem, make_ocv, docv_dsoc
from .utils import (
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

__all__ = [
    "build_chem", "make_ocv", "docv_dsoc",
    "safe_array", "downsample", "time_to_hours", "soc_to_percent",
    "rmse", "mae", "max_error", "detect_cycles", "per_cycle_stats",
    "nis_calibration", "fmt_soc", "fmt_rmse", "fmt_sigma", "summary_dict",
]
