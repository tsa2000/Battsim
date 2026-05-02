from __future__ import annotations

"""
src/pdf_report.py — BattSim Full PDF Report Generator
======================================================
يُصدِّر كل الرسوم التي ينتجها app.py في تقرير PDF كامل.

الرسوم تُحوَّل من Plotly → PNG عبر matplotlib (بدون kaleido/Chrome).
"""

import io
import numpy as np
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_img(fig, w: float = 17.0, h: float = 8.5) -> Image:
    """
    Convert a Plotly figure to a ReportLab Image via matplotlib.

    Iterates over fig.data traces and renders each as a matplotlib line.
    No external dependencies (kaleido, Chrome, orca) required.

    Parameters
    ----------
    fig : plotly.graph_objs.Figure
    w   : float  width  in cm (default 17 cm = full A4 width minus margins)
    h   : float  height in cm

    Returns
    -------
    reportlab.platypus.Image
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig_mpl, ax = plt.subplots(figsize=(w / 2.54, h / 2.54), dpi=150)

    for trace in fig.data:
        try:
            x    = list(trace.x) if trace.x is not None else []
            y    = list(trace.y) if trace.y is not None else []
            name = trace.name or ""
            dash = getattr(getattr(trace, "line", None), "dash", None)
            ls   = "--" if dash in ("dash", "dot") else "-"
            ax.plot(x, y, label=name, linewidth=1.5, linestyle=ls)
        except Exception:
            pass

    # Axis labels
    try:
        ax.set_xlabel(fig.layout.xaxis.title.text or "")
    except Exception:
        pass
    try:
        ax.set_ylabel(fig.layout.yaxis.title.text or "")
    except Exception:
        pass

    # Title — strip any Plotly HTML tags
    try:
        title = fig.layout.title.text or ""
        title = title.split("<br>")[0].replace("<b>", "").replace("</b>", "")
        ax.set_title(title, fontsize=9)
    except Exception:
        pass

    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig_mpl)
    buf.seek(0)
    return Image(buf, width=w * cm, height=h * cm)


def _section(title: str, styles) -> list:
    """Return a styled section heading with horizontal rule."""
    return [
        Spacer(1, 0.35 * cm),
        Paragraph(title, styles["Heading2"]),
        HRFlowable(width="100%", thickness=0.5,
                   color=colors.HexColor("#6366f1")),
        Spacer(1, 0.15 * cm),
    ]


def _fig_block(fig, caption: str, styles,
               w: float = 17.0, h: float = 8.5) -> list:
    """Return a figure image + italic caption as a list of flowables."""
    items = [_fig_to_img(fig, w, h)]
    if caption:
        items.append(Paragraph(
            f"<i>{caption}</i>",
            ParagraphStyle(
                "cap", parent=styles["Normal"],
                fontSize=8, textColor=colors.HexColor("#64748b"),
                spaceBefore=2, spaceAfter=6,
            ),
        ))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Table style helpers
# ─────────────────────────────────────────────────────────────────────────────

_HEADER_VIOLET = colors.HexColor("#6366f1")
_HEADER_SLATE  = colors.HexColor("#334155")
_ROW_ALT       = colors.HexColor("#f8fafc")
_GRID_COLOR    = colors.HexColor("#e2e8f0")

def _base_table_style(header_color=_HEADER_VIOLET) -> list:
    return [
        ("BACKGROUND",    (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_ROW_ALT, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, _GRID_COLOR),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_pdf_report(
    smry:         dict,
    cycle_stats:  list[dict],
    chem_label:   str,
    chem:         dict,
    n_cycles:     int,
    protocol:     str,
    c_rate:       float,
    noise_mv:     float,
    # ── Plotly figures from app.py ────────────────────────────────────────
    fig_voltage:          object,   # Tab 1: Voltage DFN vs AEKF
    fig_soc:              object,   # Tab 1: SOC tracking + ±2σ
    fig_temp:             object,   # Tab 1: Cell temperature
    fig_current:          object,   # Tab 1: Current profile
    fig_ut_ci:            object,   # Tab 2: UT 95% CI
    fig_ut_sigma:         object,   # Tab 2: σ_SOC UT vs EKF
    fig_ut_pvar:          object,   # Tab 2: P[0,0] variance
    fig_nis:              object,   # Tab 3: NIS time-series
    fig_innov:            object,   # Tab 3: Innovation ν
    fig_ocv:              object,   # Tab 4: OCV curve
    fig_uncertainty_prop: object | None = None,  # Tab 4: RMSE per cycle
) -> bytes:
    """
    Build a complete multi-page PDF report.

    Structure
    ---------
    Page 1  — Cover + Summary metrics + Simulation settings
    Page 2  — Voltage tracking · SOC tracking
    Page 3  — Temperature & current · Unscented Transform UQ
    Page 4  — σ_SOC comparison · P[0,0] variance
    Page 5  — NIS time-series · Innovation sequence
    Page 6  — Per-cycle stats table · RMSE per cycle · OCV curve

    Returns
    -------
    bytes  PDF file content (write directly to st.download_button)

    References
    ----------
    Plett 2004, J. Power Sources 134, 252–261
    Julier & Uhlmann 1997, SPIE Proc. 3068
    Wan & van der Merwe 2000, ASSPCC
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "TitleMain", parent=styles["Title"],
        fontSize=22, textColor=_HEADER_VIOLET, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#94a3b8"), spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9.5, leading=14,
    ))

    S = story = []

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 1 — Cover + Summary
    # ══════════════════════════════════════════════════════════════════════
    S.append(Spacer(1, 1 * cm))
    S.append(Paragraph("BattSim Digital Twin", styles["TitleMain"]))
    S.append(Paragraph(
        "Full Simulation Report — DFN (PyBaMM) + ECM 2-RC + AEKF",
        styles["Sub"],
    ))
    S.append(HRFlowable(width="100%", thickness=1.5, color=_HEADER_VIOLET))
    S.append(Spacer(1, 0.3 * cm))

    now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    S.append(Paragraph(
        f"Generated: <b>{now}</b> &nbsp;|&nbsp; "
        f"Chemistry: <b>{chem_label}</b> &nbsp;|&nbsp; "
        f"Protocol: <b>{protocol.upper()}</b> &nbsp;|&nbsp; "
        f"Cycles: <b>{n_cycles}</b> &nbsp;|&nbsp; "
        f"C-rate: <b>{c_rate:.1f}C</b> &nbsp;|&nbsp; "
        f"Noise: <b>{noise_mv:.0f} mV</b>",
        styles["Sub"],
    ))
    S.append(Spacer(1, 0.5 * cm))

    # ── Summary metrics table ─────────────────────────────────────────────
    S += _section("1. Summary Metrics", styles)
    data_smry = [
        ["Metric", "Value", "Interpretation"],
        ["SOC RMSE",         f"{smry['rmse_soc_pct']:.3f} %",
         "< 2% excellent · 2–5% good · > 5% poor"],
        ["SOC MAE",          f"{smry['mae_soc_pct']:.3f} %", "—"],
        ["Max |Error| SOC",  f"{smry['max_err_soc_pct']:.3f} %",
         "Worst-case instantaneous error"],
        ["Mean σ_SOC (EKF)", f"{smry['mean_sigma_pct']:.3f} %",
         "Analytical confidence bandwidth"],
        ["V RMSE",           f"{smry['rmse_v_mv']:.2f} mV",
         "< 10 mV excellent · 10–20 mV acceptable"],
        ["NIS (mean)",       f"{smry['nis_mean']:.4f}",
         "≈ 1.0 = well-calibrated (χ²(1) test)"],
        ["NIS verdict",      smry["nis_verdict"], "—"],
    ]
    t_smry = Table(data_smry, colWidths=[4.5*cm, 3.2*cm, 8.8*cm])
    t_smry.setStyle(TableStyle(_base_table_style(_HEADER_VIOLET)))
    S.append(t_smry)
    S.append(Spacer(1, 0.3 * cm))

    # ── Simulation settings table ─────────────────────────────────────────
    S += _section("2. Simulation Settings", styles)
    data_cfg = [
        ["Parameter", "Value"],
        ["Chemistry",        chem_label],
        ["PyBaMM param set", chem.get("pybamm", "—")],
        ["Protocol",         protocol.upper()],
        ["C-rate",           f"{c_rate:.1f} C"],
        ["Cycles",           str(n_cycles)],
        ["Sensor noise σ",   f"{noise_mv:.0f} mV"],
        ["V_min / V_max",    f"{chem['v_min']:.2f} V / {chem['v_max']:.2f} V"],
        ["ECM topology",     "2-RC Thevenin (Huria et al. 2012)"],
        ["Observer",         "Adaptive EKF with innovation-based R update"],
        ["UQ method",        "EKF covariance P[0,0] + per-cycle analysis"],
    ]
    t_cfg = Table(data_cfg, colWidths=[5*cm, 11.5*cm])
    t_cfg.setStyle(TableStyle(_base_table_style(_HEADER_SLATE)))
    S.append(t_cfg)
    S.append(Paragraph(
        "* Global RMSE excludes first 10% of timesteps (EKF warm-up). "
        "Per-cycle breakdown shown in Section 11.",
        styles["Sub"],
    ))

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 2 — Voltage + SOC tracking
    # ══════════════════════════════════════════════════════════════════════
    S.append(PageBreak())
    S += _section("3. Terminal Voltage — DFN vs AEKF", styles)
    S.append(Paragraph(
        "DFN (truth) vs measured noisy signal vs AEKF estimate. "
        "Voltage RMSE measures the ECM output accuracy.",
        styles["Body"],
    ))
    S.append(Spacer(1, 0.15 * cm))
    S += _fig_block(fig_voltage,
        "Blue: DFN truth · Grey: measured (noisy) · Red: AEKF estimate",
        styles)

    S += _section("4. SOC Tracking — DFN Truth vs AEKF + ±2σ CI", styles)
    S.append(Paragraph(
        "A well-calibrated filter keeps the DFN truth within the ±2σ shaded band "
        "(expected ≥ 95% of the time). Persistent bias outside the band signals "
        "model mismatch (ECM–DFN structural gap).",
        styles["Body"],
    ))
    S.append(Spacer(1, 0.15 * cm))
    S += _fig_block(fig_soc,
        "Blue: DFN truth · Red dashed: AEKF · Shaded: ±2σ EKF",
        styles, h=8.0)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 3 — Temperature / Current + σ comparison
    # ══════════════════════════════════════════════════════════════════════
    S.append(PageBreak())
    S += _section("5. Cell Temperature & Current Profile", styles)
    img_T = _fig_to_img(fig_temp,    w=8.3, h=6.0)
    img_I = _fig_to_img(fig_current, w=8.3, h=6.0)
    S.append(Table(
        [[img_T, img_I]],
        colWidths=[8.3*cm, 8.3*cm],
        style=[
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ],
    ))

    S += _section("6. σ_SOC Comparison — EKF Linearisation Check", styles)
    S.append(Paragraph(
        "If UT σ ≈ EKF σ → the Jacobian linearisation inside EKF is valid. "
        "Divergence indicates significant nonlinearity in the OCV–SOC curve "
        "(typical at low SOC for NMC and at the LFP plateau).",
        styles["Body"],
    ))
    S.append(Spacer(1, 0.15 * cm))
    S += _fig_block(fig_ut_sigma,
        "Red: EKF σ (Jacobian) · Orange dotted: UT σ (7 sigma points)",
        styles, h=6.5)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 4 — UT 95% CI + P[0,0] variance
    # ══════════════════════════════════════════════════════════════════════
    S.append(PageBreak())
    S += _section("7. Unscented Transform — 95% Confidence Interval on SOC", styles)
    S.append(Paragraph(
        "The Unscented Transform (Julier &amp; Uhlmann 1997) propagates 7 sigma "
        "points through the nonlinear ECM state equations. The resulting ±2σ band "
        "is a non-linear UQ estimate without Jacobian approximation.",
        styles["Body"],
    ))
    S.append(Spacer(1, 0.15 * cm))
    S += _fig_block(fig_ut_ci,
        "Orange shaded: ±2σ UT · Blue: DFN truth · Red dashed: AEKF",
        styles)

    S += _section("8. SOC Variance P[0,0] — UT vs EKF", styles)
    S.append(Paragraph(
        "Lower variance = more confident. Both estimates should decrease and "
        "plateau after the first cycle as the filter converges.",
        styles["Body"],
    ))
    S.append(Spacer(1, 0.15 * cm))
    S += _fig_block(fig_ut_pvar,
        "Both should converge after first few cycles.",
        styles, h=6.0)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 5 — NIS + Innovation
    # ══════════════════════════════════════════════════════════════════════
    S.append(PageBreak())
    S += _section("9. NIS Time-Series — Filter Consistency", styles)
    S.append(Paragraph(
        "NIS ~ χ²(1) for a well-calibrated filter. E[NIS] = 1.0 is ideal. "
        "Green band = 95% confidence bounds [0.004, 5.024] "
        "(Bar-Shalom 2001, Table B.1). "
        "Sustained NIS &gt; 5 → filter over-confident (Q/R too small). "
        "Sustained NIS &lt; 0.004 → filter under-confident (Q/R too large).",
        styles["Body"],
    ))
    S.append(Spacer(1, 0.15 * cm))
    S += _fig_block(fig_nis,
        "Raw NIS (faint) · 50-step moving average (solid) · Green band: χ²(1) 95%",
        styles)

    S += _section("10. Innovation Sequence ν = y − ŷ", styles)
    S.append(Paragraph(
        "The innovation should be zero-mean and white (uncorrelated). "
        "Systematic drift indicates model mismatch; "
        "high variance indicates under-estimated measurement noise.",
        styles["Body"],
    ))
    S.append(Spacer(1, 0.15 * cm))
    S += _fig_block(fig_innov,
        "Innovation ν in mV. Should fluctuate symmetrically around zero.",
        styles, h=6.0)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 6 — Per-cycle table + RMSE per cycle + OCV
    # ══════════════════════════════════════════════════════════════════════
    S.append(PageBreak())
    S += _section("11. Per-Cycle Uncertainty Propagation Statistics", styles)
    S.append(Paragraph(
        "Increasing RMSE across cycles = error accumulation (model drift). "
        "Flat trend = filter converged. "
        "Gap between RMSE and σ_SOC quantifies EKF over-confidence "
        "(structural model mismatch, not tuning).",
        styles["Body"],
    ))
    S.append(Spacer(1, 0.2 * cm))

    hdr = ["Cycle", "RMSE [%]", "MAE [%]", "Max Err [%]",
           "Mean σ [%]", "Max σ [%]", "NIS", "V RMSE [mV]"]
    tdata = [hdr]
    for r in cycle_stats:
        vals = list(r.values()) if isinstance(r, dict) else list(r)
        row  = []
        for val in vals:
            try:
                row.append(str(val) if isinstance(val, int)
                           else f"{float(val):.3f}")
            except Exception:
                row.append("—")
        tdata.append(row)

    cw_cycle = [1.5*cm, 2.5*cm, 2.2*cm, 2.5*cm, 2.5*cm, 2.3*cm, 1.8*cm, 2.7*cm]
    row_bgs  = [
        ("BACKGROUND", (0, i), (-1, i),
         _ROW_ALT if i % 2 == 0 else colors.white)
        for i in range(1, len(tdata))
    ]
    t_cycle = Table(tdata, colWidths=cw_cycle, repeatRows=1)
    t_cycle.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _HEADER_VIOLET),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("GRID",          (0, 0), (-1, -1), 0.3, _GRID_COLOR),
        ("TOPPADDING",    (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        *row_bgs,
    ]))
    S.append(t_cycle)

    # ── RMSE per cycle figure ──────────────────────────────────────────────
    S += _section("12. Uncertainty Propagation — RMSE & σ per Cycle", styles)
    if fig_uncertainty_prop is not None:
        S += _fig_block(fig_uncertainty_prop,
            "Cyan: SOC RMSE [%] · Red dotted: Mean σ_SOC [%]",
            styles, h=6.5)
    else:
        S.append(Paragraph(
            "Per-cycle uncertainty plot not available — "
            "run with n_cycles ≥ 2 to generate.",
            styles["Sub"],
        ))

    # ── OCV curve ─────────────────────────────────────────────────────────
    S += _section("13. OCV Curve (GITT-derived LUT)", styles)
    S += _fig_block(fig_ocv,
        "Open-Circuit Voltage vs State of Charge — cubic spline interpolation.",
        styles, h=6.5)

    # ══════════════════════════════════════════════════════════════════════
    # Footer
    # ══════════════════════════════════════════════════════════════════════
    S.append(Spacer(1, 0.8 * cm))
    S.append(HRFlowable(width="100%", thickness=0.5,
                        color=colors.HexColor("#e2e8f0")))
    S.append(Spacer(1, 0.2 * cm))
    S.append(Paragraph(
        "BattSim Digital Twin · DFN (PyBaMM) + ECM 2-RC + AEKF · "
        "Ref: Plett (2004) J. Power Sources 134 · "
        "Bar-Shalom et al. (2001) Estimation with Applications · "
        "Julier &amp; Uhlmann (1997) SPIE 3068",
        ParagraphStyle(
            "foot", parent=styles["Normal"],
            fontSize=7, textColor=colors.HexColor("#94a3b8"),
        ),
    ))

    doc.build(story)
    return buf.getvalue()
