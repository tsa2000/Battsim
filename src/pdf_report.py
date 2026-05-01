from __future__ import annotations

"""
src/pdf_report.py — Full PDF Report Generator
===============================================
يُصدِّر كل الرسوم التي ينتجها app.py في تقرير PDF كامل.
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


def _fig_to_img(fig, w=17.0, h=8.5):
    """Plotly → PNG عبر matplotlib (بدون kaleido/Chrome)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # حوّل Plotly → HTML → PNG عبر matplotlib
    # الطريقة: ارسم البيانات مباشرة عبر matplotlib
    buf = io.BytesIO()

    fig_mpl, ax = plt.subplots(figsize=(w/2.54, h/2.54), dpi=150)

    for trace in fig.data:
        try:
            x = list(trace.x) if trace.x is not None else []
            y = list(trace.y) if trace.y is not None else []
            name = trace.name or ""
            ls = "--" if getattr(trace.line, "dash", None) in ("dash","dot") else "-"
            ax.plot(x, y, label=name, linewidth=1.5, linestyle=ls)
        except Exception:
            pass

    ax.set_xlabel(fig.layout.xaxis.title.text or "")
    ax.set_ylabel(fig.layout.yaxis.title.text or "")
    title = fig.layout.title.text or ""
    if "<br>" in title:
        title = title.split("<br>")[0]
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig_mpl)
    buf.seek(0)
    return Image(buf, width=w*cm, height=h*cm)

def _section(title, styles):
    return [
        Spacer(1, 0.35*cm),
        Paragraph(title, styles["Heading2"]),
        HRFlowable(width="100%", thickness=0.5,
                   color=colors.HexColor("#6366f1")),
        Spacer(1, 0.15*cm),
    ]


def _fig_block(fig, caption, styles, w=17.0, h=8.5):
    items = [_fig_to_img(fig, w, h)]
    if caption:
        items.append(Paragraph(
            f"<i>{caption}</i>",
            ParagraphStyle("cap", parent=styles["Normal"],
                           fontSize=8, textColor=colors.HexColor("#64748b"),
                           spaceBefore=2, spaceAfter=6),
        ))
    return items


def build_pdf_report(
    smry, cycle_stats, chem_label, chem,
    n_cycles, protocol, c_rate, noise_mv,
    # ── كل الـ figures من app.py ──────────────────────────────────────────
    fig_voltage,   # Tab1: Voltage DFN vs AEKF
    fig_soc,       # Tab1: SOC Tracking + ±2σ + UT
    fig_temp,      # Tab1: Cell Temperature
    fig_current,   # Tab1: Current profile
    fig_ut_ci,     # Tab2: UT 95% CI
    fig_ut_sigma,  # Tab2: σ_SOC UT vs EKF
    fig_ut_pvar,   # Tab2: P[0,0] variance
    fig_nis,       # Tab3: NIS time-series
    fig_innov,     # Tab3: Innovation ν
    fig_ocv,       # Tab4: OCV curve
    fig_uncertainty_prop = None,
) -> bytes:

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleMain", parent=styles["Title"],
        fontSize=22, textColor=colors.HexColor("#6366f1"), spaceAfter=4))
    styles.add(ParagraphStyle("Sub", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#94a3b8"), spaceAfter=3))
    styles.add(ParagraphStyle("Body", parent=styles["Normal"],
        fontSize=9.5, leading=14))

    S = story = []

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 1 — Cover + Summary
    # ══════════════════════════════════════════════════════════════════════
    S.append(Spacer(1, 1*cm))
    S.append(Paragraph("🔋 BattSim Digital Twin", styles["TitleMain"]))
    S.append(Paragraph(
        "Full Simulation Report — DFN (PyBaMM) + ECM 2-RC + AEKF + Unscented Transform UQ",
        styles["Sub"]))
    S.append(HRFlowable(width="100%", thickness=1.5,
                        color=colors.HexColor("#6366f1")))
    S.append(Spacer(1, 0.3*cm))
    now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    S.append(Paragraph(
        f"Generated: <b>{now}</b> &nbsp;|&nbsp; "
        f"Chemistry: <b>{chem_label}</b> &nbsp;|&nbsp; "
        f"Protocol: <b>{protocol.upper()}</b> &nbsp;|&nbsp; "
        f"Cycles: <b>{n_cycles}</b> &nbsp;|&nbsp; "
        f"C-rate: <b>{c_rate:.1f}C</b> &nbsp;|&nbsp; "
        f"Noise: <b>{noise_mv:.0f} mV</b>",
        styles["Sub"]))
    S.append(Spacer(1, 0.5*cm))

    # Summary metrics table
    S += _section("1. Summary Metrics", styles)
    data = [
        ["Metric", "Value", "Interpretation"],
        ["SOC RMSE",        f"{smry['rmse_soc_pct']:.3f} %",
         "< 2% excellent · 2–5% good · > 5% poor"],
        ["SOC MAE",         f"{smry['mae_soc_pct']:.3f} %",  "—"],
        ["Max |Error| SOC", f"{smry['max_err_soc_pct']:.3f} %", "Worst-case error"],
        ["Mean σ_SOC (EKF)",f"{smry['mean_sigma_pct']:.3f} %",
         "Analytical confidence band"],
        ["V RMSE",          f"{smry['rmse_v_mv']:.2f} mV",
         "< 10 mV excellent"],
        ["NIS (mean)",      f"{smry['nis_mean']:.4f}",
         "≈ 1.0 = well-calibrated filter"],
        ["NIS verdict",     smry["nis_verdict"], "—"],
    ]
    t = Table(data, colWidths=[4.5*cm, 3.2*cm, 8.8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#6366f1")),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",    (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
    ]))
    S.append(t)

    # Sim settings
    S += _section("2. Simulation Settings", styles)
    data2 = [
        ["Parameter", "Value"],
        ["Chemistry",        chem_label],
        ["PyBaMM param set", chem.get("pybamm", "—")],
        ["Protocol",         protocol.upper()],
        ["C-rate",           f"{c_rate:.1f} C"],
        ["Cycles",           str(n_cycles)],
        ["Sensor noise σ",   f"{noise_mv:.0f} mV"],
        ["V_min / V_max",    f"{chem['v_min']:.2f} V / {chem['v_max']:.2f} V"],
        ["ECM topology",     "2-RC Thevenin"],
        ["UQ method",        "Unscented Transform — 7 sigma points"],
        ["References",
         "Plett (2004) J.Power Sources 134 · Julier & Uhlmann (1997)"],
    ]
    t2 = Table(data2, colWidths=[5*cm, 11.5*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#334155")),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",    (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
    ]))
    S.append(t2)
    S.append(Paragraph(
        "* Global RMSE computed over all timesteps across all cycles. "
        "Per-cycle breakdown shown in Section 11.",
        styles["Sub"]
    ))


    # ══════════════════════════════════════════════════════════════════════
    # PAGE 2 — Tab 1: Overview
    # ══════════════════════════════════════════════════════════════════════
    S.append(PageBreak())
    S += _section("3. Terminal Voltage — DFN vs AEKF", styles)
    S += _fig_block(fig_voltage, "DFN (truth) · Measured (noisy) · AEKF estimate", styles)

    S += _section("4. SOC Tracking — DFN Truth vs AEKF + UT", styles)
    S += _fig_block(fig_soc,
        "Blue: DFN truth · Red dashed: AEKF · Orange dotted: UT · Shaded: ±2σ EKF",
        styles, h=8.0)

    S.append(PageBreak())
    S += _section("5. Cell Temperature & Current Profile", styles)
    # side by side
    img_T = _fig_to_img(fig_temp,    w=8.3, h=6.0)
    img_I = _fig_to_img(fig_current, w=8.3, h=6.0)
    S.append(Table([[img_T, img_I]],
                   colWidths=[8.3*cm, 8.3*cm],
                   style=[("VALIGN",(0,0),(-1,-1),"TOP"),
                          ("LEFTPADDING",(0,0),(-1,-1),0),
                          ("RIGHTPADDING",(0,0),(-1,-1),6)]))

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 3 — Tab 2: UT Uncertainty
    # ══════════════════════════════════════════════════════════════════════
    S.append(PageBreak())
    S += _section("6. Unscented Transform — 95% CI on SOC", styles)
    S.append(Paragraph(
        "The UT propagates 7 sigma points through the nonlinear ECM equations "
        "(Julier &amp; Uhlmann 1997). Shaded area = 95% confidence interval. "
        "All DFN truth values should lie within the ±2σ band.",
        styles["Body"]))
    S.append(Spacer(1, 0.2*cm))
    S += _fig_block(fig_ut_ci, "Orange shaded: ±2σ UT · Blue: DFN truth · Orange: UT · Red dashed: AEKF", styles)

    S += _section("7. σ_SOC — UT vs EKF Linearisation Check", styles)
    S.append(Paragraph(
        "If UT σ ≈ EKF σ → the EKF Jacobian linearisation is valid. "
        "If UT σ &gt; EKF σ → EKF underestimates uncertainty (over-confident).",
        styles["Body"]))
    S.append(Spacer(1, 0.2*cm))
    S += _fig_block(fig_ut_sigma, "Red: EKF σ (Jacobian) · Orange dotted: UT σ (7 sigma points)", styles, h=6.5)

    S += _section("8. SOC Variance P[0,0] — UT vs EKF", styles)
    S += _fig_block(fig_ut_pvar, "Lower = more confident. Both should converge after first few cycles.", styles, h=6.0)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 4 — Tab 3: NIS & Calibration
    # ══════════════════════════════════════════════════════════════════════
    S.append(PageBreak())
    S += _section("9. NIS Time-Series — Filter Consistency", styles)
    S.append(Paragraph(
        "NIS ~ χ²(1) for a well-calibrated filter. E[NIS] = 1.0 is ideal. "
        "Green band = 95% confidence bounds [0.004, 5.024]. "
        "Consistent NIS ≈ 1.0 confirms the AEKF neither over- nor under-estimates its uncertainty.",
        styles["Body"]))
    S.append(Spacer(1, 0.2*cm))
    S += _fig_block(fig_nis, "Raw NIS (faint) · 50-step moving average (solid) · Green: χ²(1) 95% band", styles)

    S += _section("10. Innovation Sequence ν = y − ŷ", styles)
    S.append(Paragraph(
        "The innovation should be zero-mean and white. "
        "Systematic drift indicates model mismatch; high variance indicates "
        "measurement noise underestimation.",
        styles["Body"]))
    S.append(Spacer(1, 0.2*cm))
    S += _fig_block(fig_innov, "Innovation in mV. Should fluctuate around zero.", styles, h=6.0)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 5 — Tab 4: Per-Cycle + OCV
    # ══════════════════════════════════════════════════════════════════════
    S.append(PageBreak())
    S += _section("11. Per-Cycle EKF Statistics", styles)

    hdr = ["Cycle","RMSE SOC [%]","MAE SOC [%]","Max Err [%]",
           "Mean σ [%]","Max σ [%]","NIS","RMSE V [mV]"]
    tdata = [hdr]
    for r in cycle_stats:
        row = []
        # cycle_stats قد يكون list أو dict أو tuple
        if isinstance(r, dict):
            vals = list(r.values())
        else:
            vals = list(r)
        for val in vals:
            try:
                if isinstance(val, int):
                    row.append(str(val))
                elif isinstance(val, (float, np.floating)):
                    row.append(f"{float(val):.3f}")
                else:
                    row.append(str(val))
            except Exception:
                row.append("—")
        tdata.append(row)



    cw = [1.5*cm, 2.7*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.3*cm, 1.8*cm, 2.7*cm]
    row_bgs = [("BACKGROUND",(0,i),(-1,i),
                colors.HexColor("#f8fafc") if i%2==0 else colors.white)
               for i in range(1, len(tdata))]
    t3 = Table(tdata, colWidths=cw, repeatRows=1)
    t3.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#6366f1")),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 7.5),
        ("ALIGN",         (1,0),(-1,-1), "CENTER"),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",    (0,0),(-1,-1), 2.5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 2.5),
        *row_bgs,
    ]))
    S.append(t3)
    
    S += _section("12. Uncertainty Propagation — RMSE per Cycle", styles)
    S.append(Paragraph(
        "SOC RMSE and σ_SOC per cycle. Increasing RMSE → error accumulation. "
        "Flat trend → filter converged. Gap between RMSE and σ quantifies EKF over-confidence.",
        styles["Body"]))
    S.append(Spacer(1, 0.2*cm))
    if fig_uncertainty_prop is not None:
        S += _fig_block(fig_uncertainty_prop,
            "Cyan: SOC RMSE [%] · Red dotted: Mean σ_SOC [%]", styles, h=6.5)

    S += _section("13. OCV Curve (GITT-derived LUT)", styles)
    S += _fig_block(fig_ocv, "Open-Circuit Voltage vs State of Charge", styles, h=6.5)

    # ══════════════════════════════════════════════════════════════════════
    # Footer
    # ══════════════════════════════════════════════════════════════════════
    S.append(Spacer(1, 0.8*cm))
    S.append(HRFlowable(width="100%", thickness=0.5,
                        color=colors.HexColor("#e2e8f0")))
    S.append(Spacer(1, 0.2*cm))
    S.append(Paragraph(
        "BattSim Digital Twin · DFN (PyBaMM) + ECM 2-RC + AEKF + Unscented Transform UQ · "
        "Ref: Plett (2004) · Julier &amp; Uhlmann (1997) · Wan &amp; van der Merwe (2000)",
        ParagraphStyle("foot", parent=styles["Normal"],
                       fontSize=7, textColor=colors.HexColor("#94a3b8")),
    ))

    doc.build(story)
    return buf.getvalue()
