from __future__ import annotations

"""
app.py — BattSim Digital Twin · Streamlit UI
=============================================
Machine 1 : DFN (PyBaMM)
Machine 2 : ECM 2-RC + AEKF
UQ Layer  : Unscented Transform + PCRLB + NIS
"""

import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.chemistry     import build_chem, make_ocv
from src.machine1_dfn  import run_dfn
from src.machine2_ekf  import run_cosim
from src.utils         import (
    downsample, time_to_hours, soc_to_percent,
    summary_dict, per_cycle_stats, nis_calibration,
    fmt_soc, fmt_rmse, fmt_sigma,
)
from src.unscented_uq  import unscented_uq

IS_CLOUD = (
    os.environ.get("STREAMLIT_SHARING_MODE") is not None
    or os.environ.get("HOME") == "/home/appuser"
)

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BattSim Digital Twin",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
  .block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

if IS_CLOUD:
    st.info("☁️ Cloud mode — simulation may be slow on first run.", icon="☁️")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔋 BattSim")
    st.caption("DFN ↔ AEKF Digital Twin | UT-UQ")

    st.header("⚙️ Cell Chemistry")
    chem_all   = build_chem()
    chem_label = st.selectbox("Chemistry", list(chem_all.keys()), index=0)
    chem       = chem_all[chem_label]
    st.caption(chem["desc"])

    st.header("🔄 Simulation")
    n_cycles = st.slider("Number of Cycles", 1, 5, 1)
    c_rate   = st.slider("C-rate (discharge)", 0.1, 3.0, 1.0, 0.1)
    protocol = st.selectbox(
        "Charging Protocol",
        ["cccv", "cc", "hppc"],
        format_func=lambda x: {
            "cccv": "CC-CV (IEC 62660-1)",
            "cc":   "CC/CC (simple)",
            "hppc": "HPPC (USCAR)",
        }[x],
    )

    st.header("📡 Sensor & EKF")
    noise_mv  = st.slider("Sensor Noise σ [mV]", 1, 50, 10)
    noise_std = noise_mv / 1000.0
    p0_scale  = st.select_slider(
        "Initial P₀",
        options=[1e-5, 1e-4, 1e-3, 1e-2],
        value=1e-4,
        format_func=lambda x: "{:.0e}".format(x),
    )
    q_scale = st.slider("Q scale", 0.1, 5.0, 1.0, 0.1)
    r_scale = st.slider("R scale", 0.1, 5.0, 1.0, 0.1)

    st.header("🔺 Unscented Transform UQ")
    run_ut = st.checkbox("Enable UT-UQ", value=True)
    st.caption("7 sigma points · instant · no MC needed")

    st.divider()
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    try:
        with st.spinner("⚙️ Machine 1: Running DFN (PyBaMM)…"):
            t, V, I, soc, T, Q_nom = run_dfn(
                pset_name=chem["pybamm"],
                n_cycles=n_cycles,
                c_rate=c_rate,
                protocol=protocol,
                v_min=chem["v_min"],
                v_max=chem["v_max"],
            )

        with st.spinner("🧠 Machine 2: Running AEKF…"):
            log = run_cosim(
                t=t, V_true=V, I_true=I, soc_true=soc, T_true=T,
                Q_nom=Q_nom, chem=chem,
                noise_std=noise_std,
                p0_scale=p0_scale, q_scale=q_scale, r_scale=r_scale,
                seed=42,
            )

        ut_result = None
        if run_ut:
            with st.spinner("🔺 Unscented Transform UQ…"):
                soc_ut, sigma_ut, ci_up, ci_lo, p_soc_ut = unscented_uq(
                    chem=chem, Q_nom=Q_nom, log=log, noise_std=noise_std,
                )
                ut_result = dict(
                    soc_ut=soc_ut, sigma_ut=sigma_ut,
                    ci_upper=ci_up, ci_lower=ci_lo,
                    p_soc=p_soc_ut,
                )

        st.session_state.results = dict(
            t=t, V=V, I=I, soc=soc, T=T, Q_nom=Q_nom,
            log=log, ut=ut_result, chem=chem, n_cycles=n_cycles,
        )
        st.success("✅ Simulation complete.")

    except Exception as e:
        st.error("Simulation failed: {}".format(e))
        st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
res = st.session_state.results
if res is None:
    st.info("Configure parameters in the sidebar, then click **▶ Run Simulation**.")
    st.stop()

log      = res["log"]
t_h      = time_to_hours(log["t"])
chem     = res["chem"]
n_cycles = res["n_cycles"]
COLOR    = chem["color"]
smry     = summary_dict(log)
ut       = res.get("ut")

DS = 2000
def ds(arr): return downsample(np.asarray(arr), DS)
t_ds = ds(t_h)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔺 UT Uncertainty",
    "📐 NIS & Calibration",
    "📋 Per-Cycle Stats",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Summary Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("SOC RMSE",    fmt_rmse(smry["rmse_soc_pct"]))
    c2.metric("SOC MAE",     fmt_rmse(smry["mae_soc_pct"]))
    c3.metric("Max |Error|", fmt_rmse(smry["max_err_soc_pct"]))
    c4.metric("Mean σ_SOC",  fmt_sigma(smry["mean_sigma_pct"] / 100))
    c5.metric("V RMSE",      "{:.2f} mV".format(smry["rmse_v_mv"]))
    c6.metric("NIS",         "{:.3f}".format(smry["nis_mean"]),
              delta=smry["nis_verdict"], delta_color="off")

    # Voltage
    st.subheader("Terminal Voltage — DFN vs AEKF")
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=t_ds, y=ds(log["V_true"]),
        name="DFN (truth)", line=dict(color=COLOR, width=2)))
    fig_v.add_trace(go.Scatter(x=t_ds, y=ds(log["V_meas"]),
        name="Measured (noisy)", line=dict(color="gray", width=1, dash="dot"), opacity=0.5))
    fig_v.add_trace(go.Scatter(x=t_ds, y=ds(log["V_est"]),
        name="AEKF estimate", line=dict(color="#ff6b6b", width=2, dash="dash")))
    fig_v.update_layout(xaxis_title="Time [h]", yaxis_title="Voltage [V]",
        legend=dict(orientation="h"), height=350, template="plotly_dark")
    st.plotly_chart(fig_v, use_container_width=True)

    # SOC
    st.subheader("SOC Tracking")
    soc_t = soc_to_percent(ds(log["soc_true"]))
    soc_e = soc_to_percent(ds(log["soc_est"]))
    sig_e = soc_to_percent(ds(log["sigma_soc"]))

    fig_soc = go.Figure()
    fig_soc.add_trace(go.Scatter(
        x=np.concatenate([t_ds, t_ds[::-1]]),
        y=np.concatenate([soc_e + 2*sig_e, (soc_e - 2*sig_e)[::-1]]),
        fill="toself", fillcolor="rgba(255,107,107,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="±2σ EKF"))
    fig_soc.add_trace(go.Scatter(x=t_ds, y=soc_t,
        name="DFN (truth)", line=dict(color=COLOR, width=2)))
    fig_soc.add_trace(go.Scatter(x=t_ds, y=soc_e,
        name="AEKF", line=dict(color="#ff6b6b", width=2, dash="dash")))
    if ut is not None:
        soc_ut_pct = soc_to_percent(ds(ut["soc_ut"]))
        fig_soc.add_trace(go.Scatter(x=t_ds, y=soc_ut_pct,
            name="UT estimate", line=dict(color="#f4a261", width=2, dash="dot")))
    fig_soc.update_layout(xaxis_title="Time [h]", yaxis_title="SOC [%]",
        legend=dict(orientation="h"), height=380, template="plotly_dark")
    st.plotly_chart(fig_soc, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        fig_T = go.Figure(go.Scatter(x=t_ds, y=ds(log["T_true"]),
            line=dict(color="#f4a261"), fill="tozeroy",
            fillcolor="rgba(244,162,97,0.1)"))
        fig_T.update_layout(title="Cell Temperature [°C]",
            xaxis_title="Time [h]", yaxis_title="T [°C]",
            height=260, template="plotly_dark")
        st.plotly_chart(fig_T, use_container_width=True)
    with col_r:
        fig_I = go.Figure(go.Scatter(x=t_ds, y=ds(log["I_true"]),
            line=dict(color="#48cae4")))
        fig_I.update_layout(title="Current [A]",
            xaxis_title="Time [h]", yaxis_title="I [A]",
            height=260, template="plotly_dark")
        st.plotly_chart(fig_I, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — UT Uncertainty
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    if ut is None:
        st.info("Enable UT-UQ in the sidebar and re-run.")
        st.stop()

    st.subheader("Unscented Transform — SOC Uncertainty Propagation")
    st.caption(
        "7 sigma points propagated through the nonlinear ECM state equations. "
        "Compare UT σ vs EKF σ — if close → EKF linearisation is valid."
    )

    soc_ut_p  = soc_to_percent(ds(ut["soc_ut"]))
    sig_ut_p  = soc_to_percent(ds(ut["sigma_ut"]))
    ci_up_p   = soc_to_percent(ds(ut["ci_upper"]))
    ci_lo_p   = soc_to_percent(ds(ut["ci_lower"]))
    soc_tr_p  = soc_to_percent(ds(log["soc_true"]))

    # ± 2σ UT band
    fig_ut = go.Figure()
    fig_ut.add_trace(go.Scatter(
        x=np.concatenate([t_ds, t_ds[::-1]]),
        y=np.concatenate([ci_up_p, ci_lo_p[::-1]]),
        fill="toself", fillcolor="rgba(244,162,97,0.18)",
        line=dict(color="rgba(0,0,0,0)"), name="±2σ UT (95% CI)"))
    fig_ut.add_trace(go.Scatter(x=t_ds, y=soc_tr_p,
        name="DFN Truth", line=dict(color=COLOR, width=2)))
    fig_ut.add_trace(go.Scatter(x=t_ds, y=soc_ut_p,
        name="UT SOC estimate", line=dict(color="#f4a261", width=2)))
    fig_ut.add_trace(go.Scatter(x=t_ds, y=soc_to_percent(ds(log["soc_est"])),
        name="AEKF SOC estimate", line=dict(color="#ff6b6b", width=2, dash="dash")))
    fig_ut.update_layout(xaxis_title="Time [h]", yaxis_title="SOC [%]",
        legend=dict(orientation="h"), height=420, template="plotly_dark",
        title="UT 95% Confidence Interval on SOC")
    st.plotly_chart(fig_ut, use_container_width=True)

    # σ comparison: UT vs EKF
    st.subheader("σ_SOC: UT vs EKF — Linearisation Validity Check")
    st.caption(
        "If UT σ ≈ EKF σ → EKF linearisation (Jacobian) is valid for this OCV curve. "
        "If UT σ > EKF σ → EKF underestimates uncertainty (over-confident)."
    )
    sig_ekf_p = soc_to_percent(ds(log["sigma_soc"]))

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(x=t_ds, y=sig_ekf_p,
        name="EKF σ_SOC (Jacobian)", line=dict(color="#ff6b6b", width=2)))
    fig_sig.add_trace(go.Scatter(x=t_ds, y=sig_ut_p,
        name="UT σ_SOC (7 points)", line=dict(color="#f4a261", width=2, dash="dot")))
    fig_sig.update_layout(xaxis_title="Time [h]", yaxis_title="σ_SOC [%]",
        legend=dict(orientation="h"), height=320, template="plotly_dark")
    st.plotly_chart(fig_sig, use_container_width=True)

    # Metrics
    from src.utils import rmse as _rmse, mae as _mae
    soc_tr_full = soc_to_percent(log["soc_true"])
    soc_ut_full = soc_to_percent(ut["soc_ut"])
    soc_ek_full = soc_to_percent(log["soc_est"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("UT RMSE SOC",  "{:.2f} %".format(_rmse(soc_tr_full, soc_ut_full)))
    col2.metric("EKF RMSE SOC", "{:.2f} %".format(_rmse(soc_tr_full, soc_ek_full)))
    col3.metric("UT Mean σ",    "{:.2f} %".format(float(np.mean(soc_to_percent(ut["sigma_ut"])))))
    col4.metric("EKF Mean σ",   "{:.2f} %".format(float(np.mean(soc_to_percent(log["sigma_soc"])))))

    # P_SOC over time
    st.subheader("SOC Variance P[0,0] — UT vs EKF")
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=t_ds, y=ds(ut["p_soc"]),
        name="UT P_SOC", line=dict(color="#f4a261", width=2)))
    fig_p.add_trace(go.Scatter(x=t_ds, y=ds(log["P_soc"]),
        name="EKF P_SOC", line=dict(color="#ff6b6b", width=2, dash="dash")))
    fig_p.update_layout(xaxis_title="Time [h]", yaxis_title="P_SOC [dimensionless²]",
        legend=dict(orientation="h"), height=300, template="plotly_dark",
        title="SOC Variance — lower = more confident")
    st.plotly_chart(fig_p, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — NIS & Calibration
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("NIS Time-Series — Filter Consistency")
    st.caption(
        "NIS ~ χ²(1) for a well-calibrated filter. "
        "Expected value = 1.0. 95% bounds: [0.004, 5.024]"
    )
    fig_nis = go.Figure()
    fig_nis.add_trace(go.Scatter(x=t_ds, y=ds(log["NIS"]),
        name="NIS", line=dict(color=COLOR, width=1), opacity=0.6))
    fig_nis.add_trace(go.Scatter(
        x=t_ds, y=downsample(
            np.convolve(log["NIS"], np.ones(50)/50, mode="same"), DS),
        name="NIS (50-step avg)", line=dict(color="#ff6b6b", width=2)))
    fig_nis.add_hrect(y0=0.004, y1=5.024,
        fillcolor="rgba(45,198,83,0.08)", line_width=0,
        annotation_text="χ²(1) 95% CI")
    fig_nis.add_hline(y=1.0, line_dash="dash", line_color="white",
        annotation_text="ideal NIS=1")
    fig_nis.update_layout(xaxis_title="Time [h]", yaxis_title="NIS",
        height=360, template="plotly_dark")
    st.plotly_chart(fig_nis, use_container_width=True)

    # Calibration summary
    nis_stats = nis_calibration(log["NIS"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean NIS",      "{:.3f}".format(nis_stats["mean_nis"]))
    col2.metric("% inside 95% CI", "{:.1f} %".format(nis_stats["pct_in_band"]))
    col3.metric("Verdict", nis_stats["verdict"])

    # Innovation time-series
    st.subheader("Innovation ν = y − ŷ")
    fig_nu = go.Figure()
    fig_nu.add_trace(go.Scatter(x=t_ds, y=ds(log["innov"]) * 1000,
        name="Innovation [mV]", line=dict(color="#48cae4", width=1)))
    fig_nu.add_hline(y=0, line_dash="dash", line_color="white")
    fig_nu.update_layout(xaxis_title="Time [h]", yaxis_title="ν [mV]",
        height=280, template="plotly_dark")
    st.plotly_chart(fig_nu, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Per-Cycle Stats
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    import pandas as pd
    st.subheader("Per-Cycle EKF Performance")
    cycle_stats = per_cycle_stats(log, n_cycles=n_cycles)
    df = pd.DataFrame(cycle_stats)
    df.columns = ["Cycle","RMSE SOC [%]","MAE SOC [%]","Max |Err| [%]",
                  "Mean σ [%]","Max σ [%]","Mean NIS","RMSE V [mV]"]
    st.dataframe(
        df.round(3).style
          .background_gradient(subset=["RMSE SOC [%]"], cmap="RdYlGn_r")
          .background_gradient(subset=["Mean NIS"], cmap="RdYlGn_r", vmin=0.5, vmax=2.0),
        use_container_width=True, hide_index=True,
    )

    st.subheader("OCV Curve (GITT-derived LUT)")
    fig_ocv = go.Figure(go.Scatter(
        x=[s*100 for s in chem["soc_lut"]], y=chem["ocv_lut"],
        mode="lines+markers", line=dict(color=COLOR, width=2), marker=dict(size=4)))
    fig_ocv.update_layout(xaxis_title="SOC [%]", yaxis_title="OCV [V]",
        height=300, template="plotly_dark",
        title="OCV — {}".format(chem_label))
    st.plotly_chart(fig_ocv, use_container_width=True)

    st.subheader("2-RC ECM Parameters")
    ecm_cols = st.columns(5)
    for col, (k, v) in zip(ecm_cols, {
        "R₀ [Ω]": chem["R0"], "R₁ [Ω]": chem["R1"], "C₁ [F]": chem["C1"],
        "R₂ [Ω]": chem["R2"], "C₂ [F]": chem["C2"],
    }.items()):
        col.metric(k, "{:.4f}".format(v) if v < 10 else "{:.1f}".format(v))
