from __future__ import annotations

"""
app.py — BattSim Digital Twin · Streamlit UI
"""

import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.chemistry    import build_chem, make_ocv
from src.machine1_dfn import run_dfn
from src.machine2_ekf import run_cosim
from src.utils        import (
    downsample, time_to_hours, soc_to_percent,
    summary_dict, per_cycle_stats, nis_calibration,
    fmt_soc, fmt_rmse, fmt_sigma,
)

try:
    from src.mc_uq import run_mc_ekf, compute_pcrlb, anees_per_cycle
    _MC_AVAILABLE = True
except ImportError:
    _MC_AVAILABLE = False

IS_CLOUD = (
    os.environ.get("STREAMLIT_SHARING_MODE") is not None
    or os.environ.get("HOME") == "/home/appuser"
)


# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BattSim Digital Twin",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark-theme style injection ────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
  .block-container { padding-top: 1.2rem; }
  div[data-testid="column"] { padding: 0.3rem 0.6rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔋 BattSim")
    st.caption("DFN ↔ AEKF Digital Twin | UQ Module")

    st.header("⚙️ Cell Chemistry")
    chem_all   = build_chem()
    chem_label = st.selectbox(
        "Chemistry / Parameter Set",
        options=list(chem_all.keys()),
        index=0,
    )
    chem = chem_all[chem_label]
    st.caption(chem["desc"])

    st.header("🔄 Simulation")
    n_cycles = st.slider("Number of Cycles", 1, 5, 2)
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
        "Initial P₀ (SOC variance)",
        options=[1e-5, 1e-4, 1e-3, 1e-2],
        value=1e-4,
        format_func=lambda x: f"{x:.0e}",
    )
    q_scale   = st.slider("Q scale (process noise)", 0.1, 5.0, 1.0, 0.1)
    r_scale   = st.slider("R scale (meas. noise)", 0.1, 5.0, 1.0, 0.1)

    st.header("🎲 Monte Carlo UQ")
    run_mc    = st.checkbox("Enable Monte Carlo UQ", value=True)
    n_mc_runs = st.slider("MC runs (N)", 20, 500, 100,
                           disabled=not run_mc)
    run_pcrlb = st.checkbox("Compute PCRLB", value=True,
                             disabled=not run_mc)

    st.divider()
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State — persist results across reruns
# ─────────────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None


# ─────────────────────────────────────────────────────────────────────────────
# Run Simulation
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    try:
        # ── Machine 1: DFN ────────────────────────────────────────────────────
        with st.spinner("⚙️ Machine 1: Running DFN (PyBaMM)…"):
            t, V, I, soc, T, Q_nom = run_dfn(
                pset_name=chem["pybamm"],
                n_cycles=n_cycles,
                c_rate=c_rate,
                protocol=protocol,
                v_min=chem["v_min"],
                v_max=chem["v_max"],
            )

        # ── Machine 2: EKF co-simulation ──────────────────────────────────────
        with st.spinner("🧠 Machine 2: Running AEKF co-simulation…"):
            log = run_cosim(
                t=t, V_true=V, I_true=I, soc_true=soc, T_true=T,
                Q_nom=Q_nom, chem=chem,
                noise_std=noise_std,
                p0_scale=p0_scale, q_scale=q_scale, r_scale=r_scale,
                seed=42,
            )

        mc_result  = None
        pcrlb_res  = None

        # ── MC UQ ─────────────────────────────────────────────────────────────
        if run_mc:
            with st.spinner(f"🎲 Monte Carlo UQ: {n_mc_runs} runs…"):
                mc_result = run_mc_ekf(
                    dfn_log={"t":t,"V_true":V,"I_true":I,
                             "soc_true":soc,"T_true":T},
                    Q_nom=Q_nom, chem=chem,
                    noise_std=noise_std,
                    n_runs=n_mc_runs,
                    p0_scale=p0_scale, q_scale=q_scale, r_scale=r_scale,
                )

        # ── PCRLB ─────────────────────────────────────────────────────────────
        if run_mc and run_pcrlb:
            with st.spinner("📐 Computing PCRLB (Fisher Information)…"):
                ocv_fn    = make_ocv(chem)
                pcrlb_res = compute_pcrlb(
                    log=log, chem=chem, Q_nom=Q_nom,
                    noise_var=noise_std**2, ocv_fn=ocv_fn,
                )

        st.session_state.results = dict(
            t=t, V=V, I=I, soc=soc, T=T, Q_nom=Q_nom,
            log=log, mc=mc_result, pcrlb=pcrlb_res,
            chem=chem, n_cycles=n_cycles,
        )
        st.success("✅ Simulation complete.")

    except Exception as e:
        st.error(f"Simulation failed: {e}")
        st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# Results Display
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

smry = summary_dict(log)


# ════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🎲 Monte Carlo UQ",
    "📐 PCRLB & ANEES",
    "📋 Per-Cycle Stats",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Summary Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("SOC RMSE",      fmt_rmse(smry["rmse_soc_pct"]))
    c2.metric("SOC MAE",       fmt_rmse(smry["mae_soc_pct"]))
    c3.metric("Max |Error|",   fmt_rmse(smry["max_err_soc_pct"]))
    c4.metric("Mean σ_SOC",    fmt_sigma(smry["mean_sigma_pct"]/100))
    c5.metric("V RMSE",        f'{smry["rmse_v_mv"]:.2f} mV')
    c6.metric("NIS (mean)",    f'{smry["nis_mean"]:.3f}',
              delta=smry["nis_verdict"], delta_color="off")

    # ── Voltage plot ──────────────────────────────────────────────────────────
    st.subheader("Terminal Voltage — DFN vs AEKF Estimate")
    DS = 2000  # downsample target
    t_ds = downsample(t_h, DS)

    def ds(arr): return downsample(np.asarray(arr), DS)

    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(
        x=t_ds, y=ds(log["V_true"]),
        name="DFN (truth)", line=dict(color=COLOR, width=2)
    ))
    fig_v.add_trace(go.Scatter(
        x=t_ds, y=ds(log["V_meas"]),
        name="Measured (noisy)", line=dict(color="gray", width=1, dash="dot"),
        opacity=0.5,
    ))
    fig_v.add_trace(go.Scatter(
        x=t_ds, y=ds(log["V_est"]),
        name="AEKF estimate", line=dict(color="#ff6b6b", width=2, dash="dash")
    ))
    fig_v.update_layout(
        xaxis_title="Time [h]", yaxis_title="Voltage [V]",
        legend=dict(orientation="h"), height=350,
        template="plotly_dark",
    )
    st.plotly_chart(fig_v, use_container_width=True)

    # ── SOC tracking ──────────────────────────────────────────────────────────
    st.subheader("SOC Tracking — DFN Truth vs AEKF")
    soc_true_pct = soc_to_percent(ds(log["soc_true"]))
    soc_est_pct  = soc_to_percent(ds(log["soc_est"]))
    sigma_pct    = soc_to_percent(ds(log["sigma_soc"]))

    fig_soc = go.Figure()
    fig_soc.add_trace(go.Scatter(
        x=t_ds, y=soc_true_pct,
        name="DFN SOC (truth)", line=dict(color=COLOR, width=2)
    ))
    # ±2σ confidence band (analytical — from EKF P matrix)
    fig_soc.add_trace(go.Scatter(
        x=np.concatenate([t_ds, t_ds[::-1]]),
        y=np.concatenate([soc_est_pct + 2*sigma_pct,
                          (soc_est_pct - 2*sigma_pct)[::-1]]),
        fill="toself", fillcolor="rgba(255,107,107,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="±2σ (EKF analytical)", showlegend=True,
    ))
    fig_soc.add_trace(go.Scatter(
        x=t_ds, y=soc_est_pct,
        name="AEKF estimate", line=dict(color="#ff6b6b", width=2, dash="dash")
    ))
    fig_soc.update_layout(
        xaxis_title="Time [h]", yaxis_title="SOC [%]",
        legend=dict(orientation="h"), height=350,
        template="plotly_dark",
    )
    st.plotly_chart(fig_soc, use_container_width=True)

    # ── Temperature & Current ─────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        fig_T = go.Figure(go.Scatter(
            x=t_ds, y=ds(log["T_true"]),
            line=dict(color="#f4a261"), fill="tozeroy",
            fillcolor="rgba(244,162,97,0.1)"
        ))
        fig_T.update_layout(
            title="Cell Temperature [°C]",
            xaxis_title="Time [h]", yaxis_title="T [°C]",
            height=280, template="plotly_dark"
        )
        st.plotly_chart(fig_T, use_container_width=True)

    with col_r:
        fig_I = go.Figure(go.Scatter(
            x=t_ds, y=ds(log["I_true"]),
            line=dict(color="#48cae4")
        ))
        fig_I.update_layout(
            title="Current [A]  (+ = discharge)",
            xaxis_title="Time [h]", yaxis_title="I [A]",
            height=280, template="plotly_dark"
        )
        st.plotly_chart(fig_I, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Monte Carlo UQ
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    mc = res["mc"]
    if mc is None:
        st.info("Enable Monte Carlo UQ in the sidebar and re-run.")
        st.stop()

    st.subheader("Monte Carlo SOC Uncertainty Envelope")

    t_ds  = downsample(mc.t / 3600.0, 2000)
    def ds_mc(arr): return downsample(arr, 2000)

    true_pct = soc_to_percent(ds_mc(mc.soc_true))
    mean_pct = soc_to_percent(ds_mc(mc.soc_mean))
    ci95_up  = soc_to_percent(ds_mc(mc.ci_95_upper))
    ci95_lo  = soc_to_percent(ds_mc(mc.ci_95_lower))
    ci68_up  = soc_to_percent(ds_mc(mc.ci_68_upper))
    ci68_lo  = soc_to_percent(ds_mc(mc.ci_68_lower))

    fig_mc = go.Figure()

    # 95% CI band
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([t_ds, t_ds[::-1]]),
        y=np.concatenate([ci95_up, ci95_lo[::-1]]),
        fill="toself", fillcolor="rgba(255,107,107,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI (Monte Carlo)",
    ))
    # 68% CI band
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([t_ds, t_ds[::-1]]),
        y=np.concatenate([ci68_up, ci68_lo[::-1]]),
        fill="toself", fillcolor="rgba(255,107,107,0.25)",
        line=dict(color="rgba(0,0,0,0)"),
        name="68% CI (Monte Carlo)",
    ))
    # Sample trajectories (first 20 runs, faint)
    for i in range(min(20, mc.soc_matrix.shape[0])):
        fig_mc.add_trace(go.Scatter(
            x=t_ds,
            y=soc_to_percent(ds_mc(mc.soc_matrix[i])),
            line=dict(color="rgba(255,107,107,0.08)", width=1),
            showlegend=False,
        ))
    fig_mc.add_trace(go.Scatter(
        x=t_ds, y=mean_pct,
        name="MC Ensemble Mean", line=dict(color="#ff6b6b", width=2)
    ))
    fig_mc.add_trace(go.Scatter(
        x=t_ds, y=true_pct,
        name="DFN Truth", line=dict(color=COLOR, width=2)
    ))
    fig_mc.update_layout(
        xaxis_title="Time [h]", yaxis_title="SOC [%]",
        legend=dict(orientation="h"), height=420,
        template="plotly_dark",
        title=f"MC SOC Distribution  (N={len(mc.seeds)} runs)",
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # ── MC Metrics ────────────────────────────────────────────────────────────
    st.subheader("MC Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ANEES", f"{mc.anees:.3f}",
                delta="≈1.0 = calibrated", delta_color="off")
    col2.metric("Convergence", f"{mc.converge_steps} steps",
                delta=f"≈{mc.converge_steps*10/60:.1f} min", delta_color="off")
    col3.metric("Mean RMSE (ensemble)", f"{np.mean(mc.rmse_ensemble):.2f} %")
    col4.metric("Max RMSE (worst run)", f"{np.max(mc.rmse_ensemble):.2f} %")

    # ── RMSE Distribution ─────────────────────────────────────────────────────
    st.subheader("SOC RMSE Distribution across MC Runs")
    fig_rmse = go.Figure(go.Histogram(
        x=mc.rmse_ensemble, nbinsx=30,
        marker_color=COLOR, opacity=0.8,
        name="RMSE [%]",
    ))
    fig_rmse.add_vline(
        x=float(np.mean(mc.rmse_ensemble)),
        line_dash="dash", line_color="white",
        annotation_text=f"μ={np.mean(mc.rmse_ensemble):.2f}%",
        annotation_position="top right",
    )
    fig_rmse.update_layout(
        xaxis_title="SOC RMSE [%]", yaxis_title="Count",
        height=300, template="plotly_dark",
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

    # ── Empirical σ vs Analytical σ (EKF) ────────────────────────────────────
    st.subheader("Empirical σ (MC) vs Analytical σ (EKF P-matrix)")
    sigma_ekf_pct = soc_to_percent(ds_mc(log["sigma_soc"]))
    sigma_mc_pct  = soc_to_percent(ds_mc(mc.soc_std))

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(
        x=t_ds, y=sigma_ekf_pct,
        name="EKF analytical σ", line=dict(color="#ff6b6b", width=2)
    ))
    fig_sig.add_trace(go.Scatter(
        x=t_ds, y=sigma_mc_pct,
        name="MC empirical σ", line=dict(color="#00b4d8", width=2, dash="dash")
    ))
    fig_sig.update_layout(
        xaxis_title="Time [h]", yaxis_title="σ_SOC [%]",
        legend=dict(orientation="h"), height=300,
        template="plotly_dark",
        title="Calibration Check: If EKF σ ≈ MC σ → filter is consistent",
    )
    st.plotly_chart(fig_sig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PCRLB & ANEES
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    mc    = res["mc"]
    pcrlb = res["pcrlb"]

    if pcrlb is None:
        st.info("Enable 'Compute PCRLB' in the sidebar and re-run.")
    else:
        st.subheader("Posterior Cramér-Rao Lower Bound on SOC Variance")
        st.caption(
            "PCRLB = theoretical minimum achievable Var(SOC). "
            "If P_SOC (EKF) ≈ PCRLB → filter is near-optimal."
        )
        t_ds2 = downsample(pcrlb.t / 3600.0, 2000)

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=t_ds2,
            y=soc_to_percent(downsample(np.sqrt(pcrlb.pcrlb), 2000)),
            name="PCRLB  √[J⁻¹]₀₀ [%]",
            line=dict(color="#f4a261", width=2, dash="dot"),
        ))
        fig_p.add_trace(go.Scatter(
            x=t_ds2,
            y=soc_to_percent(downsample(log["sigma_soc"], 2000)),
            name="EKF σ_SOC [%]",
            line=dict(color="#ff6b6b", width=2),
        ))
        if mc is not None:
            fig_p.add_trace(go.Scatter(
                x=t_ds2,
                y=soc_to_percent(downsample(mc.soc_std, 2000)),
                name="MC empirical σ [%]",
                line=dict(color="#00b4d8", width=2, dash="dash"),
            ))
        fig_p.update_layout(
            xaxis_title="Time [h]", yaxis_title="σ_SOC [%]",
            legend=dict(orientation="h"), height=380,
            template="plotly_dark",
        )
        st.plotly_chart(fig_p, use_container_width=True)

    # ── ANEES per cycle ────────────────────────────────────────────────────────
    if mc is not None:
        st.subheader("ANEES per Cycle — Filter Health over Ageing")
        anees_data = anees_per_cycle(mc, n_cycles=n_cycles)

        fig_anees = go.Figure()
        fig_anees.add_trace(go.Bar(
            x=[f"Cycle {d['cycle']}" for d in anees_data],
            y=[d["anees"] for d in anees_data],
            marker_color=[COLOR] * len(anees_data),
            text=[f"{d['anees']:.3f}" for d in anees_data],
            textposition="outside",
        ))
        fig_anees.add_hline(
            y=1.0, line_dash="dash", line_color="white",
            annotation_text="ANEES = 1 (ideal)",
        )
        fig_anees.update_layout(
            yaxis_title="ANEES", height=320,
            template="plotly_dark",
            title="ANEES ≈ 1 → well-calibrated | >1 → over-confident | <1 → under-confident",
        )
        st.plotly_chart(fig_anees, use_container_width=True)

        # NIS time-series
        st.subheader("NIS Time-series (Filter Consistency)")
        nis_mean_ts = np.mean(mc.nis_matrix, axis=0)
        t_ds3       = downsample(t_h, 2000)

        fig_nis = go.Figure()
        fig_nis.add_trace(go.Scatter(
            x=t_ds3, y=downsample(nis_mean_ts, 2000),
            name="Mean NIS", line=dict(color=COLOR)
        ))
        fig_nis.add_hrect(
            y0=0.004, y1=5.024,
            fillcolor="rgba(45,198,83,0.1)",
            line_width=0,
            annotation_text="χ²(1) 95% bounds [0.004, 5.024]",
        )
        fig_nis.add_hline(y=1.0, line_dash="dash", line_color="white")
        fig_nis.update_layout(
            xaxis_title="Time [h]", yaxis_title="NIS",
            height=300, template="plotly_dark",
        )
        st.plotly_chart(fig_nis, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Per-Cycle Statistics
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Per-Cycle EKF Performance Statistics")
    cycle_stats = per_cycle_stats(log, n_cycles=n_cycles)

    import pandas as pd
    df = pd.DataFrame(cycle_stats)
    df.columns = [
        "Cycle", "RMSE SOC [%]", "MAE SOC [%]", "Max |Err| SOC [%]",
        "Mean σ [%]", "Max σ [%]", "Mean NIS", "RMSE V [mV]",
    ]
    df = df.round(3)

    # Colour-code RMSE column
    st.dataframe(
        df.style.background_gradient(
            subset=["RMSE SOC [%]"], cmap="RdYlGn_r"
        ).background_gradient(
            subset=["Mean NIS"], cmap="RdYlGn_r", vmin=0.5, vmax=2.0
        ),
        use_container_width=True, hide_index=True,
    )

    # ── OCV curve ─────────────────────────────────────────────────────────────
    st.subheader("OCV Curve (LUT — GITT-derived)")
    soc_lut = chem["soc_lut"]
    ocv_lut = chem["ocv_lut"]
    fig_ocv = go.Figure(go.Scatter(
        x=[s * 100 for s in soc_lut], y=ocv_lut,
        mode="lines+markers",
        line=dict(color=COLOR, width=2),
        marker=dict(size=4),
    ))
    fig_ocv.update_layout(
        xaxis_title="SOC [%]", yaxis_title="OCV [V]",
        height=320, template="plotly_dark",
        title=f"OCV(SOC) — {chem_label}",
    )
    st.plotly_chart(fig_ocv, use_container_width=True)

    # ── ECM Parameters display ────────────────────────────────────────────────
    st.subheader("2-RC ECM Parameters")
    ecm_col = st.columns(5)
    for col, (k, v) in zip(ecm_col, {
        "R₀ [Ω]": chem["R0"],
        "R₁ [Ω]": chem["R1"],
        "C₁ [F]": chem["C1"],
        "R₂ [Ω]": chem["R2"],
        "C₂ [F]": chem["C2"],
    }.items()):
        col.metric(k, f"{v:.4f}" if v < 10 else f"{v:.1f}")
