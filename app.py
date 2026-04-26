import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title  = "BattSim v4.2",
    page_icon   = "⚡",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Theme ─────────────────────────────────────────────────────
C_BG     = "#0d1117"
C_SURF   = "#161b22"
C_SURF2  = "#21262d"
C_BORDER = "#30363d"
C_TEXT   = "#e6edf3"
C_MUTED  = "#8b949e"
C_TEAL   = "#00b4d8"
C_GREEN  = "#2dc653"
C_ORANGE = "#f77f00"
C_RED    = "#f85149"
C_PURPLE = "#c77dff"
C_YELLOW = "#e3b341"

BASE_LAYOUT = dict(
    paper_bgcolor = C_BG,
    plot_bgcolor  = C_BG,
    font          = dict(family="monospace", size=11, color=C_TEXT),
    margin        = dict(l=52, r=20, t=44, b=44),
    legend        = dict(bgcolor=C_SURF, bordercolor=C_BORDER,
                         borderwidth=1, font=dict(size=10)),
    xaxis         = dict(gridcolor=C_BORDER, gridwidth=0.5,
                         linecolor=C_BORDER, tickfont=dict(size=10)),
    yaxis         = dict(gridcolor=C_BORDER, gridwidth=0.5,
                         linecolor=C_BORDER, tickfont=dict(size=10)),
)

def ax(title, fmt=None):
    d = dict(title=dict(text=title, font=dict(size=11)),
             gridcolor=C_BORDER, gridwidth=0.5,
             linecolor=C_BORDER, tickfont=dict(size=10))
    if fmt:
        d["tickformat"] = fmt
    return d

# ── CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {{
    background-color: {C_BG} !important;
    color: {C_TEXT} !important;
    font-family: 'Inter', sans-serif;
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: {C_SURF} !important;
    border-right: 1px solid {C_BORDER};
  }}

  /* Header */
  .app-header {{
    background: linear-gradient(135deg, {C_SURF} 0%, #0d1117 100%);
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
  }}
  .app-logo {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: {C_TEAL};
    letter-spacing: -1px;
  }}
  .app-sub {{
    color: {C_MUTED};
    font-size: 0.82rem;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
  }}

  /* Machine boxes */
  .machine-box {{
    background: {C_SURF};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    padding: 16px 20px;
    margin: 14px 0 10px 0;
  }}
  .machine-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.0rem;
    font-weight: 700;
    margin-bottom: 4px;
  }}

  /* KPI cards */
  .kpi-row {{ display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }}
  .kpi-card {{
    background: {C_SURF2};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 14px 18px;
    min-width: 160px;
    flex: 1;
  }}
  .kpi-label {{
    font-size: 0.72rem;
    color: {C_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
  }}
  .kpi-value {{
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
  }}
  .kpi-unit {{
    font-size: 0.72rem;
    color: {C_MUTED};
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
  }}

  /* Assessment flags */
  .flag-error   {{ background:#2d1217; border-left:3px solid {C_RED};    padding:10px 14px; border-radius:6px; margin:6px 0; font-size:0.84rem; }}
  .flag-warning {{ background:#2a1f0e; border-left:3px solid {C_YELLOW}; padding:10px 14px; border-radius:6px; margin:6px 0; font-size:0.84rem; }}
  .flag-ok      {{ background:#0d1f12; border-left:3px solid {C_GREEN};  padding:10px 14px; border-radius:6px; margin:6px 0; font-size:0.84rem; }}

  /* Dividers */
  .section-divider {{
    border: none;
    border-top: 1px solid {C_BORDER};
    margin: 28px 0;
  }}

  /* Tables */
  .dataframe {{ background: {C_SURF} !important; }}

  /* Buttons */
  .stButton > button {{
    background: {C_TEAL} !important;
    color: #000 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 8px 20px !important;
  }}
  .stButton > button:hover {{
    background: #00c8f0 !important;
  }}

  /* Slider labels */
  .stSlider label, .stSelectbox label {{
    color: {C_MUTED} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  /* Section headers */
  h3 {{ color: {C_TEXT} !important; font-family: 'JetBrains Mono', monospace; font-size: 0.95rem !important; }}
</style>
""", unsafe_allow_html=True)

# ── Imports (lazy — only when simulation runs) ────────────────
@st.cache_resource(show_spinner="Loading BattSim modules…")
def load_modules():
    from src.chemistry    import build_chem, make_ocv, docv_dsoc
    from src.machine1_dfn import run_dfn
    from src.machine2_ekf import EKF, run_cosim
    from src.machine3_pinn import run_pinn
    from src.utils         import compute_metrics, cycle_table, ekf_assessment
    return (build_chem, make_ocv, docv_dsoc,
            run_dfn, run_cosim, run_pinn,
            compute_metrics, cycle_table, ekf_assessment)

(build_chem, make_ocv, docv_dsoc,
 run_dfn, run_cosim, run_pinn,
 compute_metrics, cycle_table, ekf_assessment) = load_modules()

@st.cache_data(show_spinner="Building chemistry database from PyBaMM…")
def get_chem():
    return build_chem()

CHEM = get_chem()

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <div>
    <div class="app-logo">⚡ BattSim v4.2</div>
    <div class="app-sub">
      DFN Physical Asset  ↔  ECM+EKF Digital Observer  ↔  PINN Residual Corrector<br>
      PyBaMM · Chen2020 · Prada2013 · OKane2022 · Plett2004 · Nature Comms 2026
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<div style='font-family:JetBrains Mono;font-size:0.72rem;color:{C_MUTED};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:16px'>Configuration</div>", unsafe_allow_html=True)

    chem_name = st.selectbox("Cell Chemistry", list(CHEM.keys()))
    chem      = CHEM[chem_name]
    cc        = chem["color"]

    st.markdown("<hr style='border-color:#30363d;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:JetBrains Mono;font-size:0.72rem;color:{C_MUTED};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px'>Simulation</div>", unsafe_allow_html=True)

    n_cycles  = st.slider("Cycles",          3,  50,  20)
    c_rate    = st.select_slider("C-Rate",
        options=[0.5, 1.0, 1.5, 2.0, 3.0], value=1.0)
    protocol  = st.selectbox("Protocol",
        ["cc", "cccv", "hppc"],
        format_func=lambda x: {"cc":"Constant Current","cccv":"CC-CV","hppc":"HPPC"}[x])
    noise_mv  = st.select_slider("Sensor Noise σ [mV]",
        options=[5, 10, 20, 30, 50], value=10)

    st.markdown("<hr style='border-color:#30363d;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:JetBrains Mono;font-size:0.72rem;color:{C_MUTED};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px'>EKF Tuning</div>", unsafe_allow_html=True)

    p0_scale = st.select_slider("P₀ scale",
        options=[1e-6,1e-5,1e-4,1e-3,1e-2], value=1e-4,
        format_func=lambda x: f"{x:.0e}")
    q_scale  = st.select_slider("Q scale",
        options=[0.01,0.1,0.5,1.0,2.0,5.0], value=0.1)
    r_scale  = st.select_slider("R scale",
        options=[0.1,0.5,1.0,2.0,5.0,10.0], value=2.0)

    st.markdown("<hr style='border-color:#30363d;margin:12px 0'>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Simulation", use_container_width=True)

    st.markdown("<hr style='border-color:#30363d;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.68rem;color:{C_MUTED};font-family:JetBrains Mono'>"
                f"v_min {chem['v_min']} V · v_max {chem['v_max']} V<br>"
                f"Q = {chem['Q']} Ah · R0 = {chem['R0']*1000:.1f} mΩ<br>"
                f"<span style='color:{C_MUTED}'>{chem['desc']}</span>"
                f"</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# RUN SIMULATION
# ════════════════════════════════════════════════════════════════
if run_btn:
    noise_std = noise_mv / 1000.0

    with st.spinner("Machine 1 — Running PyBaMM DFN…"):
        t, V, I, soc, T, Q_nom = run_dfn(
            pset_name = chem["pybamm"],
            n_cycles  = n_cycles,
            c_rate    = c_rate,
            protocol  = protocol,
            v_min     = chem["v_min"],
            v_max     = chem["v_max"],
        )

    with st.spinner("Machine 2 — Running ECM + Adaptive EKF…"):
        log = run_cosim(
            t=t, V_true=V, I_true=I, soc_true=soc,
            T_true=T, Q_nom=Q_nom, chem=chem,
            noise_std=noise_std,
            p0_scale=p0_scale, q_scale=q_scale, r_scale=r_scale,
        )

    metrics = compute_metrics(log)
    cycles  = cycle_table(log)
    flags   = ekf_assessment(metrics)

    st.session_state.update(dict(
        t=t, V=V, I=I, soc=soc, T=T, Q_nom=Q_nom,
        log=log, metrics=metrics, cycles=cycles, flags=flags,
        chem=chem, cc=cc, chem_name=chem_name,
        noise_std=noise_std, n_cycles=n_cycles, c_rate=c_rate,
    ))
    st.session_state.pop("pinn_done", None)

# ════════════════════════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════════════════════════
if "log" not in st.session_state:
    st.markdown(f"""
    <div style='text-align:center;padding:80px 0;color:{C_MUTED}'>
      <div style='font-size:3rem'>⚡</div>
      <div style='font-family:JetBrains Mono;font-size:0.9rem;margin-top:12px'>
        Configure simulation parameters and press Run
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

log     = st.session_state["log"]
metrics = st.session_state["metrics"]
cycles  = st.session_state["cycles"]
flags   = st.session_state["flags"]
chem    = st.session_state["chem"]
cc      = st.session_state["cc"]
t_h     = log["t"] / 3600.0

# ════════════════════════════════════════════════════════════════
# MACHINE 1
# ════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="machine-box">
  <div class="machine-title" style="color:{cc}">■ Machine 1 — PyBaMM DFN Physical Asset</div>
  <small style="color:{C_MUTED}">
    Ground-truth electrochemical states · Machine 2 observes only
    V_noisy = V_true + η  (σ = {st.session_state['noise_std']*1000:.0f} mV)
  </small>
</div>
""", unsafe_allow_html=True)

m1c1, m1c2, m1c3, m1c4, m1c5 = st.columns(5)
def kpi(col, label, value, unit, color=C_TEAL):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color}">{value}</div>
      <div class="kpi-unit">{unit}</div>
    </div>""", unsafe_allow_html=True)

kpi(m1c1, "Q Nominal",    f"{st.session_state['Q_nom']:.2f}", "Ah")
kpi(m1c2, "Total Time",   f"{log['t'][-1]/3600:.2f}", "h")
kpi(m1c3, "Data Points",  f"{len(log['t']):,}", "@ dt=10 s")
kpi(m1c4, "V Range",      f"{V.min():.3f}–{V.max():.3f}", "V")
kpi(m1c5, "T Range",      f"{log['T_true'].min():.1f}–{log['T_true'].max():.1f}", "°C")

f1c1, f1c2 = st.columns(2)
with f1c1:
    f = go.Figure()
    f.add_trace(go.Scatter(x=t_h, y=log["V_true"], mode="lines",
        name="V_true (DFN)", line=dict(color=cc, width=2)))
    f.add_trace(go.Scatter(x=t_h, y=log["V_meas"], mode="lines",
        name=f"V_noisy (σ={st.session_state['noise_std']*1000:.0f} mV)",
        line=dict(color=C_MUTED, width=0.6), opacity=0.5))
    f.update_layout(title="① Voltage — DFN Ground Truth vs Noisy Sensor",
        xaxis=ax("Time [h]"), yaxis=ax("Voltage [V]"), **BASE_LAYOUT)
    st.plotly_chart(f, use_container_width=True)

with f1c2:
    f = go.Figure()
    f.add_trace(go.Scatter(x=t_h, y=log["soc_true"]*100, mode="lines",
        name="SOC_true (DFN)", line=dict(color=cc, width=2)))
    f.add_trace(go.Scatter(x=t_h, y=log["I_true"], mode="lines",
        name="Current [A]", line=dict(color=C_MUTED, width=1),
        yaxis="y2"))
    f.update_layout(
        title="② SOC & Current — DFN Ground Truth",
        xaxis=ax("Time [h]"),
        yaxis=ax("SOC [%]"),
        yaxis2=dict(title="Current [A]", overlaying="y", side="right",
                    gridcolor=C_BORDER, tickfont=dict(size=10)),
        **BASE_LAYOUT)
    st.plotly_chart(f, use_container_width=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# MACHINE 2
# ════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="machine-box">
  <div class="machine-title" style="color:{C_ORANGE}">■ Machine 2 — 2-RC ECM + Adaptive EKF</div>
  <small style="color:{C_MUTED}">
    Reconstructs [SOC, V_RC1, V_RC2] from noisy voltage & current only ·
    Adaptive Q (Mehra 1972) · Adaptive R (eIAEKF 2026) · Joseph Form · η=0.9
  </small>
</div>
""", unsafe_allow_html=True)

m2c1,m2c2,m2c3,m2c4,m2c5,m2c6 = st.columns(6)
kpi(m2c1, "V RMSE",       f"{metrics['v_rmse_ekf']:.2f}",  "mV",
    C_RED if metrics['v_rmse_ekf']>20 else C_GREEN)
kpi(m2c2, "SOC RMSE",     f"{metrics['s_rmse_ekf']:.3f}",  "%",
    C_RED if metrics['s_rmse_ekf']>5 else C_GREEN)
kpi(m2c3, "Max SOC err",  f"{metrics['s_max_ekf']:.3f}",   "%")
kpi(m2c4, "Peak tr(P)",   f"{metrics['tr_P_init']:.2e}",   "state UQ")
kpi(m2c5, "Inn/Noise",    f"{metrics['inn_noise_ratio']:.2f}×",
    C_RED if metrics['inn_noise_ratio']>1.5 else C_GREEN)
kpi(m2c6, "NIS mean",     f"{metrics['nis_mean']:.3f}",
    C_GREEN if 0.5<metrics['nis_mean']<2.0 else C_YELLOW)

f2c1, f2c2 = st.columns(2)
with f2c1:
    f = go.Figure()
    f.add_trace(go.Scatter(x=t_h, y=log["V_true"], mode="lines",
        name="V_true (DFN)", line=dict(color=cc, width=2.2)))
    f.add_trace(go.Scatter(x=t_h, y=log["V_est"], mode="lines",
        name=f"V_est EKF ({metrics['v_rmse_ekf']:.1f} mV)",
        line=dict(color=C_ORANGE, width=1.5, dash="dash")))
    f.update_layout(title="③ Voltage Tracking — DFN vs EKF Observer",
        xaxis=ax("Time [h]"), yaxis=ax("Voltage [V]"), **BASE_LAYOUT)
    st.plotly_chart(f, use_container_width=True)

with f2c2:
    soc_band_hi = np.clip(log["soc_est"] + 2*np.sqrt(log["P_soc"]), 0, 1)*100
    soc_band_lo = np.clip(log["soc_est"] - 2*np.sqrt(log["P_soc"]), 0, 1)*100
    f = go.Figure()
    f.add_trace(go.Scatter(
        x=np.concatenate([t_h, t_h[::-1]]),
        y=np.concatenate([soc_band_hi, soc_band_lo[::-1]]),
        fill="toself", fillcolor=f"rgba(247,127,0,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="±2σ band"))
    f.add_trace(go.Scatter(x=t_h, y=log["soc_true"]*100, mode="lines",
        name="SOC_true (DFN)", line=dict(color=cc, width=2.2)))
    f.add_trace(go.Scatter(x=t_h, y=log["soc_est"]*100, mode="lines",
        name=f"SOC_est EKF ({metrics['s_rmse_ekf']:.2f}%)",
        line=dict(color=C_ORANGE, width=1.5, dash="dash")))
    f.update_layout(title="④ SOC Estimation with ±2σ Uncertainty Band",
        xaxis=ax("Time [h]"), yaxis=ax("SOC [%]"), **BASE_LAYOUT)
    st.plotly_chart(f, use_container_width=True)

f3c1, f3c2 = st.columns(2)
with f3c1:
    sigma2 = 2 * st.session_state["noise_std"] * 1000
    f = go.Figure()
    f.add_trace(go.Scatter(x=t_h, y=log["innov"]*1000, mode="lines",
        name="Innovation ν(k)", line=dict(color=C_TEAL, width=0.8)))
    f.add_hline(y= sigma2, line=dict(color=C_RED, dash="dot", width=1),
        annotation_text=f"+2σ={sigma2:.0f}mV")
    f.add_hline(y=-sigma2, line=dict(color=C_RED, dash="dot", width=1),
        annotation_text=f"-2σ={-sigma2:.0f}mV")
    f.update_layout(title="⑤ Innovation Residuals ν(k) — Whiteness Test",
        xaxis=ax("Time [h]"), yaxis=ax("ν(k) [mV]"), **BASE_LAYOUT)
    st.plotly_chart(f, use_container_width=True)

with f3c2:
    cyc_nums  = [r["cycle"]   for r in cycles]
    cyc_peaks = [r["peak_P"]  for r in cycles]
    cyc_cols  = [C_GREEN if r["status"]=="Stable"
                 else C_YELLOW if r["status"]=="Warning"
                 else C_RED for r in cycles]
    f = go.Figure()
    f.add_hline(y=cycles[0]["peak_P"] if cycles else 0,
        line=dict(color=C_MUTED, dash="dot", width=1),
        annotation_text="Cycle 1 baseline")
    f.add_trace(go.Scatter(x=cyc_nums, y=cyc_peaks, mode="lines+markers",
        name="Peak tr(P)", line=dict(color=C_PURPLE, width=2),
        marker=dict(color=cyc_cols, size=7, line=dict(width=1.2,
                    color=C_BORDER))))
    f.update_layout(title="⑥ Cycle-by-Cycle Uncertainty Growth — Peak tr(P)",
        xaxis=ax("Cycle Number"), yaxis=ax("Peak tr(P)", ".2e"),
        **BASE_LAYOUT)
    st.plotly_chart(f, use_container_width=True)

# EKF parameter table
with st.expander("EKF Internal Parameters"):
    st.markdown(f"""
    <div style='font-family:JetBrains Mono;font-size:0.78rem;color:{C_MUTED};line-height:2'>
    ∂h/∂SOC &nbsp; {log['V_true'][1] - log['V_true'][0]:.5f} &nbsp;|&nbsp;
    R0 = {chem['R0']*1000:.2f} mΩ &nbsp;|&nbsp;
    R1 = {chem['R1']*1000:.2f} mΩ &nbsp;|&nbsp; C1 = {chem['C1']:.0f} F<br>
    R2 = {chem['R2']*1000:.2f} mΩ &nbsp;|&nbsp; C2 = {chem['C2']:.0f} F &nbsp;|&nbsp;
    tr(P)_init = {metrics['tr_P_init']:.3e} &nbsp;|&nbsp;
    tr(P)_final = {metrics['tr_P_final']:.3e} &nbsp;|&nbsp;
    Convergence = {metrics['convergence_pct']:.1f}%
    </div>
    """, unsafe_allow_html=True)

# Cycle table
with st.expander("Cycle-by-Cycle Uncertainty Table"):
    df_cyc = pd.DataFrame(cycles).rename(columns={
        "cycle":"Cycle","soc_start":"SOC_i[%]","soc_min":"SOC_min[%]",
        "soc_end":"SOC_end[%]","dur_min":"Dur[min]","peak_P":"Peak tr(P)",
        "delta_pct":"Δ vs C1","peak_soc_pct":"Peak σ_SOC[%]","status":"Status"})
    st.dataframe(df_cyc, use_container_width=True, hide_index=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# ENGINEERING ASSESSMENT
# ════════════════════════════════════════════════════════════════
st.markdown("### ■ Engineering Assessment")
for fl in flags:
    cls = f"flag-{fl['level']}"
    icon = {"error":"🔴","warning":"🟡","ok":"🟢"}[fl["level"]]
    st.markdown(f"<div class='{cls}'>{icon} {fl['message']}</div>",
                unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# MACHINE 3 — PINN
# ════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="machine-box">
  <div class="machine-title" style="color:{C_PURPLE}">🧠 Machine 3 — PINN Residual Corrector</div>
  <small style="color:{C_MUTED}">
    Mechanistically-Guided Residual Learner · Nature Comms (2026) ·
    Learns V_true − V_ecm · Physics loss: dSOC/dt = −ηI/Q ·
    ResNet skip · AdamW · Cosine LR · Gradient clipping
  </small>
</div>
""", unsafe_allow_html=True)

pc1,pc2,pc3,pc4 = st.columns(4)
with pc1: p_ep  = st.slider("Training Epochs", 500, 3000, 2000, 500)
with pc2: p_lay = st.select_slider("Hidden Layers",   [1,2,3,4], 3)
with pc3: p_neu = st.select_slider("Neurons / Layer", [16,32,64,128], 64)
with pc4: p_lam = st.select_slider("Physics λ",
    [0.0, 0.001, 0.01, 0.1], 0.01)

if st.button("🚀  Train PINN Corrector", use_container_width=False):
    prog   = st.progress(0)
    status = st.empty()

    def cb(ep, total, dl, pl):
        prog.progress(ep / total)
        status.markdown(
            f"<div style='font-family:JetBrains Mono;font-size:0.78rem;color:{C_MUTED}'>"
            f"Epoch {ep}/{total} — Data: {dl:.5f} | Physics: {pl:.6f}</div>",
            unsafe_allow_html=True)

    V_corr, soc_corr, l_hist, p_hist, pmet = run_pinn(
        log=log, chem=chem, Q_nom=st.session_state["Q_nom"],
        n_epochs=p_ep, n_layers=p_lay, n_neurons=p_neu,
        lambda_phys=p_lam, progress_cb=cb)

    prog.empty(); status.empty()
    st.session_state.update(dict(
        pinn_done=True, pinn_V=V_corr, pinn_soc=soc_corr,
        pinn_lh=l_hist, pinn_ph=p_hist, pinn_met=pmet))

if st.session_state.get("pinn_done"):
    V_corr   = st.session_state["pinn_V"]
    soc_corr = st.session_state["pinn_soc"]
    l_hist   = st.session_state["pinn_lh"]
    p_hist   = st.session_state["pinn_ph"]
    pmet     = st.session_state["pinn_met"]

    pp1,pp2,pp3,pp4 = st.columns(4)
    def kpi_delta(col, label, before, after, unit):
        delta = after - before
        color = C_GREEN if delta < 0 else C_RED
        sign  = "▼" if delta < 0 else "▲"
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="color:{color}">{after:.3f}</div>
          <div class="kpi-unit">{unit} &nbsp;
            <span style="color:{color}">{sign}{abs(delta):.3f}</span>
            vs EKF {before:.3f}
          </div>
        </div>""", unsafe_allow_html=True)

    kpi_delta(pp1,"V RMSE (PINN)",  pmet["v_rmse_ecm"],  pmet["v_rmse_pinn"], "mV")
    kpi_delta(pp2,"SOC RMSE (PINN)",pmet["s_rmse_ekf"],  pmet["s_rmse_pinn"], "%")
    pp3.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Voltage Improvement</div>
      <div class="kpi-value" style="color:{C_GREEN}">{pmet['improv_v']:.1f}%</div>
      <div class="kpi-unit">vs ECM baseline</div>
    </div>""", unsafe_allow_html=True)
    pp4.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Inn/Noise (PINN)</div>
      <div class="kpi-value" style="color:{C_GREEN if pmet['inn_noise_ratio']<1.5 else C_YELLOW}">{pmet['inn_noise_ratio']:.2f}×</div>
      <div class="kpi-unit">target &lt; 1.5×</div>
    </div>""", unsafe_allow_html=True)

    pf1,pf2 = st.columns(2)
    with pf1:
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=log["V_true"], mode="lines",
            name="V_true (DFN)", line=dict(color=cc, width=2.2)))
        f.add_trace(go.Scatter(x=t_h, y=log["V_est"], mode="lines",
            name=f"V_est EKF ({pmet['v_rmse_ecm']:.1f} mV)",
            line=dict(color=C_ORANGE, width=1.2, dash="dash")))
        f.add_trace(go.Scatter(x=t_h, y=V_corr, mode="lines",
            name=f"V_PINN ({pmet['v_rmse_pinn']:.1f} mV)",
            line=dict(color=C_PURPLE, width=2.0)))
        f.update_layout(title="⑦ Voltage: DFN vs EKF vs PINN Corrected",
            xaxis=ax("Time [h]"), yaxis=ax("Voltage [V]"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    with pf2:
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=log["soc_true"]*100, mode="lines",
            name="SOC_true (DFN)", line=dict(color=cc, width=2.2)))
        f.add_trace(go.Scatter(x=t_h, y=log["soc_est"]*100, mode="lines",
            name=f"SOC EKF ({pmet['s_rmse_ekf']:.2f}%)",
            line=dict(color=C_ORANGE, width=1.2, dash="dash")))
        f.add_trace(go.Scatter(x=t_h, y=soc_corr*100, mode="lines",
            name=f"SOC PINN ({pmet['s_rmse_pinn']:.2f}%)",
            line=dict(color=C_PURPLE, width=2.0)))
        f.update_layout(title="⑧ SOC: DFN vs EKF vs PINN Corrected",
            xaxis=ax("Time [h]"), yaxis=ax("SOC [%]"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    pf3,pf4 = st.columns(2)
    with pf3:
        f = go.Figure()
        f.add_trace(go.Scatter(
            x=list(range(1, len(l_hist)+1)), y=l_hist,
            mode="lines", name="Data Loss",
            line=dict(color=C_PURPLE, width=2)))
        f.add_trace(go.Scatter(
            x=list(range(1, len(p_hist)+1)), y=p_hist,
            mode="lines", name="Physics Loss",
            line=dict(color=C_ORANGE, width=1.5, dash="dash")))
        f.update_layout(title="⑨ PINN Training Loss Convergence",
            xaxis=ax("Epoch"), yaxis=ax("Loss", ".2e"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    with pf4:
        ocv_fn = make_ocv(chem)
        V_ocv  = np.array([float(ocv_fn(np.clip(s,0.01,0.99)))
                           for s in log["soc_est"]])
        V_ecm_b = V_ocv - log["I_true"] * chem["R0"]
        res_true = (log["V_true"] - V_ecm_b) * 1000
        res_pred = (V_corr       - V_ecm_b) * 1000
        f = go.Figure()
        f.add_trace(go.Scatter(x=t_h, y=res_true, mode="lines",
            name="True Residual (DFN − ECM)",
            line=dict(color=cc, width=1.8)))
        f.add_trace(go.Scatter(x=t_h, y=res_pred, mode="lines",
            name="PINN Predicted Residual",
            line=dict(color=C_PURPLE, width=1.8, dash="dash")))
        f.update_layout(title="⑩ Residual: True vs PINN Prediction",
            xaxis=ax("Time [h]"), yaxis=ax("Residual [mV]"), **BASE_LAYOUT)
        st.plotly_chart(f, use_container_width=True)

    with st.expander("PINN Architecture Summary"):
        st.code(f"""
Input Layer   : 4 features [t_norm, I_norm, SOC_est, V_ecm_norm]
Hidden Layers : {p_lay} × {p_neu} neurons  (Tanh + ResNet skip)
Output Layer  : 1  →  residual correction [mV]
Physics Loss  : dSOC/dt = −ηI/Q  (η=0.9 charge / 1.0 discharge)
Total Loss    : L = MSE(data) + {p_lam} × MSE(physics)
Optimizer     : AdamW  (lr=1e-3, wd=1e-4)
Scheduler     : CosineAnnealingLR  (T_max={p_ep}, η_min=1e-5)
Grad Clipping : max_norm = 1.0
Init          : Xavier uniform
Params        : {pmet['n_params']:,}
─────────────────────────────────────────
ECM Baseline V RMSE : {pmet['v_rmse_ecm']:.3f} mV
PINN Corrected RMSE : {pmet['v_rmse_pinn']:.3f} mV
Voltage Improvement : {pmet['improv_v']:.1f}%
EKF SOC RMSE        : {pmet['s_rmse_ekf']:.4f}%
PINN SOC RMSE       : {pmet['s_rmse_pinn']:.4f}%
Inn/Noise (PINN)    : {pmet['inn_noise_ratio']:.3f}×
""", language="")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='text-align:center;padding:20px 0;color:{C_MUTED};
     font-family:JetBrains Mono;font-size:0.70rem;line-height:2'>
  BattSim v4.2 — Digital Twin Co-Simulation Framework<br>
  Designed &amp; Developed by Eng. Thaer Abushawar<br>
  Plett (2004) J. Power Sources 134 ·
  Chen et al. (2020) J. Electrochem. Soc. 167 ·
  Yue et al. (2026) Mech. Sys. Signal Process. ·
  Nature Comms (2026) doi:10.1038/s41467
</div>
""", unsafe_allow_html=True)

