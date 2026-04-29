import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from weasyprint import HTML

# Import your backend modules
import chemistry
import machine1_dfn
import machine2_ekf
import utils

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Battery Digital Twin",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔋 Battery Digital Twin: DFN Asset & EKF Observer")
st.markdown("""
This dashboard simulates a physical battery asset using a high-fidelity **Doyle-Fuller-Newman (DFN)** model (Machine 1) 
and estimates its internal State of Charge (SOC) in real-time using an **Adaptive Extended Kalman Filter (AEKF)** (Machine 2).
""")

# ─── PDF Generation Logic ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_pdf_report(log, summary, metadata):
    """Generates a PDF report from the simulation log and returns it as bytes."""
    # 1. Create static charts for the PDF using Matplotlib
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    t_h = log['t'] / 3600
    
    # Voltage tracking plot
    ax1.plot(t_h, log['V_true'], 'k-', label='True Voltage (DFN)')
    ax1.plot(t_h, log['V_est'], 'r--', label='EKF Estimated')
    ax1.set_ylabel('Voltage [V]')
    ax1.set_title('Voltage Tracking Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # SOC tracking plot
    ax2.plot(t_h, log['soc_true'] * 100, 'k-', label='True SOC')
    ax2.plot(t_h, log['soc_est'] * 100, 'b--', label='EKF Estimate')
    ax2.fill_between(t_h, log['ci_lower']*100, log['ci_upper']*100, color='blue', alpha=0.1, label='95% Confidence (±2σ)')
    ax2.set_ylabel('SOC [%]')
    ax2.set_xlabel('Time [Hours]')
    ax2.set_title('SOC Estimation & Uncertainty Propagation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Convert plot to Base64 for HTML embedding
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    # 2. Build the HTML template
    html_content = f"""
    <html>
    <head>
        <style>
            @page {{ size: A4; margin: 20mm; background-color: #ffffff; }}
            body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; line-height: 1.6; }}
            .header {{ background-color: #1e3a8a; color: white; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
            h1 {{ margin: 0; font-size: 22pt; }}
            h2 {{ color: #1e3a8a; border-bottom: 2px solid #1e3a8a; padding-bottom: 5px; font-size: 16pt; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #e2e8f0; }}
            th {{ background-color: #f8fafc; color: #475569; }}
            .metrics-grid {{ display: flex; gap: 15px; margin-top: 20px; }}
            .metric-box {{ background-color: #f1f5f9; padding: 15px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #e2e8f0; }}
            .metric-value {{ font-size: 18pt; font-weight: bold; color: #2563eb; }}
            .metric-label {{ font-size: 10pt; color: #64748b; margin-top: 5px; }}
            img {{ width: 100%; height: auto; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Battery Digital Twin Report</h1>
            <p>DFN Physical Asset & AEKF Observer Analysis</p>
        </div>

        <h2>1. Simulation Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Cell Chemistry</td><td>{metadata['chem_name']}</td></tr>
            <tr><td>Cycle Protocol</td><td>{metadata['protocol'].upper()}</td></tr>
            <tr><td>C-Rate</td><td>{metadata['c_rate']} C</td></tr>
            <tr><td>Sensor Noise (Std. Dev.)</td><td>{metadata['noise_std']} V</td></tr>
            <tr><td>Process Noise Scale (Q)</td><td>{metadata['q_scale']}</td></tr>
        </table>

        <h2>2. Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-value">{summary['rmse_soc_pct']:.2f} %</div>
                <div class="metric-label">SOC RMSE</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">±{summary['mean_sigma_pct']:.2f} %</div>
                <div class="metric-label">Mean Uncertainty (1σ)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{summary['nis_mean']:.2f}</div>
                <div class="metric-label">Avg NIS Score</div>
            </div>
        </div>

        <h2>3. Graphical Analysis</h2>
        <img src="data:image/png;base64,{chart_base64}">
    </body>
    </html>
    """
    # 3. Render HTML to PDF bytes
    pdf_bytes = HTML(string=html_content).write_pdf()
    return pdf_bytes

# ─── Data Initialization ───────────────────────────────────────────────────────
@st.cache_resource
def load_chemistry():
    return chemistry.build_chem()

chems = load_chemistry()

# ─── Sidebar Configuration ─────────────────────────────────────────────────────
st.sidebar.header("⚙️ 1. Physical Asset (Machine 1)")
chem_label = st.sidebar.selectbox("Cell Chemistry", list(chems.keys()))
selected_chem = chems[chem_label]

protocol = st.sidebar.selectbox("Cycle Protocol", ["cccv", "cc", "hppc"], index=0)
n_cycles = st.sidebar.number_input("Number of Cycles", min_value=1, max_value=20, value=2, step=1)
c_rate = st.sidebar.slider("C-Rate", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
noise_std = st.sidebar.number_input("Sensor Noise Std. Dev [V]", min_value=0.001, max_value=0.1, value=0.01, step=0.005, format="%.3f")

st.sidebar.header("🧮 2. Digital Twin (Machine 2)")
st.sidebar.markdown("Tune the Adaptive Extended Kalman Filter (EKF):")
q_scale = st.sidebar.number_input("Process Noise Scale (Q)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
r_scale = st.sidebar.number_input("Measurement Noise Scale (R)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
p0_scale = st.sidebar.number_input("Initial Covariance Scale (P0)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)

# ─── Simulation Caching ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_physical_asset(pset_name, cycles, rate, prot, vmin, vmax):
    """Runs the heavy PyBaMM DFN simulation and caches the result."""
    return machine1_dfn.run_dfn(
        pset_name=pset_name, 
        n_cycles=cycles, 
        c_rate=rate, 
        protocol=prot, 
        v_min=vmin, 
        v_max=vmax
    )

# ─── Execution ─────────────────────────────────────────────────────────────────
if st.sidebar.button("🚀 Run Co-Simulation", type="primary", use_container_width=True):
    
    with st.spinner("Simulating Physical Asset (PyBaMM DFN)... This may take a moment."):
        t_u, V_u, I_u, soc_u, T_u, Q_nom = run_physical_asset(
            selected_chem["pybamm"], 
            n_cycles, 
            c_rate, 
            protocol, 
            selected_chem["v_min"], 
            selected_chem["v_max"]
        )
    
    with st.spinner("Running EKF Digital Twin Observer..."):
        log = machine2_ekf.run_cosim(
            t=t_u,
            V_true=V_u,
            I_true=I_u,
            soc_true=soc_u,
            T_true=T_u,
            Q_nom=Q_nom,
            chem=selected_chem,
            noise_std=noise_std,
            p0_scale=p0_scale,
            q_scale=q_scale,
            r_scale=r_scale
        )

    # ─── Compute Metrics ───────────────────────────────────────────────────────
    summary = utils.summary_dict(log)
    
    # ─── PDF Report Generation ─────────────────────────────────────────────────
    metadata = {
        "chem_name": chem_label,
        "protocol": protocol,
        "c_rate": c_rate,
        "noise_std": noise_std,
        "q_scale": q_scale
    }
    
    with st.spinner("Generating PDF Report..."):
        pdf_file = generate_pdf_report(log, summary, metadata)

    st.markdown("### 📊 Real-Time Uncertainty & Performance Metrics")
    
    # Place metrics and the download button
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SOC RMSE", utils.fmt_rmse(summary["rmse_soc_pct"], "%"))
    m2.metric("Max SOC Error", utils.fmt_rmse(summary["max_err_soc_pct"], "%"))
    m3.metric("Mean Confidence (±1σ)", utils.fmt_sigma(summary["mean_sigma_pct"] / 100))
    m4.metric("NIS Calibration", f"{summary['nis_mean']:.2f}", delta=summary['nis_verdict'], delta_color="off")
    
    st.download_button(
        label="📄 Download Full PDF Report",
        data=pdf_file,
        file_name="Battery_Digital_Twin_Report.pdf",
        mime="application/pdf",
        type="primary"
    )
    
    st.markdown("---")

    # ─── Plotting (Interactive Dashboards) ─────────────────────────────────────
    t_hours = utils.time_to_hours(log["t"])
    
    # Downsample for faster UI rendering without losing shape
    idx = np.linspace(0, len(t_hours) - 1, min(len(t_hours), 3000), dtype=int)
    t_plot = t_hours[idx]

    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=("Voltage Tracking: True vs Measured vs EKF", 
                        "SOC Estimation with Uncertainty Propagation (±2σ)", 
                        "Estimation Error & Filter Consistency (NIS)")
    )

    # 1. Voltage Plot
    fig.add_trace(go.Scatter(x=t_plot, y=log["V_true"][idx], name="True V (DFN)", line=dict(color="black", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_plot, y=log["V_meas"][idx], name="Measured V (Noisy)", mode="markers", marker=dict(color="gray", size=3, opacity=0.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_plot, y=log["V_est"][idx], name="EKF Estimated V", line=dict(color="red", width=1.5, dash="dash")), row=1, col=1)

    # 2. SOC Plot with Confidence Intervals
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_plot, t_plot[::-1]]),
        y=np.concatenate([log["ci_upper"][idx] * 100, (log["ci_lower"][idx] * 100)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence (±2σ)"
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=t_plot, y=log["soc_true"][idx] * 100, name="True SOC (DFN)", line=dict(color="black", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_plot, y=log["soc_est"][idx] * 100, name="EKF Estimated SOC", line=dict(color="dodgerblue", width=2)), row=2, col=1)

    # 3. Error and NIS Plot
    soc_error = (log["soc_est"] - log["soc_true"]) * 100
    fig.add_trace(go.Scatter(x=t_plot, y=soc_error[idx], name="SOC Error (%)", line=dict(color="purple", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=t_plot, y=2*(log["sigma_soc"][idx]*100), name="+2σ Bound", line=dict(color="gray", width=1, dash="dot")), row=3, col=1)
    fig.add_trace(go.Scatter(x=t_plot, y=-2*(log["sigma_soc"][idx]*100), name="-2σ Bound", line=dict(color="gray", width=1, dash="dot")), row=3, col=1)

    fig.update_layout(height=900, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
    
    fig.update_yaxes(title_text="Voltage [V]", row=1, col=1)
    fig.update_yaxes(title_text="SOC [%]", range=[-5, 105], row=2, col=1)
    fig.update_yaxes(title_text="SOC Error [%]", row=3, col=1)
    fig.update_xaxes(title_text="Time [Hours]", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ─── Raw Data Expander ─────────────────────────────────────────────────────
    with st.expander("🔍 View Per-Cycle Statistics & Raw Data"):
        st.dataframe(utils.per_cycle_stats(log), use_container_width=True)
else:
    st.info("👈 Set your simulation parameters in the sidebar and click **Run Co-Simulation** to start.")
