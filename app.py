"""
╔══════════════════════════════════════════════════════════════════╗
║   BattSim v5.0 — Digital Twin Co-Simulation                      ║
║   Streamlit Application                                           ║
║   Author: Eng. Thaer Abushawar | Thaer199@gmail.com              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64, io, json, datetime, textwrap, math
from pathlib import Path

# ─── Page Config (MUST be first Streamlit call) ───────────────────
st.set_page_config(
    page_title="BattSim v5.0 — Digital Twin",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "mailto:Thaer199@gmail.com",
        "About": "BattSim v5.0 | Eng. Thaer Abushawar | Thaer199@gmail.com",
    },
)

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

/* Header */
.battsim-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f4c81 100%);
    border-radius: 14px; padding: 28px 36px 22px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
    position: relative; overflow: hidden;
}
.battsim-header::before {
    content: ""; position: absolute; top: -40px; right: -40px;
    width: 180px; height: 180px; border-radius: 50%;
    background: rgba(56,189,248,0.08);
}
.battsim-header::after {
    content: ""; position: absolute; bottom: -30px; left: 120px;
    width: 120px; height: 120px; border-radius: 50%;
    background: rgba(56,189,248,0.05);
}
.header-title {
    font-size: 2.1rem; font-weight: 700; color: #f8fafc;
    margin: 0 0 4px; letter-spacing: -0.02em;
}
.header-subtitle { font-size: 0.95rem; color: #94a3b8; margin: 0; }
.header-badge {
    display: inline-block; background: rgba(56,189,248,0.15);
    border: 1px solid rgba(56,189,248,0.3); color: #38bdf8;
    border-radius: 20px; padding: 3px 12px; font-size: 0.78rem;
    font-weight: 600; margin-left: 10px; vertical-align: middle;
}
.header-author {
    font-size: 0.82rem; color: #64748b; margin-top: 10px;
    font-family: 'JetBrains Mono', monospace;
}

/* KPI Cards */
.kpi-row { display: flex; gap: 14px; margin-bottom: 1.2rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 140px;
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 16px 18px;
    transition: box-shadow 0.2s ease;
}
.kpi-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.kpi-label {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: #64748b; font-weight: 600; margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.85rem; font-weight: 700; color: #0f172a;
    font-family: 'JetBrains Mono', monospace; line-height: 1;
}
.kpi-unit { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; }
.kpi-ok   { border-top: 3px solid #22c55e; }
.kpi-warn { border-top: 3px solid #f59e0b; }
.kpi-err  { border-top: 3px solid #ef4444; }
.kpi-blue { border-top: 3px solid #38bdf8; }
.kpi-purple { border-top: 3px solid #a855f7; }

/* Section Headers */
.section-header {
    font-size: 1.05rem; font-weight: 700; color: #0f172a;
    border-left: 4px solid #38bdf8; padding-left: 12px;
    margin: 1.4rem 0 0.8rem;
}

/* Status Badges */
.badge { display:inline-block; border-radius:5px; padding:2px 9px; font-size:0.75rem; font-weight:700; }
.badge-ok   { background:#dcfce7; color:#166534; }
.badge-warn { background:#fef9c3; color:#854d0e; }
.badge-err  { background:#fee2e2; color:#991b1b; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: #f1f5f9;
    border-radius: 10px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 6px 18px;
    font-weight: 500; font-size: 0.88rem;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #f8fafc; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* Progress */
.stProgress > div > div { background: #38bdf8 !important; }

/* Run button */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 0.6rem 2rem !important; font-size: 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(3,105,161,0.3) !important;
}
div[data-testid="stButton"] button:hover {
    box-shadow: 0 4px 16px rgba(3,105,161,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Divider */
.custom-divider { border: none; border-top: 1px solid #e2e8f0; margin: 1.2rem 0; }

/* Footer */
.app-footer {
    text-align: center; color: #94a3b8; font-size: 0.78rem;
    margin-top: 3rem; padding-top: 1.5rem;
    border-top: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SIMULATION ENGINE (self-contained — no external src/ needed)
# ═══════════════════════════════════════════════════════════════════

# ── Chemistry Definitions ─────────────────────────────────────────
CHEMISTRY_DB = {
    "NMC — Chen2020 (LG M50 21700)": {
        "label": "NMC Chen2020",
        "color": "#3b82f6",
        "Q": 5.0, "v_min": 2.5, "v_max": 4.2,
        "R0": 0.010, "R1": 0.015, "C1": 3000.0,
        "R2": 0.008, "C2": 8000.0,
        "soc_lut": [0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,
                    0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,
                    0.80,0.85,0.90,0.95,1.00],
        "ocv_lut": [2.80,3.10,3.20,3.30,3.36,3.41,3.46,3.50,
                    3.54,3.58,3.62,3.66,3.71,3.76,3.80,3.85,
                    3.90,3.95,4.00,4.10,4.20],
    },
    "LFP — Prada2013 (ANR26650)": {
        "label": "LFP Prada2013",
        "color": "#10b981",
        "Q": 2.3, "v_min": 2.5, "v_max": 3.65,
        "R0": 0.012, "R1": 0.020, "C1": 2500.0,
        "R2": 0.010, "C2": 6000.0,
        "soc_lut": [0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,
                    0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,
                    0.80,0.85,0.90,0.95,1.00],
        "ocv_lut": [2.80,3.10,3.18,3.22,3.26,3.28,3.30,3.31,
                    3.32,3.32,3.33,3.33,3.34,3.34,3.35,3.36,
                    3.38,3.42,3.48,3.55,3.65],
    },
    "NMA — OKane2022 (Kokam SLPB)": {
        "label": "NMA OKane2022",
        "color": "#f59e0b",
        "Q": 3.5, "v_min": 2.7, "v_max": 4.3,
        "R0": 0.008, "R1": 0.012, "C1": 4000.0,
        "R2": 0.006, "C2": 9000.0,
        "soc_lut": [0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,
                    0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,
                    0.80,0.85,0.90,0.95,1.00],
        "ocv_lut": [2.90,3.12,3.25,3.35,3.42,3.48,3.53,3.57,
                    3.61,3.65,3.70,3.75,3.80,3.84,3.88,3.93,
                    3.98,4.04,4.12,4.20,4.30],
    },
}

def make_ocv_fn(chem):
    from scipy.interpolate import interp1d
    return interp1d(chem["soc_lut"], chem["ocv_lut"],
                    kind="linear", fill_value="extrapolate")

def docv_dsoc_num(ocv_fn, soc, dsoc=1e-5):
    return (ocv_fn(soc + dsoc) - ocv_fn(soc - dsoc)) / (2 * dsoc)


# ── Machine 1: DFN-surrogate (PyBaMM-matched ECM) ────────────────
def run_machine1(chem, n_cycles, c_rate, protocol="cccv", dt=10.0, seed=42):
    """Ground-truth battery simulator (DFN surrogate via 2-RC ECM)."""
    np.random.seed(seed)
    ocv_fn = make_ocv_fn(chem)
    Q   = chem["Q"]
    R0  = chem["R0"]; R1 = chem["R1"]; C1 = chem["C1"]
    R2  = chem["R2"]; C2 = chem["C2"]
    v_min = chem["v_min"]; v_max = chem["v_max"]
    I_chg = c_rate * Q
    I_dis = -c_rate * Q

    t_list=[]; V_list=[]; I_list=[]; soc_list=[]; T_list=[]
    t = 0.0; soc = 1.0; vrc1 = 0.0; vrc2 = 0.0; T = 25.0
    tau1 = R1*C1; tau2 = R2*C2

    for cyc in range(n_cycles):
        # ── Discharge ─────────────────────────────
        phase = "discharge"
        while True:
            I = I_dis
            ocv = float(ocv_fn(np.clip(soc,0,1)))
            V   = ocv + I*R0 + vrc1 + vrc2
            t_list.append(t); V_list.append(V); I_list.append(I)
            soc_list.append(soc); T_list.append(T)
            if V <= v_min or soc <= 0.02:
                break
            soc   = soc + I / Q * dt / 3600
            vrc1  = vrc1 * np.exp(-dt/tau1) + I*R1*(1-np.exp(-dt/tau1))
            vrc2  = vrc2 * np.exp(-dt/tau2) + I*R2*(1-np.exp(-dt/tau2))
            t    += dt

        # ── Rest 1 ────────────────────────────────
        for _ in range(int(300/dt)):
            I = 0.0
            ocv = float(ocv_fn(np.clip(soc,0,1)))
            V   = ocv + vrc1 + vrc2
            t_list.append(t); V_list.append(V); I_list.append(I)
            soc_list.append(soc); T_list.append(T)
            vrc1 *= np.exp(-dt/tau1); vrc2 *= np.exp(-dt/tau2)
            t += dt

        # ── CC Charge ─────────────────────────────
        while soc < 0.99 and float(ocv_fn(np.clip(soc,0,1))) < v_max - 0.01:
            I = I_chg
            ocv = float(ocv_fn(np.clip(soc,0,1)))
            V   = ocv + I*R0 + vrc1 + vrc2
            t_list.append(t); V_list.append(V); I_list.append(I)
            soc_list.append(soc); T_list.append(T)
            soc  = min(soc + I/Q*dt/3600, 1.0)
            vrc1 = vrc1*np.exp(-dt/tau1) + I*R1*(1-np.exp(-dt/tau1))
            vrc2 = vrc2*np.exp(-dt/tau2) + I*R2*(1-np.exp(-dt/tau2))
            t   += dt

        # ── CV tail (if cccv) ─────────────────────
        if protocol == "cccv":
            I_cv = I_chg
            for _ in range(int(1800/dt)):
                if I_cv < 0.05*I_chg:
                    break
                ocv = float(ocv_fn(np.clip(soc,0,1)))
                V   = v_max
                I_cv = (v_max - ocv - vrc1 - vrc2) / R0
                I_cv = max(I_cv, 0)
                t_list.append(t); V_list.append(V)
                I_list.append(I_cv); soc_list.append(soc); T_list.append(T)
                soc  = min(soc + I_cv/Q*dt/3600, 1.0)
                vrc1 = vrc1*np.exp(-dt/tau1) + I_cv*R1*(1-np.exp(-dt/tau1))
                vrc2 = vrc2*np.exp(-dt/tau2) + I_cv*R2*(1-np.exp(-dt/tau2))
                t   += dt

        # ── Rest 2 ────────────────────────────────
        for _ in range(int(300/dt)):
            I = 0.0
            ocv = float(ocv_fn(np.clip(soc,0,1)))
            V   = ocv + vrc1 + vrc2
            t_list.append(t); V_list.append(V); I_list.append(I)
            soc_list.append(soc); T_list.append(T)
            vrc1 *= np.exp(-dt/tau1); vrc2 *= np.exp(-dt/tau2)
            t += dt

    t   = np.array(t_list,   dtype=float)
    V   = np.array(V_list,   dtype=float)
    I   = np.array(I_list,   dtype=float)
    soc = np.array(soc_list, dtype=float)
    T   = np.array(T_list,   dtype=float)
    return t, V, I, soc, T, Q


# ── Machine 2: EKF Observer ───────────────────────────────────────
class EKF:
    def __init__(self, Q_nom, chem, P0_scale=1e-3, Q_scale=1.0, R_scale=1.0, dt=10.0):
        self.Q_nom = Q_nom; self.dt = dt
        self.ocv_fn = make_ocv_fn(chem)
        R0 = chem["R0"]; R1 = chem["R1"]; C1 = chem["C1"]
        R2 = chem["R2"]; C2 = chem["C2"]
        tau1 = R1*C1; tau2 = R2*C2
        self.R0 = R0
        self.A = np.diag([1.0, np.exp(-dt/tau1), np.exp(-dt/tau2)])
        self.B = np.array([-dt/(Q_nom*3600), R1*(1-np.exp(-dt/tau1)), R2*(1-np.exp(-dt/tau2))])
        self.x = np.array([1.0, 0.0, 0.0])
        self.P = np.eye(3) * P0_scale
        Q_diag = np.array([1e-6, 1e-8, 1e-8]) * Q_scale
        self.Q_n = np.diag(Q_diag)
        self.R_n = (0.010 ** 2) * R_scale

    def step(self, v_meas, I, T):
        # Predict
        x_p = self.A @ self.x + self.B * I
        x_p[0] = np.clip(x_p[0], 0.0, 1.0)
        P_p = self.A @ self.P @ self.A.T + self.Q_n
        # Linearise
        soc_p = x_p[0]
        h_soc = float(docv_dsoc_num(self.ocv_fn, soc_p))
        C_k = np.array([h_soc, -1.0, -1.0])
        # Innovation
        v_hat = float(self.ocv_fn(soc_p)) - x_p[1] - x_p[2] + I*self.R0
        nu    = v_meas - v_hat
        # Update
        S  = float(C_k @ P_p @ C_k) + self.R_n
        K  = P_p @ C_k / S
        self.x = x_p + K * nu
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        IKC   = np.eye(3) - np.outer(K, C_k)
        self.P = IKC @ P_p @ IKC.T + np.outer(K, K) * self.R_n
        # Outputs
        v_est    = float(self.ocv_fn(self.x[0])) - self.x[1] - self.x[2] + I*self.R0
        tr_P     = float(np.trace(self.P))
        P_soc    = float(self.P[0,0])
        sigma    = float(np.sqrt(max(P_soc, 0)))
        NIS      = float(nu**2 / S)
        return v_est, float(self.x[0]), tr_P, P_soc, float(nu), NIS, sigma


def run_cosim(t, V_true, I_true, soc_true, T_true, Q_nom, chem,
              noise_std=0.005, P0_scale=1e-3, Q_scale=1.0, R_scale=1.0, seed=42):
    np.random.seed(seed)
    n = len(t)
    dt = float(t[1]-t[0]) if n > 1 else 10.0
    ekf = EKF(Q_nom, chem, P0_scale=P0_scale, Q_scale=Q_scale, R_scale=R_scale, dt=dt)

    V_meas  = V_true + np.random.normal(0, noise_std, n)
    V_est   = np.zeros(n); soc_est  = np.zeros(n)
    P_tr    = np.zeros(n); P_soc_a  = np.zeros(n)
    sigma_s = np.zeros(n); innov    = np.zeros(n); NIS = np.zeros(n)

    for k in range(n):
        ve, se, tp, ps, nu, nis, sg = ekf.step(V_meas[k], I_true[k], T_true[k])
        V_est[k]=ve; soc_est[k]=se; P_tr[k]=tp; P_soc_a[k]=ps
        sigma_s[k]=sg; innov[k]=nu; NIS[k]=nis

    return {
        "t":t, "V_true":V_true, "V_meas":V_meas, "V_est":V_est,
        "I_true":I_true, "soc_true":soc_true, "soc_est":soc_est,
        "T_true":T_true, "P_tr":P_tr, "P_soc":P_soc_a,
        "sigma_soc":sigma_s, "innov":innov, "NIS":NIS,
        "ci_upper": soc_est + 2*sigma_s,
        "ci_lower": soc_est - 2*sigma_s,
        "Q_nom": Q_nom,
    }


# ── Per-cycle stats ───────────────────────────────────────────────
def detect_cycles(soc, threshold=0.7):
    cycles, in_c, start = [], False, 0
    for i in range(1, len(soc)):
        if not in_c and soc[i] >= threshold and soc[i-1] < threshold:
            start = i; in_c = True
        elif in_c and soc[i] < threshold and soc[i-1] >= threshold:
            cycles.append((start, i)); in_c = False
    if in_c:
        cycles.append((start, len(soc)-1))
    return cycles if cycles else [(0, len(soc)-1)]

def cycle_stats(log):
    t      = log["t"]/3600
    st     = log["soc_true"]*100
    se     = log["soc_est"]*100
    sig    = log["sigma_soc"]*100
    nis    = log["NIS"]
    vt     = log["V_true"]*1000
    ve     = log["V_est"]*1000
    cycles = detect_cycles(log["soc_true"])
    rows   = []
    for i,(s,e) in enumerate(cycles):
        sl   = slice(s,e+1)
        err  = np.abs(st[sl]-se[sl])
        rmse = float(np.sqrt(np.mean(err**2)))
        rows.append({
            "Cycle": i+1,
            "SOC Start %": round(float(st[s]),1),
            "SOC Min %":   round(float(st[s:e+1].min()),1),
            "SOC End %":   round(float(st[e]),1),
            "RMSE SOC %":  round(rmse,3),
            "Max Err SOC %": round(float(err.max()),3),
            "Mean σ %":    round(float(sig[sl].mean()),3),
            "Max σ %":     round(float(sig[sl].max()),3),
            "Mean NIS":    round(float(nis[sl].mean()),3),
            "Peak trP":    f"{log['P_tr'][sl].max():.3e}",
            "Dur (min)":   round((t[e]-t[s])*60,1),
        })
    return pd.DataFrame(rows)

def summary_metrics(log):
    st  = log["soc_true"]*100; se = log["soc_est"]*100
    vt  = log["V_true"]*1000;  ve = log["V_est"]*1000
    err = np.abs(st-se)
    v_rmse = float(np.sqrt(np.mean((vt-ve)**2)))
    nis    = log["NIS"]
    mn     = float(nis.mean())
    cal    = abs(mn-1.0) < 0.35 and float(np.mean((nis>0.004)&(nis<5.024)))>0.88
    inno   = log["innov"]*1000
    return {
        "v_rmse": round(v_rmse,3),
        "soc_rmse": round(float(np.sqrt(np.mean(err**2))),4),
        "soc_max":  round(float(err.max()),4),
        "sigma_mean": round(float(log["sigma_soc"].mean()*100),4),
        "sigma_max":  round(float(log["sigma_soc"].max()*100),4),
        "trP_init": float(log["P_tr"][0]),
        "trP_final":float(log["P_tr"][-1]),
        "trP_conv": round(float(log["P_tr"][-1]/log["P_tr"][0]*100),2),
        "nis_mean": round(mn,4),
        "nis_cal":  cal,
        "inn_rms":  round(float(np.sqrt(np.mean(inno**2))),3),
        "inn_mean": round(float(inno.mean()),3),
        "inn_ratio":round(float(np.sqrt(np.mean(inno**2)))/(log.get("noise_std",0.005)*1000+1e-9),3),
    }


# ═══════════════════════════════════════════════════════════════════
# PLOTLY CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════
COLORS = {
    "true":  "#0369a1", "est": "#f97316", "meas": "#94a3b8",
    "ci":    "rgba(249,115,22,0.12)", "sigma":"#a855f7",
    "trP":   "#22c55e",  "nis": "#f59e0b", "innov":"#ef4444",
    "grid":  "#e2e8f0",  "bg":  "#ffffff",
}

LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
    font=dict(family="Inter, sans-serif", size=12, color="#374151"),
    margin=dict(l=54, r=24, t=48, b=48),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#e2e8f0",
        borderwidth=1, font=dict(size=11),
    ),
    hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0",
                    font=dict(family="JetBrains Mono", size=11)),
)

def axis_style(title, unit=""):
    lbl = f"{title} [{unit}]" if unit else title
    return dict(title=lbl, gridcolor=COLORS["grid"], gridwidth=1,
                showline=True, linecolor="#cbd5e1", linewidth=1,
                tickfont=dict(size=11))

def fig_voltage(log):
    t = log["t"]/3600
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=t, y=log["V_meas"]*1000, name="V measured",
        line=dict(color=COLORS["meas"],width=1), opacity=0.55, mode="lines"))
    fig.add_trace(go.Scattergl(x=t, y=log["V_true"]*1000, name="V true (DFN)",
        line=dict(color=COLORS["true"],width=2.2), mode="lines"))
    fig.add_trace(go.Scattergl(x=t, y=log["V_est"]*1000, name="V estimated (EKF)",
        line=dict(color=COLORS["est"],width=1.8,dash="dash"), mode="lines"))
    fig.update_layout(**LAYOUT_BASE, title="<b>Voltage Tracking</b> — DFN Ground Truth vs EKF Reconstruction",
        xaxis=axis_style("Time","h"), yaxis=axis_style("Voltage","mV"), height=360)
    return fig

def fig_soc(log):
    t = log["t"]/3600
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([log["ci_upper"]*100, log["ci_lower"][::-1]*100]),
        fill="toself", fillcolor=COLORS["ci"], line=dict(width=0),
        name="95% CI (±2σ)", showlegend=True))
    fig.add_trace(go.Scattergl(x=t, y=log["soc_true"]*100, name="SOC true (DFN)",
        line=dict(color=COLORS["true"],width=2.2), mode="lines"))
    fig.add_trace(go.Scattergl(x=t, y=log["soc_est"]*100, name="SOC estimated (EKF)",
        line=dict(color=COLORS["est"],width=1.8,dash="dash"), mode="lines"))
    fig.update_layout(**LAYOUT_BASE, title="<b>SOC Estimation</b> — with 95% Confidence Band",
        xaxis=axis_style("Time","h"), yaxis=axis_style("SOC","%"), height=360)
    return fig

def fig_uncertainty(log):
    t = log["t"]/3600
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("State Covariance Trace P — Convergence",
                        "SOC Uncertainty σ(SOC) over Time"),
        vertical_spacing=0.12)
    fig.add_trace(go.Scattergl(x=t, y=log["P_tr"], name="trace(P)",
        line=dict(color=COLORS["trP"],width=2), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=t, y=log["sigma_soc"]*100, name="σ(SOC)",
        line=dict(color=COLORS["sigma"],width=2), mode="lines"), row=2, col=1)
    fig.update_layout(**LAYOUT_BASE, height=440,
        title="<b>Uncertainty Propagation</b> — EKF Covariance Dynamics",
        showlegend=True)
    fig.update_yaxes(title_text="trace(P)", row=1, col=1, **{k:v for k,v in axis_style("").items() if k!="title"})
    fig.update_yaxes(title_text="σ(SOC) [%]", row=2, col=1, **{k:v for k,v in axis_style("").items() if k!="title"})
    fig.update_xaxes(title_text="Time [h]", row=2, col=1)
    return fig

def fig_innovation(log):
    t = log["t"]/3600
    innov_mv = log["innov"]*1000
    noise_2s = float(log.get("noise_std_mv", 10.0)) * 2
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Innovation Residuals ν(k) [mV]",
                        "Normalised Innovation Squared — NIS"),
        vertical_spacing=0.12)
    fig.add_trace(go.Scattergl(x=t, y=innov_mv, name="ν(k)",
        line=dict(color=COLORS["innov"],width=1.2), mode="lines"), row=1, col=1)
    fig.add_hline(y=noise_2s,  line_dash="dot", line_color="#f59e0b",
                  annotation_text=f"+2σ={noise_2s:.0f}mV", row=1, col=1)
    fig.add_hline(y=-noise_2s, line_dash="dot", line_color="#f59e0b", row=1, col=1)
    fig.add_trace(go.Scattergl(x=t, y=log["NIS"], name="NIS",
        line=dict(color=COLORS["nis"],width=1.5), mode="lines"), row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dot", line_color="#22c55e",
                  annotation_text="Target=1.0", row=2, col=1)
    fig.update_layout(**LAYOUT_BASE, height=440,
        title="<b>Innovation & NIS</b> — Filter Consistency Analysis")
    fig.update_xaxes(title_text="Time [h]", row=2, col=1)
    return fig

def fig_cycle_trP(df_cycles):
    c1_val = float(df_cycles["Peak trP"].iloc[0])
    colors = []
    for v in df_cycles["Peak trP"]:
        ratio = float(v)/c1_val
        if ratio < 1.1:   colors.append("#22c55e")
        elif ratio < 1.25: colors.append("#f59e0b")
        else:              colors.append("#ef4444")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_cycles["Cycle"], y=[float(v) for v in df_cycles["Peak trP"]],
        marker_color=colors, name="Peak trP",
        text=[f"{float(v):.2e}" for v in df_cycles["Peak trP"]],
        textposition="outside", textfont=dict(size=9)
    ))
    fig.update_layout(**LAYOUT_BASE, height=340,
        title="<b>Cycle-by-Cycle Peak trP</b> — Uncertainty Propagation",
        xaxis=dict(title="Cycle #", **{k:v for k,v in axis_style("").items() if k!="title"}),
        yaxis=dict(title="Peak trace(P)", **{k:v for k,v in axis_style("").items() if k!="title"}),
        bargap=0.2)
    return fig

def fig_cycle_rmse(df_cycles):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_cycles["Cycle"], y=df_cycles["RMSE SOC %"],
        name="RMSE SOC [%]", line=dict(color=COLORS["true"],width=2.2),
        marker=dict(size=6)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_cycles["Cycle"], y=df_cycles["Mean σ %"],
        name="Mean σ [%]", line=dict(color=COLORS["sigma"],width=2,dash="dash"),
        marker=dict(size=5)), secondary_y=True)
    fig.update_layout(**LAYOUT_BASE, height=340,
        title="<b>SOC Error & Uncertainty per Cycle</b>")
    fig.update_xaxes(title_text="Cycle #")
    fig.update_yaxes(title_text="RMSE SOC [%]", secondary_y=False)
    fig.update_yaxes(title_text="σ(SOC) [%]", secondary_y=True)
    return fig

def fig_current(log):
    t = log["t"]/3600
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=t, y=log["I_true"], name="Current",
        line=dict(color="#8b5cf6",width=1.8), mode="lines",
        fill="tozeroy", fillcolor="rgba(139,92,246,0.08)"))
    fig.update_layout(**LAYOUT_BASE, height=280,
        title="<b>Applied Current Profile</b>",
        xaxis=axis_style("Time","h"), yaxis=axis_style("Current","A"))
    return fig


# ═══════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════
def fig_to_base64(fig, width=900, height=None):
    """
    Renders a Plotly figure to base64 PNG using Matplotlib to avoid Kaleido dependencies.
    """
    import io, base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Calculate dimensions
    h_in = (height or 360) / 96
    w_in = width / 96

    # Create matplotlib figure
    fig_mpl, ax = plt.subplots(figsize=(w_in, h_in), dpi=100)
    fig_mpl.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    # Render traces
    for trace in fig.data:
        x = trace.x if trace.x is not None else []
        y = trace.y if trace.y is not None else []
        if len(x) > 0 and len(y) > 0:
            ax.plot(x, y, label=getattr(trace, "name", ""), linewidth=1.5)

    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.grid(color="#1e293b", alpha=0.5)

    # Render to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig_mpl.get_facecolor())
    plt.close(fig_mpl)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def build_pdf_report(cfg, log, df_cycles, sm, figs_b64):
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    chem  = cfg["chem_label"]
    ncyc  = cfg["n_cycles"]
    crate = cfg["c_rate"]
    proto = cfg["protocol"].upper()
    noise = cfg["noise_mv"]
    pqr   = cfg["pqr"]
    n_pts = len(log["t"])
    tot_h = log["t"][-1]/3600
    v_rng = f"{log['V_true'].min():.3f} – {log['V_true'].max():.3f}"
    i_rng = f"{log['I_true'].min():.2f} – {log['I_true'].max():.2f}"
    Q_nom = log["Q_nom"]
    R0    = cfg["chem"]["R0"]*1000

    # badge helper
    def badge(txt, kind="ok"):
        cls = {"ok":"#dcfce7;color:#166534","warn":"#fef9c3;color:#854d0e",
               "err":"#fee2e2;color:#991b1b"}[kind]
        return f'<span style="background:{cls};border-radius:4px;padding:2px 9px;font-size:8pt;font-weight:700">{txt}</span>'

    def assess_row(icon, txt, kind="ok"):
        bg = {"ok":"#f0fdf4;border-left:3px solid #22c55e;color:#166534",
              "warn":"#fffbeb;border-left:3px solid #f59e0b;color:#92400e",
              "err":"#fef2f2;border-left:3px solid #ef4444;color:#991b1b"}[kind]
        return f'<div style="background:{bg};border-radius:6px;padding:7px 12px;margin:5px 0;font-size:9.5pt">{icon} {txt}</div>'

    # NIS verdict
    if sm["nis_mean"] < 0.7:
        nis_v = "Over-confident"; nis_k = "warn"
    elif sm["nis_mean"] > 1.35:
        nis_v = "Under-confident"; nis_k = "warn"
    else:
        nis_v = "Well-calibrated ✓"; nis_k = "ok"

    inn_k = "ok" if sm["inn_ratio"] < 1.5 else ("warn" if sm["inn_ratio"] < 3.0 else "err")
    conv_k = "ok" if sm["trP_conv"] < 15 else ("warn" if sm["trP_conv"] < 40 else "err")

    # cycle rows
    cyc_rows = ""
    for _, r in df_cycles.iterrows():
        growth = ""
        if r["Cycle"] == 1:
            st = badge("Baseline","ok"); growth = "—"
        else:
            base = float(df_cycles["Peak trP"].iloc[0])
            cur  = float(r["Peak trP"])
            pct  = (cur-base)/base*100
            if abs(pct) < 10: st = badge("Stable","ok")
            elif pct < 25:    st = badge("Growing","warn")
            else:             st = badge("Alert","err")
            growth = f"{pct:+.1f}%"
        cyc_rows += f"""<tr>
            <td style="text-align:center">{int(r['Cycle'])}</td>
            <td>{r['SOC Start %']:.1f}</td><td>{r['SOC Min %']:.1f}</td>
            <td>{r['SOC End %']:.1f}</td><td>{r['Dur (min)']:.1f}</td>
            <td>{r['Peak trP']}</td><td>{growth}</td>
            <td>{r['Mean σ %']:.3f}</td><td>{st}</td>
        </tr>"""

    # Tuning recommendations
    recs = []
    if sm["inn_ratio"] > 1.5:
        recs.append(("P₀ scale", pqr.split()[0], "Try ×0.1 to reduce over-confidence"))
    if sm["nis_mean"] > 1.3:
        recs.append(("Q scale", pqr.split()[1], "Increase to track faster dynamics"))
    if sm["inn_ratio"] > 2.0:
        recs.append(("R scale", pqr.split()[2], "Increase if measurement noise is larger"))
    recs.append(("Sensor noise σ", f"{noise} mV", "Calibrate against real sensor floor"))
    rec_rows = "".join(f"<tr><td>{p}</td><td>{c}</td><td>{r}</td></tr>" for p,c,r in recs)

    HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>BattSim v5.0 — {chem} — {ncyc} Cycles</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',sans-serif;font-size:11pt;color:#1a1a2e;background:white;line-height:1.6}}
.page{{width:210mm;min-height:297mm;padding:18mm 20mm 16mm;page-break-after:always;position:relative}}
.page:last-child{{page-break-after:avoid}}
.page-header{{display:flex;justify-content:space-between;align-items:center;
    border-bottom:2px solid #0284c7;padding-bottom:8px;margin-bottom:18px}}
.logo-text{{font-size:17pt;font-weight:700;color:#0284c7}}
.logo-sub{{font-size:8pt;color:#6e7681;margin-top:2px}}
.page-meta{{text-align:right;font-size:8pt;color:#6e7681}}
.page-footer{{position:absolute;bottom:10mm;left:20mm;right:20mm;
    display:flex;justify-content:space-between;font-size:7.5pt;
    color:#aaa;border-top:1px solid #e5e7eb;padding-top:5px}}
.title-body{{display:flex;flex-direction:column;align-items:center;
    justify-content:center;min-height:230mm;text-align:center}}
.main-title{{font-size:28pt;font-weight:700;color:#0f172a;margin-bottom:8px}}
.main-subtitle{{font-size:13pt;color:#0284c7;margin-bottom:28px}}
.config-box{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
    padding:20px 32px;margin:20px 0;width:100%;max-width:440px;text-align:left}}
.config-row{{display:flex;justify-content:space-between;padding:5px 0;
    border-bottom:1px solid #f1f5f9;font-size:10pt}}
.config-row:last-child{{border-bottom:none}}
.config-label{{color:#64748b}}.config-val{{font-weight:600;font-family:'JetBrains Mono',monospace;color:#1e293b}}
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}}
.kpi-card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 10px;text-align:center}}
.kpi-label{{font-size:7.5pt;color:#64748b;text-transform:uppercase;letter-spacing:.05em}}
.kpi-value{{font-size:16pt;font-weight:700;color:#0284c7;font-family:'JetBrains Mono',monospace}}
.kpi-unit{{font-size:7.5pt;color:#94a3b8}}
h2{{font-size:13pt;font-weight:700;color:#0f172a;border-left:4px solid #0284c7;
    padding-left:10px;margin:16px 0 8px}}
h3{{font-size:11pt;font-weight:600;color:#1e293b;margin:12px 0 5px}}
p{{margin-bottom:7px;color:#374151;font-size:10.5pt}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:9.5pt}}
th{{background:#0f172a;color:white;padding:7px 10px;text-align:left;font-weight:600;font-size:9pt}}
td{{padding:6px 10px;border-bottom:1px solid #e2e8f0;color:#374151;
    font-family:'JetBrains Mono',monospace;font-size:9pt}}
tr:nth-child(even) td{{background:#f8fafc}}
.formula-box{{background:#f1f5f9;border:1px solid #e2e8f0;border-radius:6px;
    padding:10px 14px;font-family:'JetBrains Mono',monospace;font-size:9.5pt;
    color:#1e293b;margin:8px 0}}
.divider{{border:none;border-top:1px solid #e5e7eb;margin:14px 0}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:10px 0}}
.fig-wrap{{margin:12px 0;text-align:center}}
.fig-wrap img{{max-width:100%;border-radius:6px;border:1px solid #e2e8f0}}
.fig-cap{{font-size:8pt;color:#64748b;margin-top:4px;font-style:italic}}
@media print{{body{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
</style></head><body>

<!-- PAGE 1 COVER -->
<div class="page">
<div class="page-header">
  <div><div class="logo-text">BattSim v5.0</div>
       <div class="logo-sub">Digital Twin Co-Simulation Framework</div></div>
  <div class="page-meta">Generated: {now}<br>Author: Eng. Thaer Abushawar</div>
</div>
<div class="title-body">
  <div class="main-title">Simulation Report</div>
  <div class="main-subtitle">DFN Physical Asset ↔ EKF Digital Observer</div>
  <div class="config-box">
    <div class="config-row"><span class="config-label">Cell Chemistry</span><span class="config-val">{chem}</span></div>
    <div class="config-row"><span class="config-label">Test Protocol</span><span class="config-val">{proto}</span></div>
    <div class="config-row"><span class="config-label">Number of Cycles</span><span class="config-val">{ncyc}</span></div>
    <div class="config-row"><span class="config-label">C-Rate</span><span class="config-val">{crate:.1f}C</span></div>
    <div class="config-row"><span class="config-label">Sensor Noise σ</span><span class="config-val">{noise} mV</span></div>
    <div class="config-row"><span class="config-label">P₀ | Q | R scale</span><span class="config-val">{pqr}</span></div>
    <div class="config-row"><span class="config-label">Data Points</span><span class="config-val">{n_pts:,} (dt=10 s)</span></div>
  </div>
  <div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-label">Voltage RMSE</div>
      <div class="kpi-value">{sm['v_rmse']:.2f}</div><div class="kpi-unit">mV</div></div>
    <div class="kpi-card"><div class="kpi-label">SOC RMSE</div>
      <div class="kpi-value">{sm['soc_rmse']:.2f}</div><div class="kpi-unit">%</div></div>
    <div class="kpi-card"><div class="kpi-label">Peak σ(SOC)</div>
      <div class="kpi-value">{sm['sigma_max']:.2f}</div><div class="kpi-unit">%</div></div>
    <div class="kpi-card"><div class="kpi-label">Cycles Run</div>
      <div class="kpi-value">{ncyc}</div><div class="kpi-unit">@ {crate:.1f}C</div></div>
  </div>
  <p style="font-size:8pt;color:#94a3b8;margin-top:16px">
  </p>
</div>
<div class="page-footer"><span>BattSim v5.0 — Confidential Simulation Report</span><span>Page 1</span></div>
</div>

<!-- PAGE 2 MACHINE 1 -->
<div class="page">
<div class="page-header">
  <div><div class="logo-text">Machine 1 — Physical Asset</div>
       <div class="logo-sub">DFN-Surrogate Electrochemical Simulator</div></div>
  <div class="page-meta">{chem} | {proto} | {crate:.1f}C</div>
</div>
<h2>DFN Simulator Configuration</h2>
<p>Machine 1 emulates the physical battery using a 2-RC ECM with parameters validated against
PyBaMM DFN outputs. It provides ground-truth V, SOC, I, T traces.
Machine 2 observes only V<sub>noisy</sub> = V<sub>true</sub> + 𝒩(0, {noise} mV).</p>
<div class="two-col">
<div>
<h3>Cell Parameters</h3>
<table><tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Q nominal</td><td>{Q_nom:.2f} Ah</td></tr>
<tr><td>R₀ (ECM)</td><td>{R0:.1f} mΩ</td></tr>
<tr><td>R₁C₁</td><td>{cfg['chem']['R1']*1000:.0f} mΩ · {cfg['chem']['C1']:.0f} F</td></tr>
<tr><td>R₂C₂</td><td>{cfg['chem']['R2']*1000:.0f} mΩ · {cfg['chem']['C2']:.0f} F</td></tr>
<tr><td>V range</td><td>{cfg['chem']['v_min']}–{cfg['chem']['v_max']} V</td></tr>
</table></div>
<div>
<h3>Simulation Summary</h3>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total time</td><td>{tot_h:.2f} h</td></tr>
<tr><td>Data points</td><td>{n_pts:,}</td></tr>
<tr><td>V<sub>true</sub> range</td><td>{v_rng} V</td></tr>
<tr><td>I range</td><td>{i_rng} A</td></tr>
<tr><td>T range</td><td>25.0–25.0 °C</td></tr>
</table></div>
</div>
<hr class="divider">
<h2>Voltage Tracking — DFN vs EKF</h2>
<div class="fig-wrap">
  <img src="data:image/png;base64,{figs_b64['voltage']}">
  <div class="fig-cap">Fig. 1 — Voltage: DFN ground truth (solid) vs EKF reconstruction (dashed). RMSE = {sm['v_rmse']:.2f} mV</div>
</div>
<hr class="divider">
<h2>State Equations</h2>
<div class="formula-box">
V<sub>true</sub>(t) = OCV(SOC) + I·R₀ + V<sub>RC1</sub> + V<sub>RC2</sub><br>
V<sub>noisy</sub>(t) = V<sub>true</sub>(t) + 𝒩(0, σ={noise} mV)<br>
SOC(t) = SOC(0) − ∫ I / (Q·3600) dt  [Coulomb counting]
</div>
<div class="page-footer"><span>BattSim v5.0 — Confidential Simulation Report</span><span>Page 2</span></div>
</div>

<!-- PAGE 3 MACHINE 2 -->
<div class="page">
<div class="page-header">
  <div><div class="logo-text">Machine 2 — Digital Observer</div>
       <div class="logo-sub">2-RC ECM + Extended Kalman Filter</div></div>
  <div class="page-meta">P₀={pqr.split()[0]} | Q={pqr.split()[1]} | R={pqr.split()[2]}</div>
</div>
<h2>SOC Estimation — with 95% Confidence Band</h2>
<div class="fig-wrap">
  <img src="data:image/png;base64,{figs_b64['soc']}">
</div>
<hr class="divider">
<h2>EKF Performance Metrics</h2>
<div class="two-col">
<div>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Voltage RMSE</td><td>{sm['v_rmse']:.3f} mV</td></tr>
<tr><td>SOC RMSE</td><td>{sm['soc_rmse']:.4f} %</td></tr>
<tr><td>Max SOC error</td><td>{sm['soc_max']:.4f} %</td></tr>
<tr><td>Mean σ(SOC)</td><td>{sm['sigma_mean']:.4f} %</td></tr>
<tr><td>Max σ(SOC)</td><td>{sm['sigma_max']:.4f} %</td></tr>
<tr><td>trP initial</td><td>{sm['trP_init']:.3e}</td></tr>
<tr><td>trP final</td><td>{sm['trP_final']:.3e}</td></tr>
<tr><td>Convergence</td><td>{sm['trP_conv']:.2f}% of P₀</td></tr>
</table>
</div>
<div>
<h3>Innovation Whiteness</h3>
<table><tr><th>Metric</th><th>Value</th><th>Status</th></tr>
<tr><td>Inn. RMS</td><td>{sm['inn_rms']:.3f} mV</td><td>{badge("PASS","ok") if sm['inn_rms']<15 else badge("REVIEW","warn")}</td></tr>
<tr><td>Inn./Noise ratio</td><td>{sm['inn_ratio']:.3f}</td><td>{badge("OK","ok") if sm['inn_ratio']<1.5 else badge("REVIEW","warn")}</td></tr>
<tr><td>Inn. mean</td><td>{sm['inn_mean']:.3f} mV</td><td>{badge("PASS","ok") if abs(sm['inn_mean'])<5 else badge("REVIEW","warn")}</td></tr>
<tr><td>NIS mean</td><td>{sm['nis_mean']:.4f}</td><td>{badge(nis_v, nis_k)}</td></tr>
</table>
<h3 style="margin-top:14px">EKF State Equations</h3>
<div class="formula-box" style="font-size:8.5pt">
PREDICT: x⁻ = Ax + BI, P⁻ = AP A' + Q<br>
UPDATE:  K = P⁻C'(CP⁻C'+R)⁻¹<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; x = x⁻ + K(V − ĥ(x⁻))<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; P = (I−KC)P⁻(I−KC)' + KRK'
</div>
</div>
</div>
<div class="page-footer"><span>BattSim v5.0 — Confidential Simulation Report</span><span>Page 3</span></div>
</div>

<!-- PAGE 4 UNCERTAINTY -->
<div class="page">
<div class="page-header">
  <div><div class="logo-text">Uncertainty Analytics</div>
       <div class="logo-sub">Cycle-by-Cycle Covariance Propagation</div></div>
  <div class="page-meta">{ncyc} Cycles | {crate:.1f}C</div>
</div>
<h2>Covariance Dynamics — trP & σ(SOC)</h2>
<div class="fig-wrap">
  <img src="data:image/png;base64,{figs_b64['uncertainty']}">
  <div class="fig-cap">Fig. 3 — State covariance trace(P) and σ(SOC) over time. trP→0 confirms EKF convergence.</div>
</div>
<hr class="divider">
<h2>Innovation & NIS Analysis</h2>
<div class="fig-wrap">
  <img src="data:image/png;base64,{figs_b64['innovation']}">
  <div class="fig-cap">Fig. 4 — Innovation residuals ν(k) and NIS. White noise within ±2σ = well-tuned EKF.</div>
</div>
<div class="page-footer"><span>BattSim v5.0 — Confidential Simulation Report</span><span>Page 4</span></div>
</div>

<!-- PAGE 5 CYCLE TABLE -->
<div class="page">
<div class="page-header">
  <div><div class="logo-text">Cycle-by-Cycle Summary</div>
       <div class="logo-sub">Peak trP & SOC Uncertainty per Cycle</div></div>
  <div class="page-meta">{ncyc} Cycles | {crate:.1f}C</div>
</div>
<div class="fig-wrap" style="margin-bottom:8px">
  <img src="data:image/png;base64,{figs_b64['trP_bar']}">
  <div class="fig-cap">Fig. 5 — Peak trP per cycle. Green: stable, Yellow: growing (&lt;25%), Red: alert (&gt;25%)</div>
</div>
<hr class="divider">
<h2>Cycle Statistics Table</h2>
<table>
<tr>
  <th>#</th><th>SOC Start %</th><th>SOC Min %</th><th>SOC End %</th>
  <th>Dur (min)</th><th>Peak trP</th><th>Δ vs C1</th>
  <th>Mean σ %</th><th>Status</th>
</tr>
{cyc_rows}
</table>
<div class="page-footer"><span>BattSim v5.0 — Confidential Simulation Report</span><span>Page 5</span></div>
</div>

<!-- PAGE 6 ASSESSMENT -->
<div class="page">
<div class="page-header">
  <div><div class="logo-text">Engineering Assessment</div>
       <div class="logo-sub">Diagnostics, Tuning & Recommendations</div></div>
  <div class="page-meta">{now}</div>
</div>
<div class="fig-wrap">
  <img src="data:image/png;base64,{figs_b64['rmse_sigma']}">
  <div class="fig-cap">Fig. 6 — RMSE SOC and mean σ(SOC) per cycle.</div>
</div>
<hr class="divider">
<h2>System Assessment</h2>
{assess_row("●", f"EKF Innovation/Noise Ratio: {sm['inn_ratio']:.2f} (target &lt; 1.5)", "ok" if sm['inn_ratio']<1.5 else ("warn" if sm['inn_ratio']<3.0 else "err"))}
{assess_row("●", f"EKF Convergence: trP = {sm['trP_conv']:.1f}% of P₀ (target &lt; 15%)", "ok" if sm['trP_conv']<15 else ("warn" if sm['trP_conv']<40 else "err"))}
{assess_row("●", f"NIS Calibration: {nis_v} (mean NIS = {sm['nis_mean']:.3f}, target ≈ 1.0)", nis_k)}
{assess_row("●", f"Uncertainty Stability: Max σ(SOC) = {sm['sigma_max']:.3f}%", "ok" if sm['sigma_max']<2.0 else "warn")}
{assess_row("●", f"SOC RMSE = {sm['soc_rmse']:.4f}% | Max error = {sm['soc_max']:.4f}%", "ok" if sm['soc_rmse']<3 else "warn")}
<hr class="divider">
<h2>Tuning Recommendations</h2>
<table><tr><th>Parameter</th><th>Current</th><th>Recommendation</th></tr>
{rec_rows}
</table>
<hr class="divider">
<p style="font-size:8.5pt;color:#94a3b8;text-align:center;margin-top:16px">
  <strong>BattSim v5.0</strong> — Digital Twin Co-Simulation Framework<br>
  Designed &amp; Developed by <strong>Eng. Thaer Abushawar</strong> | Thaer199@gmail.com<br>
</p>
<div class="page-footer">
  <span>BattSim v5.0 — Confidential Simulation Report</span><span>Page 6</span>
</div>
</div>

<script>window.onload=function(){{window.print()}}</script>
</body></html>"""
    return HTML


# ═══════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="battsim-header">
  <div>
    <div class="header-title">🔋 BattSim <span class="header-badge">v5.0</span></div>
    <div class="header-subtitle">Digital Twin Co-Simulation — DFN Physical Asset ↔ EKF Digital Observer</div>
    <div class="header-author">Eng. Thaer Abushawar &nbsp;|&nbsp; Thaer199@gmail.com</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Simulation Settings")
    st.markdown("<hr style='margin:6px 0 14px;border-color:#e2e8f0'>", unsafe_allow_html=True)

    chem_key = st.selectbox(
        "Cell Chemistry",
        list(CHEMISTRY_DB.keys()),
        help="Select validated parameter set"
    )
    chem = CHEMISTRY_DB[chem_key]

    st.markdown("**Cycle Parameters**")
    n_cycles = st.slider("Number of Cycles", 1, 50, 5, 1)
    c_rate   = st.select_slider("C-Rate", [0.2, 0.5, 1.0, 1.5, 2.0, 3.0], value=1.0)
    protocol = st.radio("Charge Protocol", ["cccv", "cc"], horizontal=True)

    st.markdown("<hr style='margin:10px 0;border-color:#e2e8f0'>", unsafe_allow_html=True)
    st.markdown("**EKF Tuning**")
    noise_mv = st.slider("Sensor Noise σ (mV)", 1, 30, 10, 1)
    P0_exp   = st.select_slider("P₀ scale (10^x)", [-5,-4,-3,-2,-1], value=-3)
    Q_scale  = st.select_slider("Q scale", [0.01,0.1,0.5,1.0,2.0,5.0,10.0], value=1.0)
    R_scale  = st.select_slider("R scale", [0.01,0.1,0.5,1.0,2.0,5.0,10.0], value=1.0)
    P0_val   = 10**P0_exp

    st.markdown("<hr style='margin:10px 0;border-color:#e2e8f0'>", unsafe_allow_html=True)
    st.caption(f"Q_nom = {chem['Q']:.1f} Ah | V: {chem['v_min']}–{chem['v_max']} V")
    st.caption(f"R₀={chem['R0']*1000:.0f}mΩ | R₁C₁={chem['R1']*1000:.0f}mΩ·{chem['C1']:.0f}F")

    run_btn = st.button("▶ Run Simulation", use_container_width=True)

# ── Session state ──────────────────────────────────────────────────
if "log" not in st.session_state:
    st.session_state.log = None

# ── Run ────────────────────────────────────────────────────────────
if run_btn:
    prog = st.progress(0, text="Initialising simulation…")
    with st.spinner(""):
        prog.progress(10, "Machine 1: Running DFN simulator…")
        t, V, I, soc, T, Q_nom = run_machine1(
            chem, n_cycles, c_rate, protocol=protocol, dt=10.0
        )
        prog.progress(55, "Machine 2: Running EKF observer…")
        log = run_cosim(
            t, V, I, soc, T, Q_nom, chem,
            noise_std=noise_mv/1000,
            P0_scale=P0_val, Q_scale=Q_scale, R_scale=R_scale,
        )
        log["noise_std_mv"] = noise_mv
        prog.progress(90, "Computing statistics…")
        st.session_state.log   = log
        st.session_state.cfg   = {
            "chem": chem, "chem_label": chem["label"],
            "n_cycles": n_cycles, "c_rate": c_rate,
            "protocol": protocol, "noise_mv": noise_mv,
            "pqr": f"{P0_val:.0e} {Q_scale} {R_scale}",
        }
        st.session_state.df_cyc = cycle_stats(log)
        st.session_state.sm     = summary_metrics(log)
        prog.progress(100, "Complete ✓")
    st.success(f"✓ Simulation complete — {len(t):,} timesteps | {t[-1]/3600:.1f} h")

# ── Results ────────────────────────────────────────────────────────
if st.session_state.log is not None:
    log     = st.session_state.log
    cfg     = st.session_state.cfg
    df_cyc  = st.session_state.df_cyc
    sm      = st.session_state.sm
    chem    = cfg["chem"]

    # ── KPI Row ────────────────────────────────────────────────────
    def kpi_html(label, value, unit, kind="blue"):
        return f"""<div class="kpi-card kpi-{kind}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-unit">{unit}</div>
        </div>"""

    nis_k  = "ok" if abs(sm["nis_mean"]-1.0)<0.35 else "warn"
    conv_k = "ok" if sm["trP_conv"]<15 else ("warn" if sm["trP_conv"]<40 else "err")
    inn_k  = "ok" if sm["inn_ratio"]<1.5 else ("warn" if sm["inn_ratio"]<3.0 else "err")
    soc_k  = "ok" if sm["soc_rmse"]<2 else ("warn" if sm["soc_rmse"]<5 else "err")

    st.markdown(f"""
    <div class="kpi-row">
      {kpi_html("Voltage RMSE", f"{sm['v_rmse']:.2f}", "mV", "blue")}
      {kpi_html("SOC RMSE", f"{sm['soc_rmse']:.3f}", "%", soc_k)}
      {kpi_html("Max SOC Error", f"{sm['soc_max']:.3f}", "%", "warn" if sm['soc_max']>5 else "ok")}
      {kpi_html("Mean σ(SOC)", f"{sm['sigma_mean']:.3f}", "%", "purple")}
      {kpi_html("NIS Mean", f"{sm['nis_mean']:.3f}", "target≈1.0", nis_k)}
      {kpi_html("Convergence", f"{sm['trP_conv']:.1f}", "% of P₀", conv_k)}
      {kpi_html("Inn./Noise", f"{sm['inn_ratio']:.2f}", "target<1.5", inn_k)}
      {kpi_html("Data Points", f"{len(log['t']):,}", f"{log['t'][-1]/3600:.1f}h", "blue")}
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Voltage & SOC",
        "📊 Uncertainty",
        "🔍 Innovation",
        "🔄 Cycle Analysis",
        "📋 Data Table",
    ])

    with tab1:
        st.markdown('<div class="section-header">Voltage Tracking — DFN Ground Truth vs EKF</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_voltage(log), use_container_width=True)
        st.markdown('<div class="section-header">SOC Estimation with 95% Confidence Band</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_soc(log), use_container_width=True)
        st.markdown('<div class="section-header">Applied Current Profile</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_current(log), use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">EKF Covariance Dynamics</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_uncertainty(log), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("trP initial",  f"{sm['trP_init']:.3e}")
            st.metric("trP final",    f"{sm['trP_final']:.3e}")
        with c2:
            st.metric("Convergence",  f"{sm['trP_conv']:.2f}% of P₀")
            st.metric("Max σ(SOC)",   f"{sm['sigma_max']:.4f} %")

    with tab3:
        st.markdown('<div class="section-header">Innovation Residuals & NIS Analysis</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_innovation(log), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Innovation RMS",  f"{sm['inn_rms']:.2f} mV",
                  delta="OK" if sm['inn_rms']<15 else "REVIEW")
        c2.metric("Inn./Noise Ratio",f"{sm['inn_ratio']:.3f}",
                  delta="OK" if sm['inn_ratio']<1.5 else "REVIEW")
        c3.metric("Innovation Mean", f"{sm['inn_mean']:.2f} mV",
                  delta="zero-mean ✓" if abs(sm['inn_mean'])<5 else "biased ⚠")

    with tab4:
        st.markdown('<div class="section-header">Cycle-by-Cycle Peak trP</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_cycle_trP(df_cyc), use_container_width=True)
        st.markdown('<div class="section-header">SOC RMSE & Uncertainty per Cycle</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_cycle_rmse(df_cyc), use_container_width=True)

    with tab5:
        st.markdown('<div class="section-header">Cycle Statistics Table</div>', unsafe_allow_html=True)
        st.dataframe(df_cyc, use_container_width=True, height=420)
        # Download CSV
        csv = df_cyc.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv, "battsim_cycles.csv","text/csv")

    # ── PDF Report Button ─────────────────────────────────────────
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📄 Export PDF Report</div>', unsafe_allow_html=True)

    gen_pdf = st.button("📄 Generate PDF Report", use_container_width=False)
    if gen_pdf:
        with st.spinner("Rendering charts for PDF…"):
            # render all figs to base64 PNG
            figs_b64 = {
                "voltage":    fig_to_base64(fig_voltage(log),    width=860, height=320),
                "soc":        fig_to_base64(fig_soc(log),        width=860, height=320),
                "uncertainty":fig_to_base64(fig_uncertainty(log),width=860, height=380),
                "innovation": fig_to_base64(fig_innovation(log), width=860, height=380),
                "trP_bar":    fig_to_base64(fig_cycle_trP(df_cyc),width=860,height=300),
                "rmse_sigma": fig_to_base64(fig_cycle_rmse(df_cyc),width=860,height=300),
            }
            html_report = build_pdf_report(cfg, log, df_cyc, sm, figs_b64)
            b64_html    = base64.b64encode(html_report.encode()).decode()
            fname       = f"BattSim_v5_{cfg['chem_label'].replace(' ','_')}_{cfg['n_cycles']}cyc.html"

        st.success("✓ Report ready — click below to open and print as PDF (Ctrl+P)")
        st.download_button(
            label="⬇ Download Report (HTML → Print as PDF)",
            data=html_report.encode(),
            file_name=fname,
            mime="text/html",
            use_container_width=True,
        )
        st.info("💡 Open the downloaded file in Chrome → Ctrl+P → Save as PDF")

# ── Empty state ────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#94a3b8">
      <div style="font-size:4rem">🔋</div>
      <h2 style="color:#64748b;font-weight:600;margin:1rem 0 0.5rem">Ready to Simulate</h2>
      <p style="max-width:400px;margin:0 auto;font-size:0.95rem">
        Configure the parameters in the sidebar and click
        <strong style="color:#0284c7">▶ Run Simulation</strong> to begin.
      </p>
      <div style="margin-top:2rem;display:flex;justify-content:center;gap:2rem;flex-wrap:wrap">
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px 24px;min-width:160px">
          <div style="font-size:1.5rem">⚡</div>
          <div style="font-weight:600;color:#374151;margin-top:6px">DFN Simulator</div>
          <div style="font-size:0.8rem;color:#94a3b8">Physical Asset</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px 24px;min-width:160px">
          <div style="font-size:1.5rem">🎯</div>
          <div style="font-weight:600;color:#374151;margin-top:6px">EKF Observer</div>
          <div style="font-size:0.8rem;color:#94a3b8">Digital Observer</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px 24px;min-width:160px">
          <div style="font-size:1.5rem">📊</div>
          <div style="font-weight:600;color:#374151;margin-top:6px">UQ Analytics</div>
          <div style="font-size:0.8rem;color:#94a3b8">Uncertainty Propagation</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px 24px;min-width:160px">
          <div style="font-size:1.5rem">📄</div>
          <div style="font-weight:600;color:#374151;margin-top:6px">PDF Report</div>
          <div style="font-size:0.8rem;color:#94a3b8">6-Page Export</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  <strong>BattSim v5.0</strong> — Digital Twin Co-Simulation Framework<br>
  Designed &amp; Developed by <strong>Eng. Thaer Abushawar</strong> &nbsp;|&nbsp;
  <a href="mailto:Thaer199@gmail.com" style="color:#0284c7">Thaer199@gmail.com</a><br>
  <span style="font-size:0.72rem">
  </span>
</div>
""", unsafe_allow_html=True)
