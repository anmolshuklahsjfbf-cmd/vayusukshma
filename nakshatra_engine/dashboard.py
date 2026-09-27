"""
dashboard.py
============
Streamlit dashboard for the UAV aero-engine Digital Twin.

Two ways to run it:

1. STANDALONE (recommended for a quick demo, no separate backend needed):
   Runs the Pipeline directly in-process, ticking it forward each
   Streamlit rerun. Good for local development.

       streamlit run dashboard.py

2. CONNECTED TO BACKEND (closer to the real deployment architecture --
   simulator/backend runs independently, dashboard is a pure client over
   WebSocket, exactly as it would be talking to real CAN/FADEC telemetry
   relayed by backend.py):

       # terminal 1
       uvicorn backend:app --port 8000
       # terminal 2
       streamlit run dashboard.py -- --mode backend

Both modes render identical UI -- only the data source differs, which is
the point: swapping the simulator for real hardware later only touches
backend.py / pipeline.py, never this file.
"""

from __future__ import annotations

import sys
import time
import json

from collections import deque

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="NAKSHATRA-ENGINE | MALE-UAV Aero Engine Digital Twin",
    page_icon="🛰️",
    layout="wide",
)

# --------------------------------------------------------------------
# Mission-control dark theme
# --------------------------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #0a0e14; color: #d6e4f0; }
[data-testid="stMetricValue"] { color: #6fffe9; font-family: 'Courier New', monospace; }
[data-testid="stMetricLabel"] { color: #8aa0b8; }
h1, h2, h3 { color: #6fffe9 !important; font-family: 'Courier New', monospace; }
.advisory-continue { background:#0f3d24; border:1px solid #2ecc71; padding:14px; border-radius:6px; color:#2ecc71; font-weight:bold; }
.advisory-monitor { background:#3d3a0f; border:1px solid #f1c40f; padding:14px; border-radius:6px; color:#f1c40f; font-weight:bold; }
.advisory-divert { background:#3d260f; border:1px solid #e67e22; padding:14px; border-radius:6px; color:#e67e22; font-weight:bold; }
.advisory-abort, .advisory-rtb { background:#3d0f0f; border:1px solid #e74c3c; padding:14px; border-radius:6px; color:#e74c3c; font-weight:bold; }
.fault-chip { display:inline-block; background:#241a3d; border:1px solid #9b59b6; color:#c39bd3;
              padding:3px 10px; margin:3px; border-radius:12px; font-size:0.8em; font-family: monospace;}
</style>
""", unsafe_allow_html=True)


ADVISORY_CLASS = {
    "CONTINUE": "advisory-continue",
    "MONITOR": "advisory-monitor",
    "DIVERT_ADVISORY": "advisory-divert",
    "ABORT_RECOMMENDED": "advisory-abort",
    "RTB_IMMEDIATE": "advisory-rtb",
}


# --------------------------------------------------------------------
# Data source setup
# --------------------------------------------------------------------
MODE = "standalone"
if "--mode" in sys.argv:
    idx = sys.argv.index("--mode")
    if idx + 1 < len(sys.argv):
        MODE = sys.argv[idx + 1]

MAX_HISTORY = 300

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=MAX_HISTORY)
if "pipeline" not in st.session_state and MODE == "standalone":
    from pipeline import Pipeline
    st.session_state.pipeline = Pipeline(mission_name="normal_cruise", db_path="dashboard_twin.db", dt=0.4)
    st.session_state.active_faults_ui = []


def standalone_tick(n_ticks: int = 3):
    p = st.session_state.pipeline
    for _ in range(n_ticks):
        rec = p.step()
        st.session_state.history.append(rec)
    return st.session_state.history[-1]


def backend_fetch_history(limit=MAX_HISTORY):
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://localhost:8000/api/history?limit={limit}", timeout=2) as r:
            data = json.loads(r.read())
        recs = []
        for row in data:
            recs.append({
                "telemetry": row["telemetry"],
                "twin": row["twin"],
                "ml": row["ml"],
                "decision": row["decision"],
                "fault_labels": row["fault_labels"],
                "link_up": True,
            })
        return recs
    except Exception as e:
        st.error(f"Could not reach backend at localhost:8000 ({e}). Falling back to standalone mode.")
        return None


# --------------------------------------------------------------------
# Sidebar: mission control panel
# --------------------------------------------------------------------
st.sidebar.markdown("## 🛰️ MISSION CONTROL")
st.sidebar.markdown(f"**Mode:** `{MODE}`")

if MODE == "standalone":
    mission_choice = st.sidebar.selectbox(
        "Mission Profile",
        ["normal_cruise", "high_altitude", "hot_weather", "endurance", "rapid_throttle_transitions"],
    )
    if st.sidebar.button("Apply Mission Profile"):
        st.session_state.pipeline.set_mission(mission_choice)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚠️ Inject Fault")
    fault_type = st.sidebar.selectbox(
        "Fault type",
        [
            "misfire", "injector_abnormality", "cooling_degradation",
            "lubrication_degradation", "sensor_drift", "combustion_instability",
            "overheating_trend", "abnormal_vibration",
        ],
    )
    ramp_s = st.sidebar.slider("Ramp duration (s)", 10, 300, 120)
    magnitude = st.sidebar.slider("Magnitude", 0.1, 1.5, 1.0)
    extra_kwargs = {}
    if fault_type == "sensor_drift":
        extra_kwargs["channel"] = st.sidebar.selectbox(
            "Drifting channel", ["oil_pressure_kpa", "cht_c", "egt_c", "vibration_g"]
        )
    if fault_type == "injector_abnormality":
        extra_kwargs["rich"] = st.sidebar.checkbox("Rich (over-fuel) — else lean", value=False)

    if st.sidebar.button("🔴 Inject Fault Now"):
        p = st.session_state.pipeline
        current_t = p.sim._elapsed
        p.add_fault(fault_type, start_t=current_t + 2.0, ramp_seconds=ramp_s,
                    magnitude=magnitude, **extra_kwargs)
        st.session_state.active_faults_ui.append(fault_type)
        st.sidebar.success(f"Injected: {fault_type}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📡 Ground Link (degraded mode demo)")
    link_col1, link_col2 = st.sidebar.columns(2)
    if link_col1.button("Lose Link"):
        st.session_state.pipeline.degraded_mode.link_lost()
    if link_col2.button("Restore Link"):
        drained = st.session_state.pipeline.degraded_mode.link_restored()
        if drained:
            st.sidebar.info(f"Synced {len(drained)} buffered summaries from degraded-mode.")
        else:
            st.sidebar.info("Link restored. No buffered summaries (link was already up, or nothing changed).")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Simulation"):
        from pipeline import Pipeline
        st.session_state.pipeline.storage.clear()
        st.session_state.pipeline = Pipeline(mission_name="normal_cruise", db_path="dashboard_twin.db", dt=0.4)
        st.session_state.history.clear()
        st.session_state.active_faults_ui = []

auto_refresh = st.sidebar.checkbox("Auto-refresh (live stream)", value=True)


# --------------------------------------------------------------------
# Pull latest data
# --------------------------------------------------------------------
if MODE == "standalone":
    latest = standalone_tick(n_ticks=3)
    hist = list(st.session_state.history)
else:
    recs = backend_fetch_history()
    if recs is None:
        st.stop()
    hist = recs
    latest = hist[-1] if hist else None

if latest is None:
    st.warning("Waiting for first telemetry sample...")
    st.stop()

t_now = latest["telemetry"]["t"]
link_up = latest.get("link_up", True)


# --------------------------------------------------------------------
# Header
# --------------------------------------------------------------------
h1, h2, h3 = st.columns([3, 1, 1])
with h1:
    st.markdown("# 🛰️ PROJECT NAKSHATRA-ENGINE")
    st.caption("AI-Enabled Digital Twin — MALE-UAV Aero Piston Engine | Health Monitoring · Fault Prediction · Mission Reliability")
with h2:
    st.metric("Mission Time", f"{t_now:0.1f} s")
with h3:
    link_label = "🟢 LINK UP" if link_up else "🔴 LINK LOST (degraded mode)"
    st.metric("Ground Link", link_label)

st.markdown(f"**Mission Profile:** `{latest['telemetry']['mission_profile']}`")

# --------------------------------------------------------------------
# Advisory banner
# --------------------------------------------------------------------
decision = latest["decision"]
advisory = decision["advisory"]
css_class = ADVISORY_CLASS.get(advisory, "advisory-monitor")
st.markdown(
    f'<div class="{css_class}">🛡️ ADVISORY: {advisory} — {decision["reason"]}</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# --------------------------------------------------------------------
# Live engine health metrics
# --------------------------------------------------------------------
tel = latest["telemetry"]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("RPM", f"{tel['rpm']:.0f}")
c2.metric("CHT (°C)", f"{tel['cht_c']:.1f}")
c3.metric("EGT (°C)", f"{tel['egt_c']:.1f}")
c4.metric("Oil Press (kPa)", f"{tel['oil_pressure_kpa']:.1f}")
c5.metric("Oil Temp (°C)", f"{tel['oil_temp_c']:.1f}")
c6.metric("Vibration (g)", f"{tel['vibration_g']:.3f}")

c7, c8, c9, c10 = st.columns(4)
c7.metric("Fuel Flow (L/h)", f"{tel['fuel_flow_lph']:.2f}")
c8.metric("Bus Voltage (V)", f"{tel['alt_voltage_v']:.2f}")
c9.metric("EGT Cyl. Variance", f"{tel['egt_cyl_variance_c']:.1f}")
ml = latest["ml"]
c10.metric("Anomaly Score", f"{ml['anomaly_score']:.3f}",
           delta="ANOMALY" if ml["is_anomaly"] else "nominal")

# --------------------------------------------------------------------
# Active faults (ground truth chips -- shown for demo transparency)
# --------------------------------------------------------------------
fault_labels = latest.get("fault_labels", {})
if fault_labels:
    chips = "".join(f'<span class="fault-chip">{k}: {v:.2f}</span>' for k, v in fault_labels.items())
    st.markdown(f"**Ground-truth active faults (demo only):** {chips}", unsafe_allow_html=True)
else:
    st.markdown("**Ground-truth active faults:** _none_")

st.markdown("---")

# --------------------------------------------------------------------
# Time-series plots
# --------------------------------------------------------------------
if len(hist) >= 2:
    df = pd.DataFrame([{
        "t": h["telemetry"]["t"],
        "cht": h["telemetry"]["cht_c"],
        "cht_expected": h["twin"]["expected"]["cht_c"],
        "egt": h["telemetry"]["egt_c"],
        "egt_expected": h["twin"]["expected"]["egt_c"],
        "oil_pressure": h["telemetry"]["oil_pressure_kpa"],
        "oil_pressure_expected": h["twin"]["expected"]["oil_pressure_kpa"],
        "vibration": h["telemetry"]["vibration_g"],
        "anomaly_score": h["ml"]["anomaly_score"],
        "rpm": h["telemetry"]["rpm"],
    } for h in hist])

    col_a, col_b = st.columns(2)

    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.t, y=df.cht, name="CHT actual", line=dict(color="#e74c3c")))
        fig.add_trace(go.Scatter(x=df.t, y=df.cht_expected, name="CHT expected (twin)",
                                  line=dict(color="#6fffe9", dash="dot")))
        fig.update_layout(title="Cylinder Head Temp: Actual vs Digital-Twin Expected",
                           template="plotly_dark", height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.t, y=df.oil_pressure, name="Oil Pressure actual", line=dict(color="#f1c40f")))
        fig2.add_trace(go.Scatter(x=df.t, y=df.oil_pressure_expected, name="Oil Pressure expected",
                                   line=dict(color="#6fffe9", dash="dot")))
        fig2.update_layout(title="Oil Pressure: Actual vs Expected", template="plotly_dark",
                            height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.t, y=df.egt, name="EGT actual", line=dict(color="#e67e22")))
        fig3.add_trace(go.Scatter(x=df.t, y=df.egt_expected, name="EGT expected",
                                   line=dict(color="#6fffe9", dash="dot")))
        fig3.update_layout(title="Exhaust Gas Temp: Actual vs Expected", template="plotly_dark",
                            height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df.t, y=df.anomaly_score, name="Anomaly Score",
                                   line=dict(color="#9b59b6"), fill="tozeroy"))
        fig4.add_hline(y=0, line_dash="dot", line_color="#555")
        fig4.update_layout(title="AI Anomaly Score (Isolation Forest, higher = more anomalous)",
                            template="plotly_dark", height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------
# RUL panel with confidence band + SHAP explainability
# --------------------------------------------------------------------
rul_col, shap_col = st.columns(2)

with rul_col:
    st.markdown("### ⏳ Remaining Useful Life (RUL)")
    rul = ml.get("rul")
    if rul:
        rul_min = rul["rul_seconds"] / 60.0
        ci_lo, ci_hi = rul["rul_ci90"][0] / 60.0, rul["rul_ci90"][1] / 60.0
        st.metric("RUL Estimate", f"{rul_min:.1f} min",
                   delta=f"90% CI [{ci_lo:.1f}, {ci_hi:.1f}] min")
        st.progress(min(1.0, rul["current_index"] / rul["failure_threshold"]))
        st.caption(
            f"Degradation index (CHT residual): {rul['current_index']:.1f} / "
            f"{rul['failure_threshold']:.1f} (failure threshold)"
        )
    else:
        st.info("No degradation trend detected above the noise floor yet — engine health nominal, "
                "or ML layer still in training/warm-up window.")

with shap_col:
    st.markdown("### 🔎 Anomaly Explainability (SHAP)")
    top_features = ml.get("top_features") or []
    if top_features:
        exp_df = pd.DataFrame(top_features, columns=["feature", "attribution"])
        fig5 = go.Figure(go.Bar(
            x=exp_df.attribution, y=exp_df.feature, orientation="h",
            marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in exp_df.attribution],
        ))
        fig5.update_layout(title="Top features driving current anomaly score",
                            template="plotly_dark", height=250, margin=dict(t=40, b=20))
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("No active anomaly to explain right now.")

# --------------------------------------------------------------------
# Sensor drift (CUSUM) panel
# --------------------------------------------------------------------
st.markdown("### 📉 Sensor Drift Monitor (CUSUM)")
drift_flags = ml.get("drift_flags", {})
if drift_flags:
    dcols = st.columns(len(drift_flags))
    for i, (ch, v) in enumerate(drift_flags.items()):
        with dcols[i]:
            flagged = v.get("drift_flag", False)
            st.metric(ch, "DRIFT" if flagged else "stable",
                       delta=f"S+={v.get('sh', 0):.1f} / S-={v.get('sl', 0):.1f}")

st.caption(
    "Prototype for SIH / DRDO problem statement: AI-enabled Digital Twin for aero piston engine "
    "health monitoring on MALE-UAVs. All telemetry is physics-grounded simulation — no real "
    "engine hardware used. See technical_doc.md for architecture, fault models, and edge/security/"
    "fleet-learning roadmap."
)

if auto_refresh:
    time.sleep(0.4)
    st.rerun()
