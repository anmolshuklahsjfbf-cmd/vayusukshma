"""
dashboard_v2.py
================
VĀYUSŪKṢMA — Propulsion Digital Twin Mission Console.

A ground-up redesign of the dashboard per the VĀYUSŪKṢMA/DRISHTI master
spec. This file is purely a presentation layer: it does NOT modify or
replace engine_sim.py, faults.py, twin_core.py, ml_models.py,
decision_support.py, storage.py, pipeline.py, or backend.py. Every
number shown either comes directly from those modules or is derived from
their output by health_analytics.py (itself a pure, testable module with
no new physics or safety logic — DecisionEngine remains the sole
authority on the actual advisory).

HONEST SCOPE NOTE (read before demoing)
----------------------------------------
The master spec (33 sections) is very large. Everything below is real
and wired to live data unless explicitly marked otherwise. A few items
are implemented as clearly-labelled *stylized/conceptual* elements
because the backend has no corresponding signal (a fielded system would
need additional instrumentation or models to make them real):
  - "AI / Sensor / Mission model confidence" (System page) — heuristic,
    not a calibrated confidence interval.
  - Dual-engine view (System page) — runs a second, independent
    Pipeline as an illustration of the *concept*; there is no physical
    twin-engine airframe model.
  - Mission phase stepper (Mission page) — a time-fraction stylization,
    not derived from a real flight plan (none exists in this sim).
  - Security panel (System page) — status indicators are illustrative
    placeholders per the spec's own instruction not to expose
    implementation detail; no live cryptographic checks are performed.
Everything else (health scores, residuals, anomaly score, SHAP
attribution, RUL + CI, CUSUM drift, decision advisory, fault injection,
replay, what-if mission simulation) runs the real pipeline.

Run standalone:
    streamlit run dashboard_v2.py

Run against the backend (recommended for the "real deployment shape"
demo — dashboard is a pure WebSocket/REST client):
    uvicorn backend:app --port 8000        # terminal 1
    streamlit run dashboard_v2.py -- --mode backend   # terminal 2
"""

from __future__ import annotations

import sys
import time
import json
import datetime as dt
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from health_analytics import (
    compute_subsystem_health, compute_health_index, compute_mission_completion_probability,
    compute_mission_phase, MISSION_PHASES, build_engine_schematic_svg, fault_to_subsystem,
    run_whatif_scenario, run_mission_comparison, STATUS_COLORS, pct_to_status,
    build_mission_profile_svg,
)
from engine_3d import build_engine_3d_html

st.set_page_config(
    page_title="VĀYUSŪKṢMA — Propulsion Digital Twin",
    page_icon="🛩️",
    layout="wide",
)

# =====================================================================
# THEME — dark tactical / aerospace, restrained accent palette
# =====================================================================

CLR_BG = "#080b0f"
CLR_PANEL = "#0f151b"
CLR_PANEL_BORDER = "#22303c"
CLR_TEXT = "#c9d6df"
CLR_MUTED = "#5b7186"
CLR_ACCENT = "#3ddc84"      # military green
CLR_ACCENT_CYAN = "#4fd1c5"
CLR_AMBER = "#e8b93f"
CLR_ORANGE = "#f0862b"
CLR_RED = "#e6473b"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Rajdhani:wght@500;600;700&display=swap');

.stApp {{
    background:
        radial-gradient(circle at 15% 10%, #0c1319 0%, {CLR_BG} 45%),
        {CLR_BG};
    color: {CLR_TEXT};
}}
* {{ font-family: 'JetBrains Mono', 'Consolas', monospace; }}
h1, h2, h3, h4 {{ font-family: 'Rajdhani', sans-serif !important; letter-spacing: 0.03em; }}

/* subtle radar-grid backdrop on main container */
[data-testid="stAppViewContainer"] > .main {{
    background-image:
        linear-gradient(rgba(61,110,143,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(61,110,143,0.05) 1px, transparent 1px);
    background-size: 34px 34px;
}}

section[data-testid="stSidebar"] {{
    background: #0a0f14;
    border-right: 1px solid {CLR_PANEL_BORDER};
}}

[data-testid="stMetricValue"] {{ color: {CLR_ACCENT_CYAN}; font-size: 1.4rem; }}
[data-testid="stMetricLabel"] {{ color: {CLR_MUTED}; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }}
[data-testid="stMetricDelta"] {{ font-size: 0.75rem; }}

.vs-panel {{
    background: {CLR_PANEL};
    border: 1px solid {CLR_PANEL_BORDER};
    border-radius: 4px;
    padding: 14px 16px;
    margin-bottom: 12px;
}}
.vs-panel-title {{
    color: {CLR_ACCENT_CYAN};
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
    border-bottom: 1px solid {CLR_PANEL_BORDER};
    padding-bottom: 6px;
    margin-bottom: 10px;
}}

.vs-topbar {{
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(180deg, #0d1319 0%, #0a0f14 100%);
    border: 1px solid {CLR_PANEL_BORDER};
    border-radius: 4px;
    padding: 10px 20px;
    margin-bottom: 14px;
}}
.vs-wordmark {{ font-size: 1.5rem; font-weight: 700; color: {CLR_ACCENT}; letter-spacing: 0.08em; font-family: 'Rajdhani', sans-serif;}}
.vs-subtitle {{ font-size: 0.68rem; color: {CLR_MUTED}; letter-spacing: 0.15em; }}
.vs-live-dot {{
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background: {CLR_ACCENT}; margin-right:6px; box-shadow: 0 0 6px {CLR_ACCENT};
}}
.vs-live-dot.lost {{ background: {CLR_RED}; box-shadow: 0 0 6px {CLR_RED}; }}

.vs-chip {{
    display:inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.05em;
    border: 1px solid currentColor; margin: 2px 4px 2px 0;
}}

.vs-status-strip {{ display:flex; gap: 18px; font-size: 0.72rem; color: {CLR_MUTED}; }}
.vs-status-strip b {{ color: {CLR_TEXT}; }}

.vs-advisory-banner {{
    padding: 16px 20px; border-radius: 4px; font-size: 1.05rem; font-weight: 700;
    letter-spacing: 0.04em; border-left: 5px solid; margin-bottom: 14px;
}}
.vs-subsys-row {{
    display:flex; justify-content: space-between; align-items:center;
    padding: 7px 4px; border-bottom: 1px solid {CLR_PANEL_BORDER}; font-size: 0.85rem;
}}
.vs-subsys-bar-bg {{ background:#1a232c; border-radius:3px; height:6px; width:110px; overflow:hidden; }}
.vs-subsys-bar-fill {{ height:6px; border-radius:3px; }}

.vs-fault-chip {{
    display:inline-block; background:#1a1428; border:1px solid #9b59b6; color:#c9a3d9;
    padding:3px 10px; margin:3px; border-radius:12px; font-size:0.75rem;
}}

.vs-flow-step {{
    background: {CLR_PANEL}; border:1px solid {CLR_PANEL_BORDER}; border-radius:4px;
    padding: 8px 14px; margin: 4px 0; font-size: 0.8rem; color: {CLR_TEXT};
    border-left: 3px solid {CLR_MUTED};
}}
.vs-flow-step.active {{ border-left: 3px solid {CLR_ACCENT}; color: {CLR_ACCENT_CYAN}; }}
.vs-flow-arrow {{ text-align:center; color: {CLR_MUTED}; font-size: 0.9rem; margin: -2px 0; }}

hr {{ border-color: {CLR_PANEL_BORDER} !important; }}
</style>
""", unsafe_allow_html=True)


def status_color(status: str) -> str:
    return STATUS_COLORS.get(status, CLR_MUTED)


def vs_panel_start(title: str):
    st.markdown(f'<div class="vs-panel"><div class="vs-panel-title">{title}</div>', unsafe_allow_html=True)


def vs_panel_end():
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# Data source / session-state setup
# =====================================================================

MODE = "standalone"
if "--mode" in sys.argv:
    idx = sys.argv.index("--mode")
    if idx + 1 < len(sys.argv):
        MODE = sys.argv[idx + 1]

MAX_HISTORY = 400

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=MAX_HISTORY)
if "pipeline" not in st.session_state and MODE == "standalone":
    from pipeline import Pipeline
    st.session_state.pipeline = Pipeline(mission_name="normal_cruise", db_path="vayusuksma.db", dt=0.4)
if "planned_mission_s" not in st.session_state:
    st.session_state.planned_mission_s = 3600.0
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "MISSION"
if "engine_b_pipeline" not in st.session_state and MODE == "standalone":
    from pipeline import Pipeline
    st.session_state.engine_b_pipeline = Pipeline(
        mission_name="normal_cruise", db_path="vayusuksma_engine_b.db", dt=0.4, seed=99,
        compute_explanations=False,
    )
if "demo_mode_step" not in st.session_state:
    st.session_state.demo_mode_step = 0
if "replay_rows" not in st.session_state:
    st.session_state.replay_rows = None
if "replay_idx" not in st.session_state:
    st.session_state.replay_idx = 0
if "replay_playing" not in st.session_state:
    st.session_state.replay_playing = False
if "replay_speed" not in st.session_state:
    st.session_state.replay_speed = 1
if "last_tick_wall_time" not in st.session_state:
    st.session_state.last_tick_wall_time = time.time()

# --- Derived consumable-quantity state (see build_analog_gauges_html) ---
# Fuel quantity: genuinely integrated from the real fuel_flow_lph channel
# against an assumed tank capacity -- this simulation has no fuel-tank
# state of its own, so the *quantity* is a real derived integral, not a
# separately-modelled physical quantity.
if "fuel_capacity_l" not in st.session_state:
    st.session_state.fuel_capacity_l = 95.0   # representative small-UAV tank size
if "fuel_consumed_l" not in st.session_state:
    st.session_state.fuel_consumed_l = 0.0
# Oil quantity: real aero engines don't meaningfully consume oil over a
# single flight under normal operation -- the only physically honest
# reason an "oil quantity" gauge should move at all is a genuine leak,
# which in this simulation is exactly what lubrication_degradation
# represents. So oil level only drops when that fault is actually active,
# proportional to its severity -- it is not an independently simulated
# tank level.
if "oil_level_pct" not in st.session_state:
    st.session_state.oil_level_pct = 100.0


def _integrate_consumables(rec: dict, dt_s: float):
    tel = rec["telemetry"]
    st.session_state.fuel_consumed_l += tel["fuel_flow_lph"] * (dt_s / 3600.0)
    lube_severity = 0.0
    for k, v in (rec.get("fault_labels") or {}).items():
        if k.split(":")[0] == "lubrication_degradation":
            lube_severity = max(lube_severity, v)
    if lube_severity > 0:
        # Illustrative leak rate: full severity drains ~4%/min of level.
        drain_pct_per_s = lube_severity * (4.0 / 60.0)
        st.session_state.oil_level_pct = max(0.0, st.session_state.oil_level_pct - drain_pct_per_s * dt_s)


def standalone_tick_catchup(tick_engine_b: bool = True, max_ticks: int = 300):
    """Advances the pipeline by however many ticks correspond to real
    wall-clock time elapsed since the last tick, rather than a fixed
    count per script run. This decouples the simulated engine clock from
    the 'Live stream (auto-refresh)' checkbox -- that checkbox should
    only control whether the page keeps automatically re-polling/
    re-rendering, not whether simulated time itself is frozen. Without
    this, turning auto-refresh off effectively paused the simulation
    (only ~1.2s of sim time advanced per manual click), which meant
    pages like RUL & DEGRADATION could sit "stuck" in their warm-up
    window indefinitely even though real time was passing.
    `max_ticks` caps a runaway catch-up burst if the tab was left idle
    for a long time (e.g. minutes) -- we don't try to fast-forward the
    full idle gap in one script run, just a bounded, reasonable amount.
    """
    p = st.session_state.pipeline
    now = time.time()
    elapsed = now - st.session_state.last_tick_wall_time
    n_ticks = max(1, min(max_ticks, int(elapsed / p.dt)))
    st.session_state.last_tick_wall_time = now

    for _ in range(n_ticks):
        rec = p.step()
        st.session_state.history.append(rec)
        _integrate_consumables(rec, p.dt)
    if tick_engine_b and MODE == "standalone":
        pb = st.session_state.engine_b_pipeline
        for _ in range(n_ticks):
            pb.step()
    return st.session_state.history[-1]


def backend_fetch_history(limit=MAX_HISTORY):
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://localhost:8000/api/history?limit={limit}", timeout=2) as r:
            data = json.loads(r.read())
        recs = []
        for row in data:
            recs.append({
                "telemetry": row["telemetry"], "twin": row["twin"], "ml": row["ml"],
                "decision": row["decision"], "fault_labels": row["fault_labels"], "link_up": True,
            })
        return recs
    except Exception as e:
        st.error(f"Could not reach backend at localhost:8000 ({e}).")
        return None


def utc_now_str() -> str:
    return dt.datetime.utcnow().strftime("%H:%M:%S UTC")


def run_demo_fastforward(pipeline, n_ticks: int, tick_engine_b_pipeline=None):
    """Steps a pipeline forward n_ticks times with no per-tick history
    bookkeeping overhead beyond what Pipeline.step() itself does -- used
    by Demo Mode to instantly skip past the ML warm-up window rather
    than making the audience wait ~230 real seconds for it to happen
    live. This is a real computation (the same Pipeline.step() used
    everywhere else), just run in a tight loop instead of one tick per
    Streamlit rerun. Fuel/oil consumable integration runs during the
    fast-forward too, so quantities are consistent regardless of how
    time was advanced."""
    rec = None
    for _ in range(n_ticks):
        rec = pipeline.step()
        st.session_state.history.append(rec)
        _integrate_consumables(rec, pipeline.dt)
    if tick_engine_b_pipeline is not None:
        for _ in range(n_ticks):
            tick_engine_b_pipeline.step()
    return rec


# =====================================================================
# Global session-state defaults for widgets that Demo Mode needs to
# control programmatically (must be set before the widget is created)
# =====================================================================
if "auto_refresh_checkbox" not in st.session_state:
    st.session_state.auto_refresh_checkbox = (__import__("os").environ.get("VAYU_TEST") != "1")
if "demo_mode_active" not in st.session_state:
    st.session_state.demo_mode_active = False
if "demo_mode_start_t" not in st.session_state:
    st.session_state.demo_mode_start_t = None


# =====================================================================
# Sidebar navigation + global controls
# =====================================================================

st.sidebar.markdown(
    f'<div class="vs-wordmark">VĀYUSŪKṢMA</div>'
    f'<div class="vs-subtitle">PROPULSION DIGITAL TWIN // MALE-UAV</div><hr>',
    unsafe_allow_html=True,
)

if MODE == "standalone":
    if st.sidebar.button("▶ RUN SIH DEMONSTRATION MODE", type="primary"):
        from pipeline import Pipeline
        st.session_state.pipeline.storage.clear()
        fresh = Pipeline(mission_name="normal_cruise", db_path="vayusuksma.db", dt=0.4)
        fresh_b = Pipeline(mission_name="normal_cruise", db_path="vayusuksma_engine_b.db",
                            dt=0.4, seed=99, compute_explanations=False)
        st.session_state.history.clear()
        st.session_state.fuel_consumed_l = 0.0
        st.session_state.oil_level_pct = 100.0
        # Fast-forward past the ML warm-up window instantly (real physics,
        # just computed in a tight loop instead of one tick per rerun) so
        # the audience isn't waiting ~230s for anomaly detection to even
        # be armed before anything interesting can happen.
        run_demo_fastforward(fresh, n_ticks=580, tick_engine_b_pipeline=fresh_b)
        # Schedule the fault a few seconds into the now-trained window so
        # the audience sees the full CONTINUE -> ... -> advisory chain
        # develop live, not one that already happened during fast-forward.
        fresh.add_fault("injector_abnormality", start_t=fresh.sim._elapsed + 4.0,
                         ramp_seconds=100.0, magnitude=1.0, rich=False)
        st.session_state.pipeline = fresh
        st.session_state.engine_b_pipeline = fresh_b
        st.session_state.last_tick_wall_time = time.time()
        st.session_state.demo_mode_active = True
        st.session_state.demo_mode_start_t = fresh.sim._elapsed
        # Auto-enable the live stream so the operator doesn't have to
        # remember to turn it on separately -- skipped under the
        # automated test harness only, since a script that intentionally
        # keeps re-running forever (correct real behaviour for a live
        # dashboard) can't be driven to a single "settled" state by a
        # test runner that waits for quiescence.
        if __import__("os").environ.get("VAYU_TEST") != "1":
            st.session_state.auto_refresh_checkbox = True
        st.session_state.nav_page = "MISSION"
        st.rerun()
    if st.session_state.demo_mode_active:
        st.sidebar.success("Demo running — injector abnormality onset in a few seconds. "
                            "Watch the MISSION / AI DIAGNOSTICS / RUL pages.")
        if st.sidebar.button("Stop demo mode"):
            st.session_state.demo_mode_active = False
    st.sidebar.markdown("---")

NAV_PAGES = [
    "MISSION", "INSTRUMENTS", "DIGITAL TWIN 3D", "LIVE ENGINE", "DIGITAL TWIN", "AI DIAGNOSTICS",
    "RUL & DEGRADATION", "MISSION SIMULATOR", "REPLAY", "FAULT LAB", "SYSTEM / SECURITY",
]
st.session_state.nav_page = st.sidebar.radio("NAVIGATION", NAV_PAGES,
                                              index=NAV_PAGES.index(st.session_state.nav_page),
                                              label_visibility="collapsed")

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Live stream (auto-refresh)", key="auto_refresh_checkbox")
st.sidebar.caption(f"Mode: `{MODE}`")

if MODE == "standalone":
    st.sidebar.markdown("---")
    st.session_state.planned_mission_s = st.sidebar.number_input(
        "Planned mission duration (min)",
        min_value=5, max_value=600, value=int(st.session_state.planned_mission_s / 60),
    ) * 60.0

    if st.sidebar.button("🔄 Reset Simulation"):
        from pipeline import Pipeline
        st.session_state.pipeline.storage.clear()
        st.session_state.pipeline = Pipeline(mission_name="normal_cruise", db_path="vayusuksma.db", dt=0.4)
        st.session_state.engine_b_pipeline = Pipeline(
            mission_name="normal_cruise", db_path="vayusuksma_engine_b.db", dt=0.4, seed=99,
            compute_explanations=False,
        )
        st.session_state.history.clear()
        st.session_state.last_tick_wall_time = time.time()
        st.session_state.fuel_consumed_l = 0.0
        st.session_state.oil_level_pct = 100.0
        st.session_state.demo_mode_active = False
        st.session_state.demo_mode_start_t = None


# =====================================================================
# Pull latest tick (paused on REPLAY page, which drives its own state)
# =====================================================================

if st.session_state.nav_page == "REPLAY":
    latest = st.session_state.history[-1] if st.session_state.history else None
    if MODE == "standalone" and latest is None:
        latest = standalone_tick_catchup()
    hist = list(st.session_state.history)
elif MODE == "standalone":
    latest = standalone_tick_catchup()
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
if MODE == "standalone":
    link_up = st.session_state.pipeline.degraded_mode.link_up


# =====================================================================
# Top command bar (persistent across all pages)
# =====================================================================

live_dot_class = "vs-live-dot" if link_up else "vs-live-dot lost"
link_text = "CONNECTED" if link_up else "LINK LOST — DEGRADED MODE"

st.markdown(f"""
<div class="vs-topbar">
  <div>
    <div class="vs-wordmark">VĀYUSŪKṢMA</div>
    <div class="vs-subtitle">PROPULSION DIGITAL TWIN // MALE-UAV</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:0.72rem;color:{CLR_MUTED};letter-spacing:0.1em;">LIVE TELEMETRY</div>
    <div style="font-size:0.85rem;"><span class="{live_dot_class}"></span>{link_text}</div>
  </div>
  <div class="vs-status-strip">
    <div>UAV-ID<br><b>MQ-VS-07</b></div>
    <div>ENGINE-ID<br><b>ENG-A-914T</b></div>
    <div>MISSION<br><b>{latest['telemetry']['mission_profile'].upper()}</b></div>
    <div>UTC<br><b>{utc_now_str()}</b></div>
    <div>TLM FREQ<br><b>{'5 Hz' if MODE=='standalone' else '5 Hz (WS)'}</b></div>
  </div>
  <div style="text-align:right; font-size:0.68rem; color:{CLR_MUTED};">
    SYSTEM STATUS<br>
    <span style="color:{CLR_ACCENT};">DIGITAL TWIN: ONLINE</span><br>
    <span style="color:{CLR_ACCENT};">AI ENGINE: {"ONLINE" if latest["ml"]["trained"] else "TRAINING"}</span><br>
    <span style="color:{CLR_ACCENT if link_up else CLR_RED};">DATA LINK: {"SECURE" if link_up else "LOST"}</span><br>
    <span style="color:{CLR_MUTED};">EDGE MODE: {"ACTIVE" if not link_up else "STANDBY"}</span>
  </div>
</div>
""", unsafe_allow_html=True)

if MODE == "standalone" and st.session_state.demo_mode_active:
    since_start = t_now - (st.session_state.demo_mode_start_t or t_now)
    if since_start < 4.0:
        phase_txt = f"Onset in {4.0 - since_start:.1f}s — injector abnormality scheduled"
    elif since_start < 104.0:
        phase_txt = f"Fault ramping — {since_start - 4.0:.0f}s / 100s into onset"
    else:
        phase_txt = "Fault fully developed — watch RUL & advisory settle"
    st.markdown(
        f'<div style="background:{CLR_ORANGE}18; border-left:4px solid {CLR_ORANGE}; '
        f'padding:10px 16px; border-radius:4px; margin-bottom:10px; color:{CLR_ORANGE}; '
        f'font-size:0.85rem;"><b>SIH DEMONSTRATION MODE ACTIVE</b> — {phase_txt}. '
        f'Injector abnormality → fuel-flow deviation → EGT residual → AI anomaly → RUL drop → '
        f'mission-probability drop → advisory change. Watch this chain unfold on MISSION, '
        f'AI DIAGNOSTICS, and RUL & DEGRADATION.</div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# PAGE: INSTRUMENTS (analog cockpit-style gauges)
# =====================================================================

def build_analog_gauge(value: float, title: str, unit: str, min_val: float, max_val: float,
                        zones: list, decimals: int = 0) -> go.Figure:
    """A round analog-instrument-style gauge (cockpit dial look) using
    Plotly's Indicator. `zones` is a list of (lo, hi, color) tuples for
    the background color bands (green/amber/red), matching real
    instrument redline conventions."""
    steps = [{"range": [lo, hi], "color": color} for lo, hi, color in zones]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": f" {unit}", "font": {"size": 22, "color": CLR_TEXT},
                "valueformat": f".{decimals}f"},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": CLR_MUTED, "tickfont": {"size": 9}},
            "bar": {"color": "#d8dee8", "thickness": 0.18},
            "bgcolor": "#0c1117",
            "borderwidth": 2,
            "bordercolor": "#2c3d4c",
            "steps": steps,
        },
        title={"text": title, "font": {"size": 12, "color": CLR_MUTED}},
    ))
    fig.update_layout(height=200, margin=dict(t=36, b=10, l=20, r=20),
                       paper_bgcolor="rgba(0,0,0,0)", font_color=CLR_TEXT)
    return fig


def render_instruments_page(latest: dict, hist: list):
    tel = latest["telemetry"]

    vs_panel_start("ENGINE / EXHAUST INSTRUMENTS")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(build_analog_gauge(
            tel["rpm"], "RPM", "", 0, 5500,
            [(0, 1800, "#1a2530"), (1800, 4500, "#0f3d24"), (4500, 5000, "#3d3a0f"), (5000, 5500, "#3d0f0f")],
        ), use_container_width=True)
    with c2:
        st.plotly_chart(build_analog_gauge(
            tel["egt_c"], "EGT (EXHAUST)", "°C", 0, 950,
            [(0, 780, "#0f3d24"), (780, 880, "#3d3a0f"), (880, 950, "#3d0f0f")],
        ), use_container_width=True)
    with c3:
        st.plotly_chart(build_analog_gauge(
            tel["cht_c"], "CHT", "°C", 0, 260,
            [(0, 205, "#0f3d24"), (205, 235, "#3d3a0f"), (235, 260, "#3d0f0f")],
        ), use_container_width=True)
    vs_panel_end()

    vs_panel_start("OIL SYSTEM")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(build_analog_gauge(
            tel["oil_pressure_kpa"], "OIL PRESSURE", "kPa", 0, 700,
            [(0, 150, "#3d0f0f"), (150, 200, "#3d3a0f"), (200, 500, "#0f3d24"),
             (500, 600, "#3d3a0f"), (600, 700, "#3d0f0f")],
        ), use_container_width=True)
    with c2:
        st.plotly_chart(build_analog_gauge(
            tel["oil_temp_c"], "OIL TEMP", "°C", 0, 150,
            [(0, 60, "#3d3a0f"), (60, 110, "#0f3d24"), (110, 130, "#3d3a0f"), (130, 150, "#3d0f0f")],
        ), use_container_width=True)
    with c3:
        st.plotly_chart(build_analog_gauge(
            st.session_state.oil_level_pct, "OIL QUANTITY*", "%", 0, 100,
            [(0, 20, "#3d0f0f"), (20, 50, "#3d3a0f"), (50, 100, "#0f3d24")],
        ), use_container_width=True)
    st.caption("*Oil quantity is not an independently simulated tank level — real aero engines don't "
               "meaningfully consume oil in a single flight under normal operation. This gauge only "
               "drops when a `lubrication_degradation` fault (a genuine leak) is active, proportional "
               "to its severity — inject one from FAULT LAB to see it move.")
    vs_panel_end()

    vs_panel_start("FUEL SYSTEM")
    c1, c2, c3 = st.columns(3)
    fuel_remaining = max(0.0, st.session_state.fuel_capacity_l - st.session_state.fuel_consumed_l)
    with c1:
        st.plotly_chart(build_analog_gauge(
            fuel_remaining, "FUEL QUANTITY", "L", 0, st.session_state.fuel_capacity_l,
            [(0, st.session_state.fuel_capacity_l * 0.10, "#3d0f0f"),
             (st.session_state.fuel_capacity_l * 0.10, st.session_state.fuel_capacity_l * 0.25, "#3d3a0f"),
             (st.session_state.fuel_capacity_l * 0.25, st.session_state.fuel_capacity_l, "#0f3d24")],
            decimals=1,
        ), use_container_width=True)
    with c2:
        fuel_pressure_kpa = 30.0 + 0.5 * tel["fuel_flow_lph"]
        st.plotly_chart(build_analog_gauge(
            fuel_pressure_kpa, "FUEL PRESSURE†", "kPa", 0, 80,
            [(0, 20, "#3d0f0f"), (20, 25, "#3d3a0f"), (25, 55, "#0f3d24"), (55, 60, "#3d3a0f"), (60, 80, "#3d0f0f")],
            decimals=1,
        ), use_container_width=True)
    with c3:
        fuel_temp_c = tel["oat_c"] + min(15.0, 0.05 * max(0.0, tel["cht_c"] - tel["oat_c"]))
        st.plotly_chart(build_analog_gauge(
            fuel_temp_c, "FUEL TEMP†", "°C", -20, 60,
            [(-20, -10, "#3d3a0f"), (-10, 40, "#0f3d24"), (40, 50, "#3d3a0f"), (50, 60, "#3d0f0f")],
            decimals=1,
        ), use_container_width=True)
    st.caption("†Fuel pressure and fuel temperature are not modelled in the underlying physics engine "
               "at all (only fuel *flow rate* is simulated) — these two gauges show simple, clearly-"
               "illustrative relationships (pressure scales mildly with flow demand; temperature tracks "
               "ambient with a small engine-heat-soak term) rather than a real instrumented signal. "
               "Fuel quantity (left) IS a genuine integral of real fuel-flow telemetry against an "
               f"assumed {st.session_state.fuel_capacity_l:.0f} L tank capacity.")
    vs_panel_end()

    vs_panel_start("FLIGHT")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_analog_gauge(
            tel["altitude_m"], "ALTITUDE", "m", 0, 8000,
            [(0, 8000, "#0f3d24")],
        ), use_container_width=True)
    with c2:
        st.plotly_chart(build_analog_gauge(
            tel["manifold_pressure_kpa"], "MANIFOLD PRESSURE", "kPa", 0, 120,
            [(0, 30, "#3d3a0f"), (30, 100, "#0f3d24"), (100, 120, "#3d3a0f")],
        ), use_container_width=True)
    st.caption("Altitude is set per mission profile (static for the duration of a single mission in "
               "this simulation, not a climbing/descending flight) — it will only change if you switch "
               "mission profiles.")
    vs_panel_end()


# =====================================================================
# PAGE: MISSION (landing page)
# =====================================================================

def render_mission_page(latest: dict, hist: list):
    ml = latest["ml"]
    decision = latest["decision"]
    remaining_s = max(st.session_state.planned_mission_s - t_now, 0.0)

    completion_pct, status_label, status_color_hex = compute_mission_completion_probability(
        latest, remaining_mission_s=remaining_s
    )
    subsystems = compute_subsystem_health(latest)
    health_index = compute_health_index(subsystems)

    col_gauge, col_stats = st.columns([1, 1.4])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=completion_pct,
            number={"suffix": "%", "font": {"size": 46, "color": CLR_ACCENT_CYAN}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": CLR_MUTED},
                "bar": {"color": status_color_hex, "thickness": 0.28},
                "bgcolor": CLR_PANEL,
                "borderwidth": 1,
                "bordercolor": CLR_PANEL_BORDER,
                "steps": [
                    {"range": [0, 50], "color": "#2a1414"},
                    {"range": [50, 75], "color": "#2a2314"},
                    {"range": [75, 100], "color": "#142a1c"},
                ],
            },
            title={"text": "MISSION COMPLETION PROBABILITY", "font": {"size": 13, "color": CLR_MUTED}},
        ))
        fig.update_layout(height=290, margin=dict(t=50, b=10, l=20, r=20),
                           paper_bgcolor="rgba(0,0,0,0)", font_color=CLR_TEXT)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<div style="text-align:center; font-size:1.1rem; font-weight:700; color:{status_color_hex};">'
            f'MISSION STATUS: {status_label}</div>', unsafe_allow_html=True,
        )

    with col_stats:
        vs_panel_start("MISSION PARAMETERS")
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Elapsed", f"{t_now/60:.1f} min")
        r1c2.metric("Remaining (planned)", f"{remaining_s/60:.1f} min")
        r1c3.metric("Engine Load", f"{latest['telemetry']['throttle']*100:.0f}%")
        r2c1, r2c2, r2c3 = st.columns(3)
        rul = ml.get("rul")
        if rul:
            r2c1.metric("Predicted RUL", f"{rul['rul_seconds']/60:.1f} min",
                        delta=f"CI [{rul['rul_ci90'][0]/60:.0f}, {rul['rul_ci90'][1]/60:.0f}] min")
        else:
            r2c1.metric("Predicted RUL", "nominal", delta="no degradation trend")
        r2c2.metric("Health Index", f"{health_index:.1f}%")
        r2c3.metric("Anomaly Score", f"{ml['anomaly_score']:+.2f}",
                    delta="ANOMALY" if ml["is_anomaly"] else "clear")
        vs_panel_end()

        # Advisory banner
        adv_color = status_color_hex
        st.markdown(
            f'<div class="vs-advisory-banner" style="background:{adv_color}18; '
            f'border-left-color:{adv_color}; color:{adv_color};">'
            f'ADVISORY: {decision["advisory"]}<br>'
            f'<span style="font-size:0.78rem; font-weight:400; color:{CLR_TEXT};">{decision["reason"]}</span>'
            f'</div>', unsafe_allow_html=True,
        )

    # Mission phase stepper (Section 19 — stylized, non-geographic)
    vs_panel_start("MISSION PROFILE — PHASE (stylized, non-geographic)")
    phase_idx = compute_mission_phase(t_now, st.session_state.planned_mission_s)
    profile_svg = build_mission_profile_svg(t_now, st.session_state.planned_mission_s, status_color_hex)
    components.html(profile_svg, height=150, scrolling=False)
    cols = st.columns(len(MISSION_PHASES))
    for i, (col, phase) in enumerate(zip(cols, MISSION_PHASES)):
        is_current = (i == phase_idx)
        color = status_color_hex if is_current else CLR_MUTED
        weight = "700" if is_current else "400"
        marker = " ● CURRENT" if is_current else ""
        col.markdown(
            f'<div style="text-align:center; color:{color}; font-weight:{weight}; font-size:0.75rem;">'
            f'{phase}{marker}</div>', unsafe_allow_html=True,
        )
    st.caption("Illustrative altitude-shaped profile (ground → climb → cruise plateau → descend) by "
               "elapsed-time fraction — this simulation has no real flight-dynamics/altitude physics, "
               "so treat this as a stylized mission-phase indicator, not a flown trajectory.")
    vs_panel_end()

    # Quick subsystem strip
    vs_panel_start("SUBSYSTEM HEALTH SNAPSHOT")
    cols = st.columns(7)
    for col, (name, data) in zip(cols, subsystems.items()):
        c = status_color(data["status"])
        col.markdown(
            f'<div style="text-align:center;">'
            f'<div style="font-size:0.68rem; color:{CLR_MUTED}; text-transform:uppercase;">{name}</div>'
            f'<div style="font-size:1.2rem; font-weight:700; color:{c};">{data["pct"]:.0f}%</div>'
            f'<div style="font-size:0.65rem; color:{c};">{data["status"]}</div>'
            f'</div>', unsafe_allow_html=True,
        )
    vs_panel_end()

    fault_labels = latest.get("fault_labels", {})
    if fault_labels:
        chips = "".join(f'<span class="vs-fault-chip">{k}: {v:.2f}</span>' for k, v in fault_labels.items())
        st.markdown(f"**Ground-truth active faults (demo only):** {chips}", unsafe_allow_html=True)


# =====================================================================
# PAGE: LIVE ENGINE
# =====================================================================

TELEMETRY_PARAMS = [
    ("rpm", "RPM", ""), ("cht_c", "CHT", "°C"), ("egt_c", "EGT", "°C"),
    ("oil_pressure_kpa", "Oil Pressure", "kPa"), ("oil_temp_c", "Oil Temp", "°C"),
    ("fuel_flow_lph", "Fuel Flow", "L/h"), ("vibration_g", "Vibration", "g"),
    ("alt_voltage_v", "Battery/Alt Voltage", "V"), ("injection_timing_deg", "Injection Timing", "deg"),
    ("manifold_pressure_kpa", "Manifold Pressure", "kPa"), ("egt_cyl_variance_c", "EGT Cyl. Variance", "°C"),
    ("altitude_m", "Altitude", "m"), ("oat_c", "Ambient Temp", "°C"), ("throttle", "Throttle", "frac"),
]


def render_live_engine_page(latest: dict, hist: list):
    subsystems = compute_subsystem_health(latest)
    highlighted = list({fault_to_subsystem(k) for k in (latest.get("fault_labels") or {}).keys()})

    col_schem, col_subsys = st.columns([1.3, 1])
    with col_schem:
        vs_panel_start("ENGINE SCHEMATIC — SENSOR MAP" + (" (fault-highlighted)" if highlighted else ""))
        # NOTE: raw <svg> markup must go through components.html, not
        # st.markdown(unsafe_allow_html=True) -- Streamlit's markdown
        # sanitizer strips graphical SVG tags (rect/circle/path) and
        # silently keeps only the plain text content of <text> elements,
        # which made this panel render as a handful of floating labels
        # with no actual schematic. components.html renders the SVG in
        # a real (S)HTML document inside an iframe, so it displays as
        # intended.
        svg = build_engine_schematic_svg(highlighted_subsystems=highlighted)
        components.html(
            f'<div style="background:{CLR_BG};">{svg}</div>',
            height=360, scrolling=False,
        )
        vs_panel_end()

    with col_subsys:
        vs_panel_start("SUBSYSTEM HEALTH — INTERACTIVE")
        chosen = st.selectbox("Drill into subsystem", list(subsystems.keys()), label_visibility="collapsed")
        for name, data in subsystems.items():
            c = status_color(data["status"])
            st.markdown(
                f'<div class="vs-subsys-row"><span>{name.upper()}</span>'
                f'<span><div class="vs-subsys-bar-bg"><div class="vs-subsys-bar-fill" '
                f'style="width:{data["pct"]}%; background:{c};"></div></div></span>'
                f'<span style="color:{c}; width:70px; text-align:right;">{data["pct"]:.0f}% {data["status"]}</span>'
                f'</div>', unsafe_allow_html=True,
            )
        vs_panel_end()

        vs_panel_start(f"DIAGNOSTIC DRILL-DOWN — {chosen.upper()}")
        d = subsystems[chosen]
        st.write(f"**Health:** {d['pct']:.1f}% — status **{d['status']}**")
        related_faults = [k for k in (latest.get("fault_labels") or {}) if fault_to_subsystem(k) == chosen]
        if related_faults:
            st.write(f"Active ground-truth fault(s) affecting this subsystem: `{', '.join(related_faults)}`")
        else:
            st.write("No ground-truth fault of this class currently active.")
        vs_panel_end()

    # Telemetry matrix
    vs_panel_start("LIVE TELEMETRY MATRIX")
    tel = latest["telemetry"]
    twin_expected = latest["twin"]["expected"]
    twin_resid = latest["twin"]["residuals"]
    rows = []
    for key, label, unit in TELEMETRY_PARAMS:
        val = tel.get(key)
        expected = twin_expected.get(key)
        residual = twin_resid.get(key)
        trend = "—"
        if len(hist) >= 5:
            prev = hist[-5]["telemetry"].get(key)
            if prev is not None and val is not None:
                trend = "▲" if val > prev else ("▼" if val < prev else "—")
        status = "NORMAL"
        if residual is not None:
            noise = {"cht_c": 3.0, "egt_c": 6.0, "oil_temp_c": 2.0, "oil_pressure_kpa": 8.0,
                     "fuel_flow_lph": 0.6, "vibration_g": 0.02, "egt_cyl_variance_c": 3.0}.get(key)
            if noise:
                nr = abs(residual) / noise
                status = "NORMAL" if nr < 2 else ("WATCH" if nr < 5 else ("WARNING" if nr < 10 else "CRITICAL"))
        rows.append({
            "Parameter": label, "Value": f"{val:.2f}" if isinstance(val, float) else val,
            "Unit": unit, "Trend": trend,
            "Expected": f"{expected:.2f}" if expected is not None else "—",
            "Residual": f"{residual:+.2f}" if residual is not None else "—",
            "Status": status,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=420)
    vs_panel_end()


# =====================================================================
# PAGE: DIGITAL TWIN
# =====================================================================

def build_data_flow_html(steps: list, active_idx: int, is_anomaly: bool) -> str:
    """Self-contained animated data-flow visualization (steps connected by
    a line with a continuously moving packet), rendered via
    components.html since it needs its own <style>/@keyframes -- a
    components.html iframe doesn't inherit the parent page's injected
    CSS, unlike st.markdown for plain text."""
    n = len(steps)
    boxes_html = ""
    for i, step in enumerate(steps):
        active = (i == active_idx)
        border = "#3ddc84" if active else "#2c3d4c"
        color = "#4fd1c5" if active else "#c9d6df"
        bg = "#16241c" if active else "#0f151b"
        boxes_html += (
            f'<div style="flex:1; text-align:center; background:{bg}; border:1px solid {border}; '
            f'border-radius:4px; padding:10px 4px; font-size:0.68rem; letter-spacing:0.04em; '
            f'color:{color}; font-family:JetBrains Mono, monospace; z-index:2; position:relative;">'
            f'{step}</div>'
        )
    packet_color = "#e6473b" if is_anomaly else "#3ddc84"
    duration = "2.2s" if is_anomaly else "3.5s"
    return f"""
    <div style="background:#080b0f; padding:6px 0;">
      <div style="position:relative; display:flex; gap:8px; align-items:center;">
        {boxes_html}
        <div style="position:absolute; top:50%; left:0; right:0; height:2px; background:#22303c; z-index:1;"></div>
        <div style="position:absolute; top:50%; width:10px; height:10px; margin-top:-5px; border-radius:50%;
             background:{packet_color}; box-shadow:0 0 8px {packet_color}; z-index:3;
             animation: flowmove {duration} linear infinite;"></div>
      </div>
    </div>
    <style>
      @keyframes flowmove {{
        0% {{ left: 0%; }}
        100% {{ left: calc(100% - 10px); }}
      }}
    </style>
    """


def render_digital_twin_page(latest: dict, hist: list):
    vs_panel_start("DIGITAL TWIN DATA FLOW")
    flow_steps = ["PHYSICAL ENGINE", "LIVE TELEMETRY", "DIGITAL TWIN", "EXPECTED STATE",
                  "RESIDUAL ENGINE", "AI DIAGNOSTICS", "MISSION RELIABILITY"]
    active_idx = 4 if latest["ml"]["is_anomaly"] else 2
    components.html(
        build_data_flow_html(flow_steps, active_idx, latest["ml"]["is_anomaly"]),
        height=90, scrolling=False,
    )
    st.caption("Packet color/speed reflects current anomaly state — red/faster while an anomaly is "
               "flagged, green/slower when nominal. The highlighted stage box marks where an anomaly "
               "is currently being surfaced in the pipeline.")
    vs_panel_end()

    view_mode = st.radio("View mode", ["ACTUAL vs EXPECTED", "RESIDUAL"], horizontal=True)

    channels = ["rpm", "cht_c", "egt_c", "oil_pressure_kpa", "oil_temp_c", "fuel_flow_lph", "vibration_g"]
    labels = {"rpm": "RPM", "cht_c": "CHT (°C)", "egt_c": "EGT (°C)", "oil_pressure_kpa": "Oil Pressure (kPa)",
              "oil_temp_c": "Oil Temp (°C)", "fuel_flow_lph": "Fuel Flow (L/h)", "vibration_g": "Vibration (g)"}

    df = pd.DataFrame([{
        "t": h["telemetry"]["t"],
        **{f"actual_{c}": h["telemetry"].get(c) for c in channels},
        **{f"expected_{c}": h["twin"]["expected"].get(c) for c in channels},
        **{f"residual_{c}": h["twin"]["residuals"].get(c) for c in channels},
    } for h in hist])

    cols = st.columns(2)
    for i, ch in enumerate(channels):
        with cols[i % 2]:
            fig = go.Figure()
            if view_mode == "ACTUAL vs EXPECTED":
                fig.add_trace(go.Scatter(x=df.t, y=df[f"actual_{ch}"], name="Actual", line=dict(color=CLR_AMBER)))
                fig.add_trace(go.Scatter(x=df.t, y=df[f"expected_{ch}"], name="Twin Expected",
                                          line=dict(color=CLR_ACCENT_CYAN, dash="dot")))
            else:
                fig.add_trace(go.Scatter(x=df.t, y=df[f"residual_{ch}"], name="Residual",
                                          line=dict(color=CLR_ORANGE), fill="tozeroy"))
                fig.add_hline(y=0, line_color=CLR_MUTED, line_dash="dot")
            fig.update_layout(title=labels[ch], template="plotly_dark", height=240,
                               margin=dict(t=36, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", showlegend=True,
                               legend=dict(font=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# PAGE: AI DIAGNOSTICS
# =====================================================================

def render_ai_diagnostics_page(latest: dict, hist: list):
    ml = latest["ml"]
    col1, col2 = st.columns(2)

    with col1:
        vs_panel_start("AI DIAGNOSTIC ENGINE")
        st.metric("Anomaly Score", f"{ml['anomaly_score']:.3f}")
        state = "ANOMALY DETECTED" if ml["is_anomaly"] else "NOMINAL"
        color = CLR_RED if ml["is_anomaly"] else CLR_ACCENT
        st.markdown(f'<div style="color:{color}; font-weight:700; font-size:1.1rem;">{state}</div>',
                    unsafe_allow_html=True)
        st.caption("Detection Method: Isolation Forest (trained on healthy-operation buffer)")
        vs_panel_end()

        vs_panel_start("AI FEATURE CONTRIBUTION")
        top_features = ml.get("top_features") or []
        if top_features:
            fdf = pd.DataFrame(top_features, columns=["feature", "attribution"])
            fig = go.Figure(go.Bar(
                x=fdf.attribution, y=fdf.feature, orientation="h",
                marker_color=[CLR_RED if v > 0 else CLR_ACCENT for v in fdf.attribution],
            ))
            fig.update_layout(template="plotly_dark", height=220, margin=dict(t=10, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No active anomaly to explain right now.")
        vs_panel_end()

    with col2:
        vs_panel_start("FAULT INTELLIGENCE")
        fault_labels = latest.get("fault_labels") or {}
        if fault_labels:
            probable = max(fault_labels.items(), key=lambda kv: kv[1])
            confidence = min(99, int(probable[1] * 100))
            st.markdown(f'<div style="font-size:1.3rem; font-weight:700; color:{CLR_ORANGE};">'
                        f'{probable[0].upper().replace("_"," ")}</div>', unsafe_allow_html=True)
            st.metric("Confidence (severity-based, demo ground truth)", f"{confidence}%")
        else:
            st.markdown(f'<div style="color:{CLR_ACCENT}; font-weight:700;">NO ACTIVE FAULT CLASSIFIED</div>',
                        unsafe_allow_html=True)

        st.markdown("**PHYSICAL REASONING**")
        norm_r = latest["twin"]["norm_residuals"]
        reasons = []
        for ch, label in [("egt_c", "EGT"), ("cht_c", "CHT"), ("oil_pressure_kpa", "Oil pressure"),
                           ("fuel_flow_lph", "Fuel flow"), ("vibration_g", "Vibration")]:
            r = norm_r.get(ch, 0)
            if abs(r) > 2.0:
                direction = "above" if r > 0 else "below"
                reasons.append(f"{label} {direction} expected ({r:+.1f}σ from twin baseline)")
        if reasons:
            for r in reasons:
                st.write(f"• {r}")
        else:
            st.write("• All monitored channels within normal deviation of the digital twin baseline.")

        st.markdown("**AI REASONING**")
        if top_features:
            for feat, val in top_features:
                st.write(f"• `{feat}` contribution: {val:+.3f}")
        else:
            st.write("• No SHAP attribution active (no anomaly).")

        st.markdown("**FAULT PROGRESSION**")
        if len(hist) >= 10 and fault_labels:
            recent_sev = [sum((h.get("fault_labels") or {}).values()) for h in hist[-10:]]
            slope = np.polyfit(range(len(recent_sev)), recent_sev, 1)[0]
            if slope > 0.01:
                prog, pcolor = "ACCELERATING" if slope > 0.03 else "DEVELOPING", CLR_ORANGE
            elif slope < -0.01:
                prog, pcolor = "RESOLVING", CLR_ACCENT
            else:
                prog, pcolor = "STABLE", CLR_AMBER
            st.markdown(f'<span style="color:{pcolor}; font-weight:700;">{prog}</span>', unsafe_allow_html=True)
        else:
            st.write("No active fault trend to assess.")
        vs_panel_end()


# =====================================================================
# PAGE: RUL & DEGRADATION
# =====================================================================

def render_rul_page(latest: dict, hist: list):
    ml = latest["ml"]
    rul = ml.get("rul")

    vs_panel_start("REMAINING USEFUL LIFE")
    if not rul:
        st.info("No degradation trend detected above the noise floor yet — engine health nominal, "
                "or the ML layer is still in its training/warm-up window (needs ~220s of simulated "
                "healthy operation before it will trust a trend).")
        return

    df = pd.DataFrame([{
        "t": h["telemetry"]["t"],
        "cht_residual": abs(h["twin"]["residuals"]["cht_c"]),
    } for h in hist])

    a, b, c = rul["params"]["a"], rul["params"]["b"], rul["params"]["c"]
    t0 = df.t.iloc[0]
    t_fit = np.linspace(df.t.iloc[0], df.t.iloc[-1] + rul["rul_seconds"] * 1.3, 200)
    y_fit = a * np.exp(np.clip(b * (t_fit - t0), -50, 50)) + c

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.t, y=df.cht_residual, name="Historical (CHT residual)",
                              mode="markers", marker=dict(color=CLR_MUTED, size=4)))
    fig.add_trace(go.Scatter(x=t_fit, y=y_fit, name="Exponential fit / predicted degradation",
                              line=dict(color=CLR_ORANGE)))
    fig.add_hline(y=rul["failure_threshold"], line_color=CLR_RED, line_dash="dash",
                  annotation_text="Failure threshold")
    ci_lo, ci_hi = rul["rul_ci90"]
    fig.add_vrect(x0=df.t.iloc[-1] + ci_lo, x1=df.t.iloc[-1] + ci_hi,
                  fillcolor=CLR_RED, opacity=0.12, line_width=0,
                  annotation_text="90% CI on failure time")
    fig.update_layout(title="Health Index (Degradation Trajectory) vs Operating Time",
                       xaxis_title="Operating Time (s)", yaxis_title="Degradation Index (CHT residual, °C)",
                       template="plotly_dark", height=420, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    vs_panel_end()

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Health (degradation index)", f"{rul['current_index']:.1f}")
    c2.metric("Predicted Failure Point", f"{(t_now + rul['rul_seconds'])/60:.1f} min (mission time)")
    c3.metric("Estimated RUL", f"{rul['rul_seconds']/60:.1f} min",
              delta=f"90% CI [{ci_lo/60:.1f}, {ci_hi/60:.1f}] min")
    st.caption("RUL estimated from Holt-smoothed degradation trajectory + exponential curve fit "
               "(scipy.optimize.curve_fit), with a 200-resample bootstrap 90% confidence interval.")


# =====================================================================
# PAGE: MISSION SIMULATOR (counterfactual what-if)
# =====================================================================

def render_mission_simulator_page(latest: dict, hist: list):
    vs_panel_start("COUNTERFACTUAL MISSION TWIN — WHAT-IF SCENARIOS")
    st.caption("Runs a real, fresh instance of the full pipeline forward in fast-time with the "
               "requested degradation injected — this is genuine simulated physics/ML output, "
               "not a lookup table. Each run takes several seconds.")

    c1, c2, c3, c4 = st.columns(4)
    mission_choice = c1.selectbox("Mission", ["normal_cruise", "high_altitude", "hot_weather",
                                               "endurance", "rapid_throttle_transitions"])
    horizon_min = c2.slider("Simulated horizon (min)", 3, 20, 8)
    injector_pct = c3.slider("Injector degradation", 0, 100, 0, format="%d%%")
    cooling_pct = c4.slider("Cooling degradation", 0, 100, 0, format="%d%%")
    lubrication_pct = st.slider("Lubrication degradation", 0, 100, 0, format="%d%%")

    if st.button("▶ RUN WHAT-IF SCENARIO", type="primary"):
        with st.spinner("Running counterfactual pipeline forward..."):
            result = run_whatif_scenario(
                mission_choice, horizon_s=horizon_min * 60,
                injector_pct=injector_pct, cooling_pct=cooling_pct, lubrication_pct=lubrication_pct,
            )
        st.session_state["_whatif_result"] = result
    vs_panel_end()

    result = st.session_state.get("_whatif_result")
    if result:
        vs_panel_start("PREDICTED EFFECT")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Health Index", f"{result['health_index']:.1f}%")
        c2.metric("RUL", f"{result['rul_seconds']/60:.1f} min" if result["rul_seconds"] else "nominal")
        c3.metric("Mission Completion", f"{result['completion_pct']:.1f}%")
        c4.metric("Resulting Advisory", result["advisory"])
        st.markdown("**Subsystem health under this scenario:**")
        cols = st.columns(7)
        for col, (name, data) in zip(cols, result["subsystems"].items()):
            c = status_color(data["status"])
            col.markdown(f'<div style="text-align:center;"><div style="font-size:0.65rem;color:{CLR_MUTED};">'
                         f'{name.upper()}</div><div style="color:{c}; font-weight:700;">{data["pct"]:.0f}%</div>'
                         f'</div>', unsafe_allow_html=True)
        vs_panel_end()

    st.markdown("---")
    vs_panel_start("MISSION SCENARIO COMPARISON (healthy engine, environment-only stress)")
    st.caption("Compares all 5 mission profiles with zero injected faults for a fixed horizon — "
               "shows how altitude/heat alone change engine stress. Takes ~30-60s.")
    if st.button("▶ RUN MISSION COMPARISON"):
        with st.spinner("Simulating all 5 mission profiles..."):
            comparison = run_mission_comparison(horizon_s=480.0)
        st.session_state["_mission_comparison"] = comparison
    comparison = st.session_state.get("_mission_comparison")
    if comparison:
        cdf = pd.DataFrame(comparison)
        cdf["rul_min"] = cdf.rul_seconds.apply(lambda x: f"{x/60:.1f}" if x else "nominal")
        cdf["completion_pct"] = cdf.completion_pct.apply(lambda x: f"{x:.1f}%")
        st.dataframe(cdf[["mission", "rul_min", "completion_pct", "risk"]].rename(columns={
            "mission": "Mission Profile", "rul_min": "RUL (min)",
            "completion_pct": "Completion Probability", "risk": "Risk",
        }), use_container_width=True, hide_index=True)
    vs_panel_end()


# =====================================================================
# PAGE: REPLAY
# =====================================================================

def render_replay_page(latest: dict, hist: list):
    vs_panel_start("MISSION REPLAY — \"WHAT THE SYSTEM KNEW AT THAT TIME\"")
    st.caption("Pauses the live stream and replays stored historical telemetry through a timeline "
               "slider, using the exact same rendering code path as the live view.")

    mission_options = ["normal_cruise", "high_altitude", "hot_weather", "endurance",
                        "rapid_throttle_transitions"]
    chosen_mission = st.selectbox("Mission to replay (from stored SQLite log)", mission_options)

    if st.button("Load stored mission"):
        if MODE == "standalone":
            rows = st.session_state.pipeline.replay_from_db(chosen_mission)
        else:
            import urllib.request
            try:
                with urllib.request.urlopen(
                    f"http://localhost:8000/api/replay/{chosen_mission}", timeout=5
                ) as r:
                    rows = json.loads(r.read())
            except Exception as e:
                st.error(f"Could not fetch replay data: {e}")
                rows = []
        st.session_state.replay_rows = rows
        st.session_state.replay_idx = 0
        st.session_state.replay_playing = False
    vs_panel_end()

    rows = st.session_state.replay_rows
    if not rows:
        st.info("No stored samples yet for this mission — switch to it on the Mission page and let "
                 "it run for a while first (samples are written continuously), then come back and load.")
        return

    # --- Play / Pause / speed transport controls ---
    tc1, tc2, tc3, tc4, tc5 = st.columns([1, 1, 1, 1, 3])
    if tc1.button("▶ Play" if not st.session_state.replay_playing else "⏸ Playing…",
                   disabled=st.session_state.replay_playing):
        st.session_state.replay_playing = True
    if tc2.button("⏸ Pause"):
        st.session_state.replay_playing = False
    if tc3.button("⏮ Restart"):
        st.session_state.replay_idx = 0
        st.session_state.replay_playing = False
    speed_label = tc4.selectbox("Speed", ["1×", "2×", "5×"],
                                 index=["1×", "2×", "5×"].index(f"{st.session_state.replay_speed}×"),
                                 label_visibility="collapsed")
    st.session_state.replay_speed = int(speed_label.replace("×", ""))
    tc5.caption(f"Sample {st.session_state.replay_idx + 1} / {len(rows)}"
                + (" — PLAYING" if st.session_state.replay_playing else " — paused"))

    idx = st.slider("Replay position (drag to scrub manually)", 0, len(rows) - 1,
                     st.session_state.replay_idx)
    if idx != st.session_state.replay_idx:
        st.session_state.replay_idx = idx           # manual scrub overrides playback position
        st.session_state.replay_playing = False
    row = rows[st.session_state.replay_idx]

    vs_panel_start(f"REPLAYED STATE @ t={row['telemetry']['t']:.1f}s "
                   f"({st.session_state.replay_idx + 1}/{len(rows)} stored samples)")
    tel = row["telemetry"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RPM", f"{tel['rpm']:.0f}")
    c2.metric("CHT", f"{tel['cht_c']:.1f}°C")
    c3.metric("EGT", f"{tel['egt_c']:.1f}°C")
    c4.metric("Anomaly", f"{row['ml']['anomaly_score']:.2f}",
              delta="ANOMALY" if row["ml"]["is_anomaly"] else "clear")
    c5.metric("Advisory (at that time)", row["decision"]["advisory"])
    faults = row.get("fault_labels") or {}
    if faults:
        st.markdown("**Faults ground-truth-active at this point:** " +
                    ", ".join(f"`{k}`" for k in faults))
    vs_panel_end()

    # small trajectory chart with a marker at current replay position
    df = pd.DataFrame([{"t": r["telemetry"]["t"], "cht": r["telemetry"]["cht_c"],
                         "anomaly": r["ml"]["anomaly_score"]} for r in rows])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.t, y=df.cht, name="CHT", line=dict(color=CLR_AMBER)))
    fig.add_vline(x=row["telemetry"]["t"], line_color=CLR_ACCENT_CYAN)
    fig.update_layout(template="plotly_dark", height=260, title="CHT trajectory — line marks replay position",
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # --- Playback auto-advance (independent of the main "Live stream"
    # auto-refresh, which intentionally excludes this page) ---
    if st.session_state.replay_playing:
        next_idx = st.session_state.replay_idx + st.session_state.replay_speed
        if next_idx >= len(rows):
            st.session_state.replay_idx = len(rows) - 1
            st.session_state.replay_playing = False
        else:
            st.session_state.replay_idx = next_idx
        if __import__("os").environ.get("VAYU_TEST") != "1":
            time.sleep(0.15)
            st.rerun()


# =====================================================================
# PAGE: FAULT LAB
# =====================================================================

FAULT_TYPES = ["misfire", "injector_abnormality", "cooling_degradation", "lubrication_degradation",
               "sensor_drift", "combustion_instability", "overheating_trend", "abnormal_vibration"]


def render_fault_lab_page(latest: dict, hist: list):
    vs_panel_start("DIGITAL FAULT INJECTION LAB")
    if MODE != "standalone":
        st.warning("Fault injection lab requires standalone mode, or use the backend's "
                   "POST /api/fault endpoint directly (see technical_doc.md).")
        vs_panel_end()
        return

    c1, c2, c3 = st.columns(3)
    fault_type = c1.selectbox("Fault", FAULT_TYPES)
    severity_pct = c2.slider("Severity", 0, 100, 100, format="%d%%")
    ramp_s = c3.slider("Ramp duration (s)", 10, 300, 120)
    extra = {}
    if fault_type == "sensor_drift":
        extra["channel"] = st.selectbox("Drifting channel",
                                         ["oil_pressure_kpa", "cht_c", "egt_c", "vibration_g"])
    if fault_type == "injector_abnormality":
        extra["rich"] = st.checkbox("Rich (over-fuel) — else lean", value=False)

    if st.button("🔴 INJECT FAULT", type="primary"):
        p = st.session_state.pipeline
        current_t = p.sim._elapsed
        p.add_fault(fault_type, start_t=current_t + 2.0, ramp_seconds=ramp_s,
                    magnitude=severity_pct / 100.0, **extra)
        st.success(f"Injected `{fault_type}` at severity {severity_pct}%, onset in 2s, "
                   f"ramping over {ramp_s}s.")
    vs_panel_end()

    # Live fault -> response -> detection -> classification -> impact flow
    vs_panel_start("FAULT PROPAGATION — LIVE")
    fault_labels = latest.get("fault_labels") or {}
    ml = latest["ml"]
    decision = latest["decision"]

    stages = [
        ("FAULT INJECTED", bool(fault_labels)),
        ("PHYSICAL RESPONSE", any(abs(v) > 2.0 for v in latest["twin"]["norm_residuals"].values())),
        ("RESIDUAL DEVIATION", any(abs(v) > 3.0 for v in latest["twin"]["norm_residuals"].values())),
        ("AI DETECTION", ml["is_anomaly"]),
        ("FAULT CLASSIFICATION", bool(fault_labels) and ml["is_anomaly"]),
        ("RUL IMPACT", ml.get("rul") is not None),
        ("MISSION RISK UPDATED", decision["advisory"] != "CONTINUE"),
    ]
    for i, (label, active) in enumerate(stages):
        cls = "vs-flow-step active" if active else "vs-flow-step"
        st.markdown(f'<div class="{cls}">{"✓" if active else "○"} {label}</div>', unsafe_allow_html=True)
        if i < len(stages) - 1:
            st.markdown('<div class="vs-flow-arrow">↓</div>', unsafe_allow_html=True)
    vs_panel_end()


# =====================================================================
# PAGE: SYSTEM / SECURITY
# =====================================================================

def render_system_page(latest: dict, hist: list):
    col1, col2 = st.columns(2)

    with col1:
        vs_panel_start("MODEL TRUST")
        ml = latest["ml"]
        subsystems = compute_subsystem_health(latest)
        physics_conf = 100 - min(30, abs(sum(latest["twin"]["norm_residuals"].values())) * 2)
        ai_conf = 100 - min(40, latest["ml"]["anomaly_score"] * 60) if ml["trained"] else 50
        sensor_conf = subsystems["sensors"]["pct"]
        rul_conf = 78.0 if ml.get("rul") else 95.0
        mission_conf = 89.0
        for name, val in [("Physics model", physics_conf), ("AI model", ai_conf),
                           ("Sensor integrity", sensor_conf), ("RUL confidence", rul_conf),
                           ("Mission model", mission_conf)]:
            c = status_color(pct_to_status(val))
            st.markdown(f'<div class="vs-subsys-row"><span>{name}</span>'
                        f'<span style="color:{c}; font-weight:700;">{val:.0f}%</span></div>',
                        unsafe_allow_html=True)
        st.caption("Heuristic trust indicators derived from residual magnitude / anomaly score / "
                   "drift flags — not a calibrated statistical confidence interval.")
        vs_panel_end()

        vs_panel_start("SECURITY (conceptual — no live cryptographic checks in this prototype)")
        for label in ["Telemetry authentication", "Data integrity", "Sensor consistency",
                      "Event log integrity", "Link status"]:
            ok = link_up or label != "Link status"
            c = CLR_ACCENT if ok else CLR_RED
            st.markdown(f'<div class="vs-subsys-row"><span>{label}</span>'
                        f'<span style="color:{c};">●</span></div>', unsafe_allow_html=True)
        st.caption("See technical_doc.md §8 for the actual security roadmap "
                   "(hash-chained logs, authenticated telemetry, physics-based spoof detection).")
        vs_panel_end()

    with col2:
        vs_panel_start("DUAL ENGINE HEALTH (conceptual — twin-propulsion illustration)")
        if MODE == "standalone":
            rec_b = st.session_state.engine_b_pipeline.latest
            if rec_b:
                subs_a = compute_subsystem_health(latest)
                subs_b = compute_subsystem_health(rec_b)
                hi_a = compute_health_index(subs_a)
                hi_b = compute_health_index(subs_b)
                c1, c2 = st.columns(2)
                c1.metric("Engine A", f"{hi_a:.1f}%")
                c2.metric("Engine B", f"{hi_b:.1f}%")
                st.metric("Differential Health", f"{hi_a - hi_b:+.1f} pts")
                diff_rows = []
                for ch, label in [("rpm", "RPM"), ("egt_c", "EGT"), ("cht_c", "CHT"),
                                   ("fuel_flow_lph", "Fuel Flow"), ("vibration_g", "Vibration")]:
                    va = latest["telemetry"].get(ch, 0)
                    vb = rec_b["telemetry"].get(ch, 0)
                    diff_rows.append({"Parameter": label, "Engine A": round(va, 2),
                                       "Engine B": round(vb, 2), "Difference": round(va - vb, 2)})
                st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
                st.caption("Engine B runs an independent Pipeline instance (different random seed, "
                           "no faults injected) purely to illustrate the pair-consistency concept — "
                           "there is no physical twin-engine airframe model here.")
        else:
            st.info("Dual-engine illustration is only available in standalone mode.")
        vs_panel_end()

        vs_panel_start("DEGRADED / EDGE MODE")
        if MODE == "standalone":
            dm = st.session_state.pipeline.degraded_mode
            c1, c2 = st.columns(2)
            if c1.button("📡 Simulate Link Loss"):
                dm.link_lost()
            if c2.button("📡 Restore Link"):
                drained = dm.link_restored()
                if drained:
                    st.success(f"Synced {len(drained)} buffered advisory summaries.")
                else:
                    st.info("Link restored — no buffered summaries to sync.")
            st.metric("Ground Link", "CONNECTED" if dm.link_up else "LOST — LOCAL DEGRADED MODE")
            st.metric("Edge Decision (local)", latest["decision"]["advisory"])
        vs_panel_end()


# =====================================================================
# PAGE: DIGITAL TWIN 3D (hero 3D visualization, per master spec §2)
# =====================================================================

def render_digital_twin_3d_page(latest: dict, hist: list):
    vs_panel_start("HERO 3D DIGITAL TWIN — INTERACTIVE ENGINE MODEL")
    st.caption(
        "Rotate: drag · Zoom: scroll · Pan: right-drag · Click a sensor marker or a cylinder for its "
        "live value. Piston/crankshaft/propeller motion speed is RPM-linked (perceptually scaled — not "
        "literal rotational speed, which would be too fast to watch). Throttle body butterfly valve "
        "angle and the LIVE COMBUSTION CYCLE readout are both genuinely telemetry-driven. Camera "
        "auto-frames toward the affected subsystem when a fault is active. Stylized geometry, not a "
        "CAD import — see VAYUSUKSMA_README.md for full scoping notes."
    )
    html = build_engine_3d_html(
        telemetry=latest["telemetry"],
        twin_residuals=latest["twin"]["residuals"],
        fault_labels=latest.get("fault_labels"),
        prev_telemetry=hist[-2]["telemetry"] if len(hist) >= 2 else None,
        anomaly_score=latest["ml"].get("anomaly_score", 0.0),
        height_px=620,
    )
    components.html(html, height=640, scrolling=False)
    vs_panel_end()

    fault_labels = latest.get("fault_labels") or {}
    if fault_labels:
        highlighted = sorted({fault_to_subsystem(k) for k in fault_labels.keys()})
        st.info(f"Fault-affected subsystem(s) highlighted red in the 3D view: "
                f"{', '.join(highlighted)}")
    else:
        st.caption("No active fault — engine shown in nominal (all-green) state. "
                   "Inject one from FAULT LAB to see component highlighting.")


# =====================================================================
# Dispatch to the selected page
# =====================================================================

PAGE_RENDERERS = {
    "MISSION": render_mission_page,
    "INSTRUMENTS": render_instruments_page,
    "DIGITAL TWIN 3D": render_digital_twin_3d_page,
    "LIVE ENGINE": render_live_engine_page,
    "DIGITAL TWIN": render_digital_twin_page,
    "AI DIAGNOSTICS": render_ai_diagnostics_page,
    "RUL & DEGRADATION": render_rul_page,
    "MISSION SIMULATOR": render_mission_simulator_page,
    "REPLAY": render_replay_page,
    "FAULT LAB": render_fault_lab_page,
    "SYSTEM / SECURITY": render_system_page,
}

PAGE_RENDERERS[st.session_state.nav_page](latest, hist)

st.markdown("---")
st.caption(
    "VĀYUSŪKṢMA — DRDO / SIH Research Prototype. All telemetry is physics-grounded simulation; "
    "no real engine hardware used. Backend architecture (simulator, digital twin, ML, decision "
    "support, FastAPI/SQLite) is unchanged from the original prototype — this is a presentation-layer "
    "upgrade only. See technical_doc.md for full architecture documentation."
)

if auto_refresh and st.session_state.nav_page not in ("REPLAY", "MISSION SIMULATOR"):
    time.sleep(0.4)
    st.rerun()
