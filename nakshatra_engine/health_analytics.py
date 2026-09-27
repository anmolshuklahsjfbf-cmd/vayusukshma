"""
health_analytics.py
====================
Pure-Python analytics layer for the VĀYUSŪKṢMA dashboard. Everything here
is independently testable without Streamlit -- it only consumes the
Pipeline's existing record shape (telemetry / twin / ml / decision /
fault_labels), never mutates the backend, and never invents new
physics -- it re-derives presentation-layer numbers (subsystem health %,
mission completion probability, fault-to-subsystem highlighting) from
values the existing pipeline already produces.

Where the master spec asks for something the backend has no direct
signal for (e.g. "AI confidence", "sensor confidence"), this module
computes an honest, clearly-labelled heuristic from data that does
exist (anomaly score, drift flags, residual magnitudes) rather than
fabricating a number with no basis. Every such heuristic is documented
inline so it's clear to a reviewer what's measured vs. estimated.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Status thresholds & colors -- single source of truth for the whole UI
# ---------------------------------------------------------------------

STATUS_COLORS = {
    "NORMAL": "#3ddc84",
    "WATCH": "#e8b93f",
    "WARNING": "#f0862b",
    "CRITICAL": "#e6473b",
    "DEGRADED": "#e6473b",
    "INFO": "#4fc3f7",
}

ADVISORY_TO_MISSION_STATUS = {
    "CONTINUE": ("GO", "#3ddc84"),
    "MONITOR": ("CAUTION", "#e8b93f"),
    "DIVERT_ADVISORY": ("CAUTION", "#f0862b"),
    "ABORT_RECOMMENDED": ("ABORT / DIVERT", "#e6473b"),
    "RTB_IMMEDIATE": ("ABORT / DIVERT", "#e6473b"),
}


def pct_to_status(pct: float) -> str:
    if pct >= 90:
        return "NORMAL"
    if pct >= 75:
        return "WATCH"
    if pct >= 50:
        return "WARNING"
    return "CRITICAL"


# ---------------------------------------------------------------------
# Subsystem health scoring
# ---------------------------------------------------------------------
# Each subsystem score is 100 minus a penalty derived from the twin's
# *normalized* residual for the channel(s) that subsystem owns, plus a
# direct penalty if a fault of that class is actively injected (ground
# truth is available in this demo; a fielded system would rely on the
# residual/anomaly signal alone). This keeps the mapping transparent:
# every score traces back to a specific existing signal, not a new model.

_FAULT_SUBSYSTEM_MAP = {
    "misfire": "combustion",
    "injector_abnormality": "fuel",
    "cooling_degradation": "thermal",
    "lubrication_degradation": "lubrication",
    "sensor_drift": "sensors",
    "combustion_instability": "combustion",
    "overheating_trend": "thermal",
    "abnormal_vibration": "mechanical",
}


def fault_to_subsystem(fault_label_key: str) -> str:
    """fault_label_key may be 'misfire' or 'sensor_drift:oil_pressure_kpa'
    (sensor drift labels embed the channel) -- strip to the base name."""
    base = fault_label_key.split(":")[0]
    return _FAULT_SUBSYSTEM_MAP.get(base, "sensors")


def _penalty_from_norm_residual(norm_residual: float, scale: float = 10.0, deadband: float = 2.0) -> float:
    """Maps a normalized residual (roughly in units of noise-floor std
    devs) to a 0-100 penalty. Residuals within `deadband` std-devs incur
    no penalty at all -- this absorbs the twin's own small, expected
    model-form mismatch (a simplified steady-state model will never
    perfectly match the higher-fidelity simulator even when the engine
    is genuinely healthy) so a fully healthy engine reads ~100%, not a
    permanent false "WATCH". Beyond the deadband, penalty saturates
    toward 100 via an exponential curve, never overshooting."""
    effective = max(0.0, abs(norm_residual) - deadband)
    return 100.0 * (1.0 - math.exp(-effective / scale))


def compute_subsystem_health(record: dict) -> Dict[str, dict]:
    """Returns {subsystem: {"pct": float, "status": str}} for the 7
    subsystems named in the spec: combustion, thermal, lubrication, fuel,
    mechanical, electrical, sensors."""
    twin = record.get("twin", {})
    norm_r = twin.get("norm_residuals", {})
    ml = record.get("ml", {})
    tel = record.get("telemetry", {})
    fault_labels = record.get("fault_labels", {}) or {}
    drift_flags = ml.get("drift_flags", {}) or {}

    active_subsystems = {fault_to_subsystem(k) for k in fault_labels.keys()}

    def score(channels: List[str], subsystem: str, base_penalty_scale: float = 12.0) -> dict:
        # Worst-channel-determines-subsystem-health: a subsystem is only
        # as healthy as its most-deviant monitored parameter. Summing
        # penalties across channels would double-count a single fault
        # that happens to perturb two related channels at once (e.g.
        # cooling degradation raises both CHT and oil temp residuals).
        penalty = max(
            (_penalty_from_norm_residual(norm_r.get(ch, 0.0), base_penalty_scale) for ch in channels),
            default=0.0,
        )
        if subsystem in active_subsystems:
            penalty += 12.0  # ground-truth fault of this class is live
        pct = float(np.clip(100.0 - penalty, 0.0, 100.0))
        return {"pct": round(pct, 1), "status": pct_to_status(pct)}

    combustion = score(["egt_c", "egt_cyl_variance_c"], "combustion")
    thermal = score(["cht_c"], "thermal")
    lubrication = score(["oil_pressure_kpa", "oil_temp_c"], "lubrication")
    fuel = score(["fuel_flow_lph"], "fuel")
    mechanical = score(["vibration_g"], "mechanical")

    # Electrical: no twin-modelled expectation for bus voltage, so we
    # score it against a realistic nominal operating band (typical small
    # aero-engine alternator/battery bus spec) rather than a single point
    # value, since bus voltage legitimately varies with RPM/load even
    # when everything is healthy.
    volt = tel.get("alt_voltage_v", 13.8)
    band_lo, band_hi = 13.2, 14.4
    volt_dev = max(0.0, band_lo - volt, volt - band_hi)
    electrical_pct = float(np.clip(100.0 - volt_dev * 40.0, 0.0, 100.0))
    electrical = {"pct": round(electrical_pct, 1), "status": pct_to_status(electrical_pct)}

    # Sensors: penalized by active CUSUM drift flags and any sensor_drift
    # fault label, since this subsystem represents "do I trust the
    # instrumentation", not the physical plant.
    n_drift = sum(1 for v in drift_flags.values() if v.get("drift_flag"))
    sensor_penalty = n_drift * 20.0
    if "sensors" in active_subsystems:
        sensor_penalty += 15.0
    sensors_pct = float(np.clip(100.0 - sensor_penalty, 0.0, 100.0))
    sensors = {"pct": round(sensors_pct, 1), "status": pct_to_status(sensors_pct)}

    return {
        "combustion": combustion,
        "thermal": thermal,
        "lubrication": lubrication,
        "fuel": fuel,
        "mechanical": mechanical,
        "electrical": electrical,
        "sensors": sensors,
    }


def compute_health_index(subsystems: Dict[str, dict]) -> float:
    """Simple weighted average -- thermal/lubrication/combustion weighted
    slightly higher since they're the classic leading indicators of aero
    piston engine failure."""
    weights = {
        "combustion": 1.2, "thermal": 1.3, "lubrication": 1.3,
        "fuel": 1.0, "mechanical": 1.0, "electrical": 0.7, "sensors": 0.5,
    }
    total_w = sum(weights.values())
    total = sum(subsystems[k]["pct"] * w for k, w in weights.items())
    return round(total / total_w, 1)


# ---------------------------------------------------------------------
# Mission completion probability
# ---------------------------------------------------------------------
# Heuristic combining: (a) overall health index, (b) RUL margin ratio
# vs. remaining-mission-time + safety margin (the same quantity the real
# DecisionEngine uses), and (c) anomaly score. This is presentation-layer
# only -- it does NOT feed back into DecisionEngine's actual advisory,
# which remains the authoritative safety logic; this number exists to
# give the operator a single at-a-glance figure consistent with that
# logic, not to replace it.

def compute_mission_completion_probability(
    record: dict, remaining_mission_s: float, safety_margin_s: float = 600.0
) -> Tuple[float, str, str]:
    ml = record.get("ml", {})
    decision = record.get("decision", {})
    health_index = compute_health_index(compute_subsystem_health(record))

    rul = ml.get("rul")
    if rul is not None and rul.get("rul_seconds") is not None:
        required = remaining_mission_s + safety_margin_s
        rul_ratio = float(np.clip(rul["rul_seconds"] / max(required, 1.0), 0.0, 2.0))
        rul_component = float(np.clip(rul_ratio * 60.0, 0.0, 60.0))  # up to 60 pts
    else:
        rul_component = 60.0  # no degradation trend detected -> full credit

    anomaly_score = ml.get("anomaly_score", 0.0)
    anomaly_penalty = float(np.clip(anomaly_score, 0.0, 1.0)) * 15.0

    health_component = health_index * 0.40  # up to 40 pts

    pct = float(np.clip(rul_component + health_component - anomaly_penalty, 0.0, 100.0))

    advisory = decision.get("advisory", "CONTINUE")
    status_label, status_color = ADVISORY_TO_MISSION_STATUS.get(advisory, ("CAUTION", "#e8b93f"))
    return round(pct, 1), status_label, status_color


# ---------------------------------------------------------------------
# Mission phase (stylized, non-geographic) -- Section 19
# ---------------------------------------------------------------------

MISSION_PHASES = ["BASE", "CLIMB", "TRANSIT", "ISR LOITER", "RETURN", "RTB"]


def compute_mission_phase(elapsed_s: float, total_planned_s: float) -> int:
    """Splits the planned mission duration into 6 stylized phases by
    simple time fraction. This is illustrative flight-profile structure,
    not derived from real flight-plan waypoints (none exist in this
    simulation) -- documented as such in the UI."""
    if total_planned_s <= 0:
        return 0
    frac = float(np.clip(elapsed_s / total_planned_s, 0.0, 0.999))
    # weight loiter as the longest phase, climb/return shorter
    bounds = [0.0, 0.12, 0.30, 0.30, 0.70, 0.85, 1.0]
    # bounds has 7 edges for 6 phases: BASE(0-0.12) CLIMB(0.12-0.30)
    # TRANSIT(0.30-0.30 -- collapsed, see below) ... simplify:
    edges = [0.0, 0.12, 0.28, 0.68, 0.80, 1.0]
    for i in range(len(edges) - 1):
        if edges[i] <= frac < edges[i + 1]:
            return i
    return len(MISSION_PHASES) - 1


_MISSION_PHASE_EDGES = [0.0, 0.12, 0.28, 0.68, 0.80, 1.0]
# Altitude shape (0-1 normalized) at each edge: ground -> climb -> cruise
# plateau (spans transit+loiter) -> descend -> ground. Purely illustrative
# of a generic long-endurance profile shape, not simulated flight dynamics.
_MISSION_ALTITUDE_KEYFRAMES = [0.05, 0.05, 0.85, 0.85, 0.85, 0.05, 0.05]
_MISSION_ALTITUDE_FRACS =     [0.0,  0.12, 0.28, 0.68, 0.80, 1.0, 1.0]


def build_mission_profile_svg(elapsed_s: float, total_planned_s: float, status_color_hex: str = "#3ddc84") -> str:
    """Small SVG: an illustrative altitude-shaped mission profile line
    (ground -> climb -> cruise plateau -> descend -> ground) with a
    marker positioned at the current elapsed-time fraction. Same
    honesty caveat as compute_mission_phase: this is a stylized
    time-fraction shape, not derived from real flight dynamics (this
    simulation has no altitude/attitude physics)."""
    frac = float(np.clip(elapsed_s / max(total_planned_s, 1.0), 0.0, 1.0))

    W, H = 760, 160
    pad_x, pad_y = 30, 20
    plot_w, plot_h = W - 2 * pad_x, H - 2 * pad_y

    xs = [pad_x + f * plot_w for f in _MISSION_ALTITUDE_FRACS]
    ys = [pad_y + (1 - a) * plot_h for a in _MISSION_ALTITUDE_KEYFRAMES]
    path_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    # Interpolate marker position along the same piecewise-linear shape.
    marker_x = pad_x + frac * plot_w
    # find segment for y-interpolation
    marker_y = ys[-1]
    for i in range(len(_MISSION_ALTITUDE_FRACS) - 1):
        f0, f1 = _MISSION_ALTITUDE_FRACS[i], _MISSION_ALTITUDE_FRACS[i + 1]
        if f0 <= frac <= f1 and f1 > f0:
            t = (frac - f0) / (f1 - f0)
            marker_y = ys[i] + t * (ys[i + 1] - ys[i])
            break

    phase_labels = ["BASE", "CLIMB", "TRANSIT", "LOITER", "RETURN", "RTB"]
    label_xs = [pad_x + f * plot_w for f in [0.0, 0.12, 0.28, 0.68, 0.80, 1.0]]

    labels_svg = "".join(
        f'<text x="{lx:.1f}" y="{H-4}" font-size="9" fill="#5b7186" '
        f'text-anchor="middle" font-family="Consolas,monospace">{lbl}</text>'
        for lx, lbl in zip(label_xs, phase_labels)
    )

    return f"""
<svg viewBox="0 0 {W} {H}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="{W}" height="{H}" fill="#080b0f"/>
  <polyline points="{path_pts}" fill="none" stroke="#3d6f8f" stroke-width="2"/>
  <circle cx="{marker_x:.1f}" cy="{marker_y:.1f}" r="6" fill="{status_color_hex}">
    <animate attributeName="r" values="5;7;5" dur="1.6s" repeatCount="indefinite"/>
  </circle>
  <line x1="{marker_x:.1f}" y1="{marker_y:.1f}" x2="{marker_x:.1f}" y2="{H-16}"
        stroke="{status_color_hex}" stroke-width="1" stroke-dasharray="3,3" opacity="0.5"/>
  {labels_svg}
</svg>
"""


# ---------------------------------------------------------------------
# Engine schematic (stylized SVG, not photorealistic) -- Section 18
# ---------------------------------------------------------------------

_SUBSYSTEM_BOX_IDS = {
    "combustion": "box-combustion",
    "thermal": "box-cooling",
    "lubrication": "box-oil",
    "fuel": "box-fuel",
    "mechanical": "box-crank",
    "electrical": "box-electrical",
    "sensors": "box-sensors",
}


def build_engine_schematic_svg(highlighted_subsystems: Optional[List[str]] = None) -> str:
    """Returns a self-contained stylized SVG cutaway of a 4-cylinder aero
    piston engine with labelled sensor points. `highlighted_subsystems`
    is a list of subsystem keys (from _FAULT_SUBSYSTEM_MAP values) whose
    boxes should be drawn with a red alert outline -- used by the Fault
    Lab and Live Engine pages to visually tie an active fault to the
    physical subsystem it affects."""
    highlighted = set(highlighted_subsystems or [])

    def stroke(subsystem: str, normal="#3d6f8f") -> str:
        return "#e6473b" if subsystem in highlighted else normal

    def width(subsystem: str) -> str:
        return "3.5" if subsystem in highlighted else "1.5"

    return f"""
<svg viewBox="0 0 900 420" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" font-family="Consolas, monospace">
  <rect x="0" y="0" width="900" height="420" fill="#0b0f14"/>

  <!-- crankcase / mechanical -->
  <rect x="60" y="300" width="500" height="70" rx="6" fill="#141b24"
        stroke="{stroke('mechanical')}" stroke-width="{width('mechanical')}"/>
  <text x="80" y="325" fill="#8aa0b8" font-size="11">CRANKCASE / CRANKSHAFT</text>
  <circle cx="150" cy="350" r="14" fill="none" stroke="{stroke('mechanical')}" stroke-width="2"/>
  <circle cx="280" cy="350" r="14" fill="none" stroke="{stroke('mechanical')}" stroke-width="2"/>
  <circle cx="410" cy="350" r="14" fill="none" stroke="{stroke('mechanical')}" stroke-width="2"/>
  <text x="80" y="365" fill="#5b7a99" font-size="9">VIBRATION SENSOR ▸</text>

  <!-- 4 cylinders -->
  {"".join(_cylinder(120 + i*110, 130, i==1, stroke('combustion'), width('combustion')) for i in range(4))}

  <!-- cooling path (right side loop) -->
  <path d="M 600 140 C 700 140, 700 380, 600 380" fill="none"
        stroke="{stroke('thermal')}" stroke-width="{width('thermal')}" stroke-dasharray="6,4"/>
  <text x="690" y="130" fill="#8aa0b8" font-size="11">COOLING PATH</text>
  <text x="612" y="160" fill="#5b7a99" font-size="9">◂ CHT sensor</text>

  <!-- oil circuit (left side loop) -->
  <path d="M 60 140 C -20 140, -20 380, 60 380" fill="none"
        stroke="{stroke('lubrication')}" stroke-width="{width('lubrication')}" stroke-dasharray="6,4"/>
  <text x="0" y="130" fill="#8aa0b8" font-size="10">OIL CIRCUIT</text>
  <text x="0" y="200" fill="#5b7a99" font-size="9">Oil pressure /</text>
  <text x="0" y="212" fill="#5b7a99" font-size="9">temp sensor ▸</text>

  <!-- fuel/injector rail -->
  <rect x="120" y="60" width="440" height="18" rx="4" fill="#141b24"
        stroke="{stroke('fuel')}" stroke-width="{width('fuel')}"/>
  <text x="130" y="52" fill="#8aa0b8" font-size="11">FUEL / INJECTOR RAIL</text>

  <!-- exhaust manifold -->
  <rect x="120" y="392" width="440" height="14" rx="4" fill="#141b24"
        stroke="{stroke('combustion')}" stroke-width="{width('combustion')}"/>
  <text x="130" y="418" fill="#8aa0b8" font-size="10">EXHAUST MANIFOLD — EGT sensor</text>

  <!-- electrical -->
  <rect x="700" y="300" width="150" height="70" rx="6" fill="#141b24"
        stroke="{stroke('electrical')}" stroke-width="{width('electrical')}"/>
  <text x="712" y="322" fill="#8aa0b8" font-size="10">ALTERNATOR /</text>
  <text x="712" y="336" fill="#8aa0b8" font-size="10">BUS VOLTAGE</text>

  <!-- sensors / ECU -->
  <rect x="700" y="60" width="150" height="60" rx="6" fill="#141b24"
        stroke="{stroke('sensors')}" stroke-width="{width('sensors')}"/>
  <text x="715" y="85" fill="#8aa0b8" font-size="10">SENSOR / ECU</text>
  <text x="715" y="100" fill="#8aa0b8" font-size="10">TELEMETRY BUS</text>

</svg>
"""


def _cylinder(x: int, y: int, mid: bool, stroke_color: str, stroke_w: str) -> str:
    label = "INJ" if not mid else "SPK"
    return f"""
  <rect x="{x}" y="{y}" width="70" height="150" rx="4" fill="#141b24"
        stroke="{stroke_color}" stroke-width="{stroke_w}"/>
  <rect x="{x+20}" y="{y+40}" width="30" height="60" rx="3" fill="#1c2733"
        stroke="{stroke_color}" stroke-width="1"/>
  <text x="{x+8}" y="{y+20}" fill="#5b7a99" font-size="8">{label}</text>
"""


# ---------------------------------------------------------------------
# What-if / counterfactual mission simulator -- Sections 12 & 13
# ---------------------------------------------------------------------
# Runs a *real* fresh Pipeline forward (not a lookup table): applies the
# requested degradation as an actual fault on a clean pipeline instance
# and reads back the genuine RUL/health/anomaly output. This reuses 100%
# of the existing physics/ML/decision code -- it is not a separate model.

def run_whatif_scenario(
    mission_name: str,
    horizon_s: float,
    injector_pct: float = 0.0,
    cooling_pct: float = 0.0,
    lubrication_pct: float = 0.0,
    dt: float = 0.5,
) -> dict:
    """Runs a fresh, throwaway Pipeline instance forward in fast-time to
    project the effect of a hypothetical degradation. Explanation (SHAP)
    is disabled for these runs -- it's not needed for the summary metrics
    returned here and is the dominant cost of a live tick, so disabling
    it is what makes a 15-30 minute simulated horizon feasible within a
    dashboard button click."""
    from pipeline import Pipeline  # local import to avoid circulars at module load

    p = Pipeline(mission_name=mission_name, db_path=":memory:", dt=dt,
                 healthy_training_seconds=min(120.0, horizon_s * 0.3),
                 compute_explanations=False)

    if injector_pct > 0:
        p.add_fault("injector_abnormality", start_t=5.0, ramp_seconds=horizon_s * 0.5,
                    magnitude=injector_pct / 100.0)
    if cooling_pct > 0:
        p.add_fault("cooling_degradation", start_t=5.0, ramp_seconds=horizon_s * 0.5,
                    magnitude=cooling_pct / 100.0)
    if lubrication_pct > 0:
        p.add_fault("lubrication_degradation", start_t=5.0, ramp_seconds=horizon_s * 0.5,
                    magnitude=lubrication_pct / 100.0)

    n_steps = int(horizon_s / dt)
    last_rec = None
    for _ in range(n_steps):
        last_rec = p.step()

    subsystems = compute_subsystem_health(last_rec)
    health_index = compute_health_index(subsystems)
    completion_pct, status_label, status_color = compute_mission_completion_probability(
        last_rec, remaining_mission_s=max(horizon_s * 0.3, 300.0)
    )
    rul = last_rec["ml"].get("rul")
    return {
        "health_index": health_index,
        "subsystems": subsystems,
        "completion_pct": completion_pct,
        "status_label": status_label,
        "status_color": status_color,
        "rul_seconds": rul["rul_seconds"] if rul else None,
        "advisory": last_rec["decision"]["advisory"],
        "final_record": last_rec,
    }


def run_mission_comparison(horizon_s: float = 900.0, dt: float = 0.5) -> List[dict]:
    """Runs all 5 mission profiles healthy (no injected fault) for a
    fixed horizon and reports the resulting RUL/completion/risk -- shows
    how mission *environment alone* (altitude, heat) changes engine
    stress even with zero faults, which is genuine simulated behaviour,
    not a lookup table."""
    missions = ["normal_cruise", "high_altitude", "hot_weather", "endurance",
                "rapid_throttle_transitions"]
    results = []
    for m in missions:
        r = run_whatif_scenario(m, horizon_s=horizon_s, dt=dt)
        risk = "LOW"
        if r["completion_pct"] < 90:
            risk = "MEDIUM"
        if r["completion_pct"] < 75:
            risk = "HIGH"
        results.append({
            "mission": m,
            "rul_seconds": r["rul_seconds"],
            "completion_pct": r["completion_pct"],
            "risk": risk,
        })
    return results
