"""
faults.py
=========
Physically-grounded fault models.

IMPORTANT DESIGN NOTE ON WHY OFFSETS, NOT DIRECT MUTATION
-----------------------------------------------------------
Several engine channels (CHT, EGT, oil temp, RPM) are modelled in
engine_sim.py as first-order lag integrators: `state.x += (equilibrium -
state.x) * (dt/tau)`. They carry memory across ticks by design (thermal
inertia, mechanical inertia).

A fault must therefore express its effect as a shift of the *equilibrium
target* that channel is lagging toward (e.g. "cooling degradation raises
the CHT equilibrium by 38 degrees"), NOT as a direct additive mutation of
the already-integrated state value on every tick. The latter would
compound: adding a fixed delta every 0.5s tick with only a weak (~1-2%
per tick) restoring pull from the lag ODE produces unbounded runaway
rather than a new, higher, physically-bounded steady state.

Channels that are recomputed fresh from a formula every tick with no
memory (fuel flow, oil pressure, vibration, cylinder EGT variance,
injection timing) are safe to bias directly/additively each tick, since
they are not integrators -- each tick's fault contribution simply adds to
that tick's freshly computed baseline rather than stacking on itself.

Each fault therefore implements `get_offsets(t) -> dict[str, float]`,
returning the CURRENT (severity-scaled) offset it contributes this
instant -- not a cumulative delta. `engine_sim.EngineSimulator.step()`
sums offsets from all active faults and applies them at the correct
point in the physics: equilibrium offsets before the lag integration,
instantaneous offsets after the fresh per-tick formulas.

Supported offset keys
----------------------
Equilibrium biases (fed into the lag ODEs):
    "cht_equilibrium", "egt_equilibrium", "oil_temp_equilibrium",
    "rpm_target"
Instantaneous biases (added to freshly-computed per-tick values):
    "fuel_flow", "oil_pressure", "vibration", "egt_variance"

Available faults
-----------------
1. Misfire                  -> cylinder torque loss => RPM target droops
                                under load, EGT cylinder-to-cylinder
                                variance rises, average EGT dips
                                (unburned fuel effect).
2. InjectorAbnormality      -> fuel flow drifts away from the RPM*throttle
                                relationship (over- or under-fueling),
                                dragging EGT the opposite direction.
3. CoolingDegradation       -> reduced effective cooling => CHT
                                equilibrium rises (coolant/airflow
                                blockage, fin fouling, fan degradation).
4. LubricationDegradation   -> oil pressure decays faster than the temp-
                                driven viscosity model alone predicts
                                (pump wear / leak / filter clogging); oil
                                temp equilibrium also rises (friction).
5. SensorDrift              -> a chosen sensor channel accumulates a slow
                                additive bias post-hoc (thermocouple
                                drift, pressure transducer zero-shift) --
                                the *true* engine is fine, only the
                                sensor reading is wrong. This one still
                                mutates the reported value directly
                                (by design -- it corrupts the *sensor*,
                                not the physical plant), but the bias is
                                itself bounded by a ramp x elapsed-time
                                cap so it does not grow forever.
6. CombustionInstability    -> cyclic variability increases: RPM and EGT
                                targets develop a higher-frequency
                                oscillation (partial misfire / detonation
                                borderline), distinct from sustained
                                misfire.
7. OverheatingTrend         -> compound cooling+load fault: cooling
                                efficiency collapses non-linearly past a
                                threshold, raising the CHT/EGT
                                equilibria non-linearly with severity
                                (bounded at full severity, not
                                open-ended).
8. AbnormalVibration        -> mechanical fault (bearing wear, prop
                                imbalance, mount fatigue) -> vibration
                                RMS rises with a periodic component,
                                applied as an instantaneous bias.

Each fault has a severity ramp (0 -> 1 over `ramp_seconds`) so faults
look like real degradation processes rather than step-function alarms,
matching how prognostics benchmarks (e.g. C-MAPSS) structure gradual
degradation trajectories. Severity is capped at 1.0 x magnitude, so every
offset above is inherently bounded.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


class _BaseFault:
    name: str = "base_fault"

    def __init__(self, start_t: float, ramp_seconds: float = 120.0, magnitude: float = 1.0,
                 rng_seed: Optional[int] = None):
        self.start_t = start_t
        self.ramp_seconds = max(ramp_seconds, 1e-3)
        self.magnitude = magnitude
        self.rng = np.random.default_rng(rng_seed)
        self._last_labels: Dict[str, float] = {}

    def severity(self, t: float) -> float:
        if t < self.start_t:
            return 0.0
        frac = (t - self.start_t) / self.ramp_seconds
        return float(np.clip(frac, 0.0, 1.0)) * self.magnitude

    def compute_offsets(self, t: float) -> Dict[str, float]:
        raise NotImplementedError

    def get_offsets(self, t: float) -> Dict[str, float]:
        sev = self.severity(t)
        if sev <= 0.0:
            self._last_labels = {}
            return {}
        offsets = self.compute_offsets(t)
        self._last_labels = {self.name: round(sev, 4)}
        return offsets

    @property
    def last_labels(self) -> Dict[str, float]:
        return self._last_labels


class MisfireFault(_BaseFault):
    """Cylinder misfire: torque loss lowers the RPM the governor can hold
    at a given throttle (target droop), unburned fuel escapes -> cylinder
    -to-cylinder EGT variance spikes, average EGT dips slightly due to
    incomplete combustion."""
    name = "misfire"

    def compute_offsets(self, t: float) -> Dict[str, float]:
        sev = self.severity(t)
        return {
            "rpm_target": -sev * 220.0,
            "egt_variance": sev * 140.0 + abs(self.rng.normal(0, 3.0)) * sev,
            "egt_equilibrium": -sev * 35.0,
            "vibration": sev * 0.06,
        }


class InjectorAbnormalityFault(_BaseFault):
    """Injector fouling/leak: fuel delivery drifts off the nominal
    RPM*throttle relationship. `rich=True` -> over-fueling (fuel flow up,
    EGT down as excess fuel cools exhaust); rich=False -> lean/under-
    fueling (fuel flow down, EGT up, risk of detonation)."""
    name = "injector_abnormality"

    def __init__(self, *args, rich: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.rich = rich

    def compute_offsets(self, t: float) -> Dict[str, float]:
        sev = self.severity(t)
        direction = 1.0 if self.rich else -1.0
        return {
            "fuel_flow": direction * sev * 3.0,
            "egt_equilibrium": -direction * sev * 55.0,
            "egt_variance": sev * 20.0,
        }


class CoolingDegradationFault(_BaseFault):
    """Cooling airflow/coolant path degradation (fin fouling, coolant
    leak, fan bearing wear): effective cooling efficiency drops, so CHT's
    equilibrium rises. Oil is partly cooled by engine case, so oil temp
    equilibrium rises less."""
    name = "cooling_degradation"

    def compute_offsets(self, t: float) -> Dict[str, float]:
        sev = self.severity(t)
        return {
            "cht_equilibrium": sev * 38.0,
            "oil_temp_equilibrium": sev * 12.0,
        }


class LubricationDegradationFault(_BaseFault):
    """Oil pump wear / filter clogging / slow leak: oil pressure decays
    beyond what the viscosity-temperature model alone would predict
    (instantaneous bias, since oil pressure is recomputed fresh each
    tick), and increased internal friction nudges oil temp equilibrium
    upward."""
    name = "lubrication_degradation"

    def compute_offsets(self, t: float) -> Dict[str, float]:
        sev = self.severity(t)
        return {
            "oil_pressure": -sev * 90.0,
            "oil_temp_equilibrium": sev * 8.0,
            "vibration": sev * 0.015,
        }


class SensorDriftFault(_BaseFault):
    """A single sensor channel accumulates additive bias while the true
    engine physics are unaffected. Models thermocouple aging / pressure
    transducer zero-drift. `channel` must be an EngineState attribute
    name. The bias is bounded: it ramps with severity AND is capped by
    `max_drift`, so it never runs away even though it grows with elapsed
    fault duration (representative of real sensor drift, which saturates)."""
    name = "sensor_drift"

    def __init__(self, *args, channel: str = "oil_pressure_kpa", drift_per_sec: float = 0.15,
                 max_drift: float = 60.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel = channel
        self.drift_per_sec = drift_per_sec
        self.max_drift = max_drift

    def current_bias(self, t: float) -> float:
        sev = self.severity(t)
        if sev <= 0:
            return 0.0
        elapsed = t - self.start_t
        raw = sev * self.drift_per_sec * elapsed
        return float(np.clip(raw, -self.max_drift, self.max_drift))

    def compute_offsets(self, t: float) -> Dict[str, float]:
        # Sensor drift is applied directly to the reported channel value
        # post-physics (it corrupts the *reading*, not the plant), handled
        # specially by EngineSimulator via `sensor_drift_faults`.
        return {}

    def get_offsets(self, t: float) -> Dict[str, float]:
        sev = self.severity(t)
        if sev <= 0:
            self._last_labels = {}
        else:
            self._last_labels = {f"{self.name}:{self.channel}": round(sev, 4)}
        return {}


class CombustionInstabilityFault(_BaseFault):
    """Cyclic combustion variability (borderline detonation / partial
    misfire without full cylinder dropout): adds a higher-frequency
    oscillatory component to the RPM and EGT *targets* (still bounded by
    the lag ODE, so it manifests as genuine oscillation rather than
    runaway), plus elevated, noisy cylinder EGT variance."""
    name = "combustion_instability"

    def compute_offsets(self, t: float) -> Dict[str, float]:
        sev = self.severity(t)
        osc = math.sin(t * 2.3) * sev
        return {
            "rpm_target": osc * 90.0,
            "egt_equilibrium": osc * 30.0,
            "egt_variance": sev * 45.0 + abs(self.rng.normal(0, 5.0)) * sev,
        }


class OverheatingTrendFault(_BaseFault):
    """Compound thermal-runaway fault: past a severity threshold, cooling
    efficiency collapses non-linearly (vapor lock / coolant boil-off
    risk), so the CHT/EGT *equilibria* accelerate upward non-linearly
    with severity -- bounded at full severity, unlike an unbounded
    additive-per-tick model. This is the "silent killer" trend fault RUL
    estimation is meant to catch early via the widening residual, not via
    literal runaway telemetry."""
    name = "overheating_trend"

    def compute_offsets(self, t: float) -> Dict[str, float]:
        sev = self.severity(t)
        runaway = sev ** 2  # nonlinear acceleration as severity approaches 1, but still bounded
        return {
            "cht_equilibrium": sev * 25.0 + runaway * 45.0,
            "egt_equilibrium": sev * 15.0 + runaway * 30.0,
            "oil_temp_equilibrium": sev * 10.0,
        }


class AbnormalVibrationFault(_BaseFault):
    """Mechanical fault: bearing wear / prop imbalance / mount fatigue.
    Adds both a sustained RMS rise and a periodic component (imbalance
    signature), applied as an instantaneous bias since vibration is
    recomputed fresh each tick."""
    name = "abnormal_vibration"

    def compute_offsets(self, t: float) -> Dict[str, float]:
        sev = self.severity(t)
        periodic = 0.04 * math.sin(t * 4.0) * sev
        return {
            "vibration": sev * 0.12 + periodic + abs(self.rng.normal(0, 0.01)) * sev,
        }


FAULT_REGISTRY = {
    "misfire": MisfireFault,
    "injector_abnormality": InjectorAbnormalityFault,
    "cooling_degradation": CoolingDegradationFault,
    "lubrication_degradation": LubricationDegradationFault,
    "sensor_drift": SensorDriftFault,
    "combustion_instability": CombustionInstabilityFault,
    "overheating_trend": OverheatingTrendFault,
    "abnormal_vibration": AbnormalVibrationFault,
}


def build_fault(fault_type: str, **kwargs):
    """Factory: build_fault('misfire', start_t=60, ramp_seconds=180)"""
    if fault_type not in FAULT_REGISTRY:
        raise ValueError(f"Unknown fault type '{fault_type}'. Options: {list(FAULT_REGISTRY)}")
    return FAULT_REGISTRY[fault_type](**kwargs)
