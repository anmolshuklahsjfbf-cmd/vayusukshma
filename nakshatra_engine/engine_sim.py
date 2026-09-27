"""
engine_sim.py
=============
Physically-grounded synthetic data simulator for a MALE-UAV aero piston
engine (modelled loosely on a Rotax-914 / UAV-class 4-cyl turbocharged
piston engine).

Design intent
-------------
Every output is derived from a small set of coupled first-order ODEs and
steady-state thermodynamic relations, NOT random noise. RPM and throttle
are the primary driving inputs; CHT, EGT, oil temp/pressure, fuel flow,
vibration and alternator load are all *derived* from them with realistic
time constants, so that a fault injected upstream (Section faults.py)
propagates through the same physical relationships a real engine would
show.

This module is intentionally the single "ground truth" physical model.
The Digital Twin (twin_core.py) runs an *independent, imperfect* copy of
a subset of this physics to generate residuals -- it must not import
internal state from here, only the streamed telemetry.
"""

from __future__ import annotations

import math
import time
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# Ambient / mission environment
# --------------------------------------------------------------------------

@dataclass
class Environment:
    """Ambient conditions the engine breathes and cools against."""
    altitude_m: float = 1500.0          # density altitude
    oat_c: float = 15.0                 # outside air temperature
    headwind_kts: float = 0.0

    def density_ratio(self) -> float:
        """ISA-ish air density ratio sigma = rho/rho0 (simplified barometric)."""
        # Standard atmosphere approx: sigma ~ (1 - 2.2557e-5 * h)^4.2561
        h = max(self.altitude_m, 0.0)
        sigma = max((1 - 2.2557e-5 * h) ** 4.2561, 0.15)
        return sigma

    def cooling_efficiency(self) -> float:
        """Air-cooling effectiveness: worse at high altitude (thin air) and
        high OAT (less delta-T to ambient)."""
        sigma = self.density_ratio()
        oat_penalty = max(0.0, (self.oat_c - 15.0)) * 0.01  # hotter day -> worse cooling
        eff = 0.55 * sigma + 0.45  # never fully collapses, some ram effect
        eff -= oat_penalty
        return float(np.clip(eff, 0.35, 1.05))


# --------------------------------------------------------------------------
# Mission profile: defines throttle(t) and environment(t)
# --------------------------------------------------------------------------

@dataclass
class MissionProfile:
    """A named mission profile producing throttle-fraction and environment
    as a function of elapsed mission time (seconds)."""
    name: str = "normal_cruise"

    def throttle_at(self, t: float) -> float:
        if self.name == "normal_cruise":
            return 0.62 + 0.03 * math.sin(t / 45.0)
        if self.name == "high_altitude":
            return 0.78 + 0.03 * math.sin(t / 60.0)
        if self.name == "hot_weather":
            return 0.65 + 0.03 * math.sin(t / 50.0)
        if self.name == "endurance":
            return 0.55 + 0.02 * math.sin(t / 90.0)
        if self.name == "rapid_throttle_transitions":
            # Square-wave-ish throttle steps every ~40s
            cycle = (t % 80.0)
            return 0.85 if cycle < 40.0 else 0.35
        return 0.6

    def environment_at(self, t: float) -> Environment:
        if self.name == "high_altitude":
            return Environment(altitude_m=5500.0, oat_c=-5.0)
        if self.name == "hot_weather":
            return Environment(altitude_m=800.0, oat_c=42.0)
        if self.name == "endurance":
            return Environment(altitude_m=2500.0, oat_c=20.0)
        if self.name == "rapid_throttle_transitions":
            return Environment(altitude_m=1800.0, oat_c=18.0)
        return Environment(altitude_m=1500.0, oat_c=15.0)


# --------------------------------------------------------------------------
# Engine state
# --------------------------------------------------------------------------

@dataclass
class EngineState:
    """Full instrumented state vector at one instant. This is what gets
    streamed over the virtual CAN/MQTT bus."""
    t: float = 0.0
    rpm: float = 2200.0
    throttle: float = 0.6
    cht_c: float = 120.0          # cylinder head temp
    egt_c: float = 650.0          # exhaust gas temp
    oil_temp_c: float = 90.0
    oil_pressure_kpa: float = 350.0
    fuel_flow_lph: float = 14.0
    vibration_g: float = 0.15     # RMS vibration
    alt_voltage_v: float = 13.8   # alternator/battery bus voltage
    injection_timing_deg: float = 22.0  # deg BTDC
    manifold_pressure_kpa: float = 95.0
    egt_cyl_variance_c: float = 8.0     # spread across cylinders (misfire signal)
    mission_profile: str = "normal_cruise"
    fault_labels: dict = field(default_factory=dict)  # ground-truth injected faults (for eval only)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d


# --------------------------------------------------------------------------
# The simulator itself
# --------------------------------------------------------------------------

class EngineSimulator:
    """
    Integrates a coupled first-order thermal/mechanical model forward in
    time. Call `.step(dt)` at 1-10 Hz to advance state.

    Physical relationships modelled (simplified but directionally correct):
      * RPM tracks a throttle-commanded target with a mechanical time lag.
      * Fuel flow ~ proportional to RPM * throttle (roughly matches BSFC
        behaviour of a carbureted/injected small aero engine).
      * CHT rises toward a throttle/RPM-dependent equilibrium, moderated
        by cooling_efficiency() from ambient; has thermal inertia (slow).
      * EGT tracks combustion energy release ~ f(RPM, throttle, mixture);
        faster time constant than CHT.
      * Oil temp follows CHT with lag and offset; oil pressure follows an
        RPM-dependent pump curve, degraded by oil temp (viscosity drop).
      * Manifold pressure ~ throttle position (naturally aspirated approx).
      * Vibration baseline scales mildly with RPM (mechanical imbalance
        naturally rises with speed) plus fault contributions injected
        externally by faults.py.
      * Alternator voltage bus is RPM-dependent (belt-driven) with a
        battery-buffered floor.
    """

    def __init__(self, mission: MissionProfile, seed: Optional[int] = 42):
        self.mission = mission
        self.rng = np.random.default_rng(seed)
        self.state = EngineState(mission_profile=mission.name)
        self._elapsed = 0.0
        # active fault objects; faults.py instances are appended here.
        # Each must implement .get_offsets(t) -> dict and .last_labels
        self.fault_hooks: list = []

    def _collect_offsets(self) -> dict:
        """Sum severity-scaled offsets from all active faults, and merge
        their ground-truth labels into state.fault_labels. Sensor-drift
        faults are handled separately (see step()) since they corrupt the
        reported value post-physics rather than shifting plant behaviour."""
        totals = {
            "cht_equilibrium": 0.0, "egt_equilibrium": 0.0,
            "oil_temp_equilibrium": 0.0, "rpm_target": 0.0,
            "fuel_flow": 0.0, "oil_pressure": 0.0,
            "vibration": 0.0, "egt_variance": 0.0,
        }
        labels = {}
        for fault in self.fault_hooks:
            offs = fault.get_offsets(self._elapsed)
            for k, v in offs.items():
                totals[k] = totals.get(k, 0.0) + v
            labels.update(fault.last_labels)
        self.state.fault_labels.update(labels)
        return totals

    def reset(self):
        self._elapsed = 0.0
        self.state = EngineState(mission_profile=self.mission.name)

    def _target_rpm(self, throttle: float) -> float:
        # Idle 1800, redline-ish max continuous 5200 for a UAV piston engine
        return 1800.0 + throttle * (5200.0 - 1800.0)

    def step(self, dt: float = 0.5) -> EngineState:
        s = self.state
        self._elapsed += dt
        s.t = self._elapsed
        s.fault_labels = {}

        # Collect this tick's fault offsets FIRST -- equilibrium offsets
        # get folded into the targets the lag ODEs pull toward; instant-
        # aneous offsets get added to the fresh per-tick formulas below.
        offs = self._collect_offsets()

        env = self.mission.environment_at(self._elapsed)
        throttle_cmd = self.mission.throttle_at(self._elapsed)
        # small actuator noise (sensor/actuator jitter, not a "fault")
        throttle_cmd = float(np.clip(throttle_cmd + self.rng.normal(0, 0.004), 0.0, 1.0))
        s.throttle = throttle_cmd

        # --- RPM dynamics: first-order lag toward (possibly fault-biased) target ---
        tau_rpm = 3.0  # seconds, mechanical/governor response
        target_rpm = self._target_rpm(throttle_cmd) + offs["rpm_target"]
        s.rpm += (target_rpm - s.rpm) * (dt / tau_rpm)
        s.rpm += self.rng.normal(0, 4.0)  # sensor/combustion cyclic noise

        # --- Manifold pressure: tracks throttle, density-corrected ---
        sigma = env.density_ratio()
        s.manifold_pressure_kpa = 30.0 + throttle_cmd * 70.0 * sigma + self.rng.normal(0, 0.6)

        # --- Fuel flow: ~ proportional to RPM * throttle (BSFC-ish) ---
        base_ff = 0.0021 * s.rpm * throttle_cmd + 2.0 + offs["fuel_flow"]
        s.fuel_flow_lph = max(0.0, base_ff + self.rng.normal(0, 0.15))

        # --- Combustion energy -> EGT equilibrium (fault-biased) ---
        # Leaner/higher-power operation and altitude both push EGT up.
        egt_equilibrium = (
            420.0
            + 480.0 * throttle_cmd
            + 60.0 * (1.0 - sigma)  # thinner air -> hotter EGT for same throttle
            + offs["egt_equilibrium"]
        )
        tau_egt = 4.0
        s.egt_c += (egt_equilibrium - s.egt_c) * (dt / tau_egt)
        s.egt_c += self.rng.normal(0, 2.0)

        # --- CHT equilibrium & cooling (fault-biased) ---
        cooling_eff = env.cooling_efficiency()
        cht_equilibrium = (
            70.0
            + 140.0 * throttle_cmd
            + 25.0 * (1.0 - cooling_eff)
            + offs["cht_equilibrium"]
        )
        tau_cht = 45.0  # metal has much higher thermal inertia than gas
        s.cht_c += (cht_equilibrium - s.cht_c) * (dt / tau_cht)
        s.cht_c += self.rng.normal(0, 0.4)

        # --- Oil temp follows CHT with lag + offset (fault-biased) ---
        oil_temp_equilibrium = s.cht_c * 0.55 + 25.0 + offs["oil_temp_equilibrium"]
        tau_oil = 60.0
        s.oil_temp_c += (oil_temp_equilibrium - s.oil_temp_c) * (dt / tau_oil)
        s.oil_temp_c += self.rng.normal(0, 0.3)

        # --- Oil pressure: pump curve vs RPM, degraded by high oil temp ---
        # Nominal pump curve ~ sqrt(rpm) scaled; viscosity drops with temp.
        visc_factor = float(np.clip(1.0 - (s.oil_temp_c - 90.0) * 0.004, 0.55, 1.1))
        s.oil_pressure_kpa = (
            120.0 + 0.09 * s.rpm * visc_factor + offs["oil_pressure"] + self.rng.normal(0, 3.0)
        )

        # --- Vibration baseline: rises mildly with RPM, plus fault bias ---
        vib_baseline = 0.05 + 0.00006 * (s.rpm - 1800.0) + offs["vibration"]
        s.vibration_g = max(0.0, vib_baseline + self.rng.normal(0, 0.01))

        # --- Cylinder EGT variance (misfire indicator baseline + fault bias) ---
        s.egt_cyl_variance_c = max(1.0, 6.0 + offs["egt_variance"] + self.rng.normal(0, 1.0))

        # --- Injection timing: commanded, small controller noise ---
        s.injection_timing_deg = 22.0 + throttle_cmd * 4.0 + self.rng.normal(0, 0.3)

        # --- Alternator/bus voltage: RPM-dependent, battery-buffered ---
        alt_output = 12.6 + min(1.6, (s.rpm - 1800.0) / 2200.0)
        s.alt_voltage_v = float(np.clip(alt_output + self.rng.normal(0, 0.05), 11.5, 14.6))

        # --- Sensor drift faults: corrupt the reported value post-physics,
        # applied last since they represent a lying sensor, not real plant
        # behaviour (the true engine state above is already final).
        for fault in self.fault_hooks:
            if hasattr(fault, "current_bias"):
                bias = fault.current_bias(self._elapsed)
                if bias != 0.0:
                    channel = fault.channel
                    setattr(s, channel, getattr(s, channel) + bias)

        return s
