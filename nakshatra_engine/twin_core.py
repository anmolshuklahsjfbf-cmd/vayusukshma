"""
twin_core.py
============
The Digital Twin Core. This is deliberately a SEPARATE, simplified,
steady-state physics model from the ground-truth simulator in
engine_sim.py -- exactly as a real digital twin would be: an engineering
model of "what the engine *should* be doing right now", evaluated purely
from the live telemetry stream (RPM, throttle/manifold pressure, ambient),
with no access to the simulator's internal fault state.

The residual (actual - expected) for each channel is the primary anomaly
signal consumed by the ML layer. This module is intentionally the *only*
place that knows how to compute "expected" values, so swapping the data
source (simulator -> real CAN telemetry) requires zero changes downstream:
anything that produces a telemetry dict with the same keys can be fed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


EXPECTED_CHANNELS = [
    "cht_c", "egt_c", "oil_temp_c", "oil_pressure_kpa",
    "fuel_flow_lph", "vibration_g", "egt_cyl_variance_c",
]


@dataclass
class TwinModel:
    """Steady-state expected-value model, mirroring the equilibrium
    relationships in engine_sim.py's EngineSimulator but WITHOUT thermal
    lag or fault effects -- i.e. "the healthy engine's textbook curve".
    Small deliberate model-form mismatch versus the simulator is expected
    and healthy (real physics-informed twins are never perfect); it shows
    up as a small baseline residual noise floor, which the ML layer must
    learn to treat as normal.
    """

    def expected(self, telemetry: dict) -> Dict[str, float]:
        rpm = telemetry["rpm"]
        throttle = telemetry["throttle"]
        altitude_m = telemetry.get("altitude_m", 1500.0)
        oat_c = telemetry.get("oat_c", 15.0)

        sigma = max((1 - 2.2557e-5 * max(altitude_m, 0.0)) ** 4.2561, 0.15)
        cooling_eff = float(np.clip(0.55 * sigma + 0.45 - max(0.0, oat_c - 15.0) * 0.01, 0.35, 1.05))

        cht_exp = 70.0 + 140.0 * throttle + 25.0 * (1.0 - cooling_eff)
        egt_exp = 420.0 + 480.0 * throttle + 60.0 * (1.0 - sigma)
        oil_temp_exp = cht_exp * 0.55 + 25.0
        visc_factor = float(np.clip(1.0 - (oil_temp_exp - 90.0) * 0.004, 0.55, 1.1))
        oil_pressure_exp = 120.0 + 0.09 * rpm * visc_factor
        fuel_flow_exp = 0.0021 * rpm * throttle + 2.0
        vibration_exp = 0.05 + 0.00006 * (rpm - 1800.0)
        egt_var_exp = 6.0

        return {
            "cht_c": cht_exp,
            "egt_c": egt_exp,
            "oil_temp_c": oil_temp_exp,
            "oil_pressure_kpa": oil_pressure_exp,
            "fuel_flow_lph": fuel_flow_exp,
            "vibration_g": vibration_exp,
            "egt_cyl_variance_c": egt_var_exp,
        }


class DigitalTwin:
    """
    Wraps TwinModel and maintains a light exponential smoothing of the
    "actual" side so residuals aren't dominated by single-sample sensor
    noise. This is the object the rest of the pipeline (ML layer,
    backend, dashboard) talks to.
    """

    def __init__(self, smoothing_alpha: float = 0.3):
        self.model = TwinModel()
        self.alpha = smoothing_alpha
        self._smoothed: Dict[str, float] = {}

    def _smooth(self, channel: str, value: float) -> float:
        prev = self._smoothed.get(channel, value)
        sm = self.alpha * value + (1 - self.alpha) * prev
        self._smoothed[channel] = sm
        return sm

    def compute(self, telemetry: dict) -> dict:
        """Returns a dict with expected values, residuals, and a
        normalized residual vector ready for the ML layer."""
        expected = self.model.expected(telemetry)
        residuals = {}
        norm_residuals = {}
        # Rough per-channel "typical noise floor" for normalization, so
        # residuals are comparable across channels of very different units.
        noise_floor = {
            "cht_c": 3.0, "egt_c": 6.0, "oil_temp_c": 2.0,
            "oil_pressure_kpa": 8.0, "fuel_flow_lph": 0.6,
            "vibration_g": 0.02, "egt_cyl_variance_c": 3.0,
        }
        for ch in EXPECTED_CHANNELS:
            actual = self._smooth(ch, telemetry[ch])
            r = actual - expected[ch]
            residuals[ch] = r
            norm_residuals[ch] = r / noise_floor[ch]

        return {
            "t": telemetry.get("t"),
            "expected": expected,
            "residuals": residuals,
            "norm_residuals": norm_residuals,
            "residual_vector": [norm_residuals[ch] for ch in EXPECTED_CHANNELS],
        }
