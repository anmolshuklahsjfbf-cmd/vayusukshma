"""
pipeline.py
===========
Orchestrates one end-to-end tick of the system:

    EngineSimulator (+ faults) --> telemetry dict
          |
          v
    DigitalTwin.compute()      --> expected + residuals
          |
          v
    ml_models (IsolationForest, CUSUM, RUL, SHAP)
          |
          v
    DecisionEngine / DegradedModeManager --> Advisory
          |
          v
    Storage.write_sample()

This is the single module the FastAPI backend and Streamlit dashboard
both depend on -- it is data-source agnostic: `Pipeline.step()` doesn't
care whether telemetry came from the live simulator or a replayed
historical log (see `replay_mode`), which satisfies the "simulator must
be swappable for real telemetry / replay must reuse all code paths"
requirement.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

from engine_sim import EngineSimulator, MissionProfile, Environment
from faults import build_fault
from twin_core import DigitalTwin
from ml_models import (
    AnomalyDetector, CUSUMDriftDetector, RULEstimator, HoltSmoother,
    AnomalyExplainer, build_feature_vector,
)
from decision_support import DecisionEngine, DegradedModeManager, MissionContext, Advisory
from storage import Storage


DRIFT_CHANNELS = ["oil_pressure_kpa", "cht_c", "egt_c", "vibration_g"]

# Health index used for RUL: we track CHT residual, a common leading
# indicator of both cooling degradation and overheating trend faults.
RUL_CHANNEL = "cht_c"
RUL_FAILURE_THRESHOLD = 45.0  # normalized-residual-equivalent "failure" bound (deg C over expected)


class Pipeline:
    def __init__(
        self,
        mission_name: str = "normal_cruise",
        db_path: str = "twin_data.db",
        healthy_training_seconds: float = 220.0,
        dt: float = 0.5,
        seed: int = 42,
        compute_explanations: bool = True,
    ):
        self.mission = MissionProfile(name=mission_name)
        self.sim = EngineSimulator(self.mission, seed=seed)
        self.twin = DigitalTwin()
        self.detector = AnomalyDetector(contamination=0.03)
        self.explainer = AnomalyExplainer(self.detector)
        self.cusum = CUSUMDriftDetector(DRIFT_CHANNELS)
        self.rul_estimator = RULEstimator(failure_threshold=RUL_FAILURE_THRESHOLD)
        self.holt = HoltSmoother()
        self.decision_engine = DecisionEngine()
        self.degraded_mode = DegradedModeManager()
        self.storage = Storage(db_path=db_path)

        self.dt = dt
        self.healthy_training_seconds = healthy_training_seconds
        self._trained = False
        self.compute_explanations = compute_explanations

        self.rul_hist_t: Deque[float] = deque(maxlen=1200)
        self.rul_hist_y: Deque[float] = deque(maxlen=1200)

        self.mission_context = MissionContext(
            remaining_mission_seconds=3600.0,
            distance_to_friendly_territory_km=40.0,
            cruise_speed_kmh=110.0,
        )

        self.latest: Optional[dict] = None
        self.history: Deque[dict] = deque(maxlen=2000)

        # Hysteresis buffer for anomaly flag -- IsolationForest scores can
        # flicker sample-to-sample near its decision boundary; requiring
        # a majority of the last N samples to be anomalous before acting
        # avoids advisory flapping (MONITOR <-> DIVERT_ADVISORY) on noise.
        self._anomaly_window: Deque[bool] = deque(maxlen=6)

        # SHAP's KernelExplainer is expensive (model-agnostic sampling);
        # recomputing it on every single anomalous tick would stall a
        # live 5 Hz dashboard. We refresh the explanation periodically
        # instead of every tick while an anomaly persists -- the
        # attribution doesn't need sub-second freshness to be useful.
        self._last_explain_wall_t = 0.0
        self._explain_interval_s = 2.0
        self._cached_top_features: List = []

    # ---------------------------------------------------------------
    def add_fault(self, fault_type: str, start_t: float, ramp_seconds: float = 120.0,
                  magnitude: float = 1.0, **kwargs):
        fault = build_fault(fault_type, start_t=start_t, ramp_seconds=ramp_seconds,
                             magnitude=magnitude, **kwargs)
        self.sim.fault_hooks.append(fault)
        return fault

    def set_mission(self, mission_name: str):
        self.mission = MissionProfile(name=mission_name)
        self.sim.mission = self.mission

    # ---------------------------------------------------------------
    def step(self) -> dict:
        state = self.sim.step(self.dt)
        telemetry = state.to_dict()
        telemetry["altitude_m"] = self.mission.environment_at(state.t).altitude_m
        telemetry["oat_c"] = self.mission.environment_at(state.t).oat_c

        twin_out = self.twin.compute(telemetry)
        feature_vec = build_feature_vector(telemetry, twin_out)

        # --- bootstrap training window on healthy operation ---
        if not self._trained:
            self.detector.add_training_sample(feature_vec)
            if state.t >= self.healthy_training_seconds:
                self.detector.fit()
                self._trained = True
            anomaly_score, is_anomaly = 0.0, False
            top_features: List = []
        else:
            anomaly_score, raw_is_anomaly = self.detector.score(feature_vec)
            self._anomaly_window.append(raw_is_anomaly)
            is_anomaly = sum(self._anomaly_window) >= max(3, len(self._anomaly_window) // 2 + 1)
            if is_anomaly and self.compute_explanations:
                now = time.time()
                if now - self._last_explain_wall_t >= self._explain_interval_s:
                    self._cached_top_features = self.explainer.explain(feature_vec, top_k=3)
                    self._last_explain_wall_t = now
                top_features = self._cached_top_features
            else:
                top_features = []

        # CUSUM baselines only once the engine has settled past its
        # thermal warm-up transient (same rationale as the RUL gate
        # above) -- otherwise the "healthy baseline" mean/std it locks
        # onto is itself mid-transient and every post-warm-up sample
        # looks like drift.
        if self._trained:
            drift_flags = self.cusum.update(telemetry)
        else:
            drift_flags = {c: {"drift_flag": False, "sh": 0.0, "sl": 0.0} for c in DRIFT_CHANNELS}

        # --- RUL: track CHT residual as degradation index ---
        self.rul_hist_t.append(state.t)
        self.rul_hist_y.append(abs(twin_out["residuals"][RUL_CHANNEL]))
        rul_result = None
        # Min-signal gate: the CHT residual noise floor is ~3 deg C (see
        # twin_core noise_floor table). Below ~2x that, curve-fitting an
        # exponential is fitting noise / normal warm-up transient, not a
        # real degradation trend -- so we don't trust/act on RUL until the
        # degradation index has clearly separated from the noise floor.
        MIN_SIGNAL_FLOOR = 6.0
        if self._trained and len(self.rul_hist_y) >= 20 and self.rul_hist_y[-1] > MIN_SIGNAL_FLOOR:
            y_arr = np.array(self.rul_hist_y)
            y_smooth = self.holt.smooth(y_arr)
            candidate = self.rul_estimator.fit_and_estimate(
                np.array(self.rul_hist_t), y_smooth
            )
            # Reject wildly unstable fits (huge CI relative to point estimate)
            if candidate is not None:
                lo, hi = candidate["rul_ci90"]
                point = candidate["rul_seconds"]
                if point <= 0 or (hi - lo) < max(point * 5.0, 300.0):
                    rul_result = candidate

        active_faults = list(state.fault_labels.keys())
        decision = self.decision_engine.decide(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            rul_result=rul_result,
            drift_flags=drift_flags,
            active_faults=active_faults,
            mission=self.mission_context,
        )

        self.degraded_mode.record(decision, state.t)

        ml_out = {
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "trained": self._trained,
            "top_features": top_features,
            "drift_flags": drift_flags,
            "rul": rul_result,
        }
        decision_out = {
            "advisory": decision.advisory.value,
            "reason": decision.reason,
        }

        self.storage.write_sample(telemetry, twin_out, ml_out, decision_out, state.fault_labels)

        record = {
            "telemetry": telemetry,
            "twin": twin_out,
            "ml": ml_out,
            "decision": decision_out,
            "fault_labels": state.fault_labels,
            "link_up": self.degraded_mode.link_up,
        }
        self.latest = record
        self.history.append(record)
        return record

    # ---------------------------------------------------------------
    def replay_from_db(self, mission_profile: str) -> List[dict]:
        """Replay mode: re-stream stored historical logs through the SAME
        downstream shape (does not recompute ML/decision, just re-emits
        exactly what was stored) -- demonstrates that the dashboard code
        path is identical for live vs replay."""
        return self.storage.read_all_for_mission(mission_profile)
