"""
ml_models.py
============
AI/ML layer sitting on top of the Digital Twin's residual stream.

Contents
--------
1. AnomalyDetector       - IsolationForest over [raw sensor vector +
                            physics residual vector]. Trained on a
                            healthy-operation buffer, scores every new
                            sample.
2. CUSUMDriftDetector     - classic tabular CUSUM control chart per
                            channel for sensor drift (no ML library
                            needed; this is the right tool for slow
                            additive bias detection).
3. RULEstimator           - exponential/power-law degradation curve fit
                            (scipy.optimize.curve_fit) on a chosen
                            degradation index (e.g. CHT residual trend),
                            extrapolated to a failure threshold with a
                            bootstrap confidence interval.
4. AnomalyExplainer       - SHAP KernelExplainer/TreeExplainer wrapper
                            attributing an anomaly score to the driving
                            feature(s).
5. HoltSmoother           - thin wrapper around statsmodels' Holt's
                            exponential smoothing, used to denoise the
                            degradation index before curve fitting.

These are deliberately kept as small, inspectable classes (not a black
box pipeline) so the dashboard can show *why* a decision was made.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.optimize import curve_fit

try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    _SHAP_AVAILABLE = False

try:
    from statsmodels.tsa.holtwinters import Holt
    _HOLT_AVAILABLE = True
except Exception:
    _HOLT_AVAILABLE = False


RAW_FEATURES = [
    "rpm", "cht_c", "egt_c", "oil_temp_c", "oil_pressure_kpa",
    "fuel_flow_lph", "vibration_g", "alt_voltage_v", "egt_cyl_variance_c",
]
RESIDUAL_FEATURES = [
    "cht_c", "egt_c", "oil_temp_c", "oil_pressure_kpa",
    "fuel_flow_lph", "vibration_g", "egt_cyl_variance_c",
]
FEATURE_NAMES = RAW_FEATURES + [f"resid_{c}" for c in RESIDUAL_FEATURES]


def build_feature_vector(telemetry: dict, twin_output: dict) -> np.ndarray:
    raw = [telemetry[f] for f in RAW_FEATURES]
    resid = [twin_output["norm_residuals"][c] for c in RESIDUAL_FEATURES]
    return np.array(raw + resid, dtype=float)


# --------------------------------------------------------------------------
# 1. Isolation Forest anomaly detector
# --------------------------------------------------------------------------

class AnomalyDetector:
    def __init__(self, contamination: float = 0.03, n_estimators: int = 80, random_state: int = 0):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=1,  # single-row scoring gets no benefit from joblib parallelism;
                       # it only adds dispatch overhead for a 100-tree ensemble.
        )
        self.is_fitted = False
        self.train_buffer: List[np.ndarray] = []
        self._score_offset = 0.0  # sklearn's internal threshold offset, cached after fit

    def add_training_sample(self, feature_vec: np.ndarray):
        self.train_buffer.append(feature_vec)

    def fit(self):
        if len(self.train_buffer) < 30:
            raise RuntimeError("Need at least 30 healthy samples to fit IsolationForest.")
        X = np.vstack(self.train_buffer)
        self.model.fit(X)
        self.is_fitted = True

    def score(self, feature_vec: np.ndarray) -> Tuple[float, bool]:
        """Returns (anomaly_score, is_anomaly). Score is the negative of
        sklearn's decision_function so *higher = more anomalous*, which is
        more intuitive for a dashboard. Calls decision_function only
        ONCE per sample -- IsolationForest.predict() is just `decision_function
        < 0`, so calling both (as an earlier version of this code did)
        doubled the per-tick cost for no extra information."""
        if not self.is_fitted:
            return 0.0, False
        decision = self.model.decision_function(feature_vec.reshape(1, -1))[0]
        raw_score = -decision
        return float(raw_score), bool(decision < 0)


# --------------------------------------------------------------------------
# 2. CUSUM sensor-drift detector
# --------------------------------------------------------------------------

class CUSUMDriftDetector:
    """Two-sided tabular CUSUM per channel. Flags a channel once the
    cumulative deviation from its healthy-baseline mean exceeds a
    threshold `h`, after allowing a slack `k` (in units of baseline
    std-dev) for normal noise. This is the textbook tool for slow additive
    bias / sensor drift -- much better suited than IsolationForest, which
    is tuned for point anomalies rather than slow trend accumulation.
    """

    def __init__(self, channels: List[str], k: float = 0.5, h: float = 5.0, baseline_window: int = 60):
        self.channels = channels
        self.k = k
        self.h = h
        self.baseline_window = baseline_window
        self._baseline: Dict[str, Deque[float]] = {c: deque(maxlen=baseline_window) for c in channels}
        self._mean: Dict[str, float] = {c: 0.0 for c in channels}
        self._std: Dict[str, float] = {c: 1.0 for c in channels}
        self._sh: Dict[str, float] = {c: 0.0 for c in channels}  # upper CUSUM
        self._sl: Dict[str, float] = {c: 0.0 for c in channels}  # lower CUSUM
        self._baseline_locked: Dict[str, bool] = {c: False for c in channels}

    def update(self, telemetry: dict) -> Dict[str, dict]:
        out = {}
        for c in self.channels:
            x = telemetry[c]
            if not self._baseline_locked[c]:
                self._baseline[c].append(x)
                if len(self._baseline[c]) >= self.baseline_window:
                    arr = np.array(self._baseline[c])
                    self._mean[c] = float(arr.mean())
                    self._std[c] = float(max(arr.std(), 1e-6))
                    self._baseline_locked[c] = True
                out[c] = {"drift_flag": False, "sh": 0.0, "sl": 0.0}
                continue

            std = self._std[c]
            k_val = self.k * std
            z = x - self._mean[c]
            self._sh[c] = max(0.0, self._sh[c] + z - k_val)
            self._sl[c] = min(0.0, self._sl[c] + z + k_val)
            h_val = self.h * std
            drift_flag = self._sh[c] > h_val or abs(self._sl[c]) > h_val
            out[c] = {"drift_flag": bool(drift_flag), "sh": self._sh[c], "sl": self._sl[c]}
        return out


# --------------------------------------------------------------------------
# 3. RUL estimator (exponential degradation curve fit)
# --------------------------------------------------------------------------

def _exp_growth(t, a, b, c):
    """Degradation index model: y(t) = a * exp(b*t) + c"""
    return a * np.exp(np.clip(b * t, -50, 50)) + c


class RULEstimator:
    """
    Fits an exponential degradation curve to a scalar 'health index'
    time-series (e.g. smoothed CHT residual, or oil-pressure decay), then
    extrapolates to a failure threshold to produce Remaining Useful Life
    with a bootstrap confidence interval.
    """

    def __init__(self, failure_threshold: float, min_points: int = 15):
        self.failure_threshold = failure_threshold
        self.min_points = min_points

    def fit_and_estimate(
        self, t_hist: np.ndarray, y_hist: np.ndarray, n_bootstrap: int = 200
    ) -> Optional[dict]:
        if len(t_hist) < self.min_points:
            return None

        # Sanity gate 1: require a genuine positive trend, not noise
        # wobble around a flat mean. A degradation process should show a
        # statistically real positive slope over the observation window.
        slope, _ = np.polyfit(t_hist, y_hist, 1)
        noise_std = float(np.std(y_hist - np.polyval([slope, y_hist.mean()], t_hist)))
        if slope <= 0 or slope * (t_hist[-1] - t_hist[0]) < 3.0 * max(noise_std, 0.5):
            return None

        t0 = t_hist[0]
        t_rel = t_hist - t0

        try:
            p0 = [max(y_hist[0], 1e-3), 0.001, 0.0]
            popt, pcov = curve_fit(_exp_growth, t_rel, y_hist, p0=p0, maxfev=8000)
        except Exception:
            return None

        a, b, c = popt
        rul_point = self._solve_rul(a, b, c, t_rel[-1])
        if rul_point is None:
            return None

        # Sanity gate 2: reject numerically-degenerate fits where the
        # model doesn't even track the current index, or where RUL is
        # near-zero despite the index being nowhere near the failure
        # threshold (a real "about to fail" reading should have the index
        # already well up the curve, not sitting near baseline).
        model_now = _exp_growth(t_rel[-1], a, b, c)
        if not np.isfinite(model_now) or abs(model_now - y_hist[-1]) > max(0.5 * abs(y_hist[-1]), 5.0):
            return None
        if rul_point < 60.0 and y_hist[-1] < 0.4 * self.failure_threshold:
            return None

        # Bootstrap CI: resample residual noise and refit
        residual_std = float(np.std(y_hist - _exp_growth(t_rel, *popt)))
        rul_samples = []
        rng = np.random.default_rng(0)
        for _ in range(n_bootstrap):
            y_boot = y_hist + rng.normal(0, max(residual_std, 1e-6), size=len(y_hist))
            try:
                popt_b, _ = curve_fit(_exp_growth, t_rel, y_boot, p0=popt, maxfev=4000)
                rul_b = self._solve_rul(*popt_b, t_rel[-1])
                if rul_b is not None and 0 <= rul_b < 1e6:
                    rul_samples.append(rul_b)
            except Exception:
                continue

        if len(rul_samples) < 10:
            ci_low, ci_high = rul_point, rul_point
        else:
            ci_low, ci_high = np.percentile(rul_samples, [10, 90])

        return {
            "params": {"a": a, "b": b, "c": c},
            "rul_seconds": float(rul_point),
            "rul_ci90": [float(ci_low), float(ci_high)],
            "current_index": float(y_hist[-1]),
            "failure_threshold": self.failure_threshold,
        }

    def _solve_rul(self, a, b, c, t_last_rel) -> Optional[float]:
        """Analytically solve a*exp(b*t)+c = threshold for t, return
        (t - t_last_rel) as RUL from *now*."""
        target = self.failure_threshold
        if b == 0 or a == 0:
            return None
        arg = (target - c) / a
        if arg <= 0:
            return None
        t_fail = np.log(arg) / b
        rul = t_fail - t_last_rel
        if rul < 0:
            return 0.0
        return float(rul)


# --------------------------------------------------------------------------
# 4. Holt's exponential smoothing (trend smoothing feeding the RUL fit)
# --------------------------------------------------------------------------

class HoltSmoother:
    """Thin wrapper: given a raw degradation-index series, returns a
    smoothed series + one-step-ahead trend, used to denoise the input to
    RULEstimator. Falls back to a simple EWMA if statsmodels unavailable."""

    def smooth(self, series: np.ndarray) -> np.ndarray:
        if len(series) < 4:
            return series
        if _HOLT_AVAILABLE:
            try:
                model = Holt(series, initialization_method="estimated").fit(
                    optimized=True
                )
                return np.asarray(model.fittedvalues)
            except Exception:
                pass
        # fallback EWMA
        alpha = 0.3
        out = [series[0]]
        for x in series[1:]:
            out.append(alpha * x + (1 - alpha) * out[-1])
        return np.array(out)


# --------------------------------------------------------------------------
# 5. SHAP-based explainability
# --------------------------------------------------------------------------

class AnomalyExplainer:
    """
    Wraps SHAP's KernelExplainer around the IsolationForest's decision
    function so each flagged anomaly can be attributed to driving
    feature(s), which the dashboard combines with the raw physics
    residual for a human-readable explanation.
    """

    def __init__(self, detector: AnomalyDetector, background_size: int = 40):
        self.detector = detector
        self.background_size = background_size
        self._explainer = None

    def _ensure_explainer(self):
        if self._explainer is not None:
            return
        if not _SHAP_AVAILABLE or not self.detector.is_fitted:
            return
        bg = np.vstack(self.detector.train_buffer[-self.background_size:])

        def f(X):
            return -self.detector.model.decision_function(X)

        self._explainer = shap.KernelExplainer(f, bg, silent=True)

    def explain(self, feature_vec: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        self._ensure_explainer()
        if self._explainer is None:
            # Fallback: naive attribution via |normalized residual| ranking
            resid_start = len(RAW_FEATURES)
            resid_vals = feature_vec[resid_start:]
            idxs = np.argsort(-np.abs(resid_vals))[:top_k]
            return [(FEATURE_NAMES[resid_start + i], float(resid_vals[i])) for i in idxs]

        shap_vals = self._explainer.shap_values(feature_vec.reshape(1, -1), nsamples=100, silent=True)
        shap_vals = np.asarray(shap_vals).reshape(-1)
        idxs = np.argsort(-np.abs(shap_vals))[:top_k]
        return [(FEATURE_NAMES[i], float(shap_vals[i])) for i in idxs]
