# PROJECT NAKSHATRA-ENGINE
### AI-Enabled Digital Twin for Health Monitoring, Fault Prediction & Mission Reliability of a MALE-UAV Aero Piston Engine

**SIH Problem Statement — DRDO** · Prototype software system, fully simulated, architected for later integration with real CAN/FADEC hardware.

---

## 1. Architecture Overview

```
 ┌──────────────────────┐     ┌──────────────────────────────────────────┐
 │  ENGINE SIMULATOR     │     │              DIGITAL TWIN CORE            │
 │  (engine_sim.py)      │     │              (twin_core.py)               │
 │                       │     │                                            │
 │  Coupled 1st-order    │     │  Independent steady-state physics model.  │
 │  ODEs: RPM, CHT, EGT, │────▶│  Computes "expected" value per channel    │
 │  oil T/P, fuel flow,  │ tel │  from (RPM, throttle, altitude, OAT).     │
 │  vibration, bus V     │ eme │                                            │
 │                       │ try │  residual = actual − expected             │
 │  + FAULT INJECTION    │     │  (normalized by per-channel noise floor)  │
 │    (faults.py)        │     └───────────────────┬────────────────────────┘
 │  8 physically-grounded│                          │ residual vector
 │  fault models, each   │                          ▼
 │  biasing equilibrium  │     ┌──────────────────────────────────────────┐
 │  targets (not raw     │     │                AI/ML LAYER                │
 │  state) so effects     │     │              (ml_models.py)               │
 │  are bounded & real.   │     │                                            │
 └───────────┬───────────┘     │  • IsolationForest  → point anomaly score │
             │                 │  • CUSUM chart      → sensor drift flag   │
             │ swappable for   │  • Exp curve fit     → RUL + 90% CI       │
             │ real CAN/FADEC  │  • Holt smoothing    → denoise RUL input  │
             │ telemetry later │  • SHAP              → anomaly attribution│
             │                 └───────────────────┬────────────────────────┘
             │                                     │
             ▼                                     ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                    DECISION SUPPORT / AUTONOMY LAYER                  │
 │                       (decision_support.py)                           │
 │                                                                        │
 │  DecisionEngine: RUL vs (remaining mission + margin) → Advisory       │
 │      CONTINUE → MONITOR → DIVERT_ADVISORY → ABORT_RECOMMENDED         │
 │      → RTB_IMMEDIATE                                                  │
 │                                                                        │
 │  DegradedModeManager: if ground link lost, keeps deciding locally,    │
 │  buffers compressed advisory-transition summaries, drains on         │
 │  link_restored()                                                      │
 └───────────────────────────────┬────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ┌─────────────────────┐     ┌───────────────────────────┐
        │   STORAGE (SQLite)   │     │  BACKEND (FastAPI + WS)   │
        │   (storage.py)       │◀───▶│   (backend.py)            │
        │  Time-series log:    │     │  WebSocket: live stream    │
        │  telemetry, twin,    │     │  REST: mission/fault/link  │
        │  ml, decision,       │     │  control, history, replay  │
        │  ground-truth faults │     └─────────────┬───────────────┘
        └─────────────────────┘                    │
                                                     ▼
                                    ┌───────────────────────────────┐
                                    │  DASHBOARD (Streamlit)         │
                                    │  (dashboard.py)                │
                                    │  Mission-control dark theme:   │
                                    │  live health, fault chips,     │
                                    │  SHAP explanation, RUL +CI,    │
                                    │  advisory banner, CUSUM panel  │
                                    └───────────────────────────────┘
```

**Key design property — swappability.** `twin_core.py`, `ml_models.py`, and `decision_support.py` consume a telemetry **dict** with a fixed schema. Nothing downstream of the simulator knows or cares whether that dict came from `EngineSimulator.step()` or from a real CAN/FADEC bus decoder. Replacing the simulator with real hardware means writing one new adapter that produces the same dict shape — no other file changes.

**Replay mode.** `Pipeline.replay_from_db()` re-emits exactly what was stored (telemetry + twin + ml + decision, all already computed), proving the dashboard's rendering code path is identical whether driven live or from historical logs (`main.py --replay <mission>`).

---

## 2. Module Map

| File | Responsibility |
|---|---|
| `engine_sim.py` | Ground-truth physics simulator (coupled ODEs), mission profiles, environment model |
| `faults.py` | 8 physically-grounded fault models, each returning severity-scaled **equilibrium offsets** |
| `twin_core.py` | Independent physics-informed "expected value" model + residual computation |
| `ml_models.py` | IsolationForest, CUSUM, RUL exponential-fit + bootstrap CI, Holt smoothing, SHAP |
| `decision_support.py` | `DecisionEngine` (advisory rules) + `DegradedModeManager` (link-loss autonomy) |
| `storage.py` | SQLite time-series persistence via SQLAlchemy Core |
| `pipeline.py` | Orchestrates one end-to-end tick; data-source agnostic; hosts sanity gates |
| `backend.py` | FastAPI + WebSocket live streaming; REST control endpoints |
| `dashboard.py` | Streamlit mission-control UI (standalone or backend-connected) |
| `main.py` | Headless CLI demo runner for scripted fault scenarios / replay |

---

## 3. Physics Assumptions

The simulator models a small 4-cylinder turbocharged/naturally-aspirated aero piston engine class typical of MALE-UAV propulsion (Rotax-914-class), using **coupled first-order lag ODEs**, not random noise:

- **RPM** lags a throttle-commanded target with `tau=3s` (governor/mechanical response).
- **CHT** (cylinder head temp) lags a throttle+cooling-dependent equilibrium with `tau=45s` (high thermal inertia of metal).
- **EGT** (exhaust gas temp) lags a combustion-energy equilibrium with `tau=4s` (gas responds much faster than metal).
- **Oil temp** tracks CHT with `tau=60s` and an offset (case-cooled).
- **Oil pressure**, **fuel flow**, **vibration**, **cylinder EGT variance** are recomputed fresh each tick from the current operating point (no memory) — a realistic simplification since these are dominated by instantaneous pump/injector/mechanical behaviour rather than thermal inertia.
- **Cooling efficiency** and **air density ratio** are derived from a simplified barometric/ISA model, so altitude and OAT genuinely affect CHT/EGT equilibria (colder/thinner air changes both cooling and combustion energy release).
- **Alternator/bus voltage** is RPM-dependent (belt-driven), battery-buffered within realistic bounds.

Five mission profiles (`normal_cruise`, `high_altitude`, `hot_weather`, `endurance`, `rapid_throttle_transitions`) each define a throttle(t) function and an `Environment` (altitude, OAT), so the same physics produces materially different steady-state operating points per mission — e.g. `hot_weather` degrades cooling efficiency and raises CHT equilibrium even with zero faults injected.

---

## 4. Fault Models — Cause/Effect Mapping

**Design principle:** every fault is a severity-ramped (0→1 over `ramp_seconds`) bias on the *equilibrium target* that a lag-ODE channel pulls toward — not a direct per-tick addition to already-integrated state. This was a deliberate fix during development: naively adding a fixed delta to `state.cht_c` every 0.5s tick, with only a weak (~1-2%/tick) restoring pull from the lag ODE, produces **unbounded runaway** rather than a new physically-bounded steady state. Channels without memory (fuel flow, oil pressure, vibration, EGT variance) are safe to bias directly since they're recomputed fresh each tick.

| Fault | Physical cause | Effect chain |
|---|---|---|
| **Misfire** | Cylinder fails to fire consistently | Torque loss → RPM target droops; unburned fuel → cylinder-to-cylinder EGT variance spikes; average EGT dips (incomplete combustion) |
| **Injector abnormality** | Injector fouling/leak, over- or under-fueling | Fuel flow drifts off nominal RPM×throttle curve; EGT moves opposite direction (rich cools exhaust, lean heats it); EGT variance rises |
| **Cooling degradation** | Fin fouling, coolant leak, fan wear | Effective cooling efficiency drops → CHT equilibrium rises; oil temp equilibrium rises less (partial case-cooling coupling) |
| **Lubrication degradation** | Pump wear, filter clogging, slow leak | Oil pressure decays beyond the temperature/viscosity model alone; oil temp equilibrium rises (friction); mild vibration rise |
| **Sensor drift** | Thermocouple aging, pressure transducer zero-shift | A *chosen channel's reported value* accumulates additive bias while the true plant is unaffected — models a lying sensor, not a real fault; bias is time-and-severity bounded (saturates, doesn't run away) |
| **Combustion instability** | Borderline detonation / partial misfire | Higher-frequency oscillation added to RPM/EGT *targets* (still bounded by the lag ODE — genuine oscillation, not runaway); elevated noisy EGT variance |
| **Overheating trend** | Compound cooling+load fault (vapor lock / boil-off risk) | CHT/EGT equilibria rise **non-linearly** with severity (`severity²` acceleration term) but remain bounded at full severity — the "silent killer" trend RUL estimation is meant to catch early |
| **Abnormal vibration** | Bearing wear, prop imbalance, mount fatigue | Sustained RMS rise + periodic imbalance-signature component |

Each fault records its severity into `state.fault_labels` for **ground-truth evaluation only** — this is never fed into the ML feature vector, to keep detection honest (the ML layer only ever sees telemetry + physics residuals, exactly as it would with real hardware where "which fault" is unknown).

---

## 5. AI/ML Approach

**Feature vector:** raw sensor channels (RPM, CHT, EGT, oil T/P, fuel flow, vibration, bus V, EGT variance) concatenated with the twin's **normalized physics residuals** for the subset of channels the twin models (CHT, EGT, oil T/P, fuel flow, vibration, EGT variance). Residuals are normalized by a per-channel noise floor so channels of very different units/scales are comparable.

- **Anomaly detection (primary): IsolationForest** — trained on a healthy-operation buffer (default 220s, chosen to exceed ~5× the slowest thermal time constant so the training set doesn't include the engine's own warm-up transient — an early bug we caught and fixed, see §7). Contamination=0.03. A short hysteresis window (majority of last 6 samples) prevents advisory flapping from single-sample score noise near the decision boundary.
- **Sensor drift: CUSUM control chart** — implemented directly (no library), because CUSUM is the textbook tool for slow additive bias, not IsolationForest (tuned for point anomalies). Also gated to only baseline after the training/warm-up window closes, for the same reason as above.
- **RUL estimation (primary): exponential curve fit** (`scipy.optimize.curve_fit`, model `y = a·exp(b·t) + c`) on a smoothed CHT-residual degradation index, extrapolated to a failure threshold, with a **90% bootstrap confidence interval** (200 resamples of the residual noise, refit each time). Includes sanity gates: (1) requires a statistically real positive slope over the observation window — rejects fitting noise/warm-up wobble as "degradation"; (2) rejects fits where the model doesn't track the current index; (3) rejects near-zero RUL when the index is nowhere near the failure threshold (numerically degenerate fits).
- **Trend smoothing:** Holt's exponential smoothing (`statsmodels`) denoises the degradation index before curve fitting (falls back to EWMA if statsmodels unavailable).
- **Explainability: SHAP KernelExplainer** wraps the IsolationForest decision function, attributing each flagged anomaly to its top-3 driving features. Recomputed at most every 2 seconds (wall-clock) while an anomaly persists, not every tick, since KernelExplainer sampling is too expensive for a 5 Hz live stream — a performance trade-off explicitly noted rather than hidden.

**Stretch goals (not implemented in this prototype, scoped for extension):**
- LSTM Autoencoder (PyTorch) for sequence-level reconstruction-error drift detection.
- Gradient-boosted RUL regressor (XGBoost) on engineered features (CHT rise rate, vibration RMS trend, oil pressure decay slope), matching the NASA C-MAPSS PHM benchmark approach.
- Federated learning (Flower) simulating 2-3 UAV clients training locally without sharing raw telemetry — see §8.

---

## 6. Decision Support / Autonomy Layer

`DecisionEngine.decide()` applies rules in priority order:

1. **RUL below (remaining mission time + safety margin)** → `ABORT_RECOMMENDED`, or `RTB_IMMEDIATE` if RUL is additionally below the time needed to reach friendly territory + margin.
2. **Severe active fault class + sustained anomaly** (misfire, lubrication degradation, overheating trend) → `DIVERT_ADVISORY`.
3. **Sensor drift only, no real anomaly** → `MONITOR` (flagged for maintenance log; explicitly does **not** trigger a flight-path change, since a lying sensor with a healthy engine shouldn't ground the mission).
4. **Generic elevated anomaly, unclassified** → `MONITOR` (increase logging/sampling, wait for trend confirmation).
5. Otherwise → `CONTINUE`.

Every advisory carries a human-readable `reason` string referencing the actual RUL/CI, mission-context numbers, or fault names — never a bare enum value, per the "explicit recommendation, not just a raw number" requirement.

**Degraded-mode (link-loss) autonomy:** `DegradedModeManager` runs the identical `DecisionEngine` logic onboard regardless of link state. When the link is down, it buffers only **advisory transitions** plus periodic (60s) summary snapshots — not every tick — modeling the constrained-downlink requirement. On `link_restored()`, the buffer drains (in a real deployment: transmits) and clears. This is exercised via `POST /api/link/lose` / `/api/link/restore` in the backend, or the sidebar buttons in the dashboard.

---

## 7. Notable Engineering Issues Found & Fixed During Build

Documented here deliberately, since a DRDO evaluator will want to see that the physics/fault model was actually stress-tested, not just written once and assumed correct:

1. **Fault runaway bug.** Initial fault implementation added fixed deltas directly to already-integrated state (`state.cht_c += ...`) every tick. Because the lag ODE's restoring pull is only ~1-2% of the equilibrium gap per tick, this produced CHT values exceeding 2800°C within minutes — clearly unphysical. Fixed by re-architecting faults to bias the **equilibrium target** instead (see §4), verified by rerunning the overheating-trend scenario and confirming CHT plateaus at a bounded, physically sane value (~230°C) rather than diverging.
2. **RUL false-positive on warm-up transient.** With a short healthy-training window, the RUL curve fit interpreted the engine's own normal thermal warm-up (CHT rising from ~120°C to steady-state over ~135s) as an exponential degradation trend, triggering a false `RTB_IMMEDIATE` on a perfectly healthy engine. Fixed by (a) extending the training window past the settling time, and (b) adding explicit statistical sanity gates (real positive slope required, model-fit consistency check, minimum-index-vs-threshold check) before trusting any RUL estimate.
3. **CUSUM baseline capturing transient.** Same root cause as #2 applied to the sensor-drift detector — fixed by gating CUSUM baseline collection to start only after the training window closes.
4. **Advisory flapping.** IsolationForest scores near the decision boundary flipped `is_anomaly` sample-to-sample under normal telemetry noise, causing the dashboard advisory to flicker between `MONITOR` and `DIVERT_ADVISORY`. Fixed with a 6-sample majority-vote hysteresis window.

A known remaining limitation: CUSUM occasionally flags genuine mission-profile-driven throttle oscillation (e.g. the slow sinusoidal throttle modulation in `normal_cruise`) as sensor drift on CHT/EGT. This only ever produces the mildest advisory (`MONITOR`, no flight-path change) by design, so it is a nuisance-level false positive rather than a safety-relevant one — but a production system would want a throttle-conditioned CUSUM baseline rather than an unconditional one.

---

## 8. Edge Deployment, Security, and Fleet-Learning Roadmap

*(Documentation-level, per spec — not implemented as running code in this prototype.)*

**Edge deployment.** The IsolationForest and exponential-curve-fit RUL model are both lightweight enough to run natively on Jetson-class onboard compute without acceleration. For the stretch-goal LSTM Autoencoder, the deployment path is: train offline → export to ONNX → apply post-training INT8 quantization via ONNX Runtime → validate accuracy delta on a held-out fault-injected dataset before fielding. Only summarized health indices (anomaly score, RUL point estimate + CI bounds, active advisory) — not raw telemetry — need to sync over a constrained downlink, matching the `DegradedModeManager` buffering strategy already implemented.

**Security.** Three concerns, each with a concrete mitigation direction:
- *Tamper-evident logging*: each stored telemetry row should be chained (hash of row N included in row N+1, e.g. a simple hash-chain or Merkle log) so post-incident forensics can detect retroactive log tampering. Not implemented in `storage.py` in this prototype, but the schema is additive-only (no update/delete except explicit `clear()`), which is a starting point.
- *Authenticated telemetry link*: real CAN/FADEC integration should run over an authenticated transport (e.g., signed MQTT messages or a CAN-FD frame authentication scheme) rather than raw unauthenticated bus traffic.
- *Sensor-spoofing detection via physics cross-check*: this is precisely what the Digital Twin residual already does — a spoofed sensor reading that's physically inconsistent with the engine's actual operating point (RPM, throttle) will show up as a large residual, the same signal used for fault detection. This is a natural byproduct of the twin architecture, not a bolt-on.

**Fleet-learning.** The stretch-goal Flower (`flwr`) integration would simulate 2-3 UAV "clients," each running its own `Pipeline` instance locally, periodically sending only **model weight updates** (IsolationForest doesn't federate natively — this would require substituting a federatable anomaly model, e.g. a shallow autoencoder) to a central aggregator, never raw telemetry. This lets a fleet collectively improve fault detection without any single UAV's flight data leaving its secure boundary — directly relevant to a multi-UAV DRDO deployment where telemetry sensitivity varies by mission.

**Validation plan (against real engine test-rig data).** Before fielding: (1) replace `engine_sim.py` with a rig telemetry adapter producing the same dict schema (validating the swappability property claimed in §1); (2) re-derive/re-fit the twin's equilibrium coefficients (§3) against real steady-state rig sweeps across throttle/altitude/temperature rather than the illustrative constants used here; (3) inject *real* rig faults (e.g. deliberately foul an injector) and confirm the residual/anomaly response direction matches §4's cause-effect table, not just magnitude; (4) re-tune the IsolationForest contamination rate and CUSUM `k`/`h` parameters against the rig's actual sensor noise floor rather than the simulated noise floor assumed here; (5) validate RUL point estimates against rig-induced run-to-failure trials before trusting any CI band operationally.

---

## 9. Running the Prototype

```bash
pip install -r requirements.txt

# Headless CLI demo (single fault scenario)
python main.py --mission normal_cruise --fault overheating_trend --at 260 --ramp 200 --duration 700

# Headless CLI demo (all 8 faults back-to-back)
python main.py --scenario full

# Full stack: backend + dashboard as separate processes (closer to real deployment)
uvicorn backend:app --port 8000          # terminal 1
streamlit run dashboard.py -- --mode backend   # terminal 2

# Or standalone dashboard only (simplest local demo, no backend needed)
streamlit run dashboard.py
```

Dashboard sidebar lets you: switch mission profile, inject any of the 8 fault types with adjustable ramp/magnitude, simulate ground-link loss/restore, and reset the simulation.
