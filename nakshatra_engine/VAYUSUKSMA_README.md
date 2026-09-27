# VĀYUSŪKṢMA — Propulsion Digital Twin Mission Console

`dashboard_v2.py` is a ground-up redesign of the original Streamlit dashboard into a defence/aerospace-grade
mission console, per the VĀYUSŪKṢMA/DRISHTI design spec. **It does not replace or modify any backend
module** (`engine_sim.py`, `faults.py`, `twin_core.py`, `ml_models.py`, `decision_support.py`, `storage.py`,
`pipeline.py`, `backend.py`) — this is a presentation-layer upgrade only. The original `dashboard.py` is
left untouched as a fallback/simpler reference implementation.

## Run it

```bash
# Standalone (simplest — runs the pipeline in-process)
streamlit run dashboard_v2.py

# Or against the backend (real deployment shape — dashboard is a pure client)
uvicorn backend:app --port 8000
streamlit run dashboard_v2.py -- --mode backend
```

## Pages

| Page | What it shows |
|---|---|
| **MISSION** (landing) | Radial mission-completion-probability gauge, GO/CAUTION/ABORT status, RUL/health quick stats, stylized mission-phase stepper, subsystem health snapshot |
| **INSTRUMENTS** | Analog cockpit-style gauge dials — RPM, EGT, CHT, oil pressure/temp/quantity, fuel quantity/pressure/temp, altitude, manifold pressure |
| **DIGITAL TWIN 3D** | Interactive Three.js hero engine visualization — rotate/zoom/pan, RPM-linked piston/crankshaft animation, clickable sensor markers, Normal/X-Ray/Exploded/Sensor view modes, red fault-highlighting on the affected component |
| **LIVE ENGINE** | Interactive stylized engine schematic (2D) with fault-highlighted subsystems, drill-down subsystem health, full live telemetry matrix (value/unit/trend/expected/residual/status) |
| **DIGITAL TWIN** | Data-flow visualization (Physical Engine → Telemetry → Twin → Residual → AI → Mission Reliability) with the currently-active stage highlighted, per-channel Actual/Expected/Residual toggle charts |
| **AI DIAGNOSTICS** | Anomaly score + detection state, SHAP feature-contribution chart, fault diagnosis panel (probable fault, physical reasoning, AI reasoning, progression trend) |
| **RUL & DEGRADATION** | Full RUL chart: historical degradation index, exponential fit, failure threshold, 90% confidence band, predicted failure point |
| **MISSION SIMULATOR** | Real counterfactual "what-if" — spins up a fresh Pipeline instance, injects the requested degradation %, fast-forwards it, and reports genuine simulated RUL/health/completion probability. Also runs a 5-mission environment-only comparison table. |
| **REPLAY** | Loads stored historical telemetry for a mission and scrubs through it — Play/Pause/Restart/1×/2×/5× speed transport controls, plus manual slider scrubbing — "what the system knew at that time," using the same rendering code as the live view |
| **FAULT LAB** | Fault type + severity + ramp controls, live fault-propagation flow (Injected → Physical Response → Residual → AI Detection → Classification → RUL Impact → Mission Risk) |
| **SYSTEM / SECURITY** | Model trust indicators, conceptual security panel, dual-engine illustration, degraded/edge-mode toggle |

## Honest scope notes

Everything above is wired to genuine pipeline output unless noted here. A few elements are clearly-labelled
**conceptual/stylized** because the backend has no corresponding real signal — a fielded system would need
additional instrumentation or models to make them fully real:

- **3D Digital Twin (`DIGITAL TWIN 3D` page)** — a genuine interactive Three.js scene: real rotate/zoom/pan
  via OrbitControls (with a hand-rolled fallback if the CDN is blocked), real RPM-linked piston/crankshaft/
  propeller animation, cam-driven intake/exhaust valve lift and spark-plug flash timed to a real 4-stroke
  cycle per cylinder, a live "CYL 1 — INTAKE/COMPRESSION/POWER/EXHAUST" readout, a throttle-body butterfly
  valve whose angle is genuinely driven by current throttle telemetry, click-to-inspect on both sensor
  markers (with real trend arrows computed against the previous sample, plus a heuristic confidence score)
  and on cylinders directly (shows engine-level CHT/EGT/injection timing/vibration/anomaly score — labelled
  as engine-level since this simulation has no per-cylinder instrumentation), 6 view modes (Normal, X-Ray,
  Exploded, Thermal — colors cylinders/heads/exhaust by actual CHT on a blue→red gradient, Sensor View,
  Flow — animated particles along the fuel/exhaust/oil/coolant paths), and fault-highlighting driven by the
  same `fault_to_subsystem` mapping the 2D view uses. It is **stylized geometry** (boxes/cylinders), not a
  CAD import or photorealistic render, and all rotational animation speeds (crank, propeller) are fixed
  perceptual scale-downs of RPM, not literal rotational or gear-ratio-accurate speed — a real reduction
  gearbox changes prop:crank speed ratio, which is not modelled here. The 3D scene regenerates fresh on
  every Streamlit rerun with the current telemetry baked in, so live *values* update at the dashboard's
  rerun cadence (sub-second) — camera control, view-mode switching, and the animation loop itself run
  natively in the browser and need no rerun at all. **Not yet built** from the fuller 3D spec: a separate
  3D MALE-UAV airframe view with view-mode switching (UAV→Propulsion→Engine→Sensor→DataFlow) and a smooth
  UAV→engine transition, flight/attitude instrumentation (altitude tape, artificial horizon, Mach/pitch/
  roll/heading — none of which exist in the current physics simulation at all; it only models altitude and
  OAT as static per-mission environment, not dynamic flight attitude), animated data-flow-packet
  visualization between architecture layers, camera auto-focus/fly-to on fault detection, 3D-linked mission
  replay scrubbing, a 3D twin-engine comparison, and a scripted one-button demonstration-mode sequence. The
  2D dashboard's Fault Lab, Replay, and System pages cover the same *information* as several of these today,
  just not staged inside the 3D scene or with cinematic camera work.
- **Model trust percentages** (System page) — heuristics derived from residual magnitude / anomaly score /
  drift flags, not a calibrated statistical confidence interval.
- **Dual-engine view** (System page) — runs a second, independent `Pipeline` instance as an illustration of
  the pair-consistency *concept*. There is no physical twin-engine airframe model.
- **Mission phase stepper** (Mission page) — a time-fraction stylization (BASE→CLIMB→TRANSIT→LOITER→
  RETURN→RTB), not derived from a real flight plan, since none exists in this simulation.
- **Security panel** (System page) — status indicators are illustrative placeholders, matching the spec's
  own instruction not to expose implementation detail. No live cryptographic checks run. See
  `technical_doc.md` §8 for the actual roadmap.
- **Mission completion probability** — a presentation-layer heuristic combining health index, RUL margin,
  and anomaly score. It is derived *from* the same signals `DecisionEngine` uses, but does not feed back
  into or override the actual advisory logic — `DecisionEngine` remains the sole safety-relevant authority.

Subsystem health scoring, residuals, anomaly detection, SHAP attribution, RUL + CI, CUSUM drift, the
decision advisory itself, fault injection, replay, and the what-if mission simulator are all real — they
run the existing, unmodified backend.

## SIH Demonstration Mode

The sidebar's **"▶ RUN SIH DEMONSTRATION MODE"** button (standalone mode only) is a real, working one-click
demo sequence, not a scripted video: it resets to a fresh pipeline, fast-forwards it (genuine physics
computation, just run in a tight loop instead of one tick per rerun) past the ~220-second ML warm-up
window, schedules a real `injector_abnormality` fault to begin a few seconds later, and turns on the live
stream automatically. From that point everything is the actual live system running forward: fuel-flow
deviation → EGT residual → AI anomaly flag → RUL drop → mission-completion-probability drop → advisory
escalation, all visible in real time across the MISSION, AI DIAGNOSTICS, and RUL & DEGRADATION pages. A
status banner on every page shows demo progress. Click "Stop demo mode" to dismiss the banner (does not
reset the simulation).

## Camera auto-focus in the 3D view

When a fault is active, the `DIGITAL TWIN 3D` page's camera automatically frames toward that fault's
affected subsystem (e.g. a lubrication fault frames the oil pan/crankcase area) instead of the generic
default view. This is computed at HTML-generation time from the current fault state — each Streamlit
rerun regenerates the 3D scene with the appropriate camera position baked in — rather than a live animated
fly-to triggered by a JS click event inside the iframe, which would require a custom-built Streamlit
component package rather than a plain HTML embed. With multiple simultaneous faults, the alphabetically-
first affected subsystem is used for framing, chosen deterministically so the behavior is predictable.

## Animated data-flow visualization

The `DIGITAL TWIN` page's pipeline-stage row (Physical Engine → ... → Mission Reliability) now includes a
genuinely animated packet traveling along a connecting line via CSS `@keyframes`, colored/timed by current
anomaly state (red and faster during an anomaly, green and slower when nominal) — implemented as a small
self-contained HTML/CSS snippet via `components.html`, since an iframe embed doesn't inherit the parent
page's injected stylesheet.

## Mission profile visualization

The `MISSION` page's phase stepper now includes a small SVG showing an illustrative altitude-shaped
profile (ground → climb → cruise plateau → descend → ground) with a pulsing marker positioned at the
current elapsed-time fraction. This is explicitly a stylized time-fraction shape, not derived from real
flight dynamics — this simulation has no altitude/attitude physics, only a static per-mission environment
(see the "Not yet built" note above regarding flight instrumentation).

## Analog instrument gauges (INSTRUMENTS page)

Real cockpit-style dial gauges for RPM, EGT, CHT, oil pressure/temperature, fuel flow-derived quantity,
altitude, and manifold pressure — all genuinely reading live telemetry, with color-zoned redlines matching
typical small-aero-piston-engine limits. Two categories are worth being precise about:

- **Fuel quantity** is a real derived integral: the simulator only outputs fuel *flow rate*
  (`fuel_flow_lph`), not a tank level, so the dashboard integrates that flow rate against an assumed
  95 L tank capacity (`session_state.fuel_capacity_l`) to produce a genuine, continuously-updated
  remaining-quantity figure. This is honest derived telemetry, not fabrication — it responds correctly to
  the actual simulated fuel flow, including under fault conditions that change consumption.
- **Oil quantity** is *not* an independently simulated tank level either, and — unlike fuel — real aero
  engines don't meaningfully consume oil over a single flight under normal operation, so a "quantity" gauge
  that just drains over time would be physically dishonest. Instead, oil level only decreases when a
  `lubrication_degradation` fault (a genuine leak) is actually active, proportional to that fault's
  severity — verified by test: injecting the fault visibly drops the gauge; leaving it un-injected holds
  the gauge at 100%.
- **Fuel pressure and fuel temperature are not modelled in the underlying physics engine at all** — only
  flow rate is simulated. Rather than silently fabricate plausible-looking numbers, these two gauges use
  simple, clearly-labelled (†) illustrative relationships (pressure scaling mildly with flow demand;
  temperature tracking ambient with a small engine-heat-soak term) and the page explicitly says so in a
  caption, distinguishing them from the genuinely-instrumented gauges alongside them.
- **Altitude** reads the real per-mission-profile value, but that value is static for the duration of a
  single mission in this simulation (no climb/descent physics) — it only changes if you switch mission
  profiles, which the page also states plainly.

## Performance notes

- IsolationForest was tuned (single `decision_function` call instead of `decision_function` + `predict`,
  `n_estimators` reduced from 150→80, `n_jobs=1`) after profiling showed it was the dominant per-tick cost.
  This cut live-tick time from tens of milliseconds to ~1ms, comfortably inside the 5 Hz budget, and cut a
  300-second what-if fast-forward from ~12s to ~5s.
- The what-if scenario runner and mission comparison disable SHAP explainability (`compute_explanations=False`)
  since it's not needed for the summary metrics and is the single most expensive per-tick operation when
  fast-forwarding through many simulated ticks in a few real seconds.
- What-if runs use an in-memory SQLite database (no disk I/O, no stray files) rather than the default
  persistent log file, since these are throwaway projections.

## Testing performed

All 9 pages were exercised via Streamlit's official `AppTest` framework (not just import/syntax checks) —
confirmed to run without exceptions, including: fault injection via button click, fault progression across
multiple reruns, and the what-if scenario button (spins up a nested Pipeline, confirmed ~3s runtime for a
3-minute simulated horizon with 50% injector degradation, returning consistent health/RUL/advisory output).
