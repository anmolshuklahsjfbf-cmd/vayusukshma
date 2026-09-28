# VAYUSŪKṢMA

### AI-Enabled Real-Time Digital Twin for Health Monitoring, Fault Prediction & Mission Reliability of Aero Piston Engines

> **VAYUSŪKṢMA** is a proposed AI-enabled digital-twin framework designed for real-time health monitoring, anomaly detection, fault prediction, and mission-reliability assessment of aero piston engines used in **MALE (Medium Altitude Long Endurance) UAV platforms**.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/AI%2FML-Machine%20Learning-orange)](#ai--ml)
[![Digital Twin](https://img.shields.io/badge/System-Digital%20Twin-blue)](#system-architecture)
[![Status](https://img.shields.io/badge/Status-Prototype%20%7C%20Under%20Development-yellow)](#project-status)

---

## ⚠️ Repository Verification Notice

This README has been prepared from the **currently visible GitHub repository structure** and the documented VAYUSŪKṢMA project concept.

### Currently verified from GitHub

* Repository: `vayusukshma`
* Repository visibility: **Public**
* Top-level project directory: `nakshatra_engine`
* Current visible repository history: **1 commit**
* README: **Not currently present**
* Repository description/topics: **Not currently configured**

The GitHub interface currently does not expose the contents of `nakshatra_engine` in a way that allows complete implementation-level verification. Therefore, technical details marked **`[VERIFY]`** should be checked against the actual source code before presenting this repository as a fully validated implementation.

---

# 1. Project Overview

Modern UAV missions depend heavily on the reliability of their propulsion systems. An aero piston engine operates under continuously changing thermal, mechanical, and operational conditions, making early identification of degradation critical for mission safety and availability.

**VAYUSŪKṢMA** aims to address this problem by creating a **software-defined digital representation of an aero piston engine** that continuously processes operational telemetry and estimates the engine's health condition.

The system is designed around four major capabilities:

1. **Real-Time Health Monitoring**
2. **Anomaly & Fault Detection**
3. **Fault Prediction / Prognostics**
4. **Digital-Twin-Based Mission Reliability Assessment**

The intended architecture combines physics-inspired engine modelling, statistical analysis, machine learning, explainable AI, and real-time telemetry visualization.

---

# 2. Problem Statement

Traditional engine monitoring systems primarily focus on detecting parameters that have already crossed predefined thresholds.

This creates several limitations:

* Threshold-based systems may detect failures only after significant degradation.
* Individual sensor readings may not represent the complete engine state.
* Sensor noise can generate false alarms.
* Gradual degradation can remain undetected.
* Operators may receive an alert without understanding its underlying cause.
* Mission-level consequences of an emerging fault may be difficult to estimate.

### VAYUSŪKṢMA's intended approach

Instead of treating each sensor independently, VAYUSŪKṢMA is designed to combine:

**Telemetry → Signal Processing → Digital Twin → Anomaly Detection → Fault Diagnosis → Prediction → Explainability → Mission Decision Support**

---

# 3. Core Objectives

| Objective            | Purpose                                                     |
| -------------------- | ----------------------------------------------------------- |
| Real-Time Monitoring | Continuously observe engine operating parameters            |
| Digital Twin         | Maintain a computational representation of engine behaviour |
| Anomaly Detection    | Identify deviations from expected operating behaviour       |
| Fault Detection      | Detect potentially abnormal engine conditions               |
| Fault Prediction     | Estimate developing degradation before critical failure     |
| Explainable AI       | Provide interpretable reasons behind model outputs          |
| Mission Reliability  | Assess potential impact of engine health on UAV mission     |
| Visualization        | Present health information through an operator dashboard    |

---

# 4. Conceptual System Architecture

```text
                 ┌───────────────────────────┐
                 │     UAV / Engine Data     │
                 │                           │
                 │ RPM | Temperature | Oil   │
                 │ Pressure | Vibration | ...│
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │   Data Acquisition Layer  │
                 │                           │
                 │ CAN / MQTT / Telemetry    │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Preprocessing & Validation│
                 │                           │
                 │ Filtering | Cleaning      │
                 │ Normalization | Validation│
                 └─────────────┬─────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
       ┌───────────────────┐       ┌───────────────────┐
       │   Physics /       │       │     AI / ML        │
       │   Digital Twin    │       │      Engine        │
       │                   │       │                   │
       │ Engine Dynamics   │       │ Anomaly Detection │
       │ State Estimation  │       │ Fault Detection   │
       │ Fault Injection   │       │ Prediction        │
       └─────────┬─────────┘       └─────────┬─────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                 ┌───────────────────────────┐
                 │ Explainability & Health   │
                 │ Assessment                │
                 │                           │
                 │ Health Score              │
                 │ Residual Analysis         │
                 │ SHAP / Model Explanation  │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Mission Reliability Layer │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Operator Dashboard        │
                 │                           │
                 │ Streamlit / Web Interface │
                 │ Real-Time Visualization   │
                 └───────────────────────────┘
```

> **[VERIFY]** The exact data-flow implementation, sensor interfaces, and module boundaries should be confirmed against the contents of `nakshatra_engine`.

---

# 5. Key Features

## 5.1 Real-Time Engine Monitoring

The system is intended to process continuously changing engine parameters such as:

* Engine RPM
* Temperature
* Pressure
* Vibration
* Oil-related parameters
* Fuel-related parameters
* Load
* Other engine telemetry

> **[VERIFY]** The exact telemetry parameters currently implemented in the repository.

---

## 5.2 Digital Twin

The digital-twin component is intended to maintain a computational representation of the engine's expected operating behaviour.

Conceptually:

```text
Physical Engine
      │
      │ Telemetry
      ▼
┌──────────────────┐
│ Digital Twin     │
│                  │
│ Expected State   │
│       ↓          │
│ Predicted Output │
└────────┬─────────┘
         │
         ▼
 Actual vs Expected
         │
         ▼
    Residual/Error
         │
         ▼
 Anomaly / Degradation
```

The difference between expected and observed behaviour can be used as an important health indicator.

> **[VERIFY]** Confirm the exact physical equations, state variables, and numerical model implemented in the repository.

---

# 6. AI & Machine Learning

The intended VAYUSŪKṢMA architecture supports multiple analytical approaches.

### Potential / documented techniques

* **Isolation Forest** — unsupervised anomaly detection
* **CUSUM** — change detection
* **Statistical residual analysis**
* **Curve fitting**
* **Holt smoothing**
* **Bootstrap confidence estimation**
* **SHAP** — model explainability
* **PyTorch** — optional deep-learning models
* **ONNX Runtime** — optional optimized inference
* **Federated Learning / Flower** — potential distributed-learning capability

> **[VERIFY]** Only techniques actually present and operational in the current codebase should be described as implemented features. The above list reflects the project's planned/documented technical architecture rather than confirmed repository contents.

---

# 7. Health Assessment Pipeline

A conceptual health-monitoring pipeline is:

```text
Raw Telemetry
      │
      ▼
Data Validation
      │
      ▼
Signal Processing
      │
      ▼
Feature Extraction
      │
      ▼
Digital Twin Prediction
      │
      ▼
Residual Generation
      │
      ▼
Anomaly Detection
      │
      ▼
Fault Classification
      │
      ▼
Health Assessment
      │
      ▼
Mission Reliability
```

A generalized health indicator can be represented as:

```text
Health State = f(
    Sensor Measurements,
    Operating Conditions,
    Model Residuals,
    Historical Behaviour,
    Detected Anomalies
)
```

> **[VERIFY]** The mathematical formulation and health-score calculation used by the current implementation.

---

# 8. Explainable AI

For safety-critical applications, an AI prediction should ideally not be treated as a black box.

VAYUSŪKṢMA is intended to provide explanations such as:

```text
Detected Anomaly
       │
       ▼
┌───────────────────────┐
│ Contributing Factors │
├───────────────────────┤
│ ↑ Temperature         │
│ ↑ Vibration           │
│ ↓ Oil Pressure       │
│ RPM deviation         │
└──────────┬────────────┘
           │
           ▼
     Fault Hypothesis
```

SHAP-based explanations may be used to determine which features contributed most strongly to a model prediction.

> **[VERIFY]** Confirm whether SHAP is currently implemented and exposed through the application.

---

# 9. Technology Stack

The intended technology stack includes:

### Backend

* Python
* FastAPI
* Uvicorn
* WebSockets
* Pydantic

### Data & Scientific Computing

* NumPy
* Pandas
* SciPy

### Machine Learning

* Scikit-learn
* PyTorch **[VERIFY]**
* SHAP **[VERIFY]**
* ONNX Runtime **[VERIFY]**

### Database

* SQLite **[VERIFY]**
* SQLAlchemy **[VERIFY]**
* Potential TimescaleDB / InfluxDB integration **[VERIFY]**

### Frontend / Visualization

* Streamlit
* Plotly
* HTML/CSS
* JavaScript
* Three.js **[VERIFY]**

### Communication / Telemetry

* SocketCAN / `vcan0` **[VERIFY]**
* MQTT / Paho MQTT **[VERIFY]**

---

# 10. Repository Structure

The currently visible repository structure is:

```text
vayusukshma/
│
└── nakshatra_engine/
    │
    └── [implementation files]
```

The GitHub repository currently exposes `nakshatra_engine` as the top-level project directory.

> **[VERIFY]** Expand this section after inspecting the actual contents of `nakshatra_engine`. A recommended final structure would look similar to:

```text
vayusukshma/
│
├── nakshatra_engine/
│   ├── api/
│   ├── models/
│   ├── digital_twin/
│   ├── ml/
│   ├── simulation/
│   ├── telemetry/
│   ├── database/
│   ├── dashboard/
│   └── ...
│
├── data/
├── tests/
├── docs/
├── requirements.txt
├── README.md
└── LICENSE
```

**Do not add directories to the README's actual structure section unless they exist in the repository.**

---

# 11. Installation

> **[VERIFY]** The following installation procedure should be confirmed against the repository's actual dependency files.

### 1. Clone the repository

```bash
git clone https://github.com/anmolshuklahsjfbf-cmd/vayusukshma.git
cd vayusukshma
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Install dependencies

If a `requirements.txt` exists:

```bash
pip install -r requirements.txt
```

> **[VERIFY]** Confirm whether the repository currently contains `requirements.txt`, `pyproject.toml`, `environment.yml`, or another dependency-management file.

---

# 12. Running the Application

The exact startup command must be verified against the repository.

### Possible FastAPI startup

```bash
uvicorn <module>:app --reload
```

### Possible Streamlit startup

```bash
streamlit run <dashboard_file>.py
```

> **[VERIFY]** Replace `<module>` and `<dashboard_file>` with the actual entry points from the repository.

---

# 13. Configuration

The application may require configuration for:

* Telemetry source
* Database
* API server
* Simulation parameters
* Engine operating parameters
* ML models
* Logging
* Dashboard settings

Recommended environment-file structure:

```env
APP_ENV=development

DATABASE_URL=sqlite:///./vayusukshma.db

API_HOST=127.0.0.1
API_PORT=8000

TELEMETRY_MODE=simulation
```

> **[VERIFY]** Do not add these variables unless they are supported by the actual application.

---

# 14. Simulation & Fault Injection

A major capability envisioned for VAYUSŪKṢMA is the ability to simulate engine operating conditions and introduce controlled faults.

Conceptually:

```text
Normal Engine Model
        │
        ├── Normal Operation
        │
        ├── Temperature Fault
        │
        ├── Pressure Fault
        │
        ├── Vibration Fault
        │
        ├── RPM Deviation
        │
        └── Sensor Anomaly
                │
                ▼
          Digital Twin
                │
                ▼
       AI Detection System
```

This allows the monitoring and prediction pipeline to be evaluated without requiring immediate access to a physical aircraft engine.

> **[VERIFY]** Confirm which fault-injection scenarios are currently implemented.

---

# 15. Dashboard

The intended dashboard provides an operator-facing view of engine condition.

Potential dashboard components include:

* Engine health indicator
* Live telemetry
* RPM trends
* Temperature trends
* Pressure
