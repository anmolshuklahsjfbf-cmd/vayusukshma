"""
backend.py
==========
FastAPI + WebSocket backend. Streams the Pipeline's live output to any
connected dashboard client in real time, and exposes REST endpoints to
control mission profile, inject faults, toggle replay mode, and simulate
ground-link loss/restoration (degraded-mode autonomy demo).

Run with:
    uvicorn backend:app --reload --port 8000

WebSocket endpoint: ws://localhost:8000/ws/telemetry
    Streams one JSON record per simulated tick (see Pipeline.step()
    output shape) at roughly `TICK_HZ` Hz.

REST endpoints:
    GET  /api/status                 -> current pipeline status
    GET  /api/history?limit=N        -> last N stored samples from SQLite
    POST /api/mission  {name: str}   -> switch mission profile
    POST /api/fault    {...}         -> inject a fault
    POST /api/reset                  -> reset pipeline + clear faults
    POST /api/link/lose              -> simulate ground-link loss
    POST /api/link/restore           -> restore link, drain buffered summary
    GET  /api/replay/{mission}       -> replay stored samples for a mission
"""

from __future__ import annotations

import asyncio
import json
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import Pipeline

TICK_HZ = 5.0  # streamed to dashboard at 5 Hz regardless of internal dt
TICK_DT = 0.4  # internal simulator dt per tick (physical seconds per tick)

app = FastAPI(title="UAV Aero-Engine Digital Twin Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = Pipeline(mission_name="normal_cruise", db_path="twin_data.db", dt=TICK_DT)

_ws_clients: List[WebSocket] = []
_sim_task: Optional[asyncio.Task] = None


class MissionRequest(BaseModel):
    name: str


class FaultRequest(BaseModel):
    fault_type: str
    start_offset_s: float = 5.0   # start N seconds from now (mission time)
    ramp_seconds: float = 120.0
    magnitude: float = 1.0
    channel: Optional[str] = None   # for sensor_drift
    rich: Optional[bool] = None     # for injector_abnormality


async def _simulation_loop():
    period = 1.0 / TICK_HZ
    while True:
        record = pipeline.step()
        payload = json.dumps(record, default=str)
        dead = []
        for ws in _ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in _ws_clients:
                _ws_clients.remove(ws)
        await asyncio.sleep(period)


@app.on_event("startup")
async def _startup():
    global _sim_task
    _sim_task = asyncio.create_task(_simulation_loop())


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            # keep the socket alive; client isn't expected to send anything
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


@app.get("/api/status")
def status():
    if pipeline.latest is None:
        return {"status": "starting"}
    return {
        "status": "running",
        "mission_profile": pipeline.mission.name,
        "t": pipeline.latest["telemetry"]["t"],
        "link_up": pipeline.degraded_mode.link_up,
        "trained": pipeline.latest["ml"]["trained"],
        "advisory": pipeline.latest["decision"]["advisory"],
    }


@app.get("/api/history")
def history(limit: int = 500):
    return pipeline.storage.read_recent(limit=limit)


@app.post("/api/mission")
def set_mission(req: MissionRequest):
    pipeline.set_mission(req.name)
    return {"ok": True, "mission_profile": req.name}


@app.post("/api/fault")
def inject_fault(req: FaultRequest):
    current_t = pipeline.sim._elapsed
    kwargs = {}
    if req.fault_type == "sensor_drift" and req.channel:
        kwargs["channel"] = req.channel
    if req.fault_type == "injector_abnormality" and req.rich is not None:
        kwargs["rich"] = req.rich
    fault = pipeline.add_fault(
        req.fault_type,
        start_t=current_t + req.start_offset_s,
        ramp_seconds=req.ramp_seconds,
        magnitude=req.magnitude,
        **kwargs,
    )
    return {"ok": True, "fault": req.fault_type, "starts_at_mission_t": current_t + req.start_offset_s}


@app.post("/api/reset")
def reset():
    global pipeline
    pipeline.storage.clear()
    pipeline = Pipeline(mission_name="normal_cruise", db_path="twin_data.db", dt=TICK_DT)
    return {"ok": True}


@app.post("/api/link/lose")
def link_lose():
    pipeline.degraded_mode.link_lost()
    return {"ok": True, "link_up": False}


@app.post("/api/link/restore")
def link_restore():
    drained = pipeline.degraded_mode.link_restored()
    return {"ok": True, "link_up": True, "synced_summaries": drained}


@app.get("/api/replay/{mission_profile}")
def replay(mission_profile: str):
    return pipeline.replay_from_db(mission_profile)


@app.get("/")
def root():
    return {
        "service": "UAV Aero-Engine Digital Twin",
        "websocket": "/ws/telemetry",
        "docs": "/docs",
    }
