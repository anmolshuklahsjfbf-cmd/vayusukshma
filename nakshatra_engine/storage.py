"""
storage.py
==========
Time-series persistence layer using SQLite via SQLAlchemy Core (kept
lightweight -- no ORM overhead needed for an append-mostly telemetry
log). Swappable for TimescaleDB/InfluxDB later: only this module would
need to change, since everything else talks to `Storage.write_sample()`
/ `Storage.read_range()`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, Float, String, Text, select
)


class Storage:
    def __init__(self, db_path: str = "twin_data.db"):
        self.db_path = db_path
        if db_path == ":memory:":
            from sqlalchemy.pool import StaticPool
            self.engine = create_engine(
                "sqlite:///:memory:", future=True,
                connect_args={"check_same_thread": False}, poolclass=StaticPool,
            )
        else:
            self.engine = create_engine(f"sqlite:///{db_path}", future=True)
        self.metadata = MetaData()

        self.telemetry = Table(
            "telemetry", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("t", Float, index=True),
            Column("wall_time", Float),
            Column("mission_profile", String(64)),
            Column("telemetry_json", Text),      # raw sensor snapshot
            Column("twin_json", Text),            # expected + residuals
            Column("ml_json", Text),              # anomaly score/flags, drift, RUL
            Column("decision_json", Text),        # advisory + reason
            Column("fault_labels_json", Text),    # ground-truth injected faults
        )
        self.metadata.create_all(self.engine)

    def write_sample(self, telemetry: dict, twin_out: dict, ml_out: dict, decision: dict,
                      fault_labels: dict):
        with self.engine.begin() as conn:
            conn.execute(self.telemetry.insert().values(
                t=telemetry.get("t", 0.0),
                wall_time=time.time(),
                mission_profile=telemetry.get("mission_profile", "unknown"),
                telemetry_json=json.dumps(telemetry),
                twin_json=json.dumps(twin_out),
                ml_json=json.dumps(ml_out),
                decision_json=json.dumps(decision),
                fault_labels_json=json.dumps(fault_labels),
            ))

    def read_recent(self, limit: int = 500) -> List[dict]:
        with self.engine.connect() as conn:
            stmt = select(self.telemetry).order_by(self.telemetry.c.id.desc()).limit(limit)
            rows = conn.execute(stmt).mappings().all()
        rows = list(reversed(rows))
        return [self._row_to_dict(r) for r in rows]

    def read_all_for_mission(self, mission_profile: str) -> List[dict]:
        with self.engine.connect() as conn:
            stmt = select(self.telemetry).where(
                self.telemetry.c.mission_profile == mission_profile
            ).order_by(self.telemetry.c.id.asc())
            rows = conn.execute(stmt).mappings().all()
        return [self._row_to_dict(r) for r in rows]

    def _row_to_dict(self, r) -> dict:
        return {
            "t": r["t"],
            "wall_time": r["wall_time"],
            "mission_profile": r["mission_profile"],
            "telemetry": json.loads(r["telemetry_json"]),
            "twin": json.loads(r["twin_json"]),
            "ml": json.loads(r["ml_json"]),
            "decision": json.loads(r["decision_json"]),
            "fault_labels": json.loads(r["fault_labels_json"]),
        }

    def clear(self):
        with self.engine.begin() as conn:
            conn.execute(self.telemetry.delete())
