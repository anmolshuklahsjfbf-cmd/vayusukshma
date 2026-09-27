"""
main.py
=======
Headless CLI demo runner. Exercises the full pipeline through a scripted
mission + fault-injection scenario and prints a running health/decision
log to the terminal -- useful for a live terminal demo, CI smoke-testing,
or generating a log to feed into replay mode afterwards.

Usage:
    python main.py                                  # default demo scenario
    python main.py --mission hot_weather             # different mission
    python main.py --fault overheating_trend --at 60 --ramp 180
    python main.py --scenario full                   # runs all 8 fault types back-to-back
    python main.py --replay normal_cruise             # replay a previously stored mission
"""

from __future__ import annotations

import argparse
import time

from pipeline import Pipeline

ALL_FAULTS = [
    "misfire", "injector_abnormality", "cooling_degradation",
    "lubrication_degradation", "sensor_drift", "combustion_instability",
    "overheating_trend", "abnormal_vibration",
]


def print_row(rec: dict):
    t = rec["telemetry"]
    ml = rec["ml"]
    dec = rec["decision"]
    faults = ",".join(rec["fault_labels"].keys()) or "-"
    rul = f"{ml['rul']['rul_seconds']/60:.1f}min" if ml["rul"] else "n/a"
    print(
        f"t={t['t']:7.1f}s | RPM={t['rpm']:5.0f} CHT={t['cht_c']:6.1f}C "
        f"EGT={t['egt_c']:6.1f}C OilP={t['oil_pressure_kpa']:6.1f}kPa "
        f"Vib={t['vibration_g']:.3f}g | anomaly={ml['anomaly_score']:+.2f} "
        f"({'ANOM' if ml['is_anomaly'] else 'ok  '}) | RUL={rul:>9s} | "
        f"faults=[{faults}] | ADVISORY={dec['advisory']}"
    )


def run_scenario(mission: str, fault: str | None, fault_at: float, ramp: float,
                  magnitude: float, duration_s: float, dt: float, print_every: int,
                  db_path: str):
    p = Pipeline(mission_name=mission, db_path=db_path, dt=dt)
    if fault:
        p.add_fault(fault, start_t=fault_at, ramp_seconds=ramp, magnitude=magnitude)
        print(f">>> Scheduled fault '{fault}' at t={fault_at}s, ramp={ramp}s, magnitude={magnitude}")

    print(f">>> Running mission profile '{mission}' for {duration_s}s (dt={dt}s)...")
    print("-" * 130)
    n_steps = int(duration_s / dt)
    for i in range(n_steps):
        rec = p.step()
        if i % print_every == 0:
            print_row(rec)
    print("-" * 130)
    print(f">>> Done. {n_steps} samples written to {db_path}")


def run_full_scenario(mission: str, dt: float, print_every: int, db_path: str):
    """Runs all 8 fault types back-to-back on the same mission, each with
    its own quiet healthy period before onset, for a comprehensive demo."""
    p = Pipeline(mission_name=mission, db_path=db_path, dt=dt)
    segment_s = 260.0
    for i, fault in enumerate(ALL_FAULTS):
        onset = p.sim._elapsed + 60.0
        p.add_fault(fault, start_t=onset, ramp_seconds=140.0, magnitude=1.0)
        print(f"\n>>> Segment {i+1}/{len(ALL_FAULTS)}: injecting '{fault}' at t={onset:.0f}s")
        n_steps = int(segment_s / dt)
        for j in range(n_steps):
            rec = p.step()
            if j % print_every == 0:
                print_row(rec)
    print(f"\n>>> Full 8-fault scenario complete. Data written to {db_path}")


def run_replay(mission: str, db_path: str):
    p = Pipeline(mission_name=mission, db_path=db_path)
    rows = p.replay_from_db(mission)
    if not rows:
        print(f"No stored samples found for mission '{mission}' in {db_path}.")
        return
    print(f">>> Replaying {len(rows)} stored samples for mission '{mission}'")
    print("-" * 130)
    for i, row in enumerate(rows):
        if i % 20 == 0:
            print_row({
                "telemetry": row["telemetry"], "ml": row["ml"],
                "decision": row["decision"], "fault_labels": row["fault_labels"],
            })
    print("-" * 130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="UAV Aero-Engine Digital Twin - CLI demo runner")
    ap.add_argument("--mission", default="normal_cruise",
                     choices=["normal_cruise", "high_altitude", "hot_weather",
                              "endurance", "rapid_throttle_transitions"])
    ap.add_argument("--fault", default=None, choices=ALL_FAULTS)
    ap.add_argument("--at", type=float, default=60.0, help="fault onset time (s)")
    ap.add_argument("--ramp", type=float, default=150.0, help="fault ramp duration (s)")
    ap.add_argument("--magnitude", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=600.0, help="total sim duration (s)")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--print-every", type=int, default=20)
    ap.add_argument("--db", default="demo_run.db")
    ap.add_argument("--scenario", choices=["single", "full"], default="single")
    ap.add_argument("--replay", default=None, metavar="MISSION_PROFILE",
                     help="replay previously stored samples for this mission instead of simulating")
    args = ap.parse_args()

    start = time.time()
    if args.replay:
        run_replay(args.replay, args.db)
    elif args.scenario == "full":
        run_full_scenario(args.mission, args.dt, args.print_every, args.db)
    else:
        run_scenario(args.mission, args.fault, args.at, args.ramp, args.magnitude,
                     args.duration, args.dt, args.print_every, args.db)
    print(f"\n(wall time: {time.time() - start:.1f}s)")
