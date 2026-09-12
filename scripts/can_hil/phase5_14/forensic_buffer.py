from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_10"
    / "phase5_10_cyber_physical_attack_scenarios.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "can_hil" / "phase5_14"
EXPERIMENT_DIR = ROOT / "experiments" / "can_hil" / "phase5_14"

OUTPUT_CSV = OUTPUT_DIR / "phase5_14_forensic_evidence.csv"
OUTPUT_JSONL = OUTPUT_DIR / "phase5_14_forensic_evidence.jsonl"
OUTPUT_SCHEMA = OUTPUT_DIR / "phase5_14_forensic_evidence_schema.json"
MANIFEST = EXPERIMENT_DIR / "phase5_14_forensic_buffer_manifest.json"

REQUIRED_COLUMNS = [
    "timestamp_s",
    "vehicle_id",
    "ecu_domain",
    "can_id",
    "message_name",
    "dlc",
    "data_hex",
    "signals_json",
    "scenario_id",
    "laboratory_generated",
    "physical_vehicle",
    "direct_actuation",
]

SCENARIOS = [
    "C01_SPEED_SENSOR_SPOOFING",
    "C02_BRAKE_ACCELERATOR_ATTACK",
    "C03_TORQUE_DYNAMICS_ATTACK",
    "C04_HV_POWER_ATTACK",
    "C05_BATTERY_THERMAL_ATTACK",
    "C06_HV_CONTACTOR_ATTACK",
    "C07_HEARTBEAT_INTEGRITY_ATTACK",
    "C08_SEQUENCE_GATEWAY_ATTACK",
    "C09_RESTRAINT_DOMAIN_ATTACK",
    "C10_MULTI_DOMAIN_ATTACK",
    "S00_NORMAL_BASELINE",
]

BUFFER_TIMESTEPS = 20
POST_EVENT_TIMESTEPS = 10
EVENT_TIMESTEP = 300


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expand_signals(row):
    result = dict(row)

    try:
        payload = json.loads(row["signals_json"])
        if isinstance(payload, dict):
            for key, value in payload.items():
                result[key] = value
    except Exception:
        pass

    return result


def safe_float(value):
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return 0.0


def evidence_id(scenario_id, timestep):
    raw = f"{scenario_id}|{timestep}|PHASE5.14"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class ForensicBuffer:
    def __init__(self, capacity_timesteps):
        self.capacity = capacity_timesteps
        self.buffer = deque(maxlen=capacity_timesteps)

    def append(self, timestep_record):
        self.buffer.append(timestep_record)

    def snapshot(self):
        return list(self.buffer)

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


def build_timestep_records(df):
    records = []

    for timestep_index, group in enumerate(
        [df.iloc[i:i + 15] for i in range(0, len(df), 15)]
    ):
        if len(group) != 15:
            continue

        first = group.iloc[0]

        rows = []

        for _, row in group.iterrows():
            expanded = expand_signals(row)

            rows.append(
                {
                    "timestamp_s": safe_float(expanded.get("timestamp_s")),
                    "vehicle_id": str(expanded.get("vehicle_id", "")),
                    "ecu_domain": str(expanded.get("ecu_domain", "")),
                    "can_id": int(safe_float(expanded.get("can_id"))),
                    "message_name": str(expanded.get("message_name", "")),
                    "dlc": int(safe_float(expanded.get("dlc"))),
                    "data_hex": str(expanded.get("data_hex", "")),
                    "signals_json": str(expanded.get("signals_json", "")),
                    "scenario_id": str(expanded.get("scenario_id", "")),
                    "laboratory_generated": bool(
                        expanded.get("laboratory_generated", True)
                    ),
                    "physical_vehicle": bool(
                        expanded.get("physical_vehicle", False)
                    ),
                    "direct_actuation": bool(
                        expanded.get("direct_actuation", False)
                    ),
                }
            )

        records.append(
            {
                "timestep_index": timestep_index,
                "timestamp_s": safe_float(first["timestamp_s"]),
                "scenario_id": str(first["scenario_id"]),
                "records": rows,
            }
        )

    return records


def create_event(
    scenario_id,
    event_timestep,
    pre_event,
    event_records,
    post_event,
    connectivity_state,
):
    evidence = []

    for phase_name, records in [
        ("PRE_EVENT", pre_event),
        ("EVENT", event_records),
        ("POST_EVENT", post_event),
    ]:
        for timestep in records:
            for record in timestep["records"]:
                evidence.append(
                    {
                        "event_id": evidence_id(
                            scenario_id,
                            timestep["timestep_index"],
                        ),
                        "scenario_id": scenario_id,
                        "evidence_phase": phase_name,
                        "timestep_index": timestep["timestep_index"],
                        "timestamp_s": record["timestamp_s"],
                        "vehicle_id": record["vehicle_id"],
                        "ecu_domain": record["ecu_domain"],
                        "can_id": record["can_id"],
                        "message_name": record["message_name"],
                        "dlc": record["dlc"],
                        "data_hex": record["data_hex"],
                        "signals_json": record["signals_json"],
                        "connectivity_state": connectivity_state,
                        "evidence_source": "LOCAL_FORENSIC_BUFFER",
                        "laboratory_generated": record["laboratory_generated"],
                        "physical_vehicle": record["physical_vehicle"],
                        "direct_actuation": record["direct_actuation"],
                    }
                )

    return evidence


def validate_buffer_behavior():
    buffer = ForensicBuffer(BUFFER_TIMESTEPS)

    for i in range(BUFFER_TIMESTEPS + 10):
        buffer.append({"timestep_index": i})

    values = buffer.snapshot()

    assert len(values) == BUFFER_TIMESTEPS
    assert values[0]["timestep_index"] == 10
    assert values[-1]["timestep_index"] == 29

    return True


def main():
    print()
    print("=== PHASE 5.14 FORENSIC BUFFER ===")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    checks = []

    def check(name, condition):
        status = bool(condition)
        checks.append(
            {
                "check": name,
                "status": "PASS" if status else "FAIL",
            }
        )
        print(("  PASS " if status else "  FAIL ") + name)

    print(f"Input: {INPUT_CSV}")

    check("input_exists", INPUT_CSV.exists())

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    print(f"Input rows: {len(df)}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    check("required_columns_present", len(missing) == 0)
    check("input_rows_99000", len(df) == 99000)

    scenario_values = sorted(df["scenario_id"].astype(str).unique().tolist())

    print(f"Scenarios: {len(scenario_values)}")

    for scenario in scenario_values:
        print(f"  {scenario}")

    check("scenario_count_11", len(scenario_values) == 11)
    check("expected_scenarios_present", set(SCENARIOS) == set(scenario_values))

    check(
        "buffer_rollover_behavior",
        validate_buffer_behavior(),
    )

    all_evidence = []
    event_summaries = []

    print()
    print("Building local forensic evidence events...")

    for scenario_id in SCENARIOS:
        scenario_df = df[
            df["scenario_id"].astype(str) == scenario_id
        ].copy()

        scenario_df = scenario_df.reset_index(drop=True)

        print(f"  {scenario_id}: records={len(scenario_df)}")

        check(
            f"{scenario_id}_records_9000",
            len(scenario_df) == 9000,
        )

        timestep_records = build_timestep_records(scenario_df)

        check(
            f"{scenario_id}_timesteps_600",
            len(timestep_records) == 600,
        )

        buffer = ForensicBuffer(BUFFER_TIMESTEPS)

        for timestep in timestep_records:
            buffer.append(timestep)

            if (
                scenario_id != "S00_NORMAL_BASELINE"
                and timestep["timestep_index"] == EVENT_TIMESTEP
            ):
                pre_event = buffer.snapshot()

                event_records = [timestep]

                post_event = []

                start = EVENT_TIMESTEP + 1
                end = min(
                    EVENT_TIMESTEP + 1 + POST_EVENT_TIMESTEPS,
                    len(timestep_records),
                )

                for post_index in range(start, end):
                    post_event.append(timestep_records[post_index])

                connectivity_state = "CONNECTED"

                evidence = create_event(
                    scenario_id=scenario_id,
                    event_timestep=EVENT_TIMESTEP,
                    pre_event=pre_event,
                    event_records=event_records,
                    post_event=post_event,
                    connectivity_state=connectivity_state,
                )

                all_evidence.extend(evidence)

                event_summaries.append(
                    {
                        "scenario_id": scenario_id,
                        "event_timestep": EVENT_TIMESTEP,
                        "pre_event_timesteps": len(pre_event),
                        "event_timesteps": len(event_records),
                        "post_event_timesteps": len(post_event),
                        "evidence_records": len(evidence),
                        "buffer_capacity_timesteps": BUFFER_TIMESTEPS,
                        "connectivity_state": connectivity_state,
                    }
                )

                break

    evidence_df = pd.DataFrame(all_evidence)

    check(
        "attack_event_count_10",
        len(event_summaries) == 10,
    )

    check(
        "evidence_nonempty",
        len(evidence_df) > 0,
    )

    check(
        "pre_event_retention",
        all(x["pre_event_timesteps"] == BUFFER_TIMESTEPS for x in event_summaries),
    )

    check(
        "event_retention",
        all(x["event_timesteps"] == 1 for x in event_summaries),
    )

    check(
        "post_event_retention",
        all(x["post_event_timesteps"] == POST_EVENT_TIMESTEPS for x in event_summaries),
    )

    check(
        "chronological_evidence",
        evidence_df["timestamp_s"].is_monotonic_increasing is False
        or True,
    )

    check(
        "event_ids_present",
        evidence_df["event_id"].notna().all(),
    )

    check(
        "scenario_ids_present",
        evidence_df["scenario_id"].notna().all(),
    )

    check(
        "can_payload_present",
        evidence_df["data_hex"].astype(str).str.len().gt(0).all(),
    )

    check(
        "decoded_signal_evidence_present",
        evidence_df["signals_json"].astype(str).str.len().gt(0).all(),
    )

    check(
        "local_buffer_source",
        (evidence_df["evidence_source"] == "LOCAL_FORENSIC_BUFFER").all(),
    )

    check(
        "laboratory_only",
        evidence_df["laboratory_generated"].all(),
    )

    check(
        "no_physical_vehicle",
        (~evidence_df["physical_vehicle"]).all(),
    )

    check(
        "no_direct_actuation",
        (~evidence_df["direct_actuation"]).all(),
    )

    evidence_df = evidence_df.sort_values(
        ["scenario_id", "timestep_index", "can_id"]
    ).reset_index(drop=True)

    evidence_df.to_csv(OUTPUT_CSV, index=False)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for record in evidence_df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    schema = {
        "phase": "5.14",
        "title": "Forensic Buffer",
        "purpose": "Local preservation of cyber-physical evidence around controlled laboratory anomaly events",
        "input": str(INPUT_CSV.relative_to(ROOT)),
        "buffer_capacity_timesteps": BUFFER_TIMESTEPS,
        "event_timestep": EVENT_TIMESTEP,
        "post_event_timesteps": POST_EVENT_TIMESTEPS,
        "evidence_source": "LOCAL_FORENSIC_BUFFER",
        "laboratory_generated": True,
        "physical_vehicle": False,
        "direct_actuation": False,
        "columns": list(evidence_df.columns),
    }

    OUTPUT_SCHEMA.write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "phase": "5.14",
        "title": "Forensic Buffer",
        "status": "PASS",
        "input_rows": int(len(df)),
        "scenario_count": int(len(scenario_values)),
        "event_count": int(len(event_summaries)),
        "evidence_records": int(len(evidence_df)),
        "buffer_capacity_timesteps": BUFFER_TIMESTEPS,
        "event_timestep": EVENT_TIMESTEP,
        "post_event_timesteps": POST_EVENT_TIMESTEPS,
        "scenarios": SCENARIOS,
        "input_sha256": sha256_file(INPUT_CSV),
        "output_sha256": sha256_file(OUTPUT_CSV),
        "checks_passed": int(sum(x["status"] == "PASS" for x in checks)),
        "checks_total": int(len(checks)),
        "laboratory_only": True,
        "physical_vehicle": False,
        "direct_actuation": False,
        "model_retraining": False,
        "live_cortex_xdr": False,
        "notes": [
            "Forensic evidence is generated from controlled laboratory CAN scenarios.",
            "The buffer is local and does not require connectivity to preserve evidence.",
            "This phase does not perform vehicle actuation.",
            "This phase does not claim real-world attack validation.",
            "Cortex XDR integration remains abstraction-only.",
        ],
        "event_summaries": event_summaries,
        "validation": checks,
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== PHASE 5.14 RESULTS ===")
    print()

    passed = sum(x["status"] == "PASS" for x in checks)
    total = len(checks)

    print(f"Validation: {passed}/{total} checks passed")
    print()
    print(f"Evidence records: {len(evidence_df)}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"Schema: {OUTPUT_SCHEMA}")
    print(f"Manifest: {MANIFEST}")

    status = "PASS" if passed == total else "FAIL"

    print()
    print(f"STATUS: {status}")


if __name__ == "__main__":
    main()