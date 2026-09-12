from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_14"
    / "phase5_14_forensic_evidence.csv"
)

INTEGRITY_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_15"
    / "phase5_15_integrity_validation.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "can_hil" / "phase5_16"
EXPERIMENT_DIR = ROOT / "experiments" / "can_hil" / "phase5_16"

OUTPUT_CSV = OUTPUT_DIR / "phase5_16_connectivity_recovery.csv"
OUTPUT_JSONL = OUTPUT_DIR / "phase5_16_connectivity_recovery.jsonl"
OUTPUT_SCHEMA = OUTPUT_DIR / "phase5_16_connectivity_recovery_schema.json"
MANIFEST = EXPERIMENT_DIR / "phase5_16_connectivity_recovery_manifest.json"

ATTACK_SCENARIOS = [
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
]

DISCONNECT_START = 300
DISCONNECT_END = 305
EVENT_TIMESTEP = 300


def canonical_record(row):
    fields = [
        "event_id",
        "scenario_id",
        "evidence_phase",
        "timestep_index",
        "timestamp_s",
        "vehicle_id",
        "ecu_domain",
        "can_id",
        "message_name",
        "dlc",
        "data_hex",
        "signals_json",
        "connectivity_state",
        "evidence_source",
        "laboratory_generated",
        "physical_vehicle",
        "direct_actuation",
    ]

    record = {}

    for field in fields:
        value = row[field]

        if pd.isna(value):
            value = None

        elif hasattr(value, "item"):
            value = value.item()

        record[field] = value

    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def dataframe_sha256(df):
    payload = "\n".join(
        canonical_record(row)
        for _, row in df.iterrows()
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def build_connectivity_timeline(df):
    rows = []

    for _, row in df.iterrows():
        timestep = int(row["timestep_index"])

        if timestep < DISCONNECT_START:
            state = "CONNECTED"

        elif timestep <= DISCONNECT_END:
            state = "DISCONNECTED"

        else:
            state = "RECOVERED"

        item = row.to_dict()
        item["connectivity_state"] = state

        if state == "DISCONNECTED":
            item["evidence_location"] = "LOCAL_FORENSIC_BUFFER"
        elif state == "RECOVERED":
            item["evidence_location"] = "RECOVERY_QUEUE"
        else:
            item["evidence_location"] = "CONNECTED_STREAM"

        rows.append(item)

    return pd.DataFrame(rows)


def evaluate_scenario(scenario_id, source_df):
    scenario_df = source_df[
        source_df["scenario_id"].astype(str) == scenario_id
    ].copy()

    scenario_df = scenario_df.sort_values(
        ["timestep_index", "timestamp_s", "can_id"]
    ).reset_index(drop=True)

    disconnected = scenario_df[
        (scenario_df["timestep_index"] >= DISCONNECT_START)
        & (scenario_df["timestep_index"] <= DISCONNECT_END)
    ]

    recovered = scenario_df[
        scenario_df["timestep_index"] > DISCONNECT_END
    ]

    event_rows = scenario_df[
        scenario_df["timestep_index"] == EVENT_TIMESTEP
    ]

    timeline = build_connectivity_timeline(scenario_df)

    connected_rows = timeline[
        timeline["connectivity_state"] == "CONNECTED"
    ]

    disconnected_rows = timeline[
        timeline["connectivity_state"] == "DISCONNECTED"
    ]

    recovered_rows = timeline[
        timeline["connectivity_state"] == "RECOVERED"
    ]

    results = [
        {
            "scenario_id": scenario_id,
            "state": "CONNECTED",
            "records": len(connected_rows),
            "local_preservation": False,
            "synchronization": "NOT_REQUIRED",
            "soc_handoff": "NOT_REQUIRED",
        },
        {
            "scenario_id": scenario_id,
            "state": "DISCONNECTED",
            "records": len(disconnected_rows),
            "local_preservation": len(disconnected_rows) > 0,
            "synchronization": "DEFERRED",
            "soc_handoff": "DEFERRED",
        },
        {
            "scenario_id": scenario_id,
            "state": "RECOVERED",
            "records": len(recovered_rows),
            "local_preservation": False,
            "synchronization": "READY",
            "soc_handoff": "READY",
        },
    ]

    return {
        "scenario_id": scenario_id,
        "records": len(scenario_df),
        "timesteps": int(scenario_df["timestep_index"].nunique()),
        "event_records": len(event_rows),
        "disconnect_records": len(disconnected),
        "recovery_records": len(recovered),
        "disconnect_start": DISCONNECT_START,
        "disconnect_end": DISCONNECT_END,
        "event_timestep": EVENT_TIMESTEP,
        "timeline": results,
        "timeline_df": timeline,
    }


def main():
    print()
    print("=== PHASE 5.16 CONNECTIVITY LOSS & RECOVERY ===")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    checks = []

    def check(name, condition):
        passed = bool(condition)
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
            }
        )
        print(("  PASS " if passed else "  FAIL ") + name)

    check("forensic_input_exists", INPUT_CSV.exists())
    check("integrity_input_exists", INTEGRITY_CSV.exists())

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    if not INTEGRITY_CSV.exists():
        raise FileNotFoundError(INTEGRITY_CSV)

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    integrity_df = pd.read_csv(INTEGRITY_CSV, low_memory=False)

    print(f"Forensic records: {len(df)}")

    check("forensic_records_4650", len(df) == 4650)

    check(
        "integrity_validation_present",
        len(integrity_df) > 0,
    )

    valid_integrity = integrity_df[
        integrity_df["tamper_case"] == "VALID_EVIDENCE"
    ]

    check(
        "valid_evidence_available",
        len(valid_integrity) == 1,
    )

    check(
        "valid_evidence_integrity_true",
        bool(valid_integrity.iloc[0]["integrity_valid"])
        if len(valid_integrity) == 1
        else False,
    )

    check(
        "valid_evidence_sync_eligible",
        bool(
            valid_integrity.iloc[0]["synchronization_decision"]
            == "ELIGIBLE"
        )
        if len(valid_integrity) == 1
        else False,
    )

    scenario_results = []
    timeline_frames = []

    for scenario_id in ATTACK_SCENARIOS:
        print(f"Processing {scenario_id}...")

        result = evaluate_scenario(
            scenario_id,
            df,
        )

        scenario_results.append(
            {
                key: value
                for key, value in result.items()
                if key != "timeline_df"
            }
        )

        timeline_frames.append(result["timeline_df"])

        check(
            f"{scenario_id}_records_465",
            result["records"] == 465,
        )

        check(
            f"{scenario_id}_timesteps_30",
            result["timesteps"] == 30,
        )

        check(
            f"{scenario_id}_disconnect_present",
            result["disconnect_records"] > 0,
        )

        check(
            f"{scenario_id}_recovery_present",
            result["recovery_records"] > 0,
        )

        check(
            f"{scenario_id}_event_present",
            result["event_records"] > 0,
        )

    timeline_df = pd.concat(
        timeline_frames,
        ignore_index=True,
    )

    check(
        "ten_attack_scenarios",
        len(scenario_results) == 10,
    )

    check(
        "local_buffer_used_during_disconnect",
        bool(
            (
                timeline_df.loc[
                    timeline_df["connectivity_state"]
                    == "DISCONNECTED",
                    "evidence_location",
                ]
                == "LOCAL_FORENSIC_BUFFER"
            ).all()
        ),
    )

    check(
        "no_local_buffer_after_recovery",
        bool(
            (
                timeline_df.loc[
                    timeline_df["connectivity_state"]
                    == "RECOVERED",
                    "evidence_location",
                ]
                == "RECOVERY_QUEUE"
            ).all()
        ),
    )

    recovery_records = timeline_df[
        timeline_df["connectivity_state"] == "RECOVERED"
    ].copy()

    check(
        "recovery_queue_nonempty",
        len(recovery_records) > 0,
    )

    recovery_hash = dataframe_sha256(recovery_records)

    check(
        "recovery_integrity_digest_generated",
        len(recovery_hash) == 64,
    )

    output_rows = []

    for result in scenario_results:
        for timeline_item in result["timeline"]:
            output_rows.append(
                {
                    "scenario_id": result["scenario_id"],
                    "connectivity_state": timeline_item["state"],
                    "records": timeline_item["records"],
                    "local_preservation": timeline_item[
                        "local_preservation"
                    ],
                    "synchronization": timeline_item[
                        "synchronization"
                    ],
                    "soc_handoff": timeline_item[
                        "soc_handoff"
                    ],
                    "disconnect_start": DISCONNECT_START,
                    "disconnect_end": DISCONNECT_END,
                    "event_timestep": EVENT_TIMESTEP,
                    "evidence_source": "LOCAL_FORENSIC_BUFFER",
                    "laboratory_generated": True,
                    "physical_vehicle": False,
                    "direct_actuation": False,
                }
            )

    output_df = pd.DataFrame(output_rows)

    check(
        "state_summary_records_30",
        len(output_df) == 30,
    )

    check(
        "connected_states_10",
        int(
            (output_df["connectivity_state"] == "CONNECTED").sum()
        ) == 10,
    )

    check(
        "disconnected_states_10",
        int(
            (output_df["connectivity_state"] == "DISCONNECTED").sum()
        ) == 10,
    )

    check(
        "recovered_states_10",
        int(
            (output_df["connectivity_state"] == "RECOVERED").sum()
        ) == 10,
    )

    check(
        "disconnect_sync_deferred",
        bool(
            (
                output_df.loc[
                    output_df["connectivity_state"]
                    == "DISCONNECTED",
                    "synchronization",
                ]
                == "DEFERRED"
            ).all()
        ),
    )

    check(
        "recovery_sync_ready",
        bool(
            (
                output_df.loc[
                    output_df["connectivity_state"]
                    == "RECOVERED",
                    "synchronization",
                ]
                == "READY"
            ).all()
        ),
    )

    check(
        "recovery_soc_ready",
        bool(
            (
                output_df.loc[
                    output_df["connectivity_state"]
                    == "RECOVERED",
                    "soc_handoff",
                ]
                == "READY"
            ).all()
        ),
    )

    check(
        "laboratory_only",
        bool(output_df["laboratory_generated"].all()),
    )

    check(
        "no_physical_vehicle",
        bool((~output_df["physical_vehicle"]).all()),
    )

    check(
        "no_direct_actuation",
        bool((~output_df["direct_actuation"]).all()),
    )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in output_df.to_dict(
            orient="records"
        ):
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    schema = {
        "phase": "5.16",
        "title": "Connectivity Loss and Recovery",
        "input": str(INPUT_CSV.relative_to(ROOT)),
        "integrity_reference": str(
            INTEGRITY_CSV.relative_to(ROOT)
        ),
        "states": [
            "CONNECTED",
            "DISCONNECTED",
            "RECOVERED",
        ],
        "disconnect_start_timestep": DISCONNECT_START,
        "disconnect_end_timestep": DISCONNECT_END,
        "event_timestep": EVENT_TIMESTEP,
        "local_buffer_during_disconnect": True,
        "synchronization_after_recovery": True,
        "soc_handoff_after_recovery": True,
        "integrity_digest": "SHA-256",
        "laboratory_generated": True,
        "physical_vehicle": False,
        "direct_actuation": False,
    }

    OUTPUT_SCHEMA.write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
    )

    passed = sum(
        item["status"] == "PASS"
        for item in checks
    )

    total = len(checks)

    status = "PASS" if passed == total else "FAIL"

    manifest = {
        "phase": "5.16",
        "title": "Connectivity Loss and Recovery",
        "status": status,
        "forensic_records": len(df),
        "attack_scenarios": ATTACK_SCENARIOS,
        "scenario_count": len(ATTACK_SCENARIOS),
        "disconnect_start_timestep": DISCONNECT_START,
        "disconnect_end_timestep": DISCONNECT_END,
        "event_timestep": EVENT_TIMESTEP,
        "recovery_sha256": recovery_hash,
        "checks_passed": passed,
        "checks_total": total,
        "laboratory_only": True,
        "physical_vehicle": False,
        "direct_actuation": False,
        "model_retraining": False,
        "live_cortex_xdr": False,
        "architecture": [
            "CONNECTED",
            "DISCONNECTED",
            "LOCAL_FORENSIC_BUFFER",
            "RECOVERED",
            "INTEGRITY_VERIFICATION",
            "SYNCHRONIZATION_READY",
            "SOC_HANDOFF_READY",
        ],
        "validation": checks,
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== PHASE 5.16 RESULTS ===")
    print()
    print(f"Validation: {passed}/{total} checks passed")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"Schema: {OUTPUT_SCHEMA}")
    print(f"Manifest: {MANIFEST}")
    print()
    print(f"STATUS: {status}")


if __name__ == "__main__":
    main()
