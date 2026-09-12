from pathlib import Path
import hashlib
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_16"
    / "phase5_16_connectivity_recovery.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_17"
)

EXPERIMENT_DIR = (
    ROOT
    / "experiments"
    / "can_hil"
    / "phase5_17"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "phase5_17_secure_synchronization.csv"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "phase5_17_secure_synchronization.jsonl"
)

OUTPUT_SCHEMA = (
    OUTPUT_DIR
    / "phase5_17_secure_synchronization_schema.json"
)

MANIFEST = (
    EXPERIMENT_DIR
    / "phase5_17_secure_synchronization_manifest.json"
)


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


REQUIRED_COLUMNS = [
    "scenario_id",
    "connectivity_state",
    "records",
    "local_preservation",
    "synchronization",
    "soc_handoff",
    "disconnect_start",
    "disconnect_end",
    "event_timestep",
    "evidence_source",
    "laboratory_generated",
    "physical_vehicle",
    "direct_actuation",
]


def digest_record(record):
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def build_scenario_result(
    scenario_id,
    scenario_df,
):
    scenario_df = scenario_df.copy()

    states = set(
        scenario_df[
            "connectivity_state"
        ].astype(str)
    )

    expected_states = {
        "CONNECTED",
        "DISCONNECTED",
        "RECOVERED",
    }

    state_coverage_ok = (
        states == expected_states
    )

    connected = scenario_df[
        scenario_df[
            "connectivity_state"
        ].astype(str)
        == "CONNECTED"
    ]

    disconnected = scenario_df[
        scenario_df[
            "connectivity_state"
        ].astype(str)
        == "DISCONNECTED"
    ]

    recovered = scenario_df[
        scenario_df[
            "connectivity_state"
        ].astype(str)
        == "RECOVERED"
    ]

    synchronization_values = set(
        scenario_df[
            "synchronization"
        ].astype(str)
    )

    soc_values = set(
        scenario_df[
            "soc_handoff"
        ].astype(str)
    )

    recovery_sync_present = (
        "RECOVERED" in states
        and (
            "READY"
            in synchronization_values
            or "NOT_REQUIRED"
            in synchronization_values
        )
    )

    recovered_rows_ready = True

    if len(recovered) > 0:
        recovered_rows_ready = bool(
            recovered[
                "synchronization"
            ]
            .astype(str)
            .isin(
                [
                    "READY",
                    "SYNCHRONIZED",
                    "NOT_REQUIRED",
                ]
            )
            .all()
        )

    soc_handoff_present = bool(
        len(recovered) > 0
        and (
            "READY" in soc_values
            or "ALLOWED" in soc_values
            or "SOC_HANDOFF_READY"
            in soc_values
            or "NOT_REQUIRED" in soc_values
        )
    )

    if state_coverage_ok and recovered_rows_ready:
        synchronization_state = (
            "SYNCHRONIZATION_ELIGIBLE"
        )
    else:
        synchronization_state = (
            "SYNCHRONIZATION_BLOCKED"
        )

    if (
        synchronization_state
        == "SYNCHRONIZATION_ELIGIBLE"
        and soc_handoff_present
    ):
        soc_state = "SOC_HANDOFF_ELIGIBLE"
    else:
        soc_state = "SOC_HANDOFF_BLOCKED"

    source_record = {
        "scenario_id": scenario_id,
        "states": sorted(states),
        "records": int(
            scenario_df["records"]
            .iloc[0]
        ),
        "connected_rows": int(
            len(connected)
        ),
        "disconnected_rows": int(
            len(disconnected)
        ),
        "recovered_rows": int(
            len(recovered)
        ),
        "synchronization_values": sorted(
            synchronization_values
        ),
        "soc_handoff_values": sorted(
            soc_values
        ),
    }

    synchronization_digest = digest_record(
        source_record
    )

    return {
        "scenario_id": scenario_id,
        "source_records": int(
            scenario_df["records"]
            .iloc[0]
        ),
        "state_rows": int(
            len(scenario_df)
        ),
        "connectivity_states": "|".join(
            sorted(states)
        ),
        "connected_state_present": bool(
            len(connected) > 0
        ),
        "disconnected_state_present": bool(
            len(disconnected) > 0
        ),
        "recovered_state_present": bool(
            len(recovered) > 0
        ),
        "state_coverage_valid": bool(
            state_coverage_ok
        ),
        "recovery_synchronization_present": bool(
            recovery_sync_present
        ),
        "recovered_rows_ready": bool(
            recovered_rows_ready
        ),
        "soc_handoff_present": bool(
            soc_handoff_present
        ),
        "synchronization_state": (
            synchronization_state
        ),
        "soc_state": soc_state,
        "synchronization_digest": (
            synchronization_digest
        ),
        "integration_mode": (
            "ABSTRACTION_ONLY"
        ),
        "laboratory_generated": bool(
            scenario_df[
                "laboratory_generated"
            ].astype(str).str.lower().eq(
                "true"
            ).all()
        ),
        "physical_vehicle": bool(
            scenario_df[
                "physical_vehicle"
            ].astype(str).str.lower().eq(
                "true"
            ).any()
        ),
        "direct_actuation": bool(
            scenario_df[
                "direct_actuation"
            ].astype(str).str.lower().eq(
                "true"
            ).any()
        ),
        "model_retraining": False,
        "live_cortex_xdr": False,
        "real_world_cyberattack": False,
    }


def main():
    print()
    print(
        "=== PHASE 5.17 SECURE SYNCHRONIZATION "
        "& SOC HANDOFF ==="
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checks = {}
    results = []

    checks["input_exists"] = INPUT_CSV.exists()

    if not INPUT_CSV.exists():
        print(
            "FAIL: Phase 5.16 input does not exist:"
        )
        print(INPUT_CSV)
        return

    df = pd.read_csv(INPUT_CSV)

    print(
        f"Input rows: {len(df)}"
    )

    print(
        "Input columns:",
        ", ".join(df.columns),
    )

    checks["input_rows_30"] = (
        len(df) == 30
    )

    checks["scenario_count_10"] = (
        df["scenario_id"].nunique() == 10
    )

    checks["required_columns_present"] = all(
        column in df.columns
        for column in REQUIRED_COLUMNS
    )

    if not checks["required_columns_present"]:
        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]
        print(
            "Missing columns:",
            ", ".join(missing),
        )

    for scenario_id in ATTACK_SCENARIOS:
        print()
        print(
            f"Processing {scenario_id}..."
        )

        scenario_df = df[
            df["scenario_id"].astype(str)
            == scenario_id
        ].copy()

        result = build_scenario_result(
            scenario_id,
            scenario_df,
        )

        results.append(result)

        print(
            "  state rows:",
            result["state_rows"],
        )

        print(
            "  states:",
            result["connectivity_states"],
        )

        print(
            "  synchronization:",
            result["synchronization_state"],
        )

        print(
            "  SOC:",
            result["soc_state"],
        )

        checks[
            f"{scenario_id}_state_rows_3"
        ] = result["state_rows"] == 3

        checks[
            f"{scenario_id}_source_records_positive"
        ] = result[
            "source_records"
        ] > 0

        checks[
            f"{scenario_id}_connected_present"
        ] = result[
            "connected_state_present"
        ]

        checks[
            f"{scenario_id}_disconnected_present"
        ] = result[
            "disconnected_state_present"
        ]

        checks[
            f"{scenario_id}_recovered_present"
        ] = result[
            "recovered_state_present"
        ]

        checks[
            f"{scenario_id}_state_coverage_valid"
        ] = result[
            "state_coverage_valid"
        ]

        checks[
            f"{scenario_id}_recovery_sync_present"
        ] = result[
            "recovery_synchronization_present"
        ]

        checks[
            f"{scenario_id}_recovered_rows_ready"
        ] = result[
            "recovered_rows_ready"
        ]

        checks[
            f"{scenario_id}_soc_handoff_present"
        ] = result[
            "soc_handoff_present"
        ]

        checks[
            f"{scenario_id}_sync_eligible"
        ] = (
            result["synchronization_state"]
            == "SYNCHRONIZATION_ELIGIBLE"
        )

        checks[
            f"{scenario_id}_soc_eligible"
        ] = (
            result["soc_state"]
            == "SOC_HANDOFF_ELIGIBLE"
        )

        checks[
            f"{scenario_id}_digest_64"
        ] = len(
            result["synchronization_digest"]
        ) == 64

    result_df = pd.DataFrame(results)

    checks["all_scenarios_present"] = (
        set(result_df["scenario_id"])
        == set(ATTACK_SCENARIOS)
    )

    checks["exactly_10_scenario_results"] = (
        len(result_df) == 10
    )

    checks["all_digests_unique"] = (
        result_df[
            "synchronization_digest"
        ].nunique()
        == 10
    )

    checks["all_abstraction_only"] = (
        result_df[
            "integration_mode"
        ].eq("ABSTRACTION_ONLY")
        .all()
    )

    checks["all_laboratory_generated"] = (
        result_df[
            "laboratory_generated"
        ].all()
    )

    checks["no_physical_vehicle"] = not bool(
        result_df[
            "physical_vehicle"
        ].any()
    )

    checks["no_direct_actuation"] = not bool(
        result_df[
            "direct_actuation"
        ].any()
    )

    checks["no_model_retraining"] = not bool(
        result_df[
            "model_retraining"
        ].any()
    )

    checks["no_live_cortex_xdr"] = not bool(
        result_df[
            "live_cortex_xdr"
        ].any()
    )

    checks["no_real_world_attack_claim"] = not bool(
        result_df[
            "real_world_cyberattack"
        ].any()
    )

    checks["all_sync_eligible"] = (
        result_df[
            "synchronization_state"
        ]
        .eq("SYNCHRONIZATION_ELIGIBLE")
        .all()
    )

    checks["all_soc_handoff_eligible"] = (
        result_df[
            "soc_state"
        ]
        .eq("SOC_HANDOFF_ELIGIBLE")
        .all()
    )

    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in results:
            handle.write(
                json.dumps(
                    record,
                    separators=(",", ":"),
                )
                + "\n"
            )

    schema = {
        "phase": "5.17",
        "title": (
            "Secure Synchronization and "
            "SOC Handoff"
        ),
        "input": str(
            INPUT_CSV.relative_to(ROOT)
        ),
        "output_csv": str(
            OUTPUT_CSV.relative_to(ROOT)
        ),
        "output_jsonl": str(
            OUTPUT_JSONL.relative_to(ROOT)
        ),
        "source_contract": {
            "input_rows": 30,
            "rows_per_scenario": 3,
            "scenarios": 10,
            "states": [
                "CONNECTED",
                "DISCONNECTED",
                "RECOVERED",
            ],
        },
        "decision_states": [
            "SYNCHRONIZATION_ELIGIBLE",
            "SYNCHRONIZATION_BLOCKED",
            "SOC_HANDOFF_ELIGIBLE",
            "SOC_HANDOFF_BLOCKED",
        ],
        "required_properties": [
            "scenario_id",
            "source_records",
            "state_rows",
            "connectivity_states",
            "state_coverage_valid",
            "recovery_synchronization_present",
            "synchronization_state",
            "soc_state",
            "synchronization_digest",
            "integration_mode",
            "laboratory_generated",
            "physical_vehicle",
            "direct_actuation",
            "model_retraining",
            "live_cortex_xdr",
            "real_world_cyberattack",
        ],
    }

    OUTPUT_SCHEMA.write_text(
        json.dumps(
            schema,
            indent=2,
        ),
        encoding="utf-8",
    )

    overall_pass = all(
        bool(value)
        for value in checks.values()
    )

    passed = sum(
        bool(value)
        for value in checks.values()
    )

    total = len(checks)

    manifest = {
        "phase": "5.17",
        "title": (
            "Secure Synchronization and "
            "SOC Handoff"
        ),
        "status": (
            "PASS"
            if bool(overall_pass)
            else "FAIL"
        ),
        "input": str(
            INPUT_CSV.relative_to(ROOT)
        ),
        "input_rows": int(len(df)),
        "scenario_count": int(
            df["scenario_id"].nunique()
        ),
        "results": results,
        "validation": {
            str(key): bool(value)
            for key, value in checks.items()
        },
        "outputs": {
            "csv": str(
                OUTPUT_CSV.relative_to(ROOT)
            ),
            "jsonl": str(
                OUTPUT_JSONL.relative_to(ROOT)
            ),
            "schema": str(
                OUTPUT_SCHEMA.relative_to(ROOT)
            ),
            "manifest": str(
                MANIFEST.relative_to(ROOT)
            ),
        },
        "research_boundary": {
            "laboratory_generated": True,
            "physical_vehicle": False,
            "direct_actuation": False,
            "model_retraining": False,
            "live_cortex_xdr": False,
            "real_world_cyberattack": False,
            "integration_mode": (
                "ABSTRACTION_ONLY"
            ),
        },
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=== PHASE 5.17 RESULTS ==="
    )

    print(
        f"Validation: {passed}/{total} "
        "checks passed"
    )

    print(
        "CSV:",
        OUTPUT_CSV,
    )

    print(
        "JSONL:",
        OUTPUT_JSONL,
    )

    print(
        "Schema:",
        OUTPUT_SCHEMA,
    )

    print(
        "Manifest:",
        MANIFEST,
    )

    print()
    print(
        "STATUS:",
        "PASS" if overall_pass else "FAIL",
    )


if __name__ == "__main__":
    main()
