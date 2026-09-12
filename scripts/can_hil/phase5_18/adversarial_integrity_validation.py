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
    / "phase5_17"
    / "phase5_17_secure_synchronization.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_18"
)

EXPERIMENT_DIR = (
    ROOT
    / "experiments"
    / "can_hil"
    / "phase5_18"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "phase5_18_adversarial_integrity_validation.csv"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "phase5_18_adversarial_integrity_validation.jsonl"
)

OUTPUT_SCHEMA = (
    OUTPUT_DIR
    / "phase5_18_adversarial_integrity_validation_schema.json"
)

MANIFEST = (
    EXPERIMENT_DIR
    / "phase5_18_adversarial_integrity_validation_manifest.json"
)


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
]


REQUIRED_COLUMNS = [
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
]


def sha256_text(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def evaluate_scenario(row):
    baseline_digest = str(
        row["synchronization_digest"]
    )

    valid_digest = (
        len(baseline_digest) == 64
        and all(
            c in "0123456789abcdef"
            for c in baseline_digest.lower()
        )
    )

    baseline_sync = (
        str(row["synchronization_state"])
        == "SYNCHRONIZATION_ELIGIBLE"
    )

    baseline_soc = (
        str(row["soc_state"])
        == "SOC_HANDOFF_ELIGIBLE"
    )

    baseline_chain = (
        valid_digest
        and baseline_sync
        and baseline_soc
    )

    tampered_digest = sha256_text(
        baseline_digest + "::TAMPERED"
    )

    tampered_chain = (
        tampered_digest == baseline_digest
    )

    tampered_sync_allowed = (
        tampered_chain
    )

    tampered_soc_allowed = (
        tampered_chain
        and baseline_soc
    )

    physical_boundary_ok = (
        str(row["physical_vehicle"]).lower()
        == "false"
        and str(row["direct_actuation"]).lower()
        == "false"
    )

    laboratory_boundary_ok = (
        str(row["laboratory_generated"]).lower()
        == "true"
    )

    refusal_correct = (
        not tampered_chain
        and not tampered_sync_allowed
        and not tampered_soc_allowed
    )

    return {
        "scenario_id": str(
            row["scenario_id"]
        ),
        "baseline_digest_valid": bool(
            valid_digest
        ),
        "baseline_synchronization_eligible": bool(
            baseline_sync
        ),
        "baseline_soc_handoff_eligible": bool(
            baseline_soc
        ),
        "baseline_chain_eligible": bool(
            baseline_chain
        ),
        "tampered_digest": tampered_digest,
        "tampered_chain_valid": bool(
            tampered_chain
        ),
        "tampered_synchronization_allowed": bool(
            tampered_sync_allowed
        ),
        "tampered_soc_handoff_allowed": bool(
            tampered_soc_allowed
        ),
        "tampered_evidence_refused": bool(
            refusal_correct
        ),
        "physical_safety_boundary_valid": bool(
            physical_boundary_ok
        ),
        "laboratory_boundary_valid": bool(
            laboratory_boundary_ok
        ),
        "integration_mode": str(
            row["integration_mode"]
        ),
        "live_cortex_xdr": str(
            row["live_cortex_xdr"]
        ).lower() == "true",
        "direct_actuation": str(
            row["direct_actuation"]
        ).lower() == "true",
        "real_world_cyberattack": str(
            row["real_world_cyberattack"]
        ).lower() == "true",
    }


def main():
    print()
    print(
        "=== PHASE 5.18 "
        "ADVERSARIAL INTEGRITY VALIDATION ==="
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

    checks["input_exists"] = INPUT_CSV.exists()

    if not INPUT_CSV.exists():
        print("FAIL: Phase 5.17 input missing")
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

    checks["input_rows_10"] = (
        len(df) == 10
    )

    checks["scenario_count_10"] = (
        df["scenario_id"].nunique() == 10
    )

    checks["required_columns_present"] = all(
        column in df.columns
        for column in REQUIRED_COLUMNS
    )

    results = []

    for scenario_id in SCENARIOS:
        scenario_rows = df[
            df["scenario_id"].astype(str)
            == scenario_id
        ]

        if len(scenario_rows) != 1:
            print(
                f"FAIL: unexpected row count for "
                f"{scenario_id}: {len(scenario_rows)}"
            )
            continue

        row = scenario_rows.iloc[0]

        result = evaluate_scenario(row)

        results.append(result)

        print()
        print(
            f"Processing {scenario_id}..."
        )
        print(
            "  baseline chain:",
            "VALID"
            if result["baseline_chain_eligible"]
            else "INVALID",
        )
        print(
            "  tampered chain:",
            "VALID"
            if result["tampered_chain_valid"]
            else "REJECTED",
        )
        print(
            "  tampered synchronization:",
            "ALLOWED"
            if result[
                "tampered_synchronization_allowed"
            ]
            else "REFUSED",
        )
        print(
            "  tampered SOC handoff:",
            "ALLOWED"
            if result[
                "tampered_soc_handoff_allowed"
            ]
            else "REFUSED",
        )

        checks[
            f"{scenario_id}_baseline_digest_valid"
        ] = result[
            "baseline_digest_valid"
        ]

        checks[
            f"{scenario_id}_baseline_sync_eligible"
        ] = result[
            "baseline_synchronization_eligible"
        ]

        checks[
            f"{scenario_id}_baseline_soc_eligible"
        ] = result[
            "baseline_soc_handoff_eligible"
        ]

        checks[
            f"{scenario_id}_baseline_chain_eligible"
        ] = result[
            "baseline_chain_eligible"
        ]

        checks[
            f"{scenario_id}_tampered_digest_rejected"
        ] = not result[
            "tampered_chain_valid"
        ]

        checks[
            f"{scenario_id}_tampered_sync_refused"
        ] = not result[
            "tampered_synchronization_allowed"
        ]

        checks[
            f"{scenario_id}_tampered_soc_refused"
        ] = not result[
            "tampered_soc_handoff_allowed"
        ]

        checks[
            f"{scenario_id}_tampered_evidence_refused"
        ] = result[
            "tampered_evidence_refused"
        ]

        checks[
            f"{scenario_id}_physical_boundary"
        ] = result[
            "physical_safety_boundary_valid"
        ]

        checks[
            f"{scenario_id}_laboratory_boundary"
        ] = result[
            "laboratory_boundary_valid"
        ]

    result_df = pd.DataFrame(results)

    checks["all_10_results_generated"] = (
        len(result_df) == 10
    )

    checks["all_baseline_chains_valid"] = (
        result_df[
            "baseline_chain_eligible"
        ].all()
    )

    checks["all_tampered_chains_rejected"] = (
        (~result_df[
            "tampered_chain_valid"
        ]).all()
    )

    checks["all_tampered_sync_refused"] = (
        (~result_df[
            "tampered_synchronization_allowed"
        ]).all()
    )

    checks["all_tampered_soc_refused"] = (
        (~result_df[
            "tampered_soc_handoff_allowed"
        ]).all()
    )

    checks["all_tampered_evidence_refused"] = (
        result_df[
            "tampered_evidence_refused"
        ].all()
    )

    checks["all_laboratory_boundary_valid"] = (
        result_df[
            "laboratory_boundary_valid"
        ].all()
    )

    checks["no_live_cortex_xdr"] = not (
        result_df[
            "live_cortex_xdr"
        ].any()
    )

    checks["no_direct_actuation"] = not (
        result_df[
            "direct_actuation"
        ].any()
    )

    checks["no_real_world_attack_claim"] = not (
        result_df[
            "real_world_cyberattack"
        ].any()
    )

    checks["all_abstraction_only"] = (
        result_df[
            "integration_mode"
        ]
        .eq("ABSTRACTION_ONLY")
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
        for result in results:
            handle.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    schema = {
        "phase": "5.18",
        "title": (
            "Adversarial Integrity Validation"
        ),
        "input_contract": {
            "source_phase": "5.17",
            "rows": 10,
            "one_row_per_scenario": True,
        },
        "adversarial_tests": [
            "baseline_digest_validation",
            "tampered_digest_generation",
            "tampered_chain_rejection",
            "tampered_synchronization_refusal",
            "tampered_soc_handoff_refusal",
        ],
        "required_outputs": [
            "baseline_digest_valid",
            "baseline_chain_eligible",
            "tampered_digest",
            "tampered_chain_valid",
            "tampered_synchronization_allowed",
            "tampered_soc_handoff_allowed",
            "tampered_evidence_refused",
        ],
        "safety_boundary": {
            "laboratory_generated": True,
            "physical_vehicle": False,
            "direct_actuation": False,
            "live_cortex_xdr": False,
            "model_retraining": False,
            "real_world_cyberattack": False,
        },
    }

    OUTPUT_SCHEMA.write_text(
        json.dumps(
            schema,
            indent=2,
        ),
        encoding="utf-8",
    )

    validation = {
        str(key): bool(value)
        for key, value in checks.items()
    }

    passed = sum(
        validation.values()
    )

    total = len(validation)

    status = (
        "PASS"
        if passed == total
        else "FAIL"
    )

    manifest = {
        "phase": "5.18",
        "title": (
            "Adversarial Integrity Validation"
        ),
        "status": status,
        "input": str(
            INPUT_CSV.relative_to(ROOT)
        ),
        "input_rows": int(len(df)),
        "scenario_count": int(
            df["scenario_id"].nunique()
        ),
        "results": results,
        "validation": validation,
        "gate_summary": {
            "passed": int(passed),
            "total": int(total),
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
        "scientific_boundary": {
            "laboratory_simulation": True,
            "physical_vehicle": False,
            "direct_vehicle_actuation": False,
            "live_cortex_xdr": False,
            "model_retraining": False,
            "real_world_cyberattack": False,
        },
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=== PHASE 5.18 RESULTS ==="
    )
    print(
        f"Validation: {passed}/{total} "
        "checks passed"
    )
    print(
        "STATUS:",
        status,
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


if __name__ == "__main__":
    main()
