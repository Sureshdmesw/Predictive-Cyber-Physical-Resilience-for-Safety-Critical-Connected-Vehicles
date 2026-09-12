from pathlib import Path
import hashlib
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

INPUT_17 = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_17"
    / "phase5_17_secure_synchronization.csv"
)

INPUT_18 = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_18"
    / "phase5_18_adversarial_integrity_validation.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_19"
)

EXPERIMENT_DIR = (
    ROOT
    / "experiments"
    / "can_hil"
    / "phase5_19"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "phase5_19_end_to_end_resilience_validation.csv"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "phase5_19_end_to_end_resilience_validation.jsonl"
)

OUTPUT_SCHEMA = (
    OUTPUT_DIR
    / "phase5_19_end_to_end_resilience_validation_schema.json"
)

MANIFEST = (
    EXPERIMENT_DIR
    / "phase5_19_end_to_end_resilience_validation_manifest.json"
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


def sha256_text(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def evaluate(row17, row18):
    states = set(
        str(row17["connectivity_states"]).split("|")
    )

    recovery_ok = (
        states
        == {
            "CONNECTED",
            "DISCONNECTED",
            "RECOVERED",
        }
    )

    synchronization_ok = (
        str(row17["synchronization_state"])
        == "SYNCHRONIZATION_ELIGIBLE"
    )

    soc_ok = (
        str(row17["soc_state"])
        == "SOC_HANDOFF_ELIGIBLE"
    )

    phase18_baseline_ok = (
        str(row18["baseline_chain_eligible"]).lower()
        == "true"
    )

    tamper_rejected = (
        str(row18["tampered_chain_valid"]).lower()
        == "false"
    )

    tamper_sync_refused = (
        str(
            row18[
                "tampered_synchronization_allowed"
            ]
        ).lower()
        == "false"
    )

    tamper_soc_refused = (
        str(
            row18[
                "tampered_soc_handoff_allowed"
            ]
        ).lower()
        == "false"
    )

    safety_boundary_ok = (
        str(row17["physical_vehicle"]).lower()
        == "false"
        and str(row17["direct_actuation"]).lower()
        == "false"
        and str(row17["model_retraining"]).lower()
        == "false"
        and str(row17["live_cortex_xdr"]).lower()
        == "false"
        and str(
            row17["real_world_cyberattack"]
        ).lower()
        == "false"
    )

    end_to_end_valid = all(
        [
            recovery_ok,
            synchronization_ok,
            soc_ok,
            phase18_baseline_ok,
            tamper_rejected,
            tamper_sync_refused,
            tamper_soc_refused,
            safety_boundary_ok,
        ]
    )

    chain_text = (
        "CONNECTED"
        " -> DISCONNECTED"
        " -> RECOVERED"
        " -> SYNCHRONIZATION_ELIGIBLE"
        " -> SOC_HANDOFF_ELIGIBLE"
    )

    evidence_digest = sha256_text(
        str(row17["synchronization_digest"])
        + str(row18["tampered_digest"])
        + chain_text
    )

    return {
        "scenario_id": str(
            row17["scenario_id"]
        ),
        "connectivity_recovery_valid": bool(
            recovery_ok
        ),
        "synchronization_eligible": bool(
            synchronization_ok
        ),
        "soc_handoff_eligible": bool(
            soc_ok
        ),
        "phase5_18_baseline_integrity_valid": bool(
            phase18_baseline_ok
        ),
        "tampered_evidence_rejected": bool(
            tamper_rejected
        ),
        "tampered_synchronization_refused": bool(
            tamper_sync_refused
        ),
        "tampered_soc_handoff_refused": bool(
            tamper_soc_refused
        ),
        "safety_boundary_valid": bool(
            safety_boundary_ok
        ),
        "end_to_end_chain": chain_text,
        "end_to_end_valid": bool(
            end_to_end_valid
        ),
        "evidence_digest": evidence_digest,
        "laboratory_simulation": True,
        "physical_vehicle": False,
        "direct_vehicle_actuation": False,
        "live_cortex_xdr": False,
        "model_retraining": False,
        "real_world_cyberattack": False,
    }


def main():
    print()
    print(
        "=== PHASE 5.19 "
        "END-TO-END RESILIENCE VALIDATION ==="
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

    checks["phase5_17_exists"] = INPUT_17.exists()
    checks["phase5_18_exists"] = INPUT_18.exists()

    if not INPUT_17.exists():
        print("FAIL: Phase 5.17 input missing")
        print(INPUT_17)
        return

    if not INPUT_18.exists():
        print("FAIL: Phase 5.18 input missing")
        print(INPUT_18)
        return

    df17 = pd.read_csv(INPUT_17)
    df18 = pd.read_csv(INPUT_18)

    print(
        f"Phase 5.17 rows: {len(df17)}"
    )
    print(
        f"Phase 5.18 rows: {len(df18)}"
    )

    checks["phase5_17_rows_10"] = (
        len(df17) == 10
    )

    checks["phase5_18_rows_10"] = (
        len(df18) == 10
    )

    checks["phase5_17_scenarios_10"] = (
        df17["scenario_id"].nunique()
        == 10
    )

    checks["phase5_18_scenarios_10"] = (
        df18["scenario_id"].nunique()
        == 10
    )

    results = []

    for scenario_id in SCENARIOS:
        rows17 = df17[
            df17["scenario_id"].astype(str)
            == scenario_id
        ]

        rows18 = df18[
            df18["scenario_id"].astype(str)
            == scenario_id
        ]

        print()
        print(
            f"Processing {scenario_id}..."
        )

        if len(rows17) != 1 or len(rows18) != 1:
            print(
                "  INVALID: missing scenario source row"
            )
            checks[
                f"{scenario_id}_source_rows_valid"
            ] = False
            continue

        result = evaluate(
            rows17.iloc[0],
            rows18.iloc[0],
        )

        results.append(result)

        print(
            "  recovery:",
            "PASS"
            if result[
                "connectivity_recovery_valid"
            ]
            else "FAIL",
        )

        print(
            "  synchronization:",
            "ALLOW"
            if result[
                "synchronization_eligible"
            ]
            else "REFUSE",
        )

        print(
            "  SOC handoff:",
            "ALLOW"
            if result[
                "soc_handoff_eligible"
            ]
            else "REFUSE",
        )

        print(
            "  tampered evidence:",
            "REJECTED"
            if result[
                "tampered_evidence_rejected"
            ]
            else "ACCEPTED",
        )

        print(
            "  end-to-end:",
            "PASS"
            if result["end_to_end_valid"]
            else "FAIL",
        )

        checks[
            f"{scenario_id}_source_rows_valid"
        ] = True

        checks[
            f"{scenario_id}_recovery_valid"
        ] = result[
            "connectivity_recovery_valid"
        ]

        checks[
            f"{scenario_id}_sync_valid"
        ] = result[
            "synchronization_eligible"
        ]

        checks[
            f"{scenario_id}_soc_valid"
        ] = result[
            "soc_handoff_eligible"
        ]

        checks[
            f"{scenario_id}_baseline_integrity_valid"
        ] = result[
            "phase5_18_baseline_integrity_valid"
        ]

        checks[
            f"{scenario_id}_tamper_rejected"
        ] = result[
            "tampered_evidence_rejected"
        ]

        checks[
            f"{scenario_id}_tampered_sync_refused"
        ] = result[
            "tampered_synchronization_refused"
        ]

        checks[
            f"{scenario_id}_tampered_soc_refused"
        ] = result[
            "tampered_soc_handoff_refused"
        ]

        checks[
            f"{scenario_id}_safety_boundary"
        ] = result[
            "safety_boundary_valid"
        ]

        checks[
            f"{scenario_id}_end_to_end_valid"
        ] = result[
            "end_to_end_valid"
        ]

        checks[
            f"{scenario_id}_digest_64"
        ] = len(
            result["evidence_digest"]
        ) == 64

    result_df = pd.DataFrame(results)

    checks["exactly_10_results"] = (
        len(result_df) == 10
    )

    checks["all_end_to_end_valid"] = (
        result_df[
            "end_to_end_valid"
        ].all()
    )

    checks["all_tampered_evidence_rejected"] = (
        result_df[
            "tampered_evidence_rejected"
        ].all()
    )

    checks["all_tampered_sync_refused"] = (
        result_df[
            "tampered_synchronization_refused"
        ].all()
    )

    checks["all_tampered_soc_refused"] = (
        result_df[
            "tampered_soc_handoff_refused"
        ].all()
    )

    checks["all_safety_boundaries_valid"] = (
        result_df[
            "safety_boundary_valid"
        ].all()
    )

    checks["all_digests_valid"] = (
        result_df[
            "evidence_digest"
        ].astype(str)
        .str.len()
        .eq(64)
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
        "phase": "5.19",
        "title": (
            "End-to-End CAN-HIL "
            "Resilience Validation"
        ),
        "inputs": {
            "phase5_17": str(
                INPUT_17.relative_to(ROOT)
            ),
            "phase5_18": str(
                INPUT_18.relative_to(ROOT)
            ),
        },
        "validated_chain": [
            "CONNECTED",
            "DISCONNECTED",
            "RECOVERED",
            "SYNCHRONIZATION_ELIGIBLE",
            "SOC_HANDOFF_ELIGIBLE",
        ],
        "adversarial_path": [
            "TAMPERED_EVIDENCE",
            "REJECTED",
            "SYNCHRONIZATION_REFUSED",
            "SOC_HANDOFF_REFUSED",
        ],
        "safety_boundary": {
            "laboratory_simulation": True,
            "physical_vehicle": False,
            "direct_vehicle_actuation": False,
            "live_cortex_xdr": False,
            "model_retraining": False,
            "real_world_cyberattack": False,
        },
    }

    OUTPUT_SCHEMA.write_text(
        json.dumps(
            schema,
            indent=2,
            ensure_ascii=False,
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
        "phase": "5.19",
        "title": (
            "End-to-End CAN-HIL "
            "Resilience Validation"
        ),
        "status": status,
        "input_rows": {
            "phase5_17": int(len(df17)),
            "phase5_18": int(len(df18)),
        },
        "scenario_count": 10,
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
        "=== PHASE 5.19 RESULTS ==="
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
