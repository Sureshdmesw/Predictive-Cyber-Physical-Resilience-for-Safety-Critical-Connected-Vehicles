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
    / "phase5_19"
    / "phase5_19_end_to_end_resilience_validation.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_20"
)

EXPERIMENT_DIR = (
    ROOT
    / "experiments"
    / "can_hil"
    / "phase5_20"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "phase5_20_resilience_scorecard.csv"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "phase5_20_resilience_scorecard.jsonl"
)

OUTPUT_SCHEMA = (
    OUTPUT_DIR
    / "phase5_20_resilience_scorecard_schema.json"
)

MANIFEST = (
    EXPERIMENT_DIR
    / "phase5_20_resilience_scorecard_manifest.json"
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
    "connectivity_recovery_valid",
    "synchronization_eligible",
    "soc_handoff_eligible",
    "phase5_18_baseline_integrity_valid",
    "tampered_evidence_rejected",
    "tampered_synchronization_refused",
    "tampered_soc_handoff_refused",
    "safety_boundary_valid",
    "end_to_end_chain",
    "end_to_end_valid",
    "evidence_digest",
    "laboratory_simulation",
    "physical_vehicle",
    "direct_vehicle_actuation",
    "live_cortex_xdr",
    "model_retraining",
    "real_world_cyberattack",
]


def as_bool(value):
    return str(value).strip().lower() == "true"


def sha256_text(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def evaluate(row):
    recovery = as_bool(
        row["connectivity_recovery_valid"]
    )

    synchronization = as_bool(
        row["synchronization_eligible"]
    )

    soc = as_bool(
        row["soc_handoff_eligible"]
    )

    baseline_integrity = as_bool(
        row["phase5_18_baseline_integrity_valid"]
    )

    tamper_rejected = as_bool(
        row["tampered_evidence_rejected"]
    )

    tamper_sync_refused = as_bool(
        row["tampered_synchronization_refused"]
    )

    tamper_soc_refused = as_bool(
        row["tampered_soc_handoff_refused"]
    )

    safety = as_bool(
        row["safety_boundary_valid"]
    )

    end_to_end = as_bool(
        row["end_to_end_valid"]
    )

    safety_flags = (
        not as_bool(row["physical_vehicle"])
        and not as_bool(row["direct_vehicle_actuation"])
        and not as_bool(row["live_cortex_xdr"])
        and not as_bool(row["model_retraining"])
        and not as_bool(row["real_world_cyberattack"])
    )

    recovery_score = int(recovery)
    synchronization_score = int(synchronization)
    soc_score = int(soc)
    integrity_score = int(baseline_integrity)
    adversarial_score = int(
        tamper_rejected
        and tamper_sync_refused
        and tamper_soc_refused
    )
    safety_score = int(
        safety
        and safety_flags
    )

    component_scores = [
        recovery_score,
        synchronization_score,
        soc_score,
        integrity_score,
        adversarial_score,
        safety_score,
    ]

    resilience_score = (
        sum(component_scores)
        / len(component_scores)
        * 100.0
    )

    scorecard_valid = (
        end_to_end
        and resilience_score == 100.0
    )

    evidence_digest = sha256_text(
        str(row["evidence_digest"])
        + str(row["end_to_end_chain"])
        + f"{resilience_score:.2f}"
    )

    return {
        "scenario_id": str(row["scenario_id"]),

        "recovery_valid": recovery,
        "synchronization_valid": synchronization,
        "soc_handoff_valid": soc,
        "baseline_integrity_valid": baseline_integrity,

        "tampered_evidence_rejected": tamper_rejected,
        "tampered_synchronization_refused": tamper_sync_refused,
        "tampered_soc_handoff_refused": tamper_soc_refused,

        "adversarial_resilience_valid": bool(
            tamper_rejected
            and tamper_sync_refused
            and tamper_soc_refused
        ),

        "safety_boundary_valid": bool(
            safety
            and safety_flags
        ),

        "end_to_end_valid": end_to_end,

        "recovery_score": recovery_score,
        "synchronization_score": synchronization_score,
        "soc_handoff_score": soc_score,
        "integrity_score": integrity_score,
        "adversarial_score": adversarial_score,
        "safety_score": safety_score,

        "resilience_score_percent": round(
            resilience_score,
            2,
        ),

        "scorecard_valid": bool(
            scorecard_valid
        ),

        "end_to_end_chain": str(
            row["end_to_end_chain"]
        ),

        "source_evidence_digest": str(
            row["evidence_digest"]
        ),

        "scorecard_evidence_digest": evidence_digest,

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
        "=== PHASE 5.20 "
        "RESILIENCE SCORECARD ==="
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
        print("FAIL: Phase 5.19 input missing")
        print(INPUT_CSV)
        return

    df = pd.read_csv(INPUT_CSV)

    print(
        f"Input rows: {len(df)}"
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    checks["required_columns_present"] = (
        len(missing) == 0
    )

    if missing:
        print("FAIL: Missing required columns")
        for column in missing:
            print(f"  {column}")
        return

    checks["input_rows_10"] = (
        len(df) == 10
    )

    checks["input_scenarios_10"] = (
        df["scenario_id"].nunique() == 10
    )

    checks["scenario_set_exact"] = (
        set(df["scenario_id"].astype(str))
        == set(SCENARIOS)
    )

    results = []

    for scenario_id in SCENARIOS:

        rows = df[
            df["scenario_id"].astype(str)
            == scenario_id
        ]

        print()
        print(
            f"Processing {scenario_id}..."
        )

        valid_source = len(rows) == 1

        checks[
            f"{scenario_id}_source_valid"
        ] = valid_source

        if not valid_source:
            print(
                "  source: FAIL"
            )
            continue

        result = evaluate(
            rows.iloc[0]
        )

        results.append(result)

        print(
            "  recovery:",
            "PASS"
            if result["recovery_valid"]
            else "FAIL",
        )

        print(
            "  synchronization:",
            "PASS"
            if result["synchronization_valid"]
            else "FAIL",
        )

        print(
            "  SOC handoff:",
            "PASS"
            if result["soc_handoff_valid"]
            else "FAIL",
        )

        print(
            "  baseline integrity:",
            "PASS"
            if result["baseline_integrity_valid"]
            else "FAIL",
        )

        print(
            "  adversarial resilience:",
            "PASS"
            if result["adversarial_resilience_valid"]
            else "FAIL",
        )

        print(
            "  safety boundary:",
            "PASS"
            if result["safety_boundary_valid"]
            else "FAIL",
        )

        print(
            f"  resilience score: "
            f"{result['resilience_score_percent']:.2f}%"
        )

        print(
            "  scorecard:",
            "PASS"
            if result["scorecard_valid"]
            else "FAIL",
        )

        checks[
            f"{scenario_id}_recovery"
        ] = result["recovery_valid"]

        checks[
            f"{scenario_id}_synchronization"
        ] = result["synchronization_valid"]

        checks[
            f"{scenario_id}_soc_handoff"
        ] = result["soc_handoff_valid"]

        checks[
            f"{scenario_id}_baseline_integrity"
        ] = result["baseline_integrity_valid"]

        checks[
            f"{scenario_id}_adversarial_resilience"
        ] = result[
            "adversarial_resilience_valid"
        ]

        checks[
            f"{scenario_id}_safety_boundary"
        ] = result[
            "safety_boundary_valid"
        ]

        checks[
            f"{scenario_id}_end_to_end"
        ] = result[
            "end_to_end_valid"
        ]

        checks[
            f"{scenario_id}_score_100"
        ] = (
            result[
                "resilience_score_percent"
            ] == 100.0
        )

        checks[
            f"{scenario_id}_digest_64"
        ] = (
            len(
                result[
                    "scorecard_evidence_digest"
                ]
            ) == 64
        )

    result_df = pd.DataFrame(results)

    checks["exactly_10_results"] = (
        len(result_df) == 10
    )

    if len(result_df) == 10:

        checks["all_recovery_valid"] = (
            result_df["recovery_valid"].all()
        )

        checks["all_synchronization_valid"] = (
            result_df[
                "synchronization_valid"
            ].all()
        )

        checks["all_soc_handoff_valid"] = (
            result_df[
                "soc_handoff_valid"
            ].all()
        )

        checks["all_baseline_integrity_valid"] = (
            result_df[
                "baseline_integrity_valid"
            ].all()
        )

        checks["all_adversarial_resilience_valid"] = (
            result_df[
                "adversarial_resilience_valid"
            ].all()
        )

        checks["all_safety_boundaries_valid"] = (
            result_df[
                "safety_boundary_valid"
            ].all()
        )

        checks["all_end_to_end_valid"] = (
            result_df[
                "end_to_end_valid"
            ].all()
        )

        checks["all_scores_100_percent"] = (
            result_df[
                "resilience_score_percent"
            ].eq(100.0).all()
        )

        checks["all_scorecards_valid"] = (
            result_df[
                "scorecard_valid"
            ].all()
        )

        checks["all_digests_64"] = (
            result_df[
                "scorecard_evidence_digest"
            ]
            .astype(str)
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
        "phase": "5.20",
        "title": (
            "End-to-End Resilience "
            "Scorecard"
        ),
        "input": str(
            INPUT_CSV.relative_to(ROOT)
        ),
        "scenarios": SCENARIOS,
        "score_components": [
            "recovery",
            "synchronization",
            "SOC handoff",
            "baseline integrity",
            "adversarial resilience",
            "safety boundary",
        ],
        "score_range": "0-100",
        "validated_score": 100.0,
        "scientific_boundary": {
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
        "phase": "5.20",
        "title": (
            "End-to-End Resilience "
            "Scorecard"
        ),
        "status": status,
        "input_rows": int(len(df)),
        "scenario_count": int(
            len(results)
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
        "=== PHASE 5.20 RESULTS ==="
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
