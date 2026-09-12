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
    / "phase5_20"
    / "phase5_20_resilience_scorecard.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_21"
)

EXPERIMENT_DIR = (
    ROOT
    / "experiments"
    / "can_hil"
    / "phase5_21"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "phase5_21_resilience_evidence_matrix.csv"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "phase5_21_resilience_evidence_matrix.jsonl"
)

OUTPUT_SCHEMA = (
    OUTPUT_DIR
    / "phase5_21_resilience_evidence_matrix_schema.json"
)

MANIFEST = (
    EXPERIMENT_DIR
    / "phase5_21_resilience_evidence_matrix_manifest.json"
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
    "recovery_valid",
    "synchronization_valid",
    "soc_handoff_valid",
    "baseline_integrity_valid",
    "tampered_evidence_rejected",
    "tampered_synchronization_refused",
    "tampered_soc_handoff_refused",
    "adversarial_resilience_valid",
    "safety_boundary_valid",
    "end_to_end_valid",
    "recovery_score",
    "synchronization_score",
    "soc_handoff_score",
    "integrity_score",
    "adversarial_score",
    "safety_score",
    "resilience_score_percent",
    "scorecard_valid",
    "end_to_end_chain",
    "source_evidence_digest",
    "scorecard_evidence_digest",
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

    component_names = [
        "recovery",
        "synchronization",
        "soc_handoff",
        "integrity",
        "adversarial",
        "safety",
    ]

    component_values = [
        as_bool(row["recovery_valid"]),
        as_bool(row["synchronization_valid"]),
        as_bool(row["soc_handoff_valid"]),
        as_bool(row["baseline_integrity_valid"]),
        as_bool(row["adversarial_resilience_valid"]),
        as_bool(row["safety_boundary_valid"]),
    ]

    component_scores = [
        int(value)
        for value in component_values
    ]

    failed_components = [
        name
        for name, value in zip(
            component_names,
            component_values,
        )
        if not value
    ]

    all_components_valid = all(
        component_values
    )

    end_to_end_valid = as_bool(
        row["end_to_end_valid"]
    )

    scorecard_valid = as_bool(
        row["scorecard_valid"]
    )

    safety_boundary_valid = (
        as_bool(row["physical_vehicle"]) is False
        and as_bool(row["direct_vehicle_actuation"]) is False
        and as_bool(row["live_cortex_xdr"]) is False
        and as_bool(row["model_retraining"]) is False
        and as_bool(row["real_world_cyberattack"]) is False
    )

    resilience_score = float(
        row["resilience_score_percent"]
    )

    evidence_complete = (
        len(
            str(row["source_evidence_digest"])
        ) == 64
        and len(
            str(row["scorecard_evidence_digest"])
        ) == 64
    )

    matrix_digest = sha256_text(
        str(row["scenario_id"])
        + str(row["source_evidence_digest"])
        + str(row["scorecard_evidence_digest"])
        + str(resilience_score)
        + "|".join(component_names)
        + "|".join(
            str(score)
            for score in component_scores
        )
    )

    return {
        "scenario_id": str(
            row["scenario_id"]
        ),

        "recovery": component_values[0],
        "synchronization": component_values[1],
        "soc_handoff": component_values[2],
        "baseline_integrity": component_values[3],
        "adversarial_resilience": component_values[4],
        "safety_boundary": component_values[5],

        "recovery_score": component_scores[0],
        "synchronization_score": component_scores[1],
        "soc_handoff_score": component_scores[2],
        "integrity_score": component_scores[3],
        "adversarial_score": component_scores[4],
        "safety_score": component_scores[5],

        "failed_component_count": len(
            failed_components
        ),

        "failed_components": (
            "|".join(failed_components)
            if failed_components
            else "NONE"
        ),

        "all_components_valid": bool(
            all_components_valid
        ),

        "end_to_end_valid": bool(
            end_to_end_valid
        ),

        "scorecard_valid": bool(
            scorecard_valid
        ),

        "resilience_score_percent": resilience_score,

        "evidence_complete": bool(
            evidence_complete
        ),

        "safety_boundary_revalidated": bool(
            safety_boundary_valid
        ),

        "end_to_end_chain": str(
            row["end_to_end_chain"]
        ),

        "source_evidence_digest": str(
            row["source_evidence_digest"]
        ),

        "scorecard_evidence_digest": str(
            row["scorecard_evidence_digest"]
        ),

        "matrix_evidence_digest": matrix_digest,

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
        "=== PHASE 5.21 "
        "RESILIENCE EVIDENCE MATRIX ==="
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

    checks["input_exists"] = (
        INPUT_CSV.exists()
    )

    if not INPUT_CSV.exists():
        print(
            "FAIL: Phase 5.20 input missing"
        )
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
        print(
            "FAIL: Missing required columns"
        )
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

        source_valid = len(rows) == 1

        checks[
            f"{scenario_id}_source_valid"
        ] = source_valid

        if not source_valid:
            print(
                "  source: FAIL"
            )
            continue

        result = evaluate(
            rows.iloc[0]
        )

        results.append(result)

        print(
            "  components:",
            "PASS"
            if result["all_components_valid"]
            else "FAIL",
        )

        print(
            "  evidence:",
            "COMPLETE"
            if result["evidence_complete"]
            else "INCOMPLETE",
        )

        print(
            "  safety:",
            "PASS"
            if result[
                "safety_boundary_revalidated"
            ]
            else "FAIL",
        )

        print(
            f"  resilience score: "
            f"{result['resilience_score_percent']:.2f}%"
        )

        print(
            "  matrix:",
            "PASS"
            if (
                result["all_components_valid"]
                and result["end_to_end_valid"]
                and result["scorecard_valid"]
                and result["evidence_complete"]
                and result[
                    "safety_boundary_revalidated"
                ]
            )
            else "FAIL",
        )

        checks[
            f"{scenario_id}_components"
        ] = result["all_components_valid"]

        checks[
            f"{scenario_id}_end_to_end"
        ] = result["end_to_end_valid"]

        checks[
            f"{scenario_id}_scorecard"
        ] = result["scorecard_valid"]

        checks[
            f"{scenario_id}_evidence_complete"
        ] = result["evidence_complete"]

        checks[
            f"{scenario_id}_safety"
        ] = result[
            "safety_boundary_revalidated"
        ]

        checks[
            f"{scenario_id}_score_100"
        ] = (
            result[
                "resilience_score_percent"
            ] == 100.0
        )

        checks[
            f"{scenario_id}_matrix_digest_64"
        ] = (
            len(
                result[
                    "matrix_evidence_digest"
                ]
            ) == 64
        )

    result_df = pd.DataFrame(results)

    checks["exactly_10_results"] = (
        len(result_df) == 10
    )

    if len(result_df) == 10:

        checks["all_components_valid"] = (
            result_df[
                "all_components_valid"
            ].all()
        )

        checks["all_end_to_end_valid"] = (
            result_df[
                "end_to_end_valid"
            ].all()
        )

        checks["all_scorecards_valid"] = (
            result_df[
                "scorecard_valid"
            ].all()
        )

        checks["all_evidence_complete"] = (
            result_df[
                "evidence_complete"
            ].all()
        )

        checks[
            "all_safety_boundaries_revalidated"
        ] = (
            result_df[
                "safety_boundary_revalidated"
            ].all()
        )

        checks["all_scores_100"] = (
            result_df[
                "resilience_score_percent"
            ].eq(100.0).all()
        )

        checks["all_matrix_digests_64"] = (
            result_df[
                "matrix_evidence_digest"
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
        "phase": "5.21",
        "title": (
            "Cross-Scenario Resilience "
            "Evidence Matrix"
        ),
        "input": str(
            INPUT_CSV.relative_to(ROOT)
        ),
        "scenarios": SCENARIOS,
        "evidence_dimensions": [
            "recovery",
            "synchronization",
            "soc_handoff",
            "baseline_integrity",
            "adversarial_resilience",
            "safety_boundary",
            "end_to_end_valid",
            "evidence_complete",
        ],
        "score_range": "0-100",
        "expected_score": 100.0,
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
        "phase": "5.21",
        "title": (
            "Cross-Scenario Resilience "
            "Evidence Matrix"
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
        "=== PHASE 5.21 RESULTS ==="
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
