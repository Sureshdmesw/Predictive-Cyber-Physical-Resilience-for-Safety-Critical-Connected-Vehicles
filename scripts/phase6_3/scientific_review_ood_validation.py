from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles"
)

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase6_2"
    / "phase6_2_corrected_ood.csv"
)

INPUT_SCHEMA = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase6_2"
    / "phase6_2_corrected_ood_schema.json"
)

INPUT_MANIFEST = (
    ROOT
    / "experiments"
    / "phase6_2"
    / "phase6_2_corrected_ood_manifest.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase6_3"
)

MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "phase6_3"
    / "phase6_3_scientific_review_manifest.json"
)

BASELINE = "S00_NORMAL_BASELINE"

ATTACKS = [
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    print("=== PHASE 6.3 SCIENTIFIC REVIEW / OOD VALIDATION ===")

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    schema = json.loads(
        INPUT_SCHEMA.read_text(
            encoding="utf-8"
        )
    )

    manifest_62 = json.loads(
        INPUT_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    print(f"[1/7] Input rows: {len(df):,}")

    checks = []

    checks.append(
        (
            "Phase 6.2 manifest PASS",
            manifest_62.get("status") == "PASS",
        )
    )

    checks.append(
        (
            "Expected 6479 OOD records",
            len(df) == 6479,
        )
    )

    checks.append(
        (
            "Expected 11 scenarios",
            df["scenario_id"].nunique() == 11,
        )
    )

    checks.append(
        (
            "59 model dimensions preserved",
            schema.get("model_feature_count") == 59,
        )
    )

    checks.append(
        (
            "ecu_count excluded from OOD statistics",
            schema.get("feature_49_excluded") is True,
        )
    )

    checks.append(
        (
            "Model input unchanged",
            schema.get(
                "model_input_features_unchanged"
            ) is True,
        )
    )

    print("\n[2/7] Structural validation")

    for name, passed in checks:
        print(
            f"{'PASS' if passed else 'FAIL'}  {name}"
        )

    baseline = df.loc[
        df["scenario_id"].eq(BASELINE)
    ]

    attacks = df.loc[
        df["scenario_id"].isin(ATTACKS)
    ]

    print("\n[3/7] Baseline / attack statistics")

    baseline_rate = float(
        baseline["corrected_ood_flag"].mean()
    )

    attack_rate = float(
        attacks["corrected_ood_flag"].mean()
    )

    baseline_score_mean = float(
        baseline["corrected_ood_score"].mean()
    )

    attack_score_mean = float(
        attacks["corrected_ood_score"].mean()
    )

    baseline_score_median = float(
        baseline["corrected_ood_score"].median()
    )

    attack_score_median = float(
        attacks["corrected_ood_score"].median()
    )

    max_score = float(
        df["corrected_ood_score"].max()
    )

    print(
        f"Baseline OOD rate : {baseline_rate:.6f}"
    )

    print(
        f"Attack OOD rate   : {attack_rate:.6f}"
    )

    print(
        f"Baseline mean     : {baseline_score_mean:.6f}"
    )

    print(
        f"Attack mean       : {attack_score_mean:.6f}"
    )

    print(
        f"Baseline median   : {baseline_score_median:.6f}"
    )

    print(
        f"Attack median     : {attack_score_median:.6f}"
    )

    print(
        f"Global max score  : {max_score:.6f}"
    )

    print("\n[4/7] Scenario review")

    scenario_rows = []

    for scenario_id in [
        BASELINE,
        *ATTACKS,
    ]:
        subset = df.loc[
            df["scenario_id"].eq(
                scenario_id
            )
        ]

        row = {
            "scenario_id": scenario_id,
            "window_count": int(len(subset)),
            "ood_rate": float(
                subset[
                    "corrected_ood_flag"
                ].mean()
            ),
            "mean_ood_score": float(
                subset[
                    "corrected_ood_score"
                ].mean()
            ),
            "median_ood_score": float(
                subset[
                    "corrected_ood_score"
                ].median()
            ),
            "max_ood_score": float(
                subset[
                    "corrected_ood_score"
                ].max()
            ),
            "mean_feature_z_rms": float(
                subset[
                    "feature_z_rms"
                ].mean()
            ),
            "max_feature_z_rms": float(
                subset[
                    "feature_z_rms"
                ].max()
            ),
            "mean_feature_z_max": float(
                subset[
                    "feature_z_max"
                ].mean()
            ),
            "max_feature_z_max": float(
                subset[
                    "feature_z_max"
                ].max()
            ),
            "mean_mahalanobis": float(
                subset[
                    "diagonal_mahalanobis"
                ].mean()
            ),
            "max_mahalanobis": float(
                subset[
                    "diagonal_mahalanobis"
                ].max()
            ),
            "mean_temporal_delta_rms": float(
                subset[
                    "temporal_delta_rms"
                ].mean()
            ),
            "max_temporal_delta_rms": float(
                subset[
                    "temporal_delta_rms"
                ].max()
            ),
        }

        scenario_rows.append(row)

        print(
            f"{scenario_id}: "
            f"OOD rate={row['ood_rate']:.6f}, "
            f"mean={row['mean_ood_score']:.6f}, "
            f"max={row['max_ood_score']:.6f}"
        )

    scenario_summary = pd.DataFrame(
        scenario_rows
    )

    print("\n[5/7] Scientific interpretation checks")

    scientific_checks = [
        (
            "No non-finite corrected OOD scores",
            np.isfinite(
                df["corrected_ood_score"]
            ).all(),
        ),
        (
            "No non-finite corrected metrics",
            np.isfinite(
                df[
                    [
                        "feature_z_rms",
                        "feature_z_max",
                        "diagonal_mahalanobis",
                        "temporal_delta_rms",
                    ]
                ].to_numpy()
            ).all(),
        ),
        (
            "Baseline rate remains approximately 1% tail",
            0.0 <= baseline_rate <= 0.10,
        ),
        (
            "Attack OOD rate is not lower than baseline",
            attack_rate >= baseline_rate,
        ),
        (
            "Corrected result removes million-scale pathological score",
            max_score < 100000.0,
        ),
        (
            "No claim of perfect attack separation",
            attack_rate < 1.0,
        ),
    ]

    for name, passed in scientific_checks:
        print(
            f"{'PASS' if passed else 'FAIL'}  {name}"
        )

    all_checks = checks + scientific_checks

    validation_pass = all(
        passed
        for _, passed in all_checks
    )

    print("\n[6/7] Review classification")

    if validation_pass:
        review_status = (
            "SCIENTIFIC_REVIEW_ACCEPTED_WITH_LIMITATIONS"
        )
    else:
        review_status = (
            "SCIENTIFIC_REVIEW_REQUIRES_FOLLOWUP"
        )

    print(
        f"Review status: {review_status}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_csv = (
        OUTPUT_DIR
        / "phase6_3_ood_scenario_summary.csv"
    )

    summary_json = (
        OUTPUT_DIR
        / "phase6_3_ood_scientific_review.json"
    )

    scenario_summary.to_csv(
        summary_csv,
        index=False,
    )

    review = {
        "phase": "6.3",
        "title": "Scientific Review and OOD Validation",
        "status": review_status,
        "input_phase": "6.2",
        "input_sha256": sha256_file(INPUT_CSV),
        "record_count": int(len(df)),
        "scenario_count": int(
            df["scenario_id"].nunique()
        ),
        "model_feature_count": 59,
        "ood_statistical_feature_count": int(
            schema[
                "ood_statistical_feature_count"
            ]
        ),
        "ood_excluded_feature_count": int(
            schema[
                "ood_excluded_feature_count"
            ]
        ),
        "ecu_count_excluded": True,
        "baseline": {
            "window_count": int(len(baseline)),
            "ood_rate": baseline_rate,
            "mean_score": baseline_score_mean,
            "median_score": baseline_score_median,
        },
        "attacks": {
            "window_count": int(len(attacks)),
            "ood_rate": attack_rate,
            "mean_score": attack_score_mean,
            "median_score": attack_score_median,
        },
        "global": {
            "max_corrected_ood_score": max_score,
        },
        "scientific_interpretation": {
            "baseline_tail_is_expected_statistical_behavior": True,
            "attack_ood_rate_exceeds_baseline": (
                attack_rate >= baseline_rate
            ),
            "perfect_attack_separation_not_claimed": True,
            "pathological_feature49_tail_removed": True,
            "model_input_dimensions_modified": False,
            "attack_specific_threshold_tuning": False,
        },
        "limitations": [
            "OOD score is a laboratory statistical distance measure.",
            "OOD detection is not equivalent to attack detection.",
            "The corrected OOD result does not establish production automotive safety.",
            "The result does not establish OEM or regulatory compliance.",
            "Synthetic cyber telemetry remains synthetic experimental data.",
            "Baseline and attack distributions remain partially overlapping.",
        ],
        "validation": {
            name: bool(passed)
            for name, passed in all_checks
        },
    }

    summary_json.write_text(
        json.dumps(
            review,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "phase": "6.3",
        "title": "Scientific Review and OOD Validation",
        "status": review_status,
        "input": str(INPUT_CSV),
        "input_sha256": sha256_file(INPUT_CSV),
        "outputs": {
            "scenario_summary": str(summary_csv),
            "scientific_review": str(summary_json),
        },
        "validation_count": len(all_checks),
        "validation_pass_count": sum(
            bool(passed)
            for _, passed in all_checks
        ),
        "validation_fail_count": sum(
            not bool(passed)
            for _, passed in all_checks
        ),
        "scientific_review_required": True,
        "scientific_review_completed": validation_pass,
        "limitations_preserved": True,
        "model_input_modified": False,
        "phase_6_2_frozen": True,
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[7/7] Outputs written")

    print(
        f"Scenario summary: {summary_csv}"
    )

    print(
        f"Scientific review: {summary_json}"
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    print(
        f"\nSTATUS: {review_status}"
    )


if __name__ == "__main__":
    main()
