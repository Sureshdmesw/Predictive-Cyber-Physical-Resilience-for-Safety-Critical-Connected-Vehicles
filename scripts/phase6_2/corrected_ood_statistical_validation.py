from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

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
    / "phase5_12"
    / "phase5_12_uncertainty_ood.csv"
)

OUT_DIR = ROOT / "data" / "processed" / "phase6_2"
EXP_DIR = ROOT / "experiments" / "phase6_2"

OUT_CSV = OUT_DIR / "phase6_2_corrected_ood.csv"
OUT_JSONL = OUT_DIR / "phase6_2_corrected_ood.jsonl"
OUT_SCHEMA = OUT_DIR / "phase6_2_corrected_ood_schema.json"
OUT_MANIFEST = EXP_DIR / "phase6_2_corrected_ood_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def percentile(values, q):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return float("nan")

    return float(np.percentile(values, q))


def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=== PHASE 6.2 OOD STATISTICAL REVIEW ===")

    if not INPUT_CSV.exists():
        print("[FAIL] Phase 5.12 CSV not found:")
        print(INPUT_CSV)
        return

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False
    )

    print(f"Input rows: {len(df):,}")
    print(f"Input columns: {len(df.columns)}")

    required = {
        "scenario_id",
        "window_index",
        "feature_z_rms",
        "feature_z_max",
        "diagonal_mahalanobis",
        "ood_score",
        "ood_flag",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        print("[FAIL] Required columns missing:")

        for column in missing:
            print(f"  - {column}")

        return

    scenarios = sorted(
        df["scenario_id"].dropna().unique()
    )

    print(f"Scenarios: {len(scenarios)}")

    baseline_present = (
        "S00_NORMAL_BASELINE" in scenarios
    )

    print(
        "Baseline present:",
        baseline_present
    )

    # ------------------------------------------------------------
    # Preserve Phase 5.12 values exactly.
    # ------------------------------------------------------------

    result = df.copy()

    result["phase5_12_ood_score"] = pd.to_numeric(
        result["ood_score"],
        errors="coerce"
    )

    result["phase5_12_feature_z_rms"] = pd.to_numeric(
        result["feature_z_rms"],
        errors="coerce"
    )

    result["phase5_12_feature_z_max"] = pd.to_numeric(
        result["feature_z_max"],
        errors="coerce"
    )

    result["phase5_12_diagonal_distance"] = pd.to_numeric(
        result["diagonal_mahalanobis"],
        errors="coerce"
    )

    # ------------------------------------------------------------
    # Check the mathematical relationship without assuming it.
    # ------------------------------------------------------------

    relationship_error = (
        result["phase5_12_diagonal_distance"]
        - np.square(result["phase5_12_feature_z_rms"])
    ).abs()

    relationship_valid = bool(
        relationship_error.fillna(0).max() < 1e-3
    )

    print()
    print(
        "diagonal_mahalanobis == feature_z_rms^2:",
        relationship_valid
    )

    # ------------------------------------------------------------
    # Identify the known pathological tail conservatively.
    #
    # IMPORTANT:
    # Parentheses around BOTH comparisons are required for Pandas.
    # ------------------------------------------------------------

    affected_scenarios = {
        "C07_HEARTBEAT_INTEGRITY_ATTACK",
        "C08_SEQUENCE_GATEWAY_ATTACK",
        "C10_MULTI_DOMAIN_ATTACK",
    }

    rms_flag = (
        result["phase5_12_feature_z_rms"] > 10.0
    )

    max_flag = (
        result["phase5_12_feature_z_max"] > 10.0
    )

    scenario_flag = result[
        "scenario_id"
    ].isin(
        affected_scenarios
    )

    affected_mask = (
        scenario_flag
        & (rms_flag | max_flag)
    )

    result[
        "phase6_2_statistical_review_required"
    ] = affected_mask.astype(bool)

    result[
        "phase6_2_known_variance_floor_issue"
    ] = affected_mask.astype(bool)

    # ------------------------------------------------------------
    # No fabricated corrected score.
    #
    # The Phase 5.12 table contains aggregate metrics. It does not
    # contain the underlying 59-dimensional window vectors.
    # ------------------------------------------------------------

    result[
        "corrected_ood_score"
    ] = np.nan

    result[
        "corrected_feature_z_rms"
    ] = np.nan

    result[
        "corrected_feature_z_max"
    ] = np.nan

    result[
        "corrected_diagonal_distance"
    ] = np.nan

    result[
        "phase6_2_correction_status"
    ] = (
        "RECOMPUTATION_REQUIRED_FROM_59D_FEATURE_VECTORS"
    )

    # ------------------------------------------------------------
    # Global statistics
    # ------------------------------------------------------------

    scores = result[
        "phase5_12_ood_score"
    ].dropna()

    baseline = result[
        result["scenario_id"]
        == "S00_NORMAL_BASELINE"
    ]

    attacks = result[
        result["scenario_id"]
        != "S00_NORMAL_BASELINE"
    ]

    print()
    print("=== ORIGINAL PHASE 5.12 STATISTICS ===")

    print(
        f"Mean:   {float(scores.mean()):.6f}"
    )

    print(
        f"Median: {float(scores.median()):.6f}"
    )

    print(
        f"P95:    {percentile(scores, 95):.6f}"
    )

    print(
        f"P99:    {percentile(scores, 99):.6f}"
    )

    print(
        f"Max:    {float(scores.max()):.6f}"
    )

    print()
    print("=== BASELINE ===")

    print(
        f"Rows: {len(baseline)}"
    )

    print(
        f"Mean OOD score: "
        f"{float(baseline['phase5_12_ood_score'].mean()):.6f}"
    )

    print(
        f"Max OOD score: "
        f"{float(baseline['phase5_12_ood_score'].max()):.6f}"
    )

    print()
    print("=== ATTACKS ===")

    print(
        f"Rows: {len(attacks)}"
    )

    print(
        f"Mean OOD score: "
        f"{float(attacks['phase5_12_ood_score'].mean()):.6f}"
    )

    print(
        f"Max OOD score: "
        f"{float(attacks['phase5_12_ood_score'].max()):.6f}"
    )

    print()
    print("=== EXTREME-TAIL REVIEW ===")

    review_count = int(
        result[
            "phase6_2_statistical_review_required"
        ].sum()
    )

    print(
        f"Review windows: {review_count}"
    )

    # ------------------------------------------------------------
    # Scenario statistics
    # ------------------------------------------------------------

    scenario_summary = []

    for scenario_id, group in result.groupby(
        "scenario_id",
        sort=True
    ):

        scenario_scores = group[
            "phase5_12_ood_score"
        ].dropna()

        scenario_summary.append({
            "scenario_id": scenario_id,
            "rows": int(len(group)),
            "mean_ood_score": float(
                scenario_scores.mean()
            ),
            "median_ood_score": float(
                scenario_scores.median()
            ),
            "p95_ood_score": percentile(
                scenario_scores,
                95
            ),
            "p99_ood_score": percentile(
                scenario_scores,
                99
            ),
            "max_ood_score": float(
                scenario_scores.max()
            ),
            "review_windows": int(
                group[
                    "phase6_2_statistical_review_required"
                ].sum()
            ),
        })

    scenario_summary.sort(
        key=lambda x: x["max_ood_score"],
        reverse=True
    )

    for item in scenario_summary:

        print(
            f"{item['scenario_id']}: "
            f"max={item['max_ood_score']:.6f}, "
            f"p99={item['p99_ood_score']:.6f}, "
            f"review={item['review_windows']}"
        )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    checks = []

    checks.append({
        "id": "INPUT_EXISTS",
        "status": INPUT_CSV.exists()
    })

    checks.append({
        "id": "ROWS_6479",
        "status": len(df) == 6479
    })

    checks.append({
        "id": "SCENARIOS_11",
        "status": len(scenarios) == 11
    })

    checks.append({
        "id": "BASELINE_PRESENT",
        "status": baseline_present
    })

    checks.append({
        "id": "ORIGINAL_OOD_FINITE",
        "status": bool(
            np.isfinite(
                result[
                    "phase5_12_ood_score"
                ].to_numpy()
            ).all()
        )
    })

    checks.append({
        "id": "ORIGINAL_FEATURE_RMS_FINITE",
        "status": bool(
            np.isfinite(
                result[
                    "phase5_12_feature_z_rms"
                ].to_numpy()
            ).all()
        )
    })

    checks.append({
        "id": "ORIGINAL_FEATURE_MAX_FINITE",
        "status": bool(
            np.isfinite(
                result[
                    "phase5_12_feature_z_max"
                ].to_numpy()
            ).all()
        )
    })

    checks.append({
        "id": "NO_MEMORY_INTENSIVE_MERGE",
        "status": True
    })

    checks.append({
        "id": "ORIGINAL_EVIDENCE_PRESERVED",
        "status": True
    })

    checks.append({
        "id": "EXTREME_TAIL_REVIEW_IDENTIFIED",
        "status": review_count > 0
    })

    # This must remain FALSE until we have the underlying
    # 59-dimensional feature vectors.
    checks.append({
        "id": "CORRECTED_METRIC_RECOMPUTED",
        "status": False
    })

    passed = sum(
        1
        for check in checks
        if check["status"]
    )

    total = len(checks)

    # ------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------

    result.to_csv(
        OUT_CSV,
        index=False
    )

    # ------------------------------------------------------------
    # JSONL
    # ------------------------------------------------------------

    with OUT_JSONL.open(
        "w",
        encoding="utf-8"
    ) as f:

        for record in result.to_dict(
            orient="records"
        ):

            f.write(
                json.dumps(
                    record,
                    default=str,
                    sort_keys=True
                )
                + "\n"
            )

    # ------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------

    schema = {
        "phase": "6.2",
        "title": (
            "Phase 5.12 OOD Statistical Review "
            "and Correction Readiness"
        ),

        "input": str(
            INPUT_CSV.relative_to(ROOT)
        ),

        "method": {
            "memory_safe": True,
            "phase5_10_merge": False,
            "original_ood_preserved": True,
            "score_clipping": False,
            "score_fabrication": False,
            "corrected_metric": "NOT_COMPUTED",
            "reason": (
                "Underlying 59-dimensional window vectors are "
                "required for a valid statistical recomputation."
            ),
        },

        "known_issue": {
            "near_zero_baseline_variance": True,
            "known_feature": "ecu_count",
            "previous_denominator_floor": "1e-6",
            "consequence": (
                "Artificially large standardized deviations "
                "can result when baseline variance is effectively zero."
            ),
        },

        "validation": {
            "passed": passed,
            "total": total,
            "checks": checks,
        },

        "outputs": {
            "csv": str(
                OUT_CSV.relative_to(ROOT)
            ),
            "jsonl": str(
                OUT_JSONL.relative_to(ROOT)
            ),
            "schema": str(
                OUT_SCHEMA.relative_to(ROOT)
            ),
            "manifest": str(
                OUT_MANIFEST.relative_to(ROOT)
            ),
        },
    }

    with OUT_SCHEMA.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            schema,
            f,
            indent=2
        )

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------

    manifest = {
        "phase": "6.2",
        "title": (
            "Phase 5.12 OOD Statistical Review "
            "and Correction Readiness"
        ),

        "status": "REVIEW_REQUIRED",

        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "input": {
            "path": str(
                INPUT_CSV.relative_to(ROOT)
            ),
            "sha256": sha256_file(INPUT_CSV),
            "rows": len(df),
            "columns": len(df.columns),
        },

        "validation": {
            "passed": passed,
            "total": total,
            "checks": checks,
        },

        "statistics": {
            "original_mean": float(
                scores.mean()
            ),
            "original_median": float(
                scores.median()
            ),
            "original_p95": percentile(
                scores,
                95
            ),
            "original_p99": percentile(
                scores,
                99
            ),
            "original_max": float(
                scores.max()
            ),
            "review_windows": review_count,
        },

        "scenario_summary": scenario_summary,

        "scientific_boundary": {
            "laboratory_simulation": True,
            "physical_vehicle": False,
            "direct_vehicle_actuation": False,
            "live_cortex_xdr": False,
            "model_retraining": False,
            "real_world_cyberattack": False,
        },
    }

    with OUT_MANIFEST.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )

    print()
    print("=== PHASE 6.2 RESULT ===")
    print(
        f"Validation: {passed}/{total} checks passed"
    )
    print(
        "Corrected metric: NOT COMPUTED"
    )
    print(
        "Scientific status: REVIEW_REQUIRED"
    )

    print()
    print("Outputs:")
    print(OUT_CSV)
    print(OUT_JSONL)
    print(OUT_SCHEMA)
    print(OUT_MANIFEST)


if __name__ == "__main__":
    main()
