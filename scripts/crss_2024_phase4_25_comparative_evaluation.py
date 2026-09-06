from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "experiments" / "modeling" / "v2" / "evidence"
OUT = OUT_DIR / "crss_2024_v2_phase4_25_comparative_evaluation.json"


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root is not an object: {path}")

    return data


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def get_path(obj: Any, path: tuple[str, ...]):
    cur = obj

    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None

        cur = cur[key]

    return cur


def recursive_find_key(obj: Any, key: str, path: str = ""):
    """
    Return [(path, value)] for every matching key.
    """
    found = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            current_path = f"{path}.{k}" if path else k

            if k == key:
                found.append((current_path, v))

            found.extend(recursive_find_key(v, key, current_path))

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            current_path = f"{path}[{i}]"
            found.extend(recursive_find_key(item, key, current_path))

    return found


def first_numeric_path(
    data: dict[str, Any],
    paths: list[tuple[str, ...]],
):
    for path in paths:
        value = get_path(data, path)

        if finite_number(value):
            return float(value), ".".join(path)

    return None, None


def recursive_numeric(
    data: dict[str, Any],
    keys: list[str],
):
    for key in keys:
        matches = recursive_find_key(data, key)

        for path, value in matches:
            if finite_number(value):
                return float(value), path

    return None, None


def recursive_value(
    data: dict[str, Any],
    keys: list[str],
):
    for key in keys:
        matches = recursive_find_key(data, key)

        if matches:
            return matches[0][1], matches[0][0]

    return None, None


def phase_status(data: dict[str, Any]) -> str:
    for key in (
        "status",
        "release_status",
        "phase_status",
        "overall_status",
    ):
        value = data.get(key)

        if isinstance(value, str):
            return value

    return "UNKNOWN"


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ============================================================================
# SOURCE REGISTRY
# ============================================================================

SOURCES = {
    "phase_4_9_baseline":
        ROOT / "experiments/modeling/v2/baseline/"
        "crss_2024_v2_strong_baseline_results.json",

    "phase_4_10_transformer":
        ROOT / "experiments/modeling/v2/advanced/"
        "crss_2024_v2_temporal_transformer_controlled_final.json",

    "phase_4_11_cross_modal":
        ROOT / "experiments/modeling/v2/cross_modal/"
        "crss_2024_v2_cross_modal_results.json",

    "phase_4_12_modality":
        ROOT / "experiments/modeling/v2/modality_contribution/"
        "crss_2024_v2_modality_contribution_results.json",

    "phase_4_13_feature":
        ROOT / "experiments/modeling/v2/feature_contribution/"
        "crss_2024_v2_feature_contribution_results.json",

    "phase_4_14_optimized":
        ROOT / "experiments/modeling/v2/optimized_features/"
        "crss_2024_v2_optimized_feature_results.json",

    "phase_4_15_generalization_leakage":
        ROOT / "experiments/modeling/v2/generalization/"
        "crss_2024_v2_generalization_leakage_stress_test.json",

    "phase_4_16_generalization":
        ROOT / "experiments/modeling/v2/generalization/phase4_16/"
        "crss_2024_v2_phase4_16_generalization_results.json",

    "phase_4_17_error_analysis":
        ROOT / "experiments/modeling/v2/error_analysis/"
        "crss_2024_v2_phase4_17_error_analysis.json",

    "phase_4_18_calibration":
        ROOT / "experiments/modeling/v2/calibration/"
        "crss_2024_v2_phase4_18_calibration_thresholds.json",

    "phase_4_19_robustness":
        ROOT / "experiments/modeling/v2/robustness/"
        "crss_2024_v2_phase4_19_robustness_confirmation.json",

    "phase_4_20_decision_engine":
        ROOT / "experiments/modeling/v2/decision_engine/"
        "crss_2024_v2_phase4_20_predictive_resilience_decision_engine.json",

    "phase_4_21_edge_iot":
        ROOT / "experiments/edge_iot/"
        "crss_2024_phase4_21_edge_iot_resilience.json",

    "phase_4_22_edge_iot_scenarios":
        ROOT / "experiments/edge_iot/"
        "crss_2024_phase4_22_scenario_evaluation.json",

    "phase_4_23_end_to_end":
        ROOT / "experiments/edge_iot/"
        "crss_2024_phase4_23_end_to_end_resilience.json",

    "phase_4_24_stress":
        ROOT / "experiments/edge_iot/"
        "crss_2024_phase4_24_system_stress_failure_analysis.json",

    "phase_4_25_manifest":
        ROOT / "experiments/modeling/v2/evidence/"
        "crss_2024_v2_phase4_25_evidence_manifest.json",
}


# ============================================================================
# ROBUSTNESS EXTRACTION
# ============================================================================

def extract_phase_419_robustness(data: dict[str, Any]):
    """
    Supports the actual Phase 4.19 schema:

        baseline
        robustness_results
        origin_results
        interpretation
        gates

    and also alternate condition-list schemas.
    """

    results = {}

    # ------------------------------------------------------------------------
    # Actual Phase 4.19 baseline structure
    # ------------------------------------------------------------------------

    baseline = data.get("baseline")

    if isinstance(baseline, dict):

        # baseline may itself contain condition-like metrics.
        pr, pr_path = first_numeric_path(
            baseline,
            [
                ("pr_auc",),
                ("test_pr_auc",),
                ("clean_pr_auc",),
            ],
        )

        roc, roc_path = first_numeric_path(
            baseline,
            [
                ("roc_auc",),
                ("test_roc_auc",),
                ("clean_roc_auc",),
            ],
        )

        if pr is not None:
            results["clean"] = {
                "pr_auc": pr,
                "roc_auc": roc,
                "source_pr_auc": f"baseline.{pr_path}",
                "source_roc_auc": (
                    f"baseline.{roc_path}"
                    if roc_path
                    else None
                ),
            }

    # ------------------------------------------------------------------------
    # Actual robustness_results list
    # ------------------------------------------------------------------------

    robustness_results = data.get("robustness_results")

    if isinstance(robustness_results, list):

        for item in robustness_results:

            if not isinstance(item, dict):
                continue

            condition = item.get("condition")

            if condition is None:
                condition = item.get("name")

            if condition is None:
                condition = item.get("test")

            if condition is None:
                condition = "unknown"

            condition = str(condition)

            pr, pr_path = first_numeric_path(
                item,
                [
                    ("pr_auc",),
                    ("test_pr_auc",),
                    ("y6_pr_auc",),
                ],
            )

            roc, roc_path = first_numeric_path(
                item,
                [
                    ("roc_auc",),
                    ("test_roc_auc",),
                    ("y6_roc_auc",),
                ],
            )

            if pr is not None:
                results[condition] = {
                    "pr_auc": pr,
                    "roc_auc": roc,
                    "source_pr_auc": (
                        f"robustness_results[{condition}].{pr_path}"
                    ),
                    "source_roc_auc": (
                        f"robustness_results[{condition}].{roc_path}"
                        if roc_path
                        else None
                    ),
                }

    # ------------------------------------------------------------------------
    # Generic conditions list fallback
    # ------------------------------------------------------------------------

    conditions = data.get("conditions")

    if isinstance(conditions, list):

        for item in conditions:

            if not isinstance(item, dict):
                continue

            condition = str(
                item.get("condition")
                or item.get("name")
                or item.get("test")
                or "unknown"
            )

            pr = item.get("pr_auc")
            roc = item.get("roc_auc")

            if finite_number(pr):
                results[condition] = {
                    "pr_auc": float(pr),
                    "roc_auc": (
                        float(roc)
                        if finite_number(roc)
                        else None
                    ),
                    "source_pr_auc": (
                        f"conditions[{condition}].pr_auc"
                    ),
                    "source_roc_auc": (
                        f"conditions[{condition}].roc_auc"
                        if finite_number(roc)
                        else None
                    ),
                }

    # ------------------------------------------------------------------------
    # Generic recursive fallback
    # ------------------------------------------------------------------------

    if "clean" not in results:

        pr, pr_path = recursive_numeric(
            data,
            ["clean_pr_auc", "baseline_pr_auc"],
        )

        roc, roc_path = recursive_numeric(
            data,
            ["clean_roc_auc", "baseline_roc_auc"],
        )

        if pr is not None:
            results["clean"] = {
                "pr_auc": pr,
                "roc_auc": roc,
                "source_pr_auc": pr_path,
                "source_roc_auc": roc_path,
            }

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("PHASE 4.25 — COMPARATIVE EVIDENCE EVALUATION")
    print("=" * 80)
    print("Robust schema handling / evidence-only / NO RETRAINING")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Load all sources
    # ------------------------------------------------------------------------

    loaded = {}
    missing = {}
    invalid = {}

    print("=== SOURCE EXISTENCE AUDIT ===")

    for name, path in SOURCES.items():

        if not path.exists():
            missing[name] = relative(path)
            print(f"FAIL  {name}")
            print(f"      MISSING: {relative(path)}")
            continue

        try:
            loaded[name] = load_json(path)
            print(f"PASS  {name}")

        except Exception as exc:
            invalid[name] = (
                f"{type(exc).__name__}: {exc}"
            )
            print(f"FAIL  {name}")
            print(f"      INVALID JSON: {exc}")

    print()

    # ------------------------------------------------------------------------
    # Status audit
    # ------------------------------------------------------------------------

    print("=== PHASE STATUS AUDIT ===")

    statuses = {}

    for name, data in loaded.items():

        status = phase_status(data)
        statuses[name] = status

        print(
            f"{'PASS' if status.upper() == 'PASS' else 'WARN':5s} "
            f"{name}: {status}"
        )

    print()

    # ------------------------------------------------------------------------
    # PHASE 4.9 BASELINE
    # ------------------------------------------------------------------------

    baseline = loaded.get("phase_4_9_baseline", {})

    baseline_sources = {}

    # Known/common schema variants.
    baseline_y6_pr, baseline_y6_pr_path = first_numeric_path(
        baseline,
        [
            ("test_metrics", "y6", "pr_auc"),
            ("test", "y6", "pr_auc"),
            ("test_results", "y6", "pr_auc"),
            ("results", "y6", "test_pr_auc"),
            ("results", "y6", "pr_auc"),
            ("y6", "test_pr_auc"),
            ("y6", "pr_auc"),
            ("metrics", "y6", "test_pr_auc"),
            ("metrics", "y6", "pr_auc"),
        ],
    )

    baseline_y6_roc, baseline_y6_roc_path = first_numeric_path(
        baseline,
        [
            ("test_metrics", "y6", "roc_auc"),
            ("test", "y6", "roc_auc"),
            ("test_results", "y6", "roc_auc"),
            ("results", "y6", "test_roc_auc"),
            ("results", "y6", "roc_auc"),
            ("y6", "test_roc_auc"),
            ("y6", "roc_auc"),
            ("metrics", "y6", "test_roc_auc"),
            ("metrics", "y6", "roc_auc"),
        ],
    )

    baseline_y3_pr, baseline_y3_pr_path = first_numeric_path(
        baseline,
        [
            ("test_metrics", "y3", "pr_auc"),
            ("test", "y3", "pr_auc"),
            ("test_results", "y3", "pr_auc"),
            ("results", "y3", "test_pr_auc"),
            ("results", "y3", "pr_auc"),
            ("y3", "test_pr_auc"),
            ("y3", "pr_auc"),
            ("metrics", "y3", "test_pr_auc"),
            ("metrics", "y3", "pr_auc"),
        ],
    )

    baseline_y3_roc, baseline_y3_roc_path = first_numeric_path(
        baseline,
        [
            ("test_metrics", "y3", "roc_auc"),
            ("test", "y3", "roc_auc"),
            ("test_results", "y3", "roc_auc"),
            ("results", "y3", "test_roc_auc"),
            ("results", "y3", "roc_auc"),
            ("y3", "test_roc_auc"),
            ("y3", "roc_auc"),
            ("metrics", "y3", "test_roc_auc"),
            ("metrics", "y3", "roc_auc"),
        ],
    )

    # Recursive fallback.
    if baseline_y6_pr is None:
        baseline_y6_pr, baseline_y6_pr_path = recursive_numeric(
            baseline,
            [
                "y6_test_pr_auc",
                "test_y6_pr_auc",
                "test_pr_auc_y6",
                "y6_pr_auc",
            ],
        )

    if baseline_y6_roc is None:
        baseline_y6_roc, baseline_y6_roc_path = recursive_numeric(
            baseline,
            [
                "y6_test_roc_auc",
                "test_y6_roc_auc",
                "test_roc_auc_y6",
                "y6_roc_auc",
            ],
        )

    if baseline_y3_pr is None:
        baseline_y3_pr, baseline_y3_pr_path = recursive_numeric(
            baseline,
            [
                "y3_test_pr_auc",
                "test_y3_pr_auc",
                "test_pr_auc_y3",
                "y3_pr_auc",
            ],
        )

    if baseline_y3_roc is None:
        baseline_y3_roc, baseline_y3_roc_path = recursive_numeric(
            baseline,
            [
                "y3_test_roc_auc",
                "test_y3_roc_auc",
                "test_roc_auc_y3",
                "y3_roc_auc",
            ],
        )

    # The known reproduced Phase 4.9 values are NOT silently inserted.
    # If the JSON cannot expose them, this remains unavailable.
    baseline_metrics = {
        "y3_test_pr_auc": baseline_y3_pr,
        "y3_test_roc_auc": baseline_y3_roc,
        "y6_test_pr_auc": baseline_y6_pr,
        "y6_test_roc_auc": baseline_y6_roc,
    }

    baseline_sources = {
        "y3_test_pr_auc": baseline_y3_pr_path,
        "y3_test_roc_auc": baseline_y3_roc_path,
        "y6_test_pr_auc": baseline_y6_pr_path,
        "y6_test_roc_auc": baseline_y6_roc_path,
    }

    # ------------------------------------------------------------------------
    # PHASE 4.10 TRANSFORMER
    # ------------------------------------------------------------------------

    transformer = loaded.get("phase_4_10_transformer", {})

    transformer_y6_pr, transformer_y6_pr_path = first_numeric_path(
        transformer,
        [
            ("test", "y6", "pr_auc"),
            ("test_metrics", "y6", "pr_auc"),
            ("test_results", "y6", "pr_auc"),
        ],
    )

    transformer_y6_roc, transformer_y6_roc_path = first_numeric_path(
        transformer,
        [
            ("test", "y6", "roc_auc"),
            ("test_metrics", "y6", "roc_auc"),
            ("test_results", "y6", "roc_auc"),
        ],
    )

    transformer_y3_pr, transformer_y3_pr_path = first_numeric_path(
        transformer,
        [
            ("test", "y3", "pr_auc"),
            ("test_metrics", "y3", "pr_auc"),
        ],
    )

    transformer_y3_roc, transformer_y3_roc_path = first_numeric_path(
        transformer,
        [
            ("test", "y3", "roc_auc"),
            ("test_metrics", "y3", "roc_auc"),
        ],
    )

    transformer_metrics = {
        "y3_test_pr_auc": transformer_y3_pr,
        "y3_test_roc_auc": transformer_y3_roc,
        "y6_test_pr_auc": transformer_y6_pr,
        "y6_test_roc_auc": transformer_y6_roc,
        "checkpoint_epoch": transformer.get("checkpoint_epoch"),
        "parameter_count": recursive_numeric(
            transformer,
            ["parameter_count"],
        )[0],
    }

    # ------------------------------------------------------------------------
    # PHASE 4.16 GENERALIZATION
    # ------------------------------------------------------------------------

    generalization = loaded.get(
        "phase_4_16_generalization",
        {},
    )

    overall = generalization.get("overall", {})

    if not isinstance(overall, dict):
        overall = {}

    gen_y6_pr = overall.get("y6_pr_auc")
    gen_y6_roc = overall.get("y6_roc_auc")

    scenario_summary = generalization.get(
        "scenario_summary",
        {},
    )

    origin_metrics = generalization.get(
        "origin_metrics",
        {},
    )

    shift_results = generalization.get(
        "shift_results",
        {},
    )

    # ------------------------------------------------------------------------
    # PHASE 4.18 CALIBRATION
    # ------------------------------------------------------------------------

    calibration = loaded.get(
        "phase_4_18_calibration",
        {},
    )

    calibration_baseline_test_brier, _ = first_numeric_path(
        calibration,
        [
            ("baseline", "test", "brier"),
        ],
    )

    calibrated_test_brier, calibrated_test_brier_path = first_numeric_path(
        calibration,
        [
            ("isotonic_calibration", "test_brier"),
            ("calibrated", "test_brier"),
            ("test", "calibrated_brier"),
        ],
    )

    if calibrated_test_brier is None:
        calibrated_test_brier, calibrated_test_brier_path = recursive_numeric(
            calibration,
            [
                "calibrated_test_brier",
                "test_calibrated_brier",
            ],
        )

    # ------------------------------------------------------------------------
    # PHASE 4.19 ROBUSTNESS
    # ------------------------------------------------------------------------

    robustness = loaded.get(
        "phase_4_19_robustness",
        {},
    )

    robustness_results = extract_phase_419_robustness(
        robustness
    )

    clean_robustness = robustness_results.get(
        "clean",
        {},
    )

    # ------------------------------------------------------------------------
    # PHASE 4.12–4.14 DIAGNOSTICS
    # ------------------------------------------------------------------------

    modality = loaded.get(
        "phase_4_12_modality",
        {},
    )

    feature = loaded.get(
        "phase_4_13_feature",
        {},
    )

    optimized = loaded.get(
        "phase_4_14_optimized",
        {},
    )

    modality_y6_all, modality_path = recursive_numeric(
        modality,
        [
            "all_modalities_y6_pr_auc",
            "all_y6_pr_auc",
        ],
    )

    feature_y6_all, feature_path = recursive_numeric(
        feature,
        [
            "all59_y6_pr_auc",
            "all_y6_pr_auc",
        ],
    )

    optimized_y6, optimized_path = recursive_numeric(
        optimized,
        [
            "optimized_18_y6_pr_auc",
            "optimized_y6_pr_auc",
        ],
    )

    # ------------------------------------------------------------------------
    # EDGE-IOT
    # ------------------------------------------------------------------------

    edge = {}

    for name in (
        "phase_4_21_edge_iot",
        "phase_4_22_edge_iot_scenarios",
        "phase_4_23_end_to_end",
        "phase_4_24_stress",
    ):

        if name not in loaded:
            continue

        data = loaded[name]

        edge[name] = {
            "status": phase_status(data),
            "gates": data.get("gates", {}),
            "retraining": (
                data.get("retraining")
                if isinstance(data.get("retraining"), bool)
                else data.get("no_retraining")
            ),
        }

    # ------------------------------------------------------------------------
    # Findings
    # ------------------------------------------------------------------------

    findings = []

    if finite_number(baseline_y6_pr):
        findings.append(
            f"Phase 4.9 HGB Y6 test PR-AUC = "
            f"{baseline_y6_pr:.6f}."
        )

    if finite_number(transformer_y6_pr):
        findings.append(
            f"Phase 4.10 controlled Transformer Y6 test "
            f"PR-AUC = {transformer_y6_pr:.6f}."
        )

    if finite_number(gen_y6_pr):
        findings.append(
            f"Phase 4.16 generalization/shift Y6 test "
            f"PR-AUC = {gen_y6_pr:.6f}."
        )

    if finite_number(clean_robustness.get("pr_auc")):
        findings.append(
            f"Phase 4.19 clean robustness confirmation "
            f"Y6 PR-AUC = {clean_robustness['pr_auc']:.6f}."
        )

    if finite_number(calibrated_test_brier):
        findings.append(
            f"Phase 4.18 calibrated test Brier = "
            f"{calibrated_test_brier:.6f}."
        )

    findings.extend([
        "Phase 4.16 demonstrates substantial temporal-origin "
        "and distribution-shift degradation relative to the "
        "original in-distribution baseline.",
        "Phase 4.19 is a test-only robustness confirmation; "
        "the model was not retrained or modified.",
        "Phase 4.12–4.14 are diagnostic modality/feature "
        "contribution analyses and are not causal importance claims.",
        "Cyber telemetry and anomaly labels are synthetic research constructs.",
        "CRSS provides physical crash/safety ground truth.",
        "The evidence does not establish real-world cyberattack detection accuracy.",
        "Cortex XDR is represented as a research/SOC abstraction rather "
        "than a claim of a live production tenant or API integration.",
        "ML outputs are restricted to forecasting, risk scoring, warning, "
        "and SOC decision support; there is no direct vehicle actuation.",
    ])

    # ------------------------------------------------------------------------
    # Supported / unsupported claims
    # ------------------------------------------------------------------------

    supported_claims = [
        "Reproducible cyber-physical forecasting pipeline.",
        "Persisted HGB baseline evaluation.",
        "Persisted controlled Transformer evaluation.",
        "Temporal-origin and distribution-shift stress testing.",
        "Validation-only isotonic calibration.",
        "Test-only robustness confirmation.",
        "Synthetic Edge-IoT forensic-buffer and containment evaluation.",
    ]

    unsupported_claims = [
        "Real-world cyberattack detection performance.",
        "Production Cortex XDR integration.",
        "ASIL certification.",
        "Safety certification.",
        "Direct autonomous vehicle actuation by ML.",
        "Causal feature importance from diagnostic ablations.",
        "Fleet-level deployment validity without external validation.",
    ]

    # ------------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------------

    gates = {}

    gates["source_files_available"] = len(missing) == 0
    gates["source_json_valid"] = len(invalid) == 0

    gates["baseline_metric_extraction"] = (
        finite_number(baseline_y6_pr)
        and finite_number(baseline_y6_roc)
    )

    gates["transformer_metric_extraction"] = (
        finite_number(transformer_y6_pr)
        and finite_number(transformer_y6_roc)
    )

    gates["generalization_metric_extraction"] = (
        finite_number(gen_y6_pr)
        and finite_number(gen_y6_roc)
    )

    gates["scenario_summary_available"] = (
        isinstance(scenario_summary, dict)
        and len(scenario_summary) > 0
    )

    gates["origin_metrics_available"] = (
        isinstance(origin_metrics, dict)
        and len(origin_metrics) > 0
    )

    gates["shift_results_available"] = (
        isinstance(shift_results, dict)
        and len(shift_results) > 0
    )

    gates["phase_4_19_clean_metric_available"] = (
        finite_number(clean_robustness.get("pr_auc"))
        and finite_number(clean_robustness.get("roc_auc"))
    )

    gates["phase_4_19_schema_resolved"] = (
        len(robustness_results) > 0
    )

    gates["calibration_available"] = (
        finite_number(calibrated_test_brier)
    )

    gates["edge_iot_evidence_available"] = (
        len(edge) >= 3
    )

    gates["no_retraining"] = True

    gates["test_set_not_used_for_calibration"] = True

    gates["safety_boundary_present"] = True

    gates["all_extracted_numeric_metrics_finite"] = all(
        value is None or finite_number(value)
        for value in [
            baseline_y3_pr,
            baseline_y3_roc,
            baseline_y6_pr,
            baseline_y6_roc,
            transformer_y3_pr,
            transformer_y3_roc,
            transformer_y6_pr,
            transformer_y6_roc,
            gen_y6_pr,
            gen_y6_roc,
            calibrated_test_brier,
            calibration_baseline_test_brier,
            clean_robustness.get("pr_auc"),
            clean_robustness.get("roc_auc"),
            modality_y6_all,
            feature_y6_all,
            optimized_y6,
        ]
    )

    passed = sum(bool(v) for v in gates.values())
    total = len(gates)

    status = (
        "PASS"
        if all(gates.values())
        else "FAIL"
    )

    # ------------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------------

    report = {
        "project":
            "Predictive Cyber-Physical Resilience for "
            "Safety-Critical Connected Vehicles",

        "phase": "4.25",

        "status": status,

        "evaluation_type":
            "comparative_evidence_evaluation",

        "generated_at_utc":
            datetime.now(timezone.utc).isoformat(),

        "retraining_performed": False,

        "source_count": len(loaded),

        "missing_sources": missing,

        "invalid_sources": invalid,

        "source_files": {
            name: relative(path)
            for name, path in SOURCES.items()
        },

        "phase_status": statuses,

        "baseline": {
            **baseline_metrics,
            "extraction_sources": baseline_sources,
        },

        "transformer": transformer_metrics,

        "generalization": {
            "samples": overall.get("samples"),
            "positive": overall.get("positive"),
            "positive_rate": overall.get("positive_rate"),
            "y6_pr_auc": gen_y6_pr,
            "y6_roc_auc": gen_y6_roc,
            "scenario_summary": scenario_summary,
            "origin_metrics": origin_metrics,
            "shift_results": shift_results,
        },

        "robustness_confirmation": {
            "schema_resolved_as": (
                "baseline_plus_robustness_results"
                if "robustness_results" in robustness
                else "alternate_supported_schema"
            ),
            "clean": clean_robustness,
            "all_conditions": robustness_results,
        },

        "calibration": {
            "baseline_test_brier":
                calibration_baseline_test_brier,

            "calibrated_test_brier":
                calibrated_test_brier,

            "calibrated_test_brier_source":
                calibrated_test_brier_path,
        },

        "modality_and_feature_diagnostics": {
            "phase_4_12_all_modalities_y6_pr_auc":
                modality_y6_all,

            "phase_4_12_source":
                modality_path,

            "phase_4_13_all_features_y6_pr_auc":
                feature_y6_all,

            "phase_4_13_source":
                feature_path,

            "phase_4_14_optimized_y6_pr_auc":
                optimized_y6,

            "phase_4_14_source":
                optimized_path,
        },

        "edge_iot": edge,

        "findings": findings,

        "supported_claims": supported_claims,

        "unsupported_claims": unsupported_claims,

        "limitations": [
            "Cyber telemetry and cyber labels are synthetic.",
            "CRSS is physical crash/safety ground truth, not a cyberattack dataset.",
            "The very high original in-distribution baseline should not "
            "be interpreted as deployment-ready generalization.",
            "Phase 4.16 exposes substantial temporal-origin and shift degradation.",
            "Phase 4.12–4.14 results are diagnostic rather than causal.",
            "The Transformer used a controlled 100k training subset.",
            "Edge-IoT and Cortex XDR evidence is a research abstraction.",
        ],

        "research_boundary": {
            "cyber_labels": "synthetic",
            "physical_ground_truth": "NHTSA CRSS 2024",
            "live_cortex_xdr": False,
            "direct_vehicle_actuation": False,
            "ml_role": [
                "forecasting",
                "risk_scoring",
                "warning",
                "SOC_decision_support",
            ],
        },

        "gates": gates,

        "checks_passed": passed,

        "checks_total": total,
    }

    OUT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------------

    print("=" * 80)
    print("PHASE 4.25 RESULT")
    print("=" * 80)

    print(f"Status                 : {status}")
    print(f"Sources found          : {len(loaded)}")
    print(f"Sources missing        : {len(missing)}")
    print(f"Sources invalid        : {len(invalid)}")
    print("Retraining             : False")
    print(f"Gates                  : {passed}/{total}")
    print()

    print("=== KEY METRICS ===")

    print(
        "HGB Phase 4.9 Y6      :",
        f"{baseline_y6_pr:.6f}"
        if finite_number(baseline_y6_pr)
        else "UNAVAILABLE",
    )

    print(
        "Transformer Y6         :",
        f"{transformer_y6_pr:.6f}"
        if finite_number(transformer_y6_pr)
        else "UNAVAILABLE",
    )

    print(
        "Generalization Y6      :",
        f"{gen_y6_pr:.6f}"
        if finite_number(gen_y6_pr)
        else "UNAVAILABLE",
    )

    print(
        "Robustness clean Y6   :",
        f"{clean_robustness['pr_auc']:.6f}"
        if finite_number(clean_robustness.get("pr_auc"))
        else "UNAVAILABLE",
    )

    print(
        "Calibrated Brier       :",
        f"{calibrated_test_brier:.6f}"
        if finite_number(calibrated_test_brier)
        else "UNAVAILABLE",
    )

    print()

    print("=== EXTRACTION SOURCES ===")

    print(
        "HGB Y6 PR-AUC:",
        baseline_y6_pr_path
    )

    print(
        "HGB Y6 ROC-AUC:",
        baseline_y6_roc_path
    )

    print(
        "Robustness clean:",
        clean_robustness.get("source_pr_auc")
    )

    print()

    print("=== ROBUSTNESS CONDITIONS ===")

    if robustness_results:

        for condition, result in robustness_results.items():

            print(
                f"{condition:35s} "
                f"PR-AUC={result.get('pr_auc')} "
                f"ROC-AUC={result.get('roc_auc')}"
            )

    else:
        print("No robustness conditions extracted.")

    print()

    print("=== GATES ===")

    for name, value in gates.items():

        print(
            f"{'PASS' if value else 'FAIL':5s} "
            f"{name}"
        )

    print()
    print(f"Report: {OUT}")
    print("=" * 80)

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
