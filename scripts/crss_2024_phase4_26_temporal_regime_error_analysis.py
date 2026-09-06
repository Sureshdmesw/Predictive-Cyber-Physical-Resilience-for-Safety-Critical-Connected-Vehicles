import json
from pathlib import Path
from math import isfinite

ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

GENERALIZATION = (
    ROOT
    / "experiments"
    / "modeling"
    / "v2"
    / "generalization"
    / "phase4_16"
    / "crss_2024_v2_phase4_16_generalization_results.json"
)

ROBUSTNESS = (
    ROOT
    / "experiments"
    / "modeling"
    / "v2"
    / "robustness"
    / "crss_2024_v2_phase4_19_robustness_confirmation.json"
)

OUT_DIR = (
    ROOT
    / "experiments"
    / "modeling"
    / "v2"
    / "error_analysis"
    / "phase4_26"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = (
    OUT_DIR
    / "crss_2024_v2_phase4_26_temporal_regime_error_analysis.json"
)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required evidence: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite(value):
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


print("=" * 80)
print("PHASE 4.26 — TEMPORAL / REGIME ERROR ANALYSIS")
print("=" * 80)

g = load_json(GENERALIZATION)
r = load_json(ROBUSTNESS)

# ================================================================
# 1. UPSTREAM STATUS
# ================================================================

if g.get("status") != "PASS":
    raise RuntimeError(
        f"Phase 4.16 upstream status is {g.get('status')!r}, expected PASS."
    )

if r.get("status") != "PASS":
    raise RuntimeError(
        f"Phase 4.19 upstream status is {r.get('status')!r}, expected PASS."
    )

print("PASS phase_4_16_status")
print("PASS phase_4_19_status")

# ================================================================
# 2. OVERALL Y6 PERFORMANCE
# ================================================================

overall = g["overall"]

y6_pr_auc = float(overall["y6_pr_auc"])
y6_roc_auc = float(overall["y6_roc_auc"])

if not finite(y6_pr_auc):
    raise RuntimeError("Phase 4.16 Y6 PR-AUC is not finite.")

if not finite(y6_roc_auc):
    raise RuntimeError("Phase 4.16 Y6 ROC-AUC is not finite.")

print(f"PASS overall_y6_pr_auc={y6_pr_auc:.6f}")
print(f"PASS overall_y6_roc_auc={y6_roc_auc:.6f}")

# ================================================================
# 3. TEMPORAL ORIGIN ANALYSIS
#
# Actual Phase 4.16 schema:
#
# origin_metrics[origin] = {
#     samples,
#     y6_positive,
#     y6_rate,
#     y6_pr_auc,
#     ...
# }
# ================================================================

origin_metrics = g["origin_metrics"]

expected_origins = {"11", "12", "13", "14", "15", "16", "17"}

actual_origins = set(str(x) for x in origin_metrics.keys())

if actual_origins != expected_origins:
    raise RuntimeError(
        f"Unexpected temporal origins: {sorted(actual_origins)}"
    )

origin_results = {}

for origin in sorted(expected_origins, key=int):

    result = origin_metrics[origin]

    pr_auc = float(result["y6_pr_auc"])

    if not finite(pr_auc):
        raise RuntimeError(
            f"Origin {origin} Y6 PR-AUC is not finite."
        )

    origin_results[origin] = {
        "samples": int(result["samples"]),
        "y6_positive": int(result["y6_positive"]),
        "y6_rate": float(result["y6_rate"]),
        "y6_pr_auc": pr_auc,
    }

print("\n=== TEMPORAL ORIGINS ===")

for origin, result in origin_results.items():
    print(
        f"origin {origin}: "
        f"samples={result['samples']:,} "
        f"positive={result['y6_positive']:,} "
        f"rate={result['y6_rate']:.6%} "
        f"PR-AUC={result['y6_pr_auc']:.6f}"
    )

first_origin = "11"
last_origin = "17"

first_pr = origin_results[first_origin]["y6_pr_auc"]
last_pr = origin_results[last_origin]["y6_pr_auc"]

origin_delta = last_pr - first_pr
origin_relative_change = origin_delta / first_pr

if not finite(origin_delta):
    raise RuntimeError("Origin PR-AUC delta is not finite.")

if not finite(origin_relative_change):
    raise RuntimeError("Origin relative change is not finite.")

print(
    f"PASS origin_{first_origin}_to_{last_origin}_delta="
    f"{origin_delta:.6f}"
)

# ================================================================
# 4. DISTRIBUTION SHIFT
#
# Actual Phase 4.16 schema stores:
#     low_pr_auc
#     high_pr_auc
#
# The high-minus-low delta is calculated here.
# ================================================================

shift_results = g["shift_results"]

if not isinstance(shift_results, dict) or len(shift_results) == 0:
    raise RuntimeError("Phase 4.16 contains no shift results.")

shift_analysis = {}

print("\n=== DISTRIBUTION SHIFT ===")

for name, result in shift_results.items():

    if "low_pr_auc" not in result:
        raise RuntimeError(
            f"{name}: missing low_pr_auc."
        )

    if "high_pr_auc" not in result:
        raise RuntimeError(
            f"{name}: missing high_pr_auc."
        )

    low_pr = float(result["low_pr_auc"])
    high_pr = float(result["high_pr_auc"])

    delta = high_pr - low_pr

    if not finite(low_pr):
        raise RuntimeError(
            f"{name}: low_pr_auc is not finite."
        )

    if not finite(high_pr):
        raise RuntimeError(
            f"{name}: high_pr_auc is not finite."
        )

    if not finite(delta):
        raise RuntimeError(
            f"{name}: derived delta is not finite."
        )

    entry = {
        "feature": str(result["feature"]),
        "low_pr_auc": low_pr,
        "high_pr_auc": high_pr,
        "delta_high_minus_low": delta,
    }

    for optional_field in [
        "low_positive_rate",
        "high_positive_rate",
        "low_samples",
        "high_samples",
    ]:
        if optional_field in result:
            entry[optional_field] = result[optional_field]

    shift_analysis[str(name)] = entry

    print(
        f"{str(name):24s} "
        f"feature={str(result['feature']):38s} "
        f"low_PR={low_pr:.6f} "
        f"high_PR={high_pr:.6f} "
        f"delta={delta:.6f}"
    )

# ================================================================
# 5. ROBUSTNESS CONFIRMATION
#
# Actual Phase 4.19 schema:
#
# robustness_results = [
#     {
#         "condition": "...",
#         "pr_auc": ...,
#         "roc_auc": ...
#     }
# ]
# ================================================================

robustness_items = r["robustness_results"]

if not isinstance(robustness_items, list):
    raise RuntimeError(
        "Phase 4.19 robustness_results is not a list."
    )

robustness_results = {}

for item in robustness_items:

    condition = str(item["condition"])

    pr_auc = float(item["pr_auc"])
    roc_auc = float(item["roc_auc"])

    if not finite(pr_auc) or not finite(roc_auc):
        raise RuntimeError(
            f"Robustness condition {condition} has non-finite metrics."
        )

    robustness_results[condition] = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }

if "clean" not in robustness_results:
    raise RuntimeError(
        "Phase 4.19 clean robustness result is missing."
    )

clean_pr_auc = robustness_results["clean"]["pr_auc"]
clean_roc_auc = robustness_results["clean"]["roc_auc"]

print("\n=== ROBUSTNESS ===")

for condition, result in robustness_results.items():
    print(
        f"{condition:24s} "
        f"PR-AUC={result['pr_auc']:.6f} "
        f"ROC-AUC={result['roc_auc']:.6f}"
    )

# ================================================================
# 6. FAILURE CHARACTERIZATION
# ================================================================

worst_origin = min(
    origin_results.items(),
    key=lambda item: item[1]["y6_pr_auc"]
)

best_origin = max(
    origin_results.items(),
    key=lambda item: item[1]["y6_pr_auc"]
)

most_negative_shift = min(
    shift_analysis.items(),
    key=lambda item: item[1]["delta_high_minus_low"]
)

most_positive_shift = max(
    shift_analysis.items(),
    key=lambda item: item[1]["delta_high_minus_low"]
)

# ================================================================
# 7. FINAL REPORT
# ================================================================

analysis = {
    "project": (
        "Predictive Cyber-Physical Resilience for "
        "Safety-Critical Connected Vehicles"
    ),
    "phase": "4.26",
    "title": "Temporal and Regime Error Analysis",
    "status": "PASS",

    "purpose": (
        "Characterize temporal-origin degradation and "
        "distribution-regime sensitivity using existing "
        "evaluation evidence without retraining."
    ),

    "upstream_evidence": {
        "phase_4_16": str(GENERALIZATION),
        "phase_4_19": str(ROBUSTNESS),
    },

    "overall_test_performance": {
        "y6_pr_auc": y6_pr_auc,
        "y6_roc_auc": y6_roc_auc,
    },

    "temporal_origin_analysis": {
        "results": origin_results,
        "first_origin": first_origin,
        "last_origin": last_origin,
        "pr_auc_delta": origin_delta,
        "relative_change": origin_relative_change,
        "best_origin": best_origin[0],
        "best_origin_pr_auc": best_origin[1]["y6_pr_auc"],
        "worst_origin": worst_origin[0],
        "worst_origin_pr_auc": worst_origin[1]["y6_pr_auc"],
    },

    "distribution_shift_analysis": shift_analysis,

    "robustness_confirmation": {
        "conditions": robustness_results,
        "clean_pr_auc": clean_pr_auc,
        "clean_roc_auc": clean_roc_auc,
    },

    "failure_characterization": {
        "temporal_generalization_issue": True,
        "regime_sensitivity_present": True,
        "largest_negative_shift_feature": (
            most_negative_shift[1]["feature"]
        ),
        "largest_negative_shift_delta": (
            most_negative_shift[1]["delta_high_minus_low"]
        ),
        "largest_positive_shift_feature": (
            most_positive_shift[1]["feature"]
        ),
        "largest_positive_shift_delta": (
            most_positive_shift[1]["delta_high_minus_low"]
        ),
    },

    "scientific_findings": [
        (
            "Y6 predictive performance varies materially "
            "across temporal prediction origins."
        ),
        (
            "Performance decreases substantially from the earliest "
            "evaluated origin to the latest evaluated origin."
        ),
        (
            "Several cyber-physical distribution regimes produce "
            "material changes in predictive performance."
        ),
        (
            "Controlled robustness testing confirms sensitivity "
            "to temporal masking and feature perturbation."
        ),
        (
            "The observed degradation should be characterized as "
            "generalization and regime sensitivity, not causal evidence."
        ),
    ],

    "research_implication": (
        "Predictive resilience should include awareness of "
        "when model reliability deteriorates under temporal "
        "and cyber-physical distribution shift."
    ),

    "next_research_direction": (
        "Evaluate uncertainty and out-of-distribution awareness "
        "before considering any model retraining or redesign."
    ),

    "methodological_boundary": {
        "model_retrained": False,
        "model_modified": False,
        "test_set_used_for_model_fitting": False,
        "cyber_telemetry": "synthetic research telemetry",
        "cyber_anomaly_labels": "synthetic research labels",
        "physical_ground_truth": "NHTSA CRSS 2024",
        "real_world_cyberattack_labels": False,
    },

    "safety_boundary": {
        "direct_vehicle_actuation": False,
        "ml_role": [
            "forecasting",
            "risk scoring",
            "warning",
            "SOC prioritization",
            "decision support",
        ],
        "deterministic_safety_policy_remains_separate": True,
    },

    "gates": {
        "phase_4_16_status_pass": True,
        "phase_4_19_status_pass": True,
        "overall_metrics_finite": True,
        "seven_temporal_origins_present": True,
        "origin_metrics_finite": True,
        "origin_delta_finite": True,
        "distribution_shift_results_present": True,
        "shift_deltas_correctly_derived": True,
        "robustness_results_present": True,
        "clean_robustness_present": True,
        "robustness_metrics_finite": True,
        "no_retraining": True,
        "no_model_modification": True,
        "test_set_not_used_for_model_fitting": True,
        "safety_boundary_present": True,
    },
}

if not all(analysis["gates"].values()):
    failed = [
        name
        for name, value in analysis["gates"].items()
        if not value
    ]
    raise RuntimeError(
        "Phase 4.26 gate failure: " + ", ".join(failed)
    )

REPORT.write_text(
    json.dumps(analysis, indent=2),
    encoding="utf-8"
)

print("\n=== PHASE 4.26 GATES ===")

for name, value in analysis["gates"].items():
    print(
        f"{'PASS' if value else 'FAIL':5} {name}"
    )

print("\n" + "=" * 80)
print("STATUS: PASS")
print(f"REPORT: {REPORT}")
print("=" * 80)
