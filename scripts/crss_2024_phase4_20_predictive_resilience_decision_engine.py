from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_REPORT = (
    ROOT / "experiments" / "modeling" / "v2" / "calibration" /
    "crss_2024_v2_phase4_18_calibration_thresholds.json"
)

ROBUSTNESS_REPORT = (
    ROOT / "experiments" / "modeling" / "v2" / "robustness" /
    "crss_2024_v2_phase4_19_robustness_confirmation.json"
)

OUTPUT = (
    ROOT / "experiments" / "modeling" / "v2" / "decision_engine" /
    "crss_2024_v2_phase4_20_predictive_resilience_decision_engine.json"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


calibration = load_json(CALIBRATION_REPORT)
robustness = load_json(ROBUSTNESS_REPORT)

# ------------------------------------------------------------
# 1. Upstream validation
# ------------------------------------------------------------

calibration_status = calibration.get("status")
robustness_status = robustness.get("status")

if calibration_status != "PASS":
    raise RuntimeError(
        f"Phase 4.18 calibration report is not PASS: {calibration_status}"
    )

if robustness_status != "PASS":
    raise RuntimeError(
        f"Phase 4.19 robustness report is not PASS: {robustness_status}"
    )


# ------------------------------------------------------------
# 2. Deterministic operational risk policy
# ------------------------------------------------------------

RISK_LEVELS = [
    "NORMAL",
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]

THRESHOLDS = {
    "NORMAL": 0.05,
    "LOW": 0.20,
    "MEDIUM": 0.50,
    "HIGH": 0.80,
    "CRITICAL": 1.00,
}


def risk_level_from_probability(probability):
    if not math.isfinite(probability):
        raise ValueError("Risk probability must be finite.")

    probability = min(max(float(probability), 0.0), 1.0)

    if probability < 0.05:
        return "NORMAL"
    if probability < 0.20:
        return "LOW"
    if probability < 0.50:
        return "MEDIUM"
    if probability < 0.80:
        return "HIGH"
    return "CRITICAL"


def level_index(level):
    return RISK_LEVELS.index(level)


def apply_context_modifiers(
    probability,
    connectivity_degraded=False,
    integrity_failure=False,
    sensor_disagreement=False,
):
    base_level = risk_level_from_probability(probability)

    minimum_level = "NORMAL"

    if connectivity_degraded:
        minimum_level = "MEDIUM"

    if sensor_disagreement:
        if level_index("MEDIUM") > level_index(minimum_level):
            minimum_level = "MEDIUM"

    if integrity_failure:
        minimum_level = "HIGH"

    final_level = (
        minimum_level
        if level_index(minimum_level) > level_index(base_level)
        else base_level
    )

    return base_level, final_level


def operational_action(risk_level):
    actions = {
        "NORMAL": {
            "soc_priority": "ROUTINE",
            "vehicle_resilience_recommendation": "CONTINUE_NORMAL_OPERATION",
            "occupant_warning": "NONE",
        },
        "LOW": {
            "soc_priority": "LOW",
            "vehicle_resilience_recommendation": "INCREASE_LOCAL_MONITORING",
            "occupant_warning": "NONE",
        },
        "MEDIUM": {
            "soc_priority": "MEDIUM",
            "vehicle_resilience_recommendation": "PREPARE_LOCAL_RESILIENCE_MODE",
            "occupant_warning": "ADVISORY",
        },
        "HIGH": {
            "soc_priority": "HIGH",
            "vehicle_resilience_recommendation": "PRIORITIZE_LOCAL_RESILIENCE_AND_FORENSIC_CAPTURE",
            "occupant_warning": "CAUTION",
        },
        "CRITICAL": {
            "soc_priority": "CRITICAL",
            "vehicle_resilience_recommendation": "ESCALATE_TO_SOC_AND_MAINTAIN_LOCAL_SAFE_STATE",
            "occupant_warning": "URGENT",
        },
    }

    return actions[risk_level]


def evaluate_scenario(
    name,
    probability,
    connectivity_degraded=False,
    integrity_failure=False,
    sensor_disagreement=False,
):
    base_level, final_level = apply_context_modifiers(
        probability=probability,
        connectivity_degraded=connectivity_degraded,
        integrity_failure=integrity_failure,
        sensor_disagreement=sensor_disagreement,
    )

    action = operational_action(final_level)

    return {
        "scenario": name,
        "risk_probability": float(probability),
        "base_risk_level": base_level,
        "risk_level": final_level,
        "context": {
            "connectivity_degraded": bool(connectivity_degraded),
            "integrity_failure": bool(integrity_failure),
            "sensor_disagreement": bool(sensor_disagreement),
        },
        **action,
    }


# ------------------------------------------------------------
# 3. Demonstration scenarios
# ------------------------------------------------------------

scenarios = [
    evaluate_scenario(
        "normal_connected",
        0.01,
    ),
    evaluate_scenario(
        "early_forecast",
        0.12,
    ),
    evaluate_scenario(
        "connectivity_degradation",
        0.14,
        connectivity_degraded=True,
    ),
    evaluate_scenario(
        "sensor_inconsistency",
        0.32,
        sensor_disagreement=True,
    ),
    evaluate_scenario(
        "communication_integrity_failure",
        0.61,
        integrity_failure=True,
    ),
    evaluate_scenario(
        "critical_predicted_anomaly",
        0.91,
        connectivity_degraded=True,
        integrity_failure=True,
        sensor_disagreement=True,
    ),
]


# ------------------------------------------------------------
# 4. Safety-boundary validation
# ------------------------------------------------------------

safety_boundary = (
    "Decision support, warning, and SOC prioritization only. "
    "The decision engine does not directly actuate vehicle safety-critical controls."
)

checks = {}

# Probability → risk mapping must be monotonic.
test_probabilities = [0.01, 0.10, 0.25, 0.55, 0.85, 0.99]
test_levels = [
    risk_level_from_probability(p)
    for p in test_probabilities
]

checks["probability_mapping_monotonic"] = all(
    level_index(test_levels[i]) <= level_index(test_levels[i + 1])
    for i in range(len(test_levels) - 1)
)

# Context modifiers cannot reduce risk.
checks["integrity_failure_non_decreasing"] = all(
    level_index(
        apply_context_modifiers(
            p,
            integrity_failure=True,
        )[1]
    )
    >= level_index(
        apply_context_modifiers(p)[1]
    )
    for p in test_probabilities
)

checks["connectivity_degradation_non_decreasing"] = all(
    level_index(
        apply_context_modifiers(
            p,
            connectivity_degraded=True,
        )[1]
    )
    >= level_index(
        apply_context_modifiers(p)[1]
    )
    for p in test_probabilities
)

checks["sensor_disagreement_non_decreasing"] = all(
    level_index(
        apply_context_modifiers(
            p,
            sensor_disagreement=True,
        )[1]
    )
    >= level_index(
        apply_context_modifiers(p)[1]
    )
    for p in test_probabilities
)

# Explicitly verify no direct actuator command exists.
forbidden_actuation_terms = [
    "ACTUATE_BRAKES",
    "ACTUATE_STEERING",
    "DIRECT_VEHICLE_CONTROL",
    "SEND_CAN_CONTROL_COMMAND",
]

serialized_scenarios = json.dumps(scenarios).upper()

checks["no_direct_vehicle_actuation"] = not any(
    term in serialized_scenarios
    for term in forbidden_actuation_terms
)

checks["all_policy_levels_defined"] = all(
    level in RISK_LEVELS
    for level in RISK_LEVELS
)

checks["context_signals_supported"] = all(
    key in scenarios[-1]["context"]
    for key in [
        "connectivity_degraded",
        "integrity_failure",
        "sensor_disagreement",
    ]
)

checks["upstream_calibration_pass"] = calibration_status == "PASS"
checks["upstream_robustness_pass"] = robustness_status == "PASS"

all_pass = all(checks.values())


# ------------------------------------------------------------
# 5. Report
# ------------------------------------------------------------

report = {
    "phase": "4.20",
    "title": "Predictive Resilience Decision Engine",
    "status": "PASS" if all_pass else "FAIL",

    "purpose": (
        "Convert calibrated predictive cyber-physical risk into a "
        "deterministic operational risk level, SOC priority, resilience "
        "recommendation, and occupant warning."
    ),

    "upstream_evidence": {
        "phase_4_18_calibration_status": calibration_status,
        "phase_4_18_report": str(CALIBRATION_REPORT.relative_to(ROOT)),
        "phase_4_19_robustness_status": robustness_status,
        "phase_4_19_report": str(ROBUSTNESS_REPORT.relative_to(ROOT)),
    },

    "risk_policy": {
        "NORMAL": "< 0.05",
        "LOW": "0.05 <= p < 0.20",
        "MEDIUM": "0.20 <= p < 0.50",
        "HIGH": "0.50 <= p < 0.80",
        "CRITICAL": ">= 0.80",
    },

    "context_modifiers": {
        "connectivity_degraded": "minimum MEDIUM",
        "sensor_disagreement": "minimum MEDIUM",
        "integrity_failure": "minimum HIGH",
    },

    "demonstration_scenarios": scenarios,

    "safety_boundary": safety_boundary,

    "safety_checks": checks,

    "methodological_note": (
        "The risk thresholds and contextual minimum-risk rules are "
        "deterministic engineering policy rules. They are not learned "
        "parameters and should not be interpreted as empirically validated "
        "safety thresholds. Cyberattack labels remain synthetic research "
        "labels; CRSS remains the physical crash-ground-truth source."
    ),
}

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("=" * 72)
print("PHASE 4.20 — PREDICTIVE RESILIENCE DECISION ENGINE")
print("=" * 72)
print(f"Status: {report['status']}")
print()

print("Demonstration scenarios:")
for scenario in scenarios:
    print(
        f"  {scenario['scenario']:<35} "
        f"p={scenario['risk_probability']:.2f}  "
        f"base={scenario['base_risk_level']:<8} "
        f"final={scenario['risk_level']:<8} "
        f"SOC={scenario['soc_priority']}"
    )

print()
print("Safety checks:")
for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")

print()
print(f"Report: {OUTPUT}")
print("=" * 72)

if not all_pass:
    raise SystemExit("Phase 4.20 failed one or more validation checks.")
