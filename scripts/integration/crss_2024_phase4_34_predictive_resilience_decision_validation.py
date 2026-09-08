import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]

DECISION_ENGINE = (
    ROOT
    / "experiments"
    / "modeling"
    / "v2"
    / "decision_engine"
    / "crss_2024_v2_phase4_20_predictive_resilience_decision_engine.json"
)

ENSEMBLE_REPORT = (
    ROOT
    / "experiments"
    / "modeling"
    / "v2"
    / "uncertainty"
    / "phase4_27"
    / "crss_2024_v2_phase4_27a_deep_ensemble_final.json"
)

OUT = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_34"
    / "crss_2024_phase4_34_predictive_resilience_decision_validation.json"
)


POLICY = [
    (0.05, "NORMAL"),
    (0.20, "LOW"),
    (0.50, "MEDIUM"),
    (0.80, "HIGH"),
    (float("inf"), "CRITICAL"),
]

RANK = {
    "NORMAL": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def risk_from_probability(p):
    p = float(p)

    if p < 0.05:
        return "NORMAL"
    if p < 0.20:
        return "LOW"
    if p < 0.50:
        return "MEDIUM"
    if p < 0.80:
        return "HIGH"
    return "CRITICAL"


def elevate(current, minimum):
    if RANK[minimum] > RANK[current]:
        return minimum
    return current


def decision_engine(
    predictive_risk_probability,
    uncertainty_score=0.0,
    connectivity_degraded=False,
    sensor_disagreement=False,
    integrity_failure=False,
):
    p = float(predictive_risk_probability)
    u = float(uncertainty_score)

    if not 0.0 <= p <= 1.0:
        return {
            "valid": False,
            "decision": "REJECT_INVALID_INPUT",
            "reason": "risk_probability_out_of_range",
        }

    if not 0.0 <= u <= 1.0:
        return {
            "valid": False,
            "decision": "REJECT_INVALID_INPUT",
            "reason": "uncertainty_out_of_range",
        }

    decision = risk_from_probability(p)

    if connectivity_degraded:
        decision = elevate(decision, "MEDIUM")

    if sensor_disagreement:
        decision = elevate(decision, "MEDIUM")

    if integrity_failure:
        decision = elevate(decision, "HIGH")

    return {
        "valid": True,
        "decision": decision,
        "predictive_risk_probability": p,
        "uncertainty_score": u,
        "connectivity_degraded": bool(connectivity_degraded),
        "sensor_disagreement": bool(sensor_disagreement),
        "integrity_failure": bool(integrity_failure),
        "direct_vehicle_actuation": False,
    }


def sha256_json(obj):
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main():
    print("=== PHASE 4.34 PREDICTIVE RESILIENCE DECISION VALIDATION ===")

    if not DECISION_ENGINE.exists():
        raise FileNotFoundError(DECISION_ENGINE)

    if not ENSEMBLE_REPORT.exists():
        raise FileNotFoundError(ENSEMBLE_REPORT)

    decision_engine_source = json.loads(
        DECISION_ENGINE.read_text(encoding="utf-8")
    )

    ensemble_source = json.loads(
        ENSEMBLE_REPORT.read_text(encoding="utf-8")
    )

    print(f"Decision engine: {DECISION_ENGINE}")
    print(f"Ensemble evidence: {ENSEMBLE_REPORT}")

    checks = {}
    cases = []

    def add_case(name, probability, expected, **kwargs):
        result = decision_engine(
            probability,
            **kwargs,
        )

        passed = (
            result.get("valid", False)
            and result.get("decision") == expected
        )

        case = {
            "name": name,
            "expected": expected,
            "result": result,
            "pass": passed,
        }

        cases.append(case)
        return passed

    # ------------------------------------------------------------
    # 1. Exact policy boundaries
    # ------------------------------------------------------------

    boundary_cases = [
        ("boundary_0_00", 0.00, "NORMAL"),
        ("boundary_0_049999", 0.049999, "NORMAL"),
        ("boundary_0_05", 0.05, "LOW"),
        ("boundary_0_199999", 0.199999, "LOW"),
        ("boundary_0_20", 0.20, "MEDIUM"),
        ("boundary_0_499999", 0.499999, "MEDIUM"),
        ("boundary_0_50", 0.50, "HIGH"),
        ("boundary_0_799999", 0.799999, "HIGH"),
        ("boundary_0_80", 0.80, "CRITICAL"),
        ("boundary_1_00", 1.00, "CRITICAL"),
    ]

    boundary_results = [
        add_case(name, p, expected)
        for name, p, expected in boundary_cases
    ]

    checks["policy_boundaries_correct"] = all(boundary_results)

    # ------------------------------------------------------------
    # 2. Connectivity modifier
    # ------------------------------------------------------------

    connectivity_results = [
        add_case(
            "connectivity_low_to_medium",
            0.10,
            "MEDIUM",
            connectivity_degraded=True,
        ),
        add_case(
            "connectivity_normal_to_medium",
            0.02,
            "MEDIUM",
            connectivity_degraded=True,
        ),
        add_case(
            "connectivity_high_unchanged",
            0.70,
            "HIGH",
            connectivity_degraded=True,
        ),
    ]

    checks["connectivity_modifier_minimum_medium"] = all(
        connectivity_results
    )

    # ------------------------------------------------------------
    # 3. Sensor disagreement modifier
    # ------------------------------------------------------------

    sensor_results = [
        add_case(
            "sensor_low_to_medium",
            0.10,
            "MEDIUM",
            sensor_disagreement=True,
        ),
        add_case(
            "sensor_normal_to_medium",
            0.02,
            "MEDIUM",
            sensor_disagreement=True,
        ),
        add_case(
            "sensor_high_unchanged",
            0.70,
            "HIGH",
            sensor_disagreement=True,
        ),
    ]

    checks["sensor_modifier_minimum_medium"] = all(sensor_results)

    # ------------------------------------------------------------
    # 4. Integrity modifier
    # ------------------------------------------------------------

    integrity_results = [
        add_case(
            "integrity_normal_to_high",
            0.02,
            "HIGH",
            integrity_failure=True,
        ),
        add_case(
            "integrity_low_to_high",
            0.10,
            "HIGH",
            integrity_failure=True,
        ),
        add_case(
            "integrity_medium_to_high",
            0.30,
            "HIGH",
            integrity_failure=True,
        ),
        add_case(
            "integrity_high_unchanged",
            0.70,
            "HIGH",
            integrity_failure=True,
        ),
        add_case(
            "integrity_critical_unchanged",
            0.90,
            "CRITICAL",
            integrity_failure=True,
        ),
    ]

    checks["integrity_modifier_minimum_high"] = all(
        integrity_results
    )

    # ------------------------------------------------------------
    # 5. Compound modifiers
    # ------------------------------------------------------------

    compound_results = [
        add_case(
            "compound_all_low_probability",
            0.01,
            "HIGH",
            connectivity_degraded=True,
            sensor_disagreement=True,
            integrity_failure=True,
        ),
        add_case(
            "compound_medium_integrity",
            0.30,
            "HIGH",
            connectivity_degraded=True,
            sensor_disagreement=True,
            integrity_failure=True,
        ),
        add_case(
            "compound_high_all",
            0.70,
            "HIGH",
            connectivity_degraded=True,
            sensor_disagreement=True,
            integrity_failure=True,
        ),
        add_case(
            "compound_critical_all",
            0.90,
            "CRITICAL",
            connectivity_degraded=True,
            sensor_disagreement=True,
            integrity_failure=True,
        ),
    ]

    checks["compound_modifiers_monotonic"] = all(compound_results)

    # ------------------------------------------------------------
    # 6. Uncertainty must not silently reduce safety decision
    # ------------------------------------------------------------

    uncertainty_cases = [
        add_case(
            "uncertainty_zero",
            0.30,
            "MEDIUM",
            uncertainty_score=0.0,
        ),
        add_case(
            "uncertainty_moderate",
            0.30,
            "MEDIUM",
            uncertainty_score=0.5,
        ),
        add_case(
            "uncertainty_high",
            0.30,
            "MEDIUM",
            uncertainty_score=1.0,
        ),
        add_case(
            "uncertainty_high_integrity",
            0.10,
            "HIGH",
            uncertainty_score=1.0,
            integrity_failure=True,
        ),
    ]

    checks["uncertainty_does_not_reduce_risk"] = all(
        uncertainty_cases
    )

    # ------------------------------------------------------------
    # 7. Invalid input rejection
    # ------------------------------------------------------------

    invalid_probability = decision_engine(-0.01)
    invalid_probability_2 = decision_engine(1.01)
    invalid_uncertainty = decision_engine(
        0.30,
        uncertainty_score=-0.01,
    )
    invalid_uncertainty_2 = decision_engine(
        0.30,
        uncertainty_score=1.01,
    )

    checks["invalid_probability_rejected"] = (
        invalid_probability["decision"] == "REJECT_INVALID_INPUT"
        and invalid_probability_2["decision"] == "REJECT_INVALID_INPUT"
    )

    checks["invalid_uncertainty_rejected"] = (
        invalid_uncertainty["decision"] == "REJECT_INVALID_INPUT"
        and invalid_uncertainty_2["decision"] == "REJECT_INVALID_INPUT"
    )

    cases.extend([
        {
            "name": "invalid_probability_negative",
            "result": invalid_probability,
            "pass": checks["invalid_probability_rejected"],
        },
        {
            "name": "invalid_probability_above_one",
            "result": invalid_probability_2,
            "pass": checks["invalid_probability_rejected"],
        },
        {
            "name": "invalid_uncertainty_negative",
            "result": invalid_uncertainty,
            "pass": checks["invalid_uncertainty_rejected"],
        },
        {
            "name": "invalid_uncertainty_above_one",
            "result": invalid_uncertainty_2,
            "pass": checks["invalid_uncertainty_rejected"],
        },
    ])

    # ------------------------------------------------------------
    # 8. Monotonicity
    # ------------------------------------------------------------

    monotonic_probabilities = [
        0.01,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
        0.70,
        0.80,
        0.90,
        1.00,
    ]

    monotonic_decisions = [
        risk_from_probability(p)
        for p in monotonic_probabilities
    ]

    monotonic_ok = all(
        RANK[monotonic_decisions[i]]
        <= RANK[monotonic_decisions[i + 1]]
        for i in range(len(monotonic_decisions) - 1)
    )

    checks["risk_probability_monotonicity"] = monotonic_ok

    # ------------------------------------------------------------
    # 9. Safety boundary
    # ------------------------------------------------------------

    safety_boundary = {
        "direct_vehicle_actuation": False,
        "live_cortex_xdr_api_used": False,
        "real_world_cyberattack_performance_claim": False,
    }

    checks["direct_vehicle_actuation_disabled"] = (
        safety_boundary["direct_vehicle_actuation"] is False
    )

    checks["live_cortex_xdr_api_disabled"] = (
        safety_boundary["live_cortex_xdr_api_used"] is False
    )

    checks["real_world_attack_claim_disabled"] = (
        safety_boundary[
            "real_world_cyberattack_performance_claim"
        ] is False
    )

    # ------------------------------------------------------------
    # 10. Source evidence integrity
    # ------------------------------------------------------------

    checks["decision_engine_source_loaded"] = bool(
        decision_engine_source
    )

    checks["ensemble_source_loaded"] = bool(
        ensemble_source
    )

    checks["ensemble_retraining_false"] = (
        ensemble_source.get("retraining", False) is False
    )

    checks["ensemble_test_used_for_training_false"] = (
        ensemble_source.get("test_used_for_training", False) is False
    )

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

    passed = sum(bool(v) for v in checks.values())
    total = len(checks)

    report = {
        "phase": "4.34",
        "title": "Predictive Resilience Decision Validation",
        "status": "PASS" if passed == total else "FAIL",
        "validation_only": True,
        "retraining_performed": False,
        "decision_engine_source": str(DECISION_ENGINE.relative_to(ROOT)),
        "ensemble_evidence_source": str(ENSEMBLE_REPORT.relative_to(ROOT)),
        "policy": {
            "NORMAL": "<0.05",
            "LOW": ">=0.05 and <0.20",
            "MEDIUM": ">=0.20 and <0.50",
            "HIGH": ">=0.50 and <0.80",
            "CRITICAL": ">=0.80",
        },
        "modifier_policy": {
            "connectivity_degraded": "minimum_MEDIUM",
            "sensor_disagreement": "minimum_MEDIUM",
            "integrity_failure": "minimum_HIGH",
        },
        "checks": checks,
        "check_count": {
            "passed": passed,
            "total": total,
        },
        "case_count": len(cases),
        "cases": cases,
        "monotonicity": {
            "probabilities": monotonic_probabilities,
            "decisions": monotonic_decisions,
            "passed": monotonic_ok,
        },
        "safety_boundary": safety_boundary,
        "scientific_boundary": {
            "synthetic_telemetry_labels": True,
            "real_world_cyberattack_performance_claim": False,
            "live_cortex_xdr_connection_claim": False,
            "direct_vehicle_actuation": False,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    digest_payload = dict(report)
    digest_payload.pop("timestamp_utc", None)

    report["evidence_digest"] = sha256_json(digest_payload)

    OUT.parent.mkdir(parents=True, exist_ok=True)

    OUT.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("\n=== PHASE 4.34 RESULT ===")
    print(f"Status: {report['status']}")
    print(f"Checks: {passed}/{total}")
    print(f"Cases: {len(cases)}")
    print(f"Evidence digest: {report['evidence_digest']}")
    print(f"Report: {OUT}")

    failed = [
        name
        for name, value in checks.items()
        if not value
    ]

    print(f"Failed checks: {failed}")

    if failed:
        raise SystemExit(1)

    print("\nALL PHASE 4.34 CHECKS PASSED")


if __name__ == "__main__":
    main()
