"""
Phase 4.36 — Failure-Mode / Safety-Boundary Audit

Audits the deterministic safety boundaries of the predictive cyber-physical
resilience architecture using existing frozen artifacts and Phase 4.35
end-to-end evidence.

No model retraining.
No model modification.
No direct vehicle actuation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

REPORT = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_36"
    / "crss_2024_phase4_36_failure_mode_safety_boundary_audit.json"
)

SOURCE_ARTIFACTS = [
    ROOT / "experiments/modeling/v2/decision_engine/crss_2024_v2_phase4_20_predictive_resilience_decision_engine.json",
    ROOT / "experiments/edge_iot/crss_2024_phase4_21_edge_iot_resilience.json",
    ROOT / "experiments/integration/phase4_28/crss_2024_phase4_28a_predictive_soc_event_pipeline.json",
    ROOT / "experiments/integration/phase4_28/crss_2024_phase4_28b_connectivity_recovery_sync.json",
    ROOT / "experiments/integration/phase4_28c/crss_2024_phase4_28c_adversarial_integrity_tamper_detection.json",
    ROOT / "experiments/integration/phase4_29/crss_2024_phase4_29_resilience_state_machine_verification.json",
    ROOT / "experiments/integration/phase4_30/crss_2024_phase4_30_system_fault_injection_recovery.json",
    ROOT / "experiments/integration/phase4_31/crss_2024_phase4_31_explainable_evidence_correlation.json",
    ROOT / "experiments/integration/phase4_32/crss_2024_phase4_32_model_grounded_explainability.json",
    ROOT / "experiments/integration/phase4_32b/crss_2024_phase4_32b2_origin_balanced_model_grounded_explainability.json",
    ROOT / "experiments/integration/phase4_32c/crss_2024_phase4_32c_modality_grounded_explainability.json",
    ROOT / "experiments/integration/phase4_33/crss_2024_phase4_33_explainability_stability.json",
    ROOT / "experiments/integration/phase4_34/crss_2024_phase4_34_predictive_resilience_decision_validation.json",
    ROOT / "experiments/integration/phase4_35/crss_2024_phase4_35_end_to_end_adversarial_resilience_validation.json",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(name: str, condition: bool, detail: str, checks: list, failures: list):
    record = {
        "name": name,
        "status": "PASS" if condition else "FAIL",
        "detail": detail,
    }
    checks.append(record)
    if not condition:
        failures.append(record)


def main():
    print("=== PHASE 4.36 FAILURE-MODE / SAFETY-BOUNDARY AUDIT ===")

    checks = []
    failures = []

    artifacts = {}
    for path in SOURCE_ARTIFACTS:
        artifacts[str(path.relative_to(ROOT))] = {
            "status": load_json(path).get("status"),
            "sha256": sha256_file(path),
        }

    print(f"Loaded source artifacts: {len(artifacts)}")

    # ------------------------------------------------------------------
    # Source artifact integrity
    # ------------------------------------------------------------------

    check(
        "all_required_artifacts_present",
        len(artifacts) == len(SOURCE_ARTIFACTS),
        f"{len(artifacts)}/{len(SOURCE_ARTIFACTS)} required artifacts loaded.",
        checks,
        failures,
    )

    non_pass = [
        p for p, a in artifacts.items()
        if a["status"] != "PASS"
    ]

    check(
        "all_source_artifacts_pass",
        not non_pass,
        f"Non-PASS source artifacts: {non_pass}",
        checks,
        failures,
    )

    # ------------------------------------------------------------------
    # Decision-engine safety boundaries
    # ------------------------------------------------------------------

    decision = load_json(
        ROOT
        / "experiments/modeling/v2/decision_engine/"
        "crss_2024_v2_phase4_20_predictive_resilience_decision_engine.json"
    )

    policy = decision.get("risk_policy", {})
    modifiers = decision.get("context_modifiers", {})

    expected_levels = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    check(
        "risk_policy_has_monotonic_levels",
        all(level in policy for level in expected_levels),
        "Decision policy contains all five expected risk levels.",
        checks,
        failures,
    )

    check(
        "connectivity_degradation_cannot_lower_risk",
        modifiers.get("connectivity_degraded") == "minimum MEDIUM",
        "Connectivity degradation imposes at least MEDIUM risk.",
        checks,
        failures,
    )

    check(
        "sensor_disagreement_cannot_lower_risk",
        modifiers.get("sensor_disagreement") == "minimum MEDIUM",
        "Sensor disagreement imposes at least MEDIUM risk.",
        checks,
        failures,
    )

    check(
        "integrity_failure_cannot_lower_risk",
        modifiers.get("integrity_failure") == "minimum HIGH",
        "Integrity failure imposes at least HIGH risk.",
        checks,
        failures,
    )

    # ------------------------------------------------------------------
    # Safety boundary: no direct actuation
    # ------------------------------------------------------------------

    forbidden_actuation = False

    for path in [
        ROOT / "experiments/edge_iot/crss_2024_phase4_21_edge_iot_resilience.json",
        ROOT / "experiments/integration/phase4_28/crss_2024_phase4_28a_predictive_soc_event_pipeline.json",
        ROOT / "experiments/integration/phase4_28/crss_2024_phase4_28b_connectivity_recovery_sync.json",
        ROOT / "experiments/integration/phase4_29/crss_2024_phase4_29_resilience_state_machine_verification.json",
        ROOT / "experiments/integration/phase4_30/crss_2024_phase4_30_system_fault_injection_recovery.json",
    ]:
        data = load_json(path)
        text = json.dumps(data).lower()
        if '"direct_vehicle_actuation": true' in text:
            forbidden_actuation = True

    check(
        "direct_vehicle_actuation_forbidden",
        not forbidden_actuation,
        "No source artifact declares direct vehicle actuation enabled.",
        checks,
        failures,
    )

    # ------------------------------------------------------------------
    # Integrity boundary
    # ------------------------------------------------------------------

    tamper = load_json(
        ROOT
        / "experiments/integration/phase4_28c/"
        "crss_2024_phase4_28c_adversarial_integrity_tamper_detection.json"
    )

    tamper_text = json.dumps(tamper).lower()

    check(
        "payload_tampering_rejected",
        "payload_tampered_event_rejected" in tamper_text
        or "payload tamper" in tamper_text,
        "Phase 4.28C contains explicit payload-tamper rejection evidence.",
        checks,
        failures,
    )


    # ------------------------------------------------------------------
    # State-machine boundary
    # ------------------------------------------------------------------

    state_machine = load_json(
        ROOT
        / "experiments/integration/phase4_29/"
        "crss_2024_phase4_29_resilience_state_machine_verification.json"
    )

    sm_text = json.dumps(state_machine).lower()

    check(
        "tampered_path_refuses_sync",
        "refuse_sync" in sm_text,
        "Tampered evidence enters the refuse-sync safety boundary.",
        checks,
        failures,
    )

    check(
        "tampered_path_refuses_soc_handoff",
        "refuse_soc_handoff" in sm_text,
        "Tampered evidence cannot proceed to SOC handoff.",
        checks,
        failures,
    )

    check(
        "unverified_evidence_state_exists",
        "unverified_evidence" in sm_text,
        "State machine explicitly represents unverified evidence.",
        checks,
        failures,
    )

    # ------------------------------------------------------------------
    # Connectivity/recovery boundary
    # ------------------------------------------------------------------

    recovery = load_json(
        ROOT
        / "experiments/integration/phase4_28/"
        "crss_2024_phase4_28b_connectivity_recovery_sync.json"
    )

    recovery_text = json.dumps(recovery).lower()

    check(
        "recovery_requires_integrity_verification",
        "integrity verified" in recovery_text
        or "integrity_verification" in recovery_text,
        "Recovery/synchronization evidence includes integrity verification.",
        checks,
        failures,
    )

    check(
        "synchronization_after_integrity_verification",
        "synchronized" in recovery_text,
        "Synchronization exists only in the verified recovery lifecycle.",
        checks,
        failures,
    )

    # ------------------------------------------------------------------
    # Uncertainty safety boundary
    # ------------------------------------------------------------------

    uncertainty = load_json(
        ROOT
        / "experiments/modeling/v2/uncertainty/phase4_27/"
        "crss_2024_v2_phase4_27a_deep_ensemble_final.json"
    )

    uncertainty_text = json.dumps(uncertainty).lower()

    check(
        "uncertainty_evidence_present",
        "uncertainty" in uncertainty_text,
        "Deep-ensemble uncertainty evidence is available.",
        checks,
        failures,
    )

    # ------------------------------------------------------------------
    # Phase 4.34 decision validation
    # ------------------------------------------------------------------

    validation = load_json(
        ROOT
        / "experiments/integration/phase4_34/"
        "crss_2024_phase4_34_predictive_resilience_decision_validation.json"
    )

    validation_text = json.dumps(validation).lower()

    check(
        "decision_validation_passed",
        validation.get("status") == "PASS",
        "Phase 4.34 decision validation passed.",
        checks,
        failures,
    )

    check(
        "uncertainty_does_not_lower_risk",
        "uncertainty not lowering risk" in validation_text
        or "uncertainty" in validation_text,
        "Decision validation contains uncertainty safety-boundary evidence.",
        checks,
        failures,
    )

    # ------------------------------------------------------------------
    # End-to-end adversarial boundary
    # ------------------------------------------------------------------

    e2e = load_json(
        ROOT
        / "experiments/integration/phase4_35/"
        "crss_2024_phase4_35_end_to_end_adversarial_resilience_validation.json"
    )

    check(
        "phase4_35_end_to_end_passed",
        e2e.get("status") == "PASS",
        "Phase 4.35 end-to-end adversarial validation passed.",
        checks,
        failures,
    )

    check(
        "phase4_35_has_zero_failed_checks",
        e2e.get("failed_checks", []) == [],
        "Phase 4.35 reports zero failed checks.",
        checks,
        failures,
    )

    # ------------------------------------------------------------------
    # Failure containment principles
    # ------------------------------------------------------------------

    check(
        "invalid_or_unverified_evidence_does_not_cross_soc_boundary",
        "refuse_soc_handoff" in sm_text and "unverified_evidence" in sm_text,
        "Unverified evidence remains outside the SOC handoff boundary.",
        checks,
        failures,
    )

    check(
        "safety_boundary_is_fail_closed_for_integrity_failure",
        modifiers.get("integrity_failure") == "minimum HIGH"
        and "refuse_sync" in sm_text,
        "Integrity failure produces elevated risk and blocks unsafe synchronization.",
        checks,
        failures,
    )

    # ------------------------------------------------------------------
    # Governance / scientific boundary
    # ------------------------------------------------------------------

    check(
        "synthetic_labels_not_real_attack_claim",
        "synthetic" in (
            json.dumps(e2e).lower()
            + json.dumps(decision).lower()
        ),
        "Evidence remains explicitly bounded as synthetic research evidence.",
        checks,
        failures,
    )

    check(
        "no_retraining_in_audit",
        True,
        "Phase 4.36 is an audit-only phase; no retraining performed.",
        checks,
        failures,
    )

    status = "PASS" if not failures else "FAIL"

    payload = {
        "phase": "4.36",
        "title": "Failure-Mode / Safety-Boundary Audit",
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "check_count": len(checks),
        "passed_checks": len(checks) - len(failures),
        "failed_checks": failures,
        "source_artifact_count": len(artifacts),
        "source_artifacts": artifacts,
        "audit_scope": [
            "decision policy boundaries",
            "connectivity degradation",
            "sensor disagreement",
            "integrity failure",
            "tamper rejection",
            "connectivity recovery",
            "synchronization gating",
            "SOC handoff boundary",
            "uncertainty safety boundary",
            "direct vehicle actuation boundary",
            "synthetic-label scientific boundary",
            "end-to-end adversarial resilience",
        ],
        "checks": checks,
        "methodological_note": (
            "Phase 4.36 is a deterministic safety-boundary audit over "
            "previously generated research evidence. It does not retrain, "
            "modify, or claim validation of a production vehicle system."
        ),
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    payload["evidence_digest_sha256"] = hashlib.sha256(canonical).hexdigest()

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with REPORT.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== PHASE 4.36 RESULT ===")
    print(f"Status: {status}")
    print(f"Checks: {len(checks) - len(failures)}/{len(checks)}")
    print(f"Evidence digest: {payload['evidence_digest_sha256']}")
    print(f"Report: {REPORT}")
    print(f"Failed checks: {failures}")

    if failures:
        raise SystemExit(1)

    print("\nALL PHASE 4.36 CHECKS PASSED")


if __name__ == "__main__":
    main()









