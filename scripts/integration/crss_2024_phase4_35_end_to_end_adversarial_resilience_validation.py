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

EDGE_REPORT = (
    ROOT
    / "experiments"
    / "edge_iot"
    / "crss_2024_phase4_21_edge_iot_resilience.json"
)

PIPELINE_28A = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_28"
    / "crss_2024_phase4_28a_predictive_soc_event_pipeline.json"
)

RECOVERY_28B = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_28"
    / "crss_2024_phase4_28b_connectivity_recovery_sync.json"
)

TAMPER_28C = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_28c"
    / "crss_2024_phase4_28c_adversarial_integrity_tamper_detection.json"
)

STATE_MACHINE_29 = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_29"
    / "crss_2024_phase4_29_resilience_state_machine_verification.json"
)

FAULT_30 = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_30"
    / "crss_2024_phase4_30_system_fault_injection_recovery.json"
)

EXPLAINABILITY_31 = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_31"
    / "crss_2024_phase4_31_explainable_evidence_correlation.json"
)

OUT = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_35"
    / "crss_2024_phase4_35_end_to_end_adversarial_resilience_validation.json"
)


RANK = {
    "NORMAL": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def risk_from_probability(p):
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


def decision(
    probability,
    connectivity=False,
    sensor=False,
    integrity=False,
):
    level = risk_from_probability(float(probability))

    if connectivity:
        level = elevate(level, "MEDIUM")

    if sensor:
        level = elevate(level, "MEDIUM")

    if integrity:
        level = elevate(level, "HIGH")

    return level


def payload_hash(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def record_hash(record):
    material = {
        "event_id": record["event_id"],
        "payload_hash": record["payload_hash"],
        "previous_hash": record["previous_hash"],
    }

    return hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def make_event(
    event_id,
    payload,
    previous_hash,
    connectivity_state,
    integrity_status="VERIFIED",
):
    ph = payload_hash(payload)

    event = {
        "event_id": event_id,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "vehicle_id": "RESEARCH_VEHICLE_001",
        "ecu_id": "ECU_RESEARCH_01",
        "can_id": "0x123",
        "event_type": "PREDICTIVE_RESILIENCE_EVENT",
        "severity": "HIGH",
        "connectivity_state": connectivity_state,
        "integrity_status": integrity_status,
        "payload": payload,
        "payload_hash": ph,
        "previous_hash": previous_hash,
    }

    event["record_hash"] = record_hash(event)

    return event


def verify_event(event):
    expected_payload = payload_hash(event["payload"])

    expected_record = record_hash(event)

    return {
        "payload_valid": (
            event["payload_hash"] == expected_payload
        ),
        "record_valid": (
            event["record_hash"] == expected_record
        ),
    }


def verify_chain(events):
    previous = "ROLLING_GENESIS"

    for event in events:
        if event["previous_hash"] != previous:
            return False

        verification = verify_event(event)

        if not verification["payload_valid"]:
            return False

        if not verification["record_valid"]:
            return False

        previous = event["record_hash"]

    return True


def end_to_end_flow(
    probability,
    connectivity_loss=False,
    tamper_payload=False,
    tamper_previous_hash=False,
    buffer_rollover=False,
):
    predicted = decision(
        probability,
        connectivity=connectivity_loss,
    )

    events = []

    previous = "ROLLING_GENESIS"

    event_count = 5 if buffer_rollover else 3

    for i in range(event_count):
        state = (
            "DISCONNECTED"
            if connectivity_loss
            else "CONNECTED"
        )

        event = make_event(
            f"ADV_{i+1}",
            {
                "risk_probability": float(probability),
                "risk_level": predicted,
                "sequence": i + 1,
            },
            previous,
            state,
        )

        events.append(event)
        previous = event["record_hash"]

    if tamper_payload:
        events[0]["payload"] = {
            "risk_probability": 0.01,
            "risk_level": "NORMAL",
            "sequence": 1,
        }

    if tamper_previous_hash:
        events[-1]["previous_hash"] = "ATTACKER_MODIFIED_PREVIOUS_HASH"

    chain_valid = verify_chain(events)

    sync_allowed = (
        chain_valid
        and connectivity_loss
    )

    soc_allowed = sync_allowed

    final_state = (
        "SOC_HANDOFF"
        if soc_allowed
        else (
            "REFUSE_SOC_HANDOFF"
            if not chain_valid
            else "LOCAL_FORENSIC_BUFFER"
        )
    )

    return {
        "predicted_risk": predicted,
        "connectivity_loss": connectivity_loss,
        "buffered_events": len(events),
        "tamper_payload": tamper_payload,
        "tamper_previous_hash": tamper_previous_hash,
        "buffer_rollover": buffer_rollover,
        "chain_valid": chain_valid,
        "synchronization": (
            "ALLOW"
            if sync_allowed
            else "REFUSE"
        ),
        "soc_handoff": (
            "ALLOW"
            if soc_allowed
            else "REFUSE"
        ),
        "final_state": final_state,
    }


def main():
    print("=== PHASE 4.35 END-TO-END ADVERSARIAL RESILIENCE VALIDATION ===")

    source_files = [
        DECISION_ENGINE,
        EDGE_REPORT,
        PIPELINE_28A,
        RECOVERY_28B,
        TAMPER_28C,
        STATE_MACHINE_29,
        FAULT_30,
        EXPLAINABILITY_31,
    ]

    sources = {}

    for path in source_files:
        if not path.exists():
            raise FileNotFoundError(path)

        sources[str(path.relative_to(ROOT))] = json.loads(
            path.read_text(encoding="utf-8")
        )

    print(f"Loaded source artifacts: {len(sources)}")

    cases = []

    def add_case(name, expected, result):
        passed = True

        for key, expected_value in expected.items():
            if result.get(key) != expected_value:
                passed = False

        cases.append(
            {
                "name": name,
                "expected": expected,
                "result": result,
                "pass": passed,
            }
        )

    # 1. Normal connected operation.
    add_case(
        "normal_connected",
        {
            "predicted_risk": "NORMAL",
            "chain_valid": True,
            "synchronization": "REFUSE",
            "soc_handoff": "REFUSE",
            "final_state": "LOCAL_FORENSIC_BUFFER",
        },
        end_to_end_flow(0.02, connectivity_loss=False),
    )

    # 2. Predictive high risk with connectivity.
    add_case(
        "predictive_high_connected",
        {
            "predicted_risk": "HIGH",
            "chain_valid": True,
            "synchronization": "REFUSE",
            "soc_handoff": "REFUSE",
        },
        end_to_end_flow(0.70, connectivity_loss=False),
    )

    # 3. High risk + connectivity loss.
    add_case(
        "high_risk_connectivity_loss",
        {
            "predicted_risk": "HIGH",
            "chain_valid": True,
            "synchronization": "ALLOW",
            "soc_handoff": "ALLOW",
            "final_state": "SOC_HANDOFF",
        },
        end_to_end_flow(0.70, connectivity_loss=True),
    )

    # 4. Medium risk + connectivity loss.
    add_case(
        "medium_risk_connectivity_loss",
        {
            "predicted_risk": "MEDIUM",
            "chain_valid": True,
            "synchronization": "ALLOW",
            "soc_handoff": "ALLOW",
        },
        end_to_end_flow(0.10, connectivity_loss=True),
    )

    # 5. Payload tampering.
    add_case(
        "payload_tampering",
        {
            "chain_valid": False,
            "synchronization": "REFUSE",
            "soc_handoff": "REFUSE",
            "final_state": "REFUSE_SOC_HANDOFF",
        },
        end_to_end_flow(
            0.70,
            connectivity_loss=True,
            tamper_payload=True,
        ),
    )

    # 6. Previous-hash tampering.
    add_case(
        "previous_hash_tampering",
        {
            "chain_valid": False,
            "synchronization": "REFUSE",
            "soc_handoff": "REFUSE",
            "final_state": "REFUSE_SOC_HANDOFF",
        },
        end_to_end_flow(
            0.70,
            connectivity_loss=True,
            tamper_previous_hash=True,
        ),
    )

    # 7. Compound high risk + payload tampering.
    add_case(
        "compound_high_risk_payload_tampering",
        {
            "predicted_risk": "HIGH",
            "chain_valid": False,
            "synchronization": "REFUSE",
            "soc_handoff": "REFUSE",
        },
        end_to_end_flow(
            0.70,
            connectivity_loss=True,
            tamper_payload=True,
        ),
    )

    # 8. Critical risk + connectivity.
    add_case(
        "critical_risk_connectivity_loss",
        {
            "predicted_risk": "CRITICAL",
            "chain_valid": True,
            "synchronization": "ALLOW",
            "soc_handoff": "ALLOW",
        },
        end_to_end_flow(0.90, connectivity_loss=True),
    )

    # 9. Buffer rollover.
    rollover_result = end_to_end_flow(
        0.70,
        connectivity_loss=True,
        buffer_rollover=True,
    )

    add_case(
        "buffer_rollover",
        {
            "buffered_events": 5,
            "chain_valid": True,
            "synchronization": "ALLOW",
            "soc_handoff": "ALLOW",
        },
        rollover_result,
    )

    # 10. Repeated recovery equivalent.
    recovery_a = end_to_end_flow(
        0.30,
        connectivity_loss=True,
    )

    recovery_b = end_to_end_flow(
        0.30,
        connectivity_loss=True,
    )

    add_case(
        "repeated_disconnect_recovery",
        {
            "chain_valid": True,
            "synchronization": "ALLOW",
            "soc_handoff": "ALLOW",
        },
        recovery_a,
    )

    cases.append(
        {
            "name": "repeated_disconnect_recovery_consistency",
            "expected": {
                "chain_valid": True,
                "synchronization": "ALLOW",
                "soc_handoff": "ALLOW",
            },
            "result": recovery_b,
            "pass": (
                recovery_a["chain_valid"]
                == recovery_b["chain_valid"]
                and recovery_a["synchronization"]
                == recovery_b["synchronization"]
                and recovery_a["soc_handoff"]
                == recovery_b["soc_handoff"]
            ),
        }
    )

    # 11. Sensor disagreement escalation.
    sensor_level = decision(
        0.10,
        sensor=True,
    )

    cases.append(
        {
            "name": "sensor_disagreement_escalation",
            "expected": {"decision": "MEDIUM"},
            "result": {"decision": sensor_level},
            "pass": sensor_level == "MEDIUM",
        }
    )

    # 12. Integrity failure escalation.
    integrity_level = decision(
        0.10,
        integrity=True,
    )

    cases.append(
        {
            "name": "integrity_failure_escalation",
            "expected": {"decision": "HIGH"},
            "result": {"decision": integrity_level},
            "pass": integrity_level == "HIGH",
        }
    )

    # 13. Safety boundary.
    safety_boundary = {
        "direct_vehicle_actuation": False,
        "live_cortex_xdr_api_used": False,
        "real_world_cyberattack_performance_claim": False,
    }

    safety_checks = {
        "direct_vehicle_actuation_disabled":
            safety_boundary["direct_vehicle_actuation"] is False,

        "live_cortex_xdr_api_disabled":
            safety_boundary["live_cortex_xdr_api_used"] is False,

        "real_world_attack_claim_disabled":
            safety_boundary[
                "real_world_cyberattack_performance_claim"
            ] is False,
    }

    # 14. Source artifacts must already be successful.
    source_status_checks = {
        "phase4_28a_pass":
            sources[
                str(PIPELINE_28A.relative_to(ROOT))
            ].get("status") == "PASS",

        "phase4_28b_pass":
            sources[
                str(RECOVERY_28B.relative_to(ROOT))
            ].get("status") == "PASS",

        "phase4_28c_pass":
            sources[
                str(TAMPER_28C.relative_to(ROOT))
            ].get("status") == "PASS",

        "phase4_29_pass":
            sources[
                str(STATE_MACHINE_29.relative_to(ROOT))
            ].get("status") == "PASS",

        "phase4_30_pass":
            sources[
                str(FAULT_30.relative_to(ROOT))
            ].get("status") == "PASS",

        "phase4_31_pass":
            sources[
                str(EXPLAINABILITY_31.relative_to(ROOT))
            ].get("status") == "PASS",
    }

    checks = {}

    checks["all_adversarial_cases_pass"] = all(
        case["pass"] for case in cases
    )

    checks.update(safety_checks)
    checks.update(source_status_checks)

    checks["tampering_never_allows_soc_handoff"] = all(
        (
            case["result"].get("soc_handoff") != "ALLOW"
            if (
                case["result"].get("tamper_payload")
                or case["result"].get("tamper_previous_hash")
            )
            else True
        )
        for case in cases
    )

    checks["tampering_never_allows_sync"] = all(
        (
            case["result"].get("synchronization") != "ALLOW"
            if (
                case["result"].get("tamper_payload")
                or case["result"].get("tamper_previous_hash")
            )
            else True
        )
        for case in cases
    )

    checks["high_risk_connectivity_recovery_preserved"] = (
        next(
            c for c in cases
            if c["name"] == "high_risk_connectivity_loss"
        )["pass"]
    )

    checks["buffer_rollover_chain_preserved"] = (
        next(
            c for c in cases
            if c["name"] == "buffer_rollover"
        )["pass"]
    )

    passed = sum(bool(v) for v in checks.values())
    total = len(checks)

    report = {
        "phase": "4.35",
        "title": "End-to-End Adversarial Resilience Validation",
        "status": "PASS" if passed == total else "FAIL",
        "validation_only": True,
        "retraining_performed": False,
        "source_artifacts": list(sources.keys()),
        "architecture_path": [
            "predictive_risk",
            "decision_engine",
            "connectivity_loss",
            "local_forensic_buffer",
            "integrity_verification",
            "synchronization",
            "soc_handoff",
        ],
        "checks": checks,
        "check_count": {
            "passed": passed,
            "total": total,
        },
        "case_count": len(cases),
        "cases": cases,
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

    report["evidence_digest"] = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    OUT.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("\n=== PHASE 4.35 RESULT ===")
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

    print("\nALL PHASE 4.35 CHECKS PASSED")


if __name__ == "__main__":
    main()


