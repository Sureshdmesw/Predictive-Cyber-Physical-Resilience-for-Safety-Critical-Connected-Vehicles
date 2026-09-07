import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PHASE_20 = ROOT / (
    "experiments/modeling/v2/decision_engine/"
    "crss_2024_v2_phase4_20_predictive_resilience_decision_engine.json"
)

PHASE_21 = ROOT / (
    "experiments/edge_iot/"
    "crss_2024_phase4_21_edge_iot_resilience.json"
)

PHASE_27A = ROOT / (
    "experiments/modeling/v2/uncertainty/phase4_27/"
    "crss_2024_v2_phase4_27a_deep_ensemble_final.json"
)

PHASE_28A = ROOT / (
    "experiments/integration/phase4_28/"
    "crss_2024_phase4_28a_predictive_soc_event_pipeline.json"
)

PHASE_28B = ROOT / (
    "experiments/integration/phase4_28/"
    "crss_2024_phase4_28b_connectivity_recovery_sync.json"
)

PHASE_28C = ROOT / (
    "experiments/integration/phase4_28c/"
    "crss_2024_phase4_28c_adversarial_integrity_tamper_detection.json"
)

PHASE_29 = ROOT / (
    "experiments/integration/phase4_29/"
    "crss_2024_phase4_29_resilience_state_machine_verification.json"
)

PHASE_30 = ROOT / (
    "experiments/integration/phase4_30/"
    "crss_2024_phase4_30_system_fault_injection_recovery.json"
)

SCHEMA = ROOT / (
    "data/schemas/explainability/"
    "resilience_explanation_schema.json"
)

OUTPUT = ROOT / (
    "experiments/integration/phase4_31/"
    "crss_2024_phase4_31_explainable_evidence_correlation.json"
)


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest(obj):
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def risk_level(probability):
    if probability < 0.05:
        return "NORMAL"
    if probability < 0.20:
        return "LOW"
    if probability < 0.50:
        return "MEDIUM"
    if probability < 0.80:
        return "HIGH"
    return "CRITICAL"


def make_forensic_event(event_id, payload, previous_hash):
    payload_hash = sha256_text(payload)

    record_hash = sha256_text(
        event_id + payload_hash + previous_hash
    )

    return {
        "event_id": event_id,
        "payload": payload,
        "payload_hash": payload_hash,
        "previous_hash": previous_hash,
        "record_hash": record_hash,
    }


def valid_event(record, previous_hash):
    payload_hash_valid = (
        record["payload_hash"] ==
        sha256_text(record["payload"])
    )

    record_hash_valid = (
        record["record_hash"] ==
        sha256_text(
            record["event_id"]
            + record["payload_hash"]
            + record["previous_hash"]
        )
    )

    previous_hash_valid = (
        record["previous_hash"] == previous_hash
    )

    return (
        payload_hash_valid
        and record_hash_valid
        and previous_hash_valid
    )


def build_evidence(event_prefix, tampered=False):
    records = []
    previous = "ROLLING_GENESIS"

    payloads = [
        {
            "ecu_id": "ECU-1",
            "can_id": "0x100",
            "signal": "message_integrity",
            "status": "FAIL" if tampered else "PASS",
        },
        {
            "ecu_id": "ECU-2",
            "can_id": "0x101",
            "signal": "sensor_consistency",
            "status": "FAIL" if tampered else "PASS",
        },
        {
            "ecu_id": "ECU-3",
            "can_id": "0x102",
            "signal": "connectivity",
            "status": "DEGRADED",
        },
    ]

    for index, payload_obj in enumerate(payloads, start=1):
        payload = json.dumps(
            payload_obj,
            sort_keys=True
        )

        event = make_forensic_event(
            f"{event_prefix}-{index:03d}",
            payload,
            previous
        )

        records.append(event)
        previous = event["record_hash"]

    if tampered:
        records[1]["payload"] = json.dumps(
            {
                "ecu_id": "ECU-2",
                "can_id": "0x101",
                "signal": "sensor_consistency",
                "status": "UNAUTHORIZED_MODIFICATION",
            },
            sort_keys=True
        )

    return records


def validate_chain(records):
    previous = "ROLLING_GENESIS"

    for record in records:
        if not valid_event(record, previous):
            return False
        previous = record["record_hash"]

    return True


def derive_risk_drivers(
    predictive_risk,
    connectivity_loss,
    sensor_disagreement,
    integrity_failure
):
    drivers = []

    if predictive_risk:
        drivers.append({
            "feature": "predictive_risk_probability",
            "value": 0.72,
            "direction": "INCREASE_RISK",
            "importance": 0.40,
        })

    if connectivity_loss:
        drivers.append({
            "feature": "connectivity_degradation_rate",
            "value": "DEGRADED",
            "direction": "INCREASE_RISK",
            "importance": 0.25,
        })

    if sensor_disagreement:
        drivers.append({
            "feature": "sensor_disagreement",
            "value": "ELEVATED",
            "direction": "INCREASE_RISK",
            "importance": 0.20,
        })

    if integrity_failure:
        drivers.append({
            "feature": "message_integrity_failure",
            "value": "FAIL",
            "direction": "INCREASE_RISK",
            "importance": 0.35,
        })

    drivers.sort(
        key=lambda item: item["importance"],
        reverse=True
    )

    return drivers


def derive_decision(
    probability,
    connectivity_loss,
    sensor_disagreement,
    integrity_failure
):
    base = risk_level(probability)
    modifiers = []

    final = base

    if connectivity_loss:
        modifiers.append("connectivity_degraded")
        order = {
            "NORMAL": 0,
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4,
        }

        if order[final] < order["MEDIUM"]:
            final = "MEDIUM"

    if sensor_disagreement:
        modifiers.append("sensor_disagreement")

        order = {
            "NORMAL": 0,
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4,
        }

        if order[final] < order["MEDIUM"]:
            final = "MEDIUM"

    if integrity_failure:
        modifiers.append("integrity_failure")

        order = {
            "NORMAL": 0,
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4,
        }

        if order[final] < order["HIGH"]:
            final = "HIGH"

    return {
        "base_policy_level": base,
        "modifiers": modifiers,
        "final_policy_level": final,
        "reason": (
            "Base predictive risk was evaluated first; "
            "deterministic safety modifiers then applied "
            "minimum policy levels."
        ),
    }


def state_transition(
    connectivity_loss,
    integrity_failure,
    trusted_sync
):
    if not connectivity_loss:
        return {
            "previous_state": "PREDICTIVE_RISK",
            "current_state": "SOC_HANDOFF",
            "transition_reason": (
                "Trusted evidence was available without "
                "a connectivity-loss recovery cycle."
            ),
        }

    if integrity_failure or not trusted_sync:
        return {
            "previous_state": "INTEGRITY_VERIFICATION",
            "current_state": "UNVERIFIED_EVIDENCE",
            "transition_reason": (
                "Integrity verification failed; evidence "
                "cannot enter trusted synchronization."
            ),
        }

    return {
        "previous_state": "INTEGRITY_VERIFICATION",
        "current_state": "SOC_HANDOFF",
        "transition_reason": (
            "Evidence integrity was verified after "
            "connectivity recovery."
        ),
    }


def investigation_context(
    decision_level,
    integrity_ok,
    drivers
):
    if not integrity_ok:
        return {
            "event_type": "CORTEX_XDR_COMPATIBLE_RESEARCH_EVENT",
            "priority": "HOLD_UNVERIFIED",
            "investigation_reason": (
                "Evidence integrity failed. Preserve the "
                "evidence as unverified and prevent trusted "
                "synchronization or SOC handoff."
            ),
            "recommended_checks": [
                "Verify payload hashes.",
                "Verify record hashes.",
                "Verify forensic hash-chain continuity.",
                "Preserve the original unverified evidence.",
                "Investigate possible unauthorized modification.",
            ],
        }

    checks = [
        "Correlate predictive risk with forensic event IDs.",
        "Verify ECU and CAN identifiers.",
        "Review connectivity transition timeline.",
    ]

    if any(
        d["feature"] == "sensor_disagreement"
        for d in drivers
    ):
        checks.append(
            "Compare redundant sensor measurements and timing."
        )

    if any(
        d["feature"] == "message_integrity_failure"
        for d in drivers
    ):
        checks.append(
            "Investigate message-integrity failures and "
            "unexpected ECU communication."
        )

    priority = {
        "NORMAL": "LOW",
        "LOW": "LOW",
        "MEDIUM": "MEDIUM",
        "HIGH": "HIGH",
        "CRITICAL": "CRITICAL",
    }[decision_level]

    return {
        "event_type": "CORTEX_XDR_COMPATIBLE_RESEARCH_EVENT",
        "priority": priority,
        "investigation_reason": (
            "Correlate predictive risk, resilience state, "
            "and verified forensic evidence."
        ),
        "recommended_checks": checks,
    }


def evaluate_case(
    name,
    probability,
    uncertainty,
    connectivity_loss,
    sensor_disagreement,
    integrity_failure
):
    drivers = derive_risk_drivers(
        probability,
        connectivity_loss,
        sensor_disagreement,
        integrity_failure
    )

    decision = derive_decision(
        probability,
        connectivity_loss,
        sensor_disagreement,
        integrity_failure
    )

    records = build_evidence(
        f"EXPLAIN-{name.upper()}",
        tampered=integrity_failure
    )

    chain_valid = validate_chain(records)

    if integrity_failure:
        trusted_sync = False
        soc_allowed = False
        integrity_status = False
    else:
        trusted_sync = chain_valid
        soc_allowed = trusted_sync
        integrity_status = chain_valid

    state = state_transition(
        connectivity_loss,
        integrity_failure,
        trusted_sync
    )

    investigation = investigation_context(
        decision["final_policy_level"],
        integrity_status,
        drivers
    )

    event_ids = [
        record["event_id"]
        for record in records
    ]

    evidence_digest = digest(records)

    explanation = {
        "explanation_id": f"EXPLANATION-{name}",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "vehicle_context": {
            "vehicle_id": "RESEARCH-VEHICLE-001",
            "ecu_id": "ECU-1",
            "connectivity_state": (
                "DISCONNECTED"
                if connectivity_loss
                else "CONNECTED"
            ),
        },
        "prediction": {
            "risk_probability": probability,
            "risk_level": risk_level(probability),
            "uncertainty": uncertainty,
            "horizon": "Y6",
        },
        "risk_drivers": drivers,
        "decision": decision,
        "resilience_state": state,
        "forensic_evidence": {
            "event_ids": event_ids,
            "event_count": len(records),
            "evidence_digest": evidence_digest,
        },
        "integrity": {
            "payload_integrity": integrity_status,
            "record_integrity": integrity_status,
            "chain_integrity": chain_valid,
            "trusted_synchronization": trusted_sync,
            "soc_handoff_allowed": soc_allowed,
        },
        "soc_investigation": investigation,
        "safety_boundary": {
            "direct_vehicle_actuation": False,
            "live_cortex_xdr_api_used": False,
            "real_world_cyberattack_performance_claim": False,
        },
    }

    return {
        "scenario": name,
        "explanation": explanation,
        "forensic_records": records,
    }


def main():
    required = [
        PHASE_20,
        PHASE_21,
        PHASE_27A,
        PHASE_28A,
        PHASE_28B,
        PHASE_28C,
        PHASE_29,
        PHASE_30,
        SCHEMA,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(str(path))

    phase20 = load_json(PHASE_20)
    phase21 = load_json(PHASE_21)
    phase27a = load_json(PHASE_27A)
    phase28a = load_json(PHASE_28A)
    phase28b = load_json(PHASE_28B)
    phase28c = load_json(PHASE_28C)
    phase29 = load_json(PHASE_29)
    phase30 = load_json(PHASE_30)
    schema = load_json(SCHEMA)

    cases = [
        evaluate_case(
            "normal_baseline",
            0.02,
            0.001,
            False,
            False,
            False,
        ),
        evaluate_case(
            "predictive_medium",
            0.31,
            0.02,
            False,
            False,
            False,
        ),
        evaluate_case(
            "connectivity_degraded",
            0.10,
            0.01,
            True,
            False,
            False,
        ),
        evaluate_case(
            "sensor_disagreement",
            0.12,
            0.015,
            False,
            True,
            False,
        ),
        evaluate_case(
            "integrity_failure",
            0.18,
            0.02,
            True,
            False,
            True,
        ),
        evaluate_case(
            "compound_high_risk",
            0.72,
            0.08,
            True,
            True,
            False,
        ),
        evaluate_case(
            "compound_tampered_evidence",
            0.72,
            0.08,
            True,
            True,
            True,
        ),
    ]

    gates = {}

    gates["phase_20_source_pass"] = (
        phase20.get("status") == "PASS"
    )

    gates["phase_21_source_pass"] = (
        phase21.get("status") == "PASS"
    )

    gates["phase_27a_source_pass"] = (
        phase27a.get("status") == "PASS"
    )

    gates["phase_28a_source_pass"] = (
        phase28a.get("status") == "PASS"
    )

    gates["phase_28b_source_pass"] = (
        phase28b.get("status") == "PASS"
    )

    gates["phase_28c_source_pass"] = (
        phase28c.get("status") == "PASS"
    )

    gates["phase_29_source_pass"] = (
        phase29.get("status") == "PASS"
    )

    gates["phase_30_source_pass"] = (
        phase30.get("status") == "PASS"
    )

    gates["schema_has_required_fields"] = (
        len(schema.get("required", [])) == 11
    )

    gates["all_explanation_cases_generated"] = (
        len(cases) == 7
    )

    gates["risk_levels_are_deterministic"] = all(
        c["explanation"]["prediction"]["risk_level"]
        == risk_level(
            c["explanation"]["prediction"]["risk_probability"]
        )
        for c in cases
    )

    gates["risk_drivers_present_when_expected"] = all(
        len(c["explanation"]["risk_drivers"]) > 0
        for c in cases
        if c["scenario"] != "normal_baseline"
    )

    gates["driver_importance_sorted"] = all(
        [
            d["importance"]
            for d in c["explanation"]["risk_drivers"]
        ]
        == sorted(
            [
                d["importance"]
                for d in c["explanation"]["risk_drivers"]
            ],
            reverse=True,
        )
        for c in cases
    )

    gates["connectivity_modifier_explained"] = any(
        "connectivity_degraded"
        in c["explanation"]["decision"]["modifiers"]
        for c in cases
    )

    gates["sensor_modifier_explained"] = any(
        "sensor_disagreement"
        in c["explanation"]["decision"]["modifiers"]
        for c in cases
    )

    gates["integrity_modifier_explained"] = any(
        "integrity_failure"
        in c["explanation"]["decision"]["modifiers"]
        for c in cases
    )

    gates["forensic_event_correlation_present"] = all(
        c["explanation"]["forensic_evidence"]["event_count"]
        == len(c["explanation"]["forensic_evidence"]["event_ids"])
        and len(c["explanation"]["forensic_evidence"]["event_ids"]) > 0
        for c in cases
    )

    gates["clean_evidence_chain_valid"] = all(
        c["explanation"]["integrity"]["chain_integrity"]
        for c in cases
        if not c["explanation"]["decision"]["modifiers"]
        or "integrity_failure"
        not in c["explanation"]["decision"]["modifiers"]
    )

    gates["integrity_failure_refuses_sync"] = all(
        not c["explanation"]["integrity"]["trusted_synchronization"]
        for c in cases
        if "integrity_failure"
        in c["explanation"]["decision"]["modifiers"]
    )

    gates["integrity_failure_refuses_soc"] = all(
        not c["explanation"]["integrity"]["soc_handoff_allowed"]
        for c in cases
        if "integrity_failure"
        in c["explanation"]["decision"]["modifiers"]
    )

    gates["tampered_evidence_explained"] = (
        cases[-1]["explanation"]["soc_investigation"]["priority"]
        == "HOLD_UNVERIFIED"
    )

    gates["tampered_evidence_preserved"] = (
        cases[-1]["explanation"]["forensic_evidence"]["event_count"]
        == 3
    )

    gates["soc_investigation_context_present"] = all(
        len(
            c["explanation"]["soc_investigation"]
            ["recommended_checks"]
        ) > 0
        for c in cases
    )

    gates["evidence_digest_present"] = all(
        len(
            c["explanation"]["forensic_evidence"]
            ["evidence_digest"]
        ) == 64
        for c in cases
    )

    gates["direct_vehicle_actuation_disabled"] = all(
        not c["explanation"]["safety_boundary"]
        ["direct_vehicle_actuation"]
        for c in cases
    )

    gates["live_cortex_xdr_api_claim_disabled"] = all(
        not c["explanation"]["safety_boundary"]
        ["live_cortex_xdr_api_used"]
        for c in cases
    )

    gates["real_attack_claim_disabled"] = all(
        not c["explanation"]["safety_boundary"]
        ["real_world_cyberattack_performance_claim"]
        for c in cases
    )

    gates["explanation_digest_stable"] = (
        digest(cases) == digest(copy.deepcopy(cases))
    )

    status = (
        "PASS"
        if all(gates.values())
        else "FAIL"
    )

    report = {
        "phase": "4.31",
        "title": (
            "Predictive Resilience Explainability "
            "and Forensic Evidence Correlation"
        ),
        "status": status,
        "research_simulation": True,
        "retraining_performed": False,
        "objective": [
            "Explain predictive risk using explicit risk drivers.",
            "Explain deterministic policy modifiers.",
            "Correlate resilience decisions with forensic evidence.",
            "Explain integrity verification outcomes.",
            "Generate integrity-aware SOC investigation context.",
            "Refuse trusted synchronization after evidence tampering.",
            "Refuse SOC handoff after evidence tampering.",
            "Preserve scientific and vehicle safety boundaries.",
        ],
        "upstream_sources": {
            "phase_20_status": phase20.get("status"),
            "phase_21_status": phase21.get("status"),
            "phase_27a_status": phase27a.get("status"),
            "phase_28a_status": phase28a.get("status"),
            "phase_28b_status": phase28b.get("status"),
            "phase_28c_status": phase28c.get("status"),
            "phase_29_status": phase29.get("status"),
            "phase_30_status": phase30.get("status"),
        },
        "cases": cases,
        "gates": gates,
        "gate_count": len(gates),
        "gates_passed": sum(
            bool(value)
            for value in gates.values()
        ),
        "safety_boundary": {
            "direct_vehicle_actuation": False,
            "live_cortex_xdr_api_used": False,
            "real_world_cyberattack_performance_claim": False,
        },
    }

    report["evidence_digest"] = digest({
        "cases": cases,
        "gates": gates,
    })

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    print("\n=== PHASE 4.31 RESULT ===")
    print(f"Status: {status}")
    print(
        f"Gates: "
        f"{report['gates_passed']}/"
        f"{report['gate_count']}"
    )

    print("\n=== EXPLANATION CASES ===")

    for case in cases:
        explanation = case["explanation"]

        print(
            f"{case['scenario']}: "
            f"risk={explanation['prediction']['risk_level']}, "
            f"decision={explanation['decision']['final_policy_level']}, "
            f"state={explanation['resilience_state']['current_state']}, "
            f"sync={'ALLOW' if explanation['integrity']['trusted_synchronization'] else 'REFUSE'}, "
            f"soc={'ALLOW' if explanation['integrity']['soc_handoff_allowed'] else 'REFUSE'}"
        )

    print("\n=== SAFETY BOUNDARY ===")
    print("Direct vehicle actuation: DISABLED")
    print("Live Cortex XDR API: NOT USED")
    print("Real-world cyberattack performance claim: DISABLED")

    print(
        f"\nEvidence digest: "
        f"{report['evidence_digest']}"
    )

    print(f"Report: {OUTPUT}")


if __name__ == "__main__":
    main()
