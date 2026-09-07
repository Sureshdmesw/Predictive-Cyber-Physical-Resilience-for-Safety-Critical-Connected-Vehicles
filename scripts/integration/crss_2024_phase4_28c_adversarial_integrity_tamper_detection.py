import json
import hashlib
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]

INPUT_4_28A = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_28"
    / "crss_2024_phase4_28a_predictive_soc_event_pipeline.json"
)

INPUT_4_28B = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_28"
    / "crss_2024_phase4_28b_connectivity_recovery_sync.json"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_28c"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "crss_2024_phase4_28c_adversarial_integrity_tamper_detection.json"
)


def canonical_json(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_object(obj):
    return sha256_text(canonical_json(obj))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def recompute_payload_hash(event):
    payload = event.get("payload", {})
    return sha256_object(payload)


def recompute_record_hash(event):
    material = {
        "event_id": event.get("event_id"),
        "timestamp": event.get("timestamp"),
        "vehicle_id": event.get("vehicle_id"),
        "ecu_id": event.get("ecu_id"),
        "can_id": event.get("can_id"),
        "event_type": event.get("event_type"),
        "severity": event.get("severity"),
        "connectivity_state": event.get("connectivity_state"),
        "integrity_status": event.get("integrity_status"),
        "sensor_consistency_status": event.get(
            "sensor_consistency_status"
        ),
        "payload_hash": event.get("payload_hash"),
        "previous_hash": event.get("previous_hash"),
    }
    return sha256_object(material)


def validate_event(event):
    required = [
        "event_id",
        "timestamp",
        "vehicle_id",
        "ecu_id",
        "can_id",
        "event_type",
        "severity",
        "connectivity_state",
        "integrity_status",
        "sensor_consistency_status",
        "payload_hash",
        "previous_hash",
        "record_hash",
    ]

    missing = [
        field for field in required
        if field not in event
    ]

    payload_hash_expected = recompute_payload_hash(event)
    record_hash_expected = recompute_record_hash(event)

    payload_valid = (
        event.get("payload_hash") == payload_hash_expected
    )

    record_valid = (
        event.get("record_hash") == record_hash_expected
    )

    return {
        "missing_required_fields": missing,
        "payload_hash_valid": payload_valid,
        "record_hash_valid": record_valid,
        "event_valid": (
            len(missing) == 0
            and payload_valid
            and record_valid
        ),
    }


def validate_chain(events):
    previous = "ROLLING_GENESIS"
    results = []

    for event in events:
        previous_matches = (
            event.get("previous_hash") == previous
        )

        validation = validate_event(event)

        results.append({
            "event_id": event.get("event_id"),
            "previous_hash_matches": previous_matches,
            "payload_hash_valid": validation["payload_hash_valid"],
            "record_hash_valid": validation["record_hash_valid"],
            "event_valid": validation["event_valid"],
            "chain_valid": (
                previous_matches
                and validation["event_valid"]
            ),
        })

        previous = event.get("record_hash")

    return {
        "chain_valid": all(
            result["chain_valid"]
            for result in results
        ),
        "results": results,
    }


def extract_forensic_events(report):
    candidates = []

    def walk(value):
        if isinstance(value, dict):
            if (
                "event_id" in value
                and "record_hash" in value
                and "payload_hash" in value
            ):
                candidates.append(value)

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(report)

    unique = {}
    for event in candidates:
        unique[event["event_id"]] = event

    return list(unique.values())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_4_28A, "r", encoding="utf-8") as f:
        report_4_28a = json.load(f)

    with open(INPUT_4_28B, "r", encoding="utf-8") as f:
        report_4_28b = json.load(f)

    events_a = extract_forensic_events(report_4_28a)
    events_b = extract_forensic_events(report_4_28b)

    source_events = events_b if events_b else events_a

    if not source_events:
        raise RuntimeError(
            "No forensic events were found in Phase 4.28A/B reports."
        )

    # Prefer an event with an actual payload.
    source_event = None

    for event in source_events:
        if isinstance(event.get("payload"), dict):
            source_event = event
            break

    if source_event is None:
        raise RuntimeError(
            "No forensic event containing a payload was found."
        )

    # ------------------------------------------------------------------
    # BASELINE
    # ------------------------------------------------------------------

    baseline_event = deepcopy(source_event)

    baseline_validation = validate_event(baseline_event)

    baseline_chain = validate_chain(
        [baseline_event]
    )

    # ------------------------------------------------------------------
    # TAMPER SCENARIO
    # ------------------------------------------------------------------

    tampered_event = deepcopy(source_event)

    original_payload = deepcopy(
        tampered_event.get("payload", {})
    )

    if isinstance(tampered_event.get("payload"), dict):
        tampered_event["payload"]["tamper_test_marker"] = (
            "ADVERSARIAL_MODIFICATION"
        )
    else:
        tampered_event["payload"] = (
            "ADVERSARIAL_MODIFICATION"
        )

    tamper_payload_hash = recompute_payload_hash(
        tampered_event
    )

    # Intentionally DO NOT update the stored payload_hash.
    # This models evidence modification after recording.

    tampered_validation = validate_event(
        tampered_event
    )

    tampered_chain = validate_chain(
        [tampered_event]
    )

    # ------------------------------------------------------------------
    # CHAIN-LINK TAMPER SCENARIO
    # ------------------------------------------------------------------

    chain_tampered_event = deepcopy(source_event)

    original_previous_hash = chain_tampered_event.get(
        "previous_hash"
    )

    chain_tampered_event["previous_hash"] = (
        "0" * 64
    )

    chain_tampered_validation = validate_event(
        chain_tampered_event
    )

    chain_tampered_chain = validate_chain(
        [chain_tampered_event]
    )

    # ------------------------------------------------------------------
    # SYNCHRONIZATION POLICY
    # ------------------------------------------------------------------

    def synchronization_decision(validation, chain):
        verified = (
            validation["event_valid"]
            and chain["chain_valid"]
        )

        return {
            "integrity_verified": verified,
            "synchronization_allowed": verified,
            "soc_handoff_allowed": verified,
            "evidence_state": (
                "VERIFIED"
                if verified
                else "UNVERIFIED"
            ),
            "refusal_reason": (
                None
                if verified
                else "INTEGRITY_VERIFICATION_FAILED"
            ),
        }

    baseline_sync = synchronization_decision(
        baseline_validation,
        baseline_chain,
    )

    tampered_sync = synchronization_decision(
        tampered_validation,
        tampered_chain,
    )

    chain_tampered_sync = synchronization_decision(
        chain_tampered_validation,
        chain_tampered_chain,
    )

    # ------------------------------------------------------------------
    # GATES
    # ------------------------------------------------------------------

    gates = {
        "source_4_28a_pass": (
            report_4_28a.get("status") == "PASS"
        ),

        "source_4_28b_pass": (
            report_4_28b.get("status") == "PASS"
        ),

        "baseline_payload_integrity_valid": (
            baseline_validation["payload_hash_valid"]
        ),

        "baseline_record_integrity_valid": (
            baseline_validation["record_hash_valid"]
        ),

        "baseline_chain_valid": (
            baseline_chain["chain_valid"]
        ),

        "baseline_synchronization_allowed": (
            baseline_sync["synchronization_allowed"]
        ),

        "baseline_soc_handoff_allowed": (
            baseline_sync["soc_handoff_allowed"]
        ),

        "payload_tampering_detected": (
            not tampered_validation["payload_hash_valid"]
        ),

        "payload_tampered_event_rejected": (
            not tampered_validation["event_valid"]
        ),

        "payload_tampered_chain_rejected": (
            not tampered_chain["chain_valid"]
        ),

        "tampered_evidence_marked_unverified": (
            tampered_sync["evidence_state"] == "UNVERIFIED"
        ),

        "tampered_synchronization_refused": (
            not tampered_sync["synchronization_allowed"]
        ),

        "tampered_soc_handoff_refused": (
            not tampered_sync["soc_handoff_allowed"]
        ),

        "previous_hash_tampering_detected": (
            not chain_tampered_chain["chain_valid"]
        ),

        "previous_hash_tampered_sync_refused": (
            not chain_tampered_sync[
                "synchronization_allowed"
            ]
        ),

        "direct_vehicle_actuation_disabled": True,

        "scientific_real_attack_claim_disabled": True,

        "live_cortex_xdr_api_claim_disabled": True,
    }

    passed = sum(
        1 for value in gates.values()
        if value
    )

    total = len(gates)

    status = (
        "PASS"
        if passed == total
        else "FAIL"
    )

    report = {
        "phase": "4.28C",
        "title": (
            "Adversarial Integrity and Tamper Detection"
        ),
        "status": status,
        "timestamp_utc": utc_now(),
        "research_simulation": True,
        "purpose": (
            "Demonstrate that modification of buffered "
            "forensic evidence is detected and that "
            "unverified evidence cannot be synchronized "
            "or handed to the SOC."
        ),
        "source_artifacts": {
            "phase_4_28a": str(INPUT_4_28A),
            "phase_4_28b": str(INPUT_4_28B),
        },
        "source_event": {
            "event_id": source_event.get("event_id"),
            "vehicle_id": source_event.get("vehicle_id"),
            "ecu_id": source_event.get("ecu_id"),
        },
        "baseline": {
            "validation": baseline_validation,
            "chain": baseline_chain,
            "synchronization": baseline_sync,
        },
        "payload_tamper_test": {
            "original_payload": original_payload,
            "tampered_payload": tampered_event.get(
                "payload"
            ),
            "tampered_payload_hash_recomputed": (
                tamper_payload_hash
            ),
            "stored_payload_hash": tampered_event.get(
                "payload_hash"
            ),
            "validation": tampered_validation,
            "chain": tampered_chain,
            "synchronization": tampered_sync,
        },
        "previous_hash_tamper_test": {
            "original_previous_hash": original_previous_hash,
            "tampered_previous_hash": (
                chain_tampered_event.get(
                    "previous_hash"
                )
            ),
            "validation": chain_tampered_validation,
            "chain": chain_tampered_chain,
            "synchronization": chain_tampered_sync,
        },
        "security_policy": {
            "unverified_evidence": (
                "REFUSE_SYNCHRONIZATION_AND_SOC_HANDOFF"
            ),
            "verified_evidence": (
                "ALLOW_SYNCHRONIZATION_AND_SOC_HANDOFF"
            ),
            "direct_vehicle_actuation": False,
        },
        "scientific_boundary": {
            "synthetic_telemetry_labels": True,
            "real_world_cyberattack_performance_claim": False,
            "live_cortex_xdr_api_used": False,
            "direct_vehicle_actuation": False,
        },
        "gates": gates,
        "gate_summary": {
            "passed": passed,
            "total": total,
        },
    }

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== PHASE 4.28C ===")
    print(
        f"Status: {status}"
    )
    print(
        f"Gates: {passed}/{total}"
    )
    print(
        "Baseline chain valid:",
        baseline_chain["chain_valid"],
    )
    print(
        "Payload tampering detected:",
        not tampered_validation[
            "payload_hash_valid"
        ],
    )
    print(
        "Tampered synchronization allowed:",
        tampered_sync[
            "synchronization_allowed"
        ],
    )
    print(
        "Tampered SOC handoff allowed:",
        tampered_sync[
            "soc_handoff_allowed"
        ],
    )
    print(
        "Previous-hash tampering detected:",
        not chain_tampered_chain[
            "chain_valid"
        ],
    )
    print(
        "Report:",
        OUTPUT_REPORT,
    )


if __name__ == "__main__":
    main()

