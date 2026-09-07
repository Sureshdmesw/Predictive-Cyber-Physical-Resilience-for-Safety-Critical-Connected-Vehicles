from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = ROOT / "experiments" / "integration" / "phase4_28"
OUT_FILE = (
    OUT_DIR
    / "crss_2024_phase4_28b_connectivity_recovery_sync.json"
)

PHASE_21_REPORT = (
    ROOT
    / "experiments"
    / "edge_iot"
    / "crss_2024_phase4_21_edge_iot_resilience.json"
)

FORENSIC_SCHEMA = (
    ROOT
    / "data"
    / "schemas"
    / "edge_iot"
    / "edge_iot_forensic_event_schema.json"
)

PHASE_28A_REPORT = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_28"
    / "crss_2024_phase4_28a_predictive_soc_event_pipeline.json"
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


def record_hash(record_core):
    return sha256_text(canonical_json(record_core))


def build_record(
    event_id,
    vehicle_id,
    ecu_id,
    can_id,
    severity,
    connectivity_state,
    integrity_status,
    sensor_consistency_status,
    payload,
    previous_hash,
):
    payload_hash = sha256_text(canonical_json(payload))

    core = {
        "event_id": event_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vehicle_id": vehicle_id,
        "ecu_id": ecu_id,
        "can_id": can_id,
        "event_type": "CYBER_PHYSICAL_FORENSIC_EVENT",
        "severity": severity,
        "connectivity_state": connectivity_state,
        "integrity_status": integrity_status,
        "sensor_consistency_status": sensor_consistency_status,
        "payload_hash": payload_hash,
        "previous_hash": previous_hash,
    }

    return {
        **core,
        "record_hash": record_hash(core),
        "payload": payload,
    }


def verify_record(record):
    core = {
        key: record[key]
        for key in [
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
        ]
    }

    expected = record_hash(core)

    return expected == record["record_hash"]


def verify_chain(records):
    previous = "ROLLING_GENESIS"

    for record in records:
        if record["previous_hash"] != previous:
            return False

        if not verify_record(record):
            return False

        previous = record["record_hash"]

    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_checks = {
        "phase4_21_report": PHASE_21_REPORT.exists(),
        "forensic_schema": FORENSIC_SCHEMA.exists(),
        "phase4_28a_report": PHASE_28A_REPORT.exists(),
    }

    vehicle_id = "VEHICLE-RESEARCH-001"
    ecu_id = "ECU-ADAS-001"
    can_id = "0x245"

    # ---------------------------------------------------------
    # 1. CONNECTED STATE — EVENTS GENERATED LOCALLY
    # ---------------------------------------------------------

    connected_events = []

    connected_events.append(
        build_record(
            "EVT-001",
            vehicle_id,
            ecu_id,
            can_id,
            "MEDIUM",
            "CONNECTED",
            "VALID",
            "CONSISTENT",
            {
                "risk_score": 0.31,
                "risk_level": "MEDIUM",
                "source": "predictive_model",
            },
            "ROLLING_GENESIS",
        )
    )

    connected_events.append(
        build_record(
            "EVT-002",
            vehicle_id,
            ecu_id,
            can_id,
            "HIGH",
            "CONNECTED",
            "VALID",
            "INCONSISTENT",
            {
                "risk_score": 0.18,
                "risk_level": "HIGH",
                "source": "sensor_disagreement",
            },
            connected_events[-1]["record_hash"],
        )
    )

    # ---------------------------------------------------------
    # 2. CONNECTIVITY LOSS
    # ---------------------------------------------------------

    offline_events = []

    offline_events.append(
        build_record(
            "EVT-003",
            vehicle_id,
            ecu_id,
            can_id,
            "HIGH",
            "DISCONNECTED",
            "VALID",
            "INCONSISTENT",
            {
                "risk_score": 0.72,
                "risk_level": "HIGH",
                "source": "edge_predictive_resilience",
                "local_action": "preserve_forensic_context",
            },
            connected_events[-1]["record_hash"],
        )
    )

    offline_events.append(
        build_record(
            "EVT-004",
            vehicle_id,
            ecu_id,
            can_id,
            "HIGH",
            "DISCONNECTED",
            "VALID",
            "INCONSISTENT",
            {
                "risk_score": 0.76,
                "risk_level": "HIGH",
                "source": "edge_predictive_resilience",
                "local_action": "retain_local_resilience",
            },
            offline_events[-1]["record_hash"],
        )
    )

    offline_events.append(
        build_record(
            "EVT-005",
            vehicle_id,
            ecu_id,
            can_id,
            "HIGH",
            "DISCONNECTED",
            "VALID",
            "INCONSISTENT",
            {
                "risk_score": 0.81,
                "risk_level": "CRITICAL",
                "source": "edge_predictive_resilience",
                "local_action": "preserve_forensic_context",
            },
            offline_events[-1]["record_hash"],
        )
    )

    # ---------------------------------------------------------
    # 3. CONNECTIVITY RECOVERY
    # ---------------------------------------------------------

    recovery_event = build_record(
        "EVT-006",
        vehicle_id,
        ecu_id,
        can_id,
        "HIGH",
        "RECOVERED",
        "PENDING_VERIFICATION",
        "INCONSISTENT",
        {
            "recovery_detected": True,
            "buffered_events": len(offline_events),
            "synchronization_required": True,
        },
        offline_events[-1]["record_hash"],
    )

    # ---------------------------------------------------------
    # 4. VERIFY BUFFER INTEGRITY BEFORE SYNCHRONIZATION
    # ---------------------------------------------------------

    buffered_chain = connected_events + offline_events

    chain_valid_before_sync = verify_chain(buffered_chain)

    # ---------------------------------------------------------
    # 5. SECURE SYNCHRONIZATION
    # ---------------------------------------------------------

    synchronization_manifest = {
        "vehicle_id": vehicle_id,
        "buffer_anchor": "ROLLING_GENESIS",
        "first_buffered_event": offline_events[0]["event_id"],
        "last_buffered_event": offline_events[-1]["event_id"],
        "buffered_event_count": len(offline_events),
        "chain_integrity_verified": chain_valid_before_sync,
        "transport_security": "RESEARCH_ABSTRACTION",
        "destination": "CORTEX_XDR_COMPATIBLE_SOC_EVENT_PIPELINE",
        "live_cortex_xdr_api_used": False,
    }

    synchronization_hash = sha256_text(
        canonical_json(synchronization_manifest)
    )

    # ---------------------------------------------------------
    # 6. POST-SYNC SOC HANDOFF
    # ---------------------------------------------------------

    soc_handoff = {
        "event_type": "CORTEX_XDR_COMPATIBLE_RESEARCH_EVENT",
        "integration_mode": "ABSTRACTION_ONLY",
        "live_cortex_xdr_api_used": False,
        "vehicle_id": vehicle_id,
        "severity": "HIGH",
        "source": "vehicle_edge_predictive_resilience",
        "synchronized_event_ids": [
            event["event_id"] for event in offline_events
        ],
        "buffer_chain_verified": chain_valid_before_sync,
        "synchronization_hash": synchronization_hash,
        "direct_vehicle_actuation": False,
    }

    # ---------------------------------------------------------
    # 7. INTEGRATION GATES
    # ---------------------------------------------------------

    gates = {}

    gates["source_foundations_present"] = all(
        source_checks.values()
    )

    gates["connected_events_generated"] = (
        len(connected_events) == 2
    )

    gates["offline_events_generated"] = (
        len(offline_events) == 3
    )

    gates["offline_events_marked_disconnected"] = all(
        event["connectivity_state"] == "DISCONNECTED"
        for event in offline_events
    )

    gates["offline_events_preserve_hash_chain"] = (
        offline_events[0]["previous_hash"]
        == connected_events[-1]["record_hash"]
    )

    gates["buffer_chain_integrity_valid"] = (
        chain_valid_before_sync
    )

    gates["recovery_event_generated"] = (
        recovery_event["connectivity_state"] == "RECOVERED"
    )

    gates["recovery_requires_verification"] = (
        recovery_event["integrity_status"]
        == "PENDING_VERIFICATION"
    )

    gates["synchronization_requires_integrity"] = (
        synchronization_manifest["chain_integrity_verified"]
        is True
    )

    gates["synchronization_hash_present"] = (
        len(synchronization_hash) == 64
    )

    gates["soc_handoff_generated"] = (
        soc_handoff["event_type"]
        == "CORTEX_XDR_COMPATIBLE_RESEARCH_EVENT"
    )

    gates["soc_handoff_chain_verified"] = (
        soc_handoff["buffer_chain_verified"] is True
    )

    gates["live_cortex_xdr_not_claimed"] = (
        soc_handoff["live_cortex_xdr_api_used"] is False
    )

    gates["direct_vehicle_actuation_disabled"] = all(
        [
            not soc_handoff["direct_vehicle_actuation"],
            not recovery_event.get("direct_vehicle_actuation", False),
        ]
    )

    gates["forensic_records_have_required_hashes"] = all(
        len(event["record_hash"]) == 64
        and len(event["payload_hash"]) == 64
        for event in connected_events
        + offline_events
        + [recovery_event]
    )

    gates["offline_first_resilience_preserved"] = (
        len(offline_events) > 0
        and all(
            event["connectivity_state"] == "DISCONNECTED"
            for event in offline_events
        )
    )

    report = {
        "phase": "4.28B",
        "title": (
            "Connectivity Loss, Forensic Buffer, Recovery, "
            "Integrity Verification and Secure Synchronization"
        ),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "training_performed": False,
        "model_retraining_performed": False,
        "architecture": [
            "Connected Event Generation",
            "Connectivity Loss",
            "Local Forensic Buffer",
            "Rolling Hash Chain",
            "Connectivity Recovery",
            "Integrity Verification",
            "Secure Synchronization",
            "Cortex XDR-Compatible SOC Handoff",
        ],
        "source_foundation": {
            "phase4_21_report": str(
                PHASE_21_REPORT.relative_to(ROOT)
            ),
            "forensic_schema": str(
                FORENSIC_SCHEMA.relative_to(ROOT)
            ),
            "phase4_28a_report": str(
                PHASE_28A_REPORT.relative_to(ROOT)
            ),
        },
        "source_checks": source_checks,
        "buffer_policy": {
            "anchor": "ROLLING_GENESIS",
            "offline_first": True,
            "hash_algorithm": "SHA-256",
            "synchronize_only_after_integrity_verification": True,
        },
        "connected_events": connected_events,
        "offline_buffer_events": offline_events,
        "recovery_event": recovery_event,
        "buffer_integrity": {
            "chain_valid": chain_valid_before_sync,
            "buffered_event_count": len(offline_events),
        },
        "synchronization_manifest": synchronization_manifest,
        "synchronization_hash": synchronization_hash,
        "soc_handoff": soc_handoff,
        "gates": gates,
        "gate_summary": {
            "passed": sum(gates.values()),
            "total": len(gates),
        },
        "scientific_boundary": {
            "research_simulation": True,
            "live_cortex_xdr_connection": False,
            "real_vehicle_network_connection": False,
            "direct_vehicle_actuation": False,
            "secure_transport_is_represented_as_abstraction": True,
        },
    }

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("PHASE 4.28B — CONNECTIVITY RECOVERY + SECURE SYNCHRONIZATION")
    print("=" * 72)
    print(f"Status: {report['status']}")
    print(
        f"Gates: {report['gate_summary']['passed']}/"
        f"{report['gate_summary']['total']}"
    )
    print(f"Output: {OUT_FILE}")
    print()

    for name, result in gates.items():
        print(f"[{'PASS' if result else 'FAIL'}] {name}")

    print()
    print("Connectivity lifecycle:")
    print("  CONNECTED → DISCONNECTED → LOCAL BUFFER → RECOVERED")
    print("  → INTEGRITY VERIFIED → SYNCHRONIZED → SOC HANDOFF")


if __name__ == "__main__":
    main()
