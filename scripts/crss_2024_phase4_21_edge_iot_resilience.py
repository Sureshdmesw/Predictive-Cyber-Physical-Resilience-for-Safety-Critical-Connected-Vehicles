from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "edge_iot"
SCHEMA_DIR = PROJECT_ROOT / "data" / "schemas" / "edge_iot"

REPORT_PATH = EXPERIMENT_DIR / "crss_2024_phase4_21_edge_iot_resilience.json"
SCHEMA_PATH = SCHEMA_DIR / "edge_iot_forensic_event_schema.json"


@dataclass
class EdgeEvent:
    event_id: str
    timestamp: float
    vehicle_id: str
    ecu_id: str
    can_id: str
    event_type: str
    severity: str
    connectivity_state: str
    integrity_status: str
    sensor_consistency_status: str
    payload_hash: str
    previous_hash: str
    record_hash: str


class TamperEvidentForensicBuffer:
    """Bounded local forensic buffer using a hash chain.

    This is a research simulation. It does not control a real vehicle,
    transmit CAN commands, or actuate safety-critical functions.
    """

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self.records: deque[EdgeEvent] = deque(maxlen=capacity)

    @staticmethod
    def _hash_payload(payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def append(self, event: dict[str, Any]) -> EdgeEvent:
        previous_hash = (
            self.records[-1].record_hash if self.records else "GENESIS"
        )

        payload_hash = self._hash_payload(event)

        record_material = {
            "event": event,
            "payload_hash": payload_hash,
            "previous_hash": previous_hash,
        }

        record_hash = self._hash_payload(record_material)

        record = EdgeEvent(
            event_id=str(event["event_id"]),
            timestamp=float(event["timestamp"]),
            vehicle_id=str(event["vehicle_id"]),
            ecu_id=str(event["ecu_id"]),
            can_id=str(event["can_id"]),
            event_type=str(event["event_type"]),
            severity=str(event["severity"]),
            connectivity_state=str(event["connectivity_state"]),
            integrity_status=str(event["integrity_status"]),
            sensor_consistency_status=str(
                event["sensor_consistency_status"]
            ),
            payload_hash=payload_hash,
            previous_hash=previous_hash,
            record_hash=record_hash,
        )

        self.records.append(record)
        return record

    def verify_integrity(self) -> bool:
        previous_hash = "GENESIS"

        for record in self.records:
            if record.previous_hash != previous_hash:
                return False

            event = {
                "event_id": record.event_id,
                "timestamp": record.timestamp,
                "vehicle_id": record.vehicle_id,
                "ecu_id": record.ecu_id,
                "can_id": record.can_id,
                "event_type": record.event_type,
                "severity": record.severity,
                "connectivity_state": record.connectivity_state,
                "integrity_status": record.integrity_status,
                "sensor_consistency_status": record.sensor_consistency_status,
            }

            payload_hash = self._hash_payload(event)

            if payload_hash != record.payload_hash:
                return False

            record_material = {
                "event": event,
                "payload_hash": payload_hash,
                "previous_hash": record.previous_hash,
            }

            expected_record_hash = self._hash_payload(record_material)

            if expected_record_hash != record.record_hash:
                return False

            previous_hash = record.record_hash

        return True

    def export(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]


def deterministic_containment_policy(event: dict[str, Any]) -> dict[str, Any]:
    """Map security observations to bounded containment recommendations.

    No direct vehicle actuation is performed.
    """

    integrity_failure = event["integrity_status"] == "FAIL"
    sensor_disagreement = event["sensor_consistency_status"] == "DISAGREEMENT"
    connectivity_degraded = event["connectivity_state"] != "CONNECTED"

    if integrity_failure:
        level = "HIGH"
        action = "ISOLATE_SUSPECT_COMMUNICATION_PATH"
        soc_priority = "HIGH"
    elif sensor_disagreement:
        level = "MEDIUM"
        action = "QUARANTINE_SUSPECT_TELEMETRY_SOURCE"
        soc_priority = "MEDIUM"
    elif connectivity_degraded:
        level = "MEDIUM"
        action = "ENTER_LOCAL_RESILIENCE_MODE"
        soc_priority = "MEDIUM"
    else:
        level = "NORMAL"
        action = "CONTINUE_MONITORING"
        soc_priority = "ROUTINE"

    return {
        "containment_level": level,
        "containment_recommendation": action,
        "soc_priority": soc_priority,
        "direct_vehicle_actuation": False,
    }


def build_cortex_xdr_event(
    event: dict[str, Any],
    policy: dict[str, Any],
    record_hash: str,
) -> dict[str, Any]:
    """Create a Cortex XDR-compatible research abstraction.

    This does not require or claim access to a Cortex XDR tenant/API.
    """

    return {
        "event_source": "EDGE_IOT_RESEARCH_SIMULATION",
        "target_platform": "Cortex XDR",
        "vehicle_id": event["vehicle_id"],
        "ecu_id": event["ecu_id"],
        "can_id": event["can_id"],
        "event_type": event["event_type"],
        "severity": event["severity"],
        "connectivity_state": event["connectivity_state"],
        "integrity_status": event["integrity_status"],
        "sensor_consistency_status": event[
            "sensor_consistency_status"
        ],
        "containment_level": policy["containment_level"],
        "containment_recommendation": policy[
            "containment_recommendation"
        ],
        "soc_priority": policy["soc_priority"],
        "forensic_record_hash": record_hash,
        "direct_vehicle_actuation": False,
    }


def create_synthetic_events() -> list[dict[str, Any]]:
    base_time = time.time()

    return [
        {
            "event_id": "EDGE-0001",
            "timestamp": base_time,
            "vehicle_id": "SIM-VEH-001",
            "ecu_id": "ECU-GW",
            "can_id": "0x120",
            "event_type": "NORMAL_CAN_TRAFFIC",
            "severity": "INFO",
            "connectivity_state": "CONNECTED",
            "integrity_status": "PASS",
            "sensor_consistency_status": "CONSISTENT",
        },
        {
            "event_id": "EDGE-0002",
            "timestamp": base_time + 5,
            "vehicle_id": "SIM-VEH-001",
            "ecu_id": "ECU-GW",
            "can_id": "0x120",
            "event_type": "CONNECTIVITY_DEGRADATION",
            "severity": "MEDIUM",
            "connectivity_state": "DEGRADED",
            "integrity_status": "PASS",
            "sensor_consistency_status": "CONSISTENT",
        },
        {
            "event_id": "EDGE-0003",
            "timestamp": base_time + 10,
            "vehicle_id": "SIM-VEH-001",
            "ecu_id": "ECU-ADAS",
            "can_id": "0x245",
            "event_type": "SENSOR_DISAGREEMENT",
            "severity": "MEDIUM",
            "connectivity_state": "DEGRADED",
            "integrity_status": "PASS",
            "sensor_consistency_status": "DISAGREEMENT",
        },
        {
            "event_id": "EDGE-0004",
            "timestamp": base_time + 15,
            "vehicle_id": "SIM-VEH-001",
            "ecu_id": "ECU-GW",
            "can_id": "0x310",
            "event_type": "MESSAGE_INTEGRITY_FAILURE",
            "severity": "HIGH",
            "connectivity_state": "OFFLINE",
            "integrity_status": "FAIL",
            "sensor_consistency_status": "DISAGREEMENT",
        },
    ]


def main() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

    buffer = TamperEvidentForensicBuffer(capacity=256)

    event_results = []
    cortex_events = []

    for event in create_synthetic_events():
        policy = deterministic_containment_policy(event)
        record = buffer.append(event)

        cortex_event = build_cortex_xdr_event(
            event,
            policy,
            record.record_hash,
        )

        event_results.append(
            {
                "event_id": event["event_id"],
                "policy": policy,
                "forensic_record_hash": record.record_hash,
            }
        )

        cortex_events.append(cortex_event)

    integrity_before_sync = buffer.verify_integrity()

    # Simulate connectivity restoration and secure synchronization.
    connectivity_restored = True
    synchronized_records = []

    if connectivity_restored and integrity_before_sync:
        synchronized_records = buffer.export()

    integrity_after_sync = buffer.verify_integrity()

    checks = {
        "events_generated": len(event_results) == 4,
        "forensic_buffer_nonempty": len(buffer.records) == 4,
        "hash_chain_integrity_before_sync": integrity_before_sync,
        "connectivity_restoration_detected": connectivity_restored,
        "secure_sync_requires_integrity": (
            integrity_before_sync and len(synchronized_records) == 4
        ),
        "hash_chain_integrity_after_sync": integrity_after_sync,
        "cortex_xdr_events_generated": len(cortex_events) == 4,
        "no_direct_vehicle_actuation": all(
            not item["direct_vehicle_actuation"]
            for item in cortex_events
        ),
        "offline_first_behavior": any(
            item["connectivity_state"] == "OFFLINE"
            for item in cortex_events
        ),
    }

    report = {
        "phase": "4.21",
        "title": "Edge-IoT Local Containment and Tamper-Evident Forensic Buffer",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "research_simulation": True,
        "real_vehicle_actuation": False,
        "cortex_xdr_integration_mode": "ABSTRACTION_ONLY",
        "architecture": [
            "Edge-IoT / ECU / CAN event observation",
            "Local deterministic containment recommendation",
            "Bounded tamper-evident forensic buffer",
            "Connectivity restoration",
            "Integrity verification",
            "Secure synchronization",
            "Cortex XDR event abstraction",
            "SOC investigation",
        ],
        "buffer_capacity": buffer.capacity,
        "event_count": len(event_results),
        "checks": checks,
        "event_results": event_results,
        "cortex_xdr_events": cortex_events,
        "synchronized_forensic_records": synchronized_records,
        "safety_boundary": (
            "ML and this research simulation provide decision support, "
            "warning, containment recommendation, and SOC prioritization. "
            "No direct safety-critical vehicle actuation is performed."
        ),
        "limitations": [
            "Synthetic Edge-IoT/CAN events are used.",
            "Cyberattack labels are simulated research labels.",
            "Cortex XDR is represented as an ingestion/event abstraction.",
            "Containment policy is deterministic engineering logic, "
            "not empirically validated vehicle control logic.",
        ],
    }

    schema = {
        "schema_name": "edge_iot_forensic_event",
        "schema_version": "1.0",
        "description": (
            "Research schema for offline-first Edge-IoT forensic events "
            "and Cortex XDR ingestion abstraction."
        ),
        "required_fields": [
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
        ],
        "safety_boundary": {
            "direct_vehicle_actuation": False,
            "purpose": [
                "detection",
                "forensics",
                "containment_recommendation",
                "SOC_prioritization",
            ],
        },
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    SCHEMA_PATH.write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("PHASE 4.21 EDGE-IOT RESILIENCE")
    print("=" * 72)
    print(f"Status: {report['status']}")
    print(f"Events: {len(event_results)}")
    print(f"Forensic records: {len(synchronized_records)}")
    print(f"Integrity before sync: {integrity_before_sync}")
    print(f"Integrity after sync:  {integrity_after_sync}")
    print(f"Cortex XDR abstractions: {len(cortex_events)}")
    print(f"Direct vehicle actuation: {report['real_vehicle_actuation']}")
    print()
    print(f"Report: {REPORT_PATH}")
    print(f"Schema: {SCHEMA_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
