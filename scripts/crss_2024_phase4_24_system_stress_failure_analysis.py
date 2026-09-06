from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE_421_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "crss_2024_phase4_21_edge_iot_resilience.py"
)

PHASE_423_REPORT = (
    PROJECT_ROOT
    / "experiments"
    / "edge_iot"
    / "crss_2024_phase4_23_end_to_end_resilience.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "edge_iot"
    / "crss_2024_phase4_24_system_stress_failure_analysis.json"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_event(
    event_id: str,
    connectivity: str = "CONNECTED",
    integrity: str = "PASS",
    sensor: str = "CONSISTENT",
    severity: str = "INFO",
) -> dict[str, Any]:

    return {
        "event_id": event_id,
        "timestamp": 0.0,
        "vehicle_id": "STRESS-VEH-001",
        "ecu_id": "ECU-GW",
        "can_id": "0x310",
        "event_type": "SYSTEM_STRESS_EVENT",
        "severity": severity,
        "connectivity_state": connectivity,
        "integrity_status": integrity,
        "sensor_consistency_status": sensor,
    }


def evaluate_event(edge_module, event):
    policy = edge_module.deterministic_containment_policy(event)

    return {
        "event_id": event["event_id"],
        "connectivity": event["connectivity_state"],
        "integrity": event["integrity_status"],
        "sensor": event["sensor_consistency_status"],
        "containment_level": policy["containment_level"],
        "containment_recommendation": policy[
            "containment_recommendation"
        ],
        "soc_priority": policy["soc_priority"],
        "direct_vehicle_actuation": policy[
            "direct_vehicle_actuation"
        ],
    }


def run_buffer_capacity_test(edge_module):
    buffer = edge_module.TamperEvidentForensicBuffer(
        capacity=16
    )

    for index in range(100):
        buffer.append(
            build_event(
                f"CAP-{index:03d}"
            )
        )

    return {
        "capacity": buffer.capacity,
        "events_inserted": 100,
        "records_retained": len(buffer.records),
        "bounded": len(buffer.records) <= buffer.capacity,
        "integrity_valid": buffer.verify_integrity(),
    }


def run_repeated_anomaly_test(edge_module):
    buffer = edge_module.TamperEvidentForensicBuffer(
        capacity=64
    )

    results = []

    for index in range(20):
        event = build_event(
            f"REP-{index:03d}",
            connectivity="OFFLINE",
            integrity="FAIL",
            sensor="DISAGREEMENT",
            severity="HIGH",
        )

        policy = edge_module.deterministic_containment_policy(
            event
        )

        buffer.append(event)

        results.append(
            policy["containment_level"]
        )

    return {
        "events": 20,
        "all_high_containment": all(
            level == "HIGH"
            for level in results
        ),
        "integrity_valid": buffer.verify_integrity(),
    }


def run_recovery_test(edge_module):
    buffer = edge_module.TamperEvidentForensicBuffer(
        capacity=64
    )

    offline_event = build_event(
        "RECOVERY-001",
        connectivity="OFFLINE",
        integrity="FAIL",
        sensor="DISAGREEMENT",
        severity="HIGH",
    )

    recovery_event = build_event(
        "RECOVERY-002",
        connectivity="CONNECTED",
        integrity="PASS",
        sensor="CONSISTENT",
        severity="INFO",
    )

    buffer.append(offline_event)
    integrity_during_failure = buffer.verify_integrity()

    buffer.append(recovery_event)
    integrity_after_recovery = buffer.verify_integrity()

    return {
        "integrity_during_failure": integrity_during_failure,
        "integrity_after_recovery": integrity_after_recovery,
        "records_preserved": len(buffer.records) == 2,
        "recovery_integrity_valid": (
            integrity_during_failure
            and integrity_after_recovery
        ),
    }


def run_corrupted_payload_test(edge_module):
    buffer = edge_module.TamperEvidentForensicBuffer(
        capacity=64
    )

    for index in range(5):
        buffer.append(
            build_event(
                f"CORRUPT-{index:03d}"
            )
        )

    integrity_before = buffer.verify_integrity()

    # Simulated telemetry corruption.
    buffer.records[2].severity = "CRITICAL"

    integrity_after = buffer.verify_integrity()

    return {
        "integrity_before_corruption": integrity_before,
        "integrity_after_corruption": integrity_after,
        "corruption_detected": (
            integrity_before
            and not integrity_after
        ),
    }


def run_compound_failure_test(edge_module):
    event = build_event(
        "COMPOUND-001",
        connectivity="OFFLINE",
        integrity="FAIL",
        sensor="DISAGREEMENT",
        severity="CRITICAL",
    )

    result = evaluate_event(
        edge_module,
        event,
    )

    return {
        "result": result,
        "high_containment": (
            result["containment_level"] == "HIGH"
        ),
        "no_direct_actuation": (
            result["direct_vehicle_actuation"] is False
        ),
    }


def main():

    edge_module = load_module(
        PHASE_421_SCRIPT,
        "phase421_stress",
    )

    phase423_report = read_json(
        PHASE_423_REPORT
    )

    upstream_pass = (
        phase423_report.get("status") == "PASS"
    )

    scenarios = {}

    scenarios["connectivity_loss_during_anomaly"] = evaluate_event(
        edge_module,
        build_event(
            "STRESS-001",
            connectivity="OFFLINE",
            integrity="FAIL",
            sensor="CONSISTENT",
            severity="HIGH",
        ),
    )

    scenarios["sensor_disagreement_during_degraded_connectivity"] = (
        evaluate_event(
            edge_module,
            build_event(
                "STRESS-002",
                connectivity="DEGRADED",
                integrity="PASS",
                sensor="DISAGREEMENT",
                severity="MEDIUM",
            ),
        )
    )

    scenarios["simultaneous_cyber_physical_failure"] = evaluate_event(
        edge_module,
        build_event(
            "STRESS-003",
            connectivity="OFFLINE",
            integrity="FAIL",
            sensor="DISAGREEMENT",
            severity="CRITICAL",
        ),
    )

    scenarios["connectivity_recovery_after_failure"] = evaluate_event(
        edge_module,
        build_event(
            "STRESS-004",
            connectivity="CONNECTED",
            integrity="PASS",
            sensor="CONSISTENT",
            severity="INFO",
        ),
    )

    capacity_test = run_buffer_capacity_test(
        edge_module
    )

    repeated_anomaly_test = run_repeated_anomaly_test(
        edge_module
    )

    recovery_test = run_recovery_test(
        edge_module
    )

    corruption_test = run_corrupted_payload_test(
        edge_module
    )

    compound_test = run_compound_failure_test(
        edge_module
    )

    gates = {
        "upstream_phase_4_23_pass": upstream_pass,

        "connectivity_loss_escalates": (
            scenarios[
                "connectivity_loss_during_anomaly"
            ]["containment_level"]
            == "HIGH"
        ),

        "sensor_degradation_escalates": (
            scenarios[
                "sensor_disagreement_during_degraded_connectivity"
            ]["containment_level"]
            == "MEDIUM"
        ),

        "compound_failure_escalates": (
            scenarios[
                "simultaneous_cyber_physical_failure"
            ]["containment_level"]
            == "HIGH"
        ),

        "recovery_returns_to_normal_policy": (
            scenarios[
                "connectivity_recovery_after_failure"
            ]["containment_level"]
            == "NORMAL"
        ),

        "buffer_remains_bounded": (
            capacity_test["bounded"]
            and capacity_test["records_retained"]
            == capacity_test["capacity"]
        ),

        "buffer_integrity_after_capacity_pressure": (
            capacity_test["integrity_valid"]
        ),

        "repeated_anomalies_preserve_integrity": (
            repeated_anomaly_test["all_high_containment"]
            and repeated_anomaly_test["integrity_valid"]
        ),

        "recovery_preserves_forensics": (
            recovery_test["recovery_integrity_valid"]
            and recovery_test["records_preserved"]
        ),

        "payload_corruption_detected": (
            corruption_test["corruption_detected"]
        ),

        "compound_failure_has_no_direct_actuation": (
            compound_test["no_direct_actuation"]
        ),

        "no_retraining": True,
    }

    report = {
        "phase": "4.24",
        "title": (
            "System-Level Stress and Failure Analysis"
        ),
        "status": (
            "PASS"
            if all(gates.values())
            else "FAIL"
        ),
        "research_simulation": True,
        "retraining_performed": False,
        "scenarios": scenarios,
        "buffer_capacity_test": capacity_test,
        "repeated_anomaly_test": repeated_anomaly_test,
        "recovery_test": recovery_test,
        "corruption_test": corruption_test,
        "compound_failure_test": compound_test,
        "gates": gates,
        "stress_dimensions": [
            "connectivity loss",
            "sensor disagreement",
            "communication integrity failure",
            "compound cyber-physical failure",
            "repeated anomalies",
            "buffer capacity pressure",
            "recovery",
            "telemetry corruption",
        ],
        "safety_boundary": (
            "Stress testing evaluates deterministic research-policy "
            "behavior only. No CAN commands are transmitted and no "
            "safety-critical vehicle actuation is performed."
        ),
        "limitations": [
            "Stress scenarios are synthetic.",
            "No real vehicle or ECU is connected.",
            "The forensic buffer is a research implementation.",
            "Containment recommendations are not production vehicle-control logic.",
            "Cortex XDR remains an abstraction.",
        ],
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("PHASE 4.24 SYSTEM STRESS & FAILURE ANALYSIS")
    print("=" * 72)
    print(f"Status: {report['status']}")
    print(f"Retraining: {report['retraining_performed']}")
    print()

    for name, result in scenarios.items():
        print(
            f"{name}: "
            f"containment={result['containment_level']} "
            f"SOC={result['soc_priority']}"
        )

    print()
    print(
        "BUFFER CAPACITY: "
        f"{capacity_test['records_retained']}/"
        f"{capacity_test['capacity']} retained "
        f"integrity={capacity_test['integrity_valid']}"
    )

    print(
        "RECOVERY: "
        f"integrity={recovery_test['recovery_integrity_valid']} "
        f"records={recovery_test['records_preserved']}"
    )

    print(
        "CORRUPTION: "
        f"detected={corruption_test['corruption_detected']}"
    )

    print()
    print("=== GATES ===")

    for name, passed in gates.items():
        print(
            f"{'PASS' if passed else 'FAIL'}: {name}"
        )

    print()
    print(f"Report: {REPORT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
