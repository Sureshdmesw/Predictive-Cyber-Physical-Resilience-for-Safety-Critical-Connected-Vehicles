from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE_421_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "crss_2024_phase4_21_edge_iot_resilience.py"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "edge_iot"
    / "crss_2024_phase4_22_scenario_evaluation.json"
)


def hash_payload(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_phase_421():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "phase421",
        PHASE_421_SCRIPT,
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules["phase421"] = module
    spec.loader.exec_module(module)

    return module


def run_scenario(module, name, modifications):
    events = module.create_synthetic_events()

    for event in events:
        modifications(event)

    buffer = module.TamperEvidentForensicBuffer(capacity=256)

    for event in events:
        buffer.append(event)

    policy_results = [
        module.deterministic_containment_policy(event)
        for event in events
    ]

    integrity = buffer.verify_integrity()

    return {
        "scenario": name,
        "event_count": len(events),
        "integrity_valid": integrity,
        "containment_levels": [
            p["containment_level"] for p in policy_results
        ],
        "containment_recommendations": [
            p["containment_recommendation"]
            for p in policy_results
        ],
        "soc_priorities": [
            p["soc_priority"] for p in policy_results
        ],
        "direct_vehicle_actuation": any(
            p["direct_vehicle_actuation"]
            for p in policy_results
        ),
    }


def tamper_buffer_test(module):
    events = module.create_synthetic_events()

    buffer = module.TamperEvidentForensicBuffer(capacity=256)

    for event in events:
        buffer.append(event)

    integrity_before = buffer.verify_integrity()

    # Simulate unauthorized modification of one forensic record.
    original_hash = buffer.records[2].record_hash

    tampered = buffer.records[2]
    tampered.event_type = "UNAUTHORIZED_MODIFICATION"

    integrity_after = buffer.verify_integrity()

    return {
        "scenario": "FORENSIC_BUFFER_TAMPERING",
        "integrity_before_tampering": integrity_before,
        "integrity_after_tampering": integrity_after,
        "tamper_detected": (
            integrity_before is True
            and integrity_after is False
        ),
        "original_record_hash": original_hash,
        "tampered_record_hash": tampered.record_hash,
    }


def main():
    module = load_phase_421()

    scenarios = []

    scenarios.append(
        run_scenario(
            module,
            "NORMAL_CONNECTED",
            lambda e: None,
        )
    )

    scenarios.append(
        run_scenario(
            module,
            "CONNECTIVITY_DEGRADATION",
            lambda e: e.update(
                connectivity_state="DEGRADED"
            ),
        )
    )

    scenarios.append(
        run_scenario(
            module,
            "CONNECTIVITY_LOSS",
            lambda e: e.update(
                connectivity_state="OFFLINE"
            ),
        )
    )

    scenarios.append(
        run_scenario(
            module,
            "MESSAGE_INTEGRITY_FAILURE",
            lambda e: e.update(
                integrity_status="FAIL"
            ),
        )
    )

    scenarios.append(
        run_scenario(
            module,
            "SENSOR_DISAGREEMENT",
            lambda e: e.update(
                sensor_consistency_status="DISAGREEMENT"
            ),
        )
    )

    scenarios.append(
        run_scenario(
            module,
            "CYBER_CONNECTIVITY_COMBINATION",
            lambda e: e.update(
                connectivity_state="OFFLINE",
                integrity_status="FAIL",
                sensor_consistency_status="DISAGREEMENT",
            ),
        )
    )

    scenarios.append(
        run_scenario(
            module,
            "RECOVERY_AND_SYNCHRONIZATION",
            lambda e: e.update(
                connectivity_state="CONNECTED"
            ),
        )
    )

    tamper_result = tamper_buffer_test(module)

    gates = {
        "normal_operation_safe": (
            scenarios[0]["integrity_valid"]
            and not scenarios[0]["direct_vehicle_actuation"]
        ),
        "connectivity_degradation_detected": (
            "MEDIUM" in scenarios[1]["containment_levels"]
        ),
        "connectivity_loss_detected": (
            "MEDIUM" in scenarios[2]["containment_levels"]
        ),
        "integrity_failure_escalates": (
            "HIGH" in scenarios[3]["containment_levels"]
        ),
        "sensor_disagreement_escalates": (
            "MEDIUM" in scenarios[4]["containment_levels"]
        ),
        "combined_attack_escalates": (
            "HIGH" in scenarios[5]["containment_levels"]
        ),
        "recovery_integrity_valid": (
            scenarios[6]["integrity_valid"]
        ),
        "forensic_tampering_detected": (
            tamper_result["tamper_detected"]
        ),
        "no_direct_vehicle_actuation": all(
            not scenario["direct_vehicle_actuation"]
            for scenario in scenarios
        ),
    }

    report = {
        "phase": "4.22",
        "title": "Edge-IoT Attack and Telemetry Scenario Evaluation",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "research_simulation": True,
        "retraining_performed": False,
        "scenario_count": len(scenarios) + 1,
        "scenarios": scenarios,
        "forensic_tampering_test": tamper_result,
        "gates": gates,
        "architecture_under_test": [
            "Edge-IoT / ECU / CAN observation",
            "Local deterministic containment",
            "Tamper-evident forensic buffer",
            "Connectivity recovery",
            "Integrity verification",
            "Secure synchronization",
            "Cortex XDR abstraction",
            "SOC investigation",
        ],
        "safety_boundary": (
            "This evaluation does not transmit CAN commands, "
            "control a vehicle, or perform safety-critical actuation. "
            "Containment is simulated deterministic policy logic."
        ),
        "limitations": [
            "All attack and telemetry scenarios are synthetic.",
            "No real vehicle or ECU is connected.",
            "Cortex XDR remains an abstraction.",
            "Results demonstrate software-policy behavior, "
            "not production vehicle safety validation.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("PHASE 4.22 EDGE-IOT SCENARIO EVALUATION")
    print("=" * 72)
    print(f"Status: {report['status']}")
    print(f"Scenarios evaluated: {report['scenario_count']}")
    print()

    for scenario in scenarios:
        print(
            f"{scenario['scenario']}: "
            f"integrity={scenario['integrity_valid']} "
            f"levels={scenario['containment_levels']}"
        )

    print()
    print(
        "FORENSIC_BUFFER_TAMPERING: "
        f"detected={tamper_result['tamper_detected']}"
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
