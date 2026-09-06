from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE_420_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "crss_2024_phase4_20_predictive_resilience_decision_engine.py"
)

PHASE_421_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "crss_2024_phase4_21_edge_iot_resilience.py"
)

PHASE_420_REPORT = (
    PROJECT_ROOT
    / "experiments"
    / "modeling"
    / "v2"
    / "decision_engine"
    / "crss_2024_v2_phase4_20_predictive_resilience_decision_engine.json"
)

PHASE_421_REPORT = (
    PROJECT_ROOT
    / "experiments"
    / "edge_iot"
    / "crss_2024_phase4_21_edge_iot_resilience.json"
)

PHASE_422_REPORT = (
    PROJECT_ROOT
    / "experiments"
    / "edge_iot"
    / "crss_2024_phase4_22_scenario_evaluation.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "edge_iot"
    / "crss_2024_phase4_23_end_to_end_resilience.json"
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


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def risk_level(probability: float) -> str:
    if probability < 0.05:
        return "NORMAL"
    if probability < 0.20:
        return "LOW"
    if probability < 0.50:
        return "MEDIUM"
    if probability < 0.80:
        return "HIGH"

    return "CRITICAL"


def apply_context_modifiers(
    base_level: str,
    connectivity_degraded: bool,
    integrity_failure: bool,
    sensor_disagreement: bool,
) -> str:

    levels = {
        "NORMAL": 0,
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }

    reverse = {value: key for key, value in levels.items()}

    final_value = levels[base_level]

    if connectivity_degraded:
        final_value = max(final_value, levels["MEDIUM"])

    if sensor_disagreement:
        final_value = max(final_value, levels["MEDIUM"])

    if integrity_failure:
        final_value = max(final_value, levels["HIGH"])

    return reverse[final_value]


def soc_priority(level: str) -> str:
    return {
        "NORMAL": "ROUTINE",
        "LOW": "LOW",
        "MEDIUM": "MEDIUM",
        "HIGH": "HIGH",
        "CRITICAL": "CRITICAL",
    }[level]


def evaluate_prediction(
    probability: float,
    connectivity_degraded: bool = False,
    integrity_failure: bool = False,
    sensor_disagreement: bool = False,
) -> dict[str, Any]:

    base_level = risk_level(probability)

    final_level = apply_context_modifiers(
        base_level=base_level,
        connectivity_degraded=connectivity_degraded,
        integrity_failure=integrity_failure,
        sensor_disagreement=sensor_disagreement,
    )

    occupant_warning = final_level in {
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }

    return {
        "risk_probability": probability,
        "base_risk_level": base_level,
        "final_risk_level": final_level,
        "soc_priority": soc_priority(final_level),
        "vehicle_resilience_recommendation": (
            "ROUTINE_MONITORING"
            if final_level == "NORMAL"
            else "EARLY_WARNING"
            if final_level == "LOW"
            else "ENTER_LOCAL_RESILIENCE_MODE"
            if final_level == "MEDIUM"
            else "PRIORITIZE_CONTAINMENT_AND_SOC_INVESTIGATION"
            if final_level == "HIGH"
            else "CRITICAL_SOC_ESCALATION"
        ),
        "occupant_warning": occupant_warning,
        "direct_vehicle_actuation": False,
        "context": {
            "connectivity_degraded": connectivity_degraded,
            "integrity_failure": integrity_failure,
            "sensor_disagreement": sensor_disagreement,
        },
    }


def evaluate_case(
    edge_module,
    name: str,
    probability: float,
    connectivity_degraded: bool = False,
    integrity_failure: bool = False,
    sensor_disagreement: bool = False,
) -> dict[str, Any]:

    decision = evaluate_prediction(
        probability=probability,
        connectivity_degraded=connectivity_degraded,
        integrity_failure=integrity_failure,
        sensor_disagreement=sensor_disagreement,
    )

    event = {
        "event_id": f"E2E-{name}",
        "timestamp": 0.0,
        "vehicle_id": "SIM-E2E-001",
        "ecu_id": "ECU-GW",
        "can_id": "0x310",
        "event_type": f"E2E_{name}",
        "severity": decision["final_risk_level"],
        "connectivity_state": (
            "DEGRADED"
            if connectivity_degraded
            else "CONNECTED"
        ),
        "integrity_status": (
            "FAIL"
            if integrity_failure
            else "PASS"
        ),
        "sensor_consistency_status": (
            "DISAGREEMENT"
            if sensor_disagreement
            else "CONSISTENT"
        ),
    }

    containment = edge_module.deterministic_containment_policy(
        event
    )

    buffer = edge_module.TamperEvidentForensicBuffer(
        capacity=64
    )

    record = buffer.append(event)

    forensic_integrity = buffer.verify_integrity()

    cortex_event = edge_module.build_cortex_xdr_event(
        event,
        containment,
        record.record_hash,
    )

    direct_actuation = (
        decision["direct_vehicle_actuation"]
        or containment["direct_vehicle_actuation"]
        or cortex_event["direct_vehicle_actuation"]
    )

    return {
        "case": name,
        "decision": decision,
        "edge_containment": containment,
        "forensic_record_hash": record.record_hash,
        "forensic_integrity_valid": forensic_integrity,
        "cortex_xdr_event": cortex_event,
        "direct_vehicle_actuation": direct_actuation,
    }


def main():

    decision_module = load_module(
        PHASE_420_SCRIPT,
        "phase420",
    )

    edge_module = load_module(
        PHASE_421_SCRIPT,
        "phase421",
    )

    phase420_report = read_json(PHASE_420_REPORT)
    phase421_report = read_json(PHASE_421_REPORT)
    phase422_report = read_json(PHASE_422_REPORT)

    upstream_pass = (
        phase420_report.get("status") == "PASS"
        and phase421_report.get("status") == "PASS"
        and phase422_report.get("status") == "PASS"
    )

    cases = [
        evaluate_case(
            edge_module,
            "NORMAL",
            0.01,
        ),
        evaluate_case(
            edge_module,
            "EARLY_WARNING",
            0.12,
        ),
        evaluate_case(
            edge_module,
            "CONNECTIVITY_DEGRADATION",
            0.14,
            connectivity_degraded=True,
        ),
        evaluate_case(
            edge_module,
            "SENSOR_INCONSISTENCY",
            0.32,
            sensor_disagreement=True,
        ),
        evaluate_case(
            edge_module,
            "INTEGRITY_FAILURE",
            0.61,
            integrity_failure=True,
        ),
        evaluate_case(
            edge_module,
            "CRITICAL_COMBINED",
            0.91,
            connectivity_degraded=True,
            integrity_failure=True,
            sensor_disagreement=True,
        ),
    ]

    gates = {
        "upstream_phases_4_20_to_4_22_pass": upstream_pass,

        "six_end_to_end_cases_executed": (
            len(cases) == 6
        ),

        "risk_policy_sequence_valid": (
            cases[0]["decision"]["final_risk_level"] == "NORMAL"
            and cases[1]["decision"]["final_risk_level"] == "LOW"
            and cases[2]["decision"]["final_risk_level"] == "MEDIUM"
            and cases[3]["decision"]["final_risk_level"] == "MEDIUM"
            and cases[4]["decision"]["final_risk_level"] == "HIGH"
            and cases[5]["decision"]["final_risk_level"] == "CRITICAL"
        ),

        "connectivity_reaches_edge_layer": (
            cases[2]["edge_containment"]["containment_level"]
            == "MEDIUM"
        ),

        "sensor_disagreement_reaches_edge_layer": (
            cases[3]["edge_containment"]["containment_level"]
            == "MEDIUM"
        ),

        "integrity_failure_reaches_high_containment": (
            cases[4]["edge_containment"]["containment_level"]
            == "HIGH"
        ),

        "combined_condition_reaches_high_containment": (
            cases[5]["edge_containment"]["containment_level"]
            == "HIGH"
        ),

        "forensic_record_created_for_every_case": (
            len({
                case["forensic_record_hash"]
                for case in cases
            }) == 6
        ),

        "forensic_integrity_valid_for_every_case": (
            all(
                case["forensic_integrity_valid"]
                for case in cases
            )
        ),

        "cortex_xdr_abstraction_complete": (
            all(
                case["cortex_xdr_event"]["target_platform"]
                == "Cortex XDR"
                and case["cortex_xdr_event"]["forensic_record_hash"]
                == case["forensic_record_hash"]
                for case in cases
            )
        ),

        "no_direct_vehicle_actuation": (
            all(
                not case["direct_vehicle_actuation"]
                for case in cases
            )
        ),

        "no_retraining": True,
    }

    report = {
        "phase": "4.23",
        "title": (
            "End-to-End Predictive Cyber-Physical "
            "Resilience Pipeline"
        ),
        "status": (
            "PASS"
            if all(gates.values())
            else "FAIL"
        ),
        "research_simulation": True,
        "retraining_performed": False,
        "upstream_phases": {
            "phase_4_20": phase420_report.get("status"),
            "phase_4_21": phase421_report.get("status"),
            "phase_4_22": phase422_report.get("status"),
        },
        "pipeline": [
            "Predictive ML risk probability",
            "Operational risk policy",
            "Predictive resilience decision engine",
            "Edge-IoT deterministic containment",
            "Tamper-evident forensic recording",
            "Integrity verification",
            "Cortex XDR abstraction",
            "SOC investigation",
        ],
        "cases": cases,
        "gates": gates,
        "architecture_hash": sha256_json([
            "predictive_risk",
            "risk_policy",
            "decision_engine",
            "edge_containment",
            "forensic_buffer",
            "integrity_verification",
            "cortex_xdr_abstraction",
            "soc_investigation",
        ]),
        "safety_boundary": (
            "Predictive ML provides forecasting and decision support. "
            "Edge containment is deterministic and simulated. "
            "No component transmits CAN commands or performs direct "
            "safety-critical vehicle actuation."
        ),
        "limitations": [
            "The pipeline is a research simulation.",
            "Cyber telemetry and attack conditions are synthetic.",
            "CRSS provides physical safety ground truth, not cyberattack labels.",
            "Cortex XDR is represented as an abstraction rather than a live tenant.",
            "Containment recommendations are not validated vehicle-control logic.",
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
    print("PHASE 4.23 END-TO-END PREDICTIVE RESILIENCE")
    print("=" * 72)
    print(f"Status: {report['status']}")
    print(f"Cases: {len(cases)}")
    print(f"Retraining: {report['retraining_performed']}")
    print()

    for case in cases:
        print(
            f"{case['case']}: "
            f"risk={case['decision']['risk_probability']:.2f} "
            f"decision={case['decision']['final_risk_level']} "
            f"edge={case['edge_containment']['containment_level']} "
            f"forensic={case['forensic_integrity_valid']}"
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
