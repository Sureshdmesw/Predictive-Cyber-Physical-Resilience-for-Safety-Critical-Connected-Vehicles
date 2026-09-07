from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = ROOT / "experiments" / "integration" / "phase4_28"
OUT_FILE = OUT_DIR / "crss_2024_phase4_28a_predictive_soc_event_pipeline.json"

DECISION_ENGINE = (
    ROOT / "experiments" / "modeling" / "v2" / "decision_engine"
    / "crss_2024_v2_phase4_20_predictive_resilience_decision_engine.json"
)

EDGE_REPORT = (
    ROOT / "experiments" / "edge_iot"
    / "crss_2024_phase4_21_edge_iot_resilience.json"
)

FORENSIC_SCHEMA = (
    ROOT / "data" / "schemas" / "edge_iot"
    / "edge_iot_forensic_event_schema.json"
)

ENSEMBLE_REPORT = (
    ROOT / "experiments" / "modeling" / "v2" / "uncertainty"
    / "phase4_27"
    / "crss_2024_v2_phase4_27a_deep_ensemble_final.json"
)


RISK_THRESHOLDS = {
    "NORMAL": 0.05,
    "LOW": 0.20,
    "MEDIUM": 0.50,
    "HIGH": 0.80,
}

MODIFIER_MINIMUMS = {
    "connectivity_degraded": "MEDIUM",
    "sensor_disagreement": "MEDIUM",
    "integrity_failure": "HIGH",
}

SEVERITY_ORDER = {
    "NORMAL": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def classify_risk(score: float) -> str:
    if score < RISK_THRESHOLDS["NORMAL"]:
        return "NORMAL"
    if score < RISK_THRESHOLDS["LOW"]:
        return "LOW"
    if score < RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    if score < RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    return "CRITICAL"


def raise_minimum(current: str, minimum: str) -> str:
    if SEVERITY_ORDER[minimum] > SEVERITY_ORDER[current]:
        return minimum
    return current


def decision_for_scenario(
    risk_score: float,
    connectivity_degraded: bool,
    sensor_disagreement: bool,
    integrity_failure: bool,
):
    base = classify_risk(risk_score)
    final = base

    if connectivity_degraded:
        final = raise_minimum(
            final, MODIFIER_MINIMUMS["connectivity_degraded"]
        )

    if sensor_disagreement:
        final = raise_minimum(
            final, MODIFIER_MINIMUMS["sensor_disagreement"]
        )

    if integrity_failure:
        final = raise_minimum(
            final, MODIFIER_MINIMUMS["integrity_failure"]
        )

    actions = []

    if final in {"MEDIUM", "HIGH", "CRITICAL"}:
        actions.append("increase_monitoring")

    if final in {"HIGH", "CRITICAL"}:
        actions.append("preserve_forensic_context")

    if final == "CRITICAL":
        actions.append("escalate_to_soc")

    if connectivity_degraded:
        actions.append("retain_local_resilience")

    if integrity_failure:
        actions.append("integrity_verification_required")

    return {
        "base_risk_level": base,
        "final_risk_level": final,
        "actions": actions,
        "direct_vehicle_actuation": False,
    }


def canonical_json(obj) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_json(obj) -> str:
    return hashlib.sha256(
        canonical_json(obj).encode("utf-8")
    ).hexdigest()


def make_forensic_event(
    scenario,
    decision,
    risk_score,
    timestamp,
    previous_hash="ROLLING_GENESIS",
):
    event_core = {
        "schema_version": "1.1",
        "event_type": "PREDICTIVE_CYBER_PHYSICAL_RISK",
        "event_id": f"PCR-{scenario['scenario_id']}",
        "timestamp_utc": timestamp,
        "scenario_id": scenario["scenario_id"],
        "risk_score": risk_score,
        "base_risk_level": decision["base_risk_level"],
        "final_risk_level": decision["final_risk_level"],
        "connectivity_degraded": scenario["connectivity_degraded"],
        "sensor_disagreement": scenario["sensor_disagreement"],
        "integrity_failure": scenario["integrity_failure"],
        "decision_actions": decision["actions"],
        "direct_vehicle_actuation": False,
        "previous_hash": previous_hash,
    }

    event_hash = sha256_json(event_core)

    return {
        **event_core,
        "event_hash": event_hash,
    }


def make_soc_event(forensic_event):
    return {
        "event_type": "CORTEX_XDR_COMPATIBLE_RESEARCH_EVENT",
        "integration_mode": "ABSTRACTION_ONLY",
        "live_cortex_xdr_api_used": False,
        "event_id": forensic_event["event_id"],
        "timestamp_utc": forensic_event["timestamp_utc"],
        "severity": forensic_event["final_risk_level"],
        "risk_score": forensic_event["risk_score"],
        "source": "vehicle_edge_predictive_resilience",
        "scenario_id": forensic_event["scenario_id"],
        "connectivity_degraded": forensic_event["connectivity_degraded"],
        "sensor_disagreement": forensic_event["sensor_disagreement"],
        "integrity_failure": forensic_event["integrity_failure"],
        "forensic_hash": forensic_event["event_hash"],
        "direct_vehicle_actuation": False,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    required_files = {
        "decision_engine": DECISION_ENGINE,
        "edge_report": EDGE_REPORT,
        "forensic_schema": FORENSIC_SCHEMA,
        "ensemble_report": ENSEMBLE_REPORT,
    }

    source_checks = {}

    for name, path in required_files.items():
        source_checks[name] = {
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
        }

    scenarios = [
        {
            "scenario_id": "normal_connected",
            "risk_score": 0.02,
            "connectivity_degraded": False,
            "sensor_disagreement": False,
            "integrity_failure": False,
        },
        {
            "scenario_id": "predictive_medium",
            "risk_score": 0.31,
            "connectivity_degraded": False,
            "sensor_disagreement": False,
            "integrity_failure": False,
        },
        {
            "scenario_id": "connectivity_degraded",
            "risk_score": 0.10,
            "connectivity_degraded": True,
            "sensor_disagreement": False,
            "integrity_failure": False,
        },
        {
            "scenario_id": "sensor_disagreement",
            "risk_score": 0.12,
            "connectivity_degraded": False,
            "sensor_disagreement": True,
            "integrity_failure": False,
        },
        {
            "scenario_id": "integrity_failure",
            "risk_score": 0.18,
            "connectivity_degraded": False,
            "sensor_disagreement": False,
            "integrity_failure": True,
        },
        {
            "scenario_id": "combined_high_risk",
            "risk_score": 0.72,
            "connectivity_degraded": True,
            "sensor_disagreement": True,
            "integrity_failure": True,
        },
    ]

    timestamp = datetime.now(timezone.utc).isoformat()

    records = []
    previous_hash = "ROLLING_GENESIS"

    for scenario in scenarios:
        decision = decision_for_scenario(
            scenario["risk_score"],
            scenario["connectivity_degraded"],
            scenario["sensor_disagreement"],
            scenario["integrity_failure"],
        )

        forensic = make_forensic_event(
            scenario,
            decision,
            scenario["risk_score"],
            timestamp,
            previous_hash,
        )

        soc = make_soc_event(forensic)

        records.append(
            {
                "scenario": scenario,
                "decision": decision,
                "forensic_event": forensic,
                "soc_event": soc,
            }
        )

        previous_hash = forensic["event_hash"]

    gates = {}

    gates["required_source_files_present"] = all(
        item["exists"] for item in source_checks.values()
    )

    gates["six_scenarios_generated"] = len(records) == 6

    gates["normal_risk_classification"] = (
        records[0]["decision"]["final_risk_level"] == "NORMAL"
    )

    gates["predictive_medium_classification"] = (
        records[1]["decision"]["final_risk_level"] == "MEDIUM"
    )

    gates["connectivity_minimum_medium"] = (
        records[2]["decision"]["final_risk_level"] == "MEDIUM"
    )

    gates["sensor_disagreement_minimum_medium"] = (
        records[3]["decision"]["final_risk_level"] == "MEDIUM"
    )

    gates["integrity_failure_minimum_high"] = (
        records[4]["decision"]["final_risk_level"] == "HIGH"
    )

    gates["combined_high_risk_escalates"] = (
        records[5]["decision"]["final_risk_level"] == "HIGH"
    )

    gates["direct_vehicle_actuation_disabled"] = all(
        not r["decision"]["direct_vehicle_actuation"]
        for r in records
    )

    gates["forensic_hashes_present"] = all(
        len(r["forensic_event"]["event_hash"]) == 64
        for r in records
    )

    gates["forensic_hash_chain_valid"] = all(
        records[i]["forensic_event"]["previous_hash"]
        == (
            "ROLLING_GENESIS"
            if i == 0
            else records[i - 1]["forensic_event"]["event_hash"]
        )
        for i in range(len(records))
    )

    gates["soc_events_present"] = all(
        r["soc_event"]["event_type"]
        == "CORTEX_XDR_COMPATIBLE_RESEARCH_EVENT"
        for r in records
    )

    gates["live_cortex_xdr_not_claimed"] = all(
        not r["soc_event"]["live_cortex_xdr_api_used"]
        for r in records
    )

    gates["integrity_failure_requires_verification"] = (
        "integrity_verification_required"
        in records[4]["decision"]["actions"]
    )

    gates["connectivity_retains_local_resilience"] = (
        "retain_local_resilience"
        in records[2]["decision"]["actions"]
    )

    gates["all_events_have_required_identity"] = all(
        r["forensic_event"]["event_id"]
        and r["forensic_event"]["scenario_id"]
        and r["forensic_event"]["timestamp_utc"]
        for r in records
    )

    report = {
        "phase": "4.28A",
        "title": "Predictive Risk to SOC Event Integration Pipeline",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "training_performed": False,
        "model_retraining_performed": False,
        "purpose": [
            "connect predictive risk output to deterministic decision policy",
            "generate tamper-evident forensic events",
            "represent connectivity-aware local resilience",
            "prepare Cortex XDR-compatible SOC events",
        ],
        "architecture": [
            "Predictive Model",
            "Risk + Uncertainty",
            "Decision Engine",
            "Forensic Event",
            "Tamper-Evident Buffer",
            "SOC Event Abstraction",
        ],
        "policy": {
            "risk_thresholds": RISK_THRESHOLDS,
            "modifier_minimums": MODIFIER_MINIMUMS,
            "direct_vehicle_actuation": False,
        },
        "source_foundation": source_checks,
        "scenario_results": records,
        "gates": gates,
        "gate_summary": {
            "passed": sum(gates.values()),
            "total": len(gates),
        },
        "scientific_boundary": {
            "synthetic_telemetry_labels": True,
            "real_world_cyberattack_performance_claim": False,
            "live_cortex_xdr_connection_claim": False,
            "direct_vehicle_actuation": False,
        },
    }

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("PHASE 4.28A — PREDICTIVE → SOC EVENT PIPELINE")
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
    print("Scenario decisions:")
    for record in records:
        print(
            f"  {record['scenario']['scenario_id']}: "
            f"risk={record['scenario']['risk_score']:.2f} → "
            f"{record['decision']['final_risk_level']}"
        )


if __name__ == "__main__":
    main()
