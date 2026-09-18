import csv
import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    ROOT / "data" / "processed" / "occupant_hmi" / "phase10_4"
    / "phase10_4_restraint_event_hmi.csv"
)

OUTPUT_CSV = (
    ROOT / "data" / "processed" / "occupant_hmi" / "phase10_5"
    / "phase10_5_hmi_evidence_package.csv"
)

SUMMARY_JSON = (
    ROOT / "data" / "processed" / "occupant_hmi" / "phase10_5"
    / "phase10_5_hmi_state_summary.json"
)

MANIFEST = (
    ROOT / "experiments" / "occupant_hmi" / "phase10_5"
    / "phase10_5_hmi_evidence_manifest.json"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 3:
        raise RuntimeError(f"Expected 3 Phase 10.4 records, found {len(rows)}")

    expected = ["CONNECTED", "DISCONNECTED", "RECOVERED"]

    actual = [row["connectivity_state"] for row in rows]

    if actual != expected:
        raise RuntimeError(
            f"Unexpected state sequence: {actual}"
        )

    evidence = []

    for index, row in enumerate(rows, start=1):

        evidence.append({
            "sequence": index,
            "scenario_id": row["scenario_id"],
            "connectivity_state": row["connectivity_state"],
            "event_state": row["event_state"],
            "hmi_status": row["hmi_status"],
            "hmi_message": row["hmi_message"],
            "forensic_state": row["forensic_state"],
            "synchronization_state": row["synchronization_state"],
            "soc_handoff_state": row["soc_handoff_state"],
            "records": row["records"],
            "local_preservation": row["local_preservation"],
            "event_timestep": row["event_timestep"],
            "disconnect_start": row["disconnect_start"],
            "disconnect_end": row["disconnect_end"],
            "evidence_source": row["evidence_source"],
            "laboratory_generated": row["laboratory_generated"],
            "physical_vehicle": row["physical_vehicle"],
            "direct_actuation": row["direct_actuation"],
        })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    fields = list(evidence[0].keys())

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(evidence)

    state_summary = {
        "scenario_id": "C09_RESTRAINT_DOMAIN_ATTACK",
        "state_sequence": expected,
        "event_sequence": [
            "EVENT_OBSERVED",
            "RESILIENCE_MODE",
            "RECOVERY_READY",
        ],
        "hmi_sequence": [
            "SAFETY_EVENT_MONITORED",
            "CONNECTIVITY_DEGRADED",
            "RECOVERY_READY",
        ],
        "forensic_sequence": [
            "CONNECTED_STREAM",
            "LOCAL_FORENSIC_BUFFER",
            "RECOVERY_QUEUE",
        ],
        "synchronization_sequence": [
            "NOT_REQUIRED",
            "DEFERRED",
            "READY",
        ],
        "soc_handoff_sequence": [
            "NOT_REQUIRED",
            "DEFERRED",
            "READY",
        ],
        "evidence_records": len(evidence),
        "laboratory_generated": True,
        "physical_vehicle": False,
        "direct_actuation": False,
        "can_transmission": False,
        "read_only": True,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(state_summary, f, indent=2)

    checks = {
        "input_exists": INPUT_CSV.exists(),
        "three_records": len(evidence) == 3,
        "connected_first": actual[0] == "CONNECTED",
        "disconnected_second": actual[1] == "DISCONNECTED",
        "recovered_third": actual[2] == "RECOVERED",
        "event_sequence_valid":
            [x["event_state"] for x in evidence] ==
            ["EVENT_OBSERVED", "RESILIENCE_MODE", "RECOVERY_READY"],
        "hmi_sequence_valid":
            [x["hmi_status"] for x in evidence] ==
            [
                "SAFETY_EVENT_MONITORED",
                "CONNECTIVITY_DEGRADED",
                "RECOVERY_READY",
            ],
        "forensic_sequence_valid":
            [x["forensic_state"] for x in evidence] ==
            [
                "CONNECTED_STREAM",
                "LOCAL_FORENSIC_BUFFER",
                "RECOVERY_QUEUE",
            ],
        "sync_deferred_during_disconnect":
            evidence[1]["synchronization_state"] == "DEFERRED",
        "sync_ready_after_recovery":
            evidence[2]["synchronization_state"] == "READY",
        "soc_deferred_during_disconnect":
            evidence[1]["soc_handoff_state"] == "DEFERRED",
        "soc_ready_after_recovery":
            evidence[2]["soc_handoff_state"] == "READY",
        "local_preservation_during_disconnect":
            evidence[1]["local_preservation"] == "True",
        "laboratory_only":
            all(x["laboratory_generated"] == "True" for x in evidence),
        "no_physical_vehicle":
            all(x["physical_vehicle"] == "False" for x in evidence),
        "no_direct_actuation":
            all(x["direct_actuation"] == "False" for x in evidence),
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    manifest = {
        "phase": "10.5",
        "title": "HMI Evidence Package and State Visualization",
        "status": status,
        "input": str(INPUT_CSV.relative_to(ROOT)),
        "input_sha256": sha256_file(INPUT_CSV),
        "output_csv": str(OUTPUT_CSV.relative_to(ROOT)),
        "output_csv_sha256": sha256_file(OUTPUT_CSV),
        "summary_json": str(SUMMARY_JSON.relative_to(ROOT)),
        "summary_json_sha256": sha256_file(SUMMARY_JSON),
        "scenario_id": "C09_RESTRAINT_DOMAIN_ATTACK",
        "state_sequence": expected,
        "records": len(evidence),
        "read_only": True,
        "can_transmission": False,
        "direct_vehicle_actuation": False,
        "physical_vehicle": False,
        "live_cortex_xdr": False,
        "frozen_transformer_modified": False,
        "hcrl_fused": False,
        "laboratory_generated": True,
        "validation": checks,
    }

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=== PHASE 10.5 HMI EVIDENCE PACKAGE ===")
    print(f"Scenario: {manifest['scenario_id']}")
    print(f"Records: {len(evidence)}")
    print("State sequence: CONNECTED -> DISCONNECTED -> RECOVERED")
    print("Event sequence: EVENT_OBSERVED -> RESILIENCE_MODE -> RECOVERY_READY")
    print("HMI sequence: SAFETY_EVENT_MONITORED -> CONNECTIVITY_DEGRADED -> RECOVERY_READY")
    print(f"Status: {status}")
    print()
    print("Artifacts:")
    print(f"  CSV: {OUTPUT_CSV}")
    print(f"  Summary: {SUMMARY_JSON}")
    print(f"  Manifest: {MANIFEST}")
    print()
    print("Safety boundary:")
    print("  Read-only: True")
    print("  CAN transmission: False")
    print("  Direct vehicle actuation: False")
    print("  Physical vehicle: False")
    print("  Live Cortex XDR: False")
    print("  Frozen Transformer modified: False")
    print("  HCRL fused: False")
    print("  Laboratory generated: True")


if __name__ == "__main__":
    main()
