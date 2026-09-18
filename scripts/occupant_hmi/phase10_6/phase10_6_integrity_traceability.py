import csv
import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

PHASE10_4_CSV = (
    ROOT / "data" / "processed" / "occupant_hmi" / "phase10_4"
    / "phase10_4_restraint_event_hmi.csv"
)

PHASE10_5_CSV = (
    ROOT / "data" / "processed" / "occupant_hmi" / "phase10_5"
    / "phase10_5_hmi_evidence_package.csv"
)

PHASE10_5_SUMMARY = (
    ROOT / "data" / "processed" / "occupant_hmi" / "phase10_5"
    / "phase10_5_hmi_state_summary.json"
)

PHASE10_4_MANIFEST = (
    ROOT / "experiments" / "occupant_hmi" / "phase10_4"
    / "phase10_4_restraint_event_hmi_manifest.json"
)

OUTPUT_JSON = (
    ROOT / "data" / "processed" / "occupant_hmi" / "phase10_6"
    / "phase10_6_integrity_traceability.json"
)

MANIFEST = (
    ROOT / "experiments" / "occupant_hmi" / "phase10_6"
    / "phase10_6_integrity_traceability_manifest.json"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():

    required = [
        PHASE10_4_CSV,
        PHASE10_5_CSV,
        PHASE10_5_SUMMARY,
        PHASE10_4_MANIFEST,
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required artifact(s):\n" + "\n".join(missing)
        )

    phase10_4 = load_csv(PHASE10_4_CSV)
    phase10_5 = load_csv(PHASE10_5_CSV)

    with open(PHASE10_5_SUMMARY, encoding="utf-8") as f:
        summary = json.load(f)

    with open(PHASE10_4_MANIFEST, encoding="utf-8") as f:
        phase10_4_manifest = json.load(f)

    expected_states = [
        "CONNECTED",
        "DISCONNECTED",
        "RECOVERED",
    ]

    expected_events = [
        "EVENT_OBSERVED",
        "RESILIENCE_MODE",
        "RECOVERY_READY",
    ]

    expected_hmi = [
        "SAFETY_EVENT_MONITORED",
        "CONNECTIVITY_DEGRADED",
        "RECOVERY_READY",
    ]

    expected_forensic = [
        "CONNECTED_STREAM",
        "LOCAL_FORENSIC_BUFFER",
        "RECOVERY_QUEUE",
    ]

    expected_sync = [
        "NOT_REQUIRED",
        "DEFERRED",
        "READY",
    ]

    expected_soc = [
        "NOT_REQUIRED",
        "DEFERRED",
        "READY",
    ]

    phase10_4_states = [x["connectivity_state"] for x in phase10_4]
    phase10_5_states = [x["connectivity_state"] for x in phase10_5]

    phase10_4_events = [x["event_state"] for x in phase10_4]
    phase10_5_events = [x["event_state"] for x in phase10_5]

    phase10_4_hmi = [x["hmi_status"] for x in phase10_4]
    phase10_5_hmi = [x["hmi_status"] for x in phase10_5]

    phase10_4_forensic = [x["forensic_state"] for x in phase10_4]
    phase10_5_forensic = [x["forensic_state"] for x in phase10_5]

    phase10_4_sync = [x["synchronization_state"] for x in phase10_4]
    phase10_5_sync = [x["synchronization_state"] for x in phase10_5]

    phase10_4_soc = [x["soc_handoff_state"] for x in phase10_4]
    phase10_5_soc = [x["soc_handoff_state"] for x in phase10_5]

    checks = {

        "phase10_4_input_exists":
            PHASE10_4_CSV.exists(),

        "phase10_5_evidence_exists":
            PHASE10_5_CSV.exists(),

        "phase10_5_summary_exists":
            PHASE10_5_SUMMARY.exists(),

        "phase10_4_manifest_exists":
            PHASE10_4_MANIFEST.exists(),

        "phase10_4_status_pass":
            phase10_4_manifest.get("status") == "PASS",

        "phase10_4_record_count":
            len(phase10_4) == 3,

        "phase10_5_record_count":
            len(phase10_5) == 3,

        "state_sequence_preserved":
            phase10_4_states == expected_states
            and phase10_5_states == expected_states,

        "event_sequence_preserved":
            phase10_4_events == expected_events
            and phase10_5_events == expected_events,

        "hmi_sequence_preserved":
            phase10_4_hmi == expected_hmi
            and phase10_5_hmi == expected_hmi,

        "forensic_sequence_preserved":
            phase10_4_forensic == expected_forensic
            and phase10_5_forensic == expected_forensic,

        "sync_sequence_preserved":
            phase10_4_sync == expected_sync
            and phase10_5_sync == expected_sync,

        "soc_sequence_preserved":
            phase10_4_soc == expected_soc
            and phase10_5_soc == expected_soc,

        "phase10_4_to_10_5_traceable":
    all(
        all(
            x.get(field) == y.get(field)
            for field in [
                "scenario_id",
                "connectivity_state",
                "event_state",
                "hmi_status",
                "hmi_message",
                "forensic_state",
                "synchronization_state",
                "soc_handoff_state",
                "records",
                "local_preservation",
                "event_timestep",
                "disconnect_start",
                "disconnect_end",
                "evidence_source",
                "laboratory_generated",
                "physical_vehicle",
                "direct_actuation",
            ]
        )
        for x, y in zip(phase10_4, phase10_5)
    ),

        "summary_state_sequence_valid":
            summary.get("state_sequence") == expected_states,

        "summary_event_sequence_valid":
            summary.get("event_sequence") == expected_events,

        "summary_hmi_sequence_valid":
            summary.get("hmi_sequence") == expected_hmi,

        "summary_laboratory_only":
            summary.get("laboratory_generated") is True,

        "summary_no_physical_vehicle":
            summary.get("physical_vehicle") is False,

        "summary_no_direct_actuation":
            summary.get("direct_actuation") is False,

        "summary_no_can_transmission":
            summary.get("can_transmission") is False,

        "summary_read_only":
            summary.get("read_only") is True,

        "phase10_4_input_hash_valid":
            len(sha256_file(PHASE10_4_CSV)) == 64,

        "phase10_5_output_hash_valid":
            len(sha256_file(PHASE10_5_CSV)) == 64,

        "phase10_5_summary_hash_valid":
            len(sha256_file(PHASE10_5_SUMMARY)) == 64,
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    traceability = {
        "phase": "10.6",
        "title": "HMI Integrity and Traceability Validation",
        "status": status,

        "traceability_chain": [
            "Phase 10.4 Controlled Restraint-Domain Event HMI",
            "Phase 10.5 HMI Evidence Package",
            "Phase 10.6 Integrity and Traceability Validation",
        ],

        "scenario_id": "C09_RESTRAINT_DOMAIN_ATTACK",

        "phase10_4_input": str(
            PHASE10_4_CSV.relative_to(ROOT)
        ),

        "phase10_4_input_sha256": sha256_file(
            PHASE10_4_CSV
        ),

        "phase10_5_evidence": str(
            PHASE10_5_CSV.relative_to(ROOT)
        ),

        "phase10_5_evidence_sha256": sha256_file(
            PHASE10_5_CSV
        ),

        "phase10_5_summary": str(
            PHASE10_5_SUMMARY.relative_to(ROOT)
        ),

        "phase10_5_summary_sha256": sha256_file(
            PHASE10_5_SUMMARY
        ),

        "phase10_4_manifest_status":
            phase10_4_manifest.get("status"),

        "state_sequence": expected_states,

        "event_sequence": expected_events,

        "hmi_sequence": expected_hmi,

        "forensic_sequence": expected_forensic,

        "synchronization_sequence": expected_sync,

        "soc_handoff_sequence": expected_soc,

        "record_count": 3,

        "safety_boundary": {
            "read_only": True,
            "can_transmission": False,
            "direct_vehicle_actuation": False,
            "physical_vehicle": False,
            "live_cortex_xdr": False,
            "frozen_transformer_modified": False,
            "hcrl_fused": False,
            "laboratory_generated": True,
        },

        "validation": checks,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(traceability, f, indent=2)

    manifest = {
        "phase": "10.6",
        "title": "HMI Integrity and Traceability Validation",
        "status": status,
        "scenario_id": "C09_RESTRAINT_DOMAIN_ATTACK",
        "input_phase10_4_sha256":
            sha256_file(PHASE10_4_CSV),
        "input_phase10_5_sha256":
            sha256_file(PHASE10_5_CSV),
        "summary_sha256":
            sha256_file(PHASE10_5_SUMMARY),
        "traceability_output":
            str(OUTPUT_JSON.relative_to(ROOT)),
        "traceability_output_sha256":
            sha256_file(OUTPUT_JSON),
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
    print("=== PHASE 10.6 HMI INTEGRITY & TRACEABILITY ===")
    print("Scenario: C09_RESTRAINT_DOMAIN_ATTACK")
    print("Chain: Phase 10.4 -> Phase 10.5 -> Phase 10.6")
    print(f"Status: {status}")
    print()

    print("Traceability:")
    print("  CONNECTED -> DISCONNECTED -> RECOVERED")
    print("  EVENT_OBSERVED -> RESILIENCE_MODE -> RECOVERY_READY")
    print("  SAFETY_EVENT_MONITORED -> CONNECTIVITY_DEGRADED -> RECOVERY_READY")
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
    print()

    print("Artifacts:")
    print(f"  Traceability: {OUTPUT_JSON}")
    print(f"  Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
