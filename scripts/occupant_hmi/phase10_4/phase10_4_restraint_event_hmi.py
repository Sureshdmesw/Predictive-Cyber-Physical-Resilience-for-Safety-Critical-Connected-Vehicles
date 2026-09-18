import csv
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    ROOT / "data" / "processed" / "occupant_hmi" / "phase10_3"
    / "phase10_3_connectivity_aware_hmi.csv"
)

OUTPUT_CSV = (
    ROOT / "data" / "processed" / "occupant_hmi" / "phase10_4"
    / "phase10_4_restraint_event_hmi.csv"
)

MANIFEST = (
    ROOT / "experiments" / "occupant_hmi" / "phase10_4"
    / "phase10_4_restraint_event_hmi_manifest.json"
)

TARGET_SCENARIO = "C09_RESTRAINT_DOMAIN_ATTACK"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def event_message(state):
    messages = {
        "CONNECTED":
            "Restraint-domain event scenario observed while telemetry is connected.",
        "DISCONNECTED":
            "Connectivity unavailable. Preserve safety evidence locally.",
        "RECOVERED":
            "Connectivity recovered. Buffered evidence is ready for synchronization and SOC handoff.",
    }
    return messages[state]


def main():

    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    target = [
        row for row in rows
        if row["scenario_id"] == TARGET_SCENARIO
    ]

    if len(target) != 3:
        raise RuntimeError(
            f"Expected 3 C09 state records, found {len(target)}"
        )

    states = {
        row["connectivity_state"]
        for row in target
    }

    expected_states = {
        "CONNECTED",
        "DISCONNECTED",
        "RECOVERED",
    }

    if states != expected_states:
        raise RuntimeError(
            f"Unexpected C09 states: {sorted(states)}"
        )

    output = []

    for row in target:

        state = row["connectivity_state"]

        if state == "CONNECTED":
            event_state = "EVENT_OBSERVED"
            hmi_status = "SAFETY_EVENT_MONITORED"
            forensic_action = "CONNECTED_STREAM"
            sync_action = "NOT_REQUIRED"
            soc_action = "NOT_REQUIRED"

        elif state == "DISCONNECTED":
            event_state = "RESILIENCE_MODE"
            hmi_status = "CONNECTIVITY_DEGRADED"
            forensic_action = "LOCAL_FORENSIC_BUFFER"
            sync_action = "DEFERRED"
            soc_action = "DEFERRED"

        else:
            event_state = "RECOVERY_READY"
            hmi_status = "RECOVERY_READY"
            forensic_action = "RECOVERY_QUEUE"
            sync_action = "READY"
            soc_action = "READY"

        output.append({
            "scenario_id": TARGET_SCENARIO,
            "connectivity_state": state,
            "event_state": event_state,
            "hmi_status": hmi_status,
            "hmi_message": event_message(state),
            "forensic_state": forensic_action,
            "synchronization_state": sync_action,
            "soc_handoff_state": soc_action,
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

    fields = list(output[0].keys())

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)

    checks = {
        "target_scenario_present":
            len(target) == 3,

        "connected_present":
            sum(x["connectivity_state"] == "CONNECTED" for x in output) == 1,

        "disconnected_present":
            sum(x["connectivity_state"] == "DISCONNECTED" for x in output) == 1,

        "recovered_present":
            sum(x["connectivity_state"] == "RECOVERED" for x in output) == 1,

        "event_observed":
            any(x["event_state"] == "EVENT_OBSERVED" for x in output),

        "local_forensic_buffer":
            any(
                x["forensic_state"] == "LOCAL_FORENSIC_BUFFER"
                for x in output
            ),

        "sync_deferred":
            any(
                x["synchronization_state"] == "DEFERRED"
                for x in output
            ),

        "soc_deferred":
            any(
                x["soc_handoff_state"] == "DEFERRED"
                for x in output
            ),

        "recovery_queue":
            any(
                x["forensic_state"] == "RECOVERY_QUEUE"
                for x in output
            ),

        "sync_ready":
            any(
                x["synchronization_state"] == "READY"
                for x in output
            ),

        "soc_ready":
            any(
                x["soc_handoff_state"] == "READY"
                for x in output
            ),

        "laboratory_only":
            all(x["laboratory_generated"] == "True" for x in output),

        "no_physical_vehicle":
            all(x["physical_vehicle"] == "False" for x in output),

        "no_direct_actuation":
            all(x["direct_actuation"] == "False" for x in output),
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    manifest = {
        "phase": "10.4",
        "title": "Controlled Restraint-Domain Event HMI",
        "status": status,
        "target_scenario": TARGET_SCENARIO,
        "input": str(INPUT_CSV.relative_to(ROOT)),
        "input_sha256": sha256_file(INPUT_CSV),
        "output": str(OUTPUT_CSV.relative_to(ROOT)),
        "records": len(output),
        "event_states": [
            "EVENT_OBSERVED",
            "RESILIENCE_MODE",
            "RECOVERY_READY",
        ],
        "connectivity_states": [
            "CONNECTED",
            "DISCONNECTED",
            "RECOVERED",
        ],
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
    print("=== PHASE 10.4 CONTROLLED RESTRAINT-DOMAIN EVENT HMI ===")
    print(f"Scenario: {TARGET_SCENARIO}")
    print(f"Records: {len(output)}")
    print("States: CONNECTED -> DISCONNECTED -> RECOVERED")
    print(f"Status: {status}")
    print()
    print("Safety boundary:")
    print("  Read-only: True")
    print("  CAN transmission: False")
    print("  Direct vehicle actuation: False")
    print("  Physical vehicle: False")
    print("  Live Cortex XDR: False")
    print("  Frozen Transformer modified: False")
    print("  HCRL fused: False")


if __name__ == "__main__":
    main()
