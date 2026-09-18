import csv
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

HMI_CSV = (
    ROOT / "data" / "processed" / "occupant_hmi" / "phase10_2"
    / "phase10_2_occupant_hmi_states.csv"
)

CONNECTIVITY_CSV = (
    ROOT / "data" / "processed" / "can_hil" / "phase5_16"
    / "phase5_16_connectivity_recovery.csv"
)

OUTPUT_CSV = (
    ROOT / "data" / "processed" / "occupant_hmi" / "phase10_3"
    / "phase10_3_connectivity_aware_hmi.csv"
)

MANIFEST = (
    ROOT / "experiments" / "occupant_hmi" / "phase10_3"
    / "phase10_3_connectivity_aware_hmi_manifest.json"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hmi_connectivity_message(state):
    messages = {
        "CONNECTED": "Safety telemetry connected.",
        "DISCONNECTED": "Safety telemetry connection unavailable. Local evidence preservation active.",
        "RECOVERED": "Connectivity recovered. Evidence synchronization and SOC handoff ready.",
    }
    return messages[state]


def main():

    with open(HMI_CSV, newline="", encoding="utf-8-sig") as f:
        hmi_rows = list(csv.DictReader(f))

    with open(CONNECTIVITY_CSV, newline="", encoding="utf-8-sig") as f:
        connectivity_rows = list(csv.DictReader(f))

    required_states = {
        "CONNECTED",
        "DISCONNECTED",
        "RECOVERED",
    }

    actual_states = {
        row["connectivity_state"]
        for row in connectivity_rows
    }

    if actual_states != required_states:
        raise RuntimeError(
            f"Unexpected connectivity states: {sorted(actual_states)}"
        )

    output = []

    for row in connectivity_rows:

        state = row["connectivity_state"]

        if state == "CONNECTED":
            forensic_state = "CONNECTED_STREAM"
            sync_state = "NOT_REQUIRED"
            soc_state = "NOT_REQUIRED"
            hmi_status = "NORMAL"

        elif state == "DISCONNECTED":
            forensic_state = "LOCAL_FORENSIC_BUFFER"
            sync_state = "DEFERRED"
            soc_state = "DEFERRED"
            hmi_status = "CONNECTIVITY_DEGRADED"

        elif state == "RECOVERED":
            forensic_state = "RECOVERY_QUEUE"
            sync_state = "READY"
            soc_state = "READY"
            hmi_status = "RECOVERY_READY"

        output.append({
            "scenario_id": row["scenario_id"],
            "connectivity_state": state,
            "hmi_status": hmi_status,
            "hmi_message": hmi_connectivity_message(state),
            "forensic_state": forensic_state,
            "synchronization_state": sync_state,
            "soc_handoff_state": soc_state,
            "records": int(row["records"]),
            "local_preservation": row["local_preservation"],
            "disconnect_start": int(row["disconnect_start"]),
            "disconnect_end": int(row["disconnect_end"]),
            "event_timestep": int(row["event_timestep"]),
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

    state_counts = {}

    for state in required_states:
        state_counts[state] = sum(
            x["connectivity_state"] == state
            for x in output
        )

    checks = {
        "input_hmi_exists": HMI_CSV.exists(),
        "input_connectivity_exists": CONNECTIVITY_CSV.exists(),
        "three_connectivity_states": actual_states == required_states,
        "state_records_30": len(output) == 30,
        "connected_states_10": state_counts["CONNECTED"] == 10,
        "disconnected_states_10": state_counts["DISCONNECTED"] == 10,
        "recovered_states_10": state_counts["RECOVERED"] == 10,
        "disconnect_local_buffer": all(
            x["forensic_state"] == "LOCAL_FORENSIC_BUFFER"
            for x in output
            if x["connectivity_state"] == "DISCONNECTED"
        ),
        "disconnect_sync_deferred": all(
            x["synchronization_state"] == "DEFERRED"
            for x in output
            if x["connectivity_state"] == "DISCONNECTED"
        ),
        "disconnect_soc_deferred": all(
            x["soc_handoff_state"] == "DEFERRED"
            for x in output
            if x["connectivity_state"] == "DISCONNECTED"
        ),
        "recovery_queue": all(
            x["forensic_state"] == "RECOVERY_QUEUE"
            for x in output
            if x["connectivity_state"] == "RECOVERED"
        ),
        "recovery_sync_ready": all(
            x["synchronization_state"] == "READY"
            for x in output
            if x["connectivity_state"] == "RECOVERED"
        ),
        "recovery_soc_ready": all(
            x["soc_handoff_state"] == "READY"
            for x in output
            if x["connectivity_state"] == "RECOVERED"
        ),
        "laboratory_only": all(
            x["laboratory_generated"] == "True"
            for x in output
        ),
        "no_physical_vehicle": all(
            x["physical_vehicle"] == "False"
            for x in output
        ),
        "no_direct_actuation": all(
            x["direct_actuation"] == "False"
            for x in output
        ),
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    manifest = {
        "phase": "10.3",
        "title": "Connectivity-Aware Occupant HMI",
        "status": status,
        "inputs": {
            "phase10_2_hmi": str(HMI_CSV.relative_to(ROOT)),
            "phase10_2_hmi_sha256": sha256_file(HMI_CSV),
            "phase5_16_connectivity": str(CONNECTIVITY_CSV.relative_to(ROOT)),
            "phase5_16_connectivity_sha256": sha256_file(CONNECTIVITY_CSV),
        },
        "output": str(OUTPUT_CSV.relative_to(ROOT)),
        "records": len(output),
        "state_counts": state_counts,
        "hmi_states": [
            "NORMAL",
            "CONNECTIVITY_DEGRADED",
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
    print("=== PHASE 10.3 CONNECTIVITY-AWARE OCCUPANT HMI ===")
    print(f"Records: {len(output)}")
    print(f"State counts: {state_counts}")
    print("HMI states: NORMAL, CONNECTIVITY_DEGRADED, RECOVERY_READY")
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
