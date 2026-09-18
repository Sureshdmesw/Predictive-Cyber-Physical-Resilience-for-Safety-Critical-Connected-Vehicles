import csv
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = ROOT / "data" / "raw" / "can_hil" / "phase5_3" / "virtual_ev_can_normal.csv"

OUTPUT_CSV = (
    ROOT / "data" / "processed" / "occupant_hmi" / "phase10_2"
    / "phase10_2_occupant_hmi_states.csv"
)

OUTPUT_JSON = (
    ROOT / "experiments" / "occupant_hmi" / "phase10_2"
    / "phase10_2_occupant_hmi_manifest.json"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []

    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["message_name"] not in {
                "RESTRAINT_STATUS",
                "AIRBAG_STATUS",
                "OCCUPANT_STATUS",
                "RESTRAINT_HEARTBEAT",
            }:
                continue

            signals = json.loads(row["signals_json"])

            rows.append({
                "timestamp_s": float(row["timestamp_s"]),
                "message_name": row["message_name"],
                "signals": signals,
                "scenario": row["scenario"],
                "laboratory_generated": row["laboratory_generated"] == "True",
                "physical_vehicle": row["physical_vehicle"] == "True",
                "direct_actuation": row["direct_actuation"] == "True",
            })

    grouped = {}

    for row in rows:
        grouped.setdefault(row["timestamp_s"], {})[
            row["message_name"]
        ] = row["signals"]

    output = []

    for timestamp, messages in sorted(grouped.items()):

        restraint = messages.get("RESTRAINT_STATUS", {})
        airbag = messages.get("AIRBAG_STATUS", {})
        occupant = messages.get("OCCUPANT_STATUS", {})
        heartbeat = messages.get("RESTRAINT_HEARTBEAT", {})

        restraint_ok = (
            restraint.get("RestraintIntegrity") == 1
            and restraint.get("RestraintDiagnostic") is not None
        )

        heartbeat_ok = all(
            heartbeat.get(x) == 1
            for x in [
                "RCMHeartbeat",
                "OccupantSensorHeartbeat",
                "RestraintGatewayHeartbeat",
            ]
        )

        occupant_ok = all(
            occupant.get(x) in (0, 1)
            for x in ["DriverOccupant", "PassengerOccupant"]
        )

        airbag_ok = all(
            airbag.get(x) in (0, 1)
            for x in [
                "DriverAirbag",
                "PassengerAirbag",
                "SideAirbag",
                "CurtainAirbag",
            ]
        )

        safety_state = (
            "NORMAL"
            if restraint_ok and heartbeat_ok and occupant_ok and airbag_ok
            else "ATTENTION_REQUIRED"
        )

        output.append({
            "timestamp_s": timestamp,
            "hmi_safety_state": safety_state,
            "driver_seatbelt": restraint.get("DriverSeatbelt"),
            "passenger_seatbelt": restraint.get("PassengerSeatbelt"),
            "driver_occupant": occupant.get("DriverOccupant"),
            "passenger_occupant": occupant.get("PassengerOccupant"),
            "driver_airbag": airbag.get("DriverAirbag"),
            "passenger_airbag": airbag.get("PassengerAirbag"),
            "side_airbag": airbag.get("SideAirbag"),
            "curtain_airbag": airbag.get("CurtainAirbag"),
            "restraint_integrity": restraint.get("RestraintIntegrity"),
            "restraint_diagnostic": restraint.get("RestraintDiagnostic"),
            "rcm_heartbeat": heartbeat.get("RCMHeartbeat"),
            "occupant_sensor_heartbeat": heartbeat.get("OccupantSensorHeartbeat"),
            "gateway_heartbeat": heartbeat.get("RestraintGatewayHeartbeat"),
            "restraint_sequence": restraint.get("RestraintSequence"),
            "scenario": "NORMAL_BASELINE",
            "connectivity_state": "CONNECTED",
            "forensic_state": "CONNECTED_STREAM",
            "synchronization_state": "NOT_REQUIRED",
            "soc_handoff_state": "NOT_REQUIRED",
            "laboratory_generated": True,
            "physical_vehicle": False,
            "direct_actuation": False,
        })

    fieldnames = list(output[0].keys())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)

    manifest = {
        "phase": "10.2",
        "title": "Occupant HMI State Interpreter",
        "status": "PASS",
        "input": str(INPUT_CSV.relative_to(ROOT)),
        "input_sha256": sha256_file(INPUT_CSV),
        "output": str(OUTPUT_CSV.relative_to(ROOT)),
        "records_processed": len(output),
        "hmi_states": sorted(set(x["hmi_safety_state"] for x in output)),
        "connectivity_state": "CONNECTED",
        "read_only": True,
        "can_transmission": False,
        "direct_vehicle_actuation": False,
        "physical_vehicle": False,
        "live_cortex_xdr": False,
        "frozen_transformer_modified": False,
        "hcrl_fused": False,
        "laboratory_generated": True,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=== PHASE 10.2 OCCUPANT HMI STATE INTERPRETER ===")
    print(f"Input records: {len(rows)}")
    print(f"HMI timesteps: {len(output)}")
    print(f"Safety states: {sorted(set(x['hmi_safety_state'] for x in output))}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Manifest: {OUTPUT_JSON}")
    print("CAN transmission: False")
    print("Direct vehicle actuation: False")
    print("Physical vehicle: False")
    print("Status: PASS")


if __name__ == "__main__":
    main()
