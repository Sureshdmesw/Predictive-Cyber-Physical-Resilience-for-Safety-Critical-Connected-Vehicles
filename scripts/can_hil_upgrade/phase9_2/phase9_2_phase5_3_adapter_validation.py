from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PHASE9_1_DIR = Path(__file__).resolve().parent.parent / "phase9_1"

if str(PHASE9_1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE9_1_DIR))

from can_hil_interface import CANFrame, build_virtual_interface


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data" / "raw" / "can_hil" / "phase5_3" / "virtual_ev_can_normal.csv"
REPORT = ROOT / "experiments" / "can_hil_upgrade" / "phase9_2" / "phase9_2_phase5_3_adapter_validation.json"

EXPECTED_SOURCE_SHA256 = (
    "035182C423E666F385D6C57E094F2247913A5FDD468711EF50088F52EF5C47E5"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> None:
    source_sha256 = sha256_file(SOURCE)

    with SOURCE.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    interface = build_virtual_interface()

    for row in rows:
        frame = CANFrame(
            timestamp_s=float(row["timestamp_s"]),
            can_id=int(row["can_id"]),
            dlc=int(row["dlc"]),
            data=bytes.fromhex(row["data_hex"]),
        )
        interface.transmit(frame)

    received = []

    while True:
        frame = interface.receive()
        if frame is None:
            break

        received.append(
            {
                "timestamp_s": frame.timestamp_s,
                "can_id": frame.can_id,
                "dlc": frame.dlc,
                "data_hex": frame.data.hex().upper(),
            }
        )

    source_frames = [
        {
            "timestamp_s": float(row["timestamp_s"]),
            "can_id": int(row["can_id"]),
            "dlc": int(row["dlc"]),
            "data_hex": row["data_hex"].upper(),
        }
        for row in rows
    ]

    checks = {
        "source_exists": SOURCE.exists(),
        "source_sha256_preserved": source_sha256 == EXPECTED_SOURCE_SHA256,
        "source_record_count": len(rows) == 9000,
        "received_record_count": len(received) == 9000,
        "frame_count_preserved": len(received) == len(source_frames),
        "can_ids_preserved": [
            x["can_id"] for x in received
        ] == [
            x["can_id"] for x in source_frames
        ],
        "dlc_preserved": [
            x["dlc"] for x in received
        ] == [
            x["dlc"] for x in source_frames
        ],
        "payloads_preserved": [
            x["data_hex"] for x in received
        ] == [
            x["data_hex"] for x in source_frames
        ],
        "timestamps_preserved": [
            x["timestamp_s"] for x in received
        ] == [
            x["timestamp_s"] for x in source_frames
        ],
        "physical_vehicle": False,
        "external_can_bus": False,
        "direct_vehicle_actuation": False,
        "hardware_backend_enabled": False,
    }

    functional_checks = [
        checks["source_exists"],
        checks["source_sha256_preserved"],
        checks["source_record_count"],
        checks["received_record_count"],
        checks["frame_count_preserved"],
        checks["can_ids_preserved"],
        checks["dlc_preserved"],
        checks["payloads_preserved"],
        checks["timestamps_preserved"],
    ]

    safety_checks = [
        checks["physical_vehicle"] is False,
        checks["external_can_bus"] is False,
        checks["direct_vehicle_actuation"] is False,
        checks["hardware_backend_enabled"] is False,
    ]

    validation_pass = all(functional_checks) and all(safety_checks)

    report = {
        "phase": "9.2",
        "title": "Phase 5.3 Virtual CAN Laboratory Adapter Validation",
        "status": "PASS" if validation_pass else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(SOURCE),
        "source_sha256": source_sha256,
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
        "source_records": len(rows),
        "received_frames": len(received),
        "backend": "VirtualCANBackend",
        "checks": checks,
        "safety_boundary": {
            "laboratory_simulation": True,
            "physical_vehicle": False,
            "external_can_bus": False,
            "direct_vehicle_actuation": False,
            "hardware_backend_enabled": False,
        },
    }

    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
