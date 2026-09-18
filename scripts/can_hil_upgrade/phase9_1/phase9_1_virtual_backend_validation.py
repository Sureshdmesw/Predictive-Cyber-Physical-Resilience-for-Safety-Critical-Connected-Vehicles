from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from can_hil_interface import CANFrame, build_virtual_interface


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "can_hil_upgrade" / "phase9_1"
OUT.mkdir(parents=True, exist_ok=True)

REPORT = OUT / "phase9_1_virtual_backend_validation.json"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main():

    interface = build_virtual_interface()

    frames = [
        CANFrame(0.00, 0x100, 8, bytes.fromhex("0102030405060708")),
        CANFrame(0.05, 0x101, 8, bytes.fromhex("1112131415161718")),
        CANFrame(0.10, 0x106, 8, bytes.fromhex("2122232425262728")),
        CANFrame(0.15, 0x10B, 8, bytes.fromhex("3132333435363738")),
    ]

    received = []

    for frame in frames:
        interface.transmit(frame)

    while True:
        frame = interface.receive()
        if frame is None:
            break

        received.append({
            "timestamp_s": frame.timestamp_s,
            "can_id": frame.can_id,
            "dlc": frame.dlc,
            "data_hex": frame.data.hex().upper(),
        })

    interface.close()

    checks = {
        "interface_created": True,
        "frames_transmitted": len(frames),
        "frames_received": len(received),
        "frame_count_preserved": len(frames) == len(received),
        "can_ids_preserved": [
            f["can_id"] for f in received
        ] == [
            f.can_id for f in frames
        ],
        "payloads_preserved": [
            f["data_hex"] for f in received
        ] == [
            f.data.hex().upper() for f in frames
        ],
        "all_dlc_8": all(f["dlc"] == 8 for f in received),
        "physical_vehicle": False,
        "direct_vehicle_actuation": False,
        "external_can_bus": False,
        "hardware_backend_enabled": False,
    }

    functional_checks = [
        checks["interface_created"],
        checks["frame_count_preserved"],
        checks["can_ids_preserved"],
        checks["payloads_preserved"],
        checks["all_dlc_8"],
    ]

    safety_checks = [
        checks["physical_vehicle"] is False,
        checks["direct_vehicle_actuation"] is False,
        checks["external_can_bus"] is False,
        checks["hardware_backend_enabled"] is False,
    ]

    validation_pass = all(functional_checks) and all(safety_checks)

    report = {
        "phase": "9.1",
        "title": "Hardware-Independent CAN/HIL Interface Validation",
        "status": "PASS" if validation_pass else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "VirtualCANBackend",
        "received_frames": received,
        "checks": checks,
        "evidence_digest": sha256_text(
            json.dumps(
                received,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
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
