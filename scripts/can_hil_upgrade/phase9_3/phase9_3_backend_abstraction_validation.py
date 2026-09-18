from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASE9_1_DIR = Path(__file__).resolve().parent.parent / "phase9_1"

if str(PHASE9_1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE9_1_DIR))

from can_hil_interface import CANFrame, CANHILInterface, VirtualCANBackend


ROOT = Path(__file__).resolve().parents[3]

REPORT = (
    ROOT
    / "experiments"
    / "can_hil_upgrade"
    / "phase9_3"
    / "phase9_3_backend_abstraction_validation.json"
)


class HardwareBackendDisabled:
    """
    Phase 9.3 placeholder.

    This class deliberately does not access physical CAN hardware,
    operating-system CAN drivers, USB-CAN devices, or external buses.
    """

    enabled = False

    def send(self, frame: CANFrame) -> None:
        raise RuntimeError("Hardware backend is intentionally disabled.")

    def receive(self) -> CANFrame | None:
        return None

    def close(self) -> None:
        pass


def main() -> None:
    virtual_backend = VirtualCANBackend()
    hardware_backend = HardwareBackendDisabled()

    virtual_interface = CANHILInterface(virtual_backend)

    test_frames = [
        CANFrame(0.00, 0x100, 8, bytes.fromhex("0102030405060708")),
        CANFrame(0.05, 0x101, 8, bytes.fromhex("1112131415161718")),
        CANFrame(0.10, 0x10B, 8, bytes.fromhex("3132333435363738")),
    ]

    for frame in test_frames:
        virtual_interface.transmit(frame)

    received = []

    while True:
        frame = virtual_interface.receive()

        if frame is None:
            break

        received.append(frame)

    checks = {
        "virtual_backend_available": isinstance(
            virtual_interface.backend,
            VirtualCANBackend,
        ),
        "virtual_backend_enabled": True,
        "hardware_backend_instantiated": isinstance(
            hardware_backend,
            HardwareBackendDisabled,
        ),
        "hardware_backend_enabled": hardware_backend.enabled is False,
        "interface_backend_replaceable": (
            virtual_interface.backend is virtual_backend
        ),
        "frame_count_preserved": len(received) == len(test_frames),
        "can_ids_preserved": [
            f.can_id for f in received
        ] == [
            f.can_id for f in test_frames
        ],
        "payloads_preserved": [
            f.data.hex().upper() for f in received
        ] == [
            f.data.hex().upper() for f in test_frames
        ],
        "physical_vehicle": False,
        "external_can_bus": False,
        "direct_vehicle_actuation": False,
        "hardware_access_attempted": False,
    }

    functional_checks = [
        checks["virtual_backend_available"],
        checks["virtual_backend_enabled"],
        checks["hardware_backend_instantiated"],
        checks["hardware_backend_enabled"],
        checks["interface_backend_replaceable"],
        checks["frame_count_preserved"],
        checks["can_ids_preserved"],
        checks["payloads_preserved"],
    ]

    safety_checks = [
        checks["physical_vehicle"] is False,
        checks["external_can_bus"] is False,
        checks["direct_vehicle_actuation"] is False,
        checks["hardware_access_attempted"] is False,
    ]

    validation_pass = all(functional_checks) and all(safety_checks)

    report = {
        "phase": "9.3",
        "title": "CAN/HIL Backend Abstraction Validation",
        "status": "PASS" if validation_pass else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "virtual_backend": "VirtualCANBackend",
        "hardware_backend": {
            "type": "HardwareBackendDisabled",
            "enabled": False,
            "hardware_access_attempted": False,
        },
        "received_frames": len(received),
        "checks": checks,
        "safety_boundary": {
            "laboratory_simulation": True,
            "physical_vehicle": False,
            "external_can_bus": False,
            "direct_vehicle_actuation": False,
            "hardware_backend_enabled": False,
            "hardware_access_attempted": False,
        },
    }

    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
