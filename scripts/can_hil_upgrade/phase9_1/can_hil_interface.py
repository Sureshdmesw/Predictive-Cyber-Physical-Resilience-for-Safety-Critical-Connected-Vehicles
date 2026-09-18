from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CANFrame:
    timestamp_s: float
    can_id: int
    dlc: int
    data: bytes


class CANBackend(Protocol):

    def send(self, frame: CANFrame) -> None:
        ...

    def receive(self) -> CANFrame | None:
        ...

    def close(self) -> None:
        ...


class VirtualCANBackend:

    def __init__(self):
        self._rx: list[CANFrame] = []

    def send(self, frame: CANFrame) -> None:
        self._rx.append(frame)

    def receive(self) -> CANFrame | None:
        if not self._rx:
            return None
        return self._rx.pop(0)

    def close(self) -> None:
        self._rx.clear()


class CANHILInterface:

    def __init__(self, backend: CANBackend):
        self.backend = backend

    def transmit(self, frame: CANFrame) -> None:
        self.backend.send(frame)

    def receive(self) -> CANFrame | None:
        return self.backend.receive()

    def close(self) -> None:
        self.backend.close()


def build_virtual_interface() -> CANHILInterface:
    return CANHILInterface(VirtualCANBackend())
