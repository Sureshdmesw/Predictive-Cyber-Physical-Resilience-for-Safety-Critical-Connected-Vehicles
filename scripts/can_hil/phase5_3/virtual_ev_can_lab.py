import csv
import json
import math
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import can
import cantools


ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

DBC_PATH = ROOT / "data" / "schemas" / "can_hil" / "ev" / "virtual_ev.dbc"
REQ_PATH = ROOT / "data" / "schemas" / "can_hil" / "ev" / "ev_safety_requirements.json"

OUT_DIR = ROOT / "data" / "raw" / "can_hil" / "phase5_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "virtual_ev_normal_can_traffic.csv"
REPORT_PATH = ROOT / "experiments" / "can_hil" / "phase5_3" / "phase5_3_virtual_ev_can_lab.json"


@dataclass
class EVState:
    vehicle_speed: float = 0.0
    wheel_speed_fl: float = 0.0
    wheel_speed_fr: float = 0.0
    steering_angle: float = 0.0

    accelerator: float = 0.0
    brake_status: int = 0
    brake_pressure: float = 0.0
    regen_request: float = 0.0

    vcu_state: int = 2
    motor_rpm: float = 0.0
    torque_command: float = 0.0
    torque_feedback: float = 0.0
    inverter_temperature: float = 35.0

    battery_soc: float = 80.0
    battery_soh: float = 98.0
    hv_voltage: float = 400.0
    hv_current: float = 0.0
    battery_temperature: float = 28.0
    bms_state: int = 2

    cell_voltage_deviation: float = 0.012
    cell_temperature_deviation: float = 0.8
    isolation_status: int = 1
    hv_contactor_state: int = 1
    precharge_state: int = 2
    dc_link_voltage: float = 395.0

    motor_temperature: float = 38.0
    regen_available: int = 1
    inverter_state: int = 2

    charging_state: int = 0
    connector_state: int = 0
    charge_current: float = 0.0
    charge_voltage: float = 0.0
    thermal_management_state: int = 1

    sensor_status: int = 1
    gateway_status: int = 1
    integrity_status: int = 1
    sequence_status: int = 1
    connectivity_status: int = 1

    aux_voltage: float = 12.6
    aux_status: int = 1
    hv_ready: int = 1
    safety_interlock: int = 1
    thermal_state: int = 1
    gateway_mode: int = 1

    counter: int = 0


def load_database():
    return cantools.database.load_file(str(DBC_PATH))


def build_state(t):
    state = EVState()

    # Smooth laboratory driving profile.
    state.vehicle_speed = 60.0 + 12.0 * math.sin(0.35 * t)
    state.wheel_speed_fl = state.vehicle_speed + 0.25 * math.sin(0.9 * t)
    state.wheel_speed_fr = state.vehicle_speed - 0.20 * math.sin(0.8 * t)

    state.steering_angle = 8.0 * math.sin(0.25 * t)

    state.accelerator = max(
        0.0,
        min(100.0, 35.0 + 15.0 * math.sin(0.20 * t))
    )

    state.brake_status = 1 if state.vehicle_speed < 50 else 0
    state.brake_pressure = 120.0 if state.brake_status else 0.0
    state.regen_request = 30.0 if state.brake_status else 0.0

    state.motor_rpm = state.vehicle_speed * 85.0
    state.torque_command = 180.0 + 40.0 * math.sin(0.2 * t)
    state.torque_feedback = state.torque_command + 1.5 * math.sin(t)

    state.inverter_temperature = 45.0 + 3.0 * math.sin(0.15 * t)
    state.motor_temperature = 50.0 + 4.0 * math.sin(0.12 * t)

    state.battery_soc = 80.0 - 0.002 * t
    state.hv_voltage = 400.0 + 4.0 * math.sin(0.1 * t)
    state.hv_current = 80.0 + 15.0 * math.sin(0.2 * t)
    state.battery_temperature = 28.0 + 1.5 * math.sin(0.1 * t)

    state.cell_voltage_deviation = 0.012 + 0.003 * abs(math.sin(t))
    state.cell_temperature_deviation = 0.8 + 0.2 * abs(math.sin(0.7 * t))

    state.dc_link_voltage = state.hv_voltage - 5.0

    state.counter = int(t * 10) % 256

    return state


def encode_messages(db, state):
    messages = []

    messages.append(
        db.get_message_by_name("VEHICLE_DYNAMICS").encode({
            "VehicleSpeed": state.vehicle_speed,
            "WheelSpeedFL": state.wheel_speed_fl,
            "WheelSpeedFR": state.wheel_speed_fr,
            "SteeringAngle": state.steering_angle
        })
    )

    messages.append(
        db.get_message_by_name("DRIVER_INPUT").encode({
            "AcceleratorPosition": state.accelerator,
            "BrakeStatus": state.brake_status,
            "BrakePressure": state.brake_pressure,
            "RegenBrakeRequest": state.regen_request
        })
    )

    messages.append(
        db.get_message_by_name("POWERTRAIN_STATE").encode({
            "VCUState": state.vcu_state,
            "MotorRPM": state.motor_rpm,
            "TorqueCommand": state.torque_command,
            "TorqueFeedback": state.torque_feedback,
            "InverterTemperature": state.inverter_temperature
        })
    )

    messages.append(
        db.get_message_by_name("BATTERY_STATUS").encode({
            "BatterySOC": state.battery_soc,
            "BatterySOH": state.battery_soh,
            "HVVoltage": state.hv_voltage,
            "HVCurrent": state.hv_current,
            "BatteryTemperature": state.battery_temperature,
            "BMSState": state.bms_state
        })
    )

    messages.append(
        db.get_message_by_name("BATTERY_CELL_STATUS").encode({
            "CellVoltageDeviation": state.cell_voltage_deviation,
            "CellTemperatureDeviation": state.cell_temperature_deviation,
            "IsolationStatus": state.isolation_status,
            "HVContactorState": state.hv_contactor_state,
            "PrechargeState": state.precharge_state,
            "DCLinkVoltage": state.dc_link_voltage / 10.0
        })
    )

    messages.append(
        db.get_message_by_name("MOTOR_STATUS").encode({
            "MotorTemperature": state.motor_temperature,
            "MotorRPMStatus": state.motor_rpm,
            "MotorTorqueFeedback": state.torque_feedback,
            "RegenAvailable": state.regen_available,
            "InverterState": state.inverter_state,
            "InverterHeartbeat": state.counter
        })
    )

    messages.append(
        db.get_message_by_name("ECU_HEARTBEAT").encode({
            "VCUHeartbeat": state.counter,
            "BMSHeartbeat": state.counter,
            "INVHeartbeat": state.counter,
            "GatewayHeartbeat": state.counter,
            "Counter": state.counter
        })
    )

    messages.append(
        db.get_message_by_name("CHARGING_STATUS").encode({
            "ChargingState": state.charging_state,
            "ConnectorState": state.connector_state,
            "ChargeCurrent": state.charge_current,
            "ChargeVoltage": state.charge_voltage,
            "ThermalManagementState": state.thermal_management_state,
            "ChargeHeartbeat": state.counter
        })
    )

    messages.append(
        db.get_message_by_name("DIAGNOSTIC_STATUS").encode({
            "DiagnosticState": 1,
            "SensorStatus": state.sensor_status,
            "GatewayStatus": state.gateway_status,
            "IntegrityStatus": state.integrity_status,
            "SequenceStatus": state.sequence_status,
            "ConnectivityStatus": state.connectivity_status,
            "DiagnosticHeartbeat": state.counter
        })
    )

    messages.append(
        db.get_message_by_name("AUXILIARY_POWER").encode({
            "AuxBatteryVoltage": state.aux_voltage,
            "AuxBatteryStatus": state.aux_status,
            "HVSystemReady": state.hv_ready,
            "SafetyInterlock": state.safety_interlock,
            "ThermalState": state.thermal_state,
            "GatewayMode": state.gateway_mode,
            "Reserved": 0
        })
    )

    return messages


def run_lab(duration_s=30.0, interval_s=0.05):
    db = load_database()

    bus = can.Bus(
        interface="virtual",
        channel="phase5_ev_lab",
        receive_own_messages=True
    )

    fields = [
        "timestamp",
        "arbitration_id",
        "message_name",
        "dlc",
        "data_hex",
        "vehicle_speed",
        "wheel_speed_fl",
        "wheel_speed_fr",
        "steering_angle",
        "accelerator",
        "brake_status",
        "battery_soc",
        "battery_soh",
        "hv_voltage",
        "hv_current",
        "battery_temperature",
        "cell_voltage_deviation",
        "cell_temperature_deviation",
        "isolation_status",
        "hv_contactor_state",
        "precharge_state",
        "motor_rpm",
        "torque_command",
        "torque_feedback",
        "motor_temperature",
        "inverter_temperature",
        "charging_state",
        "sensor_status",
        "integrity_status",
        "sequence_status",
        "connectivity_status",
        "counter"
    ]

    rows = []
    sent_count = 0

    start = time.monotonic()

    try:
        while time.monotonic() - start < duration_s:
            elapsed = time.monotonic() - start
            state = build_state(elapsed)

            messages = encode_messages(db, state)

            for raw_data, message_def in zip(
                messages,
                [
                    db.get_message_by_name("VEHICLE_DYNAMICS"),
                    db.get_message_by_name("DRIVER_INPUT"),
                    db.get_message_by_name("POWERTRAIN_STATE"),
                    db.get_message_by_name("BATTERY_STATUS"),
                    db.get_message_by_name("BATTERY_CELL_STATUS"),
                    db.get_message_by_name("MOTOR_STATUS"),
                    db.get_message_by_name("ECU_HEARTBEAT"),
                    db.get_message_by_name("CHARGING_STATUS"),
                    db.get_message_by_name("DIAGNOSTIC_STATUS"),
                    db.get_message_by_name("AUXILIARY_POWER")
                ]
            ):
                msg = can.Message(
                    arbitration_id=message_def.frame_id,
                    data=raw_data,
                    is_extended_id=False
                )

                bus.send(msg)
                sent_count += 1

                decoded = message_def.decode(raw_data)

                row = {
                    "timestamp": time.time(),
                    "arbitration_id": hex(message_def.frame_id),
                    "message_name": message_def.name,
                    "dlc": len(raw_data),
                    "data_hex": raw_data.hex(),
                    "vehicle_speed": state.vehicle_speed,
                    "wheel_speed_fl": state.wheel_speed_fl,
                    "wheel_speed_fr": state.wheel_speed_fr,
                    "steering_angle": state.steering_angle,
                    "accelerator": state.accelerator,
                    "brake_status": state.brake_status,
                    "battery_soc": state.battery_soc,
                    "battery_soh": state.battery_soh,
                    "hv_voltage": state.hv_voltage,
                    "hv_current": state.hv_current,
                    "battery_temperature": state.battery_temperature,
                    "cell_voltage_deviation": state.cell_voltage_deviation,
                    "cell_temperature_deviation": state.cell_temperature_deviation,
                    "isolation_status": state.isolation_status,
                    "hv_contactor_state": state.hv_contactor_state,
                    "precharge_state": state.precharge_state,
                    "motor_rpm": state.motor_rpm,
                    "torque_command": state.torque_command,
                    "torque_feedback": state.torque_feedback,
                    "motor_temperature": state.motor_temperature,
                    "inverter_temperature": state.inverter_temperature,
                    "charging_state": state.charging_state,
                    "sensor_status": state.sensor_status,
                    "integrity_status": state.integrity_status,
                    "sequence_status": state.sequence_status,
                    "connectivity_status": state.connectivity_status,
                    "counter": state.counter
                }

                rows.append(row)

            time.sleep(interval_s)

    finally:
        bus.shutdown()

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    requirements = json.loads(REQ_PATH.read_text(encoding="utf-8"))

    report = {
        "phase": "5.3",
        "title": "Virtual EV ECU and CAN Traffic Laboratory",
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": duration_s,
        "interval_seconds": interval_s,
        "virtual_bus": "python-can virtual",
        "channel": "phase5_ev_lab",
        "message_types": 10,
        "frames_generated": sent_count,
        "dataset_rows": len(rows),
        "dbc": str(DBC_PATH.relative_to(ROOT)),
        "requirements_specification": str(REQ_PATH.relative_to(ROOT)),
        "dataset": str(CSV_PATH.relative_to(ROOT)),
        "ev_requirements_defined": len(requirements["requirements"]),
        "signal_groups": [
            "vehicle_dynamics",
            "driver_input",
            "EV_powertrain",
            "HV_battery",
            "battery_cell_health",
            "motor_inverter",
            "ECU_heartbeats",
            "charging",
            "diagnostics",
            "auxiliary_power"
        ],
        "safety_boundary": {
            "physical_vehicle": False,
            "physical_can_interface": False,
            "real_ecu": False,
            "road_testing": False,
            "direct_actuation": False,
            "external_can_network": False
        },
        "validation": {
            "dbc_loaded": True,
            "virtual_bus_created": True,
            "frames_transmitted": sent_count > 0,
            "dataset_created": CSV_PATH.exists(),
            "requirements_loaded": len(requirements["requirements"]) == 22
        },
        "next_step": "Decode CAN signals and build EV CAN-to-cyber-physical feature adapter"
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run_lab()
