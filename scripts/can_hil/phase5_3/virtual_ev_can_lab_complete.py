from pathlib import Path
import csv
import json
import math
import random
from datetime import datetime, timezone

import can
import cantools


ROOT = Path(__file__).resolve().parents[3]

DBC_PATH = ROOT / "data" / "schemas" / "can_hil" / "ev" / "virtual_ev_complete.dbc"
OUT_DIR = ROOT / "data" / "raw" / "can_hil" / "phase5_3"
REPORT_DIR = ROOT / "experiments" / "can_hil" / "phase5_3"

CSV_PATH = OUT_DIR / "virtual_ev_can_normal.csv"
JSONL_PATH = OUT_DIR / "virtual_ev_can_normal.jsonl"
REPORT_PATH = REPORT_DIR / "phase5_3_virtual_ev_can_lab.json"


OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42
DURATION_SECONDS = 30.0
STEP_SECONDS = 0.05


random.seed(RANDOM_SEED)


db = cantools.database.load_file(str(DBC_PATH))


def clamp(value, low, high):
    return max(low, min(high, value))


def encode_message(name, signals):
    message = db.get_message_by_name(name)
    data = message.encode(signals)

    return {
        "can_id": message.frame_id,
        "message_name": message.name,
        "dlc": len(data),
        "data_hex": data.hex().upper(),
        "signals": signals,
    }


records = []

sequence = 0
previous_speed = 0.0
previous_motor_rpm = 0.0


steps = int(DURATION_SECONDS / STEP_SECONDS)

for i in range(steps):
    t = i * STEP_SECONDS

    # ------------------------------------------------------------
    # Vehicle dynamics
    # ------------------------------------------------------------

    speed = clamp(
        60.0
        + 8.0 * math.sin(t / 4.0)
        + 2.0 * math.sin(t / 1.7),
        0.0,
        120.0,
    )

    wheel_fl = speed + 0.25 + random.gauss(0, 0.08)
    wheel_fr = speed - 0.20 + random.gauss(0, 0.08)

    steering = (
        4.0 * math.sin(t / 3.0)
        + 1.0 * math.sin(t / 1.4)
    )

    acceleration = (speed - previous_speed) / STEP_SECONDS
    acceleration_g = clamp(acceleration / 9.80665, -2.0, 2.0)

    lateral_g = clamp(
        0.015 * math.sin(t / 2.0) + random.gauss(0, 0.002),
        -1.5,
        1.5,
    )

    yaw_rate = clamp(
        0.8 * steering + random.gauss(0, 0.05),
        -120.0,
        120.0,
    )

    # ------------------------------------------------------------
    # Driver inputs
    # ------------------------------------------------------------

    accelerator = clamp(
        32.0
        + 8.0 * math.sin(t / 4.0)
        + random.gauss(0, 0.8),
        0.0,
        100.0,
    )

    brake_status = 1 if acceleration_g < -0.025 else 0

    brake_pressure = (
        clamp((-acceleration_g) * 25.0, 0.0, 100.0)
        if brake_status
        else 0.0
    )

    regen_command = (
        clamp(brake_pressure * 0.8, 0.0, 100.0)
        if brake_status
        else 5.0
    )

    # ------------------------------------------------------------
    # EV powertrain
    # ------------------------------------------------------------

    motor_rpm = clamp(speed * 55.0, 0.0, 20000.0)

    torque_command = clamp(
        80.0
        + accelerator * 1.2
        - regen_command * 0.8,
        -300.0,
        500.0,
    )

    torque_feedback = clamp(
        torque_command + random.gauss(0, 1.5),
        -300.0,
        500.0,
    )

    regen_available = 1 if speed > 5.0 else 0

    # ------------------------------------------------------------
    # Battery / BMS
    # ------------------------------------------------------------

    soc = clamp(
        80.0 - 0.015 * t + 0.05 * math.sin(t / 5.0),
        0.0,
        100.0,
    )

    soh = 98.0

    hv_voltage = clamp(
        400.0
        - 0.012 * abs(torque_command)
        + random.gauss(0, 0.4),
        300.0,
        450.0,
    )

    hv_current = clamp(
        torque_command * 0.7
        + random.gauss(0, 1.0),
        -300.0,
        500.0,
    )

    battery_temperature = clamp(
        28.0
        + 0.03 * abs(hv_current)
        + 1.0 * math.sin(t / 8.0)
        + random.gauss(0, 0.15),
        -20.0,
        80.0,
    )

    cell_voltage_deviation = clamp(
        0.012 + random.gauss(0, 0.002),
        0.0,
        0.5,
    )

    cell_temperature_deviation = clamp(
        2.0 + random.gauss(0, 0.15),
        0.0,
        30.0,
    )

    isolation_status = 1
    contactor_state = 1
    precharge_state = 2
    bms_state = 2

    # ------------------------------------------------------------
    # Motor / inverter
    # ------------------------------------------------------------

    motor_temperature = clamp(
        50.0
        + 0.008 * abs(motor_rpm)
        + random.gauss(0, 0.5),
        -20.0,
        150.0,
    )

    inverter_temperature = clamp(
        45.0
        + 0.006 * abs(motor_rpm)
        + random.gauss(0, 0.5),
        -20.0,
        150.0,
    )

    inverter_state = 2

    # ------------------------------------------------------------
    # Charging
    # ------------------------------------------------------------

    charging_state = 0
    connector_state = 0
    charge_current = 0.0

    # ------------------------------------------------------------
    # Diagnostics / integrity
    # ------------------------------------------------------------

    diagnostic_state = 0
    integrity_status = 1
    gateway_integrity = 1

    aux_voltage = clamp(
        12.6 + random.gauss(0, 0.04),
        11.5,
        14.5,
    )

    aux_status = 1

    # ------------------------------------------------------------
    # Restraint / crash dynamics
    # ------------------------------------------------------------

    crash_event_status = 0
    impact_severity = 0
    rollover_status = 0

    driver_seatbelt = 1
    passenger_seatbelt = 1

    driver_pretensioner = 0
    passenger_pretensioner = 0

    restraint_diagnostic = 0
    restraint_integrity = 1

    driver_airbag = 0
    passenger_airbag = 0
    side_airbag = 0
    curtain_airbag = 0

    driver_occupant = 1
    passenger_occupant = 1

    driver_position = 50
    passenger_position = 50

    # ------------------------------------------------------------
    # Heartbeats / counters
    # ------------------------------------------------------------

    sequence = sequence % 256

    vcu_hb = 1
    bms_hb = 1
    inverter_hb = 1
    gateway_hb = 1

    rcm_hb = 1
    occupant_sensor_hb = 1
    restraint_gateway_hb = 1

    restraint_sequence = sequence

    timestamp = i * STEP_SECONDS

    message_values = [
        (
            "VEHICLE_DYNAMICS",
            {
                "VehicleSpeed": speed,
                "WheelSpeedFL": wheel_fl,
                "WheelSpeedFR": wheel_fr,
                "SteeringAngle": steering,
            },
        ),
        (
            "DRIVER_INPUT",
            {
                "Accelerator": accelerator,
                "BrakeStatus": brake_status,
                "BrakePressure": brake_pressure,
                "RegenCommand": regen_command,
            },
        ),
        (
            "POWERTRAIN_STATE",
            {
                "VCUState": 2,
                "MotorRPM": motor_rpm,
                "TorqueCommand": torque_command,
                "RegenAvailable": regen_available,
            },
        ),
        (
            "BATTERY_STATUS",
            {
                "SOC": soc,
                "SOH": soh,
                "HVVoltage": hv_voltage,
                "HVCurrent": hv_current,
                "BatteryTemperature": battery_temperature,
                "BMSState": bms_state,
            },
        ),
        (
            "BATTERY_CELL_STATUS",
            {
                "CellVoltageDeviation": cell_voltage_deviation,
                "CellTemperatureDeviation": cell_temperature_deviation,
                "IsolationStatus": isolation_status,
                "ContactorState": contactor_state,
                "PrechargeState": precharge_state,
            },
        ),
        (
            "MOTOR_STATUS",
            {
                "TorqueFeedback": torque_feedback,
                "MotorTemperature": motor_temperature,
                "InverterTemperature": inverter_temperature,
                "InverterState": inverter_state,
            },
        ),
        (
            "ECU_HEARTBEAT",
            {
                "VCUHeartbeat": vcu_hb,
                "BMSHeartbeat": bms_hb,
                "InverterHeartbeat": inverter_hb,
                "GatewayHeartbeat": gateway_hb,
                "SequenceCounter": sequence,
            },
        ),
        (
            "CHARGING_STATUS",
            {
                "ChargingState": charging_state,
                "ConnectorState": connector_state,
                "ChargeCurrent": charge_current,
            },
        ),
        (
            "DIAGNOSTIC_STATUS",
            {
                "DiagnosticState": diagnostic_state,
                "IntegrityStatus": integrity_status,
                "GatewayIntegrity": gateway_integrity,
            },
        ),
        (
            "AUXILIARY_POWER",
            {
                "AuxBatteryVoltage": aux_voltage,
                "AuxBatteryStatus": aux_status,
            },
        ),
        (
            "RESCUE_DYNAMICS",
            {
                "LongitudinalAcceleration": acceleration_g,
                "LateralAcceleration": lateral_g,
                "YawRate": yaw_rate,
                "CrashEventStatus": crash_event_status,
            },
        ),
        (
            "RESTRAINT_STATUS",
            {
                "DriverSeatbelt": driver_seatbelt,
                "PassengerSeatbelt": passenger_seatbelt,
                "DriverPretensioner": driver_pretensioner,
                "PassengerPretensioner": passenger_pretensioner,
                "RolloverStatus": rollover_status,
                "RestraintDiagnostic": restraint_diagnostic,
                "RestraintIntegrity": restraint_integrity,
                "RestraintSequence": restraint_sequence,
            },
        ),
        (
            "AIRBAG_STATUS",
            {
                "DriverAirbag": driver_airbag,
                "PassengerAirbag": passenger_airbag,
                "SideAirbag": side_airbag,
                "CurtainAirbag": curtain_airbag,
                "ImpactSeverity": impact_severity,
            },
        ),
        (
            "OCCUPANT_STATUS",
            {
                "DriverOccupant": driver_occupant,
                "PassengerOccupant": passenger_occupant,
                "DriverPosition": driver_position,
                "PassengerPosition": passenger_position,
            },
        ),
        (
            "RESTRAINT_HEARTBEAT",
            {
                "RCMHeartbeat": rcm_hb,
                "OccupantSensorHeartbeat": occupant_sensor_hb,
                "RestraintGatewayHeartbeat": restraint_gateway_hb,
                "RestraintSequenceCounter": restraint_sequence,
            },
        ),
    ]

    for message_name, signals in message_values:
        encoded = encode_message(message_name, signals)

        record = {
            "timestamp_s": round(timestamp, 3),
            "vehicle_id": "EV-LAB-001",
            "ecu_domain": message_name,
            "can_id": encoded["can_id"],
            "message_name": encoded["message_name"],
            "dlc": encoded["dlc"],
            "data_hex": encoded["data_hex"],
            "signals_json": json.dumps(signals, sort_keys=True),
            "scenario": "NORMAL_BASELINE",
            "laboratory_generated": True,
            "physical_vehicle": False,
            "direct_actuation": False,
        }

        records.append(record)

    previous_speed = speed
    previous_motor_rpm = motor_rpm
    sequence += 1


# ------------------------------------------------------------
# Write CSV
# ------------------------------------------------------------

fieldnames = [
    "timestamp_s",
    "vehicle_id",
    "ecu_domain",
    "can_id",
    "message_name",
    "dlc",
    "data_hex",
    "signals_json",
    "scenario",
    "laboratory_generated",
    "physical_vehicle",
    "direct_actuation",
]

with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)


# ------------------------------------------------------------
# Write JSONL
# ------------------------------------------------------------

with JSONL_PATH.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, sort_keys=True) + "\n")


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

unique_ids = sorted(set(r["can_id"] for r in records))
unique_messages = sorted(set(r["message_name"] for r in records))

expected_ids = list(range(0x100, 0x10F))

all_dlc_8 = all(r["dlc"] == 8 for r in records)
all_lab = all(r["laboratory_generated"] for r in records)
all_no_vehicle = all(not r["physical_vehicle"] for r in records)
all_no_actuation = all(not r["direct_actuation"] for r in records)

expected_records = steps * 15

checks = {
    "record_count": len(records) == expected_records,
    "message_count": len(unique_messages) == 15,
    "can_id_count": len(unique_ids) == 15,
    "can_ids_contiguous": unique_ids == expected_ids,
    "all_dlc_8": all_dlc_8,
    "all_laboratory_generated": all_lab,
    "no_physical_vehicle": all_no_vehicle,
    "no_direct_actuation": all_no_actuation,
}

status = all(checks.values())


report = {
    "phase": "5.3",
    "status": "PASS" if status else "FAIL",
    "scenario": "NORMAL_BASELINE",
    "random_seed": RANDOM_SEED,
    "duration_seconds": DURATION_SECONDS,
    "step_seconds": STEP_SECONDS,
    "expected_records": expected_records,
    "actual_records": len(records),
    "can_messages_per_step": 15,
    "unique_can_messages": unique_messages,
    "unique_can_ids": unique_ids,
    "checks": checks,
    "artifacts": {
        "dbc": str(DBC_PATH),
        "csv": str(CSV_PATH),
        "jsonl": str(JSONL_PATH),
    },
    "safety_boundary": {
        "physical_vehicle_access": False,
        "external_can_bus": False,
        "direct_vehicle_actuation": False,
        "production_dbc": False,
        "laboratory_generated_signals": True,
    },
}


REPORT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)


print("=== PHASE 5.3 VIRTUAL EV CAN LAB ===")
print(f"Duration: {DURATION_SECONDS} s")
print(f"Time step: {STEP_SECONDS} s")
print(f"CAN messages per step: 15")
print(f"Expected records: {expected_records}")
print(f"Actual records: {len(records)}")
print(f"Unique CAN messages: {len(unique_messages)}")
print(f"Unique CAN IDs: {len(unique_ids)}")
print(f"CSV: {CSV_PATH}")
print(f"JSONL: {JSONL_PATH}")
print(f"Report: {REPORT_PATH}")
print("")
print("Validation checks:")

for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'} {name}")

print("")
print(f"STATUS: {'PASS' if status else 'FAIL'}")