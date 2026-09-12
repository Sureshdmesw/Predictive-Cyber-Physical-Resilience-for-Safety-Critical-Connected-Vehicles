from pathlib import Path
import csv
import json
import math
import statistics
import hashlib

import cantools


ROOT = Path(__file__).resolve().parents[3]

DBC_PATH = ROOT / "data" / "schemas" / "can_hil" / "ev" / "virtual_ev_complete.dbc"
INPUT_CSV = ROOT / "data" / "raw" / "can_hil" / "phase5_3" / "virtual_ev_can_normal.csv"

OUTPUT_CSV = ROOT / "data" / "processed" / "can_hil" / "phase5_4" / "virtual_ev_decoded_features.csv"
OUTPUT_JSONL = ROOT / "data" / "processed" / "can_hil" / "phase5_4" / "virtual_ev_decoded_features.jsonl"
REPORT_PATH = ROOT / "experiments" / "can_hil" / "phase5_4" / "phase5_4_signal_decoding_feature_extraction.json"


EXPECTED_MESSAGES = {
    0x100: "VEHICLE_DYNAMICS",
    0x101: "DRIVER_INPUT",
    0x102: "POWERTRAIN_STATE",
    0x103: "BATTERY_STATUS",
    0x104: "BATTERY_CELL_STATUS",
    0x105: "MOTOR_STATUS",
    0x106: "ECU_HEARTBEAT",
    0x107: "CHARGING_STATUS",
    0x108: "DIAGNOSTIC_STATUS",
    0x109: "AUXILIARY_POWER",
    0x10A: "RESCUE_DYNAMICS",
    0x10B: "RESTRAINT_STATUS",
    0x10C: "AIRBAG_STATUS",
    0x10D: "OCCUPANT_STATUS",
    0x10E: "RESTRAINT_HEARTBEAT",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def decode_value(value):
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return float(value)

    return str(value)


def decode_row(db, row):
    # Phase 5.3 writes CAN IDs as decimal strings.
    can_id = int(row["can_id"])

    message = db.get_message_by_frame_id(can_id)

    raw_hex = row["data_hex"].strip()

    data = bytes.fromhex(raw_hex)

    decoded = message.decode(
        data,
        decode_choices=False,
        scaling=True,
    )

    return can_id, message.name, decoded


def numeric(row, name):
    value = row.get(name)

    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)

    return None


def main():

    print("=== PHASE 5.4 CAN SIGNAL DECODING & FEATURE EXTRACTION ===")

    if not DBC_PATH.exists():
        raise FileNotFoundError(f"DBC not found: {DBC_PATH}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CAN CSV not found: {INPUT_CSV}")

    db = cantools.database.load_file(str(DBC_PATH))

    print(f"DBC messages: {len(db.messages)}")

    decoded_records = []

    message_counts = {}
    signal_counts = {}
    decode_failures = []

    with INPUT_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row_number, row in enumerate(reader, start=2):

            try:

                can_id, message_name, decoded = decode_row(
                    db,
                    row,
                )

                output = {
                    "timestamp_s": float(row["timestamp_s"]),
                    "step": int(
                        round(
                            float(row["timestamp_s"]) / 0.05
                        )
                    ),
                    "vehicle_id": row["vehicle_id"],
                    "ecu_domain": row["ecu_domain"],
                    "can_id": f"0x{can_id:03X}",
                    "message_name": message_name,
                    "dlc": int(row["dlc"]),
                    "scenario": row["scenario"],
                    "laboratory_generated": (
                        row["laboratory_generated"].lower()
                        == "true"
                    ),
                    "physical_vehicle": (
                        row["physical_vehicle"].lower()
                        == "true"
                    ),
                    "direct_actuation": (
                        row["direct_actuation"].lower()
                        == "true"
                    ),
                }

                for signal_name, value in decoded.items():

                    output[signal_name] = decode_value(value)

                    signal_counts[signal_name] = (
                        signal_counts.get(signal_name, 0) + 1
                    )

                message_counts[message_name] = (
                    message_counts.get(message_name, 0) + 1
                )

                decoded_records.append(output)

            except Exception as exc:

                decode_failures.append(
                    {
                        "row": row_number,
                        "can_id": row.get("can_id"),
                        "message_name": row.get("message_name"),
                        "error": str(exc),
                    }
                )

    # ---------------------------------------------------------------
    # Aggregate all CAN messages belonging to each timestep.
    # ---------------------------------------------------------------

    timesteps = {}

    for record in decoded_records:

        timestamp = round(
            float(record["timestamp_s"]),
            6,
        )

        if timestamp not in timesteps:

            timesteps[timestamp] = {
                "timestamp_s": timestamp,
                "step": record["step"],
                "vehicle_id": record["vehicle_id"],
                "scenario": record["scenario"],
                "laboratory_generated": record[
                    "laboratory_generated"
                ],
                "physical_vehicle": record[
                    "physical_vehicle"
                ],
                "direct_actuation": record[
                    "direct_actuation"
                ],
            }

        target = timesteps[timestamp]

        for key, value in record.items():

            if key in {
                "timestamp_s",
                "step",
                "vehicle_id",
                "ecu_domain",
                "can_id",
                "message_name",
                "dlc",
                "scenario",
                "laboratory_generated",
                "physical_vehicle",
                "direct_actuation",
            }:
                continue

            # Signals are uniquely named in the current DBC.
            target[key] = value

    unified_rows = [
        timesteps[key]
        for key in sorted(timesteps)
    ]

    # ---------------------------------------------------------------
    # Actual signal names from the DBC.
    # ---------------------------------------------------------------

    dbc_signals = []

    for message in db.messages:

        for signal in message.signals:

            dbc_signals.append(signal.name)

    dbc_signals = sorted(set(dbc_signals))

    missing_signals = []

    for signal in dbc_signals:

        if not any(signal in row for row in unified_rows):

            missing_signals.append(signal)

    # ---------------------------------------------------------------
    # Derived cyber-physical features.
    # ---------------------------------------------------------------

    derived_rows = []

    for row in unified_rows:

        output = dict(row)

        # Vehicle / wheel-speed consistency
        vehicle_speed = numeric(
            row,
            "VehicleSpeed",
        )

        wheel_fl = numeric(
            row,
            "WheelSpeedFL",
        )

        wheel_fr = numeric(
            row,
            "WheelSpeedFR",
        )

        wheel_values = [
            value
            for value in [
                wheel_fl,
                wheel_fr,
            ]
            if value is not None
        ]

        if vehicle_speed is not None and wheel_values:

            wheel_mean = statistics.fmean(
                wheel_values
            )

            output["wheel_speed_mean"] = wheel_mean

            output["speed_wheel_speed_error"] = abs(
                vehicle_speed - wheel_mean
            )

        # Steering / yaw consistency
        steering = numeric(
            row,
            "SteeringAngle",
        )

        yaw = numeric(
            row,
            "YawRate",
        )

        if steering is not None and yaw is not None:

            output["steering_yaw_joint_magnitude"] = (
                abs(steering) + abs(yaw)
            )

        # Accelerator / torque relationship
        accelerator = numeric(
            row,
            "Accelerator",
        )

        torque_command = numeric(
            row,
            "MotorTorqueCommand",
        )

        torque_feedback = numeric(
            row,
            "MotorTorqueFeedback",
        )

        if (
            torque_command is not None
            and torque_feedback is not None
        ):

            output[
                "motor_torque_tracking_error"
            ] = abs(
                torque_command
                - torque_feedback
            )

        if (
            accelerator is not None
            and torque_command is not None
        ):

            output["accelerator_torque_joint"] = (
                abs(accelerator)
                * abs(torque_command)
            )

        # HV power
        hv_voltage = numeric(
            row,
            "HVVoltage",
        )

        hv_current = numeric(
            row,
            "HVCurrent",
        )

        if (
            hv_voltage is not None
            and hv_current is not None
        ):

            output["estimated_hv_power_kw"] = (
                abs(
                    hv_voltage
                    * hv_current
                )
                / 1000.0
            )

        # Battery consistency
        battery_temperature = numeric(
            row,
            "BatteryTemperature",
        )

        if battery_temperature is not None:

            output["battery_temperature_abs"] = (
                abs(battery_temperature)
            )

        cell_voltage_deviation = numeric(
            row,
            "CellVoltageDeviation",
        )

        if cell_voltage_deviation is not None:

            output[
                "cell_voltage_deviation_abs"
            ] = abs(
                cell_voltage_deviation
            )

        cell_temperature_deviation = numeric(
            row,
            "CellTemperatureDeviation",
        )

        if cell_temperature_deviation is not None:

            output[
                "cell_temperature_deviation_abs"
            ] = abs(
                cell_temperature_deviation
            )

        # Fixed laboratory CAN period from Phase 5.3.
        output["can_frame_interval_s"] = 0.05

        derived_rows.append(output)

    # ---------------------------------------------------------------
    # Output columns.
    # ---------------------------------------------------------------

    columns = []

    for row in derived_rows:

        for key in row:

            if key not in columns:

                columns.append(key)

    preferred = [
        "timestamp_s",
        "step",
        "vehicle_id",
        "scenario",

        # Vehicle dynamics
        "VehicleSpeed",
        "WheelSpeedFL",
        "WheelSpeedFR",
        "SteeringAngle",

        # Driver
        "Accelerator",
        "BrakePressure",
        "BrakeStatus",
        "RegenCommand",

        # Powertrain
        "VCUState",
        "MotorRPM",
        "MotorTorqueCommand",
        "MotorTorqueFeedback",

        # Battery
        "BatterySOC",
        "BatterySOH",
        "HVVoltage",
        "HVCurrent",
        "BatteryTemperature",
        "CellVoltageDeviation",
        "CellTemperatureDeviation",

        # HV
        "BMSState",
        "HVContactorState",
        "PrechargeState",
        "DCLinkVoltage",

        # Inverter
        "InverterTemperature",

        # Charging
        "ChargingState",
        "ChargeConnectorState",
        "HVIsolationStatus",

        # Restraints
        "RestraintECUState",
        "CrashEventStatus",
        "ImpactSeverity",
        "DriverOccupantStatus",
        "PassengerOccupantStatus",
        "DriverSeatbeltStatus",
        "PassengerSeatbeltStatus",
        "DriverPretensionerStatus",
        "PassengerPretensionerStatus",

        # Airbags
        "DriverAirbagStatus",
        "PassengerAirbagStatus",
        "SideAirbagStatus",
        "CurtainAirbagStatus",
        "RolloverStatus",

        # Dynamics
        "LongitudinalAcceleration",
        "LateralAcceleration",
        "YawRate",

        # Heartbeats
        "VCUHeartbeat",
        "BMSHeartbeat",
        "InverterHeartbeat",
        "GatewayHeartbeat",
        "RestraintHeartbeat",

        # Diagnostics
        "DiagnosticState",

        # Derived
        "wheel_speed_mean",
        "speed_wheel_speed_error",
        "motor_torque_tracking_error",
        "estimated_hv_power_kw",
        "battery_temperature_abs",
        "cell_voltage_deviation_abs",
        "cell_temperature_deviation_abs",
        "steering_yaw_joint_magnitude",
        "accelerator_torque_joint",
        "can_frame_interval_s",
    ]

    ordered_columns = [
        column
        for column in preferred
        if column in columns
    ]

    ordered_columns += [
        column
        for column in columns
        if column not in ordered_columns
    ]

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=ordered_columns,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in derived_rows:

            writer.writerow(row)

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in derived_rows:

            clean_row = {}

            for key, value in row.items():

                if isinstance(value, float):

                    if not math.isfinite(value):

                        value = None

                clean_row[key] = value

            f.write(
                json.dumps(
                    clean_row,
                    separators=(",", ":"),
                )
                + "\n"
            )

    # ---------------------------------------------------------------
    # Statistics.
    # ---------------------------------------------------------------

    numeric_statistics = {}

    for feature in ordered_columns:

        values = []

        for row in derived_rows:

            value = row.get(feature)

            if isinstance(value, (int, float)):

                value = float(value)

                if math.isfinite(value):

                    values.append(value)

        if values:

            numeric_statistics[feature] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
            }

    # ---------------------------------------------------------------
    # Validation.
    # ---------------------------------------------------------------

    checks = {

        "input_records_9000":
            sum(
                1
                for _ in INPUT_CSV.open(
                    "r",
                    encoding="utf-8",
                )
            ) - 1 == 9000,

        "decoded_records_9000":
            len(decoded_records) == 9000,

        "unified_timesteps_600":
            len(unified_rows) == 600,

        "dbc_messages_15":
            len(db.messages) == 15,

        "decoded_message_types_15":
            len(message_counts) == 15,

        "decode_failures_zero":
            len(decode_failures) == 0,

        "all_expected_dbc_signals_present":
            len(missing_signals) == 0,

        "derived_feature_table_600":
            len(derived_rows) == 600,

        "decoded_csv_exists":
            OUTPUT_CSV.exists(),

        "decoded_jsonl_exists":
            OUTPUT_JSONL.exists(),

        "laboratory_only":
            all(
                row["laboratory_generated"]
                and not row["physical_vehicle"]
                and not row["direct_actuation"]
                for row in decoded_records
            ),
    }

    status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    report = {

        "phase": "5.4",

        "title":
            "CAN Signal Decoding and EV Feature Extraction",

        "status": status,

        "input": {
            "path": str(INPUT_CSV),
            "sha256": sha256_file(INPUT_CSV),
            "records": len(decoded_records),
        },

        "dbc": {
            "path": str(DBC_PATH),
            "sha256": sha256_file(DBC_PATH),
            "message_count": len(db.messages),
            "messages": [
                {
                    "can_id":
                        f"0x{message.frame_id:03X}",
                    "name": message.name,
                    "signal_count":
                        len(message.signals),
                    "signals": [
                        signal.name
                        for signal in message.signals
                    ],
                }
                for message in db.messages
            ],
        },

        "decoding": {
            "decoded_records":
                len(decoded_records),

            "decode_failures":
                len(decode_failures),

            "message_counts":
                message_counts,

            "signal_counts":
                signal_counts,
        },

        "feature_extraction": {

            "unified_timesteps":
                len(unified_rows),

            "dbc_signal_count":
                len(dbc_signals),

            "dbc_signals":
                dbc_signals,

            "missing_signals":
                missing_signals,

            "output_feature_columns":
                len(ordered_columns),

            "derived_features": [
                "wheel_speed_mean",
                "speed_wheel_speed_error",
                "motor_torque_tracking_error",
                "estimated_hv_power_kw",
                "battery_temperature_abs",
                "cell_voltage_deviation_abs",
                "cell_temperature_deviation_abs",
                "steering_yaw_joint_magnitude",
                "accelerator_torque_joint",
                "can_frame_interval_s",
            ],
        },

        "numeric_statistics":
            numeric_statistics,

        "checks":
            checks,

        "safety_boundary": {

            "physical_vehicle_access":
                False,

            "external_can_bus":
                False,

            "direct_vehicle_actuation":
                False,

            "production_dbc":
                False,

            "laboratory_generated_data":
                True,

            "research_validation_only":
                True,
        },

        "outputs": {

            "decoded_csv":
                str(OUTPUT_CSV),

            "decoded_jsonl":
                str(OUTPUT_JSONL),

            "report":
                str(REPORT_PATH),
        },

        "decode_failures_detail":
            decode_failures[:20],
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("=== PHASE 5.4 RESULTS ===")

    print(
        f"Input CAN records: "
        f"{sum(1 for _ in INPUT_CSV.open('r', encoding='utf-8')) - 1}"
    )

    print(
        f"Decoded CAN records: "
        f"{len(decoded_records)}"
    )

    print(
        f"Unified timesteps: "
        f"{len(unified_rows)}"
    )

    print(
        f"Message types: "
        f"{len(message_counts)}"
    )

    print(
        f"DBC signals: "
        f"{len(dbc_signals)}"
    )

    print(
        f"Output feature columns: "
        f"{len(ordered_columns)}"
    )

    print(
        f"Decode failures: "
        f"{len(decode_failures)}"
    )

    print()
    print("Validation checks:")

    for name, passed in checks.items():

        print(
            f"  {'PASS' if passed else 'FAIL'} "
            f"{name}"
        )

    if missing_signals:

        print()
        print("Missing DBC signals:")

        for signal in missing_signals:

            print(f"  - {signal}")

    print()
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"Report: {REPORT_PATH}")

    print()
    print(f"STATUS: {status}")


if __name__ == "__main__":
    main()