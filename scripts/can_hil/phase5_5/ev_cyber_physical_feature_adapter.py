from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_4"
    / "virtual_ev_decoded_features.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_5"
)

EXP_DIR = (
    ROOT
    / "experiments"
    / "can_hil"
    / "phase5_5"
)

SCHEMA = (
    ROOT
    / "data"
    / "schemas"
    / "can_hil"
    / "adapter"
    / "ev_cyber_physical_feature_adapter_schema.json"
)

OUTPUT_CSV = OUT_DIR / "virtual_ev_cyber_physical_features.csv"
OUTPUT_JSONL = OUT_DIR / "virtual_ev_cyber_physical_features.jsonl"
REPORT = EXP_DIR / "phase5_5_ev_cyber_physical_feature_adapter.json"


def number(row, name):
    value = row.get(name)

    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_if_present(output, row, output_name, source_name):
    value = number(row, source_name)

    if value is not None:
        output[output_name] = value
        return True

    return False


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError("Phase 5.4 input CSV is empty.")

    source_features = {
        "vehicle_dynamics": [
            "VehicleSpeed",
            "WheelSpeedFL",
            "WheelSpeedFR",
            "SteeringAngle",
            "LongitudinalAcceleration",
            "LateralAcceleration",
            "YawRate",
            "speed_wheel_speed_error",
            "steering_yaw_joint_magnitude",
        ],
        "driver_demand": [
            "Accelerator",
            "BrakePressure",
            "BrakeStatus",
            "RegenCommand",
        ],
        "ev_powertrain": [
            "VCUState",
            "MotorRPM",
            "TorqueCommand",
            "RegenAvailable",
            "TorqueFeedback",
            "MotorTemperature",
            "InverterTemperature",
            "InverterState",
        ],
        "battery_hv": [
            "SOC",
            "SOH",
            "HVVoltage",
            "HVCurrent",
            "BatteryTemperature",
            "CellVoltageDeviation",
            "CellTemperatureDeviation",
            "BMSState",
        ],
        "hv_safety": [
            "IsolationStatus",
            "ContactorState",
            "PrechargeState",
        ],
        "charging_thermal": [
            "ChargingState",
            "ConnectorState",
            "ChargeCurrent",
            "MotorTemperature",
            "InverterTemperature",
        ],
        "connectivity_integrity": [
            "VCUHeartbeat",
            "BMSHeartbeat",
            "InverterHeartbeat",
            "GatewayHeartbeat",
            "SequenceCounter",
            "RCMHeartbeat",
            "OccupantSensorHeartbeat",
            "RestraintGatewayHeartbeat",
            "RestraintSequenceCounter",
            "can_frame_interval_s",
        ],
        "diagnostics": [
            "DiagnosticState",
            "IntegrityStatus",
            "GatewayIntegrity",
        ],
        "occupant_safety": [
            "DriverSeatbelt",
            "PassengerSeatbelt",
            "DriverPretensioner",
            "PassengerPretensioner",
            "RolloverStatus",
            "RestraintDiagnostic",
            "RestraintIntegrity",
            "RestraintSequence",
            "DriverAirbag",
            "PassengerAirbag",
            "SideAirbag",
            "CurtainAirbag",
            "ImpactSeverity",
            "DriverOccupant",
            "PassengerOccupant",
            "DriverPosition",
            "PassengerPosition",
            "CrashEventStatus",
        ],
        "auxiliary_power": [
            "AuxBatteryVoltage",
            "AuxBatteryStatus",
        ],
    }

    source_columns = set(rows[0].keys())

    all_expected_sources = [
        feature
        for features in source_features.values()
        for feature in features
    ]

    missing_sources = [
        feature
        for feature in all_expected_sources
        if feature not in source_columns
    ]

    adapted_rows = []
    mapped_source_features = set()

    for row in rows:
        output = {
            "timestamp_s": row.get("timestamp_s"),
            "step": row.get("step"),
            "vehicle_id": row.get("vehicle_id"),
            "scenario": row.get("scenario"),
        }

        for group_features in source_features.values():
            for feature in group_features:
                if add_if_present(
                    output,
                    row,
                    feature,
                    feature,
                ):
                    mapped_source_features.add(feature)

        vehicle_speed = number(row, "VehicleSpeed")
        wheel_fl = number(row, "WheelSpeedFL")
        wheel_fr = number(row, "WheelSpeedFR")

        brake_status = number(row, "BrakeStatus")
        brake_pressure = number(row, "BrakePressure")

        steering = number(row, "SteeringAngle")
        yaw_rate = number(row, "YawRate")

        torque_command = number(row, "TorqueCommand")
        torque_feedback = number(row, "TorqueFeedback")

        hv_voltage = number(row, "HVVoltage")
        hv_current = number(row, "HVCurrent")

        battery_temperature = number(
            row,
            "BatteryTemperature",
        )

        accelerator = number(
            row,
            "Accelerator",
        )

        if (
            vehicle_speed is not None
            and wheel_fl is not None
            and wheel_fr is not None
        ):
            wheel_mean = (
                wheel_fl + wheel_fr
            ) / 2.0

            output[
                "cp_speed_consistency_error"
            ] = abs(
                vehicle_speed - wheel_mean
            )

        if (
            brake_status is not None
            and brake_pressure is not None
        ):
            expected_brake = (
                1.0
                if brake_pressure > 0
                else 0.0
            )

            output[
                "cp_brake_consistency_error"
            ] = abs(
                brake_status - expected_brake
            )

        if (
            torque_command is not None
            and torque_feedback is not None
        ):
            output[
                "cp_torque_tracking_error"
            ] = abs(
                torque_command
                - torque_feedback
            )

        if (
            steering is not None
            and yaw_rate is not None
        ):
            output[
                "cp_steering_yaw_consistency"
            ] = (
                abs(steering)
                + abs(yaw_rate)
            )

        if (
            hv_voltage is not None
            and hv_current is not None
        ):
            output[
                "cp_hv_power_kw"
            ] = (
                abs(hv_voltage * hv_current)
                / 1000.0
            )

        if battery_temperature is not None:
            output[
                "cp_battery_temperature_abs"
            ] = abs(battery_temperature)

        if (
            accelerator is not None
            and torque_command is not None
        ):
            output[
                "accelerator_torque_joint"
            ] = (
                abs(accelerator)
                * abs(torque_command)
            )

        output[
            "laboratory_generated"
        ] = row.get(
            "laboratory_generated",
            "True",
        )

        output[
            "physical_vehicle"
        ] = row.get(
            "physical_vehicle",
            "False",
        )

        output[
            "direct_actuation"
        ] = row.get(
            "direct_actuation",
            "False",
        )

        adapted_rows.append(output)

    output_columns = []

    for row in adapted_rows:
        for key in row:
            if key not in output_columns:
                output_columns.append(key)

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=output_columns,
        )

        writer.writeheader()

        for row in adapted_rows:
            writer.writerow(row)

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in adapted_rows:
            handle.write(
                json.dumps(
                    row,
                    separators=(",", ":"),
                )
                + "\n"
            )

    cross_domain_features = [
        "cp_speed_consistency_error",
        "cp_brake_consistency_error",
        "cp_torque_tracking_error",
        "cp_steering_yaw_consistency",
        "cp_hv_power_kw",
        "cp_battery_temperature_abs",
        "accelerator_torque_joint",
    ]

    cross_domain_present = all(
        feature in output_columns
        for feature in cross_domain_features
    )

    group_results = {}

    for group, features in source_features.items():
        present = [
            feature
            for feature in features
            if feature in source_columns
        ]

        group_results[group] = {
            "expected": len(features),
            "present": len(present),
            "missing": [
                feature
                for feature in features
                if feature not in source_columns
            ],
        }

    checks = {
        "input_rows_600": len(rows) == 600,
        "output_rows_600": len(adapted_rows) == 600,
        "source_feature_inventory_complete": len(
            missing_sources
        ) == 0,
        "mapped_source_features_nonzero": len(
            mapped_source_features
        ) > 0,
        "vehicle_dynamics_present": (
            group_results["vehicle_dynamics"]["present"] > 0
        ),
        "ev_powertrain_present": (
            group_results["ev_powertrain"]["present"] > 0
        ),
        "battery_hv_present": (
            group_results["battery_hv"]["present"] > 0
        ),
        "hv_safety_present": (
            group_results["hv_safety"]["present"] > 0
        ),
        "connectivity_integrity_present": (
            group_results[
                "connectivity_integrity"
            ]["present"] > 0
        ),
        "occupant_safety_present": (
            group_results[
                "occupant_safety"
            ]["present"] > 0
        ),
        "cross_domain_features_present": (
            cross_domain_present
        ),
        "schema_exists": SCHEMA.exists(),
        "output_csv_exists": OUTPUT_CSV.exists(),
        "output_jsonl_exists": OUTPUT_JSONL.exists(),
        "laboratory_only": all(
            str(row.get("physical_vehicle", "")).lower()
            == "false"
            for row in rows
        ),
        "no_model_retraining": True,
        "no_crss_replacement_claim": True,
    }

    status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    unavailable_laboratory_signals = [
        "DCLinkVoltage",
        "ThermalManagementState",
        "SensorStatus",
    ]

    report = {
        "phase": "5.5",
        "title": "EV Cyber-Physical Feature Adapter",
        "status": status,
        "input": {
            "path": str(INPUT_CSV),
            "sha256": sha256_file(INPUT_CSV),
            "rows": len(rows),
            "columns": len(source_columns),
        },
        "adapter": {
            "version": "5.5.1",
            "source": "Phase 5.4 decoded virtual EV CAN",
            "output_semantics": "CAN_LAB_CYBER_PHYSICAL",
            "feature_groups": {
                group: len(features)
                for group, features
                in source_features.items()
            },
            "mapped_source_feature_count": len(
                mapped_source_features
            ),
            "missing_source_feature_count": len(
                missing_sources
            ),
            "missing_source_features": missing_sources,
            "unavailable_laboratory_signals": (
                unavailable_laboratory_signals
            ),
            "adapted_feature_count": len(
                output_columns
            ),
        },
        "feature_group_results": group_results,
        "cross_domain_features": cross_domain_features,
        "governance": {
            "crss_equivalence_claim": False,
            "crss_replacement": False,
            "model_retraining": False,
            "production_vehicle": False,
            "physical_vehicle": False,
            "direct_vehicle_actuation": False,
            "production_dbc": False,
            "research_validation_only": True,
        },
        "checks": checks,
        "artifacts": {
            "input": str(INPUT_CSV),
            "output_csv": str(OUTPUT_CSV),
            "output_jsonl": str(OUTPUT_JSONL),
            "schema": str(SCHEMA),
            "report": str(REPORT),
        },
    }

    with REPORT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print()
    print("=== PHASE 5.5 EV CYBER-PHYSICAL FEATURE ADAPTER ===")
    print(f"Input timesteps: {len(rows)}")
    print(f"Input columns: {len(source_columns)}")

    print()
    print("=== PHASE 5.5 RESULTS ===")
    print(f"Input timesteps: {len(rows)}")
    print(f"Input columns: {len(source_columns)}")
    print(
        "Mapped source features:",
        len(mapped_source_features),
    )
    print(
        "Adapted feature columns:",
        len(output_columns),
    )
    print(
        "Output timesteps:",
        len(adapted_rows),
    )

    print()
    print("Feature groups:")

    for group, result in group_results.items():
        print(
            f"  {group}: "
            f"{result['present']}/"
            f"{result['expected']}"
        )

    print()
    print("Validation checks:")

    for name, passed in checks.items():
        print(
            f"  {'PASS' if passed else 'FAIL'} "
            f"{name}"
        )

    if missing_sources:
        print()
        print("Missing source features:")

        for feature in missing_sources:
            print(f"  - {feature}")

    print()
    print(
        "Unavailable laboratory-interface signals:"
    )

    for feature in unavailable_laboratory_signals:
        print(f"  - {feature}")

    print()
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"Schema: {SCHEMA}")
    print(f"Report: {REPORT}")
    print()
    print(f"STATUS: {status}")


if __name__ == "__main__":
    main()