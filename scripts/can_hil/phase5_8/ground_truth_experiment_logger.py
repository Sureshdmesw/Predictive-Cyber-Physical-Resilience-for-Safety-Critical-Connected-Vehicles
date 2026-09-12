#!/usr/bin/env python3
"""
Phase 5.8 â€” Ground-Truth Experiment Logger

Purpose
-------
Build a reproducible, laboratory-only ground-truth/provenance layer that
connects:

    Phase 5.5 EV cyber-physical features
        -> Phase 5.6 safety-requirement evaluations
        -> Phase 5.7 frozen Transformer predictions

Important research boundaries
-----------------------------
- Laboratory-generated CAN data only.
- No physical vehicle.
- No direct vehicle actuation.
- No real-world cyberattack labels.
- No live Cortex XDR API.
- No model retraining.
- Ground-truth classes below are deterministic laboratory state labels,
  not claims about real-world attacks.

The Phase 5.6 file contains one row per requirement evaluation
(600 timesteps x 40 requirements = 24,000 rows). This script aggregates
those 40 evaluations into exactly one requirement-summary row per timestep
before joining to the 600-row Phase 5.5 feature table. This prevents
many-to-one timestamp joins from duplicating experiment rows.

Outputs
-------
data/processed/can_hil/phase5_8/
    phase5_8_ground_truth_experiment_log.csv
    phase5_8_ground_truth_experiment_log.jsonl
    phase5_8_ground_truth_schema.json

experiments/can_hil/phase5_8/
    phase5_8_ground_truth_experiment_logger.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

INPUT_FEATURES = (
    ROOT / "data" / "processed" / "can_hil" / "phase5_5"
    / "virtual_ev_cyber_physical_features.csv"
)

INPUT_REQUIREMENTS = (
    ROOT / "data" / "processed" / "can_hil" / "phase5_6"
    / "ev_safety_requirements_evaluation.csv"
)

INPUT_PREDICTIONS = (
    ROOT / "data" / "processed" / "can_hil" / "phase5_7"
    / "can_ev_frozen_transformer_predictions.csv"
)

OUT_DIR = ROOT / "data" / "processed" / "can_hil" / "phase5_8"
EXP_DIR = ROOT / "experiments" / "can_hil" / "phase5_8"

OUT_CSV = OUT_DIR / "phase5_8_ground_truth_experiment_log.csv"
OUT_JSONL = OUT_DIR / "phase5_8_ground_truth_experiment_log.jsonl"
OUT_SCHEMA = OUT_DIR / "phase5_8_ground_truth_schema.json"
OUT_REPORT = EXP_DIR / "phase5_8_ground_truth_experiment_logger.json"

GENESIS = "GROUND_TRUTH_GENESIS"

EXPECTED_FEATURE_ROWS = 600
EXPECTED_PREDICTION_ROWS = 589
EXPECTED_REQUIREMENT_EVALUATIONS = 24000
EXPECTED_REQUIREMENTS_PER_STEP = 40

NUMERIC_FIELDS = {
    "timestamp_s",
    "step",
    "VehicleSpeed",
    "WheelSpeedFL",
    "WheelSpeedFR",
    "SteeringAngle",
    "Accelerator",
    "BrakePressure",
    "BrakeStatus",
    "RegenCommand",
    "VCUState",
    "MotorRPM",
    "HVVoltage",
    "HVCurrent",
    "BatteryTemperature",
    "CellVoltageDeviation",
    "CellTemperatureDeviation",
    "BMSState",
    "PrechargeState",
    "InverterTemperature",
    "ChargingState",
    "CrashEventStatus",
    "ImpactSeverity",
    "RolloverStatus",
    "LongitudinalAcceleration",
    "LateralAcceleration",
    "YawRate",
    "VCUHeartbeat",
    "BMSHeartbeat",
    "InverterHeartbeat",
    "GatewayHeartbeat",
    "DiagnosticState",
    "wheel_speed_mean",
    "speed_wheel_speed_error",
    "estimated_hv_power_kw",
    "battery_temperature_abs",
    "cell_voltage_deviation_abs",
    "cell_temperature_deviation_abs",
    "steering_yaw_joint_magnitude",
    "can_frame_interval_s",
    "TorqueCommand",
    "RegenAvailable",
    "SOC",
    "SOH",
    "IsolationStatus",
    "ContactorState",
    "TorqueFeedback",
    "MotorTemperature",
    "InverterState",
    "SequenceCounter",
    "ConnectorState",
    "ChargeCurrent",
    "IntegrityStatus",
    "GatewayIntegrity",
    "AuxBatteryVoltage",
    "AuxBatteryStatus",
    "DriverSeatbelt",
    "PassengerSeatbelt",
    "DriverPretensioner",
    "PassengerPretensioner",
    "RestraintDiagnostic",
    "RestraintIntegrity",
    "RestraintSequence",
    "DriverAirbag",
    "PassengerAirbag",
    "SideAirbag",
    "CurtainAirbag",
    "DriverOccupant",
    "PassengerOccupant",
    "DriverPosition",
    "PassengerPosition",
    "RCMHeartbeat",
    "OccupantSensorHeartbeat",
    "RestraintGatewayHeartbeat",
    "RestraintSequenceCounter",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_boolish(value: Any) -> bool:
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "pass", "passed", "ok", "valid"}


def find_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    lower = {f.lower(): f for f in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def find_semantic_column(fieldnames: list[str], patterns: list[str]) -> str | None:
    for field in fieldnames:
        normalized = re.sub(r"[^a-z0-9]+", "_", field.lower()).strip("_")
        for pattern in patterns:
            if re.search(pattern, normalized):
                return field
    return None


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        fail(f"Required input file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            fail(f"CSV has no header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def normalize_step(row: dict[str, str]) -> int:
    if "step" in row:
        return parse_int(row["step"])
    if "Step" in row:
        return parse_int(row["Step"])
    return parse_int(row.get("timestamp_s", 0) / 0.05)


def requirement_status(row: dict[str, str], status_col: str | None) -> str:
    if status_col is not None:
        value = str(row.get(status_col, "")).strip().lower()
        if value in {"pass", "passed", "ok", "true", "1", "satisfied"}:
            return "PASS"
        if value in {"fail", "failed", "false", "0", "violation", "unsafe"}:
            return "FAIL"
        if value in {"not_evaluable", "not evaluable", "na", "n/a"}:
            return "NOT_EVALUABLE"

    # Fall back to common boolean/numeric result columns.
    for key, value in row.items():
        normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        if normalized in {
            "passed",
            "pass",
            "is_pass",
            "requirement_pass",
            "requirement_passed",
            "satisfied",
            "is_satisfied",
        }:
            return "PASS" if parse_boolish(value) else "FAIL"

    return "UNKNOWN"


def aggregate_requirements(
    fieldnames: list[str], rows: list[dict[str, str]]
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    step_col = find_column(fieldnames, ["step", "Step", "timestep", "time_step"])
    timestamp_col = find_column(fieldnames, ["timestamp_s", "timestamp", "time_s"])
    status_col = find_column(
        fieldnames,
        [
            "status",
            "requirement_status",
            "result",
            "evaluation_status",
            "passed",
            "pass",
        ],
    )
    req_id_col = find_column(
        fieldnames,
        [
            "requirement_id",
            "requirement",
            "req_id",
            "id",
            "requirement_name",
            "name",
        ],
    )
    category_col = find_column(
        fieldnames,
        ["category", "requirement_category", "domain", "group", "feature_group"],
    )

    groups: dict[int, list[dict[str, str]]] = {}

    for row in rows:
        if step_col:
            step = parse_int(row.get(step_col))
        elif timestamp_col:
            step = round(parse_float(row.get(timestamp_col)) / 0.05)
        else:
            fail(
                "Phase 5.6 CSV has neither a step-like nor timestamp-like "
                "column; cannot align requirements to CAN timesteps."
            )
        groups.setdefault(step, []).append(row)

    summary: dict[int, dict[str, Any]] = {}

    for step, items in sorted(groups.items()):
        statuses = [requirement_status(r, status_col) for r in items]
        passed = sum(s == "PASS" for s in statuses)
        failed = sum(s == "FAIL" for s in statuses)
        not_eval = sum(s == "NOT_EVALUABLE" for s in statuses)
        unknown = sum(s == "UNKNOWN" for s in statuses)

        categories: dict[str, dict[str, int]] = {}
        if category_col:
            for item, status in zip(items, statuses):
                category = str(item.get(category_col, "UNKNOWN")).strip() or "UNKNOWN"
                categories.setdefault(
                    category,
                    {"evaluated": 0, "passed": 0, "failed": 0, "not_evaluable": 0},
                )
                categories[category]["evaluated"] += 1
                if status == "PASS":
                    categories[category]["passed"] += 1
                elif status == "FAIL":
                    categories[category]["failed"] += 1
                elif status == "NOT_EVALUABLE":
                    categories[category]["not_evaluable"] += 1

        summary[step] = {
            "requirement_evaluation_count": len(items),
            "requirement_pass_count": passed,
            "requirement_fail_count": failed,
            "requirement_not_evaluable_count": not_eval,
            "requirement_unknown_count": unknown,
            "all_requirements_pass": (
                len(items) > 0 and failed == 0 and not_eval == 0 and unknown == 0
            ),
            "requirement_categories": categories,
            "requirement_ids": (
                [str(r.get(req_id_col, "")).strip() for r in items]
                if req_id_col
                else []
            ),
        }

    metadata = {
        "step_column": step_col,
        "timestamp_column": timestamp_col,
        "status_column": status_col,
        "requirement_id_column": req_id_col,
        "category_column": category_col,
        "unique_requirement_steps": len(summary),
        "max_requirements_per_step": max(
            (v["requirement_evaluation_count"] for v in summary.values()), default=0
        ),
        "min_requirements_per_step": min(
            (v["requirement_evaluation_count"] for v in summary.values()), default=0
        ),
    }
    return summary, metadata


def laboratory_ground_truth(row: dict[str, str]) -> tuple[str, list[str]]:
    """
    Deterministic laboratory state classification.

    Priority is deliberately conservative:
    physical safety -> integrity -> sensor consistency -> battery/HV ->
    diagnostics/timing -> normal.

    These are experimental state labels, not real-world cyberattack labels.
    """
    reasons: list[str] = []

    crash = parse_int(row.get("CrashEventStatus"))
    impact = parse_float(row.get("ImpactSeverity"))
    rollover = parse_int(row.get("RolloverStatus"))
    accel = abs(parse_float(row.get("LongitudinalAcceleration")))
    lat_accel = abs(parse_float(row.get("LateralAcceleration")))

    if crash != 0 or impact > 0 or rollover != 0 or accel > 15 or lat_accel > 15:
        if crash != 0:
            reasons.append("CrashEventStatus")
        if impact > 0:
            reasons.append("ImpactSeverity")
        if rollover != 0:
            reasons.append("RolloverStatus")
        if accel > 15:
            reasons.append("LongitudinalAcceleration")
        if lat_accel > 15:
            reasons.append("LateralAcceleration")
        return "PHYSICAL_SAFETY_EVENT", reasons

    integrity_values = [
        parse_int(row.get("IntegrityStatus")),
        parse_int(row.get("GatewayIntegrity")),
        parse_int(row.get("RestraintIntegrity")),
        parse_int(row.get("IsolationStatus")),
    ]
    if any(v == 0 for v in integrity_values):
        for name, value in zip(
            [
                "IntegrityStatus",
                "GatewayIntegrity",
                "RestraintIntegrity",
                "IsolationStatus",
            ],
            integrity_values,
        ):
            if value == 0:
                reasons.append(name)
        return "INTEGRITY_ANOMALY", reasons

    speed_error = abs(parse_float(row.get("speed_wheel_speed_error")))
    steering_joint = abs(parse_float(row.get("steering_yaw_joint_magnitude")))
    if speed_error > 2.0 or steering_joint > 20.0:
        if speed_error > 2.0:
            reasons.append("speed_wheel_speed_error")
        if steering_joint > 20.0:
            reasons.append("steering_yaw_joint_magnitude")
        return "SENSOR_CONSISTENCY_ANOMALY", reasons

    batt_temp = parse_float(row.get("BatteryTemperature"))
    cell_vdev = parse_float(row.get("CellVoltageDeviation"))
    cell_tdev = parse_float(row.get("CellTemperatureDeviation"))
    if (
        batt_temp < -30
        or batt_temp > 60
        or cell_vdev < 0
        or cell_vdev > 0.1
        or cell_tdev < 0
        or cell_tdev > 10
    ):
        reasons.append("battery_or_cell_plausibility")
        return "BATTERY_SAFETY_ANOMALY", reasons

    hv_v = parse_float(row.get("HVVoltage"))
    hv_i = abs(parse_float(row.get("HVCurrent")))
    precharge = parse_int(row.get("PrechargeState"))
    contactor = parse_int(row.get("ContactorState"))
    if hv_v < 250 or hv_v > 450 or hv_i > 400 or precharge not in {0, 1, 2} or contactor not in {0, 1}:
        reasons.append("HV_plausibility")
        return "HV_SAFETY_ANOMALY", reasons

    diagnostic = parse_int(row.get("DiagnosticState"))
    can_dt = parse_float(row.get("can_frame_interval_s"))
    if diagnostic != 0 or not math.isfinite(can_dt) or abs(can_dt - 0.05) > 0.01:
        if diagnostic != 0:
            reasons.append("DiagnosticState")
        if not math.isfinite(can_dt) or abs(can_dt - 0.05) > 0.01:
            reasons.append("can_frame_interval_s")
        return "DIAGNOSTIC_ANOMALY", reasons

    brake = parse_int(row.get("BrakeStatus"))
    brake_pressure = parse_float(row.get("BrakePressure"))
    regen = parse_float(row.get("RegenCommand"))
    if brake != 0 and regen > 0 and brake_pressure <= 0:
        reasons.extend(["BrakeStatus", "RegenCommand"])
        return "BRAKE_CONSISTENCY_ANOMALY", reasons

    return "NORMAL_LABORATORY_STATE", ["all_lab_rules_within_thresholds"]


def load_prediction_rows() -> tuple[list[str], list[dict[str, str]]]:
    if not INPUT_PREDICTIONS.exists():
        return [], []

    return read_csv(INPUT_PREDICTIONS)


def prediction_lookup(
    fieldnames: list[str], rows: list[dict[str, str]]
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    source_col = find_column(
        fieldnames,
        ["source_row", "source_index", "input_row", "row_index", "step"],
    )
    step_col = find_column(fieldnames, ["step", "Step", "timestep", "time_step"])
    y3_col = find_column(
        fieldnames,
        ["y3_probability", "y3_prob", "y3", "y3_prediction", "probability_y3"],
    )
    y6_col = find_column(
        fieldnames,
        ["y6_probability", "y6_prob", "y6", "y6_prediction", "probability_y6"],
    )

    lookup: dict[int, dict[str, Any]] = {}

    for idx, row in enumerate(rows):
        if source_col:
            source = parse_int(row.get(source_col), default=-1)
        elif step_col:
            source = parse_int(row.get(step_col), default=-1)
        else:
            source = idx + 11  # fallback for a 12-step observation window

        if source < 0:
            continue

        lookup[source] = {
            "prediction_available": True,
            "prediction_source_row": source,
            "y3_probability": parse_float(row.get(y3_col)) if y3_col else float("nan"),
            "y6_probability": parse_float(row.get(y6_col)) if y6_col else float("nan"),
        }

    return lookup, {
        "source_column": source_col,
        "step_column": step_col,
        "y3_column": y3_col,
        "y6_column": y6_col,
        "prediction_rows": len(rows),
    }


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    feature_fields, feature_rows = read_csv(INPUT_FEATURES)
    req_fields, req_rows = read_csv(INPUT_REQUIREMENTS)
    pred_fields, pred_rows = load_prediction_rows()

    if len(feature_rows) != EXPECTED_FEATURE_ROWS:
        fail(
            f"Expected {EXPECTED_FEATURE_ROWS} Phase 5.5 rows, got {len(feature_rows)}."
        )

    if len(req_rows) != EXPECTED_REQUIREMENT_EVALUATIONS:
        fail(
            f"Expected {EXPECTED_REQUIREMENT_EVALUATIONS} Phase 5.6 evaluations, "
            f"got {len(req_rows)}."
        )

    requirement_summary, req_meta = aggregate_requirements(req_fields, req_rows)

    if len(requirement_summary) != EXPECTED_FEATURE_ROWS:
        fail(
            f"Expected one requirement summary per {EXPECTED_FEATURE_ROWS} timestep; "
            f"got {len(requirement_summary)} unique steps."
        )

    bad_counts = {
        step: item["requirement_evaluation_count"]
        for step, item in requirement_summary.items()
        if item["requirement_evaluation_count"] != EXPECTED_REQUIREMENTS_PER_STEP
    }
    if bad_counts:
        fail(
            "Phase 5.6 did not contain exactly 40 requirement evaluations per "
            f"timestep. First mismatches: {dict(list(bad_counts.items())[:5])}"
        )

    predictions, pred_meta = prediction_lookup(pred_fields, pred_rows)

    output_rows: list[dict[str, Any]] = []
    previous_hash = GENESIS
    prediction_join_count = 0
    requirement_join_count = 0

    for row_index, src in enumerate(feature_rows):
        step = normalize_step(src)
        timestamp_s = parse_float(src.get("timestamp_s"))

        if step not in requirement_summary:
            fail(f"No Phase 5.6 requirement summary for step {step}.")

        req = requirement_summary[step]
        requirement_join_count += 1

        pred = predictions.get(step, {
            "prediction_available": False,
            "prediction_source_row": "",
            "y3_probability": float("nan"),
            "y6_probability": float("nan"),
        })
        if pred["prediction_available"]:
            prediction_join_count += 1

        label, reasons = laboratory_ground_truth(src)

        experiment_id = f"P5_8_{step:04d}"

        record: dict[str, Any] = {
            "experiment_id": experiment_id,
            "row_index": row_index,
            "step": step,
            "timestamp_s": timestamp_s,
            "vehicle_id": src.get("vehicle_id", ""),
            "scenario": src.get("scenario", "normal"),
            "ground_truth_class": label,
            "ground_truth_reason": "|".join(reasons),
            "ground_truth_label_source": "DETERMINISTIC_LABORATORY_RULES",
            "real_world_cyberattack_label": False,
            "synthetic_cyber_label": False,
            "laboratory_generated": True,
            "physical_vehicle": False,
            "direct_vehicle_actuation": False,
            "live_cortex_xdr_api": False,
            "model_retraining": False,
            "requirement_evaluation_count": req["requirement_evaluation_count"],
            "requirement_pass_count": req["requirement_pass_count"],
            "requirement_fail_count": req["requirement_fail_count"],
            "requirement_not_evaluable_count": req["requirement_not_evaluable_count"],
            "requirement_unknown_count": req["requirement_unknown_count"],
            "all_requirements_pass": req["all_requirements_pass"],
            "prediction_available": pred["prediction_available"],
            "prediction_source_row": pred["prediction_source_row"],
            "y3_probability": pred["y3_probability"],
            "y6_probability": pred["y6_probability"],
            "source_feature_sha256": "",
            "source_requirement_sha256": sha256_file(INPUT_REQUIREMENTS),
            "source_prediction_sha256": (
                sha256_file(INPUT_PREDICTIONS) if INPUT_PREDICTIONS.exists() else ""
            ),
        }

        # Include the complete Phase 5.5 feature vector in provenance.
        for field in feature_fields:
            if field in {"vehicle_id", "scenario"}:
                continue
            if field in NUMERIC_FIELDS:
                value = parse_float(src.get(field))
                record[f"feature__{field}"] = value
            else:
                record[f"feature__{field}"] = src.get(field, "")

        # Record-level source hash binds the original Phase 5.5 row.
        source_canonical = canonical_json(src).encode("utf-8")
        record["source_feature_sha256"] = sha256_bytes(source_canonical)

        # Hash chain is computed over the complete record except its own hash.
        record["previous_record_hash"] = previous_hash
        record_for_hash = dict(record)
        record_hash = sha256_bytes(canonical_json(record_for_hash).encode("utf-8"))
        record["record_hash"] = record_hash
        previous_hash = record_hash

        output_rows.append(record)

    if len(output_rows) != EXPECTED_FEATURE_ROWS:
        fail(f"Output row count mismatch: {len(output_rows)}")

    if prediction_join_count not in {0, EXPECTED_PREDICTION_ROWS}:
        fail(
            f"Prediction join count was {prediction_join_count}; expected either "
            f"0 (prediction file unavailable) or {EXPECTED_PREDICTION_ROWS}."
        )

    # CSV
    fieldnames = list(output_rows[0].keys())
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(output_rows)

    # JSONL
    # Convert non-finite floating-point values (NaN/Infinity) to JSON null.
    # This occurs legitimately for Phase 5.7 prediction fields during the
    # initial observation window before a prediction is available.
    def json_safe_record(record):
        safe = {}
        for key, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                safe[key] = None
            else:
                safe[key] = value
        return safe

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for record in output_rows:
            safe_record = json_safe_record(record)
            f.write(
                json.dumps(
                    safe_record,
                    ensure_ascii=False,
                    allow_nan=False,
                    default=str,
                )
            )
            f.write("\n")

    schema = {
        "phase": "5.8",
        "name": "ground_truth_experiment_logger",
        "status": "BUILT",
        "scope": "Virtual EV CAN laboratory",
        "purpose": (
            "Reproducible laboratory ground-truth and provenance layer linking "
            "Phase 5.5 cyber-physical features, aggregated Phase 5.6 safety "
            "requirements, and Phase 5.7 frozen Transformer predictions."
        ),
        "ground_truth_semantics": {
            "label_source": "DETERMINISTIC_LABORATORY_RULES",
            "real_world_cyberattack_labels": False,
            "synthetic_cyber_labels": False,
            "manual_labels": False,
            "physical_vehicle": False,
            "direct_vehicle_actuation": False,
        },
        "aggregation": {
            "phase_5_5_rows": len(feature_rows),
            "phase_5_6_raw_evaluations": len(req_rows),
            "phase_5_6_evaluations_per_timestep": EXPECTED_REQUIREMENTS_PER_STEP,
            "phase_5_6_timestep_summaries": len(requirement_summary),
            "phase_5_7_prediction_rows": len(pred_rows),
            "phase_5_7_prediction_rows_joined": prediction_join_count,
        },
        "ground_truth_classes": [
            "PHYSICAL_SAFETY_EVENT",
            "INTEGRITY_ANOMALY",
            "SENSOR_CONSISTENCY_ANOMALY",
            "BATTERY_SAFETY_ANOMALY",
            "HV_SAFETY_ANOMALY",
            "DIAGNOSTIC_ANOMALY",
            "BRAKE_CONSISTENCY_ANOMALY",
            "NORMAL_LABORATORY_STATE",
        ],
        "provenance": {
            "hash_algorithm": "SHA-256",
            "chain_genesis": GENESIS,
            "record_hash_chain": True,
        },
        "source_files": {
            "phase_5_5_features": str(INPUT_FEATURES.relative_to(ROOT)),
            "phase_5_6_requirements": str(INPUT_REQUIREMENTS.relative_to(ROOT)),
            "phase_5_7_predictions": (
                str(INPUT_PREDICTIONS.relative_to(ROOT))
                if INPUT_PREDICTIONS.exists()
                else None
            ),
        },
        "output_files": [
            str(OUT_CSV.relative_to(ROOT)),
            str(OUT_JSONL.relative_to(ROOT)),
            str(OUT_SCHEMA.relative_to(ROOT)),
        ],
    }

    OUT_SCHEMA.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    checks = {
        "phase_5_5_input_exists": INPUT_FEATURES.exists(),
        "phase_5_6_input_exists": INPUT_REQUIREMENTS.exists(),
        "phase_5_5_row_count_600": len(feature_rows) == 600,
        "phase_5_6_evaluation_count_24000": len(req_rows) == 24000,
        "phase_5_6_unique_timesteps_600": len(requirement_summary) == 600,
        "phase_5_6_exactly_40_requirements_per_timestep": (
            req_meta["min_requirements_per_step"] == 40
            and req_meta["max_requirements_per_step"] == 40
        ),
        "ground_truth_rows_600": len(output_rows) == 600,
        "experiment_ids_unique": len(
            {r["experiment_id"] for r in output_rows}
        ) == 600,
        "record_hashes_unique": len(
            {r["record_hash"] for r in output_rows}
        ) == 600,
        "first_record_anchored": (
            output_rows[0]["previous_record_hash"] == GENESIS
        ),
        "hash_chain_continuous": all(
            output_rows[i]["previous_record_hash"] == output_rows[i - 1]["record_hash"]
            for i in range(1, len(output_rows))
        ),
        "requirements_joined_600": requirement_join_count == 600,
        "predictions_joined_589_or_absent": (
            prediction_join_count in {0, 589}
        ),
        "laboratory_only": all(r["laboratory_generated"] for r in output_rows),
        "no_physical_vehicle": all(not r["physical_vehicle"] for r in output_rows),
        "no_direct_actuation": all(
            not r["direct_vehicle_actuation"] for r in output_rows
        ),
        "no_real_world_cyberattack_labels": all(
            not r["real_world_cyberattack_label"] for r in output_rows
        ),
        "no_synthetic_cyber_labels": all(
            not r["synthetic_cyber_label"] for r in output_rows
        ),
        "no_model_retraining": all(not r["model_retraining"] for r in output_rows),
        "no_live_cortex_xdr_api": all(
            not r["live_cortex_xdr_api"] for r in output_rows
        ),
        "csv_exists": OUT_CSV.exists(),
        "jsonl_exists": OUT_JSONL.exists(),
        "schema_exists": OUT_SCHEMA.exists(),
    }

    passed = sum(checks.values())
    failed = len(checks) - passed

    report = {
        "phase": "5.8",
        "status": "PASS" if failed == 0 else "FAIL",
        "summary": {
            "checks": len(checks),
            "pass": passed,
            "fail": failed,
        },
        "checks": checks,
        "input_counts": {
            "phase_5_5_rows": len(feature_rows),
            "phase_5_6_evaluations": len(req_rows),
            "phase_5_6_unique_steps": len(requirement_summary),
            "phase_5_7_prediction_rows": len(pred_rows),
            "phase_5_7_predictions_joined": prediction_join_count,
            "output_rows": len(output_rows),
        },
        "phase_5_6_aggregation": req_meta,
        "prediction_mapping": pred_meta,
        "output_sha256": {
            "csv": sha256_file(OUT_CSV),
            "jsonl": sha256_file(OUT_JSONL),
            "schema": sha256_file(OUT_SCHEMA),
        },
        "research_boundaries": {
            "laboratory_generated": True,
            "physical_vehicle_access": False,
            "direct_vehicle_actuation": False,
            "real_world_cyberattack_labels": False,
            "synthetic_cyber_labels": False,
            "live_cortex_xdr_api": False,
            "model_retraining": False,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    OUT_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== PHASE 5.8 GROUND-TRUTH EXPERIMENT LOGGER ===")
    print(f"Phase 5.5 feature rows: {len(feature_rows)}")
    print(f"Phase 5.6 raw requirement evaluations: {len(req_rows)}")
    print(f"Phase 5.6 timestep summaries: {len(requirement_summary)}")
    print(
        "Phase 5.6 evaluations/timestep: "
        f"{req_meta['min_requirements_per_step']}.."
        f"{req_meta['max_requirements_per_step']}"
    )
    print(f"Phase 5.7 prediction rows: {len(pred_rows)}")
    print(f"Phase 5.7 predictions joined: {prediction_join_count}")
    print(f"Ground-truth output rows: {len(output_rows)}")
    print(f"Hash chain genesis: {GENESIS}")
    print(f"Validation checks: {passed}/{len(checks)} PASS")
    print(f"CSV: {OUT_CSV}")
    print(f"JSONL: {OUT_JSONL}")
    print(f"Schema: {OUT_SCHEMA}")
    print(f"Report: {OUT_REPORT}")
    print(f"STATUS: {'PASS' if failed == 0 else 'FAIL'}")

    if failed:
        print("\nFAILED CHECKS:")
        for name, ok in checks.items():
            if not ok:
                print(f"  - {name}")


if __name__ == "__main__":
    build()


