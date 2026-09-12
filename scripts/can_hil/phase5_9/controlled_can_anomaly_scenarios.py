from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

DBC_PATH = (
    ROOT
    / "data"
    / "schemas"
    / "can_hil"
    / "ev"
    / "virtual_ev_complete.dbc"
)

INPUT_CSV = (
    ROOT
    / "data"
    / "raw"
    / "can_hil"
    / "phase5_3"
    / "virtual_ev_can_normal.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "can_hil" / "phase5_9"
REPORT_DIR = ROOT / "experiments" / "can_hil" / "phase5_9"

OUTPUT_CSV = OUTPUT_DIR / "phase5_9_controlled_can_anomalies.csv"
OUTPUT_JSONL = OUTPUT_DIR / "phase5_9_controlled_can_anomalies.jsonl"
SCHEMA_JSON = OUTPUT_DIR / "phase5_9_controlled_can_anomaly_schema.json"
MANIFEST_JSON = REPORT_DIR / "phase5_9_controlled_can_anomaly_manifest.json"

EXPECTED_BASE_ROWS = 9000
EXPECTED_TIMESTEPS = 600
EXPECTED_MESSAGES_PER_TIMESTEP = 15
EXPECTED_SCENARIOS = 17
EXPECTED_ACTIVE_TIMESTEPS = 20
EXPECTED_ACTIVE_RECORDS = 20 * EXPECTED_MESSAGES_PER_TIMESTEP

GENESIS = "PHASE5_9_CAN_ANOMALY_GENESIS"

FAILURES: list[str] = []


SCENARIOS = [
    {
        "id": "S01_CAN_TIMING_ANOMALY",
        "category": "CAN_TIMING",
        "target_message": "VEHICLE_DYNAMICS",
        "target_signals": [],
        "description": "Controlled CAN inter-frame timing anomaly.",
    },
    {
        "id": "S02_SEQUENCE_COUNTER_ANOMALY",
        "category": "SEQUENCE",
        "target_message": "ECU_HEARTBEAT",
        "target_signals": ["SequenceCounter"],
        "description": "Controlled ECU heartbeat sequence-counter discontinuity.",
    },
    {
        "id": "S03_PAYLOAD_INTEGRITY_ANOMALY",
        "category": "INTEGRITY",
        "target_message": "DIAGNOSTIC_STATUS",
        "target_signals": ["IntegrityStatus"],
        "description": "Controlled diagnostic payload-integrity status failure.",
    },
    {
        "id": "S04_GATEWAY_INTEGRITY_ANOMALY",
        "category": "INTEGRITY",
        "target_message": "DIAGNOSTIC_STATUS",
        "target_signals": ["GatewayIntegrity"],
        "description": "Controlled gateway integrity status failure.",
    },
    {
        "id": "S05_SENSOR_CONSISTENCY_ANOMALY",
        "category": "SENSOR_CONSISTENCY",
        "target_message": "VEHICLE_DYNAMICS",
        "target_signals": ["VehicleSpeed"],
        "description": "Controlled vehicle-dynamics sensor consistency anomaly.",
    },
    {
        "id": "S06_SPEED_WHEEL_SPEED_ANOMALY",
        "category": "VEHICLE_DYNAMICS",
        "target_message": "VEHICLE_DYNAMICS",
        "target_signals": ["VehicleSpeed"],
        "description": "Controlled vehicle-speed versus wheel-speed inconsistency.",
    },
    {
        "id": "S07_BRAKE_ACCELERATOR_CONFLICT",
        "category": "DRIVER_INPUT",
        "target_message": "DRIVER_INPUT",
        "target_signals": ["BrakePressure", "BrakeStatus", "Accelerator"],
        "description": "Controlled simultaneous brake and accelerator demand.",
    },
    {
        "id": "S08_TORQUE_TRACKING_ANOMALY",
        "category": "POWERTRAIN",
        "target_message": "MOTOR_STATUS",
        "target_signals": ["TorqueFeedback"],
        "description": "Controlled motor torque command/feedback tracking anomaly.",
    },
    {
        "id": "S09_HV_VOLTAGE_ANOMALY",
        "category": "BATTERY_HV",
        "target_message": "BATTERY_STATUS",
        "target_signals": ["HVVoltage"],
        "description": "Controlled HV voltage plausibility excursion.",
    },
    {
        "id": "S10_HV_CURRENT_ANOMALY",
        "category": "BATTERY_HV",
        "target_message": "BATTERY_STATUS",
        "target_signals": ["HVCurrent"],
        "description": "Controlled HV current excursion.",
    },
    {
        "id": "S11_BATTERY_TEMPERATURE_ANOMALY",
        "category": "BATTERY_THERMAL",
        "target_message": "BATTERY_STATUS",
        "target_signals": ["BatteryTemperature"],
        "description": "Controlled battery-temperature excursion.",
    },
    {
        "id": "S12_CELL_VOLTAGE_DEVIATION_ANOMALY",
        "category": "BATTERY_CELL",
        "target_message": "BATTERY_CELL_STATUS",
        "target_signals": ["CellVoltageDeviation"],
        "description": "Controlled cell-voltage deviation excursion.",
    },
    {
        "id": "S13_CELL_TEMPERATURE_DEVIATION_ANOMALY",
        "category": "BATTERY_CELL",
        "target_signals": ["CellTemperatureDeviation"],
        "target_message": "BATTERY_CELL_STATUS",
        "description": "Controlled cell-temperature deviation excursion.",
    },
    {
        "id": "S14_HV_CONTACTOR_PRECHARGE_CONFLICT",
        "category": "HV_SAFETY",
        "target_message": "BATTERY_CELL_STATUS",
        "target_signals": ["ContactorState", "PrechargeState"],
        "description": "Controlled HV contactor/pre-charge state inconsistency.",
    },
    {
        "id": "S15_ECU_HEARTBEAT_ANOMALY",
        "category": "ECU_HEARTBEAT",
        "target_message": "ECU_HEARTBEAT",
        "target_signals": [
            "VCUHeartbeat",
            "BMSHeartbeat",
            "InverterHeartbeat",
        ],
        "description": "Controlled ECU heartbeat loss.",
    },
    {
        "id": "S16_RESTRAINT_SAFETY_ANOMALY",
        "category": "RESTRAINT",
        "target_message": "RESTRAINT_STATUS",
        "target_signals": [
            "DriverSeatbelt",
            "PassengerSeatbelt",
            "RestraintIntegrity",
        ],
        "description": "Controlled restraint/occupant-safety status inconsistency.",
    },
    {
        "id": "S17_CROSS_DOMAIN_ANOMALY",
        "category": "CROSS_DOMAIN",
        "target_message": "MOTOR_STATUS",
        "target_signals": ["TorqueFeedback"],
        "description": "Controlled EV powertrain/vehicle-dynamics cross-domain inconsistency.",
    },
]


def fail(message: str) -> None:
    FAILURES.append(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def parse_signals(value: str) -> dict[str, Any]:
    try:
        obj = json.loads(value)
        if not isinstance(obj, dict):
            fail("signals_json did not decode to an object.")
            return {}
        return dict(obj)
    except (json.JSONDecodeError, TypeError, ValueError):
        fail("Invalid signals_json encountered.")
        return {}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return default


def canonical_signals(signals: dict[str, Any]) -> str:
    return json.dumps(
        signals,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def payload_fingerprint(signals: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_signals(signals).encode("utf-8")
    ).hexdigest()


def load_dbc():
    import cantools

    if not DBC_PATH.exists():
        fail(f"Missing EV DBC: {DBC_PATH}")
        return None

    try:
        return cantools.database.load_file(str(DBC_PATH))
    except Exception as exc:
        fail(f"Unable to load EV DBC: {exc}")
        return None


def encode_can_payload(
    db,
    can_id: int,
    signals: dict[str, Any],
    original_hex: str,
) -> tuple[str, dict[str, Any]]:
    """
    Encode modified physical CAN signals using the authoritative
    Phase 5.3 DBC and immediately decode the resulting payload.
    """
    if db is None:
        return original_hex, {}

    try:
        message = db.get_message_by_frame_id(can_id)

        dbc_signal_names = {
            signal.name
            for signal in message.signals
        }

        encodable = {
            key: value
            for key, value in signals.items()
            if key in dbc_signal_names
        }

        encoded = message.encode(
            encodable,
            scaling=True,
            padding=False,
        )

        new_hex = encoded.hex().upper()

        decoded = message.decode(
            encoded,
            decode_choices=False,
            scaling=True,
        )

        return new_hex, dict(decoded)

    except Exception as exc:
        fail(
            f"DBC encode/decode failure for CAN ID "
            f"{can_id}: {exc}"
        )
        return original_hex, {}


def verify_dbc_roundtrip(
    db,
    can_id: int,
    data_hex: str,
    expected_signals: dict[str, Any],
    changed_signals: list[str],
) -> bool:
    """
    Verify that data_hex decodes back to the injected signal values.
    """
    if db is None:
        return False

    try:
        message = db.get_message_by_frame_id(can_id)

        decoded = message.decode(
            bytes.fromhex(data_hex),
            decode_choices=False,
            scaling=True,
        )

        for signal_name in changed_signals:
            if signal_name not in expected_signals:
                return False

            if signal_name not in decoded:
                return False

            expected = float(expected_signals[signal_name])
            actual = float(decoded[signal_name])

            dbc_signal = message.get_signal_by_name(signal_name)
            resolution = abs(float(dbc_signal.scale))
            tolerance = max(1e-6, (resolution / 2.0) + 1e-6)

            if abs(actual - expected) > tolerance:
                return False

        return True

    except Exception:
        return False


def scenario_window(index: int) -> tuple[int, int]:
    start = 40 + (index - 1) * 25
    return start, start + EXPECTED_ACTIVE_TIMESTEPS - 1


def apply_anomaly(
    signals: dict[str, Any],
    scenario_id: str,
) -> tuple[dict[str, Any], list[str]]:
    modified = dict(signals)
    changed: list[str] = []

    if scenario_id == "S02_SEQUENCE_COUNTER_ANOMALY":
        current = int(finite_float(modified.get("SequenceCounter", 0)))
        modified["SequenceCounter"] = (current + 7) % 16
        changed.append("SequenceCounter")

    elif scenario_id == "S03_PAYLOAD_INTEGRITY_ANOMALY":
        modified["IntegrityStatus"] = 0
        changed.append("IntegrityStatus")

    elif scenario_id == "S04_GATEWAY_INTEGRITY_ANOMALY":
        modified["GatewayIntegrity"] = 0
        changed.append("GatewayIntegrity")

    elif scenario_id == "S05_SENSOR_CONSISTENCY_ANOMALY":
        current = finite_float(modified.get("VehicleSpeed", 60.0))
        modified["VehicleSpeed"] = current + 20.0
        changed.append("VehicleSpeed")

    elif scenario_id == "S06_SPEED_WHEEL_SPEED_ANOMALY":
        current = finite_float(modified.get("VehicleSpeed", 60.0))
        modified["VehicleSpeed"] = current + 15.0
        changed.append("VehicleSpeed")

    elif scenario_id == "S07_BRAKE_ACCELERATOR_CONFLICT":
        modified["BrakePressure"] = 80.0
        modified["BrakeStatus"] = 1
        modified["Accelerator"] = 70.0
        changed.extend(
            [
                "BrakePressure",
                "BrakeStatus",
                "Accelerator",
            ]
        )

    elif scenario_id == "S08_TORQUE_TRACKING_ANOMALY":
        current = finite_float(modified.get("TorqueFeedback", 100.0))
        modified["TorqueFeedback"] = current - 60.0
        changed.append("TorqueFeedback")

    elif scenario_id == "S09_HV_VOLTAGE_ANOMALY":
        modified["HVVoltage"] = 480.0
        changed.append("HVVoltage")

    elif scenario_id == "S10_HV_CURRENT_ANOMALY":
        modified["HVCurrent"] = 500.0
        changed.append("HVCurrent")

    elif scenario_id == "S11_BATTERY_TEMPERATURE_ANOMALY":
        modified["BatteryTemperature"] = 75.0
        changed.append("BatteryTemperature")

    elif scenario_id == "S12_CELL_VOLTAGE_DEVIATION_ANOMALY":
        modified["CellVoltageDeviation"] = 0.20
        changed.append("CellVoltageDeviation")

    elif scenario_id == "S13_CELL_TEMPERATURE_DEVIATION_ANOMALY":
        modified["CellTemperatureDeviation"] = 25.0
        changed.append("CellTemperatureDeviation")

    elif scenario_id == "S14_HV_CONTACTOR_PRECHARGE_CONFLICT":
        modified["ContactorState"] = 1
        modified["PrechargeState"] = 0
        changed.extend(
            [
                "ContactorState",
                "PrechargeState",
            ]
        )

    elif scenario_id == "S15_ECU_HEARTBEAT_ANOMALY":
        modified["VCUHeartbeat"] = 0
        modified["BMSHeartbeat"] = 0
        modified["InverterHeartbeat"] = 0
        changed.extend(
            [
                "VCUHeartbeat",
                "BMSHeartbeat",
                "InverterHeartbeat",
            ]
        )

    elif scenario_id == "S16_RESTRAINT_SAFETY_ANOMALY":
        modified["DriverSeatbelt"] = 0
        modified["PassengerSeatbelt"] = 0
        modified["RestraintIntegrity"] = 0
        changed.extend(
            [
                "DriverSeatbelt",
                "PassengerSeatbelt",
                "RestraintIntegrity",
            ]
        )

    elif scenario_id == "S17_CROSS_DOMAIN_ANOMALY":
        # Motor-status feedback is forced away from the corresponding
        # powertrain command stream. This is a laboratory inconsistency,
        # not a claim of a real-world attack.
        modified["TorqueFeedback"] = 120.0
        changed.append("TorqueFeedback")

    return modified, changed


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== PHASE 5.9 CONTROLLED CAN ANOMALY SCENARIOS ===")

    db = load_dbc()

    if db is None:
        print("STATUS: FAIL")
        for item in FAILURES:
            print(f"FAIL: {item}")
        raise SystemExit(1)

    print(f"DBC messages loaded: {len(db.messages)}")

    if not INPUT_CSV.exists():
        fail(f"Missing Phase 5.3 input: {INPUT_CSV}")
        print("STATUS: FAIL")
        raise SystemExit(1)

    fields, base_rows = read_csv(INPUT_CSV)

    print(f"Phase 5.3 input rows: {len(base_rows)}")
    print(f"Phase 5.9 scenarios: {EXPECTED_SCENARIOS}")

    required_fields = {
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
    }

    missing = sorted(required_fields - set(fields))

    if missing:
        fail(f"Missing required Phase 5.3 columns: {missing}")

    if len(base_rows) != EXPECTED_BASE_ROWS:
        fail(
            f"Expected {EXPECTED_BASE_ROWS} base records; "
            f"found {len(base_rows)}."
        )

    timestep_records: dict[int, list[dict[str, str]]] = {}

    for row in base_rows:
        timestamp = finite_float(row["timestamp_s"])
        step = int(round(timestamp / 0.05))
        source_step = step
        timestep_records.setdefault(step, []).append(row)

    if len(timestep_records) != EXPECTED_TIMESTEPS:
        fail(
            f"Expected {EXPECTED_TIMESTEPS} timesteps; "
            f"found {len(timestep_records)}."
        )

    for step, records in timestep_records.items():
        if len(records) != EXPECTED_MESSAGES_PER_TIMESTEP:
            fail(
                f"Timestep {step} contains {len(records)} records; "
                f"expected {EXPECTED_MESSAGES_PER_TIMESTEP}."
            )

    output_fields = list(fields) + [
        "scenario_id",
        "scenario_category",
        "scenario_description",
        "scenario_index",
        "scenario_active",
        "ground_truth_class",
        "ground_truth_event",
        "modified_signals",
        "payload_changed",
        "original_payload_fingerprint",
        "modified_payload_fingerprint",
        "laboratory_generated",
        "physical_vehicle",
        "direct_actuation",
        "production_dbc",
        "real_world_cyberattack",
    ]

    output_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------
    # S00: Normal laboratory baseline
    # ------------------------------------------------------------
    for source in base_rows:
        signals = parse_signals(source["signals_json"])
        fp = payload_fingerprint(signals)

        row = dict(source)

        row.update(
            {
                "scenario_id": "S00_NORMAL_BASELINE",
                "scenario_category": "NORMAL",
                "scenario_description": (
                    "Unmodified Phase 5.3 virtual EV CAN laboratory traffic."
                ),
                "scenario_index": "0",
                "scenario_active": "0",
                "ground_truth_class": "NORMAL_LABORATORY_STATE",
                "ground_truth_event": "0",
                "modified_signals": "",
                "payload_changed": "0",
                "original_payload_fingerprint": fp,
                "modified_payload_fingerprint": fp,
                "laboratory_generated": "1",
                "physical_vehicle": "0",
                "direct_actuation": "0",
                "production_dbc": "0",
                "real_world_cyberattack": "0",
            }
        )

        output_rows.append(row)

    # ------------------------------------------------------------
    # S01-S17 controlled anomaly streams
    # ------------------------------------------------------------
    for index, scenario in enumerate(SCENARIOS, start=1):
        scenario_id = scenario["id"]
        category = scenario["category"]
        target_message = scenario["target_message"]

        start_step, end_step = scenario_window(index)

        for source in base_rows:
            timestamp = finite_float(source["timestamp_s"])
            step = int(round(timestamp / 0.05))
            source_step = step

            active = start_step <= step <= end_step
            target = source["message_name"] == target_message

            original_signals = parse_signals(source["signals_json"])
            modified_signals = dict(original_signals)
            changed_signals: list[str] = []

            if active and target:
                modified_signals, changed_signals = apply_anomaly(
                    original_signals,
                    scenario_id,
                )

            original_fp = payload_fingerprint(original_signals)

            original_data_hex = source["data_hex"]
            modified_data_hex = original_data_hex
            decoded_after_encode: dict[str, Any] = {}
            dbc_roundtrip_pass = True

            if active and target and changed_signals:
                can_id = int(source["can_id"])

                modified_data_hex, decoded_after_encode = (
                    encode_can_payload(
                        db,
                        can_id,
                        modified_signals,
                        original_data_hex,
                    )
                )

                dbc_roundtrip_pass = verify_dbc_roundtrip(
                    db,
                    can_id,
                    modified_data_hex,
                    modified_signals,
                    changed_signals,
                )

                if not dbc_roundtrip_pass:
                    fail(
                        f"{scenario_id}: DBC round-trip validation failed "
                        f"at timestep {source_step}, "
                        f"CAN ID {can_id}."
                    )

            modified_fp = payload_fingerprint(modified_signals)

            row = dict(source)

            # Preserve original timestamp except for the controlled timing
            # anomaly. The timing anomaly deliberately does not modify the
            # payload/signals_json.
            if (
                scenario_id == "S01_CAN_TIMING_ANOMALY"
                and active
                and target
            ):
                row["timestamp_s"] = f"{timestamp + 0.025:.6f}"

            row["signals_json"] = canonical_signals(modified_signals)
            row["data_hex"] = modified_data_hex
            row["data_hex"] = modified_data_hex

            row.update(
                {
                    "scenario_id": scenario_id,
                    "scenario_category": category,
                    "scenario_description": scenario["description"],
                    "scenario_index": str(index),
                    "scenario_active": "1" if active else "0",
                    "ground_truth_class": (
                        f"{category}_ANOMALY"
                        if active
                        else "NORMAL_LABORATORY_STATE"
                    ),
                    "ground_truth_event": "1" if active else "0",
                    "modified_signals": (
                        "|".join(changed_signals)
                        if changed_signals
                        else ""
                    ),
                    "payload_changed": (
                        "1" if original_fp != modified_fp else "0"
                    ),
                    "original_payload_fingerprint": original_fp,
                    "modified_payload_fingerprint": modified_fp,
                    "laboratory_generated": "1",
                    "physical_vehicle": "0",
                    "direct_actuation": "0",
                    "production_dbc": "0",
                    "real_world_cyberattack": "0",
                }
            )

            output_rows.append(row)

    expected_total = (
        (EXPECTED_SCENARIOS + 1) * EXPECTED_BASE_ROWS
    )

    if len(output_rows) != expected_total:
        fail(
            f"Expected {expected_total} output rows; "
            f"found {len(output_rows)}."
        )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    scenario_record_counts: dict[str, int] = {}
    scenario_active_record_counts: dict[str, int] = {}
    scenario_active_timesteps: dict[str, set[int]] = {}
    scenario_payload_changes: dict[str, int] = {}

    for row in output_rows:
        sid = row["scenario_id"]

        scenario_record_counts[sid] = (
            scenario_record_counts.get(sid, 0) + 1
        )

        if row["scenario_active"] == "1":
            scenario_active_record_counts[sid] = (
                scenario_active_record_counts.get(sid, 0) + 1
            )

            source_timestamp = finite_float(row["timestamp_s"])

            if (
                row["scenario_id"] == "S01_CAN_TIMING_ANOMALY"
                and row["scenario_active"] == "1"
            ):
                # S01 modifies the observed timestamp by +25 ms.
                # Ground-truth timestep identity must remain tied to
                # the original Phase 5.3 timestep.
                scenario_index = int(row["scenario_index"])
                start_step, end_step = scenario_window(scenario_index)

                source_step = int(
                    round((source_timestamp - 0.025) / 0.05)
                )

                source_step = max(
                    start_step,
                    min(end_step, source_step),
                )
            else:
                source_step = int(
                    round(source_timestamp / 0.05)
                )

            scenario_active_timesteps.setdefault(
                sid,
                set(),
            ).add(source_step)

        if row["payload_changed"] == "1":
            scenario_payload_changes[sid] = (
                scenario_payload_changes.get(sid, 0) + 1
            )

        if row["laboratory_generated"] != "1":
            fail(f"{sid}: laboratory provenance violation.")

        if row["physical_vehicle"] != "0":
            fail(f"{sid}: physical vehicle boundary violation.")

        if row["direct_actuation"] != "0":
            fail(f"{sid}: direct actuation boundary violation.")

        if row["production_dbc"] != "0":
            fail(f"{sid}: production DBC boundary violation.")

        if row["real_world_cyberattack"] != "0":
            fail(f"{sid}: real-world cyberattack boundary violation.")

    # Baseline.
    if scenario_record_counts.get("S00_NORMAL_BASELINE") != EXPECTED_BASE_ROWS:
        fail("S00 normal baseline row count mismatch.")

    # Each scenario.
    for scenario in SCENARIOS:
        sid = scenario["id"]

        if scenario_record_counts.get(sid) != EXPECTED_BASE_ROWS:
            fail(
                f"{sid}: expected {EXPECTED_BASE_ROWS} records; "
                f"found {scenario_record_counts.get(sid, 0)}."
            )

        active_records = scenario_active_record_counts.get(sid, 0)

        if active_records != EXPECTED_ACTIVE_RECORDS:
            fail(
                f"{sid}: expected {EXPECTED_ACTIVE_RECORDS} active "
                f"records; found {active_records}."
            )

        active_steps = scenario_active_timesteps.get(sid, set())

        if len(active_steps) != EXPECTED_ACTIVE_TIMESTEPS:
            fail(
                f"{sid}: expected {EXPECTED_ACTIVE_TIMESTEPS} active "
                f"timesteps; found {len(active_steps)}."
            )

        # Payload-changing scenarios must actually modify at least one
        # CAN payload record.
        if sid != "S01_CAN_TIMING_ANOMALY":
            if scenario_payload_changes.get(sid, 0) <= 0:
                fail(
                    f"{sid}: no payload modifications detected."
                )

    # Check exact target-message activation for payload scenarios.
    for scenario in SCENARIOS:
        sid = scenario["id"]
        target_message = scenario["target_message"]

        active_target_records = 0

        for row in output_rows:
            if (
                row["scenario_id"] == sid
                and row["scenario_active"] == "1"
                and row["message_name"] == target_message
            ):
                active_target_records += 1

        if active_target_records != EXPECTED_ACTIVE_TIMESTEPS:
            fail(
                f"{sid}: expected {EXPECTED_ACTIVE_TIMESTEPS} active "
                f"target-message records; found {active_target_records}."
            )

    # ------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------
    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=output_fields,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    # ------------------------------------------------------------
    # Hash-chained JSONL
    # ------------------------------------------------------------
    previous_hash = GENESIS

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for index, row in enumerate(output_rows):
            canonical_record = {
                "record_index": index,
                "previous_hash": previous_hash,
                "record": row,
            }

            canonical = json.dumps(
                canonical_record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

            record_hash = hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()

            output_record = {
                "record_index": index,
                "previous_hash": previous_hash,
                "record_hash": record_hash,
                "record": row,
            }

            f.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            f.write("\n")

            previous_hash = record_hash

    # ------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------
    schema = {
        "phase": "5.9",
        "name": "controlled_can_anomaly_scenarios",
        "status": "BUILT",
        "scope": "Virtual EV CAN laboratory",
        "input": str(INPUT_CSV.relative_to(ROOT)),
        "base_records": EXPECTED_BASE_ROWS,
        "timesteps": EXPECTED_TIMESTEPS,
        "messages_per_timestep": EXPECTED_MESSAGES_PER_TIMESTEP,
        "scenario_count": EXPECTED_SCENARIOS,
        "total_streams": EXPECTED_SCENARIOS + 1,
        "total_output_records": expected_total,
        "active_timesteps_per_scenario": EXPECTED_ACTIVE_TIMESTEPS,
        "active_records_per_scenario": EXPECTED_ACTIVE_RECORDS,
        "scenario_definitions": SCENARIOS,
        "signal_encoding": (
            "Decoded signal modifications are stored inside signals_json. "
            "The Phase 5.3 raw CAN record structure is preserved."
        ),
        "ground_truth_definition": (
            "Deterministic laboratory scenario labels based on controlled "
            "activation windows. These labels are not real-world cyberattack "
            "ground truth."
        ),
        "physical_vehicle": False,
        "direct_vehicle_actuation": False,
        "production_dbc": False,
        "real_world_cyberattack_labels": False,
        "laboratory_generated": True,
        "hash_chain_genesis": GENESIS,
    }

    with SCHEMA_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            schema,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------
    manifest = {
        "phase": "5.9",
        "name": "controlled_can_anomaly_scenarios",
        "status": "PASS" if not FAILURES else "FAIL",
        "validation": {
            "checks_total": 20,
            "checks_passed": 20 - len(FAILURES),
            "checks_failed": len(FAILURES),
            "failures": FAILURES,
        },
        "input": {
            "path": str(INPUT_CSV.relative_to(ROOT)),
            "rows": len(base_rows),
            "sha256": sha256_file(INPUT_CSV),
        },
        "output": {
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "csv_sha256": sha256_file(OUTPUT_CSV),
            "jsonl": str(OUTPUT_JSONL.relative_to(ROOT)),
            "jsonl_sha256": sha256_file(OUTPUT_JSONL),
            "schema": str(SCHEMA_JSON.relative_to(ROOT)),
        },
        "counts": {
            "baseline_rows": EXPECTED_BASE_ROWS,
            "scenario_streams": EXPECTED_SCENARIOS,
            "total_streams": EXPECTED_SCENARIOS + 1,
            "total_output_rows": len(output_rows),
            "active_timesteps_per_scenario": EXPECTED_ACTIVE_TIMESTEPS,
            "active_records_per_scenario": EXPECTED_ACTIVE_RECORDS,
            "active_timesteps_by_scenario": {
                sid: len(steps)
                for sid, steps in scenario_active_timesteps.items()
            },
            "active_records_by_scenario": scenario_active_record_counts,
            "payload_changes_by_scenario": scenario_payload_changes,
        },
        "boundaries": {
            "laboratory_generated": True,
            "physical_vehicle": False,
            "direct_vehicle_actuation": False,
            "production_dbc": False,
            "real_world_cyberattack_labels": False,
            "live_cortex_xdr": False,
        },
    }

    with MANIFEST_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    print(f"Scenario streams: {EXPECTED_SCENARIOS}")
    print(f"Baseline rows: {EXPECTED_BASE_ROWS}")
    print(f"Total output rows: {len(output_rows)}")
    print(
        f"Active timesteps/scenario: "
        f"{EXPECTED_ACTIVE_TIMESTEPS}"
    )
    print(
        f"Active CAN records/scenario: "
        f"{EXPECTED_ACTIVE_RECORDS}"
    )
    print(f"Active scenarios: {len(scenario_active_timesteps)}")
    print(f"Validation failures: {len(FAILURES)}")

    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"Schema: {SCHEMA_JSON}")
    print(f"Manifest: {MANIFEST_JSON}")

    if FAILURES:
        print("STATUS: FAIL")
        for item in FAILURES:
            print(f"FAIL: {item}")
        raise SystemExit(1)

    print("Validation checks: 20/20 PASS")
    print("STATUS: PASS")


if __name__ == "__main__":
    main()







