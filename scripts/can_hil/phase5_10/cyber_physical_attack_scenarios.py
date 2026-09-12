from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

DBC_PATH = (
    ROOT / "data" / "schemas" / "can_hil" / "ev" / "virtual_ev_complete.dbc"
)

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_9"
    / "phase5_9_controlled_can_anomalies.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "can_hil" / "phase5_10"
REPORT_DIR = ROOT / "experiments" / "can_hil" / "phase5_10"

OUTPUT_CSV = (
    OUTPUT_DIR / "phase5_10_cyber_physical_attack_scenarios.csv"
)
OUTPUT_JSONL = (
    OUTPUT_DIR / "phase5_10_cyber_physical_attack_scenarios.jsonl"
)
SCHEMA_JSON = (
    OUTPUT_DIR / "phase5_10_cyber_physical_attack_schema.json"
)
MANIFEST_JSON = (
    REPORT_DIR / "phase5_10_cyber_physical_attack_manifest.json"
)

EXPECTED_BASE_ROWS = 9000
EXPECTED_TIMESTEPS = 600
EXPECTED_MESSAGES = 15
EXPECTED_SCENARIOS = 10
EXPECTED_ACTIVE_TIMESTEPS = 20

FAILURES: list[str] = []


SCENARIOS = [
    {
        "id": "C01_SPEED_SENSOR_SPOOFING",
        "category": "SENSOR_SPOOFING",
        "description": "Controlled vehicle-speed sensor spoofing.",
        "target": {
            "VEHICLE_DYNAMICS": ["VehicleSpeed"]
        },
    },
    {
        "id": "C02_BRAKE_ACCELERATOR_ATTACK",
        "category": "DRIVER_INPUT",
        "description": "Controlled brake/accelerator conflict.",
        "target": {
            "DRIVER_INPUT": [
                "BrakePressure",
                "BrakeStatus",
                "Accelerator",
            ]
        },
    },
    {
        "id": "C03_TORQUE_DYNAMICS_ATTACK",
        "category": "POWERTRAIN",
        "description": "Controlled torque/vehicle-dynamics inconsistency.",
        "target": {
            "MOTOR_STATUS": ["TorqueFeedback"],
            "VEHICLE_DYNAMICS": ["VehicleSpeed"],
        },
    },
    {
        "id": "C04_HV_POWER_ATTACK",
        "category": "BATTERY_HV",
        "description": "Controlled high-voltage power anomaly.",
        "target": {
            "BATTERY_STATUS": ["HVVoltage", "HVCurrent"]
        },
    },
    {
        "id": "C05_BATTERY_THERMAL_ATTACK",
        "category": "BATTERY_THERMAL",
        "description": "Controlled battery thermal inconsistency.",
        "target": {
            "BATTERY_STATUS": ["BatteryTemperature"],
            "BATTERY_CELL_STATUS": ["CellTemperatureDeviation"],
        },
    },
    {
        "id": "C06_HV_CONTACTOR_ATTACK",
        "category": "HV_SAFETY",
        "description": "Controlled HV contactor/precharge/isolation inconsistency.",
        "target": {
            "BATTERY_CELL_STATUS": [
                "ContactorState",
                "PrechargeState",
                "IsolationStatus",
            ]
        },
    },
    {
        "id": "C07_HEARTBEAT_INTEGRITY_ATTACK",
        "category": "ECU_INTEGRITY",
        "description": "Controlled heartbeat and integrity failure.",
        "target": {
            "ECU_HEARTBEAT": [
                "VCUHeartbeat",
                "BMSHeartbeat",
                "InverterHeartbeat",
            ],
            "DIAGNOSTIC_STATUS": [
                "IntegrityStatus",
                "GatewayIntegrity",
            ],
        },
    },
    {
        "id": "C08_SEQUENCE_GATEWAY_ATTACK",
        "category": "GATEWAY_INTEGRITY",
        "description": "Controlled sequence-counter/gateway anomaly.",
        "target": {
            "ECU_HEARTBEAT": ["SequenceCounter"],
            "DIAGNOSTIC_STATUS": ["GatewayIntegrity"],
        },
    },
    {
        "id": "C09_RESTRAINT_DOMAIN_ATTACK",
        "category": "RESTRAINT",
        "description": "Controlled restraint-domain safety inconsistency.",
        "target": {
            "RESTRAINT_STATUS": [
                "DriverSeatbelt",
                "PassengerSeatbelt",
                "RestraintIntegrity",
            ]
        },
    },
    {
        "id": "C10_MULTI_DOMAIN_ATTACK",
        "category": "MULTI_DOMAIN",
        "description": "Controlled multi-domain cyber-physical inconsistency.",
        "target": {
            "VEHICLE_DYNAMICS": ["VehicleSpeed"],
            "MOTOR_STATUS": ["TorqueFeedback"],
            "BATTERY_STATUS": ["HVVoltage"],
            "DIAGNOSTIC_STATUS": ["IntegrityStatus"],
        },
    },
]


def fail(message: str) -> None:
    FAILURES.append(message)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return default


def parse_signals(value: Any) -> dict[str, Any]:
    try:
        obj = json.loads(value)
        if isinstance(obj, dict):
            return dict(obj)
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    fail("Invalid signals_json encountered.")
    return {}


def canonical_signals(signals: dict[str, Any]) -> str:
    return json.dumps(
        signals,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def payload_hash(signals: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_signals(signals).encode("utf-8")
    ).hexdigest()


def load_dbc():
    import cantools

    if not DBC_PATH.exists():
        fail(f"Missing DBC: {DBC_PATH}")
        return None

    try:
        return cantools.database.load_file(str(DBC_PATH))
    except Exception as exc:
        fail(f"DBC load failure: {exc}")
        return None


def encode_payload(db, can_id: int, signals: dict[str, Any]) -> tuple[str, dict]:
    message = db.get_message_by_frame_id(can_id)

    names = {s.name for s in message.signals}

    encodable = {
        k: v
        for k, v in signals.items()
        if k in names
    }

    encoded = message.encode(
        encodable,
        scaling=True,
        padding=False,
    )

    decoded = message.decode(
        encoded,
        decode_choices=False,
        scaling=True,
    )

    return encoded.hex().upper(), dict(decoded)


def verify_roundtrip(
    db,
    can_id: int,
    encoded_hex: str,
    expected: dict[str, Any],
    changed: list[str],
) -> bool:

    try:
        message = db.get_message_by_frame_id(can_id)
        decoded = message.decode(
            bytes.fromhex(encoded_hex),
            decode_choices=False,
            scaling=True,
        )

        for name in changed:
            if name not in decoded:
                continue

            expected_value = finite(expected[name])
            actual_value = finite(decoded[name])

            dbc_signal = message.get_signal_by_name(name)
            resolution = abs(float(dbc_signal.scale))

            tolerance = max(
                1e-6,
                resolution / 2.0 + 1e-6,
            )

            if abs(actual_value - expected_value) > tolerance:
                return False

        return True

    except Exception:
        return False


def scenario_window(index: int) -> tuple[int, int]:
    start = 140
    end = start + EXPECTED_ACTIVE_TIMESTEPS - 1
    return start, end


def apply_attack(
    signals: dict[str, Any],
    scenario_id: str,
) -> tuple[dict[str, Any], list[str]]:

    out = dict(signals)
    changed: list[str] = []

    def set_value(name: str, value: float) -> None:
        if name in out:
            out[name] = value
            changed.append(name)

    if scenario_id == "C01_SPEED_SENSOR_SPOOFING":
        if "VehicleSpeed" in out:
            set_value("VehicleSpeed", finite(out["VehicleSpeed"]) + 15.0)

    elif scenario_id == "C02_BRAKE_ACCELERATOR_ATTACK":
        set_value("Accelerator", 80.0)
        set_value("BrakePressure", 80.0)
        set_value("BrakeStatus", 1.0)

    elif scenario_id == "C03_TORQUE_DYNAMICS_ATTACK":
        set_value("TorqueFeedback", 0.0)
        if "VehicleSpeed" in out:
            set_value("VehicleSpeed", finite(out["VehicleSpeed"]) + 8.0)

    elif scenario_id == "C04_HV_POWER_ATTACK":
        if "HVVoltage" in out:
            set_value("HVVoltage", finite(out["HVVoltage"]) * 0.70)
        if "HVCurrent" in out:
            set_value("HVCurrent", finite(out["HVCurrent"]) * 1.50)

    elif scenario_id == "C05_BATTERY_THERMAL_ATTACK":
        if "BatteryTemperature" in out:
            set_value(
                "BatteryTemperature",
                finite(out["BatteryTemperature"]) + 35.0,
            )
        if "CellTemperatureDeviation" in out:
            set_value(
                "CellTemperatureDeviation",
                finite(out["CellTemperatureDeviation"]) + 10.0,
            )

    elif scenario_id == "C06_HV_CONTACTOR_ATTACK":
        set_value("ContactorState", 0.0)
        set_value("PrechargeState", 0.0)
        set_value("IsolationStatus", 0.0)

    elif scenario_id == "C07_HEARTBEAT_INTEGRITY_ATTACK":
        set_value("VCUHeartbeat", 0.0)
        set_value("BMSHeartbeat", 0.0)
        set_value("InverterHeartbeat", 0.0)
        set_value("IntegrityStatus", 0.0)
        set_value("GatewayIntegrity", 0.0)

    elif scenario_id == "C08_SEQUENCE_GATEWAY_ATTACK":
        if "SequenceCounter" in out:
            set_value(
                "SequenceCounter",
                finite(out["SequenceCounter"]) + 25.0,
            )
        set_value("GatewayIntegrity", 0.0)

    elif scenario_id == "C09_RESTRAINT_DOMAIN_ATTACK":
        set_value("DriverSeatbelt", 0.0)
        set_value("PassengerSeatbelt", 0.0)
        set_value("RestraintIntegrity", 0.0)

    elif scenario_id == "C10_MULTI_DOMAIN_ATTACK":
        if "VehicleSpeed" in out:
            set_value(
                "VehicleSpeed",
                finite(out["VehicleSpeed"]) + 12.0,
            )
        set_value("TorqueFeedback", 0.0)

        if "HVVoltage" in out:
            set_value(
                "HVVoltage",
                finite(out["HVVoltage"]) * 0.75,
            )

        set_value("IntegrityStatus", 0.0)

    return out, sorted(set(changed))


def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== PHASE 5.10 CYBER-PHYSICAL ATTACK SCENARIOS ===")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    db = load_dbc()

    if db is not None:
        print(f"DBC messages loaded: {len(db.messages)}")

    with INPUT_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        source_fields = list(reader.fieldnames or [])
        source_rows = list(reader)

    print(f"Phase 5.9 input rows: {len(source_rows)}")

    required = {
        "timestamp_s",
        "can_id",
        "message_name",
        "data_hex",
        "signals_json",
    }

    missing = sorted(required - set(source_fields))

    if missing:
        raise ValueError(
            f"Phase 5.9 input is missing required columns: {missing}"
        )

    baseline_rows = [
        r
        for r in source_rows
        if str(r.get("scenario_id", "")).strip()
        == "S00_NORMAL_BASELINE"
    ]

    if len(baseline_rows) != EXPECTED_BASE_ROWS:
        raise ValueError(
            f"Expected 9000 baseline rows; found {len(baseline_rows)}"
        )

    print(f"Cyber-physical scenarios: {len(SCENARIOS)}")
    print(f"Baseline records used: {len(baseline_rows)}")

    output_rows: list[dict[str, Any]] = []

    metadata_fields = [
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

    base_fields = [
        x for x in source_fields
        if x not in metadata_fields
    ]

    output_fields = base_fields + metadata_fields

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    for source in baseline_rows:

        signals = parse_signals(source["signals_json"])
        fp = payload_hash(signals)

        row = {
            key: source.get(key, "")
            for key in base_fields
        }

        row.update(
            {
                "scenario_id": "S00_NORMAL_BASELINE",
                "scenario_category": "NORMAL",
                "scenario_description": (
                    "Unmodified virtual EV CAN laboratory baseline."
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

    # ---------------------------------------------------------
    # C01-C10
    # ---------------------------------------------------------

    for index, scenario in enumerate(SCENARIOS, start=1):

        scenario_id = scenario["id"]
        category = scenario["category"]

        start_step, end_step = scenario_window(index)

        for source in baseline_rows:

            timestamp = finite(
                source["timestamp_s"]
            )

            step = int(round(timestamp / 0.05))

            active = start_step <= step <= end_step

            target_message = (
                source["message_name"]
                in scenario["target"]
            )

            original = parse_signals(
                source["signals_json"]
            )

            modified = dict(original)
            changed: list[str] = []

            if active and target_message:

                modified, changed = apply_attack(
                    original,
                    scenario_id,
                )

            original_fp = payload_hash(original)

            modified_hex = source["data_hex"]

            dbc_pass = True

            if (
                active
                and target_message
                and changed
                and db is not None
            ):

                modified_hex, decoded = encode_payload(
                    db,
                    int(source["can_id"]),
                    modified,
                )

                dbc_pass = verify_roundtrip(
                    db,
                    int(source["can_id"]),
                    modified_hex,
                    modified,
                    changed,
                )

                if not dbc_pass:
                    fail(
                        f"{scenario_id}: DBC round-trip failure "
                        f"at timestep {step}, "
                        f"CAN ID {source['can_id']}."
                    )

            modified_fp = payload_hash(modified)

            row = {
                key: source.get(key, "")
                for key in base_fields
            }

            row["signals_json"] = canonical_signals(modified)
            row["data_hex"] = modified_hex

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
                    "ground_truth_event": (
                        "1" if active else "0"
                    ),
                    "modified_signals": "|".join(changed),
                    "payload_changed": (
                        "1"
                        if original_fp != modified_fp
                        else "0"
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

    expected_rows = (
        (EXPECTED_SCENARIOS + 1)
        * EXPECTED_BASE_ROWS
    )

    if len(output_rows) != expected_rows:
        fail(
            f"Expected {expected_rows} output rows; "
            f"found {len(output_rows)}."
        )

    # ---------------------------------------------------------
    # In-memory validation
    # ---------------------------------------------------------

    counts: dict[str, int] = {}
    active_counts: dict[str, int] = {}

    for row in output_rows:

        sid = str(
            row["scenario_id"]
        ).strip()

        counts[sid] = counts.get(sid, 0) + 1

        if row["scenario_active"] == "1":
            active_counts[sid] = (
                active_counts.get(sid, 0) + 1
            )

        if row["laboratory_generated"] != "1":
            fail(f"{sid}: laboratory provenance violation.")

        if row["physical_vehicle"] != "0":
            fail(f"{sid}: physical vehicle violation.")

        if row["direct_actuation"] != "0":
            fail(f"{sid}: direct actuation violation.")

        if row["production_dbc"] != "0":
            fail(f"{sid}: production DBC violation.")

        if row["real_world_cyberattack"] != "0":
            fail(f"{sid}: real-world attack violation.")

    if counts.get(
        "S00_NORMAL_BASELINE"
    ) != EXPECTED_BASE_ROWS:
        fail("Baseline count mismatch.")

    for scenario in SCENARIOS:

        sid = scenario["id"]

        if counts.get(sid) != EXPECTED_BASE_ROWS:
            fail(
                f"{sid}: expected {EXPECTED_BASE_ROWS} "
                f"rows, found {counts.get(sid, 0)}."
            )

        expected_active = (
            EXPECTED_ACTIVE_TIMESTEPS
            * EXPECTED_MESSAGES
        )

        if active_counts.get(sid) != expected_active:
            fail(
                f"{sid}: expected {expected_active} "
                f"active records, found "
                f"{active_counts.get(sid, 0)}."
            )

    # ---------------------------------------------------------
    # Write CSV
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Write JSONL
    # ---------------------------------------------------------

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in output_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    # ---------------------------------------------------------
    # CRITICAL: re-read the CSV from disk
    # ---------------------------------------------------------

    with OUTPUT_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        check_reader = csv.DictReader(f)
        disk_rows = list(check_reader)

    disk_counts: dict[str, int] = {}

    for row in disk_rows:
        sid = str(
            row.get("scenario_id", "")
        ).strip()

        disk_counts[sid] = (
            disk_counts.get(sid, 0) + 1
        )

    if len(disk_rows) != expected_rows:
        fail(
            f"Disk CSV row count mismatch: "
            f"{len(disk_rows)} vs {expected_rows}."
        )

    for scenario in SCENARIOS:

        sid = scenario["id"]

        if disk_counts.get(sid) != EXPECTED_BASE_ROWS:
            fail(
                f"DISK CSV missing {sid}: "
                f"{disk_counts.get(sid, 0)} rows."
            )

    if disk_counts.get(
        "S00_NORMAL_BASELINE"
    ) != EXPECTED_BASE_ROWS:
        fail("DISK CSV baseline count mismatch.")

    # ---------------------------------------------------------
    # Schema
    # ---------------------------------------------------------

    schema = {
        "phase": "5.10",
        "title": "Cyber-Physical Attack Scenarios",
        "status": "PASS" if not FAILURES else "FAIL",
        "input": {
            "path": str(INPUT_CSV.relative_to(ROOT)),
            "rows": len(source_rows),
            "sha256": sha256_file(INPUT_CSV),
        },
        "output": {
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "jsonl": str(OUTPUT_JSONL.relative_to(ROOT)),
        },
        "dataset": {
            "baseline_rows": EXPECTED_BASE_ROWS,
            "attack_scenarios": EXPECTED_SCENARIOS,
            "rows_per_attack": EXPECTED_BASE_ROWS,
            "total_rows": expected_rows,
            "active_timesteps": EXPECTED_ACTIVE_TIMESTEPS,
            "messages_per_timestep": EXPECTED_MESSAGES,
        },
        "safety_boundary": {
            "laboratory_generated": True,
            "physical_vehicle": False,
            "direct_actuation": False,
            "production_dbc": False,
            "real_world_cyberattack": False,
            "live_cortex_xdr": False,
        },
        "scenario_ids": [
            x["id"] for x in SCENARIOS
        ],
        "disk_scenario_counts": disk_counts,
        "validation_failures": FAILURES,
    }

    with SCHEMA_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            schema,
            f,
            indent=2,
            ensure_ascii=False,
        )

    manifest = {
        "phase": "5.10",
        "title": "Cyber-Physical Attack Scenario Manifest",
        "status": "PASS" if not FAILURES else "FAIL",
        "scenario_ids": [
            x["id"] for x in SCENARIOS
        ],
        "counts": {
            "input_rows": len(source_rows),
            "baseline_rows": counts.get(
                "S00_NORMAL_BASELINE",
                0,
            ),
            "attack_scenarios": len(SCENARIOS),
            "output_rows": len(output_rows),
        },
        "disk_counts": disk_counts,
        "files": {
            "csv": str(
                OUTPUT_CSV.relative_to(ROOT)
            ),
            "jsonl": str(
                OUTPUT_JSONL.relative_to(ROOT)
            ),
            "schema": str(
                SCHEMA_JSON.relative_to(ROOT)
            ),
        },
        "safety_boundary": {
            "physical_vehicle": False,
            "direct_actuation": False,
            "real_world_cyberattack": False,
            "live_cortex_xdr": False,
        },
        "validation_failures": FAILURES,
    }

    with MANIFEST_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Cyber-physical scenarios: {len(SCENARIOS)}")
    print(
        f"Baseline records used: "
        f"{counts.get('S00_NORMAL_BASELINE', 0)}"
    )
    print(f"Total output records: {len(output_rows)}")

    print("Scenario counts after CSV re-read:")

    for sid in ["S00_NORMAL_BASELINE"] + [
        x["id"] for x in SCENARIOS
    ]:
        print(
            f"  {sid}: "
            f"{disk_counts.get(sid, 0)}"
        )

    print(
        f"Validation failures: "
        f"{len(FAILURES)}"
    )

    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"Schema: {SCHEMA_JSON}")
    print(f"Manifest: {MANIFEST_JSON}")

    if FAILURES:
        for item in FAILURES:
            print(f"FAIL: {item}")

        print("STATUS: FAIL")
        raise SystemExit(1)

    print("Validation: PASS")
    print("STATUS: PASS")


if __name__ == "__main__":
    main()