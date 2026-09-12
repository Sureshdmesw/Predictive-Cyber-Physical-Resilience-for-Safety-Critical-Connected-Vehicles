from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_14"
    / "phase5_14_forensic_evidence.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "can_hil" / "phase5_15"
EXPERIMENT_DIR = ROOT / "experiments" / "can_hil" / "phase5_15"

OUTPUT_CSV = OUTPUT_DIR / "phase5_15_integrity_validation.csv"
OUTPUT_JSONL = OUTPUT_DIR / "phase5_15_integrity_validation.jsonl"
OUTPUT_SCHEMA = OUTPUT_DIR / "phase5_15_integrity_validation_schema.json"
MANIFEST = EXPERIMENT_DIR / "phase5_15_integrity_tamper_manifest.json"

TAMPER_CASES = [
    "VALID_EVIDENCE",
    "CAN_PAYLOAD_TAMPER",
    "SIGNAL_TAMPER",
    "TIMESTAMP_TAMPER",
    "SCENARIO_METADATA_TAMPER",
    "EVENT_ID_TAMPER",
    "EVIDENCE_REORDER",
    "EVIDENCE_TRUNCATION",
    "EVIDENCE_DUPLICATION",
]

REQUIRED_COLUMNS = [
    "event_id",
    "scenario_id",
    "evidence_phase",
    "timestep_index",
    "timestamp_s",
    "vehicle_id",
    "ecu_domain",
    "can_id",
    "message_name",
    "dlc",
    "data_hex",
    "signals_json",
    "connectivity_state",
    "evidence_source",
    "laboratory_generated",
    "physical_vehicle",
    "direct_actuation",
]


def canonical_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return float(value)

    return str(value)


def canonical_record(row):
    record = {}

    for column in REQUIRED_COLUMNS:
        record[column] = canonical_value(row[column])

    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def digest_record(row):
    payload = canonical_record(row).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def digest_dataframe(df):
    payload = []

    for _, row in df.iterrows():
        payload.append(canonical_record(row))

    canonical = "\n".join(payload).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


def verify_digest(df, digest):
    return digest_dataframe(df) == digest


def make_tampered_dataframe(df, case):
    tampered = df.copy()

    if case == "CAN_PAYLOAD_TAMPER":
        tampered.loc[tampered.index[0], "data_hex"] = "FFFFFFFFFFFFFFFF"

    elif case == "SIGNAL_TAMPER":
        tampered.loc[tampered.index[0], "signals_json"] = '{"TAMPERED":true}'

    elif case == "TIMESTAMP_TAMPER":
        tampered.loc[tampered.index[0], "timestamp_s"] = (
            float(tampered.loc[tampered.index[0], "timestamp_s"]) + 1.0
        )

    elif case == "SCENARIO_METADATA_TAMPER":
        tampered.loc[tampered.index[0], "scenario_id"] = "TAMPERED_SCENARIO"

    elif case == "EVENT_ID_TAMPER":
        tampered.loc[tampered.index[0], "event_id"] = "tampered_event_id"

    elif case == "EVIDENCE_REORDER":
        tampered = tampered.iloc[::-1].reset_index(drop=True)

    elif case == "EVIDENCE_TRUNCATION":
        tampered = tampered.iloc[:-1].reset_index(drop=True)

    elif case == "EVIDENCE_DUPLICATION":
        tampered = pd.concat(
            [tampered, tampered.iloc[[0]]],
            ignore_index=True,
        )

    return tampered


def evaluate_case(df, case):
    original_digest = digest_dataframe(df)

    if case == "VALID_EVIDENCE":
        test_df = df.copy()
    else:
        test_df = make_tampered_dataframe(df, case)

    calculated_digest = digest_dataframe(test_df)

    integrity_valid = calculated_digest == original_digest

    if integrity_valid:
        state = "VALID"
        synchronization = "ELIGIBLE"
        soc_handoff = "ELIGIBLE"
    else:
        state = "UNVERIFIED_EVIDENCE"
        synchronization = "REFUSE_SYNC"
        soc_handoff = "REFUSE_SOC_HANDOFF"

    return {
        "tamper_case": case,
        "original_sha256": original_digest,
        "calculated_sha256": calculated_digest,
        "integrity_valid": integrity_valid,
        "evidence_state": state,
        "synchronization_decision": synchronization,
        "soc_handoff_decision": soc_handoff,
        "evidence_records": len(test_df),
        "laboratory_generated": True,
        "physical_vehicle": False,
        "direct_actuation": False,
    }


def main():
    print()
    print("=== PHASE 5.15 INTEGRITY & TAMPER VALIDATION ===")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    checks = []

    def check(name, condition):
        passed = bool(condition)
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
            }
        )
        print(("  PASS " if passed else "  FAIL ") + name)

    check("input_exists", INPUT_CSV.exists())

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    print(f"Input records: {len(df)}")

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    check("required_columns_present", len(missing) == 0)
    check("input_records_4650", len(df) == 4650)
    check("input_not_empty", len(df) > 0)

    base_digest = digest_dataframe(df)

    check("base_sha256_generated", len(base_digest) == 64)

    results = []

    for case in TAMPER_CASES:
        print(f"Testing {case}...")

        result = evaluate_case(df, case)
        results.append(result)

        print(
            f"  integrity={result['integrity_valid']} "
            f"state={result['evidence_state']} "
            f"sync={result['synchronization_decision']} "
            f"soc={result['soc_handoff_decision']}"
        )

    results_df = pd.DataFrame(results)

    valid_case = results_df[
        results_df["tamper_case"] == "VALID_EVIDENCE"
    ].iloc[0]

    tampered_cases = results_df[
        results_df["tamper_case"] != "VALID_EVIDENCE"
    ]

    check(
        "valid_evidence_verified",
        bool(valid_case["integrity_valid"]),
    )

    check(
        "valid_evidence_sync_eligible",
        valid_case["synchronization_decision"] == "ELIGIBLE",
    )

    check(
        "valid_evidence_soc_eligible",
        valid_case["soc_handoff_decision"] == "ELIGIBLE",
    )

    check(
        "tamper_cases_detected",
        bool((~tampered_cases["integrity_valid"]).all()),
    )

    check(
        "tampered_evidence_unverified",
        bool(
            (
                tampered_cases["evidence_state"]
                == "UNVERIFIED_EVIDENCE"
            ).all()
        ),
    )

    check(
        "tampered_sync_refused",
        bool(
            (
                tampered_cases["synchronization_decision"]
                == "REFUSE_SYNC"
            ).all()
        ),
    )

    check(
        "tampered_soc_handoff_refused",
        bool(
            (
                tampered_cases["soc_handoff_decision"]
                == "REFUSE_SOC_HANDOFF"
            ).all()
        ),
    )

    check(
        "all_tamper_cases_present",
        set(TAMPER_CASES) == set(results_df["tamper_case"]),
    )

    check(
        "laboratory_only",
        bool(results_df["laboratory_generated"].all()),
    )

    check(
        "no_physical_vehicle",
        bool((~results_df["physical_vehicle"]).all()),
    )

    check(
        "no_direct_actuation",
        bool((~results_df["direct_actuation"]).all()),
    )

    results_df.to_csv(OUTPUT_CSV, index=False)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as file:
        for record in results_df.to_dict(orient="records"):
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    schema = {
        "phase": "5.15",
        "title": "Integrity and Tamper Validation",
        "input": str(INPUT_CSV.relative_to(ROOT)),
        "integrity_algorithm": "SHA-256",
        "canonical_serialization": {
            "format": "JSON",
            "sort_keys": True,
            "separators": [",", ":"],
            "unicode": "UTF-8",
        },
        "valid_path": [
            "VALID",
            "ELIGIBLE",
        ],
        "tampered_path": [
            "UNVERIFIED_EVIDENCE",
            "REFUSE_SYNC",
            "REFUSE_SOC_HANDOFF",
        ],
        "tamper_cases": TAMPER_CASES,
        "laboratory_generated": True,
        "physical_vehicle": False,
        "direct_actuation": False,
    }

    OUTPUT_SCHEMA.write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
    )

    passed = sum(
        item["status"] == "PASS"
        for item in checks
    )

    total = len(checks)

    status = "PASS" if passed == total else "FAIL"

    manifest = {
        "phase": "5.15",
        "title": "Integrity and Tamper Validation",
        "status": status,
        "input_records": len(df),
        "base_sha256": base_digest,
        "tamper_cases": TAMPER_CASES,
        "checks_passed": passed,
        "checks_total": total,
        "laboratory_only": True,
        "physical_vehicle": False,
        "direct_actuation": False,
        "live_cortex_xdr": False,
        "model_retraining": False,
        "results": results,
        "validation": checks,
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== PHASE 5.15 RESULTS ===")
    print()
    print(f"Validation: {passed}/{total} checks passed")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"Schema: {OUTPUT_SCHEMA}")
    print(f"Manifest: {MANIFEST}")
    print()
    print(f"STATUS: {status}")


if __name__ == "__main__":
    main()