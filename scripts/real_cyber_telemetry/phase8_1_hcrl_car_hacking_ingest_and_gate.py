"""
Phase 8.1 -- Real Cyber Telemetry Ingestion: HCRL Car-Hacking Dataset
======================================================================

STATUS: v2. v1 assumed a header row and failed against the real
dataset (0/5 files ingested -- confirmed by direct run against the
real downloaded files on 2026-09-15). This version fixes that.

Objective
---------
Ingest the real, labeled HCRL Car-Hacking dataset (real vehicle CAN
captures, Hyundai YF Sonata, logged via OBD-II) into a canonical schema
that is explicitly tagged as REAL_CYBER_TELEMETRY_BASELINE, so it can
never be silently blended with this project's existing SYNTHETIC cyber
telemetry.

Confirmed real format (verified against actual downloaded file output,
2026-09-15 -- this is evidence, not assumption):
    No header row. Each data row is:
        timestamp, CAN_ID (hex, no 0x prefix), DLC, <DLC data bytes (hex)>, Flag
    Row width VARIES per row because DLC varies (row length = 3 + DLC + 1).
    Flag "T" marks an injected/attack frame, "R" marks a normal frame.
    Example real row: 1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R

Usage
-----
1. Raw files already placed under:
       data/raw/real_cyber_telemetry/hcrl_car_hacking/
   (DoS_dataset.csv, Fuzzy_dataset.csv, gear_dataset.csv, RPM_dataset.csv)
   normal_run_data.txt is present alongside but is NOT processed by this
   script -- different format, out of scope for Phase 8.1.
2. Run:  python scripts/real_cyber_telemetry/phase8_1_hcrl_car_hacking_ingest_and_gate.py

Output
------
- data/processed/real_cyber_telemetry/hcrl_car_hacking/hcrl_canonical.csv
- experiments/real_cyber_telemetry/phase8_1_hcrl_ingest_gate.json
"""

from pathlib import Path
import csv
import json
import hashlib
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "real_cyber_telemetry" / "hcrl_car_hacking"
OUT_DIR = ROOT / "data" / "processed" / "real_cyber_telemetry" / "hcrl_car_hacking"
REPORT_DIR = ROOT / "experiments" / "real_cyber_telemetry"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_CSV = OUT_DIR / "hcrl_canonical.csv"
REPORT_PATH = REPORT_DIR / "phase8_1_hcrl_ingest_gate.json"

DATA_CLASS = "REAL_CYBER_TELEMETRY_BASELINE"

SOURCE_FILES = {
    "DoS_dataset.csv": "DOS",
    "Fuzzy_dataset.csv": "FUZZY",
    "gear_dataset.csv": "GEAR_SPOOFING",
    "RPM_dataset.csv": "RPM_SPOOFING",
}

# A negative timestamp jump larger than this (seconds) is flagged as a
# real monotonicity violation rather than capture jitter.
TIMESTAMP_JITTER_TOLERANCE_S = 1.0


def sha256_row(row):
    return hashlib.sha256(
        json.dumps(row, sort_keys=True).encode("utf-8")
    ).hexdigest()


checks = []


def add_check(name, passed, detail=""):
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def looks_like_header(first_row):
    """The confirmed real format has NO header. Defensively support a
    headered mirror too: if the first field doesn't parse as a
    timestamp, treat that single row as a header and skip it."""
    if not first_row:
        return False
    try:
        float(first_row[0])
        return False
    except ValueError:
        return True


def parse_data_row(row):
    """Positional parse per the confirmed real format:
    [timestamp, can_id, dlc, <dlc data bytes>, flag]
    Row width varies with dlc. Returns a dict of parsed fields plus a
    'width_ok' flag, or None if the row is too short to parse at all.
    """
    if len(row) < 4:
        return None

    ts_raw = row[0]
    can_id_raw = row[1].strip()
    dlc_raw = row[2]

    try:
        dlc_val = int(dlc_raw)
    except ValueError:
        dlc_val = None

    if dlc_val is not None:
        expected_len = 3 + dlc_val + 1
        width_ok = len(row) == expected_len
    else:
        expected_len = None
        width_ok = False

    # Best-effort extraction even when width is off, so we can still
    # report what's wrong rather than dropping the row silently.
    data_bytes = row[3:-1]
    flag_raw = row[-1]

    return {
        "timestamp": ts_raw,
        "can_id": can_id_raw,
        "dlc_raw": dlc_raw,
        "dlc_val": dlc_val,
        "data_bytes": data_bytes,
        "flag_raw": flag_raw,
        "width_ok": width_ok,
        "raw_width": len(row),
        "expected_width": expected_len,
    }


def inspect_and_ingest():
    canonical_rows = []
    per_file_summary = {}
    seen_hashes = set()
    duplicate_count = 0

    for filename, attack_label in SOURCE_FILES.items():
        path = RAW_DIR / filename
        file_exists = path.exists()
        add_check(f"file_present::{filename}", file_exists, str(path))
        if not file_exists:
            per_file_summary[filename] = {"status": "MISSING"}
            continue

        with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)

            row_count = 0
            monotonic_violations = 0
            invalid_dlc = 0
            invalid_can_id = 0
            unexpected_flag = 0
            width_mismatches = 0
            label_counter = Counter()
            prev_ts = None
            first_row_checked = False

            for raw_row in reader:
                if not first_row_checked:
                    first_row_checked = True
                    if looks_like_header(raw_row):
                        continue  # skip a header row if this mirror has one

                parsed = parse_data_row(raw_row)
                if parsed is None:
                    continue

                row_count += 1

                try:
                    ts = float(parsed["timestamp"])
                except ValueError:
                    ts = None

                if ts is not None and prev_ts is not None and ts < prev_ts - TIMESTAMP_JITTER_TOLERANCE_S:
                    monotonic_violations += 1
                if ts is not None:
                    prev_ts = ts

                try:
                    int(parsed["can_id"], 16)
                except ValueError:
                    invalid_can_id += 1

                if parsed["dlc_val"] is None or not (0 <= parsed["dlc_val"] <= 8):
                    invalid_dlc += 1

                if not parsed["width_ok"]:
                    width_mismatches += 1

                flag_val = parsed["flag_raw"].strip().upper()
                is_attack = flag_val in ("T", "1", "ATTACK")
                is_normal = flag_val in ("R", "0", "NORMAL")
                if not (is_attack or is_normal):
                    unexpected_flag += 1

                label = attack_label if is_attack else "NORMAL"
                label_counter[label] += 1

                canonical_row = {
                    "data_class": DATA_CLASS,
                    "source_file": filename,
                    "timestamp": parsed["timestamp"],
                    "can_id": parsed["can_id"],
                    "dlc": parsed["dlc_raw"],
                    "data_bytes": "|".join(parsed["data_bytes"]),
                    "label": label,
                    "is_attack": is_attack,
                }

                row_hash = sha256_row(canonical_row)
                if row_hash in seen_hashes:
                    duplicate_count += 1
                else:
                    seen_hashes.add(row_hash)

                canonical_rows.append(canonical_row)

            add_check(
                f"timestamp_monotonic::{filename}",
                monotonic_violations == 0,
                f"{monotonic_violations} violations out of {row_count} rows "
                f"(tolerance {TIMESTAMP_JITTER_TOLERANCE_S}s)",
            )
            add_check(
                f"can_id_valid_hex::{filename}",
                invalid_can_id == 0,
                f"{invalid_can_id} invalid CAN IDs out of {row_count} rows",
            )
            add_check(
                f"dlc_in_range::{filename}",
                invalid_dlc == 0,
                f"{invalid_dlc} out-of-range/unparseable DLC values out of {row_count} rows",
            )
            add_check(
                f"row_width_matches_dlc::{filename}",
                width_mismatches == 0,
                f"{width_mismatches} rows where column count didn't match 3+DLC+1, "
                f"out of {row_count} rows",
            )
            add_check(
                f"flag_value_known::{filename}",
                unexpected_flag == 0,
                f"{unexpected_flag} unrecognized flag values out of {row_count} rows",
            )

            per_file_summary[filename] = {
                "status": "INGESTED" if row_count > 0 else "EMPTY",
                "row_count": row_count,
                "label_counts": dict(label_counter),
            }

    add_check(
        "no_exact_duplicate_rows",
        duplicate_count == 0,
        f"{duplicate_count} exact-duplicate rows detected across all files",
    )
    add_check(
        "at_least_one_file_ingested",
        any(v.get("status") == "INGESTED" for v in per_file_summary.values()),
        "no source file was successfully ingested",
    )

    return canonical_rows, per_file_summary


def write_canonical_csv(rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(CANONICAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    canonical_rows, per_file_summary = inspect_and_ingest()
    write_canonical_csv(canonical_rows)

    passed = sum(1 for c in checks if c["passed"])
    failed = sum(1 for c in checks if not c["passed"])
    verdict = "PASS" if failed == 0 and canonical_rows else "FAIL"

    report = {
        "phase": "8.1",
        "name": "HCRL Car-Hacking real cyber telemetry ingest + gate (v2)",
        "data_class": DATA_CLASS,
        "status": verdict,
        "summary": {"pass": passed, "fail": failed, "total_checks": len(checks)},
        "checks": checks,
        "per_file_summary": per_file_summary,
        "canonical_row_count": len(canonical_rows),
        "canonical_output": str(CANONICAL_CSV),
        "note": (
            "This is a real, labeled dataset (not synthetic). "
            "Do not merge canonical_output with any SYNTHETIC-tagged "
            "telemetry file without an explicit, separately-reviewed "
            "fusion step."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("PHASE 8.1 -- HCRL CAR-HACKING REAL TELEMETRY INGEST GATE (v2)")
    print("=" * 80)
    print()
    print("STATUS :", verdict)
    print("CHECKS :", len(checks))
    print("PASS   :", passed)
    print("FAIL   :", failed)
    print("ROWS   :", len(canonical_rows))
    print()
    print("-" * 80)
    print("PER-FILE SUMMARY")
    print("-" * 80)
    for fname, summ in per_file_summary.items():
        print(fname, ":", summ)
    print()
    print("-" * 80)
    print("FAILURES")
    print("-" * 80)
    failures = [c for c in checks if not c["passed"]]
    if failures:
        for item in failures:
            print("[FAIL]", item["name"], ":", item["detail"])
    else:
        print("NONE")
    print()
    print("Report:", REPORT_PATH)
    print("Canonical CSV:", CANONICAL_CSV)
    print("=" * 80)


if __name__ == "__main__":
    main()
