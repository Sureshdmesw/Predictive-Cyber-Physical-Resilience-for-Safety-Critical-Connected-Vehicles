from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

SCRIPT_DIR = ROOT / "scripts" / "phase6_1"
DATA_DIR = ROOT / "data" / "processed" / "phase6_1"
EXP_DIR = ROOT / "experiments" / "phase6_1"

CSV_OUT = DATA_DIR / "phase6_1_integrated_validation.csv"
JSONL_OUT = DATA_DIR / "phase6_1_integrated_validation.jsonl"
SCHEMA_OUT = DATA_DIR / "phase6_1_integrated_validation_schema.json"
MANIFEST_OUT = EXP_DIR / "phase6_1_integrated_validation_manifest.json"


REQUIRED_DOCS = [
    "docs/threat_model.md",
    "docs/system_architecture.md",
    "docs/data_architecture.md",
    "docs/safety_architecture.md",
    "docs/connectivity_resilience.md",
    "docs/cortex_xdr_integration.md",
    "docs/research_questions.md",
]

EXPECTED_PHASES = list(range(1, 22))

SAFETY_EXPECTATIONS = {
    "laboratory_simulation": True,
    "physical_vehicle": False,
    "direct_vehicle_actuation": False,
    "live_cortex_xdr": False,
    "model_retraining": False,
    "real_world_cyberattack": False,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_phase_artifacts(phase: int):
    name = f"phase5_{phase}"

    script_dir = ROOT / "scripts" / "can_hil" / name
    data_dir = ROOT / "data" / "processed" / "can_hil" / name
    exp_dir = ROOT / "experiments" / "can_hil" / name

    script_files = list(script_dir.glob("*.py")) if script_dir.exists() else []
    data_files = list(data_dir.iterdir()) if data_dir.exists() else []
    manifest_files = list(exp_dir.glob("*.json")) if exp_dir.exists() else []

    return {
        "phase": name,
        "script_directory": script_dir.exists(),
        "data_directory": data_dir.exists(),
        "experiment_directory": exp_dir.exists(),
        "script_file_count": len(script_files),
        "data_file_count": len(data_files),
        "manifest_file_count": len(manifest_files),
        "manifest_status": (
            "PRESENT"
            if manifest_files
            else "MISSING"
        ),
    }


def read_phase_manifest(phase: int):
    exp_dir = ROOT / "experiments" / "can_hil" / f"phase5_{phase}"

    manifests = sorted(exp_dir.glob("*.json")) if exp_dir.exists() else []

    if not manifests:
        return None, None

    # Prefer a manifest containing "manifest" in its name.
    preferred = [
        p for p in manifests
        if "manifest" in p.name.lower()
    ]

    path = preferred[0] if preferred else manifests[0]

    try:
        return path, load_json(path)
    except Exception:
        return path, None


def validate_phase5_21():
    path, manifest = read_phase_manifest(21)

    result = {
        "manifest_exists": path is not None,
        "manifest_valid_json": manifest is not None,
        "status_pass": False,
        "input_rows": None,
        "scenario_count": None,
        "gate_passed": None,
        "gate_total": None,
        "all_components_valid": False,
        "all_end_to_end_valid": False,
        "all_evidence_complete": False,
        "all_scores_100": False,
        "safety_boundaries_valid": False,
    }

    if manifest is None:
        return result

    result["status_pass"] = manifest.get("status") == "PASS"
    result["input_rows"] = manifest.get("input_rows")
    result["scenario_count"] = manifest.get("scenario_count")

    gate_summary = manifest.get("gate_summary", {})
    result["gate_passed"] = gate_summary.get("passed")
    result["gate_total"] = gate_summary.get("total")

    validation = manifest.get("validation", {})

    result["all_components_valid"] = (
        validation.get("all_components_valid") is True
    )
    result["all_end_to_end_valid"] = (
        validation.get("all_end_to_end_valid") is True
    )
    result["all_evidence_complete"] = (
        validation.get("all_evidence_complete") is True
    )
    result["all_scores_100"] = (
        validation.get("all_scores_100") is True
    )
    result["safety_boundaries_valid"] = (
        validation.get("all_safety_boundaries_revalidated") is True
    )

    return result


def validate_safety_flags():
    failures = []

    csv_candidates = sorted(
        (ROOT / "data" / "processed" / "can_hil").rglob("*.csv")
    )

    checked_files = 0
    checked_rows = 0

    relevant_names = [
        "phase5_17_secure_synchronization.csv",
        "phase5_18_adversarial_integrity_validation.csv",
        "phase5_19_end_to_end_resilience_validation.csv",
        "phase5_20_resilience_scorecard.csv",
        "phase5_21_resilience_evidence_matrix.csv",
    ]

    for path in csv_candidates:
        if path.name not in relevant_names:
            continue

        checked_files += 1

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    checked_rows += 1

                    for key, expected in SAFETY_EXPECTATIONS.items():
                        if key not in row:
                            continue

                        value = str(row[key]).strip().lower()

                        expected_text = "true" if expected else "false"

                        if value != expected_text:
                            failures.append({
                                "file": str(path.relative_to(ROOT)),
                                "field": key,
                                "actual": row[key],
                                "expected": expected_text,
                            })

        except Exception as exc:
            failures.append({
                "file": str(path.relative_to(ROOT)),
                "error": str(exc),
            })

    return {
        "files_checked": checked_files,
        "rows_checked": checked_rows,
        "failure_count": len(failures),
        "failures": failures,
        "valid": len(failures) == 0,
    }


def main():
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    checks = []

    # ------------------------------------------------------------
    # Project root
    # ------------------------------------------------------------

    checks.append({
        "check_id": "PROJECT_ROOT",
        "category": "PROJECT",
        "item": "Project root exists",
        "status": ROOT.exists(),
        "details": str(ROOT),
    })

    # ------------------------------------------------------------
    # Documentation
    # ------------------------------------------------------------

    for doc in REQUIRED_DOCS:
        path = ROOT / doc

        checks.append({
            "check_id": f"DOC_{Path(doc).stem.upper()}",
            "category": "DOCUMENTATION",
            "item": doc,
            "status": path.exists() and path.is_file(),
            "details": (
                f"{path.stat().st_size} bytes"
                if path.exists()
                else "MISSING"
            ),
        })

    # ------------------------------------------------------------
    # Phase 5 artifact coverage
    # ------------------------------------------------------------

    phase_results = []

    for phase in EXPECTED_PHASES:
        result = check_phase_artifacts(phase)
        phase_results.append(result)

        # Phase 1 and 2 have known architecture/experiment-only artifacts.
        # Therefore the integrated audit treats experiment evidence as
        # sufficient for those two historical phases.
        if phase in (1, 2):
            status = result["experiment_directory"] and (
                result["manifest_file_count"] > 0
            )
        else:
            status = (
                result["script_directory"]
                and result["data_directory"]
                and result["experiment_directory"]
            )

        checks.append({
            "check_id": f"PHASE5_{phase:02d}_ARTIFACT_COVERAGE",
            "category": "PHASE_5",
            "item": f"Phase 5.{phase} artifact coverage",
            "status": status,
            "details": json.dumps(result, sort_keys=True),
        })

    # ------------------------------------------------------------
    # Phase 5.21 final gate
    # ------------------------------------------------------------

    p521 = validate_phase5_21()

    p521_checks = {
        "P521_MANIFEST_EXISTS": p521["manifest_exists"],
        "P521_MANIFEST_VALID_JSON": p521["manifest_valid_json"],
        "P521_STATUS_PASS": p521["status_pass"],
        "P521_SCENARIO_COUNT_10": p521["scenario_count"] == 10,
        "P521_GATE_93_OF_93": (
            p521["gate_passed"] == 93
            and p521["gate_total"] == 93
        ),
        "P521_COMPONENTS_VALID": p521["all_components_valid"],
        "P521_END_TO_END_VALID": p521["all_end_to_end_valid"],
        "P521_EVIDENCE_COMPLETE": p521["all_evidence_complete"],
        "P521_SCORES_100": p521["all_scores_100"],
        "P521_SAFETY_VALID": p521["safety_boundaries_valid"],
    }

    for check_id, status in p521_checks.items():
        checks.append({
            "check_id": check_id,
            "category": "PHASE_5_21",
            "item": check_id,
            "status": bool(status),
            "details": json.dumps(p521, sort_keys=True),
        })

    # ------------------------------------------------------------
    # Safety boundary audit
    # ------------------------------------------------------------

    safety = validate_safety_flags()

    checks.append({
        "check_id": "SAFETY_BOUNDARY_AUDIT",
        "category": "SAFETY",
        "item": "Phase 5 safety boundary consistency",
        "status": safety["valid"],
        "details": json.dumps(safety, sort_keys=True),
    })

    # ------------------------------------------------------------
    # Required top-level architecture directories
    # ------------------------------------------------------------

    required_dirs = [
        "ai",
        "attacks",
        "cortex_xdr",
        "detection",
        "edge",
        "fleet",
        "forensics",
        "models",
        "resilience",
        "scripts",
        "tests",
    ]

    for dirname in required_dirs:
        path = ROOT / dirname

        checks.append({
            "check_id": f"DIR_{dirname.upper()}",
            "category": "ARCHITECTURE",
            "item": f"Top-level directory: {dirname}",
            "status": path.exists() and path.is_dir(),
            "details": str(path),
        })

    # ------------------------------------------------------------
    # Phase 5.12 known statistical limitation detection
    # ------------------------------------------------------------

    p512_csv = (
        ROOT
        / "data"
        / "processed"
        / "can_hil"
        / "phase5_12"
        / "phase5_12_uncertainty_ood.csv"
    )

    phase512_warning = False
    phase512_details = "Phase 5.12 output not found."

    if p512_csv.exists():
        try:
            with p512_csv.open(
                "r",
                encoding="utf-8-sig",
                newline=""
            ) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            max_ood = None

            if "ood_score" in reader.fieldnames:
                values = []

                for row in rows:
                    try:
                        values.append(float(row["ood_score"]))
                    except (ValueError, TypeError):
                        pass

                if values:
                    max_ood = max(values)

            phase512_warning = (
                max_ood is not None
                and max_ood > 100000
            )

            phase512_details = json.dumps({
                "rows": len(rows),
                "max_ood_score": max_ood,
                "review_required": phase512_warning,
                "reason": (
                    "Extreme OOD score detected; Phase 5.12 statistical "
                    "methodology requires scientific review before claiming "
                    "production-grade OOD robustness."
                    if phase512_warning
                    else
                    "No extreme OOD threshold trigger detected."
                ),
            })

        except Exception as exc:
            phase512_warning = True
            phase512_details = f"Unable to inspect Phase 5.12 output: {exc}"

    checks.append({
        "check_id": "P512_STATISTICAL_REVIEW",
        "category": "SCIENTIFIC_REVIEW",
        "item": "Phase 5.12 OOD methodology review",
        "status": not phase512_warning,
        "details": phase512_details,
    })

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    passed = sum(1 for c in checks if c["status"])
    total = len(checks)
    failed = total - passed

    # Scientific review is deliberately separated from engineering gates.
    # A failed review does not erase already validated Phase 5 evidence.
    engineering_checks = [
        c for c in checks
        if c["category"] != "SCIENTIFIC_REVIEW"
    ]

    engineering_passed = sum(
        1 for c in engineering_checks if c["status"]
    )
    engineering_total = len(engineering_checks)

    engineering_valid = (
        engineering_passed == engineering_total
    )

    overall_status = (
        "PASS_WITH_SCIENTIFIC_REVIEW_REQUIRED"
        if engineering_valid and phase512_warning
        else
        "PASS"
        if engineering_valid and not phase512_warning
        else
        "FAIL"
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    records = []

    for check in checks:
        records.append({
            "timestamp_utc": timestamp,
            "check_id": check["check_id"],
            "category": check["category"],
            "item": check["item"],
            "status": "PASS" if check["status"] else "FAIL",
            "details": check["details"],
        })

    with CSV_OUT.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "check_id",
                "category",
                "item",
                "status",
                "details",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    with JSONL_OUT.open(
        "w",
        encoding="utf-8"
    ) as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    schema = {
        "phase": "6.1",
        "title": "Integrated Research System Validation & Completion Audit",
        "format": "CSV/JSONL",
        "columns": {
            "timestamp_utc": "UTC execution timestamp",
            "check_id": "Unique validation check identifier",
            "category": "Validation category",
            "item": "Validation item",
            "status": "PASS or FAIL",
            "details": "Validation evidence/details",
        },
    }

    with SCHEMA_OUT.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(schema, f, indent=2)

    manifest = {
        "phase": "6.1",
        "title": "Integrated Research System Validation & Completion Audit",
        "status": overall_status,
        "timestamp_utc": timestamp,

        "validation": {
            "checks_passed": passed,
            "checks_total": total,
            "checks_failed": failed,
            "engineering_checks_passed": engineering_passed,
            "engineering_checks_total": engineering_total,
            "engineering_valid": engineering_valid,
            "scientific_review_required": phase512_warning,
        },

        "phase5": {
            "phases_audited": EXPECTED_PHASES,
            "phase5_21": p521,
            "phase5_21_status": "PASS" if (
                p521["status_pass"]
                and p521["gate_passed"] == 93
                and p521["gate_total"] == 93
            ) else "REVIEW",
        },

        "documentation": {
            "required_documents": REQUIRED_DOCS,
            "all_present": all(
                (ROOT / doc).is_file()
                for doc in REQUIRED_DOCS
            ),
        },

        "safety_boundary": SAFETY_EXPECTATIONS,

        "phase5_12_scientific_review": {
            "required": phase512_warning,
            "details": phase512_details,
        },

        "outputs": {
            "csv": str(CSV_OUT.relative_to(ROOT)),
            "jsonl": str(JSONL_OUT.relative_to(ROOT)),
            "schema": str(SCHEMA_OUT.relative_to(ROOT)),
            "manifest": str(MANIFEST_OUT.relative_to(ROOT)),
        },

        "scientific_boundary": {
            "laboratory_simulation": True,
            "physical_vehicle": False,
            "direct_vehicle_actuation": False,
            "live_cortex_xdr": False,
            "model_retraining": False,
            "real_world_cyberattack": False,
        },
    }

    with MANIFEST_OUT.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=== PHASE 6.1 INTEGRATED VALIDATION ===")
    print(f"Checks passed: {passed}/{total}")
    print(f"Engineering:   {engineering_passed}/{engineering_total}")
    print(f"Phase 5.21:    {p521['gate_passed']}/{p521['gate_total']}")

    if phase512_warning:
        print(
            "Scientific review: REQUIRED "
            "(Phase 5.12 OOD statistical issue detected)"
        )
    else:
        print("Scientific review: no automatic trigger")

    print(f"STATUS: {overall_status}")

    print()
    print("Outputs:")
    print(CSV_OUT)
    print(JSONL_OUT)
    print(SCHEMA_OUT)
    print(MANIFEST_OUT)


if __name__ == "__main__":
    main()
