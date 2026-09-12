from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone


ROOT = Path(
    r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles"
)

EVIDENCE_JSON = (
    ROOT
    / "experiments"
    / "final_release"
    / "final_evidence_consolidation.json"
)

OUTPUT_DIR = ROOT / "experiments" / "final_release"

OUTPUT_JSON = (
    OUTPUT_DIR
    / "final_reproducibility_audit.json"
)


EXPECTED_MODEL_SHA256 = (
    "d54cfc5fb5613fef5bdfd6ac4b2ce780ebd2e64ca47b7e57c630dae3d323a754"
)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_json(path):
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        return {
            "__error__": str(exc)
        }


def add_check(checks, name, passed, details):
    checks.append(
        {
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )


def check_consolidated_evidence(checks):

    exists = EVIDENCE_JSON.exists()

    add_check(
        checks,
        "evidence_consolidation_exists",
        exists,
        str(EVIDENCE_JSON.relative_to(ROOT))
    )

    if not exists:
        return None

    data = load_json(EVIDENCE_JSON)

    valid_json = "__error__" not in data

    add_check(
        checks,
        "evidence_consolidation_json_valid",
        valid_json,
        "JSON parsing successful"
        if valid_json
        else data.get("__error__", "Unknown JSON error")
    )

    if not valid_json:
        return None

    records = data.get(
        "evidence_records",
        []
    )

    add_check(
        checks,
        "evidence_records_present",
        isinstance(records, list) and len(records) > 0,
        f"records={len(records)}"
    )

    return data


def verify_evidence_hashes(checks, data):

    records = data.get(
        "evidence_records",
        []
    )

    missing = []
    mismatched = []
    verified = 0

    for record in records:

        relative = record.get("path")

        if not relative:
            missing.append(
                "<missing path>"
            )
            continue

        path = ROOT / relative

        if not path.exists():
            missing.append(relative)
            continue

        expected = record.get("sha256")

        if not expected:
            missing.append(
                f"{relative} [missing recorded hash]"
            )
            continue

        actual = sha256_file(path)

        if actual != expected:
            mismatched.append(
                {
                    "path": relative,
                    "expected": expected,
                    "actual": actual,
                }
            )
        else:
            verified += 1

    add_check(
        checks,
        "all_consolidated_evidence_files_present",
        len(missing) == 0,
        f"missing={len(missing)}"
    )

    add_check(
        checks,
        "all_consolidated_sha256_hashes_match",
        len(mismatched) == 0,
        f"verified={verified}, mismatched={len(mismatched)}"
    )

    return {
        "records_total": len(records),
        "verified": verified,
        "missing": missing,
        "mismatched": mismatched,
    }


def verify_manifests(checks):

    manifests = []

    experiments = ROOT / "experiments"

    if experiments.exists():

        for path in experiments.rglob("*.json"):

            if "manifest" not in path.name.lower():
                continue

            manifests.append(path)

    invalid = []

    for path in manifests:

        data = load_json(path)

        if "__error__" in data:

            invalid.append(
                {
                    "path": str(
                        path.relative_to(ROOT)
                    ),
                    "error": data["__error__"],
                }
            )

    add_check(
        checks,
        "all_manifest_json_valid",
        len(invalid) == 0,
        f"manifests={len(manifests)}, invalid={len(invalid)}"
    )

    return {
        "total": len(manifests),
        "invalid": invalid,
    }


def discover_phase_manifests():

    phase5 = []
    phase6 = []

    experiments = ROOT / "experiments"

    if not experiments.exists():
        return phase5, phase6

    for path in experiments.rglob("*manifest.json"):

        relative = (
            path.relative_to(ROOT)
            .as_posix()
            .lower()
        )

        if "phase5_" in relative:
            phase5.append(path)

        if "phase6_" in relative:
            phase6.append(path)

    return (
        sorted(phase5),
        sorted(phase6),
    )


def verify_phase_coverage(checks):

    phase5, phase6 = discover_phase_manifests()

    add_check(
        checks,
        "phase5_manifest_coverage",
        len(phase5) >= 14,
        f"phase5_manifests={len(phase5)}"
    )

    add_check(
        checks,
        "phase6_manifest_coverage",
        len(phase6) >= 4,
        f"phase6_manifests={len(phase6)}"
    )

    return {
        "phase5": [
            str(p.relative_to(ROOT))
            for p in phase5
        ],
        "phase6": [
            str(p.relative_to(ROOT))
            for p in phase6
        ],
    }


def verify_required_scripts(checks):

    required = [
        ROOT
        / "scripts"
        / "phase6_1"
        / "integrated_research_system_validation.py",

        ROOT
        / "scripts"
        / "phase6_2"
        / "corrected_ood_recomputation.py",

        ROOT
        / "scripts"
        / "phase6_3"
        / "scientific_review_ood_validation.py",
    ]

    missing = [
        str(p.relative_to(ROOT))
        for p in required
        if not p.exists()
    ]

    add_check(
        checks,
        "required_phase6_scripts_present",
        len(missing) == 0,
        f"required={len(required)}, missing={len(missing)}"
    )

    return {
        "required": [
            str(p.relative_to(ROOT))
            for p in required
        ],
        "missing": missing,
    }


def verify_phase6_chain(checks):

    chain = [
        ROOT
        / "data"
        / "processed"
        / "can_hil"
        / "phase6_2"
        / "phase6_2_corrected_ood.csv",

        ROOT
        / "data"
        / "processed"
        / "can_hil"
        / "phase6_3"
        / "phase6_3_ood_scientific_review.json",

        ROOT
        / "data"
        / "processed"
        / "can_hil"
        / "phase6_4"
        / "phase6_4_uncertainty_calibration.json",
    ]

    missing = [
        str(p.relative_to(ROOT))
        for p in chain
        if not p.exists()
    ]

    add_check(
        checks,
        "phase6_scientific_chain_present",
        len(missing) == 0,
        f"chain_artifacts={len(chain)}, missing={len(missing)}"
    )

    return {
        "chain": [
            str(p.relative_to(ROOT))
            for p in chain
        ],
        "missing": missing,
    }


def verify_model_checkpoint(checks):

    model = (
        ROOT
        / "models"
        / "v2"
        / "advanced"
        / "crss_2024_v2_temporal_transformer_best.pt"
    )

    exists = model.exists()

    add_check(
        checks,
        "frozen_transformer_checkpoint_present",
        exists,
        str(model.relative_to(ROOT))
    )

    if not exists:
        return {
            "path": str(model.relative_to(ROOT)),
            "exists": False,
            "sha256": None,
            "expected_sha256": EXPECTED_MODEL_SHA256,
            "hash_match": False,
        }

    actual = sha256_file(model)

    match = actual == EXPECTED_MODEL_SHA256

    add_check(
        checks,
        "frozen_transformer_checkpoint_hash_match",
        match,
        f"actual={actual}"
    )

    return {
        "path": str(model.relative_to(ROOT)),
        "exists": True,
        "sha256": actual,
        "expected_sha256": EXPECTED_MODEL_SHA256,
        "hash_match": match,
    }


def verify_duplicate_phase62_outputs(checks):

    targets = [
        "phase6_2_corrected_ood.csv",
        "phase6_2_corrected_ood.jsonl",
        "phase6_2_corrected_ood_schema.json",
    ]

    locations = []

    for target in targets:

        matches = []

        for path in ROOT.rglob(target):

            if path.is_file():
                matches.append(
                    str(path.relative_to(ROOT))
                )

        locations.append(
            {
                "filename": target,
                "matches": sorted(matches),
                "count": len(matches),
            }
        )

    total_duplicates = sum(
        max(0, item["count"] - 1)
        for item in locations
    )

    add_check(
        checks,
        "duplicate_phase62_artifacts_identified",
        True,
        f"duplicate_or_multi_location_files={total_duplicates}"
    )

    return locations


def build_summary(checks):

    passed = sum(
        1
        for check in checks
        if check["status"] == "PASS"
    )

    failed = sum(
        1
        for check in checks
        if check["status"] == "FAIL"
    )

    total = len(checks)

    return {
        "status": (
            "PASS"
            if failed == 0
            else "FAIL"
        ),
        "checks_passed": passed,
        "checks_failed": failed,
        "checks_total": total,
        "completion_percentage": round(
            (passed / total) * 100,
            2
        ) if total else 0.0,
    }


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checks = []

    evidence_data = check_consolidated_evidence(
        checks
    )

    hash_results = None

    if evidence_data is not None:

        hash_results = verify_evidence_hashes(
            checks,
            evidence_data
        )

    manifest_results = verify_manifests(
        checks
    )

    coverage_results = verify_phase_coverage(
        checks
    )

    script_results = verify_required_scripts(
        checks
    )

    chain_results = verify_phase6_chain(
        checks
    )

    model_results = verify_model_checkpoint(
        checks
    )

    duplicate_results = verify_duplicate_phase62_outputs(
        checks
    )

    summary = build_summary(checks)

    result = {
        "reproducibility_audit": {
            "name": "FINAL_RESEARCH_RELEASE_REPRODUCIBILITY_AUDIT",
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "project_root": str(ROOT),
            "source_evidence": (
                "experiments/final_release/"
                "final_evidence_consolidation.json"
            ),
            "read_only_source_validation": True,
        },

        "summary": summary,

        "checks": checks,

        "hash_verification": hash_results,

        "manifest_validation": manifest_results,

        "phase_coverage": coverage_results,

        "required_scripts": script_results,

        "phase6_scientific_chain": chain_results,

        "model_checkpoint": model_results,

        "phase6_2_duplicate_locations": duplicate_results,
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 80)
    print("FINAL RESEARCH RELEASE — REPRODUCIBILITY AUDIT")
    print("=" * 80)

    print(
        f"Status              : "
        f"{summary['status']}"
    )

    print(
        f"Checks              : "
        f"{summary['checks_passed']}/"
        f"{summary['checks_total']}"
    )

    print(
        f"Completion          : "
        f"{summary['completion_percentage']:.2f}%"
    )

    print()
    print("REPRODUCIBILITY CHECKS")
    print("-" * 80)

    for check in checks:

        print(
            f"{check['status']:<6} "
            f"{check['name']:<45} "
            f"{check['details']}"
        )

    print()
    print("MODEL CHECKPOINT")
    print("-" * 80)

    if model_results:
        print(
            f"Path   : "
            f"{model_results['path']}"
        )

        print(
            f"SHA256 : "
            f"{model_results['sha256']}"
        )

        print(
            f"Match  : "
            f"{model_results['hash_match']}"
        )

    print()
    print("PHASE 6.2 ARTIFACT LOCATIONS")
    print("-" * 80)

    for item in duplicate_results:

        print(
            f"{item['filename']}: "
            f"{item['count']} location(s)"
        )

        for location in item["matches"]:
            print(
                f"    {location}"
            )

    print()
    print(
        f"Audit JSON: "
        f"{OUTPUT_JSON.relative_to(ROOT)}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()

