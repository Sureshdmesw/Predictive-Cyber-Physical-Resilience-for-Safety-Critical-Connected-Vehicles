from pathlib import Path
import hashlib
import json
import py_compile
import sys


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "experiments" / "final_release"

EXPECTED_CHECKPOINT_SHA256 = (
    "d54cfc5fb5613fef5bdfd6ac4b2ce780ebd2e64ca47b7e57c630dae3d323a754"
)

EXPECTED_PHASE6_STATUSES = {
    "phase6_1": "PASS_WITH_SCIENTIFIC_REVIEW_REQUIRED",
    "phase6_2": "PASS",
    "phase6_3": "SCIENTIFIC_REVIEW_ACCEPTED_WITH_LIMITATIONS",
    "phase6_4": "PASS_WITH_CALIBRATION_LIMITATIONS",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_values(obj, key):
    values = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                values.append(v)
            values.extend(find_values(v, key))

    elif isinstance(obj, list):
        for item in obj:
            values.extend(find_values(item, key))

    return values


def find_first(obj, keys):
    for key in keys:
        values = find_values(obj, key)
        if values:
            return values[0]
    return None


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def check(condition, name, details):
    return {
        "name": name,
        "status": "PASS" if condition else "FAIL",
        "details": details,
    }


def main():
    checks = []

    required_files = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "requirements.txt",
        ROOT / "requirements-freeze.txt",

        ROOT / "docs" / "FINAL_RESEARCH_DOCUMENTATION.md",
        ROOT / "docs" / "final_research_claims_and_limitations.md",

        FINAL / "final_integrated_audit_v2.json",
        FINAL / "final_evidence_consolidation.json",
        FINAL / "final_evidence_index.csv",
        FINAL / "final_reproducibility_audit.json",
        FINAL / "final_claims_limitations_audit.json",
        FINAL / "publication_thesis_evidence_package.json",
        FINAL / "publication_thesis_evidence_index.csv",

        ROOT / "scripts" / "final_release" / "final_integrated_audit_v2.py",
        ROOT / "scripts" / "final_release" / "final_evidence_consolidation.py",
        ROOT / "scripts" / "final_release" / "final_reproducibility_audit.py",
        ROOT / "scripts" / "final_release" / "final_claims_limitations_audit.py",
        ROOT / "scripts" / "final_release" / "build_publication_thesis_evidence_package.py",
        ROOT / "scripts" / "final_release" / "sync_final_readme.py",
    ]

    missing = [
        str(p.relative_to(ROOT))
        for p in required_files
        if not p.exists()
    ]

    checks.append(
        check(
            len(missing) == 0,
            "Required final artifacts",
            {
                "missing": missing,
                "count_required": len(required_files),
            },
        )
    )

    # ------------------------------------------------------------
    # README
    # ------------------------------------------------------------

    readme = ROOT / "README.md"

    if readme.exists():
        text = readme.read_text(encoding="utf-8")

        readme_checks = {
            "final_release_marker":
                "<!-- FINAL_RESEARCH_RELEASE_START -->" in text,

            "final_release_end_marker":
                "<!-- FINAL_RESEARCH_RELEASE_END -->" in text,

            "synthetic_boundary":
                "synthetic" in text.lower(),

            "production_safety_boundary":
                "production" in text.lower() and
                "road" in text.lower(),

            "cortex_xdr_boundary":
                "Cortex XDR" in text or
                "CORTEX_XDR" in text,

            "reproducibility":
                "reproduc" in text.lower(),

            "claims_limitations":
                "limitation" in text.lower(),
        }

        checks.append(
            check(
                all(readme_checks.values()),
                "README final research release",
                readme_checks,
            )
        )

    # ------------------------------------------------------------
    # Integrated audit
    # ------------------------------------------------------------

    integrated_path = FINAL / "final_integrated_audit_v2.json"

    if integrated_path.exists():
        try:
            integrated = load_json(integrated_path)

            status = find_first(integrated, ["Status", "status"])
            completion = find_first(
                integrated,
                ["Completion", "completion"]
            )

            valid = (
                str(status).upper() == "PASS"
                and (
                    completion is None
                    or float(completion) >= 100.0
                )
            )

            checks.append(
                check(
                    valid,
                    "Integrated audit",
                    {
                        "status": status,
                        "completion": completion,
                    },
                )
            )

        except Exception as exc:
            checks.append(
                check(
                    False,
                    "Integrated audit JSON",
                    {"error": str(exc)},
                )
            )

    # ------------------------------------------------------------
    # Evidence consolidation
    # ------------------------------------------------------------

    evidence_path = FINAL / "final_evidence_consolidation.json"

    if evidence_path.exists():
        try:
            evidence = load_json(evidence_path)

            total = find_first(
                evidence,
                ["Total evidence files", "total_evidence_files",
                 "total_files", "Evidence files"]
            )

            invalid = find_first(
                evidence,
                ["Invalid manifests", "invalid_manifests",
                 "invalid"]
            )

            valid_manifest = find_first(
                evidence,
                ["Valid manifests", "valid_manifests"]
            )

            total_ok = total is None or int(total) > 0
            invalid_ok = invalid is None or int(invalid) == 0
            valid_manifest_ok = (
                valid_manifest is None
                or int(valid_manifest) > 0
            )

            checks.append(
                check(
                    total_ok and invalid_ok and valid_manifest_ok,
                    "Evidence consolidation",
                    {
                        "total_evidence_files": total,
                        "valid_manifests": valid_manifest,
                        "invalid_manifests": invalid,
                    },
                )
            )

        except Exception as exc:
            checks.append(
                check(
                    False,
                    "Evidence consolidation JSON",
                    {"error": str(exc)},
                )
            )

    # ------------------------------------------------------------
    # Reproducibility audit
    # ------------------------------------------------------------

    repro_path = FINAL / "final_reproducibility_audit.json"

    if repro_path.exists():
        try:
            repro = load_json(repro_path)

            status = find_first(repro, ["Status", "status"])
            hash_match = find_first(
                repro,
                [
                    "all_hashes_match",
                    "All hashes match",
                    "hashes_match",
                ],
            )

            valid = (
                str(status).upper() == "PASS"
                and (
                    hash_match is None
                    or bool(hash_match) is True
                    or str(hash_match).upper() == "PASS"
                )
            )

            checks.append(
                check(
                    valid,
                    "Reproducibility audit",
                    {
                        "status": status,
                        "hash_match": hash_match,
                    },
                )
            )

        except Exception as exc:
            checks.append(
                check(
                    False,
                    "Reproducibility audit JSON",
                    {"error": str(exc)},
                )
            )

    # ------------------------------------------------------------
    # Claims and limitations audit
    # ------------------------------------------------------------

    claims_path = FINAL / "final_claims_limitations_audit.json"

    if claims_path.exists():
        try:
            claims = load_json(claims_path)

            status = find_first(claims, ["Status", "status"])

            unsupported = find_first(
                claims,
                [
                    "unsupported",
                    "unsupported_claims",
                    "Unsupported claims",
                ],
            )

            unsupported_ok = (
                unsupported is None
                or unsupported == []
                or unsupported == {}
                or unsupported == ""
                or str(unsupported).lower() == "none"
            )

            valid = (
                str(status).upper() == "PASS"
                and unsupported_ok
            )

            checks.append(
                check(
                    valid,
                    "Claims and limitations audit",
                    {
                        "status": status,
                        "unsupported_claims": unsupported,
                    },
                )
            )

        except Exception as exc:
            checks.append(
                check(
                    False,
                    "Claims audit JSON",
                    {"error": str(exc)},
                )
            )

    # ------------------------------------------------------------
    # Publication / thesis evidence package
    # ------------------------------------------------------------

    publication_path = (
        FINAL / "publication_thesis_evidence_package.json"
    )

    if publication_path.exists():
        try:
            publication = load_json(publication_path)

            status = find_first(publication, ["Status", "status"])
            sections = find_first(
                publication,
                ["Sections", "sections"]
            )
            evidence_refs = find_first(
                publication,
                ["Evidence references", "evidence_references"]
            )
            existing_refs = find_first(
                publication,
                ["Existing references", "existing_references"]
            )
            missing_refs = find_first(
                publication,
                ["Missing references", "missing_references"]
            )
            coverage = find_first(
                publication,
                ["Coverage", "coverage"]
            )

            status_ok = (
                status is None
                or str(status).upper() == "PASS"
            )

            sections_ok = (
                sections is None
                or int(sections) == 13
            )

            refs_ok = (
                evidence_refs is None
                or existing_refs is None
                or int(evidence_refs) == int(existing_refs)
            )

            missing_ok = (
                missing_refs is None
                or int(missing_refs) == 0
            )

            coverage_ok = (
                coverage is None
                or float(str(coverage).replace("%", "")) >= 100.0
            )

            valid = (
                status_ok
                and sections_ok
                and refs_ok
                and missing_ok
                and coverage_ok
            )

            checks.append(
                check(
                    valid,
                    "Publication/thesis evidence package",
                    {
                        "status": status,
                        "sections": sections,
                        "evidence_references": evidence_refs,
                        "existing_references": existing_refs,
                        "missing_references": missing_refs,
                        "coverage": coverage,
                    },
                )
            )

        except Exception as exc:
            checks.append(
                check(
                    False,
                    "Publication package JSON",
                    {"error": str(exc)},
                )
            )

    # ------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------

    checkpoint_candidates = [
        ROOT / "models" / "v2" / "advanced"
        / "crss_2024_v2_temporal_transformer_best.pt"
    ]

    checkpoint = None

    for candidate in checkpoint_candidates:
        if candidate.exists():
            checkpoint = candidate
            break

    if checkpoint is not None:
        actual_hash = sha256(checkpoint)

        checks.append(
            check(
                actual_hash == EXPECTED_CHECKPOINT_SHA256,
                "Frozen Transformer checkpoint",
                {
                    "path": str(checkpoint.relative_to(ROOT)),
                    "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
                    "actual_sha256": actual_hash,
                },
            )
        )
    else:
        checks.append(
            check(
                False,
                "Frozen Transformer checkpoint",
                {"error": "Expected checkpoint not found"},
            )
        )

    # ------------------------------------------------------------
    # Phase 6 manifests
    # ------------------------------------------------------------

    phase6_paths = {
        "phase6_1": FINAL.parent / "phase6_1"
        / "phase6_1_integrated_validation_manifest.json",

        "phase6_2": FINAL.parent / "phase6_2"
        / "phase6_2_corrected_ood_manifest.json",

        "phase6_3": FINAL.parent / "phase6_3"
        / "phase6_3_scientific_review_manifest.json",

        "phase6_4": FINAL.parent / "phase6_4"
        / "phase6_4_uncertainty_calibration_manifest.json",
    }

    phase6_results = {}

    for phase, path in phase6_paths.items():
        if not path.exists():
            phase6_results[phase] = {
                "status": "MISSING",
                "expected": EXPECTED_PHASE6_STATUSES[phase],
            }
            continue

        try:
            obj = load_json(path)
            status = find_first(obj, ["Status", "status"])

            phase6_results[phase] = {
                "status": status,
                "expected": EXPECTED_PHASE6_STATUSES[phase],
            }

        except Exception as exc:
            phase6_results[phase] = {
                "status": "INVALID_JSON",
                "expected": EXPECTED_PHASE6_STATUSES[phase],
                "error": str(exc),
            }

    phase6_ok = all(
        result["status"] == result["expected"]
        for result in phase6_results.values()
    )

    checks.append(
        check(
            phase6_ok,
            "Phase 6 status preservation",
            phase6_results,
        )
    )

    # ------------------------------------------------------------
    # Final release scripts compile
    # ------------------------------------------------------------

    release_scripts = sorted(
        (ROOT / "scripts" / "final_release").glob("*.py")
    )

    compile_results = {}

    for script in release_scripts:
        try:
            py_compile.compile(
                str(script),
                doraise=True,
            )
            compile_results[
                str(script.relative_to(ROOT))
            ] = "PASS"

        except Exception as exc:
            compile_results[
                str(script.relative_to(ROOT))
            ] = f"FAIL: {exc}"

    checks.append(
        check(
            len(release_scripts) > 0
            and all(v == "PASS" for v in compile_results.values()),
            "Final release scripts compile",
            compile_results,
        )
    )

    # ------------------------------------------------------------
    # Git staging size safety
    # ------------------------------------------------------------

    staged_files = []

    try:
        import subprocess

        result = subprocess.run(
            [
                "git",
                "diff",
                "--cached",
                "--name-only",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        staged_files = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]

    except Exception:
        staged_files = []

    oversized = []

    for relative in staged_files:
        path = ROOT / relative

        if path.exists() and path.is_file():
            if path.stat().st_size >= 100 * 1024 * 1024:
                oversized.append({
                    "path": relative,
                    "size_mb": round(
                        path.stat().st_size / (1024 * 1024),
                        2,
                    ),
                })

    checks.append(
        check(
            len(oversized) == 0,
            "GitHub file-size safety",
            {
                "staged_files": len(staged_files),
                "oversized_files": oversized,
            },
        )
    )

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

    passed = sum(
        1 for item in checks
        if item["status"] == "PASS"
    )

    failed = sum(
        1 for item in checks
        if item["status"] == "FAIL"
    )

    status = "PASS" if failed == 0 else "FAIL"

    result = {
        "Status": status,
        "Checks": len(checks),
        "Passed": passed,
        "Failed": failed,
        "Completion": round(
            100.0 * passed / len(checks),
            2,
        ) if checks else 0.0,
        "ValidationType": "FINAL_RELEASE_INTEGRITY_VERIFICATION",
        "ChecksDetail": checks,
    }

    output = FINAL / "final_release_integrity_verification.json"

    with output.open("w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    print("=" * 68)
    print("FINAL RELEASE INTEGRITY VERIFICATION")
    print("=" * 68)
    print(f"Status     : {status}")
    print(f"Checks     : {len(checks)}")
    print(f"Passed     : {passed}")
    print(f"Failed     : {failed}")
    print(f"Completion : {result['Completion']:.2f}%")
    print(f"Output     : {output.relative_to(ROOT)}")
    print("=" * 68)

    print("")
    for item in checks:
        print(
            f"[{item['status']}] "
            f"{item['name']}"
        )

    print("")
    if failed == 0:
        print("FINAL RELEASE STATUS : PASS")
    else:
        print("FINAL RELEASE STATUS : FAIL")
        print("")
        print("Failed checks:")
        for item in checks:
            if item["status"] == "FAIL":
                print(
                    f"- {item['name']}: "
                    f"{item['details']}"
                )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

