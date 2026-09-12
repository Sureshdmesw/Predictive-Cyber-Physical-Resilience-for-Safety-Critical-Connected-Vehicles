from pathlib import Path
import json
import re
from datetime import datetime, timezone


ROOT = Path(
    r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles"
)

OUTPUT_DIR = ROOT / "experiments" / "final_release"

OUTPUT_JSON = (
    OUTPUT_DIR
    / "final_claims_limitations_audit.json"
)

EVIDENCE_JSON = (
    OUTPUT_DIR
    / "final_evidence_consolidation.json"
)


EXPECTED_PHASE6_STATUSES = {
    "phase6_1": "PASS_WITH_SCIENTIFIC_REVIEW_REQUIRED",
    "phase6_2": "PASS",
    "phase6_3": "SCIENTIFIC_REVIEW_ACCEPTED_WITH_LIMITATIONS",
    "phase6_4": "PASS_WITH_CALIBRATION_LIMITATIONS",
}


LIMITATIONS = {
    "synthetic_cyber_telemetry_not_real_world_attack_evidence": [
        "synthetic cyber telemetry",
        "synthetic",
        "real-world cyberattack",
        "real world cyberattack",
        "cyberattack labels",
    ],
    "no_physical_vehicle_validation": [
        "physical vehicle",
        "physical deployment",
        "road validation",
        "road testing",
        "vehicle deployment",
    ],
    "no_production_ecu_can": [
        "production ECU",
        "production CAN",
        "production vehicle",
        "production deployment",
        "ECU/CAN",
    ],
    "no_live_cortex_xdr": [
        "Cortex XDR",
        "live Cortex XDR",
        "live API",
        "API deployment",
    ],
    "no_direct_vehicle_actuation": [
        "direct actuation",
        "vehicle actuation",
        "direct vehicle control",
        "actuation",
    ],
    "no_iso_26262_asil_certification": [
        "ISO 26262",
        "ASIL",
        "certification",
        "certified",
    ],
    "ood_is_incomplete": [
        "OOD",
        "out-of-distribution",
        "uncertainty",
    ],
    "ood_calibration_has_limitations": [
        "calibration",
        "limitation",
        "scientific review",
    ],
}


# These patterns intentionally detect only affirmative unsupported claims.
# Explicit limitation language such as "does not claim" is therefore safe.
UNSUPPORTED_AFFIRMATIVE_PATTERNS = [
    r"\b(?:we|this project|the project|the system|the model)\s+(?:have|has|is|are|was|were)\s+(?:proven|validated|certified)\s+[^.\n]{0,120}\breal[-\s]?world\b",
    r"\b(?:we|this project|the project|the system|the model)\s+(?:is|are)\s+production[-\s]?ready\b",
    r"\bproduction\s+vehicle\s+deployment\s+(?:has\s+been\s+)?validated\b",
    r"\blive\s+cortex\s+xdr\s+deployment\s+(?:has\s+been\s+)?(?:validated|completed|demonstrated)\b",
    r"\b(?:we|this project|the project|the system)\s+(?:is|are)\s+iso\s*26262\s+certified\b",
    r"\b(?:we|this project|the project|the system)\s+(?:is|are)\s+asil\s+certified\b",
    r"\broad[-\s]?validated\s+vehicle\s+system\b",
]


def load_text(path):
    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return ""


def load_json(path):
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except Exception:
        return None


def add_check(checks, name, passed, details):
    checks.append(
        {
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )


def discover_docs():
    paths = []

    docs = ROOT / "docs"

    if docs.exists():
        for path in docs.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in {
                    ".md",
                    ".txt",
                    ".rst",
                }
            ):
                paths.append(path)

    readme = ROOT / "README.md"

    if readme.exists():
        paths.append(readme)

    return sorted(set(paths))


def discover_manifests():
    manifests = []

    experiments = ROOT / "experiments"

    if experiments.exists():
        for path in experiments.rglob("*manifest.json"):
            manifests.append(path)

    return sorted(set(manifests))


def combined_text(paths):
    chunks = []

    for path in paths:
        text = load_text(path)

        if text:
            chunks.append(
                f"\n===== {path.relative_to(ROOT)} =====\n"
            )
            chunks.append(text)

    return "\n".join(chunks)


def find_terms(text, terms):
    found = []

    lower_text = text.lower()

    for term in terms:
        if term.lower() in lower_text:
            found.append(term)

    return found


def audit_limitation_concepts(docs, manifests):
    documentation_text = combined_text(docs)
    manifest_text = combined_text(manifests)

    results = {}

    for name, terms in LIMITATIONS.items():

        doc_found = find_terms(
            documentation_text,
            terms,
        )

        manifest_found = find_terms(
            manifest_text,
            terms,
        )

        evidence = []

        if doc_found:
            evidence.append("documentation")

        if manifest_found:
            evidence.append("manifests")

        results[name] = {
            "documented_terms_found": doc_found,
            "manifest_terms_found": manifest_found,
            "evidence_sources": evidence,
            "supported": len(evidence) > 0,
        }

    return results


def audit_phase6_statuses(manifests):
    text = combined_text(manifests)

    results = {}

    for phase, expected in EXPECTED_PHASE6_STATUSES.items():

        found = expected.lower() in text.lower()

        results[phase] = {
            "expected": expected,
            "found": found,
        }

    return results


def audit_unsupported_claims(docs):
    text = combined_text(docs)

    results = {}

    for pattern in UNSUPPORTED_AFFIRMATIVE_PATTERNS:

        matches = []

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            matches.append(
                match.group(0)
            )

        results[pattern] = matches

    return results


def audit_evidence_consolidation():

    if not EVIDENCE_JSON.exists():
        return {
            "exists": False,
            "valid": False,
            "records": 0,
        }

    data = load_json(EVIDENCE_JSON)

    if not isinstance(data, dict):
        return {
            "exists": True,
            "valid": False,
            "records": 0,
        }

    records = data.get(
        "evidence_records",
        [],
    )

    return {
        "exists": True,
        "valid": True,
        "records": len(records),
    }


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
            2,
        ) if total else 0.0,
    }


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checks = []

    docs = discover_docs()
    manifests = discover_manifests()

    evidence = audit_evidence_consolidation()

    add_check(
        checks,
        "evidence_consolidation_valid",
        evidence["exists"]
        and evidence["valid"]
        and evidence["records"] > 0,
        f"records={evidence['records']}",
    )

    add_check(
        checks,
        "documentation_available",
        len(docs) > 0,
        f"documentation_files={len(docs)}",
    )

    phase6_manifests = [
        path
        for path in manifests
        if "phase6_" in (
            path.relative_to(ROOT)
            .as_posix()
            .lower()
        )
    ]

    add_check(
        checks,
        "phase6_manifests_available",
        len(phase6_manifests) >= 4,
        f"phase6_manifests={len(phase6_manifests)}",
    )

    limitation_results = audit_limitation_concepts(
        docs,
        phase6_manifests,
    )

    unsupported_limitation_concepts = [
        name
        for name, result
        in limitation_results.items()
        if not result["supported"]
    ]

    add_check(
        checks,
        "research_limitations_supported_by_evidence",
        len(unsupported_limitation_concepts) == 0,
        f"unsupported={unsupported_limitation_concepts}",
    )

    phase6_status_results = audit_phase6_statuses(
        phase6_manifests,
    )

    missing_statuses = [
        phase
        for phase, result
        in phase6_status_results.items()
        if not result["found"]
    ]

    add_check(
        checks,
        "phase6_statuses_preserved",
        len(missing_statuses) == 0,
        f"missing={missing_statuses}",
    )

    docs_text = combined_text(docs)

    synthetic_present = (
        "synthetic" in docs_text.lower()
    )

    add_check(
        checks,
        "synthetic_telemetry_boundary_supported",
        synthetic_present,
        f"documentation_mentions_synthetic={synthetic_present}",
    )

    cortex_present = (
        "cortex xdr" in docs_text.lower()
    )

    add_check(
        checks,
        "cortex_xdr_architecture_boundary_supported",
        cortex_present,
        f"documentation_mentions_cortex_xdr={cortex_present}",
    )

    ood_present = (
        re.search(
            r"\bood\b|out[-\s]?of[-\s]?distribution",
            docs_text,
            flags=re.IGNORECASE,
        )
        is not None
    )

    calibration_present = (
        "calibration" in docs_text.lower()
        or "uncertainty" in docs_text.lower()
    )

    add_check(
        checks,
        "ood_and_calibration_boundary_supported",
        ood_present and calibration_present,
        (
            f"ood={ood_present}, "
            f"calibration_or_uncertainty={calibration_present}"
        ),
    )

    unsupported_claims = audit_unsupported_claims(
        docs,
    )

    present_claims = {
        pattern: matches
        for pattern, matches
        in unsupported_claims.items()
        if matches
    }

    add_check(
        checks,
        "no_unsupported_unqualified_claims",
        len(present_claims) == 0,
        f"unsupported_claim_patterns={len(present_claims)}",
    )

    limitation_statuses = [
        EXPECTED_PHASE6_STATUSES["phase6_1"],
        EXPECTED_PHASE6_STATUSES["phase6_3"],
        EXPECTED_PHASE6_STATUSES["phase6_4"],
    ]

    status_text = combined_text(
        phase6_manifests,
    )

    preserved_limited_statuses = [
        status
        for status in limitation_statuses
        if status.lower() in status_text.lower()
    ]

    add_check(
        checks,
        "limited_phase6_results_not_upgraded",
        len(preserved_limited_statuses)
        == len(limitation_statuses),
        (
            f"preserved={len(preserved_limited_statuses)}/"
            f"{len(limitation_statuses)}"
        ),
    )

    summary = build_summary(checks)

    result = {
        "claims_limitations_audit": {
            "name": (
                "FINAL_RESEARCH_RELEASE_"
                "CLAIMS_LIMITATIONS_AUDIT"
            ),
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "project_root": str(ROOT),
            "read_only_source_validation": True,
            "audit_method": (
                "evidence_and_governance_based_claim_validation"
            ),
            "unsupported_claim_detection": (
                "affirmative_claim_patterns_only"
            ),
        },
        "summary": summary,
        "checks": checks,
        "limitation_results": limitation_results,
        "phase6_status_results": phase6_status_results,
        "unsupported_claim_results": unsupported_claims,
        "documentation_files": [
            str(path.relative_to(ROOT))
            for path in docs
        ],
        "phase6_manifests": [
            str(path.relative_to(ROOT))
            for path in phase6_manifests
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(
        "FINAL RESEARCH RELEASE — "
        "CLAIMS & LIMITATIONS AUDIT"
    )
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
    print("CLAIMS / LIMITATIONS CHECKS")
    print("-" * 80)

    for check in checks:
        print(
            f"{check['status']:<6} "
            f"{check['name']:<55} "
            f"{check['details']}"
        )

    print()
    print("OUTPUT")
    print("-" * 80)
    print(
        OUTPUT_JSON.relative_to(ROOT)
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
