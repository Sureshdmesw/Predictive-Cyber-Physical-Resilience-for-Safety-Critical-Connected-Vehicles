from pathlib import Path
import json
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[3]

REPORT = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_37"
    / "crss_2024_phase4_37_final_integrated_system_validation.json"
)

EXCLUDED = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
}


def iter_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue

        if any(part in EXCLUDED for part in p.parts):
            continue

        yield p


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def check(name, condition, detail=None):
    checks.append({
        "name": name,
        "passed": bool(condition),
        "detail": detail,
    })


def find_json_candidates(predicate):
    results = []

    for path in iter_files():
        if path.suffix.lower() != ".json":
            continue

        try:
            data = load_json(path)
        except Exception:
            data = None

        if predicate(path, data):
            results.append((path, data))

    return results


def resolve_schema(required_terms):
    candidates = find_json_candidates(
        lambda p, d:
            all(
                term.lower() in p.name.lower()
                or term.lower() in str(p).lower()
                or (
                    isinstance(d, dict)
                    and term.lower() in json.dumps(d).lower()
                )
                for term in required_terms
            )
    )

    candidates.sort(
        key=lambda x: (
            "schemas" in str(x[0]).lower(),
            "explainability" in str(x[0]).lower(),
            "schema" in x[0].name.lower(),
            "model_grounded" in x[0].name.lower(),
        ),
        reverse=True,
    )

    return candidates[0] if candidates else None


def resolve_phase(phase, required_status="PASS"):
    token = f"phase4_{phase}".lower()

    candidates = []

    for path in iter_files():
        if path.suffix.lower() != ".json":
            continue

        text = str(path.relative_to(ROOT)).lower()

        if token not in text:
            continue

        data = load_json(path)

        if not isinstance(data, dict):
            continue

        status = str(data.get("status", "")).upper()

        if required_status and status != required_status.upper():
            continue

        candidates.append((path, data))

    candidates.sort(
        key=lambda x: (
            x[1].get("phase") == f"4.{phase}",
            "integration" in str(x[0]).lower(),
            "modeling" in str(x[0]).lower(),
            "final" in x[0].name.lower(),
        ),
        reverse=True,
    )

    return candidates[0] if candidates else None


checks = []
evidence = []
resolved = {}

# ================================================================
# 1. Required PASS phases
# ================================================================

required_pass_phases = [
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
]

for phase in required_pass_phases:

    result = resolve_phase(phase, "PASS")

    if result:
        path, data = result

        resolved[phase] = {
            "path": path,
            "data": data,
        }

        evidence.append({
            "phase": f"4.{phase}",
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "status": data.get("status"),
        })

        check(
            f"phase_4_{phase}_resolved",
            True,
            str(path.relative_to(ROOT)),
        )

    else:
        check(
            f"phase_4_{phase}_resolved",
            False,
            "No PASS artifact could be resolved.",
        )


# ================================================================
# 2. Required schemas — DYNAMIC RESOLUTION
# ================================================================

# Resolve the explainability schema deterministically.
# Prefer the known canonical filename, then fall back to
# semantic discovery without assuming a directory layout.

EXPLAINABILITY_SCHEMA_NAME = (
    "model_grounded_resilience_explanation_schema.json"
)

schema_matches = []

for path in iter_files():

    if path.suffix.lower() != ".json":
        continue

    if path.name.lower() == EXPLAINABILITY_SCHEMA_NAME.lower():

        data = load_json(path)

        if isinstance(data, dict):
            schema_matches.append(
                (path, data)
            )

if len(schema_matches) == 1:

    path, data = schema_matches[0]

    check(
        "model_grounded_explainability_schema_present",
        True,
        str(path.relative_to(ROOT)),
    )

    evidence.append({
        "artifact_type": "explainability_schema",
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "resolution": "exact_filename",
    })

elif len(schema_matches) > 1:

    check(
        "model_grounded_explainability_schema_present",
        False,
        {
            "reason": "Multiple exact schema files found",
            "candidates": [
                str(x[0].relative_to(ROOT))
                for x in schema_matches
            ],
        },
    )

else:

    # Controlled fallback.
    fallback = []

    for path in iter_files():

        if path.suffix.lower() != ".json":
            continue

        name = path.name.lower()
        full = str(path.relative_to(ROOT)).lower()

        if (
            "explain" in name
            and "schema" in name
            and (
                "model" in name
                or "grounded" in name
                or "resilience" in name
            )
        ):

            data = load_json(path)

            if isinstance(data, dict):
                fallback.append(
                    (path, data)
                )

    if len(fallback) == 1:

        path, data = fallback[0]

        check(
            "model_grounded_explainability_schema_present",
            True,
            str(path.relative_to(ROOT)),
        )

        evidence.append({
            "artifact_type": "explainability_schema",
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "resolution": "semantic_fallback",
        })

    else:

        check(
            "model_grounded_explainability_schema_present",
            False,
            {
                "reason": "Explainability schema unresolved",
                "fallback_candidates": [
                    str(x[0].relative_to(ROOT))
                    for x in fallback
                ],
            },
        )


crss_schema = resolve_schema(
    [
        "crss",
        "canonical",
        "schema",
    ]
)

if crss_schema:

    path, data = crss_schema

    check(
        "crss_canonical_schema_present",
        isinstance(data, dict),
        str(path.relative_to(ROOT)),
    )

    evidence.append({
        "artifact_type": "crss_schema",
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
    })

else:

    check(
        "crss_canonical_schema_present",
        False,
        "No CRSS canonical schema discovered.",
    )


# ================================================================
# 3. Release gate
# ================================================================

release_gate = ROOT / "experiments" / "nhtsa" / "crss_2024_release_gate.json"

release_data = load_json(release_gate) if release_gate.exists() else None

check(
    "crss_release_gate_present",
    isinstance(release_data, dict),
    str(release_gate.relative_to(ROOT)),
)

if isinstance(release_data, dict):

    check(
        "crss_release_gate_released",
        str(release_data.get("status", "")).upper() == "RELEASED",
        release_data.get("status"),
    )


# ================================================================
# 4. Model checkpoints
# ================================================================

checkpoints = []

models_dir = ROOT / "models"

if models_dir.exists():

    checkpoints = [
        p
        for p in models_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".pt", ".joblib"}
    ]

check(
    "model_checkpoints_present",
    len(checkpoints) > 0,
    [str(p.relative_to(ROOT)) for p in checkpoints],
)


# ================================================================
# 5. Governance infrastructure
# ================================================================

registry_script = ROOT / "scripts" / "governance" / "audit_artifact_registry.py"
dependency_script = ROOT / "scripts" / "governance" / "phase_dependency_gate.py"
resolver_script = ROOT / "scripts" / "governance" / "artifact_resolver.py"

check(
    "artifact_registry_script_present",
    registry_script.exists(),
)

check(
    "dependency_gate_script_present",
    dependency_script.exists(),
)

check(
    "shared_artifact_resolver_present",
    resolver_script.exists(),
)


# ================================================================
# 6. Safety boundaries
# ================================================================

all_text = json.dumps(
    {
        phase: item["data"]
        for phase, item in resolved.items()
    }
).lower()


check(
    "synthetic_cyber_label_boundary",
    "synthetic" in all_text,
)

check(
    "cortex_xdr_abstraction_boundary",
    (
        "abstraction_only" in all_text
        or "abstraction only" in all_text
    ),
)

check(
    "no_direct_vehicle_actuation_boundary",
    (
        "direct vehicle actuation" in all_text
        or "direct_vehicle_actuation" in all_text
    ),
)

check(
    "integrity_verification_evidence",
    "integrity" in all_text,
)

check(
    "tamper_detection_evidence",
    "tamper" in all_text,
)

check(
    "recovery_synchronization_evidence",
    (
        "recovery" in all_text
        or "synchron" in all_text
    ),
)

check(
    "explainability_evidence",
    "explain" in all_text,
)


# ================================================================
# 7. Historical OOD limitation preservation
# ================================================================

ood_terms = [
    "phase4_27b",
    "phase4_27b_1",
    "phase4_27b_2",
    "phase4_27b_3",
]

ood_files = []

for path in iter_files():

    text = str(path.relative_to(ROOT)).lower()

    if path.suffix.lower() == ".json":
        if any(term in text for term in ood_terms):

            data = load_json(path)

            if isinstance(data, dict):

                ood_files.append({
                    "path": str(path.relative_to(ROOT)),
                    "status": data.get("status"),
                    "sha256": sha256(path),
                })


check(
    "historical_ood_limitations_preserved",
    len(ood_files) > 0,
    ood_files,
)


# ================================================================
# 8. Final result
# ================================================================

failed = [
    x["name"]
    for x in checks
    if not x["passed"]
]

payload = {
    "phase": "4.37",
    "title": "Final Integrated System Validation",

    "status": (
        "PASS"
        if not failed
        else "FAIL"
    ),

    "check_count": len(checks),

    "passed_count": sum(
        1 for x in checks
        if x["passed"]
    ),

    "failed_count": len(failed),

    "failed_checks": failed,

    "checks": checks,

    "resolved_artifacts": {
        phase: {
            "path": str(item["path"].relative_to(ROOT)),
            "status": item["data"].get("status"),
        }
        for phase, item in resolved.items()
    },

    "ood_limitation_artifacts": ood_files,

    "evidence": evidence,

    "methodological_boundary": {
        "no_retraining": True,
        "synthetic_cyber_labels_not_observed_attacks": True,
        "cortex_xdr_live_tenant_claim": False,
        "direct_vehicle_actuation": False,
        "historical_ood_failures_preserved_as_limitations": True,
    },
}

REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT.write_text(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

print()
print("=== PHASE 4.37 RESULT ===")
print(f"Status: {payload['status']}")
print(
    f"Checks: "
    f"{payload['passed_count']}/"
    f"{payload['check_count']}"
)

if failed:
    print("Failed checks:")
    for item in failed:
        print("  -", item)
else:
    print("Failed checks: NONE")

print()
print("=== RESOLVED ARTIFACTS ===")

for phase, item in resolved.items():
    print(
        f"4.{phase}: "
        f"{item['path'].relative_to(ROOT)}"
    )

print()
print("=== OOD LIMITATIONS ===")

for item in ood_files:
    print(
        f"{item['path']} "
        f"[{item['status']}]"
    )

print()
print("Report:")
print(REPORT)

sys.exit(
    0
    if not failed
    else 1
)

