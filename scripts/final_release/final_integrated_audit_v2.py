from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"__error__": str(exc)}

def get_value(data, names):
    if isinstance(data, dict):
        for name in names:
            if name in data:
                return data[name]

        for value in data.values():
            result = get_value(value, names)
            if result is not None:
                return result

    if isinstance(data, list):
        for value in data:
            result = get_value(value, names)
            if result is not None:
                return result

    return None

def discover_manifests(phase_prefix):
    found = []

    experiments = ROOT / "experiments"

    if not experiments.exists():
        return found

    for path in experiments.rglob("*.json"):
        relative = path.relative_to(ROOT).as_posix().lower()

        if phase_prefix.lower() in relative and "manifest" in path.name.lower():
            data = load_json(path)

            found.append({
                "path": str(path.relative_to(ROOT)),
                "filename": path.name,
                "status": get_value(
                    data,
                    ["Status", "status", "release_status", "validation_status"]
                ),
                "checks": get_value(
                    data,
                    ["Checks", "checks", "check_count", "passed_checks"]
                ),
                "gate": get_value(
                    data,
                    ["Gate", "gate", "validation_gate"]
                ),
                "json_valid": "__error__" not in data,
                "size_bytes": path.stat().st_size,
            })

    return sorted(found, key=lambda x: x["path"])

def discover_phase_directories():
    result = []

    for base in [
        ROOT / "experiments",
        ROOT / "data" / "processed",
    ]:
        if not base.exists():
            continue

        for path in base.rglob("*"):
            if path.is_dir() and path.name.lower().startswith("phase"):
                result.append(str(path.relative_to(ROOT)))

    return sorted(set(result))

def required_files():
    return {
        "documentation": [
            ROOT / "README.md",
            ROOT / "docs" / "connectivity_resilience.md",
            ROOT / "docs" / "cortex_xdr_integration.md",
            ROOT / "docs" / "data_architecture.md",
            ROOT / "docs" / "research_questions.md",
            ROOT / "docs" / "safety_architecture.md",
            ROOT / "docs" / "system_architecture.md",
            ROOT / "docs" / "threat_model.md",
        ],

        "phase6_scripts": [
            ROOT / "scripts" / "phase6_1" / "integrated_research_system_validation.py",
            ROOT / "scripts" / "phase6_2" / "corrected_ood_recomputation.py",
            ROOT / "scripts" / "phase6_3" / "scientific_review_ood_validation.py",
        ],

        "phase6_outputs": [
            ROOT / "data" / "processed" / "can_hil" / "phase6_2" / "phase6_2_corrected_ood.csv",
            ROOT / "data" / "processed" / "can_hil" / "phase6_2" / "phase6_2_corrected_ood.jsonl",
            ROOT / "data" / "processed" / "can_hil" / "phase6_2" / "phase6_2_corrected_ood_schema.json",
            ROOT / "data" / "processed" / "can_hil" / "phase6_3" / "phase6_3_ood_scenario_summary.csv",
            ROOT / "data" / "processed" / "can_hil" / "phase6_3" / "phase6_3_ood_scientific_review.json",
            ROOT / "data" / "processed" / "can_hil" / "phase6_4" / "phase6_4_ood_calibration_summary.csv",
            ROOT / "data" / "processed" / "can_hil" / "phase6_4" / "phase6_4_uncertainty_calibration.json",
        ],

        "model_checkpoints": [
            ROOT / "models" / "v2" / "advanced" / "crss_2024_v2_temporal_transformer_best.pt",
        ],
    }

def check_files(paths):
    result = []

    for path in paths:
        result.append({
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        })

    return result

def main():
    phase5 = discover_manifests("phase5_")
    phase6 = discover_manifests("phase6_")

    files = required_files()

    documentation = check_files(files["documentation"])
    phase6_scripts = check_files(files["phase6_scripts"])
    phase6_outputs = check_files(files["phase6_outputs"])
    checkpoints = check_files(files["model_checkpoints"])

    for item in checkpoints:
        if item["exists"]:
            item["sha256"] = sha256_file(ROOT / item["path"])

    phase5_numbers = sorted({
        item["path"].split("/")[2]
        for item in phase5
        if len(item["path"].split("/")) > 2
        and item["path"].split("/")[2].lower().startswith("phase5_")
    })

    phase6_numbers = sorted({
        item["path"].split("/")[1]
        for item in phase6
        if len(item["path"].split("/")) > 1
        and item["path"].split("/")[1].lower().startswith("phase6_")
    })

    phase521 = [
        x for x in phase5
        if "phase5_21" in x["path"].lower()
    ]

    phase64 = [
        x for x in phase6
        if "phase6_4" in x["path"].lower()
    ]

    critical = {
        "project_root_exists": ROOT.exists(),
        "phase5_manifests_discovered": len(phase5) >= 13,
        "phase5_21_manifest_present": len(phase521) >= 1,
        "phase6_manifests_discovered": len(phase6) >= 4,
        "phase6_4_manifest_present": len(phase64) >= 1,
        "required_documentation_present": all(x["exists"] for x in documentation),
        "required_phase6_scripts_present": all(x["exists"] for x in phase6_scripts),
        "required_phase6_outputs_present": all(x["exists"] for x in phase6_outputs),
        "frozen_model_checkpoint_present": all(x["exists"] for x in checkpoints),
        "manifest_json_valid": all(
            x["json_valid"] for x in phase5 + phase6
        ),
    }

    passed = sum(bool(v) for v in critical.values())
    total = len(critical)

    result = {
        "audit": {
            "name": "FINAL_RESEARCH_RELEASE_INTEGRATED_AUDIT_V2",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(ROOT),
            "read_only": True,
        },

        "discovered_phase5_manifests": phase5,
        "discovered_phase6_manifests": phase6,

        "phase_directories": discover_phase_directories(),

        "documentation": documentation,
        "phase6_scripts": phase6_scripts,
        "phase6_outputs": phase6_outputs,
        "model_checkpoints": checkpoints,

        "critical_checks": critical,

        "summary": {
            "status": "PASS" if passed == total else "PASS_WITH_GAPS",
            "checks_passed": passed,
            "checks_total": total,
            "completion_percentage": round(
                (passed / total) * 100, 2
            ),
            "phase5_manifest_count": len(phase5),
            "phase6_manifest_count": len(phase6),
        },
    }

    return result

if __name__ == "__main__":
    result = main()

    output_dir = ROOT / "experiments" / "final_release"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "final_integrated_audit_v2.json"

    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print()
    print("=" * 80)
    print("FINAL RESEARCH RELEASE — INTEGRATED AUDIT V2")
    print("=" * 80)

    print(
        f"Status              : "
        f"{result['summary']['status']}"
    )

    print(
        f"Checks              : "
        f"{result['summary']['checks_passed']}/"
        f"{result['summary']['checks_total']}"
    )

    print(
        f"Completion          : "
        f"{result['summary']['completion_percentage']:.2f}%"
    )

    print(
        f"Phase 5 manifests   : "
        f"{result['summary']['phase5_manifest_count']}"
    )

    print(
        f"Phase 6 manifests   : "
        f"{result['summary']['phase6_manifest_count']}"
    )

    print()
    print("CRITICAL CHECKS")
    print("-" * 80)

    for name, value in result["critical_checks"].items():
        print(
            f"{name:<45} : "
            f"{'PASS' if value else 'FAIL'}"
        )

    print()
    print("PHASE 5 MANIFESTS DISCOVERED")
    print("-" * 80)

    for item in result["discovered_phase5_manifests"]:
        print(
            f"{item['path']:<75} "
            f"{item['status']}"
        )

    print()
    print("PHASE 6 MANIFESTS DISCOVERED")
    print("-" * 80)

    for item in result["discovered_phase6_manifests"]:
        print(
            f"{item['path']:<75} "
            f"{item['status']}"
        )

    print()
    print("PHASE 6 OUTPUTS")
    print("-" * 80)

    for item in result["phase6_outputs"]:
        print(
            f"{'PASS' if item['exists'] else 'FAIL':<6} "
            f"{item['path']}"
        )

    print()
    print("MODEL CHECKPOINT")
    print("-" * 80)

    for item in result["model_checkpoints"]:
        print(
            f"{'PASS' if item['exists'] else 'FAIL':<6} "
            f"{item['path']}"
        )

        if item.get("sha256"):
            print(f"       SHA256: {item['sha256']}")

    print()
    print(f"Audit JSON: {output_path.relative_to(ROOT)}")
    print("=" * 80)
