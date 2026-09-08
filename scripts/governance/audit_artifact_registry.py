import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]

REGISTRY = (
    ROOT
    / "experiments"
    / "governance"
    / "artifact_registry.json"
)

REQUIRED_ARTIFACTS = {
    "phase4_20_decision_engine":
        "experiments/modeling/v2/decision_engine/crss_2024_v2_phase4_20_predictive_resilience_decision_engine.json",

    "phase4_21_edge_iot":
        "experiments/edge_iot/crss_2024_phase4_21_edge_iot_resilience.json",

    "phase4_27a_deep_ensemble":
        "experiments/modeling/v2/uncertainty/phase4_27/crss_2024_v2_phase4_27a_deep_ensemble_final.json",

    "phase4_28a_predictive_soc":
        "experiments/integration/phase4_28/crss_2024_phase4_28a_predictive_soc_event_pipeline.json",

    "phase4_28b_connectivity_recovery":
        "experiments/integration/phase4_28/crss_2024_phase4_28b_connectivity_recovery_sync.json",

    "phase4_28c_adversarial_integrity":
        "experiments/integration/phase4_28c/crss_2024_phase4_28c_adversarial_integrity_tamper_detection.json",

    "phase4_29_state_machine":
        "experiments/integration/phase4_29/crss_2024_phase4_29_resilience_state_machine_verification.json",

    "phase4_30_fault_recovery":
        "experiments/integration/phase4_30/crss_2024_phase4_30_system_fault_injection_recovery.json",

    "phase4_31_explainability":
        "experiments/integration/phase4_31/crss_2024_phase4_31_explainable_evidence_correlation.json",

    "phase4_33_explainability_stability":
        "experiments/integration/phase4_33/crss_2024_phase4_33_explainability_stability.json",

    "phase4_34_decision_validation":
        "experiments/integration/phase4_34/crss_2024_phase4_34_predictive_resilience_decision_validation.json",
}


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def inspect_artifact(key, relative_path):
    path = ROOT / relative_path

    result = {
        "key": key,
        "path": relative_path,
        "exists": path.exists(),
        "json_valid": False,
        "status": None,
        "sha256": None,
        "size_bytes": None,
    }

    if not path.exists():
        return result

    result["size_bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        result["json_valid"] = True
        result["status"] = data.get("status")

    except Exception:
        return result

    return result


def main():
    print("=== PROJECT ARTIFACT REGISTRY AUDIT ===")

    artifacts = []

    for key, relative_path in REQUIRED_ARTIFACTS.items():

        result = inspect_artifact(
            key,
            relative_path,
        )

        artifacts.append(result)

        print(
            f"{key}: "
            f"exists={result['exists']} "
            f"json={result['json_valid']} "
            f"status={result['status']}"
        )

    missing = [
        a["key"]
        for a in artifacts
        if not a["exists"]
    ]

    invalid_json = [
        a["key"]
        for a in artifacts
        if a["exists"] and not a["json_valid"]
    ]

    non_pass = [
        a["key"]
        for a in artifacts
        if a["json_valid"]
        and a["status"] != "PASS"
    ]

    report = {
        "registry_version": "1.1",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),

        "artifact_count": len(artifacts),

        "artifacts": artifacts,

        "missing_artifacts": missing,

        "invalid_json_artifacts": invalid_json,

        "non_pass_artifacts": non_pass,

        "status": (
            "PASS"
            if (
                not missing
                and not invalid_json
                and not non_pass
            )
            else "FAIL"
        ),
    }

    REGISTRY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REGISTRY.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("\n=== REGISTRY RESULT ===")

    print(
        f"Status: {report['status']}"
    )

    print(
        f"Artifacts: {len(artifacts)}"
    )

    print(
        f"Missing: {missing}"
    )

    print(
        f"Invalid JSON: {invalid_json}"
    )

    print(
        f"Non-PASS: {non_pass}"
    )

    print(
        f"Registry: {REGISTRY}"
    )

    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
