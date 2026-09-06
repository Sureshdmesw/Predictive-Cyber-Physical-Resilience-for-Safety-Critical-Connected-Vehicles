from pathlib import Path
import json
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS = {
    "phase_4_9_strong_baseline":
        PROJECT_ROOT / "experiments/modeling/v2/baseline/crss_2024_v2_strong_baseline_results.json",

    "phase_4_10_transformer":
        PROJECT_ROOT / "experiments/modeling/v2/advanced/crss_2024_v2_temporal_transformer_controlled_final.json",

    "phase_4_10b_label_separability":
        PROJECT_ROOT / "experiments/modeling/v2/advanced/crss_2024_v2_synthetic_label_separability_audit.json",

    "phase_4_10c_robustness_ablation":
        PROJECT_ROOT / "experiments/modeling/v2/advanced/crss_2024_v2_robustness_ablation_final.json",

    "phase_4_11_cross_modal":
        PROJECT_ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_cross_modal_results.json",

    "phase_4_12_modality_contribution":
        PROJECT_ROOT / "experiments/modeling/v2/modality_contribution/crss_2024_v2_modality_contribution_results.json",

    "phase_4_13_feature_contribution":
        PROJECT_ROOT / "experiments/modeling/v2/feature_contribution/crss_2024_v2_feature_contribution_results.json",

    "phase_4_14_optimized_features":
        PROJECT_ROOT / "experiments/modeling/v2/optimized_features/crss_2024_v2_optimized_feature_results.json",

    "phase_4_15_leakage_stress":
        PROJECT_ROOT / "experiments/modeling/v2/generalization/crss_2024_v2_generalization_leakage_stress_test.json",

    "phase_4_16_generalization":
        PROJECT_ROOT / "experiments/modeling/v2/generalization/phase4_16/crss_2024_v2_phase4_16_generalization_results.json",

    "phase_4_17_error_analysis":
        PROJECT_ROOT / "experiments/modeling/v2/error_analysis/crss_2024_v2_phase4_17_error_analysis.json",

    "phase_4_18_calibration":
        PROJECT_ROOT / "experiments/modeling/v2/calibration/crss_2024_v2_phase4_18_calibration_thresholds.json",

    "phase_4_19_robustness_confirmation":
        PROJECT_ROOT / "experiments/modeling/v2/robustness/crss_2024_v2_phase4_19_robustness_confirmation.json",

    "phase_4_20_decision_engine":
        PROJECT_ROOT / "experiments/modeling/v2/decision_engine/crss_2024_v2_phase4_20_predictive_resilience_decision_engine.json",

    "phase_4_21_edge_iot":
        PROJECT_ROOT / "experiments/edge_iot/crss_2024_phase4_21_edge_iot_resilience.json",

    "phase_4_22_edge_iot_scenarios":
        PROJECT_ROOT / "experiments/edge_iot/crss_2024_phase4_22_scenario_evaluation.json",

    "phase_4_23_end_to_end":
        PROJECT_ROOT / "experiments/edge_iot/crss_2024_phase4_23_end_to_end_resilience.json",

    "phase_4_24_stress_failure":
        PROJECT_ROOT / "experiments/edge_iot/crss_2024_phase4_24_system_stress_failure_analysis.json",
}

OUTPUT_DIR = PROJECT_ROOT / "experiments/modeling/v2/evidence"
OUTPUT_PATH = OUTPUT_DIR / "crss_2024_v2_phase4_25_evidence_manifest.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    evidence = {}
    missing = []

    for name, path in REPORTS.items():
        if not path.exists():
            missing.append({
                "name": name,
                "path": str(path),
            })
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))

            evidence[name] = {
                "path": str(path),
                "status": data.get("status"),
                "phase": data.get("phase"),
                "title": data.get("title"),
            }

        except Exception as exc:
            missing.append({
                "name": name,
                "path": str(path),
                "error": str(exc),
            })

    all_present = len(missing) == 0
    all_pass = all(
        item.get("status") == "PASS"
        for item in evidence.values()
    )

    report = {
        "phase": "4.25",
        "title": "Research Evidence and Comparative Evaluation — Evidence Manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_present and all_pass else "FAIL",
        "retraining_performed": False,
        "evidence_count": len(evidence),
        "missing_count": len(missing),
        "evidence": evidence,
        "missing": missing,
        "methodological_boundary": {
            "cyber_labels_are_synthetic": True,
            "crss_is_physical_safety_ground_truth": True,
            "cyber_telemetry_is_simulated": True,
            "cortex_xdr_is_research_abstraction": True,
            "direct_vehicle_actuation": False,
            "model_retraining": False,
        },
        "evaluation_principles": [
            "Keep previously validated models frozen.",
            "Separate synthetic-label separability from generalization evidence.",
            "Do not treat CRSS as a cyberattack dataset.",
            "Do not interpret Cortex XDR abstraction as live production integration.",
            "Do not claim direct safety-critical vehicle control.",
            "Preserve temporal generalization failures as research findings.",
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("PHASE 4.25 EVIDENCE MANIFEST")
    print("=" * 72)
    print(f"Status: {report['status']}")
    print(f"Evidence reports found: {len(evidence)}")
    print(f"Missing reports: {len(missing)}")
    print(f"Retraining: {report['retraining_performed']}")

    if missing:
        print("\nMISSING:")
        for item in missing:
            print(f"  {item['name']}: {item['path']}")

    print(f"\nReport: {OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
