from pathlib import Path
import json
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[3]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

checks = {}
artifacts = {}

required = [
    ("phase4_20", ROOT / "experiments/modeling/v2/decision_engine/crss_2024_v2_phase4_20_predictive_resilience_decision_engine.json"),
    ("phase4_21", ROOT / "experiments/edge_iot/crss_2024_phase4_21_edge_iot_resilience.json"),
    ("phase4_22", ROOT / "experiments/edge_iot/crss_2024_phase4_22_scenario_evaluation.json"),
    ("phase4_23", ROOT / "experiments/edge_iot/crss_2024_phase4_23_end_to_end_resilience.json"),
    ("phase4_24", ROOT / "experiments/edge_iot/crss_2024_phase4_24_system_stress_failure_analysis.json"),
    ("phase4_25", ROOT / "experiments/modeling/v2/evidence/crss_2024_v2_phase4_25_evidence_manifest.json"),
    ("phase4_26", ROOT / "experiments/modeling/v2/error_analysis/phase4_26/crss_2024_v2_phase4_26_temporal_regime_error_analysis.json"),
    ("phase4_27a", ROOT / "experiments/modeling/v2/uncertainty/phase4_27/crss_2024_v2_phase4_27a_deep_ensemble_final.json"),
    ("phase4_27b", ROOT / "experiments/modeling/v2/uncertainty/phase4_27b/crss_2024_v2_phase4_27b_ood_regime_uncertainty_final.json"),
    ("phase4_28a", ROOT / "experiments/integration/phase4_28/crss_2024_phase4_28a_predictive_soc_event_pipeline.json"),
    ("phase4_28b", ROOT / "experiments/integration/phase4_28/crss_2024_phase4_28b_connectivity_recovery_sync.json"),
    ("phase4_28c", ROOT / "experiments/integration/phase4_28c/crss_2024_phase4_28c_adversarial_integrity_tamper_detection.json"),
    ("phase4_29", ROOT / "experiments/integration/phase4_29/crss_2024_phase4_29_resilience_state_machine_verification.json"),
    ("phase4_30", ROOT / "experiments/integration/phase4_30/crss_2024_phase4_30_system_fault_injection_recovery.json"),
    ("phase4_31", ROOT / "experiments/integration/phase4_31/crss_2024_phase4_31_explainable_evidence_correlation.json"),
    ("phase4_32", ROOT / "experiments/integration/phase4_32/crss_2024_phase4_32_model_grounded_explainability.json"),
    ("phase4_32b2", ROOT / "experiments/integration/phase4_32b/crss_2024_phase4_32b2_origin_balanced_model_grounded_explainability.json"),
    ("phase4_32c", ROOT / "experiments/integration/phase4_32c/crss_2024_phase4_32c_modality_grounded_explainability.json"),
    ("phase4_33", ROOT / "experiments/integration/phase4_33/crss_2024_phase4_33_explainability_stability.json"),
    ("phase4_34", ROOT / "experiments/integration/phase4_34/crss_2024_phase4_34_predictive_resilience_decision_validation.json"),
    ("phase4_35", ROOT / "experiments/integration/phase4_35/crss_2024_phase4_35_end_to_end_adversarial_resilience_validation.json"),
    ("phase4_36", ROOT / "experiments/integration/phase4_36/crss_2024_phase4_36_failure_mode_safety_boundary_audit.json"),
    ("phase4_37", ROOT / "experiments/integration/phase4_37/crss_2024_phase4_37_final_integrated_system_validation.json"),
]

for name, path in required:
    exists = path.exists()
    checks[f"{name}_exists"] = exists

    if exists:
        try:
            data = load_json(path)
            artifacts[name] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "json_valid": True,
                "status": data.get("status"),
                "checks": data.get("checks"),
            }
            checks[f"{name}_json_valid"] = True
        except Exception as e:
            artifacts[name] = {
                "path": str(path.relative_to(ROOT)),
                "json_valid": False,
                "error": str(e),
            }
            checks[f"{name}_json_valid"] = False

# Critical final validation
p437 = required[-1][1]
p437_data = load_json(p437) if p437.exists() else {}
checks["phase4_37_status_pass"] = p437_data.get("status") == "PASS"
checks["phase4_37_checks_32_of_32"] = (isinstance(p437_data.get("checks"), list) and len(p437_data.get("checks")) == 32 and all(item.get("passed") is True for item in p437_data.get("checks")))

# Canonical explainability schema
schema = ROOT / "data/schemas/explainability/model_grounded_resilience_explanation_schema.json"
checks["canonical_explainability_schema_exists"] = schema.exists()
if schema.exists():
    try:
        load_json(schema)
        checks["canonical_explainability_schema_valid"] = True
        artifacts["canonical_explainability_schema"] = {
            "path": str(schema.relative_to(ROOT)),
            "sha256": sha256(schema),
            "json_valid": True,
        }
    except Exception:
        checks["canonical_explainability_schema_valid"] = False
else:
    checks["canonical_explainability_schema_valid"] = False

# Core modeling artifacts
model_files = [
    ROOT / "models/v2/uncertainty/phase4_27/transformer_y6_seed_42_best.pt",
    ROOT / "models/v2/uncertainty/phase4_27/transformer_y6_seed_52_best.pt",
    ROOT / "models/v2/uncertainty/phase4_27/transformer_y6_seed_62_best.pt",
]

checks["all_frozen_model_checkpoints_present"] = all(p.exists() for p in model_files)

for p in model_files:
    if p.exists():
        artifacts[p.name] = {
            "path": str(p.relative_to(ROOT)),
            "sha256": sha256(p),
            "size_bytes": p.stat().st_size,
        }

# Dataset/tensor artifacts
tensor_dir = ROOT / "data/processed/modeling/sequences/v2_forecasting"
checks["forecasting_tensor_directory_exists"] = tensor_dir.exists()

tensor_names = [
    "X_train_v2_forecasting.npy",
    "X_validation_v2_forecasting.npy",
    "X_test_v2_forecasting.npy",
    "y_3step_train_v2_forecasting.npy",
    "y_3step_validation_v2_forecasting.npy",
    "y_3step_test_v2_forecasting.npy",
    "y_6step_train_v2_forecasting.npy",
    "y_6step_validation_v2_forecasting.npy",
    "y_6step_test_v2_forecasting.npy",
    "scenario_ids_train_v2_forecasting.npy",
    "scenario_ids_validation_v2_forecasting.npy",
    "scenario_ids_test_v2_forecasting.npy",
    "prediction_origins_train_v2_forecasting.npy",
    "prediction_origins_validation_v2_forecasting.npy",
    "prediction_origins_test_v2_forecasting.npy",
]

checks["all_forecasting_tensors_present"] = all(
    (tensor_dir / n).exists() for n in tensor_names
)

# Evidence digest over audit results
digest_payload = json.dumps(
    {"checks": checks, "artifacts": artifacts},
    sort_keys=True,
    default=str,
).encode("utf-8")

evidence_digest = hashlib.sha256(digest_payload).hexdigest()

status = "PASS" if all(checks.values()) else "FAIL"

report = {
    "phase": "4.38",
    "title": "Reproducibility and Evidence Audit",
    "status": status,
    "checks_passed": sum(bool(v) for v in checks.values()),
    "checks_total": len(checks),
    "checks": checks,
    "artifacts": artifacts,
    "evidence_digest": evidence_digest,
    "scientific_boundary": {
        "retraining_performed": False,
        "test_set_used_for_calibration": False,
        "synthetic_cyber_labels": True,
        "real_world_cyberattack_claim": False,
        "live_cortex_xdr_integration_claim": False,
    },
}

out = ROOT / "experiments/integration/phase4_38/crss_2024_phase4_38_reproducibility_evidence_audit.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n=== PHASE 4.38 RESULT ===")
print("Status:", status)
print(f"Checks: {report['checks_passed']}/{report['checks_total']}")
if status != "PASS":
    print("Failed checks:")
    for k, v in checks.items():
        if not v:
            print("  -", k)
print("Evidence digest:", evidence_digest)
print("Report:", out)

if status != "PASS":
    sys.exit(1)


