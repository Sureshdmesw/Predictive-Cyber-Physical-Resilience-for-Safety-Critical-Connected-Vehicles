from pathlib import Path
import json
import hashlib
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

checks = {}
manifest = {}

# Final audit reports
p38 = ROOT / "experiments/integration/phase4_38/crss_2024_phase4_38_reproducibility_evidence_audit.json"
p39 = ROOT / "experiments/integration/phase4_39/crss_2024_phase4_39_research_governance_safety_claims_audit.json"
p37 = ROOT / "experiments/integration/phase4_37/crss_2024_phase4_37_final_integrated_system_validation.json"

for label, p in [
    ("phase4_37", p37),
    ("phase4_38", p38),
    ("phase4_39", p39),
]:
    checks[f"{label}_report_present"] = p.exists()

    if p.exists():
        d = load(p)
        checks[f"{label}_status_pass"] = d.get("status") == "PASS"
        manifest[label] = {
            "path": str(p.relative_to(ROOT)),
            "sha256": sha256(p),
            "status": d.get("status"),
        }

# Canonical schema
schema = ROOT / "data/schemas/explainability/model_grounded_resilience_explanation_schema.json"
checks["canonical_explainability_schema_present"] = schema.exists()

if schema.exists():
    try:
        load(schema)
        checks["canonical_explainability_schema_valid"] = True
        manifest["canonical_explainability_schema"] = {
            "path": str(schema.relative_to(ROOT)),
            "sha256": sha256(schema),
        }
    except Exception:
        checks["canonical_explainability_schema_valid"] = False

# Governance registry
registry = ROOT / "experiments/governance/artifact_registry.json"
checks["artifact_registry_present"] = registry.exists()

if registry.exists():
    try:
        d = load(registry)
        checks["artifact_registry_status_pass"] = d.get("status") == "PASS"
        manifest["artifact_registry"] = {
            "path": str(registry.relative_to(ROOT)),
            "sha256": sha256(registry),
            "status": d.get("status"),
        }
    except Exception:
        checks["artifact_registry_status_pass"] = False

# Frozen checkpoints
models = [
    ROOT / "models/v2/uncertainty/phase4_27/transformer_y6_seed_42_best.pt",
    ROOT / "models/v2/uncertainty/phase4_27/transformer_y6_seed_52_best.pt",
    ROOT / "models/v2/uncertainty/phase4_27/transformer_y6_seed_62_best.pt",
]

checks["all_frozen_checkpoints_present"] = all(p.exists() for p in models)

for p in models:
    if p.exists():
        manifest[p.name] = {
            "path": str(p.relative_to(ROOT)),
            "sha256": sha256(p),
            "size_bytes": p.stat().st_size,
        }

# Forecasting tensors
tensor_dir = ROOT / "data/processed/modeling/sequences/v2_forecasting"

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

# No unresolved merge conflicts
conflicts = []
for p in ROOT.rglob("*"):
    if (
        p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and p != Path(__file__).resolve()
        and p.suffix.lower() not in {".pyc", ".pyo"}
    ):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "<<<<<<< HEAD" in text or ">>>>>>> " in text:
                conflicts.append(str(p.relative_to(ROOT)))
        except Exception:
            pass

checks["no_unresolved_merge_conflicts"] = len(conflicts) == 0

# Git state snapshot
try:
    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
except Exception:
    git_status = []

manifest["git_status_before_final_commit"] = git_status

# Release boundary
release_boundary = {
    "research_release": True,
    "production_vehicle_deployment": False,
    "direct_vehicle_actuation": False,
    "live_cortex_xdr_api": False,
    "real_world_cyberattack_labels": False,
    "synthetic_cyber_telemetry": True,
    "test_set_calibration": False,
    "final_release_retraining": False,
}

payload = json.dumps(
    {
        "checks": checks,
        "manifest": manifest,
        "release_boundary": release_boundary,
    },
    sort_keys=True,
    default=str,
).encode("utf-8")

digest = hashlib.sha256(payload).hexdigest()

status = "PASS" if all(checks.values()) else "FAIL"

report = {
    "phase": "4.40",
    "title": "Final Release Manifest and Reproducibility Package",
    "status": status,
    "checks_passed": sum(bool(v) for v in checks.values()),
    "checks_total": len(checks),
    "checks": checks,
    "manifest": manifest,
    "release_boundary": release_boundary,
    "evidence_digest": digest,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
}

out = ROOT / "experiments/integration/phase4_40/crss_2024_phase4_40_final_release_manifest.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

# Also create a release-level copy
release = ROOT / "experiments/release/final_release_manifest.json"
release.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n=== PHASE 4.40 RESULT ===")
print("Status:", status)
print(f"Checks: {report['checks_passed']}/{report['checks_total']}")
if status != "PASS":
    print("Failed checks:")
    for k, v in checks.items():
        if not v:
            print("  -", k)
print("Evidence digest:", digest)
print("Report:", out)
print("Release manifest:", release)

if conflicts:
    print("\nUnresolved conflicts:")
    for p in conflicts:
        print("  ", p)

if status != "PASS":
    sys.exit(1)

