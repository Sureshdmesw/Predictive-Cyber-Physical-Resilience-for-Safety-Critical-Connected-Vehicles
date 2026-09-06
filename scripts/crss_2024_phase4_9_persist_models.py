from pathlib import Path
import json
import joblib
import runpy

ROOT = Path.cwd()

SCRIPT = ROOT / "scripts" / "crss_2024_phase4_9_strong_baseline.py"
MODEL_DIR = ROOT / "models" / "v2" / "baseline"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("PHASE 4.9 MODEL PERSISTENCE REBUILD")
print("=" * 100)
print(f"Script: {SCRIPT}")
print(f"Output: {MODEL_DIR}")

# Execute the existing Phase 4.9 script in its own namespace.
# This reproduces the already validated methodology exactly.
ns = runpy.run_path(str(SCRIPT), run_name="__main__")

models = ns.get("models")
if not isinstance(models, dict) or "y3" not in models or "y6" not in models:
    raise RuntimeError("Phase 4.9 script did not expose trained y3/y6 models.")

# Save the exact trained models produced by Phase 4.9.
y3_path = MODEL_DIR / "crss_2024_v2_hgb_y3.joblib"
y6_path = MODEL_DIR / "crss_2024_v2_hgb_y6.joblib"

joblib.dump(models["y3"], y3_path, compress=3)
joblib.dump(models["y6"], y6_path, compress=3)

# Save the normalization statistics and representation metadata if exposed.
artifact = {
    "phase": "4.9",
    "algorithm": "HistGradientBoostingClassifier",
    "targets": ["y3", "y6"],
    "feature_count": 59,
    "representation_dimensions": 472,
    "representation_statistics": [
        "last",
        "mean",
        "std",
        "minimum",
        "maximum",
        "last_minus_first",
        "max_absolute_step_change",
        "last_step_change",
    ],
    "normalization": "z_score_train_only",
    "model_files": {
        "y3": str(y3_path.relative_to(ROOT)),
        "y6": str(y6_path.relative_to(ROOT)),
    },
    "source_script": str(SCRIPT.relative_to(ROOT)),
}

artifact_path = MODEL_DIR / "crss_2024_v2_hgb_artifact_manifest.json"
artifact_path.write_text(
    json.dumps(artifact, indent=2),
    encoding="utf-8",
)

print()
print("MODEL ARTIFACTS CREATED")
print(f"Y3: {y3_path} ({y3_path.stat().st_size:,} bytes)")
print(f"Y6: {y6_path} ({y6_path.stat().st_size:,} bytes)")
print(f"Manifest: {artifact_path} ({artifact_path.stat().st_size:,} bytes)")

# Immediate load test.
loaded_y3 = joblib.load(y3_path)
loaded_y6 = joblib.load(y6_path)

if type(loaded_y3).__name__ != "HistGradientBoostingClassifier":
    raise RuntimeError("Y3 artifact type mismatch.")

if type(loaded_y6).__name__ != "HistGradientBoostingClassifier":
    raise RuntimeError("Y6 artifact type mismatch.")

print()
print("LOAD TEST: PASS")
print(f"Y3 estimator: {type(loaded_y3).__name__}")
print(f"Y6 estimator: {type(loaded_y6).__name__}")

print()
print("=" * 100)
print("PHASE 4.9 MODEL PERSISTENCE: PASS")
print("=" * 100)
