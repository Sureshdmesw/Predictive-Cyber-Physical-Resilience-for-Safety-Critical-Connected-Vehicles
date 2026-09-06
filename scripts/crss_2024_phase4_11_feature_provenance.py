from pathlib import Path
import re
import json

ROOT = Path(__file__).resolve().parents[1]

spec_path = ROOT / "experiments/modeling/v2/crss_2024_v2_feature_treatment_spec.json"
out_path = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_feature_provenance_audit.json"

spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
features = spec["final_features"]

script_files = list((ROOT / "scripts").glob("*.py"))

results = []

for feature in features:
    matches = []

    for path in script_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if feature in text:
            lines = text.splitlines()

            for i, line in enumerate(lines):
                if feature in line:
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)

                    matches.append({
                        "script": str(path.relative_to(ROOT)),
                        "line": i + 1,
                        "context": lines[start:end]
                    })

    results.append({
        "feature": feature,
        "matches": matches
    })

result = {
    "phase": "4.11",
    "purpose": "Trace every final model feature to its actual project definition before assigning cyber/physical modality.",
    "feature_count": len(features),
    "features": results
}

out_path.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

print("=" * 72)
print("PHASE 4.11 — FEATURE PROVENANCE AUDIT")
print("=" * 72)

print("Final features:", len(features))

for item in results:
    print()
    print(item["feature"])

    if not item["matches"]:
        print("  [NO SCRIPT MATCH]")
    else:
        unique_scripts = sorted(set(x["script"] for x in item["matches"]))
        print("  Scripts:", ", ".join(unique_scripts))

print()
print("=" * 72)
print("AUDIT SAVED")
print("=" * 72)
print("Output:", out_path)
