import json
from pathlib import Path

ROOT = Path.cwd()

META = ROOT / "data/processed/modeling/sequences/v2_forecasting/crss_2024_forecasting_tensor_metadata.json"
TREATMENT = ROOT / "experiments/modeling/v2/crss_2024_v2_feature_treatment_spec.json"
MAP = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_modality_map.json"

with open(META, "r", encoding="utf-8") as f:
    meta = json.load(f)

with open(TREATMENT, "r", encoding="utf-8") as f:
    treatment = json.load(f)

with open(MAP, "r", encoding="utf-8") as f:
    modality = json.load(f)

original_features = meta["features"]

# Find final feature list from the treatment specification.
final_features = None

for key in [
    "final_features",
    "features_final",
    "retained_features",
    "selected_features"
]:
    if key in treatment and isinstance(treatment[key], list):
        final_features = treatment[key]
        break

if final_features is None:
    print("TREATMENT TOP-LEVEL KEYS:")
    for k, v in treatment.items():
        print(" -", k, type(v).__name__, len(v) if hasattr(v, "__len__") else "")
    raise RuntimeError(
        "Could not locate final feature list in treatment specification."
    )

mapped_features = (
    modality["cyber_features"]
    + modality["physical_features"]
    + modality["derived_temporal_resilience_features"]
)

original_set = set(original_features)
final_set = set(final_features)
mapped_set = set(mapped_features)

removed_by_treatment = original_set - final_set
missing_from_original = mapped_set - original_set
missing_from_final = mapped_set - final_set
unexpected_final = final_set - mapped_set

checks = {
    "original_count_63": len(original_features) == 63,
    "final_count_59": len(final_features) == 59,
    "mapped_count_59": len(mapped_features) == 59,
    "four_features_removed": len(removed_by_treatment) == 4,
    "mapped_all_in_original": len(missing_from_original) == 0,
    "mapped_all_in_final": len(missing_from_final) == 0,
    "no_unexpected_final_features": len(unexpected_final) == 0,
    "unique_final_features": len(final_features) == len(final_set),
    "unique_mapped_features": len(mapped_features) == len(mapped_set),
}

status = "PASS" if all(checks.values()) else "HOLD"

print("=" * 72)
print("PHASE 4.11 — FINAL FEATURE/TENSOR INTEGRITY")
print("=" * 72)

print("Original tensor features :", len(original_features))
print("Final treated features  :", len(final_features))
print("Modality-map features   :", len(mapped_features))
print()

print("Features removed by treatment:")
for x in sorted(removed_by_treatment):
    print("  -", x)

print()
print("CHECKS:")
for name, value in checks.items():
    print(f"  {name:<32}: {'PASS' if value else 'FAIL'}")

print()
print("=" * 72)
print(f"FINAL FEATURE/TENSOR INTEGRITY {status}")
print("=" * 72)

if missing_from_final:
    print("Missing from final:")
    for x in sorted(missing_from_final):
        print("  -", x)

if unexpected_final:
    print("Unexpected final:")
    for x in sorted(unexpected_final):
        print("  -", x)
