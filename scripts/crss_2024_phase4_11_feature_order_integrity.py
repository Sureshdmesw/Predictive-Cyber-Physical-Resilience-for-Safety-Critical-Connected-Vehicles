import json
from pathlib import Path

ROOT = Path.cwd()

SEQ_DIR = ROOT / "data/processed/modeling/sequences/v2_forecasting"
MAP = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_modality_map.json"
OUT = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_feature_order_integrity.json"

with open(SEQ_DIR / "crss_2024_forecasting_tensor_metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

with open(MAP, "r", encoding="utf-8") as f:
    modality = json.load(f)

tensor_features = meta["feature_names"]

mapped_features = (
    modality["cyber_features"]
    + modality["physical_features"]
    + modality["derived_temporal_resilience_features"]
)

tensor_set = set(tensor_features)
mapped_set = set(mapped_features)

missing_from_tensor = sorted(mapped_set - tensor_set)
extra_in_tensor = sorted(tensor_set - mapped_set)

checks = {
    "tensor_feature_count": len(tensor_features) == 59,
    "mapped_feature_count": len(mapped_features) == 59,
    "no_missing_features": len(missing_from_tensor) == 0,
    "no_extra_features": len(extra_in_tensor) == 0,
    "unique_tensor_features": len(tensor_features) == len(set(tensor_features)),
    "unique_mapped_features": len(mapped_features) == len(set(mapped_features)),
}

status = "PASS" if all(checks.values()) else "HOLD"

result = {
    "phase": "4.11",
    "status": status,
    "tensor_feature_count": len(tensor_features),
    "mapped_feature_count": len(mapped_features),
    "missing_from_tensor": missing_from_tensor,
    "extra_in_tensor": extra_in_tensor,
    "checks": checks,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("=" * 72)
print("PHASE 4.11 — FEATURE ORDER INTEGRITY")
print("=" * 72)
print("Tensor features :", len(tensor_features))
print("Mapped features :", len(mapped_features))
print("Missing         :", len(missing_from_tensor))
print("Extra           :", len(extra_in_tensor))
print()
print("CHECKS:")
for name, value in checks.items():
    print(f"  {name:<28}: {'PASS' if value else 'FAIL'}")

print()
print("=" * 72)
print(f"FEATURE ORDER INTEGRITY {status}")
print("=" * 72)
print("Output:", OUT)
