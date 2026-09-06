import json
from pathlib import Path

ROOT = Path.cwd()

MAP = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_modality_map.json"
OUT = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_cross_modal_experiment_spec.json"

with open(MAP, "r", encoding="utf-8") as f:
    m = json.load(f)

cyber = m["cyber_features"]
physical = m["physical_features"]
derived = m["derived_temporal_resilience_features"]

composite = {
    "cyber_instability_index",
    "resilience_degradation_rate",
    "state_transition_magnitude",
}

derived_primary = [x for x in derived if x not in composite]

experiments = {
    "A_cyber_only": cyber,
    "B_physical_only": physical,
    "C_cyber_physical": cyber + physical,
    "D_cyber_physical_temporal": cyber + physical + derived_primary,
}

spec = {
    "phase": "4.11",
    "status": "DESIGN_READY",
    "base_feature_count": 59,
    "cyber_count": len(cyber),
    "physical_count": len(physical),
    "derived_count": len(derived),
    "excluded_composite_features": sorted(composite),
    "primary_derived_count": len(derived_primary),
    "experiments": {
        name: {
            "feature_count": len(features),
            "features": features
        }
        for name, features in experiments.items()
    },
    "targets": [
        "Y3",
        "Y6"
    ],
    "observation_window_steps": 12,
    "observation_window_seconds": 60,
    "controlled_subset": 100000,
    "same_split_across_experiments": True,
    "same_target_weights": True,
    "test_set_untouched": True,
    "model_family": "lightweight_GRU",
    "safety_boundary": "prediction_and_decision_support_only"
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(spec, f, indent=2)

print("=" * 72)
print("PHASE 4.11 — CROSS-MODAL EXPERIMENT SPEC")
print("=" * 72)

for name, features in experiments.items():
    print(f"{name:<30} : {len(features)} features")

print()
print("Excluded composite features:")
for x in sorted(composite):
    print("  -", x)

print()
print("Primary derived features:", len(derived_primary))
print("Total base features     :", len(cyber) + len(physical) + len(derived))
print()
print("=" * 72)
print("CROSS-MODAL EXPERIMENT SPEC PASS")
print("=" * 72)
print("Output:", OUT)
