import json
from pathlib import Path

ROOT = Path.cwd()

MAP = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_modality_map.json"
OUT = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_modality_sanity_audit.json"

with open(MAP, "r", encoding="utf-8") as f:
    data = json.load(f)

cyber = data["cyber_features"]
physical = data["physical_features"]
derived = data["derived_temporal_resilience_features"]

all_features = set(cyber) | set(physical) | set(derived)

expected_total = data["total_features"]

composite = {
    "cyber_instability_index",
    "resilience_degradation_rate",
    "state_transition_magnitude",
}

rolling = {
    x for x in derived
    if x.startswith("rolling_mean_") or x.startswith("rolling_std_")
}

delta = {
    x for x in derived
    if x.startswith("d1_")
}

checks = {
    "total_count": len(all_features) == expected_total,
    "cyber_count": len(cyber) == data["cyber_feature_count"],
    "physical_count": len(physical) == data["physical_feature_count"],
    "derived_count": len(derived) == data["derived_temporal_resilience_feature_count"],
    "no_overlap": (
        not (set(cyber) & set(physical))
        and not (set(cyber) & set(derived))
        and not (set(physical) & set(derived))
    ),
    "no_unmapped": len(all_features) == expected_total,
}

status = "PASS" if all(checks.values()) else "HOLD"

audit = {
    "phase": "4.11",
    "status": status,
    "checks": checks,
    "total_features": len(all_features),
    "cyber_count": len(cyber),
    "physical_count": len(physical),
    "derived_count": len(derived),
    "composite_derived_features": sorted(composite & set(derived)),
    "rolling_derived_features": sorted(rolling),
    "delta_derived_features": sorted(delta),
    "scientific_notes": [
        "Cyber/Edge represents ECU, CAN, diagnostic, connectivity, and communication-integrity telemetry.",
        "Physical/Safety represents CRSS-derived physical and safety observations.",
        "Derived/Temporal/Resilience represents downstream temporal transforms and resilience-state constructs.",
        "Derived features should be evaluated separately from raw modality features.",
        "Synthetic cyber telemetry is research-generated and is not observed cyberattack ground truth."
    ]
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)

print("=" * 72)
print("PHASE 4.11 — MODALITY SANITY AUDIT")
print("=" * 72)

print("Expected total :", expected_total)
print("Mapped total   :", len(all_features))
print("Cyber          :", len(cyber))
print("Physical       :", len(physical))
print("Derived        :", len(derived))
print()

print("CHECKS:")
for name, result in checks.items():
    print(f"  {name:<20} : {'PASS' if result else 'FAIL'}")

print()
print("Composite derived features:")
for x in sorted(composite & set(derived)):
    print("  -", x)

print()
print("Rolling derived :", len(rolling))
print("Delta derived   :", len(delta))

print()
print("=" * 72)
print(f"MODALITY SANITY AUDIT {status}")
print("=" * 72)
print("Output:", OUT)
