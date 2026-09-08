import json
from pathlib import Path
from collections import Counter

ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

report_path = ROOT / "experiments/integration/phase4_32b/crss_2024_phase4_32b_unique_scenario_model_grounded_explainability.json"

with open(report_path, encoding="utf-8") as f:
    report = json.load(f)

cases = report["cases"]

scenario_ids = [
    c["scenario_id"]
    for c in cases
]

origins = [
    c["prediction_origin"]
    for c in cases
]

targets = [
    c["target_y6"]
    for c in cases
]

print("Cases:", len(cases))
print("Unique scenarios:", len(set(scenario_ids)))
print("Targets:", Counter(targets))
print("Origins:", Counter(origins))

print("\n=== ORIGIN DISTRIBUTION BY TARGET ===")

for target in [0, 1]:
    print(
        f"target={target}:",
        Counter(
            c["prediction_origin"]
            for c in cases
            if c["target_y6"] == target
        )
    )

print("\n=== SCENARIO UNIQUENESS ===")
print(
    "Unique scenario check:",
    len(set(scenario_ids)) == len(scenario_ids)
)

print("\n=== ORIGIN COVERAGE ===")
print(
    "Origins present:",
    sorted(set(origins))
)

print(
    "All origins 11-17 represented:",
    set(origins) == set(range(11, 18))
)

print("\n=== DECISION ===")

if set(origins) == set(range(11, 18)):
    print("PASS: origin coverage is complete.")
else:
    print(
        "FAIL: Phase 4.32B needs origin-balanced resampling "
        "before final evidence commitment."
    )
