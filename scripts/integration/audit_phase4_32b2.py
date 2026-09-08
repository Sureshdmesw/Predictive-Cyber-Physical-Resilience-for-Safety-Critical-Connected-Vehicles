import json
from pathlib import Path
from collections import Counter

REPORT = Path(
    r"experiments\integration\phase4_32b\crss_2024_phase4_32b2_origin_balanced_model_grounded_explainability.json"
)

d = json.loads(REPORT.read_text(encoding="utf-8"))

print("=== PHASE 4.32B.2 ARTIFACT AUDIT ===")
print("Report:", REPORT)
print("Phase:", d.get("phase"))
print("Status:", d.get("status"))
print("Cases:", d.get("cases"))
print("Unique scenarios:", d.get("unique_scenarios"))
print("Targets:", d.get("targets"))
print("Origins:", d.get("origins"))

checks = {}

checks["status_pass"] = d.get("status") == "PASS"
checks["cases_50"] = d.get("cases") == 50
checks["unique_scenarios_50"] = d.get("unique_scenarios") == 50

targets = {int(k): v for k, v in d.get("targets", {}).items()}
checks["targets_25_25"] = (
    targets.get(0) == 25 and
    targets.get(1) == 25
)

origins = {int(k): v for k, v in d.get("origins", {}).items()}
checks["origins_11_17"] = set(origins) == set(range(11, 18))

print("\n=== CORE CHECKS ===")
for name, value in checks.items():
    print(f"{name}: {'PASS' if value else 'FAIL'}")

cases = (
    d.get("case_results")
    or d.get("cases_detail")
    or d.get("results")
    or []
)

print("\n=== CASE STRUCTURE ===")
print("Case records:", len(cases))

if cases:
    scenario_ids = []
    case_origins = []
    case_targets = []

    for c in cases:
        scenario_ids.append(
            c.get("scenario_id")
            or c.get("scenario")
            or c.get("vehicle_scenario_id")
        )

        case_origins.append(
            c.get("prediction_origin", c.get("origin"))
        )

        case_targets.append(
            c.get("target", c.get("label"))
        )

    scenario_counter = Counter(scenario_ids)
    origin_counter = Counter(case_origins)
    target_counter = Counter(case_targets)

    duplicate_scenarios = {
        k: v for k, v in scenario_counter.items()
        if v > 1
    }

    print("Unique case scenarios:", len(scenario_counter))
    print("Duplicate scenarios:", len(duplicate_scenarios))
    print("Case targets:", dict(target_counter))
    print("Case origins:", dict(sorted(origin_counter.items())))

    checks["case_count_50"] = len(cases) == 50
    checks["case_scenarios_unique"] = len(scenario_counter) == 50
    checks["case_targets_25_25"] = (
        target_counter.get(0, 0) == 25 and
        target_counter.get(1, 0) == 25
    )
    checks["case_origins_all_11_17"] = (
        set(origin_counter) == set(range(11, 18))
    )

    print("\nCase-level checks:")
    for name in [
        "case_count_50",
        "case_scenarios_unique",
        "case_targets_25_25",
        "case_origins_all_11_17",
    ]:
        print(f"{name}: {'PASS' if checks[name] else 'FAIL'}")

print("\n=== MODEL-GROUNDED EVIDENCE ===")

for key in [
    "model_grounded",
    "retraining",
    "attribution_method",
    "temporal_attribution",
    "counterfactual_method",
    "checkpoint_hashes",
    "deterministic_digest",
    "causal_claim",
]:
    print(f"{key}: {d.get(key)}")

print("\n=== SAFETY BOUNDARY ===")

safety = d.get("safety_boundary", {})

for key in [
    "direct_vehicle_actuation",
    "live_cortex_xdr_api_used",
    "real_world_cyberattack_performance_claim",
]:
    print(f"{key}: {safety.get(key)}")

print("\n=== TOP-LEVEL KEYS ===")
for key in d.keys():
    print(" ", key)

print("\n=== FINAL AUDIT ===")

failed = [
    name for name, value in checks.items()
    if not value
]

if failed:
    print("AUDIT STATUS: FAIL")
    print("Failed checks:")
    for name in failed:
        print(" -", name)
else:
    print("AUDIT STATUS: PASS")
    print("All structural, uniqueness, balance, and origin-coverage checks passed.")

print("\nEvidence digest:", d.get("evidence_digest"))
