from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/temporal/crss_2024_temporal_cyber_physical_states.parquet"
OUTPUT = ROOT / "data/schemas/modeling/crss_2024_feature_registry.json"
REPORT = ROOT / "experiments/modeling/leakage/crss_2024_leakage_audit.json"

print("=" * 100)
print("CRSS 2024 MODEL FEATURE / LEAKAGE AUDIT")
print("=" * 100)

if not INPUT.exists():
    raise FileNotFoundError(f"Temporal dataset not found: {INPUT}")

df = pd.read_parquet(INPUT)

# ------------------------------------------------------------
# TARGETS
# ------------------------------------------------------------

TARGETS = {
    "future_anomaly_3step",
    "future_anomaly_6step",
}

# ------------------------------------------------------------
# ABSOLUTELY FORBIDDEN
# These directly reveal the synthetic cyber state/target.
# ------------------------------------------------------------

DIRECT_FORBIDDEN = {
    "synthetic_cyber_anomaly",
    "cyber_anomaly_severity",
    "future_anomaly_3step",
    "future_anomaly_6step",
}

# ------------------------------------------------------------
# HIGH-RISK DERIVED FEATURES
# These require explicit leakage review because their
# construction may contain anomaly/resilience information.
# ------------------------------------------------------------

HIGH_RISK_DERIVED = {
    "physical_cyber_interaction_score",
    "connectivity_resilience_score",
    "cyber_physical_resilience_score",
    "predictive_warning_score",
    "predictive_warning_state",
}

# ------------------------------------------------------------
# IDENTIFIERS / TIME INDEX
# ------------------------------------------------------------

IDENTIFIERS = {
    "CASENUM",
    "VEH_NO",
    "telemetry_scenario_id",
    "time_step",
    "relative_time_sec",
    "sequence_length",
}

# ------------------------------------------------------------
# TEMPORAL DERIVED FEATURES
# ------------------------------------------------------------

TEMPORAL_DERIVED = {
    c for c in df.columns
    if (
        c.startswith("d1_")
        or c.startswith("rolling_mean_")
        or c.startswith("rolling_std_")
    )
}

features = []

for col in df.columns:

    if col in TARGETS:
        role = "target"
        leakage_status = "FORBIDDEN"

    elif col in DIRECT_FORBIDDEN:
        role = "target_derived"
        leakage_status = "FORBIDDEN"

    elif col in HIGH_RISK_DERIVED:
        role = "high_risk_derived"
        leakage_status = "REVIEW_REQUIRED"

    elif col in IDENTIFIERS:
        role = "identifier_or_temporal_index"
        leakage_status = "NOT_A_PREDICTOR"

    elif col in TEMPORAL_DERIVED:
        role = "temporal_predictor_candidate"
        leakage_status = "CANDIDATE"

    else:
        role = "raw_predictor_candidate"
        leakage_status = "CANDIDATE"

    features.append({
        "feature": col,
        "role": role,
        "leakage_status": leakage_status,
        "dtype": str(df[col].dtype),
        "null_count": int(df[col].isna().sum()),
        "unique_values": int(df[col].nunique(dropna=False))
    })

candidate_predictors = [
    x["feature"]
    for x in features
    if x["leakage_status"] == "CANDIDATE"
]

forbidden_features = [
    x["feature"]
    for x in features
    if x["leakage_status"] == "FORBIDDEN"
]

review_features = [
    x["feature"]
    for x in features
    if x["leakage_status"] == "REVIEW_REQUIRED"
]

# ------------------------------------------------------------
# REQUIRED COLUMN CHECKS
# ------------------------------------------------------------

required_targets_present = TARGETS.issubset(df.columns)

check_results = {
    "dataset_exists": True,
    "dataset_nonempty": len(df) > 0,
    "targets_present": required_targets_present,
    "candidate_predictors_exist": len(candidate_predictors) > 0,
    "direct_forbidden_identified": all(
        x in forbidden_features for x in DIRECT_FORBIDDEN
        if x in df.columns
    ),
}

# ------------------------------------------------------------
# LEAKAGE POLICY
# ------------------------------------------------------------

policy = {
    "prediction_time": "t",
    "prediction_horizons": {
        "short": "t+1 through t+3",
        "long": "t+1 through t+6"
    },
    "split_unit": "telemetry_scenario_id",
    "row_level_random_split_forbidden": True,
    "future_information_forbidden": True,
    "synthetic_target_as_predictor_forbidden": True,
    "target_derived_features_forbidden": True,
    "high_risk_features_require_review": True
}

# ------------------------------------------------------------
# WRITE FEATURE REGISTRY
# ------------------------------------------------------------

registry = {
    "dataset": "CRSS 2024 Temporal Cyber-Physical State Dataset",
    "version": "phase4_v1",
    "prediction_problem": {
        "task": "predict future cyber anomaly from information available at time t",
        "targets": sorted(TARGETS)
    },
    "policy": policy,
    "summary": {
        "total_columns": len(df.columns),
        "candidate_predictors": len(candidate_predictors),
        "forbidden_features": len(forbidden_features),
        "review_required_features": len(review_features)
    },
    "features": features
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)

# ------------------------------------------------------------
# WRITE AUDIT REPORT
# ------------------------------------------------------------

report = {
    "status": (
        "PASS"
        if all(check_results.values())
        else "FAIL"
    ),
    "checks": check_results,
    "summary": registry["summary"],
    "targets": sorted(TARGETS),
    "forbidden_features": sorted(forbidden_features),
    "review_required_features": sorted(review_features),
    "candidate_predictors": sorted(candidate_predictors),
    "policy": policy
}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

# ------------------------------------------------------------
# CONSOLE REPORT
# ------------------------------------------------------------

print()
print("DATASET ROWS              :", f"{len(df):,}")
print("DATASET COLUMNS           :", len(df.columns))
print("CANDIDATE PREDICTORS      :", len(candidate_predictors))
print("FORBIDDEN FEATURES        :", len(forbidden_features))
print("REVIEW REQUIRED           :", len(review_features))

print()
print("FORBIDDEN FEATURES:")
for x in sorted(forbidden_features):
    print("  [FORBIDDEN]", x)

print()
print("REVIEW REQUIRED:")
for x in sorted(review_features):
    print("  [REVIEW]", x)

print()
print("CHECKS:")
for name, passed in check_results.items():
    print(
        f"  [{'PASS' if passed else 'FAIL'}] {name}"
    )

print()
print("FEATURE REGISTRY:")
print(OUTPUT)

print("LEAKAGE REPORT:")
print(REPORT)

print()
print("=" * 100)

if not all(check_results.values()):
    raise RuntimeError("Leakage audit failed.")

print("LEAKAGE AUDIT: PASS")
print("=" * 100)
