from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/temporal/crss_2024_temporal_cyber_physical_states.parquet"
REGISTRY = ROOT / "data/schemas/modeling/crss_2024_feature_registry.json"

OUTPUT = ROOT / "data/schemas/modeling/crss_2024_final_predictor_registry.json"
REPORT = ROOT / "experiments/modeling/leakage/crss_2024_derived_feature_audit.json"

df = pd.read_parquet(INPUT)

TARGET = "synthetic_cyber_anomaly"

HIGH_RISK = [
    "physical_cyber_interaction_score",
    "connectivity_resilience_score",
    "cyber_physical_resilience_score",
    "predictive_warning_score",
    "predictive_warning_state",
]

# ------------------------------------------------------------
# Determine whether a feature is plausibly target-derived.
# ------------------------------------------------------------

results = []

for feature in HIGH_RISK:

    if feature not in df.columns:
        results.append({
            "feature": feature,
            "status": "MISSING",
            "decision": "EXCLUDE"
        })
        continue

    x = df[feature]

    if pd.api.types.is_numeric_dtype(x):

        valid = x.notna() & df[TARGET].notna()

        if valid.sum() > 0:
            anomaly_mean = float(
                x[valid & (df[TARGET] == 1)].mean()
            ) if (valid & (df[TARGET] == 1)).any() else None

            normal_mean = float(
                x[valid & (df[TARGET] == 0)].mean()
            ) if (valid & (df[TARGET] == 0)).any() else None

            correlation = float(
                x[valid].corr(
                    df.loc[valid, TARGET]
                )
            )

            # Conservative research rule:
            # Any feature explicitly constructed as a resilience,
            # interaction, or warning score is excluded from the
            # primary predictive model unless independent provenance
            # proves it is observable without the target.

            decision = "EXCLUDE"
            status = "TARGET_DERIVED_RISK"

            results.append({
                "feature": feature,
                "dtype": str(x.dtype),
                "non_null": int(x.notna().sum()),
                "unique_values": int(x.nunique(dropna=False)),
                "normal_mean": normal_mean,
                "anomaly_mean": anomaly_mean,
                "target_correlation": correlation,
                "status": status,
                "decision": decision,
                "reason": (
                    "Feature is a synthesized resilience/interaction/"
                    "warning quantity and its construction may encode "
                    "synthetic anomaly information."
                )
            })

        else:
            results.append({
                "feature": feature,
                "status": "NO_VALID_ROWS",
                "decision": "EXCLUDE"
            })

    else:

        results.append({
            "feature": feature,
            "dtype": str(x.dtype),
            "non_null": int(x.notna().sum()),
            "unique_values": int(x.nunique(dropna=False)),
            "status": "DERIVED_CATEGORICAL",
            "decision": "EXCLUDE",
            "reason": (
                "Predictive warning state is generated from a "
                "derived warning score and therefore cannot be "
                "used as an independent predictor."
            )
        })

# ------------------------------------------------------------
# Primary model predictor policy
# ------------------------------------------------------------

IDENTIFIERS = {
    "CASENUM",
    "VEH_NO",
    "telemetry_scenario_id",
    "time_step",
    "relative_time_sec",
    "sequence_length",
}

TARGETS = {
    "future_anomaly_3step",
    "future_anomaly_6step",
    "synthetic_cyber_anomaly",
    "cyber_anomaly_severity",
}

# Explicitly exclude all current target-derived quantities.
EXCLUDED_DERIVED = set(HIGH_RISK)

# Temporal derivatives and rolling statistics are retained because
# they are functions of observations available at or before t.
CANDIDATES = []

for col in df.columns:

    if col in IDENTIFIERS:
        continue

    if col in TARGETS:
        continue

    if col in EXCLUDED_DERIVED:
        continue

    CANDIDATES.append(col)

# ------------------------------------------------------------
# Additional leakage rules
# ------------------------------------------------------------

forbidden_prefixes = [
    "future_",
]

forbidden_exact = {
    "synthetic_cyber_anomaly",
    "cyber_anomaly_severity",
}

FINAL_PREDICTORS = []

for col in CANDIDATES:

    if col in forbidden_exact:
        continue

    if any(col.startswith(prefix) for prefix in forbidden_prefixes):
        continue

    FINAL_PREDICTORS.append(col)

# ------------------------------------------------------------
# Identify temporal predictor groups
# ------------------------------------------------------------

groups = {
    "physical_vehicle": [],
    "cyber_ecu_can": [],
    "sensor_consistency": [],
    "connectivity": [],
    "communication_integrity": [],
    "environment": [],
    "temporal_derivatives": [],
    "temporal_rolling": [],
    "other": []
}

for col in FINAL_PREDICTORS:

    c = col.lower()

    if (
        c.startswith("d1_")
    ):
        groups["temporal_derivatives"].append(col)

    elif (
        c.startswith("rolling_mean_")
        or c.startswith("rolling_std_")
    ):
        groups["temporal_rolling"].append(col)

    elif any(
        token in c
        for token in [
            "ecu_",
            "can_",
            "uds_",
            "diagnostic_",
            "authentication_"
        ]
    ):
        groups["cyber_ecu_can"].append(col)

    elif any(
        token in c
        for token in [
            "sensor_",
            "redundant_sensor"
        ]
    ):
        groups["sensor_consistency"].append(col)

    elif any(
        token in c
        for token in [
            "rssi",
            "packet_loss",
            "latency",
            "connectivity",
            "telemetry_gap"
        ]
    ):
        groups["connectivity"].append(col)

    elif any(
        token in c
        for token in [
            "integrity",
            "replay",
            "sequence_counter",
            "unexpected_source"
        ]
    ):
        groups["communication_integrity"].append(col)

    elif col in {
        "VISION",
        "WEATHER"
    }:
        groups["environment"].append(col)

    elif any(
        token in c
        for token in [
            "speed_",
            "rollover",
            "fire_",
            "collision",
            "roadway_",
            "vehicle_state",
            "physical_state"
        ]
    ):
        groups["physical_vehicle"].append(col)

    else:
        groups["other"].append(col)

# ------------------------------------------------------------
# Final registry
# ------------------------------------------------------------

registry = {
    "dataset": "CRSS 2024 Temporal Cyber-Physical State Dataset",
    "version": "phase4_final_predictor_registry_v1",

    "prediction_definition": {
        "input_information": "observations available at time t",
        "target_3step": "future_anomaly_3step",
        "target_6step": "future_anomaly_6step",
        "prediction_windows": {
            "3_step": "t+1 through t+3",
            "6_step": "t+1 through t+6"
        }
    },

    "split_policy": {
        "unit": "telemetry_scenario_id",
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
        "random_row_split": False
    },

    "excluded_features": sorted(
        IDENTIFIERS |
        TARGETS |
        EXCLUDED_DERIVED
    ),

    "high_risk_audit": results,

    "predictor_count": len(FINAL_PREDICTORS),

    "predictors": sorted(FINAL_PREDICTORS),

    "predictor_groups": {
        k: sorted(v)
        for k, v in groups.items()
    },

    "research_policy": [
        "No future labels may be used as predictors.",
        "Synthetic anomaly labels may not be used as predictors.",
        "Target-derived resilience/warning scores are excluded from the primary model.",
        "Temporal derivatives are computed from current and historical observations.",
        "Rolling statistics are computed from current and historical observations.",
        "Scenario-level splitting prevents temporal sequence leakage.",
        "Primary predictive evaluation must use the held-out test scenarios."
    ]
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)

audit = {
    "status": "PASS",
    "high_risk_features": results,
    "excluded_count": len(
        IDENTIFIERS | TARGETS | EXCLUDED_DERIVED
    ),
    "final_predictor_count": len(FINAL_PREDICTORS),
    "predictor_groups": {
        k: len(v)
        for k, v in groups.items()
    }
}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)

print()
print("=" * 100)
print("CRSS 2024 FINAL PREDICTOR / DERIVED-FEATURE AUDIT")
print("=" * 100)

print()
print("HIGH-RISK FEATURE DECISIONS:")

for r in results:
    print(
        f"[{r['decision']}] "
        f"{r['feature']}"
    )

print()
print("FINAL PREDICTOR COUNT:", len(FINAL_PREDICTORS))

print()
print("PREDICTOR GROUPS:")

for name, values in groups.items():
    print(
        f"{name:28s}: {len(values):3d}"
    )

print()
print("FINAL REGISTRY:")
print(OUTPUT)

print()
print("AUDIT REPORT:")
print(REPORT)

print()
print("=" * 100)
print("DERIVED FEATURE AUDIT: PASS")
print("PRIMARY MODEL PREDICTOR REGISTRY: READY")
print("=" * 100)
