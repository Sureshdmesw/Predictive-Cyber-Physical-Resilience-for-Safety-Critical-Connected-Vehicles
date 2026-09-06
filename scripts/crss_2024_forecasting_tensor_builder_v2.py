from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data/processed/temporal/v2/crss_2024_temporal_cyber_physical_states_v2.parquet"
SPLIT = ROOT / "data/processed/modeling/splits/crss_2024_scenario_split.csv"
REGISTRY = ROOT / "data/schemas/modeling/crss_2024_final_predictor_registry.json"

OUT = ROOT / "data/processed/modeling/sequences/v2_forecasting"
META = OUT / "crss_2024_forecasting_tensor_metadata.json"

OBSERVATION_WINDOW = 12
HORIZON_3 = 3
HORIZON_6 = 6
STEP_SECONDS = 5

print("=" * 100)
print("CRSS 2024 — V2 ROLLING FORECASTING TENSOR CONSTRUCTION")
print("=" * 100)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_parquet(DATA)
split = pd.read_csv(SPLIT)

print(f"Rows loaded: {len(df):,}")

# ------------------------------------------------------------
# PREDICTOR REGISTRY
# ------------------------------------------------------------

with open(REGISTRY, "r", encoding="utf-8") as f:
    registry = json.load(f)

if isinstance(registry, dict):

    if "predictors" in registry:
        predictors = registry["predictors"]

    elif "features" in registry:
        predictors = registry["features"]

    elif "final_predictors" in registry:
        predictors = registry["final_predictors"]

    else:
        predictors = []

        for value in registry.values():
            if isinstance(value, list):
                predictors.extend(value)

else:
    predictors = registry

predictors = list(dict.fromkeys(predictors))

# Explicit leakage / identifier exclusion
excluded = {
    "synthetic_cyber_anomaly",
    "cyber_anomaly_severity",
    "future_anomaly_3step",
    "future_anomaly_6step",

    "connectivity_resilience_score",
    "cyber_physical_resilience_score",
    "physical_cyber_interaction_score",

    "predictive_warning_score",
    "predictive_warning_state",

    "temporal_phase",

    "CASENUM",
    "VEH_NO",
    "telemetry_scenario_id",
    "time_step",
    "relative_time_sec",
    "sequence_length",
}

predictors = [
    p for p in predictors
    if p in df.columns and p not in excluded
]

if not predictors:
    raise RuntimeError("No valid predictors found.")

print(f"Predictors: {len(predictors)}")

# ------------------------------------------------------------
# MERGE SPLITS
# ------------------------------------------------------------

split_map = (
    split[
        ["telemetry_scenario_id", "split"]
    ]
    .drop_duplicates()
)

if not split_map["telemetry_scenario_id"].is_unique:
    raise RuntimeError("Duplicate scenario IDs in split manifest.")

df = df.merge(
    split_map,
    on="telemetry_scenario_id",
    how="left",
    validate="many_to_one"
)

if df["split"].isna().any():
    raise RuntimeError(
        f"Rows without split: {df['split'].isna().sum():,}"
    )

# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

df = df.sort_values(
    ["telemetry_scenario_id", "time_step"]
).reset_index(drop=True)

# ------------------------------------------------------------
# BUILD ROLLING FORECASTING WINDOWS
#
# Input:
#   12 historical steps
#
# Target:
#   future anomaly in next 3 steps
#   future anomaly in next 6 steps
#
# Last valid prediction origin:
#   t = 17
#
# because t+1 ... t+6 must exist.
# ------------------------------------------------------------

X_windows = []
Y3_windows = []
Y6_windows = []
scenario_windows = []
origin_windows = []
split_windows = []

scenario_count = 0

for scenario_id, group in df.groupby(
    "telemetry_scenario_id",
    sort=False
):

    group = group.sort_values("time_step")

    if len(group) != 24:
        raise RuntimeError(
            f"Scenario {scenario_id} has {len(group)} rows."
        )

    values = (
        group[predictors]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )

    anomaly = (
        group["synthetic_cyber_anomaly"]
        .to_numpy(dtype=np.int8)
    )

    scenario_split = group["split"].iloc[0]

    # --------------------------------------------------------
    # Rolling prediction origins
    # --------------------------------------------------------

    for origin in range(
        OBSERVATION_WINDOW - 1,
        24 - HORIZON_6
    ):

        start = origin - OBSERVATION_WINDOW + 1
        end = origin + 1

        # Historical observation window
        X_windows.append(
            values[start:end]
        )

        # Future 3-step label:
        # t+1, t+2, t+3
        y3 = int(
            anomaly[
                origin + 1:
                origin + HORIZON_3 + 1
            ].max()
        )

        # Future 6-step label:
        # t+1 ... t+6
        y6 = int(
            anomaly[
                origin + 1:
                origin + HORIZON_6 + 1
            ].max()
        )

        Y3_windows.append(y3)
        Y6_windows.append(y6)

        scenario_windows.append(scenario_id)
        origin_windows.append(origin)
        split_windows.append(scenario_split)

    scenario_count += 1

# ------------------------------------------------------------
# CONVERT TO ARRAYS
# ------------------------------------------------------------

X = np.asarray(
    X_windows,
    dtype=np.float32
)

Y3 = np.asarray(
    Y3_windows,
    dtype=np.int8
)

Y6 = np.asarray(
    Y6_windows,
    dtype=np.int8
)

scenario_ids = np.asarray(
    scenario_windows
)

prediction_origins = np.asarray(
    origin_windows,
    dtype=np.int8
)

window_splits = np.asarray(
    split_windows
)

print()
print("FULL FORECASTING TENSOR")
print("-" * 100)
print(f"X shape : {X.shape}")
print(f"Y3     : {Y3.shape}")
print(f"Y6     : {Y6.shape}")

# ------------------------------------------------------------
# SPLIT
# ------------------------------------------------------------

OUT.mkdir(
    parents=True,
    exist_ok=True
)

for split_name in [
    "train",
    "validation",
    "test"
]:

    mask = (
        window_splits ==
        split_name
    )

    np.save(
        OUT / f"X_{split_name}_v2_forecasting.npy",
        X[mask]
    )

    np.save(
        OUT / f"y_3step_{split_name}_v2_forecasting.npy",
        Y3[mask]
    )

    np.save(
        OUT / f"y_6step_{split_name}_v2_forecasting.npy",
        Y6[mask]
    )

    np.save(
        OUT / f"scenario_ids_{split_name}_v2_forecasting.npy",
        scenario_ids[mask]
    )

    np.save(
        OUT / f"prediction_origins_{split_name}_v2_forecasting.npy",
        prediction_origins[mask]
    )

    print(
        f"{split_name:12s}: "
        f"samples={mask.sum():,} "
        f"X={X[mask].shape} "
        f"Y3+={Y3[mask].sum():,} "
        f"Y6+={Y6[mask].sum():,}"
    )

# ------------------------------------------------------------
# QUALITY GATE
# ------------------------------------------------------------

checks = []

def check(name, condition, detail):
    checks.append({
        "name": name,
        "passed": bool(condition),
        "detail": str(detail)
    })

check(
    "nonempty",
    len(X) > 0,
    f"samples={len(X):,}"
)

check(
    "observation_window",
    X.shape[1] == OBSERVATION_WINDOW,
    X.shape
)

check(
    "feature_count",
    X.shape[2] == len(predictors),
    f"expected={len(predictors)}, actual={X.shape[2]}"
)

check(
    "binary_y3",
    set(np.unique(Y3)).issubset({0,1}),
    np.unique(Y3).tolist()
)

check(
    "binary_y6",
    set(np.unique(Y6)).issubset({0,1}),
    np.unique(Y6).tolist()
)

check(
    "horizon_distinct",
    not np.array_equal(Y3, Y6),
    f"Y3={Y3.sum():,}, Y6={Y6.sum():,}"
)

check(
    "no_nan",
    not np.isnan(X).any(),
    f"NaNs={np.isnan(X).sum():,}"
)

check(
    "no_inf",
    not np.isinf(X).any(),
    f"Infs={np.isinf(X).sum():,}"
)

check(
    "train_present",
    (window_splits == "train").any(),
    "train samples exist"
)

check(
    "validation_present",
    (window_splits == "validation").any(),
    "validation samples exist"
)

check(
    "test_present",
    (window_splits == "test").any(),
    "test samples exist"
)

# ------------------------------------------------------------
# SCENARIO INTEGRITY
# ------------------------------------------------------------

scenario_split_pairs = {}

for sid in np.unique(scenario_ids):

    splits_for_sid = np.unique(
        window_splits[
            scenario_ids == sid
        ]
    )

    scenario_split_pairs[sid] = splits_for_sid.tolist()

scenario_overlap = [
    sid
    for sid, values in scenario_split_pairs.items()
    if len(values) != 1
]

check(
    "no_scenario_split_overlap",
    len(scenario_overlap) == 0,
    f"overlap={len(scenario_overlap)}"
)

# ------------------------------------------------------------
# HORIZON STATISTICS
# ------------------------------------------------------------

stats = {}

for split_name in [
    "train",
    "validation",
    "test"
]:

    mask = (
        window_splits ==
        split_name
    )

    n_samples = int(mask.sum())

    p3 = int(Y3[mask].sum())
    p6 = int(Y6[mask].sum())

    stats[split_name] = {
        "samples": n_samples,
        "y3_positive": p3,
        "y3_rate": float(
            p3 / max(n_samples, 1)
        ),
        "y6_positive": p6,
        "y6_rate": float(
            p6 / max(n_samples, 1)
        ),
        "y6_only": int(
            ((Y6 == 1) &
             (Y3 == 0) &
             mask).sum()
        ),
        "y3_only": int(
            ((Y3 == 1) &
             (Y6 == 0) &
             mask).sum()
        )
    }

# ------------------------------------------------------------
# METADATA
# ------------------------------------------------------------

metadata = {
    "dataset": "CRSS 2024 Temporal Cyber-Physical States V2",
    "task": "rolling multi-horizon forecasting",
    "observation_window_steps": OBSERVATION_WINDOW,
    "observation_window_seconds": (
        OBSERVATION_WINDOW *
        STEP_SECONDS
    ),
    "horizon_3_steps": HORIZON_3,
    "horizon_3_seconds": (
        HORIZON_3 *
        STEP_SECONDS
    ),
    "horizon_6_steps": HORIZON_6,
    "horizon_6_seconds": (
        HORIZON_6 *
        STEP_SECONDS
    ),
    "feature_count": len(predictors),
    "features": predictors,
    "scenario_count": scenario_count,
    "sample_count": len(X),
    "statistics": stats,
    "excluded_features": sorted(excluded),
    "label_definition": {
        "y_3step": (
            "1 if any synthetic cyber anomaly occurs "
            "within t+1 through t+3."
        ),
        "y_6step": (
            "1 if any synthetic cyber anomaly occurs "
            "within t+1 through t+6."
        )
    },
    "methodological_note": (
        "Prediction samples are generated from rolling "
        "prediction origins within scenario-level partitions. "
        "Scenario identities never cross train, validation, "
        "and test partitions."
    )
}

with open(
    META,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )

# ------------------------------------------------------------
# QUALITY REPORT
# ------------------------------------------------------------

passed = sum(
    c["passed"]
    for c in checks
)

failed = len(checks) - passed

quality = {
    "status": (
        "PASS"
        if failed == 0
        else "FAIL"
    ),
    "checks": checks,
    "shape": list(X.shape),
    "statistics": stats
}

quality_path = (
    ROOT /
    "experiments/temporal/v2/"
    "crss_2024_v2_forecasting_tensor_quality.json"
)

with open(
    quality_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        quality,
        f,
        indent=2
    )

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

print()
print("=" * 100)
print("V2 FORECASTING TENSOR QUALITY GATE")
print("=" * 100)

for c in checks:
    print(
        f"[{'PASS' if c['passed'] else 'FAIL'}] "
        f"{c['name']} -- {c['detail']}"
    )

print()
print("HORIZON DISTRIBUTION")
print("-" * 100)

for split_name, s in stats.items():

    print(
        f"{split_name:12s} "
        f"samples={s['samples']:,} "
        f"Y3={s['y3_positive']:,} "
        f"({s['y3_rate']:.6%}) "
        f"Y6={s['y6_positive']:,} "
        f"({s['y6_rate']:.6%}) "
        f"Y6-only={s['y6_only']:,}"
    )

print()
print("=" * 100)
print(
    f"STATUS : "
    f"{'PASS' if failed == 0 else 'FAIL'}"
)
print(f"CHECKS : {len(checks)}")
print(f"PASS   : {passed}")
print(f"FAIL   : {failed}")
print("=" * 100)

if failed:
    raise RuntimeError(
        f"Forecasting tensor quality gate failed: "
        f"{failed} failure(s)."
    )

print()
print("PHASE 4.4 V2 FORECASTING: PASS")
print(f"OUTPUT: {OUT}")
print(f"QUALITY: {quality_path}")
