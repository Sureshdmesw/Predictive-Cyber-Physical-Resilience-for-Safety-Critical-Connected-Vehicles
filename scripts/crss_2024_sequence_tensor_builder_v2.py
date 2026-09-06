from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data/processed/temporal/v2/crss_2024_temporal_cyber_physical_states_v2.parquet"
SPLIT = ROOT / "data/processed/modeling/splits/crss_2024_scenario_split.csv"

OUT = ROOT / "data/processed/modeling/sequences/v2"
META = OUT / "crss_2024_sequence_metadata_v2.json"

SEQUENCE_LENGTH = 24

print("=" * 100)
print("CRSS 2024 — PHASE 4.4 V2 SEQUENCE TENSOR CONSTRUCTION")
print("=" * 100)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_parquet(DATA)
split = pd.read_csv(SPLIT)

print(f"Rows loaded: {len(df):,}")

if "telemetry_scenario_id" not in df.columns:
    raise RuntimeError("telemetry_scenario_id missing")

if "telemetry_scenario_id" not in split.columns:
    raise RuntimeError("telemetry_scenario_id missing from split")

# ------------------------------------------------------------
# FINAL PREDICTOR SET
# ------------------------------------------------------------

registry_path = (
    ROOT /
    "data/schemas/modeling/crss_2024_final_predictor_registry.json"
)

with open(registry_path, "r", encoding="utf-8") as f:
    registry = json.load(f)

# Support both common registry structures.
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

# Explicit exclusions
forbidden = {
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
    "telemetry_scenario_id",
    "CASENUM",
    "VEH_NO",
    "time_step",
    "relative_time_sec",
    "sequence_length",
}

predictors = [
    p for p in predictors
    if p in df.columns and p not in forbidden
]

if not predictors:
    raise RuntimeError(
        "No usable predictors were found in final predictor registry."
    )

missing_predictors = [
    p for p in predictors
    if p not in df.columns
]

if missing_predictors:
    raise RuntimeError(
        f"Predictors missing from dataset: {missing_predictors}"
    )

print(f"Predictors selected: {len(predictors)}")

# ------------------------------------------------------------
# MERGE SCENARIO SPLITS
# ------------------------------------------------------------

split_map = split[
    ["telemetry_scenario_id", "split"]
].drop_duplicates()

if not split_map["telemetry_scenario_id"].is_unique:
    raise RuntimeError(
        "Scenario split contains duplicate scenario IDs."
    )

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
# VALIDATE SEQUENCE STRUCTURE
# ------------------------------------------------------------

counts = (
    df.groupby("telemetry_scenario_id")
      .size()
)

if not counts.eq(SEQUENCE_LENGTH).all():
    bad = counts[counts != SEQUENCE_LENGTH]
    raise RuntimeError(
        f"Invalid sequence lengths: {bad.head().to_dict()}"
    )

timesteps = (
    df.groupby("telemetry_scenario_id")["time_step"]
      .agg(list)
)

bad_timesteps = [
    sid for sid, values in timesteps.items()
    if values != list(range(SEQUENCE_LENGTH))
]

if bad_timesteps:
    raise RuntimeError(
        f"Invalid timestep sequence for {len(bad_timesteps)} scenarios."
    )

# ------------------------------------------------------------
# BUILD ARRAYS
# ------------------------------------------------------------

scenario_ids = (
    df["telemetry_scenario_id"]
    .drop_duplicates()
    .to_numpy()
)

scenario_split = (
    df.groupby("telemetry_scenario_id")["split"]
      .first()
      .reindex(scenario_ids)
      .to_numpy()
)

n_scenarios = len(scenario_ids)
n_features = len(predictors)

X = np.empty(
    (
        n_scenarios,
        SEQUENCE_LENGTH,
        n_features
    ),
    dtype=np.float32
)

Y3 = np.empty(
    (n_scenarios,),
    dtype=np.int8
)

Y6 = np.empty(
    (n_scenarios,),
    dtype=np.int8
)

# ------------------------------------------------------------
# VECTORIZED TENSOR CONSTRUCTION
# ------------------------------------------------------------

feature_matrix = (
    df[predictors]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .to_numpy(dtype=np.float32)
)

feature_matrix = feature_matrix.reshape(
    n_scenarios,
    SEQUENCE_LENGTH,
    n_features
)

X[:] = feature_matrix

scenario_target = (
    df.groupby("telemetry_scenario_id")
      .agg(
          y3=("future_anomaly_3step", "max"),
          y6=("future_anomaly_6step", "max")
      )
      .reindex(scenario_ids)
)

Y3[:] = scenario_target["y3"].to_numpy(dtype=np.int8)
Y6[:] = scenario_target["y6"].to_numpy(dtype=np.int8)

# ------------------------------------------------------------
# SPLIT TENSORS
# ------------------------------------------------------------

OUT.mkdir(parents=True, exist_ok=True)

for split_name in ["train", "validation", "test"]:

    mask = scenario_split == split_name

    np.save(
        OUT / f"X_{split_name}_v2.npy",
        X[mask]
    )

    np.save(
        OUT / f"y_3step_{split_name}_v2.npy",
        Y3[mask]
    )

    np.save(
        OUT / f"y_6step_{split_name}_v2.npy",
        Y6[mask]
    )

    np.save(
        OUT / f"scenario_ids_{split_name}_v2.npy",
        scenario_ids[mask]
    )

    print(
        f"{split_name:12s}: "
        f"scenarios={mask.sum():,} "
        f"X={X[mask].shape}"
    )

# ------------------------------------------------------------
# METADATA
# ------------------------------------------------------------

metadata = {
    "dataset": "CRSS 2024 Temporal Cyber-Physical States V2",
    "source": str(DATA),
    "sequence_length": SEQUENCE_LENGTH,
    "feature_count": n_features,
    "features": predictors,
    "scenario_count": n_scenarios,
    "tensor_dtype": "float32",
    "target_dtype": "int8",
    "targets": [
        "future_anomaly_3step",
        "future_anomaly_6step"
    ],
    "excluded_features": sorted(forbidden),
    "splits": {
        name: int((scenario_split == name).sum())
        for name in ["train", "validation", "test"]
    },
    "target_distribution": {
        "3step_positive_scenarios": int(Y3.sum()),
        "3step_negative_scenarios": int((Y3 == 0).sum()),
        "6step_positive_scenarios": int(Y6.sum()),
        "6step_negative_scenarios": int((Y6 == 0).sum())
    }
}

with open(META, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

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
    "tensor_nonempty",
    X.shape[0] > 0,
    X.shape
)

check(
    "sequence_length_24",
    X.shape[1] == 24,
    X.shape
)

check(
    "feature_count_positive",
    X.shape[2] > 0,
    X.shape
)

check(
    "train_exists",
    (scenario_split == "train").sum() > 0,
    int((scenario_split == "train").sum())
)

check(
    "validation_exists",
    (scenario_split == "validation").sum() > 0,
    int((scenario_split == "validation").sum())
)

check(
    "test_exists",
    (scenario_split == "test").sum() > 0,
    int((scenario_split == "test").sum())
)

check(
    "target_3step_binary",
    set(np.unique(Y3)).issubset({0, 1}),
    np.unique(Y3).tolist()
)

check(
    "target_6step_binary",
    set(np.unique(Y6)).issubset({0, 1}),
    np.unique(Y6).tolist()
)

check(
    "horizon_targets_distinct",
    not np.array_equal(Y3, Y6),
    f"3step={Y3.sum()}, 6step={Y6.sum()}"
)

check(
    "no_nan",
    not np.isnan(X).any(),
    f"NaNs={np.isnan(X).sum()}"
)

check(
    "no_inf",
    not np.isinf(X).any(),
    f"Infs={np.isinf(X).sum()}"
)

passed = sum(c["passed"] for c in checks)
failed = len(checks) - passed

print()
print("QUALITY GATE")
print("-" * 100)

for c in checks:
    print(
        f"[{'PASS' if c['passed'] else 'FAIL'}] "
        f"{c['name']} -- {c['detail']}"
    )

print()
print("=" * 100)
print(f"STATUS : {'PASS' if failed == 0 else 'FAIL'}")
print(f"CHECKS : {len(checks)}")
print(f"PASS   : {passed}")
print(f"FAIL   : {failed}")
print("=" * 100)

if failed:
    raise RuntimeError(
        f"V2 tensor quality gate failed: {failed} failure(s)."
    )

print()
print("PHASE 4.4 V2: PASS")
print(f"OUTPUT: {OUT}")
