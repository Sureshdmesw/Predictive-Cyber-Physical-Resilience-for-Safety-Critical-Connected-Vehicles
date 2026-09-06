from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/temporal/crss_2024_temporal_cyber_physical_states.parquet"
REGISTRY = ROOT / "data/schemas/modeling/crss_2024_final_predictor_registry.json"
SPLIT = ROOT / "data/processed/modeling/splits/crss_2024_scenario_split.csv"

OUT_DIR = ROOT / "data/processed/modeling/sequences"
REPORT = ROOT / "experiments/modeling/sequence_tensor_quality.json"
SCALER_FILE = ROOT / "data/schemas/modeling/crss_2024_training_scaler.json"

SEQUENCE_LENGTH = 24

print("=" * 100)
print("CRSS 2024 PHASE 4.4 — SEQUENCE TENSOR CONSTRUCTION")
print("=" * 100)

# ------------------------------------------------------------
# LOAD REGISTRY
# ------------------------------------------------------------

with open(REGISTRY, "r", encoding="utf-8") as f:
    registry = json.load(f)

predictors = registry["predictors"]

if not predictors:
    raise RuntimeError("No predictors found in final registry.")

# Safety check: targets must never be predictors.
forbidden = {
    "synthetic_cyber_anomaly",
    "cyber_anomaly_severity",
    "future_anomaly_3step",
    "future_anomaly_6step",
    "physical_cyber_interaction_score",
    "connectivity_resilience_score",
    "cyber_physical_resilience_score",
    "predictive_warning_score",
    "predictive_warning_state",
}

leaked_predictors = sorted(set(predictors) & forbidden)

if leaked_predictors:
    raise RuntimeError(
        f"LEAKAGE DETECTED IN PREDICTORS: {leaked_predictors}"
    )

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

columns = (
    ["telemetry_scenario_id", "time_step",
     "future_anomaly_3step", "future_anomaly_6step"]
    + predictors
)

df = pd.read_parquet(INPUT, columns=columns)

print("Rows loaded:", f"{len(df):,}")
print("Predictors :", len(predictors))

# ------------------------------------------------------------
# VALIDATE SEQUENCE STRUCTURE
# ------------------------------------------------------------

df = df.sort_values(
    ["telemetry_scenario_id", "time_step"],
    kind="stable"
).reset_index(drop=True)

sequence_counts = (
    df.groupby("telemetry_scenario_id")["time_step"]
      .nunique()
)

if not (sequence_counts == SEQUENCE_LENGTH).all():
    bad = int((sequence_counts != SEQUENCE_LENGTH).sum())
    raise RuntimeError(
        f"{bad} sequences do not contain exactly {SEQUENCE_LENGTH} steps."
    )

# ------------------------------------------------------------
# LOAD SCENARIO SPLIT
# ------------------------------------------------------------

split_df = pd.read_csv(SPLIT)

if not split_df["telemetry_scenario_id"].is_unique:
    raise RuntimeError("Duplicate scenario IDs in split manifest.")

if set(split_df["split"]) != {"train", "validation", "test"}:
    raise RuntimeError("Split manifest does not contain train/validation/test.")

df = df.merge(
    split_df[["telemetry_scenario_id", "split"]],
    on="telemetry_scenario_id",
    how="left",
    validate="many_to_one"
)

if df["split"].isna().any():
    raise RuntimeError("Some temporal rows have no scenario split.")

# ------------------------------------------------------------
# CHECK NUMERIC PREDICTORS
# ------------------------------------------------------------

non_numeric = [
    c for c in predictors
    if not pd.api.types.is_numeric_dtype(df[c])
]

if non_numeric:
    raise RuntimeError(
        f"Non-numeric predictors require encoding: {non_numeric}"
    )

# ------------------------------------------------------------
# IMPUTE USING TRAINING STATISTICS ONLY
# ------------------------------------------------------------

train_rows = df[df["split"] == "train"]

train_medians = train_rows[predictors].median()

df[predictors] = df[predictors].fillna(train_medians)

remaining_nulls = int(
    df[predictors].isna().sum().sum()
)

if remaining_nulls:
    raise RuntimeError(
        f"Remaining predictor nulls after training-median imputation: "
        f"{remaining_nulls}"
    )

# ------------------------------------------------------------
# NORMALIZATION
# TRAINING DATA ONLY
# ------------------------------------------------------------

train_mean = train_rows[predictors].median()
train_std = train_rows[predictors].std(ddof=0)

# Recalculate after imputation.
train_imputed = train_rows[predictors].fillna(train_medians)

train_mean = train_imputed.mean()
train_std = train_imputed.std(ddof=0)

# Prevent division by zero.
train_std = train_std.replace(0, 1.0)

df[predictors] = (
    df[predictors] - train_mean
) / train_std

# ------------------------------------------------------------
# SCALER SERIALIZATION
# ------------------------------------------------------------

scaler = {
    "method": "standardization",
    "fit_scope": "train scenarios only",
    "sequence_length": SEQUENCE_LENGTH,
    "features": {
        feature: {
            "mean": float(train_mean[feature]),
            "std": float(train_std[feature])
        }
        for feature in predictors
    }
}

SCALER_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(SCALER_FILE, "w", encoding="utf-8") as f:
    json.dump(scaler, f, indent=2)

# ------------------------------------------------------------
# BUILD SEQUENCE ARRAYS
# ------------------------------------------------------------

feature_count = len(predictors)

sequence_arrays = {}
target3_arrays = {}
target6_arrays = {}

for split_name in ["train", "validation", "test"]:

    subset = df[df["split"] == split_name]

    X_list = []
    y3_list = []
    y6_list = []
    ids = []

    for scenario_id, grp in subset.groupby(
        "telemetry_scenario_id",
        sort=False
    ):

        grp = grp.sort_values("time_step")

        if len(grp) != SEQUENCE_LENGTH:
            raise RuntimeError(
                f"Invalid sequence length for {scenario_id}: {len(grp)}"
            )

        X_list.append(
            grp[predictors].to_numpy(dtype=np.float32)
        )

        # Sequence-level prediction target:
        # Does an anomaly occur in the future window of the sequence?
        y3_list.append(
            int(grp["future_anomaly_3step"].max())
        )

        y6_list.append(
            int(grp["future_anomaly_6step"].max())
        )

        ids.append(scenario_id)

    X = np.stack(X_list).astype(np.float32)
    y3 = np.asarray(y3_list, dtype=np.int8)
    y6 = np.asarray(y6_list, dtype=np.int8)
    ids = np.asarray(ids)

    sequence_arrays[split_name] = X
    target3_arrays[split_name] = y3
    target6_arrays[split_name] = y6

    np.save(
        OUT_DIR / f"X_{split_name}.npy",
        X
    )

    np.save(
        OUT_DIR / f"y_3step_{split_name}.npy",
        y3
    )

    np.save(
        OUT_DIR / f"y_6step_{split_name}.npy",
        y6
    )

    np.save(
        OUT_DIR / f"scenario_ids_{split_name}.npy",
        ids
    )

    print()
    print(
        f"{split_name.upper():12s}: "
        f"X={X.shape}, "
        f"y3={y3.shape}, "
        f"y6={y6.shape}"
    )

# ------------------------------------------------------------
# CHECK SCENARIO OVERLAP
# ------------------------------------------------------------

train_ids = set(
    sequence_arrays and
    np.load(OUT_DIR / "scenario_ids_train.npy").tolist()
)

validation_ids = set(
    np.load(OUT_DIR / "scenario_ids_validation.npy").tolist()
)

test_ids = set(
    np.load(OUT_DIR / "scenario_ids_test.npy").tolist()
)

overlap_tv = train_ids & validation_ids
overlap_tt = train_ids & test_ids
overlap_vt = validation_ids & test_ids

if overlap_tv or overlap_tt or overlap_vt:
    raise RuntimeError("SCENARIO LEAKAGE DETECTED.")

# ------------------------------------------------------------
# TARGET DISTRIBUTIONS
# ------------------------------------------------------------

target_stats = {}

for split_name in ["train", "validation", "test"]:

    y3 = target3_arrays[split_name]
    y6 = target6_arrays[split_name]

    target_stats[split_name] = {
        "3step": {
            "negative": int((y3 == 0).sum()),
            "positive": int((y3 == 1).sum()),
            "positive_rate": float(y3.mean())
        },
        "6step": {
            "negative": int((y6 == 0).sum()),
            "positive": int((y6 == 1).sum()),
            "positive_rate": float(y6.mean())
        }
    }

# ------------------------------------------------------------
# FEATURE RANGE SANITY
# ------------------------------------------------------------

all_finite = True

for split_name in ["train", "validation", "test"]:

    X = sequence_arrays[split_name]

    if not np.isfinite(X).all():
        all_finite = False

if not all_finite:
    raise RuntimeError("Non-finite values detected in model tensors.")

# ------------------------------------------------------------
# SAVE METADATA
# ------------------------------------------------------------

metadata = {
    "dataset": "CRSS 2024 Temporal Cyber-Physical State Dataset",
    "sequence_length": SEQUENCE_LENGTH,
    "predictor_count": feature_count,
    "predictors": predictors,
    "normalization": {
        "method": "standardization",
        "fit_on": "training scenarios only"
    },
    "targets": {
        "3step": "future_anomaly_3step",
        "6step": "future_anomaly_6step"
    },
    "split_unit": "telemetry_scenario_id",
    "splits": {
        name: {
            "sequences": int(len(sequence_arrays[name])),
            "rows": int(len(sequence_arrays[name]) * SEQUENCE_LENGTH)
        }
        for name in ["train", "validation", "test"]
    },
    "target_statistics": target_stats
}

with open(
    OUT_DIR / "crss_2024_sequence_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, indent=2)

# ------------------------------------------------------------
# QUALITY REPORT
# ------------------------------------------------------------

checks = {
    "predictor_registry_loaded": True,
    "no_forbidden_predictors": len(leaked_predictors) == 0,
    "all_sequences_length_24": bool(
        (sequence_counts == SEQUENCE_LENGTH).all()
    ),
    "train_validation_overlap": len(overlap_tv) == 0,
    "train_test_overlap": len(overlap_tt) == 0,
    "validation_test_overlap": len(overlap_vt) == 0,
    "all_tensors_finite": all_finite,
    "normalization_train_only": True,
    "train_sequence_count": len(train_ids),
    "validation_sequence_count": len(validation_ids),
    "test_sequence_count": len(test_ids),
}

passed = sum(
    bool(v) for k, v in checks.items()
    if isinstance(v, bool)
)

failed = sum(
    not bool(v) for k, v in checks.items()
    if isinstance(v, bool)
)

report = {
    "status": "PASS" if failed == 0 else "FAIL",
    "checks": checks,
    "predictor_count": feature_count,
    "sequence_length": SEQUENCE_LENGTH,
    "target_statistics": target_stats,
    "artifacts": [
        str(OUT_DIR / "X_train.npy"),
        str(OUT_DIR / "X_validation.npy"),
        str(OUT_DIR / "X_test.npy"),
        str(OUT_DIR / "y_3step_train.npy"),
        str(OUT_DIR / "y_3step_validation.npy"),
        str(OUT_DIR / "y_3step_test.npy"),
        str(OUT_DIR / "y_6step_train.npy"),
        str(OUT_DIR / "y_6step_validation.npy"),
        str(OUT_DIR / "y_6step_test.npy"),
        str(SCALER_FILE),
        str(OUT_DIR / "crss_2024_sequence_metadata.json")
    ]
}

REPORT.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print()
print("=" * 100)
print("PHASE 4.4 RESULT")
print("=" * 100)

print("STATUS           :", report["status"])
print("PREDICTORS       :", feature_count)
print("SEQUENCE LENGTH  :", SEQUENCE_LENGTH)

print()
print("SEQUENCES:")
print("TRAIN            :", f"{len(train_ids):,}")
print("VALIDATION       :", f"{len(validation_ids):,}")
print("TEST             :", f"{len(test_ids):,}")

print()
print("TARGET DISTRIBUTION:")

for split_name, stats in target_stats.items():
    print()
    print(split_name.upper())
    print(
        "  3-step:",
        stats["3step"]
    )
    print(
        "  6-step:",
        stats["6step"]
    )

print()
print("CHECKS:")

for name, value in checks.items():

    if isinstance(value, bool):
        state = "PASS" if value else "FAIL"
        print(f"[{state}] {name}")
    else:
        print(f"[INFO] {name}: {value}")

print()
print("SEQUENCE ARTIFACTS:")
print(OUT_DIR)

print()
print("SCALER:")
print(SCALER_FILE)

print()
print("QUALITY REPORT:")
print(REPORT)

print()
print("=" * 100)

if failed:
    raise RuntimeError(
        f"Sequence tensor quality gate failed: {failed} failure(s)."
    )

print("SEQUENCE TENSOR PIPELINE: PASS")
print("=" * 100)
