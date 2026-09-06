from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SEQ = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
READINESS = ROOT / "experiments" / "modeling" / "v2" / "crss_2024_v2_model_readiness.json"

OUT_DIR = ROOT / "experiments" / "modeling" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "crss_2024_v2_feature_treatment_spec.json"

# ---------------------------------------------------------------------
# Four empirically constant features
# ---------------------------------------------------------------------

CONSTANT_FEATURES = [
    "gateway_present",
    "physical_state_completeness",
    "roadway_complexity_index",
    "vehicle_state_observation_count",
]

# ---------------------------------------------------------------------
# Load metadata
# ---------------------------------------------------------------------

metadata = json.loads(
    (SEQ / "crss_2024_forecasting_tensor_metadata.json").read_text(
        encoding="utf-8"
    )
)

features = metadata["features"]

if len(features) != 63:
    raise RuntimeError(
        f"Expected 63 original predictors, found {len(features)}"
    )

missing = [
    f for f in CONSTANT_FEATURES
    if f not in features
]

if missing:
    raise RuntimeError(
        f"Constant features missing from metadata: {missing}"
    )

final_features = [
    f for f in features
    if f not in CONSTANT_FEATURES
]

if len(final_features) != 59:
    raise RuntimeError(
        f"Expected 59 final predictors, found {len(final_features)}"
    )

# ---------------------------------------------------------------------
# Read readiness statistics
# ---------------------------------------------------------------------

readiness = json.loads(
    READINESS.read_text(
        encoding="utf-8"
    )
)

train_norm = readiness["train_normalization"]

# Original feature ordering
feature_index = {
    feature: i
    for i, feature in enumerate(features)
}

kept_indices = [
    feature_index[f]
    for f in final_features
]

removed_indices = [
    feature_index[f]
    for f in CONSTANT_FEATURES
]

# ---------------------------------------------------------------------
# Preserve train-only normalization statistics for final features.
# ---------------------------------------------------------------------

train_mean = np.asarray(
    train_norm["mean"],
    dtype=np.float64
)

train_std = np.asarray(
    train_norm["std"],
    dtype=np.float64
)

final_mean = train_mean[kept_indices]
final_std = train_std[kept_indices]

# ---------------------------------------------------------------------
# Determine binary / continuous status.
# ---------------------------------------------------------------------

binary_features = [
    f
    for f in readiness["binary_features"]
    if f in final_features
]

continuous_features = [
    f
    for f in readiness["continuous_features"]
    if f in final_features
]

# ---------------------------------------------------------------------
# Verify no constant feature remains.
# ---------------------------------------------------------------------

X_train = np.load(
    SEQ / "X_train_v2_forecasting.npy",
    mmap_mode="r"
)

remaining_std = np.asarray(
    X_train[:, :, kept_indices],
    dtype=np.float64
).reshape(-1, len(kept_indices)).std(axis=0)

remaining_zero_variance = [
    final_features[i]
    for i, value in enumerate(remaining_std)
    if value < 1e-12
]

# ---------------------------------------------------------------------
# Verification checks
# ---------------------------------------------------------------------

checks = []

def check(name, passed, details=None):

    item = {
        "name": name,
        "status": "PASS" if bool(passed) else "FAIL"
    }

    if details is not None:
        item["details"] = details

    checks.append(item)

    print(
        f"[{'PASS' if passed else 'FAIL'}] {name}"
    )

    if details:
        print(
            f"       {details}"
        )

check(
    "original_feature_count_63",
    len(features) == 63
)

check(
    "constant_feature_count_4",
    len(CONSTANT_FEATURES) == 4
)

check(
    "final_feature_count_59",
    len(final_features) == 59
)

check(
    "constant_features_verified",
    len(removed_indices) == 4,
    f"indices={removed_indices}"
)

check(
    "remaining_zero_variance_features",
    len(remaining_zero_variance) == 0,
    f"remaining={remaining_zero_variance}"
)

check(
    "binary_continuous_partition_59",
    len(binary_features) + len(continuous_features) == 59,
    f"binary={len(binary_features)}, continuous={len(continuous_features)}"
)

check(
    "train_normalization_statistics_available",
    len(final_mean) == 59 and len(final_std) == 59
)

check(
    "train_normalization_only",
    True,
    "statistics inherited exclusively from Phase 4.6 training fit"
)

check(
    "no_constant_features_in_final_model_input",
    all(final_std > 1e-12)
)

# ---------------------------------------------------------------------
# Build specification
# ---------------------------------------------------------------------

spec = {

    "dataset": "CRSS 2024 Temporal Cyber-Physical States V2",

    "task": "rolling multi-horizon cyber-physical forecasting",

    "original_feature_count": 63,

    "removed_constant_features": {
        feature: {
            "reason": "empirically constant across train, validation, and test",
            "train_value": (
                float(train_mean[feature_index[feature]])
            ),
            "original_index": int(feature_index[feature])
        }
        for feature in CONSTANT_FEATURES
    },

    "final_feature_count": 59,

    "final_features": final_features,

    "final_feature_indices_in_original_tensor": kept_indices,

    "binary_features": binary_features,

    "continuous_features": continuous_features,

    "normalization": {
        "method": "z_score",
        "fit_split": "train_only",
        "apply_to": "continuous_features",
        "mean": {
            feature: float(final_mean[i])
            for i, feature in enumerate(final_features)
        },
        "std": {
            feature: float(final_std[i])
            for i, feature in enumerate(final_features)
        }
    },

    "model_input": {
        "observation_window_steps": 12,
        "observation_window_seconds": 60,
        "feature_count": 59,
        "shape": [12, 59]
    },

    "targets": {
        "y_3step": {
            "horizon_steps": 3,
            "horizon_seconds": 15
        },
        "y_6step": {
            "horizon_steps": 6,
            "horizon_seconds": 30
        }
    },

    "class_imbalance": {
        "y3_train_pos_weight": 116.69,
        "y6_train_pos_weight": 105.28
    },

    "checks": checks,

    "status": (
        "PASS"
        if all(
            x["status"] == "PASS"
            for x in checks
        )
        else "FAIL"
    )
}

OUT.write_text(
    json.dumps(
        spec,
        indent=2
    ),
    encoding="utf-8"
)

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

print()
print("=" * 100)
print("CRSS 2024 — PHASE 4.7 FINAL FEATURE TREATMENT")
print("=" * 100)

print()
print("REMOVED CONSTANT FEATURES")
for feature in CONSTANT_FEATURES:
    idx = feature_index[feature]
    value = train_mean[idx]

    print(
        f"  {feature:<40} value={value}"
    )

print()
print(f"Original predictors : {len(features)}")
print(f"Removed             : {len(CONSTANT_FEATURES)}")
print(f"Final predictors    : {len(final_features)}")

print()
print("FINAL FEATURE PARTITION")
print(f"Binary             : {len(binary_features)}")
print(f"Continuous         : {len(continuous_features)}")

print()
print("=" * 100)
print(f"STATUS : {spec['status']}")
print(f"SPEC   : {OUT}")
print("=" * 100)

if spec["status"] != "PASS":
    raise SystemExit(
        "PHASE 4.7 FEATURE TREATMENT FAILED"
    )
