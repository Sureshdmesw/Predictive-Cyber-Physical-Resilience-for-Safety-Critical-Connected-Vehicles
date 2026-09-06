from pathlib import Path
import json
import hashlib
import platform
import sys
import random

import numpy as np


# =====================================================================
# CRSS 2024 — PHASE 4.8 TRAINING INFRASTRUCTURE
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_SPEC = (
    ROOT
    / "experiments"
    / "modeling"
    / "v2"
    / "crss_2024_v2_feature_treatment_spec.json"
)

METADATA = (
    ROOT
    / "data"
    / "processed"
    / "modeling"
    / "sequences"
    / "v2_forecasting"
    / "crss_2024_forecasting_tensor_metadata.json"
)

TENSOR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "modeling"
    / "sequences"
    / "v2_forecasting"
)

OUT_DIR = (
    ROOT
    / "experiments"
    / "modeling"
    / "v2"
    / "training"
)

OUT = OUT_DIR / "crss_2024_v2_training_infrastructure_spec.json"


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------
# Expected tensors
# ---------------------------------------------------------------------

EXPECTED_FILES = [
    "X_train_v2_forecasting.npy",
    "X_validation_v2_forecasting.npy",
    "X_test_v2_forecasting.npy",
    "y_3step_train_v2_forecasting.npy",
    "y_3step_validation_v2_forecasting.npy",
    "y_3step_test_v2_forecasting.npy",
    "y_6step_train_v2_forecasting.npy",
    "y_6step_validation_v2_forecasting.npy",
    "y_6step_test_v2_forecasting.npy",
    "scenario_ids_train_v2_forecasting.npy",
    "scenario_ids_validation_v2_forecasting.npy",
    "scenario_ids_test_v2_forecasting.npy",
    "prediction_origins_train_v2_forecasting.npy",
    "prediction_origins_validation_v2_forecasting.npy",
    "prediction_origins_test_v2_forecasting.npy",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def check(condition, name, detail=""):
    status = "PASS" if condition else "FAIL"

    print(
        f"[{status}] {name}"
        + (f"\n       {detail}" if detail else "")
    )

    return {
        "name": name,
        "status": status,
        "detail": detail,
    }


# ---------------------------------------------------------------------
# Load Phase 4.7 specification
# ---------------------------------------------------------------------

if not FEATURE_SPEC.exists():
    raise SystemExit(
        f"Missing Phase 4.7 feature specification:\n{FEATURE_SPEC}"
    )

with open(FEATURE_SPEC, encoding="utf-8") as f:
    feature_spec = json.load(f)


# ---------------------------------------------------------------------
# Load forecasting metadata
# ---------------------------------------------------------------------

if not METADATA.exists():
    raise SystemExit(
        f"Missing forecasting metadata:\n{METADATA}"
    )

with open(METADATA, encoding="utf-8") as f:
    metadata = json.load(f)


features = metadata["features"]

final_features = feature_spec["final_features"]
final_indices = feature_spec[
    "final_feature_indices_in_original_tensor"
]

binary_features = feature_spec["binary_features"]
continuous_features = feature_spec["continuous_features"]

checks = []


# =====================================================================
# 1. Feature treatment validation
# =====================================================================

checks.append(
    check(
        feature_spec["status"] == "PASS",
        "phase_4_7_status",
        feature_spec["status"],
    )
)

checks.append(
    check(
        len(features) == 63,
        "original_tensor_feature_count",
        f"metadata={len(features)}",
    )
)

checks.append(
    check(
        len(final_features) == 59,
        "final_model_feature_count",
        f"final={len(final_features)}",
    )
)

checks.append(
    check(
        len(final_indices) == 59,
        "final_feature_index_count",
        f"indices={len(final_indices)}",
    )
)

checks.append(
    check(
        len(binary_features) == 27
        and len(continuous_features) == 32,
        "binary_continuous_partition",
        f"binary={len(binary_features)}, continuous={len(continuous_features)}",
    )
)


# ---------------------------------------------------------------------
# Verify feature indices
# ---------------------------------------------------------------------

reconstructed = [
    features[i]
    for i in final_indices
]

checks.append(
    check(
        reconstructed == final_features,
        "feature_index_mapping",
        "59 final features correctly map to original 63-feature tensor",
    )
)


# =====================================================================
# 2. Verify tensors
# =====================================================================

tensor_status = {}

for filename in EXPECTED_FILES:

    path = TENSOR_DIR / filename

    exists = path.exists()

    checks.append(
        check(
            exists,
            f"tensor_exists_{filename}",
            str(path) if exists else "MISSING",
        )
    )

    tensor_status[filename] = {
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else None,
    }


# =====================================================================
# 3. Tensor shape validation
# =====================================================================

expected_shapes = {
    "X_train_v2_forecasting.npy": (444052, 12, 63),
    "X_validation_v2_forecasting.npy": (94584, 12, 63),
    "X_test_v2_forecasting.npy": (95851, 12, 63),
}

for filename, expected_shape in expected_shapes.items():

    path = TENSOR_DIR / filename

    if not path.exists():
        continue

    arr = np.load(path, mmap_mode="r")

    checks.append(
        check(
            tuple(arr.shape) == expected_shape,
            f"shape_{filename}",
            f"actual={tuple(arr.shape)}, expected={expected_shape}",
        )
    )


# =====================================================================
# 4. Target validation
# =====================================================================

target_specs = {
    "y_3step_train_v2_forecasting.npy": 444052,
    "y_3step_validation_v2_forecasting.npy": 94584,
    "y_3step_test_v2_forecasting.npy": 95851,

    "y_6step_train_v2_forecasting.npy": 444052,
    "y_6step_validation_v2_forecasting.npy": 94584,
    "y_6step_test_v2_forecasting.npy": 95851,
}

target_statistics = {}

for filename, expected_rows in target_specs.items():

    path = TENSOR_DIR / filename

    if not path.exists():
        continue

    arr = np.load(path, mmap_mode="r")

    unique = np.unique(arr)

    valid_binary = set(unique.tolist()).issubset({0, 1})

    positives = int(np.sum(arr))
    total = int(arr.shape[0])

    target_statistics[filename] = {
        "rows": total,
        "positives": positives,
        "positive_rate": positives / total,
        "unique_values": unique.tolist(),
    }

    checks.append(
        check(
            total == expected_rows,
            f"target_rows_{filename}",
            f"rows={total}",
        )
    )

    checks.append(
        check(
            valid_binary,
            f"target_binary_{filename}",
            f"unique={unique.tolist()}",
        )
    )


# =====================================================================
# 5. Scenario isolation validation
# =====================================================================

scenario_arrays = {}

for split in ["train", "validation", "test"]:

    filename = f"scenario_ids_{split}_v2_forecasting.npy"
    path = TENSOR_DIR / filename

    if not path.exists():
        continue

    arr = np.load(path, mmap_mode="r")

    scenario_arrays[split] = np.asarray(arr)

    checks.append(
        check(
            len(np.unique(arr)) > 0,
            f"scenario_ids_nonempty_{split}",
            f"unique={len(np.unique(arr))}",
        )
    )


if len(scenario_arrays) == 3:

    train_ids = set(scenario_arrays["train"].tolist())
    val_ids = set(scenario_arrays["validation"].tolist())
    test_ids = set(scenario_arrays["test"].tolist())

    checks.append(
        check(
            len(train_ids & val_ids) == 0,
            "scenario_leakage_train_validation",
            f"overlap={len(train_ids & val_ids)}",
        )
    )

    checks.append(
        check(
            len(train_ids & test_ids) == 0,
            "scenario_leakage_train_test",
            f"overlap={len(train_ids & test_ids)}",
        )
    )

    checks.append(
        check(
            len(val_ids & test_ids) == 0,
            "scenario_leakage_validation_test",
            f"overlap={len(val_ids & test_ids)}",
        )
    )


# =====================================================================
# 6. Prediction-origin validation
# =====================================================================

origin_statistics = {}

for split in ["train", "validation", "test"]:

    filename = (
        f"prediction_origins_{split}_v2_forecasting.npy"
    )

    path = TENSOR_DIR / filename

    if not path.exists():
        continue

    arr = np.load(path, mmap_mode="r")

    unique = np.unique(arr)

    origin_statistics[split] = {
        "unique_origins": unique.tolist(),
        "count": len(unique),
    }

    checks.append(
        check(
            np.array_equal(
                unique,
                np.arange(11, 18)
            ),
            f"prediction_origins_{split}",
            f"origins={unique.tolist()}",
        )
    )


# =====================================================================
# 7. Normalization policy
# =====================================================================

normalization = feature_spec["normalization"]

checks.append(
    check(
        normalization["method"] == "z_score",
        "normalization_method",
        normalization["method"],
    )
)

checks.append(
    check(
        normalization["fit_split"] == "train_only",
        "normalization_train_only",
        normalization["fit_split"],
    )
)

mean_dict = normalization["mean"]
std_dict = normalization["std"]

checks.append(
    check(
        all(
            feature in mean_dict
            for feature in continuous_features
        ),
        "continuous_feature_means_available",
        f"required={len(continuous_features)}",
    )
)

checks.append(
    check(
        all(
            feature in std_dict
            for feature in continuous_features
        ),
        "continuous_feature_stds_available",
        f"required={len(continuous_features)}",
    )
)

checks.append(
    check(
        all(
            np.isfinite(mean_dict[feature])
            and np.isfinite(std_dict[feature])
            and std_dict[feature] > 0
            for feature in continuous_features
        ),
        "normalization_statistics_valid",
        "all continuous training statistics finite with std > 0",
    )
)


# =====================================================================
# 8. Class weighting
# =====================================================================

class_weights = {
    "y3": float(
        feature_spec["class_imbalance"]["y3_train_pos_weight"]
    ),
    "y6": float(
        feature_spec["class_imbalance"]["y6_train_pos_weight"]
    ),
}

checks.append(
    check(
        class_weights["y3"] > 1
        and class_weights["y6"] > 1,
        "positive_class_weighting_required",
        f"y3={class_weights['y3']}, y6={class_weights['y6']}",
    )
)


# =====================================================================
# 9. Training policy
# =====================================================================

training_policy = {
    "seed": SEED,

    "task": "rolling_multi_horizon_binary_forecasting",

    "input": {
        "observation_window_steps": 12,
        "observation_window_seconds": 60,
        "feature_count": 59,
        "shape": [12, 59],
    },

    "targets": {
        "y3": {
            "horizon_steps": 3,
            "horizon_seconds": 15,
            "loss": "weighted_binary_cross_entropy",
            "positive_class_weight": class_weights["y3"],
        },
        "y6": {
            "horizon_steps": 6,
            "horizon_seconds": 30,
            "loss": "weighted_binary_cross_entropy",
            "positive_class_weight": class_weights["y6"],
        },
    },

    "feature_processing": {
        "binary_features": binary_features,
        "continuous_features": continuous_features,
        "continuous_normalization": "train_only_z_score",
        "constant_features_removed": feature_spec[
            "removed_constant_features"
        ],
    },

    "data_split": {
        "unit": "telemetry_scenario_id",
        "train_validation_test_isolation": True,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
    },

    "optimization": {
        "optimizer": "AdamW",
        "gradient_clipping": True,
        "gradient_clip_norm": 1.0,
        "mixed_precision": True,
        "early_stopping": True,
        "checkpoint_best_validation_metric": True,
    },

    "primary_selection_metric": {
        "name": "y6_pr_auc",
        "direction": "maximize",
        "reason": (
            "rare-event forecasting requires precision-recall "
            "sensitivity rather than ROC-AUC alone"
        ),
    },

    "secondary_metrics": [
        "y3_pr_auc",
        "y6_pr_auc",
        "y3_roc_auc",
        "y6_roc_auc",
        "y3_precision",
        "y6_precision",
        "y3_recall",
        "y6_recall",
        "y3_f1",
        "y6_f1",
        "y3_brier",
        "y6_brier",
        "y3_ece",
        "y6_ece",
        "low_fpr_recall",
        "warning_lead_time",
        "false_alarm_persistence",
        "missed_event_rate",
    ],

    "validation_policy": {
        "used_for": [
            "early_stopping",
            "hyperparameter_selection",
            "threshold_selection",
            "calibration_selection",
        ],
        "test_used": False,
    },

    "test_policy": {
        "used_only_after_model_freeze": True,
        "used_for_final_evaluation": True,
        "used_for_hyperparameter_selection": False,
    },

    "safety_boundary": {
        "ml_direct_actuation_authority": False,
        "ml_role": [
            "prediction",
            "risk_scoring",
            "warning",
            "decision_support",
        ],
        "deterministic_safety_policy_remains_separate": True,
    },
}


# =====================================================================
# 10. Environment record
# =====================================================================

environment = {
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
}


# =====================================================================
# 11. Final specification
# =====================================================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

spec = {
    "project": (
        "Predictive Cyber-Physical Resilience "
        "for Safety-Critical Connected Vehicles"
    ),

    "dataset": "CRSS 2024 + synthetic cyber telemetry",

    "phase": "4.8",

    "name": "training_infrastructure",

    "status": (
        "PASS"
        if all(x["status"] == "PASS" for x in checks)
        else "FAIL"
    ),

    "seed": SEED,

    "feature_treatment": {
        "original_feature_count": len(features),
        "final_feature_count": len(final_features),
        "binary_feature_count": len(binary_features),
        "continuous_feature_count": len(continuous_features),
        "final_features": final_features,
        "final_indices": final_indices,
    },

    "tensor_directory": str(TENSOR_DIR),

    "tensor_status": tensor_status,

    "target_statistics": target_statistics,

    "origin_statistics": origin_statistics,

    "normalization": normalization,

    "class_weights": class_weights,

    "training_policy": training_policy,

    "environment": environment,

    "checks": checks,
}


OUT.write_text(
    json.dumps(
        spec,
        indent=2,
        default=lambda x: (
            x.item()
            if hasattr(x, "item")
            else str(x)
        ),
    ),
    encoding="utf-8",
)


# =====================================================================
# Output
# =====================================================================

print()
print("=" * 100)
print("CRSS 2024 — PHASE 4.8 TRAINING INFRASTRUCTURE")
print("=" * 100)

print()
print("MODEL INPUT")
print(f"  Original predictors : {len(features)}")
print(f"  Final predictors    : {len(final_features)}")
print(f"  Binary              : {len(binary_features)}")
print(f"  Continuous          : {len(continuous_features)}")
print("  Shape               : [12, 59]")

print()
print("TARGETS")
print(
    f"  Y3 : 15 sec horizon | "
    f"positive weight={class_weights['y3']}"
)
print(
    f"  Y6 : 30 sec horizon | "
    f"positive weight={class_weights['y6']}"
)

print()
print("TRAINING POLICY")
print("  Optimizer           : AdamW")
print("  Mixed precision     : enabled")
print("  Gradient clipping   : 1.0")
print("  Early stopping      : enabled")
print("  Selection metric    : Y6 PR-AUC")
print("  Test isolation      : enforced")

print()
print("=" * 100)
print(f"STATUS : {spec['status']}")
print(f"SPEC   : {OUT}")
print("=" * 100)

if spec["status"] != "PASS":
    raise SystemExit(
        "PHASE 4.8 TRAINING INFRASTRUCTURE FAILED"
    )
