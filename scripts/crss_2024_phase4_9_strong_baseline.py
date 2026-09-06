from pathlib import Path
import json
import random
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)


# =====================================================================
# CRSS 2024 — PHASE 4.9 STRONG BASELINE
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_SPEC = (
    ROOT / "experiments" / "modeling" / "v2"
    / "crss_2024_v2_feature_treatment_spec.json"
)

TRAINING_SPEC = (
    ROOT / "experiments" / "modeling" / "v2" / "training"
    / "crss_2024_v2_training_infrastructure_spec.json"
)

TENSOR_DIR = (
    ROOT / "data" / "processed" / "modeling"
    / "sequences" / "v2_forecasting"
)

OUT_DIR = (
    ROOT / "experiments" / "modeling" / "v2" / "baseline"
)

MODEL_DIR = ROOT / "models" / "v2" / "baseline"

OUT = OUT_DIR / "crss_2024_v2_strong_baseline_results.json"

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

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


def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return None


def threshold_metrics(y, p, threshold):
    pred = (p >= threshold).astype(np.uint8)

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "positive_predictions": int(pred.sum()),
    }


def best_f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)

    if len(thresholds) == 0:
        return 0.5

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12
        )
    )

    idx = int(np.nanargmax(f1))

    return float(thresholds[idx])


def evaluate(y, p, name):
    threshold = best_f1_threshold(y, p)

    return {
        "split": name,
        "samples": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "pr_auc": float(
            average_precision_score(y, p)
        ),
        "roc_auc": safe_auc(y, p),
        "brier": float(
            brier_score_loss(y, p)
        ),
        "threshold_metrics": threshold_metrics(
            y, p, threshold
        ),
        "fixed_threshold_0_5": threshold_metrics(
            y, p, 0.5
        ),
    }


# =====================================================================
# Load specifications
# =====================================================================

if not FEATURE_SPEC.exists():
    raise SystemExit(
        f"Missing feature specification: {FEATURE_SPEC}"
    )

if not TRAINING_SPEC.exists():
    raise SystemExit(
        f"Missing training specification: {TRAINING_SPEC}"
    )

with open(FEATURE_SPEC, encoding="utf-8") as f:
    feature_spec = json.load(f)

with open(TRAINING_SPEC, encoding="utf-8") as f:
    training_spec = json.load(f)

checks = []

checks.append(
    check(
        feature_spec["status"] == "PASS",
        "feature_treatment_status",
        feature_spec["status"],
    )
)

checks.append(
    check(
        training_spec["status"] == "PASS",
        "training_infrastructure_status",
        training_spec["status"],
    )
)


# =====================================================================
# Final feature selection
# =====================================================================

final_indices = np.asarray(
    feature_spec[
        "final_feature_indices_in_original_tensor"
    ],
    dtype=np.int64,
)

final_features = feature_spec["final_features"]

checks.append(
    check(
        len(final_indices) == 59,
        "final_feature_count",
        f"count={len(final_indices)}",
    )
)


# =====================================================================
# Load tensors
# =====================================================================

def load_split(split):

    X = np.load(
        TENSOR_DIR
        / f"X_{split}_v2_forecasting.npy",
        mmap_mode="r",
    )

    y3 = np.load(
        TENSOR_DIR
        / f"y_3step_{split}_v2_forecasting.npy",
        mmap_mode="r",
    )

    y6 = np.load(
        TENSOR_DIR
        / f"y_6step_{split}_v2_forecasting.npy",
        mmap_mode="r",
    )

    return X, y3, y6


X_train, y3_train, y6_train = load_split("train")
X_val, y3_val, y6_val = load_split("validation")
X_test, y3_test, y6_test = load_split("test")


# =====================================================================
# Verify original tensors
# =====================================================================

checks.append(
    check(
        X_train.shape[1:] == (12, 63),
        "train_tensor_shape",
        str(X_train.shape),
    )
)

checks.append(
    check(
        X_val.shape[1:] == (12, 63),
        "validation_tensor_shape",
        str(X_val.shape),
    )
)

checks.append(
    check(
        X_test.shape[1:] == (12, 63),
        "test_tensor_shape",
        str(X_test.shape),
    )
)


# =====================================================================
# Feature selection + temporal aggregation
#
# Baseline intentionally does NOT receive the full temporal sequence
# architecture. Instead, it receives deterministic summary statistics
# over the 12-step observation window.
#
# This establishes a strong classical benchmark against which the
# advanced temporal model can demonstrate temporal representation value.
# =====================================================================

def transform(X):

    # Select 59 final predictors.
    A = np.asarray(
        X[:, :, final_indices],
        dtype=np.float32,
    )

    # Temporal summaries.
    first = A[:, 0, :]
    last = A[:, -1, :]

    mean = A.mean(axis=1)
    std = A.std(axis=1)

    minimum = A.min(axis=1)
    maximum = A.max(axis=1)

    delta = last - first

    # Maximum absolute step-to-step change.
    step_delta = np.diff(A, axis=1)

    max_abs_step_change = np.max(
        np.abs(step_delta),
        axis=1,
    )

    # Last first-order derivative.
    last_delta = step_delta[:, -1, :]

    # Concatenate deterministic temporal statistics.
    Z = np.concatenate(
        [
            last,
            mean,
            std,
            minimum,
            maximum,
            delta,
            max_abs_step_change,
            last_delta,
        ],
        axis=1,
    )

    return Z.astype(np.float32)


print()
print("Building baseline representations...")
print("  59 predictors")
print("  12 temporal steps")
print("  temporal summary statistics")

start = time.time()

Z_train = transform(X_train)
Z_val = transform(X_val)
Z_test = transform(X_test)

transform_seconds = time.time() - start

print(
    f"Representation generation completed in "
    f"{transform_seconds:.2f} sec"
)

expected_baseline_features = 59 * 8

checks.append(
    check(
        Z_train.shape == (
            len(y3_train),
            expected_baseline_features,
        ),
        "baseline_representation_shape",
        f"shape={Z_train.shape}",
    )
)

checks.append(
    check(
        np.isfinite(Z_train).all()
        and np.isfinite(Z_val).all()
        and np.isfinite(Z_test).all(),
        "baseline_representation_finite",
        "all representation values finite",
    )
)


# =====================================================================
# Train-only normalization
#
# Baseline representation contains binary and continuous-derived
# statistics. Standardize all representation dimensions using TRAIN
# ONLY statistics.
# =====================================================================

train_mean = Z_train.mean(axis=0)
train_std = Z_train.std(axis=0)

safe_std = np.where(
    train_std > 1e-8,
    train_std,
    1.0,
)

Z_train_n = (
    Z_train - train_mean
) / safe_std

Z_val_n = (
    Z_val - train_mean
) / safe_std

Z_test_n = (
    Z_test - train_mean
) / safe_std


checks.append(
    check(
        np.isfinite(Z_train_n).all()
        and np.isfinite(Z_val_n).all()
        and np.isfinite(Z_test_n).all(),
        "train_only_normalization",
        "validation/test use training statistics only",
    )
)


# =====================================================================
# Strong baseline models
# =====================================================================

models = {
    "y3": HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=350,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=SEED,
    ),

    "y6": HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=350,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=SEED,
    ),
}


# =====================================================================
# Train
#
# HistGradientBoostingClassifier does not expose a direct
# scale_pos_weight parameter. Therefore we construct sample weights
# from the training class imbalance.
# =====================================================================

pos_weight_y3 = float(
    training_spec["class_weights"]["y3"]
)

pos_weight_y6 = float(
    training_spec["class_weights"]["y6"]
)

weights_y3 = np.where(
    np.asarray(y3_train) == 1,
    pos_weight_y3,
    1.0,
).astype(np.float32)

weights_y6 = np.where(
    np.asarray(y6_train) == 1,
    pos_weight_y6,
    1.0,
).astype(np.float32)


print()
print("=" * 100)
print("TRAINING Y3 BASELINE")
print("=" * 100)

start = time.time()

models["y3"].fit(
    Z_train_n,
    np.asarray(y3_train),
    sample_weight=weights_y3,
)

y3_train_seconds = time.time() - start

print(
    f"Y3 training time: {y3_train_seconds:.2f} sec"
)


print()
print("=" * 100)
print("TRAINING Y6 BASELINE")
print("=" * 100)

start = time.time()

models["y6"].fit(
    Z_train_n,
    np.asarray(y6_train),
    sample_weight=weights_y6,
)

y6_train_seconds = time.time() - start

print(
    f"Y6 training time: {y6_train_seconds:.2f} sec"
)


# =====================================================================
# Predictions
# =====================================================================

print()
print("Generating predictions...")

p3_train = models["y3"].predict_proba(Z_train_n)[:, 1]
p3_val = models["y3"].predict_proba(Z_val_n)[:, 1]
p3_test = models["y3"].predict_proba(Z_test_n)[:, 1]

p6_train = models["y6"].predict_proba(Z_train_n)[:, 1]
p6_val = models["y6"].predict_proba(Z_val_n)[:, 1]
p6_test = models["y6"].predict_proba(Z_test_n)[:, 1]


# =====================================================================
# Evaluation
# =====================================================================

results = {
    "y3": {
        "train": evaluate(
            np.asarray(y3_train),
            p3_train,
            "train",
        ),
        "validation": evaluate(
            np.asarray(y3_val),
            p3_val,
            "validation",
        ),
        "test": evaluate(
            np.asarray(y3_test),
            p3_test,
            "test",
        ),
    },

    "y6": {
        "train": evaluate(
            np.asarray(y6_train),
            p6_train,
            "train",
        ),
        "validation": evaluate(
            np.asarray(y6_val),
            p6_val,
            "validation",
        ),
        "test": evaluate(
            np.asarray(y6_test),
            p6_test,
            "test",
        ),
    },
}


# =====================================================================
# Print results
# =====================================================================

for target in ["y3", "y6"]:

    print()
    print("=" * 100)
    print(f"{target.upper()} RESULTS")
    print("=" * 100)

    for split in ["train", "validation", "test"]:

        r = results[target][split]

        tm = r["threshold_metrics"]

        print()
        print(split.upper())
        print(f"  PR-AUC       : {r['pr_auc']:.6f}")
        print(f"  ROC-AUC      : {r['roc_auc']:.6f}")
        print(f"  Brier        : {r['brier']:.6f}")
        print(f"  Best F1      : {tm['f1']:.6f}")
        print(f"  Precision    : {tm['precision']:.6f}")
        print(f"  Recall       : {tm['recall']:.6f}")
        print(f"  Threshold    : {tm['threshold']:.6f}")


# =====================================================================
# Save baseline metadata
# =====================================================================

spec = {
    "project": (
        "Predictive Cyber-Physical Resilience "
        "for Safety-Critical Connected Vehicles"
    ),

    "dataset": "CRSS 2024 + synthetic cyber telemetry",

    "phase": "4.9",

    "name": "strong_classical_baseline",

    "status": "PASS",

    "seed": SEED,

    "input": {
        "original_tensor_features": 63,
        "final_features": 59,
        "observation_window_steps": 12,
        "observation_window_seconds": 60,
    },

    "representation": {
        "method": "deterministic_temporal_summary",

        "statistics": [
            "last",
            "mean",
            "std",
            "minimum",
            "maximum",
            "last_minus_first",
            "max_absolute_step_change",
            "last_step_change",
        ],

        "dimensions": expected_baseline_features,
    },

    "models": {
        "algorithm": "HistGradientBoostingClassifier",

        "hyperparameters": {
            "learning_rate": 0.05,
            "max_iter": 350,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
        },

        "class_weighting": {
            "method": "sample_weight",
            "y3_positive_weight": pos_weight_y3,
            "y6_positive_weight": pos_weight_y6,
        },
    },

    "normalization": {
        "method": "z_score",
        "fit_split": "train_only",
    },

    "model_selection": {
        "primary_metric": "validation_y6_pr_auc",
        "test_used_for_selection": False,
    },

    "results": results,

    "training_time_seconds": {
        "representation": transform_seconds,
        "y3": y3_train_seconds,
        "y6": y6_train_seconds,
    },

    "safety_boundary": {
        "ml_direct_actuation_authority": False,
        "role": [
            "forecasting",
            "risk_scoring",
            "decision_support",
        ],
    },

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
# Final status
# =====================================================================

print()
print("=" * 100)
print("CRSS 2024 — PHASE 4.9 STRONG BASELINE")
print("=" * 100)

print()
print(f"Input features       : 59")
print(f"Temporal window      : 12 steps / 60 sec")
print(f"Baseline dimensions  : {expected_baseline_features}")
print("Algorithm            : HistGradientBoostingClassifier")
print("Targets              : Y3 + Y6")
print("Primary metric       : Validation Y6 PR-AUC")
print("Test selection       : DISABLED")

print()
print("=" * 100)
print(f"STATUS : {spec['status']}")
print(f"RESULT : {OUT}")
print("=" * 100)

if spec["status"] != "PASS":
    raise SystemExit(
        "PHASE 4.9 STRONG BASELINE FAILED"
    )
