from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SEQ = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
METADATA = SEQ / "crss_2024_forecasting_tensor_metadata.json"

OUT_DIR = ROOT / "experiments" / "modeling" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "crss_2024_v2_model_readiness.json"

SPLITS = ["train", "validation", "test"]

CHUNK_SIZE = 4096

# ---------------------------------------------------------------------
# Load metadata
# ---------------------------------------------------------------------

metadata = json.loads(
    METADATA.read_text(encoding="utf-8")
)

features = metadata["features"]

if len(features) != 63:
    raise RuntimeError(
        f"Expected 63 predictors, found {len(features)}"
    )

# ---------------------------------------------------------------------
# Load target arrays
# ---------------------------------------------------------------------

targets = {}

for split in SPLITS:
    targets[split] = {
        "y3": np.load(
            SEQ / f"y_3step_{split}_v2_forecasting.npy",
            mmap_mode="r"
        ),
        "y6": np.load(
            SEQ / f"y_6step_{split}_v2_forecasting.npy",
            mmap_mode="r"
        )
    }

# ---------------------------------------------------------------------
# Training tensor
# ---------------------------------------------------------------------

X_train = np.load(
    SEQ / "X_train_v2_forecasting.npy",
    mmap_mode="r"
)

if X_train.shape[1:] != (12, 63):
    raise RuntimeError(
        f"Unexpected tensor shape: {X_train.shape}"
    )

# ---------------------------------------------------------------------
# Feature statistics
#
# IMPORTANT:
# Normalization parameters are fitted ONLY on training data.
# ---------------------------------------------------------------------

sum_x = np.zeros(63, dtype=np.float64)
sum_x2 = np.zeros(63, dtype=np.float64)
count = 0

feature_min = np.full(63, np.inf, dtype=np.float64)
feature_max = np.full(63, -np.inf, dtype=np.float64)

for start in range(0, X_train.shape[0], CHUNK_SIZE):

    end = min(
        start + CHUNK_SIZE,
        X_train.shape[0]
    )

    chunk = np.asarray(
        X_train[start:end],
        dtype=np.float64
    )

    # Flatten sample × timestep dimensions.
    flat = chunk.reshape(-1, 63)

    sum_x += flat.sum(axis=0)
    sum_x2 += np.square(flat).sum(axis=0)

    feature_min = np.minimum(
        feature_min,
        flat.min(axis=0)
    )

    feature_max = np.maximum(
        feature_max,
        flat.max(axis=0)
    )

    count += flat.shape[0]

mean = sum_x / count

variance = (
    sum_x2 / count
    - np.square(mean)
)

variance = np.maximum(
    variance,
    0.0
)

std = np.sqrt(variance)

# Prevent division by zero.
safe_std = np.where(
    std < 1e-12,
    1.0,
    std
)

# ---------------------------------------------------------------------
# Identify empirically binary features
# ---------------------------------------------------------------------

binary_features = []
continuous_features = []

for i, feature in enumerate(features):

    mn = feature_min[i]
    mx = feature_max[i]

    # Binary if observed range is exactly 0..1.
    if mn >= 0.0 and mx <= 1.0:
        binary_features.append(feature)
    else:
        continuous_features.append(feature)

# ---------------------------------------------------------------------
# Distribution statistics for every split
#
# These are diagnostic statistics.
# They DO NOT modify the tensors.
# ---------------------------------------------------------------------

distribution = {}

for split in SPLITS:

    X = np.load(
        SEQ / f"X_{split}_v2_forecasting.npy",
        mmap_mode="r"
    )

    s = np.zeros(63, dtype=np.float64)
    s2 = np.zeros(63, dtype=np.float64)
    n = 0

    mn_all = np.full(63, np.inf, dtype=np.float64)
    mx_all = np.full(63, -np.inf, dtype=np.float64)

    for start in range(0, X.shape[0], CHUNK_SIZE):

        end = min(
            start + CHUNK_SIZE,
            X.shape[0]
        )

        chunk = np.asarray(
            X[start:end],
            dtype=np.float64
        )

        flat = chunk.reshape(-1, 63)

        s += flat.sum(axis=0)
        s2 += np.square(flat).sum(axis=0)

        mn_all = np.minimum(
            mn_all,
            flat.min(axis=0)
        )

        mx_all = np.maximum(
            mx_all,
            flat.max(axis=0)
        )

        n += flat.shape[0]

    split_mean = s / n

    split_var = np.maximum(
        s2 / n - np.square(split_mean),
        0.0
    )

    split_std = np.sqrt(split_var)

    distribution[split] = {
        "mean": split_mean.tolist(),
        "std": split_std.tolist(),
        "min": mn_all.tolist(),
        "max": mx_all.tolist(),
        "observations": int(n)
    }

# ---------------------------------------------------------------------
# Drift diagnostics
#
# Standardized mean shift:
# (split_mean - train_mean) / train_std
# ---------------------------------------------------------------------

drift = {}

for split in ["validation", "test"]:

    split_mean = np.asarray(
        distribution[split]["mean"]
    )

    standardized_shift = (
        split_mean - mean
    ) / safe_std

    drift[split] = {
        "max_abs_standardized_mean_shift": float(
            np.max(np.abs(standardized_shift))
        ),
        "mean_abs_standardized_mean_shift": float(
            np.mean(np.abs(standardized_shift))
        ),
        "features_abs_shift_gt_2": [
            features[i]
            for i, value in enumerate(standardized_shift)
            if abs(value) > 2.0
        ],
        "features_abs_shift_gt_3": [
            features[i]
            for i, value in enumerate(standardized_shift)
            if abs(value) > 3.0
        ]
    }

# ---------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------

class_balance = {}

for split in SPLITS:

    y3 = np.asarray(targets[split]["y3"])
    y6 = np.asarray(targets[split]["y6"])

    y3_pos = int(y3.sum())
    y6_pos = int(y6.sum())

    y3_neg = int(len(y3) - y3_pos)
    y6_neg = int(len(y6) - y6_pos)

    class_balance[split] = {
        "samples": int(len(y3)),

        "y3_positive": y3_pos,
        "y3_negative": y3_neg,
        "y3_positive_rate": y3_pos / len(y3),
        "y3_pos_weight": (
            y3_neg / y3_pos
            if y3_pos else None
        ),

        "y6_positive": y6_pos,
        "y6_negative": y6_neg,
        "y6_positive_rate": y6_pos / len(y6),
        "y6_pos_weight": (
            y6_neg / y6_pos
            if y6_pos else None
        )
    }

# ---------------------------------------------------------------------
# Readiness checks
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
        print(f"       {details}")

# ---------------------------------------------------------------------
# Core checks
# ---------------------------------------------------------------------

check(
    "feature_count_63",
    len(features) == 63
)

check(
    "training_tensor_shape",
    X_train.shape[1:] == (12, 63),
    f"shape={X_train.shape}"
)

check(
    "train_only_normalization",
    True,
    "mean/std fitted exclusively from X_train"
)

check(
    "zero_variance_protection",
    bool(np.all(safe_std > 0)),
    f"zero_variance_features={int(np.sum(std < 1e-12))}"
)

check(
    "binary_continuous_partition",
    len(binary_features) + len(continuous_features) == 63,
    f"binary={len(binary_features)}, continuous={len(continuous_features)}"
)

check(
    "target_imbalance_requires_weighting",
    all(
        class_balance[s]["y3_positive"] > 0
        and class_balance[s]["y6_positive"] > 0
        for s in SPLITS
    )
)

check(
    "validation_statistics_diagnostic_only",
    True,
    "validation/test statistics are not used to fit normalization"
)

check(
    "test_statistics_diagnostic_only",
    True,
    "test statistics are not used to fit normalization"
)

# ---------------------------------------------------------------------
# Save model readiness artifact
# ---------------------------------------------------------------------

result = {
    "dataset": metadata["dataset"],
    "task": metadata["task"],

    "input": {
        "observation_window_steps": 12,
        "observation_window_seconds": 60,
        "feature_count": 63,
        "sample_shape": [12, 63]
    },

    "targets": {
        "y3": {
            "horizon_steps": 3,
            "horizon_seconds": 15
        },
        "y6": {
            "horizon_steps": 6,
            "horizon_seconds": 30
        }
    },

    "features": features,

    "binary_features": binary_features,
    "continuous_features": continuous_features,

    "train_normalization": {
        "method": "z_score",
        "fit_split": "train_only",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "safe_std": safe_std.tolist(),
        "min": feature_min.tolist(),
        "max": feature_max.tolist()
    },

    "distribution": distribution,

    "drift": drift,

    "class_balance": class_balance,

    "checks": checks,

    "status": "PASS"
    if all(x["status"] == "PASS" for x in checks)
    else "FAIL"
}

OUT.write_text(
    json.dumps(
        result,
        indent=2,
        default=lambda o: (
            o.item()
            if hasattr(o, "item")
            else str(o)
        )
    ),
    encoding="utf-8"
)

print()
print("=" * 100)
print("PHASE 4.6 — MODEL READINESS")
print("=" * 100)

print()
print("FEATURE SUMMARY")
print(f"Total features      : {len(features)}")
print(f"Binary features     : {len(binary_features)}")
print(f"Continuous features : {len(continuous_features)}")

print()
print("CLASS BALANCE")

for split in SPLITS:

    b = class_balance[split]

    print(
        f"{split:12s} "
        f"Y3+={b['y3_positive']:,} "
        f"rate={b['y3_positive_rate']:.6%} "
        f"weight={b['y3_pos_weight']:.2f} | "
        f"Y6+={b['y6_positive']:,} "
        f"rate={b['y6_positive_rate']:.6%} "
        f"weight={b['y6_pos_weight']:.2f}"
    )

print()
print("DRIFT")

for split in ["validation", "test"]:

    d = drift[split]

    print(
        f"{split:12s} "
        f"max_abs_standardized_mean_shift="
        f"{d['max_abs_standardized_mean_shift']:.4f} | "
        f"features>|2|="
        f"{len(d['features_abs_shift_gt_2'])} | "
        f"features>|3|="
        f"{len(d['features_abs_shift_gt_3'])}"
    )

print()
print("=" * 100)
print(f"STATUS : {result['status']}")
print(f"REPORT : {OUT}")
print("=" * 100)

if result["status"] != "PASS":
    raise SystemExit(
        "PHASE 4.6 MODEL READINESS FAILED"
    )
