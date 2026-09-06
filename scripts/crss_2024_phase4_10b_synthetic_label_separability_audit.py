from pathlib import Path
import json
import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

TENSOR_DIR = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
FEATURE_SPEC = ROOT / "experiments" / "modeling" / "v2" / "crss_2024_v2_feature_treatment_spec.json"
METADATA = TENSOR_DIR / "crss_2024_forecasting_tensor_metadata.json"

OUT_DIR = ROOT / "experiments" / "modeling" / "v2" / "advanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "crss_2024_v2_synthetic_label_separability_audit.json"

MAX_TRAIN_SAMPLES = 120_000
RANDOM_SEED = 42


def check(name, condition, detail=""):
    return {
        "name": name,
        "status": "PASS" if condition else "FAIL",
        "detail": detail,
    }


print()
print("=" * 100)
print("CRSS 2024 — PHASE 4.10B SYNTHETIC CYBER-LABEL SEPARABILITY AUDIT")
print("=" * 100)
print()


# ============================================================================
# LOAD AUTHORITATIVE FEATURE SPEC
# ============================================================================

feature_spec = json.loads(
    FEATURE_SPEC.read_text(encoding="utf-8-sig")
)

metadata = json.loads(
    METADATA.read_text(encoding="utf-8-sig")
)

checks = []

features = list(metadata["features"])

final_features = list(
    feature_spec["final_features"]
)

final_indices = list(
    feature_spec["final_feature_indices_in_original_tensor"]
)

removed_constants = dict(
    feature_spec["removed_constant_features"]
)

checks.append(
    check(
        "original_feature_count",
        len(features) == 63,
        f"count={len(features)}",
    )
)

checks.append(
    check(
        "removed_constant_features",
        len(removed_constants) == 4,
        f"count={len(removed_constants)}",
    )
)

checks.append(
    check(
        "final_feature_count",
        len(final_features) == 59
        and len(final_indices) == 59,
        f"count={len(final_features)}",
    )
)

checks.append(
    check(
        "feature_index_alignment",
        all(
            features[idx] == feature
            for idx, feature in zip(
                final_indices,
                final_features,
            )
        ),
        "feature names match authoritative tensor indices",
    )
)


# ============================================================================
# LOAD DATA
# ============================================================================

print("[INFO] Loading forecasting tensors...")

X_train = np.load(
    TENSOR_DIR / "X_train_v2_forecasting.npy",
    mmap_mode="r",
)

X_val = np.load(
    TENSOR_DIR / "X_validation_v2_forecasting.npy",
    mmap_mode="r",
)

X_test = np.load(
    TENSOR_DIR / "X_test_v2_forecasting.npy",
    mmap_mode="r",
)

y3_train = np.load(
    TENSOR_DIR / "y_3step_train_v2_forecasting.npy"
)

y3_val = np.load(
    TENSOR_DIR / "y_3step_validation_v2_forecasting.npy"
)

y3_test = np.load(
    TENSOR_DIR / "y_3step_test_v2_forecasting.npy"
)

y6_train = np.load(
    TENSOR_DIR / "y_6step_train_v2_forecasting.npy"
)

y6_val = np.load(
    TENSOR_DIR / "y_6step_validation_v2_forecasting.npy"
)

y6_test = np.load(
    TENSOR_DIR / "y_6step_test_v2_forecasting.npy"
)


checks.append(
    check(
        "tensor_shapes",
        X_train.shape[1:] == (12, 63)
        and X_val.shape[1:] == (12, 63)
        and X_test.shape[1:] == (12, 63),
        (
            f"train={X_train.shape}; "
            f"val={X_val.shape}; "
            f"test={X_test.shape}"
        ),
    )
)


# ============================================================================
# FINAL 59 FEATURES
# ============================================================================

print("[INFO] Selecting authoritative 59 features...")

X_train59 = np.asarray(
    X_train[:, :, final_indices],
    dtype=np.float32,
)

X_val59 = np.asarray(
    X_val[:, :, final_indices],
    dtype=np.float32,
)

X_test59 = np.asarray(
    X_test[:, :, final_indices],
    dtype=np.float32,
)


checks.append(
    check(
        "final_input_shape",
        X_train59.shape[1:] == (12, 59)
        and X_val59.shape[1:] == (12, 59)
        and X_test59.shape[1:] == (12, 59),
        (
            f"train={X_train59.shape}; "
            f"val={X_val59.shape}; "
            f"test={X_test59.shape}"
        ),
    )
)


# ============================================================================
# TEMPORAL SUMMARY
# ============================================================================

def summarize(X):

    last = X[:, -1, :]

    mean = np.mean(
        X,
        axis=1,
    )

    std = np.std(
        X,
        axis=1,
    )

    minimum = np.min(
        X,
        axis=1,
    )

    maximum = np.max(
        X,
        axis=1,
    )

    delta = (
        X[:, -1, :]
        -
        X[:, 0, :]
    )

    step_delta = np.diff(
        X,
        axis=1,
    )

    max_abs_step = np.max(
        np.abs(step_delta),
        axis=1,
    )

    last_step = (
        X[:, -1, :]
        -
        X[:, -2, :]
    )

    return np.concatenate(
        [
            last,
            mean,
            std,
            minimum,
            maximum,
            delta,
            max_abs_step,
            last_step,
        ],
        axis=1,
    ).astype(np.float32)


print("[INFO] Building temporal summaries...")

t0 = time.perf_counter()

S_train = summarize(X_train59)
S_val = summarize(X_val59)
S_test = summarize(X_test59)

summary_seconds = time.perf_counter() - t0

print(
    f"[INFO] Summary construction: "
    f"{summary_seconds:.2f} sec"
)

checks.append(
    check(
        "summary_dimensions",
        S_train.shape[1] == 472
        and S_val.shape[1] == 472
        and S_test.shape[1] == 472,
        f"dimension={S_train.shape[1]}",
    )
)

checks.append(
    check(
        "summary_finite",
        np.isfinite(S_train).all()
        and np.isfinite(S_val).all()
        and np.isfinite(S_test).all(),
    )
)


# ============================================================================
# TRAIN-ONLY STANDARDIZATION
# ============================================================================

train_mean = np.mean(
    S_train,
    axis=0,
)

train_std = np.std(
    S_train,
    axis=0,
)

train_std_safe = np.where(
    train_std < 1e-8,
    1.0,
    train_std,
)

S_train_z = (
    (S_train - train_mean)
    /
    train_std_safe
).astype(np.float32)

S_val_z = (
    (S_val - train_mean)
    /
    train_std_safe
).astype(np.float32)

S_test_z = (
    (S_test - train_mean)
    /
    train_std_safe
).astype(np.float32)


checks.append(
    check(
        "train_only_standardization",
        np.isfinite(S_train_z).all()
        and np.isfinite(S_val_z).all()
        and np.isfinite(S_test_z).all(),
    )
)


# ============================================================================
# MEMORY-SAFE BALANCED SUBSAMPLE
#
# Use the full validation/test sets.
# Only the classifier's training subset is bounded.
# ============================================================================

def balanced_indices(y, max_samples, seed):

    rng = np.random.default_rng(seed)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    # Preserve all positives whenever feasible.
    max_pos = len(pos_idx)

    target_pos = min(
        max_pos,
        max_samples // 2,
    )

    target_neg = min(
        len(neg_idx),
        max_samples - target_pos,
    )

    selected_pos = rng.choice(
        pos_idx,
        size=target_pos,
        replace=False,
    )

    selected_neg = rng.choice(
        neg_idx,
        size=target_neg,
        replace=False,
    )

    selected = np.concatenate(
        [
            selected_pos,
            selected_neg,
        ]
    )

    rng.shuffle(selected)

    return selected


# ============================================================================
# MODEL-BASED SEPARABILITY
# ============================================================================

results = {}

for horizon, y_train, y_val, y_test in [
    (
        "y3",
        y3_train,
        y3_val,
        y3_test,
    ),
    (
        "y6",
        y6_train,
        y6_val,
        y6_test,
    ),
]:

    print()
    print("-" * 100)
    print(f"[INFO] Evaluating {horizon}")
    print("-" * 100)

    train_pos = int(
        np.sum(y_train)
    )

    val_pos = int(
        np.sum(y_val)
    )

    test_pos = int(
        np.sum(y_test)
    )

    train_rate = float(
        np.mean(y_train)
    )

    val_rate = float(
        np.mean(y_val)
    )

    test_rate = float(
        np.mean(y_test)
    )

    selected = balanced_indices(
        y_train,
        MAX_TRAIN_SAMPLES,
        RANDOM_SEED,
    )

    X_sub = S_train_z[selected]
    y_sub = y_train[selected]

    sub_pos = int(
        np.sum(y_sub)
    )

    sub_neg = int(
        len(y_sub) - sub_pos
    )

    sample_weight = np.where(
        y_sub == 1,
        max(sub_neg, 1) / max(sub_pos, 1),
        1.0,
    ).astype(np.float32)

    print(
        f"[INFO] Classifier subset: "
        f"{len(selected):,} samples "
        f"({sub_pos:,} positive / "
        f"{sub_neg:,} negative)"
    )

    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
        max_bins=64,
    )

    t_train = time.perf_counter()

    clf.fit(
        X_sub,
        y_sub,
        sample_weight=sample_weight,
    )

    train_seconds = (
        time.perf_counter()
        -
        t_train
    )

    p_train = clf.predict_proba(
        X_sub
    )[:, 1]

    p_val = clf.predict_proba(
        S_val_z
    )[:, 1]

    p_test = clf.predict_proba(
        S_test_z
    )[:, 1]

    train_pr = average_precision_score(
        y_sub,
        p_train,
    )

    val_pr = average_precision_score(
        y_val,
        p_val,
    )

    test_pr = average_precision_score(
        y_test,
        p_test,
    )

    train_roc = roc_auc_score(
        y_sub,
        p_train,
    )

    val_roc = roc_auc_score(
        y_val,
        p_val,
    )

    test_roc = roc_auc_score(
        y_test,
        p_test,
    )

    results[horizon] = {
        "full_train_positive_count": train_pos,
        "validation_positive_count": val_pos,
        "test_positive_count": test_pos,
        "full_train_positive_rate": train_rate,
        "validation_positive_rate": val_rate,
        "test_positive_rate": test_rate,
        "classifier_train_samples": int(len(selected)),
        "classifier_train_positive_count": sub_pos,
        "classifier_train_negative_count": sub_neg,
        "train_pr_auc": float(train_pr),
        "validation_pr_auc": float(val_pr),
        "test_pr_auc": float(test_pr),
        "train_roc_auc": float(train_roc),
        "validation_roc_auc": float(val_roc),
        "test_roc_auc": float(test_roc),
        "training_seconds": float(train_seconds),
    }

    print(
        f"Train PR-AUC : {train_pr:.6f}"
    )

    print(
        f"Val PR-AUC   : {val_pr:.6f}"
    )

    print(
        f"Test PR-AUC  : {test_pr:.6f}"
    )

    print(
        f"Train ROC-AUC: {train_roc:.6f}"
    )

    print(
        f"Val ROC-AUC  : {val_roc:.6f}"
    )

    print(
        f"Test ROC-AUC : {test_roc:.6f}"
    )


# ============================================================================
# COMPLETE UNIVARIATE SEPARABILITY ANALYSIS
#
# Y6 is used because it is the longer forecasting horizon.
# ============================================================================

print()
print("=" * 100)
print("UNIVARIATE LAST-STEP SEPARABILITY — Y6")
print("=" * 100)

last_train = X_train59[:, -1, :]

pos_mask = (
    y6_train == 1
)

neg_mask = (
    y6_train == 0
)

feature_separation = []


for i, feature in enumerate(final_features):

    values = last_train[:, i]

    pos_values = values[pos_mask]
    neg_values = values[neg_mask]

    pos_mean = float(
        np.mean(pos_values)
    )

    neg_mean = float(
        np.mean(neg_values)
    )

    pos_std = float(
        np.std(pos_values)
    )

    neg_std = float(
        np.std(neg_values)
    )

    pooled_std = np.sqrt(
        (
            pos_std ** 2
            +
            neg_std ** 2
        )
        /
        2.0
    )

    if pooled_std < 1e-8:
        effect_size = 0.0
    else:
        effect_size = abs(
            pos_mean - neg_mean
        ) / pooled_std

    feature_separation.append(
        {
            "feature": feature,
            "effect_size": float(effect_size),
            "positive_mean": pos_mean,
            "negative_mean": neg_mean,
            "positive_std": pos_std,
            "negative_std": neg_std,
        }
    )


feature_separation.sort(
    key=lambda x: x["effect_size"],
    reverse=True,
)

top_20 = feature_separation[:20]

effect_ge_1 = [
    x
    for x in feature_separation
    if x["effect_size"] >= 1.0
]

effect_ge_2 = [
    x
    for x in feature_separation
    if x["effect_size"] >= 2.0
]

effect_ge_3 = [
    x
    for x in feature_separation
    if x["effect_size"] >= 3.0
]


for item in top_20:

    print(
        f"{item['feature']:<45}"
        f"effect={item['effect_size']:.4f}"
    )


checks.append(
    check(
        "univariate_features_complete",
        len(feature_separation) == 59,
        f"analyzed={len(feature_separation)}",
    )
)


# ============================================================================
# INTERPRETATION
# ============================================================================

y3_test_pr = results["y3"]["test_pr_auc"]
y6_test_pr = results["y6"]["test_pr_auc"]

high_separability = (
    y3_test_pr >= 0.95
    or
    y6_test_pr >= 0.95
)

if high_separability:

    interpretation = (
        "HIGH_SYNTHETIC_SEPARABILITY. "
        "The synthetic telemetry generation process produces "
        "exceptionally separable future-anomaly targets. "
        "The observed predictive performance must therefore be "
        "interpreted as a controlled synthetic forecasting result, "
        "not as validated real-world cyberattack detection. "
        "The research must next evaluate feature ablation, "
        "telemetry corruption, temporal perturbation, "
        "synthetic-generation sensitivity, and robustness."
    )

else:

    interpretation = (
        "MODERATE_SYNTHETIC_SEPARABILITY. "
        "The bounded classifier diagnostic does not indicate "
        "exceptionally high separability."
    )


checks.append(
    check(
        "metrics_finite",
        all(
            np.isfinite(
                [
                    results[h][metric]
                    for h in results
                    for metric in [
                        "train_pr_auc",
                        "validation_pr_auc",
                        "test_pr_auc",
                        "train_roc_auc",
                        "validation_roc_auc",
                        "test_roc_auc",
                    ]
                ]
            )
        ),
    )
)


# ============================================================================
# FINAL JSON
# ============================================================================

status = (
    "PASS"
    if all(
        c["status"] == "PASS"
        for c in checks
    )
    else
    "FAIL"
)


result = {
    "phase": "4.10B",

    "title": (
        "Synthetic Cyber-Label Separability Audit"
    ),

    "status": status,

    "purpose": (
        "Determine whether unusually strong forecasting "
        "performance is caused by intrinsic separability "
        "in the synthetic cyber telemetry generation process."
    ),

    "dataset": (
        "CRSS 2024 Temporal Cyber-Physical States V2"
    ),

    "feature_contract": {
        "original_feature_count": len(features),
        "removed_constant_features": removed_constants,
        "final_feature_count": len(final_features),
        "final_features": final_features,
        "final_indices": final_indices,
    },

    "summary_representation": {
        "statistics_per_feature": 8,
        "dimension": 472,
        "summary_seconds": float(summary_seconds),
    },

    "classifier": {
        "algorithm": "HistGradientBoostingClassifier",
        "max_training_samples": MAX_TRAIN_SAMPLES,
        "random_seed": RANDOM_SEED,
        "bounded_training": True,
        "full_validation_used": True,
        "full_test_used": True,
    },

    "results": results,

    "univariate_last_step_y6": {
        "features_analyzed": len(feature_separation),
        "effect_size_ge_1": len(effect_ge_1),
        "effect_size_ge_2": len(effect_ge_2),
        "effect_size_ge_3": len(effect_ge_3),
        "top_20": top_20,
    },

    "interpretation": interpretation,

    "methodological_boundary": (
        "The cyber anomaly labels are synthetic research labels. "
        "CRSS supplies physical crash and safety ground truth "
        "but is not a cyberattack dataset. Results must not be "
        "presented as validated real-world cyberattack detection "
        "performance."
    ),

    "checks": checks,
}


OUT.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 100)
print("PHASE 4.10B COMPLETE")
print("=" * 100)

print()
print(
    f"Y3 Test PR-AUC : "
    f"{y3_test_pr:.6f}"
)

print(
    f"Y6 Test PR-AUC : "
    f"{y6_test_pr:.6f}"
)

print()
print(
    "Effect size >= 1.0:",
    len(effect_ge_1),
)

print(
    "Effect size >= 2.0:",
    len(effect_ge_2),
)

print(
    "Effect size >= 3.0:",
    len(effect_ge_3),
)

print()
print("Interpretation:")
print(interpretation)

print()
print("=" * 100)

print(
    f"STATUS : {status}"
)

print(
    f"RESULT : {OUT}"
)

print(
    f"CHECKS : "
    f"{sum(c['status'] == 'PASS' for c in checks)} / "
    f"{len(checks)}"
)

print("=" * 100)


if status != "PASS":

    raise SystemExit(
        "PHASE 4.10B SYNTHETIC LABEL SEPARABILITY AUDIT FAILED"
    )
