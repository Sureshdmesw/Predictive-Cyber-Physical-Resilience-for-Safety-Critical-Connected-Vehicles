from pathlib import Path
import json
import numpy as np
import joblib
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]

SEQ = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
MODEL = ROOT / "models" / "v2" / "baseline" / "crss_2024_v2_hgb_y6.joblib"

OUT_DIR = ROOT / "experiments" / "modeling" / "v2" / "calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = OUT_DIR / "crss_2024_v2_phase4_18_calibration_thresholds.json"

print("=" * 72)
print("PHASE 4.18 — CALIBRATION + OPERATIONAL THRESHOLDS")
print("=" * 72)

# ---------------------------------------------------------------
# Load validation/test data
# ---------------------------------------------------------------

X_val = np.load(
    SEQ / "X_validation_v2_forecasting.npy",
    mmap_mode="r"
)

y_val = np.load(
    SEQ / "y_6step_validation_v2_forecasting.npy",
    mmap_mode="r"
)

X_test = np.load(
    SEQ / "X_test_v2_forecasting.npy",
    mmap_mode="r"
)

y_test = np.load(
    SEQ / "y_6step_test_v2_forecasting.npy",
    mmap_mode="r"
)

print("\n=== DATA ===")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ---------------------------------------------------------------
# Exact Phase 4.9 representation
# ---------------------------------------------------------------

FINAL_INDICES = [
    i for i in range(63)
    if i not in {30, 35, 39, 62}
]

def temporal_summary(X):
    X = X[:, :, FINAL_INDICES]

    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    minimum = X.min(axis=1)
    maximum = X.max(axis=1)
    last_minus_first = X[:, -1, :] - X[:, 0, :]
    max_absolute_step_change = np.abs(np.diff(X, axis=1)).max(axis=1)
    last_step_change = X[:, -1, :] - X[:, -2, :]

    return np.concatenate(
        [
            last,
            mean,
            std,
            minimum,
            maximum,
            last_minus_first,
            max_absolute_step_change,
            last_step_change,
        ],
        axis=1,
    )

print("\n=== REPRESENTATION ===")

X_val_repr = temporal_summary(X_val)
X_test_repr = temporal_summary(X_test)

print("Validation representation:", X_val_repr.shape)
print("Test representation:", X_test_repr.shape)

if X_val_repr.shape[1] != 472:
    raise RuntimeError(
        f"Expected validation representation of 472 dimensions, "
        f"got {X_val_repr.shape[1]}"
    )

if X_test_repr.shape[1] != 472:
    raise RuntimeError(
        f"Expected test representation of 472 dimensions, "
        f"got {X_test_repr.shape[1]}"
    )

# ---------------------------------------------------------------
# Load persisted HGB Y6 model
# ---------------------------------------------------------------

print("\n=== MODEL ===")

artifact = joblib.load(MODEL)

if hasattr(artifact, "predict_proba"):
    model = artifact
else:
    model = artifact["model"]

print("Model loaded:", MODEL)

p_val = model.predict_proba(X_val_repr)[:, 1]
p_test = model.predict_proba(X_test_repr)[:, 1]

if not np.all(np.isfinite(p_val)):
    raise RuntimeError("Validation probabilities contain non-finite values.")

if not np.all(np.isfinite(p_test)):
    raise RuntimeError("Test probabilities contain non-finite values.")

# ---------------------------------------------------------------
# Baseline validation/test metrics
# ---------------------------------------------------------------

print("\n=== BASELINE METRICS ===")

val_pr = average_precision_score(y_val, p_val)
test_pr = average_precision_score(y_test, p_test)

val_roc = roc_auc_score(y_val, p_val)
test_roc = roc_auc_score(y_test, p_test)

val_brier = brier_score_loss(y_val, p_val)
test_brier = brier_score_loss(y_test, p_test)

print(f"Validation PR-AUC : {val_pr:.6f}")
print(f"Test PR-AUC       : {test_pr:.6f}")
print(f"Validation ROC-AUC: {val_roc:.6f}")
print(f"Test ROC-AUC      : {test_roc:.6f}")
print(f"Validation Brier  : {val_brier:.6f}")
print(f"Test Brier        : {test_brier:.6f}")

# ---------------------------------------------------------------
# Calibration
#
# IMPORTANT:
# Fit calibration ONLY on validation predictions.
# Test remains untouched for final evaluation.
# ---------------------------------------------------------------

print("\n=== ISOTONIC CALIBRATION ===")

iso = IsotonicRegression(
    y_min=0.0,
    y_max=1.0,
    out_of_bounds="clip"
)

iso.fit(p_val, y_val)

p_val_cal = iso.predict(p_val)
p_test_cal = iso.predict(p_test)

cal_val_brier = brier_score_loss(y_val, p_val_cal)
cal_test_brier = brier_score_loss(y_test, p_test_cal)

print(f"Calibrated validation Brier: {cal_val_brier:.6f}")
print(f"Calibrated test Brier      : {cal_test_brier:.6f}")

# ---------------------------------------------------------------
# Threshold analysis
#
# Thresholds are selected using validation data only.
# Test metrics are reported afterward.
# ---------------------------------------------------------------

print("\n=== THRESHOLD ANALYSIS ===")

thresholds = np.array([
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
], dtype=float)

threshold_results = []

for threshold in thresholds:

    pred_val = (p_val_cal >= threshold).astype(np.int8)

    tp = int(np.sum((pred_val == 1) & (y_val == 1)))
    fp = int(np.sum((pred_val == 1) & (y_val == 0)))
    fn = int(np.sum((pred_val == 0) & (y_val == 1)))
    tn = int(np.sum((pred_val == 0) & (y_val == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    pred_test = (p_test_cal >= threshold).astype(np.int8)

    test_tp = int(np.sum((pred_test == 1) & (y_test == 1)))
    test_fp = int(np.sum((pred_test == 1) & (y_test == 0)))
    test_fn = int(np.sum((pred_test == 0) & (y_test == 1)))

    test_precision = (
        test_tp / (test_tp + test_fp)
        if (test_tp + test_fp)
        else 0.0
    )

    test_recall = (
        test_tp / (test_tp + test_fn)
        if (test_tp + test_fn)
        else 0.0
    )

    test_f1 = (
        2 * test_precision * test_recall /
        (test_precision + test_recall)
        if (test_precision + test_recall)
        else 0.0
    )

    threshold_results.append({
        "threshold": float(threshold),

        "validation": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "alert_rate": float(np.mean(pred_val))
        },

        "test": {
            "tp": test_tp,
            "fp": test_fp,
            "fn": test_fn,
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1": float(test_f1),
            "alert_rate": float(np.mean(pred_test))
        }
    })

# ---------------------------------------------------------------
# Operational policy
#
# This is a decision-support policy, NOT direct vehicle actuation.
# ---------------------------------------------------------------

def operational_level(p):

    if p >= 0.80:
        return "CRITICAL"

    if p >= 0.50:
        return "HIGH"

    if p >= 0.20:
        return "MEDIUM"

    if p >= 0.05:
        return "LOW"

    return "NORMAL"

policy_counts = {}

for level in [
    "NORMAL",
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL"
]:
    policy_counts[level] = int(
        np.sum(
            np.array(
                [operational_level(x) for x in p_test_cal],
                dtype=object
            ) == level
        )
    )

# ---------------------------------------------------------------
# Reliability bins
# ---------------------------------------------------------------

print("\n=== CALIBRATION BINS ===")

fraction_pos, mean_pred = calibration_curve(
    y_test,
    p_test_cal,
    n_bins=10,
    strategy="quantile"
)

calibration_bins = []

for mp, fp in zip(mean_pred, fraction_pos):
    calibration_bins.append({
        "mean_predicted_probability": float(mp),
        "observed_positive_fraction": float(fp),
        "absolute_calibration_error": float(abs(mp - fp))
    })

# ---------------------------------------------------------------
# Gates
# ---------------------------------------------------------------

print("\n=== GATES ===")

gates = {
    "validation_samples_positive": bool(len(y_val) > 0 and np.sum(y_val) > 0),
    "test_samples_positive": bool(len(y_test) > 0 and np.sum(y_test) > 0),
    "finite_validation_probabilities": bool(np.all(np.isfinite(p_val))),
    "finite_test_probabilities": bool(np.all(np.isfinite(p_test))),
    "exact_472_validation_representation": bool(X_val_repr.shape[1] == 472),
    "exact_472_test_representation": bool(X_test_repr.shape[1] == 472),
    "calibration_fitted_on_validation_only": True,
    "test_isolation_preserved": True,
    "threshold_analysis_complete": bool(len(threshold_results) == len(thresholds)),
    "operational_policy_defined": True,
}

for name, passed in gates.items():
    print(f"{'PASS' if passed else 'FAIL':5} {name}")

if not all(gates.values()):
    raise RuntimeError("Phase 4.18 gate failure.")

# ---------------------------------------------------------------
# Report
# ---------------------------------------------------------------

report = {
    "phase": "4.18",
    "status": "PASS",

    "task": "Calibration and operational threshold analysis",

    "model": str(MODEL),

    "representation": {
        "original_features": 63,
        "final_features": 59,
        "temporal_summary_dimensions": 472,
        "statistics": [
            "last",
            "mean",
            "std",
            "minimum",
            "maximum",
            "last_minus_first",
            "max_absolute_step_change",
            "last_step_change"
        ]
    },

    "baseline": {
        "validation": {
            "pr_auc": float(val_pr),
            "roc_auc": float(val_roc),
            "brier": float(val_brier)
        },
        "test": {
            "pr_auc": float(test_pr),
            "roc_auc": float(test_roc),
            "brier": float(test_brier)
        }
    },

    "isotonic_calibration": {
        "validation_brier": float(cal_val_brier),
        "test_brier": float(cal_test_brier),
        "calibration_fit_split": "validation_only"
    },

    "thresholds": threshold_results,

    "operational_policy": {
        "NORMAL": "<0.05",
        "LOW": "0.05-<0.20",
        "MEDIUM": "0.20-<0.50",
        "HIGH": "0.50-<0.80",
        "CRITICAL": ">=0.80",
        "test_counts": policy_counts,
        "safety_boundary": "decision support / warning / SOC prioritization only; no direct vehicle actuation"
    },

    "calibration_bins": calibration_bins,

    "gates": gates
}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 72)
print("STATUS: PASS")
print(f"REPORT: {REPORT}")
print("=" * 72)
