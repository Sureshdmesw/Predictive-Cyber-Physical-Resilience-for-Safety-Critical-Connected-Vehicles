from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

SEQ = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
MODEL = ROOT / "models" / "v2" / "baseline" / "crss_2024_v2_hgb_y6.joblib"

OUT_DIR = ROOT / "experiments" / "modeling" / "v2" / "robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = OUT_DIR / "crss_2024_v2_phase4_19_robustness_confirmation.json"

print("=" * 72)
print("PHASE 4.19 — ROBUSTNESS CONFIRMATION")
print("=" * 72)

# ---------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------

X_test = np.load(
    SEQ / "X_test_v2_forecasting.npy",
    mmap_mode="r"
)

y_test = np.load(
    SEQ / "y_6step_test_v2_forecasting.npy",
    mmap_mode="r"
)

scenario_test = np.load(
    SEQ / "scenario_ids_test_v2_forecasting.npy",
    mmap_mode="r"
)

origin_test = np.load(
    SEQ / "prediction_origins_test_v2_forecasting.npy",
    mmap_mode="r"
)

print("\n=== DATA CONTRACT ===")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("scenario_test:", scenario_test.shape)
print("origin_test:", origin_test.shape)

if X_test.shape != (95851, 12, 63):
    raise RuntimeError(f"Unexpected X_test shape: {X_test.shape}")

if y_test.shape != (95851,):
    raise RuntimeError(f"Unexpected y_test shape: {y_test.shape}")

if scenario_test.shape != (95851,):
    raise RuntimeError(f"Unexpected scenario shape: {scenario_test.shape}")

if origin_test.shape != (95851,):
    raise RuntimeError(f"Unexpected origin shape: {origin_test.shape}")

if scenario_test.dtype.kind not in {"U", "S", "O"}:
    raise RuntimeError(
        f"Scenario IDs must remain strings, got {scenario_test.dtype}"
    )

if set(np.unique(origin_test).tolist()) != set(range(11, 18)):
    raise RuntimeError("Origins must be exactly 11..17.")

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

X_repr = temporal_summary(X_test)

print("\nRepresentation:", X_repr.shape)

if X_repr.shape != (95851, 472):
    raise RuntimeError(
        f"Expected (95851,472), got {X_repr.shape}"
    )

# ---------------------------------------------------------------
# Load model
# ---------------------------------------------------------------

artifact = joblib.load(MODEL)

if hasattr(artifact, "predict_proba"):
    model = artifact
else:
    model = artifact["model"]

p_base = model.predict_proba(X_repr)[:, 1]

if not np.all(np.isfinite(p_base)):
    raise RuntimeError("Base probabilities contain non-finite values.")

# ---------------------------------------------------------------
# Controlled corruption experiments
#
# IMPORTANT:
# These are robustness tests only.
# The trained model is NOT modified or retrained.
# ---------------------------------------------------------------

rng = np.random.default_rng(20260906)

corruption_levels = [0.05, 0.10, 0.20]

results = []

def evaluate(name, X):
    representation = temporal_summary(X)
    probability = model.predict_proba(representation)[:, 1]

    return {
        "condition": name,
        "pr_auc": float(average_precision_score(y_test, probability)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "mean_probability": float(np.mean(probability)),
        "positive_prediction_rate_050": float(np.mean(probability >= 0.50)),
    }

# Baseline
print("\n=== BASELINE ===")

baseline = {
    "condition": "clean",
    "pr_auc": float(average_precision_score(y_test, p_base)),
    "roc_auc": float(roc_auc_score(y_test, p_base)),
    "mean_probability": float(np.mean(p_base)),
    "positive_prediction_rate_050": float(np.mean(p_base >= 0.50)),
}

results.append(baseline)

print(
    f"clean PR-AUC={baseline['pr_auc']:.6f} "
    f"ROC-AUC={baseline['roc_auc']:.6f}"
)

# ---------------------------------------------------------------
# Feature noise corruption
# ---------------------------------------------------------------

for level in corruption_levels:

    print(f"\n=== FEATURE NOISE {level:.0%} ===")

    # Copy only test data.
    X_corrupt = np.array(X_test, dtype=np.float32, copy=True)

    # Robust multiplicative perturbation.
    noise = rng.normal(
        loc=0.0,
        scale=level,
        size=X_corrupt.shape
    ).astype(np.float32)

    X_corrupt *= (1.0 + noise)

    result = evaluate(
        f"multiplicative_noise_{level:.0%}",
        X_corrupt
    )

    results.append(result)

    print(
        f"PR-AUC={result['pr_auc']:.6f} "
        f"ROC-AUC={result['roc_auc']:.6f}"
    )

# ---------------------------------------------------------------
# Temporal masking
# ---------------------------------------------------------------

for steps in [1, 2, 3]:

    print(f"\n=== TEMPORAL MASK: LAST {steps} STEP(S) ===")

    X_masked = np.array(X_test, dtype=np.float32, copy=True)

    # Replace recent observations with the preceding observation.
    for k in range(1, steps + 1):
        X_masked[:, -k, :] = X_masked[:, -steps - 1, :]

    result = evaluate(
        f"temporal_mask_last_{steps}_steps",
        X_masked
    )

    results.append(result)

    print(
        f"PR-AUC={result['pr_auc']:.6f} "
        f"ROC-AUC={result['roc_auc']:.6f}"
    )

# ---------------------------------------------------------------
# Feature-group perturbation
#
# These indices correspond to the 63-feature original tensor.
# We deliberately perturb broad contiguous groups rather than
# claiming causal importance.
# ---------------------------------------------------------------

groups = {
    "early_features": list(range(0, 15)),
    "middle_features": list(range(15, 35)),
    "late_features": list(range(35, 63)),
}

for group_name, indices in groups.items():

    print(f"\n=== GROUP PERTURBATION: {group_name} ===")

    X_group = np.array(X_test, dtype=np.float32, copy=True)

    group_indices = [
        i for i in indices
        if i not in {30, 35, 39, 62}
    ]

    X_group[:, :, group_indices] *= 0.90

    result = evaluate(
        f"group_scale_{group_name}",
        X_group
    )

    results.append(result)

    print(
        f"PR-AUC={result['pr_auc']:.6f} "
        f"ROC-AUC={result['roc_auc']:.6f}"
    )

# ---------------------------------------------------------------
# Degradation analysis
# ---------------------------------------------------------------

for result in results:
    result["pr_auc_delta_vs_clean"] = (
        result["pr_auc"] - baseline["pr_auc"]
    )

    result["roc_auc_delta_vs_clean"] = (
        result["roc_auc"] - baseline["roc_auc"]
    )

# ---------------------------------------------------------------
# Origin robustness
# ---------------------------------------------------------------

origin_results = {}

for origin in sorted(np.unique(origin_test)):

    mask = origin_test == origin

    y = y_test[mask]
    p = p_base[mask]

    if len(np.unique(y)) < 2:
        continue

    origin_results[str(int(origin))] = {
        "samples": int(np.sum(mask)),
        "positives": int(np.sum(y)),
        "positive_rate": float(np.mean(y)),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
    }

# ---------------------------------------------------------------
# Gates
# ---------------------------------------------------------------

print("\n=== GATES ===")

gates = {
    "data_shapes_valid": True,
    "scenario_identifiers_valid": True,
    "origins_valid": True,
    "exact_472_representation": True,
    "baseline_probabilities_finite": bool(np.all(np.isfinite(p_base))),
    "baseline_metrics_valid": bool(
        np.isfinite(baseline["pr_auc"]) and
        np.isfinite(baseline["roc_auc"])
    ),
    "corruption_tests_complete": bool(
        len(results) == 1 + 3 + 3 + 3
    ),
    "origin_analysis_complete": bool(len(origin_results) > 0),
    "model_unchanged": True,
    "no_retraining": True,
}

for name, passed in gates.items():
    print(f"{'PASS' if passed else 'FAIL':5} {name}")

if not all(gates.values()):
    raise RuntimeError("Phase 4.19 gate failure.")

# ---------------------------------------------------------------
# Report
# ---------------------------------------------------------------

report = {
    "phase": "4.19",
    "status": "PASS",
    "task": "Robustness confirmation",

    "model": str(MODEL),

    "evaluation_boundary": {
        "test_only": True,
        "model_retrained": False,
        "model_modified": False,
        "purpose": "stress-test existing baseline under controlled input perturbations"
    },

    "baseline": baseline,

    "robustness_results": results,

    "origin_results": origin_results,

    "interpretation": {
        "note": "Performance degradation under corruption is characterized, not treated as causal evidence.",
        "synthetic_data_warning": "Cyber telemetry and anomaly labels are synthetic research constructs.",
        "safety_boundary": "Model outputs are decision-support signals only and must not directly actuate safety-critical vehicle controls."
    },

    "gates": gates
}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 72)
print("STATUS: PASS")
print(f"REPORT: {REPORT}")
print("=" * 72)
