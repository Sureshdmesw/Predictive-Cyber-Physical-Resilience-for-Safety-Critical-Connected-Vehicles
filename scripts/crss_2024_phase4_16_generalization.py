from pathlib import Path
import json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib

ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

SEQ = ROOT / "data/processed/modeling/sequences/v2_forecasting"
OUT = ROOT / "experiments/modeling/v2/generalization/phase4_16"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

X_train_path = SEQ / "X_train_v2_forecasting.npy"
X_test_path = SEQ / "X_test_v2_forecasting.npy"

y6_train_path = SEQ / "y_6step_train_v2_forecasting.npy"
y6_test_path = SEQ / "y_6step_test_v2_forecasting.npy"

y3_test_path = SEQ / "y_3step_test_v2_forecasting.npy"

scenario_train_path = SEQ / "scenario_ids_train_v2_forecasting.npy"
scenario_test_path = SEQ / "scenario_ids_test_v2_forecasting.npy"

origin_test_path = SEQ / "prediction_origins_test_v2_forecasting.npy"

baseline_json = (
    ROOT
    / "experiments/modeling/v2/baseline/"
      "crss_2024_v2_strong_baseline_results.json"
)

# ------------------------------------------------------------
# Locate HGB model artifacts
# ------------------------------------------------------------

candidate_models = []

for root in [
    ROOT / "models/v2/baseline",
    ROOT / "models/v2",
]:
    if root.exists():
        candidate_models.extend(
            list(root.rglob("*.joblib"))
            + list(root.rglob("*.pkl"))
        )

print("=" * 72)
print("PHASE 4.16 — SCENARIO GENERALIZATION / SHIFT EVALUATION")
print("=" * 72)

print("\n=== MODEL ARTIFACTS ===")

for p in candidate_models:
    print(p)

if not candidate_models:
    raise FileNotFoundError(
        "No .joblib or .pkl HGB model artifact was found."
    )

# Prefer files whose names suggest Y6.
y6_candidates = [
    p for p in candidate_models
    if "y6" in p.name.lower()
    or "6step" in p.name.lower()
]

if y6_candidates:
    model_path = y6_candidates[0]
else:
    model_path = candidate_models[0]

print(f"\nSelected model: {model_path}")

model = joblib.load(model_path)

# ------------------------------------------------------------
# Load arrays
# ------------------------------------------------------------

X_train = np.load(X_train_path, mmap_mode="r")
X_test = np.load(X_test_path, mmap_mode="r")

y6_train = np.load(y6_train_path, mmap_mode="r")
y6_test = np.load(y6_test_path, mmap_mode="r")
y3_test = np.load(y3_test_path, mmap_mode="r")

scenario_train = np.load(
    scenario_train_path,
    mmap_mode="r"
)

scenario_test = np.load(
    scenario_test_path,
    mmap_mode="r"
)

origins_test = np.load(
    origin_test_path,
    mmap_mode="r"
)

print("\n=== SHAPES ===")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y6_train:", y6_train.shape)
print("y6_test :", y6_test.shape)
print("scenario_train:", scenario_train.shape)
print("scenario_test :", scenario_test.shape)

# ------------------------------------------------------------
# Build the same 472-dimensional HGB representation:
# 8 statistics × 59 features
#
# mean/std/min/max/last/delta plus two additional temporal
# summary statistics.
# ------------------------------------------------------------

FINAL_FEATURE_INDICES = [
    i for i in range(63)
    if i not in {30, 35, 39, 62}
]

if len(FINAL_FEATURE_INDICES) != 59:
    raise RuntimeError(
        f"Expected 59 final features, got {len(FINAL_FEATURE_INDICES)}"
    )


def temporal_summary(X):
    """
    EXACT Phase 4.9 representation.

    First applies the exact Phase 4.7 treatment:
    63 original features -> 59 final features.

    Then:
    59 features × 8 temporal statistics = 472 dimensions.

    59 treated features × 8 statistics = 472 dimensions.

    Statistics, in the exact Phase 4.9 order:
      1. last
      2. mean
      3. std
      4. minimum
      5. maximum
      6. last_minus_first
      7. max_absolute_step_change
      8. last_step_change
    """
    X = np.asarray(X, dtype=np.float32)

    # Apply the exact Phase 4.7 feature treatment used by Phase 4.9.
    # Removed constant features:
    #   30 gateway_present
    #   35 physical_state_completeness
    #   39 roadway_complexity_index
    #   62 vehicle_state_observation_count
    if X.shape[-1] == 63:
        X = X[:, :, FINAL_FEATURE_INDICES]
    elif X.shape[-1] != 59:
        raise ValueError(
            f"Expected 63 original or 59 treated features, got {X.shape[-1]}"
        )

    if X.shape[-1] != 59:
        raise RuntimeError(
            f"Feature treatment failed: expected 59, got {X.shape[-1]}"
        )

    last = X[:, -1, :]
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    minimum = np.min(X, axis=1)
    maximum = np.max(X, axis=1)
    last_minus_first = X[:, -1, :] - X[:, 0, :]

    step_change = np.diff(X, axis=1)

    max_absolute_step_change = np.max(
        np.abs(step_change),
        axis=1,
    )

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
    ).astype(np.float32)

print("\nBuilding test representation...")

X_test_repr = temporal_summary(X_test)

if X_test_repr.shape[1] != 472:
    raise RuntimeError(
        f"Phase 4.9 compatibility failure: expected 472 representation "
        f"features, got {X_test_repr.shape[1]}"
    )

print("X_test_repr:", X_test_repr.shape)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

if hasattr(model, "predict_proba"):
    test_prob = model.predict_proba(X_test_repr)[:, 1]
else:
    test_prob = model.predict(X_test_repr)

test_prob = np.asarray(test_prob, dtype=np.float64)

# ------------------------------------------------------------
# Overall metrics
# ------------------------------------------------------------

overall = {
    "samples": int(len(y6_test)),
    "positive": int(np.sum(y6_test)),
    "positive_rate": float(np.mean(y6_test)),
    "y6_pr_auc": float(
        average_precision_score(y6_test, test_prob)
    ),
    "y6_roc_auc": float(
        roc_auc_score(y6_test, test_prob)
    ),
}

# ------------------------------------------------------------
# Scenario-level evaluation
# ------------------------------------------------------------

scenario_metrics = []

unique_scenarios = np.unique(scenario_test)

for scenario_id in unique_scenarios:

    mask = scenario_test == scenario_id

    y = np.asarray(y6_test[mask])
    p = test_prob[mask]

    # Each scenario has exactly 7 origins.
    if len(np.unique(origins_test[mask])) != 7:
        continue

    # AP requires at least one positive or one negative.
    if len(np.unique(y)) < 2:
        continue

    scenario_metrics.append(
        {
            "scenario_id": str(scenario_id),
            "samples": int(mask.sum()),
            "positive": int(y.sum()),
            "pr_auc": float(
                average_precision_score(y, p)
            ),
        }
    )

scenario_pr = np.array(
    [x["pr_auc"] for x in scenario_metrics],
    dtype=np.float64,
)

scenario_summary = {
    "eligible_scenarios": int(len(scenario_metrics)),
    "total_test_scenarios": int(len(unique_scenarios)),
    "mean_pr_auc": float(np.mean(scenario_pr))
        if len(scenario_pr) else None,
    "median_pr_auc": float(np.median(scenario_pr))
        if len(scenario_pr) else None,
    "p10_pr_auc": float(np.percentile(scenario_pr, 10))
        if len(scenario_pr) else None,
    "p25_pr_auc": float(np.percentile(scenario_pr, 25))
        if len(scenario_pr) else None,
    "p75_pr_auc": float(np.percentile(scenario_pr, 75))
        if len(scenario_pr) else None,
    "p90_pr_auc": float(np.percentile(scenario_pr, 90))
        if len(scenario_pr) else None,
}

# ------------------------------------------------------------
# Temporal-origin analysis
# ------------------------------------------------------------

origin_metrics = {}

for origin in sorted(np.unique(origins_test)):

    mask = origins_test == origin

    y6 = np.asarray(y6_test[mask])
    p = test_prob[mask]

    y3 = np.asarray(y3_test[mask])

    entry = {
        "samples": int(mask.sum()),
        "y6_positive": int(y6.sum()),
        "y6_rate": float(np.mean(y6)),
        "y3_positive": int(y3.sum()),
        "y3_rate": float(np.mean(y3)),
    }

    if len(np.unique(y6)) >= 2:
        entry["y6_pr_auc"] = float(
            average_precision_score(y6, p)
        )
        entry["y6_roc_auc"] = float(
            roc_auc_score(y6, p)
        )
    else:
        entry["y6_pr_auc"] = None
        entry["y6_roc_auc"] = None

    origin_metrics[str(int(origin))] = entry

# ------------------------------------------------------------
# Test-set cyber-shift proxies
#
# Use model-independent feature groups available in the
# 59-feature tensor. We evaluate high vs low regimes using
# train-derived quartile thresholds.
# ------------------------------------------------------------

metadata_path = SEQ / "crss_2024_forecasting_tensor_metadata.json"
metadata = json.loads(
    metadata_path.read_text(encoding="utf-8")
)

features = metadata["features"]

feature_index = {
    name: i for i, name in enumerate(features)
}

shift_features = {
    "cyber_instability": "cyber_instability_index",
    "packet_loss": "packet_loss_rate",
    "latency_jitter": "latency_jitter_ms",
    "message_integrity": "message_integrity_failure_rate",
    "sensor_disagreement": "sensor_disagreement_rate",
    "speed_deviation_abs": "speed_limit_deviation_abs",
}

X_train_last = np.asarray(
    X_train[:, -1, :],
    dtype=np.float32
)

X_test_last = np.asarray(
    X_test[:, -1, :],
    dtype=np.float32
)

shift_results = {}

for group_name, feature_name in shift_features.items():

    idx = feature_index[feature_name]

    train_values = X_train_last[:, idx]
    test_values = X_test_last[:, idx]

    q25, q75 = np.percentile(
        train_values,
        [25, 75]
    )

    low_mask = test_values <= q25
    high_mask = test_values >= q75

    entry = {
        "feature": feature_name,
        "train_q25": float(q25),
        "train_q75": float(q75),
        "low_samples": int(low_mask.sum()),
        "high_samples": int(high_mask.sum()),
    }

    for regime_name, mask in [
        ("low", low_mask),
        ("high", high_mask),
    ]:

        y = np.asarray(y6_test[mask])
        p = test_prob[mask]

        entry[f"{regime_name}_positive_rate"] = (
            float(np.mean(y))
            if len(y) else None
        )

        if len(y) and len(np.unique(y)) >= 2:
            entry[f"{regime_name}_pr_auc"] = float(
                average_precision_score(y, p)
            )
            entry[f"{regime_name}_roc_auc"] = float(
                roc_auc_score(y, p)
            )
        else:
            entry[f"{regime_name}_pr_auc"] = None
            entry[f"{regime_name}_roc_auc"] = None

    shift_results[group_name] = entry

# ------------------------------------------------------------
# Generalization stress summary
# ------------------------------------------------------------

origin_pr_values = [
    x["y6_pr_auc"]
    for x in origin_metrics.values()
    if x["y6_pr_auc"] is not None
]

high_low_deltas = {}

for name, result in shift_results.items():

    low = result.get("low_pr_auc")
    high = result.get("high_pr_auc")

    if low is not None and high is not None:
        high_low_deltas[name] = float(high - low)

summary = {
    "overall_y6_pr_auc": overall["y6_pr_auc"],
    "scenario_mean_pr_auc": scenario_summary["mean_pr_auc"],
    "scenario_median_pr_auc": scenario_summary["median_pr_auc"],
    "scenario_p10_pr_auc": scenario_summary["p10_pr_auc"],
    "origin_mean_pr_auc": (
        float(np.mean(origin_pr_values))
        if origin_pr_values else None
    ),
    "origin_min_pr_auc": (
        float(np.min(origin_pr_values))
        if origin_pr_values else None
    ),
    "origin_max_pr_auc": (
        float(np.max(origin_pr_values))
        if origin_pr_values else None
    ),
    "shift_high_minus_low_pr_auc": high_low_deltas,
}

# ------------------------------------------------------------
# Gates
# ------------------------------------------------------------

gates = {
    "test_samples_positive":
        len(y6_test) > 0 and int(np.sum(y6_test)) > 0,

    "test_probabilities_finite":
        bool(np.isfinite(test_prob).all()),

    "scenario_count_expected":
        len(unique_scenarios) == 13693,

    "seven_origins_per_scenario":
        all(
            len(np.unique(origins_test[
                scenario_test == sid
            ])) == 7
            for sid in unique_scenarios
        ),

    "overall_pr_auc_valid":
        np.isfinite(overall["y6_pr_auc"]),

    "scenario_metrics_available":
        len(scenario_metrics) > 0,

    "origin_metrics_available":
        len(origin_metrics) == 7,
}

status = "PASS" if all(gates.values()) else "HOLD"

# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

report = {
    "phase": "4.16",
    "title":
        "Scenario-Level Generalization and Distribution-Shift Evaluation",
    "status": status,
    "model_path": str(model_path),
    "overall": overall,
    "scenario_summary": scenario_summary,
    "scenario_metrics": scenario_metrics,
    "origin_metrics": origin_metrics,
    "shift_results": shift_results,
    "summary": summary,
    "gates": gates,
    "methodological_notes": [
        "Evaluation uses the existing HGB baseline; no model retraining.",
        "Scenario IDs are the partition identity.",
        "Each scenario contributes exactly seven prediction origins.",
        "prediction_origins 11-17 are local temporal positions.",
        "Shift analysis uses train-derived quartile thresholds.",
        "Cyber telemetry and cyber anomaly labels remain synthetic research signals.",
        "Results should not be interpreted as real-world cyberattack detection performance."
    ],
}

out_path = (
    OUT
    / "crss_2024_v2_phase4_16_generalization_results.json"
)

out_path.write_text(
    json.dumps(
    report,
    indent=2,
    default=lambda o: (
        bool(o) if type(o).__module__ == "numpy" and type(o).__name__ == "bool_"
        else int(o) if type(o).__module__ == "numpy" and type(o).__name__.startswith("int")
        else float(o) if type(o).__module__ == "numpy" and type(o).__name__.startswith("float")
        else o.tolist() if hasattr(o, "tolist") else str(o)
    )
),
    encoding="utf-8"
)

# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------

print("\n=== OVERALL Y6 ===")
print(f"Samples:       {overall['samples']:,}")
print(f"Positive:      {overall['positive']:,}")
print(f"Positive rate: {overall['positive_rate']:.6%}")
print(f"PR-AUC:        {overall['y6_pr_auc']:.6f}")
print(f"ROC-AUC:       {overall['y6_roc_auc']:.6f}")

print("\n=== SCENARIO-LEVEL Y6 ===")
for k, v in scenario_summary.items():
    print(f"{k}: {v}")

print("\n=== ORIGIN-LEVEL Y6 ===")

for origin, result in origin_metrics.items():
    print(
        f"origin {origin}: "
        f"samples={result['samples']:,} "
        f"positive={result['y6_positive']:,} "
        f"rate={result['y6_rate']:.6%} "
        f"PR-AUC={result['y6_pr_auc']}"
    )

print("\n=== DISTRIBUTION-SHIFT REGIMES ===")

for name, result in shift_results.items():

    print(
        f"{name:22s} "
        f"feature={result['feature']:38s} "
        f"low_PR={result['low_pr_auc']} "
        f"high_PR={result['high_pr_auc']} "
        f"delta={high_low_deltas.get(name)}"
    )

print("\n=== GATES ===")

for name, value in gates.items():
    print(
        f"{'PASS' if value else 'FAIL'}  {name}"
    )

print(f"\nSTATUS: {status}")
print(f"REPORT: {out_path}")

print("=" * 72)
