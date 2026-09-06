
"""
PHASE 4.17 — ERROR ANALYSIS / FAILURE CHARACTERIZATION

Purpose:
    Characterize where and why the validated HGB Y6 predictor
    loses performance under scenario/temporal distribution shift.

No model retraining is performed.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SEQ = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
GENERALIZATION = (
    ROOT / "experiments" / "modeling" / "v2" / "generalization"
    / "phase4_16"
    / "crss_2024_v2_phase4_16_generalization_results.json"
)

OUTDIR = (
    ROOT / "experiments" / "modeling" / "v2"
    / "error_analysis"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("PHASE 4.17 — ERROR ANALYSIS / FAILURE CHARACTERIZATION")
print("=" * 72)

report = {
    "phase": "4.17",
    "title": "Error Analysis / Failure Characterization",
    "model": "Phase 4.9 HGB Y6",
    "retraining": False,
    "status": "STARTED",
}

if not GENERALIZATION.exists():
    raise FileNotFoundError(
        f"Missing Phase 4.16 report: {GENERALIZATION}"
    )

with GENERALIZATION.open("r", encoding="utf-8") as f:
    phase416 = json.load(f)

report["phase4_16_reference"] = phase416

# ------------------------------------------------------------------
# Load test artifacts
# ------------------------------------------------------------------

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

# Scenario IDs are opaque identifiers and must remain strings.
if scenario_test.dtype.kind not in {"U", "S", "O"}:
    raise RuntimeError(
        f"Scenario IDs must be string/object identifiers, got dtype={scenario_test.dtype}"
    )


origin_test = np.load(
    SEQ / "prediction_origins_test_v2_forecasting.npy",
    mmap_mode="r"
)

print("\n=== DATA ===")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("scenario_test:", scenario_test.shape)
print("origin_test:", origin_test.shape)

if len(y_test) != len(scenario_test):
    raise RuntimeError("Test/scenario length mismatch.")

if len(y_test) != len(origin_test):
    raise RuntimeError("Test/origin length mismatch.")

# ------------------------------------------------------------------
# Rebuild exact Phase 4.9-compatible representation
# ------------------------------------------------------------------

FINAL_INDICES = [
    i for i in range(63)
    if i not in {30, 35, 39, 62}
]

def temporal_summary(X):
    X = np.asarray(X, dtype=np.float32)
    X = X[:, :, FINAL_INDICES]

    last = X[:, -1, :]
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    minimum = np.min(X, axis=1)
    maximum = np.max(X, axis=1)

    last_minus_first = X[:, -1, :] - X[:, 0, :]

    step_change = np.diff(X, axis=1)

    max_absolute_step_change = np.max(
        np.abs(step_change),
        axis=1
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
    )

X_repr = temporal_summary(X_test)

print("\nRepresentation:", X_repr.shape)

if X_repr.shape[1] != 472:
    raise RuntimeError(
        f"Expected 472 dimensions, got {X_repr.shape[1]}"
    )

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------

import joblib

MODEL = (
    ROOT / "models" / "v2" / "baseline"
    / "crss_2024_v2_hgb_y6.joblib"
)

model = joblib.load(MODEL)

prob = model.predict_proba(X_repr)[:, 1]

print("Predictions:", prob.shape)

if not np.all(np.isfinite(prob)):
    raise RuntimeError("Non-finite predictions detected.")

# ------------------------------------------------------------------
# Basic error characterization
# ------------------------------------------------------------------

pred = (prob >= 0.5).astype(np.int8)

tp = int(np.sum((pred == 1) & (y_test == 1)))
tn = int(np.sum((pred == 0) & (y_test == 0)))
fp = int(np.sum((pred == 1) & (y_test == 0)))
fn = int(np.sum((pred == 0) & (y_test == 1)))

report["confusion_matrix_threshold_0_5"] = {
    "tp": tp,
    "tn": tn,
    "fp": fp,
    "fn": fn,
}

report["prediction_summary"] = {
    "min": float(np.min(prob)),
    "max": float(np.max(prob)),
    "mean": float(np.mean(prob)),
    "median": float(np.median(prob)),
}

# ------------------------------------------------------------------
# Origin analysis
# ------------------------------------------------------------------

origin_rows = []

for origin in sorted(np.unique(origin_test)):
    mask = origin_test == origin

    y = y_test[mask]
    p = prob[mask]

    row = {
        "origin": int(origin),
        "samples": int(mask.sum()),
        "positives": int(y.sum()),
        "positive_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "median_probability": float(np.median(p)),
        "mean_positive_probability": (
            float(np.mean(p[y == 1]))
            if np.any(y == 1) else None
        ),
        "mean_negative_probability": (
            float(np.mean(p[y == 0]))
            if np.any(y == 0) else None
        ),
    }

    origin_rows.append(row)

report["origin_error_analysis"] = origin_rows

# ------------------------------------------------------------------
# Scenario analysis
# ------------------------------------------------------------------

scenario_rows = []

for scenario in np.unique(scenario_test):
    mask = scenario_test == scenario

    y = y_test[mask]
    p = prob[mask]

    if len(y) == 0:
        continue

    scenario_rows.append({
        "scenario": str(scenario),
        "samples": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "max_probability": float(np.max(p)),
        "mean_positive_probability": (
            float(np.mean(p[y == 1]))
            if np.any(y == 1) else None
        ),
        "mean_negative_probability": (
            float(np.mean(p[y == 0]))
            if np.any(y == 0) else None
        ),
    })

scenario_df = pd.DataFrame(scenario_rows)

if not scenario_df.empty:
    report["scenario_error_analysis"] = {
        "count": int(len(scenario_df)),
        "hardest_mean_probability": (
            scenario_df
            .sort_values("mean_probability")
            .head(10)
            .to_dict(orient="records")
        ),
        "highest_probability": (
            scenario_df
            .sort_values("max_probability", ascending=False)
            .head(10)
            .to_dict(orient="records")
        ),
    }

# ------------------------------------------------------------------
# Probability separation
# ------------------------------------------------------------------

positive_prob = prob[y_test == 1]
negative_prob = prob[y_test == 0]

report["class_probability_separation"] = {
    "positive_count": int(len(positive_prob)),
    "negative_count": int(len(negative_prob)),
    "positive_mean": float(np.mean(positive_prob)),
    "negative_mean": float(np.mean(negative_prob)),
    "positive_median": float(np.median(positive_prob)),
    "negative_median": float(np.median(negative_prob)),
}

# ------------------------------------------------------------------
# Save tabular origin/scenario outputs
# ------------------------------------------------------------------

pd.DataFrame(origin_rows).to_csv(
    OUTDIR / "origin_error_analysis.csv",
    index=False
)

scenario_df.to_csv(
    OUTDIR / "scenario_error_analysis.csv",
    index=False
)

report["status"] = "PASS"

out = OUTDIR / "crss_2024_v2_phase4_17_error_analysis.json"

with out.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n=== GATES ===")
print("PASS  exact_472_dimension_representation")
print("PASS  finite_probabilities")
print("PASS  origin_analysis")
print("PASS  scenario_analysis")
print("PASS  error_characterization")
print("PASS  report_written")

print("\nSTATUS: PASS")
print("REPORT:", out)

