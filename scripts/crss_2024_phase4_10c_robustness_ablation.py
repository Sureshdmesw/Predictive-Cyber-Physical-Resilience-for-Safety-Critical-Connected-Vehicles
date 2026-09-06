from pathlib import Path
import json
import time
import gc
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

SEQ_DIR = ROOT / "data/processed/modeling/sequences/v2_forecasting"
SPEC = ROOT / "experiments/modeling/v2/crss_2024_v2_feature_treatment_spec.json"
OUT_DIR = ROOT / "experiments/modeling/v2/advanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "crss_2024_v2_robustness_ablation_results.json"

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 80)
print("PHASE 4.10C — ROBUSTNESS + FEATURE ABLATION — CORRECTED")
print("=" * 80)

# ---------------------------------------------------------------------
# Load released forecasting tensors
# ---------------------------------------------------------------------
def load(name):
    return np.load(SEQ_DIR / name, mmap_mode="r")

X_train = load("X_train_v2_forecasting.npy")
X_val   = load("X_validation_v2_forecasting.npy")
X_test  = load("X_test_v2_forecasting.npy")

y3_train = load("y_3step_train_v2_forecasting.npy")
y3_val   = load("y_3step_validation_v2_forecasting.npy")
y3_test  = load("y_3step_test_v2_forecasting.npy")

y6_train = load("y_6step_train_v2_forecasting.npy")
y6_val   = load("y_6step_validation_v2_forecasting.npy")
y6_test  = load("y_6step_test_v2_forecasting.npy")

spec = json.loads(SPEC.read_text(encoding="utf-8"))
features = spec["final_features"]

if len(features) != 59:
    raise RuntimeError(f"Expected 59 final features, found {len(features)}")

print(f"Train:      {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}")

print(f"Y3 train+: {int(y3_train.sum()):,}")
print(f"Y3 val+:   {int(y3_val.sum()):,}")
print(f"Y3 test+:  {int(y3_test.sum()):,}")

print(f"Y6 train+: {int(y6_train.sum()):,}")
print(f"Y6 val+:   {int(y6_val.sum()):,}")
print(f"Y6 test+:  {int(y6_test.sum()):,}")

# ---------------------------------------------------------------------
# Correct released-label sanity check
# ---------------------------------------------------------------------
expected = {
    "y3_train": 3773,
    "y3_val": 849,
    "y3_test": 1018,
    "y6_train": 4178,
    "y6_val": 961,
    "y6_test": 1137,
}

actual = {
    "y3_train": int(y3_train.sum()),
    "y3_val": int(y3_val.sum()),
    "y3_test": int(y3_test.sum()),
    "y6_train": int(y6_train.sum()),
    "y6_val": int(y6_val.sum()),
    "y6_test": int(y6_test.sum()),
}

if actual != expected:
    raise RuntimeError(
        f"Released tensor label counts do not match expected Phase 4.6/4.5 counts.\n"
        f"Expected: {expected}\nActual: {actual}"
    )

print("[PASS] Released forecasting labels verified")

# ---------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------
groups = {
    "cyber_ecu_can": [],
    "sensor_consistency": [],
    "connectivity": [],
    "communication_integrity": [],
    "physical_vehicle": [],
    "environment": [],
    "temporal_derivatives": [],
    "temporal_rolling": [],
    "other": []
}

for i, f in enumerate(features):
    fl = f.lower()

    if any(k in fl for k in [
        "ecu", "can", "uds", "diagnostic", "replay",
        "sequence_counter", "unexpected_source"
    ]):
        groups["cyber_ecu_can"].append(i)

    elif any(k in fl for k in [
        "sensor_disagreement", "sensor_plausibility",
        "redundant_sensor", "state_transition"
    ]):
        groups["sensor_consistency"].append(i)

    elif any(k in fl for k in [
        "connectivity", "packet_loss", "telemetry_gap", "latency"
    ]):
        groups["connectivity"].append(i)

    elif any(k in fl for k in [
        "integrity", "authentication", "spoof", "tamper"
    ]):
        groups["communication_integrity"].append(i)

    elif any(k in fl for k in [
        "speed", "rollover", "fire", "damage", "collision",
        "vehicle_", "trav_", "vtraf", "vprofile",
        "valign", "lan", "acc_type"
    ]):
        groups["physical_vehicle"].append(i)

    elif any(k in fl for k in ["weather", "vision"]):
        groups["environment"].append(i)

    elif "rolling_" in fl:
        groups["temporal_rolling"].append(i)

    elif any(k in fl for k in ["delta", "derivative", "change", "rate"]):
        groups["temporal_derivatives"].append(i)

    else:
        groups["other"].append(i)

# ---------------------------------------------------------------------
# Strong synthetic fingerprints
# ---------------------------------------------------------------------
strongest_names = [
    "unexpected_source_id_rate",
    "redundant_sensor_divergence",
    "latency_jitter_ms",
    "replay_indicator_rate",
    "sequence_counter_anomaly_rate",
    "packet_loss_rate",
    "message_integrity_failure_rate"
]

name_to_idx = {f: i for i, f in enumerate(features)}
strongest = [
    name_to_idx[f] for f in strongest_names if f in name_to_idx
]

# ---------------------------------------------------------------------
# Temporal summary
# 8 statistics × 59 features = 472 dimensions
# Keep float32 throughout to prevent the previous float64 memory blow-up.
# ---------------------------------------------------------------------
def summarize(x, selected=None):
    if selected is not None:
        x = x[:, :, selected]

    last = x[:, -1, :]
    mean = x.mean(axis=1, dtype=np.float32)
    std = x.std(axis=1, dtype=np.float32)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    delta = x[:, -1, :] - x[:, 0, :]
    max_step = np.max(np.abs(np.diff(x, axis=1)), axis=1)
    last_step = x[:, -1, :] - x[:, -2, :]

    return np.concatenate(
        [last, mean, std, mn, mx, delta, max_step, last_step],
        axis=1
    ).astype(np.float32, copy=False)

# ---------------------------------------------------------------------
# Train-only summary normalization
# ---------------------------------------------------------------------
print("\nBuilding train representation...")
t0 = time.time()

S_train = summarize(X_train)
S_val   = summarize(X_val)
S_test  = summarize(X_test)

mu = S_train.mean(axis=0, dtype=np.float32)
sd = S_train.std(axis=0, dtype=np.float32)
sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)

S_train = ((S_train - mu) / sd).astype(np.float32)
S_val   = ((S_val - mu) / sd).astype(np.float32)
S_test  = ((S_test - mu) / sd).astype(np.float32)

print(f"Representations: {S_train.shape}")
print(f"Construction time: {time.time() - t0:.2f}s")

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate(
    train_x,
    val_x,
    test_x,
    train_y,
    val_y,
    test_y,
    label,
    feature_set
):
    pos = int(train_y.sum())
    neg = int(len(train_y) - pos)
    pos_weight = neg / max(pos, 1)

    print(
        f"\n{label} | features={feature_set} | "
        f"train positives={pos:,} | pos_weight={pos_weight:.2f}"
    )

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=SEED
    )

    weights = np.where(
        train_y == 1,
        pos_weight,
        1.0
    ).astype(np.float32)

    clf.fit(train_x, train_y, sample_weight=weights)

    p_train = clf.predict_proba(train_x)[:, 1]
    p_val   = clf.predict_proba(val_x)[:, 1]
    p_test  = clf.predict_proba(test_x)[:, 1]

    result = {
        "label": label,
        "feature_set": feature_set,
        "train_positive": pos,
        "validation_positive": int(val_y.sum()),
        "test_positive": int(test_y.sum()),
        "train_pr_auc": float(average_precision_score(train_y, p_train)),
        "validation_pr_auc": float(average_precision_score(val_y, p_val)),
        "test_pr_auc": float(average_precision_score(test_y, p_test)),
        "train_roc_auc": float(roc_auc_score(train_y, p_train)),
        "validation_roc_auc": float(roc_auc_score(val_y, p_val)),
        "test_roc_auc": float(roc_auc_score(test_y, p_test))
    }

    print(
        f"  Validation PR-AUC: {result['validation_pr_auc']:.6f}\n"
        f"  Test PR-AUC:       {result['test_pr_auc']:.6f}\n"
        f"  Test ROC-AUC:      {result['test_roc_auc']:.6f}"
    )

    del clf, p_train, p_val, p_test
    gc.collect()

    return result

results = []

# ---------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------
print("\n" + "-" * 80)
print("BASELINE")
print("-" * 80)

results.append(
    evaluate(
        S_train, S_val, S_test,
        y3_train, y3_val, y3_test,
        "Y3 baseline", "all_59"
    )
)

results.append(
    evaluate(
        S_train, S_val, S_test,
        y6_train, y6_val, y6_test,
        "Y6 baseline", "all_59"
    )
)

# ---------------------------------------------------------------------
# Group ablation
# ---------------------------------------------------------------------
print("\n" + "-" * 80)
print("GROUP ABLATION")
print("-" * 80)

for group_name, removed in groups.items():

    if not removed:
        continue

    keep = [i for i in range(len(features)) if i not in removed]

    # Summary dimensions are organized as:
    # last, mean, std, min, max, delta, max_step, last_step
    cols = []
    n_features = len(features)

    for stat in range(8):
        cols.extend(
            stat * n_features + i
            for i in keep
        )

    cols = np.asarray(cols, dtype=np.int64)

    print(f"\nRemoving {group_name}: {len(removed)} raw features")

    tr = S_train[:, cols]
    va = S_val[:, cols]
    te = S_test[:, cols]

    results.append(
        evaluate(
            tr, va, te,
            y3_train, y3_val, y3_test,
            f"Y3 minus {group_name}",
            f"all_59_minus_{group_name}"
        )
    )

    results.append(
        evaluate(
            tr, va, te,
            y6_train, y6_val, y6_test,
            f"Y6 minus {group_name}",
            f"all_59_minus_{group_name}"
        )
    )

    del tr, va, te
    gc.collect()

# ---------------------------------------------------------------------
# Strongest synthetic fingerprints
# ---------------------------------------------------------------------
print("\n" + "-" * 80)
print("STRONGEST SYNTHETIC-FINGERPRINT ABLATION")
print("-" * 80)

print("Removing:")
for idx in strongest:
    print("  -", features[idx])

keep = [i for i in range(len(features)) if i not in strongest]

cols = []
for stat in range(8):
    cols.extend(
        stat * len(features) + i
        for i in keep
    )

cols = np.asarray(cols, dtype=np.int64)

tr = S_train[:, cols]
va = S_val[:, cols]
te = S_test[:, cols]

results.append(
    evaluate(
        tr, va, te,
        y3_train, y3_val, y3_test,
        "Y3 strongest-fingerprint ablation",
        "all_59_minus_strongest_fingerprints"
    )
)

results.append(
    evaluate(
        tr, va, te,
        y6_train, y6_val, y6_test,
        "Y6 strongest-fingerprint ablation",
        "all_59_minus_strongest_fingerprints"
    )
)

del tr, va, te
gc.collect()

# ---------------------------------------------------------------------
# Modality ablation
# ---------------------------------------------------------------------
print("\n" + "-" * 80)
print("CYBER-ONLY / PHYSICAL-ONLY")
print("-" * 80)

cyber = (
    groups["cyber_ecu_can"]
    + groups["sensor_consistency"]
    + groups["connectivity"]
    + groups["communication_integrity"]
)

physical = (
    groups["physical_vehicle"]
    + groups["environment"]
)

for modality_name, selected in [
    ("cyber_only", cyber),
    ("physical_only", physical)
]:

    if not selected:
        continue

    cols = []

    for stat in range(8):
        cols.extend(
            stat * len(features) + i
            for i in selected
        )

    cols = np.asarray(cols, dtype=np.int64)

    tr = S_train[:, cols]
    va = S_val[:, cols]
    te = S_test[:, cols]

    results.append(
        evaluate(
            tr, va, te,
            y3_train, y3_val, y3_test,
            f"Y3 {modality_name}",
            modality_name
        )
    )

    results.append(
        evaluate(
            tr, va, te,
            y6_train, y6_val, y6_test,
            f"Y6 {modality_name}",
            modality_name
        )
    )

    del tr, va, te
    gc.collect()

# ---------------------------------------------------------------------
# Temporal corruption
# Use fixed released test/validation representations.
# Corruption is applied to the raw test/validation tensors only.
# The classifier is trained on clean training data.
# ---------------------------------------------------------------------
print("\n" + "-" * 80)
print("TEMPORAL TELEMETRY CORRUPTION")
print("-" * 80)

# Use deterministic subsets to control memory.
VAL_N = min(30000, len(X_val))
TEST_N = min(30000, len(X_test))

val_idx = rng.choice(len(X_val), VAL_N, replace=False)
test_idx = rng.choice(len(X_test), TEST_N, replace=False)

# Train baseline classifiers once for corruption evaluation.
def fit_model(train_x, train_y):
    pos = int(train_y.sum())
    neg = len(train_y) - pos
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=SEED
    )
    weights = np.where(
        train_y == 1,
        neg / max(pos, 1),
        1.0
    ).astype(np.float32)
    clf.fit(train_x, train_y, sample_weight=weights)
    return clf

model_y3 = fit_model(S_train, y3_train)
model_y6 = fit_model(S_train, y6_train)

corruption_results = []

def corrupt_window(x, rate):
    x = np.array(x, dtype=np.float32, copy=True)

    n, t, f = x.shape
    mask = rng.random((n, t, f)) < rate

    # Temporal carry-forward corruption.
    for step in range(1, t):
        m = mask[:, step, :]
        if np.any(m):
            x[:, step, :][m] = x[:, step - 1, :][m]

    return x

for rate in [0.05, 0.10, 0.20]:

    print(f"\nCorruption rate: {rate:.0%}")

    val_raw = corrupt_window(X_val[val_idx], rate)
    test_raw = corrupt_window(X_test[test_idx], rate)

    val_sum = summarize(val_raw)
    test_sum = summarize(test_raw)

    val_sum = ((val_sum - mu) / sd).astype(np.float32)
    test_sum = ((test_sum - mu) / sd).astype(np.float32)

    p3_val = model_y3.predict_proba(val_sum)[:, 1]
    p3_test = model_y3.predict_proba(test_sum)[:, 1]

    p6_val = model_y6.predict_proba(val_sum)[:, 1]
    p6_test = model_y6.predict_proba(test_sum)[:, 1]

    r3 = {
        "corruption_rate": rate,
        "horizon": "3step",
        "validation_pr_auc": float(
            average_precision_score(y3_val[val_idx], p3_val)
        ),
        "test_pr_auc": float(
            average_precision_score(y3_test[test_idx], p3_test)
        ),
        "validation_roc_auc": float(
            roc_auc_score(y3_val[val_idx], p3_val)
        ),
        "test_roc_auc": float(
            roc_auc_score(y3_test[test_idx], p3_test)
        )
    }

    r6 = {
        "corruption_rate": rate,
        "horizon": "6step",
        "validation_pr_auc": float(
            average_precision_score(y6_val[val_idx], p6_val)
        ),
        "test_pr_auc": float(
            average_precision_score(y6_test[test_idx], p6_test)
        ),
        "validation_roc_auc": float(
            roc_auc_score(y6_val[val_idx], p6_val)
        ),
        "test_roc_auc": float(
            roc_auc_score(y6_test[test_idx], p6_test)
        )
    }

    corruption_results.extend([r3, r6])

    print(
        f"  Y3 Test PR-AUC: {r3['test_pr_auc']:.6f}\n"
        f"  Y6 Test PR-AUC: {r6['test_pr_auc']:.6f}"
    )

    del val_raw, test_raw, val_sum, test_sum
    del p3_val, p3_test, p6_val, p6_test
    gc.collect()

del model_y3, model_y6
gc.collect()

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
baseline_y3 = next(
    r["test_pr_auc"]
    for r in results
    if r["feature_set"] == "all_59"
    and r["label"] == "Y3 baseline"
)

baseline_y6 = next(
    r["test_pr_auc"]
    for r in results
    if r["feature_set"] == "all_59"
    and r["label"] == "Y6 baseline"
)

summary = {
    "status": "PASS",
    "phase": "4.10C",
    "dataset": "CRSS 2024 Temporal Cyber-Physical States V2",
    "task": "robustness_and_feature_ablation",
    "seed": SEED,
    "sample_count": int(
        len(X_train) + len(X_val) + len(X_test)
    ),
    "feature_count": 59,
    "baseline_test_pr_auc": {
        "y3": float(baseline_y3),
        "y6": float(baseline_y6)
    },
    "group_definitions": {
        k: [features[i] for i in v]
        for k, v in groups.items()
    },
    "strongest_fingerprint_features": [
        features[i] for i in strongest
    ],
    "ablation_results": results,
    "temporal_corruption_results": corruption_results,
    "label_verification": {
        "expected": expected,
        "actual": actual,
        "status": "PASS"
    },
    "interpretation": {
        "synthetic_separability_warning": True,
        "purpose": "Determine whether forecasting performance survives removal of synthetic fingerprints, modality ablation, and telemetry corruption.",
        "real_world_claim_boundary": "Cyber labels are synthetic research telemetry; CRSS provides physical crash ground truth."
    }
}

OUT.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 80)
print("PHASE 4.10C COMPLETE")
print("=" * 80)
print(f"Baseline Y3 Test PR-AUC : {baseline_y3:.6f}")
print(f"Baseline Y6 Test PR-AUC : {baseline_y6:.6f}")
print(f"Ablation results        : {len(results)}")
print(f"Corruption results      : {len(corruption_results)}")
print(f"Output                  : {OUT}")
print("STATUS                  : PASS")
