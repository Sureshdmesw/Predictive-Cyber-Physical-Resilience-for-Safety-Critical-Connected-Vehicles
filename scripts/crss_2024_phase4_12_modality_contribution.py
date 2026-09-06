from pathlib import Path
import json
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression

ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

SEQ_DIR = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
MAP_FILE = ROOT / "experiments" / "modeling" / "v2" / "cross_modal" / "crss_2024_v2_modality_map.json"
OUT = ROOT / "experiments" / "modeling" / "v2" / "modality_contribution" / "crss_2024_v2_modality_contribution_results.json"

X_TRAIN = np.load(SEQ_DIR / "X_train_v2_forecasting.npy", mmap_mode="r")
X_TEST = np.load(SEQ_DIR / "X_test_v2_forecasting.npy", mmap_mode="r")
Y6_TRAIN = np.load(SEQ_DIR / "y_6step_train_v2_forecasting.npy", mmap_mode="r")
Y6_TEST = np.load(SEQ_DIR / "y_6step_test_v2_forecasting.npy", mmap_mode="r")

META = json.loads(
    (SEQ_DIR / "crss_2024_forecasting_tensor_metadata.json").read_text(encoding="utf-8")
)

FEATURES_63 = META["features"]

modality = json.loads(MAP_FILE.read_text(encoding="utf-8"))

cyber = modality["cyber_features"]
physical = modality["physical_features"]
derived = modality["derived_temporal_resilience_features"]

FINAL_FEATURES = [
    f for f in FEATURES_63
    if f not in {
        "gateway_present",
        "physical_state_completeness",
        "roadway_complexity_index",
        "vehicle_state_observation_count",
    }
]

assert len(FINAL_FEATURES) == 59

feature_index = {f: i for i, f in enumerate(FEATURES_63)}

cyber_idx = [feature_index[f] for f in cyber]
physical_idx = [feature_index[f] for f in physical]
derived_idx = [feature_index[f] for f in derived]

# Use a controlled, deterministic subset to keep memory/runtime manageable.
rng = np.random.default_rng(42)

N = min(100000, X_TRAIN.shape[0])
train_idx = rng.choice(X_TRAIN.shape[0], size=N, replace=False)

# Ensure all positive Y6 examples are retained.
positive_idx = np.flatnonzero(Y6_TRAIN == 1)

if len(positive_idx) > N:
    raise RuntimeError("Positive count exceeds controlled subset size.")

negative_needed = N - len(positive_idx)
negative_idx = np.flatnonzero(Y6_TRAIN == 0)
negative_idx = rng.choice(negative_idx, size=negative_needed, replace=False)

train_idx = np.concatenate([positive_idx, negative_idx])
rng.shuffle(train_idx)

y_train = np.asarray(Y6_TRAIN[train_idx]).astype(np.int8)
y_test = np.asarray(Y6_TEST).astype(np.int8)

def flatten_last_step(X, indices):
    return np.asarray(X[:, -1, indices], dtype=np.float32)

def evaluate_group(name, indices):
    Xtr = flatten_last_step(X_TRAIN[train_idx], indices)
    Xte = flatten_last_step(X_TEST, indices)

    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    model.fit(Xtr, y_train)
    pred = model.predict_proba(Xte)[:, 1]

    pr = average_precision_score(y_test, pred)

    return {
        "name": name,
        "feature_count": len(indices),
        "test_y6_pr_auc": float(pr),
    }

results = {
    "phase": "4.12",
    "status": "PASS",
    "task": "Modality Contribution Analysis",
    "target": "Y6",
    "controlled_train_samples": int(len(train_idx)),
    "test_samples": int(len(y_test)),
    "seed": 42,
    "groups": {},
    "individual_physical_features": [],
    "individual_cyber_features": [],
}

# Group-level baselines.
for name, indices in [
    ("cyber_only", cyber_idx),
    ("physical_only", physical_idx),
    ("derived_temporal_resilience", derived_idx),
    ("cyber_plus_physical", cyber_idx + physical_idx),
    ("cyber_plus_derived", cyber_idx + derived_idx),
    ("physical_plus_derived", physical_idx + derived_idx),
    ("cyber_plus_physical_plus_derived", cyber_idx + physical_idx + derived_idx),
]:
    results["groups"][name] = evaluate_group(name, indices)

# Individual physical feature contribution.
for feature in physical:
    idx = feature_index[feature]

    single = evaluate_group(
        f"physical_single::{feature}",
        [idx],
    )

    cyber_plus = evaluate_group(
        f"cyber_plus::{feature}",
        cyber_idx + [idx],
    )

    cyber_baseline = results["groups"]["cyber_only"]["test_y6_pr_auc"]

    single["feature"] = feature
    single["delta_vs_cyber"] = float(
        single["test_y6_pr_auc"] - cyber_baseline
    )

    cyber_plus["feature"] = feature
    cyber_plus["delta_vs_cyber"] = float(
        cyber_plus["test_y6_pr_auc"] - cyber_baseline
    )

    results["individual_physical_features"].append({
        "feature": feature,
        "physical_only_pr_auc": single["test_y6_pr_auc"],
        "physical_only_delta_vs_cyber": single["delta_vs_cyber"],
        "cyber_plus_feature_pr_auc": cyber_plus["test_y6_pr_auc"],
        "cyber_plus_feature_delta_vs_cyber": cyber_plus["delta_vs_cyber"],
    })

# Individual cyber feature contribution.
for feature in cyber:
    idx = feature_index[feature]

    single = evaluate_group(
        f"cyber_single::{feature}",
        [idx],
    )

    single["feature"] = feature
    single["delta_vs_cyber_group"] = float(
        single["test_y6_pr_auc"]
        - results["groups"]["cyber_only"]["test_y6_pr_auc"]
    )

    results["individual_cyber_features"].append(single)

# Sort physical features by effect when added to cyber.
results["individual_physical_features"].sort(
    key=lambda x: x["cyber_plus_feature_delta_vs_cyber"],
    reverse=True,
)

results["individual_cyber_features"].sort(
    key=lambda x: x["test_y6_pr_auc"],
    reverse=True,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print("=" * 72)
print("PHASE 4.12 — MODALITY CONTRIBUTION ANALYSIS")
print("=" * 72)

print("\nGROUP Y6 PR-AUC:")
for name, result in results["groups"].items():
    print(f"  {name:35s}: {result['test_y6_pr_auc']:.6f}")

print("\nTOP PHYSICAL FEATURES WHEN ADDED TO CYBER:")
for row in results["individual_physical_features"][:12]:
    print(
        f"  {row['feature']:40s} "
        f"{row['cyber_plus_feature_pr_auc']:.6f} "
        f"delta={row['cyber_plus_feature_delta_vs_cyber']:+.6f}"
    )

print("\nTOP CYBER FEATURES:")
for row in results["individual_cyber_features"][:12]:
    print(
        f"  {row['feature']:40s} "
        f"{row['test_y6_pr_auc']:.6f}"
    )

print("\n" + "=" * 72)
print("PHASE 4.12 MODALITY CONTRIBUTION ANALYSIS PASS")
print("=" * 72)
print("Output:", OUT)
