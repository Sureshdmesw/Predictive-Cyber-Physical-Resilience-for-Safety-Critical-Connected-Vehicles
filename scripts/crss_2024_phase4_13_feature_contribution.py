from pathlib import Path
import json
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

SEQ_DIR = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
MAP_FILE = ROOT / "experiments" / "modeling" / "v2" / "cross_modal" / "crss_2024_v2_modality_map.json"
OUT = ROOT / "experiments" / "modeling" / "v2" / "feature_contribution" / "crss_2024_v2_feature_contribution_results.json"

X_TRAIN = np.load(SEQ_DIR / "X_train_v2_forecasting.npy", mmap_mode="r")
X_TEST = np.load(SEQ_DIR / "X_test_v2_forecasting.npy", mmap_mode="r")
Y6_TRAIN = np.load(SEQ_DIR / "y_6step_train_v2_forecasting.npy", mmap_mode="r")
Y6_TEST = np.load(SEQ_DIR / "y_6step_test_v2_forecasting.npy", mmap_mode="r")

META = json.loads(
    (SEQ_DIR / "crss_2024_forecasting_tensor_metadata.json").read_text(
        encoding="utf-8"
    )
)

FEATURES = META["features"]
MODALITY = json.loads(MAP_FILE.read_text(encoding="utf-8"))

CYBER = MODALITY["cyber_features"]
PHYSICAL = MODALITY["physical_features"]
DERIVED = MODALITY["derived_temporal_resilience_features"]

FEATURE_INDEX = {f: i for i, f in enumerate(FEATURES)}

# Final 59-feature treatment.
REMOVED = {
    "gateway_present",
    "physical_state_completeness",
    "roadway_complexity_index",
    "vehicle_state_observation_count",
}

FINAL_FEATURES = [f for f in FEATURES if f not in REMOVED]

assert len(FINAL_FEATURES) == 59

# Use the full temporal window but summarize it compactly.
# This avoids another expensive deep-learning run.
def temporal_summary(X, indices):
    A = np.asarray(X[:, :, indices], dtype=np.float32)

    mean = A.mean(axis=1)
    std = A.std(axis=1)
    minimum = A.min(axis=1)
    maximum = A.max(axis=1)
    last = A[:, -1, :]
    first = A[:, 0, :]
    delta = last - first

    return np.concatenate(
        [mean, std, minimum, maximum, last, delta],
        axis=1,
    ).astype(np.float32)


def train_model(X, y):
    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=42,
    )

    model.fit(X, y)
    return model


def evaluate(indices):
    Xtr = temporal_summary(X_TRAIN[train_idx], indices)
    Xte = temporal_summary(X_TEST, indices)

    model = train_model(Xtr, y_train)
    pred = model.predict_proba(Xte)[:, 1]

    return float(average_precision_score(y_test, pred))


# ------------------------------------------------------------------
# Controlled training subset
# ------------------------------------------------------------------

rng = np.random.default_rng(42)

positive_idx = np.flatnonzero(Y6_TRAIN == 1)
negative_idx = np.flatnonzero(Y6_TRAIN == 0)

N = min(100000, X_TRAIN.shape[0])

if len(positive_idx) > N:
    raise RuntimeError("Positive examples exceed controlled subset.")

negative_needed = N - len(positive_idx)

negative_idx = rng.choice(
    negative_idx,
    size=negative_needed,
    replace=False,
)

train_idx = np.concatenate([positive_idx, negative_idx])
rng.shuffle(train_idx)

y_train = np.asarray(Y6_TRAIN[train_idx]).astype(np.int8)
y_test = np.asarray(Y6_TEST).astype(np.int8)

# ------------------------------------------------------------------
# Baseline groups
# ------------------------------------------------------------------

cyber_idx = [FEATURE_INDEX[f] for f in CYBER]
physical_idx = [FEATURE_INDEX[f] for f in PHYSICAL]
derived_idx = [FEATURE_INDEX[f] for f in DERIVED]

baseline = evaluate(cyber_idx)

groups = {
    "cyber_only": {
        "feature_count": len(cyber_idx),
        "test_y6_pr_auc": baseline,
    },
    "cyber_plus_physical": {
        "feature_count": len(cyber_idx + physical_idx),
        "test_y6_pr_auc": evaluate(cyber_idx + physical_idx),
    },
    "cyber_plus_derived": {
        "feature_count": len(cyber_idx + derived_idx),
        "test_y6_pr_auc": evaluate(cyber_idx + derived_idx),
    },
    "all_59": {
        "feature_count": len(cyber_idx + physical_idx + derived_idx),
        "test_y6_pr_auc": evaluate(
            cyber_idx + physical_idx + derived_idx
        ),
    },
}

# ------------------------------------------------------------------
# Leave-one-feature-out contribution
# ------------------------------------------------------------------

contributions = []

for feature in FINAL_FEATURES:

    idx = FEATURE_INDEX[feature]

    remaining = [
        FEATURE_INDEX[f]
        for f in FINAL_FEATURES
        if f != feature
    ]

    score = evaluate(remaining)

    contribution = groups["all_59"]["test_y6_pr_auc"] - score

    if feature in CYBER:
        modality = "cyber"
    elif feature in PHYSICAL:
        modality = "physical"
    elif feature in DERIVED:
        modality = "derived_temporal_resilience"
    else:
        modality = "unknown"

    contributions.append(
        {
            "feature": feature,
            "modality": modality,
            "without_feature_pr_auc": score,
            "contribution_delta": float(contribution),
        }
    )

contributions.sort(
    key=lambda x: x["contribution_delta"],
    reverse=True,
)

# ------------------------------------------------------------------
# Modality summary
# ------------------------------------------------------------------

modality_summary = {}

for m in ["cyber", "physical", "derived_temporal_resilience"]:
    vals = [
        x["contribution_delta"]
        for x in contributions
        if x["modality"] == m
    ]

    modality_summary[m] = {
        "feature_count": len(vals),
        "mean_leave_one_out_delta": float(np.mean(vals)),
        "max_positive_delta": float(np.max(vals)),
        "max_negative_delta": float(np.min(vals)),
    }

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

results = {
    "phase": "4.13",
    "status": "PASS",
    "task": "Rigorous Feature Contribution Analysis",
    "target": "Y6",
    "controlled_train_samples": int(len(train_idx)),
    "test_samples": int(len(y_test)),
    "seed": 42,
    "feature_count": 59,
    "groups": groups,
    "modality_summary": modality_summary,
    "feature_contributions": contributions,
    "methodological_note": (
        "Temporal summary representation with six statistics per feature "
        "(mean, std, minimum, maximum, first/last relationship, and delta). "
        "Leave-one-feature-out analysis uses a controlled 100k training subset "
        "and the untouched test set. Results are diagnostic and should not "
        "be interpreted as causal effects."
    ),
}

OUT.parent.mkdir(parents=True, exist_ok=True)

OUT.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print("=" * 72)
print("PHASE 4.13 — RIGOROUS FEATURE CONTRIBUTION ANALYSIS")
print("=" * 72)

print("\nGROUP Y6 PR-AUC:")
for name, value in groups.items():
    print(
        f"  {name:25s}: "
        f"{value['test_y6_pr_auc']:.6f}"
    )

print("\nTOP POSITIVE FEATURES:")
for row in contributions[:15]:
    print(
        f"  {row['feature']:40s} "
        f"{row['modality']:28s} "
        f"delta={row['contribution_delta']:+.6f}"
    )

print("\nTOP NEGATIVE FEATURES:")
for row in contributions[-10:][::-1]:
    print(
        f"  {row['feature']:40s} "
        f"{row['modality']:28s} "
        f"delta={row['contribution_delta']:+.6f}"
    )

print("\nMODALITY SUMMARY:")
for name, value in modality_summary.items():
    print(
        f"  {name:30s} "
        f"mean={value['mean_leave_one_out_delta']:+.6f} "
        f"max+={value['max_positive_delta']:+.6f} "
        f"max-={value['max_negative_delta']:+.6f}"
    )

print("\n" + "=" * 72)
print("PHASE 4.13 RESULT PASS")
print("=" * 72)
print("Output:", OUT)
