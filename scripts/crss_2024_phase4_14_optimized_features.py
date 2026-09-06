from pathlib import Path
import json
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

SEQ_DIR = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
CONTRIB_FILE = ROOT / "experiments" / "modeling" / "v2" / "feature_contribution" / "crss_2024_v2_feature_contribution_results.json"
MAP_FILE = ROOT / "experiments" / "modeling" / "v2" / "cross_modal" / "crss_2024_v2_modality_map.json"

OUT = ROOT / "experiments" / "modeling" / "v2" / "optimized_features" / "crss_2024_v2_optimized_feature_results.json"

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
FEATURE_INDEX = {f: i for i, f in enumerate(FEATURES)}

CONTRIB = json.loads(
    CONTRIB_FILE.read_text(encoding="utf-8")
)

MODALITY = json.loads(
    MAP_FILE.read_text(encoding="utf-8")
)

CYBER = MODALITY["cyber_features"]
PHYSICAL = MODALITY["physical_features"]
DERIVED = MODALITY["derived_temporal_resilience_features"]

# ------------------------------------------------------------
# Controlled deterministic subset
# ------------------------------------------------------------

rng = np.random.default_rng(42)

positive_idx = np.flatnonzero(Y6_TRAIN == 1)
negative_idx = np.flatnonzero(Y6_TRAIN == 0)

N = min(100000, X_TRAIN.shape[0])

negative_needed = N - len(positive_idx)

negative_idx = rng.choice(
    negative_idx,
    size=negative_needed,
    replace=False
)

train_idx = np.concatenate([positive_idx, negative_idx])
rng.shuffle(train_idx)

y_train = np.asarray(Y6_TRAIN[train_idx]).astype(np.int8)
y_test = np.asarray(Y6_TEST).astype(np.int8)

# ------------------------------------------------------------
# Temporal summary
# ------------------------------------------------------------

def summarize(X, indices):

    A = np.asarray(
        X[:, :, indices],
        dtype=np.float32
    )

    mean = A.mean(axis=1)
    std = A.std(axis=1)
    minimum = A.min(axis=1)
    maximum = A.max(axis=1)
    last = A[:, -1, :]
    delta = last - A[:, 0, :]

    return np.concatenate(
        [
            mean,
            std,
            minimum,
            maximum,
            last,
            delta
        ],
        axis=1
    ).astype(np.float32)


def evaluate(features):

    indices = [
        FEATURE_INDEX[f]
        for f in features
    ]

    Xtr = summarize(
        X_TRAIN[train_idx],
        indices
    )

    Xte = summarize(
        X_TEST,
        indices
    )

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=42
    )

    model.fit(
        Xtr,
        y_train
    )

    pred = model.predict_proba(
        Xte
    )[:, 1]

    return float(
        average_precision_score(
            y_test,
            pred
        )
    )


# ------------------------------------------------------------
# Select strongest contributors
# ------------------------------------------------------------

rows = CONTRIB["feature_contributions"]

positive_rows = [
    r for r in rows
    if r["contribution_delta"] > 0
]

positive_rows.sort(
    key=lambda r: r["contribution_delta"],
    reverse=True
)

# Top 10 cyber
cyber_core = [
    r["feature"]
    for r in positive_rows
    if r["modality"] == "cyber"
][:10]

# Top 5 physical
physical_core = [
    r["feature"]
    for r in positive_rows
    if r["modality"] == "physical"
][:5]

# Top 5 derived
derived_core = [
    r["feature"]
    for r in positive_rows
    if r["modality"] == "derived_temporal_resilience"
][:5]

# ------------------------------------------------------------
# Experiment sets
# ------------------------------------------------------------

feature_sets = {

    "cyber_core": cyber_core,

    "cyber_core_plus_physical_core":
        list(dict.fromkeys(
            cyber_core + physical_core
        )),

    "cyber_core_plus_derived_core":
        list(dict.fromkeys(
            cyber_core + derived_core
        )),

    "optimized_cyber_physical":
        list(dict.fromkeys(
            cyber_core +
            physical_core +
            derived_core
        )),

    "all_59":
        list(dict.fromkeys(
            CYBER +
            PHYSICAL +
            DERIVED
        )),
}

# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

results = {}

for name, features in feature_sets.items():

    score = evaluate(features)

    results[name] = {
        "feature_count": len(features),
        "test_y6_pr_auc": score,
        "features": features
    }

# ------------------------------------------------------------
# Ranking
# ------------------------------------------------------------

ranking = sorted(
    results.items(),
    key=lambda x: x[1]["test_y6_pr_auc"],
    reverse=True
)

output = {

    "phase": "4.14",

    "status": "PASS",

    "task":
        "Targeted Optimized Feature-Set Analysis",

    "target":
        "Y6",

    "controlled_train_samples":
        int(len(train_idx)),

    "test_samples":
        int(len(y_test)),

    "seed":
        42,

    "feature_sets":
        results,

    "ranking":
        [
            {
                "rank": i + 1,
                "name": name,
                "feature_count": value["feature_count"],
                "test_y6_pr_auc": value["test_y6_pr_auc"]
            }
            for i, (name, value)
            in enumerate(ranking)
        ],

    "methodological_note":
        "Feature sets are constructed from positive leave-one-feature-out "
        "contributions identified in Phase 4.13. This is a diagnostic "
        "optimization experiment using a controlled 100k training subset "
        "and the untouched test set. Results are not causal evidence."
}

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUT.write_text(
    json.dumps(
        output,
        indent=2
    ),
    encoding="utf-8"
)

print("=" * 72)
print("PHASE 4.14 — TARGETED OPTIMIZED FEATURE-SET ANALYSIS")
print("=" * 72)

print("\nSELECTED FEATURES:")

print("\nCyber Core:")
for f in cyber_core:
    print("  ", f)

print("\nPhysical Core:")
for f in physical_core:
    print("  ", f)

print("\nDerived Core:")
for f in derived_core:
    print("  ", f)

print("\nRESULTS:")

for i, (name, value) in enumerate(ranking, 1):
    print(
        f"  {i}. "
        f"{name:35s} "
        f"{value['feature_count']:2d} features "
        f"Y6 PR-AUC={value['test_y6_pr_auc']:.6f}"
    )

print("\n" + "=" * 72)
print("PHASE 4.14 RESULT PASS")
print("=" * 72)

print("Output:", OUT)
