from pathlib import Path
import json
import time
import gc
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]

SEQ = ROOT / "data/processed/modeling/sequences/v2_forecasting"
SPEC = ROOT / "experiments/modeling/v2/crss_2024_v2_feature_treatment_spec.json"
OUT = ROOT / "experiments/modeling/v2/advanced/crss_2024_v2_robustness_ablation_final.json"


def load_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


spec = load_json(SPEC)

keep = np.asarray(
    spec["final_feature_indices_in_original_tensor"],
    dtype=np.int64
)

features = list(spec["final_features"])

assert spec["original_feature_count"] == 63
assert len(keep) == 59
assert len(features) == 59


def load_split(name):
    X = np.load(
        SEQ / f"X_{name}_v2_forecasting.npy",
        mmap_mode="r"
    )

    y3 = np.load(
        SEQ / f"y_3step_{name}_v2_forecasting.npy",
        mmap_mode="r"
    )

    y6 = np.load(
        SEQ / f"y_6step_{name}_v2_forecasting.npy",
        mmap_mode="r"
    )

    return X, y3, y6


Xtr0, y3tr, y6tr = load_split("train")
Xva0, y3va, y6va = load_split("validation")
Xte0, y3te, y6te = load_split("test")

print("Original tensors:")
print(" train:", Xtr0.shape)
print(" validation:", Xva0.shape)
print(" test:", Xte0.shape)


# ------------------------------------------------------------
# FINAL 59-FEATURE TREATMENT
# ------------------------------------------------------------

Xtr = np.asarray(
    Xtr0[:, :, keep],
    dtype=np.float32
)

Xva = np.asarray(
    Xva0[:, :, keep],
    dtype=np.float32
)

Xte = np.asarray(
    Xte0[:, :, keep],
    dtype=np.float32
)

assert Xtr.shape == (len(y3tr), 12, 59)
assert Xva.shape == (len(y3va), 12, 59)
assert Xte.shape == (len(y3te), 12, 59)

print("Final model tensors:")
print(" train:", Xtr.shape)
print(" validation:", Xva.shape)
print(" test:", Xte.shape)


# ------------------------------------------------------------
# TEMPORAL SUMMARY
# 8 statistics × 59 features = 472
# ------------------------------------------------------------

def summarize(X):

    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)

    delta = X[:, -1, :] - X[:, 0, :]

    step = np.diff(X, axis=1)

    max_abs_step = np.max(
        np.abs(step),
        axis=1
    )

    last_step = X[:, -1, :] - X[:, -2, :]

    return np.concatenate(
        [
            last,
            mean,
            std,
            mn,
            mx,
            delta,
            max_abs_step,
            last_step
        ],
        axis=1
    ).astype(np.float32)


print()
print("Building 472-dimensional representation...")

t0 = time.time()

Str = summarize(Xtr)
Sva = summarize(Xva)
Ste = summarize(Xte)

assert Str.shape[1] == 472
assert Sva.shape[1] == 472
assert Ste.shape[1] == 472

print("Representation:", Str.shape)
print(
    "Construction time:",
    round(time.time() - t0, 2),
    "sec"
)


# ------------------------------------------------------------
# TRAIN-ONLY NORMALIZATION
# ------------------------------------------------------------

mu = Str.mean(axis=0)
sd = Str.std(axis=0)

sd[sd < 1e-8] = 1.0

Str = ((Str - mu) / sd).astype(np.float32)
Sva = ((Sva - mu) / sd).astype(np.float32)
Ste = ((Ste - mu) / sd).astype(np.float32)


# ------------------------------------------------------------
# MEMORY-SAFE STRATIFIED TRAINING SUBSETS
# ------------------------------------------------------------

def stratified_subset(y, max_samples=60000, seed=42):

    y = np.asarray(y)

    rng = np.random.default_rng(seed)

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    if len(pos) >= max_samples:
        chosen_pos = rng.choice(
            pos,
            size=max_samples // 2,
            replace=False
        )
        chosen_neg = rng.choice(
            neg,
            size=max_samples - len(chosen_pos),
            replace=False
        )
    else:
        chosen_pos = pos

        remaining = max_samples - len(chosen_pos)

        chosen_neg = rng.choice(
            neg,
            size=min(remaining, len(neg)),
            replace=False
        )

    idx = np.concatenate(
        [chosen_pos, chosen_neg]
    )

    rng.shuffle(idx)

    return idx


idx3 = stratified_subset(
    y3tr,
    max_samples=60000,
    seed=42
)

idx6 = stratified_subset(
    y6tr,
    max_samples=60000,
    seed=43
)

print()
print("Memory-safe training subsets:")
print(
    " Y3:",
    len(idx3),
    "samples; positives:",
    int(y3tr[idx3].sum())
)

print(
    " Y6:",
    len(idx6),
    "samples; positives:",
    int(y6tr[idx6].sum())
)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

def make_model():

    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        max_bins=128,
        l2_regularization=1.0,
        random_state=42
    )


def fit_eval(
    Atr,
    ytr,
    Ava,
    yva,
    Ate,
    yte
):

    model = make_model()

    model.fit(
        Atr,
        ytr
    )

    pva = model.predict_proba(
        Ava
    )[:, 1]

    pte = model.predict_proba(
        Ate
    )[:, 1]

    result = {
        "validation_pr_auc":
            float(
                average_precision_score(
                    yva,
                    pva
                )
            ),

        "validation_roc_auc":
            float(
                roc_auc_score(
                    yva,
                    pva
                )
            ),

        "test_pr_auc":
            float(
                average_precision_score(
                    yte,
                    pte
                )
            ),

        "test_roc_auc":
            float(
                roc_auc_score(
                    yte,
                    pte
                )
            )
    }

    del model
    del pva
    del pte

    gc.collect()

    return result


# ------------------------------------------------------------
# RESULTS OBJECT
# ------------------------------------------------------------

results = {

    "status": "PASS",

    "dataset":
        "CRSS 2024 Temporal Cyber-Physical States V2",

    "task":
        "rolling multi-horizon forecasting",

    "feature_treatment": {

        "original_features": 63,

        "removed_constant_features":
            spec["removed_constant_features"],

        "final_features": 59,

        "summary_dimensions": 472,

        "normalization":
            "train_only_z_score"
    },

    "training_strategy": {

        "method":
            "memory_safe_stratified_subset",

        "maximum_training_samples_per_target":
            60000,

        "test_set_unchanged":
            True,

        "validation_set_unchanged":
            True
    },

    "baseline": {},

    "group_ablation": {},

    "fingerprint_ablation": {},

    "corruption": {}
}


# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

print()
print("=" * 70)
print("BASELINE Y3")
print("=" * 70)

results["baseline"]["y3"] = fit_eval(
    Str[idx3],
    y3tr[idx3],
    Sva,
    y3va,
    Ste,
    y3te
)

print(
    results["baseline"]["y3"]
)


print()
print("=" * 70)
print("BASELINE Y6")
print("=" * 70)

results["baseline"]["y6"] = fit_eval(
    Str[idx6],
    y6tr[idx6],
    Sva,
    y6va,
    Ste,
    y6te
)

print(
    results["baseline"]["y6"]
)


# ------------------------------------------------------------
# FEATURE GROUPS
# ------------------------------------------------------------

groups = {}

for i, name in enumerate(features):

    if name.startswith((
        "can_",
        "ecu_",
        "uds_",
        "replay_",
        "sequence_counter_",
        "unexpected_source_",
        "message_integrity_",
        "packet_loss_",
        "latency_",
        "communication_",
        "connectivity_"
    )):

        groups.setdefault(
            "cyber_ecu_can",
            []
        ).append(i)

    elif name.startswith((
        "sensor_",
        "redundant_",
        "plausibility_"
    )):

        groups.setdefault(
            "sensor_consistency",
            []
        ).append(i)

    elif name.startswith((
        "physical_",
        "vehicle_",
        "speed_",
        "rollover_",
        "fire_",
        "damage_",
        "collision_",
        "occupant_"
    )):

        groups.setdefault(
            "physical_vehicle",
            []
        ).append(i)

    elif name.startswith((
        "weather",
        "vision"
    )):

        groups.setdefault(
            "environment",
            []
        ).append(i)

    elif name.startswith((
        "rolling_",
        "state_transition_",
        "telemetry_gap_"
    )):

        groups.setdefault(
            "temporal",
            []
        ).append(i)


# ------------------------------------------------------------
# REMOVE FEATURE BLOCKS
# ------------------------------------------------------------

def remove_features(
    S,
    remove_idx
):

    remove = set(remove_idx)

    keep_idx = [
        i
        for i in range(59)
        if i not in remove
    ]

    cols = []

    for block in range(8):

        offset = block * 59

        cols.extend(
            offset + i
            for i in keep_idx
        )

    return S[:, cols]


# ------------------------------------------------------------
# GROUP ABLATIONS
# ------------------------------------------------------------

for group_name in [
    "cyber_ecu_can",
    "sensor_consistency",
    "physical_vehicle",
    "environment",
    "temporal"
]:

    idx = groups.get(
        group_name,
        []
    )

    if not idx:
        print(
            "Skipping empty group:",
            group_name
        )
        continue

    print()
    print("=" * 70)
    print(
        "ABLATION:",
        group_name,
        "(",
        len(idx),
        "features)"
    )
    print("=" * 70)

    results["group_ablation"][
        group_name
    ] = {

        "removed_feature_count":
            len(idx),

        "removed_features":
            [features[i] for i in idx]
    }

    Atr3 = remove_features(
        Str[idx3],
        idx
    )

    Atr6 = remove_features(
        Str[idx6],
        idx
    )

    Ava = remove_features(
        Sva,
        idx
    )

    Ate = remove_features(
        Ste,
        idx
    )

    print("Y3...")

    results["group_ablation"][
        group_name
    ]["y3"] = fit_eval(
        Atr3,
        y3tr[idx3],
        Ava,
        y3va,
        Ate,
        y3te
    )

    print(
        results["group_ablation"][
            group_name
        ]["y3"]
    )

    del Atr3
    gc.collect()

    print("Y6...")

    results["group_ablation"][
        group_name
    ]["y6"] = fit_eval(
        Atr6,
        y6tr[idx6],
        Ava,
        y6va,
        Ate,
        y6te
    )

    print(
        results["group_ablation"][
            group_name
        ]["y6"]
    )

    del Atr6
    del Ava
    del Ate

    gc.collect()


# ------------------------------------------------------------
# STRONG SYNTHETIC FINGERPRINT ABLATION
# ------------------------------------------------------------

fingerprints = [

    "unexpected_source_id_rate",

    "redundant_sensor_divergence",

    "latency_jitter_ms",

    "replay_indicator_rate",

    "sequence_counter_anomaly_rate",

    "packet_loss_rate",

    "message_integrity_failure_rate"
]


fp_idx = [
    features.index(name)
    for name in fingerprints
    if name in features
]


print()
print("=" * 70)
print("SYNTHETIC FINGERPRINT ABLATION")
print("=" * 70)

print(
    "Features removed:",
    len(fp_idx)
)

results["fingerprint_ablation"][
    "removed_features"
] = fingerprints

results["fingerprint_ablation"][
    "removed_feature_count"
] = len(fp_idx)


Atr3 = remove_features(
    Str[idx3],
    fp_idx
)

Atr6 = remove_features(
    Str[idx6],
    fp_idx
)

Ava = remove_features(
    Sva,
    fp_idx
)

Ate = remove_features(
    Ste,
    fp_idx
)


print("Y3...")

results["fingerprint_ablation"][
    "y3"
] = fit_eval(
    Atr3,
    y3tr[idx3],
    Ava,
    y3va,
    Ate,
    y3te
)

print(
    results["fingerprint_ablation"]["y3"]
)


del Atr3
gc.collect()


print("Y6...")

results["fingerprint_ablation"][
    "y6"
] = fit_eval(
    Atr6,
    y6tr[idx6],
    Ava,
    y6va,
    Ate,
    y6te
)

print(
    results["fingerprint_ablation"]["y6"]
)


del Atr6
del Ava
del Ate

gc.collect()


# ------------------------------------------------------------
# TEST TELEMETRY CORRUPTION
# ------------------------------------------------------------

print()
print("=" * 70)
print("TEST TELEMETRY CORRUPTION")
print("=" * 70)

rng = np.random.default_rng(42)


for rate in [
    0.05,
    0.10,
    0.20
]:

    print()
    print(
        "Corruption:",
        f"{rate:.0%}"
    )

    corrupted = np.array(
        Xte,
        dtype=np.float32,
        copy=True
    )

    mask = (
        rng.random(
            corrupted.shape
        )
        < rate
    )

    corrupted[mask] = 0.0

    Sc = summarize(
        corrupted
    )

    Sc = (
        (Sc - mu) / sd
    ).astype(np.float32)

    del corrupted
    del mask

    gc.collect()


    results[
        "corruption"
    ][str(rate)] = {}


    # Y3 model
    model3 = make_model()

    model3.fit(
        Str[idx3],
        y3tr[idx3]
    )

    p3 = model3.predict_proba(
        Sc
    )[:, 1]

    results[
        "corruption"
    ][str(rate)]["y3"] = {

        "test_pr_auc":
            float(
                average_precision_score(
                    y3te,
                    p3
                )
            ),

        "test_roc_auc":
            float(
                roc_auc_score(
                    y3te,
                    p3
                )
            )
    }

    del model3
    del p3

    gc.collect()


    # Y6 model
    model6 = make_model()

    model6.fit(
        Str[idx6],
        y6tr[idx6]
    )

    p6 = model6.predict_proba(
        Sc
    )[:, 1]

    results[
        "corruption"
    ][str(rate)]["y6"] = {

        "test_pr_auc":
            float(
                average_precision_score(
                    y6te,
                    p6
                )
            ),

        "test_roc_auc":
            float(
                roc_auc_score(
                    y6te,
                    p6
                )
            )
    }

    print(
        "Y3:",
        results[
            "corruption"
        ][str(rate)]["y3"]
    )

    print(
        "Y6:",
        results[
            "corruption"
        ][str(rate)]["y6"]
    )

    del model6
    del p6
    del Sc

    gc.collect()


# ------------------------------------------------------------
# RELEASE CHECKS
# ------------------------------------------------------------

checks = {

    "original_feature_count_63":
        spec["original_feature_count"] == 63,

    "final_feature_count_59":
        len(features) == 59,

    "summary_dimensions_472":
        Str.shape[1] == 472,

    "train_only_normalization":
        True,

    "baseline_y3_present":
        "y3" in results["baseline"],

    "baseline_y6_present":
        "y6" in results["baseline"],

    "group_ablation_completed":
        len(results["group_ablation"]) >= 3,

    "fingerprint_ablation_completed":
        "y3" in results["fingerprint_ablation"]
        and
        "y6" in results["fingerprint_ablation"],

    "corruption_completed":
        len(results["corruption"]) == 3
}


results["checks"] = checks

results["checks_passed"] = sum(
    bool(v)
    for v in checks.values()
)

results["checks_total"] = len(checks)

results["release_status"] = (
    "PASS"
    if all(checks.values())
    else
    "HOLD"
)


# ------------------------------------------------------------
# WRITE ARTIFACT
# ------------------------------------------------------------

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print()
print("=" * 80)
print("PHASE 4.10C MEMORY-SAFE FINAL")
print("=" * 80)

print(
    "Feature input:",
    59
)

print(
    "Summary dimensions:",
    472
)

print(
    "Checks:",
    results["checks_passed"],
    "/",
    results["checks_total"]
)

print(
    "STATUS:",
    results["release_status"]
)

print(
    "Output:",
    OUT
)
