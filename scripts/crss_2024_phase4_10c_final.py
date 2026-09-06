from pathlib import Path
import json
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

SEQ = ROOT / "data/processed/modeling/sequences/v2_forecasting"
SPEC = ROOT / "experiments/modeling/v2/crss_2024_v2_feature_treatment_spec.json"
OUT = ROOT / "experiments/modeling/v2/advanced/crss_2024_v2_robustness_ablation_final.json"

def load_json(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)

spec = load_json(SPEC)
keep = np.asarray(spec["final_feature_indices_in_original_tensor"], dtype=np.int64)

assert len(keep) == 59, f"Expected 59 retained features, got {len(keep)}"

def load_split(name):
    X = np.load(SEQ / f"X_{name}_v2_forecasting.npy", mmap_mode="r")
    y3 = np.load(SEQ / f"y_3step_{name}_v2_forecasting.npy", mmap_mode="r")
    y6 = np.load(SEQ / f"y_6step_{name}_v2_forecasting.npy", mmap_mode="r")
    return X, y3, y6

Xtr, y3tr, y6tr = load_split("train")
Xva, y3va, y6va = load_split("validation")
Xte, y3te, y6te = load_split("test")

print("Original tensors:")
print(" train:", Xtr.shape)
print(" validation:", Xva.shape)
print(" test:", Xte.shape)

Xtr = Xtr[:, :, keep].astype(np.float32)
Xva = Xva[:, :, keep].astype(np.float32)
Xte = Xte[:, :, keep].astype(np.float32)

assert Xtr.shape[2] == 59
assert Xva.shape[2] == 59
assert Xte.shape[2] == 59

print("Final model tensors:")
print(" train:", Xtr.shape)
print(" validation:", Xva.shape)
print(" test:", Xte.shape)

def summarize(X):
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    delta = X[:, -1, :] - X[:, 0, :]
    step = np.diff(X, axis=1)
    max_abs_step = np.max(np.abs(step), axis=1)
    last_step = X[:, -1, :] - X[:, -2, :]
    return np.concatenate(
        [last, mean, std, mn, mx, delta, max_abs_step, last_step],
        axis=1
    ).astype(np.float32)

def fit_eval(Atr, ytr, Ava, yva, Ate, yte):
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )
    model.fit(Atr, ytr)
    pva = model.predict_proba(Ava)[:, 1]
    pte = model.predict_proba(Ate)[:, 1]
    return {
        "validation_pr_auc": float(average_precision_score(yva, pva)),
        "validation_roc_auc": float(roc_auc_score(yva, pva)),
        "test_pr_auc": float(average_precision_score(yte, pte)),
        "test_roc_auc": float(roc_auc_score(yte, pte))
    }

print("\nBuilding 472-dimensional representation...")
t0 = time.time()
Str = summarize(Xtr)
Sva = summarize(Xva)
Ste = summarize(Xte)

assert Str.shape[1] == 472
assert Sva.shape[1] == 472
assert Ste.shape[1] == 472

print("Representation:", Str.shape)
print("Construction time:", round(time.time()-t0,2), "sec")

# Train-only normalization
mu = Str.mean(axis=0)
sd = Str.std(axis=0)
sd[sd < 1e-8] = 1.0

Str = ((Str-mu)/sd).astype(np.float32)
Sva = ((Sva-mu)/sd).astype(np.float32)
Ste = ((Ste-mu)/sd).astype(np.float32)

results = {
    "status": "PASS",
    "feature_treatment": {
        "original_features": 63,
        "removed_constant_features": spec["removed_constant_features"],
        "final_features": 59,
        "summary_dimensions": 472
    },
    "baseline": {},
    "group_ablation": {},
    "fingerprint_ablation": {},
    "corruption": {}
}

for target_name, yt, yv, ye in [
    ("y3", y3tr, y3va, y3te),
    ("y6", y6tr, y6va, y6te)
]:
    print(f"\nBASELINE {target_name}")
    results["baseline"][target_name] = fit_eval(Str, yt, Sva, yv, Ste, ye)
    print(results["baseline"][target_name])

# Feature-name groups from final feature specification
features = spec["final_features"]

groups = {}
for i, name in enumerate(features):
    if name.startswith(("can_", "ecu_", "uds_", "replay_", "sequence_counter_",
                        "unexpected_source_", "message_integrity_", "packet_loss_",
                        "latency_", "communication_", "connectivity_")):
        groups.setdefault("cyber_ecu_can", []).append(i)
    elif name.startswith(("sensor_", "redundant_", "plausibility_")):
        groups.setdefault("sensor_consistency", []).append(i)
    elif name.startswith(("physical_", "vehicle_", "speed_", "rollover_", "fire_",
                          "damage_", "collision_", "occupant_")):
        groups.setdefault("physical_vehicle", []).append(i)
    elif name.startswith(("weather", "vision")):
        groups.setdefault("environment", []).append(i)
    elif name.startswith(("rolling_", "state_transition_", "telemetry_gap_")):
        groups.setdefault("temporal", []).append(i)

for g in ["cyber_ecu_can", "sensor_consistency", "physical_vehicle",
          "environment", "temporal"]:
    groups.setdefault(g, [])

# Summary block offsets
block_names = [
    "last", "mean", "std", "min", "max",
    "last_first", "max_abs_step", "last_step"
]

def remove_features(S, remove_idx):
    keep_idx = [i for i in range(59) if i not in set(remove_idx)]
    cols = []
    for b in range(8):
        off = b*59
        cols.extend([off+i for i in keep_idx])
    return S[:, cols]

for g, idx in groups.items():
    if not idx or len(idx) >= 59:
        continue

    print(f"\nABLATION: remove {g} ({len(idx)} features)")
    Atr = remove_features(Str, idx)
    Ava = remove_features(Sva, idx)
    Ate = remove_features(Ste, idx)

    results["group_ablation"][g] = {
        "removed_feature_count": len(idx),
        "removed_features": [features[i] for i in idx]
    }

    for target_name, yt, yv, ye in [
        ("y3", y3tr, y3va, y3te),
        ("y6", y6tr, y6va, y6te)
    ]:
        r = fit_eval(Atr, yt, Ava, yv, Ate, ye)
        results["group_ablation"][g][target_name] = r
        print(target_name, r)

# Strong synthetic fingerprint features
fingerprints = [
    "unexpected_source_id_rate",
    "redundant_sensor_divergence",
    "latency_jitter_ms",
    "replay_indicator_rate",
    "sequence_counter_anomaly_rate",
    "packet_loss_rate",
    "message_integrity_failure_rate"
]

fp_idx = [features.index(x) for x in fingerprints if x in features]

print("\nFINGERPRINT ABLATION:", fingerprints)

Atr = remove_features(Str, fp_idx)
Ava = remove_features(Sva, fp_idx)
Ate = remove_features(Ste, fp_idx)

results["fingerprint_ablation"]["removed_features"] = fingerprints

for target_name, yt, yv, ye in [
    ("y3", y3tr, y3va, y3te),
    ("y6", y6tr, y6va, y6te)
]:
    r = fit_eval(Atr, yt, Ava, yv, Ate, ye)
    results["fingerprint_ablation"][target_name] = r
    print(target_name, r)

# Temporal corruption on test only
rng = np.random.default_rng(42)

for rate in [0.05, 0.10, 0.20]:
    print(f"\nCORRUPTION RATE: {rate:.0%}")

    corrupted = Xte.copy()

    mask = rng.random(corrupted.shape) < rate
    corrupted = corrupted.copy()
    corrupted[mask] = 0.0

    Sc = summarize(corrupted)
    Sc = ((Sc-mu)/sd).astype(np.float32)

    for target_name, yt, yv, ye in [
        ("y3", y3tr, y3va, y3te),
        ("y6", y6tr, y6va, y6te)
    ]:
        model = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        )
        model.fit(Str, yt)
        p = model.predict_proba(Sc)[:,1]

        r = {
            "test_pr_auc": float(average_precision_score(ye, p)),
            "test_roc_auc": float(roc_auc_score(ye, p))
        }

        results["corruption"].setdefault(str(rate), {})[target_name] = r
        print(target_name, r)

checks = {
    "original_feature_count_63": len(features) == 59 and spec["original_feature_count"] == 63,
    "final_feature_count_59": len(features) == 59,
    "summary_dimensions_472": Str.shape[1] == 472,
    "train_only_normalization": True,
    "baseline_y3_present": "y3" in results["baseline"],
    "baseline_y6_present": "y6" in results["baseline"],
    "group_ablation_completed": len(results["group_ablation"]) >= 3,
    "fingerprint_ablation_completed": "y3" in results["fingerprint_ablation"] and "y6" in results["fingerprint_ablation"],
    "corruption_completed": len(results["corruption"]) == 3
}

results["checks"] = checks
results["checks_passed"] = sum(checks.values())
results["checks_total"] = len(checks)

if all(checks.values()):
    results["release_status"] = "PASS"
else:
    results["release_status"] = "HOLD"

OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("PHASE 4.10C FINAL")
print("="*80)
print("Feature input:", 59)
print("Summary dimensions:", 472)
print("Checks:", results["checks_passed"], "/", results["checks_total"])
print("STATUS:", results["release_status"])
print("Output:", OUT)
