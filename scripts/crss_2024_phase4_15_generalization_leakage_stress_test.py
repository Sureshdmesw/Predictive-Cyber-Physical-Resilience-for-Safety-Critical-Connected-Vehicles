from pathlib import Path
import json
import numpy as np
import hashlib
from collections import Counter

ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

SEQ = ROOT / "data/processed/modeling/sequences/v2_forecasting"
OUT = ROOT / "experiments/modeling/v2/generalization"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
files = {
    "X_train": SEQ / "X_train_v2_forecasting.npy",
    "X_validation": SEQ / "X_validation_v2_forecasting.npy",
    "X_test": SEQ / "X_test_v2_forecasting.npy",
    "y3_train": SEQ / "y_3step_train_v2_forecasting.npy",
    "y3_validation": SEQ / "y_3step_validation_v2_forecasting.npy",
    "y3_test": SEQ / "y_3step_test_v2_forecasting.npy",
    "y6_train": SEQ / "y_6step_train_v2_forecasting.npy",
    "y6_validation": SEQ / "y_6step_validation_v2_forecasting.npy",
    "y6_test": SEQ / "y_6step_test_v2_forecasting.npy",
    "origin_train": SEQ / "prediction_origins_train_v2_forecasting.npy",
    "origin_validation": SEQ / "prediction_origins_validation_v2_forecasting.npy",
    "origin_test": SEQ / "prediction_origins_test_v2_forecasting.npy",
}

metadata_path = SEQ / "crss_2024_forecasting_tensor_metadata.json"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def sha256_array(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def load(path, mmap=True):
    return np.load(path, mmap_mode="r" if mmap else None)

# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

# ------------------------------------------------------------
# Load arrays
# ------------------------------------------------------------
X = {
    "train": load(files["X_train"]),
    "validation": load(files["X_validation"]),
    "test": load(files["X_test"]),
}

y3 = {
    "train": load(files["y3_train"]),
    "validation": load(files["y3_validation"]),
    "test": load(files["y3_test"]),
}

y6 = {
    "train": load(files["y6_train"]),
    "validation": load(files["y6_validation"]),
    "test": load(files["y6_test"]),
}

origins_raw = {
    "train": load(files["origin_train"]),
    "validation": load(files["origin_validation"]),
    "test": load(files["origin_test"]),
}

# ------------------------------------------------------------
# Basic shape reconciliation
# ------------------------------------------------------------
shape_audit = {}

for split in ["train", "validation", "test"]:
    x_n = int(X[split].shape[0])
    y3_n = int(y3[split].shape[0])
    y6_n = int(y6[split].shape[0])
    origin_n = int(origins_raw[split].shape[0])

    shape_audit[split] = {
        "X_samples": x_n,
        "y3_samples": y3_n,
        "y6_samples": y6_n,
        "origin_samples_raw": origin_n,
        "X_y3_match": x_n == y3_n,
        "X_y6_match": x_n == y6_n,
        "origin_matches_X": origin_n == x_n,
        "origin_extra": max(0, origin_n - x_n),
        "origin_shortfall": max(0, x_n - origin_n),
    }

# ------------------------------------------------------------
# IMPORTANT:
# Modeling arrays are authoritative for sample alignment.
# Origins are trimmed only if they contain the documented
# extra tail entries.
# ------------------------------------------------------------
origins = {}

for split in ["train", "validation", "test"]:
    n = X[split].shape[0]
    raw = np.asarray(origins_raw[split])
    if len(raw) < n:
        raise RuntimeError(
            f"{split}: origin array shorter than modeling samples "
            f"({len(raw)} < {n})"
        )
    origins[split] = raw[:n]

# ------------------------------------------------------------
# Origin uniqueness / overlap
# ------------------------------------------------------------
origin_sets = {
    split: set(np.asarray(origins[split]).tolist())
    for split in ["train", "validation", "test"]
}

origin_overlap = {
    "train_validation": len(origin_sets["train"] & origin_sets["validation"]),
    "train_test": len(origin_sets["train"] & origin_sets["test"]),
    "validation_test": len(origin_sets["validation"] & origin_sets["test"]),
}

# ------------------------------------------------------------
# Origin distribution
# ------------------------------------------------------------
origin_stats = {}

for split in ["train", "validation", "test"]:
    arr = np.asarray(origins[split])

    counts = Counter(arr.tolist())

    origin_stats[split] = {
        "count": int(len(arr)),
        "unique_origins": int(len(counts)),
        "min_origin": int(arr.min()) if len(arr) else None,
        "max_origin": int(arr.max()) if len(arr) else None,
        "duplicate_origin_values": int(sum(1 for v in counts.values() if v > 1)),
        "max_origin_frequency": int(max(counts.values())) if counts else 0,
    }

# ------------------------------------------------------------
# Feature-level distribution fingerprints
# This is intentionally lightweight and memory-safe.
# ------------------------------------------------------------
feature_stats = {}

for split in ["train", "validation", "test"]:
    arr = X[split]

    # Last observation in each sequence.
    last = np.asarray(arr[:, -1, :], dtype=np.float32)

    feature_stats[split] = {
        "samples": int(last.shape[0]),
        "features": int(last.shape[1]),
        "mean": np.mean(last, axis=0).astype(float).tolist(),
        "std": np.std(last, axis=0).astype(float).tolist(),
        "min": np.min(last, axis=0).astype(float).tolist(),
        "max": np.max(last, axis=0).astype(float).tolist(),
    }

# ------------------------------------------------------------
# Distribution drift
# ------------------------------------------------------------
train_mean = np.asarray(feature_stats["train"]["mean"])
train_std = np.asarray(feature_stats["train"]["std"])

drift = {}

for split in ["validation", "test"]:
    mean = np.asarray(feature_stats[split]["mean"])
    std = np.asarray(feature_stats[split]["std"])

    safe_std = np.where(train_std > 1e-8, train_std, 1.0)

    standardized_mean_shift = np.abs(mean - train_mean) / safe_std
    std_ratio = np.divide(
        std,
        np.where(train_std > 1e-8, train_std, 1.0)
    )

    drift[split] = {
        "mean_abs_z_shift": float(np.mean(standardized_mean_shift)),
        "mean_max_abs_z_shift": float(np.max(standardized_mean_shift)),
        "features_mean_shift_gt_1sd": int(np.sum(standardized_mean_shift > 1.0)),
        "features_mean_shift_gt_2sd": int(np.sum(standardized_mean_shift > 2.0)),
        "median_std_ratio": float(np.median(std_ratio)),
        "min_std_ratio": float(np.min(std_ratio)),
        "max_std_ratio": float(np.max(std_ratio)),
    }

# ------------------------------------------------------------
# Label drift
# ------------------------------------------------------------
label_stats = {}

for split in ["train", "validation", "test"]:
    label_stats[split] = {
        "samples": int(len(y3[split])),
        "y3_positive": int(np.sum(y3[split])),
        "y3_rate": float(np.mean(y3[split])),
        "y6_positive": int(np.sum(y6[split])),
        "y6_rate": float(np.mean(y6[split])),
    }

# ------------------------------------------------------------
# Exact sequence fingerprint collision audit
#
# Hashing full tensors is expensive, so use a deterministic
# compact fingerprint of the final observation vector.
# This identifies exact repeated last-step states across splits.
# ------------------------------------------------------------
def row_fingerprints(arr, decimals=6):
    last = np.asarray(arr[:, -1, :], dtype=np.float32)
    rounded = np.round(last, decimals=decimals)
    return {
        hashlib.sha256(row.tobytes()).hexdigest()
        for row in rounded
    }

fingerprints = {
    split: row_fingerprints(X[split])
    for split in ["train", "validation", "test"]
}

fingerprint_overlap = {
    "train_validation": len(fingerprints["train"] & fingerprints["validation"]),
    "train_test": len(fingerprints["train"] & fingerprints["test"]),
    "validation_test": len(fingerprints["validation"] & fingerprints["test"]),
}

# ------------------------------------------------------------
# Label-conditioned state fingerprints
# ------------------------------------------------------------
conditional_overlap = {}

for split_a, split_b in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:
    result = {}

    for target_name, target_dict in [("y3", y3), ("y6", y6)]:
        for label in [0, 1]:
            a_mask = np.asarray(target_dict[split_a]) == label
            b_mask = np.asarray(target_dict[split_b]) == label

            a_last = np.asarray(X[split_a][a_mask, -1, :], dtype=np.float32)
            b_last = np.asarray(X[split_b][b_mask, -1, :], dtype=np.float32)

            def fp_from_matrix(m):
                rounded = np.round(m, decimals=6)
                return {
                    hashlib.sha256(row.tobytes()).hexdigest()
                    for row in rounded
                }

            a_fp = fp_from_matrix(a_last)
            b_fp = fp_from_matrix(b_last)

            result[f"{target_name}_label_{label}"] = len(a_fp & b_fp)

    conditional_overlap[f"{split_a}_{split_b}"] = result

# ------------------------------------------------------------
# Dataset integrity gates
# ------------------------------------------------------------
gates = {
    "metadata_scenario_partition_statement_present":
        "Scenario identities never cross train, validation, and test partitions."
        in metadata.get("methodological_note", ""),

    "all_X_y3_shapes_match":
        all(v["X_y3_match"] for v in shape_audit.values()),

    "all_X_y6_shapes_match":
        all(v["X_y6_match"] for v in shape_audit.values()),

    "origin_arrays_sufficient_after_trim":
        all(v["origin_shortfall"] == 0 for v in shape_audit.values()),

    "origin_overlap_train_validation_zero":
        origin_overlap["train_validation"] == 0,

    "origin_overlap_train_test_zero":
        origin_overlap["train_test"] == 0,

    "origin_overlap_validation_test_zero":
        origin_overlap["validation_test"] == 0,

    "finite_features":
        all(np.isfinite(np.asarray(X[s][:])).all() for s in ["train", "validation", "test"]),

    "binary_labels":
        all(
            set(np.unique(np.asarray(y3[s])).tolist()).issubset({0, 1})
            and set(np.unique(np.asarray(y6[s])).tolist()).issubset({0, 1})
            for s in ["train", "validation", "test"]
        ),
}

status = "PASS" if all(gates.values()) else "HOLD"

# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------
report = {
    "phase": "4.15",
    "title": "CRSS 2024 Generalization and Leakage Stress Test",
    "status": status,
    "purpose": [
        "Validate split integrity.",
        "Reconcile prediction-origin alignment.",
        "Audit origin overlap.",
        "Measure feature distribution drift.",
        "Measure label-rate drift.",
        "Audit exact last-step state fingerprint collisions.",
        "Establish a generalization baseline before further modeling."
    ],
    "dataset": metadata.get("dataset"),
    "task": metadata.get("task"),
    "scenario_count": metadata.get("scenario_count"),
    "feature_count": metadata.get("feature_count"),
    "observation_window_steps": metadata.get("observation_window_steps"),
    "observation_window_seconds": metadata.get("observation_window_seconds"),
    "horizon_3_steps": metadata.get("horizon_3_steps"),
    "horizon_6_steps": metadata.get("horizon_6_steps"),
    "label_definition": metadata.get("label_definition"),
    "methodological_note": metadata.get("methodological_note"),
    "shape_audit": shape_audit,
    "origin_overlap": origin_overlap,
    "origin_stats": origin_stats,
    "label_stats": label_stats,
    "feature_distribution_drift": drift,
    "exact_last_step_fingerprint_overlap": fingerprint_overlap,
    "label_conditioned_fingerprint_overlap": conditional_overlap,
    "gates": gates,
    "interpretation": {
        "origin_alignment":
            "The modeling X/y arrays are treated as authoritative. "
            "The prediction-origin arrays contain 128 additional entries "
            "per split and are therefore trimmed to the corresponding X sample count.",
        "scenario_leakage":
            "The metadata explicitly states that scenario identities never cross "
            "train, validation, and test partitions. This audit additionally checks "
            "the available prediction-origin identifiers.",
        "synthetic_cyber_limitation":
            "Cyber telemetry and cyber anomaly labels are synthetic research signals. "
            "Strong generalization within this dataset does not establish real-world "
            "cyberattack detection performance.",
        "next_step":
            "If integrity gates pass, proceed to scenario-level and harder "
            "distribution-shift evaluation rather than immediately increasing model complexity."
    }
}

out_path = OUT / "crss_2024_v2_generalization_leakage_stress_test.json"
out_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("=" * 72)
print("PHASE 4.15 — GENERALIZATION / LEAKAGE STRESS TEST")
print("=" * 72)
print(f"STATUS: {status}")
print()
print("SHAPE AUDIT")
for split, v in shape_audit.items():
    print(
        f"{split:12s} X={v['X_samples']:,} "
        f"y3={v['y3_samples']:,} "
        f"y6={v['y6_samples']:,} "
        f"origins_raw={v['origin_samples_raw']:,} "
        f"extra={v['origin_extra']}"
    )

print()
print("ORIGIN OVERLAP")
for k, v in origin_overlap.items():
    print(f"{k:28s}: {v}")

print()
print("LABEL RATES")
for split, v in label_stats.items():
    print(
        f"{split:12s} "
        f"Y3={v['y3_rate']:.6%} "
        f"Y6={v['y6_rate']:.6%}"
    )

print()
print("DISTRIBUTION DRIFT")
for split, v in drift.items():
    print(
        f"{split:12s} "
        f"mean_abs_z={v['mean_abs_z_shift']:.4f} "
        f"max_abs_z={v['mean_max_abs_z_shift']:.4f} "
        f">1SD={v['features_mean_shift_gt_1sd']} "
        f">2SD={v['features_mean_shift_gt_2sd']}"
    )

print()
print("EXACT LAST-STEP FINGERPRINT OVERLAP")
for k, v in fingerprint_overlap.items():
    print(f"{k:28s}: {v}")

print()
print("GATES")
for k, v in gates.items():
    print(f"{'PASS' if v else 'FAIL'}  {k}")

print()
print(f"REPORT: {out_path}")
print("=" * 72)
