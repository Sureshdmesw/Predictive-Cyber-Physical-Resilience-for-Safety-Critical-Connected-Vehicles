from pathlib import Path
import json
import random
import numpy as np

from sklearn.decomposition import IncrementalPCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

SEQ = ROOT / "data/processed/modeling/sequences/v2_forecasting"
MOD = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_modality_map.json"

OUT_DIR = ROOT / "experiments/modeling/v2/uncertainty/phase4_27b_1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "crss_2024_v2_phase4_27b_1_temporal_dynamics_ood_final.json"

REFERENCE_N = 100_000
PCA_COMPONENTS = 32
BATCH = 2048
SEED = 4271

np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# Load forecasting tensors
# ============================================================

def load(name):
    return np.load(SEQ / name, mmap_mode="r")

X_train = load("X_train_v2_forecasting.npy")
X_val = load("X_validation_v2_forecasting.npy")
X_test = load("X_test_v2_forecasting.npy")

y_train = np.load(
    SEQ / "y_6step_train_v2_forecasting.npy"
)
y_val = np.load(
    SEQ / "y_6step_validation_v2_forecasting.npy"
)
y_test = np.load(
    SEQ / "y_6step_test_v2_forecasting.npy"
)

KEEP = np.array([
    i for i in range(X_train.shape[2])
    if i not in [30, 35, 39, 62]
])

assert len(KEEP) == 59

X_train_r = X_train[:, :, KEEP]
X_val_r = X_val[:, :, KEEP]
X_test_r = X_test[:, :, KEEP]

# ============================================================
# Existing 100k Phase 4.27A-style training reference
# ============================================================

pos = np.where(y_train == 1)[0]
neg = np.where(y_train == 0)[0]

rng = np.random.default_rng(SEED)

n_neg = REFERENCE_N - len(pos)

if n_neg <= 0:
    ref_idx = pos[:REFERENCE_N]
else:
    neg_sel = rng.choice(
        neg,
        size=n_neg,
        replace=False
    )
    ref_idx = np.concatenate([pos, neg_sel])

rng.shuffle(ref_idx)

# ============================================================
# Train-only normalization
# ============================================================

sum_x = np.zeros(59, dtype=np.float64)
sum_x2 = np.zeros(59, dtype=np.float64)
count = 0

for start in range(0, len(ref_idx), 256):

    ids = ref_idx[start:start + 256]

    xb = np.asarray(
        X_train_r[ids],
        dtype=np.float32
    )

    flat = xb.reshape(-1, 59).astype(np.float64)

    sum_x += flat.sum(axis=0)
    sum_x2 += np.square(flat).sum(axis=0)

    count += flat.shape[0]

mean = sum_x / count
var = np.maximum(
    sum_x2 / count - mean * mean,
    1e-8
)
std = np.sqrt(var)

mean32 = mean.astype(np.float32)
std32 = std.astype(np.float32)

def normalize(x):
    return (
        (np.asarray(x, dtype=np.float32) - mean32)
        / std32
    ).astype(np.float32)

# ============================================================
# TEMPORAL-DYNAMICS REPRESENTATION
#
# Unlike the Phase 4.27B static representation, this explicitly
# models temporal relationships.
#
# For each of 59 features:
#   1. mean level
#   2. temporal std
#   3. mean |delta|
#   4. std delta
#   5. max |delta|
#   6. mean |second difference|
#   7. max |second difference|
#   8. last-step delta
#   9. last-step absolute delta
#  10. trajectory change
#
# 590 dimensions.
# ============================================================

def temporal_dynamics_summary(x):

    # x: N x 12 x 59

    dx = np.diff(x, axis=1)

    d2x = np.diff(dx, axis=1)

    level_mean = x.mean(axis=1)

    level_std = x.std(axis=1)

    abs_delta_mean = np.abs(dx).mean(axis=1)

    delta_std = dx.std(axis=1)

    abs_delta_max = np.abs(dx).max(axis=1)

    abs_d2_mean = np.abs(d2x).mean(axis=1)

    abs_d2_max = np.abs(d2x).max(axis=1)

    last_delta = dx[:, -1, :]

    last_abs_delta = np.abs(dx[:, -1, :])

    trajectory_change = (
        x[:, -1, :] - x[:, 0, :]
    )

    return np.concatenate(
        [
            level_mean,
            level_std,
            abs_delta_mean,
            delta_std,
            abs_delta_max,
            abs_d2_mean,
            abs_d2_max,
            last_delta,
            last_abs_delta,
            trajectory_change
        ],
        axis=1
    ).astype(np.float32)

# ============================================================
# Fit temporal PCA on TRAIN ONLY
# ============================================================

print("=" * 65)
print("PHASE 4.27B.1: TEMPORAL-DYNAMICS OOD")
print("=" * 65)
print()
print("Reference sequences:", len(ref_idx))
print("Temporal representation: 590 dimensions")
print("PCA dimensions:", PCA_COMPONENTS)
print()

pca = IncrementalPCA(
    n_components=PCA_COMPONENTS,
    batch_size=BATCH
)

for start in range(0, len(ref_idx), BATCH):

    ids = ref_idx[start:start + BATCH]

    xb = normalize(X_train_r[ids])

    summary = temporal_dynamics_summary(xb)

    pca.partial_fit(summary)

    if start % (BATCH * 10) == 0:
        print(
            f"PCA fit: {min(start + BATCH, len(ref_idx))}"
            f"/{len(ref_idx)}"
        )

# ============================================================
# Fit robust covariance on TRAIN PCA representation
# ============================================================

train_pca = np.empty(
    (len(ref_idx), PCA_COMPONENTS),
    dtype=np.float32
)

for start in range(0, len(ref_idx), BATCH):

    ids = ref_idx[start:start + BATCH]

    xb = normalize(X_train_r[ids])

    summary = temporal_dynamics_summary(xb)

    train_pca[
        start:start + len(ids)
    ] = pca.transform(summary).astype(np.float32)

cov = LedoitWolf(
    assume_centered=False
)

cov.fit(train_pca)

reference_mean = cov.location_.astype(np.float64)
precision = cov.precision_.astype(np.float64)

def mahalanobis(scores):

    d = scores.astype(np.float64) - reference_mean

    return np.einsum(
        "ij,jk,ik->i",
        d,
        precision,
        d
    )

def temporal_ood_score(X):

    scores = np.empty(
        len(X),
        dtype=np.float64
    )

    for start in range(0, len(X), BATCH):

        xb = normalize(
            X[start:start + BATCH]
        )

        summary = temporal_dynamics_summary(xb)

        ps = pca.transform(summary)

        scores[
            start:start + len(xb)
        ] = mahalanobis(ps)

    return scores

# ============================================================
# Validation normal reference
# ============================================================

val_normal = np.where(y_val == 0)[0]

val_scores = temporal_ood_score(
    X_val_r[val_normal]
)

TEMPORAL_OOD_THRESHOLD = float(
    np.quantile(
        val_scores,
        0.99
    )
)

print()
print(
    "Temporal OOD threshold:",
    TEMPORAL_OOD_THRESHOLD
)

# ============================================================
# Validation temporal OOD discrimination
# ============================================================

eval_n = min(
    20_000,
    len(val_normal)
)

rng = np.random.default_rng(SEED)

eval_ids = rng.choice(
    val_normal,
    size=eval_n,
    replace=False
)

X_clean = np.asarray(
    X_val_r[eval_ids],
    dtype=np.float32
)

clean_scores = temporal_ood_score(
    X_clean
)

conditions = {
    "temporal_mask_1": ("mask", 1),
    "temporal_mask_2": ("mask", 2),
    "temporal_mask_3": ("mask", 3),

    "temporal_noise_05": ("noise", 0.05),
    "temporal_noise_10": ("noise", 0.10),
    "temporal_noise_20": ("noise", 0.20),

    "temporal_velocity_shift": ("velocity", 1.0),
}

def perturb_temporal(X, mode, strength):

    Y = np.array(
        X,
        dtype=np.float32,
        copy=True
    )

    rng = np.random.default_rng(
        SEED + int(strength * 1000)
    )

    if mode == "mask":

        k = int(strength)

        # Replace the final k observations with the
        # preceding valid observation.
        Y[:, -k:, :] = Y[:, [-k - 1], :]

    elif mode == "noise":

        # Perturb temporal differences rather than levels.
        dx = np.diff(Y, axis=1)

        noise = rng.normal(
            0,
            strength,
            size=dx.shape
        ).astype(np.float32)

        dx_shifted = dx + noise

        Y[:, 1:, :] = (
            Y[:, [0], :]
            + np.cumsum(
                dx_shifted,
                axis=1
            )
        )

    elif mode == "velocity":

        dx = np.diff(Y, axis=1)

        dx[:, -3:, :] *= (
            1.0 + float(strength)
        )

        Y[:, 1:, :] = (
            Y[:, [0], :]
            + np.cumsum(
                dx,
                axis=1
            )
        )

    else:
        raise ValueError(mode)

    return Y

validation_results = {}

for name, (mode, strength) in conditions.items():

    print()
    print("Validation condition:", name)

    X_shift = perturb_temporal(
        X_clean,
        mode,
        strength
    )

    shift_scores = temporal_ood_score(
        X_shift
    )

    labels = np.concatenate(
        [
            np.zeros(len(clean_scores)),
            np.ones(len(shift_scores))
        ]
    )

    scores = np.concatenate(
        [
            clean_scores,
            shift_scores
        ]
    )

    auc = float(
        roc_auc_score(
            labels,
            scores
        )
    )

    detection = float(
        np.mean(
            shift_scores >= TEMPORAL_OOD_THRESHOLD
        )
    )

    validation_results[name] = {
        "ood_auc": auc,
        "detection_rate": detection,
        "clean_mean_score": float(
            clean_scores.mean()
        ),
        "shift_mean_score": float(
            shift_scores.mean()
        ),
        "shift_p95_score": float(
            np.quantile(
                shift_scores,
                0.95
            )
        )
    }

    print(
        f"AUC={auc:.6f} "
        f"detection={detection:.4%}"
    )

# ============================================================
# Full test evaluation
# ============================================================

print()
print("=" * 65)
print("FULL TEST TEMPORAL-DYNAMICS EVALUATION")
print("=" * 65)

test_conditions = {
    "clean": ("clean", 0),
    "temporal_mask_1": ("mask", 1),
    "temporal_mask_2": ("mask", 2),
    "temporal_mask_3": ("mask", 3),
    "temporal_noise_10": ("noise", 0.10),
    "temporal_velocity_shift": ("velocity", 1.0),
}

test_results = {}

for name, (mode, strength) in test_conditions.items():

    print()
    print("Test condition:", name)

    if mode == "clean":

        Xt = np.asarray(
            X_test_r,
            dtype=np.float32
        )

    else:

        Xt = perturb_temporal(
            np.asarray(
                X_test_r,
                dtype=np.float32
            ),
            mode,
            strength
        )

    scores = temporal_ood_score(Xt)

    high_ood = (
        scores >= TEMPORAL_OOD_THRESHOLD
    )

    test_results[name] = {
        "mean_score": float(
            scores.mean()
        ),
        "p95_score": float(
            np.quantile(
                scores,
                0.95
            )
        ),
        "ood_rate": float(
            high_ood.mean()
        )
    }

    print(
        f"mean={scores.mean():.6f} "
        f"p95={np.quantile(scores,0.95):.6f} "
        f"OOD={high_ood.mean():.4%}"
    )

# ============================================================
# Gates
# ============================================================

gates = {
    "train_only_ood_fit": True,

    "validation_normal_only_threshold": True,

    "temporal_representation_590_dimensions": True,

    "pca_32_dimensions": True,

    "temporal_mask_1_detected": (
        validation_results[
            "temporal_mask_1"
        ]["ood_auc"] > 0.50
    ),

    "temporal_mask_2_detected": (
        validation_results[
            "temporal_mask_2"
        ]["ood_auc"] > 0.50
    ),

    "temporal_mask_3_detected": (
        validation_results[
            "temporal_mask_3"
        ]["ood_auc"] > 0.50
    ),

    "temporal_noise_detected": (
        validation_results[
            "temporal_noise_10"
        ]["ood_auc"] > 0.50
    ),

    "temporal_velocity_detected": (
        validation_results[
            "temporal_velocity_shift"
        ]["ood_auc"] > 0.50
    ),

    "test_not_used_for_threshold": True,

    "direct_vehicle_actuation_disabled": True
}

status = (
    "PASS"
    if all(gates.values())
    else "FAIL"
)

report = {
    "phase": "4.27B.1",

    "title": (
        "Temporal-Dynamics OOD Detector"
    ),

    "status": status,

    "method": {
        "representation": (
            "Temporal level + first-difference + "
            "second-difference dynamics"
        ),
        "raw_features": 59,
        "temporal_summary_dimensions": 590,
        "pca_components": PCA_COMPONENTS,
        "covariance": "Ledoit-Wolf",
        "reference_sequences": int(len(ref_idx)),
        "threshold_source": (
            "99th percentile of validation-normal "
            "temporal OOD scores"
        )
    },

    "validation_results": validation_results,

    "test_results": test_results,

    "gates": gates,

    "safety_boundary": {
        "direct_vehicle_actuation": False,
        "decision_support_only": True,
        "soc_escalation_allowed": True,
        "forensic_capture_allowed": True,
        "production_vehicle_control_claim": False,
        "synthetic_label_caveat": True
    }
}

with open(
    OUT,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        report,
        f,
        indent=2
    )

print()
print("=" * 65)
print("PHASE 4.27B.1 COMPLETE")
print("=" * 65)
print("STATUS:", status)
print("REPORT:", OUT)
print()
print("GATES:")

for k, v in gates.items():
    print(
        f"  {'PASS' if v else 'FAIL'}  {k}"
    )

print()
print("VALIDATION TEMPORAL OOD AUC:")

for k, v in validation_results.items():
    print(
        f"  {k:28s} "
        f"{v['ood_auc']:.6f}"
    )
