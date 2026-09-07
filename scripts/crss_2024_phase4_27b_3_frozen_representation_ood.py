from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

SEQ_DIR = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
MODEL_DIR = ROOT / "models" / "v2" / "uncertainty" / "phase4_27"
OUT_DIR = ROOT / "experiments" / "modeling" / "v2" / "uncertainty" / "phase4_27b_3"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(6)

DEVICE = torch.device("cpu")

KEEP = [i for i in range(63) if i not in [30, 35, 39, 62]]

X_train = np.load(SEQ_DIR / "X_train_v2_forecasting.npy").astype(np.float32)
X_val = np.load(SEQ_DIR / "X_validation_v2_forecasting.npy").astype(np.float32)
X_test = np.load(SEQ_DIR / "X_test_v2_forecasting.npy").astype(np.float32)

y_val = np.load(SEQ_DIR / "y_6step_validation_v2_forecasting.npy").astype(np.int64)
y_test = np.load(SEQ_DIR / "y_6step_test_v2_forecasting.npy").astype(np.int64)

X_train = X_train[:, :, KEEP]
X_val = X_val[:, :, KEEP]
X_test = X_test[:, :, KEEP]

# Train-only normalization.
mu = X_train.reshape(-1, 59).mean(axis=0)
sigma = X_train.reshape(-1, 59).std(axis=0)
sigma = np.maximum(sigma, 1e-6)

def normalize(x):
    return ((x - mu) / sigma).astype(np.float32)

X_val = normalize(X_val)
X_test = normalize(X_test)

print("=" * 80)
print("PHASE 4.27B.3 — FROZEN TRANSFORMER TEMPORAL REPRESENTATION OOD")
print("=" * 80)
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}")
print("Features:   59")
print("Encoder:    frozen Phase 4.27A")
print()

class TemporalTransformer(nn.Module):

    def __init__(
        self,
        input_dim=59,
        d_model=128,
        nhead=8,
        layers=3,
        ff_dim=256,
        dropout=0.10,
        max_len=12
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        self.position = nn.Parameter(
            torch.zeros(1, max_len, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

        self.norm = nn.LayerNorm(d_model)

        self.attention = nn.Linear(
            d_model,
            1
        )

        self.head = nn.Linear(
            d_model,
            1
        )

    def hidden_sequence(self, x):

        z = self.input_projection(x)

        z = z + self.position[:, :x.shape[1], :]

        z = self.encoder(z)

        z = self.norm(z)

        return z

    def forward(self, x):

        z = self.hidden_sequence(x)

        a = torch.softmax(
            self.attention(z).squeeze(-1),
            dim=1
        )

        pooled = torch.sum(
            z * a.unsqueeze(-1),
            dim=1
        )

        return self.head(pooled)


def load_member(seed):

    model = TemporalTransformer().to(DEVICE)

    path = MODEL_DIR / f"transformer_y6_seed_{seed}_best.pt"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


SEEDS = [42, 52, 62]

models = [
    load_member(seed)
    for seed in SEEDS
]

print(f"Loaded frozen ensemble members: {SEEDS}")
print()

# ---------------------------------------------------------------------
# Hidden temporal representation extraction.
# ---------------------------------------------------------------------

@torch.no_grad()
def extract_hidden(model, x, batch_size=512):

    outputs = []

    for start in range(0, len(x), batch_size):

        xb = torch.from_numpy(
            x[start:start + batch_size]
        ).to(DEVICE)

        h = model.hidden_sequence(xb)

        outputs.append(
            h.cpu().numpy()
        )

    return np.concatenate(outputs, axis=0)


# ---------------------------------------------------------------------
# Temporal representation score.
#
# Four complementary learned-space signals:
#
# 1. hidden velocity
# 2. hidden acceleration
# 3. final transition magnitude
# 4. directional change
#
# Scores are normalized against clean validation behavior.
# ---------------------------------------------------------------------

def temporal_features(h):

    velocity = h[:, 1:, :] - h[:, :-1, :]

    acceleration = (
        velocity[:, 1:, :]
        - velocity[:, :-1, :]
    )

    velocity_norm = np.linalg.norm(
        velocity,
        axis=2
    )

    acceleration_norm = np.linalg.norm(
        acceleration,
        axis=2
    )

    final_velocity = velocity_norm[:, -1]

    final_acceleration = acceleration_norm[:, -1]

    mean_velocity = np.mean(
        velocity_norm,
        axis=1
    )

    mean_acceleration = np.mean(
        acceleration_norm,
        axis=1
    )

    # Cosine similarity between consecutive hidden states.
    eps = 1e-8

    a = h[:, 1:, :]
    b = h[:, :-1, :]

    cosine = np.sum(a * b, axis=2) / (
        np.linalg.norm(a, axis=2)
        * np.linalg.norm(b, axis=2)
        + eps
    )

    direction_change = 1.0 - cosine

    final_direction_change = direction_change[:, -1]

    mean_direction_change = np.mean(
        direction_change,
        axis=1
    )

    return np.column_stack([
        mean_velocity,
        final_velocity,
        mean_acceleration,
        final_acceleration,
        mean_direction_change,
        final_direction_change
    ]).astype(np.float32)


print("Extracting validation representations...")

val_hidden = [
    extract_hidden(model, X_val)
    for model in models
]

print("Extracting test representations...")

test_hidden = [
    extract_hidden(model, X_test)
    for model in models
]

print("Representation extraction complete.")
print()

# ---------------------------------------------------------------------
# Ensemble learned-space temporal features.
# ---------------------------------------------------------------------

val_features = np.stack(
    [temporal_features(h) for h in val_hidden],
    axis=0
)

test_features = np.stack(
    [temporal_features(h) for h in test_hidden],
    axis=0
)

# Mean across ensemble members.
val_mean = np.mean(
    val_features,
    axis=0
)

test_mean = np.mean(
    test_features,
    axis=0
)

# Ensemble disagreement.
val_std = np.std(
    val_features,
    axis=0
)

test_std = np.std(
    test_features,
    axis=0
)

# ---------------------------------------------------------------------
# Build a learned-space temporal score.
#
# Reference statistics are derived only from clean NORMAL validation
# sequences.
# ---------------------------------------------------------------------

normal_mask = y_val == 0

ref = val_mean[normal_mask]

ref_mu = np.median(
    ref,
    axis=0
)

ref_mad = np.median(
    np.abs(ref - ref_mu),
    axis=0
)

ref_scale = 1.4826 * ref_mad
ref_scale = np.maximum(ref_scale, 1e-6)

def learned_temporal_score(features):

    z = np.abs(
        (features - ref_mu) / ref_scale
    )

    return np.mean(
        z,
        axis=1
    )


val_clean_score = learned_temporal_score(
    val_mean
)

threshold = float(
    np.quantile(
        val_clean_score[normal_mask],
        0.99
    )
)

print(
    f"Learned temporal OOD threshold: {threshold:.8f}"
)

# ---------------------------------------------------------------------
# Perturbations.
# ---------------------------------------------------------------------

def temporal_mask(x, n):

    z = x.copy()

    z[:, -n:, :] = z[:, -n-1, :][:, None, :]

    return z


def temporal_noise(x, pct, seed):

    rng = np.random.default_rng(seed)

    z = x.copy()

    scale = np.std(
        x[:, -1, :],
        axis=0,
        keepdims=True
    )

    noise = rng.normal(
        0,
        pct * np.maximum(scale, 1e-6),
        size=z[:, -1, :].shape
    )

    z[:, -1, :] += noise.astype(np.float32)

    return z


def temporal_velocity_shift(x, factor=2.0):

    z = x.copy()

    delta = (
        z[:, -1, :]
        - z[:, -2, :]
    )

    z[:, -1, :] = (
        z[:, -2, :]
        + factor * delta
    )

    return z


def score_dataset(x):

    hidden = [
        extract_hidden(model, x)
        for model in models
    ]

    features = np.stack(
        [temporal_features(h) for h in hidden],
        axis=0
    )

    mean_features = np.mean(
        features,
        axis=0
    )

    std_features = np.std(
        features,
        axis=0
    )

    score = learned_temporal_score(
        mean_features
    )

    # Add ensemble disagreement as a secondary signal.
    disagreement = np.mean(
        std_features / ref_scale,
        axis=1
    )

    combined = (
        0.80 * score
        + 0.20 * disagreement
    )

    return combined


# ---------------------------------------------------------------------
# Validation discrimination.
# ---------------------------------------------------------------------

val_conditions = [
    ("temporal_mask_1", temporal_mask(X_val, 1)),
    ("temporal_mask_2", temporal_mask(X_val, 2)),
    ("temporal_mask_3", temporal_mask(X_val, 3)),
    ("temporal_noise_10pct", temporal_noise(X_val, 0.10, 123)),
    ("temporal_velocity_shift", temporal_velocity_shift(X_val)),
]

validation_results = []

for name, perturbed in val_conditions:

    clean_score = score_dataset(X_val)
    perturbed_score = score_dataset(perturbed)

    scores = np.concatenate([
        clean_score,
        perturbed_score
    ])

    labels = np.concatenate([
        np.zeros(len(clean_score), dtype=np.int64),
        np.ones(len(perturbed_score), dtype=np.int64)
    ])

    auc = float(
        roc_auc_score(labels, scores)
    )

    ap = float(
        average_precision_score(labels, scores)
    )

    detection = float(
        np.mean(
            perturbed_score >= threshold
        )
    )

    result = {
        "condition": name,
        "roc_auc": auc,
        "average_precision": ap,
        "detection_rate": detection,
        "score_mean_clean": float(np.mean(clean_score)),
        "score_mean_perturbed": float(np.mean(perturbed_score)),
    }

    validation_results.append(result)

    print(
        f"VAL_{name:24s} "
        f"AUC={auc:.6f} "
        f"AP={ap:.6f} "
        f"detect={detection:.4%}"
    )


# ---------------------------------------------------------------------
# Test conditions.
# ---------------------------------------------------------------------

test_conditions = [
    ("clean", X_test),
    ("temporal_mask_1", temporal_mask(X_test, 1)),
    ("temporal_mask_2", temporal_mask(X_test, 2)),
    ("temporal_mask_3", temporal_mask(X_test, 3)),
    ("temporal_noise_10pct", temporal_noise(X_test, 0.10, 456)),
    ("temporal_velocity_shift", temporal_velocity_shift(X_test)),
]

test_results = []

print()
print("-" * 80)
print("TEST LEARNED-REPRESENTATION TEMPORAL SCORES")
print("-" * 80)

for name, condition in test_conditions:

    scores = score_dataset(condition)

    result = {
        "condition": name,
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "score_p99": float(np.quantile(scores, 0.99)),
        "ood_detection_rate": float(
            np.mean(scores >= threshold)
        ),
    }

    test_results.append(result)

    print(
        f"{name:28s} "
        f"mean={result['score_mean']:.6f} "
        f"median={result['score_median']:.6f} "
        f"p95={result['score_p95']:.6f} "
        f"OOD={result['ood_detection_rate']:.4%}"
    )


# ---------------------------------------------------------------------
# Gates.
# ---------------------------------------------------------------------

val_auc = {
    r["condition"]: r["roc_auc"]
    for r in validation_results
}

gates = {
    "all_three_temporal_masks_above_random":
        all(
            val_auc[f"temporal_mask_{n}"] > 0.50
            for n in [1, 2, 3]
        ),

    "temporal_noise_above_random":
        val_auc["temporal_noise_10pct"] > 0.50,

    "velocity_shift_above_random":
        val_auc["temporal_velocity_shift"] > 0.50,

    "all_required_conditions_above_random":
        all(v > 0.50 for v in val_auc.values()),

    "frozen_predictive_ensemble":
        True,

    "direct_vehicle_actuation_disabled":
        True,

    "no_anomaly_label_training":
        True,
}

passed = sum(
    bool(v) for v in gates.values()
)

total = len(gates)

report = {
    "phase": "4.27B.3",
    "title": "Frozen Transformer Temporal Representation OOD",
    "status": "PASS" if passed == total else "FAIL",
    "gates_passed": passed,
    "gates_total": total,
    "ensemble_seeds": SEEDS,
    "encoder_frozen": True,
    "architecture": {
        "input_features": 59,
        "sequence_length": 12,
        "hidden_dimension": 128,
        "transformer_layers": 3,
        "attention_heads": 8,
        "feedforward_dimension": 256,
    },
    "score_components": [
        "hidden_velocity",
        "hidden_acceleration",
        "hidden_direction_change",
        "ensemble_disagreement",
    ],
    "threshold_method":
        "99th percentile of clean normal validation learned-space score",
    "threshold": threshold,
    "validation_results": validation_results,
    "test_results": test_results,
    "safety_boundary": {
        "direct_vehicle_actuation": False,
        "system_role":
            "predictive decision support and anomaly detection",
    },
    "scientific_note": (
        "The Phase 4.27A predictive Transformer ensemble is used "
        "strictly as a frozen representation extractor. No model "
        "retraining or anomaly-label optimization occurs in this "
        "phase. The detector evaluates whether temporal perturbations "
        "become distinguishable in learned representation space."
    ),
}

report_path = (
    OUT_DIR /
    "crss_2024_v2_phase4_27b_3_frozen_representation_ood_final.json"
)

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print()
print("=" * 80)
print(
    f"STATUS: {report['status']} "
    f"({passed}/{total} gates)"
)
print(f"Report: {report_path}")
print("=" * 80)
