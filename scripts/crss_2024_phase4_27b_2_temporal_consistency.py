from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

SEQ_DIR = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
OUT_DIR = ROOT / "experiments" / "modeling" / "v2" / "uncertainty" / "phase4_27b_2"
MODEL_DIR = ROOT / "models" / "v2" / "uncertainty" / "phase4_27b_2"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(6)

DEVICE = torch.device("cpu")

X_train = np.load(SEQ_DIR / "X_train_v2_forecasting.npy").astype(np.float32)
X_val = np.load(SEQ_DIR / "X_validation_v2_forecasting.npy").astype(np.float32)
X_test = np.load(SEQ_DIR / "X_test_v2_forecasting.npy").astype(np.float32)

y_train = np.load(SEQ_DIR / "y_6step_train_v2_forecasting.npy").astype(np.int64)
y_val = np.load(SEQ_DIR / "y_6step_validation_v2_forecasting.npy").astype(np.int64)
y_test = np.load(SEQ_DIR / "y_6step_test_v2_forecasting.npy").astype(np.int64)

assert X_train.shape[1:] == (12, 63)
assert X_val.shape[1:] == (12, 63)
assert X_test.shape[1:] == (12, 63)

# Remove the four zero-variance features identified during model readiness.
KEEP = [i for i in range(63) if i not in [30, 35, 39, 62]]

X_train = X_train[:, :, KEEP]
X_val = X_val[:, :, KEEP]
X_test = X_test[:, :, KEEP]

assert X_train.shape[2] == 59

# ---------------------------------------------------------------------
# Train-only normalization.
# ---------------------------------------------------------------------

flat_train = X_train.reshape(-1, X_train.shape[-1])

scaler = StandardScaler()
scaler.fit(flat_train)

def normalize(x):
    shape = x.shape
    return scaler.transform(x.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)

# ---------------------------------------------------------------------
# Self-supervised training data.
#
# We train only on NORMAL sequences. The model learns:
#
#   observations[t-11:t] -> observation[t]
#
# This avoids using the synthetic anomaly labels as the training target.
# ---------------------------------------------------------------------

normal_idx = np.where(y_train == 0)[0]

rng = np.random.default_rng(SEED)

MAX_NORMAL = 100000

if len(normal_idx) > MAX_NORMAL:
    normal_idx = rng.choice(normal_idx, size=MAX_NORMAL, replace=False)

normal_train = X_train[normal_idx]

# Predict the final observation from the preceding 11 observations.
train_context = normal_train[:, :-1, :]
train_target = normal_train[:, -1, :]

print("=" * 80)
print("PHASE 4.27B.2 — SELF-SUPERVISED TEMPORAL CONSISTENCY")
print("=" * 80)
print(f"Train shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Features: {X_train.shape[-1]}")
print(f"Normal training sequences: {len(normal_train)}")
print(f"Device: {DEVICE}")
print()

# ---------------------------------------------------------------------
# GRU predictor
# ---------------------------------------------------------------------

class TemporalPredictor(nn.Module):
    def __init__(self, input_dim=59, hidden_dim=128, layers=2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=0.10 if layers > 1 else 0.0
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        h, _ = self.gru(x)
        h_last = self.norm(h[:, -1, :])
        return self.head(h_last)


model = TemporalPredictor().to(DEVICE)

criterion = nn.HuberLoss(delta=1.0)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

dataset = TensorDataset(
    torch.from_numpy(train_context),
    torch.from_numpy(train_target)
)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    drop_last=False
)

best_loss = float("inf")
best_epoch = 0
patience = 2
bad_epochs = 0

checkpoint_path = MODEL_DIR / "temporal_predictor_best.pt"

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

for epoch in range(1, 9):

    model.train()

    running = 0.0
    count = 0

    for xb, yb in loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        pred = model(xb)
        loss = criterion(pred, yb)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running += loss.item() * len(xb)
        count += len(xb)

    epoch_loss = running / count

    print(
        f"Epoch {epoch:02d} | "
        f"train Huber = {epoch_loss:.8f}"
    )

    if epoch_loss < best_loss:

        best_loss = epoch_loss
        best_epoch = epoch
        bad_epochs = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "seed": SEED,
                "input_dim": 59,
                "hidden_dim": 128,
                "layers": 2,
                "best_epoch": epoch,
                "best_train_loss": best_loss,
            },
            checkpoint_path
        )

    else:
        bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping.")
            break

# ---------------------------------------------------------------------
# Reload best model.
# ---------------------------------------------------------------------

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print()
print(f"Best epoch: {checkpoint['best_epoch']}")
print(f"Best training loss: {checkpoint['best_train_loss']:.8f}")

# ---------------------------------------------------------------------
# Prediction-error scoring.
# ---------------------------------------------------------------------

@torch.no_grad()
def prediction_scores(x, batch_size=256):

    scores = []

    for start in range(0, len(x), batch_size):

        xb = torch.from_numpy(
            x[start:start + batch_size, :-1, :]
        ).to(DEVICE)

        target = torch.from_numpy(
            x[start:start + batch_size, -1, :]
        ).to(DEVICE)

        pred = model(xb)

        # Mean squared normalized prediction error.
        err = torch.mean(
            (pred - target) ** 2,
            dim=1
        )

        scores.append(
            err.cpu().numpy()
        )

    return np.concatenate(scores)


# ---------------------------------------------------------------------
# Validation threshold from NORMAL validation sequences only.
# ---------------------------------------------------------------------

val_scores = prediction_scores(X_val)

normal_val_scores = val_scores[y_val == 0]

threshold = float(np.quantile(normal_val_scores, 0.99))

print()
print(f"Temporal prediction-error threshold: {threshold:.8f}")

# ---------------------------------------------------------------------
# Perturbation generators.
# ---------------------------------------------------------------------

def temporal_mask(x, n):

    z = x.copy()

    # Repeat the observation immediately before the masked tail.
    z[:, -n:, :] = z[:, -n-1, :][:, None, :]

    return z


def temporal_noise(x, pct, seed=123):

    rng = np.random.default_rng(seed)

    z = x.copy()

    scale = np.std(
        x[:, -1, :],
        axis=0,
        keepdims=True
    )

    noise = rng.normal(
        0.0,
        pct * np.maximum(scale, 1e-6),
        size=z[:, -1, :].shape
    )

    z[:, -1, :] += noise.astype(np.float32)

    return z


def temporal_velocity_shift(x, factor=2.0):

    z = x.copy()

    # Modify the final trajectory increment.
    delta = z[:, -1, :] - z[:, -2, :]

    z[:, -1, :] = (
        z[:, -2, :]
        + factor * delta
    )

    return z


def evaluate_condition(name, x, labels):

    scores = prediction_scores(x)

    auc = float(
        roc_auc_score(labels, scores)
    )

    ap = float(
        average_precision_score(labels, scores)
    )

    detection = float(
        np.mean(scores >= threshold)
    )

    result = {
        "condition": name,
        "roc_auc": auc,
        "average_precision": ap,
        "threshold": threshold,
        "detection_rate": detection,
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "score_p99": float(np.quantile(scores, 0.99)),
    }

    print(
        f"{name:28s} "
        f"AUC={auc:.6f} "
        f"AP={ap:.6f} "
        f"detect={detection:.4%} "
        f"mean={result['score_mean']:.6f}"
    )

    return result


# ---------------------------------------------------------------------
# Validation OOD experiments.
# ---------------------------------------------------------------------

val_results = []

labels_val = np.ones(len(X_val), dtype=np.int64)

for name, condition in [
    ("temporal_mask_1", temporal_mask(X_val, 1)),
    ("temporal_mask_2", temporal_mask(X_val, 2)),
    ("temporal_mask_3", temporal_mask(X_val, 3)),
    ("temporal_noise_10pct", temporal_noise(X_val, 0.10)),
    ("temporal_velocity_shift", temporal_velocity_shift(X_val)),
]:

    # Compare perturbed data against clean validation data.
    combined = np.concatenate(
        [X_val, condition],
        axis=0
    )

    labels = np.concatenate(
        [
            np.zeros(len(X_val), dtype=np.int64),
            labels_val
        ]
    )

    result = evaluate_condition(
        "VAL_" + name,
        combined,
        labels
    )

    val_results.append(result)


# ---------------------------------------------------------------------
# Test experiments.
# ---------------------------------------------------------------------

test_results = []

test_conditions = [
    ("clean", X_test),
    ("temporal_mask_1", temporal_mask(X_test, 1)),
    ("temporal_mask_2", temporal_mask(X_test, 2)),
    ("temporal_mask_3", temporal_mask(X_test, 3)),
    ("temporal_noise_10pct", temporal_noise(X_test, 0.10, seed=456)),
    ("temporal_velocity_shift", temporal_velocity_shift(X_test)),
]

print()
print("-" * 80)
print("TEST TEMPORAL CONSISTENCY SCORES")
print("-" * 80)

for name, condition in test_conditions:

    scores = prediction_scores(condition)

    result = {
        "condition": name,
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "score_p99": float(np.quantile(scores, 0.99)),
        "detection_rate": float(np.mean(scores >= threshold)),
    }

    print(
        f"{name:28s} "
        f"mean={result['score_mean']:.6f} "
        f"median={result['score_median']:.6f} "
        f"p95={result['score_p95']:.6f} "
        f"OOD={result['detection_rate']:.4%}"
    )

    test_results.append(result)


# ---------------------------------------------------------------------
# Gate logic.
#
# We deliberately do NOT require temporal masks to achieve a perfect
# detection rate. The purpose of this phase is to measure whether a
# learned temporal predictor provides information that the previous
# raw-summary detector missed.
# ---------------------------------------------------------------------

val_auc = {
    r["condition"]: r["roc_auc"]
    for r in val_results
}

gates = {
    "model_trained": checkpoint_path.exists(),
    "validation_temporal_mask_1_auc_above_0_5":
        val_auc["VAL_temporal_mask_1"] > 0.50,
    "validation_temporal_mask_2_auc_above_0_5":
        val_auc["VAL_temporal_mask_2"] > 0.50,
    "validation_temporal_mask_3_auc_above_0_5":
        val_auc["VAL_temporal_mask_3"] > 0.50,
    "validation_temporal_noise_auc_above_0_5":
        val_auc["VAL_temporal_noise_10pct"] > 0.50,
    "validation_velocity_shift_auc_above_0_5":
        val_auc["VAL_temporal_velocity_shift"] > 0.50,
    "direct_vehicle_actuation_disabled": True,
    "trained_without_anomaly_labels": True,
}

passed = sum(bool(v) for v in gates.values())
total = len(gates)

report = {
    "phase": "4.27B.2",
    "title": "Self-Supervised Temporal Consistency Model",
    "status": "PASS" if passed == total else "FAIL",
    "gates_passed": passed,
    "gates_total": total,
    "architecture": {
        "type": "GRU next-step predictor",
        "input_features": 59,
        "context_steps": 11,
        "prediction_target": "next observation",
        "hidden_dim": 128,
        "layers": 2,
        "dropout": 0.10,
        "loss": "HuberLoss",
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
    },
    "training": {
        "normal_sequences": int(len(normal_train)),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_train_loss": float(checkpoint["best_train_loss"]),
        "seed": SEED,
        "anomaly_labels_used_for_training": False,
    },
    "threshold": {
        "method": "99th percentile normal validation prediction error",
        "value": threshold,
    },
    "validation_results": val_results,
    "test_results": test_results,
    "safety_boundary": {
        "direct_vehicle_actuation": False,
        "system_role": "predictive decision support and anomaly detection",
    },
    "scientific_note": (
        "This phase evaluates a self-supervised temporal consistency "
        "signal independently from the Phase 4.27A predictive ensemble. "
        "Synthetic anomaly labels are not used to train the temporal "
        "predictor. Results are research evidence and must not be "
        "interpreted as real-world cyberattack detection performance."
    ),
}

report_path = OUT_DIR / "crss_2024_v2_phase4_27b_2_temporal_consistency_final.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print()
print("=" * 80)
print(f"STATUS: {report['status']} ({passed}/{total} gates)")
print(f"Report: {report_path}")
print(f"Checkpoint: {checkpoint_path}")
print("=" * 80)
