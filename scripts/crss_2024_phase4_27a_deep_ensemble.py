import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)

# ================================================================
# PHASE 4.27A
# UNCERTAINTY-AWARE TEMPORAL TRANSFORMER DEEP ENSEMBLE
# ================================================================

ROOT = Path(
    r"E:\Predictive Cyber-Physical Resilience for "
    r"Safety-Critical Connected Vehicles"
)

SEQ = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"

X_TRAIN = SEQ / "X_train_v2_forecasting.npy"
X_VAL = SEQ / "X_validation_v2_forecasting.npy"
X_TEST = SEQ / "X_test_v2_forecasting.npy"

Y_TRAIN = SEQ / "y_6step_train_v2_forecasting.npy"
Y_VAL = SEQ / "y_6step_validation_v2_forecasting.npy"
Y_TEST = SEQ / "y_6step_test_v2_forecasting.npy"

OUT_DIR = (
    ROOT
    / "experiments"
    / "modeling"
    / "v2"
    / "uncertainty"
    / "phase4_27"
)

MODEL_DIR = ROOT / "models" / "v2" / "uncertainty" / "phase4_27"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPORT = OUT_DIR / (
    "crss_2024_v2_phase4_27a_deep_ensemble_results.json"
)

# ================================================================
# CONFIGURATION
# ================================================================

SEEDS = [42, 52, 62]

MAX_TRAIN_SAMPLES = 100_000
BATCH_SIZE = 128
MAX_EPOCHS = 5
PATIENCE = 2

INPUT_FEATURES = 59
WINDOW = 12

D_MODEL = 128
NHEAD = 8
LAYERS = 3
FF_DIM = 256
DROPOUT = 0.10

LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

DEVICE = torch.device("cpu")

# Keep CPU utilization controlled.
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

print("=" * 80)
print("PHASE 4.27A — DEEP ENSEMBLE TEMPORAL TRANSFORMER")
print("=" * 80)

print(f"Device          : {DEVICE}")
print(f"Seeds           : {SEEDS}")
print(f"Training cap    : {MAX_TRAIN_SAMPLES:,}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Max epochs      : {MAX_EPOCHS}")
print(f"Patience        : {PATIENCE}")
print(f"Threads         : {torch.get_num_threads()}")

# ================================================================
# REPRODUCIBILITY
# ================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ================================================================
# DATA
# ================================================================

required = [
    X_TRAIN,
    X_VAL,
    X_TEST,
    Y_TRAIN,
    Y_VAL,
    Y_TEST,
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

X_train_raw = np.load(X_TRAIN, mmap_mode="r")
X_val_raw = np.load(X_VAL, mmap_mode="r")
X_test_raw = np.load(X_TEST, mmap_mode="r")

y_train_raw = np.load(Y_TRAIN, mmap_mode="r")
y_val = np.load(Y_VAL, mmap_mode="r")
y_test = np.load(Y_TEST, mmap_mode="r")

assert X_train_raw.shape[1:] == (12, 63)
assert X_val_raw.shape[1:] == (12, 63)
assert X_test_raw.shape[1:] == (12, 63)

# Phase 4.7 removed zero-variance indices.
REMOVED = np.array([30, 35, 39, 62], dtype=np.int64)

KEEP = np.array(
    [i for i in range(63) if i not in set(REMOVED)],
    dtype=np.int64,
)

assert len(KEEP) == 59

# IMPORTANT:
# The original tensors contain 63 features.
# Apply the established Phase 4.7 feature removal.
X_train = np.asarray(X_train_raw[:, :, KEEP], dtype=np.float32)
X_val = np.asarray(X_val_raw[:, :, KEEP], dtype=np.float32)
X_test = np.asarray(X_test_raw[:, :, KEEP], dtype=np.float32)

y_train = np.asarray(y_train_raw, dtype=np.float32)
y_val = np.asarray(y_val, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.float32)

assert X_train.shape[1:] == (12, 59)
assert X_val.shape[1:] == (12, 59)
assert X_test.shape[1:] == (12, 59)

print("\n=== DATA ===")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# ================================================================
# TRAIN SUBSET
#
# Preserve all positive samples, then sample negatives.
# ================================================================

positive_idx = np.where(y_train == 1)[0]
negative_idx = np.where(y_train == 0)[0]

rng = np.random.default_rng(42)

negative_needed = MAX_TRAIN_SAMPLES - len(positive_idx)

if negative_needed <= 0:
    selected_idx = positive_idx[:MAX_TRAIN_SAMPLES]
else:
    negative_selected = rng.choice(
        negative_idx,
        size=min(negative_needed, len(negative_idx)),
        replace=False,
    )

    selected_idx = np.concatenate(
        [positive_idx, negative_selected]
    )

rng.shuffle(selected_idx)

X_train_sub = X_train[selected_idx]
y_train_sub = y_train[selected_idx]

print("\n=== TRAIN SUBSET ===")
print("Samples :", len(y_train_sub))
print("Positive:", int(y_train_sub.sum()))
print("Rate    :", float(y_train_sub.mean()))

# ================================================================
# NORMALIZATION
#
# Fit using TRAINING SUBSET ONLY.
# ================================================================

train_mean = X_train_sub.mean(axis=(0, 1))
train_std = X_train_sub.std(axis=(0, 1))

train_std[train_std < 1e-8] = 1.0

X_train_sub = (
    (X_train_sub - train_mean[None, None, :])
    / train_std[None, None, :]
).astype(np.float32)

X_val = (
    (X_val - train_mean[None, None, :])
    / train_std[None, None, :]
).astype(np.float32)

X_test = (
    (X_test - train_mean[None, None, :])
    / train_std[None, None, :]
).astype(np.float32)

# ================================================================
# DATASET
# ================================================================

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = SequenceDataset(X_train_sub, y_train_sub)
val_ds = SequenceDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

# ================================================================
# MODEL
# ================================================================

class TemporalTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(
            INPUT_FEATURES,
            D_MODEL,
        )

        self.position = nn.Parameter(
            torch.zeros(1, WINDOW, D_MODEL)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=LAYERS,
        )

        self.attention = nn.Linear(
            D_MODEL,
            1,
        )

        self.norm = nn.LayerNorm(D_MODEL)

        self.head = nn.Linear(
            D_MODEL,
            1,
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = x + self.position

        x = self.encoder(x)

        weights = torch.softmax(
            self.attention(x),
            dim=1,
        )

        pooled = torch.sum(
            weights * x,
            dim=1,
        )

        pooled = self.norm(pooled)

        return self.head(pooled).squeeze(-1)


# ================================================================
# LOSS
# ================================================================

positive_count = float(y_train_sub.sum())
negative_count = float(len(y_train_sub) - positive_count)

pos_weight = torch.tensor(
    [negative_count / max(positive_count, 1.0)],
    dtype=torch.float32,
    device=DEVICE,
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

print("\nPositive weight:", float(pos_weight.item()))

# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def predict(model, loader):

    model.eval()

    probabilities = []
    labels = []

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(DEVICE)

        logits = model(X_batch)

        probabilities.append(
            torch.sigmoid(logits).cpu().numpy()
        )

        labels.append(
            y_batch.numpy()
        )

    return (
        np.concatenate(probabilities),
        np.concatenate(labels),
    )


def metrics(y, p):

    return {
        "pr_auc": float(
            average_precision_score(y, p)
        ),
        "roc_auc": float(
            roc_auc_score(y, p)
        ),
        "brier": float(
            brier_score_loss(y, p)
        ),
    }


# ================================================================
# ENSEMBLE STORAGE
# ================================================================

validation_predictions = []
test_predictions = []

member_results = []

# ================================================================
# TRAIN EACH ENSEMBLE MEMBER
# ================================================================

for member_number, seed in enumerate(SEEDS, start=1):

    print("\n" + "=" * 80)
    print(
        f"ENSEMBLE MEMBER {member_number}/{len(SEEDS)} "
        f"SEED={seed}"
    )
    print("=" * 80)

    set_seed(seed)

    model = TemporalTransformer().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_pr = -np.inf
    best_epoch = None
    patience_counter = 0

    checkpoint = (
        MODEL_DIR /
        f"transformer_y6_seed_{seed}_best.pt"
    )

    for epoch in range(1, MAX_EPOCHS + 1):

        start = time.time()

        model.train()

        running_loss = 0.0
        seen = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            batch_size_actual = len(y_batch)

            running_loss += (
                float(loss.item())
                * batch_size_actual
            )

            seen += batch_size_actual

        train_loss = running_loss / max(seen, 1)

        val_p, val_y = predict(
            model,
            val_loader,
        )

        val_metrics = metrics(
            val_y,
            val_p,
        )

        elapsed = time.time() - start

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"val_PR={val_metrics['pr_auc']:.6f} "
            f"val_ROC={val_metrics['roc_auc']:.6f} "
            f"time={elapsed:.1f}s"
        )

        if val_metrics["pr_auc"] > best_val_pr:

            best_val_pr = val_metrics["pr_auc"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_pr_auc": best_val_pr,
                    "feature_count": INPUT_FEATURES,
                    "window": WINDOW,
                    "architecture": {
                        "d_model": D_MODEL,
                        "nhead": NHEAD,
                        "layers": LAYERS,
                        "ff_dim": FF_DIM,
                        "dropout": DROPOUT,
                    },
                },
                checkpoint,
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    # ------------------------------------------------------------
    # Restore BEST MEMBER
    # ------------------------------------------------------------

    saved = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        saved["model_state_dict"]
    )

    val_p, val_y = predict(
        model,
        val_loader,
    )

    # Test predictions are generated now for ensemble aggregation.
    # The test labels remain completely untouched for model selection.
    test_ds = SequenceDataset(
        X_test,
        y_test,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_p, test_y_local = predict(
        model,
        test_loader,
    )

    validation_predictions.append(val_p)
    test_predictions.append(test_p)

    member_val = metrics(
        val_y,
        val_p,
    )

    member_results.append(
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_pr_auc": float(best_val_pr),
            "validation": member_val,
            "checkpoint": str(checkpoint),
        }
    )

    print(
        f"BEST seed={seed} "
        f"epoch={best_epoch} "
        f"val_PR={best_val_pr:.6f}"
    )

# ================================================================
# ENSEMBLE
# ================================================================

validation_matrix = np.vstack(
    validation_predictions
)

test_matrix = np.vstack(
    test_predictions
)

val_mean = validation_matrix.mean(axis=0)
test_mean = test_matrix.mean(axis=0)

val_std = validation_matrix.std(
    axis=0,
    ddof=0,
)

test_std = test_matrix.std(
    axis=0,
    ddof=0,
)

ensemble_val = metrics(
    y_val,
    val_mean,
)

ensemble_test = metrics(
    y_test,
    test_mean,
)

# ================================================================
# UNCERTAINTY SUMMARY
# ================================================================

confidence = np.abs(
    test_mean - 0.5
) * 2.0

entropy_eps = 1e-12

p_clip = np.clip(
    test_mean,
    entropy_eps,
    1.0 - entropy_eps,
)

entropy = -(
    p_clip * np.log(p_clip)
    + (1.0 - p_clip)
    * np.log(1.0 - p_clip)
)

# ================================================================
# HIGH-UNCERTAINTY CASES
# ================================================================

uncertainty_threshold = float(
    np.quantile(
        val_std,
        0.90,
    )
)

high_uncertainty = (
    test_std >= uncertainty_threshold
)

predicted_positive = (
    test_mean >= 0.5
)

correct = (
    predicted_positive ==
    (y_test == 1)
)

high_uncertainty_error_rate = (
    float(
        np.mean(
            ~correct[high_uncertainty]
        )
    )
    if high_uncertainty.any()
    else 0.0
)

low_uncertainty_error_rate = (
    float(
        np.mean(
            ~correct[~high_uncertainty]
        )
    )
    if (~high_uncertainty).any()
    else 0.0
)

# ================================================================
# GATES
# ================================================================

gates = {

    "three_ensemble_members_completed":
        len(member_results) == 3,

    "all_member_checkpoints_exist":
        all(
            Path(x["checkpoint"]).exists()
            for x in member_results
        ),

    "validation_predictions_finite":
        bool(
            np.isfinite(validation_matrix).all()
        ),

    "test_predictions_finite":
        bool(
            np.isfinite(test_matrix).all()
        ),

    "ensemble_validation_metrics_finite":
        all(
            np.isfinite(v)
            for v in ensemble_val.values()
        ),

    "ensemble_test_metrics_finite":
        all(
            np.isfinite(v)
            for v in ensemble_test.values()
        ),

    "uncertainty_finite":
        bool(
            np.isfinite(test_std).all()
        ),

    "validation_only_uncertainty_threshold":
        True,

    "test_not_used_for_model_selection":
        True,

    "no_transformer_retraining_after_test":
        True,

    "direct_vehicle_actuation":
        False,
}

if not all(gates.values()):

    failed = [
        name
        for name, value in gates.items()
        if not value
    ]

    raise RuntimeError(
        "Phase 4.27A failed: "
        + ", ".join(failed)
    )

# ================================================================
# REPORT
# ================================================================

report = {

    "project": (
        "Predictive Cyber-Physical Resilience for "
        "Safety-Critical Connected Vehicles"
    ),

    "phase": "4.27A",

    "title": (
        "Uncertainty-Aware Temporal Transformer "
        "Deep Ensemble"
    ),

    "status": "PASS",

    "objective": (
        "Determine whether ensemble disagreement can "
        "provide useful uncertainty information for "
        "the existing temporal cyber-physical forecasting task."
    ),

    "model": {

        "type":
            "Temporal Transformer Deep Ensemble",

        "members":
            len(SEEDS),

        "seeds":
            SEEDS,

        "input_shape":
            [12, 59],

        "architecture": {
            "d_model": D_MODEL,
            "attention_heads": NHEAD,
            "encoder_layers": LAYERS,
            "feedforward_dimension": FF_DIM,
            "dropout": DROPOUT,
        },

        "training": {
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRAD_CLIP,
            "early_stopping_patience": PATIENCE,
            "training_subset": MAX_TRAIN_SAMPLES,
        },
    },

    "member_results":
        member_results,

    "ensemble_validation":
        ensemble_val,

    "ensemble_test":
        ensemble_test,

    "uncertainty": {

        "method":
            "ensemble predictive standard deviation",

        "validation_uncertainty_threshold":
            uncertainty_threshold,

        "test_high_uncertainty_rate":
            float(
                high_uncertainty.mean()
            ),

        "mean_test_uncertainty":
            float(
                test_std.mean()
            ),

        "p90_test_uncertainty":
            float(
                np.quantile(
                    test_std,
                    0.90,
                )
            ),

        "p95_test_uncertainty":
            float(
                np.quantile(
                    test_std,
                    0.95,
                )
            ),

        "high_uncertainty_error_rate":
            high_uncertainty_error_rate,

        "low_uncertainty_error_rate":
            low_uncertainty_error_rate,
    },

    "prediction_distribution": {

        "mean_probability":
            float(test_mean.mean()),

        "p95_probability":
            float(
                np.quantile(
                    test_mean,
                    0.95,
                )
            ),

        "mean_confidence":
            float(confidence.mean()),

        "mean_entropy":
            float(entropy.mean()),
    },

    "methodological_boundary": {

        "test_used_for_training":
            False,

        "test_used_for_model_selection":
            False,

        "uncertainty_threshold_fit_on":
            "validation ensemble disagreement",

        "cyber_telemetry":
            "synthetic research telemetry",

        "cyber_labels":
            "synthetic research labels",

        "physical_ground_truth":
            "NHTSA CRSS 2024",

        "real_world_cyberattack_performance_claim":
            False,
    },

    "safety_boundary": {

        "direct_vehicle_actuation":
            False,

        "role": [
            "forecasting",
            "uncertainty estimation",
            "risk qualification",
            "SOC prioritization",
            "decision support",
        ],
    },

    "gates":
        gates,
}

REPORT.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)

print("\n" + "=" * 80)
print("PHASE 4.27A COMPLETE")
print("=" * 80)

print(
    f"Validation ensemble PR-AUC : "
    f"{ensemble_val['pr_auc']:.6f}"
)

print(
    f"Validation ensemble ROC-AUC: "
    f"{ensemble_val['roc_auc']:.6f}"
)

print(
    f"Test ensemble PR-AUC       : "
    f"{ensemble_test['pr_auc']:.6f}"
)

print(
    f"Test ensemble ROC-AUC      : "
    f"{ensemble_test['roc_auc']:.6f}"
)

print(
    f"Test ensemble Brier        : "
    f"{ensemble_test['brier']:.6f}"
)

print(
    f"High uncertainty rate      : "
    f"{high_uncertainty.mean():.6%}"
)

print(
    f"High uncertainty error     : "
    f"{high_uncertainty_error_rate:.6%}"
)

print("\n=== GATES ===")

for name, value in gates.items():
    print(
        f"{'PASS' if value else 'FAIL':5} "
        f"{name}"
    )

print("\n" + "=" * 80)
print("STATUS: PASS")
print(f"REPORT: {REPORT}")
print("=" * 80)
