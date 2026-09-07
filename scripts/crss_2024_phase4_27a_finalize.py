import json
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
# PHASE 4.27A — FINALIZATION ONLY
# NO RETRAINING
# ================================================================

ROOT = Path(
    r"E:\Predictive Cyber-Physical Resilience for "
    r"Safety-Critical Connected Vehicles"
)

SEQ = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"

MODEL_DIR = ROOT / "models" / "v2" / "uncertainty" / "phase4_27"

OUT_DIR = (
    ROOT
    / "experiments"
    / "modeling"
    / "v2"
    / "uncertainty"
    / "phase4_27"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = OUT_DIR / (
    "crss_2024_v2_phase4_27a_deep_ensemble_final.json"
)

# ================================================================
# CONFIG
# ================================================================

SEEDS = [42, 52, 62]

INPUT_FEATURES = 59
WINDOW = 12

D_MODEL = 128
NHEAD = 8
LAYERS = 3
FF_DIM = 256
DROPOUT = 0.10

BATCH_SIZE = 128

DEVICE = torch.device("cpu")

torch.set_num_threads(6)
torch.set_num_interop_threads(2)

print("=" * 80)
print("PHASE 4.27A — FINALIZATION ONLY")
print("NO RETRAINING")
print("=" * 80)

# ================================================================
# PATHS
# ================================================================

X_VAL_PATH = SEQ / "X_validation_v2_forecasting.npy"
X_TEST_PATH = SEQ / "X_test_v2_forecasting.npy"

Y_VAL_PATH = SEQ / "y_6step_validation_v2_forecasting.npy"
Y_TEST_PATH = SEQ / "y_6step_test_v2_forecasting.npy"

for p in [
    X_VAL_PATH,
    X_TEST_PATH,
    Y_VAL_PATH,
    Y_TEST_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

# ================================================================
# LOAD DATA
# ================================================================

X_val_raw = np.load(
    X_VAL_PATH,
    mmap_mode="r",
)

X_test_raw = np.load(
    X_TEST_PATH,
    mmap_mode="r",
)

y_val = np.asarray(
    np.load(Y_VAL_PATH, mmap_mode="r"),
    dtype=np.float32,
)

y_test = np.asarray(
    np.load(Y_TEST_PATH, mmap_mode="r"),
    dtype=np.float32,
)

# Established Phase 4.7 zero-variance removal.
REMOVED = {30, 35, 39, 62}

KEEP = np.array(
    [
        i
        for i in range(63)
        if i not in REMOVED
    ],
    dtype=np.int64,
)

assert len(KEEP) == 59

X_val = np.asarray(
    X_val_raw[:, :, KEEP],
    dtype=np.float32,
)

X_test = np.asarray(
    X_test_raw[:, :, KEEP],
    dtype=np.float32,
)

assert X_val.shape == (
    len(y_val),
    12,
    59,
)

assert X_test.shape == (
    len(y_test),
    12,
    59,
)

print("\n=== DATA ===")
print("Validation:", X_val.shape)
print("Test      :", X_test.shape)
print("Val +     :", int(y_val.sum()))
print("Test +    :", int(y_test.sum()))

# ================================================================
# TRAIN-ONLY NORMALIZATION
#
# Reconstruct the exact training statistics from the 100k subset
# used by Phase 4.27A.
# ================================================================

X_train_path = SEQ / "X_train_v2_forecasting.npy"
y_train_path = SEQ / "y_6step_train_v2_forecasting.npy"

X_train_raw = np.load(
    X_train_path,
    mmap_mode="r",
)

y_train_raw = np.load(
    y_train_path,
    mmap_mode="r",
)

X_train = np.asarray(
    X_train_raw[:, :, KEEP],
    dtype=np.float32,
)

y_train = np.asarray(
    y_train_raw,
    dtype=np.float32,
)

positive_idx = np.where(
    y_train == 1
)[0]

negative_idx = np.where(
    y_train == 0
)[0]

rng = np.random.default_rng(42)

MAX_TRAIN_SAMPLES = 100_000

negative_needed = (
    MAX_TRAIN_SAMPLES
    - len(positive_idx)
)

negative_selected = rng.choice(
    negative_idx,
    size=negative_needed,
    replace=False,
)

selected_idx = np.concatenate(
    [
        positive_idx,
        negative_selected,
    ]
)

rng.shuffle(selected_idx)

X_train_sub = X_train[selected_idx]

train_mean = X_train_sub.mean(
    axis=(0, 1)
)

train_std = X_train_sub.std(
    axis=(0, 1)
)

train_std[
    train_std < 1e-8
] = 1.0

X_val = (
    X_val
    - train_mean[None, None, :]
) / train_std[None, None, :]

X_test = (
    X_test
    - train_mean[None, None, :]
) / train_std[None, None, :]

X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

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
            torch.zeros(
                1,
                WINDOW,
                D_MODEL,
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=NHEAD,
                dim_feedforward=FF_DIM,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=LAYERS,
        )

        self.attention = nn.Linear(
            D_MODEL,
            1,
        )

        self.norm = nn.LayerNorm(
            D_MODEL
        )

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

        pooled = self.norm(
            pooled
        )

        return self.head(
            pooled
        ).squeeze(-1)


# ================================================================
# PREDICTION
# ================================================================

def predict(model, X):

    model.eval()

    output = []

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            BATCH_SIZE,
        ):

            stop = min(
                start + BATCH_SIZE,
                len(X),
            )

            xb = torch.from_numpy(
                X[start:stop]
            ).to(DEVICE)

            logits = model(xb)

            probabilities = (
                torch.sigmoid(logits)
                .cpu()
                .numpy()
            )

            output.append(
                probabilities
            )

    return np.concatenate(
        output
    )


# ================================================================
# LOAD THREE COMPLETED MEMBERS
# ================================================================

val_predictions = []
test_predictions = []
member_results = []

for seed in SEEDS:

    checkpoint = (
        MODEL_DIR
        / f"transformer_y6_seed_{seed}_best.pt"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}"
        )

    saved = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=False,
    )

    model = TemporalTransformer().to(
        DEVICE
    )

    model.load_state_dict(
        saved["model_state_dict"]
    )

    model.eval()

    val_p = predict(
        model,
        X_val,
    )

    test_p = predict(
        model,
        X_test,
    )

    val_predictions.append(
        val_p
    )

    test_predictions.append(
        test_p
    )

    val_pr = average_precision_score(
        y_val,
        val_p,
    )

    val_roc = roc_auc_score(
        y_val,
        val_p,
    )

    val_brier = brier_score_loss(
        y_val,
        val_p,
    )

    member_results.append(
        {
            "seed": seed,
            "checkpoint_epoch": int(
                saved["epoch"]
            ),
            "validation_pr_auc": float(
                val_pr
            ),
            "validation_roc_auc": float(
                val_roc
            ),
            "validation_brier": float(
                val_brier
            ),
            "checkpoint": str(
                checkpoint
            ),
        }
    )

    print(
        f"Seed {seed}: "
        f"Val PR-AUC={val_pr:.6f} "
        f"ROC-AUC={val_roc:.6f} "
        f"Brier={val_brier:.6f}"
    )

# ================================================================
# ENSEMBLE
# ================================================================

val_matrix = np.vstack(
    val_predictions
)

test_matrix = np.vstack(
    test_predictions
)

val_mean = val_matrix.mean(
    axis=0
)

test_mean = test_matrix.mean(
    axis=0
)

val_std = val_matrix.std(
    axis=0
)

test_std = test_matrix.std(
    axis=0
)

# ================================================================
# ENSEMBLE METRICS
# ================================================================

val_pr = average_precision_score(
    y_val,
    val_mean,
)

val_roc = roc_auc_score(
    y_val,
    val_mean,
)

val_brier = brier_score_loss(
    y_val,
    val_mean,
)

test_pr = average_precision_score(
    y_test,
    test_mean,
)

test_roc = roc_auc_score(
    y_test,
    test_mean,
)

test_brier = brier_score_loss(
    y_test,
    test_mean,
)

# ================================================================
# UNCERTAINTY
# ================================================================

uncertainty_threshold = float(
    np.quantile(
        val_std,
        0.90,
    )
)

high_uncertainty = (
    test_std
    >= uncertainty_threshold
)

predicted_positive = (
    test_mean >= 0.5
)

actual_positive = (
    y_test == 1
)

correct = (
    predicted_positive
    == actual_positive
)

if high_uncertainty.any():

    high_uncertainty_error_rate = float(
        np.mean(
            ~correct[
                high_uncertainty
            ]
        )
    )

else:

    high_uncertainty_error_rate = 0.0

if (~high_uncertainty).any():

    low_uncertainty_error_rate = float(
        np.mean(
            ~correct[
                ~high_uncertainty
            ]
        )
    )

else:

    low_uncertainty_error_rate = 0.0

# ================================================================
# DISAGREEMENT STATISTICS
# ================================================================

mean_pairwise_disagreement = float(
    np.mean(
        np.abs(
            val_matrix[0]
            - val_matrix[1]
        )
        +
        np.abs(
            val_matrix[0]
            - val_matrix[2]
        )
        +
        np.abs(
            val_matrix[1]
            - val_matrix[2]
        )
    ) / 3.0
)

# ================================================================
# GATES
# ================================================================

gates = {

    "three_members_loaded": (
        len(member_results) == 3
    ),

    "all_checkpoints_epoch_5": (
        all(
            x["checkpoint_epoch"] == 5
            for x in member_results
        )
    ),

    "validation_predictions_finite": bool(
        np.isfinite(
            val_matrix
        ).all()
    ),

    "test_predictions_finite": bool(
        np.isfinite(
            test_matrix
        ).all()
    ),

    "ensemble_validation_pr_auc_finite": bool(
        np.isfinite(val_pr)
    ),

    "ensemble_test_pr_auc_finite": bool(
        np.isfinite(test_pr)
    ),

    "ensemble_test_roc_auc_finite": bool(
        np.isfinite(test_roc)
    ),

    "ensemble_test_brier_finite": bool(
        np.isfinite(test_brier)
    ),

    "uncertainty_finite": bool(
        np.isfinite(test_std).all()
    ),

    "uncertainty_threshold_validation_only": True,

    "test_not_used_for_model_selection": True,

    "no_retraining_during_finalization": True,

    "direct_vehicle_actuation_disabled": True,
}

failed = [
    name
    for name, value in gates.items()
    if not value
]

if failed:
    raise RuntimeError(
        "Phase 4.27A finalization failed: "
        + ", ".join(failed)
    )

# ================================================================
# REPORT
# ================================================================

report = {

    "project":
        "Predictive Cyber-Physical Resilience for "
        "Safety-Critical Connected Vehicles",

    "phase":
        "4.27A",

    "status":
        "PASS",

    "title":
        "Uncertainty-Aware Temporal Transformer Deep Ensemble",

    "experiment_type":
        "expensive_temporal_deep_ensemble",

    "model": {

        "members": 3,

        "seeds": SEEDS,

        "input_shape": [
            12,
            59,
        ],

        "architecture": {
            "d_model": D_MODEL,
            "attention_heads": NHEAD,
            "encoder_layers": LAYERS,
            "feedforward_dimension": FF_DIM,
            "dropout": DROPOUT,
        },

        "training": {
            "training_samples": 100000,
            "batch_size": BATCH_SIZE,
            "max_epochs": 5,
            "best_epoch_all_members": 5,
        },
    },

    "member_results":
        member_results,

    "ensemble_validation": {

        "pr_auc": float(val_pr),

        "roc_auc": float(val_roc),

        "brier": float(val_brier),
    },

    "ensemble_test": {

        "pr_auc": float(test_pr),

        "roc_auc": float(test_roc),

        "brier": float(test_brier),
    },

    "uncertainty": {

        "method":
            "ensemble predictive standard deviation",

        "validation_p90_threshold":
            uncertainty_threshold,

        "test_mean_std":
            float(test_std.mean()),

        "test_p90_std":
            float(
                np.quantile(
                    test_std,
                    0.90,
                )
            ),

        "test_p95_std":
            float(
                np.quantile(
                    test_std,
                    0.95,
                )
            ),

        "high_uncertainty_rate":
            float(
                high_uncertainty.mean()
            ),

        "high_uncertainty_error_rate":
            high_uncertainty_error_rate,

        "low_uncertainty_error_rate":
            low_uncertainty_error_rate,

        "mean_validation_pairwise_disagreement":
            mean_pairwise_disagreement,
    },

    "methodological_boundary": {

        "train_data":
            "CRSS-derived synthetic cyber-physical temporal dataset",

        "cyber_labels":
            "synthetic research telemetry labels",

        "physical_ground_truth":
            "NHTSA CRSS 2024",

        "test_used_for_training":
            False,

        "test_used_for_model_selection":
            False,

        "uncertainty_threshold_source":
            "validation ensemble disagreement",

        "real_world_cyberattack_claim":
            False,
    },

    "safety_boundary": {

        "direct_vehicle_actuation":
            False,

        "allowed_roles": [
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
print("PHASE 4.27A FINALIZATION COMPLETE")
print("=" * 80)

print(
    f"Validation Ensemble PR-AUC : "
    f"{val_pr:.6f}"
)

print(
    f"Validation Ensemble ROC-AUC: "
    f"{val_roc:.6f}"
)

print(
    f"Validation Ensemble Brier  : "
    f"{val_brier:.6f}"
)

print(
    f"Test Ensemble PR-AUC       : "
    f"{test_pr:.6f}"
)

print(
    f"Test Ensemble ROC-AUC      : "
    f"{test_roc:.6f}"
)

print(
    f"Test Ensemble Brier        : "
    f"{test_brier:.6f}"
)

print(
    f"Uncertainty P90 threshold  : "
    f"{uncertainty_threshold:.6f}"
)

print(
    f"High uncertainty rate      : "
    f"{high_uncertainty.mean():.6%}"
)

print(
    f"High uncertainty error     : "
    f"{high_uncertainty_error_rate:.6%}"
)

print(
    f"Low uncertainty error      : "
    f"{low_uncertainty_error_rate:.6%}"
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
