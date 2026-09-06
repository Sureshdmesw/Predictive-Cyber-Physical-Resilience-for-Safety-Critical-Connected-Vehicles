import json
import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    brier_score_loss,
)

ROOT = Path.cwd()

DATA = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
MODEL = ROOT / "models" / "v2" / "advanced" / "crss_2024_v2_temporal_transformer_controlled_best.pt"
FEATURE_SPEC = ROOT / "experiments" / "modeling" / "v2" / "crss_2024_v2_feature_treatment_spec.json"

OUT = ROOT / "experiments" / "modeling" / "v2" / "advanced" / "crss_2024_v2_temporal_transformer_controlled_final.json"


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


class CyberPhysicalTemporalTransformer(nn.Module):

    def __init__(
        self,
        input_dim=59,
        seq_len=12,
        embedding_dim=128,
        layers=3,
        heads=8,
        ff_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            embedding_dim,
        )

        self.input_norm = nn.LayerNorm(
            embedding_dim
        )

        self.position = nn.Parameter(
            torch.zeros(
                1,
                seq_len,
                embedding_dim,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.attention_pool = nn.Linear(
            embedding_dim,
            1,
        )

        self.pool_norm = nn.LayerNorm(
            embedding_dim
        )

        self.output_norm = nn.LayerNorm(
            embedding_dim
        )

        self.y3_head = nn.Linear(
            embedding_dim,
            1,
        )

        self.y6_head = nn.Linear(
            embedding_dim,
            1,
        )

    def forward(self, x):

        x = self.input_projection(x)
        x = self.input_norm(x)

        x = x + self.position[
            :, :x.size(1), :
        ]

        x = self.encoder(x)

        scores = self.attention_pool(
            x
        ).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1,
        ).unsqueeze(-1)

        pooled = torch.sum(
            x * weights,
            dim=1,
        )

        pooled = self.pool_norm(
            pooled
        )

        pooled = self.output_norm(
            pooled
        )

        y3 = self.y3_head(
            pooled
        ).squeeze(-1)

        y6 = self.y6_head(
            pooled
        ).squeeze(-1)

        return y3, y6


def prepare(
    X,
    indices,
    feature_indices,
    continuous_positions,
    means,
    stds,
):
    x = np.asarray(
        X[indices],
        dtype=np.float32,
    )

    x = x[:, :, feature_indices].copy()

    if continuous_positions:
        x[:, :, continuous_positions] = (
            x[:, :, continuous_positions]
            - means[continuous_positions]
        ) / stds[continuous_positions]

    return torch.from_numpy(x)


def evaluate(
    model,
    X,
    y3,
    y6,
    feature_indices,
    continuous_positions,
    means,
    stds,
    batch_size=128,
):

    model.eval()

    p3 = []
    p6 = []
    t3 = []
    t6 = []

    with torch.no_grad():

        for start in range(
            0,
            len(y3),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(y3),
            )

            idx = np.arange(
                start,
                end,
            )

            xb = prepare(
                X,
                idx,
                feature_indices,
                continuous_positions,
                means,
                stds,
            )

            logits3, logits6 = model(xb)

            p3.append(
                torch.sigmoid(
                    logits3
                ).numpy()
            )

            p6.append(
                torch.sigmoid(
                    logits6
                ).numpy()
            )

            t3.append(
                np.asarray(
                    y3[start:end]
                )
            )

            t6.append(
                np.asarray(
                    y6[start:end]
                )
            )

            del xb, logits3, logits6

    p3 = np.concatenate(p3)
    p6 = np.concatenate(p6)
    t3 = np.concatenate(t3).astype(np.int8)
    t6 = np.concatenate(t6).astype(np.int8)

    def calc(y, p):

        pred = (
            p >= 0.5
        ).astype(np.int8)

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y,
                pred,
                average="binary",
                zero_division=0,
            )
        )

        return {
            "pr_auc": float(
                average_precision_score(
                    y,
                    p,
                )
            ),
            "roc_auc": float(
                roc_auc_score(
                    y,
                    p,
                )
            ),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(
                f1
            ),
            "brier": float(
                brier_score_loss(
                    y,
                    p,
                )
            ),
            "positive_count": int(
                np.sum(y == 1)
            ),
            "sample_count": int(
                len(y)
            ),
        }

    return {
        "y3": calc(t3, p3),
        "y6": calc(t6, p6),
    }


print("=" * 72)
print("PHASE 4.10D — FINAL CONTROLLED TEST EVALUATION")
print("=" * 72)

if not MODEL.exists():
    raise FileNotFoundError(
        f"Checkpoint missing: {MODEL}"
    )

feature_spec = load_json(
    FEATURE_SPEC
)

assert feature_spec["status"] == "PASS"
assert feature_spec["original_feature_count"] == 63
assert feature_spec["final_feature_count"] == 59

feature_names = list(
    feature_spec["final_features"]
)

feature_indices = list(
    feature_spec[
        "final_feature_indices_in_original_tensor"
    ]
)

binary_names = set(
    feature_spec["binary_features"]
)

continuous_names = set(
    feature_spec["continuous_features"]
)

assert len(feature_names) == 59
assert len(feature_indices) == 59
assert len(binary_names) == 27
assert len(continuous_names) == 32

continuous_positions = [
    i
    for i, name in enumerate(feature_names)
    if name in continuous_names
]

assert len(continuous_positions) == 32

checkpoint = torch.load(
    MODEL,
    map_location="cpu",
    weights_only=False,
)

means = np.asarray(
    checkpoint["means"],
    dtype=np.float32,
)

stds = np.asarray(
    checkpoint["stds"],
    dtype=np.float32,
)

assert len(means) == 59
assert len(stds) == 59

model = CyberPhysicalTemporalTransformer()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Checkpoint epoch:", checkpoint["epoch"])
print("Model parameters:",
      sum(p.numel() for p in model.parameters()))
print("Features:", len(feature_names))
print("Binary:", len(binary_names))
print("Continuous:", len(continuous_names))
print()

X_test = np.load(
    DATA / "X_test_v2_forecasting.npy",
    mmap_mode="r",
)

y3_test = np.load(
    DATA / "y_3step_test_v2_forecasting.npy",
    mmap_mode="r",
)

y6_test = np.load(
    DATA / "y_6step_test_v2_forecasting.npy",
    mmap_mode="r",
)

assert X_test.shape[1:] == (12, 63)
assert len(X_test) == len(y3_test)
assert len(X_test) == len(y6_test)

print("Test samples:", len(X_test))
print("Memory mapping: ENABLED")
print()

start = time.time()

results = evaluate(
    model,
    X_test,
    y3_test,
    y6_test,
    feature_indices,
    continuous_positions,
    means,
    stds,
    batch_size=128,
)

seconds = time.time() - start

print("=" * 72)
print("TEST RESULTS")
print("=" * 72)

for horizon in ["y3", "y6"]:

    r = results[horizon]

    print()
    print(horizon.upper())
    print("PR-AUC :", f"{r['pr_auc']:.6f}")
    print("ROC-AUC:", f"{r['roc_auc']:.6f}")
    print("F1     :", f"{r['f1']:.6f}")
    print("Prec.  :", f"{r['precision']:.6f}")
    print("Recall :", f"{r['recall']:.6f}")
    print("Brier  :", f"{r['brier']:.6f}")

final = {
    "project":
        "Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles",

    "phase":
        "4.10D",

    "status":
        "PASS",

    "experiment_type":
        "controlled_cpu_temporal_transformer",

    "checkpoint_epoch":
        int(checkpoint["epoch"]),

    "checkpoint":
        str(MODEL),

    "test_evaluation_seconds":
        seconds,

    "architecture":
        {
            "input_shape":
                [12, 59],
            "embedding_dim":
                128,
            "transformer_layers":
                3,
            "attention_heads":
                8,
            "feedforward_dimension":
                256,
            "dropout":
                0.1,
            "parameters":
                sum(
                    p.numel()
                    for p in model.parameters()
                ),
        },

    "feature_treatment":
        {
            "original":
                63,
            "final":
                59,
            "binary":
                27,
            "continuous":
                32,
            "normalization":
                "train_only_z_score_continuous_features",
        },

    "test":
        results,

    "methodological_boundary":
        {
            "cyber_labels":
                "synthetic research telemetry",
            "physical_ground_truth":
                "CRSS 2024",
            "real_world_cyberattack_labels":
                False,
            "controlled_training_subset":
                100000,
        },

    "safety_boundary":
        {
            "direct_actuation":
                False,
            "ml_role":
                [
                    "forecasting",
                    "risk scoring",
                    "warning",
                    "decision support",
                ],
        },
}

with open(
    OUT,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        final,
        f,
        indent=2,
    )

print()
print("=" * 72)
print("PHASE 4.10D FINALIZED")
print("=" * 72)
print("Status: PASS")
print("Selected checkpoint epoch:", checkpoint["epoch"])
print("Results:", OUT)
