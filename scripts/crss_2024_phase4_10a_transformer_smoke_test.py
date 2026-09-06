from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]

SPEC = ROOT / "experiments/modeling/v2/advanced/crss_2024_v2_temporal_transformer_spec.json"
FEATURE_SPEC = ROOT / "experiments/modeling/v2/crss_2024_v2_feature_treatment_spec.json"

TENSOR_DIR = ROOT / "data/processed/modeling/sequences/v2_forecasting"

OUT = ROOT / "experiments/modeling/v2/advanced/crss_2024_v2_transformer_smoke_test.json"

SEED = 42
SMOKE_SAMPLES = 2048

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(10)

DEVICE = torch.device("cpu")

checks = []


def check(condition, name, detail=""):
    status = "PASS" if condition else "FAIL"

    print(
        f"[{status}] {name}"
        + (f"\n       {detail}" if detail else "")
    )

    checks.append({
        "name": name,
        "status": status,
        "detail": detail,
    })


# =====================================================================
# Load specifications
# =====================================================================

with open(SPEC, encoding="utf-8") as f:
    spec = json.load(f)

with open(FEATURE_SPEC, encoding="utf-8") as f:
    feature_spec = json.load(f)

check(
    spec["status"] == "DESIGN_READY",
    "architecture_spec",
    spec["status"],
)

final_indices = np.asarray(
    feature_spec["final_feature_indices_in_original_tensor"],
    dtype=np.int64,
)

check(
    len(final_indices) == 59,
    "final_feature_count",
    f"count={len(final_indices)}",
)


# =====================================================================
# Load smoke-test subset
# =====================================================================

X = np.load(
    TENSOR_DIR / "X_train_v2_forecasting.npy",
    mmap_mode="r",
)

y3 = np.load(
    TENSOR_DIR / "y_3step_train_v2_forecasting.npy",
    mmap_mode="r",
)

y6 = np.load(
    TENSOR_DIR / "y_6step_train_v2_forecasting.npy",
    mmap_mode="r",
)

X = np.asarray(
    X[:SMOKE_SAMPLES, :, final_indices],
    dtype=np.float32,
)

y3 = np.asarray(
    y3[:SMOKE_SAMPLES],
    dtype=np.float32,
)

y6 = np.asarray(
    y6[:SMOKE_SAMPLES],
    dtype=np.float32,
)

check(
    X.shape == (SMOKE_SAMPLES, 12, 59),
    "smoke_input_shape",
    str(X.shape),
)

check(
    np.isfinite(X).all(),
    "smoke_input_finite",
)


# =====================================================================
# Train-only normalization from Phase 4.7
# =====================================================================

continuous_features = feature_spec["continuous_features"]

means = np.asarray([
    feature_spec["normalization"]["mean"][f]
    for f in continuous_features
], dtype=np.float32)

stds = np.asarray([
    feature_spec["normalization"]["std"][f]
    for f in continuous_features
], dtype=np.float32)

continuous_indices = [
    feature_spec["final_features"].index(f)
    for f in continuous_features
]

X_norm = X.copy()

X_norm[:, :, continuous_indices] = (
    X_norm[:, :, continuous_indices] - means
) / stds

check(
    np.isfinite(X_norm).all(),
    "train_only_normalization",
)


# =====================================================================
# Model
# =====================================================================

class CyberPhysicalTemporalTransformer(nn.Module):

    def __init__(
        self,
        input_dim=59,
        embedding_dim=128,
        heads=8,
        layers=3,
        feedforward_dim=256,
        dropout=0.10,
        max_seq_len=12,
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            embedding_dim,
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_seq_len,
                embedding_dim,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
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

        self.y3_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1),
        )

        self.y6_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = x + self.position_embedding[
            :, :x.size(1), :
        ]

        x = self.encoder(x)

        attention_scores = self.attention_pool(x)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        representation = (
            x * attention_weights
        ).sum(dim=1)

        y3 = self.y3_head(
            representation
        ).squeeze(-1)

        y6 = self.y6_head(
            representation
        ).squeeze(-1)

        return y3, y6, representation


model = CyberPhysicalTemporalTransformer().to(DEVICE)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

check(
    parameter_count > 0,
    "model_parameter_count",
    f"parameters={parameter_count:,}",
)


# =====================================================================
# Forward pass
# =====================================================================

X_tensor = torch.from_numpy(X_norm)

y3_tensor = torch.from_numpy(y3)
y6_tensor = torch.from_numpy(y6)

start = time.time()

y3_logits, y6_logits, representation = model(
    X_tensor
)

forward_seconds = time.time() - start

check(
    y3_logits.shape == (SMOKE_SAMPLES,),
    "y3_output_shape",
    str(tuple(y3_logits.shape)),
)

check(
    y6_logits.shape == (SMOKE_SAMPLES,),
    "y6_output_shape",
    str(tuple(y6_logits.shape)),
)

check(
    representation.shape == (SMOKE_SAMPLES, 128),
    "temporal_representation_shape",
    str(tuple(representation.shape)),
)

check(
    torch.isfinite(y3_logits).all()
    and torch.isfinite(y6_logits).all(),
    "forward_outputs_finite",
)


# =====================================================================
# Weighted multi-horizon loss
# =====================================================================

pos_weight_y3 = torch.tensor(116.69)
pos_weight_y6 = torch.tensor(105.28)

loss_y3 = nn.functional.binary_cross_entropy_with_logits(
    y3_logits,
    y3_tensor,
    pos_weight=pos_weight_y3,
)

loss_y6 = nn.functional.binary_cross_entropy_with_logits(
    y6_logits,
    y6_tensor,
    pos_weight=pos_weight_y6,
)

loss = loss_y3 + loss_y6

check(
    torch.isfinite(loss),
    "weighted_multi_horizon_loss",
    f"loss={float(loss):.6f}",
)


# =====================================================================
# Backpropagation
# =====================================================================

model.zero_grad(set_to_none=True)

loss.backward()

gradient_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)

check(
    torch.isfinite(torch.as_tensor(gradient_norm)),
    "gradient_backpropagation",
    f"gradient_norm={float(gradient_norm):.6f}",
)


# =====================================================================
# Optimizer
# =====================================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0003,
    weight_decay=0.0001,
)

optimizer.step()

check(
    True,
    "adamw_optimizer_step",
)


# =====================================================================
# Probability sanity checks
# =====================================================================

with torch.no_grad():

    p3 = torch.sigmoid(y3_logits).numpy()
    p6 = torch.sigmoid(y6_logits).numpy()

check(
    np.isfinite(p3).all()
    and np.isfinite(p6).all(),
    "probability_outputs_finite",
)

check(
    np.all((p3 >= 0) & (p3 <= 1))
    and np.all((p6 >= 0) & (p6 <= 1)),
    "probability_range",
)


# =====================================================================
# Final result
# =====================================================================

result = {
    "project": "Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles",

    "phase": "4.10A",

    "name": "temporal_transformer_cpu_smoke_test",

    "status": (
        "PASS"
        if all(c["status"] == "PASS" for c in checks)
        else "FAIL"
    ),

    "device": str(DEVICE),

    "seed": SEED,

    "sample_count": SMOKE_SAMPLES,

    "input_shape": [
        SMOKE_SAMPLES,
        12,
        59,
    ],

    "architecture": {
        "embedding_dim": 128,
        "attention_heads": 8,
        "transformer_layers": 3,
        "feedforward_dim": 256,
        "dropout": 0.10,
        "attention_pooling": True,
        "dual_horizon_heads": True,
    },

    "parameters": {
        "total": parameter_count,
        "trainable": trainable_parameter_count,
    },

    "loss": {
        "type": "weighted_binary_cross_entropy_with_logits",
        "y3_positive_weight": 116.69,
        "y6_positive_weight": 105.28,
        "combined_loss": float(loss),
    },

    "forward_seconds": forward_seconds,

    "checks": checks,
}


OUT.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 100)
print("CRSS 2024 — PHASE 4.10A TRANSFORMER CPU SMOKE TEST")
print("=" * 100)

print()
print(f"Device              : {DEVICE}")
print(f"Samples             : {SMOKE_SAMPLES}")
print("Input               : 12 × 59")
print("Embedding           : 128")
print("Attention heads     : 8")
print("Transformer layers  : 3")
print(f"Parameters          : {parameter_count:,}")
print(f"Forward time        : {forward_seconds:.3f} sec")
print(f"Combined loss       : {float(loss):.6f}")

print()
print("=" * 100)
print(f"STATUS : {result['status']}")
print(f"RESULT : {OUT}")
print("=" * 100)

if result["status"] != "PASS":
    raise SystemExit(
        "PHASE 4.10A TRANSFORMER SMOKE TEST FAILED"
    )
