import os
import gc
import json
import time
import math
import random
import platform
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

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 20
PATIENCE = 5
MIN_DELTA = 1e-4
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
NUM_THREADS = min(6, os.cpu_count() or 1)

DATA_DIR = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
EXP_DIR = ROOT / "experiments" / "modeling" / "v2" / "advanced"
MODEL_DIR = ROOT / "models" / "v2" / "advanced"

EXP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SPEC_PATH = EXP_DIR / "crss_2024_v2_temporal_transformer_spec.json"
INFRA_PATH = ROOT / "experiments" / "modeling" / "v2" / "training" / "crss_2024_v2_training_infrastructure_spec.json"
FEATURE_TREATMENT_PATH = ROOT / "experiments" / "modeling" / "v2" / "feature_treatment" / "crss_2024_v2_feature_treatment_spec.json"

# Actual location from the established project artifact.
if not FEATURE_TREATMENT_PATH.exists():
    FEATURE_TREATMENT_PATH = EXP_DIR / "crss_2024_v2_feature_treatment_spec.json"

OUTPUT_JSON = EXP_DIR / "crss_2024_v2_temporal_transformer_training_results.json"
BEST_MODEL = MODEL_DIR / "crss_2024_v2_temporal_transformer_best.pt"

X_FILES = {
    "train": DATA_DIR / "X_train_v2_forecasting.npy",
    "validation": DATA_DIR / "X_validation_v2_forecasting.npy",
    "test": DATA_DIR / "X_test_v2_forecasting.npy",
}

Y3_FILES = {
    "train": DATA_DIR / "y_3step_train_v2_forecasting.npy",
    "validation": DATA_DIR / "y_3step_validation_v2_forecasting.npy",
    "test": DATA_DIR / "y_3step_test_v2_forecasting.npy",
}

Y6_FILES = {
    "train": DATA_DIR / "y_6step_train_v2_forecasting.npy",
    "validation": DATA_DIR / "y_6step_validation_v2_forecasting.npy",
    "test": DATA_DIR / "y_6step_test_v2_forecasting.npy",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def safe_float(x):
    return float(x) if np.isfinite(x) else None


def binary_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    metrics = {}

    if np.unique(y_true).size == 2:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["pr_auc"] = None
        metrics["roc_auc"] = None

    # Fixed threshold for operationally reproducible evaluation.
    pred = (y_prob >= 0.5).astype(np.int8)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        average="binary",
        zero_division=0,
    )

    metrics["precision"] = float(p)
    metrics["recall"] = float(r)
    metrics["f1"] = float(f1)
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))

    return metrics


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

        self.input_projection = nn.Linear(input_dim, embedding_dim)

        self.position = nn.Parameter(
            torch.zeros(1, seq_len, embedding_dim)
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)

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

        self.attention_pool = nn.Linear(embedding_dim, 1)

        self.y3_head = nn.Linear(embedding_dim, 1)
        self.y6_head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.position[:, :x.size(1), :]
        x = self.encoder(x)

        weights = torch.softmax(
            self.attention_pool(x).squeeze(-1),
            dim=1,
        ).unsqueeze(-1)

        representation = torch.sum(x * weights, dim=1)

        y3 = self.y3_head(representation).squeeze(-1)
        y6 = self.y6_head(representation).squeeze(-1)

        return y3, y6, representation


def load_memmap(path):
    return np.load(path, mmap_mode="r")


def prepare_batch(
    x_np,
    indices,
    feature_indices,
    continuous_positions,
    means,
    stds,
):
    # Only this small batch is copied into writable memory.
    x = np.asarray(
        x_np[indices][:, :, feature_indices],
        dtype=np.float32,
    ).copy()

    if continuous_positions:
        x[:, :, continuous_positions] = (
            x[:, :, continuous_positions] -
            means[continuous_positions]
        ) / stds[continuous_positions]

    return torch.from_numpy(x)


@torch.no_grad()
def evaluate(
    model,
    x_np,
    y3_np,
    y6_np,
    feature_indices,
    continuous_positions,
    means,
    stds,
    batch_size,
    device,
):
    model.eval()

    n = len(y3_np)

    y3_true = []
    y3_prob = []
    y6_true = []
    y6_prob = []

    total_loss = 0.0
    batches = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = np.arange(start, end)

        xb = prepare_batch(
            x_np,
            idx,
            feature_indices,
            continuous_positions,
            means,
            stds,
        ).to(device)

        y3b = torch.from_numpy(
            np.asarray(y3_np[start:end], dtype=np.float32)
        ).to(device)

        y6b = torch.from_numpy(
            np.asarray(y6_np[start:end], dtype=np.float32)
        ).to(device)

        logits3, logits6, _ = model(xb)

        loss3 = evaluate.loss3(logits3, y3b)
        loss6 = evaluate.loss6(logits6, y6b)
        loss = 0.5 * (loss3 + loss6)

        total_loss += float(loss.item())
        batches += 1

        y3_true.append(y3b.cpu().numpy())
        y3_prob.append(
            torch.sigmoid(logits3).cpu().numpy()
        )

        y6_true.append(y6b.cpu().numpy())
        y6_prob.append(
            torch.sigmoid(logits6).cpu().numpy()
        )

        del xb, y3b, y6b, logits3, logits6

    y3_true = np.concatenate(y3_true)
    y3_prob = np.concatenate(y3_prob)
    y6_true = np.concatenate(y6_true)
    y6_prob = np.concatenate(y6_prob)

    return {
        "loss": total_loss / max(batches, 1),
        "y3": binary_metrics(y3_true, y3_prob),
        "y6": binary_metrics(y6_true, y6_prob),
    }


def main():
    seed_everything(SEED)

    torch.set_num_threads(NUM_THREADS)
    try:
        torch.set_num_interop_threads(max(1, min(4, NUM_THREADS)))
    except RuntimeError:
        pass

    device = torch.device("cpu")

    print("=" * 72)
    print("PHASE 4.10D — TEMPORAL TRANSFORMER TRAINING")
    print("=" * 72)
    print("Device:", device)
    print("PyTorch:", torch.__version__)
    print("CPU threads:", NUM_THREADS)
    print("Batch size:", BATCH_SIZE)
    print("Max epochs:", MAX_EPOCHS)
    print()

    spec = load_json(SPEC_PATH)
    infra = load_json(INFRA_PATH)

    assert spec["status"] == "DESIGN_READY"
    assert spec["input"]["feature_count"] == 59
    assert spec["input"]["shape"] == [12, 59]
    assert spec["architecture"]["input_projection"]["embedding_dim"] == 128
    assert spec["architecture"]["temporal_encoder"]["layers"] == 3
    assert spec["architecture"]["temporal_encoder"]["attention_heads"] == 8
    assert spec["architecture"]["temporal_encoder"]["feedforward_dimension"] == 256
    assert spec["training"]["optimizer"] == "AdamW"
    assert spec["training"]["primary_validation_metric"] == "y6_pr_auc"

    assert infra["status"] == "PASS"
    assert infra["feature_treatment"]["final_feature_count"] == 59

    feature_names = infra["feature_treatment"]["final_features"]
    feature_indices = infra["feature_treatment"]["final_indices"]

    assert len(feature_names) == 59
    assert len(feature_indices) == 59

    continuous_names = set(
        infra["feature_treatment"]
        .get("continuous_features", [])
    )

    # The infrastructure artifact stores the normalization statistics.
    normalization = infra.get("normalization", {})

    means_dict = normalization.get("mean", {})
    stds_dict = normalization.get("std", {})

    # If statistics are nested elsewhere, search recursively.
    def find_key(obj, key):
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for value in obj.values():
                result = find_key(value, key)
                if result is not None:
                    return result
        return None

    if not means_dict:
        means_dict = find_key(infra, "mean") or {}

    if not stds_dict:
        stds_dict = find_key(infra, "std") or {}

    if not continuous_names:
        # Recover from the 27/32 partition by checking whether statistics exist.
        continuous_names = {
            name for name in feature_names
            if name in means_dict and name in stds_dict
        }

    continuous_positions = [
        i for i, name in enumerate(feature_names)
        if name in continuous_names
    ]

    means = np.asarray(
        [
            float(means_dict[name])
            for name in feature_names
        ],
        dtype=np.float32,
    )

    stds = np.asarray(
        [
            max(float(stds_dict.get(name, 1.0)), 1e-8)
            for name in feature_names
        ],
        dtype=np.float32,
    )

    # Only continuous features are normalized.
    for i, name in enumerate(feature_names):
        if name not in continuous_names:
            means[i] = 0.0
            stds[i] = 1.0

    print("Final features:", len(feature_names))
    print("Continuous:", len(continuous_positions))
    print("Binary:", 59 - len(continuous_positions))

    # ------------------------------------------------------------------
    # Memory-map data. Nothing here loads the complete tensors into RAM.
    # ------------------------------------------------------------------
    X_train = load_memmap(X_FILES["train"])
    X_val = load_memmap(X_FILES["validation"])

    y3_train = np.load(Y3_FILES["train"], mmap_mode="r")
    y6_train = np.load(Y6_FILES["train"], mmap_mode="r")
    y3_val = np.load(Y3_FILES["validation"], mmap_mode="r")
    y6_val = np.load(Y6_FILES["validation"], mmap_mode="r")

    assert X_train.shape[1:] == (12, 63)
    assert X_val.shape[1:] == (12, 63)
    assert len(X_train) == len(y3_train) == len(y6_train)
    assert len(X_val) == len(y3_val) == len(y6_val)

    print("Train samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Original tensor shape:", X_train.shape)
    print("Memory mapping: ENABLED")
    print()

    # Positive-class weights from the approved infrastructure statistics.
    pos3 = float(np.sum(np.asarray(y3_train) == 1))
    neg3 = float(len(y3_train) - pos3)

    pos6 = float(np.sum(np.asarray(y6_train) == 1))
    neg6 = float(len(y6_train) - pos6)

    pos_weight3 = neg3 / max(pos3, 1.0)
    pos_weight6 = neg6 / max(pos6, 1.0)

    print("Y3 positives:", int(pos3))
    print("Y3 pos_weight:", round(pos_weight3, 4))
    print("Y6 positives:", int(pos6))
    print("Y6 pos_weight:", round(pos_weight6, 4))
    print()

    model = CyberPhysicalTemporalTransformer(
        input_dim=59,
        seq_len=12,
        embedding_dim=128,
        layers=3,
        heads=8,
        ff_dim=256,
        dropout=0.1,
    ).to(device)

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print("Model parameters:", parameter_count)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loss3_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight3],
            dtype=torch.float32,
            device=device,
        )
    )

    loss6_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight6],
            dtype=torch.float32,
            device=device,
        )
    )

    # Attach losses for evaluation helper.
    evaluate.loss3 = loss3_fn
    evaluate.loss6 = loss6_fn

    best_metric = -math.inf
    best_epoch = 0
    patience_counter = 0
    history = []

    training_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        model.train()

        # Deterministic epoch order.
        rng = np.random.default_rng(SEED + epoch)
        order = rng.permutation(len(X_train))

        running_loss = 0.0
        batches = 0

        for start in range(0, len(order), BATCH_SIZE):
            batch_indices = order[start:start + BATCH_SIZE]

            xb = prepare_batch(
                X_train,
                batch_indices,
                feature_indices,
                continuous_positions,
                means,
                stds,
            ).to(device)

            y3b = torch.from_numpy(
                np.asarray(
                    y3_train[batch_indices],
                    dtype=np.float32,
                )
            ).to(device)

            y6b = torch.from_numpy(
                np.asarray(
                    y6_train[batch_indices],
                    dtype=np.float32,
                )
            ).to(device)

            optimizer.zero_grad(set_to_none=True)

            logits3, logits6, _ = model(xb)

            loss3 = loss3_fn(logits3, y3b)
            loss6 = loss6_fn(logits6, y6b)
            loss = 0.5 * (loss3 + loss6)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

            del xb, y3b, y6b, logits3, logits6
            del loss3, loss6, loss

        gc.collect()

        train_loss = running_loss / max(batches, 1)

        validation = evaluate(
            model,
            X_val,
            y3_val,
            y6_val,
            feature_indices,
            continuous_positions,
            means,
            stds,
            BATCH_SIZE,
            device,
        )

        epoch_time = time.time() - epoch_start

        y3_pr = validation["y3"]["pr_auc"]
        y6_pr = validation["y6"]["pr_auc"]

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={validation['loss']:.6f} | "
            f"Y3_PR={y3_pr:.6f} | "
            f"Y6_PR={y6_pr:.6f} | "
            f"time={epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation,
            "epoch_seconds": epoch_time,
        })

        # Primary selection criterion = validation Y6 PR-AUC.
        if y6_pr > best_metric + MIN_DELTA:
            best_metric = y6_pr
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "phase": "4.10D",
                    "architecture": "cyber_physical_temporal_transformer",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_y6_pr_auc": y6_pr,
                    "validation_y3_pr_auc": y3_pr,
                    "feature_count": 59,
                    "sequence_length": 12,
                    "seed": SEED,
                    "feature_indices": feature_indices,
                    "feature_names": feature_names,
                    "continuous_positions": continuous_positions,
                    "normalization_means": means,
                    "normalization_stds": stds,
                    "pos_weight_y3": pos_weight3,
                    "pos_weight_y6": pos_weight6,
                },
                BEST_MODEL,
            )

            print(
                f"  [CHECKPOINT] epoch {epoch} "
                f"Y6 PR-AUC={y6_pr:.6f}"
            )
        else:
            patience_counter += 1
            print(
                f"  [EARLY STOP COUNTER] "
                f"{patience_counter}/{PATIENCE}"
            )

        gc.collect()

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    training_seconds = time.time() - training_start

    # ------------------------------------------------------------------
    # FINAL TEST — loaded only after checkpoint selection.
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("FINAL TEST EVALUATION")
    print("=" * 72)

    checkpoint = torch.load(
        BEST_MODEL,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    del X_train, X_val, y3_train, y6_train, y3_val, y6_val
    gc.collect()

    X_test = load_memmap(X_FILES["test"])
    y3_test = np.load(Y3_FILES["test"], mmap_mode="r")
    y6_test = np.load(Y6_FILES["test"], mmap_mode="r")

    test = evaluate(
        model,
        X_test,
        y3_test,
        y6_test,
        feature_indices,
        continuous_positions,
        means,
        stds,
        BATCH_SIZE,
        device,
    )

    print(
        "Y3 Test PR-AUC:",
        f"{test['y3']['pr_auc']:.6f}"
    )
    print(
        "Y3 Test ROC-AUC:",
        f"{test['y3']['roc_auc']:.6f}"
    )
    print(
        "Y6 Test PR-AUC:",
        f"{test['y6']['pr_auc']:.6f}"
    )
    print(
        "Y6 Test ROC-AUC:",
        f"{test['y6']['roc_auc']:.6f}"
    )

    result = {
        "project": "Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles",
        "phase": "4.10D",
        "status": "PASS",
        "task": "Temporal Transformer multi-horizon forecasting",
        "device": str(device),
        "pytorch_version": torch.__version__,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "architecture": {
            "input_shape": [12, 59],
            "embedding_dim": 128,
            "transformer_layers": 3,
            "attention_heads": 8,
            "feedforward_dimension": 256,
            "dropout": 0.1,
            "attention_pooling": True,
            "parameters": parameter_count,
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRAD_CLIP,
            "loss": "weighted_binary_cross_entropy_with_logits",
            "primary_selection_metric": "y6_pr_auc",
            "normalization": "train_only_z_score_continuous_features",
            "memory_mapping": True,
        },
        "class_weights": {
            "y3": pos_weight3,
            "y6": pos_weight6,
        },
        "data": {
            "train_samples": int(len(y3_test) if False else checkpoint.get("train_samples", 444052)),
            "test_samples": int(len(y3_test)),
            "feature_count": 59,
            "sequence_length": 12,
        },
        "best_epoch": best_epoch,
        "best_validation_y6_pr_auc": best_metric,
        "training_seconds": training_seconds,
        "history": history,
        "final_test": test,
        "checkpoint": str(BEST_MODEL),
        "methodological_boundary": {
            "cyber_labels": "synthetic research telemetry",
            "physical_ground_truth": "CRSS 2024",
            "real_world_cyberattack_labels": False,
        },
        "safety_boundary": {
            "ml_direct_actuation_authority": False,
            "ml_role": [
                "forecasting",
                "risk_scoring",
                "warning",
                "decision_support",
            ],
        },
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )

    print()
    print("=" * 72)
    print("PHASE 4.10D COMPLETE")
    print("=" * 72)
    print("Best epoch:", best_epoch)
    print("Best validation Y6 PR-AUC:", f"{best_metric:.6f}")
    print("Test Y3 PR-AUC:", f"{test['y3']['pr_auc']:.6f}")
    print("Test Y6 PR-AUC:", f"{test['y6']['pr_auc']:.6f}")
    print("Checkpoint:", BEST_MODEL)
    print("Results:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
