import os
import gc
import json
import time
import random
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
TRAIN_SUBSET = 100_000
BATCH_SIZE = 128
MAX_EPOCHS = 5
PATIENCE = 2
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
NUM_THREADS = min(6, os.cpu_count() or 1)

DATA_DIR = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
EXP_DIR = ROOT / "experiments" / "modeling" / "v2" / "advanced"
MODEL_DIR = ROOT / "models" / "v2" / "advanced"

EXP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SPEC = ROOT / "experiments" / "modeling" / "v2" / "crss_2024_v2_feature_treatment_spec.json"
INFRA_SPEC = ROOT / "experiments" / "modeling" / "v2" / "training" / "crss_2024_v2_training_infrastructure_spec.json"

RESULT_PATH = EXP_DIR / "crss_2024_v2_temporal_transformer_controlled_results.json"
CHECKPOINT_PATH = MODEL_DIR / "crss_2024_v2_temporal_transformer_controlled_best.pt"

X_TRAIN = DATA_DIR / "X_train_v2_forecasting.npy"
X_VAL = DATA_DIR / "X_validation_v2_forecasting.npy"
X_TEST = DATA_DIR / "X_test_v2_forecasting.npy"

Y3_TRAIN = DATA_DIR / "y_3step_train_v2_forecasting.npy"
Y3_VAL = DATA_DIR / "y_3step_validation_v2_forecasting.npy"
Y3_TEST = DATA_DIR / "y_3step_test_v2_forecasting.npy"

Y6_TRAIN = DATA_DIR / "y_6step_train_v2_forecasting.npy"
Y6_VAL = DATA_DIR / "y_6step_validation_v2_forecasting.npy"
Y6_TEST = DATA_DIR / "y_6step_test_v2_forecasting.npy"


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics(y, p):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)

    out = {}

    if len(np.unique(y)) == 2:
        out["pr_auc"] = float(average_precision_score(y, p))
        out["roc_auc"] = float(roc_auc_score(y, p))
    else:
        out["pr_auc"] = None
        out["roc_auc"] = None

    pred = (p >= 0.5).astype(np.int8)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        average="binary",
        zero_division=0,
    )

    out["precision"] = float(precision)
    out["recall"] = float(recall)
    out["f1"] = float(f1)
    out["brier"] = float(brier_score_loss(y, p))

    return out


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

        nn.init.normal_(
            self.position,
            mean=0.0,
            std=0.02,
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


def prepare_batch(
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


@torch.no_grad()
def evaluate(
    model,
    X,
    y3,
    y6,
    feature_indices,
    continuous_positions,
    means,
    stds,
    batch_size,
    device,
    loss3_fn,
    loss6_fn,
):

    model.eval()

    y3_true = []
    y3_prob = []
    y6_true = []
    y6_prob = []

    total_loss = 0.0
    count = 0

    for start in range(
        0,
        len(y3),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(y3),
        )

        indices = np.arange(
            start,
            end,
        )

        xb = prepare_batch(
            X,
            indices,
            feature_indices,
            continuous_positions,
            means,
            stds,
        ).to(device)

        y3b = torch.from_numpy(
            np.asarray(
                y3[start:end],
                dtype=np.float32,
            )
        ).to(device)

        y6b = torch.from_numpy(
            np.asarray(
                y6[start:end],
                dtype=np.float32,
            )
        ).to(device)

        logits3, logits6 = model(xb)

        loss3 = loss3_fn(
            logits3,
            y3b,
        )

        loss6 = loss6_fn(
            logits6,
            y6b,
        )

        loss = 0.5 * (
            loss3 + loss6
        )

        total_loss += (
            float(loss.item())
            * len(indices)
        )

        count += len(indices)

        y3_true.append(
            y3b.cpu().numpy()
        )

        y3_prob.append(
            torch.sigmoid(
                logits3
            ).cpu().numpy()
        )

        y6_true.append(
            y6b.cpu().numpy()
        )

        y6_prob.append(
            torch.sigmoid(
                logits6
            ).cpu().numpy()
        )

        del (
            xb,
            y3b,
            y6b,
            logits3,
            logits6,
            loss3,
            loss6,
            loss,
        )

    return {
        "loss": total_loss / max(count, 1),
        "y3": metrics(
            np.concatenate(y3_true),
            np.concatenate(y3_prob),
        ),
        "y6": metrics(
            np.concatenate(y6_true),
            np.concatenate(y6_prob),
        ),
    }


def main():

    seed_all(SEED)

    torch.set_num_threads(
        NUM_THREADS
    )

    try:
        torch.set_num_interop_threads(
            min(4, NUM_THREADS)
        )
    except RuntimeError:
        pass

    device = torch.device("cpu")

    print("=" * 72)
    print("PHASE 4.10D — CONTROLLED TEMPORAL TRANSFORMER")
    print("=" * 72)
    print("Device:", device)
    print("PyTorch:", torch.__version__)
    print("CPU threads:", NUM_THREADS)
    print("Train subset:", TRAIN_SUBSET)
    print("Batch size:", BATCH_SIZE)
    print("Max epochs:", MAX_EPOCHS)
    print()

    feature_spec = load_json(
        FEATURE_SPEC
    )

    infra = load_json(
        INFRA_SPEC
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
    assert binary_names.isdisjoint(
        continuous_names
    )

    continuous_positions = [
        i
        for i, name in enumerate(feature_names)
        if name in continuous_names
    ]

    binary_positions = [
        i
        for i, name in enumerate(feature_names)
        if name in binary_names
    ]

    assert len(continuous_positions) == 32
    assert len(binary_positions) == 27

    normalization = infra.get(
        "normalization",
        {}
    )

    means_dict = normalization.get(
        "mean",
        {}
    )

    stds_dict = normalization.get(
        "std",
        {}
    )

    if not means_dict:
        means_dict = {}

        for item in normalization.get(
            "statistics",
            []
        ):
            if isinstance(item, dict):
                name = item.get(
                    "feature"
                )

                if name:
                    means_dict[name] = item.get(
                        "mean"
                    )

    if not stds_dict:
        stds_dict = {}

        for item in normalization.get(
            "statistics",
            []
        ):
            if isinstance(item, dict):
                name = item.get(
                    "feature"
                )

                if name:
                    stds_dict[name] = item.get(
                        "std"
                    )

    # Fall back to feature-treatment normalization
    # if infrastructure uses a different nesting.
    if not means_dict:

        ft_norm = feature_spec.get(
            "normalization",
            {}
        )

        means_dict = ft_norm.get(
            "mean",
            {}
        )

        stds_dict = ft_norm.get(
            "std",
            {}
        )

    assert means_dict
    assert stds_dict

    means = np.zeros(
        59,
        dtype=np.float32,
    )

    stds = np.ones(
        59,
        dtype=np.float32,
    )

    for i, name in enumerate(
        feature_names
    ):

        if name in continuous_names:

            means[i] = float(
                means_dict[name]
            )

            stds[i] = max(
                float(stds_dict[name]),
                1e-8,
            )

    print(
        "Final features:",
        len(feature_names),
    )

    print(
        "Binary:",
        len(binary_positions),
    )

    print(
        "Continuous:",
        len(continuous_positions),
    )

    assert len(binary_positions) == 27
    assert len(continuous_positions) == 32

    # ---------------------------------------------------------------
    # Memory mapped arrays
    # ---------------------------------------------------------------

    Xtr = np.load(
        X_TRAIN,
        mmap_mode="r",
    )

    Xv = np.load(
        X_VAL,
        mmap_mode="r",
    )

    y3tr = np.load(
        Y3_TRAIN,
        mmap_mode="r",
    )

    y6tr = np.load(
        Y6_TRAIN,
        mmap_mode="r",
    )

    y3v = np.load(
        Y3_VAL,
        mmap_mode="r",
    )

    y6v = np.load(
        Y6_VAL,
        mmap_mode="r",
    )

    assert Xtr.shape[1:] == (
        12,
        63,
    )

    assert Xv.shape[1:] == (
        12,
        63,
    )

    assert len(Xtr) == len(y3tr)
    assert len(Xtr) == len(y6tr)

    assert len(Xv) == len(y3v)
    assert len(Xv) == len(y6v)

    print(
        "Full training samples:",
        len(Xtr),
    )

    print(
        "Validation samples:",
        len(Xv),
    )

    # ---------------------------------------------------------------
    # Deterministic stratified training subset.
    # Preserve all positive examples and sample negatives.
    # ---------------------------------------------------------------

    rng = np.random.default_rng(
        SEED
    )

    y3_train_np = np.asarray(
        y3tr
    )

    y6_train_np = np.asarray(
        y6tr
    )

    positive_mask = (
        (y3_train_np == 1)
        | (y6_train_np == 1)
    )

    positive_indices = np.flatnonzero(
        positive_mask
    )

    negative_indices = np.flatnonzero(
        ~positive_mask
    )

    if TRAIN_SUBSET < len(
        positive_indices
    ):
        raise RuntimeError(
            "TRAIN_SUBSET is smaller than "
            "the number of positive examples."
        )

    negative_needed = (
        TRAIN_SUBSET
        - len(positive_indices)
    )

    selected_negative = rng.choice(
        negative_indices,
        size=negative_needed,
        replace=False,
    )

    train_indices = np.concatenate(
        [
            positive_indices,
            selected_negative,
        ]
    )

    rng.shuffle(
        train_indices
    )

    print(
        "Selected training samples:",
        len(train_indices),
    )

    print(
        "Selected Y3 positives:",
        int(
            np.sum(
                y3_train_np[
                    train_indices
                ] == 1
            )
        ),
    )

    print(
        "Selected Y6 positives:",
        int(
            np.sum(
                y6_train_np[
                    train_indices
                ] == 1
            )
        ),
    )

    # ---------------------------------------------------------------
    # Class weights from selected training subset.
    # ---------------------------------------------------------------

    selected_y3 = y3_train_np[
        train_indices
    ]

    selected_y6 = y6_train_np[
        train_indices
    ]

    y3_pos = max(
        int(np.sum(selected_y3 == 1)),
        1,
    )

    y6_pos = max(
        int(np.sum(selected_y6 == 1)),
        1,
    )

    y3_neg = (
        len(selected_y3)
        - y3_pos
    )

    y6_neg = (
        len(selected_y6)
        - y6_pos
    )

    pos_weight3 = (
        y3_neg / y3_pos
    )

    pos_weight6 = (
        y6_neg / y6_pos
    )

    print(
        "Y3 pos_weight:",
        round(pos_weight3, 4),
    )

    print(
        "Y6 pos_weight:",
        round(pos_weight6, 4),
    )

    model = (
        CyberPhysicalTemporalTransformer()
        .to(device)
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model parameters:",
        parameter_count,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loss3_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight3,
            dtype=torch.float32,
            device=device,
        )
    )

    loss6_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight6,
            dtype=torch.float32,
            device=device,
        )
    )

    best_y6_pr = -np.inf
    best_epoch = 0
    patience_counter = 0
    history = []

    start_training = time.time()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = time.time()

        model.train()

        epoch_indices = (
            train_indices.copy()
        )

        rng = np.random.default_rng(
            SEED + epoch
        )

        rng.shuffle(
            epoch_indices
        )

        running_loss = 0.0
        batch_count = 0

        for start in range(
            0,
            len(epoch_indices),
            BATCH_SIZE,
        ):

            batch_idx = (
                epoch_indices[
                    start:start + BATCH_SIZE
                ]
            )

            xb = prepare_batch(
                Xtr,
                batch_idx,
                feature_indices,
                continuous_positions,
                means,
                stds,
            ).to(device)

            y3b = torch.from_numpy(
                np.asarray(
                    y3tr[batch_idx],
                    dtype=np.float32,
                )
            ).to(device)

            y6b = torch.from_numpy(
                np.asarray(
                    y6tr[batch_idx],
                    dtype=np.float32,
                )
            ).to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits3, logits6 = model(
                xb
            )

            loss3 = loss3_fn(
                logits3,
                y3b,
            )

            loss6 = loss6_fn(
                logits6,
                y6b,
            )

            loss = 0.5 * (
                loss3 + loss6
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            running_loss += (
                float(loss.item())
            )

            batch_count += 1

            del (
                xb,
                y3b,
                y6b,
                logits3,
                logits6,
                loss3,
                loss6,
                loss,
            )

        gc.collect()

        train_loss = (
            running_loss
            / max(batch_count, 1)
        )

        validation = evaluate(
            model,
            Xv,
            y3v,
            y6v,
            feature_indices,
            continuous_positions,
            means,
            stds,
            BATCH_SIZE,
            device,
            loss3_fn,
            loss6_fn,
        )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        y3_pr = validation[
            "y3"
        ]["pr_auc"]

        y6_pr = validation[
            "y6"
        ]["pr_auc"]

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={validation['loss']:.6f} | "
            f"Y3_PR={y3_pr:.6f} | "
            f"Y6_PR={y6_pr:.6f} | "
            f"time={epoch_seconds:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": validation,
                "epoch_seconds": epoch_seconds,
            }
        )

        if y6_pr > (
            best_y6_pr + 1e-4
        ):

            best_y6_pr = y6_pr
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "epoch": epoch,
                    "feature_names":
                        feature_names,
                    "feature_indices":
                        feature_indices,
                    "binary_features":
                        list(binary_names),
                    "continuous_features":
                        list(continuous_names),
                    "continuous_positions":
                        continuous_positions,
                    "means":
                        means,
                    "stds":
                        stds,
                    "y3_pos_weight":
                        pos_weight3,
                    "y6_pos_weight":
                        pos_weight6,
                    "seed": SEED,
                    "train_subset":
                        TRAIN_SUBSET,
                },
                CHECKPOINT_PATH,
            )

            print(
                "  [CHECKPOINT]"
            )

        else:

            patience_counter += 1

            print(
                f"  [PATIENCE] "
                f"{patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:

            print(
                "Early stopping."
            )

            break

    training_seconds = (
        time.time()
        - start_training
    )

    # ---------------------------------------------------------------
    # Final test evaluation.
    # ---------------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL TEST")
    print("=" * 72)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    Xte = np.load(
        X_TEST,
        mmap_mode="r",
    )

    y3te = np.load(
        Y3_TEST,
        mmap_mode="r",
    )

    y6te = np.load(
        Y6_TEST,
        mmap_mode="r",
    )

    test = evaluate(
        model,
        Xte,
        y3te,
        y6te,
        feature_indices,
        continuous_positions,
        means,
        stds,
        BATCH_SIZE,
        device,
        loss3_fn,
        loss6_fn,
    )

    print(
        "Y3 Test PR-AUC:",
        f"{test['y3']['pr_auc']:.6f}",
    )

    print(
        "Y3 Test ROC-AUC:",
        f"{test['y3']['roc_auc']:.6f}",
    )

    print(
        "Y6 Test PR-AUC:",
        f"{test['y6']['pr_auc']:.6f}",
    )

    print(
        "Y6 Test ROC-AUC:",
        f"{test['y6']['roc_auc']:.6f}",
    )

    result = {
        "project":
            "Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles",
        "phase":
            "4.10D",
        "status":
            "PASS",
        "experiment_type":
            "controlled_cpu_training",
        "interpretation":
            "Architecture validation and controlled temporal forecasting experiment; not a full-scale compute benchmark.",
        "device":
            "cpu",
        "pytorch":
            torch.__version__,
        "seed":
            SEED,
        "train_subset":
            TRAIN_SUBSET,
        "full_train_samples":
            int(len(y3tr)),
        "validation_samples":
            int(len(y3v)),
        "test_samples":
            int(len(y3te)),
        "batch_size":
            BATCH_SIZE,
        "max_epochs":
            MAX_EPOCHS,
        "best_epoch":
            best_epoch,
        "best_validation_y6_pr_auc":
            float(best_y6_pr),
        "training_seconds":
            training_seconds,
        "feature_treatment":
            {
                "original_features":
                    63,
                "final_features":
                    59,
                "binary":
                    27,
                "continuous":
                    32,
                "normalization":
                    "train_only_z_score_continuous_features",
            },
        "architecture":
            {
                "input":
                    [12, 59],
                "embedding":
                    128,
                "layers":
                    3,
                "heads":
                    8,
                "feedforward":
                    256,
                "dropout":
                    0.1,
                "parameters":
                    parameter_count,
            },
        "training":
            {
                "optimizer":
                    "AdamW",
                "learning_rate":
                    LR,
                "weight_decay":
                    WEIGHT_DECAY,
                "gradient_clip":
                    GRAD_CLIP,
                "loss":
                    "weighted BCEWithLogits",
                "primary_metric":
                    "validation Y6 PR-AUC",
                "memory_mapping":
                    True,
            },
        "class_weights":
            {
                "y3":
                    pos_weight3,
                "y6":
                    pos_weight6,
            },
        "history":
            history,
        "final_test":
            test,
        "checkpoint":
            str(CHECKPOINT_PATH),
        "methodological_boundary":
            {
                "cyber_labels":
                    "synthetic research telemetry",
                "physical_ground_truth":
                    "CRSS 2024",
                "real_world_cyberattack_labels":
                    False,
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
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("PHASE 4.10D COMPLETE")
    print("=" * 72)
    print(
        "Best epoch:",
        best_epoch,
    )
    print(
        "Best validation Y6 PR-AUC:",
        f"{best_y6_pr:.6f}",
    )
    print(
        "Test Y3 PR-AUC:",
        f"{test['y3']['pr_auc']:.6f}",
    )
    print(
        "Test Y6 PR-AUC:",
        f"{test['y6']['pr_auc']:.6f}",
    )
    print(
        "Training time:",
        f"{training_seconds:.1f}s",
    )
    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )
    print(
        "Results:",
        RESULT_PATH,
    )


if __name__ == "__main__":
    main()
