from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

ROOT = Path.cwd()

SEQ_DIR = ROOT / "data/processed/modeling/sequences/v2_forecasting"
MAP = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_modality_map.json"
OUT = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_cross_modal_results.json"

SEED = 42
SUBSET_SIZE = 100000
BATCH_SIZE = 256
MAX_EPOCHS = 5
PATIENCE = 2

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(6)

with open(SEQ_DIR / "crss_2024_forecasting_tensor_metadata.json", encoding="utf-8") as f:
    meta = json.load(f)

with open(MAP, encoding="utf-8") as f:
    modality = json.load(f)

tensor_features = meta["features"]

cyber = modality["cyber_features"]
physical = modality["physical_features"]
derived = modality["derived_temporal_resilience_features"]

excluded_composite = {
    "cyber_instability_index",
    "resilience_degradation_rate",
    "state_transition_magnitude",
}

derived_primary = [x for x in derived if x not in excluded_composite]

experiments = {
    "A_cyber_only": cyber,
    "B_physical_only": physical,
    "C_cyber_physical": cyber + physical,
    "D_cyber_physical_temporal": cyber + physical + derived_primary,
}

feature_index = {name: i for i, name in enumerate(tensor_features)}

# Confirm every requested feature exists in the original 63-feature tensor.
for name, feature_list in experiments.items():
    missing = [x for x in feature_list if x not in feature_index]
    if missing:
        raise RuntimeError(f"{name}: missing tensor features: {missing}")

X_train = np.load(SEQ_DIR / "X_train_v2_forecasting.npy", mmap_mode="r")
X_val = np.load(SEQ_DIR / "X_validation_v2_forecasting.npy", mmap_mode="r")
X_test = np.load(SEQ_DIR / "X_test_v2_forecasting.npy", mmap_mode="r")

y3_train = np.load(SEQ_DIR / "y_3step_train_v2_forecasting.npy", mmap_mode="r")
y3_val = np.load(SEQ_DIR / "y_3step_validation_v2_forecasting.npy", mmap_mode="r")
y3_test = np.load(SEQ_DIR / "y_3step_test_v2_forecasting.npy", mmap_mode="r")

y6_train = np.load(SEQ_DIR / "y_6step_train_v2_forecasting.npy", mmap_mode="r")
y6_val = np.load(SEQ_DIR / "y_6step_validation_v2_forecasting.npy", mmap_mode="r")
y6_test = np.load(SEQ_DIR / "y_6step_test_v2_forecasting.npy", mmap_mode="r")


def make_subset(y3, y6, size):
    rng = np.random.default_rng(SEED)

    positive = np.where((y3 == 1) | (y6 == 1))[0]
    negative = np.where((y3 == 0) & (y6 == 0))[0]

    if len(positive) > size:
        raise RuntimeError(
            f"Positive samples {len(positive)} exceed subset size {size}"
        )

    n_negative = size - len(positive)

    selected_negative = rng.choice(
        negative,
        size=min(n_negative, len(negative)),
        replace=False,
    )

    idx = np.concatenate([positive, selected_negative])
    rng.shuffle(idx)

    return idx.astype(np.int64)


train_idx = make_subset(y3_train, y6_train, SUBSET_SIZE)

# Use the complete validation set.
val_idx = np.arange(len(y3_val), dtype=np.int64)
test_idx = np.arange(len(y3_test), dtype=np.int64)


class TinyGRU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden = 48

        self.norm = nn.LayerNorm(input_dim)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.10)

        self.y3 = nn.Linear(hidden, 1)
        self.y6 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.norm(x)
        out, _ = self.gru(x)
        h = self.dropout(out[:, -1, :])

        return (
            self.y3(h).squeeze(-1),
            self.y6(h).squeeze(-1),
        )


def batches(X, y3, y6, indices, feature_indices):
    for start in range(0, len(indices), BATCH_SIZE):
        idx = indices[start:start + BATCH_SIZE]

        xb = np.asarray(
            X[idx][:, :, feature_indices],
            dtype=np.float32
        )

        yield (
            torch.from_numpy(xb),
            torch.from_numpy(
                np.asarray(y3[idx], dtype=np.float32)
            ),
            torch.from_numpy(
                np.asarray(y6[idx], dtype=np.float32)
            ),
        )


def evaluate(model, X, y3, y6, indices, feature_indices):
    model.eval()

    p3 = []
    p6 = []
    t3 = []
    t6 = []

    with torch.no_grad():
        for xb, y3b, y6b in batches(
            X, y3, y6, indices, feature_indices
        ):
            z3, z6 = model(xb)

            p3.append(torch.sigmoid(z3).numpy())
            p6.append(torch.sigmoid(z6).numpy())

            t3.append(y3b.numpy())
            t6.append(y6b.numpy())

    p3 = np.concatenate(p3)
    p6 = np.concatenate(p6)
    t3 = np.concatenate(t3)
    t6 = np.concatenate(t6)

    def score(y, p):
        pred = (p >= 0.5).astype(np.int32)

        return {
            "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(
                precision_score(y, pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y, pred, zero_division=0)
            ),
            "brier": float(brier_score_loss(y, p)),
        }

    return {
        "Y3": score(t3, p3),
        "Y6": score(t6, p6),
    }


def train_one(name, feature_names):
    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    feature_indices = np.array(
        [feature_index[x] for x in feature_names],
        dtype=np.int64,
    )

    print("Feature count:", len(feature_indices))

    model = TinyGRU(len(feature_indices))

    train_y3 = np.asarray(y3_train[train_idx])
    train_y6 = np.asarray(y6_train[train_idx])

    pos3 = max(float(np.sum(train_y3 == 1)), 1.0)
    neg3 = max(float(np.sum(train_y3 == 0)), 1.0)

    pos6 = max(float(np.sum(train_y6 == 1)), 1.0)
    neg6 = max(float(np.sum(train_y6 == 0)), 1.0)

    weight3 = neg3 / pos3
    weight6 = neg6 / pos6

    criterion3 = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weight3)
    )

    criterion6 = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weight6)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    best_score = -np.inf
    best_state = None
    bad_epochs = 0

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        losses = []

        for xb, y3b, y6b in batches(
            X_train,
            y3_train,
            y6_train,
            train_idx,
            feature_indices,
        ):
            optimizer.zero_grad(set_to_none=True)

            z3, z6 = model(xb)

            loss3 = criterion3(z3, y3b)
            loss6 = criterion6(z6, y6b)

            loss = loss3 + loss6

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            losses.append(float(loss.item()))

        val = evaluate(
            model,
            X_val,
            y3_val,
            y6_val,
            val_idx,
            feature_indices,
        )

        y6_pr = val["Y6"]["pr_auc"]

        print(
            f"Epoch {epoch} | "
            f"loss={np.mean(losses):.5f} | "
            f"Y3 PR-AUC={val['Y3']['pr_auc']:.6f} | "
            f"Y6 PR-AUC={y6_pr:.6f}"
        )

        if y6_pr > best_score:
            best_score = y6_pr

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            break

    model.load_state_dict(best_state)

    test = evaluate(
        model,
        X_test,
        y3_test,
        y6_test,
        test_idx,
        feature_indices,
    )

    elapsed = time.time() - start_time

    print(
        f"TEST | "
        f"Y3 PR-AUC={test['Y3']['pr_auc']:.6f} | "
        f"Y6 PR-AUC={test['Y6']['pr_auc']:.6f} | "
        f"time={elapsed:.1f}s"
    )

    return {
        "feature_count": len(feature_names),
        "features": feature_names,
        "parameters": int(
            sum(p.numel() for p in model.parameters())
        ),
        "best_validation_y6_pr_auc": float(best_score),
        "test_metrics": test,
        "training_seconds": float(elapsed),
    }


results = {
    "phase": "4.11",
    "status": "PASS",
    "seed": SEED,
    "controlled_subset_size": SUBSET_SIZE,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "observation_window_steps": 12,
    "observation_window_seconds": 60,
    "excluded_composite_features": sorted(excluded_composite),
    "experiments": {},
}

for name, feature_names in experiments.items():
    results["experiments"][name] = train_one(
        name,
        feature_names
    )

A = results["experiments"]["A_cyber_only"]["test_metrics"]
B = results["experiments"]["B_physical_only"]["test_metrics"]
C = results["experiments"]["C_cyber_physical"]["test_metrics"]
D = results["experiments"]["D_cyber_physical_temporal"]["test_metrics"]

results["comparison"] = {
    "C_vs_A_Y6_PR_AUC_delta":
        C["Y6"]["pr_auc"] - A["Y6"]["pr_auc"],

    "C_vs_B_Y6_PR_AUC_delta":
        C["Y6"]["pr_auc"] - B["Y6"]["pr_auc"],

    "D_vs_C_Y6_PR_AUC_delta":
        D["Y6"]["pr_auc"] - C["Y6"]["pr_auc"],

    "C_vs_A_Y3_PR_AUC_delta":
        C["Y3"]["pr_auc"] - A["Y3"]["pr_auc"],

    "C_vs_B_Y3_PR_AUC_delta":
        C["Y3"]["pr_auc"] - B["Y3"]["pr_auc"],

    "D_vs_C_Y3_PR_AUC_delta":
        D["Y3"]["pr_auc"] - C["Y3"]["pr_auc"],
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print()
print("=" * 72)
print("PHASE 4.11 — CROSS-MODAL EXPERIMENT COMPLETE")
print("=" * 72)

print("Output:", OUT)
print()
print("TEST Y6 PR-AUC:")

for name, result in results["experiments"].items():
    print(
        f"  {name:<32} "
        f"{result['test_metrics']['Y6']['pr_auc']:.6f}"
    )

print()
print("C vs A Y6:",
      f"{results['comparison']['C_vs_A_Y6_PR_AUC_delta']:+.6f}")

print("C vs B Y6:",
      f"{results['comparison']['C_vs_B_Y6_PR_AUC_delta']:+.6f}")

print("D vs C Y6:",
      f"{results['comparison']['D_vs_C_Y6_PR_AUC_delta']:+.6f}")

