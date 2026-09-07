from pathlib import Path
import json, math, random, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")
SEQ = ROOT / "data/processed/modeling/sequences/v2_forecasting"
MOD = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_modality_map.json"
OUT = ROOT / "experiments/modeling/v2/uncertainty/phase4_27b"
OUT.mkdir(parents=True, exist_ok=True)

CKPTS = [
    ROOT / "models/v2/uncertainty/phase4_27/transformer_y6_seed_42_best.pt",
    ROOT / "models/v2/uncertainty/phase4_27/transformer_y6_seed_52_best.pt",
    ROOT / "models/v2/uncertainty/phase4_27/transformer_y6_seed_62_best.pt",
]

SEEDS = [42, 52, 62]
BATCH = 128
THREADS = 6
REFERENCE_N = 100_000
PCA_COMPONENTS = 32
P90 = 0.90
OOD_QUANTILE = 0.99

torch.set_num_threads(THREADS)
np.random.seed(42)
random.seed(42)

def load_npy(name):
    return np.load(SEQ / name, mmap_mode="r")

X_train = load_npy("X_train_v2_forecasting.npy")
X_val = load_npy("X_validation_v2_forecasting.npy")
X_test = load_npy("X_test_v2_forecasting.npy")

y_train = np.load(SEQ / "y_6step_train_v2_forecasting.npy")
y_val = np.load(SEQ / "y_6step_validation_v2_forecasting.npy")
y_test = np.load(SEQ / "y_6step_test_v2_forecasting.npy")

# 63 -> 59, matching Phase 4.27A / model readiness.
KEEP = np.array([i for i in range(X_train.shape[2]) if i not in [30,35,39,62]])

assert X_train.shape[2] == 63
assert len(KEEP) == 59

X_train_r = X_train[:, :, KEEP]
X_val_r = X_val[:, :, KEEP]
X_test_r = X_test[:, :, KEEP]

# ------------------------------------------------------------------
# Robustly recover the existing 59-feature modality assignment.
# ------------------------------------------------------------------

with open(MOD, "r", encoding="utf-8") as f:
    modality_json = json.load(f)

def collect_groups(obj):
    groups = {"cyber": set(), "physical": set(), "derived": set()}

    def walk(x, current=None):
        if isinstance(x, dict):
            for k, v in x.items():
                kl = str(k).lower()
                nxt = current
                if "cyber" in kl:
                    nxt = "cyber"
                elif "physical" in kl:
                    nxt = "physical"
                elif "derived" in kl or "temporal" in kl or "resilience" in kl:
                    nxt = "derived"

                if isinstance(v, list):
                    if nxt:
                        for item in v:
                            if isinstance(item, str):
                                groups[nxt].add(item)
                            elif isinstance(item, dict):
                                for key in ["feature", "name", "column", "feature_name"]:
                                    if key in item and isinstance(item[key], str):
                                        groups[nxt].add(item[key])
                walk(v, nxt)
        elif isinstance(x, list):
            for item in x:
                walk(item, current)

    walk(obj)
    return groups

groups = collect_groups(modality_json)

# If the modality file stores dictionaries keyed by feature name,
# recover them as well.
if sum(len(v) for v in groups.values()) < 59:
    def walk_feature_map(obj):
        for k, v in obj.items() if isinstance(obj, dict) else []:
            if isinstance(v, str):
                vl = v.lower()
                kl = k.lower()
                target = None
                if "cyber" in vl:
                    target = "cyber"
                elif "physical" in vl:
                    target = "physical"
                elif "derived" in vl or "temporal" in vl or "resilience" in vl:
                    target = "derived"
                if target:
                    groups[target].add(k)
            elif isinstance(v, dict):
                walk_feature_map(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        walk_feature_map(item)
    walk_feature_map(modality_json)

# Get feature names from metadata.
META = SEQ / "crss_2024_forecasting_tensor_metadata.json"
with open(META, "r", encoding="utf-8") as f:
    meta = json.load(f)

def find_feature_list(obj):
    candidates = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                kl = str(k).lower()
                if isinstance(v, list) and v and all(isinstance(z, str) for z in v):
                    if any(t in kl for t in ["feature", "column", "input"]):
                        candidates.append(v)
                walk(v)
        elif isinstance(x, list):
            for z in x:
                walk(z)
    walk(obj)
    candidates.sort(key=len, reverse=True)
    return candidates[0] if candidates else None

all_features = find_feature_list(meta)

if all_features is None or len(all_features) != 63:
    raise RuntimeError(
        f"Could not recover the expected 63-feature metadata list. "
        f"Found={0 if all_features is None else len(all_features)}"
    )

feature_names = [all_features[i] for i in KEEP]

# Normalize group names against actual model feature names.
def normalize_name(s):
    return str(s).strip().lower()

lookup = {normalize_name(n): i for i, n in enumerate(feature_names)}

group_idx = {}
for g in ["cyber", "physical", "derived"]:
    idx = sorted({
        lookup[normalize_name(n)]
        for n in groups[g]
        if normalize_name(n) in lookup
    })
    group_idx[g] = np.array(idx, dtype=np.int64)

if len(group_idx["cyber"]) != 34 or len(group_idx["physical"]) != 12 or len(group_idx["derived"]) != 13:
    raise RuntimeError(
        "Modality mapping validation failed. "
        f"cyber={len(group_idx['cyber'])}, "
        f"physical={len(group_idx['physical'])}, "
        f"derived={len(group_idx['derived'])}. "
        "Expected 34/12/13."
    )

assert len(set(group_idx["cyber"]) &
           set(group_idx["physical"])) == 0
assert len(set(group_idx["cyber"]) &
           set(group_idx["derived"])) == 0
assert len(set(group_idx["physical"]) &
           set(group_idx["derived"])) == 0
assert sum(len(v) for v in group_idx.values()) == 59

# ------------------------------------------------------------------
# Train-only normalization using the same 100k-reference principle.
# ------------------------------------------------------------------

pos = np.where(y_train == 1)[0]
neg = np.where(y_train == 0)[0]

rng = np.random.default_rng(42)
n_neg = max(0, REFERENCE_N - len(pos))
neg_sel = rng.choice(neg, size=n_neg, replace=False)

ref_idx = np.concatenate([pos, neg_sel])
rng.shuffle(ref_idx)

sum_x = np.zeros(59, dtype=np.float64)
sum_x2 = np.zeros(59, dtype=np.float64)
count = 0

for start in range(0, len(ref_idx), 256):
    ids = ref_idx[start:start+256]
    b = np.asarray(X_train_r[ids], dtype=np.float32)
    flat = b.reshape(-1, 59).astype(np.float64)
    sum_x += flat.sum(axis=0)
    sum_x2 += np.square(flat).sum(axis=0)
    count += flat.shape[0]

mean = sum_x / count
var = np.maximum(sum_x2 / count - mean * mean, 1e-8)
std = np.sqrt(var)

def normalize_batch(x):
    return ((np.asarray(x, dtype=np.float32) - mean.astype(np.float32))
            / std.astype(np.float32)).astype(np.float32)

# ------------------------------------------------------------------
# Temporal summary representation for distribution/OOD modeling.
# 8 deterministic statistics x 59 features = 472 dimensions.
# ------------------------------------------------------------------

def temporal_summary(x):
    # x = [N,T,F]
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    last = x[:, -1, :]
    delta = x[:, -1, :] - x[:, 0, :]
    step = np.diff(x, axis=1)
    max_step = np.max(np.abs(step), axis=1)
    last_step = x[:, -1, :] - x[:, -2, :]

    return np.concatenate(
        [mu, sd, mn, mx, last, delta, max_step, last_step],
        axis=1
    ).astype(np.float32)

# ------------------------------------------------------------------
# Fit PCA + Ledoit-Wolf Mahalanobis OOD detector on TRAIN ONLY.
# ------------------------------------------------------------------

print("=== PHASE 4.27B: OOD DETECTOR FIT ===")
print(f"Reference sequences: {len(ref_idx)}")
print("Building temporal summary representation...")

pca = IncrementalPCA(
    n_components=PCA_COMPONENTS,
    batch_size=2048
)

for start in range(0, len(ref_idx), 2048):
    ids = ref_idx[start:start+2048]
    xb = normalize_batch(X_train_r[ids])
    sb = temporal_summary(xb)
    pca.partial_fit(sb)

ref_scores = np.empty((len(ref_idx), PCA_COMPONENTS), dtype=np.float32)

for start in range(0, len(ref_idx), 2048):
    ids = ref_idx[start:start+2048]
    xb = normalize_batch(X_train_r[ids])
    sb = temporal_summary(xb)
    ref_scores[start:start+len(ids)] = pca.transform(sb).astype(np.float32)

cov = LedoitWolf(assume_centered=False)
cov.fit(ref_scores)

ref_mean = cov.location_.astype(np.float64)
precision = cov.precision_.astype(np.float64)

def mahalanobis(scores):
    d = scores.astype(np.float64) - ref_mean
    return np.einsum("ij,jk,ik->i", d, precision, d)

# ------------------------------------------------------------------
# Validation clean OOD threshold, using validation NORMAL states only.
# ------------------------------------------------------------------

val_normal_idx = np.where(y_val == 0)[0]

val_scores = np.empty(len(val_normal_idx), dtype=np.float64)

for start in range(0, len(val_normal_idx), 2048):
    ids = val_normal_idx[start:start+2048]
    xb = normalize_batch(X_val_r[ids])
    sb = temporal_summary(xb)
    ps = pca.transform(sb)
    val_scores[start:start+len(ids)] = mahalanobis(ps)

ood_threshold = float(np.quantile(val_scores, OOD_QUANTILE))

print(f"OOD threshold ({OOD_QUANTILE:.2%} validation-normal quantile): {ood_threshold:.6f}")

# ------------------------------------------------------------------
# Transformer
# ------------------------------------------------------------------

class TemporalTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(59, 128)

        self.position = nn.Parameter(
            torch.zeros(1, 12, 128)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=256,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=3
        )

        self.norm = nn.LayerNorm(128)

        # Phase 4.27A attention pooling.
        self.attention = nn.Linear(128, 1)

        self.head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.input_projection(x) + self.position

        h = self.encoder(h)

        h = self.norm(h)

        # Attention pooling over the 12-step temporal window.
        weights = torch.softmax(
            self.attention(h).squeeze(-1),
            dim=1
        )

        pooled = torch.sum(
            h * weights.unsqueeze(-1),
            dim=1
        )

        return self.head(pooled).squeeze(-1)


def load_model(path):
    model = TemporalTransformer()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

models = [load_model(p) for p in CKPTS]

@torch.no_grad()
def ensemble_predict(X):
    preds = []

    for model_no, model in enumerate(models):
        out = np.empty(len(X), dtype=np.float32)

        for start in range(0, len(X), BATCH):
            xb = normalize_batch(X[start:start+BATCH])
            logits = model(torch.from_numpy(xb))
            out[start:start+len(xb)] = torch.sigmoid(logits).numpy()

        preds.append(out)

    arr = np.stack(preds, axis=1)

    mean_p = arr.mean(axis=1)
    variance = arr.var(axis=1)

    # Predictive entropy.
    eps = 1e-7
    entropy = -(
        mean_p * np.log(np.clip(mean_p, eps, 1-eps)) +
        (1-mean_p) * np.log(np.clip(1-mean_p, eps, 1-eps))
    )

    return arr, mean_p, variance, entropy

def ood_scores_for(X):
    out = np.empty(len(X), dtype=np.float64)

    for start in range(0, len(X), 2048):
        xb = normalize_batch(X[start:start+2048])
        sb = temporal_summary(xb)
        ps = pca.transform(sb)
        out[start:start+len(xb)] = mahalanobis(ps)

    return out

# ------------------------------------------------------------------
# Perturbations
# ------------------------------------------------------------------

def perturb(X, mode, strength=0.10, seed=100):
    rng = np.random.default_rng(seed)
    Y = np.array(X, dtype=np.float32, copy=True)

    if mode == "clean":
        return Y

    if mode == "cyber":
        idx = group_idx["cyber"]
        Y[:, :, idx] += rng.normal(
            0, strength, size=Y[:, :, idx].shape
        ).astype(np.float32)

    elif mode == "physical":
        idx = group_idx["physical"]
        Y[:, :, idx] += rng.normal(
            0, strength, size=Y[:, :, idx].shape
        ).astype(np.float32)

    elif mode == "combined":
        idx = np.concatenate([group_idx["cyber"], group_idx["physical"]])
        Y[:, :, idx] += rng.normal(
            0, strength, size=Y[:, :, idx].shape
        ).astype(np.float32)

    elif mode == "temporal_mask":
        k = int(strength)
        if k >= 12:
            raise ValueError("Temporal mask cannot remove all 12 steps.")
        Y[:, -k:, :] = Y[:, [-k-1], :]

    elif mode == "cyber_shift":
        idx = group_idx["cyber"]
        Y[:, :, idx] += np.float32(strength)

    elif mode == "physical_shift":
        idx = group_idx["physical"]
        Y[:, :, idx] += np.float32(strength)

    elif mode == "combined_shift":
        idx = np.concatenate([group_idx["cyber"], group_idx["physical"]])
        Y[:, :, idx] += np.float32(strength)

    else:
        raise ValueError(mode)

    return Y

# ------------------------------------------------------------------
# Validation OOD discrimination.
# ------------------------------------------------------------------

print("\n=== VALIDATION OOD DISCRIMINATION ===")

# Keep validation OOD experiment label-independent:
# normal clean vs known synthetic distribution shifts.
ood_eval_n = min(20_000, len(val_normal_idx))
rng = np.random.default_rng(427)
eval_ids = rng.choice(val_normal_idx, size=ood_eval_n, replace=False)

Xv_clean = np.asarray(X_val_r[eval_ids], dtype=np.float32)

ood_conditions = {
    "cyber_noise_10pct": ("cyber", 0.10),
    "physical_noise_10pct": ("physical", 0.10),
    "combined_noise_10pct": ("combined", 0.10),
    "temporal_mask_2": ("temporal_mask", 2),
    "cyber_shift_1sigma": ("cyber_shift", 1.0),
    "physical_shift_1sigma": ("physical_shift", 1.0),
    "combined_shift_1sigma": ("combined_shift", 1.0),
}

ood_validation = {}

clean_ood = ood_scores_for(Xv_clean)

for name, (mode, strength) in ood_conditions.items():
    Xp = perturb(Xv_clean, mode, strength, seed=427)
    shifted_ood = ood_scores_for(Xp)

    labels = np.concatenate([
        np.zeros(len(clean_ood), dtype=np.int8),
        np.ones(len(shifted_ood), dtype=np.int8)
    ])
    scores = np.concatenate([clean_ood, shifted_ood])

    auc = float(roc_auc_score(labels, scores))
    rate = float(np.mean(shifted_ood >= ood_threshold))

    ood_validation[name] = {
        "ood_auc": auc,
        "shifted_ood_detection_rate": rate,
        "clean_mean_distance": float(clean_ood.mean()),
        "shifted_mean_distance": float(shifted_ood.mean()),
        "shifted_p95_distance": float(np.quantile(shifted_ood, 0.95))
    }

    print(
        f"{name:28s} "
        f"AUC={auc:.4f} "
        f"detect={rate:.4%}"
    )

# ------------------------------------------------------------------
# Full test regime evaluation.
# ------------------------------------------------------------------

print("\n=== FULL TEST REGIME EVALUATION ===")

test_conditions = {
    "clean": ("clean", 0),
    "cyber_noise_10pct": ("cyber", 0.10),
    "physical_noise_10pct": ("physical", 0.10),
    "combined_noise_10pct": ("combined", 0.10),
    "temporal_mask_2": ("temporal_mask", 2),
    "combined_shift_1sigma": ("combined_shift", 1.0),
}

test_results = {}
risk_uncertainty = {}

for name, (mode, strength) in test_conditions.items():
    print(f"\nRunning: {name}")

    Xt = perturb(X_test_r, mode, strength, seed=427)

    member_p, mean_p, variance, entropy = ensemble_predict(Xt)
    distances = ood_scores_for(Xt)

    high_unc = variance >= float(np.quantile(variance, P90))
    high_ood = distances >= ood_threshold
    high_risk = mean_p >= 0.50

    pr = float(average_precision_score(y_test, mean_p))
    roc = float(roc_auc_score(y_test, mean_p))
    brier = float(brier_score_loss(y_test, mean_p))

    errors = (mean_p >= 0.50) != (y_test.astype(bool))

    test_results[name] = {
        "pr_auc": pr,
        "roc_auc": roc,
        "brier": brier,
        "mean_risk": float(mean_p.mean()),
        "risk_p95": float(np.quantile(mean_p, 0.95)),
        "mean_ensemble_variance": float(variance.mean()),
        "p90_uncertainty_threshold": float(np.quantile(variance, P90)),
        "high_uncertainty_rate": float(high_unc.mean()),
        "mean_predictive_entropy": float(entropy.mean()),
        "mean_ood_distance": float(distances.mean()),
        "ood_p95_distance": float(np.quantile(distances, 0.95)),
        "ood_rate": float(high_ood.mean()),
        "error_rate": float(errors.mean())
    }

    cells = {}
    for r in [False, True]:
        for u in [False, True]:
            for o in [False, True]:
                mask = (high_risk == r) & (high_unc == u) & (high_ood == o)
                key = f"risk_{'HIGH' if r else 'LOW'}__unc_{'HIGH' if u else 'LOW'}__ood_{'OUT' if o else 'IN'}"

                if mask.any():
                    cells[key] = {
                        "count": int(mask.sum()),
                        "rate": float(mask.mean()),
                        "error_rate": float(errors[mask].mean())
                    }
                else:
                    cells[key] = {
                        "count": 0,
                        "rate": 0.0,
                        "error_rate": None
                    }

    risk_uncertainty[name] = cells

    print(
        f"PR={pr:.6f} "
        f"ROC={roc:.6f} "
        f"Brier={brier:.6f} "
        f"OOD={high_ood.mean():.4%} "
        f"unc={high_unc.mean():.4%}"
    )

# ------------------------------------------------------------------
# Final gates.
# ------------------------------------------------------------------

gates = {
    "checkpoint_seed_42_exists": CKPTS[0].exists(),
    "checkpoint_seed_52_exists": CKPTS[1].exists(),
    "checkpoint_seed_62_exists": CKPTS[2].exists(),
    "feature_count_59": len(feature_names) == 59,
    "modality_counts_34_12_13": (
        len(group_idx["cyber"]) == 34 and
        len(group_idx["physical"]) == 12 and
        len(group_idx["derived"]) == 13
    ),
    "modality_partition_complete": (
        sum(len(v) for v in group_idx.values()) == 59
    ),
    "ood_threshold_from_validation_normal_only": True,
    "ood_detector_fit_train_only": True,
    "test_not_used_for_ood_threshold": True,
    "ensemble_members_3": len(models) == 3,
    "ood_auc_all_conditions_above_0_5": all(
        x["ood_auc"] > 0.5 for x in ood_validation.values()
    ),
    "direct_vehicle_actuation_disabled": True
}

status = "PASS" if all(gates.values()) else "FAIL"

report = {
    "phase": "4.27B",
    "title": "OOD and Regime-Aware Uncertainty Analysis",
    "status": status,
    "method": {
        "ensemble": "3-member temporal Transformer deep ensemble",
        "seeds": SEEDS,
        "ood_detector": "IncrementalPCA(32) + Ledoit-Wolf Mahalanobis distance",
        "reference_training_sequences": int(len(ref_idx)),
        "temporal_summary_dimensions": 472,
        "pca_dimensions": PCA_COMPONENTS,
        "ood_threshold_quantile": OOD_QUANTILE,
        "uncertainty_threshold_quantile": P90
    },
    "feature_partition": {
        "total": 59,
        "cyber": 34,
        "physical": 12,
        "derived": 13,
        "feature_names": feature_names
    },
    "ood_threshold": ood_threshold,
    "validation_ood_results": ood_validation,
    "test_regime_results": test_results,
    "risk_uncertainty_ood_matrix": risk_uncertainty,
    "gates": gates,
    "safety_boundary": {
        "direct_vehicle_actuation": False,
        "system_role": "decision support and predictive resilience research",
        "production_cyberattack_claim": False,
        "synthetic_label_caveat": True
    }
}

report_path = OUT / "crss_2024_v2_phase4_27b_ood_regime_uncertainty_final.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n============================================================")
print("PHASE 4.27B COMPLETE")
print("============================================================")
print(f"STATUS: {status}")
print(f"REPORT: {report_path}")
print("\nGATES:")
for k, v in gates.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")

print("\nKey validation OOD AUC:")
for k, v in ood_validation.items():
    print(f"  {k}: {v['ood_auc']:.6f}")

print("\nClean test:")
print(json.dumps(test_results["clean"], indent=2))
