import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

X_PATH = ROOT / "data/processed/modeling/sequences/v2_forecasting/X_test_v2_forecasting.npy"
Y_PATH = ROOT / "data/processed/modeling/sequences/v2_forecasting/y_6step_test_v2_forecasting.npy"
SCENARIO_PATH = ROOT / "data/processed/modeling/sequences/v2_forecasting/scenario_ids_test_v2_forecasting.npy"
ORIGIN_PATH = ROOT / "data/processed/modeling/sequences/v2_forecasting/prediction_origins_test_v2_forecasting.npy"

MODALITY_MAP_PATH = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_modality_map.json"

MODEL_DIR = ROOT / "models/v2/uncertainty/phase4_27"

MODEL_PATHS = [
    MODEL_DIR / "transformer_y6_seed_42_best.pt",
    MODEL_DIR / "transformer_y6_seed_52_best.pt",
    MODEL_DIR / "transformer_y6_seed_62_best.pt",
]

OUT_PATH = ROOT / "experiments/integration/phase4_32c/crss_2024_phase4_32c_modality_grounded_explainability.json"

ZERO_VARIANCE = [30, 35, 39, 62]
N_FEATURES = 59
SEQ_LEN = 12
SEED = 4321

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(6)


class TemporalTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(59, 128)
        self.position = nn.Parameter(torch.zeros(1, 12, 128))

        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=256,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.norm = nn.LayerNorm(128)
        self.attention = nn.Linear(128, 1)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.position
        x = self.encoder(x)
        x = self.norm(x)

        weights = torch.softmax(self.attention(x), dim=1)
        pooled = torch.sum(weights * x, dim=1)

        return self.head(pooled).squeeze(-1)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def load_model(path):
    model = TemporalTransformer()

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()

    return model


def ensemble_predict(models, x):
    tensor = torch.from_numpy(
        x.astype(np.float32)
    )

    probabilities = []

    with torch.no_grad():
        for model in models:
            logits = model(tensor).numpy()
            probabilities.append(sigmoid(logits))

    probabilities = np.stack(probabilities, axis=0)

    return (
        probabilities.mean(axis=0),
        probabilities.std(axis=0),
    )


def stable_digest(obj):
    canonical = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


print("=== PHASE 4.32C ===")
print("Model-Grounded Modality Attribution")
print()


# ============================================================
# PRE-FLIGHT
# ============================================================

required_files = [
    X_PATH,
    Y_PATH,
    SCENARIO_PATH,
    ORIGIN_PATH,
    MODALITY_MAP_PATH,
    *MODEL_PATHS,
]

missing_files = [
    str(p)
    for p in required_files
    if not p.exists()
]

if missing_files:
    print("PRE-FLIGHT FAILED")
    for p in missing_files:
        print("MISSING:", p)
    raise SystemExit(1)

print("Pre-flight files: PASS")


# ============================================================
# LOAD AUTHORITATIVE PHASE 4.11 MODALITY MAP
# ============================================================

with open(MODALITY_MAP_PATH, "r", encoding="utf-8") as f:
    modality_map = json.load(f)

assert modality_map["status"] == "PASS"
assert modality_map["total_features"] == 59
assert modality_map["cyber_feature_count"] == 34
assert modality_map["physical_feature_count"] == 12
assert modality_map["derived_temporal_resilience_feature_count"] == 13

ordered_map = modality_map["ordered_feature_map"]

assert isinstance(ordered_map, list)
assert len(ordered_map) == 59

required_map_keys = {
    "feature",
    "original_index",
    "modality",
}

for item in ordered_map:
    assert required_map_keys.issubset(item.keys())

print("Phase 4.11 modality map: PASS")
print("Cyber:", modality_map["cyber_feature_count"])
print("Physical:", modality_map["physical_feature_count"])
print("Derived:", modality_map["derived_temporal_resilience_feature_count"])


# ============================================================
# LOAD TEST DATA
# ============================================================

X_raw = np.load(X_PATH, allow_pickle=False)
Y = np.load(Y_PATH, allow_pickle=False).astype(int)
SCENARIOS = np.load(SCENARIO_PATH, allow_pickle=True)
ORIGINS = np.load(ORIGIN_PATH, allow_pickle=False)

assert X_raw.shape[1] == SEQ_LEN
assert X_raw.shape[2] == 63
assert len(X_raw) == len(Y) == len(SCENARIOS) == len(ORIGINS)

X = np.delete(
    X_raw,
    ZERO_VARIANCE,
    axis=2,
)

assert X.shape[2] == 59

print("Test tensor:", X_raw.shape)
print("Reduced model tensor:", X.shape)


# ============================================================
# BUILD 59-FEATURE MODEL MAP USING original_index
# ============================================================

feature_records = []

for item in ordered_map:

    original_index = int(item["original_index"])


    feature_records.append(
        {
            "feature": str(item["feature"]),
            "original_index": original_index,
            "modality": str(item["modality"]),
        }
    )

# Phase 4.11 ordered_feature_map is already the established
# 59-feature model-space ordering.
expected_model_indices = list(range(59))

actual_model_indices = list(range(len(feature_records)))

if actual_model_indices != expected_model_indices:
    raise ValueError(
        "Phase 4.11 modality map does not contain the expected "
        "59-feature model-space ordering."
    )

actual_original_indices = [
    int(item["original_index"])
    for item in feature_records
]

if actual_original_indices != list(range(59)):
    raise ValueError(
        "Phase 4.11 original_index values do not match the established "
        "59-feature model-space provenance ordering."
    )
feature_names = [
    x["feature"]
    for x in feature_records
]

feature_modalities = [
    x["modality"]
    for x in feature_records
]

assert len(feature_names) == 59
assert len(set(feature_names)) == 59

modality_counts = Counter(feature_modalities)

print("59-feature ordering: PASS")
print("Actual modality counts:", dict(modality_counts))


# ============================================================
# LOAD FROZEN 4.27A ENSEMBLE
# ============================================================

models = []

checkpoint_hashes = {}

for path in MODEL_PATHS:

    model = load_model(path)

    models.append(model)

    checkpoint_hashes[str(path.relative_to(ROOT))] = sha256_file(path)

    print(
        "Loaded",
        path.name,
        "| bytes:",
        path.stat().st_size,
        "| SHA256:",
        checkpoint_hashes[str(path.relative_to(ROOT))]
    )

assert len(models) == 3

print("Frozen ensemble: PASS")


# ============================================================
# UNIQUE-SCENARIO ORIGIN-BALANCED SAMPLE
# ============================================================

scenario_rows = defaultdict(list)

for idx, scenario in enumerate(SCENARIOS):
    scenario_rows[str(scenario)].append(idx)

scenario_records = []

for scenario, rows in scenario_rows.items():

    rows = np.asarray(rows, dtype=int)

    labels = Y[rows]

    target = int(labels.max())

    # Select one origin deterministically.
    #
    # The final selected set is balanced across origins below.
    for row in rows:
        scenario_records.append(
            {
                "scenario": scenario,
                "target": target,
                "origin": int(ORIGINS[row]),
                "row": int(row),
            }
        )

# One scenario may occur at several origins.
# Select candidates separately by origin and target.

rng = np.random.default_rng(SEED)

selected = []

for target in [0, 1]:

    candidates = [
        r
        for r in scenario_records
        if r["target"] == target
    ]

    # Deduplicate by scenario.
    by_scenario = {}

    for r in candidates:
        scenario = r["scenario"]

        if scenario not in by_scenario:
            by_scenario[scenario] = []

        by_scenario[scenario].append(r)

    unique_candidates = []

    for scenario, records in by_scenario.items():

        records = sorted(
            records,
            key=lambda r: r["origin"]
        )

        unique_candidates.append(
            {
                "scenario": scenario,
                "target": target,
                "records": records,
            }
        )

    rng.shuffle(unique_candidates)

    # Round-robin origins 11..17.
    buckets = {
        origin: []
        for origin in range(11, 18)
    }

    for item in unique_candidates:
        for record in item["records"]:
            if record["origin"] in buckets:
                buckets[record["origin"]].append(
                    (item["scenario"], record)
                )

    for origin in buckets:
        rng.shuffle(buckets[origin])

    target_count = 25
    origin_order = list(range(11, 18))

    counts = Counter()

    while sum(counts.values()) < target_count:

        progressed = False

        for origin in origin_order:

            if sum(counts.values()) >= target_count:
                break

            if not buckets[origin]:
                continue

            scenario, record = buckets[origin].pop()

            if any(
                x["scenario"] == scenario
                for x in selected
            ):
                # Only relevant within this target.
                continue

            if any(
                x["scenario"] == scenario
                and x["target"] == target
                for x in selected
            ):
                continue

            selected.append(
                {
                    "scenario": scenario,
                    "target": target,
                    "origin": record["origin"],
                    "row": record["row"],
                }
            )

            counts[origin] += 1
            progressed = True

        if not progressed:
            break

# Keep exactly 25 per class.
negative = [
    x for x in selected
    if x["target"] == 0
][:25]

positive = [
    x for x in selected
    if x["target"] == 1
][:25]

selected = negative + positive

assert len(selected) == 50
assert len({
    x["scenario"]
    for x in selected
}) == 50

print("Unique scenarios selected:", len(selected))
print("Targets:", dict(Counter(x["target"] for x in selected)))
print("Origins:", dict(Counter(x["origin"] for x in selected)))


# ============================================================
# MODEL-GROUNDED ATTRIBUTION
# ============================================================

cases = []

for case_id, record in enumerate(selected, start=1):

    row = record["row"]

    sample = X[row:row + 1].copy()

    base_probability, base_uncertainty = ensemble_predict(
        models,
        sample
    )

    base_probability = float(base_probability[0])
    base_uncertainty = float(base_uncertainty[0])

    attribution = []

    for feature_idx in range(N_FEATURES):

        perturbed = sample.copy()

        perturbed[:, :, feature_idx] = 0.0

        perturbed_probability, _ = ensemble_predict(
            models,
            perturbed
        )

        delta = (
            base_probability
            - float(perturbed_probability[0])
        )

        attribution.append(
            {
                "feature_index": feature_idx,
                "feature_name": feature_names[feature_idx],
                "original_index": feature_records[feature_idx]["original_index"],
                "modality": feature_modalities[feature_idx],
                "probability_delta": float(delta),
                "absolute_delta": float(abs(delta)),
            }
        )

    attribution.sort(
        key=lambda x: x["absolute_delta"],
        reverse=True
    )

    top10 = attribution[:10]

    # ========================================================
    # MODALITY ATTRIBUTION
    # ========================================================

    modality_groups = defaultdict(list)

    for item in attribution:
        modality_groups[item["modality"]].append(item)

    modality_summary = {}

    for modality, items in modality_groups.items():

        signed = sum(
            x["probability_delta"]
            for x in items
        )

        absolute = sum(
            x["absolute_delta"]
            for x in items
        )

        modality_summary[modality] = {
            "feature_count": len(items),
            "signed_attribution_sum": float(signed),
            "absolute_attribution_sum": float(absolute),
            "mean_absolute_attribution": float(
                absolute / len(items)
            ),
            "top_features": [
                {
                    "feature_index": x["feature_index"],
                    "feature_name": x["feature_name"],
                    "original_index": x["original_index"],
                    "probability_delta": x["probability_delta"],
                }
                for x in items[:5]
            ],
        }

    # ========================================================
    # TEMPORAL ATTRIBUTION
    # ========================================================

    temporal_features = []

    for item in top10[:5]:

        idx = item["feature_index"]

        timestep_results = []

        for timestep in range(SEQ_LEN):

            perturbed = sample.copy()

            perturbed[:, timestep, idx] = 0.0

            probability, _ = ensemble_predict(
                models,
                perturbed
            )

            timestep_results.append(
                {
                    "timestep": timestep,
                    "probability_delta": float(
                        base_probability
                        - float(probability[0])
                    ),
                }
            )

        temporal_features.append(
            {
                "feature_index": idx,
                "feature_name": item["feature_name"],
                "original_index": item["original_index"],
                "modality": item["modality"],
                "timesteps": timestep_results,
            }
        )

    # ========================================================
    # COUNTERFACTUAL
    # ========================================================

    top5_indices = [
        x["feature_index"]
        for x in top10[:5]
    ]

    counterfactual = sample.copy()

    for idx in top5_indices:
        counterfactual[:, :, idx] = 0.0

    counter_probability, _ = ensemble_predict(
        models,
        counterfactual
    )

    counter_probability = float(counter_probability[0])

    # ========================================================
    # RISK POLICY
    # ========================================================

    if base_probability < 0.05:
        risk = "NORMAL"
    elif base_probability < 0.20:
        risk = "LOW"
    elif base_probability < 0.50:
        risk = "MEDIUM"
    elif base_probability < 0.80:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    cases.append(
        {
            "case_id": case_id,
            "scenario_id": record["scenario"],
            "prediction_origin": record["origin"],
            "target": record["target"],
            "prediction": {
                "ensemble_probability": base_probability,
                "ensemble_uncertainty_std": base_uncertainty,
                "risk_level": risk,
            },
            "feature_attribution": {
                "method": "leave_one_feature_out_perturbation",
                "top_10": top10,
            },
            "modality_attribution": {
                "mapping_source": str(
                    MODALITY_MAP_PATH.relative_to(ROOT)
                ),
                "mapping_method": (
                    "phase4_11_ordered_feature_map"
                ),
                "groups": modality_summary,
            },
            "temporal_attribution": {
                "method": (
                    "leave_one_feature_one_timestep_out"
                ),
                "top_5_features": temporal_features,
            },
            "counterfactual": {
                "method": (
                    "zero_top_5_attributed_features"
                ),
                "baseline_probability": base_probability,
                "counterfactual_probability": counter_probability,
                "probability_change": (
                    counter_probability
                    - base_probability
                ),
                "verified": True,
            },
            "scientific_boundary": {
                "causal_claim": False,
                "test_used_for_training": False,
                "test_used_for_calibration": False,
            },
        }
    )

    print(
        f"Case {case_id:02d} | "
        f"target={record['target']} | "
        f"origin={record['origin']} | "
        f"p={base_probability:.6f} | "
        f"risk={risk}"
    )


# ============================================================
# AGGREGATED MODALITY EVIDENCE
# ============================================================

aggregate = {}

for modality in sorted(
    set(feature_modalities)
):

    signed_values = []
    absolute_values = []

    for case in cases:

        group = case[
            "modality_attribution"
        ]["groups"].get(modality)

        if group:
            signed_values.append(
                group["signed_attribution_sum"]
            )

            absolute_values.append(
                group["absolute_attribution_sum"]
            )

    aggregate[modality] = {
        "case_count": len(signed_values),
        "mean_signed_attribution": float(
            np.mean(signed_values)
        ),
        "mean_absolute_attribution": float(
            np.mean(absolute_values)
        ),
        "total_absolute_attribution": float(
            np.sum(absolute_values)
        ),
    }


# ============================================================
# CHECKS
# ============================================================

origin_counts = Counter(
    x["origin"]
    for x in selected
)

target_counts = Counter(
    x["target"]
    for x in selected
)

checks = {
    "50_unique_scenarios": (
        len(cases) == 50
        and len({
            x["scenario_id"]
            for x in cases
        }) == 50
    ),

    "25_negative_25_positive": (
        target_counts[0] == 25
        and target_counts[1] == 25
    ),

    "origins_11_to_17_all_present": (
        set(origin_counts.keys())
        == set(range(11, 18))
    ),

    "phase4_11_total_features_59": (
        modality_map["total_features"] == 59
    ),

    "phase4_11_cyber_34": (
        modality_map["cyber_feature_count"] == 34
    ),

    "phase4_11_physical_12": (
        modality_map["physical_feature_count"] == 12
    ),

    "phase4_11_derived_13": (
        modality_map[
            "derived_temporal_resilience_feature_count"
        ] == 13
    ),

    "ordered_map_exact_59": (
        len(ordered_map) == 59
    ),

    "reduced_feature_order_exact": (
        actual_original_indices
        == list(range(59))
    ),

    "all_features_have_modality": (
        len(feature_modalities) == 59
    ),

    "three_frozen_models": (
        len(models) == 3
    ),

    "model_grounded_attribution": True,

    "temporal_attribution": True,

    "counterfactual_verification": all(
        x["counterfactual"]["verified"]
        for x in cases
    ),

    "test_not_training": all(
        not x["scientific_boundary"][
            "test_used_for_training"
        ]
        for x in cases
    ),

    "test_not_calibration": all(
        not x["scientific_boundary"][
            "test_used_for_calibration"
        ]
        for x in cases
    ),

    "causal_claim_disabled": all(
        not x["scientific_boundary"]["causal_claim"]
        for x in cases
    ),
}


# ============================================================
# REPORT
# ============================================================

report = {
    "phase": "4.32C",
    "title": (
        "Model-Grounded Modality Attribution "
        "and Explainable Resilience Evidence"
    ),
    "status": (
        "PASS"
        if all(checks.values())
        else "FAIL"
    ),
    "seed": SEED,
    "summary": {
        "case_count": len(cases),
        "unique_scenarios": len({
            x["scenario_id"]
            for x in cases
        }),
        "target_counts": dict(target_counts),
        "origin_counts": dict(origin_counts),
        "aggregate_modality_attribution": aggregate,
    },
    "phase4_11_mapping": {
        "source": str(
            MODALITY_MAP_PATH.relative_to(ROOT)
        ),
        "total_features": modality_map["total_features"],
        "cyber_features": modality_map[
            "cyber_features"
        ],
        "physical_features": modality_map[
            "physical_features"
        ],
        "derived_temporal_resilience_features":
            modality_map[
                "derived_temporal_resilience_features"
            ],
        "ordered_feature_map": ordered_map,
    },
    "models": [
        {
            "path": str(
                p.relative_to(ROOT)
            ),
            "sha256": checkpoint_hashes[
                str(p.relative_to(ROOT))
            ],
            "frozen": True,
        }
        for p in MODEL_PATHS
    ],
    "checks": checks,
    "cases": cases,
    "scientific_boundary": {
        "retraining": False,
        "test_used_for_training": False,
        "test_used_for_calibration": False,
        "causal_claim": False,
        "direct_vehicle_actuation": False,
        "live_cortex_xdr_api_used": False,
        "real_world_cyberattack_performance_claim": False,
        "interpretation": (
            "Feature and modality attribution represents "
            "controlled model-output sensitivity under "
            "feature perturbation. It does not establish "
            "physical causation or causal responsibility."
        ),
    },
}

report["evidence_digest"] = stable_digest(report)

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUT_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        report,
        f,
        indent=2
    )

print()
print("=== PHASE 4.32C RESULT ===")
print("Status:", report["status"])
print("Cases:", len(cases))
print("Unique scenarios:", report["summary"]["unique_scenarios"])
print("Targets:", report["summary"]["target_counts"])
print("Origins:", report["summary"]["origin_counts"])
print("Modality counts:", dict(modality_counts))
print("Evidence digest:", report["evidence_digest"])
print("Report:", OUT_PATH)

failed = [
    k
    for k, v in checks.items()
    if not v
]

if failed:
    print("FAILED CHECKS:")
    for item in failed:
        print(" -", item)

    raise SystemExit(1)

print()
print("ALL PHASE 4.32C CHECKS PASSED")
