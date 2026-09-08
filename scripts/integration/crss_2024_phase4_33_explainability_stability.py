# Phase 4.33
# Explainability Stability and Consistency
# Frozen Phase 4.27A ensemble; no retraining.

from pathlib import Path
import json
import hashlib
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

X_PATH = ROOT / "data/processed/modeling/sequences/v2_forecasting/X_test_v2_forecasting.npy"
Y_PATH = ROOT / "data/processed/modeling/sequences/v2_forecasting/y_6step_test_v2_forecasting.npy"
SCENARIO_PATH = ROOT / "data/processed/modeling/sequences/v2_forecasting/scenario_ids_test_v2_forecasting.npy"
ORIGIN_PATH = ROOT / "data/processed/modeling/sequences/v2_forecasting/prediction_origins_test_v2_forecasting.npy"

MODEL_DIR = ROOT / "models/v2/uncertainty/phase4_27"

MODELS = [
    MODEL_DIR / "transformer_y6_seed_42_best.pt",
    MODEL_DIR / "transformer_y6_seed_52_best.pt",
    MODEL_DIR / "transformer_y6_seed_62_best.pt",
]

MODALITY_MAP_PATH = (
    ROOT / "experiments/modeling/v2/cross_modal/"
    "crss_2024_v2_modality_map.json"
)

OUT_DIR = ROOT / "experiments/integration/phase4_33"

REPORT_PATH = (
    OUT_DIR /
    "crss_2024_phase4_33_explainability_stability.json"
)

SEED = 4331
NUM_NEGATIVE = 25
NUM_POSITIVE = 25
TOP_K = 10
TOP_TEMPORAL_K = 5


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
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=3,
        )

        self.norm = nn.LayerNorm(128)
        self.attention = nn.Linear(128, 1)
        self.head = nn.Linear(128, 1)

    def forward(self, x):

        x = self.input_projection(x)
        x = x + self.position
        x = self.encoder(x)
        x = self.norm(x)

        weights = torch.softmax(
            self.attention(x),
            dim=1,
        )

        pooled = torch.sum(
            x * weights,
            dim=1,
        )

        return self.head(pooled).squeeze(-1)


def load_model(path):

    model = TemporalTransformer()

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.eval()

    return model


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def predict_probability(model, sample):

    tensor = torch.tensor(
        sample[None, :, :],
        dtype=torch.float32,
    )

    with torch.no_grad():

        probability = torch.sigmoid(
            model(tensor)
        )

    return float(
        probability.cpu().numpy()[0]
    )


def feature_attribution(model, sample):

    baseline = predict_probability(
        model,
        sample,
    )

    attribution = np.zeros(
        59,
        dtype=np.float64,
    )

    for feature in range(59):

        perturbed = sample.copy()

        perturbed[:, feature] = 0.0

        probability = predict_probability(
            model,
            perturbed,
        )

        attribution[feature] = (
            baseline - probability
        )

    return baseline, attribution


def temporal_attribution(
    model,
    sample,
    feature_indices,
):

    baseline = predict_probability(
        model,
        sample,
    )

    matrix = np.zeros(
        (12, len(feature_indices)),
        dtype=np.float64,
    )

    for t in range(12):

        for j, feature in enumerate(
            feature_indices
        ):

            perturbed = sample.copy()

            perturbed[t, feature] = 0.0

            probability = predict_probability(
                model,
                perturbed,
            )

            matrix[t, j] = (
                baseline - probability
            )

    return matrix


def deterministic_digest(payload):

    def json_safe(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        return value

    payload = json_safe(payload)

    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(text).hexdigest()


print("=== PHASE 4.33 EXPLAINABILITY STABILITY ===")

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

X_raw = np.load(X_PATH)
y = np.load(Y_PATH)

scenario_ids = np.load(
    SCENARIO_PATH,
    allow_pickle=True,
)

origins = np.load(
    ORIGIN_PATH
)

print("Raw X:", X_raw.shape)
print("Targets:", y.shape)

zero_variance_indices = [
    30,
    35,
    39,
    62,
]

keep_indices = [
    i
    for i in range(
        X_raw.shape[2]
    )
    if i not in zero_variance_indices
]

X = X_raw[:, :, keep_indices]

print("Model X:", X.shape)

if X.shape[2] != 59:
    raise ValueError(
        "Expected 59 model features."
    )

# ---------------------------------------------------------
# Modality map
# ---------------------------------------------------------

modality_map = json.loads(
    MODALITY_MAP_PATH.read_text(
        encoding="utf-8"
    )
)

ordered_map = modality_map[
    "ordered_feature_map"
]

if len(ordered_map) != 59:
    raise ValueError(
        "Phase 4.11 modality map must contain 59 entries."
    )

if sorted(
    item["original_index"]
    for item in ordered_map
) != list(range(59)):

    raise ValueError(
        "Phase 4.11 model-space indices are not 0..58."
    )

modalities = [
    item["modality"]
    for item in ordered_map
]

expected_modalities = Counter({
    "cyber": 34,
    "physical": 12,
    "derived_temporal_resilience": 13,
})

if Counter(modalities) != expected_modalities:
    raise ValueError(
        "Phase 4.11 modality counts do not match 34/12/13."
    )

print(
    "Modality counts:",
    dict(Counter(modalities))
)

# ---------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------

scenario_to_indices = {}

for i, scenario in enumerate(
    scenario_ids
):

    scenario = str(scenario)

    scenario_to_indices.setdefault(
        scenario,
        [],
    ).append(i)

scenario_targets = {}

for scenario, indices in (
    scenario_to_indices.items()
):

    scenario_targets[scenario] = int(
        np.any(
            y[indices] == 1
        )
    )

negative_scenarios = sorted(
    scenario
    for scenario, target
    in scenario_targets.items()
    if target == 0
)

positive_scenarios = sorted(
    scenario
    for scenario, target
    in scenario_targets.items()
    if target == 1
)

rng = random.Random(SEED)

rng.shuffle(
    negative_scenarios
)

rng.shuffle(
    positive_scenarios
)

selected_negative = (
    negative_scenarios[
        :NUM_NEGATIVE
    ]
)

selected_positive = (
    positive_scenarios[
        :NUM_POSITIVE
    ]
)

selected_scenarios = (
    selected_negative +
    selected_positive
)

# ---------------------------------------------------------
# Select unique scenarios with deterministic origin balance
# ---------------------------------------------------------

ORIGIN_VALUES = list(range(11, 18))

rng = random.Random(SEED)


def build_scenario_origin_rows(
    target_scenarios,
    target_label,
):
    """
    Build scenario -> origin -> candidate row mapping.

    The selected row must have the same target label as
    the scenario class being evaluated. This prevents a
    positive scenario from contributing a negative row
    merely because that row occurs at the desired origin.
    """

    mapping = {}

    for scenario in target_scenarios:
        mapping[scenario] = {
            origin: []
            for origin in ORIGIN_VALUES
        }

        for idx in scenario_to_indices[scenario]:

            # Enforce row-level target consistency.
            if int(y[idx]) != int(target_label):
                continue

            origin = int(origins[idx])

            if origin in mapping[scenario]:
                mapping[scenario][origin].append(
                    int(idx)
                )

        for origin in ORIGIN_VALUES:
            mapping[scenario][origin].sort()

    return mapping


def select_origin_balanced_class(
    target_scenarios,
    required_count,
    target_label,
):
    """
    Select exactly required_count unique scenarios.

    Selection is deterministic and explicitly attempts to
    cover prediction origins 11-17 before filling remaining
    slots. One row is selected per scenario.
    """

    target_label = int(target_label)

    scenario_origin_rows = (
        build_scenario_origin_rows(
            target_scenarios,
            target_label,
        )
    )

    eligible = [
        scenario
        for scenario in target_scenarios
        if any(
            scenario_origin_rows[scenario][origin]
            for origin in ORIGIN_VALUES
        )
    ]

    if len(eligible) < required_count:
        raise ValueError(
            f"Insufficient eligible scenarios: "
            f"required={required_count}, "
            f"eligible={len(eligible)}"
        )

    # Deterministic randomized order.
    ordered = list(eligible)
    rng.shuffle(ordered)

    selected = {}
    origin_counts = {
        origin: 0
        for origin in ORIGIN_VALUES
    }

    # -----------------------------------------------------
    # Pass 1: guarantee every origin is represented.
    # Prefer scenarios that are available at the requested
    # origin and do not already have a selected row.
    # -----------------------------------------------------

    for origin in ORIGIN_VALUES:

        candidates = [
            scenario
            for scenario in ordered
            if scenario not in selected
            and scenario_origin_rows[
                scenario
            ][origin]
        ]

        if not candidates:
            continue

        # Prefer scenarios having fewer alternative origins,
        # because they are less useful for later coverage.
        candidates.sort(
            key=lambda scenario: (
                sum(
                    bool(
                        scenario_origin_rows[
                            scenario
                        ][o]
                    )
                    for o in ORIGIN_VALUES
                ),
                ordered.index(scenario),
            )
        )

        scenario = candidates[0]

        selected[scenario] = (
            origin,
            scenario_origin_rows[
                scenario
            ][origin][0],
        )

        origin_counts[origin] += 1

    # -----------------------------------------------------
    # Pass 2: fill remaining slots while balancing origins.
    # -----------------------------------------------------

    remaining = [
        scenario
        for scenario in ordered
        if scenario not in selected
    ]

    while len(selected) < required_count:

        if not remaining:
            break

        best = None

        for scenario in remaining:

            available_origins = [
                origin
                for origin in ORIGIN_VALUES
                if scenario_origin_rows[
                    scenario
                ][origin]
            ]

            if not available_origins:
                continue

            origin = min(
                available_origins,
                key=lambda o: (
                    origin_counts[o],
                    abs(o - 14),
                    o,
                ),
            )

            candidate = (
                origin_counts[origin],
                len(available_origins),
                abs(origin - 14),
                ordered.index(scenario),
                scenario,
                origin,
            )

            if best is None or candidate < best:
                best = candidate

        if best is None:
            break

        (
            _,
            _,
            _,
            _,
            scenario,
            origin,
        ) = best

        row = scenario_origin_rows[
            scenario
        ][origin][0]

        selected[scenario] = (
            origin,
            row,
        )

        origin_counts[origin] += 1
        remaining.remove(scenario)

    if len(selected) != required_count:
        raise ValueError(
            f"Could not select {required_count} "
            f"unique scenarios; selected={len(selected)}"
        )

    return selected, origin_counts


# ---------------------------------------------------------
# Deterministic 25/25 class-balanced selection
# ---------------------------------------------------------

selected_negative_candidates = list(
    negative_scenarios
)

selected_positive_candidates = list(
    positive_scenarios
)

negative_selection, negative_origin_counts = (
    select_origin_balanced_class(
        selected_negative_candidates,
        NUM_NEGATIVE,
        0,
    )
)

positive_selection, positive_origin_counts = (
    select_origin_balanced_class(
        selected_positive_candidates,
        NUM_POSITIVE,
        1,
    )
)

selected_rows = {}

for scenario, (
    origin,
    idx,
) in negative_selection.items():

    selected_rows[scenario] = idx

for scenario, (
    origin,
    idx,
) in positive_selection.items():

    selected_rows[scenario] = idx

selected_scenarios = (
    list(negative_selection.keys())
    +
    list(positive_selection.keys())
)

if len(selected_rows) != 50:
    raise ValueError(
        f"Expected 50 selected scenarios; "
        f"got {len(selected_rows)}"
    )

selected_origin_counts = Counter(
    int(origins[idx])
    for idx in selected_rows.values()
)

if not all(
    origin in selected_origin_counts
    for origin in ORIGIN_VALUES
):
    raise ValueError(
        "Origin-balanced selection failed: "
        "not all origins 11-17 are represented."
    )

print(
    "Negative origin counts:",
    dict(
        sorted(
            negative_origin_counts.items()
        )
    ),
)

print(
    "Positive origin counts:",
    dict(
        sorted(
            positive_origin_counts.items()
        )
    ),
)

print(
    "Combined origin counts:",
    dict(
        sorted(
            selected_origin_counts.items()
        )
    ),
)
# ---------------------------------------------------------
# Load frozen ensemble
# ---------------------------------------------------------

models = []

checkpoint_hashes = {}

for path in MODELS:

    print(
        "Loading:",
        path.name,
    )

    model = load_model(path)

    models.append(model)

    checkpoint_hashes[
        path.name
    ] = sha256_file(path)

if len(models) != 3:

    raise ValueError(
        "Expected three frozen models."
    )

print(
    "Frozen ensemble:",
    len(models),
)

# ---------------------------------------------------------
# Case analysis
# ---------------------------------------------------------

cases = []

for case_number, scenario in enumerate(
    selected_scenarios,
    start=1,
):

    row = selected_rows[
        scenario
    ]

    sample = X[
        row
    ].astype(
        np.float32
    )

    target = int(
        y[row]
    )

    origin = int(
        origins[row]
    )

    print(
        f"Case {case_number:02d} | "
        f"target={target} | "
        f"origin={origin}"
    )

    model_probabilities = []

    model_attributions = []

    model_temporal = []

    for model in models:

        baseline, attribution = (
            feature_attribution(
                model,
                sample,
            )
        )

        top_features = (
            np.argsort(
                np.abs(attribution)
            )[::-1]
            [:TOP_K]
        )

        temporal = temporal_attribution(
            model,
            sample,
            top_features[
                :TOP_TEMPORAL_K
            ],
        )

        model_probabilities.append(
            baseline
        )

        model_attributions.append(
            attribution
        )

        model_temporal.append(
            temporal
        )

    model_probabilities = np.asarray(
        model_probabilities,
        dtype=np.float64,
    )

    model_attributions = np.asarray(
        model_attributions,
        dtype=np.float64,
    )

    model_temporal = np.asarray(
        model_temporal,
        dtype=np.float64,
    )

    ensemble_probability = float(
        np.mean(
            model_probabilities
        )
    )

    # -----------------------------------------------------
    # Feature stability
    # -----------------------------------------------------

    mean_attribution = np.mean(
        model_attributions,
        axis=0,
    )

    attribution_std = np.std(
        model_attributions,
        axis=0,
    )

    ensemble_rank = np.argsort(
        np.abs(
            mean_attribution
        )
    )[::-1]

    top_sets = [
        set(
            np.argsort(
                np.abs(
                    model_attributions[
                        model_index
                    ]
                )
            )[::-1][:TOP_K]
        )
        for model_index in range(3)
    ]

    jaccards = []

    for a in range(3):

        for b in range(
            a + 1,
            3,
        ):

            intersection = len(
                top_sets[a]
                & top_sets[b]
            )

            union = len(
                top_sets[a]
                | top_sets[b]
            )

            jaccards.append(
                (
                    intersection / union
                    if union
                    else 1.0
                )
            )

    top10_jaccard = float(
        np.mean(jaccards)
    )

    # -----------------------------------------------------
    # Modality stability
    # -----------------------------------------------------

    modality_scores = {}

    for modality in (
        "cyber",
        "physical",
        "derived_temporal_resilience",
    ):

        indices = [
            i
            for i, m in enumerate(
                modalities
            )
            if m == modality
        ]

        modality_scores[
            modality
        ] = {
            "feature_count":
                len(indices),

            "mean_absolute_attribution":
                float(
                    np.mean(
                        np.abs(
                            mean_attribution[
                                indices
                            ]
                        )
                    )
                ),

            "sum_absolute_attribution":
                float(
                    np.sum(
                        np.abs(
                            mean_attribution[
                                indices
                            ]
                        )
                    )
                ),
        }

    # -----------------------------------------------------
    # Temporal stability
    # -----------------------------------------------------

    temporal_mean = np.mean(
        model_temporal,
        axis=0,
    )

    temporal_step_scores = np.sum(
        np.abs(
            temporal_mean
        ),
        axis=1,
    )

    top_temporal_steps = (
        np.argsort(
            temporal_step_scores
        )[::-1]
        [:TOP_TEMPORAL_K]
    )

    # -----------------------------------------------------
    # Counterfactual stability
    # -----------------------------------------------------

    top5 = ensemble_rank[:5]

    counterfactual = sample.copy()

    counterfactual[
        :,
        top5,
    ] = 0.0

    counterfactual_probabilities = []

    for model in models:

        counterfactual_probabilities.append(
            predict_probability(
                model,
                counterfactual,
            )
        )

    counterfactual_probability = float(
        np.mean(
            counterfactual_probabilities
        )
    )

    counterfactual_delta = float(
        ensemble_probability
        - counterfactual_probability
    )

    cases.append(
        {
            "case_number":
                case_number,

            "scenario_id":
                scenario,

            "row_index":
                int(row),

            "target":
                target,

            "prediction_origin":
                origin,

            "ensemble_probability":
                ensemble_probability,

            "model_probabilities":
                model_probabilities.tolist(),

            "probability_std":
                float(
                    np.std(
                        model_probabilities
                    )
                ),

            "feature_attribution": {
                "method":
                    "leave_one_feature_out_perturbation",

                "feature_count":
                    59,

                "ensemble_top_10_features":
                    ensemble_rank[
                        :TOP_K
                    ].tolist(),

                "ensemble_top_10_absolute_attribution":
                    np.abs(
                        mean_attribution[
                            ensemble_rank[
                                :TOP_K
                            ]
                        ]
                    ).tolist(),

                "member_top_10_sets":
                    [
                        sorted(
                            list(x)
                        )
                        for x in top_sets
                    ],

                "top_10_jaccard_mean":
                    top10_jaccard,

                "mean_absolute_attribution_std":
                    float(
                        np.mean(
                            attribution_std
                        )
                    ),

                "max_absolute_attribution_std":
                    float(
                        np.max(
                            attribution_std
                        )
                    ),
            },

            "modality_stability": {
                "groups":
                    modality_scores,
            },

            "temporal_stability": {
                "method":
                    "leave_one_feature_one_timestep_out",

                "top_features":
                    top5.tolist(),

                "top_temporal_steps":
                    top_temporal_steps.tolist(),

                "temporal_step_scores":
                    temporal_step_scores.tolist(),
            },

            "counterfactual_stability": {
                "method":
                    "zero_top_5_ensemble_features",

                "top_5_features":
                    top5.tolist(),

                "original_probability":
                    ensemble_probability,

                "counterfactual_probability":
                    counterfactual_probability,

                "probability_delta":
                    counterfactual_delta,

                "verified":
                    True,
            },

            "scientific_boundary": {
                "causal_claim":
                    False,

                "test_used_for_training":
                    False,

                "test_used_for_calibration":
                    False,

                "live_cortex_xdr_api_used":
                    False,

                "direct_vehicle_actuation":
                    False,

                "real_world_cyberattack_performance_claim":
                    False,
            },
        }
    )

# ---------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------

jaccard_values = [
    case[
        "feature_attribution"
    ][
        "top_10_jaccard_mean"
    ]
    for case in cases
]

probability_std_values = [
    case[
        "probability_std"
    ]
    for case in cases
]

attribution_std_values = [
    case[
        "feature_attribution"
    ][
        "mean_absolute_attribution_std"
    ]
    for case in cases
]

feature_frequency = Counter()

for case in cases:

    feature_frequency.update(
        case[
            "feature_attribution"
        ][
            "ensemble_top_10_features"
        ]
    )

modality_totals = Counter()

for case in cases:

    for modality, data in (
        case[
            "modality_stability"
        ][
            "groups"
        ].items()
    ):

        modality_totals[
            modality
        ] += data[
            "sum_absolute_attribution"
        ]

# ---------------------------------------------------------
# Validation gates
# ---------------------------------------------------------

checks = {}

checks[
    "50_cases"
] = len(cases) == 50

checks[
    "25_negative_25_positive"
] = (
    Counter(
        int(case["target"])
        for case in cases
    )
    == Counter({
        0: 25,
        1: 25,
    })
)

checks[
    "50_unique_scenarios"
] = (
    len(
        set(
            case["scenario_id"]
            for case in cases
        )
    )
    == 50
)

checks[
    "origins_11_to_17_present"
] = (
    set(
        case["prediction_origin"]
        for case in cases
    )
    == set(range(11, 18))
)

checks[
    "three_frozen_models"
] = len(models) == 3

checks[
    "59_features"
] = all(
    case[
        "feature_attribution"
    ][
        "feature_count"
    ] == 59
    for case in cases
)

checks[
    "model_grounded_attribution"
] = all(
    case[
        "feature_attribution"
    ][
        "method"
    ]
    == "leave_one_feature_out_perturbation"
    for case in cases
)

checks[
    "temporal_attribution"
] = all(
    case[
        "temporal_stability"
    ][
        "method"
    ]
    == "leave_one_feature_one_timestep_out"
    for case in cases
)

checks[
    "counterfactual_verification"
] = all(
    case[
        "counterfactual_stability"
    ][
        "verified"
    ]
    for case in cases
)

checks[
    "modality_mapping_59"
] = (
    len(modalities) == 59
)

checks[
    "modality_mapping_34_12_13"
] = (
    Counter(modalities)
    == expected_modalities
)

checks[
    "test_not_training"
] = all(
    not case[
        "scientific_boundary"
    ][
        "test_used_for_training"
    ]
    for case in cases
)

checks[
    "test_not_calibration"
] = all(
    not case[
        "scientific_boundary"
    ][
        "test_used_for_calibration"
    ]
    for case in cases
)

checks[
    "causal_claim_disabled"
] = all(
    not case[
        "scientific_boundary"
    ][
        "causal_claim"
    ]
    for case in cases
)

checks[
    "vehicle_actuation_disabled"
] = all(
    not case[
        "scientific_boundary"
    ][
        "direct_vehicle_actuation"
    ]
    for case in cases
)

checks[
    "live_cortex_xdr_disabled"
] = all(
    not case[
        "scientific_boundary"
    ][
        "live_cortex_xdr_api_used"
    ]
    for case in cases
)

checks[
    "real_world_attack_claim_disabled"
] = all(
    not case[
        "scientific_boundary"
    ][
        "real_world_cyberattack_performance_claim"
    ]
    for case in cases
)

# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

report = {

    "phase":
        "4.33",

    "title":
        "Explainability Stability and Consistency",

    "status":
        "PASS"
        if all(checks.values())
        else "FAIL",

    "research_simulation":
        True,

    "retraining_performed":
        False,

    "sample_design": {

        "cases":
            len(cases),

        "unique_scenarios":
            len(
                set(
                    case["scenario_id"]
                    for case in cases
                )
            ),

        "targets":
            dict(
                Counter(
                    int(case["target"])
                    for case in cases
                )
            ),

        "origins":
            dict(
                sorted(
                    Counter(
                        case[
                            "prediction_origin"
                        ]
                        for case in cases
                    ).items()
                )
            ),

        "seed":
            SEED,

        "one_row_per_scenario":
            True,
    },

    "ensemble": {

        "member_count":
            3,

        "seeds":
            [
                42,
                52,
                62,
            ],

        "frozen":
            True,

        "checkpoint_hashes":
            checkpoint_hashes,
    },

    "stability_summary": {

        "mean_top_10_jaccard":
            float(
                np.mean(
                    jaccard_values
                )
            ),

        "median_top_10_jaccard":
            float(
                np.median(
                    jaccard_values
                )
            ),

        "minimum_top_10_jaccard":
            float(
                np.min(
                    jaccard_values
                )
            ),

        "mean_probability_std":
            float(
                np.mean(
                    probability_std_values
                )
            ),

        "maximum_probability_std":
            float(
                np.max(
                    probability_std_values
                )
            ),

        "mean_attribution_absolute_std":
            float(
                np.mean(
                    attribution_std_values
                )
            ),

        "maximum_attribution_absolute_std":
            float(
                np.max(
                    attribution_std_values
                )
            ),

        "most_frequent_top_features":
            [
                {
                    "feature_index":
                        int(feature),

                    "top10_case_count":
                        int(count),
                }

                for feature, count
                in feature_frequency.most_common(15)
            ],

        "modality_total_absolute_attribution":
            dict(
                sorted(
                    (
                        (
                            modality,
                            float(value),
                        )

                        for modality, value
                        in modality_totals.items()
                    ),

                    key=lambda item:
                        item[1],

                    reverse=True,
                )
            ),
    },

    "modality_definition": {

        "source":
            "Phase 4.11 modality map",

        "total_features":
            59,

        "cyber":
            34,

        "physical":
            12,

        "derived_temporal_resilience":
            13,
    },

    "checks":
        checks,

    "scientific_boundary": {

        "causal_claim":
            False,

        "test_used_for_training":
            False,

        "test_used_for_calibration":
            False,

        "live_cortex_xdr_api_used":
            False,

        "direct_vehicle_actuation":
            False,

        "real_world_cyberattack_performance_claim":
            False,
    },

    "cases":
        cases,
}

report[
    "evidence_digest"
] = deterministic_digest(
    report
)

REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
        default=lambda obj: (
            obj.item()
            if isinstance(obj, np.generic)
            else obj.tolist()
            if isinstance(obj, np.ndarray)
            else str(obj)
        ),
    ),
    encoding="utf-8",
)

failed = [
    key
    for key, value
    in checks.items()
    if not value
]

print()
print("=== PHASE 4.33 RESULT ===")
print("Status:", report["status"])
print("Cases:", len(cases))

print(
    "Targets:",
    dict(
        Counter(
            int(case["target"])
            for case in cases
        )
    )
)

print(
    "Origins:",
    dict(
        sorted(
            Counter(
                case[
                    "prediction_origin"
                ]
                for case in cases
            ).items()
        )
    )
)

print(
    "Mean top-10 Jaccard:",
    report[
        "stability_summary"
    ][
        "mean_top_10_jaccard"
    ]
)

print(
    "Median top-10 Jaccard:",
    report[
        "stability_summary"
    ][
        "median_top_10_jaccard"
    ]
)

print(
    "Mean probability std:",
    report[
        "stability_summary"
    ][
        "mean_probability_std"
    ]
)

print(
    "Failed checks:",
    failed
)

print(
    "Evidence digest:",
    report[
        "evidence_digest"
    ]
)

print(
    "Report:",
    REPORT_PATH
)

if failed:

    raise SystemExit(
        "PHASE 4.33 FAILED"
    )

print()
print(
    "ALL PHASE 4.33 CHECKS PASSED"
)







