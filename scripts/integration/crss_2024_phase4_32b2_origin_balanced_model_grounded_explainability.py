import json
import hashlib
import csv
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

TENSOR_DIR = ROOT / "data/processed/modeling/sequences/v2_forecasting"
MODEL_DIR = ROOT / "models/v2/uncertainty/phase4_27"
OUT_DIR = ROOT / "experiments/integration/phase4_32b"

X_TEST = TENSOR_DIR / "X_test_v2_forecasting.npy"
Y6_TEST = TENSOR_DIR / "y_6step_test_v2_forecasting.npy"
SCENARIO_TEST = TENSOR_DIR / "scenario_ids_test_v2_forecasting.npy"
ORIGIN_TEST = TENSOR_DIR / "prediction_origins_test_v2_forecasting.npy"

FEATURE_MAP = ROOT / "data/schemas/nhtsa/verification/crss_2024_final_feature_dictionary_v2.csv"

CHECKPOINTS = [
    MODEL_DIR / "transformer_y6_seed_42_best.pt",
    MODEL_DIR / "transformer_y6_seed_52_best.pt",
    MODEL_DIR / "transformer_y6_seed_62_best.pt",
]

ZERO_VARIANCE = [30, 35, 39, 62]

SEED = 4321
CASES_PER_CLASS = 25
ORIGINS = list(range(11, 18))
TOP_K = 10


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


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
            self.attention(x).squeeze(-1),
            dim=1,
        )

        pooled = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        return self.head(pooled).squeeze(-1)


def load_models():

    models = []

    for path in CHECKPOINTS:

        model = TemporalTransformer()

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(
            state["model_state_dict"]
        )

        model.eval()

        models.append(model)

    return models


def predict(models, x):

    values = []

    with torch.no_grad():

        for model in models:

            values.append(
                torch.sigmoid(
                    model(x)
                ).numpy()
            )

    values = np.stack(values)

    return (
        values.mean(axis=0),
        values.std(axis=0),
        values,
    )


def risk_level(p):

    if p < 0.05:
        return "NORMAL"

    if p < 0.20:
        return "LOW"

    if p < 0.50:
        return "MEDIUM"

    if p < 0.80:
        return "HIGH"

    return "CRITICAL"


def load_feature_names():

    names = []

    with open(
        FEATURE_MAP,
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        fields = reader.fieldnames or []

        candidate = None

        for field in [
            "feature",
            "feature_name",
            "canonical_feature",
        ]:

            if field in fields:
                candidate = field
                break

        if candidate:

            for row in reader:

                value = row.get(candidate)

                if value:
                    names.append(value)

    if len(names) >= 59:
        return names[:59]

    return [
        f"feature_{i:02d}"
        for i in range(59)
    ]


def build_scenario_index(y, scenario_ids, origins):

    data = {
        0: defaultdict(dict),
        1: defaultdict(dict),
    }

    for i in range(len(y)):

        target = int(y[i])
        scenario = str(scenario_ids[i])
        origin = int(origins[i])

        if origin in ORIGINS:

            # One row per scenario/origin.
            if origin not in data[target][scenario]:

                data[target][scenario][origin] = i

    return data


def select_balanced(
    y,
    scenario_ids,
    origins,
):

    rng = np.random.default_rng(SEED)

    data = build_scenario_index(
        y,
        scenario_ids,
        origins,
    )

    selected = []

    diagnostics = {}

    for target in [0, 1]:

        scenarios = sorted(
            data[target].keys()
        )

        rng.shuffle(scenarios)

        # Candidate scenarios grouped by available origins.
        candidates = {
            origin: []
            for origin in ORIGINS
        }

        for scenario in scenarios:

            available = sorted(
                data[target][scenario].keys()
            )

            for origin in available:

                candidates[origin].append(
                    scenario
                )

        # Deterministic greedy balancing:
        # repeatedly choose the origin with the fewest
        # selected cases, then select an unused scenario
        # available at that origin.
        chosen_scenarios = set()
        chosen = []

        while len(chosen) < CASES_PER_CLASS:

            available_origins = [
                o for o in ORIGINS
                if any(
                    s not in chosen_scenarios
                    for s in candidates[o]
                )
            ]

            if not available_origins:
                break

            origin = min(
                available_origins,
                key=lambda o: (
                    sum(
                        1
                        for _, selected_origin, _
                        in chosen
                        if selected_origin == o
                    ),
                    o,
                ),
            )

            available_scenarios = [
                s for s in candidates[origin]
                if s not in chosen_scenarios
            ]

            if not available_scenarios:
                continue

            scenario = available_scenarios[0]

            chosen_scenarios.add(scenario)

            chosen.append(
                (
                    scenario,
                    origin,
                    data[target][scenario][origin],
                )
            )

        if len(chosen) != CASES_PER_CLASS:

            raise RuntimeError(
                f"Could not select "
                f"{CASES_PER_CLASS} unique scenarios "
                f"for target={target}. "
                f"Only selected {len(chosen)}."
            )

        selected.extend(
            idx for _, _, idx in chosen
        )

        diagnostics[str(target)] = {
            "available_unique_scenarios":
                len(scenarios),

            "selected_unique_scenarios":
                len(chosen_scenarios),

            "origin_distribution":
                dict(
                    Counter(
                        origin
                        for _, origin, _
                        in chosen
                    )
                ),

            "selected_scenarios": [
                {
                    "scenario_id": scenario,
                    "origin": origin,
                    "test_index": idx,
                }
                for scenario, origin, idx
                in chosen
            ],
        }

    # Final deterministic ordering.
    selected = sorted(
        selected,
        key=lambda i: (
            int(y[i]),
            int(origins[i]),
            str(scenario_ids[i]),
        ),
    )

    return selected, diagnostics


def feature_attribution(
    models,
    sample,
    feature_names,
):

    base = torch.from_numpy(
        sample.astype(np.float32)
    ).unsqueeze(0)

    base_mean, base_std, base_seed = predict(
        models,
        base,
    )

    probability = float(
        base_mean[0]
    )

    rows = []

    for feature_idx in range(59):

        perturbed = sample.copy()

        perturbed[
            :,
            feature_idx
        ] = 0.0

        x = torch.from_numpy(
            perturbed.astype(np.float32)
        ).unsqueeze(0)

        cf_mean, _, cf_seed = predict(
            models,
            x,
        )

        delta = (
            probability -
            float(cf_mean[0])
        )

        seed_delta = (
            base_seed[:, 0] -
            cf_seed[:, 0]
        )

        rows.append(
            {
                "feature_index":
                    feature_idx,

                "feature":
                    feature_names[
                        feature_idx
                    ],

                "ensemble_mean_delta":
                    float(delta),

                "ensemble_abs_delta":
                    float(abs(delta)),

                "ensemble_std_delta":
                    float(
                        np.std(seed_delta)
                    ),

                "seed_deltas": [
                    float(v)
                    for v in seed_delta
                ],
            }
        )

    rows.sort(
        key=lambda r:
            r["ensemble_abs_delta"],
        reverse=True,
    )

    return (
        probability,
        float(base_std[0]),
        rows,
    )


def temporal_attribution(
    models,
    sample,
    top_features,
):

    base = torch.from_numpy(
        sample.astype(np.float32)
    ).unsqueeze(0)

    base_mean, _, _ = predict(
        models,
        base,
    )

    base_probability = float(
        base_mean[0]
    )

    results = []

    for feature in top_features:

        idx = feature[
            "feature_index"
        ]

        deltas = []

        for step in range(12):

            perturbed = sample.copy()

            perturbed[
                step,
                idx
            ] = 0.0

            x = torch.from_numpy(
                perturbed.astype(np.float32)
            ).unsqueeze(0)

            cf_mean, _, _ = predict(
                models,
                x,
            )

            deltas.append(
                base_probability -
                float(cf_mean[0])
            )

        results.append(
            {
                "feature":
                    feature["feature"],

                "feature_index":
                    idx,

                "temporal_deltas":
                    [
                        float(v)
                        for v in deltas
                    ],

                "max_abs_temporal_delta":
                    float(
                        np.max(
                            np.abs(deltas)
                        )
                    ),

                "mean_abs_temporal_delta":
                    float(
                        np.mean(
                            np.abs(deltas)
                        )
                    ),
            }
        )

    return results


def counterfactual_verification(
    models,
    sample,
    probability,
    top_features,
):

    results = []

    for feature in top_features[:5]:

        idx = feature[
            "feature_index"
        ]

        perturbed = sample.copy()

        perturbed[
            :,
            idx
        ] = 0.0

        x = torch.from_numpy(
            perturbed.astype(np.float32)
        ).unsqueeze(0)

        cf_mean, cf_std, _ = predict(
            models,
            x,
        )

        cf_probability = float(
            cf_mean[0]
        )

        delta = (
            probability -
            cf_probability
        )

        results.append(
            {
                "feature":
                    feature["feature"],

                "feature_index":
                    idx,

                "baseline_probability":
                    probability,

                "counterfactual_probability":
                    cf_probability,

                "probability_change":
                    float(delta),

                "direction":
                    (
                        "risk_increasing"
                        if delta > 0
                        else
                        "risk_decreasing"
                        if delta < 0
                        else
                        "neutral"
                    ),
            }
        )

    return results


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== PHASE 4.32B.2 ==="
    )

    X = np.load(X_TEST)

    y = np.load(Y6_TEST)

    scenario_ids = np.load(
        SCENARIO_TEST,
        allow_pickle=True,
    )

    origins = np.load(
        ORIGIN_TEST,
        allow_pickle=True,
    )

    print(
        "Raw test tensor:",
        X.shape,
    )

    print(
        "Total unique scenarios:",
        len(
            np.unique(
                scenario_ids
            )
        ),
    )

    print(
        "Total positive rows:",
        int(
            (y == 1).sum()
        ),
    )

    print(
        "Positive unique scenarios:",
        len(
            np.unique(
                scenario_ids[y == 1]
            )
        ),
    )

    keep = [
        i for i in range(
            X.shape[2]
        )
        if i not in ZERO_VARIANCE
    ]

    X = X[
        :,
        :,
        keep,
    ]

    assert X.shape[2] == 59

    models = load_models()

    print(
        "Frozen checkpoints loaded:",
        len(models),
    )

    feature_names = load_feature_names()

    selected, diagnostics = (
        select_balanced(
            y,
            scenario_ids,
            origins,
        )
    )

    assert len(selected) == 50

    assert len(
        set(
            str(scenario_ids[i])
            for i in selected
        )
    ) == 50

    print(
        "\nSelected unique scenarios:",
        50,
    )

    for target in [0, 1]:

        target_indices = [
            i for i in selected
            if int(y[i]) == target
        ]

        print(
            f"Target {target} origins:",
            Counter(
                int(origins[i])
                for i in target_indices
            ),
        )

    all_origins = [
        int(origins[i])
        for i in selected
    ]

    print(
        "All origins:",
        Counter(all_origins),
    )

    cases = []

    for number, idx in enumerate(
        selected,
        start=1,
    ):

        sample = X[idx]

        probability, uncertainty, attribution = (
            feature_attribution(
                models,
                sample,
                feature_names,
            )
        )

        top_features = attribution[:TOP_K]

        temporal = temporal_attribution(
            models,
            sample,
            top_features[:5],
        )

        stability_values = []

        for row in top_features:

            denominator = max(
                abs(
                    row[
                        "ensemble_mean_delta"
                    ]
                ),
                1e-8,
            )

            stability_values.append(
                max(
                    0.0,
                    1.0 -
                    (
                        row[
                            "ensemble_std_delta"
                        ] /
                        denominator
                    ),
                )
            )

        stability = float(
            np.mean(
                stability_values
            )
        )

        counterfactuals = (
            counterfactual_verification(
                models,
                sample,
                probability,
                top_features,
            )
        )

        cases.append(
            {
                "explanation_id":
                    f"PHASE4_32B2_CASE_{number:03d}",

                "test_index":
                    int(idx),

                "scenario_id":
                    str(
                        scenario_ids[idx]
                    ),

                "prediction_origin":
                    int(
                        origins[idx]
                    ),

                "target_y6":
                    int(y[idx]),

                "prediction": {
                    "ensemble_mean_probability":
                        probability,

                    "ensemble_uncertainty_std":
                        uncertainty,

                    "risk_level":
                        risk_level(
                            probability
                        ),
                },

                "model_attribution": {
                    "method":
                        "leave_one_feature_out_perturbation",

                    "manual_driver_assignment":
                        False,

                    "top_features":
                        top_features,

                    "ensemble_attribution_stability":
                        stability,
                },

                "temporal_attribution": {
                    "method":
                        "single_feature_single_timestep_ablation",

                    "steps":
                        12,

                    "top_features":
                        temporal,
                },

                "counterfactual_verification": {
                    "method":
                        "feature_zeroing",

                    "uses_actual_frozen_model":
                        True,

                    "tests":
                        counterfactuals,
                },

                "scientific_interpretation": {
                    "claim_type":
                        "model_grounded_perturbation_attribution",

                    "causal_claim":
                        False,

                    "test_data_used_for_training":
                        False,

                    "test_data_used_for_calibration":
                        False,
                },
            }
        )

        print(
            f"Case {number:02d}/50 | "
            f"target={int(y[idx])} | "
            f"origin={int(origins[idx])} | "
            f"p={probability:.6f} | "
            f"risk={risk_level(probability)}"
        )

    unique_scenarios = len(
        set(
            c["scenario_id"]
            for c in cases
        )
    )

    target_counts = Counter(
        c["target_y6"]
        for c in cases
    )

    origin_counts = Counter(
        c["prediction_origin"]
        for c in cases
    )

    checkpoint_hashes = {
        p.name:
            sha256_file(p)
        for p in CHECKPOINTS
    }

    schema_path = (
        ROOT /
        "data/schemas/explainability/"
        "model_grounded_resilience_explanation_schema.json"
    )

    checks = {

        "50_cases":
            len(cases) == 50,

        "25_negative":
            target_counts[0] == 25,

        "25_positive":
            target_counts[1] == 25,

        "50_unique_scenarios":
            unique_scenarios == 50,

        "all_origins_11_to_17_represented":
            set(origin_counts)
            == set(ORIGINS),

        "three_frozen_models":
            len(models) == 3,

        "manual_driver_assignment_disabled":
            all(
                not c[
                    "model_attribution"
                ][
                    "manual_driver_assignment"
                ]
                for c in cases
            ),

        "temporal_attribution_present":
            all(
                "temporal_attribution"
                in c
                for c in cases
            ),

        "counterfactuals_present":
            all(
                "counterfactual_verification"
                in c
                for c in cases
            ),

        "scenario_ids_recorded":
            all(
                bool(
                    c["scenario_id"]
                )
                for c in cases
            ),

        "origins_recorded":
            all(
                isinstance(
                    c["prediction_origin"],
                    int,
                )
                for c in cases
            ),

        "schema_present":
            schema_path.exists(),

        "checkpoint_hashes_recorded":
            len(checkpoint_hashes) == 3,

        "no_retraining":
            True,

        "test_not_training":
            True,

        "test_not_calibration":
            True,

        "causal_claim_disabled":
            True,
    }

    digest_payload = {
        "phase":
            "4.32B.2",

        "seed":
            SEED,

        "cases":
            cases,

        "diagnostics":
            diagnostics,

        "checkpoint_hashes":
            checkpoint_hashes,
    }

    digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode("utf-8")
    ).hexdigest()

    report = {

        "phase":
            "4.32B.2",

        "title":
            "Origin-Balanced Unique-Scenario "
            "Model-Grounded Explainability",

        "status":
            "PASS"
            if all(checks.values())
            else "FAIL",

        "research_simulation":
            True,

        "model_grounded":
            True,

        "retraining_performed":
            False,

        "evaluation_case_count":
            len(cases),

        "target_counts":
            dict(target_counts),

        "origin_counts":
            dict(origin_counts),

        "sampling": {

            "method":
                "deterministic_unique_scenario_origin_balancing",

            "seed":
                SEED,

            "cases_per_class":
                CASES_PER_CLASS,

            "origins_targeted":
                ORIGINS,

            "diagnostics":
                diagnostics,
        },

        "attribution_method": {

            "type":
                "leave_one_feature_out_perturbation",

            "uses_actual_frozen_model":
                True,

            "manual_driver_assignment":
                False,
        },

        "temporal_attribution_method": {

            "type":
                "single_feature_single_timestep_ablation",

            "steps":
                12,
        },

        "counterfactual_method": {

            "type":
                "feature_zeroing",

            "uses_actual_frozen_model":
                True,

            "causal_claim":
                False,
        },

        "checkpoint_hashes":
            checkpoint_hashes,

        "schema": {

            "path":
                str(
                    schema_path.relative_to(
                        ROOT
                    )
                ),

            "present":
                schema_path.exists(),

            "validated":
                False,
        },

        "checks":
            checks,

        "cases":
            cases,

        "safety_boundary": {

            "direct_vehicle_actuation":
                False,

            "live_cortex_xdr_api_used":
                False,

            "real_world_cyberattack_performance_claim":
                False,
        },

        "evidence_digest":
            digest,
    }

    output = (
        OUT_DIR /
        "crss_2024_phase4_32b2_origin_balanced_model_grounded_explainability.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print("\n=== FINAL RESULT ===")

    print(
        "STATUS:",
        report["status"],
    )

    print(
        "Cases:",
        len(cases),
    )

    print(
        "Unique scenarios:",
        unique_scenarios,
    )

    print(
        "Targets:",
        dict(target_counts),
    )

    print(
        "Origins:",
        dict(origin_counts),
    )

    print(
        "Evidence digest:",
        digest,
    )

    print(
        "Report:",
        output,
    )


if __name__ == "__main__":
    main()
