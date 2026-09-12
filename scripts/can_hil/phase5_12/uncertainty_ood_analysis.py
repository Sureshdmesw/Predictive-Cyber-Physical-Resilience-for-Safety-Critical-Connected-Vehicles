from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ================================================================
# PHASE 5.12
# UNCERTAINTY & OOD ANALYSIS
# ================================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_10"
    / "phase5_10_cyber_physical_attack_scenarios.csv"
)

CHECKPOINT = (
    ROOT
    / "models"
    / "v2"
    / "advanced"
    / "crss_2024_v2_temporal_transformer_best.pt"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_12"
)

REPORT_DIR = (
    ROOT
    / "experiments"
    / "can_hil"
    / "phase5_12"
)

OUTPUT_CSV = OUTPUT_DIR / "phase5_12_uncertainty_ood.csv"
OUTPUT_JSONL = OUTPUT_DIR / "phase5_12_uncertainty_ood.jsonl"
SCHEMA = OUTPUT_DIR / "phase5_12_uncertainty_ood_schema.json"
MANIFEST = REPORT_DIR / "phase5_12_uncertainty_ood_manifest.json"

OBSERVATION_STEPS = 12
RECORDS_PER_TIMESTEP = 15
EXPECTED_TIMESTEPS = 600
EXPECTED_WINDOWS = 589
EXPECTED_SCENARIOS = 11
EXPECTED_ATTACK_SCENARIOS = 10
FEATURE_COUNT = 59
MODEL_PARAMS = 407811

BASELINE_ID = "S00_NORMAL_BASELINE"

PHASE57_DIR = (
    ROOT
    / "scripts"
    / "can_hil"
    / "phase5_7"
)

sys.path.insert(0, str(PHASE57_DIR))

from can_ev_frozen_transformer_integration import (
    FEATURES,
    derive_model_compatible_features,
    TemporalTransformer,
)


# ================================================================
# HELPERS
# ================================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def finite_float(value, default=0.0):
    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except Exception:
        pass

    return float(default)


def parse_signals(value):
    if isinstance(value, dict):
        return value

    try:
        parsed = json.loads(str(value))

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    return {}


def expand_signals(row):
    result = row.copy()

    signals = parse_signals(
        row.get("signals_json", "{}")
    )

    for key, value in signals.items():
        result[key] = finite_float(value)

    return result


# ================================================================
# FEATURE CONSTRUCTION
# ================================================================

def build_feature_timestep(group):
    feature_rows = []

    for _, row in group.iterrows():
        expanded = expand_signals(row)

        features = derive_model_compatible_features(
            expanded
        )

        feature_rows.append(
            [
                finite_float(
                    features.get(name, 0.0)
                )
                for name in FEATURES
            ]
        )

    matrix = np.asarray(
        feature_rows,
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Invalid feature matrix."
        )

    if matrix.shape[0] != RECORDS_PER_TIMESTEP:
        raise ValueError(
            "Unexpected CAN records per timestep: "
            f"{matrix.shape[0]}"
        )

    if matrix.shape[1] != FEATURE_COUNT:
        raise ValueError(
            "Unexpected feature count: "
            f"{matrix.shape[1]}"
        )

    return matrix.mean(axis=0)


def build_stream(df):
    if len(df) != 9000:
        raise ValueError(
            f"Expected 9000 records, got {len(df)}."
        )

    timesteps = []
    timestamp_spread = []

    for start in range(
        0,
        len(df),
        RECORDS_PER_TIMESTEP,
    ):
        group = df.iloc[
            start:start + RECORDS_PER_TIMESTEP
        ]

        if len(group) != RECORDS_PER_TIMESTEP:
            raise ValueError(
                "Incomplete timestep."
            )

        timesteps.append(
            build_feature_timestep(group)
        )

        timestamps = pd.to_numeric(
            group["timestamp_s"],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        timestamps = timestamps[
            np.isfinite(timestamps)
        ]

        if len(timestamps) > 0:
            timestamp_spread.append(
                float(
                    timestamps.max()
                    - timestamps.min()
                )
            )
        else:
            timestamp_spread.append(0.0)

    X = np.asarray(
        timesteps,
        dtype=np.float32,
    )

    if X.shape != (
        EXPECTED_TIMESTEPS,
        FEATURE_COUNT,
    ):
        raise ValueError(
            "Unexpected stream tensor shape: "
            f"{X.shape}"
        )

    return (
        X,
        np.asarray(
            timestamp_spread,
            dtype=np.float32,
        ),
    )


def make_windows(X):
    windows = []

    for index in range(
        len(X) - OBSERVATION_STEPS + 1
    ):
        windows.append(
            X[
                index:
                index + OBSERVATION_STEPS
            ]
        )

    result = np.asarray(
        windows,
        dtype=np.float32,
    )

    if result.shape != (
        EXPECTED_WINDOWS,
        OBSERVATION_STEPS,
        FEATURE_COUNT,
    ):
        raise ValueError(
            "Unexpected window tensor shape: "
            f"{result.shape}"
        )

    return result


# ================================================================
# FROZEN MODEL
# ================================================================

def initialize_missing_norms(model):
    """
    The existing frozen checkpoint lacks six normalization tensors.

    These layers are initialized as identity transformations:
        weight = 1
        bias   = 0

    This prevents random initialization from changing inference
    behavior when those historical tensors are absent.
    """

    for layer_name in (
        "input_norm",
        "pool_norm",
        "output_norm",
    ):
        layer = getattr(model, layer_name)

        with torch.no_grad():
            layer.weight.fill_(1.0)
            layer.bias.zero_()


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        return checkpoint

    return checkpoint


def normalize_state_dict_keys(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[7:]] = value
        else:
            cleaned[key] = value

    return cleaned


def load_model():
    model = TemporalTransformer(
        input_dim=FEATURE_COUNT,
        embedding_dim=128,
        num_layers=3,
        num_heads=8,
        ff_dim=256,
        dropout=0.1,
        max_len=OBSERVATION_STEPS,
    )

    initialize_missing_norms(model)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = normalize_state_dict_keys(
        state_dict
    )

    missing_before = []

    model_keys = set(
        model.state_dict().keys()
    )

    checkpoint_keys = set(
        state_dict.keys()
    )

    missing_before = sorted(
        model_keys - checkpoint_keys
    )

    allowed_missing = {
        "input_norm.weight",
        "input_norm.bias",
        "pool_norm.weight",
        "pool_norm.bias",
        "output_norm.weight",
        "output_norm.bias",
    }

    unexpected_missing = (
        set(missing_before)
        - allowed_missing
    )

    if unexpected_missing:
        raise RuntimeError(
            "Unexpected missing checkpoint keys: "
            f"{sorted(unexpected_missing)}"
        )

    unexpected_checkpoint = (
        checkpoint_keys
        - model_keys
    )

    if unexpected_checkpoint:
        raise RuntimeError(
            "Unexpected checkpoint keys: "
            f"{sorted(unexpected_checkpoint)}"
        )

    load_result = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if set(load_result.missing_keys) != set(
        missing_before
    ):
        raise RuntimeError(
            "Checkpoint compatibility changed unexpectedly."
        )

    if set(load_result.unexpected_keys):
        raise RuntimeError(
            "Unexpected keys after loading: "
            f"{load_result.unexpected_keys}"
        )

    model.eval()

    print(
        "Checkpoint compatibility: PASS"
    )

    print(
        "Allowed missing normalization tensors: "
        f"{len(load_result.missing_keys)}"
    )

    return model


def predict(model, windows):
    tensor = torch.from_numpy(
        windows
    )

    with torch.no_grad():
        output = model(
            tensor
        )

    if isinstance(output, tuple):
        y3_logits = output[0]
        y6_logits = output[1]

    elif isinstance(output, dict):
        y3_logits = output["y3"]
        y6_logits = output["y6"]

    else:
        raise ValueError(
            "Unsupported model output."
        )

    y3 = torch.sigmoid(
        y3_logits
    ).reshape(-1).cpu().numpy()

    y6 = torch.sigmoid(
        y6_logits
    ).reshape(-1).cpu().numpy()

    return y3, y6


# ================================================================
# UNCERTAINTY
# ================================================================

def entropy(probability):
    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    p = np.clip(
        probability,
        1e-7,
        1.0 - 1e-7,
    )

    return (
        -p * np.log2(p)
        - (1.0 - p)
        * np.log2(1.0 - p)
    )


def get_continuous_indices():
    binary_names = {
        "authentication_failure_rate",
        "collision_event_flag",
        "connectivity_degradation_rate",
        "connectivity_drop_rate",
        "cyber_instability_index",
        "fire_explosion_flag",
        "message_integrity_failure_rate",
        "packet_loss_rate",
        "redundant_sensor_divergence",
        "replay_indicator_rate",
        "resilience_degradation_rate",
        "rolling_mean_3_message_integrity_failure_rate",
        "rolling_mean_3_packet_loss_rate",
        "rolling_mean_3_sensor_disagreement_rate",
        "rolling_std_3_message_integrity_failure_rate",
        "rolling_std_3_packet_loss_rate",
        "rolling_std_3_sensor_disagreement_rate",
        "rollover_flag",
        "safety_critical_ecu_ratio",
        "sensor_disagreement_rate",
        "sensor_plausibility_score",
        "sequence_counter_anomaly_rate",
        "speed_over_limit_flag",
        "state_transition_magnitude",
        "uds_request_rate",
        "unexpected_source_id_rate",
        "vehicle_damage_flag",
    }

    return [
        index
        for index, name in enumerate(FEATURES)
        if name not in binary_names
    ]


def local_sensitivity(
    model,
    windows,
    baseline_std,
):
    continuous_indices = (
        get_continuous_indices()
    )

    perturbations = []

    for seed in (
        11,
        22,
        33,
    ):
        rng = np.random.default_rng(
            seed
        )

        perturbed = windows.copy()

        noise = rng.normal(
            loc=0.0,
            scale=0.02,
            size=perturbed.shape,
        ).astype(
            np.float32
        )

        for index in continuous_indices:
            scale = max(
                float(
                    baseline_std[index]
                ),
                1e-6,
            )

            perturbed[
                :,
                :,
                index,
            ] += (
                noise[
                    :,
                    :,
                    index,
                ]
                * scale
            )

        y3, y6 = predict(
            model,
            perturbed,
        )

        perturbations.append(
            np.column_stack(
                [y3, y6]
            )
        )

    stacked = np.stack(
        perturbations,
        axis=0,
    )

    mean = stacked.mean(
        axis=0
    )

    std = stacked.std(
        axis=0
    )

    return (
        mean[:, 0],
        mean[:, 1],
        std[:, 0],
        std[:, 1],
    )


# ================================================================
# OOD ANALYSIS
# ================================================================

def compute_ood(
    windows,
    baseline_mean,
    baseline_std,
    baseline_delta,
):
    safe_std = np.maximum(
        baseline_std,
        1e-6,
    )

    z = (
        windows
        - baseline_mean.reshape(
            1,
            1,
            -1,
        )
    ) / safe_std.reshape(
        1,
        1,
        -1,
    )

    feature_z_rms = np.sqrt(
        np.mean(
            np.square(z),
            axis=(1, 2),
        )
    )

    feature_z_max = np.max(
        np.abs(z),
        axis=(1, 2),
    )

    diagonal_mahalanobis = np.mean(
        np.square(z),
        axis=(1, 2),
    )

    deltas = np.diff(
        windows,
        axis=1,
    )

    delta_scale = np.maximum(
        baseline_std,
        1e-6,
    ).reshape(
        1,
        1,
        -1,
    )

    delta_z = (
        deltas
        - baseline_delta.reshape(
            1,
            1,
            -1,
        )
    ) / delta_scale

    temporal_delta_rms = np.sqrt(
        np.mean(
            np.square(delta_z),
            axis=(1, 2),
        )
    )

    return (
        feature_z_rms,
        feature_z_max,
        diagonal_mahalanobis,
        temporal_delta_rms,
    )


def quantile_threshold(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        raise ValueError(
            "Cannot calculate threshold from empty data."
        )

    return float(
        np.quantile(
            values,
            0.99,
        )
    )


def finite_series(values):
    return bool(
        np.isfinite(
            np.asarray(
                values,
                dtype=float,
            )
        ).all()
    )


# ================================================================
# MAIN
# ================================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checks = {}

    print(
        "\n=== PHASE 5.12 UNCERTAINTY & OOD ANALYSIS ==="
    )

    # ------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------

    checks["input_exists"] = (
        INPUT_CSV.exists()
    )

    checks["checkpoint_exists"] = (
        CHECKPOINT.exists()
    )

    if not checks["input_exists"]:
        raise FileNotFoundError(
            INPUT_CSV
        )

    if not checks["checkpoint_exists"]:
        raise FileNotFoundError(
            CHECKPOINT
        )

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    print(
        f"Phase 5.10 input rows: {len(df)}"
    )

    checks["input_rows_99000"] = (
        len(df) == 99000
    )

    checks["scenario_id_present"] = (
        "scenario_id" in df.columns
    )

    checks["signals_json_present"] = (
        "signals_json" in df.columns
    )

    checks["scenario_active_present"] = (
        "scenario_active" in df.columns
    )

    if not all(
        (
            checks["input_rows_99000"],
            checks["scenario_id_present"],
            checks["signals_json_present"],
            checks["scenario_active_present"],
        )
    ):
        raise ValueError(
            "Phase 5.10 input contract failed."
        )

    scenario_ids = sorted(
        df["scenario_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    attack_ids = [
        value
        for value in scenario_ids
        if value.startswith("C")
    ]

    print(
        f"Scenarios found: {len(scenario_ids)}"
    )

    for scenario in scenario_ids:
        print(
            f"  {scenario}"
        )

    checks["scenario_count_11"] = (
        len(scenario_ids)
        == EXPECTED_SCENARIOS
    )

    checks["attack_scenario_count_10"] = (
        len(attack_ids)
        == EXPECTED_ATTACK_SCENARIOS
    )

    if not checks["scenario_count_11"]:
        raise ValueError(
            "Expected exactly 11 scenarios."
        )

    if not checks["attack_scenario_count_10"]:
        raise ValueError(
            "Expected exactly 10 attack scenarios."
        )

    # ------------------------------------------------------------
    # Frozen model
    # ------------------------------------------------------------

    print(
        "\nLoading frozen Transformer..."
    )

    model = load_model()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    checkpoint_hash = sha256_file(
        CHECKPOINT
    )

    print(
        f"Model parameters: {parameter_count}"
    )

    print(
        f"Checkpoint SHA256: {checkpoint_hash}"
    )

    checks["parameter_count_407811"] = (
        parameter_count == MODEL_PARAMS
    )

    checks["checkpoint_sha256_64"] = (
        len(checkpoint_hash) == 64
    )

    # ------------------------------------------------------------
    # Build streams
    # ------------------------------------------------------------

    stream_windows = {}
    stream_timestamps = {}

    for scenario in scenario_ids:
        print(
            f"\nBuilding stream: {scenario}..."
        )

        scenario_df = df[
            df["scenario_id"]
            .astype(str)
            == scenario
        ].copy()

        scenario_df = scenario_df.reset_index(
            drop=True
        )

        print(
            f"  records: {len(scenario_df)}"
        )

        if len(scenario_df) != 9000:
            raise ValueError(
                f"{scenario} has "
                f"{len(scenario_df)} records."
            )

        X, timestamp_spread = (
            build_stream(
                scenario_df
            )
        )

        windows = make_windows(
            X
        )

        stream_windows[
            scenario
        ] = windows

        stream_timestamps[
            scenario
        ] = timestamp_spread

        print(
            f"  timesteps: {len(X)}"
        )

        print(
            f"  windows: {len(windows)}"
        )

    checks["each_stream_600_timesteps"] = all(
        len(
            stream_timestamps[scenario]
        )
        == EXPECTED_TIMESTEPS
        for scenario in scenario_ids
    )

    checks["each_stream_589_windows"] = all(
        len(
            stream_windows[scenario]
        )
        == EXPECTED_WINDOWS
        for scenario in scenario_ids
    )

    # ------------------------------------------------------------
    # Baseline reference
    # ------------------------------------------------------------

    baseline_windows = (
        stream_windows[BASELINE_ID]
    )

    baseline_reference = (
        baseline_windows.reshape(
            -1,
            FEATURE_COUNT,
        )
    )

    baseline_mean = (
        baseline_reference.mean(
            axis=0
        )
    )

    baseline_std = (
        baseline_reference.std(
            axis=0
        )
    )

    baseline_std = np.maximum(
        baseline_std,
        1e-6,
    )

    # Correct temporal baseline:
    # mean transition across normal baseline windows.
    baseline_deltas = np.diff(
        baseline_windows,
        axis=1,
    )

    baseline_delta = (
        baseline_deltas.mean(
            axis=(0, 1)
        )
    )

    checks["baseline_windows_589"] = (
        len(baseline_windows)
        == EXPECTED_WINDOWS
    )

    checks["baseline_reference_finite"] = (
        finite_series(baseline_mean)
        and finite_series(baseline_std)
        and finite_series(baseline_delta)
    )

    # ------------------------------------------------------------
    # Baseline OOD thresholds
    # ------------------------------------------------------------

    (
        baseline_z_rms,
        baseline_z_max,
        baseline_mahalanobis,
        baseline_temporal,
    ) = compute_ood(
        baseline_windows,
        baseline_mean,
        baseline_std,
        baseline_delta,
    )

    thresholds = {
        "feature_z_rms_99": quantile_threshold(
            baseline_z_rms
        ),
        "feature_z_max_99": quantile_threshold(
            baseline_z_max
        ),
        "diagonal_mahalanobis_99": quantile_threshold(
            baseline_mahalanobis
        ),
        "temporal_delta_rms_99": quantile_threshold(
            baseline_temporal
        ),
    }

    print(
        "\nBaseline-derived OOD thresholds:"
    )

    for name, value in thresholds.items():
        print(
            f"  {name}: {value:.6f}"
        )

    # ------------------------------------------------------------
    # Evaluate streams
    # ------------------------------------------------------------

    records = []

    for scenario in scenario_ids:
        print(
            f"\nEvaluating {scenario}..."
        )

        windows = stream_windows[
            scenario
        ]

        y3, y6 = predict(
            model,
            windows,
        )

        (
            perturbed_y3,
            perturbed_y6,
            y3_uncertainty,
            y6_uncertainty,
        ) = local_sensitivity(
            model,
            windows,
            baseline_std,
        )

        (
            z_rms,
            z_max,
            mahalanobis,
            temporal_ood,
        ) = compute_ood(
            windows,
            baseline_mean,
            baseline_std,
            baseline_delta,
        )

        y3_entropy = entropy(
            y3
        )

        y6_entropy = entropy(
            y6
        )

        feature_z_rms_flag = (
            z_rms
            > thresholds[
                "feature_z_rms_99"
            ]
        )

        feature_z_max_flag = (
            z_max
            > thresholds[
                "feature_z_max_99"
            ]
        )

        mahalanobis_flag = (
            mahalanobis
            > thresholds[
                "diagonal_mahalanobis_99"
            ]
        )

        temporal_flag = (
            temporal_ood
            > thresholds[
                "temporal_delta_rms_99"
            ]
        )

        ood_flag = (
            feature_z_rms_flag
            | feature_z_max_flag
            | mahalanobis_flag
            | temporal_flag
        )

        ood_score = np.maximum.reduce(
            [
                z_rms
                / max(
                    thresholds[
                        "feature_z_rms_99"
                    ],
                    1e-6,
                ),
                z_max
                / max(
                    thresholds[
                        "feature_z_max_99"
                    ],
                    1e-6,
                ),
                mahalanobis
                / max(
                    thresholds[
                        "diagonal_mahalanobis_99"
                    ],
                    1e-6,
                ),
                temporal_ood
                / max(
                    thresholds[
                        "temporal_delta_rms_99"
                    ],
                    1e-6,
                ),
            ]
        )

        uncertainty_score = np.maximum.reduce(
            [
                y3_uncertainty,
                y6_uncertainty,
                y3_entropy,
                y6_entropy,
            ]
        )

        for index in range(
            EXPECTED_WINDOWS
        ):
            end_timestep = (
                index
                + OBSERVATION_STEPS
                - 1
            )

            records.append(
                {
                    "scenario_id": scenario,
                    "window_index": index,
                    "start_timestep": index,
                    "end_timestep": end_timestep,
                    "y3_probability": finite_float(
                        y3[index]
                    ),
                    "y6_probability": finite_float(
                        y6[index]
                    ),
                    "y3_entropy": finite_float(
                        y3_entropy[index]
                    ),
                    "y6_entropy": finite_float(
                        y6_entropy[index]
                    ),
                    "y3_perturbed_mean": finite_float(
                        perturbed_y3[index]
                    ),
                    "y6_perturbed_mean": finite_float(
                        perturbed_y6[index]
                    ),
                    "y3_local_sensitivity_std": finite_float(
                        y3_uncertainty[index]
                    ),
                    "y6_local_sensitivity_std": finite_float(
                        y6_uncertainty[index]
                    ),
                    "uncertainty_score": finite_float(
                        uncertainty_score[index]
                    ),
                    "feature_z_rms": finite_float(
                        z_rms[index]
                    ),
                    "feature_z_max": finite_float(
                        z_max[index]
                    ),
                    "diagonal_mahalanobis": finite_float(
                        mahalanobis[index]
                    ),
                    "temporal_delta_z_rms": finite_float(
                        temporal_ood[index]
                    ),
                    "ood_score": finite_float(
                        ood_score[index]
                    ),
                    "ood_feature_rms_flag": bool(
                        feature_z_rms_flag[index]
                    ),
                    "ood_feature_max_flag": bool(
                        feature_z_max_flag[index]
                    ),
                    "ood_mahalanobis_flag": bool(
                        mahalanobis_flag[index]
                    ),
                    "ood_temporal_flag": bool(
                        temporal_flag[index]
                    ),
                    "ood_flag": bool(
                        ood_flag[index]
                    ),
                    "laboratory_generated": True,
                    "physical_vehicle": False,
                    "direct_actuation": False,
                    "model_retraining": False,
                    "live_cortex_xdr": False,
                    "real_world_attack_claim": False,
                }
            )

    result_df = pd.DataFrame(
        records
    )

    # ------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------

    expected_rows = (
        EXPECTED_SCENARIOS
        * EXPECTED_WINDOWS
    )

    print(
        "\n=== PHASE 5.12 RESULTS ==="
    )

    print(
        f"Prediction/OOD rows: "
        f"{len(result_df)}"
    )

    checks["output_rows_6479"] = (
        len(result_df)
        == expected_rows
    )

    checks["y3_finite"] = finite_series(
        result_df["y3_probability"]
    )

    checks["y6_finite"] = finite_series(
        result_df["y6_probability"]
    )

    checks["uncertainty_finite"] = (
        finite_series(
            result_df[
                "uncertainty_score"
            ]
        )
    )

    checks["ood_finite"] = finite_series(
        result_df["ood_score"]
    )

    checks["probabilities_in_range"] = bool(
        result_df[
            "y3_probability"
        ].between(
            0.0,
            1.0,
        ).all()
        and
        result_df[
            "y6_probability"
        ].between(
            0.0,
            1.0,
        ).all()
    )

    checks["all_scenarios_present"] = (
        set(
            result_df[
                "scenario_id"
            ]
        )
        == set(scenario_ids)
    )

    counts = (
        result_df[
            "scenario_id"
        ]
        .value_counts()
    )

    checks["each_scenario_589"] = (
        len(counts) == EXPECTED_SCENARIOS
        and bool(
            (
                counts
                == EXPECTED_WINDOWS
            ).all()
        )
    )

    baseline_result = result_df[
        result_df["scenario_id"]
        == BASELINE_ID
    ]

    checks["baseline_present"] = (
        len(baseline_result)
        == EXPECTED_WINDOWS
    )

    checks["laboratory_only"] = bool(
        result_df[
            "laboratory_generated"
        ].all()
        and not result_df[
            "physical_vehicle"
        ].any()
    )

    checks["no_direct_actuation"] = (
        not bool(
            result_df[
                "direct_actuation"
            ].any()
        )
    )

    checks["no_model_retraining"] = (
        not bool(
            result_df[
                "model_retraining"
            ].any()
        )
    )

    checks["no_live_cortex_xdr"] = (
        not bool(
            result_df[
                "live_cortex_xdr"
            ].any()
        )
    )

    checks["no_real_world_attack_claim"] = (
        not bool(
            result_df[
                "real_world_attack_claim"
            ].any()
        )
    )

    # ------------------------------------------------------------
    # Disk outputs
    # ------------------------------------------------------------

    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    with open(
        OUTPUT_JSONL,
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            clean = {}

            for key, value in record.items():
                if isinstance(
                    value,
                    np.bool_,
                ):
                    value = bool(value)

                elif isinstance(
                    value,
                    np.integer,
                ):
                    value = int(value)

                elif isinstance(
                    value,
                    np.floating,
                ):
                    value = float(value)

                if isinstance(
                    value,
                    float,
                ) and not math.isfinite(value):
                    value = None

                clean[key] = value

            handle.write(
                json.dumps(
                    clean,
                    allow_nan=False,
                )
                + "\n"
            )

    schema = {
        "phase": "5.12",
        "title": "Uncertainty and OOD Analysis",
        "feature_count": FEATURE_COUNT,
        "observation_steps": OBSERVATION_STEPS,
        "records_per_timestep": RECORDS_PER_TIMESTEP,
        "expected_timesteps": EXPECTED_TIMESTEPS,
        "expected_windows_per_stream": EXPECTED_WINDOWS,
        "model_parameters": parameter_count,
        "checkpoint_sha256": checkpoint_hash,
        "baseline_reference": BASELINE_ID,
        "ood_threshold_method": (
            "99th percentile of normal laboratory "
            "baseline window distributions"
        ),
        "uncertainty_method": (
            "prediction entropy plus deterministic "
            "local perturbation sensitivity"
        ),
        "uncertainty_interpretation": (
            "proxy measure; not Bayesian uncertainty, "
            "not calibrated confidence, and not a "
            "safety guarantee"
        ),
        "ood_methods": [
            "feature_z_rms",
            "feature_z_max",
            "diagonal_mahalanobis",
            "temporal_delta_z_rms",
        ],
        "outputs": list(
            result_df.columns
        ),
        "boundary": {
            "laboratory_generated": True,
            "physical_vehicle": False,
            "direct_actuation": False,
            "model_retraining": False,
            "live_cortex_xdr": False,
            "real_world_attack_labels": False,
        },
    }

    with open(
        SCHEMA,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            schema,
            handle,
            indent=2,
        )

    # ------------------------------------------------------------
    # Re-read from disk
    # ------------------------------------------------------------

    disk_df = pd.read_csv(
        OUTPUT_CSV,
        low_memory=False,
    )

    checks["disk_output_rows_6479"] = (
        len(disk_df)
        == expected_rows
    )

    disk_counts = (
        disk_df[
            "scenario_id"
        ]
        .value_counts()
    )

    checks["disk_each_scenario_589"] = (
        len(disk_counts)
        == EXPECTED_SCENARIOS
        and bool(
            (
                disk_counts
                == EXPECTED_WINDOWS
            ).all()
        )
    )

    checks["output_csv_exists"] = (
        OUTPUT_CSV.exists()
    )

    checks["output_jsonl_exists"] = (
        OUTPUT_JSONL.exists()
    )

    checks["schema_exists"] = (
        SCHEMA.exists()
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    attack_df = result_df[
        result_df[
            "scenario_id"
        ].astype(str).str.startswith("C")
    ]

    summary = {
        "total_rows": int(
            len(result_df)
        ),
        "baseline_rows": int(
            len(baseline_result)
        ),
        "attack_rows": int(
            len(attack_df)
        ),
        "baseline_ood_rate": float(
            baseline_result[
                "ood_flag"
            ].mean()
        ),
        "attack_ood_rate": float(
            attack_df[
                "ood_flag"
            ].mean()
        ),
        "y3_mean": float(
            result_df[
                "y3_probability"
            ].mean()
        ),
        "y6_mean": float(
            result_df[
                "y6_probability"
            ].mean()
        ),
        "max_uncertainty_score": float(
            result_df[
                "uncertainty_score"
            ].max()
        ),
        "max_ood_score": float(
            result_df[
                "ood_score"
            ].max()
        ),
    }

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------

    overall_pass = all(
        checks.values()
    )

    manifest = {
        "phase": "5.12",
        "status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
        "experiment": (
            "Frozen Transformer uncertainty "
            "and out-of-distribution analysis "
            "using controlled EV laboratory scenarios."
        ),
        "input": str(
            INPUT_CSV
        ),
        "checkpoint": str(
            CHECKPOINT
        ),
        "checkpoint_sha256": checkpoint_hash,
        "model_parameters": parameter_count,
        "feature_count": FEATURE_COUNT,
        "observation_steps": OBSERVATION_STEPS,
        "records_per_timestep": RECORDS_PER_TIMESTEP,
        "scenarios": scenario_ids,
        "summary": summary,
        "ood_thresholds": thresholds,
        "validation": checks,
        "outputs": {
            "csv": str(
                OUTPUT_CSV
            ),
            "jsonl": str(
                OUTPUT_JSONL
            ),
            "schema": str(
                SCHEMA
            ),
        },
        "research_boundaries": {
            "laboratory_generated": True,
            "physical_vehicle": False,
            "direct_actuation": False,
            "model_retraining": False,
            "live_cortex_xdr": False,
            "real_world_attack_labels": False,
            "production_safety_claim": False,
            "production_ood_claim": False,
        },
        "limitations": [
            (
                "The frozen Phase 5.7 model is reused "
                "without retraining."
            ),
            (
                "The historical training scaler parameters "
                "were not reconstructed."
            ),
            (
                "The six historical normalization tensors "
                "missing from the checkpoint are treated "
                "as identity transformations."
            ),
            (
                "OOD thresholds are laboratory-baseline "
                "statistical thresholds."
            ),
            (
                "Diagonal Mahalanobis is used instead of "
                "a full covariance estimator."
            ),
            (
                "Local perturbation sensitivity is an "
                "uncertainty proxy, not Bayesian uncertainty."
            ),
            (
                "Synthetic laboratory cyber-physical "
                "scenarios do not establish real-world "
                "attack performance."
            ),
        ],
    }

    with open(
        MANIFEST,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
        )

    # ------------------------------------------------------------
    # Console results
    # ------------------------------------------------------------

    print(
        "\nValidation checks:"
    )

    for name, passed in checks.items():
        print(
            f"  "
            f"{'PASS' if passed else 'FAIL'} "
            f"{name}"
        )

    print(
        "\nSummary:"
    )

    print(
        f"  Baseline OOD rate: "
        f"{summary['baseline_ood_rate']:.6f}"
    )

    print(
        f"  Attack OOD rate: "
        f"{summary['attack_ood_rate']:.6f}"
    )

    print(
        f"  Mean Y3: "
        f"{summary['y3_mean']:.6f}"
    )

    print(
        f"  Mean Y6: "
        f"{summary['y6_mean']:.6f}"
    )

    print(
        f"  Maximum uncertainty score: "
        f"{summary['max_uncertainty_score']:.6f}"
    )

    print(
        f"  Maximum OOD score: "
        f"{summary['max_ood_score']:.6f}"
    )

    print(
        f"\nCSV: {OUTPUT_CSV}"
    )

    print(
        f"JSONL: {OUTPUT_JSONL}"
    )

    print(
        f"Schema: {SCHEMA}"
    )

    print(
        f"Manifest: {MANIFEST}"
    )

    print(
        "\nSTATUS: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )


if __name__ == "__main__":
    main()