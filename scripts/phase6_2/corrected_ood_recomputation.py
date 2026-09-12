from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles"
)

PHASE512_SCRIPT = (
    ROOT
    / "scripts"
    / "can_hil"
    / "phase5_12"
    / "uncertainty_ood_analysis.py"
)

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_10"
    / "phase5_10_cyber_physical_attack_scenarios.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase6_2"
)

MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "phase6_2"
    / "phase6_2_corrected_ood_manifest.json"
)

WINDOW_SIZE = 12

# This is a numerical-stability tolerance for identifying features
# that have no measurable variation in the baseline timestep stream.
#
# It is NOT selected from attack behavior and is NOT an OOD threshold.
BASELINE_VARIANCE_TOLERANCE = 1e-12

BASELINE_ID = "S00_NORMAL_BASELINE"


def load_phase512():
    spec = importlib.util.spec_from_file_location(
        "phase512",
        PHASE512_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load Phase 5.12 module: {PHASE512_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def build_all_streams(df: pd.DataFrame, phase512):
    scenario_ids = sorted(df["scenario_id"].dropna().unique().tolist())

    expected_scenarios = [
        "C01_SPEED_SENSOR_SPOOFING",
        "C02_BRAKE_ACCELERATOR_ATTACK",
        "C03_TORQUE_DYNAMICS_ATTACK",
        "C04_HV_POWER_ATTACK",
        "C05_BATTERY_THERMAL_ATTACK",
        "C06_HV_CONTACTOR_ATTACK",
        "C07_HEARTBEAT_INTEGRITY_ATTACK",
        "C08_SEQUENCE_GATEWAY_ATTACK",
        "C09_RESTRAINT_DOMAIN_ATTACK",
        "C10_MULTI_DOMAIN_ATTACK",
        BASELINE_ID,
    ]

    if set(scenario_ids) != set(expected_scenarios):
        raise ValueError(
            "Unexpected Phase 5.10 scenarios: "
            f"{scenario_ids}"
        )

    streams = {}
    windows = {}

    for scenario_id in expected_scenarios:
        scenario_df = df.loc[
            df["scenario_id"].eq(scenario_id)
        ].copy()

        if len(scenario_df) != 9000:
            raise ValueError(
                f"{scenario_id}: expected 9000 records, "
                f"got {len(scenario_df)}"
            )

        X, timestamp_spread = phase512.build_stream(
            scenario_df
        )

        if X.shape != (
            phase512.EXPECTED_TIMESTEPS,
            phase512.FEATURE_COUNT,
        ):
            raise ValueError(
                f"{scenario_id}: unexpected stream shape {X.shape}"
            )

        W = phase512.make_windows(X)

        expected_windows = (
            phase512.EXPECTED_WINDOWS
        )

        if W.shape != (
            expected_windows,
            WINDOW_SIZE,
            phase512.FEATURE_COUNT,
        ):
            raise ValueError(
                f"{scenario_id}: unexpected window shape {W.shape}"
            )

        streams[scenario_id] = X
        windows[scenario_id] = W

    return streams, windows


def compute_baseline_statistics(
    baseline_stream: np.ndarray,
    baseline_windows: np.ndarray,
):
    """
    Baseline statistics are established from the non-overlapping
    timestep stream.

    This avoids creating artificial variance by flattening overlapping
    windows. The 59-D model input itself remains unchanged.
    """

    timestep_mean = np.mean(
        baseline_stream,
        axis=0,
        dtype=np.float64,
    )

    timestep_std = np.std(
        baseline_stream,
        axis=0,
        dtype=np.float64,
    )

    active_mask = (
        timestep_std > BASELINE_VARIANCE_TOLERANCE
    )

    excluded_indices = np.where(
        ~active_mask
    )[0].tolist()

    active_indices = np.where(
        active_mask
    )[0].tolist()

    if len(active_indices) == 0:
        raise RuntimeError(
            "No OOD statistical features remain."
        )

    active_mean = timestep_mean[
        active_mask
    ]

    active_std = timestep_std[
        active_mask
    ]

    # Numerical protection only for active features.
    active_std = np.maximum(
        active_std,
        np.finfo(np.float64).eps,
    )

    baseline_deltas = np.diff(
        baseline_stream,
        axis=0,
    )

    active_baseline_deltas = (
        baseline_deltas[:, active_mask]
    )

    delta_mean = np.mean(
        active_baseline_deltas,
        axis=0,
        dtype=np.float64,
    )

    return {
        "timestep_mean": timestep_mean,
        "timestep_std": timestep_std,
        "active_mask": active_mask,
        "active_indices": active_indices,
        "excluded_indices": excluded_indices,
        "active_mean": active_mean,
        "active_std": active_std,
        "delta_mean": delta_mean,
        "baseline_windows": baseline_windows,
    }


def compute_window_metrics(
    windows: np.ndarray,
    stats: dict,
):
    active_mask = stats["active_mask"]
    active_mean = stats["active_mean"]
    active_std = stats["active_std"]
    delta_mean = stats["delta_mean"]

    selected = windows[:, :, active_mask].astype(
        np.float64,
        copy=False,
    )

    z = (
        selected - active_mean[None, None, :]
    ) / active_std[None, None, :]

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

    diagonal_mahalanobis = np.sqrt(
        np.mean(
            np.square(z),
            axis=1,
        ).sum(axis=1)
    )

    deltas = np.diff(
        selected,
        axis=1,
    )

    delta_residual = (
        deltas
        - delta_mean[None, None, :]
    )

    temporal_delta_rms = np.sqrt(
        np.mean(
            np.square(delta_residual),
            axis=(1, 2),
        )
    )

    return {
        "feature_z_rms": feature_z_rms,
        "feature_z_max": feature_z_max,
        "diagonal_mahalanobis": diagonal_mahalanobis,
        "temporal_delta_rms": temporal_delta_rms,
    }


def percentile99(values):
    return float(
        np.percentile(
            np.asarray(values, dtype=np.float64),
            99.0,
        )
    )


def add_normalized_scores(
    metrics: dict,
    thresholds: dict,
):
    rms = (
        metrics["feature_z_rms"]
        / thresholds["feature_z_rms_99"]
    )

    zmax = (
        metrics["feature_z_max"]
        / thresholds["feature_z_max_99"]
    )

    maha = (
        metrics["diagonal_mahalanobis"]
        / thresholds["diagonal_mahalanobis_99"]
    )

    delta = (
        metrics["temporal_delta_rms"]
        / thresholds["temporal_delta_rms_99"]
    )

    score = np.maximum.reduce(
        [rms, zmax, maha, delta]
    )

    flag = (
        (metrics["feature_z_rms"]
         > thresholds["feature_z_rms_99"])
        |
        (metrics["feature_z_max"]
         > thresholds["feature_z_max_99"])
        |
        (metrics["diagonal_mahalanobis"]
         > thresholds["diagonal_mahalanobis_99"])
        |
        (metrics["temporal_delta_rms"]
         > thresholds["temporal_delta_rms_99"])
    )

    return score, flag


def sha256_file(path: Path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    print(
        "=== PHASE 6.2 CORRECTED OOD RECOMPUTATION ==="
    )

    phase512 = load_phase512()

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    print(
        f"[1/8] Phase 5.10 rows: {len(df):,}"
    )

    streams, windows = build_all_streams(
        df,
        phase512,
    )

    print(
        "[2/8] Streams/windows reconstructed:"
    )

    for scenario_id in sorted(streams):
        print(
            f"  {scenario_id}: "
            f"stream={streams[scenario_id].shape}, "
            f"windows={windows[scenario_id].shape}"
        )

    baseline_stream = streams[BASELINE_ID]
    baseline_windows = windows[BASELINE_ID]

    stats = compute_baseline_statistics(
        baseline_stream,
        baseline_windows,
    )

    print(
        "\n[3/8] Baseline statistical feature selection"
    )

    print(
        f"Total model features       : "
        f"{phase512.FEATURE_COUNT}"
    )

    print(
        f"Active OOD statistical dims : "
        f"{len(stats['active_indices'])}"
    )

    print(
        f"Excluded constant dims      : "
        f"{len(stats['excluded_indices'])}"
    )

    print(
        f"Excluded indices            : "
        f"{stats['excluded_indices']}"
    )

    ecu_index = 49

    print(
        f"ecu_count index 49 excluded : "
        f"{ecu_index in stats['excluded_indices']}"
    )

    if ecu_index not in stats["excluded_indices"]:
        raise RuntimeError(
            "Scientific validation failed: "
            "ecu_count index 49 must be excluded because "
            "its baseline timestep variance is zero."
        )

    print(
        "\n[4/8] Computing corrected OOD metrics"
    )

    all_rows = []

    for scenario_id in [
        BASELINE_ID,
        "C01_SPEED_SENSOR_SPOOFING",
        "C02_BRAKE_ACCELERATOR_ATTACK",
        "C03_TORQUE_DYNAMICS_ATTACK",
        "C04_HV_POWER_ATTACK",
        "C05_BATTERY_THERMAL_ATTACK",
        "C06_HV_CONTACTOR_ATTACK",
        "C07_HEARTBEAT_INTEGRITY_ATTACK",
        "C08_SEQUENCE_GATEWAY_ATTACK",
        "C09_RESTRAINT_DOMAIN_ATTACK",
        "C10_MULTI_DOMAIN_ATTACK",
    ]:
        scenario_metrics = compute_window_metrics(
            windows[scenario_id],
            stats,
        )

        frame = pd.DataFrame(
            {
                "scenario_id": scenario_id,
                "window_index": np.arange(
                    len(
                        windows[scenario_id]
                    ),
                    dtype=np.int32,
                ),
                "feature_z_rms": scenario_metrics[
                    "feature_z_rms"
                ],
                "feature_z_max": scenario_metrics[
                    "feature_z_max"
                ],
                "diagonal_mahalanobis": scenario_metrics[
                    "diagonal_mahalanobis"
                ],
                "temporal_delta_rms": scenario_metrics[
                    "temporal_delta_rms"
                ],
            }
        )

        all_rows.append(frame)

    result = pd.concat(
        all_rows,
        ignore_index=True,
    )

    print(
        f"Corrected rows: {len(result):,}"
    )

    print(
        "\n[5/8] Recomputing baseline 99th-percentile thresholds"
    )

    baseline_result = result.loc[
        result["scenario_id"].eq(BASELINE_ID)
    ]

    thresholds = {
        "feature_z_rms_99": percentile99(
            baseline_result["feature_z_rms"]
        ),
        "feature_z_max_99": percentile99(
            baseline_result["feature_z_max"]
        ),
        "diagonal_mahalanobis_99": percentile99(
            baseline_result["diagonal_mahalanobis"]
        ),
        "temporal_delta_rms_99": percentile99(
            baseline_result["temporal_delta_rms"]
        ),
    }

    for key, value in thresholds.items():
        print(
            f"{key}: {value:.12g}"
        )

    print(
        "\n[6/8] Applying corrected OOD normalization"
    )

    score, flag = add_normalized_scores(
        {
            "feature_z_rms": result[
                "feature_z_rms"
            ].to_numpy(),
            "feature_z_max": result[
                "feature_z_max"
            ].to_numpy(),
            "diagonal_mahalanobis": result[
                "diagonal_mahalanobis"
            ].to_numpy(),
            "temporal_delta_rms": result[
                "temporal_delta_rms"
            ].to_numpy(),
        },
        thresholds,
    )

    result[
        "corrected_ood_score"
    ] = score

    result[
        "corrected_ood_flag"
    ] = flag.astype(bool)

    result[
        "ood_statistical_feature_count"
    ] = len(stats["active_indices"])

    result[
        "ood_excluded_feature_count"
    ] = len(stats["excluded_indices"])

    print(
        "\n[7/8] Validation"
    )

    validation = []

    validation.append(
        (
            "59 model input features preserved",
            phase512.FEATURE_COUNT == 59,
        )
    )

    validation.append(
        (
            "OOD active dimensions nonzero",
            len(stats["active_indices"]) > 0,
        )
    )

    validation.append(
        (
            "ecu_count excluded",
            49 in stats["excluded_indices"],
        )
    )

    validation.append(
        (
            "baseline windows expected",
            len(baseline_result)
            == phase512.EXPECTED_WINDOWS,
        )
    )

    validation.append(
        (
            "all scenarios expected",
            result["scenario_id"].nunique()
            == 11,
        )
    )

    validation.append(
        (
            "all corrected scores finite",
            np.isfinite(
                result["corrected_ood_score"]
            ).all(),
        )
    )

    validation.append(
        (
            "all corrected metrics finite",
            np.isfinite(
                result[
                    [
                        "feature_z_rms",
                        "feature_z_max",
                        "diagonal_mahalanobis",
                        "temporal_delta_rms",
                    ]
                ].to_numpy()
            ).all(),
        )
    )

    validation.append(
        (
            "no extreme pathological OOD tail",
            float(
                result["corrected_ood_score"].max()
            ) < 100000.0,
        )
    )

    validation.append(
        (
            "model feature dimensionality unchanged",
            len(stats["timestep_mean"])
            == phase512.FEATURE_COUNT,
        )
    )

    validation.append(
        (
            "constant-feature variance rule applied",
            np.all(
                stats["timestep_std"][
                    stats["excluded_indices"]
                ]
                <= BASELINE_VARIANCE_TOLERANCE
            ),
        )
    )

    validation_pass = all(
        passed
        for _, passed in validation
    )

    for name, passed in validation:
        print(
            f"{'PASS' if passed else 'FAIL'}  {name}"
        )

    print(
        f"\nValidation: "
        f"{sum(p for _, p in validation)}/"
        f"{len(validation)} PASS"
    )

    print(
        "\n=== CORRECTED SCENARIO SUMMARY ==="
    )

    for scenario_id in [
        BASELINE_ID,
        "C01_SPEED_SENSOR_SPOOFING",
        "C02_BRAKE_ACCELERATOR_ATTACK",
        "C03_TORQUE_DYNAMICS_ATTACK",
        "C04_HV_POWER_ATTACK",
        "C05_BATTERY_THERMAL_ATTACK",
        "C06_HV_CONTACTOR_ATTACK",
        "C07_HEARTBEAT_INTEGRITY_ATTACK",
        "C08_SEQUENCE_GATEWAY_ATTACK",
        "C09_RESTRAINT_DOMAIN_ATTACK",
        "C10_MULTI_DOMAIN_ATTACK",
    ]:
        subset = result.loc[
            result["scenario_id"].eq(
                scenario_id
            )
        ]

        print(
            f"{scenario_id}: "
            f"mean={subset['corrected_ood_score'].mean():.6f} "
            f"median={subset['corrected_ood_score'].median():.6f} "
            f"max={subset['corrected_ood_score'].max():.6f} "
            f"rate={subset['corrected_ood_flag'].mean():.6f}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_DIR
        / "phase6_2_corrected_ood.csv"
    )

    jsonl_path = (
        OUTPUT_DIR
        / "phase6_2_corrected_ood.jsonl"
    )

    schema_path = (
        OUTPUT_DIR
        / "phase6_2_corrected_ood_schema.json"
    )

    result.to_csv(
        csv_path,
        index=False,
    )

    result.to_json(
        jsonl_path,
        orient="records",
        lines=True,
    )

    schema = {
        "phase": "6.2",
        "title": "Corrected OOD Statistical Recomputation",
        "model_feature_count": int(
            phase512.FEATURE_COUNT
        ),
        "ood_statistical_feature_count": int(
            len(stats["active_indices"])
        ),
        "ood_excluded_feature_count": int(
            len(stats["excluded_indices"])
        ),
        "ood_active_feature_indices": [
            int(x)
            for x in stats["active_indices"]
        ],
        "ood_excluded_feature_indices": [
            int(x)
            for x in stats["excluded_indices"]
        ],
        "baseline_variance_tolerance": (
            BASELINE_VARIANCE_TOLERANCE
        ),
        "feature_49_name": "ecu_count",
        "feature_49_excluded": True,
        "ood_thresholds": thresholds,
        "model_input_features_unchanged": True,
        "statistical_basis": (
            "Baseline timestep stream rather than "
            "flattened overlapping windows"
        ),
        "safety_scope": (
            "Laboratory research only; not production "
            "vehicle safety validation or regulatory compliance."
        ),
    }

    schema_path.write_text(
        json.dumps(
            schema,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "phase": "6.2",
        "title": "Corrected OOD Statistical Recomputation",
        "status": (
            "PASS"
            if validation_pass
            else "FAIL"
        ),
        "input": str(INPUT_CSV),
        "outputs": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "schema": str(schema_path),
        },
        "model_feature_count": int(
            phase512.FEATURE_COUNT
        ),
        "ood_statistical_feature_count": int(
            len(stats["active_indices"])
        ),
        "ood_excluded_feature_count": int(
            len(stats["excluded_indices"])
        ),
        "ood_active_feature_indices": [
            int(x)
            for x in stats["active_indices"]
        ],
        "ood_excluded_feature_indices": [
            int(x)
            for x in stats["excluded_indices"]
        ],
        "excluded_feature_names": [
            phase512.FEATURES[i]
            for i in stats["excluded_indices"]
        ],
        "feature_49": {
            "index": 49,
            "name": "ecu_count",
            "baseline_timestep_std": float(
                stats["timestep_std"][49]
            ),
            "excluded": True,
        },
        "baseline_variance_tolerance": (
            BASELINE_VARIANCE_TOLERANCE
        ),
        "thresholds": thresholds,
        "validation": {
            name: bool(passed)
            for name, passed in validation
        },
        "input_sha256": sha256_file(
            INPUT_CSV
        ),
        "method": {
            "frozen_transformer_reused": True,
            "model_input_is_not_modified": True,
            "ood_statistics_use_timestep_baseline": True,
            "overlapping_window_flattening_avoided": True,
            "constant_baseline_features_excluded": True,
            "attack_specific_threshold_tuning": False,
        },
        "safety_scope": (
            "Laboratory research only; no production vehicle "
            "safety, OEM compliance, regulatory certification, "
            "or live vehicle deployment claim."
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n[8/8] Outputs written"
    )

    print(f"CSV     : {csv_path}")
    print(f"JSONL   : {jsonl_path}")
    print(f"Schema  : {schema_path}")
    print(f"Manifest: {MANIFEST_PATH}")

    print(
        f"\nSTATUS: "
        f"{'PASS' if validation_pass else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
