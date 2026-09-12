from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

PHASE57_DIR = (
    ROOT
    / "scripts"
    / "can_hil"
    / "phase5_7"
)

sys.path.insert(
    0,
    str(PHASE57_DIR)
)

from can_ev_frozen_transformer_integration import (
    FEATURES,
    derive_model_compatible_features,
)

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_10"
    / "phase5_10_cyber_physical_attack_scenarios.csv"
)

RECORDS_PER_TIMESTEP = 15
OBSERVATION_STEPS = 12


def expand_signals(row):

    result = row.copy()

    try:
        signals = json.loads(
            str(
                row.get(
                    "signals_json",
                    "{}"
                )
            )
        )
    except Exception:
        signals = {}

    if isinstance(signals, dict):

        for key, value in signals.items():

            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                result[key] = 0.0

    return result


def row_features(row):

    expanded = expand_signals(row)

    derived = derive_model_compatible_features(
        expanded
    )

    values = []

    for name in FEATURES:

        try:
            value = float(
                derived.get(
                    name,
                    0.0
                )
            )
        except (TypeError, ValueError):
            value = 0.0

        if not np.isfinite(value):
            value = 0.0

        values.append(value)

    return values


def build_timestep_matrix(group):

    rows = []

    for _, row in group.iterrows():

        rows.append(
            row_features(row)
        )

    matrix = np.asarray(
        rows,
        dtype=np.float64
    )

    if matrix.shape[0] != RECORDS_PER_TIMESTEP:

        raise ValueError(
            f"Expected {RECORDS_PER_TIMESTEP} "
            f"records per timestep, got "
            f"{matrix.shape[0]}"
        )

    return matrix


def build_scenario_timesteps(df):

    timesteps = []

    # The Phase 5.10 data is ordered by timestep.
    # Every timestep contains 15 CAN/HIL records.

    for start in range(
        0,
        len(df),
        RECORDS_PER_TIMESTEP
    ):

        group = df.iloc[
            start:
            start + RECORDS_PER_TIMESTEP
        ]

        if len(group) != RECORDS_PER_TIMESTEP:
            break

        matrix = build_timestep_matrix(
            group
        )

        # Collapse the 15 records into one
        # timestep feature vector.
        timestep = np.mean(
            matrix,
            axis=0
        )

        timesteps.append(
            timestep
        )

    return np.asarray(
        timesteps,
        dtype=np.float64
    )


def main():

    print("=" * 70)
    print("PHASE 5.12 EXACT WINDOW FORENSICS")
    print("=" * 70)
    print()

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False
    )

    baseline = df[
        df["scenario_id"].astype(str)
        == "S00_NORMAL_BASELINE"
    ].reset_index(drop=True)

    c07 = df[
        df["scenario_id"].astype(str)
        == "C07_HEARTBEAT_INTEGRITY_ATTACK"
    ].reset_index(drop=True)

    print(
        f"S00 rows: {len(baseline):,}"
    )

    print(
        f"C07 rows: {len(c07):,}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print()

    print(
        "Building S00 timestep matrix..."
    )

    baseline_timesteps = (
        build_scenario_timesteps(
            baseline
        )
    )

    print(
        "Building C07 timestep matrix..."
    )

    c07_timesteps = (
        build_scenario_timesteps(
            c07
        )
    )

    print()

    print(
        "S00 timestep matrix:",
        baseline_timesteps.shape
    )

    print(
        "C07 timestep matrix:",
        c07_timesteps.shape
    )

    print()

    # ------------------------------------------------------------
    # Baseline statistics
    # ------------------------------------------------------------

    baseline_mean = (
        baseline_timesteps.mean(
            axis=0
        )
    )

    baseline_std_raw = (
        baseline_timesteps.std(
            axis=0
        )
    )

    baseline_std_used = np.maximum(
        baseline_std_raw,
        1e-6
    )

    # ------------------------------------------------------------
    # Build EXACT 12-step C07 window
    # ------------------------------------------------------------

    window = c07_timesteps[
        140:
        140 + OBSERVATION_STEPS
    ]

    print(
        "Exact C07 window shape:",
        window.shape
    )

    print()

    z = (
        window
        - baseline_mean.reshape(
            1,
            -1
        )
    ) / baseline_std_used.reshape(
        1,
        -1
    )

    abs_z = np.abs(z)

    # ------------------------------------------------------------
    # Exact feature_z_rms
    # ------------------------------------------------------------

    feature_z_rms = np.sqrt(
        np.mean(
            np.square(z),
            axis=(0, 1)
        )
    )

    feature_z_max = np.max(
        abs_z,
        axis=(0, 1)
    )

    overall_feature_z_rms = np.sqrt(
        np.mean(
            np.square(z)
        )
    )

    overall_feature_z_max = np.max(
        abs_z
    )

    diagonal_mahalanobis = np.mean(
        np.square(z)
    )

    print("=" * 70)
    print("EXACT WINDOW OOD METRICS")
    print("=" * 70)

    print(
        f"feature_z_rms           = "
        f"{overall_feature_z_rms:.12g}"
    )

    print(
        f"feature_z_max           = "
        f"{overall_feature_z_max:.12g}"
    )

    print(
        f"diagonal_mahalanobis    = "
        f"{diagonal_mahalanobis:.12g}"
    )

    print()

    # ------------------------------------------------------------
    # Find exact offending feature
    # ------------------------------------------------------------

    r, f = np.unravel_index(
        np.argmax(abs_z),
        abs_z.shape
    )

    print("=" * 70)
    print("MAX FEATURE DEVIATION")
    print("=" * 70)

    print(
        f"Window timestep offset : {r}"
    )

    print(
        f"Feature index          : {f}"
    )

    print(
        f"Feature name           : "
        f"{FEATURES[f]}"
    )

    print(
        f"Z-score                : "
        f"{z[r, f]:.12g}"
    )

    print(
        f"C07 value              : "
        f"{window[r, f]:.12g}"
    )

    print(
        f"Baseline mean          : "
        f"{baseline_mean[f]:.12g}"
    )

    print(
        f"Baseline raw std       : "
        f"{baseline_std_raw[f]:.12g}"
    )

    print(
        f"Baseline std used      : "
        f"{baseline_std_used[f]:.12g}"
    )

    print()

    # ------------------------------------------------------------
    # Top 20 feature deviations
    # ------------------------------------------------------------

    print("=" * 70)
    print("TOP 20 FEATURE Z-SCORES")
    print("=" * 70)

    order = np.argsort(
        feature_z_max
    )[::-1]

    for rank, f in enumerate(
        order[:20],
        start=1
    ):

        print(
            f"{rank:02d}: "
            f"{FEATURES[f]} | "
            f"max_abs_z="
            f"{feature_z_max[f]:.12g} | "
            f"mean="
            f"{baseline_mean[f]:.12g} | "
            f"raw_std="
            f"{baseline_std_raw[f]:.12g} | "
            f"std_used="
            f"{baseline_std_used[f]:.12g}"
        )

    print()

    # ------------------------------------------------------------
    # Compare against reported Phase 5.12 value
    # ------------------------------------------------------------

    REPORTED_FEATURE_Z_RMS = 1118.960693
    REPORTED_FEATURE_Z_MAX = 5121.662598
    REPORTED_MAHALANOBIS = 1252073.125

    print("=" * 70)
    print("COMPARISON WITH PHASE 5.12")
    print("=" * 70)

    print(
        f"Reported feature_z_rms : "
        f"{REPORTED_FEATURE_Z_RMS}"
    )

    print(
        f"Reconstructed RMS      : "
        f"{overall_feature_z_rms:.12g}"
    )

    print()

    print(
        f"Reported feature_z_max : "
        f"{REPORTED_FEATURE_Z_MAX}"
    )

    print(
        f"Reconstructed max      : "
        f"{overall_feature_z_max:.12g}"
    )

    print()

    print(
        f"Reported Mahalanobis   : "
        f"{REPORTED_MAHALANOBIS}"
    )

    print(
        f"Reconstructed value    : "
        f"{diagonal_mahalanobis:.12g}"
    )

    print()

    print("=" * 70)
    print("FORENSIC ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
