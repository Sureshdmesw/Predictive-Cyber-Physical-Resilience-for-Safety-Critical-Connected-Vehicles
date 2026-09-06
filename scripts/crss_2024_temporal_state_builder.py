from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed" / "cyber_telemetry"
    / "crss_2024_cyber_telemetry.parquet"
)

OUT = ROOT / "data" / "processed" / "temporal"
SCHEMA = ROOT / "data" / "schemas" / "temporal"
REPORT = ROOT / "experiments" / "temporal"

OUT.mkdir(parents=True, exist_ok=True)
SCHEMA.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

SEED = 20260902
rng = np.random.default_rng(SEED)

df = pd.read_parquet(INPUT)

required = [
    "CASENUM",
    "VEH_NO",
    "telemetry_scenario_id",
    "synthetic_cyber_anomaly",
    "cyber_anomaly_severity",
    "physical_cyber_interaction_score",
    "connectivity_resilience_score",
    "cyber_physical_resilience_score",
    "can_message_rate",
    "can_interarrival_jitter_ms",
    "can_bus_load_pct",
    "sensor_plausibility_score",
    "sensor_disagreement_rate",
    "packet_loss_rate",
    "latency_ms",
    "latency_jitter_ms",
    "connectivity_drop_rate",
    "telemetry_gap_duration_ms",
    "message_integrity_failure_rate",
    "replay_indicator_rate",
    "sequence_counter_anomaly_rate",
    "authentication_failure_rate"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(f"Missing telemetry fields: {missing}")

# ------------------------------------------------------------------
# Vectorized temporal expansion
# ------------------------------------------------------------------

SEQUENCE_LENGTH = 24

base = df.reset_index(drop=True).copy()

n = len(base)

expanded = base.loc[
    base.index.repeat(SEQUENCE_LENGTH)
].reset_index(drop=True)

expanded["time_step"] = np.tile(
    np.arange(SEQUENCE_LENGTH),
    n
)

expanded["relative_time_sec"] = (
    expanded["time_step"] * 5
)

expanded["sequence_length"] = SEQUENCE_LENGTH

expanded["progress"] = (
    expanded["time_step"] /
    (SEQUENCE_LENGTH - 1)
)

# ------------------------------------------------------------------
# Deterministic precursor dynamics
# ------------------------------------------------------------------

anomaly = expanded[
    "synthetic_cyber_anomaly"
].astype(float)

precursor = np.where(
    anomaly > 0,
    expanded["progress"] ** 2,
    0.0
)

# One deterministic random stream per temporal state.
m = len(expanded)

noise = {
    "message": rng.normal(0, 0.18, m),
    "jitter": rng.uniform(0, 1.5, m),
    "bus": rng.normal(0, 8, m),
    "plausibility": rng.uniform(0, 0.25, m),
    "disagreement": rng.uniform(0, 0.20, m),
    "packet": rng.uniform(0, 0.15, m),
    "latency": rng.uniform(0, 3, m),
    "latency_jitter": rng.uniform(0, 3, m),
    "drop": rng.uniform(0, 0.5, m),
    "gap": rng.uniform(0, 5, m),
    "integrity": rng.uniform(0, 0.25, m),
    "replay": rng.uniform(0, 0.15, m),
    "sequence": rng.uniform(0, 0.20, m),
    "authentication": rng.uniform(0, 1.0, m),
    "severity": rng.uniform(0, 0.25, m),
    "interaction": rng.uniform(0, 0.20, m),
    "resilience": rng.uniform(0, 0.35, m),
    "connectivity": rng.uniform(0, 0.30, m)
}

# ------------------------------------------------------------------
# Temporal cyber signals
# ------------------------------------------------------------------

expanded["can_message_rate"] = (
    expanded["can_message_rate"]
    * (
        1 + precursor * noise["message"]
    )
)

expanded["can_interarrival_jitter_ms"] = (
    expanded["can_interarrival_jitter_ms"]
    * (
        1 + precursor * noise["jitter"]
    )
)

expanded["can_bus_load_pct"] = (
    expanded["can_bus_load_pct"]
    + precursor * noise["bus"]
)

expanded["sensor_plausibility_score"] = np.clip(
    expanded["sensor_plausibility_score"]
    - precursor * noise["plausibility"],
    0,
    1
)

expanded["sensor_disagreement_rate"] = np.clip(
    expanded["sensor_disagreement_rate"]
    + precursor * noise["disagreement"],
    0,
    1
)

expanded["packet_loss_rate"] = np.clip(
    expanded["packet_loss_rate"]
    + precursor * noise["packet"],
    0,
    1
)

expanded["latency_ms"] = (
    expanded["latency_ms"]
    * (
        1 + precursor * noise["latency"]
    )
)

expanded["latency_jitter_ms"] = (
    expanded["latency_jitter_ms"]
    * (
        1 + precursor * noise["latency_jitter"]
    )
)

expanded["connectivity_drop_rate"] = (
    expanded["connectivity_drop_rate"]
    + precursor * noise["drop"]
)

expanded["telemetry_gap_duration_ms"] = (
    expanded["telemetry_gap_duration_ms"]
    * (
        1 + precursor * noise["gap"]
    )
)

expanded["message_integrity_failure_rate"] = (
    expanded["message_integrity_failure_rate"]
    + precursor * noise["integrity"]
)

expanded["replay_indicator_rate"] = (
    expanded["replay_indicator_rate"]
    + precursor * noise["replay"]
)

expanded["sequence_counter_anomaly_rate"] = (
    expanded["sequence_counter_anomaly_rate"]
    + precursor * noise["sequence"]
)

expanded["authentication_failure_rate"] = (
    expanded["authentication_failure_rate"]
    + precursor * noise["authentication"]
)

expanded["cyber_anomaly_severity"] = np.clip(
    expanded["cyber_anomaly_severity"]
    + precursor * noise["severity"],
    0,
    1
)

expanded["physical_cyber_interaction_score"] = np.clip(
    expanded["physical_cyber_interaction_score"]
    + precursor * noise["interaction"],
    0,
    1
)

expanded["cyber_physical_resilience_score"] = np.clip(
    expanded["cyber_physical_resilience_score"]
    - precursor * noise["resilience"],
    0,
    1
)

expanded["connectivity_resilience_score"] = np.clip(
    expanded["connectivity_resilience_score"]
    - precursor * noise["connectivity"],
    0,
    1
)

# ------------------------------------------------------------------
# Sort before temporal operations
# ------------------------------------------------------------------

expanded = expanded.sort_values(
    ["telemetry_scenario_id", "time_step"]
).reset_index(drop=True)

g = expanded.groupby(
    "telemetry_scenario_id",
    sort=False
)

# ------------------------------------------------------------------
# Forecast labels
# ------------------------------------------------------------------

future3 = (
    g["synthetic_cyber_anomaly"]
    .transform(
        lambda x:
        x.shift(-1)
        .rolling(3, min_periods=1)
        .max()
    )
)

future6 = (
    g["synthetic_cyber_anomaly"]
    .transform(
        lambda x:
        x.shift(-1)
        .rolling(6, min_periods=1)
        .max()
    )
)

expanded["future_anomaly_3step"] = (
    future3.fillna(0).astype("int8")
)

expanded["future_anomaly_6step"] = (
    future6.fillna(0).astype("int8")
)

# ------------------------------------------------------------------
# First-order temporal derivatives
# ------------------------------------------------------------------

derivative_features = [
    "can_message_rate",
    "can_interarrival_jitter_ms",
    "can_bus_load_pct",
    "sensor_disagreement_rate",
    "packet_loss_rate",
    "latency_ms",
    "latency_jitter_ms",
    "connectivity_drop_rate",
    "telemetry_gap_duration_ms",
    "message_integrity_failure_rate",
    "replay_indicator_rate",
    "sequence_counter_anomaly_rate",
    "authentication_failure_rate",
    "cyber_anomaly_severity",
    "physical_cyber_interaction_score",
    "connectivity_resilience_score",
    "cyber_physical_resilience_score"
]

for feature in derivative_features:
    expanded[f"d1_{feature}"] = (
        g[feature].diff()
    )

# ------------------------------------------------------------------
# Rolling statistics
# ------------------------------------------------------------------

rolling_features = [
    "packet_loss_rate",
    "latency_ms",
    "sensor_disagreement_rate",
    "message_integrity_failure_rate",
    "cyber_anomaly_severity",
    "cyber_physical_resilience_score"
]

for feature in rolling_features:

    expanded[f"rolling_mean_3_{feature}"] = (
        g[feature]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    expanded[f"rolling_std_3_{feature}"] = (
        g[feature]
        .rolling(3, min_periods=1)
        .std()
        .fillna(0)
        .reset_index(level=0, drop=True)
    )

# ------------------------------------------------------------------
# Cyber instability
# ------------------------------------------------------------------

expanded["cyber_instability_index"] = (
    0.20 *
    expanded["packet_loss_rate"].clip(0, 1)
    +
    0.20 *
    (expanded["latency_ms"] / 500).clip(0, 1)
    +
    0.15 *
    expanded["sensor_disagreement_rate"].clip(0, 1)
    +
    0.15 *
    expanded[
        "message_integrity_failure_rate"
    ].clip(0, 1)
    +
    0.10 *
    expanded["replay_indicator_rate"].clip(0, 1)
    +
    0.10 *
    expanded[
        "sequence_counter_anomaly_rate"
    ].clip(0, 1)
    +
    0.10 *
    (
        expanded["authentication_failure_rate"]
        /
        (1 + expanded["authentication_failure_rate"])
    )
)

expanded["resilience_degradation_rate"] = (
    -expanded[
        "d1_cyber_physical_resilience_score"
    ]
)

expanded["connectivity_degradation_rate"] = (
    -expanded[
        "d1_connectivity_resilience_score"
    ]
)

# ------------------------------------------------------------------
# State transition magnitude
# ------------------------------------------------------------------

state_delta_columns = [
    f"d1_{x}" for x in derivative_features
]

expanded["state_transition_magnitude"] = np.sqrt(
    expanded[state_delta_columns]
    .fillna(0)
    .pow(2)
    .sum(axis=1)
)

# Normalize magnitude for numerical stability.
q99 = expanded[
    "state_transition_magnitude"
].quantile(0.99)

if q99 > 0:
    expanded[
        "state_transition_magnitude"
    ] = (
        expanded["state_transition_magnitude"]
        / q99
    ).clip(0, 1)

# ------------------------------------------------------------------
# Predictive warning score
# ------------------------------------------------------------------

expanded["predictive_warning_score"] = (
    0.30 *
    expanded["cyber_instability_index"].clip(0, 1)
    +
    0.20 *
    expanded[
        "resilience_degradation_rate"
    ].clip(lower=0).clip(upper=1)
    +
    0.15 *
    expanded[
        "connectivity_degradation_rate"
    ].clip(lower=0).clip(upper=1)
    +
    0.20 *
    expanded[
        "physical_cyber_interaction_score"
    ].clip(0, 1)
    +
    0.15 *
    expanded[
        "state_transition_magnitude"
    ].clip(0, 1)
)

expanded["predictive_warning_state"] = pd.cut(
    expanded["predictive_warning_score"],
    bins=[
        -np.inf,
        0.25,
        0.50,
        0.75,
        np.inf
    ],
    labels=[
        "NORMAL",
        "WATCH",
        "WARNING",
        "CRITICAL"
    ]
).astype(str)

# ------------------------------------------------------------------
# Remove temporary field
# ------------------------------------------------------------------

expanded = expanded.drop(
    columns=["progress"],
    errors="ignore"
)

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

output_path = (
    OUT /
    "crss_2024_temporal_cyber_physical_states.parquet"
)

expanded.to_parquet(
    output_path,
    index=False
)

# ------------------------------------------------------------------
# Sequence summary
# ------------------------------------------------------------------

summary = (
    expanded
    .groupby(
        "telemetry_scenario_id",
        as_index=False
    )
    .agg(
        sequence_length=(
            "time_step",
            "count"
        ),
        anomaly_present=(
            "synthetic_cyber_anomaly",
            "max"
        ),
        max_cyber_instability=(
            "cyber_instability_index",
            "max"
        ),
        min_resilience=(
            "cyber_physical_resilience_score",
            "min"
        ),
        max_warning_score=(
            "predictive_warning_score",
            "max"
        ),
        critical_states=(
            "predictive_warning_state",
            lambda x: int(
                (x == "CRITICAL").sum()
            )
        )
    )
)

summary.to_parquet(
    OUT /
    "crss_2024_temporal_sequence_summary.parquet",
    index=False
)

# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

schema = {
    "dataset": "CRSS 2024 Temporal Cyber-Physical State Dataset",
    "version": "2.0",
    "sequence_length": SEQUENCE_LENGTH,
    "temporal_resolution_seconds": 5,
    "time_semantics": (
        "Synthetic relative simulation time. "
        "Not observed NHTSA timestamps."
    ),
    "sequence_key": [
        "telemetry_scenario_id"
    ],
    "forecast_targets": [
        "future_anomaly_3step",
        "future_anomaly_6step"
    ],
    "temporal_features": {
        "first_order_derivatives": "d1_*",
        "rolling_statistics": [
            "rolling_mean_3_*",
            "rolling_std_3_*"
        ],
        "state_transition": [
            "state_transition_magnitude"
        ],
        "predictive": [
            "cyber_instability_index",
            "resilience_degradation_rate",
            "connectivity_degradation_rate",
            "predictive_warning_score",
            "predictive_warning_state"
        ]
    },
    "warning_states": [
        "NORMAL",
        "WATCH",
        "WARNING",
        "CRITICAL"
    ]
}

with open(
    SCHEMA /
    "crss_2024_temporal_state_schema.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(schema, f, indent=2)

# ------------------------------------------------------------------
# Quality gate
# ------------------------------------------------------------------

expected_rows = (
    df["telemetry_scenario_id"].nunique()
    * SEQUENCE_LENGTH
)

quality = {
    "status": "PASS",
    "rows": int(len(expanded)),
    "columns": int(len(expanded.columns)),
    "sequences": int(
        expanded["telemetry_scenario_id"].nunique()
    ),
    "expected_rows": int(expected_rows),
    "duplicate_rows": int(
        expanded.duplicated().sum()
    ),
    "warning_state_counts": {
        str(k): int(v)
        for k, v in expanded[
            "predictive_warning_state"
        ].value_counts().items()
    },
    "checks": {
        "nonempty": bool(len(expanded) > 0),
        "expected_sequence_size": bool(
            len(expanded) == expected_rows
        ),
        "unique_sequence_timestep": bool(
            expanded[
                [
                    "telemetry_scenario_id",
                    "time_step"
                ]
            ].duplicated().sum() == 0
        ),
        "warning_states_valid": bool(
            expanded[
                "predictive_warning_state"
            ].isin(
                [
                    "NORMAL",
                    "WATCH",
                    "WARNING",
                    "CRITICAL"
                ]
            ).all()
        ),
        "resilience_valid": bool(
            expanded[
                "cyber_physical_resilience_score"
            ].between(0, 1).all()
        ),
        "forecast_labels_valid": bool(
            expanded[
                [
                    "future_anomaly_3step",
                    "future_anomaly_6step"
                ]
            ].isin([0, 1]).all().all()
        )
    }
}

if not all(quality["checks"].values()):
    quality["status"] = "FAIL"
    raise RuntimeError(
        "Temporal quality gate failed."
    )

with open(
    REPORT /
    "crss_2024_temporal_quality.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(quality, f, indent=2)

print("=" * 80)
print("CRSS 2024 TEMPORAL CYBER-PHYSICAL STATE")
print("=" * 80)
print()
print("STATUS    :", quality["status"])
print("ROWS      :", f"{quality['rows']:,}")
print("COLUMNS   :", quality["columns"])
print("SEQUENCES :", f"{quality['sequences']:,}")
print("EXPECTED  :", f"{quality['expected_rows']:,}")
print("DUPLICATES:", quality["duplicate_rows"])
print()
print("OUTPUT:")
print(output_path)
print()
print("SCHEMA:")
print(
    SCHEMA /
    "crss_2024_temporal_state_schema.json"
)
print()
print("QUALITY:")
print(
    REPORT /
    "crss_2024_temporal_quality.json"
)
print()
print("NEXT STAGE:")
print(
    "Advanced temporal anomaly detection + "
    "predictive resilience modeling"
)
print("=" * 80)
