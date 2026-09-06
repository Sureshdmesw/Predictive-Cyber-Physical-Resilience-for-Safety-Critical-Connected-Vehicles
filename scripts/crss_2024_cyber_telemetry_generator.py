from pathlib import Path
import json
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# JSON serialization helper
# Converts NumPy/Pandas scalar types to native Python types.
# ------------------------------------------------------------------

def json_default(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if pd.isna(obj):
        return None
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


ROOT = Path(__file__).resolve().parents[1]

PHYSICAL = (
    ROOT / "data" / "processed" / "crss_2024"
    / "cyber_physical_features.parquet"
)

OUT = ROOT / "data" / "processed" / "cyber_telemetry"
SCHEMA_OUT = ROOT / "data" / "schemas" / "cyber"
REPORT_OUT = ROOT / "experiments" / "cyber_telemetry"

OUT.mkdir(parents=True, exist_ok=True)
SCHEMA_OUT.mkdir(parents=True, exist_ok=True)
REPORT_OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

SEED = 20260902
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------
# Load physical-safety feature layer
# ------------------------------------------------------------------

physical = pd.read_parquet(PHYSICAL)

required = [
    "CASENUM",
    "VEH_NO",
    "speed_limit_deviation",
    "speed_limit_deviation_abs",
    "speed_over_limit_flag",
    "rollover_flag",
    "fire_explosion_flag",
    "vehicle_damage_flag",
    "collision_event_flag",
    "roadway_complexity_index",
    "vehicle_state_observation_count",
    "VISION",
    "WEATHER",
    "physical_state_completeness"
]

missing = [c for c in required if c not in physical.columns]

if missing:
    raise RuntimeError(
        f"Missing physical features: {missing}"
    )

# ------------------------------------------------------------------
# One cyber telemetry sequence per vehicle.
#
# This is a synthetic research telemetry layer.
# It is NOT represented as observed NHTSA cyberattack data.
# ------------------------------------------------------------------

N = len(physical)

telemetry = physical[
    [
        "CASENUM",
        "VEH_NO",
        "speed_limit_deviation",
        "speed_limit_deviation_abs",
        "speed_over_limit_flag",
        "rollover_flag",
        "fire_explosion_flag",
        "vehicle_damage_flag",
        "collision_event_flag",
        "roadway_complexity_index",
        "vehicle_state_observation_count",
        "VISION",
        "WEATHER",
        "physical_state_completeness"
    ]
].copy()

# ------------------------------------------------------------------
# Vehicle / ECU topology
# ------------------------------------------------------------------

ecu_types = np.array([
    "ADAS_ECU",
    "BRAKE_ECU",
    "POWERTRAIN_ECU",
    "BODY_ECU",
    "GATEWAY_ECU",
    "TELEMATICS_ECU"
])

telemetry["ecu_count"] = rng.integers(
    6, 18, size=N
)

telemetry["can_bus_count"] = rng.integers(
    1, 4, size=N
)

telemetry["gateway_present"] = 1

telemetry["safety_critical_ecu_ratio"] = (
    rng.uniform(0.25, 0.60, size=N)
)

# ------------------------------------------------------------------
# CAN message behavior
# ------------------------------------------------------------------

telemetry["can_message_rate"] = (
    rng.normal(850, 120, size=N)
    .clip(300, 1400)
)

telemetry["can_message_rate_deviation"] = (
    rng.normal(0, 35, size=N)
)

telemetry["can_interarrival_mean_ms"] = (
    1000.0 / telemetry["can_message_rate"]
)

telemetry["can_interarrival_jitter_ms"] = (
    rng.gamma(shape=2.0, scale=0.8, size=N)
)

telemetry["can_bus_load_pct"] = (
    rng.normal(42, 10, size=N)
    .clip(5, 85)
)

# ------------------------------------------------------------------
# ECU diagnostic behavior
# ------------------------------------------------------------------

telemetry["ecu_state_transition_rate"] = (
    rng.gamma(shape=2.0, scale=0.35, size=N)
)

telemetry["diagnostic_event_rate"] = (
    rng.gamma(shape=1.5, scale=0.15, size=N)
)

telemetry["uds_request_rate"] = (
    rng.gamma(shape=1.5, scale=0.08, size=N)
)

telemetry["authentication_failure_rate"] = (
    rng.exponential(scale=0.03, size=N)
)

# ------------------------------------------------------------------
# Sensor consistency
# ------------------------------------------------------------------

telemetry["sensor_plausibility_score"] = (
    rng.beta(12, 2, size=N)
)

telemetry["sensor_disagreement_rate"] = (
    rng.beta(2, 30, size=N)
)

telemetry["redundant_sensor_divergence"] = (
    rng.normal(0.02, 0.01, size=N)
).clip(0, None)

# ------------------------------------------------------------------
# Connectivity / telemetry resilience
# ------------------------------------------------------------------

telemetry["rssi_dbm"] = (
    rng.normal(-72, 12, size=N)
).clip(-115, -35)

telemetry["packet_loss_rate"] = (
    rng.beta(1.5, 80, size=N)
)

telemetry["latency_ms"] = (
    rng.lognormal(
        mean=np.log(45),
        sigma=0.45,
        size=N
    )
).clip(5, 500)

telemetry["latency_jitter_ms"] = (
    rng.gamma(shape=2.0, scale=3.0, size=N)
)

telemetry["connectivity_drop_rate"] = (
    rng.exponential(scale=0.01, size=N)
)

telemetry["telemetry_gap_duration_ms"] = (
    rng.exponential(scale=150, size=N)
)

# ------------------------------------------------------------------
# Cryptographic / communication integrity indicators
# ------------------------------------------------------------------

telemetry["message_integrity_failure_rate"] = (
    rng.exponential(scale=0.005, size=N)
)

telemetry["replay_indicator_rate"] = (
    rng.exponential(scale=0.002, size=N)
)

telemetry["sequence_counter_anomaly_rate"] = (
    rng.exponential(scale=0.004, size=N)
)

telemetry["unexpected_source_id_rate"] = (
    rng.exponential(scale=0.002, size=N)
)

# ------------------------------------------------------------------
# Synthetic cyber anomaly state
#
# IMPORTANT:
# This is a research simulation label.
# It is not a NHTSA attack label.
# ------------------------------------------------------------------

base_anomaly_probability = (
    0.005
    + 0.025 * telemetry["packet_loss_rate"]
    + 0.020 * telemetry["sensor_disagreement_rate"]
    + 0.010 * telemetry["message_integrity_failure_rate"]
)

physical_stress = (
    0.10 * telemetry["speed_limit_deviation_abs"].fillna(0).clip(0, 40)
    + 0.20 * telemetry["rollover_flag"].fillna(0)
    + 0.15 * telemetry["collision_event_flag"].fillna(0)
)

physical_stress = physical_stress.clip(0, 1)

anomaly_probability = (
    base_anomaly_probability + 0.01 * physical_stress
).clip(0, 0.35)

telemetry["synthetic_cyber_anomaly"] = (
    rng.random(N) < anomaly_probability
).astype("int8")

# ------------------------------------------------------------------
# Inject structured anomalies into cyber telemetry.
#
# The anomaly mechanisms are intentionally interpretable:
#
# 1. message-rate anomaly
# 2. timing/jitter anomaly
# 3. sensor-consistency anomaly
# 4. connectivity degradation
# 5. authentication/integrity anomaly
# ------------------------------------------------------------------

anomaly = telemetry["synthetic_cyber_anomaly"].eq(1)

idx = telemetry.index[anomaly]

if len(idx) > 0:

    modes = rng.integers(0, 5, size=len(idx))

    # Message-rate manipulation
    m = idx[modes == 0]
    telemetry.loc[m, "can_message_rate"] *= rng.uniform(
        0.25, 2.5, size=len(m)
    )
    telemetry.loc[m, "can_message_rate_deviation"] += rng.normal(
        250, 80, size=len(m)
    )

    # Timing anomaly
    m = idx[modes == 1]
    telemetry.loc[m, "can_interarrival_jitter_ms"] *= rng.uniform(
        4, 12, size=len(m)
    )

    # Sensor disagreement
    m = idx[modes == 2]
    telemetry.loc[m, "sensor_disagreement_rate"] = rng.uniform(
        0.15, 0.60, size=len(m)
    )
    telemetry.loc[m, "sensor_plausibility_score"] = rng.uniform(
        0.25, 0.70, size=len(m)
    )

    # Connectivity degradation
    m = idx[modes == 3]
    telemetry.loc[m, "packet_loss_rate"] = rng.uniform(
        0.10, 0.65, size=len(m)
    )
    telemetry.loc[m, "latency_ms"] *= rng.uniform(
        3, 12, size=len(m)
    )
    telemetry.loc[m, "connectivity_drop_rate"] += rng.uniform(
        0.1, 0.8, size=len(m)
    )

    # Integrity/authentication anomaly
    m = idx[modes == 4]
    telemetry.loc[m, "authentication_failure_rate"] += rng.uniform(
        0.2, 2.0, size=len(m)
    )
    telemetry.loc[m, "message_integrity_failure_rate"] += rng.uniform(
        0.05, 0.50, size=len(m)
    )
    telemetry.loc[m, "replay_indicator_rate"] += rng.uniform(
        0.05, 0.30, size=len(m)
    )
    telemetry.loc[m, "sequence_counter_anomaly_rate"] += rng.uniform(
        0.05, 0.30, size=len(m)
    )

# ------------------------------------------------------------------
# Cyber anomaly severity
# ------------------------------------------------------------------

telemetry["cyber_anomaly_severity"] = (
    0.25 * telemetry["sensor_disagreement_rate"].clip(0, 1)
    + 0.20 * telemetry["packet_loss_rate"].clip(0, 1)
    + 0.20 * telemetry["message_integrity_failure_rate"].clip(0, 1)
    + 0.15 * telemetry["replay_indicator_rate"].clip(0, 1)
    + 0.10 * telemetry["sequence_counter_anomaly_rate"].clip(0, 1)
    + 0.10 * (
        telemetry["authentication_failure_rate"] /
        (1 + telemetry["authentication_failure_rate"])
    )
).clip(0, 1)

# ------------------------------------------------------------------
# Physical-cyber interaction
# ------------------------------------------------------------------

telemetry["physical_cyber_interaction_score"] = (
    telemetry["cyber_anomaly_severity"]
    * (
        0.4 * telemetry["physical_state_completeness"]
        + 0.2 * telemetry["collision_event_flag"].fillna(0)
        + 0.2 * telemetry["rollover_flag"].fillna(0)
        + 0.2 * (
            telemetry["speed_limit_deviation_abs"]
            .fillna(0)
            .clip(0, 40) / 40
        )
    )
).clip(0, 1)

# ------------------------------------------------------------------
# Preliminary resilience indicators
# ------------------------------------------------------------------

telemetry["connectivity_resilience_score"] = (
    1
    - (
        0.35 * telemetry["packet_loss_rate"].clip(0, 1)
        + 0.25 * (
            telemetry["latency_ms"] / 500
        ).clip(0, 1)
        + 0.20 * (
            telemetry["connectivity_drop_rate"] / 1
        ).clip(0, 1)
        + 0.20 * (
            telemetry["telemetry_gap_duration_ms"] / 5000
        ).clip(0, 1)
    )
).clip(0, 1)

telemetry["cyber_physical_resilience_score"] = (
    1
    - (
        0.45 * telemetry["cyber_anomaly_severity"]
        + 0.35 * (
            1 - telemetry["connectivity_resilience_score"]
        )
        + 0.20 * telemetry["physical_cyber_interaction_score"]
    )
).clip(0, 1)

# ------------------------------------------------------------------
# Deterministic research scenario identifier
# ------------------------------------------------------------------

telemetry["telemetry_scenario_id"] = (
    "CRSS24_" +
    telemetry["CASENUM"].astype(str) +
    "_V" +
    telemetry["VEH_NO"].fillna(0).astype(int).astype(str)
)

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

telemetry_path = OUT / "crss_2024_cyber_telemetry.parquet"
telemetry.to_parquet(telemetry_path, index=False)

# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

schema = {
    "dataset": "CRSS 2024 Synthetic Cyber Telemetry",
    "version": "1.0",
    "seed": SEED,
    "purpose": (
        "Synthetic edge/ECU/CAN telemetry layer for studying "
        "cyber-physical resilience. This telemetry is simulated "
        "and must not be interpreted as observed NHTSA cybersecurity data."
    ),
    "physical_ground_truth": (
        "data/processed/crss_2024/cyber_physical_features.parquet"
    ),
    "identifier": [
        "CASENUM",
        "VEH_NO",
        "telemetry_scenario_id"
    ],
    "feature_groups": {
        "vehicle_ecu_topology": [
            "ecu_count",
            "can_bus_count",
            "gateway_present",
            "safety_critical_ecu_ratio"
        ],
        "can_behavior": [
            "can_message_rate",
            "can_message_rate_deviation",
            "can_interarrival_mean_ms",
            "can_interarrival_jitter_ms",
            "can_bus_load_pct"
        ],
        "ecu_diagnostics": [
            "ecu_state_transition_rate",
            "diagnostic_event_rate",
            "uds_request_rate",
            "authentication_failure_rate"
        ],
        "sensor_consistency": [
            "sensor_plausibility_score",
            "sensor_disagreement_rate",
            "redundant_sensor_divergence"
        ],
        "connectivity": [
            "rssi_dbm",
            "packet_loss_rate",
            "latency_ms",
            "latency_jitter_ms",
            "connectivity_drop_rate",
            "telemetry_gap_duration_ms"
        ],
        "communication_integrity": [
            "message_integrity_failure_rate",
            "replay_indicator_rate",
            "sequence_counter_anomaly_rate",
            "unexpected_source_id_rate"
        ],
        "research_labels": [
            "synthetic_cyber_anomaly",
            "cyber_anomaly_severity",
            "physical_cyber_interaction_score",
            "connectivity_resilience_score",
            "cyber_physical_resilience_score"
        ]
    }
}

with open(
    SCHEMA_OUT / "crss_2024_cyber_telemetry_schema.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(schema, f, indent=2)

# ------------------------------------------------------------------
# Quality report
# ------------------------------------------------------------------

numeric_cols = telemetry.select_dtypes(
    include=[np.number]
).columns.tolist()

quality = {
    "status": "PASS",
    "rows": int(len(telemetry)),
    "columns": int(len(telemetry.columns)),
    "duplicate_rows": int(telemetry.duplicated().sum()),
    "missing_identifier_rows": int(
        telemetry[["CASENUM", "VEH_NO"]].isna().any(axis=1).sum()
    ),
    "anomaly_rows": int(
        telemetry["synthetic_cyber_anomaly"].sum()
    ),
    "anomaly_rate": float(
        telemetry["synthetic_cyber_anomaly"].mean()
    ),
    "numeric_features": len(numeric_cols),
    "checks": {
        "nonempty": len(telemetry) > 0,
        "no_duplicate_rows": telemetry.duplicated().sum() == 0,
        "identifier_present": (
            telemetry["CASENUM"].notna().all()
            and telemetry["VEH_NO"].notna().all()
        ),
        "resilience_range_valid": (
            telemetry["cyber_physical_resilience_score"]
            .between(0, 1)
            .all()
        ),
        "anomaly_severity_range_valid": (
            telemetry["cyber_anomaly_severity"]
            .between(0, 1)
            .all()
        )
    }
}

if not all(quality["checks"].values()):
    quality["status"] = "FAIL"
    raise RuntimeError(
        "Cyber telemetry quality gate failed."
    )

with open(
    REPORT_OUT / "crss_2024_cyber_telemetry_quality.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(quality, f, indent=2, default=json_default)

print("=" * 80)
print("CRSS 2024 SYNTHETIC CYBER TELEMETRY")
print("=" * 80)
print()
print("STATUS :", quality["status"])
print("ROWS   :", f"{quality['rows']:,}")
print("COLUMNS:", quality["columns"])
print("ANOMALIES:", f"{quality['anomaly_rows']:,}")
print("ANOMALY RATE:", f"{quality['anomaly_rate']:.4%}")
print()
print("OUTPUT:")
print(telemetry_path)
print()
print("SCHEMA:")
print(SCHEMA_OUT / "crss_2024_cyber_telemetry_schema.json")
print()
print("QUALITY:")
print(REPORT_OUT / "crss_2024_cyber_telemetry_quality.json")
print()
print("NEXT STAGE:")
print("Temporal cyber-physical state construction")
print("=" * 80)
