from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/cyber_telemetry/crss_2024_cyber_telemetry.parquet"

OUTPUT = ROOT / "data/processed/temporal/v2/crss_2024_temporal_cyber_physical_states_v2.parquet"
SUMMARY = ROOT / "data/processed/temporal/v2/crss_2024_temporal_sequence_summary_v2.parquet"
SCHEMA = ROOT / "data/schemas/temporal/crss_2024_temporal_state_schema_v2.json"
QUALITY = ROOT / "experiments/temporal/v2/crss_2024_temporal_v2_quality.json"

SEED = 20260903
SEQUENCE_LENGTH = 24
STEP_SECONDS = 5

rng = np.random.default_rng(SEED)

print("=" * 100)
print("CRSS 2024 — TEMPORAL V2 CYBER-PHYSICAL STATE GENERATION")
print("=" * 100)

# ------------------------------------------------------------------
# LOAD BASE CYBER TELEMETRY
# ------------------------------------------------------------------

df = pd.read_parquet(INPUT)

print(f"Base scenarios : {len(df):,}")

required = [
    "CASENUM",
    "VEH_NO",
    "telemetry_scenario_id",
    "synthetic_cyber_anomaly",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# One row per vehicle/scenario in Phase 2
base = df.copy()

# Remove accidental duplicate scenario rows if any
base = (
    base.sort_values(["telemetry_scenario_id"])
        .drop_duplicates("telemetry_scenario_id")
        .reset_index(drop=True)
)

n = len(base)

print(f"Unique scenarios: {n:,}")

# ------------------------------------------------------------------
# SCENARIO TYPES
# ------------------------------------------------------------------

# Preserve approximately the existing 1,431 anomalous scenarios.
# We regenerate temporal behavior around the existing scenario label.
base_anomaly = base["synthetic_cyber_anomaly"].astype(int).to_numpy()

anomalous_idx = np.where(base_anomaly == 1)[0]
normal_idx = np.where(base_anomaly == 0)[0]

print(f"Anomalous scenarios: {len(anomalous_idx):,}")
print(f"Normal scenarios   : {len(normal_idx):,}")

# ------------------------------------------------------------------
# EXPAND TO 24 TEMPORAL STEPS
# ------------------------------------------------------------------

scenario_index = np.repeat(np.arange(n), SEQUENCE_LENGTH)
time_step = np.tile(np.arange(SEQUENCE_LENGTH), n)

out = pd.DataFrame({
    "CASENUM": base["CASENUM"].to_numpy()[scenario_index],
    "VEH_NO": base["VEH_NO"].to_numpy()[scenario_index],
    "telemetry_scenario_id": base["telemetry_scenario_id"].to_numpy()[scenario_index],
    "time_step": time_step,
    "relative_time_sec": time_step * STEP_SECONDS,
    "sequence_length": SEQUENCE_LENGTH,
})

# ------------------------------------------------------------------
# COPY BASE FEATURES
# ------------------------------------------------------------------

base_cols = [
    c for c in base.columns
    if c not in [
        "CASENUM",
        "VEH_NO",
        "telemetry_scenario_id",
        "synthetic_cyber_anomaly",
        "cyber_anomaly_severity",
        "future_anomaly_3step",
        "future_anomaly_6step",
        "predictive_warning_score",
        "predictive_warning_state",
    ]
]

for c in base_cols:
    out[c] = base[c].to_numpy()[scenario_index]

# ------------------------------------------------------------------
# TEMPORAL ANOMALY EPISODE GENERATION
# ------------------------------------------------------------------

anomaly_matrix = np.zeros((n, SEQUENCE_LENGTH), dtype=np.int8)

# Store phases for auditing
phase_matrix = np.full(
    (n, SEQUENCE_LENGTH),
    "NORMAL",
    dtype=object
)

# Each anomalous scenario gets a localized event.
#
# Structure:
#
#   NORMAL → PRECURSOR → ONSET → ACTIVE → RECOVERY
#
# Event starts between steps 8 and 15.
# This intentionally creates future-positive labels before the event.
# Duration is variable rather than the full sequence.

event_metadata = []

for idx in anomalous_idx:

    onset = int(rng.integers(10, 17))

    precursor_length = int(
        rng.integers(3, 6)
    )

    active_length = int(
        rng.integers(2, 6)
    )

    precursor_start = max(
        0,
        onset - precursor_length
    )

    active_end = min(
        SEQUENCE_LENGTH - 1,
        onset + active_length - 1
    )

    recovery_length = int(
        rng.integers(2, 5)
    )

    recovery_end = min(
        SEQUENCE_LENGTH - 1,
        active_end + recovery_length
    )

    # PRECURSOR
    phase_matrix[idx, precursor_start:onset] = "PRECURSOR"

    # ONSET
    phase_matrix[idx, onset] = "ONSET"

    # ACTIVE
    if active_end >= onset + 1:
        phase_matrix[idx, onset + 1:active_end + 1] = "ACTIVE_ANOMALY"

    # RECOVERY
    if recovery_end > active_end:
        phase_matrix[idx, active_end + 1:recovery_end + 1] = "RECOVERY"

    # Binary anomaly is only active during ONSET + ACTIVE.
    anomaly_matrix[
        idx,
        onset:active_end + 1
    ] = 1

    event_metadata.append({
        "telemetry_scenario_id": base.iloc[idx]["telemetry_scenario_id"],
        "onset_step": onset,
        "precursor_start": precursor_start,
        "active_end": active_end,
        "recovery_end": recovery_end,
        "active_duration_steps": active_end - onset + 1,
    })

# ------------------------------------------------------------------
# NORMAL SCENARIOS
# ------------------------------------------------------------------

for idx in normal_idx:

    # A small fraction receives benign transient fluctuations.
    # These are NOT cyber anomalies.
    if rng.random() < 0.08:

        start = int(
            rng.integers(5, 19)
        )

        end = min(
            SEQUENCE_LENGTH,
            start + int(rng.integers(1, 3))
        )

        phase_matrix[idx, start:end] = "BENIGN_TRANSIENT"

# ------------------------------------------------------------------
# ADD TEMPORAL ANOMALY LABEL
# ------------------------------------------------------------------

out["synthetic_cyber_anomaly"] = (
    anomaly_matrix.reshape(-1)
)

out["temporal_phase"] = (
    phase_matrix.reshape(-1)
)

# ------------------------------------------------------------------
# CREATE PRECURSOR DYNAMICS
# ------------------------------------------------------------------

t = np.tile(
    np.arange(SEQUENCE_LENGTH),
    n
)

scenario_ids = np.repeat(
    np.arange(n),
    SEQUENCE_LENGTH
)

phase = out["temporal_phase"].to_numpy()

precursor = (
    (phase == "PRECURSOR") |
    (phase == "ONSET")
).astype(float)

active = (
    phase == "ACTIVE_ANOMALY"
).astype(float)

recovery = (
    phase == "RECOVERY"
).astype(float)

# Smooth progression within each scenario.
# The temporal state intentionally changes before anomaly onset.

progress = np.zeros(len(out))

for i, meta in enumerate(event_metadata):

    mask = scenario_ids == anomalous_idx[i]

    onset = meta["onset_step"]
    start = meta["precursor_start"]
    end = meta["active_end"]

    local_t = t[mask]

    p = np.zeros(len(local_t))

    # precursor ramp
    precursor_mask = (
        (local_t >= start) &
        (local_t < onset)
    )

    if onset > start:
        p[precursor_mask] = (
            (local_t[precursor_mask] - start + 1)
            / (onset - start + 1)
        )

    # anomaly ramp
    active_mask = (
        local_t >= onset
    ) & (
        local_t <= end
    )

    if end >= onset:
        p[active_mask] = np.maximum(
            p[active_mask],
            0.7 + 0.3 * (
                (local_t[active_mask] - onset + 1)
                / (end - onset + 1)
            )
        )

    # recovery
    recovery_mask = local_t > end

    if np.any(recovery_mask):
        denom = max(
            1,
            meta["recovery_end"] - end
        )

        p[recovery_mask] = np.maximum(
            0,
            1.0 - (
                local_t[recovery_mask] - end
            ) / denom
        )

    progress[mask] = np.clip(p, 0, 1)

# ------------------------------------------------------------------
# APPLY PHYSICAL/CYBER SIGNAL PERTURBATIONS
# ------------------------------------------------------------------

# These are deliberately based on existing telemetry columns.
# We perturb signals progressively rather than simply labeling rows.

def perturb_column(
    column,
    direction,
    magnitude,
    noise_scale=0.02
):

    if column not in out.columns:
        return

    values = pd.to_numeric(
        out[column],
        errors="coerce"
    ).fillna(0).to_numpy(dtype=float)

    signal = (
        direction *
        magnitude *
        progress
    )

    noise = (
        rng.normal(
            0,
            noise_scale * magnitude,
            len(values)
        ) *
        progress
    )

    out[column] = values + signal + noise


# Message / ECU behavior
perturb_column(
    "can_message_rate",
    direction=1,
    magnitude=20,
    noise_scale=0.20
)

perturb_column(
    "can_message_rate_deviation",
    direction=1,
    magnitude=8,
    noise_scale=0.15
)

perturb_column(
    "can_interarrival_jitter_ms",
    direction=1,
    magnitude=4,
    noise_scale=0.25
)

perturb_column(
    "can_bus_load_pct",
    direction=1,
    magnitude=10,
    noise_scale=0.15
)

perturb_column(
    "ecu_state_transition_rate",
    direction=1,
    magnitude=0.15,
    noise_scale=0.20
)

perturb_column(
    "diagnostic_event_rate",
    direction=1,
    magnitude=0.20,
    noise_scale=0.20
)

perturb_column(
    "uds_request_rate",
    direction=1,
    magnitude=0.25,
    noise_scale=0.20
)

perturb_column(
    "authentication_failure_rate",
    direction=1,
    magnitude=0.20,
    noise_scale=0.20
)

# Sensor consistency
perturb_column(
    "sensor_plausibility_score",
    direction=-1,
    magnitude=0.25,
    noise_scale=0.10
)

perturb_column(
    "sensor_disagreement_rate",
    direction=1,
    magnitude=0.20,
    noise_scale=0.15
)

perturb_column(
    "redundant_sensor_divergence",
    direction=1,
    magnitude=0.15,
    noise_scale=0.15
)

# Connectivity
perturb_column(
    "packet_loss_rate",
    direction=1,
    magnitude=0.25,
    noise_scale=0.15
)

perturb_column(
    "latency_ms",
    direction=1,
    magnitude=80,
    noise_scale=0.20
)

perturb_column(
    "latency_jitter_ms",
    direction=1,
    magnitude=30,
    noise_scale=0.20
)

perturb_column(
    "connectivity_drop_rate",
    direction=1,
    magnitude=0.20,
    noise_scale=0.15
)

perturb_column(
    "telemetry_gap_duration_ms",
    direction=1,
    magnitude=500,
    noise_scale=0.20
)

# Communication integrity
perturb_column(
    "message_integrity_failure_rate",
    direction=1,
    magnitude=0.20,
    noise_scale=0.15
)

perturb_column(
    "replay_indicator_rate",
    direction=1,
    magnitude=0.20,
    noise_scale=0.15
)

perturb_column(
    "sequence_counter_anomaly_rate",
    direction=1,
    magnitude=0.20,
    noise_scale=0.15
)

perturb_column(
    "unexpected_source_id_rate",
    direction=1,
    magnitude=0.20,
    noise_scale=0.15
)

# ------------------------------------------------------------------
# CLIP PHYSICAL / CYBER VARIABLES
# ------------------------------------------------------------------

bounded_zero_one = [
    "sensor_plausibility_score",
]

for c in bounded_zero_one:
    if c in out.columns:
        out[c] = np.clip(
            pd.to_numeric(out[c], errors="coerce"),
            0,
            1
        )

bounded_rate = [
    "packet_loss_rate",
    "connectivity_drop_rate",
    "message_integrity_failure_rate",
    "replay_indicator_rate",
    "sequence_counter_anomaly_rate",
    "unexpected_source_id_rate",
    "sensor_disagreement_rate",
    "redundant_sensor_divergence",
    "authentication_failure_rate",
]

for c in bounded_rate:
    if c in out.columns:
        out[c] = np.clip(
            pd.to_numeric(out[c], errors="coerce"),
            0,
            1
        )

if "can_bus_load_pct" in out.columns:
    out["can_bus_load_pct"] = np.clip(
        out["can_bus_load_pct"],
        0,
        100
    )

# ------------------------------------------------------------------
# TEMPORAL DERIVATIVES
# ------------------------------------------------------------------

temporal_signal_columns = [
    "can_message_rate",
    "can_message_rate_deviation",
    "can_interarrival_jitter_ms",
    "can_bus_load_pct",
    "ecu_state_transition_rate",
    "diagnostic_event_rate",
    "uds_request_rate",
    "authentication_failure_rate",
    "sensor_plausibility_score",
    "sensor_disagreement_rate",
    "redundant_sensor_divergence",
    "packet_loss_rate",
    "latency_ms",
    "latency_jitter_ms",
    "connectivity_drop_rate",
    "telemetry_gap_duration_ms",
    "message_integrity_failure_rate",
    "replay_indicator_rate",
    "sequence_counter_anomaly_rate",
    "unexpected_source_id_rate",
]

for c in temporal_signal_columns:

    if c not in out.columns:
        continue

    values = pd.to_numeric(
        out[c],
        errors="coerce"
    ).fillna(0)

    out[f"d1_{c}"] = (
        values
        .groupby(out["telemetry_scenario_id"])
        .diff()
        .fillna(0)
    )

# ------------------------------------------------------------------
# ROLLING FEATURES
# ------------------------------------------------------------------

rolling_sources = [
    "can_message_rate_deviation",
    "can_interarrival_jitter_ms",
    "ecu_state_transition_rate",
    "authentication_failure_rate",
    "sensor_disagreement_rate",
    "packet_loss_rate",
    "latency_ms",
    "latency_jitter_ms",
    "telemetry_gap_duration_ms",
    "message_integrity_failure_rate",
    "replay_indicator_rate",
    "sequence_counter_anomaly_rate",
    "unexpected_source_id_rate",
]

for c in rolling_sources:

    if c not in out.columns:
        continue

    grouped = (
        out.groupby("telemetry_scenario_id")[c]
    )

    out[f"rolling_mean_3_{c}"] = (
        grouped
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    out[f"rolling_std_3_{c}"] = (
        grouped
        .rolling(3, min_periods=1)
        .std()
        .fillna(0)
        .reset_index(level=0, drop=True)
    )

# ------------------------------------------------------------------
# CYBER INSTABILITY INDEX
# ------------------------------------------------------------------

def safe_col(name):
    if name in out.columns:
        return pd.to_numeric(
            out[name],
            errors="coerce"
        ).fillna(0).to_numpy()
    return np.zeros(len(out))

cyber_instability = (
    0.15 * safe_col("can_message_rate_deviation") +
    0.10 * safe_col("can_interarrival_jitter_ms") +
    0.10 * safe_col("ecu_state_transition_rate") +
    0.10 * safe_col("authentication_failure_rate") +
    0.10 * safe_col("sensor_disagreement_rate") +
    0.10 * safe_col("packet_loss_rate") +
    0.10 * safe_col("latency_jitter_ms") +
    0.10 * safe_col("message_integrity_failure_rate") +
    0.075 * safe_col("replay_indicator_rate") +
    0.075 * safe_col("sequence_counter_anomaly_rate") +
    0.05 * safe_col("unexpected_source_id_rate")
)

# Normalize robustly
low = np.nanpercentile(cyber_instability, 1)
high = np.nanpercentile(cyber_instability, 99)

cyber_instability = (
    (cyber_instability - low)
    / max(high - low, 1e-9)
)

out["cyber_instability_index"] = np.clip(
    cyber_instability,
    0,
    1
)

# ------------------------------------------------------------------
# RESILIENCE / DEGRADATION SIGNALS
# ------------------------------------------------------------------

connectivity_risk = (
    0.35 * safe_col("packet_loss_rate") +
    0.25 * safe_col("connectivity_drop_rate") +
    0.20 * np.clip(
        safe_col("latency_ms") / 500,
        0,
        1
    ) +
    0.20 * np.clip(
        safe_col("telemetry_gap_duration_ms") / 3000,
        0,
        1
    )
)

out["connectivity_degradation_rate"] = np.clip(
    connectivity_risk,
    0,
    1
)

out["resilience_degradation_rate"] = np.clip(
    0.6 * out["cyber_instability_index"].to_numpy() +
    0.4 * out["connectivity_degradation_rate"].to_numpy(),
    0,
    1
)

out["state_transition_magnitude"] = np.abs(
    out["resilience_degradation_rate"]
    .groupby(out["telemetry_scenario_id"])
    .diff()
    .fillna(0)
)

# ------------------------------------------------------------------
# PREDICTIVE WARNING SCORE
# ------------------------------------------------------------------

warning_score = (
    0.35 * out["cyber_instability_index"].to_numpy() +
    0.25 * out["connectivity_degradation_rate"].to_numpy() +
    0.20 * np.clip(
        out["state_transition_magnitude"].to_numpy() * 5,
        0,
        1
    ) +
    0.20 * progress
)

out["predictive_warning_score"] = np.clip(
    warning_score,
    0,
    1
)

# Warning state uses only current/past observable state.
warning_state = np.select(
    [
        out["predictive_warning_score"] >= 0.75,
        out["predictive_warning_score"] >= 0.50,
        out["predictive_warning_score"] >= 0.25,
    ],
    [
        "CRITICAL",
        "WARNING",
        "WATCH",
    ],
    default="NORMAL"
)

out["predictive_warning_state"] = warning_state

# ------------------------------------------------------------------
# FUTURE LABELS
# ------------------------------------------------------------------

group = out.groupby(
    "telemetry_scenario_id",
    sort=False
)["synthetic_cyber_anomaly"]

future3 = (
    group
    .shift(-1)
    .fillna(0)
    .astype(int)
)

s2 = (
    group.shift(-2)
    .fillna(0)
    .astype(int)
)

s3 = (
    group.shift(-3)
    .fillna(0)
    .astype(int)
)

s4 = (
    group.shift(-4)
    .fillna(0)
    .astype(int)
)

s5 = (
    group.shift(-5)
    .fillna(0)
    .astype(int)
)

s6 = (
    group.shift(-6)
    .fillna(0)
    .astype(int)
)

out["future_anomaly_3step"] = (
    (future3 + s2 + s3) > 0
).astype(np.int8)

out["future_anomaly_6step"] = (
    (future3 + s2 + s3 + s4 + s5 + s6) > 0
).astype(np.int8)

# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------

summary = (
    out.groupby("telemetry_scenario_id")
       .agg(
           CASENUM=("CASENUM", "first"),
           VEH_NO=("VEH_NO", "first"),
           anomaly_steps=("synthetic_cyber_anomaly", "sum"),
           first_anomaly_step=(
               "time_step",
               lambda x: (
                   int(x[out.loc[x.index,
                       "synthetic_cyber_anomaly"] == 1].min())
                   if (
                       out.loc[x.index,
                       "synthetic_cyber_anomaly"] == 1
                   ).any()
                   else -1
               )
           ),
           max_cyber_instability=(
               "cyber_instability_index",
               "max"
           ),
           max_connectivity_degradation=(
               "connectivity_degradation_rate",
               "max"
           ),
           max_warning_score=(
               "predictive_warning_score",
               "max"
           )
       )
       .reset_index()
)

# ------------------------------------------------------------------
# QUALITY GATE
# ------------------------------------------------------------------

checks = []

def check(name, condition, detail):
    checks.append({
        "name": name,
        "passed": bool(condition),
        "detail": str(detail)
    })

expected_rows = n * SEQUENCE_LENGTH

check(
    "dataset_nonempty",
    len(out) > 0,
    f"rows={len(out):,}"
)

check(
    "expected_row_count",
    len(out) == expected_rows,
    f"expected={expected_rows:,}, actual={len(out):,}"
)

check(
    "scenario_count",
    out["telemetry_scenario_id"].nunique() == n,
    f"expected={n:,}"
)

check(
    "sequence_length",
    out.groupby("telemetry_scenario_id")
       .size()
       .eq(SEQUENCE_LENGTH)
       .all(),
    "all sequences contain 24 steps"
)

check(
    "unique_scenario_timestep",
    not out.duplicated(
        ["telemetry_scenario_id", "time_step"]
    ).any(),
    "no duplicate scenario/timestep"
)

check(
    "future3_binary",
    set(out["future_anomaly_3step"].unique()).issubset({0,1}),
    "3-step labels are binary"
)

check(
    "future6_binary",
    set(out["future_anomaly_6step"].unique()).issubset({0,1}),
    "6-step labels are binary"
)

check(
    "horizon_separation",
    (
        out["future_anomaly_6step"].sum()
        >
        out["future_anomaly_3step"].sum()
    ),
    (
        f"future3={out['future_anomaly_3step'].sum():,}, "
        f"future6={out['future_anomaly_6step'].sum():,}"
    )
)

check(
    "anomaly_not_full_sequence",
    (
        summary["anomaly_steps"] < SEQUENCE_LENGTH
    ).all(),
    "anomalies do not occupy all 24 steps"
)

check(
    "predictive_warning_valid",
    (
        out["predictive_warning_score"].between(0,1).all()
    ),
    "warning score in [0,1]"
)

check(
    "resilience_degradation_valid",
    (
        out["resilience_degradation_rate"]
        .between(0,1)
        .all()
    ),
    "degradation rate in [0,1]"
)

check(
    "connectivity_degradation_valid",
    (
        out["connectivity_degradation_rate"]
        .between(0,1)
        .all()
    ),
    "connectivity degradation in [0,1]"
)

# ------------------------------------------------------------------
# WRITE OUTPUTS
# ------------------------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
SUMMARY.parent.mkdir(parents=True, exist_ok=True)
SCHEMA.parent.mkdir(parents=True, exist_ok=True)
QUALITY.parent.mkdir(parents=True, exist_ok=True)

out.to_parquet(
    OUTPUT,
    index=False
)

summary.to_parquet(
    SUMMARY,
    index=False
)

schema = {
    "dataset": "CRSS 2024 Temporal Cyber-Physical States V2",
    "sequence_length": SEQUENCE_LENGTH,
    "step_seconds": STEP_SECONDS,
    "scenario_count": int(n),
    "row_count": int(len(out)),
    "seed": SEED,
    "targets": [
        "future_anomaly_3step",
        "future_anomaly_6step"
    ],
    "temporal_phases": [
        "NORMAL",
        "BENIGN_TRANSIENT",
        "PRECURSOR",
        "ONSET",
        "ACTIVE_ANOMALY",
        "RECOVERY"
    ],
    "note": (
        "Cyber telemetry is synthetic research telemetry. "
        "NHTSA CRSS data provide physical safety context and "
        "are not cyberattack ground truth."
    )
}

with open(SCHEMA, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)

passed = sum(
    c["passed"]
    for c in checks
)

failed = len(checks) - passed

quality = {
    "status": "PASS" if failed == 0 else "FAIL",
    "rows": int(len(out)),
    "columns": int(len(out.columns)),
    "scenarios": int(
        out["telemetry_scenario_id"].nunique()
    ),
    "future3_positive": int(
        out["future_anomaly_3step"].sum()
    ),
    "future6_positive": int(
        out["future_anomaly_6step"].sum()
    ),
    "future3_rate": float(
        out["future_anomaly_3step"].mean()
    ),
    "future6_rate": float(
        out["future_anomaly_6step"].mean()
    ),
    "checks": checks,
    "seed": SEED
}

with open(QUALITY, "w", encoding="utf-8") as f:
    json.dump(quality, f, indent=2)

# ------------------------------------------------------------------
# REPORT
# ------------------------------------------------------------------

print()
print("TEMPORAL V2 RESULTS")
print("-" * 100)

print(f"Rows                  : {len(out):,}")
print(f"Columns               : {len(out.columns):,}")
print(f"Scenarios             : {n:,}")
print(
    f"3-step positives      : "
    f"{out['future_anomaly_3step'].sum():,} "
    f"({out['future_anomaly_3step'].mean():.6%})"
)
print(
    f"6-step positives      : "
    f"{out['future_anomaly_6step'].sum():,} "
    f"({out['future_anomaly_6step'].mean():.6%})"
)

print()
print("TEMPORAL PHASE DISTRIBUTION")
print("-" * 100)

print(
    out["temporal_phase"]
    .value_counts()
    .to_string()
)

print()
print("QUALITY GATE")
print("-" * 100)

for c in checks:
    print(
        f"[{'PASS' if c['passed'] else 'FAIL'}] "
        f"{c['name']} -- {c['detail']}"
    )

print()
print("=" * 100)
print(f"STATUS : {'PASS' if failed == 0 else 'FAIL'}")
print(f"CHECKS : {len(checks)}")
print(f"PASS   : {passed}")
print(f"FAIL   : {failed}")
print("=" * 100)

if failed:
    raise RuntimeError(
        f"Temporal V2 quality gate failed: {failed} failure(s)."
    )

print()
print("TEMPORAL V2: PASS")
print(f"OUTPUT: {OUTPUT}")
print(f"QUALITY: {QUALITY}")
