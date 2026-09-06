from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/temporal/crss_2024_temporal_cyber_physical_states.parquet"
OUTPUT = INPUT

QUALITY = ROOT / "experiments/temporal/crss_2024_temporal_quality.json"
SCHEMA = ROOT / "data/schemas/temporal/crss_2024_temporal_state_schema.json"

print("=" * 100)
print("CRSS 2024 TEMPORAL REPAIR + RELEASE GATE")
print("=" * 100)

if not INPUT.exists():
    raise FileNotFoundError(f"Missing temporal dataset: {INPUT}")

df = pd.read_parquet(INPUT)

required = [
    "telemetry_scenario_id",
    "time_step",
    "relative_time_sec",
    "sequence_length",
    "synthetic_cyber_anomaly",
    "future_anomaly_3step",
    "future_anomaly_6step",
    "cyber_physical_resilience_score",
    "cyber_instability_index",
    "state_transition_magnitude",
    "predictive_warning_score",
    "predictive_warning_state",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

df = df.sort_values(
    ["telemetry_scenario_id", "time_step"],
    kind="stable"
).reset_index(drop=True)

g = df.groupby("telemetry_scenario_id", sort=False)["synthetic_cyber_anomaly"]

# ------------------------------------------------------------------
# 1. CORRECT FUTURE LABELS
# ------------------------------------------------------------------
# True definition:
# future_anomaly_3step(t) =
#   max(anomaly(t+1), anomaly(t+2), anomaly(t+3))
#
# future_anomaly_6step(t) =
#   max(anomaly(t+1)...anomaly(t+6))
#
# The current artifact appears to have generated these in a way that
# can make the 3-step and 6-step labels unnecessarily identical.
# ------------------------------------------------------------------

future3 = (
    g.shift(-1).fillna(0)
    .astype("int8")
    .groupby(df["telemetry_scenario_id"])
    .rolling(window=3, min_periods=1)
    .max()
)

# The rolling operation above is not guaranteed to preserve the
# desired forward orientation after groupby/rolling. Therefore use
# explicit shifted columns instead.

anomaly = df["synthetic_cyber_anomaly"].astype("int8")

future3_matrix = pd.concat(
    [anomaly.groupby(df["telemetry_scenario_id"]).shift(-i)
     for i in range(1, 4)],
    axis=1
)

future6_matrix = pd.concat(
    [anomaly.groupby(df["telemetry_scenario_id"]).shift(-i)
     for i in range(1, 7)],
    axis=1
)

df["future_anomaly_3step"] = (
    future3_matrix.fillna(0).max(axis=1).astype("int8")
)

df["future_anomaly_6step"] = (
    future6_matrix.fillna(0).max(axis=1).astype("int8")
)

# ------------------------------------------------------------------
# 2. REPAIR FIRST-TIMESTEP TEMPORAL DERIVATIVES
# ------------------------------------------------------------------
# At t=0 there is no previous observation, so derivative is undefined.
# For modeling/release purposes, use zero baseline at sequence start.
# ------------------------------------------------------------------

derivative_cols = [
    c for c in df.columns
    if c.startswith("d1_")
]

for c in derivative_cols:
    df[c] = df[c].fillna(0.0)

for c in [
    "resilience_degradation_rate",
    "connectivity_degradation_rate",
    "predictive_warning_score",
]:
    if c in df.columns:
        df[c] = df[c].fillna(0.0)

# ------------------------------------------------------------------
# 3. RECOMPUTE WARNING STATE WHERE SCORE WAS MISSING
# ------------------------------------------------------------------

if "predictive_warning_score" in df.columns:

    score = df["predictive_warning_score"].astype(float)

    # Preserve the existing conceptual state thresholds.
    # These are research-state labels, not clinical/safety certification.
    df["predictive_warning_state"] = np.select(
        [
            score < 0.25,
            score < 0.50,
            score < 0.75,
        ],
        [
            "NORMAL",
            "WATCH",
            "WARNING",
        ],
        default="CRITICAL",
    )

# ------------------------------------------------------------------
# 4. STRUCTURAL VALIDATION
# ------------------------------------------------------------------

checks = []

def check(name, condition, detail):
    checks.append({
        "check": name,
        "passed": bool(condition),
        "detail": str(detail)
    })

n_rows = len(df)
n_sequences = df["telemetry_scenario_id"].nunique()

check(
    "dataset_nonempty",
    n_rows > 0,
    f"rows={n_rows:,}"
)

check(
    "expected_sequence_rows",
    n_rows == n_sequences * 24,
    f"rows={n_rows:,}; sequences={n_sequences:,}; expected={n_sequences*24:,}"
)

check(
    "sequence_length_exact",
    bool((df["sequence_length"] == 24).all()),
    str(df["sequence_length"].value_counts().to_dict())
)

check(
    "timestep_range",
    df["time_step"].min() == 0 and df["time_step"].max() == 23,
    f"{df['time_step'].min()}..{df['time_step'].max()}"
)

check(
    "unique_sequence_timestep",
    df.duplicated(["telemetry_scenario_id", "time_step"]).sum() == 0,
    f"duplicates={df.duplicated(['telemetry_scenario_id','time_step']).sum()}"
)

counts = df.groupby("telemetry_scenario_id")["time_step"].nunique()

check(
    "every_sequence_has_24_steps",
    bool((counts == 24).all()),
    f"invalid_sequences={(counts != 24).sum()}"
)

check(
    "no_missing_sequence_id",
    int(df["telemetry_scenario_id"].isna().sum()) == 0,
    f"missing={df['telemetry_scenario_id'].isna().sum()}"
)

check(
    "no_missing_timestep",
    int(df["time_step"].isna().sum()) == 0,
    f"missing={df['time_step'].isna().sum()}"
)

check(
    "resilience_range",
    bool(
        df["cyber_physical_resilience_score"].between(0, 1).all()
    ),
    (
        f"min={df['cyber_physical_resilience_score'].min():.6f}; "
        f"max={df['cyber_physical_resilience_score'].max():.6f}"
    )
)

check(
    "future_3step_binary",
    set(df["future_anomaly_3step"].unique()).issubset({0, 1}),
    str(sorted(df["future_anomaly_3step"].unique()))
)

check(
    "future_6step_binary",
    set(df["future_anomaly_6step"].unique()).issubset({0, 1}),
    str(sorted(df["future_anomaly_6step"].unique()))
)

check(
    "warning_score_finite",
    bool(np.isfinite(df["predictive_warning_score"]).all()),
    f"nonfinite={np.sum(~np.isfinite(df['predictive_warning_score']))}"
)

valid_states = {"NORMAL", "WATCH", "WARNING", "CRITICAL"}

check(
    "warning_states_valid",
    set(df["predictive_warning_state"].dropna().unique()).issubset(valid_states),
    str(sorted(df["predictive_warning_state"].dropna().unique()))
)

check(
    "no_warning_state_nulls",
    int(df["predictive_warning_state"].isna().sum()) == 0,
    f"nulls={df['predictive_warning_state'].isna().sum()}"
)

# ------------------------------------------------------------------
# 5. TEMPORAL ORDER VALIDATION
# ------------------------------------------------------------------

ordered_ok = True
bad_sequences = 0

for sid, grp in df.groupby("telemetry_scenario_id", sort=False):
    ts = grp["time_step"].to_numpy()

    if not np.array_equal(ts, np.arange(24)):
        ordered_ok = False
        bad_sequences += 1

check(
    "strict_temporal_order",
    ordered_ok,
    f"bad_sequences={bad_sequences}"
)

# ------------------------------------------------------------------
# 6. FUTURE LABEL SANITY
# ------------------------------------------------------------------

# A future label must not be positive at the final timestep unless
# there is actually a future anomaly. With the explicit formulation,
# trailing missing future observations are treated as zero.

last_steps = df[df["time_step"] >= 21]

check(
    "future_labels_trailing_boundary",
    bool(
        (
            last_steps["future_anomaly_3step"].isin([0,1])
            & last_steps["future_anomaly_6step"].isin([0,1])
        ).all()
    ),
    "trailing labels are binary and future-window bounded"
)

# ------------------------------------------------------------------
# 7. SAVE REPAIRED DATASET
# ------------------------------------------------------------------

df.to_parquet(
    OUTPUT,
    index=False,
    engine="pyarrow"
)

# ------------------------------------------------------------------
# 8. QUALITY REPORT
# ------------------------------------------------------------------

passed = sum(x["passed"] for x in checks)
failed = len(checks) - passed

report = {
    "dataset": "CRSS 2024 Temporal Cyber-Physical State Dataset",
    "status": "PASS" if failed == 0 else "FAIL",
    "rows": int(n_rows),
    "columns": int(len(df.columns)),
    "sequences": int(n_sequences),
    "sequence_length": 24,
    "time_step_range": [
        int(df["time_step"].min()),
        int(df["time_step"].max())
    ],
    "future_anomaly_3step_distribution": {
        str(k): int(v)
        for k, v in df["future_anomaly_3step"].value_counts().sort_index().items()
    },
    "future_anomaly_6step_distribution": {
        str(k): int(v)
        for k, v in df["future_anomaly_6step"].value_counts().sort_index().items()
    },
    "warning_state_distribution": {
        str(k): int(v)
        for k, v in df["predictive_warning_state"].value_counts().items()
    },
    "checks": checks,
    "passed": int(passed),
    "failed": int(failed),
    "methodological_notes": [
        "Temporal sequences are grouped by telemetry_scenario_id.",
        "Each sequence contains exactly 24 ordered observations.",
        "Future labels use strictly forward windows and exclude the current timestep.",
        "First-timestep derivatives are initialized to zero because no prior observation exists.",
        "CRSS physical variables are safety-ground-truth context; synthetic cyber telemetry is simulated research data.",
        "Synthetic target-derived fields must not be used as predictive model inputs."
    ]
}

QUALITY.parent.mkdir(parents=True, exist_ok=True)

with open(QUALITY, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

# ------------------------------------------------------------------
# 9. FINAL RELEASE DECISION
# ------------------------------------------------------------------

print()
print("=" * 100)
print("TEMPORAL RELEASE RESULT")
print("=" * 100)
print(f"STATUS       : {report['status']}")
print(f"ROWS         : {n_rows:,}")
print(f"COLUMNS      : {len(df.columns)}")
print(f"SEQUENCES    : {n_sequences:,}")
print(f"CHECKS       : {len(checks)}")
print(f"PASS         : {passed}")
print(f"FAIL         : {failed}")

print()
print("FUTURE 3-STEP LABEL:")
print(df["future_anomaly_3step"].value_counts().sort_index().to_string())

print()
print("FUTURE 6-STEP LABEL:")
print(df["future_anomaly_6step"].value_counts().sort_index().to_string())

print()
print("WARNING STATES:")
print(df["predictive_warning_state"].value_counts().to_string())

print()
for item in checks:
    status = "PASS" if item["passed"] else "FAIL"
    print(f"[{status}] {item['check']} -- {item['detail']}")

print("=" * 100)

if failed:
    raise RuntimeError(
        f"Temporal release gate failed: {failed} check(s). "
        f"See {QUALITY}"
    )

print("TEMPORAL DATASET: RELEASED")
