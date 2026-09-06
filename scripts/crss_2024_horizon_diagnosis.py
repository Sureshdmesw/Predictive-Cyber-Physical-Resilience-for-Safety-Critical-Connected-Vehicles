from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data/processed/temporal/crss_2024_temporal_cyber_physical_states.parquet"
OUT = ROOT / "experiments/modeling/crss_2024_horizon_diagnosis.json"

print("=" * 100)
print("CRSS 2024 — FUTURE HORIZON DIAGNOSIS")
print("=" * 100)

df = pd.read_parquet(
    DATA,
    columns=[
        "telemetry_scenario_id",
        "time_step",
        "synthetic_cyber_anomaly",
        "future_anomaly_3step",
        "future_anomaly_6step"
    ]
)

df = df.sort_values(
    ["telemetry_scenario_id", "time_step"]
).reset_index(drop=True)

results = {}

# ------------------------------------------------------------
# BASIC COMPARISON
# ------------------------------------------------------------

same = (
    df["future_anomaly_3step"].values ==
    df["future_anomaly_6step"].values
)

results["row_level"] = {
    "rows": int(len(df)),
    "identical": bool(same.all()),
    "different_rows": int((~same).sum()),
    "future3_positive": int(df["future_anomaly_3step"].sum()),
    "future6_positive": int(df["future_anomaly_6step"].sum())
}

# ------------------------------------------------------------
# TRUE HORIZON LABELS FROM RAW ANOMALY
# ------------------------------------------------------------

g = df.groupby("telemetry_scenario_id", sort=False)

anomaly = df["synthetic_cyber_anomaly"].astype(int)

future3_true = (
    g["synthetic_cyber_anomaly"]
    .shift(-1)
    .fillna(0)
    .astype(int)
)

s1 = future3_true

s2 = (
    g["synthetic_cyber_anomaly"]
    .shift(-2)
    .fillna(0)
    .astype(int)
)

s3 = (
    g["synthetic_cyber_anomaly"]
    .shift(-3)
    .fillna(0)
    .astype(int)
)

s4 = (
    g["synthetic_cyber_anomaly"]
    .shift(-4)
    .fillna(0)
    .astype(int)
)

s5 = (
    g["synthetic_cyber_anomaly"]
    .shift(-5)
    .fillna(0)
    .astype(int)
)

s6 = (
    g["synthetic_cyber_anomaly"]
    .shift(-6)
    .fillna(0)
    .astype(int)
)

true3 = ((s1 + s2 + s3) > 0).astype(int)
true6 = ((s1 + s2 + s3 + s4 + s5 + s6) > 0).astype(int)

results["reconstructed"] = {
    "future3_positive": int(true3.sum()),
    "future6_positive": int(true6.sum()),
    "different_rows": int((true3 != true6).sum()),
    "future6_only_rows": int(((true6 == 1) & (true3 == 0)).sum()),
    "future3_only_rows": int(((true3 == 1) & (true6 == 0)).sum())
}

# ------------------------------------------------------------
# STEP-SPECIFIC POSITIVES
# ------------------------------------------------------------

step_stats = {}

for i, s in enumerate([s1, s2, s3, s4, s5, s6], start=1):
    step_stats[f"t_plus_{i}"] = {
        "positive_rows": int(s.sum()),
        "positive_rate": float(s.mean())
    }

results["individual_future_steps"] = step_stats

# ------------------------------------------------------------
# SCENARIO-LEVEL EPISODE STRUCTURE
# ------------------------------------------------------------

scenario_rows = []

for sid, part in df.groupby("telemetry_scenario_id", sort=False):

    arr = part["synthetic_cyber_anomaly"].to_numpy()

    positions = np.where(arr == 1)[0]

    if len(positions) == 0:
        continue

    first = int(positions[0])
    last = int(positions[-1])

    scenario_rows.append({
        "scenario": sid,
        "first_anomaly_step": first,
        "last_anomaly_step": last,
        "anomaly_duration_steps": last - first + 1,
        "anomaly_count": int(len(positions))
    })

episodes = pd.DataFrame(scenario_rows)

if len(episodes):
    results["anomaly_episode_structure"] = {
        "anomalous_scenarios": int(len(episodes)),
        "mean_duration_steps": float(
            episodes["anomaly_duration_steps"].mean()
        ),
        "median_duration_steps": float(
            episodes["anomaly_duration_steps"].median()
        ),
        "max_duration_steps": int(
            episodes["anomaly_duration_steps"].max()
        ),
        "duration_distribution": {
            str(k): int(v)
            for k, v in episodes[
                "anomaly_duration_steps"
            ].value_counts().sort_index().items()
        }
    }

# ------------------------------------------------------------
# FINAL DIAGNOSIS
# ------------------------------------------------------------

if results["reconstructed"]["future6_only_rows"] > 0:
    diagnosis = (
        "VALID_HORIZON_SEPARATION: "
        "raw anomaly sequence supports additional t+4..t+6 positives."
    )
elif results["reconstructed"]["future6_positive"] == results["reconstructed"]["future3_positive"]:
    diagnosis = (
        "HORIZON_COLLAPSE: "
        "the synthetic anomaly episodes do not create additional "
        "6-step-only future positives."
    )
else:
    diagnosis = (
        "REVIEW_REQUIRED: "
        "horizon labels require further inspection."
    )

results["diagnosis"] = diagnosis

OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print()
print("ROW-LEVEL")
print("-" * 100)

for k, v in results["row_level"].items():
    print(f"{k:30s}: {v}")

print()
print("RECONSTRUCTED HORIZONS")
print("-" * 100)

for k, v in results["reconstructed"].items():
    print(f"{k:30s}: {v}")

print()
print("INDIVIDUAL FUTURE STEPS")
print("-" * 100)

for k, v in step_stats.items():
    print(
        f"{k:15s}: "
        f"positive={v['positive_rows']:,} "
        f"rate={v['positive_rate']:.6%}"
    )

if "anomaly_episode_structure" in results:

    print()
    print("ANOMALY EPISODE STRUCTURE")
    print("-" * 100)

    for k, v in results["anomaly_episode_structure"].items():
        print(f"{k:30s}: {v}")

print()
print("DIAGNOSIS")
print("-" * 100)
print(results["diagnosis"])

print()
print(f"REPORT: {OUT}")
print("=" * 100)
