from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/temporal/crss_2024_temporal_cyber_physical_states.parquet"
OUTPUT = ROOT / "data/processed/modeling/splits/crss_2024_scenario_split.csv"
REPORT = ROOT / "experiments/modeling/crss_2024_split_quality.json"

df = pd.read_parquet(
    INPUT,
    columns=[
        "telemetry_scenario_id",
        "synthetic_cyber_anomaly",
        "future_anomaly_3step",
        "future_anomaly_6step"
    ]
)

scenarios = (
    df.groupby("telemetry_scenario_id")
      .agg(
          rows=("telemetry_scenario_id", "size"),
          anomaly_rate=("synthetic_cyber_anomaly", "mean"),
          future3_rate=("future_anomaly_3step", "mean"),
          future6_rate=("future_anomaly_6step", "mean")
      )
      .reset_index()
)

def deterministic_bucket(value):
    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    return int(digest[:8], 16) / 0xFFFFFFFF

scenarios["bucket"] = (
    scenarios["telemetry_scenario_id"]
    .map(deterministic_bucket)
)

scenarios["split"] = pd.cut(
    scenarios["bucket"],
    bins=[-0.000001, 0.70, 0.85, 1.0],
    labels=["train", "validation", "test"]
).astype(str)

if not scenarios["telemetry_scenario_id"].is_unique:
    raise RuntimeError("Duplicate scenario IDs detected.")

if scenarios["split"].isna().any():
    raise RuntimeError("Unassigned scenarios detected.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

scenarios.to_csv(OUTPUT, index=False)

split_counts = (
    scenarios["split"]
    .value_counts()
    .sort_index()
)

row_counts = (
    scenarios.groupby("split")["rows"]
    .sum()
    .sort_index()
)

report = {
    "status": "PASS",
    "split_unit": "telemetry_scenario_id",
    "method": "deterministic SHA256 hash",
    "scenario_count": int(len(scenarios)),
    "scenario_counts": {
        str(k): int(v)
        for k, v in split_counts.items()
    },
    "row_counts": {
        str(k): int(v)
        for k, v in row_counts.items()
    },
    "ratios": {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15
    },
    "scenario_overlap": False
}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print()
print("=" * 100)
print("CRSS 2024 SCENARIO-LEVEL SPLIT")
print("=" * 100)

print("TOTAL SCENARIOS:", f"{len(scenarios):,}")

print()
print("SCENARIO COUNTS:")
print(split_counts.to_string())

print()
print("ROW COUNTS:")
print(row_counts.to_string())

print()
print("SPLIT METHOD: deterministic SHA256 scenario assignment")
print("SCENARIO OVERLAP: FALSE")

print()
print("OUTPUT:")
print(OUTPUT)

print()
print("QUALITY REPORT:")
print(REPORT)

print()
print("SCENARIO SPLIT: PASS")
print("=" * 100)
