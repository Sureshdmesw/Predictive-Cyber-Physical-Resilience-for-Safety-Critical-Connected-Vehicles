from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TEMPORAL = ROOT / "data/processed/temporal/crss_2024_temporal_cyber_physical_states.parquet"
SPLIT = ROOT / "data/processed/modeling/splits/crss_2024_scenario_split.csv"

REPORT = ROOT / "experiments/modeling/crss_2024_phase45_distribution_quality.json"

print("=" * 100)
print("CRSS 2024 PHASE 4.5 — MODELING DISTRIBUTION / IMBALANCE AUDIT")
print("=" * 100)

df = pd.read_parquet(
    TEMPORAL,
    columns=[
        "telemetry_scenario_id",
        "time_step",
        "synthetic_cyber_anomaly",
        "future_anomaly_3step",
        "future_anomaly_6step"
    ]
)

split = pd.read_csv(SPLIT)

df = df.merge(
    split[["telemetry_scenario_id", "split"]],
    on="telemetry_scenario_id",
    how="left",
    validate="many_to_one"
)

checks = []

def check(name, condition, detail):
    checks.append({
        "name": name,
        "passed": bool(condition),
        "detail": str(detail)
    })

# ------------------------------------------------------------
# BASIC INTEGRITY
# ------------------------------------------------------------

check(
    "all_rows_have_split",
    df["split"].notna().all(),
    f"missing={df['split'].isna().sum()}"
)

check(
    "all_scenarios_have_24_steps",
    bool(
        df.groupby("telemetry_scenario_id")["time_step"]
        .nunique()
        .eq(24)
        .all()
    ),
    "24 timesteps required"
)

check(
    "no_scenario_overlap",
    split["telemetry_scenario_id"].is_unique,
    "scenario split manifest has unique scenario IDs"
)

# ------------------------------------------------------------
# ROW-LEVEL TARGET DISTRIBUTION
# ------------------------------------------------------------

target_stats = {}

for target in [
    "synthetic_cyber_anomaly",
    "future_anomaly_3step",
    "future_anomaly_6step"
]:

    vc = df[target].value_counts().sort_index()

    total = len(df)
    positives = int(vc.get(1, 0))

    target_stats[target] = {
        "total_rows": int(total),
        "negative_rows": int(vc.get(0, 0)),
        "positive_rows": positives,
        "positive_rate": float(positives / total)
    }

# ------------------------------------------------------------
# SCENARIO-LEVEL POSITIVE DISTRIBUTION
# ------------------------------------------------------------

scenario_targets = (
    df.groupby(["telemetry_scenario_id", "split"])
      .agg(
          anomaly=("synthetic_cyber_anomaly", "max"),
          future3=("future_anomaly_3step", "max"),
          future6=("future_anomaly_6step", "max")
      )
      .reset_index()
)

scenario_stats = {}

for split_name in ["train", "validation", "test"]:

    part = scenario_targets[
        scenario_targets["split"] == split_name
    ]

    scenario_stats[split_name] = {
        "scenarios": int(len(part)),
        "anomaly_positive": int(part["anomaly"].sum()),
        "future3_positive": int(part["future3"].sum()),
        "future6_positive": int(part["future6"].sum()),
        "anomaly_rate": float(part["anomaly"].mean()),
        "future3_rate": float(part["future3"].mean()),
        "future6_rate": float(part["future6"].mean())
    }

# ------------------------------------------------------------
# TRAIN / VALIDATION / TEST DISTRIBUTION
# ------------------------------------------------------------

split_stats = {}

for split_name in ["train", "validation", "test"]:

    part = df[df["split"] == split_name]

    split_stats[split_name] = {
        "rows": int(len(part)),
        "scenarios": int(
            part["telemetry_scenario_id"].nunique()
        ),
        "future3_positive_rows": int(
            part["future_anomaly_3step"].sum()
        ),
        "future6_positive_rows": int(
            part["future_anomaly_6step"].sum()
        ),
        "future3_positive_rate": float(
            part["future_anomaly_3step"].mean()
        ),
        "future6_positive_rate": float(
            part["future_anomaly_6step"].mean()
        )
    }

# ------------------------------------------------------------
# HORIZON OVERLAP
# ------------------------------------------------------------

future3 = df["future_anomaly_3step"].astype(bool)
future6 = df["future_anomaly_6step"].astype(bool)

both = future3 & future6
six_only = future6 & ~future3
three_only = future3 & ~future6

horizon_overlap = {
    "both_positive": int(both.sum()),
    "three_only": int(three_only.sum()),
    "six_only": int(six_only.sum()),
    "overlap_rate_among_6step_positive": float(
        both.sum() / max(future6.sum(), 1)
    )
}

check(
    "future_horizons_not_identical",
    not np.array_equal(
        df["future_anomaly_3step"].values,
        df["future_anomaly_6step"].values
    ),
    f"3step_vs_6step different={not np.array_equal(df['future_anomaly_3step'].values, df['future_anomaly_6step'].values)}"
)

# ------------------------------------------------------------
# CLASS IMBALANCE
# ------------------------------------------------------------

imbalance = {}

for target in [
    "future_anomaly_3step",
    "future_anomaly_6step"
]:

    positive = int(df[target].sum())
    negative = int((df[target] == 0).sum())

    imbalance[target] = {
        "positive": positive,
        "negative": negative,
        "negative_to_positive_ratio": float(
            negative / max(positive, 1)
        )
    }

# ------------------------------------------------------------
# FINAL REPORT
# ------------------------------------------------------------

passed = sum(x["passed"] for x in checks)
failed = len(checks) - passed

report = {
    "status": "PASS" if failed == 0 else "FAIL",
    "checks": checks,
    "target_distribution": target_stats,
    "scenario_distribution": scenario_stats,
    "split_distribution": split_stats,
    "horizon_overlap": horizon_overlap,
    "class_imbalance": imbalance,
    "methodological_notes": [
        "Scenario-level split is preserved.",
        "Future targets are evaluated separately for 3-step and 6-step horizons.",
        "Class imbalance must be handled during model training.",
        "Test scenarios remain untouched during model development."
    ]
}

REPORT.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print()
print("TARGET DISTRIBUTION")
print("-" * 100)

for k, v in target_stats.items():
    print(
        f"{k:28s} "
        f"positive={v['positive_rows']:,} "
        f"rate={v['positive_rate']:.6%}"
    )

print()
print("SCENARIO DISTRIBUTION")
print("-" * 100)

for k, v in scenario_stats.items():
    print(
        f"{k:12s} "
        f"scenarios={v['scenarios']:,} "
        f"future3={v['future3_positive']:,} "
        f"future6={v['future6_positive']:,}"
    )

print()
print("TRAIN / VALIDATION / TEST")
print("-" * 100)

for k, v in split_stats.items():
    print(
        f"{k:12s} "
        f"rows={v['rows']:,} "
        f"future3={v['future3_positive_rate']:.6%} "
        f"future6={v['future6_positive_rate']:.6%}"
    )

print()
print("HORIZON OVERLAP")
print("-" * 100)

for k, v in horizon_overlap.items():
    print(f"{k:45s}: {v}")

print()
print("CLASS IMBALANCE")
print("-" * 100)

for k, v in imbalance.items():
    print(
        f"{k:28s} "
        f"ratio={v['negative_to_positive_ratio']:.2f}:1"
    )

print()
print("QUALITY CHECKS")
print("-" * 100)

for item in checks:
    print(
        f"[{'PASS' if item['passed'] else 'FAIL'}] "
        f"{item['name']} -- {item['detail']}"
    )

print()
print("=" * 100)
print(f"STATUS: {'PASS' if failed == 0 else 'FAIL'}")
print(f"CHECKS: {len(checks)}")
print(f"PASS  : {passed}")
print(f"FAIL  : {failed}")
print("=" * 100)

if failed:
    raise RuntimeError(
        f"Phase 4.5 quality gate failed: {failed} failure(s)."
    )

print("PHASE 4.5: PASS")
