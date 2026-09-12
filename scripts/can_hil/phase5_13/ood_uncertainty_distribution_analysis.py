from pathlib import Path
import json
import math
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# PHASE 5.13
# OOD / UNCERTAINTY DISTRIBUTION & SCENARIO ANALYSIS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_12"
    / "phase5_12_uncertainty_ood.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_13"
)

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "can_hil"
    / "phase5_13"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_CSV = OUTPUT_DIR / "phase5_13_scenario_summary.csv"
EXTREME_OOD_CSV = OUTPUT_DIR / "phase5_13_extreme_ood_windows.csv"
EXTREME_UNCERTAINTY_CSV = OUTPUT_DIR / "phase5_13_extreme_uncertainty_windows.csv"
SUMMARY_JSON = OUTPUT_DIR / "phase5_13_distribution_summary.json"
MANIFEST_JSON = EXPERIMENT_DIR / "phase5_13_ood_uncertainty_manifest.json"


# ------------------------------------------------------------
# PERMANENT JSON SERIALIZATION BOUNDARY
# ------------------------------------------------------------

def json_safe(value):
    """
    Recursively convert NumPy/Pandas values into native
    Python JSON-compatible values.
    """

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, np.ndarray):
        return [
            json_safe(v)
            for v in value.tolist()
        ]

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite floating-point value encountered: {value}"
            )

        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite floating-point value encountered: {value}"
            )

        return value

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def write_json_verified(path, payload):
    """
    Permanent JSON write + read-back verification.

    A phase cannot claim PASS unless the generated JSON
    can be parsed successfully after writing.
    """

    safe_payload = json_safe(payload)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            safe_payload,
            f,
            indent=2,
            allow_nan=False
        )

    # Mandatory read-back validation.
    with open(temp_path, "r", encoding="utf-8") as f:
        json.load(f)

    temp_path.replace(path)

    # Final verification of the actual destination.
    with open(path, "r", encoding="utf-8") as f:
        json.load(f)


# ------------------------------------------------------------
# COLUMN DETECTION
# ------------------------------------------------------------

def find_column(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


# ------------------------------------------------------------
# LOAD INPUT
# ------------------------------------------------------------

print()
print("=== PHASE 5.13 ===")
print("OOD / UNCERTAINTY DISTRIBUTION & SCENARIO ANALYSIS")
print()

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

if df.empty:
    raise ValueError("Input CSV contains zero rows.")

scenario_col = find_column(
    df,
    ["scenario_id", "scenario", "scenario_name"]
)

uncertainty_col = find_column(
    df,
    ["uncertainty_score", "uncertainty"]
)

ood_col = find_column(
    df,
    ["ood_score", "ood"]
)

ood_flag_col = find_column(
    df,
    ["ood_flag", "is_ood"]
)

y3_col = find_column(
    df,
    ["y3_probability", "y3_prob"]
)

y6_col = find_column(
    df,
    ["y6_probability", "y6_prob"]
)

required = {
    "scenario": scenario_col,
    "uncertainty": uncertainty_col,
    "ood_score": ood_col,
    "ood_flag": ood_flag_col,
    "y3": y3_col,
    "y6": y6_col,
}

missing = [
    name
    for name, column in required.items()
    if column is None
]

if missing:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
    )

print(f"Input rows: {len(df):,}")
print(f"Input columns: {len(df.columns)}")

print()
print("Detected columns:")
for name, column in required.items():
    print(f"  {name}: {column}")


# ------------------------------------------------------------
# NUMERIC NORMALIZATION
# ------------------------------------------------------------

df[uncertainty_col] = pd.to_numeric(
    df[uncertainty_col],
    errors="coerce"
)

df[ood_col] = pd.to_numeric(
    df[ood_col],
    errors="coerce"
)

df[y3_col] = pd.to_numeric(
    df[y3_col],
    errors="coerce"
)

df[y6_col] = pd.to_numeric(
    df[y6_col],
    errors="coerce"
)

if not np.isfinite(df[uncertainty_col].to_numpy()).all():
    raise ValueError("Non-finite uncertainty values detected.")

if not np.isfinite(df[ood_col].to_numpy()).all():
    raise ValueError("Non-finite OOD values detected.")

if not np.isfinite(df[y3_col].to_numpy()).all():
    raise ValueError("Non-finite Y3 values detected.")

if not np.isfinite(df[y6_col].to_numpy()).all():
    raise ValueError("Non-finite Y6 values detected.")


# ------------------------------------------------------------
# GLOBAL DISTRIBUTION
# ------------------------------------------------------------

def distribution(series):
    values = series.to_numpy(dtype=float)

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


global_summary = {
    "rows": int(len(df)),
    "scenarios": int(df[scenario_col].nunique()),
    "uncertainty": distribution(df[uncertainty_col]),
    "ood_score": distribution(df[ood_col]),
    "y3_probability": distribution(df[y3_col]),
    "y6_probability": distribution(df[y6_col]),
}


# ------------------------------------------------------------
# SCENARIO SUMMARY
# ------------------------------------------------------------

scenario_rows = []

for scenario, group in df.groupby(
    scenario_col,
    sort=True
):
    scenario_rows.append(
        {
            "scenario_id": str(scenario),
            "rows": int(len(group)),

            "uncertainty_mean": float(
                group[uncertainty_col].mean()
            ),
            "uncertainty_median": float(
                group[uncertainty_col].median()
            ),
            "uncertainty_p95": float(
                group[uncertainty_col].quantile(0.95)
            ),
            "uncertainty_p99": float(
                group[uncertainty_col].quantile(0.99)
            ),
            "uncertainty_max": float(
                group[uncertainty_col].max()
            ),

            "ood_mean": float(
                group[ood_col].mean()
            ),
            "ood_median": float(
                group[ood_col].median()
            ),
            "ood_p95": float(
                group[ood_col].quantile(0.95)
            ),
            "ood_p99": float(
                group[ood_col].quantile(0.99)
            ),
            "ood_max": float(
                group[ood_col].max()
            ),

            "ood_rate": float(
                group[ood_flag_col].astype(bool).mean()
            ),
        }
    )

scenario_summary = pd.DataFrame(scenario_rows)

scenario_summary.to_csv(
    SCENARIO_CSV,
    index=False
)


# ------------------------------------------------------------
# EXTREME WINDOWS
# ------------------------------------------------------------

extreme_ood = (
    df.sort_values(
        ood_col,
        ascending=False
    )
    .head(25)
    .copy()
)

extreme_uncertainty = (
    df.sort_values(
        uncertainty_col,
        ascending=False
    )
    .head(25)
    .copy()
)

extreme_ood.to_csv(
    EXTREME_OOD_CSV,
    index=False
)

extreme_uncertainty.to_csv(
    EXTREME_UNCERTAINTY_CSV,
    index=False
)


# ------------------------------------------------------------
# BASELINE VS ATTACK
# ------------------------------------------------------------

baseline_mask = (
    df[scenario_col]
    .astype(str)
    .str.startswith("S00")
)

baseline = df.loc[baseline_mask]
attack = df.loc[~baseline_mask]

baseline_ood_rate = float(
    baseline[ood_flag_col].astype(bool).mean()
) if len(baseline) else None

attack_ood_rate = float(
    attack[ood_flag_col].astype(bool).mean()
) if len(attack) else None

rate_ratio = None

if (
    baseline_ood_rate is not None
    and baseline_ood_rate > 0
    and attack_ood_rate is not None
):
    rate_ratio = float(
        attack_ood_rate / baseline_ood_rate
    )

baseline_attack = {
    "baseline_rows": int(len(baseline)),
    "attack_rows": int(len(attack)),

    "baseline_uncertainty_mean": (
        float(baseline[uncertainty_col].mean())
        if len(baseline)
        else None
    ),

    "baseline_uncertainty_std": (
        float(baseline[uncertainty_col].std())
        if len(baseline)
        else None
    ),

    "baseline_ood_mean": (
        float(baseline[ood_col].mean())
        if len(baseline)
        else None
    ),

    "baseline_ood_median": (
        float(baseline[ood_col].median())
        if len(baseline)
        else None
    ),

    "baseline_ood_p95": (
        float(baseline[ood_col].quantile(0.95))
        if len(baseline)
        else None
    ),

    "baseline_ood_p99": (
        float(baseline[ood_col].quantile(0.99))
        if len(baseline)
        else None
    ),

    "baseline_ood_max": (
        float(baseline[ood_col].max())
        if len(baseline)
        else None
    ),

    "baseline_ood_rate": baseline_ood_rate,

    "attack_uncertainty_mean": (
        float(attack[uncertainty_col].mean())
        if len(attack)
        else None
    ),

    "attack_uncertainty_std": (
        float(attack[uncertainty_col].std())
        if len(attack)
        else None
    ),

    "attack_ood_mean": (
        float(attack[ood_col].mean())
        if len(attack)
        else None
    ),

    "attack_ood_median": (
        float(attack[ood_col].median())
        if len(attack)
        else None
    ),

    "attack_ood_p95": (
        float(attack[ood_col].quantile(0.95))
        if len(attack)
        else None
    ),

    "attack_ood_p99": (
        float(attack[ood_col].quantile(0.99))
        if len(attack)
        else None
    ),

    "attack_ood_max": (
        float(attack[ood_col].max())
        if len(attack)
        else None
    ),

    "attack_ood_rate": attack_ood_rate,

    "attack_to_baseline_ood_rate_ratio": rate_ratio,
}


# ------------------------------------------------------------
# EXTREME-VALUE DIAGNOSTIC
# ------------------------------------------------------------

ood_values = df[ood_col].to_numpy(dtype=float)

ood_max = float(np.max(ood_values))
ood_p99 = float(np.percentile(ood_values, 99))
ood_p999 = float(np.percentile(ood_values, 99.9))
ood_median = float(np.median(ood_values))

extreme_diagnostic = {
    "ood_score_max": ood_max,
    "ood_score_p99": ood_p99,
    "ood_score_p999": ood_p999,

    "max_to_p99_ratio": (
        float(ood_max / ood_p99)
        if ood_p99 != 0
        else None
    ),

    "max_to_median_ratio": (
        float(ood_max / ood_median)
        if ood_median != 0
        else None
    ),
}


# ------------------------------------------------------------
# SUMMARY JSON
# ------------------------------------------------------------

summary_payload = {
    **global_summary,
    "baseline_vs_attack": baseline_attack,
    "extreme_diagnostic": extreme_diagnostic,
    "scenario_summary": scenario_summary.to_dict(
        orient="records"
    ),
}

write_json_verified(
    SUMMARY_JSON,
    summary_payload
)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

checks = {}

checks["PASS input_exists"] = INPUT_CSV.exists()
checks["PASS input_rows_positive"] = len(df) > 0
checks["PASS required_columns_detected"] = len(missing) == 0
checks["PASS uncertainty_finite"] = bool(
    np.isfinite(
        df[uncertainty_col].to_numpy()
    ).all()
)
checks["PASS ood_finite"] = bool(
    np.isfinite(
        df[ood_col].to_numpy()
    ).all()
)
checks["PASS y3_finite"] = bool(
    np.isfinite(
        df[y3_col].to_numpy()
    ).all()
)
checks["PASS y6_finite"] = bool(
    np.isfinite(
        df[y6_col].to_numpy()
    ).all()
)
checks["PASS scenarios_present"] = (
    df[scenario_col].nunique() > 0
)
checks["PASS scenario_csv_exists"] = (
    SCENARIO_CSV.exists()
)
checks["PASS extreme_ood_csv_exists"] = (
    EXTREME_OOD_CSV.exists()
)
checks["PASS extreme_uncertainty_csv_exists"] = (
    EXTREME_UNCERTAINTY_CSV.exists()
)
checks["PASS summary_json_exists"] = (
    SUMMARY_JSON.exists()
)


# ------------------------------------------------------------
# MANIFEST
# ------------------------------------------------------------

all_pass = bool(all(checks.values()))

manifest = {
    "phase": "5.13",
    "status": "PASS" if all_pass else "FAIL",

    "input_csv": str(INPUT_CSV),
    "input_rows": int(len(df)),
    "scenario_count": int(
        df[scenario_col].nunique()
    ),

    "columns": {
        "scenario": str(scenario_col),
        "uncertainty": str(uncertainty_col),
        "ood_score": str(ood_col),
        "ood_flag": str(ood_flag_col),
        "y3": str(y3_col),
        "y6": str(y6_col),
    },

    "checks": checks,

    "outputs": [
        str(SCENARIO_CSV),
        str(EXTREME_OOD_CSV),
        str(EXTREME_UNCERTAINTY_CSV),
        str(SUMMARY_JSON),
    ],
}

# The manifest itself must pass the same JSON boundary.
write_json_verified(
    MANIFEST_JSON,
    manifest
)

# Confirm final manifest status from the actual written JSON.
with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
    verified_manifest = json.load(f)

if verified_manifest.get("status") != (
    "PASS" if all_pass else "FAIL"
):
    raise RuntimeError(
        "Manifest read-back status does not match computed status."
    )


# ------------------------------------------------------------
# CONSOLE REPORT
# ------------------------------------------------------------

print()
print("=== PHASE 5.13 RESULTS ===")

print()
print("Global OOD distribution:")

for key, value in global_summary["ood_score"].items():
    print(
        f"  {key}: {value:.9f}"
        if isinstance(value, float)
        else f"  {key}: {value}"
    )

print()
print("Global uncertainty distribution:")

for key, value in global_summary["uncertainty"].items():
    print(
        f"  {key}: {value:.9f}"
        if isinstance(value, float)
        else f"  {key}: {value}"
    )

print()
print("Scenario summary:")

for _, row in scenario_summary.iterrows():
    print(
        f"  {row['scenario_id']}: "
        f"rows={int(row['rows'])}, "
        f"OOD_mean={row['ood_mean']:.6f}, "
        f"OOD_p99={row['ood_p99']:.6f}, "
        f"OOD_max={row['ood_max']:.6f}, "
        f"uncertainty_mean={row['uncertainty_mean']:.6f}, "
        f"uncertainty_max={row['uncertainty_max']:.6f}"
    )

print()
print("Extreme OOD diagnostic:")
print(
    f"  Maximum OOD:       "
    f"{extreme_diagnostic['ood_score_max']:.9f}"
)
print(
    f"  P99 OOD:           "
    f"{extreme_diagnostic['ood_score_p99']:.9f}"
)
print(
    f"  P99.9 OOD:         "
    f"{extreme_diagnostic['ood_score_p999']:.9f}"
)
print(
    f"  Max/P99 ratio:     "
    f"{extreme_diagnostic['max_to_p99_ratio']:.3f}"
)
print(
    f"  Max/median ratio:  "
    f"{extreme_diagnostic['max_to_median_ratio']:.3f}"
)

print()
print("Validation checks:")

for name, result in checks.items():
    print(
        f"  {'PASS' if bool(result) else 'FAIL'} "
        f"{name.replace('PASS ', '').replace('FAIL ', '')}"
    )

print()
print(f"Scenario CSV: {SCENARIO_CSV}")
print(f"Extreme OOD CSV: {EXTREME_OOD_CSV}")
print(
    f"Extreme uncertainty CSV: "
    f"{EXTREME_UNCERTAINTY_CSV}"
)
print(f"Summary JSON: {SUMMARY_JSON}")
print(f"Manifest: {MANIFEST_JSON}")

print()
print(
    f"STATUS: "
    f"{'PASS' if all_pass else 'FAIL'}"
)
