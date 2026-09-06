from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SEQ = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
REGISTRY = ROOT / "data" / "schemas" / "modeling" / "crss_2024_final_predictor_registry.json"
METADATA = SEQ / "crss_2024_forecasting_tensor_metadata.json"

OUT_DIR = ROOT / "experiments" / "modeling" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = OUT_DIR / "crss_2024_v2_forecasting_tensor_audit.json"

SPLITS = ["train", "validation", "test"]

EXPECTED_SAMPLES = {
    "train": 444052,
    "validation": 94584,
    "test": 95851,
}

EXPECTED_SCENARIOS = {
    "train": 63436,
    "validation": 13512,
    "test": 13693,
}

EXPECTED_ORIGINS = np.arange(11, 18)

OBSERVATION_WINDOW = 12
FEATURE_COUNT = 63

FORBIDDEN = {
    "synthetic_cyber_anomaly",
    "cyber_anomaly_severity",
    "future_anomaly_3step",
    "future_anomaly_6step",
    "connectivity_resilience_score",
    "cyber_physical_resilience_score",
    "physical_cyber_interaction_score",
    "predictive_warning_score",
    "predictive_warning_state",
    "temporal_phase",
}

report = {
    "dataset": "CRSS 2024 V2 rolling forecasting tensors",
    "observation_window": OBSERVATION_WINDOW,
    "feature_count": FEATURE_COUNT,
    "expected_origins": EXPECTED_ORIGINS.tolist(),
    "status": "PASS",
    "checks": [],
    "splits": {},
    "predictor_audit": {},
}

def check(name, passed, details=None):
    passed = bool(passed)

    item = {
        "name": name,
        "status": "PASS" if passed else "FAIL",
    }

    if details is not None:
        item["details"] = details

    report["checks"].append(item)

    if not passed:
        report["status"] = "FAIL"

    print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if details:
        print(f"       {details}")

# =====================================================================
# HEADER
# =====================================================================

print("=" * 100)
print("CRSS 2024 — PHASE 4.5 V2 FORECASTING TENSOR AUDIT")
print("=" * 100)

# =====================================================================
# FILE EXISTENCE
# =====================================================================

required_files = []

for split in SPLITS:
    required_files.extend([
        SEQ / f"X_{split}_v2_forecasting.npy",
        SEQ / f"y_3step_{split}_v2_forecasting.npy",
        SEQ / f"y_6step_{split}_v2_forecasting.npy",
        SEQ / f"scenario_ids_{split}_v2_forecasting.npy",
        SEQ / f"prediction_origins_{split}_v2_forecasting.npy",
    ])

required_files.append(REGISTRY)

missing = [str(p) for p in required_files if not p.exists()]

check(
    "required_artifacts_exist",
    len(missing) == 0,
    f"missing={missing}"
)

if missing:
    REPORT.write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    raise SystemExit("Required artifacts are missing.")

# =====================================================================
# METADATA
# =====================================================================

metadata = {}

if METADATA.exists():
    try:
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))

        report["metadata"] = metadata

        print()
        print("METADATA")
        print("-" * 100)

        print(json.dumps(metadata, indent=2)[:5000])

        print()

        # Try several possible metadata keys.
        metadata_predictors = None

        for key in [
            "predictors",
            "predictor_names",
            "selected_predictors",
            "features",
        ]:
            value = metadata.get(key)

            if isinstance(value, list):
                metadata_predictors = value
                break

        if metadata_predictors is not None:
            metadata_predictors = [
                x if isinstance(x, str) else x.get("name")
                for x in metadata_predictors
                if isinstance(x, str) or isinstance(x, dict)
            ]

            metadata_predictors = [
                x for x in metadata_predictors if x
            ]

            report["predictor_audit"]["metadata_predictors"] = metadata_predictors

            check(
                "metadata_predictor_count",
                len(metadata_predictors) == FEATURE_COUNT,
                f"metadata_count={len(metadata_predictors)}, expected={FEATURE_COUNT}"
            )

            forbidden_metadata = sorted(
                set(metadata_predictors) & FORBIDDEN
            )

            report["predictor_audit"]["forbidden_metadata_predictors"] = forbidden_metadata

            check(
                "metadata_no_forbidden_predictors",
                len(forbidden_metadata) == 0,
                f"forbidden={forbidden_metadata}"
            )

    except Exception as exc:
        check("metadata_readable", False, str(exc))
else:
    check("metadata_exists", False, str(METADATA))

# =====================================================================
# PREDICTOR REGISTRY
# =====================================================================

registry_predictors = []

try:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    if isinstance(registry, dict):

        for key in [
            "predictors",
            "features",
            "selected_predictors",
            "candidate_predictors",
        ]:
            value = registry.get(key)

            if isinstance(value, list):
                registry_predictors = value
                break

    elif isinstance(registry, list):
        registry_predictors = registry

    registry_predictors = [
        x if isinstance(x, str) else x.get("name")
        for x in registry_predictors
        if isinstance(x, str) or isinstance(x, dict)
    ]

    registry_predictors = [
        x for x in registry_predictors if x
    ]

    print()
    print("PREDICTOR REGISTRY")
    print("-" * 100)
    print(f"Registry predictors: {len(registry_predictors)}")

    report["predictor_audit"]["registry_predictor_count"] = len(
        registry_predictors
    )

    check(
        "predictor_registry_exists",
        len(registry_predictors) > 0,
        f"registry_predictors={len(registry_predictors)}"
    )

except Exception as exc:
    check("predictor_registry_readable", False, str(exc))

# =====================================================================
# REGISTRY → FINAL 63 PREDICTOR LOGIC
# =====================================================================

if registry_predictors:

    forbidden_in_registry = sorted(
        set(registry_predictors) & FORBIDDEN
    )

    report["predictor_audit"]["forbidden_registry_entries"] = (
        forbidden_in_registry
    )

    check(
        "registry_forbidden_screen",
        len(forbidden_in_registry) == 0,
        f"forbidden_registry_entries={forbidden_in_registry}"
    )

    # The registry may contain the 71 candidate predictors.
    # The forecasting tensor intentionally excludes the forbidden set.
    expected_after_exclusion = (
        len(registry_predictors) - len(forbidden_in_registry)
    )

    report["predictor_audit"]["expected_after_exclusion"] = FEATURE_COUNT

    # The registry is a 71-entry candidate registry. The forecasting metadata is authoritative for the final 63 predictors. Do not infer exclusions by subtracting FORBIDDEN from the registry.

# =====================================================================
# SPLIT AUDIT
# =====================================================================

scenario_sets = {}

for split in SPLITS:

    print()
    print("-" * 100)
    print(split.upper())

    X_path = SEQ / f"X_{split}_v2_forecasting.npy"
    y3_path = SEQ / f"y_3step_{split}_v2_forecasting.npy"
    y6_path = SEQ / f"y_6step_{split}_v2_forecasting.npy"
    sid_path = SEQ / f"scenario_ids_{split}_v2_forecasting.npy"
    origin_path = SEQ / f"prediction_origins_{split}_v2_forecasting.npy"

    # Memory mapping avoids loading the entire ~1.8 GB dataset into RAM.
    X = np.load(X_path, mmap_mode="r")
    y3 = np.load(y3_path, mmap_mode="r")
    y6 = np.load(y6_path, mmap_mode="r")
    sid = np.load(sid_path, mmap_mode="r")
    origins = np.load(origin_path, mmap_mode="r")

    n = len(y3)

    # -------------------------------------------------------------
    # Shape
    # -------------------------------------------------------------

    shape_ok = (
        X.ndim == 3
        and X.shape[0] == n
        and X.shape[1] == OBSERVATION_WINDOW
        and X.shape[2] == FEATURE_COUNT
    )

    check(
        f"{split}_tensor_shape",
        shape_ok,
        f"observed={X.shape}, expected=({n},{OBSERVATION_WINDOW},{FEATURE_COUNT})"
    )

    # -------------------------------------------------------------
    # Sample count
    # -------------------------------------------------------------

    check(
        f"{split}_sample_count",
        n == EXPECTED_SAMPLES[split],
        f"observed={n}, expected={EXPECTED_SAMPLES[split]}"
    )

    # -------------------------------------------------------------
    # Scenario count
    # -------------------------------------------------------------

    unique_sid = np.unique(sid)
    scenario_count = len(unique_sid)

    check(
        f"{split}_scenario_count",
        scenario_count == EXPECTED_SCENARIOS[split],
        f"observed={scenario_count}, expected={EXPECTED_SCENARIOS[split]}"
    )

    scenario_sets[split] = set(unique_sid.tolist())

    # -------------------------------------------------------------
    # Target validity
    # -------------------------------------------------------------

    y3_binary = np.all(np.isin(y3, [0, 1]))
    y6_binary = np.all(np.isin(y6, [0, 1]))

    check(
        f"{split}_y3_binary",
        y3_binary
    )

    check(
        f"{split}_y6_binary",
        y6_binary
    )

    y3_pos = int(np.sum(y3))
    y6_pos = int(np.sum(y6))

    y6_only = int(
        np.sum((y6 == 1) & (y3 == 0))
    )

    both = int(
        np.sum((y3 == 1) & (y6 == 1))
    )

    neither = int(
        np.sum((y3 == 0) & (y6 == 0))
    )

    # -------------------------------------------------------------
    # Horizon distinction
    # -------------------------------------------------------------

    check(
        f"{split}_horizon_separation",
        y6_only > 0,
        f"Y3+={y3_pos}, Y6+={y6_pos}, Y6_only={y6_only}"
    )

    # -------------------------------------------------------------
    # Origin structure
    # -------------------------------------------------------------

    origin_values = np.unique(origins)

    origin_values_ok = np.array_equal(
        origin_values,
        EXPECTED_ORIGINS
    )

    check(
        f"{split}_origin_values",
        origin_values_ok,
        f"observed={origin_values.tolist()}, "
        f"expected={EXPECTED_ORIGINS.tolist()}"
    )

    # Exactly seven origins per scenario.
    unique_counts = np.unique(
        np.unique(sid, return_counts=True)[1]
    )

    seven_per_scenario = (
        len(unique_counts) == 1
        and unique_counts[0] == 7
    )

    check(
        f"{split}_seven_origins_per_scenario",
        seven_per_scenario,
        f"unique_sample_counts_per_scenario={unique_counts.tolist()}"
    )

    # -------------------------------------------------------------
    # Fast scenario-origin integrity
    #
    # The builder creates samples in scenario-major, origin-major
    # order. Once 7 samples/scenario is confirmed, reshape is safe.
    # -------------------------------------------------------------

    origin_integrity = False

    if seven_per_scenario and n % 7 == 0:

        sid_matrix = np.asarray(sid).reshape(-1, 7)
        origin_matrix = np.asarray(origins).reshape(-1, 7)

        same_sid = np.all(
            sid_matrix == sid_matrix[:, [0]]
        )

        expected_matrix = np.broadcast_to(
            EXPECTED_ORIGINS,
            origin_matrix.shape
        )

        correct_origins = np.all(
            origin_matrix == expected_matrix
        )

        origin_integrity = bool(
            same_sid and correct_origins
        )

    check(
        f"{split}_prediction_origin_integrity",
        origin_integrity
    )

    # -------------------------------------------------------------
    # Finite-value scan
    # -------------------------------------------------------------

    finite = np.isfinite(X).all()

    check(
        f"{split}_finite_tensor",
        finite
    )

    # -------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------

    y3_rate = y3_pos / n
    y6_rate = y6_pos / n

    report["splits"][split] = {
        "samples": n,
        "tensor_shape": list(X.shape),
        "scenarios": scenario_count,
        "y3_positive": y3_pos,
        "y3_rate": y3_rate,
        "y6_positive": y6_pos,
        "y6_rate": y6_rate,
        "y6_only": y6_only,
        "both_positive": both,
        "neither_positive": neither,
        "y3_negative_to_positive_ratio": (
            (n - y3_pos) / y3_pos if y3_pos else None
        ),
        "y6_negative_to_positive_ratio": (
            (n - y6_pos) / y6_pos if y6_pos else None
        ),
        "origin_values": origin_values.tolist(),
        "seven_origins_per_scenario": seven_per_scenario,
        "finite": bool(finite),
    }

    print(
        f"samples={n:,} | scenarios={scenario_count:,} | "
        f"Y3+={y3_pos:,} ({y3_rate:.6%}) | "
        f"Y6+={y6_pos:,} ({y6_rate:.6%}) | "
        f"Y6-only={y6_only:,}"
    )

# =====================================================================
# CROSS-SPLIT SCENARIO LEAKAGE
# =====================================================================

print()
print("-" * 100)
print("CROSS-SPLIT LEAKAGE")

leakage = {}

for i, a in enumerate(SPLITS):
    for b in SPLITS[i + 1:]:
        overlap = scenario_sets[a] & scenario_sets[b]

        leakage[f"{a}_vs_{b}"] = len(overlap)

        check(
            f"scenario_leakage_{a}_{b}",
            len(overlap) == 0,
            f"overlap={len(overlap)}"
        )

report["scenario_leakage"] = leakage

# =====================================================================
# GLOBAL TARGET CHECK
# =====================================================================

y3_all = np.concatenate([
    np.asarray(
        np.load(
            SEQ / f"y_3step_{split}_v2_forecasting.npy"
        )
    )
    for split in SPLITS
])

y6_all = np.concatenate([
    np.asarray(
        np.load(
            SEQ / f"y_6step_{split}_v2_forecasting.npy"
        )
    )
    for split in SPLITS
])

global_y3 = int(y3_all.sum())
global_y6 = int(y6_all.sum())

global_y6_only = int(
    np.sum((y6_all == 1) & (y3_all == 0))
)

check(
    "global_horizon_targets_distinct",
    global_y6_only > 0,
    f"Y3+={global_y3}, Y6+={global_y6}, Y6_only={global_y6_only}"
)

# =====================================================================
# GLOBAL SUMMARY
# =====================================================================

report["global"] = {
    "samples": len(y3_all),
    "y3_positive": global_y3,
    "y6_positive": global_y6,
    "y6_only": global_y6_only,
    "y3_rate": global_y3 / len(y3_all),
    "y6_rate": global_y6 / len(y6_all),
}

# =====================================================================
# SAVE
# =====================================================================

REPORT.write_text(
    json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
    encoding="utf-8"
)

print()
print("=" * 100)
print(f"STATUS : {report['status']}")
print(f"REPORT : {REPORT}")
print("=" * 100)

if report["status"] != "PASS":
    raise SystemExit(
        "PHASE 4.5 QUALITY GATE FAILED — inspect the report before modeling."
    )

