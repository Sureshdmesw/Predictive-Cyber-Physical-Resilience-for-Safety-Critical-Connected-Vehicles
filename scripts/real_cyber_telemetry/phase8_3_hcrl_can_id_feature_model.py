"""
Phase 8.3 -- CAN-ID-Aware Detection Model (fixes Phase 8.2's GEAR/RPM confusion)
================================================================================

STATUS: Implemented (code). Run locally after Phase 8.1 has produced
hcrl_canonical.csv. Does NOT overwrite or modify Phase 8.2's model or
report -- both are kept as the documented baseline for comparison.

Why this phase exists (real evidence from Phase 8.2, run 2026-09-16)
----------------------------------------------------------------------
Phase 8.2's confusion matrix showed:
  - GEAR_SPOOFING -> predicted RPM_SPOOFING: 62,566 / 69,037 (90.6%)
  - RPM_SPOOFING  -> predicted NORMAL:       65,931 / 82,652 (79.8%)
  - NORMAL        -> false-flagged as GEAR_SPOOFING or RPM_SPOOFING: ~160,481

Phase 8.2's features (delta_t, freq_1s, dlc, hamming_dist, dlc_changed)
were computed PER CAN ID internally but the CAN ID itself was never
passed to the model. Gear-position and RPM spoofing attacks target
different, specific arbitration IDs on this vehicle -- without the ID,
the model can only see generic "a periodic signal's value changed"
numbers that look alike across different targeted IDs. This phase adds
CAN ID as a feature to test whether that closes the gap.

What's new vs Phase 8.2
------------------------
- Feature extraction now also captures can_id per row.
- can_id is one-hot encoded, but ONLY for the top-N most frequent
  TRAINING IDs -- everything else is bucketed as OTHER. (v1 of this
  script tried to one-hot ALL distinct IDs and crashed: the real
  dataset has 2,048 distinct IDs in training, mostly one-off random
  IDs injected by the Fuzzy attack, and tried to allocate a 22GB
  array. A one-off random ID has no repeatable pattern to learn from
  anyway, so bucketing the tail loses no real signal.)
- Everything else (model type, hyperparameters, time-based 70/30 split
  per source file) is held constant vs Phase 8.2, so any difference in
  results is attributable to the CAN-ID feature, not a confound.

Usage
-----
    python scripts/real_cyber_telemetry/phase8_3_hcrl_can_id_feature_model.py

Output
------
- data/processed/real_cyber_telemetry/hcrl_car_hacking/hcrl_features_v2_canid.csv
- models/real_cyber_telemetry_baseline/rf_v2_canid_model.joblib
- experiments/real_cyber_telemetry/phase8_3_canid_feature_model_report.json
"""

from pathlib import Path
from collections import deque
import json
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

ROOT = Path(__file__).resolve().parents[2]

CANONICAL_CSV = (
    ROOT / "data" / "processed" / "real_cyber_telemetry" / "hcrl_car_hacking" / "hcrl_canonical.csv"
)
FEATURES_CSV = (
    ROOT / "data" / "processed" / "real_cyber_telemetry" / "hcrl_car_hacking" / "hcrl_features_v2_canid.csv"
)
MODEL_DIR = ROOT / "models" / "real_cyber_telemetry_baseline"
MODEL_PATH = MODEL_DIR / "rf_v2_canid_model.joblib"
REPORT_DIR = ROOT / "experiments" / "real_cyber_telemetry"
REPORT_PATH = REPORT_DIR / "phase8_3_canid_feature_model_report.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SECONDS = 1.0
CHUNK_SIZE = 500_000
TRAIN_FRACTION = 0.70
NUMERIC_FEATURE_COLUMNS = ["delta_t", "freq_1s", "dlc", "hamming_dist", "dlc_changed"]

# One-hot encoding EVERY distinct CAN ID blew up memory: the real
# dataset has 2,048 distinct IDs in training (confirmed by direct
# run -- the Fuzzy attack injects frames with randomized IDs across
# the full 11-bit address space, most appearing only once or twice).
# A one-off random ID has no repeatable timing/identity pattern to
# learn from, so collapsing the long tail into OTHER loses nothing
# real. Only the top-N most frequent (genuinely recurring, real
# periodic) IDs are individually encoded.
TOP_N_CAN_IDS = 30

# Real Phase 8.2 results, recorded verbatim from the actual run on
# 2026-09-16 (not re-derived here) -- used only as a fixed reference
# point for the before/after comparison below.
PHASE_8_2_BASELINE = {
    "run_date": "2026-09-16",
    "f1_gear_spoofing": 0.0012,
    "f1_rpm_spoofing": 0.1609,
    "binary_tpr": 0.6634901699680038,
    "binary_fpr": 0.033804122687907716,
    "binary_precision": 0.47168670144401603,
}


def popcount(x):
    return bin(x).count("1")


def hamming_distance(bytes_a, bytes_b):
    n = min(len(bytes_a), len(bytes_b))
    return sum(popcount(bytes_a[i] ^ bytes_b[i]) for i in range(n))


def parse_data_bytes(data_bytes_str):
    if not data_bytes_str:
        return []
    result = []
    for p in str(data_bytes_str).split("|"):
        if p == "":
            continue
        try:
            result.append(int(p, 16))
        except ValueError:
            result.append(0)
    return result


def extract_features():
    if FEATURES_CSV.exists():
        FEATURES_CSV.unlink()

    last_ts, last_bytes, last_dlc, recent_ts_window = {}, {}, {}, {}
    total_rows = 0
    t_start = time.time()

    reader = pd.read_csv(
        CANONICAL_CSV,
        dtype={
            "data_class": "category",
            "source_file": "category",
            "can_id": "string",
            "data_bytes": "string",
            "label": "category",
            "is_attack": "bool",
        },
        chunksize=CHUNK_SIZE,
    )

    header_written = False

    for chunk_idx, chunk in enumerate(reader):
        out_rows = []
        for row in chunk.itertuples(index=False):
            can_id = row.can_id
            ts = float(row.timestamp)
            dlc_val = int(row.dlc)
            data_bytes = parse_data_bytes(row.data_bytes)

            prev_ts = last_ts.get(can_id)
            delta_t = (ts - prev_ts) if prev_ts is not None else 0.0

            window = recent_ts_window.setdefault(can_id, deque())
            while window and ts - window[0] > WINDOW_SECONDS:
                window.popleft()
            window.append(ts)
            freq_1s = len(window)

            prev_bytes = last_bytes.get(can_id, [])
            hdist = hamming_distance(data_bytes, prev_bytes) if prev_bytes else 0
            prev_dlc = last_dlc.get(can_id)
            dlc_changed = int(prev_dlc is not None and prev_dlc != dlc_val)

            out_rows.append(
                {
                    "source_file": row.source_file,
                    "timestamp": ts,
                    "can_id": can_id,  # NEW vs Phase 8.2
                    "delta_t": delta_t,
                    "freq_1s": freq_1s,
                    "dlc": dlc_val,
                    "hamming_dist": hdist,
                    "dlc_changed": dlc_changed,
                    "label": row.label,
                    "is_attack": bool(row.is_attack),
                }
            )

            last_ts[can_id] = ts
            last_bytes[can_id] = data_bytes
            last_dlc[can_id] = dlc_val

        out_df = pd.DataFrame(out_rows)
        out_df.to_csv(FEATURES_CSV, mode="a", header=not header_written, index=False)
        header_written = True

        total_rows += len(out_rows)
        elapsed = time.time() - t_start
        print(f"[feature extraction] chunk {chunk_idx + 1}: {total_rows:,} rows ({elapsed:.1f}s elapsed)")

    return total_rows


def time_based_split(df):
    train_parts, test_parts = [], []
    for _, group in df.groupby("source_file", observed=True):
        g = group.sort_values("timestamp")
        cut = int(len(g) * TRAIN_FRACTION)
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def encode_can_id(df, top_ids):
    """Maps any CAN ID not in top_ids to 'OTHER' before one-hot
    encoding, so the output width is fixed at len(top_ids)+1
    regardless of how many rare/random IDs exist in this split."""
    bucketed = df["can_id"].where(df["can_id"].isin(top_ids), other="OTHER")
    cat = pd.Categorical(bucketed, categories=list(top_ids) + ["OTHER"])
    return pd.get_dummies(cat, prefix="can_id")


def train_and_evaluate():
    df = pd.read_csv(
        FEATURES_CSV,
        dtype={
            "source_file": "category",
            "can_id": "string",
            "delta_t": "float32",
            "freq_1s": "int32",
            "dlc": "int32",
            "hamming_dist": "int32",
            "dlc_changed": "int8",
            "label": "category",
            "is_attack": "bool",
        },
    )

    train_df, test_df = time_based_split(df)

    # Only the top-N most frequent TRAINING IDs get their own one-hot
    # column; everything else (the long tail of near-unique, mostly
    # Fuzzy-attack random IDs) is bucketed as OTHER. Computed from
    # TRAIN only, same anti-leakage principle as before.
    id_counts = train_df["can_id"].value_counts()
    top_ids = id_counts.nlargest(TOP_N_CAN_IDS).index.tolist()

    train_id_dummies = encode_can_id(train_df, top_ids)
    test_id_dummies = encode_can_id(test_df, top_ids)

    train_other_frac = (train_id_dummies["can_id_OTHER"].sum() / len(train_id_dummies)) if len(train_id_dummies) else 0
    test_other_frac = (test_id_dummies["can_id_OTHER"].sum() / len(test_id_dummies)) if len(test_id_dummies) else 0
    print(
        f"  CAN ID encoding: {len(top_ids)} individually-encoded IDs "
        f"(+OTHER). OTHER covers {train_other_frac:.4%} of train rows, "
        f"{test_other_frac:.4%} of test rows."
    )

    X_train = pd.concat(
        [train_df[NUMERIC_FEATURE_COLUMNS].reset_index(drop=True), train_id_dummies.reset_index(drop=True)],
        axis=1,
    )
    X_test = pd.concat(
        [test_df[NUMERIC_FEATURE_COLUMNS].reset_index(drop=True), test_id_dummies.reset_index(drop=True)],
        axis=1,
    )
    y_train = train_df["label"].reset_index(drop=True)
    y_test = test_df["label"].reset_index(drop=True)

    rf_params = {"n_estimators": 50, "max_depth": 20, "n_jobs": -1, "random_state": 42}
    clf = RandomForestClassifier(**rf_params)
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_PATH)

    y_pred = clf.predict(X_test)

    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    labels_sorted = sorted(y_test.unique().tolist())
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    y_test_binary = test_df["is_attack"].values
    y_pred_binary = (y_pred != "NORMAL")

    tp = int(((y_pred_binary == True) & (y_test_binary == True)).sum())
    fp = int(((y_pred_binary == True) & (y_test_binary == False)).sum())
    fn = int(((y_pred_binary == False) & (y_test_binary == True)).sum())
    tn = int(((y_pred_binary == False) & (y_test_binary == False)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    precision_binary = tp / (tp + fp) if (tp + fp) > 0 else None

    f1_gear = class_report.get("GEAR_SPOOFING", {}).get("f1-score", 0.0)
    f1_rpm = class_report.get("RPM_SPOOFING", {}).get("f1-score", 0.0)

    comparison = {
        "phase_8_2_baseline": PHASE_8_2_BASELINE,
        "phase_8_3_result": {
            "f1_gear_spoofing": f1_gear,
            "f1_rpm_spoofing": f1_rpm,
            "binary_tpr": tpr,
            "binary_fpr": fpr,
            "binary_precision": precision_binary,
        },
        "delta": {
            "f1_gear_spoofing": f1_gear - PHASE_8_2_BASELINE["f1_gear_spoofing"],
            "f1_rpm_spoofing": f1_rpm - PHASE_8_2_BASELINE["f1_rpm_spoofing"],
            "binary_tpr": (tpr - PHASE_8_2_BASELINE["binary_tpr"]) if tpr is not None else None,
            "binary_fpr": (fpr - PHASE_8_2_BASELINE["binary_fpr"]) if fpr is not None else None,
        },
    }

    report = {
        "phase": "8.3",
        "name": "HCRL real cyber telemetry -- CAN-ID-aware detection model",
        "data_class": "REAL_DATA_BASELINE_MODEL",
        "fused_with_main_pipeline": False,
        "hypothesis": (
            "Phase 8.2's GEAR/RPM confusion was caused by omitting CAN ID "
            "from the feature set. Adding it as a one-hot feature should "
            "improve GEAR_SPOOFING and RPM_SPOOFING detection without "
            "requiring any new data source."
        ),
        "model_hyperparameters": rf_params,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "num_can_id_categories": len(top_ids) + 1,
        "can_id_other_fraction_train": train_other_frac,
        "can_id_other_fraction_test": test_other_frac,
        "classification_report": class_report,
        "confusion_matrix": {"labels": labels_sorted, "matrix": conf_matrix.tolist()},
        "binary_detection_metrics": {
            "true_positives": tp, "false_positives": fp,
            "false_negatives": fn, "true_negatives": tn,
            "true_positive_rate_recall": tpr, "false_positive_rate": fpr,
            "precision": precision_binary,
        },
        "comparison_vs_phase_8_2": comparison,
        "model_path": str(MODEL_PATH),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    print("=" * 80)
    print("PHASE 8.3 -- CAN-ID-AWARE DETECTION MODEL")
    print("=" * 80)

    if not CANONICAL_CSV.exists():
        print(f"ERROR: canonical CSV not found at {CANONICAL_CSV}")
        return

    print("\nStep 1/2: extracting features (now including can_id)...")
    total_rows = extract_features()
    print(f"Feature extraction complete: {total_rows:,} rows written to {FEATURES_CSV}")

    print("\nStep 2/2: training and evaluating CAN-ID-aware model...")
    report = train_and_evaluate()

    print()
    print("-" * 80)
    print("RESULTS")
    print("-" * 80)
    print("Train rows:", report["train_rows"], "| Test rows:", report["test_rows"])
    print("CAN ID categories used:", report["num_can_id_categories"])
    print()
    print("Per-class precision/recall/F1:")
    for label, metrics in report["classification_report"].items():
        if isinstance(metrics, dict):
            print(
                f"  {label:16s} precision={metrics.get('precision', 0):.4f} "
                f"recall={metrics.get('recall', 0):.4f} "
                f"f1={metrics.get('f1-score', 0):.4f} "
                f"support={metrics.get('support', 0)}"
            )
    print()
    bm = report["binary_detection_metrics"]
    print("Binary detection (attack vs normal, any attack type):")
    print(f"  TPR (recall) : {bm['true_positive_rate_recall']}")
    print(f"  FPR          : {bm['false_positive_rate']}")
    print(f"  Precision    : {bm['precision']}")
    print()
    print("-" * 80)
    print("BEFORE (Phase 8.2) vs AFTER (Phase 8.3)")
    print("-" * 80)
    comp = report["comparison_vs_phase_8_2"]
    print(f"  F1 GEAR_SPOOFING : {comp['phase_8_2_baseline']['f1_gear_spoofing']:.4f} -> {comp['phase_8_3_result']['f1_gear_spoofing']:.4f}  (delta {comp['delta']['f1_gear_spoofing']:+.4f})")
    print(f"  F1 RPM_SPOOFING  : {comp['phase_8_2_baseline']['f1_rpm_spoofing']:.4f} -> {comp['phase_8_3_result']['f1_rpm_spoofing']:.4f}  (delta {comp['delta']['f1_rpm_spoofing']:+.4f})")
    print(f"  Binary TPR       : {comp['phase_8_2_baseline']['binary_tpr']:.4f} -> {comp['phase_8_3_result']['binary_tpr']:.4f}  (delta {comp['delta']['binary_tpr']:+.4f})")
    print(f"  Binary FPR       : {comp['phase_8_2_baseline']['binary_fpr']:.4f} -> {comp['phase_8_3_result']['binary_fpr']:.4f}  (delta {comp['delta']['binary_fpr']:+.4f})")
    print()
    print("Report:", REPORT_PATH)
    print("Model :", MODEL_PATH)
    print("=" * 80)


if __name__ == "__main__":
    main()
