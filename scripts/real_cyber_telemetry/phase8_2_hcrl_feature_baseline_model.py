"""
Phase 8.2 -- Baseline Detection Model on Real Cyber Telemetry (HCRL)
======================================================================

STATUS: Implemented (code). Run locally after Phase 8.1 has produced
hcrl_canonical.csv. This is a SEPARATE baseline model -- it does NOT
touch, modify, or feed into the project's existing frozen Transformer
checkpoint or its evidence chain. Kept deliberately independent per
the project's own discipline: don't contaminate validated evidence
with an untested integration step.

Objective
---------
Establish an honest, real-data intrusion-detection BASELINE (Layer 3:
cybersecurity monitoring / detection -- not prediction, not fusion with
vehicle-state) on the 16.57M real CAN frames ingested in Phase 8.1.

Features (standard CAN-IDS baseline features, not vehicle-state
features -- deliberately independent of the existing 59-dim schema):
  - delta_t          : seconds since the last frame with the same CAN ID
  - freq_1s          : count of frames with the same CAN ID in the
                        preceding 1.0s window (rolling)
  - dlc              : declared data length code
  - hamming_dist      : Hamming distance between this frame's data bytes
                        and the previous frame's data bytes for the same
                        CAN ID (over the overlapping byte length)
  - dlc_changed       : 1 if this frame's DLC differs from the previous
                        frame's DLC for the same CAN ID, else 0

Model: RandomForestClassifier (simplest defensible baseline; matches
project preference for the simplest technically defensible method
before considering anything heavier).

Evaluation: precision/recall/F1 per class (NORMAL + 4 attack types),
confusion matrix, and binary detection TPR/FPR, using a TIME-BASED
split per source file (first 70% chronological = train, last 30% =
test) -- NOT a random shuffle, since CAN traffic is sequential and a
random split would leak information across the delta_t / freq_1s
features.

Usage
-----
Run after Phase 8.1 has produced:
    data/processed/real_cyber_telemetry/hcrl_car_hacking/hcrl_canonical.csv

    python scripts/real_cyber_telemetry/phase8_2_hcrl_feature_baseline_model.py

Output
------
- data/processed/real_cyber_telemetry/hcrl_car_hacking/hcrl_features.csv
- models/real_cyber_telemetry_baseline/rf_baseline_model.joblib
- experiments/real_cyber_telemetry/phase8_2_baseline_model_report.json
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
    ROOT / "data" / "processed" / "real_cyber_telemetry" / "hcrl_car_hacking" / "hcrl_features.csv"
)
MODEL_DIR = ROOT / "models" / "real_cyber_telemetry_baseline"
MODEL_PATH = MODEL_DIR / "rf_baseline_model.joblib"
REPORT_DIR = ROOT / "experiments" / "real_cyber_telemetry"
REPORT_PATH = REPORT_DIR / "phase8_2_baseline_model_report.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SECONDS = 1.0
CHUNK_SIZE = 500_000
TRAIN_FRACTION = 0.70
FEATURE_COLUMNS = ["delta_t", "freq_1s", "dlc", "hamming_dist", "dlc_changed"]


def popcount(x):
    return bin(x).count("1")


def hamming_distance(bytes_a, bytes_b):
    n = min(len(bytes_a), len(bytes_b))
    dist = 0
    for i in range(n):
        dist += popcount(bytes_a[i] ^ bytes_b[i])
    return dist


def parse_data_bytes(data_bytes_str):
    if not data_bytes_str:
        return []
    parts = str(data_bytes_str).split("|")
    result = []
    for p in parts:
        if p == "":
            continue
        try:
            result.append(int(p, 16))
        except ValueError:
            result.append(0)
    return result


def extract_features():
    """Streams the canonical CSV in chunks, maintaining per-CAN-ID
    state across chunk boundaries, and appends feature rows to
    FEATURES_CSV as it goes so memory stays bounded."""

    if FEATURES_CSV.exists():
        FEATURES_CSV.unlink()

    last_ts = {}
    last_bytes = {}
    last_dlc = {}
    recent_ts_window = {}  # can_id -> deque of timestamps within WINDOW_SECONDS

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
        out_df.to_csv(
            FEATURES_CSV,
            mode="a",
            header=not header_written,
            index=False,
        )
        header_written = True

        total_rows += len(out_rows)
        elapsed = time.time() - t_start
        print(
            f"[feature extraction] chunk {chunk_idx + 1}: "
            f"{total_rows:,} rows processed ({elapsed:.1f}s elapsed)"
        )

    return total_rows


def time_based_split(df):
    train_parts = []
    test_parts = []
    for source_file, group in df.groupby("source_file", observed=True):
        group_sorted = group.sort_values("timestamp")
        cut = int(len(group_sorted) * TRAIN_FRACTION)
        train_parts.append(group_sorted.iloc[:cut])
        test_parts.append(group_sorted.iloc[cut:])
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def train_and_evaluate():
    df = pd.read_csv(
        FEATURES_CSV,
        dtype={
            "source_file": "category",
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

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    # n_estimators/max_depth deliberately bounded: unbounded depth on
    # 11M+ rows risks very long training time and an oversized model
    # file for a first baseline. These are a documented, honest choice
    # for tractability -- not a tuned/optimized configuration.
    rf_params = {
        "n_estimators": 50,
        "max_depth": 20,
        "n_jobs": -1,
        "random_state": 42,
    }
    clf = RandomForestClassifier(**rf_params)
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_PATH)

    y_pred = clf.predict(X_test)

    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique().tolist()))

    # Binary detection view: attack vs normal, regardless of attack type
    y_test_binary = test_df["is_attack"].values
    y_pred_binary = (y_pred != "NORMAL")

    tp = int(((y_pred_binary == True) & (y_test_binary == True)).sum())
    fp = int(((y_pred_binary == True) & (y_test_binary == False)).sum())
    fn = int(((y_pred_binary == False) & (y_test_binary == True)).sum())
    tn = int(((y_pred_binary == False) & (y_test_binary == False)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    precision_binary = tp / (tp + fp) if (tp + fp) > 0 else None

    feature_importances = dict(zip(FEATURE_COLUMNS, clf.feature_importances_.tolist()))

    report = {
        "phase": "8.2",
        "name": "HCRL real cyber telemetry -- baseline RandomForest detection model",
        "data_class": "REAL_DATA_BASELINE_MODEL",
        "fused_with_main_pipeline": False,
        "note": (
            "This model is trained and evaluated independently of the "
            "project's existing frozen Transformer checkpoint. It is a "
            "detection baseline (Layer 3), not a prediction or fusion "
            "component."
        ),
        "split_method": "time-based, per source file, 70/30",
        "model_hyperparameters": rf_params,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "classification_report": class_report,
        "confusion_matrix": {
            "labels": sorted(y_test.unique().tolist()),
            "matrix": conf_matrix.tolist(),
        },
        "binary_detection_metrics": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "true_positive_rate_recall": tpr,
            "false_positive_rate": fpr,
            "precision": precision_binary,
        },
        "feature_importances": feature_importances,
        "model_path": str(MODEL_PATH),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    print("=" * 80)
    print("PHASE 8.2 -- BASELINE DETECTION MODEL ON REAL CYBER TELEMETRY")
    print("=" * 80)

    if not CANONICAL_CSV.exists():
        print(f"ERROR: canonical CSV not found at {CANONICAL_CSV}")
        print("Run Phase 8.1 first.")
        return

    print("\nStep 1/2: extracting features (streaming, this covers all 16.5M+ rows)...")
    total_rows = extract_features()
    print(f"Feature extraction complete: {total_rows:,} rows written to {FEATURES_CSV}")

    print("\nStep 2/2: training and evaluating baseline RandomForest model...")
    report = train_and_evaluate()

    print()
    print("-" * 80)
    print("RESULTS")
    print("-" * 80)
    print("Train rows:", report["train_rows"])
    print("Test rows :", report["test_rows"])
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
    print("Feature importances:", report["feature_importances"])
    print()
    print("Report:", REPORT_PATH)
    print("Model :", MODEL_PATH)
    print("=" * 80)


if __name__ == "__main__":
    main()
