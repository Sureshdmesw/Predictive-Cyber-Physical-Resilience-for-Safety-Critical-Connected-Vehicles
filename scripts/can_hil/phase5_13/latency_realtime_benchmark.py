from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_10"
    / "phase5_10_cyber_physical_attack_scenarios.csv"
)

CHECKPOINT = (
    ROOT
    / "models"
    / "v2"
    / "advanced"
    / "crss_2024_v2_temporal_transformer_best.pt"
)

PHASE512_DIR = (
    ROOT
    / "scripts"
    / "can_hil"
    / "phase5_12"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_13"
)

REPORT_DIR = (
    ROOT
    / "experiments"
    / "can_hil"
    / "phase5_13"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "phase5_13_latency_realtime.csv"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "phase5_13_latency_realtime.jsonl"
)

SCHEMA = (
    OUTPUT_DIR
    / "phase5_13_latency_realtime_schema.json"
)

MANIFEST = (
    REPORT_DIR
    / "phase5_13_latency_realtime_manifest.json"
)

OBSERVATION_STEPS = 12
RECORDS_PER_TIMESTEP = 15
EXPECTED_TIMESTEPS = 600
EXPECTED_WINDOWS = 589
EXPECTED_SCENARIOS = 11
FEATURE_COUNT = 59
MODEL_PARAMS = 407811

Y3_HORIZON_SECONDS = 15.0
Y6_HORIZON_SECONDS = 30.0

WARMUP_WINDOWS = 20
MEASURED_WINDOWS = 120
REPEATS = 3

DEVICE = torch.device("cpu")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_phase512():
    phase512_path = PHASE512_DIR.resolve()

    sys.path.insert(0, str(phase512_path))

    import uncertainty_ood_analysis as phase512

    return phase512


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)

    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
        }

    return {
        "count": int(len(arr)),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
    }


def benchmark_feature_pipeline(
    phase512,
    stream_df,
):
    timings = []
    windows = None

    for _ in range(REPEATS):
        start = time.perf_counter()

        feature_matrix, timestamp_spread = (
            phase512.build_stream(stream_df)
        )

        build_done = time.perf_counter()

        windows = phase512.make_windows(
            feature_matrix
        )

        window_done = time.perf_counter()

        timings.append(
            {
                "feature_build_ms":
                    (build_done - start) * 1000.0,
                "window_build_ms":
                    (window_done - build_done) * 1000.0,
                "feature_plus_window_ms":
                    (window_done - start) * 1000.0,
            }
        )

    return timings, windows


def benchmark_model(
    model,
    windows,
):
    warmup = windows[
        :min(WARMUP_WINDOWS, len(windows))
    ]

    warmup_tensor = torch.from_numpy(
        warmup
    )

    for _ in range(2):
        with torch.inference_mode():
            model(warmup_tensor)

    selected = windows[
        :min(MEASURED_WINDOWS, len(windows))
    ]

    inference_times = []
    end_to_end_times = []

    y3_values = []
    y6_values = []

    for window in selected:
        tensor_start = time.perf_counter()

        tensor = torch.from_numpy(
            window[np.newaxis, ...]
        )

        tensor_ready = time.perf_counter()

        with torch.inference_mode():
            output = model(tensor)

        inference_done = time.perf_counter()

        if isinstance(output, tuple):
            y3_logits = output[0]
            y6_logits = output[1]

        elif isinstance(output, dict):
            y3_logits = output["y3"]
            y6_logits = output["y6"]

        else:
            raise RuntimeError(
                "Unsupported Transformer output type."
            )

        y3 = torch.sigmoid(
            y3_logits
        ).reshape(-1)[0].item()

        y6 = torch.sigmoid(
            y6_logits
        ).reshape(-1)[0].item()

        completed = time.perf_counter()

        inference_times.append(
            (inference_done - tensor_ready)
            * 1000.0
        )

        end_to_end_times.append(
            (completed - tensor_start)
            * 1000.0
        )

        y3_values.append(float(y3))
        y6_values.append(float(y6))

    return {
        "transformer": summarize(
            inference_times
        ),
        "prediction_end_to_end": summarize(
            end_to_end_times
        ),
        "y3_mean": float(
            np.mean(y3_values)
        ),
        "y6_mean": float(
            np.mean(y6_values)
        ),
        "measured_windows": int(
            len(selected)
        ),
    }


def main():
    print()
    print(
        "=== PHASE 5.13 LATENCY & REAL-TIME BEHAVIOR ==="
    )
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            INPUT_CSV
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            CHECKPOINT
        )

    if not PHASE512_DIR.exists():
        raise FileNotFoundError(
            PHASE512_DIR
        )

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    print(
        f"Input rows: {len(df)}"
    )

    required_columns = {
        "scenario_id",
        "signals_json",
        "timestamp_s",
        "can_id",
        "message_name",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + str(sorted(missing))
        )

    scenarios = sorted(
        df[
            "scenario_id"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    print(
        f"Scenarios: {len(scenarios)}"
    )

    for scenario in scenarios:
        print(
            f"  {scenario}"
        )

    if len(scenarios) != EXPECTED_SCENARIOS:
        raise ValueError(
            "Expected 11 scenarios, got "
            f"{len(scenarios)}."
        )

    print()
    print(
        "Loading validated Phase 5.12 "
        "feature/model pipeline..."
    )

    phase512 = load_phase512()

    model = phase512.load_model()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    checkpoint_sha256 = sha256_file(
        CHECKPOINT
    )

    print(
        f"Model parameters: "
        f"{parameter_count}"
    )

    print(
        f"Checkpoint SHA256: "
        f"{checkpoint_sha256}"
    )

    if parameter_count != MODEL_PARAMS:
        raise RuntimeError(
            "Unexpected model parameter count: "
            f"{parameter_count}"
        )

    scenario_results = {}
    output_records = []

    for scenario in scenarios:
        print()
        print(
            f"Benchmarking {scenario}..."
        )

        stream = df[
            df["scenario_id"].astype(str).str.strip()
            == scenario
        ].copy()

        if len(stream) != 9000:
            raise RuntimeError(
                f"{scenario}: expected 9000 "
                f"records, got {len(stream)}."
            )

        feature_timings, windows = (
            benchmark_feature_pipeline(
                phase512,
                stream,
            )
        )

        if windows.shape != (
            EXPECTED_WINDOWS,
            OBSERVATION_STEPS,
            FEATURE_COUNT,
        ):
            raise RuntimeError(
                f"{scenario}: unexpected "
                f"window shape {windows.shape}"
            )

        feature_summary = summarize(
            [
                x["feature_build_ms"]
                for x in feature_timings
            ]
        )

        window_summary = summarize(
            [
                x["window_build_ms"]
                for x in feature_timings
            ]
        )

        feature_window_summary = summarize(
            [
                x["feature_plus_window_ms"]
                for x in feature_timings
            ]
        )

        model_result = benchmark_model(
            model,
            windows,
        )

        result = {
            "scenario_id": scenario,
            "records": int(len(stream)),
            "timesteps": int(
                len(stream)
                // RECORDS_PER_TIMESTEP
            ),
            "windows": int(len(windows)),
            "measured_windows":
                model_result["measured_windows"],
            "feature_build_mean_ms":
                feature_summary["mean_ms"],
            "feature_build_median_ms":
                feature_summary["median_ms"],
            "feature_build_p95_ms":
                feature_summary["p95_ms"],
            "feature_build_p99_ms":
                feature_summary["p99_ms"],
            "window_build_mean_ms":
                window_summary["mean_ms"],
            "window_build_median_ms":
                window_summary["median_ms"],
            "window_build_p95_ms":
                window_summary["p95_ms"],
            "window_build_p99_ms":
                window_summary["p99_ms"],
            "feature_window_mean_ms":
                feature_window_summary["mean_ms"],
            "feature_window_p95_ms":
                feature_window_summary["p95_ms"],
            "feature_window_p99_ms":
                feature_window_summary["p99_ms"],
            "transformer_mean_ms":
                model_result[
                    "transformer"
                ]["mean_ms"],
            "transformer_median_ms":
                model_result[
                    "transformer"
                ]["median_ms"],
            "transformer_p95_ms":
                model_result[
                    "transformer"
                ]["p95_ms"],
            "transformer_p99_ms":
                model_result[
                    "transformer"
                ]["p99_ms"],
            "prediction_end_to_end_mean_ms":
                model_result[
                    "prediction_end_to_end"
                ]["mean_ms"],
            "prediction_end_to_end_median_ms":
                model_result[
                    "prediction_end_to_end"
                ]["median_ms"],
            "prediction_end_to_end_p95_ms":
                model_result[
                    "prediction_end_to_end"
                ]["p95_ms"],
            "prediction_end_to_end_p99_ms":
                model_result[
                    "prediction_end_to_end"
                ]["p99_ms"],
            "prediction_end_to_end_min_ms":
                model_result[
                    "prediction_end_to_end"
                ]["min_ms"],
            "prediction_end_to_end_max_ms":
                model_result[
                    "prediction_end_to_end"
                ]["max_ms"],
            "prediction_end_to_end_std_ms":
                model_result[
                    "prediction_end_to_end"
                ]["std_ms"],
            "y3_mean":
                model_result["y3_mean"],
            "y6_mean":
                model_result["y6_mean"],
            "cpu_only": True,
            "laboratory_only": True,
            "direct_actuation": False,
            "model_retraining": False,
        }

        scenario_results[
            scenario
        ] = result

        for metric, value in result.items():
            if metric == "scenario_id":
                continue

            output_records.append(
                {
                    "scenario_id": scenario,
                    "metric": metric,
                    "value": value,
                }
            )

        print(
            f"  records={result['records']}"
        )

        print(
            f"  timesteps={result['timesteps']}"
        )

        print(
            f"  windows={result['windows']}"
        )

        print(
            "  feature_build_mean="
            f"{result['feature_build_mean_ms']:.3f} ms"
        )

        print(
            "  transformer_mean="
            f"{result['transformer_mean_ms']:.3f} ms"
        )

        print(
            "  e2e_mean="
            f"{result['prediction_end_to_end_mean_ms']:.3f} ms"
        )

        print(
            "  e2e_p99="
            f"{result['prediction_end_to_end_p99_ms']:.3f} ms"
        )

    all_e2e = []

    all_transformer = []

    all_feature = []

    all_feature_window = []

    for result in scenario_results.values():
        all_e2e.append(
            result[
                "prediction_end_to_end_mean_ms"
            ]
        )

        all_transformer.append(
            result[
                "transformer_mean_ms"
            ]
        )

        all_feature.append(
            result[
                "feature_build_mean_ms"
            ]
        )

        all_feature_window.append(
            result[
                "feature_window_mean_ms"
            ]
        )

    aggregate_e2e = summarize(
        all_e2e
    )

    aggregate_transformer = summarize(
        all_transformer
    )

    aggregate_feature = summarize(
        all_feature
    )

    aggregate_feature_window = summarize(
        all_feature_window
    )

    maximum_e2e_p99 = max(
        result[
            "prediction_end_to_end_p99_ms"
        ]
        for result in scenario_results.values()
    )

    maximum_e2e_mean = max(
        result[
            "prediction_end_to_end_mean_ms"
        ]
        for result in scenario_results.values()
    )

    maximum_transformer_p99 = max(
        result[
            "transformer_p99_ms"
        ]
        for result in scenario_results.values()
    )

    y3_fraction = (
        maximum_e2e_p99
        / 1000.0
        / Y3_HORIZON_SECONDS
    )

    y6_fraction = (
        maximum_e2e_p99
        / 1000.0
        / Y6_HORIZON_SECONDS
    )

    checks = {}

    checks[
        "input_exists"
    ] = INPUT_CSV.exists()

    checks[
        "checkpoint_exists"
    ] = CHECKPOINT.exists()

    checks[
        "phase512_pipeline_exists"
    ] = PHASE512_DIR.exists()

    checks[
        "input_rows_99000"
    ] = len(df) == 99000

    checks[
        "scenario_count_11"
    ] = len(scenarios) == 11

    checks[
        "feature_count_59"
    ] = FEATURE_COUNT == 59

    checks[
        "parameter_count_407811"
    ] = parameter_count == MODEL_PARAMS

    checks[
        "cpu_only"
    ] = DEVICE.type == "cpu"

    checks[
        "all_scenarios_9000_records"
    ] = all(
        x["records"] == 9000
        for x in scenario_results.values()
    )

    checks[
        "all_scenarios_600_timesteps"
    ] = all(
        x["timesteps"] == 600
        for x in scenario_results.values()
    )

    checks[
        "all_scenarios_589_windows"
    ] = all(
        x["windows"] == 589
        for x in scenario_results.values()
    )

    checks[
        "all_scenarios_measured"
    ] = all(
        x["measured_windows"]
        == min(
            MEASURED_WINDOWS,
            EXPECTED_WINDOWS,
        )
        for x in scenario_results.values()
    )

    checks[
        "latencies_finite"
    ] = all(
        math.isfinite(
            x[
                "prediction_end_to_end_mean_ms"
            ]
        )
        and math.isfinite(
            x[
                "prediction_end_to_end_p99_ms"
            ]
        )
        for x in scenario_results.values()
    )

    checks[
        "transformer_latencies_finite"
    ] = all(
        math.isfinite(
            x["transformer_mean_ms"]
        )
        for x in scenario_results.values()
    )

    checks[
        "feature_latencies_finite"
    ] = all(
        math.isfinite(
            x["feature_build_mean_ms"]
        )
        for x in scenario_results.values()
    )

    checks[
        "positive_latency"
    ] = (
        maximum_e2e_mean > 0
    )

    checks[
        "positive_transformer_latency"
    ] = (
        aggregate_transformer["mean_ms"]
        > 0
    )

    checks[
        "y3_horizon_defined"
    ] = (
        Y3_HORIZON_SECONDS == 15.0
    )

    checks[
        "y6_horizon_defined"
    ] = (
        Y6_HORIZON_SECONDS == 30.0
    )

    checks[
        "latency_below_y3_horizon"
    ] = (
        maximum_e2e_p99
        / 1000.0
        < Y3_HORIZON_SECONDS
    )

    checks[
        "latency_below_y6_horizon"
    ] = (
        maximum_e2e_p99
        / 1000.0
        < Y6_HORIZON_SECONDS
    )

    checks[
        "laboratory_only"
    ] = all(
        x["laboratory_only"]
        for x in scenario_results.values()
    )

    checks[
        "no_direct_actuation"
    ] = all(
        not x["direct_actuation"]
        for x in scenario_results.values()
    )

    checks[
        "no_model_retraining"
    ] = all(
        not x["model_retraining"]
        for x in scenario_results.values()
    )

    output_df = pd.DataFrame(
        output_records
    )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    with open(
        OUTPUT_JSONL,
        "w",
        encoding="utf-8",
    ) as f:
        for record in output_records:
            f.write(
                json.dumps(
                    record,
                    allow_nan=False,
                )
                + "\n"
            )

    schema = {
        "phase": "5.13",
        "title":
            "Latency and Real-Time Behavior",
        "laboratory_only": True,
        "physical_vehicle": False,
        "direct_actuation": False,
        "model_retraining": False,
        "feature_pipeline_source":
            "Phase 5.12 validated pipeline",
        "input_rows": 99000,
        "scenario_count": 11,
        "feature_count": 59,
        "sequence_length": 12,
        "records_per_timestep": 15,
        "timesteps_per_scenario": 600,
        "windows_per_scenario": 589,
        "measured_windows_per_scenario":
            MEASURED_WINDOWS,
        "prediction_horizons_seconds": {
            "Y3": 15.0,
            "Y6": 30.0,
        },
        "latency_metrics": [
            "feature_build_mean_ms",
            "feature_build_median_ms",
            "feature_build_p95_ms",
            "feature_build_p99_ms",
            "window_build_mean_ms",
            "feature_window_mean_ms",
            "transformer_mean_ms",
            "transformer_median_ms",
            "transformer_p95_ms",
            "transformer_p99_ms",
            "prediction_end_to_end_mean_ms",
            "prediction_end_to_end_median_ms",
            "prediction_end_to_end_p95_ms",
            "prediction_end_to_end_p99_ms",
            "prediction_end_to_end_min_ms",
            "prediction_end_to_end_max_ms",
            "prediction_end_to_end_std_ms",
        ],
    }

    with open(
        SCHEMA,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            schema,
            f,
            indent=2,
        )

    manifest = {
        "phase": "5.13",
        "title":
            "Latency and Real-Time Behavior",
        "status":
            "PASS"
            if all(checks.values())
            else "FAIL",
        "input": str(INPUT_CSV),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256":
            checkpoint_sha256,
        "feature_pipeline":
            "Phase 5.12 validated pipeline",
        "feature_count":
            FEATURE_COUNT,
        "parameter_count":
            parameter_count,
        "device":
            str(DEVICE),
        "python_version":
            platform.python_version(),
        "torch_version":
            torch.__version__,
        "platform":
            platform.platform(),
        "sequence_length":
            OBSERVATION_STEPS,
        "warmup_windows":
            WARMUP_WINDOWS,
        "measured_windows_per_scenario":
            MEASURED_WINDOWS,
        "repeats":
            REPEATS,
        "prediction_horizons_seconds": {
            "Y3":
                Y3_HORIZON_SECONDS,
            "Y6":
                Y6_HORIZON_SECONDS,
        },
        "aggregate": {
            "feature_build_mean_of_scenario_means_ms":
                aggregate_feature["mean_ms"],
            "feature_window_mean_of_scenario_means_ms":
                aggregate_feature_window["mean_ms"],
            "transformer_mean_of_scenario_means_ms":
                aggregate_transformer["mean_ms"],
            "end_to_end_mean_of_scenario_means_ms":
                aggregate_e2e["mean_ms"],
            "maximum_scenario_end_to_end_mean_ms":
                maximum_e2e_mean,
            "maximum_scenario_end_to_end_p99_ms":
                maximum_e2e_p99,
            "maximum_scenario_transformer_p99_ms":
                maximum_transformer_p99,
            "y3_latency_fraction":
                y3_fraction,
            "y6_latency_fraction":
                y6_fraction,
        },
        "scenario_results":
            scenario_results,
        "validation_checks":
            checks,
        "research_boundaries": {
            "laboratory_only": True,
            "physical_vehicle": False,
            "direct_actuation": False,
            "model_retraining": False,
            "live_cortex_xdr": False,
            "production_realtime_compliance":
                False,
            "oem_timing_requirement_claim":
                False,
        },
        "outputs": {
            "csv": str(OUTPUT_CSV),
            "jsonl": str(OUTPUT_JSONL),
            "schema": str(SCHEMA),
            "manifest": str(MANIFEST),
        },
    }

    with open(
        MANIFEST,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print(
        "=== PHASE 5.13 RESULTS ==="
    )
    print()

    passed = 0

    for name, result in checks.items():
        if result:
            print(
                f"  PASS {name}"
            )
            passed += 1
        else:
            print(
                f"  FAIL {name}"
            )

    print()
    print(
        f"Validation: "
        f"{passed}/{len(checks)} checks passed"
    )

    print()
    print("Aggregate latency:")
    print(
        "  Feature build mean: "
        f"{aggregate_feature['mean_ms']:.3f} ms"
    )
    print(
        "  Feature + window mean: "
        f"{aggregate_feature_window['mean_ms']:.3f} ms"
    )
    print(
        "  Transformer mean: "
        f"{aggregate_transformer['mean_ms']:.3f} ms"
    )
    print(
        "  End-to-end mean: "
        f"{aggregate_e2e['mean_ms']:.3f} ms"
    )
    print(
        "  Maximum scenario P99: "
        f"{maximum_e2e_p99:.3f} ms"
    )
    print(
        "  Maximum Transformer P99: "
        f"{maximum_transformer_p99:.3f} ms"
    )
    print(
        "  Y3 latency fraction: "
        f"{y3_fraction:.8f}"
    )
    print(
        "  Y6 latency fraction: "
        f"{y6_fraction:.8f}"
    )

    print()
    print(
        f"CSV: {OUTPUT_CSV}"
    )
    print(
        f"JSONL: {OUTPUT_JSONL}"
    )
    print(
        f"Schema: {SCHEMA}"
    )
    print(
        f"Manifest: {MANIFEST}"
    )

    final_status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    print()
    print(
        f"STATUS: {final_status}"
    )

    if final_status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()