from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]

SEQ = (
    ROOT / "data" / "processed" / "modeling"
    / "sequences" / "v2_forecasting"
)

print("=" * 72)
print("PROJECT DATA CONTRACT PREFLIGHT")
print("=" * 72)

required = {
    "X_train":
        SEQ / "X_train_v2_forecasting.npy",
    "X_validation":
        SEQ / "X_validation_v2_forecasting.npy",
    "X_test":
        SEQ / "X_test_v2_forecasting.npy",
    "Y3_train":
        SEQ / "y_3step_train_v2_forecasting.npy",
    "Y3_validation":
        SEQ / "y_3step_validation_v2_forecasting.npy",
    "Y3_test":
        SEQ / "y_3step_test_v2_forecasting.npy",
    "Y6_train":
        SEQ / "y_6step_train_v2_forecasting.npy",
    "Y6_validation":
        SEQ / "y_6step_validation_v2_forecasting.npy",
    "Y6_test":
        SEQ / "y_6step_test_v2_forecasting.npy",
    "scenario_train":
        SEQ / "scenario_ids_train_v2_forecasting.npy",
    "scenario_validation":
        SEQ / "scenario_ids_validation_v2_forecasting.npy",
    "scenario_test":
        SEQ / "scenario_ids_test_v2_forecasting.npy",
    "origin_train":
        SEQ / "prediction_origins_train_v2_forecasting.npy",
    "origin_validation":
        SEQ / "prediction_origins_validation_v2_forecasting.npy",
    "origin_test":
        SEQ / "prediction_origins_test_v2_forecasting.npy",
    "metadata":
        SEQ / "crss_2024_forecasting_tensor_metadata.json",
}

failed = False

print("\n=== FILE CONTRACT ===")

for name, path in required.items():
    exists = path.exists()

    print(
        f"{'PASS' if exists else 'FAIL':4} "
        f"{name:22} {path.name}"
    )

    if not exists:
        failed = True

if failed:
    raise SystemExit(
        "DATA CONTRACT FAILED: required artifact missing."
    )

# ---------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------

print("\n=== SHAPE CONTRACT ===")

arrays = {}

for name, path in required.items():
    if path.suffix == ".npy":
        arrays[name] = np.load(path, mmap_mode="r")

expected = {
    "X_train": (444052, 12, 63),
    "X_validation": (94584, 12, 63),
    "X_test": (95851, 12, 63),

    "Y3_train": (444052,),
    "Y3_validation": (94584,),
    "Y3_test": (95851,),

    "Y6_train": (444052,),
    "Y6_validation": (94584,),
    "Y6_test": (95851,),

    "scenario_train": (444052,),
    "scenario_validation": (94584,),
    "scenario_test": (95851,),

    "origin_train": (444052,),
    "origin_validation": (94584,),
    "origin_test": (95851,),
}

for name, shape in expected.items():
    actual = arrays[name].shape
    ok = actual == shape

    print(
        f"{'PASS' if ok else 'FAIL':4} "
        f"{name:22} expected={shape} actual={actual}"
    )

    if not ok:
        failed = True

# ---------------------------------------------------------------
# Identifier contract
# ---------------------------------------------------------------

print("\n=== IDENTIFIER CONTRACT ===")

for name in [
    "scenario_train",
    "scenario_validation",
    "scenario_test",
]:
    arr = arrays[name]

    ok = arr.dtype.kind in {"U", "S", "O"}

    print(
        f"{'PASS' if ok else 'FAIL':4} "
        f"{name:22} dtype={arr.dtype}"
    )

    if not ok:
        failed = True

# ---------------------------------------------------------------
# Origin contract
# ---------------------------------------------------------------

print("\n=== ORIGIN CONTRACT ===")

for name in [
    "origin_train",
    "origin_validation",
    "origin_test",
]:
    arr = arrays[name]

    values = np.unique(arr)

    ok = len(values) == 7 and set(values.tolist()) == set(range(11, 18))

    print(
        f"{'PASS' if ok else 'FAIL':4} "
        f"{name:22} origins={values.tolist()}"
    )

    if not ok:
        failed = True

# ---------------------------------------------------------------
# Metadata contract
# ---------------------------------------------------------------

print("\n=== METADATA CONTRACT ===")

with open(required["metadata"], "r", encoding="utf-8") as f:
    metadata = json.load(f)

feature_count = metadata["feature_count"]

ok = feature_count == 63

print(
    f"{'PASS' if ok else 'FAIL':4} "
    f"metadata feature_count={feature_count}"
)

if not ok:
    failed = True

# ---------------------------------------------------------------

print("\n" + "=" * 72)

if failed:
    print("STATUS: FAIL")
    raise SystemExit(1)

print("STATUS: PASS")
print("All tensor, identifier, origin and metadata contracts are valid.")
