import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SEQ = ROOT / "data" / "processed" / "modeling" / "sequences" / "v2_forecasting"
READINESS = ROOT / "experiments" / "modeling" / "v2" / "crss_2024_v2_model_readiness.json"

features_to_check = [
    "gateway_present",
    "physical_state_completeness",
    "roadway_complexity_index",
    "vehicle_state_observation_count",
]

metadata = json.loads(
    (SEQ / "crss_2024_forecasting_tensor_metadata.json").read_text(
        encoding="utf-8"
    )
)

features = metadata["features"]

indices = {
    feature: features.index(feature)
    for feature in features_to_check
}

print("=" * 100)
print("CRSS 2024 — PHASE 4.7 ZERO-VARIANCE FEATURE INVESTIGATION")
print("=" * 100)

print()
print("FEATURE INDICES")
print("-" * 100)

for feature, idx in indices.items():
    print(f"{idx:02d}  {feature}")

for split in ["train", "validation", "test"]:

    path = SEQ / f"X_{split}_v2_forecasting.npy"

    X = np.load(path, mmap_mode="r")

    print()
    print("=" * 100)
    print(split.upper())
    print("=" * 100)

    for feature, idx in indices.items():

        values = np.asarray(X[:, :, idx], dtype=np.float64)

        unique, counts = np.unique(
            values,
            return_counts=True
        )

        print()
        print(f"{feature}")
        print("-" * 80)
        print(f"min       : {values.min()}")
        print(f"max       : {values.max()}")
        print(f"mean      : {values.mean()}")
        print(f"std       : {values.std()}")
        print(f"unique    : {len(unique)}")

        if len(unique) <= 20:
            print("values    :")

            for value, count in zip(unique, counts):
                print(
                    f"    {value!r:<20} "
                    f"count={count:,} "
                    f"pct={count / values.size:.6%}"
                )
        else:
            print(
                f"unique range too large to display "
                f"({len(unique)} values)"
            )

print()
print("=" * 100)
print("END PHASE 4.7 INVESTIGATION")
print("=" * 100)
