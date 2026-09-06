import numpy as np
from pathlib import Path

p = Path("data/processed/modeling/sequences/v2_forecasting")

origin_names = [
    "prediction_origins_train_v2_forecasting.npy",
    "prediction_origins_validation_v2_forecasting.npy",
    "prediction_origins_test_v2_forecasting.npy",
]

scenario_names = [
    "scenario_ids_train_v2_forecasting.npy",
    "scenario_ids_validation_v2_forecasting.npy",
    "scenario_ids_test_v2_forecasting.npy",
]

print("=" * 72)
print("PHASE 4.15 — ID / ORIGIN SEMANTICS INVESTIGATION")
print("=" * 72)

print("\n=== AVAILABLE ID / ORIGIN FILES ===")

for name in origin_names + scenario_names:
    path = p / name
    if path.exists():
        arr = np.load(path, mmap_mode="r")
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    else:
        print(f"{name}: MISSING")

print("\n=== ORIGIN UNIQUE VALUES ===")

origins = []

for name in origin_names:
    path = p / name
    arr = np.load(path, mmap_mode="r")
    origins.append(np.asarray(arr))
    unique = np.unique(arr)

    print(f"\n{name}")
    print(f"unique count: {len(unique)}")
    print(f"first values: {unique[:30]}")
    print(f"min: {unique.min()}")
    print(f"max: {unique.max()}")

print("\n=== ORIGIN OVERLAP VALUES ===")

train_o, val_o, test_o = origins

tv = np.intersect1d(train_o, val_o)
tt = np.intersect1d(train_o, test_o)
vt = np.intersect1d(val_o, test_o)

print(f"train ∩ validation: {len(tv)}")
print(f"values: {tv}")

print(f"train ∩ test: {len(tt)}")
print(f"values: {tt}")

print(f"validation ∩ test: {len(vt)}")
print(f"values: {vt}")

print("\n=== SCENARIO ID FILES ===")

scenario_arrays = []
scenario_available = True

for name in scenario_names:
    path = p / name

    if not path.exists():
        print(f"{name}: MISSING")
        scenario_available = False
        continue

    arr = np.load(path, mmap_mode="r")
    scenario_arrays.append(np.asarray(arr))

    unique = np.unique(arr)

    print(f"\n{name}")
    print(f"shape: {arr.shape}")
    print(f"dtype: {arr.dtype}")
    print(f"unique scenarios: {len(unique)}")
    print(f"min: {unique.min()}")
    print(f"max: {unique.max()}")
    print(f"first values: {unique[:20]}")

if scenario_available and len(scenario_arrays) == 3:
    train_s, val_s, test_s = scenario_arrays

    print("\n=== SCENARIO ID OVERLAP ===")

    tvs = np.intersect1d(train_s, val_s)
    tts = np.intersect1d(train_s, test_s)
    vts = np.intersect1d(val_s, test_s)

    print(f"train / validation overlap: {len(tvs)}")
    if len(tvs):
        print(f"values: {tvs[:50]}")

    print(f"train / test overlap: {len(tts)}")
    if len(tts):
        print(f"values: {tts[:50]}")

    print(f"validation / test overlap: {len(vts)}")
    if len(vts):
        print(f"values: {vts[:50]}")

    print("\n=== SCENARIO ID RANGES ===")

    print(
        f"train:      min={train_s.min()} "
        f"max={train_s.max()} "
        f"unique={len(np.unique(train_s))}"
    )

    print(
        f"validation: min={val_s.min()} "
        f"max={val_s.max()} "
        f"unique={len(np.unique(val_s))}"
    )

    print(
        f"test:       min={test_s.min()} "
        f"max={test_s.max()} "
        f"unique={len(np.unique(test_s))}"
    )

    print("\n=== SCENARIO-TO-ORIGIN RELATIONSHIP ===")

    for split_name, s, o in [
        ("train", train_s, train_o),
        ("validation", val_s, val_o),
        ("test", test_s, test_o),
    ]:
        print(f"\n{split_name}")

        print("first 20 scenario IDs:")
        print(s[:20])

        print("first 20 origins:")
        print(o[:20])

        pairs = list(zip(s[:30], o[:30]))
        print("first 30 scenario/origin pairs:")
        print(pairs)

        print("origin values per first scenarios:")

        for scenario_id in np.unique(s[:100])[:10]:
            mask = s == scenario_id
            vals = np.unique(o[mask])

            print(
                f"  scenario {scenario_id}: "
                f"count={mask.sum()}, "
                f"origins={vals[:30]}"
            )

else:
    print("\nSCENARIO ID FILES ARE NOT AVAILABLE.")
    print("Do not classify the 7 origin overlaps as leakage yet.")

print("\n" + "=" * 72)
print("INVESTIGATION COMPLETE")
print("=" * 72)
