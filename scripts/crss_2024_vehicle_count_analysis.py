import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/nhtsa/CRSS/downloads/CRSS2024CSV"

accident = pd.read_csv(
    DATA / "accident.csv",
    usecols=[
        "CASENUM",
        "VE_TOTAL",
        "VE_FORMS",
        "PVH_INVL",
        "PERMVIT",
        "PERNOTMVIT"
    ],
    low_memory=False
)

vehicle = pd.read_csv(
    DATA / "vehicle.csv",
    usecols=[
        "CASENUM",
        "VEH_NO",
        "UNITTYPE"
    ],
    low_memory=False
)

vehicle_counts = (
    vehicle.groupby("CASENUM")
    .size()
    .rename("vehicle_rows")
)

check = accident.merge(
    vehicle_counts,
    on="CASENUM",
    how="left"
)

check["vehicle_rows"] = check["vehicle_rows"].fillna(0)

mismatch = check[
    check["VE_TOTAL"] != check["vehicle_rows"]
].copy()

mismatch["difference"] = (
    mismatch["VE_TOTAL"] -
    mismatch["vehicle_rows"]
)

print("=" * 80)
print("CRSS 2024 VE_TOTAL VS VEHICLE RECORD ANALYSIS")
print("=" * 80)

print(f"\nTotal crashes: {len(accident):,}")
print(f"Mismatched crashes: {len(mismatch):,}")

print("\nDifference distribution:")
print(
    mismatch["difference"]
    .value_counts()
    .sort_index()
    .head(30)
)

print("\n" + "=" * 80)
print("MISMATCH SUMMARY")
print("=" * 80)

print(
    mismatch[
        [
            "VE_TOTAL",
            "VE_FORMS",
            "PVH_INVL",
            "PERMVIT",
            "PERNOTMVIT",
            "vehicle_rows"
        ]
    ]
    .describe()
    .to_string()
)

print("\n" + "=" * 80)
print("VE_TOTAL vs VEHICLE ROWS")
print("=" * 80)

print(
    mismatch[
        [
            "CASENUM",
            "VE_TOTAL",
            "VE_FORMS",
            "PVH_INVL",
            "vehicle_rows",
            "difference"
        ]
    ]
    .head(30)
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("UNITTYPE DISTRIBUTION FOR VEHICLE TABLE")
print("=" * 80)

print(
    vehicle["UNITTYPE"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\nDone.")
