import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/nhtsa/CRSS/downloads/CRSS2024CSV"

accident = pd.read_csv(
    DATA / "accident.csv",
    usecols=["CASENUM", "VE_TOTAL", "VE_FORMS", "PVH_INVL"],
    low_memory=False
)

vehicle = pd.read_csv(
    DATA / "vehicle.csv",
    usecols=["CASENUM", "VEH_NO", "UNITTYPE"],
    low_memory=False
)

vehicle_forms = (
    vehicle[vehicle["UNITTYPE"] == 1]
    .groupby("CASENUM")
    .size()
    .rename("vehicle_form_rows")
)

check = accident.merge(
    vehicle_forms,
    on="CASENUM",
    how="left"
)

check["vehicle_form_rows"] = (
    check["vehicle_form_rows"].fillna(0)
)

mismatch = check[
    check["VE_FORMS"] != check["vehicle_form_rows"]
]

print("=" * 80)
print("VE_FORMS VS VEHICLE UNITTYPE=1")
print("=" * 80)

print(f"Total crashes: {len(check):,}")
print(f"VE_FORMS mismatches: {len(mismatch):,}")

if len(mismatch) > 0:
    print("\nMismatch examples:")
    print(
        mismatch.head(30).to_string(index=False)
    )
else:
    print("\nRESULT: PERFECT MATCH")
    print("VE_FORMS corresponds exactly to vehicle UNITTYPE=1 rows.")

print("\nDone.")
