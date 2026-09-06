import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/nhtsa/CRSS/downloads/CRSS2024CSV"

vehicle = pd.read_csv(
    DATA / "vehicle.csv",
    usecols=["CASENUM", "VEH_NO", "NUMOCCS"],
    low_memory=False
)

person = pd.read_csv(
    DATA / "person.csv",
    usecols=["CASENUM", "VEH_NO", "PER_NO", "ROLLOVER", "IMPACT1", "FIRE_EXP"],
    low_memory=False
)

print("=" * 80)
print("PERSON -> VEHICLE RELATIONSHIP SUMMARY")
print("=" * 80)

print(f"Vehicle rows: {len(vehicle):,}")
print(f"Person rows:  {len(person):,}")

vehicle_keys = set(zip(vehicle.CASENUM, vehicle.VEH_NO))

person_keys = set(zip(person.CASENUM, person.VEH_NO))

orphans = person_keys - vehicle_keys

print(f"Unique vehicle keys: {len(vehicle_keys):,}")
print(f"Unique person vehicle references: {len(person_keys):,}")
print(f"Orphan person vehicle references: {len(orphans):,}")

print("\nPerson NaNs:")
for col in ["ROLLOVER", "IMPACT1", "FIRE_EXP"]:
    print(f"{col:10s}: {person[col].isna().sum():,}")

print("\nNaN records by VEH_NO:")
nan_any = person[
    person[["ROLLOVER", "IMPACT1", "FIRE_EXP"]]
    .isna()
    .any(axis=1)
]

print(
    nan_any.groupby("VEH_NO")
    .size()
    .sort_values(ascending=False)
    .head(20)
)

print("\nNaN records by PER_TYP:")
# PER_TYP not loaded intentionally; skip if unavailable.

print("\nDone.")
