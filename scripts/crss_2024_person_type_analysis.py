import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/nhtsa/CRSS/downloads/CRSS2024CSV"

person = pd.read_csv(
    DATA / "person.csv",
    usecols=[
        "CASENUM",
        "VEH_NO",
        "PER_NO",
        "PER_TYP",
        "PER_TYPNAME",
        "ROLLOVER",
        "IMPACT1",
        "FIRE_EXP"
    ],
    low_memory=False
)

print("=" * 80)
print("CRSS 2024 PERSON TYPE / VEHICLE NUMBER ANALYSIS")
print("=" * 80)

print("\nPERSON TYPE DISTRIBUTION")
print("-" * 80)

print(
    person.groupby(
        ["PER_TYP", "PER_TYPNAME"],
        dropna=False
    )
    .size()
    .reset_index(name="records")
    .sort_values("records", ascending=False)
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("VEH_NO = 0 BY PERSON TYPE")
print("=" * 80)

veh0 = person[person["VEH_NO"] == 0]

print(f"VEH_NO = 0 records: {len(veh0):,}")

print(
    veh0.groupby(
        ["PER_TYP", "PER_TYPNAME"],
        dropna=False
    )
    .size()
    .reset_index(name="records")
    .sort_values("records", ascending=False)
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("PERSON STRUCTURAL NaNs BY PERSON TYPE")
print("=" * 80)

nan_mask = person[
    ["ROLLOVER", "IMPACT1", "FIRE_EXP"]
].isna().any(axis=1)

nan_people = person[nan_mask]

print(f"Records with structural NaNs: {len(nan_people):,}")

print(
    nan_people.groupby(
        ["VEH_NO", "PER_TYP", "PER_TYPNAME"],
        dropna=False
    )
    .size()
    .reset_index(name="records")
    .sort_values("records", ascending=False)
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("STRUCTURAL NaN PATTERN")
print("=" * 80)

print(
    nan_people[
        ["ROLLOVER", "IMPACT1", "FIRE_EXP"]
    ]
    .isna()
    .value_counts()
)

print("\nDone.")
