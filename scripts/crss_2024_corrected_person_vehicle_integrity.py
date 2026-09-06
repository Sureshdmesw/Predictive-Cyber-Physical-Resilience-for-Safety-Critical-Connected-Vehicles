import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/nhtsa/CRSS/downloads/CRSS2024CSV"

vehicle = pd.read_csv(
    DATA / "vehicle.csv",
    usecols=["CASENUM", "VEH_NO"],
    low_memory=False
)

person = pd.read_csv(
    DATA / "person.csv",
    usecols=[
        "CASENUM",
        "VEH_NO",
        "PER_NO",
        "PER_TYP",
        "PER_TYPNAME"
    ],
    low_memory=False
)

vehicle_keys = set(
    zip(vehicle["CASENUM"], vehicle["VEH_NO"])
)

print("=" * 80)
print("CORRECTED PERSON -> VEHICLE REFERENTIAL INTEGRITY")
print("=" * 80)

print(f"\nTotal person records: {len(person):,}")

print("\nVEH_NO distribution:")
print(
    person["VEH_NO"]
    .value_counts(dropna=False)
    .sort_index()
    .head(20)
)

print("\nPerson records with VEH_NO = 0:")
print((person["VEH_NO"] == 0).sum())

# Treat VEH_NO=0 separately.
non_vehicle = person[person["VEH_NO"] == 0]
vehicle_linked = person[person["VEH_NO"] != 0]

orphans = [
    pair not in vehicle_keys
    for pair in zip(
        vehicle_linked["CASENUM"],
        vehicle_linked["VEH_NO"]
    )
]

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print(f"Non-vehicle / VEH_NO=0 records: {len(non_vehicle):,}")
print(f"Vehicle-associated person records: {len(vehicle_linked):,}")
print(f"Vehicle-associated orphan references: {sum(orphans):,}")

if sum(orphans) == 0:
    print("\nRESULT: PASS")
else:
    print("\nRESULT: REVIEW")

    orphan_df = vehicle_linked.loc[
        pd.Series(orphans, index=vehicle_linked.index)
    ]

    print("\nOrphan records by PER_TYP:")
    print(
        orphan_df.groupby(
            ["PER_TYP", "PER_TYPNAME"],
            dropna=False
        )
        .size()
        .reset_index(name="records")
        .sort_values("records", ascending=False)
        .to_string(index=False)
    )

print("\nDone.")
