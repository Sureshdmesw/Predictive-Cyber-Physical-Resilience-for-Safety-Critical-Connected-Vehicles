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
        "LOCATION",
        "LOCATIONNAME",
        "HARM_EV",
        "HARM_EVNAME"
    ],
    low_memory=False
)

vehicle = pd.read_csv(
    DATA / "vehicle.csv",
    usecols=[
        "CASENUM",
        "VEH_NO",
        "UNITTYPE",
        "UNITTYPENAME",
        "HARM_EV",
        "HARM_EVNAME",
        "HIT_RUN"
    ],
    low_memory=False
)

vehicle_keys = set(
    zip(vehicle["CASENUM"], vehicle["VEH_NO"])
)

p3 = person[
    person["PER_TYP"] == 3
].copy()

p3["vehicle_exists"] = [
    (c, v) in vehicle_keys
    for c, v in zip(p3["CASENUM"], p3["VEH_NO"])
]

print("=" * 80)
print("CRSS 2024 PER_TYP=3 INVESTIGATION")
print("=" * 80)

print(f"\nTotal PER_TYP=3 records: {len(p3):,}")
print(
    f"PER_TYP=3 with vehicle record: "
    f"{p3['vehicle_exists'].sum():,}"
)
print(
    f"PER_TYP=3 without vehicle record: "
    f"{(~p3['vehicle_exists']).sum():,}"
)

orphans = p3[
    ~p3["vehicle_exists"]
].copy()

print("\n" + "=" * 80)
print("ORPHAN PER_TYP=3 DISTRIBUTION")
print("=" * 80)

print(
    orphans[
        [
            "LOCATION",
            "LOCATIONNAME",
            "HARM_EV",
            "HARM_EVNAME"
        ]
    ]
    .value_counts(dropna=False)
    .head(30)
    .to_string()
)

print("\n" + "=" * 80)
print("ORPHAN VEH_NO DISTRIBUTION")
print("=" * 80)

print(
    orphans["VEH_NO"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\n" + "=" * 80)
print("SAMPLE ORPHAN RECORDS")
print("=" * 80)

print(
    orphans[
        [
            "CASENUM",
            "VEH_NO",
            "PER_NO",
            "PER_TYP",
            "PER_TYPNAME",
            "LOCATION",
            "LOCATIONNAME",
            "HARM_EV",
            "HARM_EVNAME"
        ]
    ]
    .head(50)
    .to_string(index=False)
)

print("\nDone.")
