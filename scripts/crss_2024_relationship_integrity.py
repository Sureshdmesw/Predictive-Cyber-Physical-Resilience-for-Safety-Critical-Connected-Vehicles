import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/nhtsa/CRSS/downloads/CRSS2024CSV"
OUT = ROOT / "data/schemas/nhtsa/verification/crss_2024_relationship_integrity.csv"

print("=" * 80)
print("CRSS 2024 RELATIONSHIP INTEGRITY ANALYSIS")
print("=" * 80)

accident = pd.read_csv(
    DATA / "accident.csv",
    usecols=["CASENUM", "VE_TOTAL"],
    low_memory=False
)

vehicle = pd.read_csv(
    DATA / "vehicle.csv",
    usecols=[
        "CASENUM",
        "VEH_NO",
        "NUMOCCS",
        "ROLLOVER",
        "IMPACT1",
        "FIRE_EXP"
    ],
    low_memory=False
)

person = pd.read_csv(
    DATA / "person.csv",
    usecols=[
        "CASENUM",
        "VEH_NO",
        "PER_NO",
        "ROLLOVER",
        "IMPACT1",
        "FIRE_EXP"
    ],
    low_memory=False
)

results = []


def add_result(test, status, value, details=""):
    results.append({
        "test": test,
        "status": status,
        "value": value,
        "details": details
    })


# ------------------------------------------------------------------
# 1. Primary-key uniqueness
# ------------------------------------------------------------------

accident_dup = accident["CASENUM"].duplicated().sum()

vehicle_key = vehicle[["CASENUM", "VEH_NO"]].astype(str).agg("|".join, axis=1)
vehicle_dup = vehicle_key.duplicated().sum()

person_key = person[
    ["CASENUM", "VEH_NO", "PER_NO"]
].astype(str).agg("|".join, axis=1)
person_dup = person_key.duplicated().sum()

add_result(
    "accident CASENUM uniqueness",
    "PASS" if accident_dup == 0 else "FAIL",
    accident_dup,
    "Expected 0 duplicate crash identifiers"
)

add_result(
    "vehicle CASENUM+VEH_NO uniqueness",
    "PASS" if vehicle_dup == 0 else "FAIL",
    vehicle_dup,
    "Expected 0 duplicate vehicle identifiers"
)

add_result(
    "person CASENUM+VEH_NO+PER_NO uniqueness",
    "PASS" if person_dup == 0 else "FAIL",
    person_dup,
    "Expected 0 duplicate person identifiers"
)


# ------------------------------------------------------------------
# 2. Referential integrity
# ------------------------------------------------------------------

accident_ids = set(accident["CASENUM"])

vehicle_orphans = vehicle[
    ~vehicle["CASENUM"].isin(accident_ids)
]

vehicle_orphan_count = len(vehicle_orphans)

add_result(
    "vehicle -> accident referential integrity",
    "PASS" if vehicle_orphan_count == 0 else "FAIL",
    vehicle_orphan_count,
    "Vehicles whose CASENUM does not exist in accident"
)


vehicle_keys = set(
    zip(vehicle["CASENUM"], vehicle["VEH_NO"])
)

person_pairs = list(
    zip(person["CASENUM"], person["VEH_NO"])
)

person_orphan_mask = [
    pair not in vehicle_keys
    for pair in person_pairs
]

person_orphan_count = sum(person_orphan_mask)

add_result(
    "person -> vehicle referential integrity",
    "PASS" if person_orphan_count == 0 else "FAIL",
    person_orphan_count,
    "Persons whose CASENUM+VEH_NO does not exist in vehicle"
)


# ------------------------------------------------------------------
# 3. Vehicle counts per crash
# ------------------------------------------------------------------

vehicle_counts = (
    vehicle.groupby("CASENUM")
    .size()
    .rename("actual_vehicle_rows")
)

crash_counts = accident[
    ["CASENUM", "VE_TOTAL"]
].merge(
    vehicle_counts,
    on="CASENUM",
    how="left"
)

crash_counts["actual_vehicle_rows"] = (
    crash_counts["actual_vehicle_rows"]
    .fillna(0)
)

vehicle_count_mismatch = crash_counts[
    crash_counts["VE_TOTAL"] != crash_counts["actual_vehicle_rows"]
]

add_result(
    "accident.VE_TOTAL vs vehicle row count",
    "PASS" if len(vehicle_count_mismatch) == 0 else "REVIEW",
    len(vehicle_count_mismatch),
    "Crashes where VE_TOTAL differs from vehicle-table row count"
)


# ------------------------------------------------------------------
# 4. Occupant counts per vehicle
# ------------------------------------------------------------------

person_counts = (
    person.groupby(
        ["CASENUM", "VEH_NO"]
    )
    .size()
    .rename("actual_person_rows")
    .reset_index()
)

vehicle_occ = vehicle[
    ["CASENUM", "VEH_NO", "NUMOCCS"]
].merge(
    person_counts,
    on=["CASENUM", "VEH_NO"],
    how="left"
)

vehicle_occ["actual_person_rows"] = (
    vehicle_occ["actual_person_rows"]
    .fillna(0)
)

occupant_count_mismatch = vehicle_occ[
    vehicle_occ["NUMOCCS"] != vehicle_occ["actual_person_rows"]
]

add_result(
    "vehicle.NUMOCCS vs person row count",
    "PASS" if len(occupant_count_mismatch) == 0 else "REVIEW",
    len(occupant_count_mismatch),
    "Vehicles where NUMOCCS differs from person-table row count"
)


# ------------------------------------------------------------------
# 5. Person-level rollover / impact / fire structural analysis
# ------------------------------------------------------------------

person["vehicle_key"] = (
    person["CASENUM"].astype(str)
    + "|"
    + person["VEH_NO"].astype(str)
)

vehicle_structural = vehicle[
    [
        "CASENUM",
        "VEH_NO",
        "ROLLOVER",
        "IMPACT1",
        "FIRE_EXP"
    ]
].copy()

vehicle_structural = vehicle_structural.rename(
    columns={
        "ROLLOVER": "vehicle_ROLLOVER",
        "IMPACT1": "vehicle_IMPACT1",
        "FIRE_EXP": "vehicle_FIRE_EXP"
    }
)

person_structural = person.merge(
    vehicle_structural,
    on=["CASENUM", "VEH_NO"],
    how="left"
)

# Exact NaN pattern
for field in ["ROLLOVER", "IMPACT1", "FIRE_EXP"]:
    nan_mask = person[field].isna()

    count = nan_mask.sum()

    if count > 0:
        vehicle_values_available = (
            person_structural.loc[
                nan_mask,
                f"vehicle_{field}"
            ].notna().sum()
        )

        add_result(
            f"person.{field} NaN structural linkage",
            "REVIEW",
            count,
            f"{vehicle_values_available} of {count} NaN person records have vehicle-level {field} available"
        )
    else:
        add_result(
            f"person.{field} NaN structural linkage",
            "PASS",
            0,
            "No pandas NaN values"
        )


# ------------------------------------------------------------------
# 6. Person/vehicle value consistency where both exist
# ------------------------------------------------------------------

for field in ["ROLLOVER", "IMPACT1", "FIRE_EXP"]:

    comparable = person_structural[
        person_structural[field].notna()
        & person_structural[f"vehicle_{field}"].notna()
    ]

    mismatch = comparable[
        comparable[field] != comparable[f"vehicle_{field}"]
    ]

    add_result(
        f"person.{field} vs vehicle.{field} consistency",
        "REVIEW" if len(mismatch) > 0 else "PASS",
        len(mismatch),
        f"Compared {len(comparable)} person-vehicle records"
    )


# ------------------------------------------------------------------
# 7. Summary
# ------------------------------------------------------------------

result_df = pd.DataFrame(results)

OUT.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(
    OUT,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(result_df.to_string(index=False))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Total tests: {len(result_df)}")
print(f"PASS: {(result_df['status'] == 'PASS').sum()}")
print(f"REVIEW: {(result_df['status'] == 'REVIEW').sum()}")
print(f"FAIL: {(result_df['status'] == 'FAIL').sum()}")

print("\nOutput:")
print(OUT)
