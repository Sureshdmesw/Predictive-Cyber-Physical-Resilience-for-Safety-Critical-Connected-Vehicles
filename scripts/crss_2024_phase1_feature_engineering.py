from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw" / "nhtsa" / "CRSS" / "downloads" / "CRSS2024CSV"
OUT = ROOT / "data" / "processed" / "crss_2024"
SCHEMA = ROOT / "data" / "schemas" / "nhtsa" / "crss_2024_canonical_schema.json"
FEATURE_DICT = ROOT / "data" / "schemas" / "nhtsa" / "verification" / "crss_2024_final_feature_dictionary_v2.csv"

OUT.mkdir(parents=True, exist_ok=True)

def read_csv(name):
    path = RAW / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CRSS table: {path}")
    return pd.read_csv(path, low_memory=False)

def clean_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------------------------------------------------------
# Load released CRSS 2024 source tables
# ------------------------------------------------------------------

accident = read_csv("accident")
vehicle = read_csv("vehicle")
person = read_csv("person")
vision = read_csv("vision")
weather = read_csv("weather")

# ------------------------------------------------------------------
# Normalize identifiers
# ------------------------------------------------------------------

for df in [accident, vehicle, person, vision, weather]:
    if "CASENUM" in df.columns:
        df["CASENUM"] = df["CASENUM"].astype(str).str.strip()

for df in [vehicle, person]:
    if "VEH_NO" in df.columns:
        df["VEH_NO"] = pd.to_numeric(df["VEH_NO"], errors="coerce")

if "PER_NO" in person.columns:
    person["PER_NO"] = pd.to_numeric(person["PER_NO"], errors="coerce")

# ------------------------------------------------------------------
# Vehicle-level canonical dataset
# UNITTYPE=1 is the motor-vehicle-in-transport population.
# ------------------------------------------------------------------

vehicle_cols = [
    "CASENUM", "VEH_NO",
    "ACC_TYPE", "SPEEDREL", "TRAV_SP", "VALIGN",
    "VNUM_LAN", "VPROFILE", "VSPD_LIM", "VSURCOND",
    "VTRAFCON", "VTRAFWAY",
    "HARM_EV", "MAN_COLL", "IMPACT1",
    "ROLLOVER", "DEFORMED", "FIRE_EXP"
]

vehicle_features = vehicle[
    [c for c in vehicle_cols if c in vehicle.columns]
].copy()

if "UNITTYPE" in vehicle.columns:
    vehicle_features = vehicle_features[
        vehicle.loc[vehicle_features.index, "UNITTYPE"].eq(1)
    ].copy()

# ------------------------------------------------------------------
# Numeric conversion
# ------------------------------------------------------------------

vehicle_numeric = [
    "ACC_TYPE", "SPEEDREL", "TRAV_SP", "VALIGN",
    "VNUM_LAN", "VPROFILE", "VSPD_LIM", "VSURCOND",
    "VTRAFCON", "VTRAFWAY", "HARM_EV", "MAN_COLL",
    "IMPACT1", "ROLLOVER", "DEFORMED", "FIRE_EXP"
]

vehicle_features = clean_numeric(vehicle_features, vehicle_numeric)

# ------------------------------------------------------------------
# Vehicle cyber-physical context features
# ------------------------------------------------------------------

vehicle_features["speed_limit_deviation"] = np.where(
    vehicle_features["TRAV_SP"].notna() &
    vehicle_features["VSPD_LIM"].notna(),
    vehicle_features["TRAV_SP"] - vehicle_features["VSPD_LIM"],
    np.nan
)

vehicle_features["speed_limit_deviation_abs"] = (
    vehicle_features["speed_limit_deviation"].abs()
)

vehicle_features["speed_over_limit_flag"] = np.where(
    vehicle_features["speed_limit_deviation"].notna(),
    (vehicle_features["speed_limit_deviation"] > 0).astype("int8"),
    np.nan
)

vehicle_features["rollover_flag"] = np.where(
    vehicle_features["ROLLOVER"].isin([1, 2, 3]),
    1,
    np.where(vehicle_features["ROLLOVER"].notna(), 0, np.nan)
)

vehicle_features["fire_explosion_flag"] = np.where(
    vehicle_features["FIRE_EXP"].eq(1),
    1,
    np.where(vehicle_features["FIRE_EXP"].notna(), 0, np.nan)
)

vehicle_features["vehicle_damage_flag"] = np.where(
    vehicle_features["DEFORMED"].notna(),
    (vehicle_features["DEFORMED"] > 0).astype("int8"),
    np.nan
)

vehicle_features["collision_event_flag"] = np.where(
    vehicle_features["MAN_COLL"].notna(),
    (~vehicle_features["MAN_COLL"].isin([0, 98, 99])).astype("int8"),
    np.nan
)

vehicle_features["roadway_complexity_index"] = (
    vehicle_features[
        ["VNUM_LAN", "VALIGN", "VPROFILE", "VSURCOND", "VTRAFWAY"]
    ].notna().sum(axis=1)
)

vehicle_features["vehicle_state_observation_count"] = (
    vehicle_features[
        [
            "TRAV_SP", "VSPD_LIM", "VALIGN", "VPROFILE",
            "VSURCOND", "VTRAFCON", "VTRAFWAY"
        ]
    ].notna().sum(axis=1)
)

vehicle_features.to_parquet(
    OUT / "vehicle_features.parquet",
    index=False
)

# ------------------------------------------------------------------
# Occupant-level canonical dataset
# ------------------------------------------------------------------

person_cols = [
    "CASENUM", "VEH_NO", "PER_NO",
    "AGE", "SEX", "SEAT_POS",
    "AIR_BAG", "EJECTION",
    "REST_USE", "REST_MIS",
    "HELM_USE", "HELM_MIS",
    "INJ_SEV", "HOSPITAL",
    "HARM_EV", "MAN_COLL", "IMPACT1",
    "ROLLOVER", "FIRE_EXP",
    "PER_TYP"
]

person_features = person[
    [c for c in person_cols if c in person.columns]
].copy()

person_numeric = [
    "VEH_NO", "PER_NO", "AGE", "SEX", "SEAT_POS",
    "AIR_BAG", "EJECTION", "REST_USE", "REST_MIS",
    "HELM_USE", "HELM_MIS", "INJ_SEV", "HOSPITAL",
    "HARM_EV", "MAN_COLL", "IMPACT1",
    "ROLLOVER", "FIRE_EXP", "PER_TYP"
]

person_features = clean_numeric(person_features, person_numeric)

# ------------------------------------------------------------------
# Occupant protection / exposure features
# ------------------------------------------------------------------

person_features["restraint_available_flag"] = np.where(
    person_features["REST_USE"].notna(),
    (~person_features["REST_USE"].isin([8, 98, 99])).astype("int8"),
    np.nan
)

person_features["airbag_observed_flag"] = np.where(
    person_features["AIR_BAG"].notna(),
    (~person_features["AIR_BAG"].isin([97, 98, 99])).astype("int8"),
    np.nan
)

person_features["ejection_occurred_flag"] = np.where(
    person_features["EJECTION"].isin([1, 2, 3]),
    1,
    np.where(person_features["EJECTION"].notna(), 0, np.nan)
)

person_features["helmet_misuse_flag"] = np.where(
    person_features["HELM_MIS"].eq(1),
    1,
    np.where(person_features["HELM_MIS"].notna(), 0, np.nan)
)

person_features["hospitalization_flag"] = np.where(
    person_features["HOSPITAL"].notna(),
    (person_features["HOSPITAL"] > 0).astype("int8"),
    np.nan
)

# Preserve PER_TYP=3 as a special population rather than forcing
# an invalid vehicle join.
person_features["special_population_flag"] = (
    person_features["PER_TYP"].eq(3).astype("int8")
)

person_features.to_parquet(
    OUT / "occupant_features.parquet",
    index=False
)

# ------------------------------------------------------------------
# Crash-level context
# ------------------------------------------------------------------

crash_cols = [
    "CASENUM",
    "HARM_EV",
    "MAN_COLL",
    "PEDS"
]

crash_features = accident[
    [c for c in crash_cols if c in accident.columns]
].copy()

crash_features = clean_numeric(
    crash_features,
    ["HARM_EV", "MAN_COLL", "PEDS"]
)

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

vision_features = vision.copy()
weather_features = weather.copy()

for df in [vision_features, weather_features]:
    if "CASENUM" in df.columns:
        df["CASENUM"] = df["CASENUM"].astype(str).str.strip()

vision_keep = [c for c in ["CASENUM", "VISION"] if c in vision_features.columns]
weather_keep = [c for c in ["CASENUM", "WEATHER"] if c in weather_features.columns]

vision_features = vision_features[vision_keep].drop_duplicates("CASENUM")
weather_features = weather_features[weather_keep].drop_duplicates("CASENUM")

environment_features = vision_features.merge(
    weather_features,
    on="CASENUM",
    how="outer"
)

environment_features.to_parquet(
    OUT / "environment_features.parquet",
    index=False
)

# ------------------------------------------------------------------
# Merge crash + environment
# ------------------------------------------------------------------

crash_features = crash_features.merge(
    environment_features,
    on="CASENUM",
    how="left"
)

crash_features.to_parquet(
    OUT / "crash_features.parquet",
    index=False
)

# ------------------------------------------------------------------
# Event-level representation
# ------------------------------------------------------------------

event_parts = []

for source_name, df, level in [
    ("accident", accident, "crash"),
    ("vehicle", vehicle, "vehicle"),
    ("person", person, "person")
]:
    cols = [
        c for c in [
            "CASENUM", "VEH_NO", "PER_NO",
            "HARM_EV", "MAN_COLL", "IMPACT1",
            "ROLLOVER", "FIRE_EXP"
        ]
        if c in df.columns
    ]

    part = df[cols].copy()
    part["source_level"] = level
    part["source_table"] = source_name
    event_parts.append(part)

event_features = pd.concat(
    event_parts,
    ignore_index=True
)

event_features.to_parquet(
    OUT / "event_features.parquet",
    index=False
)

# ------------------------------------------------------------------
# Cyber-physical feature table
#
# This is deliberately an abstraction layer.
# CRSS supplies physical safety evidence; cyber telemetry will be
# attached in the next stage rather than falsely represented as CRSS.
# ------------------------------------------------------------------

cp = vehicle_features[
    [
        "CASENUM", "VEH_NO",
        "speed_limit_deviation",
        "speed_limit_deviation_abs",
        "speed_over_limit_flag",
        "rollover_flag",
        "fire_explosion_flag",
        "vehicle_damage_flag",
        "collision_event_flag",
        "roadway_complexity_index",
        "vehicle_state_observation_count"
    ]
].copy()

cp = cp.merge(
    environment_features,
    on="CASENUM",
    how="left"
)

cp["physical_state_completeness"] = (
    cp[
        [
            "speed_limit_deviation",
            "rollover_flag",
            "fire_explosion_flag",
            "vehicle_damage_flag",
            "collision_event_flag"
        ]
    ].notna().sum(axis=1) / 5.0
)

cp["cyber_physical_observation_vector"] = (
    cp[
        [
            "speed_limit_deviation",
            "speed_limit_deviation_abs",
            "speed_over_limit_flag",
            "rollover_flag",
            "fire_explosion_flag",
            "vehicle_damage_flag",
            "collision_event_flag",
            "roadway_complexity_index",
            "vehicle_state_observation_count",
            "physical_state_completeness"
        ]
    ].notna().sum(axis=1)
)

cp.to_parquet(
    OUT / "cyber_physical_features.parquet",
    index=False
)

# ------------------------------------------------------------------
# Feature specification
# ------------------------------------------------------------------

spec = {
    "dataset": "CRSS 2024 Cyber-Physical Feature Layer",
    "source": "NHTSA CRSS 2024",
    "status": "phase_1_generated",
    "source_role": {
        "CRSS": "physical safety and crash-investigation ground truth",
        "cyber_telemetry": "future synthetic/edge telemetry layer",
        "Cortex_XDR": "future SOC/security analytics integration"
    },
    "design_principles": [
        "preserve NHTSA coded missingness",
        "do not treat CRSS as a cybersecurity-labelled dataset",
        "separate predictors from downstream outcomes",
        "preserve PER_TYP=3 special population",
        "retain event relationships rather than flattening all events",
        "derived features must remain traceable to source variables"
    ],
    "outputs": [
        "crash_features.parquet",
        "vehicle_features.parquet",
        "occupant_features.parquet",
        "environment_features.parquet",
        "event_features.parquet",
        "cyber_physical_features.parquet"
    ],
    "derived_features": {
        "speed_limit_deviation": "TRAV_SP - VSPD_LIM",
        "speed_limit_deviation_abs": "absolute speed-limit deviation",
        "speed_over_limit_flag": "TRAV_SP greater than VSPD_LIM",
        "rollover_flag": "derived from CRSS ROLLOVER coding",
        "fire_explosion_flag": "FIRE_EXP == 1",
        "vehicle_damage_flag": "DEFORMED observed and positive",
        "collision_event_flag": "MAN_COLL indicates collision",
        "roadway_complexity_index": "count of observed roadway-context variables",
        "vehicle_state_observation_count": "count of observed vehicle-state variables",
        "physical_state_completeness": "fraction of core physical-state features observed"
    }
}

with open(
    OUT / "cyber_physical_feature_spec.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(spec, f, indent=2)

# ------------------------------------------------------------------
# Automated quality report
# ------------------------------------------------------------------

reports = {}

for name, df in {
    "crash": crash_features,
    "vehicle": vehicle_features,
    "occupant": person_features,
    "environment": environment_features,
    "event": event_features,
    "cyber_physical": cp
}.items():

    reports[name] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_cells": int(df.isna().sum().sum())
    }

quality = {
    "status": "PASS",
    "dataset": "CRSS 2024",
    "reports": reports,
    "checks": {
        "crash_features_nonempty": len(crash_features) > 0,
        "vehicle_features_nonempty": len(vehicle_features) > 0,
        "occupant_features_nonempty": len(person_features) > 0,
        "environment_features_nonempty": len(environment_features) > 0,
        "event_features_nonempty": len(event_features) > 0,
        "cyber_physical_features_nonempty": len(cp) > 0,
        "casenum_present": "CASENUM" in crash_features.columns,
        "vehicle_key_present": all(
            c in vehicle_features.columns for c in ["CASENUM", "VEH_NO"]
        ),
        "occupant_key_present": all(
            c in person_features.columns for c in ["CASENUM", "VEH_NO", "PER_NO"]
        )
    }
}

if not all(quality["checks"].values()):
    quality["status"] = "FAIL"
    raise RuntimeError(
        "Feature-engineering quality gate failed. "
        f"Report: {quality}"
    )

with open(
    OUT / "phase1_quality_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(quality, f, indent=2)

print("=" * 80)
print("CRSS 2024 PHASE 1 FEATURE ENGINEERING")
print("=" * 80)
print()
print("STATUS : PASS")
print()
for name, report in reports.items():
    print(
        f"{name:16s} rows={report['rows']:,} "
        f"columns={report['columns']} "
        f"duplicate_rows={report['duplicate_rows']:,}"
    )

print()
print("OUTPUT DIRECTORY:")
print(OUT)
print()
print("NEXT STAGE:")
print("Cyber telemetry synthesis + temporal cyber-physical state modeling")
print("=" * 80)
