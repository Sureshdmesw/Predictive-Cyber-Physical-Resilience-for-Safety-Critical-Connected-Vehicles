from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]

SCHEMA_DIR = ROOT / "data/schemas/nhtsa"

FEATURE_DICT = (
    SCHEMA_DIR
    / "verification"
    / "crss_2024_final_feature_dictionary_v2.csv"
)

features = pd.read_csv(FEATURE_DICT)

def normalize_table(table):
    return str(table).strip().lower().replace(".csv", "")

def logical_entity(table, variable):
    table = normalize_table(table)
    variable = str(variable).strip().upper()

    if table == "accident":
        return "crash"

    if table == "vehicle":
        if variable in {"HARM_EV", "MAN_COLL", "IMPACT1"}:
            return "event"
        if variable in {"ROLLOVER", "FIRE_EXP", "DEFORMED"}:
            return "vehicle_safety"
        return "vehicle"

    if table == "person":
        if variable in {"HARM_EV", "MAN_COLL", "IMPACT1"}:
            return "event"

        if variable in {"INJ_SEV", "HOSPITAL"}:
            return "occupant_outcome"

        if variable in {
            "EJECTION",
            "AIR_BAG",
            "REST_USE",
            "REST_MIS",
            "HELM_USE",
            "HELM_MIS",
        }:
            return "occupant_protection"

        if variable in {"ROLLOVER", "FIRE_EXP"}:
            return "vehicle_safety"

        return "person"

    if table in {"vision", "weather"}:
        return "environment"

    return "other"


records = []

for _, row in features.iterrows():

    table = str(row["table"]).strip()
    variable = str(row["variable"]).strip()

    if not table or not variable:
        continue

    records.append({
        "table": table,
        "variable": variable,
        "logical_entity": logical_entity(table, variable),
        "semantic_category": str(
            row.get("semantic_category", "")
        ).strip(),
        "verified": True,
    })


feature_map = pd.DataFrame(records)

feature_map = (
    feature_map
    .drop_duplicates(subset=["table", "variable"])
    .sort_values(
        ["logical_entity", "table", "variable"]
    )
    .reset_index(drop=True)
)


feature_map_path = (
    SCHEMA_DIR / "crss_2024_feature_map.csv"
)

feature_map.to_csv(
    feature_map_path,
    index=False
)


schema_path = (
    SCHEMA_DIR / "crss_2024_canonical_schema.json"
)

schema = {
    "dataset": "NHTSA CRSS 2024",
    "version": "2024",

    "analytical_grain": {
        "recommended": "vehicle",
        "definition": (
            "One row per in-transport vehicle involved "
            "in a CRSS crash."
        )
    },

    "entities": {
        "crash": {
            "source_table": "accident.csv",
            "primary_key": ["CASENUM"]
        },

        "vehicle": {
            "source_table": "vehicle.csv",
            "primary_key": [
                "CASENUM",
                "VEH_NO"
            ],
            "filter": "UNITTYPE = 1"
        },

        "person": {
            "source_table": "person.csv",
            "primary_key": [
                "CASENUM",
                "VEH_NO",
                "PER_NO"
            ]
        },

        "environment": {
            "source_tables": [
                "vision.csv",
                "weather.csv"
            ],
            "grain": "crash"
        },

        "event": {
            "source_tables": [
                "accident.csv",
                "vehicle.csv",
                "person.csv"
            ]
        },

        "vehicle_safety": {
            "source_table": "vehicle.csv"
        },

        "occupant_protection": {
            "source_table": "person.csv"
        },

        "occupant_outcome": {
            "source_table": "person.csv"
        }
    },

    "relationships": [
        {
            "parent": "crash",
            "child": "vehicle",
            "type": "1:N",
            "key": ["CASENUM"]
        },
        {
            "parent": "vehicle",
            "child": "person",
            "type": "1:N",
            "key": [
                "CASENUM",
                "VEH_NO"
            ]
        }
    ],

    "special_populations": {
        "non_motorists": {
            "condition": "VEH_NO = 0",
            "handling": "retain"
        },

        "non_transport_occupants": {
            "condition": "PER_TYP = 3",
            "handling": (
                "retain without requiring "
                "vehicle-table join"
            )
        }
    },

    "vehicle_count_semantics": {
        "VE_FORMS": (
            "Exact count of vehicle.csv records "
            "where UNITTYPE = 1."
        ),
        "VE_TOTAL": (
            "Crash-level total involved entities; "
            "not equivalent to vehicle.csv row count."
        )
    },

    "feature_statistics": {
        "verified_features": int(len(feature_map)),
        "logical_entity_counts": (
            feature_map["logical_entity"]
            .value_counts()
            .sort_index()
            .to_dict()
        )
    },

    "feature_map": records
}


with open(
    schema_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        schema,
        f,
        indent=2,
        ensure_ascii=False
    )


print("=" * 80)
print("CRSS 2024 CANONICAL SCHEMA — CORRECTED")
print("=" * 80)

print(f"\nVerified features: {len(feature_map):,}")

print("\nLogical entity distribution:")
print(
    feature_map["logical_entity"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nFeature map:")
print(
    feature_map.to_string(index=False)
)

print("\nGenerated:")
print(f"  {feature_map_path}")
print(f"  {schema_path}")

print("\nValidation:")
print(
    "PASS"
    if "other" not in feature_map["logical_entity"].values
    else "FAIL — unexpected 'other' entity remains"
)

print("\nDone.")
