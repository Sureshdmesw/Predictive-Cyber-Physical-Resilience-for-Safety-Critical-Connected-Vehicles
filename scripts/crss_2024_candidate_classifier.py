from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "schemas"
    / "nhtsa"
    / "crss_2024_variable_inventory.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "schemas"
    / "nhtsa"
    / "crss_2024_candidate_variables.csv"
)

# Variables that are primarily identifiers / survey-design fields
IDENTIFIER_VARS = {
    "CASENUM",
    "PSU",
    "PSU_VAR",
    "PSUSTRAT",
    "REGION",
    "STRATUM",
    "PJ",
}

SURVEY_VARS = {
    "PSU",
    "PSU_VAR",
    "PSUSTRAT",
    "STRATUM",
    "WEIGHT",
    "PJ",
}

# Human-readable companion fields
NAME_SUFFIX = "NAME"

def classify_family(table, variable):
    v = variable.upper()

    if table == "accident.csv":
        return "crash_context"

    if table in {
        "vehicle.csv",
        "vehiclesf.csv",
        "factor.csv",
        "maneuver.csv",
        "damage.csv",
        "driverrf.csv",
        "distract.csv",
        "drimpair.csv",
    }:
        return "vehicle"

    if table in {
        "person.csv",
        "personrf.csv",
        "safetyeq.csv",
        "nmcrash.csv",
        "nmdistract.csv",
        "nmimpair.csv",
        "nmprior.csv",
        "pbtype.csv",
    }:
        return "occupant_person"

    if table in {
        "cevent.csv",
        "vevent.csv",
        "vsoe.csv",
    }:
        return "event_sequence"

    if table in {
        "vision.csv",
        "weather.csv",
    }:
        return "environment"

    if table in {
        "vpicdecode.csv",
        "vpictrailerdecode.csv",
    }:
        return "vehicle_technology"

    if table in {
        "violatn.csv",
        "crashrf.csv",
    }:
        return "risk_factor"

    if table in {
        "parkwork.csv",
        "pvehiclesf.csv",
    }:
        return "special_vehicle"

    return "other"


def classify_role(table, variable):
    v = variable.upper()

    if v in SURVEY_VARS:
        return "survey_design"

    if v in IDENTIFIER_VARS:
        return "identifier"

    if v.endswith(NAME_SUFFIX):
        return "descriptive_label"

    if v in {
        "CASENUM",
        "VEH_NO",
        "PER_NO",
        "EVENTNUM",
        "VEVENTNUM",
        "VNUMBER1",
        "VNUMBER2",
        "AOI",
        "AOI1",
        "AOI2",
        "SOE",
    }:
        return "relational_identifier"

    if v in {
        "INJ_SEV",
        "MAX_SEV",
        "MAX_VSEV",
        "NUM_INJ",
        "NUM_INJV",
        "NO_INJ_IM",
        "INJSEV_IM",
        "MAXSEV_IM",
        "MXVSEV_IM",
        "NUMINJ_IM",
    }:
        return "candidate_target"

    return "candidate_feature"


def relevance_flags(table, variable):
    v = variable.upper()

    occupant_terms = {
        "AGE",
        "SEX",
        "INJ",
        "SEAT",
        "REST",
        "AIR_BAG",
        "EJECT",
        "DRINK",
        "ALC",
        "DRUG",
        "HOSPITAL",
        "HELM",
        "PB",
        "PED",
        "BIKE",
    }

    vehicle_terms = {
        "VEH",
        "VIN",
        "TRAV_SP",
        "SPEED",
        "VSPD",
        "VTRAF",
        "LAN",
        "ALIGN",
        "PROFILE",
        "SUR",
        "ROLL",
        "IMPACT",
        "DEFORM",
        "FIRE",
        "BRAKE",
        "ENGINE",
        "BATTERY",
        "ADAS",
        "LANE",
        "COLLISION",
    }

    event_terms = {
        "EVENT",
        "SOE",
        "AOI",
        "VNUMBER",
        "HARM_EV",
        "MAN_COLL",
        "ACC_TYPE",
    }

    cyber_physical_terms = {
        "SPEED",
        "TRAV_SP",
        "VSPD",
        "BRAKE",
        "LANE",
        "COLLISION",
        "ROLL",
        "IMPACT",
        "VISION",
        "ADAS",
        "WARNING",
        "ASSIST",
        "CONTROL",
        "EVENT",
        "SOE",
        "AOI",
        "ACC_TYPE",
    }

    return {
        "occupant_relevance": any(x in v for x in occupant_terms),
        "vehicle_relevance": any(x in v for x in vehicle_terms),
        "event_relevance": any(x in v for x in event_terms),
        "cyber_physical_relevance": any(
            x in v for x in cyber_physical_terms
        ),
    }


with INPUT.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

output_rows = []

for row in rows:
    table = row["table"]
    variable = row["variable"]

    role = classify_role(table, variable)
    family = classify_family(table, variable)

    flags = relevance_flags(table, variable)

    requires_verification = (
        role in {
            "candidate_feature",
            "candidate_target",
        }
        and (
            flags["cyber_physical_relevance"]
            or flags["vehicle_relevance"]
            or flags["occupant_relevance"]
            or flags["event_relevance"]
        )
    )

    output_rows.append({
        "dataset": row["dataset"],
        "year": row["year"],
        "table": table,
        "column_position": row["column_position"],
        "variable": variable,
        "role": role,
        "family": family,
        "identifier_flag": role in {
            "identifier",
            "relational_identifier",
        },
        "name_field_flag": variable.upper().endswith(NAME_SUFFIX),
        "survey_design_flag": role == "survey_design",
        "candidate_feature": role == "candidate_feature",
        "candidate_target": role == "candidate_target",
        "candidate_context": role in {
            "identifier",
            "relational_identifier",
            "descriptive_label",
        },
        "cyber_physical_relevance": flags["cyber_physical_relevance"],
        "occupant_relevance": flags["occupant_relevance"],
        "vehicle_relevance": flags["vehicle_relevance"],
        "event_relevance": flags["event_relevance"],
        "requires_codebook_verification": requires_verification,
    })


FIELDNAMES = list(output_rows[0].keys())

with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(output_rows)

print()
print("=" * 80)
print("CRSS 2024 CANDIDATE VARIABLE CLASSIFICATION")
print("=" * 80)
print()
print(f"Input variables : {len(rows)}")
print(f"Output variables: {len(output_rows)}")
print(f"Output          : {OUTPUT}")
print()

from collections import Counter

print("ROLE COUNTS")
for key, value in Counter(r["role"] for r in output_rows).most_common():
    print(f"{key:25} {value}")

print()
print("FAMILY COUNTS")
for key, value in Counter(r["family"] for r in output_rows).most_common():
    print(f"{key:25} {value}")

print()
print("REQUIRES CODEBOOK VERIFICATION")
print(
    sum(
        r["requires_codebook_verification"]
        for r in output_rows
    )
)

print()
print("DONE")
