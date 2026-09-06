import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/schemas/nhtsa/verification/crss_2024_semantic_verified_candidates.csv"
OUTPUT = ROOT / "data/schemas/nhtsa/verification/crss_2024_final_feature_dictionary.csv"

REVIEW_REQUIRED = {
    ("accident.csv", "HARM_EV"),
    ("person.csv", "HARM_EV"),
    ("person.csv", "SEAT_POS"),
    ("person.csv", "IMPACT1"),
    ("vehicle.csv", "HARM_EV"),
    ("vehicle.csv", "IMPACT1"),
    ("vehicle.csv", "VTRAFCON"),
}

# Explicitly classify fields by their CRSS relational level and research role.
CLASSIFICATION = {
    "PEDS":      ("accident", "crash_context", "predictor"),
    "HARM_EV":   ("event", "event_sequence", "predictor"),
    "MAN_COLL":  ("event", "event_sequence", "predictor"),
    "AGE":       ("person", "occupant_state", "predictor"),
    "SEX":       ("person", "occupant_state", "predictor"),
    "INJ_SEV":   ("person", "occupant_outcome", "target_candidate"),
    "SEAT_POS":  ("person", "occupant_state", "predictor"),
    "REST_USE":  ("person", "occupant_protection", "predictor"),
    "REST_MIS":  ("person", "occupant_protection", "predictor"),
    "HELM_USE":  ("person", "occupant_protection", "predictor"),
    "HELM_MIS":  ("person", "occupant_protection", "predictor"),
    "AIR_BAG":   ("person", "occupant_protection", "predictor"),
    "EJECTION":  ("person", "occupant_outcome", "target_candidate"),
    "HOSPITAL":  ("person", "occupant_outcome", "target_candidate"),
    "ROLLOVER":  ("vehicle", "vehicle_dynamics", "predictor"),
    "IMPACT1":   ("event", "event_sequence", "predictor"),
    "FIRE_EXP":  ("vehicle", "vehicle_safety_state", "target_candidate"),
    "TRAV_SP":   ("vehicle", "vehicle_dynamics", "predictor"),
    "DEFORMED":  ("vehicle", "vehicle_safety_state", "target_candidate"),
    "SPEEDREL":  ("vehicle", "vehicle_dynamics", "predictor"),
    "VTRAFWAY":  ("vehicle", "roadway_context", "predictor"),
    "VNUM_LAN":  ("vehicle", "roadway_context", "predictor"),
    "VSPD_LIM":  ("vehicle", "roadway_context", "predictor"),
    "VALIGN":    ("vehicle", "roadway_context", "predictor"),
    "VPROFILE":  ("vehicle", "roadway_context", "predictor"),
    "VSURCOND":  ("vehicle", "roadway_context", "predictor"),
    "VTRAFCON":  ("vehicle", "roadway_context", "predictor"),
    "ACC_TYPE":  ("vehicle", "crash_context", "predictor"),
    "VISION":    ("vision", "environmental_context", "predictor"),
    "WEATHER":   ("weather", "environmental_context", "predictor"),
}

CYBER_PHYSICAL = {
    "PEDS": "indirect",
    "HARM_EV": "high",
    "MAN_COLL": "high",
    "AGE": "low",
    "SEX": "low",
    "INJ_SEV": "high",
    "SEAT_POS": "high",
    "REST_USE": "high",
    "REST_MIS": "high",
    "HELM_USE": "high",
    "HELM_MIS": "high",
    "AIR_BAG": "high",
    "EJECTION": "high",
    "HOSPITAL": "high",
    "ROLLOVER": "high",
    "IMPACT1": "high",
    "FIRE_EXP": "high",
    "TRAV_SP": "high",
    "DEFORMED": "high",
    "SPEEDREL": "high",
    "VTRAFWAY": "medium",
    "VNUM_LAN": "medium",
    "VSPD_LIM": "medium",
    "VALIGN": "medium",
    "VPROFILE": "medium",
    "VSURCOND": "medium",
    "VTRAFCON": "medium",
    "ACC_TYPE": "high",
    "VISION": "medium",
    "WEATHER": "medium",
}

def feature_level(table, variable):
    if variable in {"HARM_EV", "MAN_COLL", "IMPACT1"}:
        return "relational_event"

    if table == "accident.csv":
        return "accident"

    if table == "person.csv":
        return "person"

    if table == "vehicle.csv":
        return "vehicle"

    if table == "vision.csv":
        return "environment"

    if table == "weather.csv":
        return "environment"

    return "other"

def data_type(row):
    fmt = (row.get("format") or "").lower()

    if "numeric" in fmt:
        return "numeric"
    if "subfields" in fmt:
        return "multi_subfield"
    return "categorical_or_other"

rows = list(csv.DictReader(INPUT.open(encoding="utf-8-sig")))

# Deduplicate documentation repeats.
unique = {}
for row in rows:
    key = (row["table"], row["variable"])

    # Prefer a fully verified record over a review-required record.
    if key not in unique:
        unique[key] = row
    elif "VERIFIED" in row["verification_status"]:
        unique[key] = row

output = []

for key, row in sorted(unique.items()):
    table = row["table"]
    variable = row["variable"]

    status = (
        "REVIEW_REQUIRED"
        if key in REVIEW_REQUIRED
        else "VERIFIED"
    )

    domain, category, role = CLASSIFICATION.get(
        variable,
        ("unknown", "unclassified", "review")
    )

    # Never automatically promote quarantined fields.
    if status == "REVIEW_REQUIRED":
        role = "review_required"

    output.append({
        "table": table,
        "variable": variable,
        "manual_page": row["manual_page"],
        "verification_status": status,
        "source_verification_status": row["verification_status"],
        "feature_level": feature_level(table, variable),
        "domain": domain,
        "semantic_category": category,
        "feature_role": role,
        "data_type": data_type(row),
        "format": row.get("format", ""),
        "values_length": row.get("values_length", ""),
        "definition_length": row.get("definition_length", ""),
        "cyber_physical_relevance": CYBER_PHYSICAL.get(variable, "review"),
        "predictive_use": (
            "quarantined_pending_semantic_review"
            if status == "REVIEW_REQUIRED"
            else "historical_safety_context"
        ),
        "target_candidate": (
            "yes" if role == "target_candidate" else "no"
        ),
        "duplicate_documentation": (
            "yes" if sum(
                1 for r in rows
                if r["table"] == table and r["variable"] == variable
            ) > 1 else "no"
        ),
        "review_required": (
            "yes" if status == "REVIEW_REQUIRED" else "no"
        ),
    })

fields = [
    "table",
    "variable",
    "manual_page",
    "verification_status",
    "source_verification_status",
    "feature_level",
    "domain",
    "semantic_category",
    "feature_role",
    "data_type",
    "format",
    "values_length",
    "definition_length",
    "cyber_physical_relevance",
    "predictive_use",
    "target_candidate",
    "duplicate_documentation",
    "review_required",
]

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(output)

print("=== CRSS 2024 FINAL FEATURE DICTIONARY ===")
print(f"Input records:       {len(rows)}")
print(f"Unique table.fields: {len(output)}")
print(f"Review required:     {sum(r['review_required'] == 'yes' for r in output)}")
print(f"Verified:            {sum(r['verification_status'] == 'VERIFIED' for r in output)}")
print(f"Output:              {OUTPUT}")
