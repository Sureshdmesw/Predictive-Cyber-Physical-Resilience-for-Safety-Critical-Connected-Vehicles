import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/schemas/nhtsa/verification/crss_2024_final_feature_dictionary.csv"
OUTPUT = ROOT / "data/schemas/nhtsa/verification/crss_2024_final_feature_dictionary_v2.csv"

# Fields previously quarantined but now explicitly confirmed
# by the 2024 FARS/CRSS Coding and Validation Manual.
RESOLVED = {
    ("accident.csv", "HARM_EV"),
    ("person.csv", "HARM_EV"),
    ("person.csv", "SEAT_POS"),
    ("person.csv", "IMPACT1"),
    ("vehicle.csv", "HARM_EV"),
    ("vehicle.csv", "IMPACT1"),
    ("vehicle.csv", "VTRAFCON"),
}

# Relational/event fields should remain relational.
RELATIONAL_EVENT = {
    "HARM_EV",
    "MAN_COLL",
    "IMPACT1",
}

rows = list(csv.DictReader(INPUT.open(encoding="utf-8-sig")))

for row in rows:
    key = (row["table"], row["variable"])

    if key in RESOLVED:
        row["verification_status"] = "VERIFIED_MANUAL_REVIEW"
        row["review_required"] = "no"
        row["predictive_use"] = "historical_safety_context"

    if row["variable"] in RELATIONAL_EVENT:
        row["feature_level"] = "relational_event"
        row["semantic_category"] = "event_sequence"

    if row["variable"] == "SEAT_POS":
        row["feature_level"] = "person"
        row["domain"] = "person"
        row["semantic_category"] = "occupant_state"
        row["feature_role"] = "predictor"

    if row["variable"] == "VTRAFCON":
        row["feature_level"] = "vehicle"
        row["domain"] = "vehicle"
        row["semantic_category"] = "roadway_context"
        row["feature_role"] = "predictor"

    if row["variable"] == "HARM_EV":
        row["feature_level"] = "relational_event"
        row["domain"] = "event"
        row["semantic_category"] = "event_sequence"
        row["feature_role"] = "predictor"

    if row["variable"] == "IMPACT1":
        row["feature_level"] = "relational_event"
        row["domain"] = "event"
        row["semantic_category"] = "event_sequence"
        row["feature_role"] = "predictor"

    # These are now fully resolved.
    if row["review_required"] == "no":
        row["predictive_use"] = "historical_safety_context"

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    fields = list(rows[0].keys())
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("=== CRSS 2024 FINAL SEMANTIC RESOLUTION ===")
print(f"Input records:       {len(rows)}")
print(f"Output records:      {len(rows)}")
print(f"Verified total:      {sum(r['verification_status'] != 'REVIEW_REQUIRED' for r in rows)}")
print(f"Review required:     {sum(r['review_required'] == 'yes' for r in rows)}")
print(f"Resolved manually:   {sum(r['verification_status'] == 'VERIFIED_MANUAL_REVIEW' for r in rows)}")
print(f"Output:              {OUTPUT}")
