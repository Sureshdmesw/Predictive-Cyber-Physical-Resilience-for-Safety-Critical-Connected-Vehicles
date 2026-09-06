from pathlib import Path
import json
import csv

ROOT = Path(__file__).resolve().parents[1]

PROFILE = (
    ROOT
    / "experiments"
    / "nhtsa"
    / "crss_2024_profile.json"
)

OUTPUT = (
    ROOT
    / "data"
    / "schemas"
    / "nhtsa"
    / "crss_2024_variable_inventory.csv"
)

with PROFILE.open(
    "r",
    encoding="utf-8"
) as handle:
    data = json.load(handle)

rows = []

for item in data:

    table = item["file"]

    for position, column in enumerate(
        item["columns"],
        start=1
    ):

        rows.append({
            "dataset": "CRSS",
            "year": "2024",
            "table": table,
            "column_position": position,
            "variable": column
        })

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "dataset",
            "year",
            "table",
            "column_position",
            "variable"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print()
print("=" * 80)
print("CRSS 2024 VARIABLE INVENTORY")
print("=" * 80)
print()
print(f"Tables: {len(data)}")
print(f"Variables: {len(rows)}")
print(f"Output: {OUTPUT}")
print()
