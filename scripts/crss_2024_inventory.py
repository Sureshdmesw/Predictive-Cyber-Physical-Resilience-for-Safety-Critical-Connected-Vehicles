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
    / "crss_2024_table_inventory.csv"
)

with PROFILE.open(
    "r",
    encoding="utf-8"
) as handle:

    data = json.load(handle)

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as handle:

    writer = csv.writer(handle)

    writer.writerow([
        "table",
        "size_bytes",
        "row_count",
        "column_count",
        "columns"
    ])

    for item in data:

        writer.writerow([
            item["file"],
            item["size_bytes"],
            item["row_count"],
            len(item["columns"]),
            "|".join(item["columns"])
        ])

print(f"Created: {OUTPUT}")
