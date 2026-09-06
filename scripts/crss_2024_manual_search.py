import csv
import re
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]

MANUAL = (
    ROOT
    / "data"
    / "raw"
    / "nhtsa"
    / "CRSS"
    / "documentation"
    / "FARS_CRSS_2024_Coding_Validation_Manual.pdf"
)

QUEUE = (
    ROOT
    / "data"
    / "schemas"
    / "nhtsa"
    / "verification"
    / "crss_2024_feature_verification_queue.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "schemas"
    / "nhtsa"
    / "verification"
    / "crss_2024_manual_search_results.csv"
)

print("ROOT:", ROOT)
print("MANUAL:", MANUAL)
print("QUEUE:", QUEUE)

if not MANUAL.exists():
    raise FileNotFoundError(f"Manual not found: {MANUAL}")

if not QUEUE.exists():
    raise FileNotFoundError(f"Verification queue not found: {QUEUE}")

doc = pymupdf.open(MANUAL)

# Extract page text once
pages = []
for page_number, page in enumerate(doc, start=1):
    pages.append(page.get_text("text"))

print(f"Manual pages loaded: {len(pages)}")

with open(QUEUE, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print(f"Queue variables: {len(rows)}")

output_rows = []

for row in rows:
    variable = row["variable"].strip()

    if not variable:
        continue

    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(variable)}(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )

    matched_pages = []

    for page_number, text in enumerate(pages, start=1):
        if pattern.search(text):
            matched_pages.append(page_number)

    output_rows.append({
        "dataset": row.get("dataset", ""),
        "year": row.get("year", ""),
        "table": row.get("table", ""),
        "variable": variable,
        "role": row.get("role", ""),
        "family": row.get("family", ""),
        "manual_match_count": len(matched_pages),
        "manual_pages": ";".join(map(str, matched_pages)),
        "manual_verified": "YES" if matched_pages else "NO",
        "official_definition": "",
        "official_codes": "",
        "unit_or_scale": "",
        "missing_codes": "",
        "physical_safety_relevance": "",
        "cyber_physical_relevance": "",
        "occupant_relevance": "",
        "vehicle_relevance": "",
        "adas_relevance": "",
        "feature_role": "",
        "verification_notes": "",
    })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

fieldnames = list(output_rows[0].keys()) if output_rows else [
    "dataset",
    "year",
    "table",
    "variable",
    "role",
    "family",
    "manual_match_count",
    "manual_pages",
    "manual_verified",
    "official_definition",
    "official_codes",
    "unit_or_scale",
    "missing_codes",
    "physical_safety_relevance",
    "cyber_physical_relevance",
    "occupant_relevance",
    "vehicle_relevance",
    "adas_relevance",
    "feature_role",
    "verification_notes",
]

with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print()
print("=== MANUAL SEARCH COMPLETE ===")
print("Output:", OUTPUT)
print("Variables processed:", len(output_rows))

print()
print("=== MATCH COUNTS ===")

for row in output_rows:
    print(
        f'{row["variable"]:40} '
        f'matches={row["manual_match_count"]:3} '
        f'pages={row["manual_pages"]}'
    )
