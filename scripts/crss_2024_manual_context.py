import csv
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]

MANUAL = (
    ROOT / "data" / "raw" / "nhtsa" / "CRSS" / "documentation"
    / "FARS_CRSS_2024_Coding_Validation_Manual.pdf"
)

INPUT = (
    ROOT / "data" / "schemas" / "nhtsa" / "verification"
    / "crss_2024_manual_search_results.csv"
)

OUTPUT = (
    ROOT / "data" / "schemas" / "nhtsa" / "verification"
    / "crss_2024_manual_context.csv"
)

doc = pymupdf.open(MANUAL)

with open(INPUT, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

output_rows = []

for row in rows:
    pages_text = row.get("manual_pages", "").strip()

    if not pages_text:
        continue

    pages = []

    for value in pages_text.split(";"):
        try:
            pages.append(int(value))
        except ValueError:
            pass

    for page_number in pages:
        if page_number < 1 or page_number > len(doc):
            continue

        text = doc[page_number - 1].get_text("text")

        output_rows.append({
            "dataset": row.get("dataset", ""),
            "year": row.get("year", ""),
            "table": row.get("table", ""),
            "variable": row.get("variable", ""),
            "role": row.get("role", ""),
            "family": row.get("family", ""),
            "manual_page": page_number,
            "manual_text": text.replace("\x00", " ").strip()
        })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "dataset",
    "year",
    "table",
    "variable",
    "role",
    "family",
    "manual_page",
    "manual_text"
]

with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print("=== MANUAL CONTEXT EXTRACTION COMPLETE ===")
print("Variables in search results:", len(rows))
print("Context records:", len(output_rows))
print("Output:", OUTPUT)
