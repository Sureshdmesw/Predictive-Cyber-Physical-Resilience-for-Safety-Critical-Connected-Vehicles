import json
import re
from pathlib import Path

from pypdf import PdfReader

MANUAL = Path(
    "data/raw/nhtsa/CRSS/documentation/"
    "CRSS_Analytical_Users_Manual_2016_2024.pdf"
)

FEATURE_DICT = Path(
    "data/schemas/nhtsa/verification/"
    "crss_2024_final_feature_dictionary_v2.csv"
)

OUT_JSON = Path(
    "data/schemas/nhtsa/codebook/"
    "crss_2024_codebook_context_full.json"
)

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CRSS 2024 CODEBOOK CONTEXT EXTRACTION")
print("=" * 80)

# ------------------------------------------------------------------
# Load variables from final feature dictionary
# ------------------------------------------------------------------

import csv

variables = []

with FEATURE_DICT.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        variable = row.get("variable", "").strip()

        if variable and variable not in variables:
            variables.append(variable)

print(f"Variables: {len(variables)}")

# ------------------------------------------------------------------
# Read manual
# ------------------------------------------------------------------

reader = PdfReader(str(MANUAL))
print(f"Manual pages: {len(reader.pages)}")

pages = []

for i, page in enumerate(reader.pages, start=1):
    try:
        text = page.extract_text() or ""
    except Exception as exc:
        print(f"WARNING: page {i} extraction failed: {exc}")
        text = ""

    pages.append(text)

# ------------------------------------------------------------------
# Extract larger contexts around each variable
# ------------------------------------------------------------------

results = {}

for variable in variables:

    pattern = re.compile(
        rf"\b{re.escape(variable)}\b",
        re.IGNORECASE
    )

    matches = []

    for page_num, text in enumerate(pages, start=1):

        for match in pattern.finditer(text):

            start = max(0, match.start() - 1800)
            end = min(len(text), match.end() + 5000)

            context = text[start:end]

            # Prefer contexts that contain code-related language.
            score = 0

            keywords = [
                "Attribute Codes",
                "Definition",
                "Additional Information",
                "Not Reported",
                "Unknown",
                "Reported as Unknown",
                "Not Applicable",
                "2018-Later",
                "2019-Later",
                "2020-Later",
                "2022-Later",
                "2023-Later",
                "2024"
            ]

            for keyword in keywords:
                if keyword.lower() in context.lower():
                    score += 1

            matches.append(
                {
                    "page": page_num,
                    "score": score,
                    "context": context
                }
            )

    # Remove exact duplicate contexts.
    unique = []
    seen = set()

    for item in matches:

        key = (
            item["page"],
            item["context"]
        )

        if key not in seen:
            seen.add(key)
            unique.append(item)

    unique.sort(
        key=lambda x: (
            -x["score"],
            x["page"]
        )
    )

    results[variable] = {
        "match_count": len(unique),
        "matches": unique[:20]
    }

    print(
        f"{variable:<12} "
        f"contexts={len(unique)}"
    )

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

with OUT_JSON.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

print()
print("=" * 80)
print("GENERATED")
print("=" * 80)
print(OUT_JSON)
print("Done.")
