import json
import re
from pathlib import Path

INPUT = Path(
    "data/schemas/nhtsa/codebook/"
    "crss_2024_codebook_candidates.json"
)

OUTPUT = Path(
    "data/schemas/nhtsa/codebook/"
    "crss_2024_codebook_parsed_candidates.json"
)

data = json.loads(INPUT.read_text(encoding="utf-8"))

results = {}

# ------------------------------------------------------------------
# Patterns for CRSS code/value pairs
# ------------------------------------------------------------------

code_patterns = [
    re.compile(
        r'(?<![\w.-])(\d{1,4})\s+'
        r'([A-Z][^0-9]{2,120}?)'
        r'(?=\s+\d{1,4}\s+|\s+--|\s*$)',
        re.IGNORECASE
    ),

    re.compile(
        r'(?<![\w.-])(\d{1,4})\s+'
        r'(Not Reported|Unknown|Reported as Unknown|'
        r'Not Applicable|Yes|No|'
        r'Less Than One Year|Age in Years)',
        re.IGNORECASE
    )
]

# Words that commonly indicate actual code-table rows.
code_indicators = [
    "Attribute Codes",
    "Not Reported",
    "Reported as Unknown",
    "Not Applicable",
    "Unknown",
    "2016-Later",
    "2018-Later",
    "2019-Later",
    "2020-Later",
    "2022-Later"
]

for variable, payload in data.items():

    parsed = []

    for candidate in payload.get("candidates", []):

        context = candidate["context"]
        page = candidate["page"]

        if not any(
            indicator.lower() in context.lower()
            for indicator in code_indicators
        ):
            continue

        # Normalize PDF extraction artifacts.
        text = re.sub(r"\s+", " ", context).strip()

        found = []

        for pattern in code_patterns:
            for match in pattern.finditer(text):

                code = match.group(1)
                label = match.group(2).strip()

                label = re.sub(
                    r"\s+",
                    " ",
                    label
                )

                # Avoid obvious false positives from page numbers,
                # years, and data-element IDs.
                if code in {
                    "2016", "2017", "2018", "2019",
                    "2020", "2021", "2022", "2023", "2024"
                }:
                    continue

                if len(label) < 3:
                    continue

                item = {
                    "code": code,
                    "label": label,
                    "page": page
                }

                if item not in found:
                    found.append(item)

        if found:
            parsed.append({
                "page": page,
                "codes": found,
                "context": text
            })

    results[variable] = {
        "parsed_contexts": parsed
    }

OUTPUT.write_text(
    json.dumps(
        results,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print("=" * 80)
print("CRSS 2024 PARSED CODEBOOK CANDIDATES")
print("=" * 80)

total_codes = 0

for variable, payload in results.items():

    codes = []

    for context in payload["parsed_contexts"]:
        codes.extend(context["codes"])

    # Deduplicate by code + label.
    unique = []
    seen = set()

    for item in codes:
        key = (
            item["code"],
            item["label"]
        )

        if key not in seen:
            seen.add(key)
            unique.append(item)

    total_codes += len(unique)

    print(
        f"{variable:<12} "
        f"candidate_codes={len(unique)}"
    )

print()
print("Variables:", len(results))
print("Total unique code/label candidates:", total_codes)
print()
print("Generated:")
print(OUTPUT)
print("Done.")
