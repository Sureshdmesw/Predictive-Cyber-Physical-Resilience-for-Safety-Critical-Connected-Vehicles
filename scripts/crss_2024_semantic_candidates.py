import csv
import re
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]

MANUAL = (
    ROOT / "data" / "raw" / "nhtsa" / "CRSS" / "documentation"
    / "FARS_CRSS_2024_Coding_Validation_Manual.pdf"
)

QUEUE = (
    ROOT / "data" / "schemas" / "nhtsa" / "verification"
    / "crss_2024_feature_verification_queue.csv"
)

OUTPUT = (
    ROOT / "data" / "schemas" / "nhtsa" / "verification"
    / "crss_2024_semantic_candidates.csv"
)

doc = pymupdf.open(MANUAL)

with open(QUEUE, "r", encoding="utf-8-sig", newline="") as f:
    queue = list(csv.DictReader(f))

# Cache page text
pages = {
    i + 1: doc[i].get_text("text")
    for i in range(len(doc))
}

records = []

for item in queue:
    variable = item["variable"].strip()

    # Find pages containing the variable.
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(variable)}(?![A-Za-z0-9_])",
        re.IGNORECASE
    )

    candidate_pages = [
        page_num
        for page_num, text in pages.items()
        if pattern.search(text)
    ]

    for page_num in candidate_pages:
        text = pages[page_num]

        # A true definition page normally contains several of these anchors.
        anchors = [
            "Format",
            "SAS Name",
            "Element Values",
        ]

        anchor_score = sum(
            1 for anchor in anchors
            if anchor.lower() in text.lower()
        )

        definition_match = re.search(
            r"Definition\s+(.*?)(?=\nRemarks\b|\nConsistency|\Z)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        definition = ""
        if definition_match:
            definition = re.sub(
                r"\s+",
                " ",
                definition_match.group(1)
            ).strip()

        sas_match = re.search(
            r"SAS Name\s+(.*?)(?=\nElement Values|\nDefinition|\Z)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        sas_name = ""
        if sas_match:
            sas_name = re.sub(
                r"\s+",
                " ",
                sas_match.group(1)
            ).strip()

        format_match = re.search(
            r"Format\s+(.*?)(?=\nSAS Name|\nElement Values|\Z)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        fmt = ""
        if format_match:
            fmt = re.sub(
                r"\s+",
                " ",
                format_match.group(1)
            ).strip()

        # Capture the section beginning with Element Values.
        values_match = re.search(
            r"Element Values\s+(.*?)(?=\nDefinition\b|\nRemarks\b|\Z)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        element_values = ""
        if values_match:
            element_values = re.sub(
                r"\s+",
                " ",
                values_match.group(1)
            ).strip()

        # Only retain pages that look like actual variable-definition pages.
        if anchor_score >= 2 or definition:
            records.append({
                "dataset": item.get("dataset", ""),
                "year": item.get("year", ""),
                "table": item.get("table", ""),
                "variable": variable,
                "role": item.get("role", ""),
                "family": item.get("family", ""),
                "manual_page": page_num,
                "semantic_anchor_score": anchor_score,
                "format": fmt,
                "sas_name": sas_name,
                "element_values": element_values,
                "definition": definition,
                "verification_status": (
                    "DEFINITION_CONTEXT_FOUND"
                    if definition
                    else "STRUCTURED_PAGE_FOUND"
                )
            })

fieldnames = [
    "dataset",
    "year",
    "table",
    "variable",
    "role",
    "family",
    "manual_page",
    "semantic_anchor_score",
    "format",
    "sas_name",
    "element_values",
    "definition",
    "verification_status"
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print("=== SEMANTIC CANDIDATE EXTRACTION COMPLETE ===")
print("Queue variables:", len(queue))
print("Semantic records:", len(records))
print("Output:", OUTPUT)
