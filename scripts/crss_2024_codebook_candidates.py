import json
import re
from pathlib import Path

INPUT = Path(
    "data/schemas/nhtsa/codebook/"
    "crss_2024_codebook_context_full.json"
)

OUTPUT = Path(
    "data/schemas/nhtsa/codebook/"
    "crss_2024_codebook_candidates.json"
)

data = json.loads(INPUT.read_text(encoding="utf-8"))

results = {}

for variable, payload in data.items():

    candidates = []

    for match in payload.get("matches", []):

        page = match["page"]
        context = match["context"]

        # Keep only contexts that look like actual
        # data-element definitions/code tables.
        indicators = [
            "Attribute Codes",
            "SAS Name",
            "Definition",
            "Not Reported",
            "Reported as Unknown",
            "Not Applicable",
            "Unknown"
        ]

        score = sum(
            1
            for x in indicators
            if x.lower() in context.lower()
        )

        if score == 0:
            continue

        # Normalize whitespace while preserving the actual text.
        normalized = re.sub(r"\s+", " ", context).strip()

        candidates.append({
            "page": page,
            "score": score,
            "context": normalized
        })

    # Highest-value contexts first.
    candidates.sort(
        key=lambda x: (-x["score"], x["page"])
    )

    results[variable] = {
        "candidate_context_count": len(candidates),
        "candidates": candidates[:10]
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
print("CRSS 2024 CODEBOOK CANDIDATE EXTRACTION")
print("=" * 80)

for variable, payload in results.items():
    print(
        f"{variable:<12} "
        f"candidates={payload['candidate_context_count']}"
    )

print()
print("Generated:")
print(OUTPUT)
print("Done.")
