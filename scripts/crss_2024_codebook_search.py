from pathlib import Path
import re
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MANUAL = (
    ROOT
    / "data/raw/nhtsa/CRSS/documentation"
    / "CRSS_Analytical_Users_Manual_2016_2024.pdf"
)

FEATURE_DICT = (
    ROOT
    / "data/schemas/nhtsa/verification"
    / "crss_2024_final_feature_dictionary_v2.csv"
)

OUT_DIR = ROOT / "data/schemas/nhtsa/codebook"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "crss_2024_codebook_search_context.json"

if not MANUAL.exists():
    raise FileNotFoundError(f"Manual not found: {MANUAL}")

if not FEATURE_DICT.exists():
    raise FileNotFoundError(
        f"Feature dictionary not found: {FEATURE_DICT}"
    )

try:
    from pypdf import PdfReader
except ImportError:
    raise SystemExit(
        "pypdf is not installed. Run: "
        "python -m pip install pypdf"
    )

features = pd.read_csv(FEATURE_DICT)

variables = (
    features["variable"]
    .astype(str)
    .str.strip()
    .str.upper()
    .drop_duplicates()
    .tolist()
)

print("=" * 80)
print("CRSS 2024 CODEBOOK SEARCH")
print("=" * 80)

print(f"\nManual: {MANUAL}")
print(f"Variables to search: {len(variables)}")

reader = PdfReader(str(MANUAL))

print(f"Manual pages: {len(reader.pages):,}")

results = {}

for variable in variables:

    matches = []

    pattern = re.compile(
        rf"\b{re.escape(variable)}\b",
        re.IGNORECASE
    )

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        if pattern.search(text):

            # Extract a local context window around each occurrence.
            snippets = []

            for match in pattern.finditer(text):

                start = max(0, match.start() - 700)
                end = min(len(text), match.end() + 1200)

                snippet = text[start:end]
                snippet = re.sub(
                    r"\s+",
                    " ",
                    snippet
                ).strip()

                snippets.append(snippet)

                # Avoid excessive duplicate snippets on one page.
                if len(snippets) >= 3:
                    break

            matches.append({
                "page": page_number,
                "snippets": snippets
            })

    results[variable] = {
        "match_count": len(matches),
        "matches": matches
    }

    print(
        f"{variable:<12} "
        f"{len(matches):>5} page matches"
    )


with open(
    OUT_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\n" + "=" * 80)
print("SEARCH SUMMARY")
print("=" * 80)

for variable, result in results.items():

    if result["match_count"] == 0:
        print(f"REVIEW: {variable} -> no manual match")
    else:
        pages = [
            x["page"]
            for x in result["matches"]
        ]

        print(
            f"{variable:<12} "
            f"pages={pages[:15]}"
        )

print("\nGenerated:")
print(OUT_JSON)

print("\nDone.")
