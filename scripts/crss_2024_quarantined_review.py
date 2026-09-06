import pymupdf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PDF = ROOT / "data/raw/nhtsa/CRSS/documentation/FARS_CRSS_2024_Coding_Validation_Manual.pdf"
OUT = ROOT / "data/schemas/nhtsa/verification/crss_2024_quarantined_manual_pages.txt"

pages = {
    103: "HARM_EV",
    272: "IMPACT1",
    440: "VTRAFCON",
    523: "SEAT_POS",
}

doc = pymupdf.open(PDF)

with OUT.open("w", encoding="utf-8") as f:
    f.write("CRSS 2024 QUARANTINED FIELD SEMANTIC REVIEW\n")
    f.write("=" * 80 + "\n\n")

    for page_num, variable in pages.items():
        text = doc[page_num - 1].get_text()

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"VARIABLE: {variable}\n")
        f.write(f"MANUAL PAGE: {page_num}\n")
        f.write("=" * 80 + "\n\n")
        f.write(text)
        f.write("\n")

doc.close()

print("Created:")
print(OUT)
