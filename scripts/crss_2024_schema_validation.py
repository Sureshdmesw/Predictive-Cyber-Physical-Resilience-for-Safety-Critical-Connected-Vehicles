import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/nhtsa/CRSS/downloads/CRSS2024CSV"

CHECKS = {
    "accident.csv": [
        "PEDS", "HARM_EV", "MAN_COLL"
    ],
    "person.csv": [
        "AGE", "SEX", "INJ_SEV", "SEAT_POS",
        "REST_USE", "REST_MIS",
        "HELM_USE", "HELM_MIS",
        "AIR_BAG", "EJECTION", "HOSPITAL",
        "ROLLOVER", "IMPACT1", "FIRE_EXP",
        "HARM_EV", "MAN_COLL"
    ],
    "vehicle.csv": [
        "HARM_EV", "MAN_COLL", "TRAV_SP",
        "ROLLOVER", "IMPACT1", "DEFORMED",
        "FIRE_EXP", "SPEEDREL",
        "VTRAFWAY", "VNUM_LAN", "VSPD_LIM",
        "VALIGN", "VPROFILE", "VSURCOND",
        "VTRAFCON", "ACC_TYPE"
    ],
    "vision.csv": ["VISION"],
    "weather.csv": ["WEATHER"],
}

print("=" * 80)
print("CRSS 2024 SCHEMA VALIDATION")
print("=" * 80)

all_results = []

for filename, expected_columns in CHECKS.items():
    path = DATA / filename

    if not path.exists():
        print(f"\nERROR: Missing file: {path}")
        continue

    df = pd.read_csv(path, nrows=0)

    print(f"\n[{filename}]")
    print(f"Columns in CSV: {len(df.columns)}")

    for column in expected_columns:
        exists = column in df.columns
        status = "OK" if exists else "MISSING"

        print(f"  {column:<12} {status}")

        all_results.append({
            "table": filename,
            "variable": column,
            "exists": exists,
        })

print("\n" + "=" * 80)
print("SCHEMA SUMMARY")
print("=" * 80)

total = len(all_results)
found = sum(r["exists"] for r in all_results)
missing = total - found

print(f"Expected fields checked: {total}")
print(f"Fields found:            {found}")
print(f"Fields missing:          {missing}")

if missing == 0:
    print("\nRESULT: PASS — all expected CRSS fields exist.")
else:
    print("\nRESULT: FAIL — missing fields detected.")

print("=" * 80)
