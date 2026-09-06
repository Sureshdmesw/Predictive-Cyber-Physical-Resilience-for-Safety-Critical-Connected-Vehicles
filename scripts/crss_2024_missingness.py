import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/nhtsa/CRSS/downloads/CRSS2024CSV"
OUT = ROOT / "data/schemas/nhtsa/verification/crss_2024_missingness.csv"

FIELDS = {
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

results = []

print("=" * 80)
print("CRSS 2024 MISSINGNESS ANALYSIS")
print("=" * 80)

for filename, columns in FIELDS.items():

    path = DATA / filename

    print(f"\nLoading {filename}...")

    df = pd.read_csv(
        path,
        usecols=columns,
        low_memory=False
    )

    print(f"Rows: {len(df):,}")

    for column in columns:

        missing = int(df[column].isna().sum())
        nonmissing = len(df) - missing

        missing_pct = (
            missing / len(df) * 100
            if len(df) > 0 else 0
        )

        results.append({
            "table": filename,
            "variable": column,
            "rows": len(df),
            "missing": missing,
            "nonmissing": nonmissing,
            "missing_pct": round(missing_pct, 4),
        })

        print(
            f"  {column:<12} "
            f"missing={missing:>8,} "
            f"({missing_pct:>7.2f}%)"
        )

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    ["table", "missing_pct"],
    ascending=[True, False]
)

result_df.to_csv(
    OUT,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)
print(f"Fields analyzed: {len(result_df)}")
print(f"Output: {OUT}")
