import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/nhtsa/CRSS/downloads/CRSS2024CSV"
OUT = ROOT / "data/schemas/nhtsa/verification/crss_2024_coded_missingness.csv"

FIELDS = {
    "accident.csv": ["PEDS", "HARM_EV", "MAN_COLL"],
    "person.csv": [
        "AGE", "SEX", "INJ_SEV", "SEAT_POS",
        "REST_USE", "REST_MIS", "HELM_USE", "HELM_MIS",
        "AIR_BAG", "EJECTION", "HOSPITAL",
        "ROLLOVER", "IMPACT1", "FIRE_EXP",
        "HARM_EV", "MAN_COLL"
    ],
    "vehicle.csv": [
        "HARM_EV", "MAN_COLL", "TRAV_SP", "ROLLOVER",
        "IMPACT1", "DEFORMED", "FIRE_EXP", "SPEEDREL",
        "VTRAFWAY", "VNUM_LAN", "VSPD_LIM", "VALIGN",
        "VPROFILE", "VSURCOND", "VTRAFCON", "ACC_TYPE"
    ],
    "vision.csv": ["VISION"],
    "weather.csv": ["WEATHER"],
}

# Common NHTSA coded unknown / not-reported values.
# These are NOT universally applicable; this is a screening pass.
SCREEN_CODES = {
    "8", "9",
    "98", "99",
    "998", "999",
    "997",
    "888", "889",
    "9999",
}

results = []

for filename, columns in FIELDS.items():

    path = DATA / filename
    df = pd.read_csv(path, usecols=columns, low_memory=False)

    for column in columns:

        series = df[column]

        missing = series.isna().sum()

        coded = 0

        for value in series.dropna():

            normalized = str(value).strip()

            if normalized in SCREEN_CODES:
                coded += 1

        total = len(series)

        results.append({
            "table": filename,
            "variable": column,
            "rows": total,
            "pandas_nan": int(missing),
            "coded_screening_values": int(coded),
            "physical_values": int(total - missing - coded),
            "nan_pct": round(missing / total * 100, 4),
            "coded_pct": round(coded / total * 100, 4),
        })

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUT,
    index=False,
    encoding="utf-8"
)

print("=" * 80)
print("CRSS 2024 CODED MISSINGNESS SCREEN")
print("=" * 80)

print(
    result_df[
        [
            "table",
            "variable",
            "pandas_nan",
            "coded_screening_values",
            "physical_values",
            "nan_pct",
            "coded_pct",
        ]
    ].to_string(index=False)
)

print("\nOutput:")
print(OUT)
