import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/nhtsa/CRSS/downloads/CRSS2024CSV"

FILES = [
    "accident.csv",
    "vehicle.csv",
    "person.csv",
]

print("=" * 80)
print("CRSS 2024 RELATIONAL SCHEMA INSPECTION")
print("=" * 80)

for filename in FILES:

    path = DATA / filename

    df = pd.read_csv(
        path,
        nrows=5,
        low_memory=False
    )

    print(f"\n{'=' * 80}")
    print(filename)
    print(f"Columns: {len(df.columns)}")
    print("=" * 80)

    for i, column in enumerate(df.columns, start=1):
        print(f"{i:>3}. {column}")
