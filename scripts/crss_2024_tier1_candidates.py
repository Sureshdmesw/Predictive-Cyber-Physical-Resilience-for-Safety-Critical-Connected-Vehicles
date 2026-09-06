from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "schemas"
    / "nhtsa"
    / "crss_2024_candidate_variables.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "schemas"
    / "nhtsa"
    / "crss_2024_tier1_candidates.csv"
)

# High-value variables for the project's physical-safety /
# occupant-risk / vehicle-behavior / ADAS / event-sequence layers.

TIER1 = {
    # Crash context
    "PEDS",
    "HARM_EV",
    "EVENT1_IM",
    "MAN_COLL",
    "NUM_INJ",
    "NO_INJ_IM",

    # Vehicle dynamics / roadway
    "TRAV_SP",
    "SPEEDREL",
    "VTRAFWAY",
    "VNUM_LAN",
    "VSPD_LIM",
    "VALIGN",
    "VPROFILE",
    "VSURCOND",
    "VTRAFCON",
    "ACC_TYPE",

    # Vehicle safety state
    "ROLLOVER",
    "IMPACT1",
    "DEFORMED",
    "FIRE_EXP",
    "NUM_INJV",

    # Occupant state
    "AGE",
    "SEX",
    "INJ_SEV",
    "SEAT_POS",
    "REST_USE",
    "REST_MIS",
    "HELM_USE",
    "HELM_MIS",
    "AIR_BAG",
    "EJECTION",
    "EJECT_IM",
    "HOSPITAL",

    # Environment
    "VISION",
    "WEATHER",

    # Event sequence
    "EVENTNUM",
    "VEVENTNUM",
    "VNUMBER1",
    "VNUMBER2",
    "AOI",
    "SOE",

    # ADAS / vehicle technology
    "FORWARDCOLLISIONWARNING",
    "DYNAMICBRAKESUPPORT",
    "PEDESTRIANAUTOEMERGENCYBRAKING",
    "BLINDSPOTWARNING",
    "LANEDEPARTUREWARNING",
    "LANEKEEPINGASSISTANCE",
    "LANECENTERINGASSISTANCE",
    "ADAPTIVECRUISECONTROL",
    "ANTILOCKBRAKESYSTEM",
    "ELECTRONICSTABILITYCONTROL",
    "EVENTDATARECORDER",
    "TRACTIONCONTROL",
    "AUTOPEDESTRIANALERTINGSOUND",
}

with INPUT.open(
    "r",
    encoding="utf-8",
    newline=""
) as handle:
    rows = list(csv.DictReader(handle))

selected = []

for row in rows:
    variable = row["variable"].upper()

    if variable in TIER1:
        row["tier"] = "Tier-1"
        row["verification_priority"] = "HIGH"
        selected.append(row)

fields = list(selected[0].keys())

with OUTPUT.open(
    "w",
    encoding="utf-8",
    newline=""
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(selected)

print()
print("=" * 80)
print("CRSS 2024 TIER-1 CANDIDATES")
print("=" * 80)
print()
print(f"Candidate variables: {len(selected)}")
print(f"Output: {OUTPUT}")
print()

for row in selected:
    print(
        f"{row['table']:25} "
        f"{row['variable']:45} "
        f"{row['role']}"
    )

print()
print("DONE")
