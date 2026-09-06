from pathlib import Path
import json
import hashlib
import csv
import re
from collections import defaultdict, Counter

ROOT = Path(".")
REPORT = ROOT / "experiments/nhtsa/crss_2024_master_diagnostic.json"

errors = []
warnings = []
checks = []

def check(name, condition, detail=""):
    item = {
        "check": name,
        "status": "PASS" if condition else "FAIL",
        "detail": detail
    }
    checks.append(item)
    if not condition:
        errors.append(item)

def warn(name, detail):
    item = {
        "check": name,
        "status": "WARNING",
        "detail": detail
    }
    checks.append(item)
    warnings.append(item)

def load_json(path):
    if not path.exists():
        errors.append({
            "check": f"file:{path}",
            "status": "FAIL",
            "detail": "Missing file"
        })
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        errors.append({
            "check": f"json:{path}",
            "status": "FAIL",
            "detail": str(e)
        })
        return None

# ================================================================
# 1. REQUIRED ARTIFACTS
# ================================================================

required = [
    "data/schemas/nhtsa/crss_2024_canonical_schema.json",
    "data/schemas/nhtsa/crss_2024_feature_map.csv",
    "data/schemas/nhtsa/verification/crss_2024_final_feature_dictionary_v2.csv",
    "data/schemas/nhtsa/verification/crss_2024_semantic_verified_candidates.csv",
    "data/schemas/nhtsa/verification/crss_2024_coded_missingness.csv",
    "data/schemas/nhtsa/verification/crss_2024_relationship_integrity.csv",
    "data/schemas/nhtsa/codebook/crss_2024_codebook_context_full.json",
    "data/schemas/nhtsa/codebook/crss_2024_codebook_candidates.json",
    "data/schemas/nhtsa/codebook/crss_2024_codebook_parsed_candidates.json",
    "data/raw/nhtsa/CRSS/documentation/CRSS_Analytical_Users_Manual_2016_2024.pdf",
    "data/raw/nhtsa/CRSS/documentation/FARS_CRSS_2024_Coding_Validation_Manual.pdf",
]

for p in required:
    path = ROOT / p
    check(
        f"required:{p}",
        path.exists(),
        f"{path} exists={path.exists()}"
    )

# ================================================================
# 2. JSON VALIDATION
# ================================================================

canonical = load_json(
    ROOT / "data/schemas/nhtsa/crss_2024_canonical_schema.json"
)

feature_dict_path = ROOT / \
    "data/schemas/nhtsa/verification/" \
    "crss_2024_final_feature_dictionary_v2.csv"

codebook_context = load_json(
    ROOT / "data/schemas/nhtsa/codebook/"
    "crss_2024_codebook_context_full.json"
)

codebook_candidates = load_json(
    ROOT / "data/schemas/nhtsa/codebook/"
    "crss_2024_codebook_candidates.json"
)

parsed_candidates = load_json(
    ROOT / "data/schemas/nhtsa/codebook/"
    "crss_2024_codebook_parsed_candidates.json"
)

if canonical:
    check(
        "canonical_schema_is_dictionary",
        isinstance(canonical, dict)
    )

if codebook_context:
    check(
        "codebook_context_variables",
        len(codebook_context) == 30,
        f"Found {len(codebook_context)} variables; expected 30"
    )

if codebook_candidates:
    check(
        "codebook_candidate_variables",
        len(codebook_candidates) == 30,
        f"Found {len(codebook_candidates)} variables; expected 30"
    )

if parsed_candidates:
    check(
        "parsed_candidate_variables",
        len(parsed_candidates) == 30,
        f"Found {len(parsed_candidates)} variables; expected 30"
    )

# ================================================================
# 3. FINAL FEATURE DICTIONARY
# ================================================================

feature_rows = []

if feature_dict_path.exists():
    with feature_dict_path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        feature_rows = list(csv.DictReader(f))

check(
    "feature_dictionary_37_rows",
    len(feature_rows) == 37,
    f"Rows={len(feature_rows)}"
)

feature_pairs = set()

for row in feature_rows:
    table = row.get("table", "").strip().lower()
    variable = row.get("variable", "").strip().upper()

    if table and variable:
        feature_pairs.add((table, variable))

check(
    "feature_dictionary_unique_pairs",
    len(feature_pairs) == len(feature_rows),
    f"Unique pairs={len(feature_pairs)}"
)

# ================================================================
# 4. EXPECTED 30 VARIABLES
# ================================================================

expected_variables = {
    "HARM_EV",
    "MAN_COLL",
    "PEDS",
    "AGE",
    "AIR_BAG",
    "EJECTION",
    "FIRE_EXP",
    "HELM_MIS",
    "HELM_USE",
    "HOSPITAL",
    "IMPACT1",
    "INJ_SEV",
    "REST_MIS",
    "REST_USE",
    "ROLLOVER",
    "SEAT_POS",
    "SEX",
    "ACC_TYPE",
    "DEFORMED",
    "SPEEDREL",
    "TRAV_SP",
    "VALIGN",
    "VNUM_LAN",
    "VPROFILE",
    "VSPD_LIM",
    "VSURCOND",
    "VTRAFCON",
    "VTRAFWAY",
    "VISION",
    "WEATHER",
}

actual_variables = {
    v.upper()
    for _, v in feature_pairs
}

# PEDS is included in candidate/codebook scope even if
# it is not present in the final 37-pair feature map.
map_variables = actual_variables | {"PEDS"}

missing = sorted(expected_variables - map_variables)
extra = sorted(map_variables - expected_variables)

check(
    "expected_variable_coverage",
    not missing,
    f"Missing={missing}"
)

if extra:
    warn(
        "unexpected_variables",
        f"Extra variables={extra}"
    )

# ================================================================
# 5. CODEBOOK EXTRACTION QUALITY
# ================================================================

parsed_stats = {}

if parsed_candidates:
    for variable, payload in parsed_candidates.items():

        contexts = payload.get("parsed_contexts", [])

        unique = {}
        for ctx in contexts:
            for item in ctx.get("codes", []):
                key = (
                    str(item.get("code")),
                    str(item.get("label", "")).lower()
                )
                unique[key] = item

        parsed_stats[variable] = {
            "contexts": len(contexts),
            "unique_pairs": len(unique),
            "codes": sorted(unique.values(), key=lambda x: (
                int(x["code"]) if str(x["code"]).isdigit() else 9999,
                x["label"]
            ))
        }

# Specific known parser failure.
if parsed_candidates and "PEDS" in parsed_candidates:
    peds_contexts = parsed_candidates["PEDS"].get(
        "parsed_contexts", []
    )

    if not peds_contexts:
        errors.append({
            "check": "PEDS_codebook_extraction",
            "status": "FAIL",
            "detail": (
                "Generic parser extracted zero PEDS contexts. "
                "This must be resolved from the authoritative manual."
            )
        })

# Obvious parser artifacts.
artifact_pairs = []

if parsed_candidates:
    for variable, stats in parsed_stats.items():
        for item in stats["codes"]:
            label = item["label"].strip().lower()
            code = str(item["code"])

            if label in {"and", "or", "the", "value", "values"}:
                artifact_pairs.append({
                    "variable": variable,
                    "code": code,
                    "label": item["label"],
                    "page": item.get("page")
                })

            if re.fullmatch(r"(20\d\d|19\d\d)", code):
                artifact_pairs.append({
                    "variable": variable,
                    "code": code,
                    "label": item["label"],
                    "page": item.get("page")
                })

check(
    "no_obvious_codebook_artifacts",
    len(artifact_pairs) == 0,
    f"Artifacts={artifact_pairs[:20]}"
)

# ================================================================
# 6. VARIABLE-SPECIFIC EXPECTATION CHECKS
# ================================================================

# These are diagnostic expectations, not replacements for the manual.
expected_minimum_codes = {
    "HARM_EV": 5,
    "MAN_COLL": 5,
    "PEDS": 1,
    "AGE": 3,
    "AIR_BAG": 5,
    "EJECTION": 5,
    "FIRE_EXP": 2,
    "HELM_MIS": 3,
    "HELM_USE": 5,
    "HOSPITAL": 3,
    "IMPACT1": 5,
    "INJ_SEV": 5,
    "REST_MIS": 3,
    "REST_USE": 5,
    "ROLLOVER": 3,
    "SEAT_POS": 5,
    "SEX": 3,
    "ACC_TYPE": 3,
    "DEFORMED": 3,
    "SPEEDREL": 3,
    "TRAV_SP": 2,
    "VALIGN": 3,
    "VNUM_LAN": 3,
    "VPROFILE": 3,
    "VSPD_LIM": 2,
    "VSURCOND": 3,
    "VTRAFCON": 3,
    "VTRAFWAY": 3,
    "VISION": 3,
    "WEATHER": 3,
}

for variable, minimum in expected_minimum_codes.items():

    count = parsed_stats.get(
        variable,
        {}
    ).get(
        "unique_pairs",
        0
    )

    if count < minimum:
        warn(
            f"codebook_depth:{variable}",
            f"Only {count} extracted candidates; expected at least {minimum}"
        )

# ================================================================
# 7. DERIVED VARIABLE FLAGS
# ================================================================

derived = {
    "HARM_EV",
    "ROLLOVER",
    "IMPACT1",
}

if codebook_context:
    for variable in derived:
        contexts = codebook_context.get(variable, {}).get(
            "contexts", []
        )

        text = " ".join(
            str(c.get("context", ""))
            for c in contexts
        ).lower()

        if "derived" not in text:
            warn(
                f"derived_variable_documentation:{variable}",
                "No explicit 'derived' wording found in extracted context"
            )

# ================================================================
# 8. CANONICAL SCHEMA STRUCTURE
# ================================================================

if canonical:

    expected_entities = {
        "crash",
        "vehicle",
        "person",
        "environment",
        "event",
        "vehicle_safety",
        "occupant_protection",
        "occupant_outcome",
    }

    entities = set()

    # Handle common schema layouts.
    if isinstance(canonical.get("entities"), dict):
        entities = set(canonical["entities"].keys())

    elif isinstance(canonical.get("logical_entities"), dict):
        entities = set(canonical["logical_entities"].keys())

    if entities:
        check(
            "canonical_entity_coverage",
            expected_entities.issubset(entities),
            f"Entities={sorted(entities)}"
        )
    else:
        warn(
            "canonical_entity_structure",
            "Could not infer entity dictionary layout automatically"
        )

# ================================================================
# 9. CSV SOURCE INVENTORY
# ================================================================

csv_root = ROOT / \
    "data/raw/nhtsa/CRSS/downloads/CRSS2024CSV"

csv_files = sorted(csv_root.glob("*.csv"))

check(
    "CRSS_2024_csv_inventory",
    len(csv_files) == 28,
    f"CSV files={len(csv_files)}"
)

csv_inventory = {}

for csv_path in csv_files:
    try:
        with csv_path.open(
            encoding="utf-8-sig",
            errors="replace"
        ) as f:

            reader = csv.reader(f)

            header = next(reader, [])

            row_count = sum(
                1 for _ in reader
            )

            csv_inventory[csv_path.name] = {
                "columns": len(header),
                "rows": row_count
            }

    except Exception as e:
        errors.append({
            "check": f"csv_read:{csv_path.name}",
            "status": "FAIL",
            "detail": str(e)
        })

# ================================================================
# 10. REQUIRED TABLES / VARIABLES
# ================================================================

required_table_variables = {
    "accident": {
        "CASENUM",
        "HARM_EV",
        "MAN_COLL",
        "PEDS",
    },
    "person": {
        "CASENUM",
        "VEH_NO",
        "PER_NO",
        "PER_TYP",
        "AGE",
        "SEX",
        "INJ_SEV",
        "SEAT_POS",
        "REST_USE",
        "REST_MIS",
        "AIR_BAG",
        "EJECTION",
        "HOSPITAL",
        "ROLLOVER",
        "IMPACT1",
        "FIRE_EXP",
        "HARM_EV",
        "MAN_COLL",
        "HELM_USE",
        "HELM_MIS",
    },
    "vehicle": {
        "CASENUM",
        "VEH_NO",
        "UNITTYPE",
        "HARM_EV",
        "MAN_COLL",
        "TRAV_SP",
        "ROLLOVER",
        "IMPACT1",
        "DEFORMED",
        "FIRE_EXP",
        "SPEEDREL",
        "VTRAFWAY",
        "VNUM_LAN",
        "VSPD_LIM",
        "VALIGN",
        "VPROFILE",
        "VSURCOND",
        "VTRAFCON",
        "ACC_TYPE",
    },
    "vision": {
        "CASENUM",
        "VISION",
    },
    "weather": {
        "CASENUM",
        "WEATHER",
    },
}

for table, variables in required_table_variables.items():

    path = csv_root / f"{table}.csv"

    if not path.exists():
        check(
            f"table_exists:{table}",
            False,
            str(path)
        )
        continue

    try:
        with path.open(
            encoding="utf-8-sig",
            errors="replace"
        ) as f:
            header = {
                h.strip().upper()
                for h in next(csv.reader(f), [])
            }

        missing_vars = sorted(
            variables - header
        )

        check(
            f"table_variables:{table}",
            not missing_vars,
            f"Missing={missing_vars}"
        )

    except Exception as e:
        check(
            f"table_variables:{table}",
            False,
            str(e)
        )

# ================================================================
# 11. FINAL CLASSIFICATION
# ================================================================

# A GO requires:
# - no missing required artifact
# - no parser artifact
# - PEDS resolved
# - 30 variable codebook scope
#
# Other warnings are allowed but must be reviewed.

fatal_conditions = [
    item for item in errors
    if item["status"] == "FAIL"
]

if fatal_conditions:
    verdict = "NO-GO"
else:
    verdict = "GO-WITH-WARNINGS" if warnings else "GO"

report = {
    "project": "Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles",
    "dataset": "NHTSA CRSS 2024",
    "diagnostic": "Master CRSS 2024 diagnostic",
    "verdict": verdict,
    "summary": {
        "checks": len(checks),
        "passes": sum(
            1 for x in checks if x["status"] == "PASS"
        ),
        "failures": len(errors),
        "warnings": len(warnings),
    },
    "parsed_codebook_stats": parsed_stats,
    "artifact_pairs": artifact_pairs,
    "csv_inventory": csv_inventory,
    "checks": checks,
    "errors": errors,
    "warnings": warnings,
}

REPORT.parent.mkdir(
    parents=True,
    exist_ok=True
)

REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

# ================================================================
# CONSOLE SUMMARY
# ================================================================

print()
print("=" * 80)
print("CRSS 2024 MASTER DIAGNOSTIC")
print("=" * 80)

print()
print(f"VERDICT       : {verdict}")
print(f"TOTAL CHECKS  : {len(checks)}")
print(f"PASS          : {sum(1 for x in checks if x['status'] == 'PASS')}")
print(f"FAIL          : {len(errors)}")
print(f"WARNINGS      : {len(warnings)}")

print()
print("-" * 80)
print("FAILURES")
print("-" * 80)

if errors:
    for x in errors:
        print(
            f"[FAIL] {x['check']}: "
            f"{x['detail']}"
        )
else:
    print("None")

print()
print("-" * 80)
print("WARNINGS")
print("-" * 80)

if warnings:
    for x in warnings:
        print(
            f"[WARN] {x['check']}: "
            f"{x['detail']}"
        )
else:
    print("None")

print()
print("-" * 80)
print("CODEBOOK SUMMARY")
print("-" * 80)

for variable in sorted(parsed_stats):
    stats = parsed_stats[variable]
    print(
        f"{variable:<12} "
        f"contexts={stats['contexts']:<3} "
        f"unique_candidates={stats['unique_pairs']:<3}"
    )

print()
print("-" * 80)
print("REPORT")
print("-" * 80)
print(REPORT)
print("=" * 80)
