from pathlib import Path
import json
import csv
import re
from collections import defaultdict

ROOT = Path(".")
MANUAL = ROOT / "data/raw/nhtsa/CRSS/documentation/CRSS_Analytical_Users_Manual_2016_2024.pdf"

OUTDIR = ROOT / "data/schemas/nhtsa/codebook"
REPORTDIR = ROOT / "experiments/nhtsa"

OUTDIR.mkdir(parents=True, exist_ok=True)
REPORTDIR.mkdir(parents=True, exist_ok=True)

AUTHORITATIVE_JSON = OUTDIR / "crss_2024_authoritative_codebook.json"
AUDIT_CSV = OUTDIR / "crss_2024_codebook_audit.csv"
REPORT_JSON = REPORTDIR / "crss_2024_one_shot_diagnostic.json"

VARIABLES = [
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
]

DERIVED = {
    "HARM_EV",
    "ROLLOVER",
    "IMPACT1",
}

errors = []
warnings = []
checks = []

def check(name, ok, detail=""):
    record = {
        "name": name,
        "status": "PASS" if ok else "FAIL",
        "detail": detail,
    }
    checks.append(record)
    if not ok:
        errors.append(record)

def warning(name, detail=""):
    record = {
        "name": name,
        "status": "WARNING",
        "detail": detail,
    }
    checks.append(record)
    warnings.append(record)

# ================================================================
# 1. BASIC FILE CHECKS
# ================================================================

check(
    "analytical_manual_exists",
    MANUAL.exists(),
    str(MANUAL)
)

if not MANUAL.exists():
    raise SystemExit("Analytical manual not found.")

# ================================================================
# 2. EXTRACT PDF
# ================================================================

try:
    from pypdf import PdfReader

    reader = PdfReader(str(MANUAL))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = text.replace("\x00", " ")
        text = re.sub(r"\s+", " ", text).strip()

        pages.append({
            "page": i,
            "text": text,
        })

    check(
        "manual_page_count",
        len(pages) >= 400,
        f"pages={len(pages)}"
    )

except Exception as e:
    check(
        "pdf_extraction",
        False,
        str(e)
    )
    raise

# ================================================================
# 3. FIND VARIABLE-SPECIFIC PAGES
# ================================================================

variable_pages = defaultdict(list)

for item in pages:

    text = item["text"]
    page_no = item["page"]

    for variable in VARIABLES:

        # Exact SAS Name match.
        if re.search(
            rf"\bSAS\s+Name\s+{re.escape(variable)}\b",
            text,
            re.I
        ):
            variable_pages[variable].append(page_no)

        # PEDS sometimes has formatting/context that does not
        # survive generic SAS-name matching.
        elif variable == "PEDS" and re.search(
            r"\bPEDS\b",
            text,
            re.I
        ):
            if re.search(
                r"Number of Persons Not in Motor Vehicles",
                text,
                re.I
            ):
                variable_pages[variable].append(page_no)

for variable in VARIABLES:

    check(
        f"manual_variable:{variable}",
        len(variable_pages[variable]) > 0,
        f"pages={variable_pages[variable][:20]}"
    )

# ================================================================
# 4. IDENTIFY CODE-TABLE CONTEXT
# ================================================================

def clean(text):
    text = text.replace("â€“", "-")
    text = text.replace("â€”", "-")
    text = text.replace("â€™", "'")
    text = text.replace("â€œ", '"')
    text = text.replace("â€", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_code_context(text, variable):

    text = clean(text)

    if variable == "PEDS":
        return (
            "Number of Persons Not in Motor Vehicles" in text
            or "SAS Name PEDS" in text
        )

    return bool(
        re.search(
            rf"\bSAS\s+Name\s+{re.escape(variable)}\b",
            text,
            re.I
        )
        and
        re.search(
            r"Attribute Codes",
            text,
            re.I
        )
    )

# ================================================================
# 5. EXTRACT CODE ROWS
# ================================================================

def extract_code_rows(text):

    text = clean(text)

    rows = []

    # ------------------------------------------------------------
    # IMPORTANT:
    # Do NOT use one giant nested regex here.
    #
    # The CRSS PDF contains historical year columns, prose,
    # appendix text, and table values on the same extracted line.
    # We therefore use two small, independently validated patterns.
    # ------------------------------------------------------------

    patterns = [

        # Normal code + descriptive label.
        re.compile(
            r"(?<![\w.-])"
            r"(\d{1,3})\s+"
            r"([A-Z][A-Za-z0-9 ,()/&.'?\-??]{2,120}?)"
            r"(?=\s+\d{1,3}\s+|\s*$)",
            re.I
        ),

        # Common CRSS special-value labels.
        re.compile(
            r"(?<![\w.-])"
            r"(\d{1,3})\s+"
            r"(Not Reported|"
            r"Reported as Unknown|"
            r"Not Applicable|"
            r"Unknown|"
            r"Yes|"
            r"No|"
            r"Less Than One Year|"
            r"Age in Years)",
            re.I
        ),
    ]

    bad_codes = {
        "1900", "1901", "1902", "1903", "1904", "1905",
        "1906", "1907", "1908", "1909",
        "1910", "1911", "1912", "1913", "1914", "1915",
        "1916", "1917", "1918", "1919",
        "1920", "1921", "1922", "1923", "1924", "1925",
        "1926", "1927", "1928", "1929",
        "1930", "1931", "1932", "1933", "1934", "1935",
        "1936", "1937", "1938", "1939",
        "1940", "1941", "1942", "1943", "1944", "1945",
        "1946", "1947", "1948", "1949",
        "1950", "1951", "1952", "1953", "1954", "1955",
        "1956", "1957", "1958", "1959",
        "1960", "1961", "1962", "1963", "1964", "1965",
        "1966", "1967", "1968", "1969",
        "1970", "1971", "1972", "1973", "1974", "1975",
        "1976", "1977", "1978", "1979",
        "1980", "1981", "1982", "1983", "1984", "1985",
        "1986", "1987", "1988", "1989",
        "1990", "1991", "1992", "1993", "1994", "1995",
        "1996", "1997", "1998", "1999",
        "2000", "2001", "2002", "2003", "2004", "2005",
        "2006", "2007", "2008", "2009",
        "2010", "2011", "2012", "2013", "2014", "2015",
        "2016", "2017", "2018", "2019",
        "2020", "2021", "2022", "2023", "2024",
    }

    bad_labels = {
        "and",
        "or",
        "the",
        "value",
        "values",
        "attribute",
        "attributes",
        "codes",
        "code",
        "definition",
        "additional information",
    }

    def clean_label(label):

        label = clean(label)

        label = re.sub(
            r"^(and|or|the)\s+",
            "",
            label,
            flags=re.I
        )

        label = label.strip(
            " -??:;,.|"
        )

        return label

    def valid(code, label):

        label = clean_label(label)

        if code in bad_codes:
            return False

        if len(label) < 3:
            return False

        if label.lower() in bad_labels:
            return False

        # Reject obvious prose fragments.
        if label.lower().startswith(
            (
                "and ",
                "or ",
                "the ",
                "this data element",
                "see appendix",
                "see the ",
            )
        ):
            return False

        # Reject labels consisting only of punctuation/numbers.
        if not re.search(
            r"[A-Za-z]",
            label
        ):
            return False

        return True

    # ------------------------------------------------------------
    # Extract normal rows.
    # ------------------------------------------------------------

    for pattern in patterns:

        for match in pattern.finditer(text):

            code = match.group(1)
            label = clean_label(match.group(2))

            if not valid(code, label):
                continue

            rows.append({
                "code": code,
                "label": label,
            })

    # ------------------------------------------------------------
    # Explicit CRSS range handling.
    #
    # AGE is 1-120, and this must not become 120 separate codes.
    # ------------------------------------------------------------

    range_patterns = [
        (
            re.compile(
                r"1-120\s+Age\s+in\s+Years",
                re.I
            ),
            "1-120",
            "Age in Years"
        ),
    ]

    for pattern, code, label in range_patterns:

        if pattern.search(text):

            rows.append({
                "code": code,
                "label": label,
            })

    # ------------------------------------------------------------
    # Deduplicate.
    # ------------------------------------------------------------

    unique = []
    seen = set()

    for row in rows:

        key = (
            row["code"],
            row["label"].lower()
        )

        if key not in seen:

            seen.add(key)
            unique.append(row)

    return unique


# ================================================================
# 6. BUILD AUTHORITATIVE CANDIDATES
# ================================================================

authoritative = {}

for variable in VARIABLES:

    contexts = []

    for page_no in variable_pages[variable]:

        text = pages[page_no - 1]["text"]

        if not is_code_context(text, variable):
            continue

        codes = extract_code_rows(text)

        if codes:

            contexts.append({
                "page": page_no,
                "codes": codes,
                "context": text,
            })

    # PEDS fallback:
    # use the page containing the actual PEDS definition even
    # if the generic Attribute Codes phrase was not extracted.
    if variable == "PEDS" and not contexts:

        for page_no in variable_pages[variable]:

            text = clean(
                pages[page_no - 1]["text"]
            )

            if (
                "Number of Persons Not in Motor Vehicles"
                in text
            ):

                # PEDS is a count, so preserve its range rather
                # than inventing individual values.
                contexts.append({
                    "page": page_no,
                    "codes": [
                        {
                            "code": "0-99",
                            "label": (
                                "Number of Persons Not in "
                                "Motor Vehicles"
                            ),
                        }
                    ],
                    "context": text,
                    "extraction_method": "PEDS_range_definition",
                })

                break

    authoritative[variable] = {
        "variable": variable,
        "derived": variable in DERIVED,
        "contexts": contexts,
    }

# ================================================================
# 7. CODEBOOK QUALITY CHECKS
# ================================================================

for variable in VARIABLES:

    contexts = authoritative[variable]["contexts"]

    if not contexts:
        check(
            f"authoritative_context:{variable}",
            False,
            "No authoritative code-table context found."
        )
        continue

    unique = {}

    for context in contexts:
        for code in context["codes"]:
            key = (
                code["code"],
                code["label"].lower()
            )
            unique[key] = code

    if len(unique) == 0:
        check(
            f"authoritative_codes:{variable}",
            False,
            "Context found but no codes extracted."
        )
    else:
        check(
            f"authoritative_codes:{variable}",
            True,
            f"unique_codes={len(unique)}"
        )

# ================================================================
# 8. EXPLICIT DERIVATION EVIDENCE
# ================================================================

for variable in DERIVED:

    all_text = " ".join(
        c["context"]
        for c in authoritative[variable]["contexts"]
    ).lower()

    derived_evidence = any(
        phrase in all_text
        for phrase in [
            "logic of derivation",
            "this data element is derived",
            "derived from",
            "rules for derived data elements",
        ]
    )

    if not derived_evidence:

        # Search the entire manual for the variable's derivation.
        full_text = " ".join(
            p["text"] for p in pages
        ).lower()

        derived_evidence = (
            variable.lower() in full_text
            and
            "logic of derivation" in full_text
        )

    check(
        f"derived_documentation:{variable}",
        derived_evidence,
        "Derived-variable evidence found."
        if derived_evidence
        else "No explicit derivation evidence found."
    )

# ================================================================
# 9. ARTIFACT CHECK
# ================================================================

artifacts = []

for variable, payload in authoritative.items():

    for context in payload["contexts"]:

        for code in context["codes"]:

            if code["label"].strip().lower() in {
                "and",
                "or",
                "the",
                "value",
                "values",
            }:

                artifacts.append({
                    "variable": variable,
                    "page": context["page"],
                    "code": code["code"],
                    "label": code["label"],
                })

check(
    "authoritative_no_obvious_artifacts",
    len(artifacts) == 0,
    f"artifacts={artifacts}"
)

# ================================================================
# 10. 2024 CODEBOOK AUDIT CSV
# ================================================================

audit_rows = []

for variable, payload in authoritative.items():

    seen = set()

    for context in payload["contexts"]:

        for code in context["codes"]:

            key = (
                code["code"],
                code["label"].lower()
            )

            if key in seen:
                continue

            seen.add(key)

            audit_rows.append({
                "variable": variable,
                "code": code["code"],
                "label": code["label"],
                "page": context["page"],
                "derived_variable": payload["derived"],
            })

with AUDIT_CSV.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "variable",
            "code",
            "label",
            "page",
            "derived_variable",
        ]
    )

    writer.writeheader()
    writer.writerows(audit_rows)

# ================================================================
# 11. WRITE AUTHORITATIVE JSON
# ================================================================

AUTHORITATIVE_JSON.write_text(
    json.dumps(
        authoritative,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

# ================================================================
# 12. CSV DATASET VALIDATION
# ================================================================

csv_root = (
    ROOT /
    "data/raw/nhtsa/CRSS/downloads/CRSS2024CSV"
)

csv_files = sorted(csv_root.glob("*.csv"))

check(
    "CRSS_2024_28_csv_files",
    len(csv_files) == 28,
    f"csv_files={len(csv_files)}"
)

required_headers = {

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

for table, required in required_headers.items():

    path = csv_root / f"{table}.csv"

    if not path.exists():
        check(
            f"table_exists:{table}",
            False,
            str(path)
        )
        continue

    with path.open(
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as f:

        reader_csv = csv.reader(f)
        header = {
            x.strip().upper()
            for x in next(reader_csv, [])
        }

    missing = sorted(
        required - header
    )

    check(
        f"headers:{table}",
        not missing,
        f"missing={missing}"
    )

# ================================================================
# 13. FEATURE DICTIONARY VALIDATION
# ================================================================

feature_path = (
    ROOT /
    "data/schemas/nhtsa/verification/"
    "crss_2024_final_feature_dictionary_v2.csv"
)

if feature_path.exists():

    with feature_path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:

        feature_rows = list(
            csv.DictReader(f)
        )

    check(
        "feature_dictionary_37_rows",
        len(feature_rows) == 37,
        f"rows={len(feature_rows)}"
    )

    pairs = {
        (
            r.get("table", "").strip().lower(),
            r.get("variable", "").strip().upper()
        )
        for r in feature_rows
    }

    check(
        "feature_dictionary_unique_pairs",
        len(pairs) == len(feature_rows),
        f"unique_pairs={len(pairs)}"
    )

else:

    check(
        "feature_dictionary_exists",
        False,
        str(feature_path)
    )

# ================================================================
# 14. RELATIONSHIP-INTEGRITY ARTIFACT
# ================================================================

relationship_path = (
    ROOT /
    "data/schemas/nhtsa/verification/"
    "crss_2024_relationship_integrity.csv"
)

if relationship_path.exists():

    with relationship_path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:

        relationship_rows = list(
            csv.DictReader(f)
        )

    check(
        "relationship_integrity_report_exists",
        len(relationship_rows) > 0,
        f"rows={len(relationship_rows)}"
    )

else:

    warning(
        "relationship_integrity_report_missing",
        str(relationship_path)
    )

# ================================================================
# 15. SEMANTIC FEATURE DICTIONARY
# ================================================================

semantic_path = (
    ROOT /
    "data/schemas/nhtsa/verification/"
    "crss_2024_final_feature_dictionary_v2.csv"
)

if semantic_path.exists():

    with semantic_path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    review_rows = []

    for row in rows:

        text = " ".join(
            str(v)
            for v in row.values()
        ).lower()

        if "review" in text:
            review_rows.append(row)

    check(
        "semantic_dictionary_no_review_rows",
        len(review_rows) == 0,
        f"review_rows={len(review_rows)}"
    )

# ================================================================
# 16. CANONICAL SCHEMA JSON
# ================================================================

schema_path = (
    ROOT /
    "data/schemas/nhtsa/"
    "crss_2024_canonical_schema.json"
)

if schema_path.exists():

    try:

        schema = json.loads(
            schema_path.read_text(
                encoding="utf-8"
            )
        )

        check(
            "canonical_schema_valid_json",
            isinstance(schema, dict)
        )

    except Exception as e:

        check(
            "canonical_schema_valid_json",
            False,
            str(e)
        )

else:

    check(
        "canonical_schema_exists",
        False,
        str(schema_path)
    )

# ================================================================
# 17. FINAL VERDICT
# ================================================================

fatal = [
    x for x in errors
    if x["status"] == "FAIL"
]

if fatal:
    verdict = "NO-GO"
elif warnings:
    verdict = "GO-WITH-WARNINGS"
else:
    verdict = "GO"

report = {
    "project": (
        "Predictive Cyber-Physical Resilience "
        "for Safety-Critical Connected Vehicles"
    ),
    "dataset": "NHTSA CRSS 2024",
    "diagnostic": "One-shot authoritative codebook + pipeline diagnostic",
    "verdict": verdict,

    "summary": {
        "checks": len(checks),
        "pass": sum(
            1 for x in checks
            if x["status"] == "PASS"
        ),
        "fail": len(errors),
        "warnings": len(warnings),
    },

    "manual": {
        "pages": len(pages),
        "variable_pages": dict(variable_pages),
    },

    "authoritative_codebook": {
        "variables": len(authoritative),
        "variables_with_context": sum(
            1 for x in authoritative.values()
            if x["contexts"]
        ),
        "audit_rows": len(audit_rows),
    },

    "artifacts": artifacts,

    "errors": errors,
    "warnings": warnings,

    "checks": checks,
}

REPORT_JSON.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

# ================================================================
# CONSOLE OUTPUT
# ================================================================

print()
print("=" * 80)
print("CRSS 2024 ONE-SHOT AUTHORITATIVE DIAGNOSTIC")
print("=" * 80)

print()
print("VERDICT :", verdict)
print("CHECKS  :", len(checks))
print("PASS    :", sum(
    1 for x in checks
    if x["status"] == "PASS"
))
print("FAIL    :", len(errors))
print("WARN    :", len(warnings))

print()
print("-" * 80)
print("VARIABLE COVERAGE")
print("-" * 80)

for variable in VARIABLES:

    payload = authoritative[variable]

    unique = {
        (
            x["code"],
            x["label"].lower()
        )
        for c in payload["contexts"]
        for x in c["codes"]
    }

    print(
        f"{variable:<12}"
        f"contexts={len(payload['contexts']):<3}"
        f"codes={len(unique):<3}"
        f"derived={payload['derived']}"
    )

print()
print("-" * 80)
print("FAILURES")
print("-" * 80)

if errors:
    for x in errors:
        print(
            "[FAIL]",
            x["name"],
            ":",
            x["detail"]
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
            "[WARN]",
            x["name"],
            ":",
            x["detail"]
        )
else:
    print("None")

print()
print("-" * 80)
print("OUTPUTS")
print("-" * 80)

print("Authoritative JSON:")
print(AUTHORITATIVE_JSON)

print()
print("Audit CSV:")
print(AUDIT_CSV)

print()
print("Diagnostic:")
print(REPORT_JSON)

print()
print("=" * 80)
