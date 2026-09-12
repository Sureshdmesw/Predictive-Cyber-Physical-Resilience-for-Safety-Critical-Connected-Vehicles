from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import csv


ROOT = Path(r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles")

OUTPUT_DIR = ROOT / "experiments" / "final_release"

JSON_OUTPUT = OUTPUT_DIR / "final_evidence_consolidation.json"
CSV_OUTPUT = OUTPUT_DIR / "final_evidence_index.csv"


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "__error__": str(exc)
        }


def recursive_find_value(data, names):
    if isinstance(data, dict):
        for name in names:
            if name in data:
                return data[name]

        for value in data.values():
            result = recursive_find_value(value, names)

            if result is not None:
                return result

    elif isinstance(data, list):
        for value in data:
            result = recursive_find_value(value, names)

            if result is not None:
                return result

    return None


def classify_file(path):
    relative = path.relative_to(ROOT).as_posix().lower()
    name = path.name.lower()

    if "manifest" in name:
        return "manifest"

    if "schema" in name:
        return "schema"

    if path.suffix.lower() == ".csv":
        return "dataset_csv"

    if path.suffix.lower() == ".jsonl":
        return "dataset_jsonl"

    if path.suffix.lower() == ".json":
        return "json_evidence"

    if path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return "model_checkpoint"

    if path.suffix.lower() in {".md", ".txt"}:
        return "documentation"

    if path.suffix.lower() in {".py", ".ps1"}:
        return "source_code"

    if "final_release" in relative or "release" in relative:
        return "release_artifact"

    return "other"


def classify_phase(path):
    relative = path.relative_to(ROOT).as_posix().lower()

    for phase in [
        "phase6_4",
        "phase6_3",
        "phase6_2",
        "phase6_1",
        "phase5_21",
        "phase5_20",
        "phase5_19",
        "phase5_18",
        "phase5_17",
        "phase5_16",
        "phase5_15",
        "phase5_14",
        "phase5_13",
        "phase5_12",
        "phase5_11",
        "phase5_10",
        "phase5_9",
        "phase5_8",
        "phase5_7",
        "phase5_6",
        "phase5_5",
        "phase5_4",
        "phase5_3",
        "phase5_2",
        "phase5_1",
        "phase4_40",
    ]:
        if phase in relative:
            return phase

    return "non_phase"


def collect_manifest_evidence():
    records = []

    experiments = ROOT / "experiments"

    if not experiments.exists():
        return records

    for path in experiments.rglob("*.json"):

        if "manifest" not in path.name.lower():
            continue

        data = load_json(path)

        relative = path.relative_to(ROOT).as_posix()

        records.append({
            "path": relative,
            "filename": path.name,
            "category": "manifest",
            "phase": classify_phase(path),
            "status": recursive_find_value(
                data,
                [
                    "Status",
                    "status",
                    "release_status",
                    "validation_status"
                ]
            ),
            "checks": recursive_find_value(
                data,
                [
                    "Checks",
                    "checks",
                    "check_count",
                    "passed_checks"
                ]
            ),
            "gate": recursive_find_value(
                data,
                [
                    "Gate",
                    "gate",
                    "validation_gate"
                ]
            ),
            "json_valid": "__error__" not in data,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    return records


def collect_processed_evidence():
    records = []

    processed = ROOT / "data" / "processed"

    if not processed.exists():
        return records

    for path in processed.rglob("*"):

        if not path.is_file():
            continue

        relative = path.relative_to(ROOT).as_posix()

        records.append({
            "path": relative,
            "filename": path.name,
            "category": classify_file(path),
            "phase": classify_phase(path),
            "status": "PRESENT",
            "checks": None,
            "gate": None,
            "json_valid": None,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    return records


def collect_release_evidence():
    records = []

    release_roots = [
        ROOT / "experiments" / "final_release",
        ROOT / "experiments" / "release",
    ]

    for base in release_roots:

        if not base.exists():
            continue

        for path in base.rglob("*"):

            if not path.is_file():
                continue

            relative = path.relative_to(ROOT).as_posix()

            records.append({
                "path": relative,
                "filename": path.name,
                "category": "release_artifact",
                "phase": "final_release",
                "status": "PRESENT",
                "checks": None,
                "gate": None,
                "json_valid": None,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    return records


def collect_documentation():
    records = []

    documentation_roots = [
        ROOT / "docs",
        ROOT / "README.md",
    ]

    for item in documentation_roots:

        if item.is_file():
            paths = [item]

        elif item.is_dir():
            paths = [
                p for p in item.rglob("*")
                if p.is_file()
            ]

        else:
            paths = []

        for path in paths:

            relative = path.relative_to(ROOT).as_posix()

            records.append({
                "path": relative,
                "filename": path.name,
                "category": "documentation",
                "phase": "documentation",
                "status": "PRESENT",
                "checks": None,
                "gate": None,
                "json_valid": None,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    return records


def collect_model_checkpoints():
    records = []

    model_root = ROOT / "models"

    if not model_root.exists():
        return records

    for path in model_root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".pt",
            ".pth",
            ".ckpt"
        }:
            continue

        relative = path.relative_to(ROOT).as_posix()

        records.append({
            "path": relative,
            "filename": path.name,
            "category": "model_checkpoint",
            "phase": "model",
            "status": "PRESENT",
            "checks": None,
            "gate": None,
            "json_valid": None,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    return records


def deduplicate(records):
    unique = {}

    for record in records:
        unique[record["path"]] = record

    return [
        unique[path]
        for path in sorted(unique)
    ]


def build_summary(records):

    categories = {}

    for record in records:
        category = record["category"]
        categories[category] = categories.get(category, 0) + 1

    phases = {}

    for record in records:
        phase = record["phase"]
        phases[phase] = phases.get(phase, 0) + 1

    phase5_manifests = [
        r for r in records
        if r["category"] == "manifest"
        and r["phase"].startswith("phase5_")
    ]

    phase6_manifests = [
        r for r in records
        if r["category"] == "manifest"
        and r["phase"].startswith("phase6_")
    ]

    valid_manifests = [
        r for r in records
        if r["category"] == "manifest"
        and r["json_valid"] is True
    ]

    return {
        "total_evidence_files": len(records),
        "category_counts": categories,
        "phase_counts": phases,
        "phase5_manifest_count": len(phase5_manifests),
        "phase6_manifest_count": len(phase6_manifests),
        "valid_manifest_count": len(valid_manifests),
        "invalid_manifest_count": len(
            [
                r for r in records
                if r["category"] == "manifest"
                and r["json_valid"] is False
            ]
        ),
    }


def write_csv(records):

    fieldnames = [
        "path",
        "filename",
        "category",
        "phase",
        "status",
        "checks",
        "gate",
        "json_valid",
        "size_bytes",
        "sha256",
    ]

    with CSV_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for record in records:
            writer.writerow(record)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    records.extend(
        collect_manifest_evidence()
    )

    records.extend(
        collect_processed_evidence()
    )

    records.extend(
        collect_release_evidence()
    )

    records.extend(
        collect_documentation()
    )

    records.extend(
        collect_model_checkpoints()
    )

    records = deduplicate(records)

    summary = build_summary(records)

    result = {
        "evidence_consolidation": {
            "name": "FINAL_RESEARCH_RELEASE_EVIDENCE_CONSOLIDATION",
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "project_root": str(ROOT),
            "read_only_source_scan": True,
        },

        "summary": summary,

        "evidence_records": records,
    }

    JSON_OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    write_csv(records)

    print()
    print("=" * 80)
    print("FINAL RESEARCH RELEASE — EVIDENCE CONSOLIDATION")
    print("=" * 80)

    print(
        f"Total evidence files : "
        f"{summary['total_evidence_files']}"
    )

    print(
        f"Phase 5 manifests    : "
        f"{summary['phase5_manifest_count']}"
    )

    print(
        f"Phase 6 manifests    : "
        f"{summary['phase6_manifest_count']}"
    )

    print(
        f"Valid manifests      : "
        f"{summary['valid_manifest_count']}"
    )

    print(
        f"Invalid manifests    : "
        f"{summary['invalid_manifest_count']}"
    )

    print()
    print("CATEGORY COUNTS")
    print("-" * 80)

    for category, count in sorted(
        summary["category_counts"].items()
    ):
        print(
            f"{category:<30} : {count}"
        )

    print()
    print("EVIDENCE INDEX")
    print("-" * 80)

    for record in records:
        print(
            f"{record['category']:<20} "
            f"{record['path']}"
        )

    print()
    print(f"JSON evidence index : {JSON_OUTPUT.relative_to(ROOT)}")
    print(f"CSV evidence index  : {CSV_OUTPUT.relative_to(ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
