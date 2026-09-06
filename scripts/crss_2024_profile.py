from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]

CRSS_DIR = (
    ROOT
    / "data"
    / "raw"
    / "nhtsa"
    / "CRSS"
    / "downloads"
    / "CRSS2024CSV"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "nhtsa"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_csv(path):

    result = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "columns": [],
        "row_count": 0,
        "sample_rows": []
    }

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as handle:

        reader = csv.reader(handle)

        try:
            header = next(reader)
        except StopIteration:
            return result

        result["columns"] = header

        for row_number, row in enumerate(reader, start=1):

            result["row_count"] = row_number

            if len(result["sample_rows"]) < 3:
                result["sample_rows"].append(row)

    return result


def main():

    files = sorted(
        CRSS_DIR.glob("*.csv")
    )

    print()
    print("=" * 80)
    print("CRSS 2024 DATASET PROFILER")
    print("=" * 80)
    print()

    if not files:
        raise SystemExit(
            f"No CSV files found in: {CRSS_DIR}"
        )

    results = []

    for path in files:

        print(
            f"Inspecting: {path.name}"
        )

        result = inspect_csv(path)

        results.append(result)

        print(
            f"  Rows: {result['row_count']:,}"
        )

        print(
            f"  Columns: {len(result['columns'])}"
        )

        print(
            f"  Size: {result['size_bytes']:,} bytes"
        )

        print()

    output = (
        OUTPUT_DIR
        / "crss_2024_profile.json"
    )

    with output.open(
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            results,
            handle,
            indent=2
        )

    print("=" * 80)
    print("PROFILE COMPLETE")
    print("=" * 80)
    print()
    print(f"Tables profiled: {len(results)}")
    print(f"Output: {output}")
    print()


if __name__ == "__main__":
    main()
