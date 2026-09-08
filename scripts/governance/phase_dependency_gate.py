import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REGISTRY = (
    ROOT
    / "experiments"
    / "governance"
    / "artifact_registry.json"
)


def main():
    print("=== PHASE DEPENDENCY GATE ===")

    if not REGISTRY.exists():
        print("FAIL: Artifact registry does not exist.")
        raise SystemExit(1)

    data = json.loads(
        REGISTRY.read_text(encoding="utf-8")
    )

    if data.get("status") != "PASS":
        print(
            "FAIL: Artifact registry is not PASS."
        )

        print(
            "Missing:",
            data.get("missing_artifacts"),
        )

        print(
            "Invalid JSON:",
            data.get("invalid_json_artifacts"),
        )

        print(
            "Non-PASS:",
            data.get("non_pass_artifacts"),
        )

        raise SystemExit(1)

    artifacts = data.get(
        "artifacts",
        [],
    )

    failures = []

    for artifact in artifacts:

        if not artifact.get("exists"):
            failures.append(
                f"{artifact['key']}: missing"
            )

        elif not artifact.get("json_valid"):
            failures.append(
                f"{artifact['key']}: invalid JSON"
            )

        elif artifact.get("status") != "PASS":
            failures.append(
                f"{artifact['key']}: "
                f"status={artifact.get('status')}"
            )

    if failures:
        print("FAIL:")

        for failure in failures:
            print(" -", failure)

        raise SystemExit(1)

    print(
        f"PASS: {len(artifacts)} artifacts verified."
    )


if __name__ == "__main__":
    main()
