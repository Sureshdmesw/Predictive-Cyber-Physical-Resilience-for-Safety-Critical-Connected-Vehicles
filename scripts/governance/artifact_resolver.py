from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]

EXCLUDED = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
}


def iter_files():
    for path in ROOT.rglob("*"):

        if not path.is_file():
            continue

        if any(part in EXCLUDED for part in path.parts):
            continue

        yield path


def load_json(path):

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        return None


def relative(path):
    return str(
        path.relative_to(ROOT)
    )


def find_json_files():

    return [
        p
        for p in iter_files()
        if p.suffix.lower() == ".json"
    ]


def find_exact_filename(filename):

    matches = []

    target = filename.lower()

    for path in find_json_files():

        if path.name.lower() == target:
            matches.append(path)

    return matches


def find_by_terms(
    terms,
    directories=None,
):

    terms = [
        str(x).lower()
        for x in terms
    ]

    results = []

    for path in find_json_files():

        text = relative(path).lower()

        if directories:

            directory_match = any(
                str(d).lower() in text
                for d in directories
            )

            if not directory_match:
                continue

        if all(
            term in text
            for term in terms
        ):

            results.append(path)

    return results


def resolve_schema(
    exact_names=None,
    filename_terms=None,
    directory_terms=None,
):

    # -------------------------------------------------------------
    # Priority 1: exact filename
    # -------------------------------------------------------------

    if exact_names:

        for filename in exact_names:

            matches = find_exact_filename(
                filename
            )

            valid = []

            for path in matches:

                data = load_json(path)

                if isinstance(data, dict):
                    valid.append(
                        (path, data)
                    )

            if len(valid) == 1:

                return {
                    "path": valid[0][0],
                    "data": valid[0][1],
                    "resolution": "exact_filename",
                }

            if len(valid) > 1:

                raise RuntimeError(
                    "Multiple exact schema matches: "
                    + str(
                        [
                            relative(x[0])
                            for x in valid
                        ]
                    )
                )

    # -------------------------------------------------------------
    # Priority 2: filename terms
    # -------------------------------------------------------------

    candidates = []

    if filename_terms:

        candidates = find_by_terms(
            filename_terms,
            directories=directory_terms,
        )

    valid = []

    for path in candidates:

        data = load_json(path)

        if isinstance(data, dict):

            valid.append(
                (path, data)
            )

    # -------------------------------------------------------------
    # Exactly one semantic match
    # -------------------------------------------------------------

    if len(valid) == 1:

        return {
            "path": valid[0][0],
            "data": valid[0][1],
            "resolution": "semantic_unique",
        }

    # -------------------------------------------------------------
    # Multiple matches: rank, but refuse ambiguity
    # -------------------------------------------------------------

    if len(valid) > 1:

        ranked = []

        for path, data in valid:

            text = (
                relative(path)
                + " "
                + json.dumps(data)
            ).lower()

            score = 0

            for term in (
                "schema",
                "model",
                "grounded",
                "resilience",
                "explain",
            ):

                if term in text:
                    score += 1

            if "data\\schemas" in relative(path).lower():
                score += 10

            if "explainability" in relative(path).lower():
                score += 10

            ranked.append(
                (
                    score,
                    path,
                    data,
                )
            )

        ranked.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        # Only automatically resolve if the top
        # candidate is strictly better.

        if (
            len(ranked) == 1
            or ranked[0][0] > ranked[1][0]
        ):

            return {
                "path": ranked[0][1],
                "data": ranked[0][2],
                "resolution": "semantic_ranked",
            }

        raise RuntimeError(
            "Ambiguous schema resolution. "
            "Candidates: "
            + str(
                [
                    {
                        "path": relative(x[1]),
                        "score": x[0],
                    }
                    for x in ranked
                ]
            )
        )

    return None


def find_phase_artifacts(
    phase,
):

    token = (
        f"phase4_{phase}"
        .lower()
    )

    results = []

    for path in find_json_files():

        text = relative(path).lower()

        if token not in text:
            continue

        data = load_json(path)

        results.append(
            {
                "path": path,
                "relative_path": relative(path),
                "data": data,
                "valid_json": isinstance(
                    data,
                    dict,
                ),
                "status": (
                    str(
                        data.get(
                            "status",
                            "",
                        )
                    ).upper()
                    if isinstance(
                        data,
                        dict,
                    )
                    else None
                ),
            }
        )

    return results


def resolve_phase(
    phase,
    required_status=None,
):

    candidates = find_phase_artifacts(
        phase
    )

    if required_status:

        candidates = [
            x
            for x in candidates
            if x["status"]
            == required_status.upper()
        ]

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x["valid_json"],
            x["status"] == "PASS",
            "integration"
            in x["relative_path"].lower(),
            "modeling"
            in x["relative_path"].lower(),
            "final"
            in x["relative_path"].lower(),
        ),
        reverse=True,
    )

    return candidates[0]


def require_phase(
    phase,
    required_status="PASS",
):

    result = resolve_phase(
        phase,
        required_status,
    )

    if result is None:

        candidates = find_phase_artifacts(
            phase
        )

        raise RuntimeError(
            f"Unable to resolve Phase 4.{phase}. "
            f"Candidates: "
            f"{[x['relative_path'] for x in candidates]}"
        )

    return result
