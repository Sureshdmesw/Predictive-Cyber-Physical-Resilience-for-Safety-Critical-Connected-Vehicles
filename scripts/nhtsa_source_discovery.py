from pathlib import Path
from urllib.parse import urljoin
import csv
import hashlib
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_DIR = PROJECT_ROOT / "data" / "schemas" / "nhtsa"
DISCOVERY_DIR = PROJECT_ROOT / "experiments" / "nhtsa"

SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/139 Safari/537.36"
    )
}

SOURCES = {
    "CISS": {
        "program_url": (
            "https://www.nhtsa.gov/"
            "crash-data-systems/"
            "crash-investigation-sampling-system"
        ),
        "download_url": (
            "https://www.nhtsa.gov/"
            "file-downloads?p=nhtsa/downloads/CISS/"
        ),
    },

    "CRSS": {
        "program_url": (
            "https://www.nhtsa.gov/"
            "data/crash-data-systems"
        ),
        "download_url": (
            "https://www.nhtsa.gov/"
            "file-downloads?p=nhtsa/downloads/CRSS/"
        ),
    },

    "FARS": {
        "program_url": (
            "https://www.nhtsa.gov/"
            "research-data/"
            "fatality-analysis-reporting-system-fars"
        ),
        "download_url": (
            "https://www.nhtsa.gov/"
            "file-downloads?p=nhtsa/downloads/FARS/"
        ),
    },

    "NTS": {
        "program_url": (
            "https://www.nhtsa.gov/"
            "crash-data-systems/"
            "non-traffic-surveillance"
        ),
        "download_url": (
            "https://www.nhtsa.gov/"
            "file-downloads?p=nhtsa/downloads/NTS/"
        ),
    },

    "SCI": {
        "program_url": (
            "https://www.nhtsa.gov/"
            "research-data/"
            "special-crash-investigations-sci"
        ),
    },

    "SGO_ADAS_ADS": {
        "program_url": (
            "https://www.nhtsa.gov/"
            "vehicle-safety/"
            "standing-general-order-crash-reporting"
        ),
    },
}


def fetch(url):
    print(f"  GET {url}")

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=60
    )

    response.raise_for_status()
    return response


def discover_links(url):
    response = fetch(url)

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    results = []

    for tag in soup.find_all("a"):
        href = tag.get("href")

        if not href:
            continue

        absolute = urljoin(
            response.url,
            href
        )

        text = " ".join(
            tag.stripped_strings
        )

        results.append({
            "text": text,
            "url": absolute
        })

    return results


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():

    inventory = []

    print()
    print("=" * 70)
    print(" NHTSA SOURCE DISCOVERY")
    print(" Predictive Cyber-Physical Resilience for Safety-Critical")
    print(" Connected Vehicles")
    print("=" * 70)
    print()

    for dataset, config in SOURCES.items():

        print()
        print(f"[{dataset}]")

        program_url = config["program_url"]

        try:
            response = fetch(program_url)

            print(
                f"  Program page: HTTP {response.status_code}"
            )

            inventory.append({
                "dataset": dataset,
                "source_type": "program_page",
                "source_url": program_url,
                "http_status": response.status_code,
                "content_type": response.headers.get(
                    "Content-Type",
                    ""
                ),
                "discovered_text": "",
            })

        except Exception as exc:

            print(
                f"  ERROR program page: {exc}"
            )

            inventory.append({
                "dataset": dataset,
                "source_type": "program_page",
                "source_url": program_url,
                "http_status": "",
                "content_type": "",
                "discovered_text": str(exc),
            })

        download_url = config.get("download_url")

        if not download_url:
            print(
                "  No standard file-download endpoint "
                "defined for this source."
            )
            continue

        try:

            links = discover_links(
                download_url
            )

            print(
                f"  Discovered {len(links)} links"
            )

            for item in links:

                inventory.append({
                    "dataset": dataset,
                    "source_type": "download_repository",
                    "source_url": download_url,
                    "http_status": 200,
                    "content_type": "",
                    "discovered_text": item["text"],
                    "discovered_url": item["url"],
                })

        except Exception as exc:

            print(
                f"  ERROR download repository: {exc}"
            )

            inventory.append({
                "dataset": dataset,
                "source_type": "download_repository",
                "source_url": download_url,
                "http_status": "",
                "content_type": "",
                "discovered_text": str(exc),
                "discovered_url": "",
            })

    output = (
        DISCOVERY_DIR /
        "nhtsa_source_discovery.csv"
    )

    fieldnames = [
        "dataset",
        "source_type",
        "source_url",
        "http_status",
        "content_type",
        "discovered_text",
        "discovered_url",
    ]

    with output.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            inventory
        )

    print()
    print("=" * 70)
    print("DISCOVERY COMPLETE")
    print("=" * 70)
    print()
    print(f"Output: {output}")
    print(f"Records: {len(inventory)}")
    print()


if __name__ == "__main__":
    main()
