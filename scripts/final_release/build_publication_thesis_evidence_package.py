from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone


ROOT = Path(
    r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles"
)

OUT_DIR = ROOT / "experiments" / "final_release"

OUTPUT_JSON = OUT_DIR / "publication_thesis_evidence_package.json"
OUTPUT_CSV = OUT_DIR / "publication_thesis_evidence_index.csv"


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def exists(relative_path):
    return (ROOT / relative_path).exists()


def artifact(relative_path, role, claim):
    path = ROOT / relative_path

    if not path.exists():
        return {
            "path": relative_path,
            "role": role,
            "claim_supported": claim,
            "exists": False,
            "sha256": None,
            "size_bytes": None,
        }

    return {
        "path": relative_path,
        "role": role,
        "claim_supported": claim,
        "exists": True,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sections = []

    sections.append(
        {
            "section": "Research Objective",
            "purpose": "Define the research problem and system objective.",
            "evidence": [
                artifact(
                    "docs\\research_questions.md",
                    "research_questions",
                    "Research questions and objectives",
                ),
                artifact(
                    "docs\\system_architecture.md",
                    "system_architecture",
                    "Overall research architecture",
                ),
                artifact(
                    "docs\\FINAL_RESEARCH_DOCUMENTATION.md",
                    "final_documentation",
                    "Final consolidated research description",
                ),
            ],
        }
    )

    sections.append(
        {
            "section": "Dataset Foundation",
            "purpose": "Establish the data foundation and experimental scope.",
            "evidence": [
                artifact(
                    "docs\\data_architecture.md",
                    "data_architecture",
                    "Dataset architecture and provenance",
                ),
                artifact(
                    "data\\processed\\can_hil\\phase5_10\\phase5_10_cyber_physical_attack_scenarios.csv",
                    "cyber_physical_dataset",
                    "Controlled cyber-physical experimental data",
                ),
            ],
        }
    )

    sections.append(
        {
            "section": "Predictive Modeling",
            "purpose": "Support the temporal predictive-model methodology.",
            "evidence": [
                artifact(
                    "models\\v2\\advanced\\crss_2024_v2_temporal_transformer_best.pt",
                    "frozen_checkpoint",
                    "Frozen temporal Transformer checkpoint",
                ),
                artifact(
                    "experiments\\can_hil\\phase5_11\\phase5_11_predictive_detection_manifest.json",
                    "predictive_manifest",
                    "Predictive-detection experiment provenance",
                ),
            ],
        }
    )

    sections.append(
        {
            "section": "Uncertainty and OOD",
            "purpose": "Support corrected OOD analysis and scientific interpretation.",
            "evidence": [
                artifact(
                    "data\\processed\\can_hil\\phase6_2\\phase6_2_corrected_ood.csv",
                    "corrected_ood_results",
                    "Corrected OOD results",
                ),
                artifact(
                    "experiments\\phase6_2\\phase6_2_corrected_ood_manifest.json",
                    "ood_manifest",
                    "Corrected OOD experiment provenance",
                ),
                artifact(
                    "experiments\\phase6_3\\phase6_3_scientific_review_manifest.json",
                    "scientific_review_manifest",
                    "Scientific review of OOD results",
                ),
                artifact(
                    "experiments\\phase6_4\\phase6_4_uncertainty_calibration_manifest.json",
                    "calibration_manifest",
                    "Uncertainty calibration evidence",
                ),
            ],
        }
    )

    sections.append(
        {
            "section": "Connectivity Resilience",
            "purpose": "Support connectivity-loss and recovery claims.",
            "evidence": [
                artifact(
                    "experiments\\can_hil\\phase5_16\\phase5_16_connectivity_recovery_manifest.json",
                    "connectivity_manifest",
                    "Connectivity loss and recovery validation",
                ),
                artifact(
                    "experiments\\can_hil\\phase5_17\\phase5_17_secure_synchronization_manifest.json",
                    "synchronization_manifest",
                    "Secure synchronization and SOC handoff",
                ),
            ],
        }
    )

    sections.append(
        {
            "section": "Forensic Integrity",
            "purpose": "Support evidence-integrity and tamper-rejection claims.",
            "evidence": [
                artifact(
                    "experiments\\can_hil\\phase5_15\\phase5_15_integrity_tamper_manifest.json",
                    "integrity_manifest",
                    "Integrity and tamper validation",
                ),
                artifact(
                    "experiments\\can_hil\\phase5_18\\phase5_18_adversarial_integrity_validation_manifest.json",
                    "adversarial_integrity_manifest",
                    "Adversarial integrity validation",
                ),
            ],
        }
    )

    sections.append(
        {
            "section": "End-to-End Resilience",
            "purpose": "Support integrated resilience claims.",
            "evidence": [
                artifact(
                    "experiments\\can_hil\\phase5_19\\phase5_19_end_to_end_resilience_validation_manifest.json",
                    "end_to_end_manifest",
                    "End-to-end resilience validation",
                ),
                artifact(
                    "experiments\\can_hil\\phase5_20\\phase5_20_resilience_scorecard_manifest.json",
                    "resilience_scorecard",
                    "Resilience scorecard",
                ),
                artifact(
                    "experiments\\can_hil\\phase5_21\\phase5_21_resilience_evidence_matrix_manifest.json",
                    "resilience_evidence_matrix",
                    "Scenario-level evidence matrix",
                ),
            ],
        }
    )

    sections.append(
        {
            "section": "Safety Architecture",
            "purpose": "Support safety boundaries and deterministic policy claims.",
            "evidence": [
                artifact(
                    "docs\\safety_architecture.md",
                    "safety_architecture",
                    "Safety-oriented architecture",
                ),
                artifact(
                    "docs\\threat_model.md",
                    "threat_model",
                    "Threat model and safety/security boundaries",
                ),
            ],
        }
    )

    sections.append(
        {
            "section": "SOC/XDR Architecture",
            "purpose": "Support SOC/XDR-compatible integration claims.",
            "evidence": [
                artifact(
                    "docs\\cortex_xdr_integration.md",
                    "cortex_xdr_architecture",
                    "Cortex XDR-compatible architecture",
                ),
            ],
        }
    )

    sections.append(
        {
            "section": "Explainability",
            "purpose": "Support model-grounded explainability claims.",
            "evidence": [
                artifact(
                    "docs\\FINAL_RESEARCH_DOCUMENTATION.md",
                    "explainability_documentation",
                    "Explainability scope and methodology",
                ),
            ],
        }
    )

    sections.append(
        {
            "section": "Reproducibility",
            "purpose": "Support reproducibility and artifact-integrity claims.",
            "evidence": [
                artifact(
                    "experiments\\final_release\\final_evidence_consolidation.json",
                    "evidence_consolidation",
                    "Consolidated evidence inventory",
                ),
                artifact(
                    "experiments\\final_release\\final_evidence_index.csv",
                    "evidence_index",
                    "Evidence index and artifact hashes",
                ),
                artifact(
                    "experiments\\final_release\\final_reproducibility_audit.json",
                    "reproducibility_audit",
                    "Reproducibility validation",
                ),
            ],
        }
    )

    sections.append(
        {
            "section": "Claims and Limitations",
            "purpose": "Support scientifically bounded final claims.",
            "evidence": [
                artifact(
                    "docs\\final_research_claims_and_limitations.md",
                    "claims_limitations_document",
                    "Authoritative claims and limitations",
                ),
                artifact(
                    "experiments\\final_release\\final_claims_limitations_audit.json",
                    "claims_limitations_audit",
                    "Claims and limitations governance validation",
                ),
            ],
        }
    )

    sections.append(
        {
            "section": "Final Release Governance",
            "purpose": "Support overall release-readiness claims.",
            "evidence": [
                artifact(
                    "experiments\\final_release\\final_integrated_audit_v2.json",
                    "integrated_audit",
                    "Final integrated audit",
                ),
                artifact(
                    "experiments\\final_release\\final_evidence_consolidation.json",
                    "evidence_consolidation",
                    "Final evidence consolidation",
                ),
                artifact(
                    "experiments\\final_release\\final_reproducibility_audit.json",
                    "reproducibility_audit",
                    "Final reproducibility audit",
                ),
                artifact(
                    "experiments\\final_release\\final_claims_limitations_audit.json",
                    "claims_limitations_audit",
                    "Final claims and limitations audit",
                ),
            ],
        }
    )

    all_artifacts = []

    for section in sections:
        for item in section["evidence"]:
            all_artifacts.append(item)

    existing = [
        item
        for item in all_artifacts
        if item["exists"]
    ]

    missing = [
        item
        for item in all_artifacts
        if not item["exists"]
    ]

    result = {
        "publication_thesis_evidence_package": {
            "name": "FINAL_PUBLICATION_THESIS_EVIDENCE_PACKAGE",
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "read_only_source_mapping": True,
            "purpose": (
                "Map research claims, methodology, validation, "
                "reproducibility, and limitations to authoritative evidence."
            ),
        },
        "summary": {
            "sections": len(sections),
            "evidence_references": len(all_artifacts),
            "existing_evidence_references": len(existing),
            "missing_evidence_references": len(missing),
            "coverage_percentage": round(
                (
                    len(existing)
                    / len(all_artifacts)
                    * 100
                ),
                2,
            ) if all_artifacts else 0.0,
            "status": (
                "PASS"
                if not missing
                else "PASS_WITH_MISSING_REFERENCES"
            ),
        },
        "sections": sections,
        "missing_evidence": missing,
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    csv_lines = [
        "section,role,path,exists,size_bytes,sha256,claim_supported"
    ]

    for section in sections:
        for item in section["evidence"]:
            values = [
                section["section"],
                item["role"],
                item["path"],
                str(item["exists"]),
                str(item["size_bytes"] or ""),
                item["sha256"] or "",
                item["claim_supported"],
            ]

            escaped = [
                '"' + value.replace('"', '""') + '"'
                for value in values
            ]

            csv_lines.append(",".join(escaped))

    OUTPUT_CSV.write_text(
        "\n".join(csv_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("PUBLICATION / THESIS EVIDENCE PACKAGE")
    print("=" * 80)
    print(
        f"Status                    : "
        f"{result['summary']['status']}"
    )
    print(
        f"Sections                  : "
        f"{result['summary']['sections']}"
    )
    print(
        f"Evidence references       : "
        f"{result['summary']['evidence_references']}"
    )
    print(
        f"Existing references      : "
        f"{result['summary']['existing_evidence_references']}"
    )
    print(
        f"Missing references       : "
        f"{result['summary']['missing_evidence_references']}"
    )
    print(
        f"Coverage                 : "
        f"{result['summary']['coverage_percentage']:.2f}%"
    )

    if missing:
        print()
        print("MISSING EVIDENCE")
        print("-" * 80)

        for item in missing:
            print(
                f"{item['path']} "
                f"({item['role']})"
            )

    print()
    print("OUTPUT")
    print("-" * 80)
    print(OUTPUT_JSON.relative_to(ROOT))
    print(OUTPUT_CSV.relative_to(ROOT))
    print("=" * 80)


if __name__ == "__main__":
    main()
