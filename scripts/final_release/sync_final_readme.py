from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(
    r"E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles"
)

README = ROOT / "README.md"

START_MARKER = "<!-- FINAL_RESEARCH_RELEASE_START -->"
END_MARKER = "<!-- FINAL_RESEARCH_RELEASE_END -->"


FINAL_SECTION = f"""
{START_MARKER}

# Final Research Release

## Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles

### Final Research Status

The research framework has completed integrated validation, evidence consolidation, reproducibility validation, and claims/limitations governance.

| Release Area | Result |
|---|---:|
| Final integrated audit | PASS — 10/10 |
| Evidence consolidation | 161 evidence files |
| Manifest validation | 21 valid / 0 invalid |
| Reproducibility audit | PASS — 13/13 |
| Claims & limitations audit | PASS — 10/10 |
| Phase 5 manifests | 14 |
| Phase 6 manifests | 4 |
| Model input dimensionality | 59 |
| Frozen model checkpoint | Verified by SHA-256 |

### Research Contribution

The project presents a reproducible research framework that connects:

**Observation → Prediction → Uncertainty/OOD → Deterministic Decision → Forensic Preservation → Integrity Verification → Connectivity-Resilient Recovery → SOC/XDR-Compatible Investigation**

The research goes beyond isolated anomaly classification by integrating predictive cyber-physical analysis with resilience-state management, forensic evidence preservation, integrity verification, connectivity recovery, secure synchronization, and SOC-compatible investigation.

### Predictive Model

The implemented temporal Transformer uses:

- 59 input features
- 128-dimensional embedding
- 3 Transformer layers
- 8 attention heads
- 256-dimensional feed-forward network
- dropout = 0.10
- 12-timestep temporal windows
- attention pooling
- Y3 and Y6 prediction horizons
- 407,811 trainable parameters

Approximate prediction horizons:

- Y3 = 15 seconds
- Y6 = 30 seconds

Frozen checkpoint:

`models\\v2\\advanced\\crss_2024_v2_temporal_transformer_best.pt`

Checkpoint SHA-256:

`d54cfc5fb5613fef5b5dfd6ac4b2ce780ebd2e64ca47b7e57c630dae3d323a754`

### Experimental Evidence

The controlled cyber-physical scenarios include:

- S00_NORMAL_BASELINE
- C01_SPEED_SENSOR_SPOOFING
- C02_BRAKE_ACCELERATOR_ATTACK
- C03_TORQUE_DYNAMICS_ATTACK
- C04_HV_POWER_ATTACK
- C05_BATTERY_THERMAL_ATTACK
- C06_HV_CONTACTOR_ATTACK
- C07_HEARTBEAT_INTEGRITY_ATTACK
- C08_SEQUENCE_GATEWAY_ATTACK
- C09_RESTRAINT_DOMAIN_ATTACK
- C10_MULTI_DOMAIN_ATTACK

The Phase 5.10 cyber-physical dataset contains 99,000 records.

The Phase 5.11 predictive-detection dataset contains 6,479 predictions.

### OOD and Uncertainty

The OOD analysis identified and corrected a numerical artifact associated with the constant `ecu_count` feature.

The correction preserves the 59-dimensional Transformer input while excluding zero-variance dimensions only from statistical OOD calculations.

The corrected OOD analysis is a statistical research evaluation.

It is not a complete OOD-detection solution and does not establish universal real-world OOD reliability or uncertainty calibration.

The Phase 6 research statuses remain authoritative:

- Phase 6.1 — `PASS_WITH_SCIENTIFIC_REVIEW_REQUIRED`
- Phase 6.2 — `PASS`
- Phase 6.3 — `SCIENTIFIC_REVIEW_ACCEPTED_WITH_LIMITATIONS`
- Phase 6.4 — `PASS_WITH_CALIBRATION_LIMITATIONS`

### Resilience Architecture

The validated resilience state machine is:

`CONNECTED → DISCONNECTED → LOCAL BUFFER → RECOVERED → INTEGRITY VERIFIED → SYNCHRONIZED → SOC HANDOFF`

Invalid evidence follows:

`UNVERIFIED_EVIDENCE → REFUSE_SYNC → REFUSE_SOC_HANDOFF`

This establishes the research principle that unverifiable evidence must not be silently synchronized or handed to downstream SOC workflows as trusted evidence.

### Cortex XDR Boundary

The project uses:

`CORTEX_XDR_COMPATIBLE_RESEARCH_EVENT`

with:

`ABSTRACTION_ONLY`

The project does not claim live Cortex XDR deployment or live Cortex XDR API integration.

### Research Boundaries

This project does **not** claim:

- physical road deployment;
- production ECU/CAN validation;
- production-fleet validation;
- real-world cyberattack-data validation;
- live Cortex XDR deployment;
- uncontrolled direct vehicle actuation;
- ISO 26262 certification;
- ASIL certification;
- complete OOD detection;
- universal uncertainty calibration; or
- complete real-world automotive cybersecurity validation.

High predictive performance within controlled experimental data does not establish real-world cyberattack-detection capability.

### Reproducibility

The final release contains:

- canonical schemas;
- feature definitions;
- experiment manifests;
- deterministic validation scripts;
- frozen model checkpoints;
- SHA-256 artifact hashes;
- evidence indexes;
- experiment reports;
- phase-specific validation evidence;
- dependency validation;
- release audits; and
- explicit research-claim governance.

### Final Evidence

Authoritative final-release artifacts:

- `experiments\\final_release\\final_integrated_audit_v2.json`
- `experiments\\final_release\\final_evidence_consolidation.json`
- `experiments\\final_release\\final_evidence_index.csv`
- `experiments\\final_release\\final_reproducibility_audit.json`
- `experiments\\final_release\\final_claims_limitations_audit.json`

Authoritative documentation:

- `docs\\FINAL_RESEARCH_DOCUMENTATION.md`
- `docs\\final_research_claims_and_limitations.md`

### Final Research Statement

This project presents a reproducible research framework for predictive cyber-physical resilience in safety-critical connected vehicles.

Using NHTSA CRSS 2024 as a crash-investigation foundation and controlled synthetic cyber telemetry for reproducible cybersecurity experimentation, the framework evaluates temporal risk forecasting, uncertainty and OOD analysis, deterministic resilience decisions, local forensic buffering, evidence-integrity verification, connectivity recovery, secure synchronization, SOC/XDR-compatible research-event generation, and model-grounded explainability.

The completed evidence demonstrates that these architectural components can be implemented and evaluated under defined experimental conditions.

The results are bounded research findings. They do not constitute physical vehicle validation, production ECU/CAN validation, road validation, real-world cyberattack detection validation, live Cortex XDR deployment, direct vehicle actuation, ISO 26262 certification, ASIL certification, complete OOD detection, universal uncertainty calibration, or production-fleet validation.

The project therefore represents an evidence-backed research framework rather than a production safety or cybersecurity certification claim.

### Release Governance

A passing experimental gate means that the defined research validation condition passed.

It does not automatically establish production readiness, road readiness, real-world generalization, safety certification, cybersecurity certification, or universal model validity.

All conclusions must remain bounded by the datasets, experimental scenarios, model configuration, statistical methodology, validation conditions, and documented limitations.

{END_MARKER}
"""


if not README.exists():
    print("ERROR: README.md does not exist.")
else:
    original = README.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if START_MARKER in original and END_MARKER in original:
        start = original.index(START_MARKER)
        end = original.index(END_MARKER) + len(END_MARKER)

        updated = (
            original[:start]
            + FINAL_SECTION.strip()
            + original[end:]
        )

        action = "updated_existing_final_release_section"

    else:
        separator = "\n\n"
        updated = (
            original.rstrip()
            + separator
            + FINAL_SECTION.strip()
            + "\n"
        )

        action = "appended_final_release_section"

    README.write_text(
        updated,
        encoding="utf-8",
    )

    print("=" * 80)
    print("README FINAL RELEASE SYNCHRONIZATION")
    print("=" * 80)
    print(f"Status : PASS")
    print(f"Action : {action}")
    print(f"README : {README.relative_to(ROOT)}")
    print(f"Length : {len(updated):,} characters")
    print("=" * 80)
