# Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles

## Final Research Documentation

---

## 1. Research Objective

This project develops a reproducible research framework for predictive cyber-physical resilience in safety-critical connected vehicles.

The framework integrates:

- automotive cybersecurity;
- CAN/ECU telemetry analysis;
- temporal predictive risk forecasting;
- uncertainty and out-of-distribution analysis;
- deterministic resilience decision logic;
- forensic evidence preservation;
- evidence-integrity verification;
- connectivity-loss resilience;
- secure synchronization;
- SOC/XDR-compatible research-event generation;
- model-grounded explainability; and
- reproducible research governance.

The central architectural principle is:

Observation → Prediction → Uncertainty/OOD → Deterministic Decision → Forensic Preservation → Integrity Verification → Connectivity-Resilient Recovery → SOC/XDR-Compatible Investigation

---

## 2. Research Architecture

The system separates probabilistic machine-learning inference from deterministic safety-oriented decision logic.

The architecture does not rely entirely on remote connectivity.

During connectivity loss, local evidence can be preserved and processed through the resilience state machine:

CONNECTED
→ DISCONNECTED
→ LOCAL BUFFER
→ RECOVERED
→ INTEGRITY VERIFIED
→ SYNCHRONIZED
→ SOC HANDOFF

Evidence that fails integrity verification follows:

UNVERIFIED_EVIDENCE
→ REFUSE_SYNC
→ REFUSE_SOC_HANDOFF

This separation prevents unverifiable evidence from being silently treated as trusted evidence.

---

## 3. Dataset Foundation

The research foundation includes NHTSA CRSS 2024 and controlled synthetic cyber-physical telemetry.

CRSS 2024 is used as a real-world crash-investigation foundation.

The synthetic cyber telemetry is used for reproducible controlled cybersecurity experimentation.

Synthetic cyber telemetry is not real-world cyberattack evidence.

The project does not claim that performance on the synthetic attack scenarios establishes validated real-world cyberattack detection capability.

---

## 4. Predictive Model

The implemented temporal Transformer uses:

- 59 input features;
- 128-dimensional embedding;
- 3 Transformer layers;
- 8 attention heads;
- 256-dimensional feed-forward network;
- dropout of 0.10;
- 12-timestep temporal windows;
- attention pooling;
- Y3 and Y6 prediction horizons;
- 407,811 trainable parameters.

The predictive horizons represent approximately:

- Y3: 15 seconds;
- Y6: 30 seconds.

The frozen research checkpoint is:

`models\v2\advanced\crss_2024_v2_temporal_transformer_best.pt`

Checkpoint SHA-256:

`d54cfc5fb5613fef5b5dfd6ac4b2ce780ebd2e64ca47b7e57c630dae3d323a754`

---

## 5. Cyber-Physical Attack Scenarios

The controlled cyber-physical experiments contain:

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

The Phase 5.10 dataset contains 99,000 records.

The Phase 5.11 predictive-detection dataset contains 6,479 predictions.

---

## 6. Uncertainty and OOD Analysis

The original OOD analysis identified a numerical artifact associated with a constant `ecu_count` feature.

The issue was not hidden or discarded.

The corrected methodology determines constant dimensions from the non-overlapping baseline timestep stream.

The 59-dimensional Transformer model input remains unchanged.

Constant statistical dimensions are excluded only from OOD statistical calculations when their baseline variance is zero.

The corrected OOD analysis uses the active statistical dimensions while preserving model compatibility.

This correction prevents constant-feature floating-point artifacts from dominating the OOD score.

---

## 7. OOD Scientific Interpretation

The corrected OOD results are a statistical research evaluation.

They are not a complete OOD-detection solution.

The results do not establish that the system can detect every unseen real-world distribution shift, cyberattack, sensor failure, environmental condition, or vehicle operating condition.

OOD interpretation remains dependent on:

- available research data;
- baseline distribution;
- feature representation;
- statistical methodology;
- temporal representation; and
- experimental scenario coverage.

The project therefore preserves the scientific-review and calibration limitations associated with the Phase 6 results.

---

## 8. Phase 6 Results

The Phase 6 research chain is:

Phase 6.1:
`PASS_WITH_SCIENTIFIC_REVIEW_REQUIRED`

Phase 6.2:
`PASS`

Phase 6.3:
`SCIENTIFIC_REVIEW_ACCEPTED_WITH_LIMITATIONS`

Phase 6.4:
`PASS_WITH_CALIBRATION_LIMITATIONS`

These statuses are authoritative research-governance results.

They must not be converted into unconditional claims of validated uncertainty, complete OOD detection, or universal calibration.

---

## 9. Connectivity Resilience

Phase 5.16 validated connectivity-loss and recovery behavior.

The corrected event semantics identify:

- CONNECTED state before disconnection;
- DISCONNECTED state during the defined loss interval;
- RECOVERED state after connectivity restoration.

The validated resilience workflow preserves local evidence during connectivity loss and permits synchronization only after integrity verification.

---

## 10. Secure Synchronization and SOC Handoff

Phase 5.17 validates secure synchronization and SOC-handoff eligibility for the defined research scenarios.

The research event format is:

`CORTEX_XDR_COMPATIBLE_RESEARCH_EVENT`

with:

`ABSTRACTION_ONLY`

The project does not claim live Cortex XDR deployment.

The project does not claim live Cortex XDR API integration.

The SOC/XDR component demonstrates architectural compatibility and research-event handoff rather than operational deployment into a live security platform.

---

## 11. Adversarial Integrity Validation

Phase 5.18 evaluates adversarial modification of forensic evidence.

The baseline evidence chain remains valid.

Tampered evidence is rejected.

Tampered synchronization and SOC handoff are refused.

The validated behavior demonstrates the research governance principle that integrity failure must prevent untrusted evidence from entering downstream synchronization or SOC workflows.

---

## 12. End-to-End Resilience Validation

Phase 5.19 validates the integrated resilience path across the defined research scenarios.

The final resilience scorecard and evidence matrix provide scenario-level evidence for:

- connectivity handling;
- local buffering;
- recovery;
- integrity verification;
- synchronization;
- SOC handoff;
- evidence completeness; and
- safety-boundary compliance.

---

## 13. Explainability

The research framework includes model-grounded explainability.

The explainability design includes:

- feature attribution;
- temporal attribution;
- counterfactual analysis;
- modality grounding; and
- attribution stability.

Explainability is intended to support research interpretation and investigation rather than establish causal certainty.

---

## 14. Safety Architecture

The safety-oriented architecture separates:

1. probabilistic AI outputs;
2. uncertainty/OOD information;
3. deterministic safety/resilience policy;
4. evidence integrity;
5. connectivity state; and
6. downstream SOC handoff.

The framework does not provide uncontrolled direct vehicle actuation.

The research does not constitute ISO 26262 certification or ASIL certification.

---

## 15. Validation Evidence

The final research release includes evidence from:

- CRSS 2024 release validation;
- controlled CAN anomaly experiments;
- cyber-physical attack experiments;
- predictive detection;
- uncertainty/OOD analysis;
- OOD correction;
- scientific review;
- OOD calibration;
- forensic buffering;
- integrity/tamper validation;
- connectivity recovery;
- secure synchronization;
- adversarial integrity validation;
- end-to-end resilience validation;
- resilience scorecard;
- resilience evidence matrix;
- final integrated audit;
- evidence consolidation;
- reproducibility audit; and
- claims/limitations audit.

---

## 16. Final Audit Status

The final integrated audit passed:

`10/10`

The evidence consolidation contains:

`161 evidence files`

including:

- 14 Phase 5 manifests;
- 4 Phase 6 manifests;
- 20 valid manifests;
- 0 invalid manifests;
- datasets;
- schemas;
- documentation;
- JSON evidence;
- model checkpoints; and
- release artifacts.

The reproducibility audit passed:

`13/13`

with all evidence hashes matching the consolidated evidence index.

The claims and limitations audit passed:

`10/10`

with no unsupported affirmative research claims detected.

---

## 17. Reproducibility

The final research release uses:

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

The reproducibility chain is designed so that research outputs can be independently inspected against their recorded evidence.

---

## 18. Research Boundaries

The project does not claim:

- physical road deployment;
- production ECU/CAN validation;
- production-fleet validation;
- real-world cyberattack-data validation;
- live Cortex XDR deployment;
- uncontrolled direct vehicle actuation;
- ISO 26262 certification;
- ASIL certification;
- complete OOD detection;
- universal uncertainty calibration;
- complete real-world automotive cybersecurity validation.

High predictive performance within controlled experimental data does not establish real-world cyberattack-detection capability.

---

## 19. Scientific Contribution

The principal contribution is the integration of predictive cyber-physical analysis with deterministic resilience and evidence governance.

The framework does not treat anomaly detection as an isolated classifier.

Instead, it connects:

prediction
→ uncertainty
→ deterministic decision
→ evidence preservation
→ integrity verification
→ connectivity recovery
→ secure synchronization
→ SOC-compatible investigation.

This provides a research architecture for studying how a safety-critical connected vehicle can continue producing trustworthy evidence and resilience decisions when connectivity is degraded or unavailable.

---

## 20. Final Research Statement

This project presents a reproducible research framework for predictive cyber-physical resilience in safety-critical connected vehicles.

Using NHTSA CRSS 2024 as a crash-investigation foundation and controlled synthetic cyber telemetry for reproducible cybersecurity experimentation, the framework evaluates temporal risk forecasting, uncertainty and OOD analysis, deterministic resilience decisions, local forensic buffering, evidence-integrity verification, connectivity recovery, secure synchronization, SOC/XDR-compatible research-event generation, and model-grounded explainability.

The completed research evidence demonstrates that these architectural components can be implemented and evaluated under defined experimental conditions.

The results do not constitute physical vehicle validation, production ECU/CAN validation, road validation, real-world cyberattack detection validation, live Cortex XDR deployment, direct vehicle actuation, ISO 26262 certification, ASIL certification, complete OOD detection, universal uncertainty calibration, or production-fleet validation.

The project therefore presents an evidence-backed research framework rather than a production safety or cybersecurity certification claim.

---

## 21. Final Governance Principle

A passing experimental gate means that the defined research validation condition passed.

It does not automatically establish:

- production readiness;
- road readiness;
- real-world generalization;
- safety certification;
- cybersecurity certification; or
- universal model validity.

All final research conclusions must remain bounded by the datasets, experimental scenarios, model configuration, statistical methodology, validation conditions, and documented limitations.

---

## 22. Authoritative Supporting Documents

Core technical documentation:

- `docs\system_architecture.md`
- `docs\data_architecture.md`
- `docs\safety_architecture.md`
- `docs\connectivity_resilience.md`
- `docs\cortex_xdr_integration.md`
- `docs\threat_model.md`
- `docs\research_questions.md`
- `docs\final_research_claims_and_limitations.md`

Final release evidence:

- `experiments\final_release\final_integrated_audit_v2.json`
- `experiments\final_release\final_evidence_consolidation.json`
- `experiments\final_release\final_evidence_index.csv`
- `experiments\final_release\final_reproducibility_audit.json`
- `experiments\final_release\final_claims_limitations_audit.json`

---

## 23. Release Readiness

The research framework has completed:

1. integrated validation;
2. evidence consolidation;
3. reproducibility validation;
4. claims and limitations governance.

The remaining release work is documentation finalization, README/research-statement synchronization, and preparation of the publication/thesis evidence package.

