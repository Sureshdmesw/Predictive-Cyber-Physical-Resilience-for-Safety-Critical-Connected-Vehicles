# Final Research Claims, Boundaries, and Limitations

## 1. Research Scope

This project is a research prototype for predictive cyber-physical resilience in safety-critical connected vehicles.

The framework combines temporal risk forecasting, uncertainty and out-of-distribution analysis, deterministic resilience decisions, local forensic evidence preservation, integrity verification, connectivity-loss recovery, secure synchronization, and SOC/XDR-compatible event generation.

The results are intended to support reproducible research and architectural evaluation. They are not a production vehicle validation or certification claim.

---

## 2. Cybersecurity Data Boundary

The cyber telemetry used in the controlled cyber-physical attack experiments is synthetic research telemetry.

Synthetic cyber telemetry is not real-world cyberattack evidence.

The project does not claim that the observed predictive-detection or anomaly-detection performance constitutes validated real-world cyberattack detection capability.

The project does not use real-world cyberattack labels as the basis for the reported cyber-physical attack experiments.

The controlled scenarios are research scenarios used to evaluate the behavior of the proposed architecture under reproducible experimental conditions.

---

## 3. Physical Vehicle and Road-Validation Boundary

The project has not been physically deployed on a road vehicle.

There is no physical vehicle validation, road testing, production-fleet validation, or road-validated vehicle-system claim.

The research does not establish performance under uncontrolled real-world vehicle operation.

The CAN/ECU experiments are research and HIL-oriented validation activities and must not be interpreted as validation of a production vehicle platform.

---

## 4. Production ECU/CAN Boundary

The project does not claim validation on production ECUs or production CAN networks.

The research does not establish production-vehicle deployment readiness.

Results obtained from the research datasets, synthetic telemetry, software experiments, and HIL-oriented experiments do not constitute evidence of production ECU/CAN safety or cybersecurity performance.

---

## 5. Direct Vehicle Actuation Boundary

The resilience decision architecture does not provide uncontrolled direct vehicle actuation.

The deterministic resilience engine is separated from probabilistic AI outputs.

The research framework is intended to support monitoring, prediction, evidence preservation, integrity verification, resilience-state management, and investigation workflows rather than unrestricted direct control of safety-critical vehicle functions.

---

## 6. Functional-Safety and Certification Boundary

This research does not constitute ISO 26262 certification.

The project does not claim ASIL certification or certified functional-safety compliance.

Safety-oriented architecture, deterministic decision logic, requirements-style controls, and validation gates are research mechanisms and should not be interpreted as evidence of formal automotive safety certification.

---

## 7. OOD and Uncertainty Boundary

The project includes uncertainty and out-of-distribution (OOD) analysis.

The OOD methodology was corrected to prevent constant-feature numerical artifacts from dominating the statistical analysis.

The Transformer model input remains 59-dimensional. Constant dimensions are excluded only from the statistical OOD calculation when the baseline timestep stream establishes that those dimensions have zero variance.

The corrected OOD analysis is therefore a statistical research evaluation and not a complete OOD-detection solution.

The Phase 6 OOD results must not be interpreted as proof that the system can reliably detect every unseen real-world distribution shift, cyberattack, vehicle condition, sensor failure, or environmental condition.

OOD performance remains subject to the limitations of the available research data, baseline distribution, feature representation, and statistical methodology.

---

## 8. OOD Calibration Boundary

OOD calibration and uncertainty interpretation remain subject to scientific limitations.

Phase 6.1 is retained as:

`PASS_WITH_SCIENTIFIC_REVIEW_REQUIRED`

Phase 6.2 is retained as:

`PASS`

Phase 6.3 is retained as:

`SCIENTIFIC_REVIEW_ACCEPTED_WITH_LIMITATIONS`

Phase 6.4 is retained as:

`PASS_WITH_CALIBRATION_LIMITATIONS`

These statuses are part of the research evidence and must not be upgraded into unconditional claims of validated uncertainty or OOD performance.

The corrected Phase 6.2 results remove the identified constant-feature numerical artifact, but they do not establish universal calibration or real-world OOD reliability.

---

## 9. Predictive Performance Boundary

High predictive performance within the controlled experimental data does not establish real-world cyberattack-detection capability.

The reported results should be interpreted within the scope of the datasets, feature construction, temporal windows, synthetic cyber-physical scenarios, model checkpoint, and experimental methodology used in this research.

No claim is made that the model has been validated across all real-world vehicle architectures, ECU configurations, CAN traffic distributions, cyberattack types, road environments, or operating conditions.

---

## 10. Cortex XDR Boundary

The Cortex XDR component is an integration architecture and event-schema abstraction.

The project generates a:

`CORTEX_XDR_COMPATIBLE_RESEARCH_EVENT`

with:

`ABSTRACTION_ONLY`

The project does not claim live Cortex XDR deployment.

The project does not claim a live Cortex XDR API integration or production SOC deployment.

The SOC/XDR portion demonstrates architectural compatibility and research-event handoff concepts rather than operational deployment into a live security platform.

---

## 11. Connectivity and Resilience Boundary

The connectivity-resilience experiments evaluate the research state machine:

`CONNECTED → DISCONNECTED → LOCAL BUFFER → RECOVERED → INTEGRITY VERIFIED → SYNCHRONIZED → SOC HANDOFF`

The experiments demonstrate deterministic research behavior for connectivity loss, local buffering, recovery, integrity verification, synchronization, and SOC handoff under the defined scenarios.

They do not establish production-fleet resilience under every possible telecommunications failure, ECU failure, gateway failure, cloud failure, or adversarial operating condition.

---

## 12. Forensic Integrity Boundary

The forensic evidence path includes integrity verification and refusal behavior for invalid evidence.

An invalid evidence chain is treated as:

`UNVERIFIED_EVIDENCE → REFUSE_SYNC → REFUSE_SOC_HANDOFF`

This demonstrates the research governance principle that unverifiable evidence must not be silently synchronized or handed to the SOC as trusted evidence.

The integrity experiments do not constitute a formal certification of a production automotive forensic system.

---

## 13. Scope of Validation

The final research evidence covers controlled software and HIL-oriented experiments, reproducibility checks, integrity validation, resilience-state validation, uncertainty/OOD analysis, calibration analysis, and governance checks.

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

---

## 14. Interpretation of Final Research Results

The final results support the claim that the proposed research framework can be implemented and evaluated reproducibly under the defined experimental conditions.

They support architectural evidence for:

1. temporal predictive risk analysis;
2. uncertainty and OOD analysis;
3. deterministic resilience decision logic;
4. local forensic buffering;
5. evidence-integrity verification;
6. connectivity-loss recovery;
7. secure synchronization;
8. SOC/XDR-compatible research-event generation;
9. model-grounded explainability; and
10. reproducible evidence and governance workflows.

They do not support claims of production deployment readiness, real-world cyberattack detection validation, physical vehicle safety validation, or automotive safety certification.

---

## 15. Governance Rule for Research Claims

Experiment PASS status means that the corresponding defined research validation gate passed.

A PASS status does not automatically convert the result into a production, road, real-world, certification, or universal-performance claim.

Phase-specific limitations and scientific-review requirements must remain attached to the corresponding results in the final research release.

The final research statement must preserve these boundaries.
