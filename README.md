#Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles

Research Project

This project presents a reproducible research framework for predictive cyber-physical resilience in safety-critical connected vehicles. It combines temporal AI, cyber telemetry, physical/safety signals, uncertainty estimation, deterministic resilience decisions, offline forensic buffering, integrity verification, connectivity recovery, SOC/XDR-compatible event generation, and model-grounded explainability.

Research boundary: The current implemented pipeline uses NHTSA CRSS 2024 as its physical-world safety/crash-data foundation together with synthetic cyber telemetry. CISS, FARS, SCI, NTS/NiTS, and ADAS/ADS reporting sources were investigated as possible future sources but are not integrated into the current modeling pipeline. This is a research framework, not a production vehicle cybersecurity system.

1. Research Objective

Traditional security monitoring is largely reactive:

Event â†’ Detection â†’ Alert â†’ Investigation

This research investigates a predictive resilience pipeline:

Cyber + Physical + Temporal Telemetry
                â†“
        Temporal Risk Forecast
                â†“
      Ensemble Uncertainty / OOD
                â†“
      Calibrated Risk Assessment
                â†“
   Deterministic Resilience Policy
                â†“
 â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
 â†“                             â†“
Local Resilience          SOC/XDR Event
+ Forensics               Abstraction
 â†“                             â†“
Connectivity Recovery          â”‚
 â†“                             â”‚
Integrity Verification         â”‚
 â†“                             â”‚
Secure Synchronization â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                â†“
      Model-Grounded Evidence

The central principle is:

AI predicts risk; deterministic policy interprets risk; local resilience preserves security evidence; integrity verification controls synchronization; SOC/XDR-compatible events support investigation.

The framework does not directly actuate a vehicle.

2. Data Foundation

NHTSA CRSS 2024

The implemented physical-world safety foundation is the National Highway Traffic Safety Administration Crash Report Sampling System (CRSS) 2024.

The project performed:

CRSS 2024 extraction and inspection

feature dictionary construction

semantic feature verification

canonical schema creation

primary-key validation

relational-integrity validation

vehicle/person relationship analysis

VE_FORMS consistency validation

release-gate testing

temporal feature construction

model-readiness validation

CRSS release gate

21 / 21 checks passed.

A notable relationship issue involving VEH_NO = 0 was investigated and corrected without blindly discarding semantically meaningful records. Vehicle-associated orphan references were characterized and retained where appropriate.

Additional NHTSA sources

The following were researched as future extensions:

CISS â€” Crash Investigation Sampling System

FARS â€” Fatality Analysis Reporting System

SCI â€” Special Crash Investigations

NTS/NiTS â€” Non-Traffic Surveillance

NHTSA ADAS/ADS Standing General Order reporting

They are not part of the current implemented modeling dataset.

3. Research Architecture

              NHTSA CRSS 2024
                    â”‚
                    â–¼
        Data + Semantic Validation
                    â”‚
                    â–¼
      Cyber / Physical / Temporal Model
                    â”‚
                    â–¼
          Temporal Transformer
                    â”‚
             â”Œâ”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”
             â–¼             â–¼
       Risk Forecast   Uncertainty
             â”‚             â”‚
             â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜
                    â–¼
       Deterministic Risk Policy
                    â”‚
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
          â–¼                   â–¼
   Edge Resilience        SOC/XDR
   + Forensics            Abstraction
          â”‚                   â”‚
          â–¼                   â”‚
 Connectivity Loss            â”‚
          â”‚                   â”‚
 Local Forensic Buffer         â”‚
          â”‚                   â”‚
 Recovery                     â”‚
          â”‚                   â”‚
 Integrity Verification â—„â”€â”€â”€â”€â”€â”˜
          â”‚
          â–¼
 Secure Synchronization
          â”‚
          â–¼
 SOC Investigation + Explainability

4. Phase-by-Phase Research Record

Phase 4.1 â€” Research Foundation

Defined predictive cyber-physical resilience as the core problem rather than treating vehicle cybersecurity as isolated intrusion detection.

The research scope connects:

automotive cybersecurity

AI/ML

embedded and edge systems

physical safety signals

connectivity resilience

forensic evidence

integrity verification

SOC investigation

explainability

Contribution: established the integrated research direction.

Phase 4.2 â€” Safety/Cyber-Physical Data Strategy

Defined the separation between physical-world safety evidence and synthetic cyber telemetry. This distinction prevents synthetic attack labels from being presented as real-world cybersecurity ground truth.

Phase 4.3 â€” NHTSA Safety Intelligence Foundation

Investigated NHTSA sources and selected CRSS 2024 for the implemented safety-data foundation. Other sources remain documented future extensions.

Phase 4.4 â€” CRSS Data Engineering and Semantic Validation

Built the CRSS engineering layer, including 28 CSV tables, feature dictionary, canonical schema, key validation, relational checks, and a 21/21 release gate.

Contribution: data semantics and relationships were validated before ML experimentation.

Phase 4.5 â€” Temporal Forecasting Tensor Audit

Converted event-level information into temporal sequences capable of representing evolving cyber instability, physical disagreement, packet loss, latency, integrity degradation, and state transitions.

Contribution: shifted the research problem from static classification toward temporal resilience forecasting.

Phase 4.6 â€” Model Readiness

Validated that labels, features, temporal construction, schemas, and training inputs were sufficiently controlled before training.

Phase 4.7 â€” Feature Treatment

Established the three-modality feature organization:

Modality

Features

Cyber

34

Physical

12

Derived temporal resilience

13

Total

59

Phase 4.8 â€” Training Infrastructure

Established controlled CPU training, reproducibility configuration, checkpointing, and experiment controls.

5. Predictive Modeling

Phase 4.9 â€” Strong Gradient-Boosting Baseline

A Histogram Gradient Boosting baseline achieved very high scores on the original synthetic-label distribution:

Target

Validation PR-AUC

Test PR-AUC

Y3

0.997166

0.996728

Y6

0.993042

0.989544

A later synthetic-label separability audit showed that the synthetic labels contained highly learnable fingerprints. Therefore these scores are not interpreted as real-world cyberattack detection performance.

Contribution: the baseline established the initial predictive benchmark and exposed the need for stronger validation.

Phase 4.10 â€” Temporal Transformer

Implemented a temporal Transformer with:

59 input features

128-dimensional embedding

learned positional encoding

sequence length 12

3 Transformer encoder layers

8 attention heads

256-dimensional feed-forward layer

0.10 dropout

GELU

attention pooling

separate Y3/Y6 heads

Representative controlled test results:

Target

PR-AUC

ROC-AUC

F1

Brier

Y3

0.778333

0.998590

0.878722

0.002866

Y6

0.950336

0.998166

0.881502

0.003042

Phase 4.10C â€” Robustness and Ablation

Evaluated cyber-only, physical-only, temporal, fingerprint, and corruption conditions. Increasing corruption produced progressively lower predictive performance, demonstrating that telemetry quality is a resilience concern rather than merely a preprocessing concern.

Representative Y3 PR-AUC under corruption was approximately 0.942 at 5%, 0.884 at 10%, and 0.776 at 20% corruption.

Phase 4.11 â€” Three-Modality Representation

Formalized the cyber + physical + temporal representation of 59 features.

Phase 4.12 â€” Modality Contribution

Y6 test PR-AUC by modality combination:

Modality

PR-AUC

Cyber only

0.763062

Physical only

0.626046

Derived temporal only

0.608888

Cyber + physical

0.766964

Cyber + derived temporal

0.784837

Physical + derived temporal

0.661303

All 59

0.786068

This supports the research hypothesis that cyber, physical, and temporal information can provide complementary predictive information.

Phase 4.13 â€” Feature Contribution

Investigated physical signals such as sensor_disagreement_rate and sensor_plausibility_score within cyber-plus-feature experiments, demonstrating measurable incremental contribution from physical-state evidence.

Phase 4.14 â€” Feature Optimization

Refined the representation for prediction, uncertainty, explainability, and resilience experiments rather than optimizing only one benchmark metric.

Phase 4.15 â€” Scenario and Leakage Stress Testing

Stress-tested scenario structure, temporal effects, and synthetic-label leakage concerns.

6. Generalization, Calibration, and Uncertainty

Phase 4.16 â€” Generalization

Under a more challenging generalization evaluation, Y6 test PR-AUC dropped to:

0.134872

This is a critical result: performance on the original synthetic distribution does not automatically generalize to shifted conditions.

Scenario-level and origin-level analysis showed substantial distribution sensitivity.

Contribution: the project preserved this negative result instead of reporting only the high original benchmark scores.

Phase 4.17 â€” Error Analysis

Analyzed prediction errors across changing cyber/physical operating regimes and distribution conditions.

Phase 4.18 â€” Calibration

Applied validation-only isotonic calibration.

Metric

Validation

Test

PR-AUC

0.177382

0.134872

ROC-AUC

0.678982

0.666617

Brier after calibration

0.008394

0.010358

Risk policy:

NORMAL    < 0.05
LOW       0.05 â€“ <0.20
MEDIUM    0.20 â€“ <0.50
HIGH      0.50 â€“ <0.80
CRITICAL  >=0.80

Phase 4.19 â€” Robustness Confirmation

Confirmed the major robustness and generalization observations and retained unfavorable findings as part of the evidence trail.

7. Predictive Resilience Decision Engine

Phase 4.20 â€” Deterministic Decision Layer

Separated probabilistic ML output from deterministic resilience policy.

Context rules include:

connectivity_degraded â†’ minimum MEDIUM
sensor_disagreement   â†’ minimum MEDIUM
integrity_failure     â†’ minimum HIGH

The architecture explicitly enforces:

direct_vehicle_actuation = false

Contribution: prediction and safety-oriented policy are separated rather than allowing a neural network to directly control the vehicle.

8. Edge-IoT and Offline Resilience

Phase 4.21 â€” Edge-IoT Forensic Resilience

Implemented a research abstraction for local containment and forensic buffering with hash-linked records.

Core fields include:

event_id
timestamp
vehicle_id
ecu_id
can_id
event_type
severity
connectivity_state
integrity_status
sensor_consistency_status
payload_hash
previous_hash
record_hash

The bounded rolling buffer uses a ROLLING_GENESIS re-anchor to preserve hash-chain validity after rollover.

The Cortex XDR relationship is explicitly research-compatible abstraction only; no live tenant/API integration is claimed.

Phase 4.22 â€” Edge-IoT Scenario Evaluation

Evaluated representative connectivity, local operation, evidence-preservation, and recovery scenarios.

Phase 4.23 â€” End-to-End Edge-IoT Resilience

Connected local detection, forensic buffering, recovery, integrity verification, synchronization, and SOC handoff into one lifecycle.

Phase 4.24 â€” Stress and Failure Analysis

Examined resilience behavior under failure conditions and emphasized safe refusal rather than assuming every subsystem is trustworthy.

9. Evidence, Uncertainty, and OOD

Phase 4.25 â€” Evidence Manifest

Established an evidence-manifest approach for reproducible experimental artifacts.

Phase 4.26 â€” Temporal Regime Error Analysis

Extended error analysis into temporal operating regimes.

Phase 4.27A â€” Deep Ensemble Uncertainty

Built a three-seed Transformer ensemble using seeds 42, 52, and 62.

Representative ensemble test results:

PR-AUC: 0.975914

ROC-AUC: 0.997649

Brier: 0.002313

P90 uncertainty threshold: approximately 0.000658

high-uncertainty rate: approximately 9.71%

The ensemble demonstrated that model predictions should not be treated as equally trustworthy for every input.

Phase 4.27B â€” Static OOD

Evaluated cyber, physical, combined, temporal-mask, and distribution-shift conditions.

The final static OOD artifact did not pass every check.

Phase 4.27B.1 â€” Temporal Dynamics OOD

Tested temporal masking and temporal perturbations. The final artifact did not pass every detection check.

Phase 4.27B.2 â€” Temporal Consistency OOD

Added GRU-based temporal consistency evaluation. The final artifact did not pass every check.

Phase 4.27B.3 â€” Frozen Representation OOD

Evaluated OOD behavior using frozen Transformer representations. The final artifact did not pass every check.

Scientific importance

These negative results are intentionally preserved. They demonstrate that:

synthetic separability can inflate apparent performance,

distribution shift is difficult,

temporal OOD is not solved by ordinary feature perturbation,

uncertainty estimation does not automatically provide reliable OOD detection.

10. SOC/XDR Integration and Recovery

Phase 4.28A â€” Predictive-to-SOC Pipeline

Created the research event abstraction:

CORTEX_XDR_COMPATIBLE_RESEARCH_EVENT

with:

integration_mode = ABSTRACTION_ONLY
live_cortex_xdr = false

The system does not claim a live Cortex XDR deployment.

Phase 4.28B â€” Connectivity Recovery and Synchronization

Formalized:

CONNECTED
 â†’ DISCONNECTED
 â†’ LOCAL BUFFER
 â†’ RECOVERED
 â†’ INTEGRITY VERIFIED
 â†’ SYNCHRONIZED
 â†’ SOC HANDOFF

Phase 4.28C â€” Adversarial Integrity/Tamper Detection

Tested payload tampering and previous-hash tampering.

Tampered evidence is routed toward:

UNVERIFIED EVIDENCE
 â†’ REFUSE SYNC
 â†’ REFUSE SOC HANDOFF

Contribution: cryptographic evidence integrity is tied directly to operational recovery behavior.

11. Formal Resilience and Fault Validation

Phase 4.29 â€” Resilience State Machine

The validated lifecycle is:

NORMAL
 â†’ PREDICTIVE_RISK
 â†’ CONNECTIVITY_DEGRADED
 â†’ DISCONNECTED
 â†’ LOCAL_FORENSIC_BUFFER
 â†’ RECOVERY
 â†’ INTEGRITY_VERIFICATION
 â†’ SYNCHRONIZED
 â†’ SOC_HANDOFF

Tamper path:

UNVERIFIED_EVIDENCE
 â†’ REFUSE_SYNC
 â†’ REFUSE_SOC_HANDOFF

14 / 14 checks passed.

Phase 4.30 â€” Fault Injection and Recovery

Evaluated seven representative fault/recovery scenarios.

16 / 16 checks passed.

The evaluator is explicitly synthetic/deterministic and does not constitute physical-vehicle fault validation.

12. Model-Grounded Explainability

Phase 4.31 â€” Explainable Evidence Correlation

Established the first explanation/evidence layer and identified the limitation of manually assigned risk drivers and fixed scenario probabilities.

This motivated actual model-grounded explanations.

Phase 4.32 â€” Model-Grounded Explainability

Implemented explanations derived from the frozen predictive model, including:

model attribution

temporal evidence

counterfactual reasoning

model/checkpoint identity

reproducibility metadata

The canonical schema uses strict validation and rejects undeclared properties.

Phase 4.32B â€” Origin-Balanced Explainability

Evaluated 50 unique scenarios, with 25 negative and 25 positive targets across multiple origins:

Origin 11 â†’ 8
Origin 12 â†’ 8
Origin 13 â†’ 8
Origin 14 â†’ 8
Origin 15 â†’ 6
Origin 16 â†’ 6
Origin 17 â†’ 6

Phase 4.32C â€” Modality-Grounded Explainability

Grounded explanations in the same research modalities:

Physical = 12
Cyber = 34
Derived temporal resilience = 13
Total = 59

Contribution: explainability is tied to the actual cyber-physical representation rather than being a generic feature-ranking exercise.

13. Explainability Stability and Final Validation

Phase 4.33 â€” Explainability Stability

Final evaluation:

50 scenarios

25 negative / 25 positive

mean top-10 Jaccard: 0.743192

median top-10 Jaccard: 0.717172

mean probability standard deviation: 0.122652

status: PASS

Phase 4.34 â€” Predictive Resilience Decision Validation

16 / 16 checks passed.

Validated risk levels, contextual modifiers, safety boundaries, and refusal behavior.

Phase 4.35 â€” End-to-End Adversarial Resilience

14 / 14 checks passed.

Validated the complete predictive-risk â†’ resilience â†’ integrity â†’ recovery â†’ SOC path.

Phase 4.36 â€” Failure-Mode and Safety-Boundary Audit

22 / 22 checks passed.

Audited failure modes, evidence integrity, direct-actuation restrictions, recovery, synchronization refusal, and research/production boundaries.

Phase 4.37 â€” Final Integrated System Validation

32 / 32 checks passed.

Known OOD limitations remain documented as limitations rather than being converted into PASS results.

Phase 4.38 â€” Reproducibility Evidence Audit

53 / 53 checks passed.

Audited the reproducibility evidence required for the research release.

Phase 4.39 â€” Research Governance and Safety Claims Audit

19 / 19 checks passed.

Verified that public claims remain within the evidence actually produced.

Phase 4.40 â€” Final Release Manifest

13 / 13 checks passed.

The final release manifest records the research boundary, required artifacts, schemas, model checkpoints, validation reports, and governance evidence.

14. Main Research Contributions

1. Predictive cyber-physical resilience

Moves the research question from reactive intrusion detection toward prediction of evolving cyber-physical risk.

2. Cyber + physical + temporal intelligence

Combines 34 cyber, 12 physical, and 13 derived temporal-resilience features.

3. Predictive risk before connectivity failure

Creates a conceptual early-warning layer that can inform SOC awareness and local resilience preparation.

4. AI separated from safety control

The ML model predicts risk; deterministic policy handles resilience decisions; direct vehicle actuation remains disabled.

5. Offline-first forensic resilience

Connectivity loss does not automatically mean loss of security evidence.

6. Tamper-aware recovery

Recovered evidence must pass integrity verification before synchronization and SOC handoff.

7. SOC/XDR-compatible research abstraction

Provides a Cortex XDR-compatible event abstraction without falsely claiming a live Cortex XDR deployment.

8. Model-grounded explainability

Uses actual frozen-model evidence, temporal reasoning, counterfactuals, and modality grounding.

9. Explainability stability

Introduces explicit stability testing rather than assuming one generated explanation is trustworthy.

10. Negative-result preservation

OOD and generalization failures remain part of the research evidence. This prevents benchmark performance from being mistaken for general real-world capability.

15. What Was Actually Built

The final research system can be summarized as five layers:

1. SAFETY INTELLIGENCE
   NHTSA CRSS 2024
        â†“
   validated safety representation

2. PREDICTIVE INTELLIGENCE
   59 features
        â†“
   Temporal Transformer
        â†“
   risk + ensemble uncertainty

3. RESILIENCE INTELLIGENCE
   calibrated probability
        â†“
   deterministic risk policy

4. OFFLINE CYBER-PHYSICAL RESILIENCE
   connectivity loss
        â†“
   local forensic buffer
        â†“
   hash-chain integrity
        â†“
   recovery + verification
        â†“
   secure synchronization

5. SOC + EXPLAINABILITY
   trusted evidence
        â†“
   XDR-compatible research event
        â†“
   model-grounded explanation

16. Scientific and Engineering Significance

The strongest contribution is not a single accuracy number. It is the integration of research problems that are often treated separately:

Safety data
 +
Cyber telemetry
 +
Temporal AI
 +
Uncertainty
 +
Deterministic resilience
 +
Offline forensics
 +
Integrity verification
 +
Connectivity recovery
 +
SOC/XDR integration
 +
Explainability
 +
Safety-boundary verification
 +
Reproducibility

The resulting framework is an auditable research architecture for predictive cyber-physical resilience.

The research also demonstrates an important methodological principle:

A safety-critical cyber-physical AI system should not be evaluated only by predictive accuracy. It should also be evaluated under uncertainty, distribution shift, connectivity loss, evidence tampering, subsystem failure, recovery, explainability instability, and explicit safety boundaries.

17. Limitations

The current release does not provide:

a physical vehicle

production ECU/CAN integration

live vehicle-network traffic

real-world connected-vehicle cyberattack labels

live Cortex XDR tenant/API deployment

direct vehicle actuation

production fleet deployment

ISO 26262 certification

ASIL certification

complete OOD detection

multi-source NHTSA modeling

real-world cybersecurity accuracy validation

The cyber labels used by the implemented predictive pipeline are synthetic.

Therefore:

High predictive performance on synthetic data must not be interpreted as proof of real-world connected-vehicle attack detection capability.

The generalization and OOD experiments are intentionally included to expose this limitation.

18. Reproducibility and Governance

The repository maintains:

canonical schemas

CRSS feature dictionary

experiment reports

frozen model checkpoints

SHA-256 hashes

artifact registry

phase evidence

dependency validation

final release manifest

compilation validation

safety-claim governance

explicit research boundaries

The final release was subjected to integrated validation through Phases 4.37â€“4.40.

19. Repository Structure

Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles/
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â”‚   â””â”€â”€ nhtsa/CRSS/
â”‚   â””â”€â”€ schemas/
â”‚       â”œâ”€â”€ nhtsa/
â”‚       â”œâ”€â”€ edge_iot/
â”‚       â””â”€â”€ explainability/
â”‚
â”œâ”€â”€ experiments/
â”‚   â”œâ”€â”€ nhtsa/
â”‚   â”œâ”€â”€ modeling/
â”‚   â”œâ”€â”€ edge_iot/
â”‚   â”œâ”€â”€ integration/
â”‚   â”œâ”€â”€ governance/
â”‚   â””â”€â”€ release/
â”‚
â”œâ”€â”€ models/
â”‚   â””â”€â”€ v2/
â”‚
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ governance/
â”‚   â”œâ”€â”€ integration/
â”‚   â””â”€â”€ ...
â”‚
â”œâ”€â”€ LICENSE
â””â”€â”€ README.md

20. Frozen Model Checkpoints

The final release preserves three frozen Transformer checkpoints.

Seed 42
7f1e099b2287d45daa5c241b30219100e4a5e4b1cff5649182e1c481c64e3c6a

Seed 52
7aaa5a29168f4b5cbcec14008c4dad17a4d9a8111690ddb1eecdc24e1327e844

Seed 62
47212e284dc47c1351fbf1aada838e40041cf69f9d91498c5bdf428b99ee8ac0

These hashes support reproducibility and model identity tracking.

21. Final Validation Summary

Validation

Result

CRSS 2024 release gate

21 / 21 PASS

Resilience state machine

14 / 14 PASS

Fault injection/recovery

16 / 16 PASS

End-to-end adversarial resilience

14 / 14 PASS

Safety-boundary audit

22 / 22 PASS

Final integrated validation

32 / 32 PASS

Reproducibility evidence audit

53 / 53 PASS

Governance/safety-claims audit

19 / 19 PASS

Final release manifest

13 / 13 PASS

Explainability stability

PASS

Static/temporal OOD experiments

Limitations preserved; not all checks passed

22. Final Research Statement

This project presents a reproducible research framework for predictive cyber-physical resilience in safety-critical connected vehicles. The current release uses NHTSA CRSS 2024 as its physical-world safety-data foundation and combines it with synthetic cyber telemetry to investigate temporal risk forecasting, uncertainty, deterministic resilience decisions, offline forensic buffering, integrity verification, connectivity recovery, SOC/XDR-compatible event generation, and model-grounded explainability. The framework is experimentally validated as a research system, while real-world cyberattack data, additional NHTSA datasets, live Cortex XDR integration, physical vehicle deployment, and direct vehicle actuation remain outside the current release boundary.

23. Future Research

The strongest next steps are:

Real cybersecurity data â€” integrate appropriately labeled connected-vehicle/automotive cyber telemetry.

Multi-source NHTSA validation â€” integrate CISS, FARS, SCI, NTS/NiTS, or ADAS/ADS sources only after semantic and sampling validation.

ECU/CAN validation â€” evaluate controlled ECU, CAN/CAN-FD, gateway, sensor, and edge-device scenarios.

Stronger OOD methods â€” investigate conformal prediction, energy-based OOD, representation methods, temporal contrastive learning, and distribution-aware ensembles.

Automotive safety/cybersecurity processes â€” map the research to appropriate ISO 26262 and ISO/SAE 21434 workflows.

Authorized SOC/XDR integration â€” connect the research event abstraction to an enterprise XDR/SOC environment without coupling it to direct vehicle actuation.

24. Closing Perspective

The project evolved from an automotive cybersecurity problem into a broader predictive cyber-physical resilience framework.

Its central architecture is:

AI prediction
      +
uncertainty
      +
deterministic policy
      +
edge resilience
      +
forensic integrity
      +
connectivity recovery
      +
SOC investigation
      +
explainability
      +
formal validation
      +
reproducibility

The final research lesson is:

Resilience in a safety-critical cyber-physical AI system is not demonstrated by accuracy alone. It is demonstrated by how the system behaves when predictions are uncertain, distributions shift, connectivity disappears, evidence is tampered with, components fail, recovery occurs, explanations vary, and safety boundaries must be enforced.

Project Status

Research Release: COMPLETE

Current physical-world dataset: NHTSA CRSS 2024
Cyber telemetry: Synthetic research telemetry
Production vehicle deployment: No
Direct vehicle actuation: No
Live Cortex XDR API: No
Research validation: Complete within the documented release boundary

<!-- FINAL_RESEARCH_RELEASE_START -->

# Final Research Release

## Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles

### Final Research Status

The research framework has completed integrated validation, evidence consolidation, reproducibility validation, and claims/limitations governance.

| Release Area | Result |
|---|---:|
| Final integrated audit | PASS â€” 10/10 |
| Evidence consolidation | 161 evidence files |
| Manifest validation | 21 valid / 0 invalid |
| Reproducibility audit | PASS â€” 13/13 |
| Claims & limitations audit | PASS â€” 10/10 |
| Phase 5 manifests | 14 |
| Phase 6 manifests | 4 |
| Model input dimensionality | 59 |
| Frozen model checkpoint | Verified by SHA-256 |

### Research Contribution

The project presents a reproducible research framework that connects:

**Observation â†’ Prediction â†’ Uncertainty/OOD â†’ Deterministic Decision â†’ Forensic Preservation â†’ Integrity Verification â†’ Connectivity-Resilient Recovery â†’ SOC/XDR-Compatible Investigation**

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

`models\v2\advanced\crss_2024_v2_temporal_transformer_best.pt`

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

- Phase 6.1 â€” `PASS_WITH_SCIENTIFIC_REVIEW_REQUIRED`
- Phase 6.2 â€” `PASS`
- Phase 6.3 â€” `SCIENTIFIC_REVIEW_ACCEPTED_WITH_LIMITATIONS`
- Phase 6.4 â€” `PASS_WITH_CALIBRATION_LIMITATIONS`

### Resilience Architecture

The validated resilience state machine is:

`CONNECTED â†’ DISCONNECTED â†’ LOCAL BUFFER â†’ RECOVERED â†’ INTEGRITY VERIFIED â†’ SYNCHRONIZED â†’ SOC HANDOFF`

Invalid evidence follows:

`UNVERIFIED_EVIDENCE â†’ REFUSE_SYNC â†’ REFUSE_SOC_HANDOFF`

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

- `experiments\final_release\final_integrated_audit_v2.json`
- `experiments\final_release\final_evidence_consolidation.json`
- `experiments\final_release\final_evidence_index.csv`
- `experiments\final_release\final_reproducibility_audit.json`
- `experiments\final_release\final_claims_limitations_audit.json`

Authoritative documentation:

- `docs\FINAL_RESEARCH_DOCUMENTATION.md`
- `docs\final_research_claims_and_limitations.md`

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

<!-- FINAL_RESEARCH_RELEASE_END -->


