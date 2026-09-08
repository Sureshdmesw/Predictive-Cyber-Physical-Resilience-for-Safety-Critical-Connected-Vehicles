#Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles

Research Project

This project presents a reproducible research framework for predictive cyber-physical resilience in safety-critical connected vehicles. It combines temporal AI, cyber telemetry, physical/safety signals, uncertainty estimation, deterministic resilience decisions, offline forensic buffering, integrity verification, connectivity recovery, SOC/XDR-compatible event generation, and model-grounded explainability.

Research boundary: The current implemented pipeline uses NHTSA CRSS 2024 as its physical-world safety/crash-data foundation together with synthetic cyber telemetry. CISS, FARS, SCI, NTS/NiTS, and ADAS/ADS reporting sources were investigated as possible future sources but are not integrated into the current modeling pipeline. This is a research framework, not a production vehicle cybersecurity system.

1. Research Objective

Traditional security monitoring is largely reactive:

Event → Detection → Alert → Investigation

This research investigates a predictive resilience pipeline:

Cyber + Physical + Temporal Telemetry
                ↓
        Temporal Risk Forecast
                ↓
      Ensemble Uncertainty / OOD
                ↓
      Calibrated Risk Assessment
                ↓
   Deterministic Resilience Policy
                ↓
 ┌──────────────┴──────────────┐
 ↓                             ↓
Local Resilience          SOC/XDR Event
+ Forensics               Abstraction
 ↓                             ↓
Connectivity Recovery          │
 ↓                             │
Integrity Verification         │
 ↓                             │
Secure Synchronization ────────┘
                ↓
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

CISS — Crash Investigation Sampling System

FARS — Fatality Analysis Reporting System

SCI — Special Crash Investigations

NTS/NiTS — Non-Traffic Surveillance

NHTSA ADAS/ADS Standing General Order reporting

They are not part of the current implemented modeling dataset.

3. Research Architecture

              NHTSA CRSS 2024
                    │
                    ▼
        Data + Semantic Validation
                    │
                    ▼
      Cyber / Physical / Temporal Model
                    │
                    ▼
          Temporal Transformer
                    │
             ┌──────┴──────┐
             ▼             ▼
       Risk Forecast   Uncertainty
             │             │
             └──────┬──────┘
                    ▼
       Deterministic Risk Policy
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Edge Resilience        SOC/XDR
   + Forensics            Abstraction
          │                   │
          ▼                   │
 Connectivity Loss            │
          │                   │
 Local Forensic Buffer         │
          │                   │
 Recovery                     │
          │                   │
 Integrity Verification ◄─────┘
          │
          ▼
 Secure Synchronization
          │
          ▼
 SOC Investigation + Explainability

4. Phase-by-Phase Research Record

Phase 4.1 — Research Foundation

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

Phase 4.2 — Safety/Cyber-Physical Data Strategy

Defined the separation between physical-world safety evidence and synthetic cyber telemetry. This distinction prevents synthetic attack labels from being presented as real-world cybersecurity ground truth.

Phase 4.3 — NHTSA Safety Intelligence Foundation

Investigated NHTSA sources and selected CRSS 2024 for the implemented safety-data foundation. Other sources remain documented future extensions.

Phase 4.4 — CRSS Data Engineering and Semantic Validation

Built the CRSS engineering layer, including 28 CSV tables, feature dictionary, canonical schema, key validation, relational checks, and a 21/21 release gate.

Contribution: data semantics and relationships were validated before ML experimentation.

Phase 4.5 — Temporal Forecasting Tensor Audit

Converted event-level information into temporal sequences capable of representing evolving cyber instability, physical disagreement, packet loss, latency, integrity degradation, and state transitions.

Contribution: shifted the research problem from static classification toward temporal resilience forecasting.

Phase 4.6 — Model Readiness

Validated that labels, features, temporal construction, schemas, and training inputs were sufficiently controlled before training.

Phase 4.7 — Feature Treatment

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

Phase 4.8 — Training Infrastructure

Established controlled CPU training, reproducibility configuration, checkpointing, and experiment controls.

5. Predictive Modeling

Phase 4.9 — Strong Gradient-Boosting Baseline

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

Phase 4.10 — Temporal Transformer

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

Phase 4.10C — Robustness and Ablation

Evaluated cyber-only, physical-only, temporal, fingerprint, and corruption conditions. Increasing corruption produced progressively lower predictive performance, demonstrating that telemetry quality is a resilience concern rather than merely a preprocessing concern.

Representative Y3 PR-AUC under corruption was approximately 0.942 at 5%, 0.884 at 10%, and 0.776 at 20% corruption.

Phase 4.11 — Three-Modality Representation

Formalized the cyber + physical + temporal representation of 59 features.

Phase 4.12 — Modality Contribution

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

Phase 4.13 — Feature Contribution

Investigated physical signals such as sensor_disagreement_rate and sensor_plausibility_score within cyber-plus-feature experiments, demonstrating measurable incremental contribution from physical-state evidence.

Phase 4.14 — Feature Optimization

Refined the representation for prediction, uncertainty, explainability, and resilience experiments rather than optimizing only one benchmark metric.

Phase 4.15 — Scenario and Leakage Stress Testing

Stress-tested scenario structure, temporal effects, and synthetic-label leakage concerns.

6. Generalization, Calibration, and Uncertainty

Phase 4.16 — Generalization

Under a more challenging generalization evaluation, Y6 test PR-AUC dropped to:

0.134872

This is a critical result: performance on the original synthetic distribution does not automatically generalize to shifted conditions.

Scenario-level and origin-level analysis showed substantial distribution sensitivity.

Contribution: the project preserved this negative result instead of reporting only the high original benchmark scores.

Phase 4.17 — Error Analysis

Analyzed prediction errors across changing cyber/physical operating regimes and distribution conditions.

Phase 4.18 — Calibration

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
LOW       0.05 – <0.20
MEDIUM    0.20 – <0.50
HIGH      0.50 – <0.80
CRITICAL  >=0.80

Phase 4.19 — Robustness Confirmation

Confirmed the major robustness and generalization observations and retained unfavorable findings as part of the evidence trail.

7. Predictive Resilience Decision Engine

Phase 4.20 — Deterministic Decision Layer

Separated probabilistic ML output from deterministic resilience policy.

Context rules include:

connectivity_degraded → minimum MEDIUM
sensor_disagreement   → minimum MEDIUM
integrity_failure     → minimum HIGH

The architecture explicitly enforces:

direct_vehicle_actuation = false

Contribution: prediction and safety-oriented policy are separated rather than allowing a neural network to directly control the vehicle.

8. Edge-IoT and Offline Resilience

Phase 4.21 — Edge-IoT Forensic Resilience

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

Phase 4.22 — Edge-IoT Scenario Evaluation

Evaluated representative connectivity, local operation, evidence-preservation, and recovery scenarios.

Phase 4.23 — End-to-End Edge-IoT Resilience

Connected local detection, forensic buffering, recovery, integrity verification, synchronization, and SOC handoff into one lifecycle.

Phase 4.24 — Stress and Failure Analysis

Examined resilience behavior under failure conditions and emphasized safe refusal rather than assuming every subsystem is trustworthy.

9. Evidence, Uncertainty, and OOD

Phase 4.25 — Evidence Manifest

Established an evidence-manifest approach for reproducible experimental artifacts.

Phase 4.26 — Temporal Regime Error Analysis

Extended error analysis into temporal operating regimes.

Phase 4.27A — Deep Ensemble Uncertainty

Built a three-seed Transformer ensemble using seeds 42, 52, and 62.

Representative ensemble test results:

PR-AUC: 0.975914

ROC-AUC: 0.997649

Brier: 0.002313

P90 uncertainty threshold: approximately 0.000658

high-uncertainty rate: approximately 9.71%

The ensemble demonstrated that model predictions should not be treated as equally trustworthy for every input.

Phase 4.27B — Static OOD

Evaluated cyber, physical, combined, temporal-mask, and distribution-shift conditions.

The final static OOD artifact did not pass every check.

Phase 4.27B.1 — Temporal Dynamics OOD

Tested temporal masking and temporal perturbations. The final artifact did not pass every detection check.

Phase 4.27B.2 — Temporal Consistency OOD

Added GRU-based temporal consistency evaluation. The final artifact did not pass every check.

Phase 4.27B.3 — Frozen Representation OOD

Evaluated OOD behavior using frozen Transformer representations. The final artifact did not pass every check.

Scientific importance

These negative results are intentionally preserved. They demonstrate that:

synthetic separability can inflate apparent performance,

distribution shift is difficult,

temporal OOD is not solved by ordinary feature perturbation,

uncertainty estimation does not automatically provide reliable OOD detection.

10. SOC/XDR Integration and Recovery

Phase 4.28A — Predictive-to-SOC Pipeline

Created the research event abstraction:

CORTEX_XDR_COMPATIBLE_RESEARCH_EVENT

with:

integration_mode = ABSTRACTION_ONLY
live_cortex_xdr = false

The system does not claim a live Cortex XDR deployment.

Phase 4.28B — Connectivity Recovery and Synchronization

Formalized:

CONNECTED
 → DISCONNECTED
 → LOCAL BUFFER
 → RECOVERED
 → INTEGRITY VERIFIED
 → SYNCHRONIZED
 → SOC HANDOFF

Phase 4.28C — Adversarial Integrity/Tamper Detection

Tested payload tampering and previous-hash tampering.

Tampered evidence is routed toward:

UNVERIFIED EVIDENCE
 → REFUSE SYNC
 → REFUSE SOC HANDOFF

Contribution: cryptographic evidence integrity is tied directly to operational recovery behavior.

11. Formal Resilience and Fault Validation

Phase 4.29 — Resilience State Machine

The validated lifecycle is:

NORMAL
 → PREDICTIVE_RISK
 → CONNECTIVITY_DEGRADED
 → DISCONNECTED
 → LOCAL_FORENSIC_BUFFER
 → RECOVERY
 → INTEGRITY_VERIFICATION
 → SYNCHRONIZED
 → SOC_HANDOFF

Tamper path:

UNVERIFIED_EVIDENCE
 → REFUSE_SYNC
 → REFUSE_SOC_HANDOFF

14 / 14 checks passed.

Phase 4.30 — Fault Injection and Recovery

Evaluated seven representative fault/recovery scenarios.

16 / 16 checks passed.

The evaluator is explicitly synthetic/deterministic and does not constitute physical-vehicle fault validation.

12. Model-Grounded Explainability

Phase 4.31 — Explainable Evidence Correlation

Established the first explanation/evidence layer and identified the limitation of manually assigned risk drivers and fixed scenario probabilities.

This motivated actual model-grounded explanations.

Phase 4.32 — Model-Grounded Explainability

Implemented explanations derived from the frozen predictive model, including:

model attribution

temporal evidence

counterfactual reasoning

model/checkpoint identity

reproducibility metadata

The canonical schema uses strict validation and rejects undeclared properties.

Phase 4.32B — Origin-Balanced Explainability

Evaluated 50 unique scenarios, with 25 negative and 25 positive targets across multiple origins:

Origin 11 → 8
Origin 12 → 8
Origin 13 → 8
Origin 14 → 8
Origin 15 → 6
Origin 16 → 6
Origin 17 → 6

Phase 4.32C — Modality-Grounded Explainability

Grounded explanations in the same research modalities:

Physical = 12
Cyber = 34
Derived temporal resilience = 13
Total = 59

Contribution: explainability is tied to the actual cyber-physical representation rather than being a generic feature-ranking exercise.

13. Explainability Stability and Final Validation

Phase 4.33 — Explainability Stability

Final evaluation:

50 scenarios

25 negative / 25 positive

mean top-10 Jaccard: 0.743192

median top-10 Jaccard: 0.717172

mean probability standard deviation: 0.122652

status: PASS

Phase 4.34 — Predictive Resilience Decision Validation

16 / 16 checks passed.

Validated risk levels, contextual modifiers, safety boundaries, and refusal behavior.

Phase 4.35 — End-to-End Adversarial Resilience

14 / 14 checks passed.

Validated the complete predictive-risk → resilience → integrity → recovery → SOC path.

Phase 4.36 — Failure-Mode and Safety-Boundary Audit

22 / 22 checks passed.

Audited failure modes, evidence integrity, direct-actuation restrictions, recovery, synchronization refusal, and research/production boundaries.

Phase 4.37 — Final Integrated System Validation

32 / 32 checks passed.

Known OOD limitations remain documented as limitations rather than being converted into PASS results.

Phase 4.38 — Reproducibility Evidence Audit

53 / 53 checks passed.

Audited the reproducibility evidence required for the research release.

Phase 4.39 — Research Governance and Safety Claims Audit

19 / 19 checks passed.

Verified that public claims remain within the evidence actually produced.

Phase 4.40 — Final Release Manifest

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
        ↓
   validated safety representation

2. PREDICTIVE INTELLIGENCE
   59 features
        ↓
   Temporal Transformer
        ↓
   risk + ensemble uncertainty

3. RESILIENCE INTELLIGENCE
   calibrated probability
        ↓
   deterministic risk policy

4. OFFLINE CYBER-PHYSICAL RESILIENCE
   connectivity loss
        ↓
   local forensic buffer
        ↓
   hash-chain integrity
        ↓
   recovery + verification
        ↓
   secure synchronization

5. SOC + EXPLAINABILITY
   trusted evidence
        ↓
   XDR-compatible research event
        ↓
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

The final release was subjected to integrated validation through Phases 4.37–4.40.

19. Repository Structure

Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles/
│
├── data/
│   ├── raw/
│   │   └── nhtsa/CRSS/
│   └── schemas/
│       ├── nhtsa/
│       ├── edge_iot/
│       └── explainability/
│
├── experiments/
│   ├── nhtsa/
│   ├── modeling/
│   ├── edge_iot/
│   ├── integration/
│   ├── governance/
│   └── release/
│
├── models/
│   └── v2/
│
├── scripts/
│   ├── governance/
│   ├── integration/
│   └── ...
│
├── LICENSE
└── README.md

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

Real cybersecurity data — integrate appropriately labeled connected-vehicle/automotive cyber telemetry.

Multi-source NHTSA validation — integrate CISS, FARS, SCI, NTS/NiTS, or ADAS/ADS sources only after semantic and sampling validation.

ECU/CAN validation — evaluate controlled ECU, CAN/CAN-FD, gateway, sensor, and edge-device scenarios.

Stronger OOD methods — investigate conformal prediction, energy-based OOD, representation methods, temporal contrastive learning, and distribution-aware ensembles.

Automotive safety/cybersecurity processes — map the research to appropriate ISO 26262 and ISO/SAE 21434 workflows.

Authorized SOC/XDR integration — connect the research event abstraction to an enterprise XDR/SOC environment without coupling it to direct vehicle actuation.

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
