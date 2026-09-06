# Research Questions

## Primary Research Question

Can an Edge-IoT cyber-physical resilience architecture detect and classify attacks against safety-critical connected vehicle systems, predict impending communication degradation or loss, provide sufficient warning to enterprise security operations, transition safety-critical telemetry to an independent resilient communication path, preserve forensic evidence, and improve fleet-level intelligence using historical NHTSA safety data?

## Secondary Research Questions

### RQ1 — Cyber-Physical Detection

Can behavioral models detect coordinated anomalies across:

- CAN communications
- ECU behavior
- physical sensors
- occupant state
- ADAS signals
- vehicle dynamics
- diagnostic activity

rather than treating each telemetry source independently?

### RQ2 — Cyber-Physical Correlation

Can apparently independent anomalies be correlated into a unified cyber-physical incident?

### RQ3 — Predictive Connectivity Resilience

Can the system predict the probability and approximate time window of primary telemetry-channel degradation or loss before complete communication failure?

### RQ4 — Attack vs Failure

Can the system distinguish probable malicious communication suppression from benign connectivity degradation?

### RQ5 — Edge Resilience

Can the vehicle maintain local detection, containment, safety telemetry and forensic recording when enterprise connectivity becomes unavailable?

### RQ6 — Auxiliary Communication

Can safety-critical telemetry be dynamically prioritized and transferred to an independent auxiliary communication path before primary connectivity is lost?

### RQ7 — AI Security Reasoning

Can an AI reasoning layer explain correlated cyber-physical incidents and provide evidence-based response recommendations without directly controlling safety-critical actuation?

### RQ8 — Enterprise Security Operations

Can Cortex XDR provide an enterprise-level view of vehicle cyber-physical incidents and predictive connectivity threats?

### RQ9 — Safety Intelligence

Can historical NHTSA datasets provide contextual information for estimating potential safety consequences associated with cyber-physical anomalies?

### RQ10 — Fleet Learning

Can patterns observed across multiple simulated vehicles improve anomaly detection, threat classification and resilience decisions across the fleet?

## Core Hypothesis

A layered Edge-first architecture combining behavioral cyber-physical detection, predictive connectivity analysis, AI-assisted security reasoning, deterministic safety-constrained response, resilient communication, forensic preservation, and NHTSA-derived safety intelligence can provide earlier and more reliable response to cyber-physical threats than a connectivity-dependent detection architecture.

## Safety Constraint

AI components shall not possess unrestricted authority over safety-critical vehicle actuation.

AI shall provide analysis, classification, explanation and recommendations.

Deterministic safety policies shall govern safety-critical actions.
