# System Architecture

# Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles

## 1. Architectural Objective

The system provides an Edge-first cyber-physical security architecture for connected vehicles.

The architecture combines:

- Automotive network telemetry
- ECU behavior
- Physical sensors
- Occupant-state information
- ADAS information
- Vehicle dynamics
- Connectivity telemetry
- Behavioral anomaly detection
- Predictive connectivity analysis
- AI-assisted security reasoning
- Deterministic response
- Auxiliary communication
- Forensic evidence preservation
- Cortex XDR enterprise security operations
- NHTSA safety intelligence
- Fleet-level learning

## 2. Architectural Principle

The vehicle must remain capable of detecting and responding to security threats even when external enterprise connectivity is degraded, unavailable or intentionally suppressed.

Cloud connectivity is therefore an enhancement to vehicle security operations, not a prerequisite for local safety.

## 3. Major Layers

### Layer 1 — Vehicle Telemetry

Sources include:

- CAN / CAN-FD
- ECU communications
- Diagnostic traffic
- Occupant sensors
- Restraint state
- IMU
- Vehicle dynamics
- ADAS signals
- Gateway state
- Primary network state
- Auxiliary network state

### Layer 2 — Edge-IoT Gateway

Responsibilities:

- Telemetry collection
- Normalization
- Timestamping
- Feature extraction
- Local buffering
- Secure event generation
- Connectivity monitoring

### Layer 3 — Cyber-Physical Detection

Responsibilities:

- Behavioral modeling
- Temporal analysis
- ECU communication modeling
- Sensor consistency analysis
- Cross-domain anomaly correlation
- Cyber-physical incident generation

### Layer 4 — Predictive Connectivity Resilience

Responsibilities:

- Monitor communication quality
- Estimate probability of connectivity loss
- Estimate failure horizon
- Identify degradation trends
- Assess possible malicious suppression
- Generate early-warning events
- Trigger resilience preparation

### Layer 5 — Multimodal Risk Fusion

Inputs:

- Cyber anomaly risk
- Physical anomaly risk
- Occupant-state risk
- Connectivity risk
- Vehicle-state risk
- Historical safety context

Output:

- Cyber-physical risk state
- Safety impact estimate
- Response priority

### Layer 6 — AI Security Reasoning

Responsibilities:

- Correlate structured evidence
- Generate attack hypotheses
- Explain detection decisions
- Assess potential impact
- Recommend investigation actions
- Recommend approved response playbooks

AI shall not directly control safety-critical actuators.

### Layer 7 — Deterministic Response

Potential actions include:

- ECU isolation
- Communication restriction
- Unauthorized command rejection
- Safety-state transition
- Auxiliary-channel activation
- Safety telemetry prioritization
- Forensic preservation

### Layer 8 — Forensic Evidence

The system records:

- Timestamp
- Vehicle identifier
- ECU
- Network event
- Sensor state
- Occupant state
- Connectivity state
- Model version
- Anomaly score
- AI assessment
- Response policy
- Response action
- Evidence integrity information

### Layer 9 — Enterprise Security Operations

Cortex XDR receives normalized security events and provides:

- Incident visibility
- Correlation
- Investigation
- Alert prioritization
- Security operations workflows

### Layer 10 — Fleet Intelligence

Aggregates privacy-conscious security and safety information across vehicles to identify:

- Repeated attack patterns
- New behavioral deviations
- Connectivity threats
- Fleet-wide anomalies
- Model improvement opportunities

## 4. NHTSA Safety Intelligence

The system will investigate the following NHTSA datasets:

- CISS
- CRSS
- FARS
- SCI
- ADAS
- NiTS

These datasets provide historical safety, crash, occupant, vehicle and injury context.

They shall not be represented as cybersecurity telemetry.

Their role is to support safety-context modeling and potential consequence analysis.

## 5. Communication Architecture

### Primary Channel

Vehicle Edge Gateway
    |
Primary telemetry network
    |
Enterprise infrastructure
    |
Cortex XDR / SOC

### Auxiliary Safety Channel

Safety-critical vehicle information
    |
Independent auxiliary communication path
    |
Safety telemetry service

The auxiliary path shall prioritize critical safety and cybersecurity information rather than attempting to replicate all normal telemetry.

## 6. Predictive Resilience States

NORMAL

    |
    v

DEGRADED

    |
    v

PRE-FAILURE

    |
    v

OFFLINE / COMPROMISED

    |
    v

RECOVERY

The transition logic will be defined separately in the connectivity resilience specification.

## 7. Core Design Principle

Detect -> Correlate -> Predict -> Warn -> Prepare -> Prioritize -> Contain -> Preserve -> Recover -> Learn
