# Threat Model

# Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles

## 1. Objective

Identify threats capable of compromising vehicle cybersecurity, safety telemetry, cyber-physical state estimation, enterprise visibility or forensic evidence.

## 2. Protected Assets

### Vehicle Assets

- ECUs
- CAN/CAN-FD communications
- Diagnostic interfaces
- Sensors
- Occupant-state systems
- ADAS systems
- Gateway
- Safety controllers
- Local forensic storage

### Communication Assets

- Primary telemetry channel
- Auxiliary safety channel
- Vehicle-to-cloud communication
- Authentication credentials
- Communication metadata

### Enterprise Assets

- Security events
- Cortex XDR incidents
- Fleet intelligence
- Detection models
- Response policies
- Forensic records

## 3. Threat Classes

### T1 — CAN Injection

Unauthorized messages introduced into vehicle communication.

### T2 — Replay

Previously valid messages replayed at an inappropriate time.

### T3 — ECU Impersonation

An unauthorized device attempts to appear as a trusted ECU.

### T4 — Message Timing Manipulation

Communication timing is altered to produce abnormal or unsafe behavior.

### T5 — Sensor Spoofing

Physical sensor observations are manipulated or fabricated.

### T6 — Cross-Domain Manipulation

Cyber and physical signals are manipulated to create a misleading system state.

### T7 — Diagnostic Abuse

Unauthorized diagnostic commands or diagnostic-state manipulation.

### T8 — Telemetry Suppression

External security telemetry is intentionally delayed, degraded or prevented from reaching the SOC.

### T9 — Gateway Compromise

The Edge gateway is compromised or manipulated.

### T10 — Coordinated Cyber-Physical Attack

Multiple attack techniques are combined against a safety-critical function.

### T11 — Forensic Manipulation

Security evidence is deleted, altered or reordered.

### T12 — Fleet-Level Attack

A common attack pattern affects multiple vehicles or attempts to propagate through fleet infrastructure.

## 4. Connectivity Threat Model

The system must distinguish:

- Normal connectivity loss
- Infrastructure failure
- Temporary degradation
- Hardware failure
- Network congestion
- Malicious suppression
- Deliberate communication interference

## 5. Safety Principle

A cybersecurity response must not introduce a greater safety risk than the detected threat.

Safety-critical response decisions must therefore remain constrained by deterministic policies.

## 6. Trust Boundaries

### TB1

Vehicle sensors -> Edge Gateway

### TB2

Vehicle network -> Enterprise network

### TB3

Primary communication -> Cortex XDR

### TB4

Auxiliary communication -> Safety service

### TB5

Fleet vehicles -> Fleet intelligence

### TB6

AI reasoning -> Deterministic policy engine

## 7. AI Threats

The AI reasoning layer must be evaluated against:

- Incorrect classification
- Hallucinated evidence
- Overconfident conclusions
- Prompt manipulation
- Malicious telemetry content
- Data poisoning
- Model drift
- Adversarial inputs

The AI shall not be trusted as the sole authority for safety-critical decisions.

## 8. Resilience Objective

If primary connectivity is compromised:

1. Predict degradation where possible.
2. Notify the SOC before complete loss.
3. Prepare the auxiliary channel.
4. Prioritize safety-critical telemetry.
5. Maintain local detection.
6. Maintain deterministic containment.
7. Preserve forensic evidence.
8. Synchronize evidence after recovery.
