# Safety Architecture

# Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles

## 1. Safety Objective

The system shall improve cyber-physical resilience without allowing cybersecurity automation to introduce unsafe vehicle behavior.

The architecture therefore separates:

- Detection
- Prediction
- AI reasoning
- Response recommendation
- Deterministic safety policy
- Safety-critical actuation

AI is an intelligence component, not the ultimate authority for safety-critical actuation.

---

## 2. Safety-Critical Design Principle

The fundamental control boundary is:

Vehicle telemetry
        |
        v
Detection
        |
        v
Risk assessment
        |
        v
AI reasoning
        |
        v
Recommended response
        |
        v
Deterministic safety policy
        |
        v
Approved action
        |
        v
Safety controller

The AI reasoning layer cannot directly command safety-critical actuators.

---

## 3. Vehicle Safety Domains

### Occupant Protection

Relevant signals include:

- Occupant position
- Seat position
- Restraint state
- Seatbelt state
- Airbag/occupant protection state
- Crash-state information

### Vehicle Dynamics

Relevant signals include:

- Speed
- Acceleration
- Deceleration
- Yaw
- Vehicle motion
- Stability-related information

### ADAS

Relevant information may include:

- ADAS operating state
- Sensor-derived environment state
- Driver-assistance state
- Safety warnings
- System availability

### Automotive Network

Relevant information includes:

- ECU communication
- CAN/CAN-FD traffic
- Message timing
- Message frequency
- Diagnostic activity
- Gateway state

---

## 4. Cyber-Physical Safety Correlation

The system shall not rely solely on cybersecurity indicators.

Example:

Normal CAN timing
        +
Normal occupant position
        +
Normal vehicle dynamics

        =

Low cyber-physical concern


Example:

Abnormal CAN timing
        +
Unexpected ECU command
        +
Occupant-position inconsistency
        +
Abnormal vehicle state

        =

Elevated cyber-physical risk

The purpose is to identify relationships between cyber events and physical consequences.

---

## 5. Risk Model

The architecture separates at least three dimensions:

Cyber Risk
    |
    +-- likelihood of malicious activity

Physical Risk
    |
    +-- deviation from expected physical behavior

Safety Risk
    |
    +-- potential consequence to occupants or vehicle safety

A combined cyber-physical safety state can be calculated from these dimensions.

The exact mathematical formulation will be evaluated experimentally.

---

## 6. Safety Risk States

### S0 — NORMAL

No significant cyber-physical anomaly.

Actions:

- Continue normal operation
- Continue monitoring

### S1 — OBSERVE

Potential anomaly detected.

Actions:

- Increase monitoring
- Collect additional evidence
- Continue normal safety operation

### S2 — CAUTION

Correlated cyber-physical anomaly detected.

Actions:

- Increase evidence collection
- Restrict nonessential suspicious communication where safe
- Notify security operations
- Evaluate safety impact

### S3 — PROTECT

High-confidence threat with potential safety consequences.

Actions:

- Apply approved containment policy
- Protect safety-critical communication
- Activate resilience mechanisms
- Preserve forensic evidence
- Notify SOC

### S4 — FAIL-SAFE

Unsafe or compromised operating condition.

Actions:

- Execute predetermined safety policy
- Preserve safety-critical functionality
- Prevent unauthorized commands
- Maintain local operation
- Use auxiliary communication where applicable

AI does not directly select arbitrary actuator commands.

---

## 7. Predictive Safety

The architecture shall attempt to identify threats before physical consequences occur.

Conceptual flow:

Cyber anomaly
      |
      v
Cyber-physical correlation
      |
      v
Potential safety consequence
      |
      v
Predictive risk assessment
      |
      v
Early warning
      |
      v
Deterministic safety policy

The objective is to maximize useful warning time while controlling false alarms.

---

## 8. Connectivity Loss and Safety

Loss of Internet or enterprise connectivity must not disable local safety mechanisms.

If connectivity degrades:

1. Detect degradation.
2. Estimate probability of complete loss.
3. Estimate time-to-loss.
4. Notify SOC.
5. Prepare auxiliary communication.
6. Prioritize safety-critical telemetry.
7. Continue local detection.
8. Continue deterministic containment.
9. Buffer forensic evidence.

---

## 9. Auxiliary Communication

The auxiliary communication path is intended to preserve critical information when the primary telemetry path becomes unavailable.

It should prioritize:

### Priority 0

- Crash state
- Occupant state
- Restraint state
- Critical ECU state
- Critical safety events
- High-severity cyber-physical incidents

### Priority 1

- Security alerts
- Attack indicators
- Containment events
- Forensic metadata

Noncritical bulk telemetry shall not consume capacity needed for safety-critical events.

---

## 10. Predictive Transition

Example:

NORMAL
   |
   | connectivity degradation detected
   v
DEGRADED
   |
   | predicted loss probability increasing
   v
PRE-FAILURE
   |
   | primary channel unavailable
   v
OFFLINE
   |
   | primary channel verified healthy
   v
RECOVERY
   |
   v
NORMAL

The transition thresholds will be experimentally determined.

---

## 11. AI Safety Boundary

The AI reasoning layer may:

- Analyze structured evidence
- Correlate events
- Classify probable attack types
- Explain anomalies
- Estimate contextual risk
- Recommend approved playbooks
- Produce SOC investigation guidance

The AI reasoning layer shall not:

- Directly deploy airbags
- Directly control brakes
- Directly control steering
- Directly modify safety-critical ECU firmware
- Override deterministic safety policies
- Disable safety mechanisms based solely on an AI-generated conclusion

---

## 12. Deterministic Response

Safety-critical actions must be implemented through deterministic policies.

Example:

IF

    cyber_risk >= threshold
AND
    safety_risk >= threshold
AND
    evidence_quality >= minimum

THEN

    execute predefined safety containment policy.

The exact policies will be defined and tested experimentally.

---

## 13. Forensic Safety Record

Every safety-critical security decision should generate a structured record containing:

- Event timestamp
- Vehicle identifier
- ECU identifier
- Relevant telemetry
- Anomaly score
- Cyber-risk estimate
- Safety-risk estimate
- Connectivity state
- AI model/version
- AI reasoning reference
- Deterministic policy identifier
- Action executed
- Evidence integrity information

---

## 14. Recovery

When normal connectivity returns:

Primary channel restored
        |
        v
Integrity verification
        |
        v
Forensic buffer verification
        |
        v
Secure synchronization
        |
        v
Cortex XDR
        |
        v
SOC investigation
        |
        v
Incident reconstruction

The system shall not assume that restored connectivity automatically means restored trust.

---

## 15. Safety Evaluation

The project will evaluate:

- Detection accuracy
- False-positive rate
- False-negative rate
- Prediction lead time
- Containment latency
- Safety telemetry survival
- Auxiliary-channel effectiveness
- Forensic completeness
- Recovery integrity
- Edge resource consumption

---

## 16. Core Safety Principle

The system follows:

Detect
  ->
Correlate
  ->
Predict
  ->
Warn
  ->
Prepare
  ->
Prioritize
  ->
Contain
  ->
Preserve
  ->
Recover
  ->
Learn

while maintaining:

AI intelligence

separate from

deterministic safety authority.
