# Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles

## Project Overview

An Edge-IoT and AI-driven cyber-physical resilience architecture for safety-critical connected vehicles.

The system investigates the ability to:

- Detect cyber-physical attacks
- Correlate ECU, CAN, physical-sensor, occupant and ADAS anomalies
- Predict impending communication degradation or loss
- Provide advance warning to enterprise security operations
- Transition safety-critical telemetry to a resilient auxiliary communication path
- Maintain local detection and containment during connectivity loss
- Preserve tamper-evident forensic evidence
- Integrate with Cortex XDR for enterprise security monitoring
- Incorporate NHTSA safety intelligence
- Improve detection through fleet-level learning

## NHTSA Safety Intelligence

The project will investigate:

- CISS
- CRSS
- FARS
- SCI
- ADAS
- NiTS

These datasets provide historical crash, injury, vehicle, occupant and safety context.

They are not treated as cybersecurity telemetry.

## Core Architecture

Vehicle / Edge IoT
    |
    +-- CAN / ECU
    +-- Occupant sensors
    +-- Vehicle dynamics
    +-- ADAS
    +-- Connectivity telemetry
            |
            v
Cyber-Physical Detection
            |
            v
Predictive Connectivity Resilience
            |
            v
AI Security Reasoning
            |
            v
Deterministic Response
            |
            +-- ECU isolation
            +-- Command rejection
            +-- Safety state
            +-- Auxiliary communication
            |
            v
Forensic Evidence
            |
            v
Cortex XDR / SOC
            |
            v
Fleet Intelligence

## Safety Principle

AI reasoning must not have unrestricted authority over safety-critical actuation.

AI provides analysis, classification, explanation and recommendations.

Deterministic safety policies remain responsible for safety-critical actions.

## Development Status

Stage 0 - Project foundation

[ ] Repository structure
[ ] Threat model
[ ] System requirements
[ ] Data architecture
[ ] Safety architecture
[ ] Predictive connectivity architecture
[ ] Cortex XDR integration architecture
[ ] NHTSA data acquisition plan

## Project Title

Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles
