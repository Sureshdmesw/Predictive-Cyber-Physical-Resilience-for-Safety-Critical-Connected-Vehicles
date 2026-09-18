# Phase 9 — CAN/HIL Upgrade

## Objective

Upgrade the existing virtual CAN/ECU laboratory toward a hardware-ready
HIL architecture while preserving the existing validated virtual
laboratory and evidence chain.

## Boundary

This phase does not modify the frozen Transformer checkpoint.

It does not merge the real HCRL cybersecurity dataset with the existing
synthetic cyber-physical telemetry.

The existing virtual CAN/HIL laboratory remains the controlled baseline.

## Planned architecture

Existing:

Synthetic scenario
    ↓
Virtual CAN traffic
    ↓
Virtual ECU
    ↓
Detection / safety logic
    ↓
Evidence artifacts

Upgrade target:

Scenario generator
    ↓
CAN/HIL abstraction layer
    ↓
Virtual CAN / simulated ECU
    ↓
Hardware-interface boundary
    ↓
Future physical CAN interface
    ↓
Controlled ECU/HIL observation
    ↓
Evidence capture

## Initial milestone

Establish a hardware-independent CAN/HIL abstraction boundary.

The same test scenarios should be capable of operating against the
existing virtual laboratory without requiring physical hardware.

## Validation requirements

1. Existing virtual CAN/HIL behavior remains reproducible.
2. Existing evidence artifacts remain unchanged.
3. Hardware-specific functionality is isolated behind an interface.
4. No production-vehicle connection is required.
5. No safety-critical deployment claim is made from laboratory results.

## Future physical-HIL work

Physical CAN interface integration is a separate controlled laboratory
step and must be validated independently before being used for any
safety-related experiment.

## Status

Phase 9 preparation: STARTED
Physical hardware integration: NOT STARTED
Virtual laboratory baseline: PRESERVED
