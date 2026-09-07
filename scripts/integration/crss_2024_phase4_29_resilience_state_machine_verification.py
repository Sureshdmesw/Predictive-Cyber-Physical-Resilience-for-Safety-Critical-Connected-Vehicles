import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


ROOT = Path(__file__).resolve().parents[2]

PHASE_28A = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_28"
    / "crss_2024_phase4_28a_predictive_soc_event_pipeline.json"
)

PHASE_28B = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_28"
    / "crss_2024_phase4_28b_connectivity_recovery_sync.json"
)

PHASE_28C = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_28c"
    / "crss_2024_phase4_28c_adversarial_integrity_tamper_detection.json"
)

OUTPUT = (
    ROOT
    / "experiments"
    / "integration"
    / "phase4_29"
    / "crss_2024_phase4_29_resilience_state_machine_verification.json"
)


class State(str, Enum):
    NORMAL = "NORMAL"
    PREDICTIVE_RISK = "PREDICTIVE_RISK"
    CONNECTIVITY_DEGRADED = "CONNECTIVITY_DEGRADED"
    DISCONNECTED = "DISCONNECTED"
    LOCAL_FORENSIC_BUFFER = "LOCAL_FORENSIC_BUFFER"
    RECOVERY = "RECOVERY"
    INTEGRITY_VERIFICATION = "INTEGRITY_VERIFICATION"
    SYNCHRONIZED = "SYNCHRONIZED"
    SOC_HANDOFF = "SOC_HANDOFF"
    REFUSE_SYNC = "REFUSE_SYNC"
    REFUSE_SOC_HANDOFF = "REFUSE_SOC_HANDOFF"
    UNVERIFIED_EVIDENCE = "UNVERIFIED_EVIDENCE"


@dataclass
class Transition:
    source: str
    event: str
    destination: str
    allowed: bool
    reason: str


class ResilienceStateMachine:
    def __init__(self):
        self.state = State.NORMAL
        self.history = [self.state.value]

    def transition(self, event: str, integrity_valid=True):
        current = self.state

        if current == State.NORMAL and event == "PREDICTIVE_RISK":
            self.state = State.PREDICTIVE_RISK
            reason = "Predictive risk detected before active containment."

        elif current in {
            State.NORMAL,
            State.PREDICTIVE_RISK,
        } and event == "CONNECTIVITY_DEGRADED":
            self.state = State.CONNECTIVITY_DEGRADED
            reason = "Connectivity degradation requires resilience mode."

        elif current in {
            State.NORMAL,
            State.PREDICTIVE_RISK,
            State.CONNECTIVITY_DEGRADED,
        } and event == "DISCONNECT":
            self.state = State.DISCONNECTED
            reason = "Connectivity loss requires offline-first operation."

        elif current == State.DISCONNECTED and event == "BUFFER_FORENSIC":
            self.state = State.LOCAL_FORENSIC_BUFFER
            reason = "Forensic events remain local while disconnected."

        elif current == State.LOCAL_FORENSIC_BUFFER and event == "CONNECTIVITY_RECOVERED":
            self.state = State.RECOVERY
            reason = "Connectivity returned; evidence remains local pending verification."

        elif current == State.RECOVERY and event == "VERIFY_INTEGRITY":
            self.state = State.INTEGRITY_VERIFICATION
            reason = "Recovered evidence requires integrity verification before synchronization."

        elif current == State.INTEGRITY_VERIFICATION and event == "INTEGRITY_VALID":
            if not integrity_valid:
                self.state = State.UNVERIFIED_EVIDENCE
                reason = "Integrity verification failed; trusted synchronization is prohibited."
            else:
                self.state = State.SYNCHRONIZED
                reason = "Integrity verified; synchronized evidence is permitted."

        elif current == State.INTEGRITY_VERIFICATION and event == "INTEGRITY_TAMPERED":
            self.state = State.UNVERIFIED_EVIDENCE
            reason = "Tampering detected; evidence is explicitly unverified."

        elif current == State.SYNCHRONIZED and event == "SOC_HANDOFF":
            self.state = State.SOC_HANDOFF
            reason = "Verified evidence may be handed to the SOC abstraction."

        elif current == State.UNVERIFIED_EVIDENCE and event == "ATTEMPT_SYNC":
            self.state = State.REFUSE_SYNC
            reason = "Synchronization refused because evidence integrity is unverified."

        elif current in {
            State.REFUSE_SYNC,
            State.UNVERIFIED_EVIDENCE,
        } and event == "ATTEMPT_SOC_HANDOFF":
            self.state = State.REFUSE_SOC_HANDOFF
            reason = "SOC handoff refused because trusted evidence was not established."

        else:
            return Transition(
                source=current.value,
                event=event,
                destination=current.value,
                allowed=False,
                reason="Transition is not permitted by the deterministic safety state machine.",
            )

        self.history.append(self.state.value)

        return Transition(
            source=current.value,
            event=event,
            destination=self.state.value,
            allowed=True,
            reason=reason,
        )


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_json(obj):
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_valid_path():
    sm = ResilienceStateMachine()

    events = [
        ("PREDICTIVE_RISK", True),
        ("CONNECTIVITY_DEGRADED", True),
        ("DISCONNECT", True),
        ("BUFFER_FORENSIC", True),
        ("CONNECTIVITY_RECOVERED", True),
        ("VERIFY_INTEGRITY", True),
        ("INTEGRITY_VALID", True),
        ("SOC_HANDOFF", True),
    ]

    transitions = []

    for event, integrity in events:
        result = sm.transition(event, integrity)
        transitions.append(asdict(result))

    return sm, transitions


def run_tampered_path():
    sm = ResilienceStateMachine()

    events = [
        ("PREDICTIVE_RISK", True),
        ("CONNECTIVITY_DEGRADED", True),
        ("DISCONNECT", True),
        ("BUFFER_FORENSIC", True),
        ("CONNECTIVITY_RECOVERED", True),
        ("VERIFY_INTEGRITY", True),
        ("INTEGRITY_TAMPERED", False),
        ("ATTEMPT_SYNC", False),
        ("ATTEMPT_SOC_HANDOFF", False),
    ]

    transitions = []

    for event, integrity in events:
        result = sm.transition(event, integrity)
        transitions.append(asdict(result))

    return sm, transitions


def run_invalid_bypass_tests():
    tests = []

    sm = ResilienceStateMachine()

    result = sm.transition("SOC_HANDOFF")

    tests.append({
        "name": "soc_handoff_from_normal_blocked",
        "passed": not result.allowed,
        "result": asdict(result),
    })

    sm = ResilienceStateMachine()

    result = sm.transition("ATTEMPT_SYNC")

    tests.append({
        "name": "sync_from_normal_blocked",
        "passed": not result.allowed,
        "result": asdict(result),
    })

    sm = ResilienceStateMachine()

    for event in [
        "PREDICTIVE_RISK",
        "CONNECTIVITY_DEGRADED",
        "DISCONNECT",
        "BUFFER_FORENSIC",
        "CONNECTIVITY_RECOVERED",
        "VERIFY_INTEGRITY",
        "INTEGRITY_TAMPERED",
    ]:
        sm.transition(event)

    result = sm.transition("SOC_HANDOFF")

    tests.append({
        "name": "tampered_evidence_soc_bypass_blocked",
        "passed": not result.allowed,
        "result": asdict(result),
    })

    return tests


def main():
    required = [
        PHASE_28A,
        PHASE_28B,
        PHASE_28C,
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required Phase 4.28 artifacts:\n"
            + "\n".join(missing)
        )

    phase28a = load_json(PHASE_28A)
    phase28b = load_json(PHASE_28B)
    phase28c = load_json(PHASE_28C)

    valid_sm, valid_transitions = run_valid_path()
    tampered_sm, tampered_transitions = run_tampered_path()
    bypass_tests = run_invalid_bypass_tests()

    gates = {
        "source_4_28a_present": PHASE_28A.exists(),
        "source_4_28b_present": PHASE_28B.exists(),
        "source_4_28c_present": PHASE_28C.exists(),

        "valid_path_reaches_soc_handoff":
            valid_sm.state == State.SOC_HANDOFF,

        "valid_path_all_transitions_allowed":
            all(t["allowed"] for t in valid_transitions),

        "tampered_path_reaches_unverified":
            "UNVERIFIED_EVIDENCE" in tampered_sm.history,

        "tampered_path_refuses_sync":
            "REFUSE_SYNC" in tampered_sm.history,

        "tampered_path_refuses_soc_handoff":
            "REFUSE_SOC_HANDOFF" in tampered_sm.history,

        "tampered_path_never_reaches_synchronized":
            State.SYNCHRONIZED.value not in tampered_sm.history,

        "tampered_path_never_reaches_soc_handoff":
            State.SOC_HANDOFF.value not in tampered_sm.history,

        "bypass_tests_all_pass":
            all(t["passed"] for t in bypass_tests),

        "direct_vehicle_actuation_disabled": True,

        "scientific_real_attack_claim_disabled": True,

        "live_cortex_xdr_api_claim_disabled": True,
    }

    status = "PASS" if all(gates.values()) else "FAIL"

    report = {
        "phase": "4.29",
        "title": "Resilience State-Machine and Safety-Boundary Verification",
        "status": status,
        "research_simulation": True,

        "state_machine": {
            "states": [s.value for s in State],
            "valid_path": valid_sm.history,
            "tampered_path": tampered_sm.history,
        },

        "valid_path_transitions": valid_transitions,
        "tampered_path_transitions": tampered_transitions,
        "bypass_tests": bypass_tests,

        "safety_boundary": {
            "direct_vehicle_actuation": False,
            "real_world_cyberattack_performance_claim": False,
            "live_cortex_xdr_api_used": False,
        },

        "source_status": {
            "phase_4_28a_status": phase28a.get("status"),
            "phase_4_28b_status": phase28b.get("status"),
            "phase_4_28c_status": phase28c.get("status"),
        },

        "gates": gates,
        "gate_count": len(gates),
        "gates_passed": sum(bool(v) for v in gates.values()),

        "evidence_digest": sha256_json({
            "valid_path": valid_transitions,
            "tampered_path": tampered_transitions,
            "bypass_tests": bypass_tests,
        }),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== PHASE 4.29 RESULT ===")
    print(f"Status: {status}")
    print(
        f"Gates: {report['gates_passed']}/{report['gate_count']}"
    )

    print("\nValid path:")
    print(" -> ".join(valid_sm.history))

    print("\nTampered path:")
    print(" -> ".join(tampered_sm.history))

    print("\nBypass tests:")
    for test in bypass_tests:
        print(
            f"  {'PASS' if test['passed'] else 'FAIL'} "
            f"{test['name']}"
        )

    print(f"\nEvidence digest: {report['evidence_digest']}")
    print(f"Report: {OUTPUT}")


if __name__ == "__main__":
    main()
