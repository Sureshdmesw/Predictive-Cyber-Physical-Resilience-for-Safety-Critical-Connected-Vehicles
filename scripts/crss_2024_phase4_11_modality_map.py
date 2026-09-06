import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPEC = ROOT / "experiments/modeling/v2/crss_2024_v2_feature_treatment_spec.json"
OUT = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_modality_map.json"

spec = json.loads(SPEC.read_text(encoding="utf-8-sig"))

features = spec["final_features"]

# Authoritative cyber modality.
cyber = {
    "authentication_failure_rate",
    "can_bus_count",
    "can_bus_load_pct",
    "can_interarrival_jitter_ms",
    "can_interarrival_mean_ms",
    "can_message_rate",
    "can_message_rate_deviation",
    "connectivity_degradation_rate",
    "connectivity_drop_rate",
    "d1_authentication_failure_rate",
    "d1_can_bus_load_pct",
    "d1_can_interarrival_jitter_ms",
    "d1_can_message_rate",
    "d1_connectivity_drop_rate",
    "d1_latency_jitter_ms",
    "d1_latency_ms",
    "d1_message_integrity_failure_rate",
    "d1_packet_loss_rate",
    "d1_replay_indicator_rate",
    "d1_sequence_counter_anomaly_rate",
    "d1_telemetry_gap_duration_ms",
    "diagnostic_event_rate",
    "ecu_count",
    "ecu_state_transition_rate",
    "latency_jitter_ms",
    "latency_ms",
    "message_integrity_failure_rate",
    "packet_loss_rate",
    "replay_indicator_rate",
    "rolling_mean_3_latency_ms",
    "rolling_mean_3_message_integrity_failure_rate",
    "rolling_mean_3_packet_loss_rate",
    "rolling_std_3_latency_ms",
    "rolling_std_3_message_integrity_failure_rate",
    "rolling_std_3_packet_loss_rate",
    "safety_critical_ecu_ratio",
    "sequence_counter_anomaly_rate",
    "telemetry_gap_duration_ms",
    "uds_request_rate",
    "unexpected_source_id_rate",
}

# Physical vehicle / environment signals.
# These are intentionally separated from cyber-network telemetry.
physical = {
    "VISION",
    "WEATHER",
    "collision_event_flag",
    "fire_explosion_flag",
    "redundant_sensor_divergence",
    "rolling_mean_3_sensor_disagreement_rate",
    "rolling_std_3_sensor_disagreement_rate",
    "rollover_flag",
    "rssi_dbm",
    "resilience_degradation_rate",
    "sensor_disagreement_rate",
    "sensor_plausibility_score",
    "speed_limit_deviation",
    "speed_limit_deviation_abs",
    "speed_over_limit_flag",
    "state_transition_magnitude",
    "vehicle_damage_flag",
}

# Check for missing features.
mapped = cyber | physical
missing = [f for f in features if f not in mapped]
extra_cyber = sorted(cyber - set(features))
extra_physical = sorted(physical - set(features))

print("=" * 72)
print("PHASE 4.11 — AUTHORITATIVE MODALITY MAP")
print("=" * 72)

print("Final features :", len(features))
print("Cyber mapped   :", len(cyber & set(features)))
print("Physical mapped:", len(physical & set(features)))
print("Mapped total   :", len(mapped & set(features)))

if missing:
    print("\nUNMAPPED FEATURES:")
    for f in missing:
        print("  -", f)

if extra_cyber:
    print("\nCYBER FEATURES NOT IN FINAL SPEC:")
    for f in extra_cyber:
        print("  -", f)

if extra_physical:
    print("\nPHYSICAL FEATURES NOT IN FINAL SPEC:")
    for f in extra_physical:
        print("  -", f)

if len(features) != 59:
    raise RuntimeError(f"Expected 59 final features, got {len(features)}")

if len(cyber & set(features)) != 27:
    raise RuntimeError(
        f"Expected exactly 27 cyber features, got {len(cyber & set(features))}"
    )

if len(physical & set(features)) != 32:
    raise RuntimeError(
        f"Expected exactly 32 physical features, got {len(physical & set(features))}"
    )

if missing:
    raise RuntimeError(f"{len(missing)} features are not assigned to a modality")

if len((cyber & set(features)) & (physical & set(features))):
    raise RuntimeError("Feature assigned to both modalities")

ordered_modalities = []

for feature in features:
    if feature in cyber:
        modality = "cyber"
    elif feature in physical:
        modality = "physical"
    else:
        raise RuntimeError(f"Unmapped feature: {feature}")

    ordered_modalities.append({
        "feature": feature,
        "original_index": features.index(feature),
        "modality": modality
    })

result = {
    "phase": "4.11",
    "status": "PASS",
    "total_features": len(features),
    "cyber_feature_count": len(cyber & set(features)),
    "physical_feature_count": len(physical & set(features)),
    "cyber_features": [
        x["feature"] for x in ordered_modalities
        if x["modality"] == "cyber"
    ],
    "physical_features": [
        x["feature"] for x in ordered_modalities
        if x["modality"] == "physical"
    ],
    "ordered_feature_map": ordered_modalities,
    "methodological_note": (
        "Cyber modality represents ECU/CAN/connectivity/communication telemetry. "
        "Physical modality represents vehicle dynamics, sensor consistency, "
        "environment, safety state, and derived physical resilience signals. "
        "Cyber labels remain synthetic research telemetry; CRSS provides "
        "physical crash/safety ground truth."
    )
}

OUT.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 72)
print("MODALITY MAP PASS")
print("=" * 72)
print("Cyber   :", result["cyber_feature_count"])
print("Physical:", result["physical_feature_count"])
print("Total   :", result["total_features"])
print("Output  :", OUT)
