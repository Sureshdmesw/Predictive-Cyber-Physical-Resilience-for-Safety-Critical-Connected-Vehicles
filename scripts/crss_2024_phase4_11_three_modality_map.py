import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPEC = ROOT / "experiments/modeling/v2/crss_2024_v2_feature_treatment_spec.json"
OUT = ROOT / "experiments/modeling/v2/cross_modal/crss_2024_v2_modality_map.json"

spec = json.loads(SPEC.read_text(encoding="utf-8-sig"))
features = spec["final_features"]

# ------------------------------------------------------------
# CYBER / EDGE TELEMETRY
# ------------------------------------------------------------

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
    "safety_critical_ecu_ratio",
    "sequence_counter_anomaly_rate",
    "telemetry_gap_duration_ms",
    "uds_request_rate",
    "unexpected_source_id_rate",
}

# ------------------------------------------------------------
# PHYSICAL / SAFETY
# ------------------------------------------------------------

physical = {
    "VISION",
    "WEATHER",
    "collision_event_flag",
    "fire_explosion_flag",
    "redundant_sensor_divergence",
    "sensor_disagreement_rate",
    "sensor_plausibility_score",
    "rollover_flag",
    "speed_limit_deviation",
    "speed_limit_deviation_abs",
    "speed_over_limit_flag",
    "vehicle_damage_flag",
}

# ------------------------------------------------------------
# DERIVED / TEMPORAL / RESILIENCE
# ------------------------------------------------------------

derived = {
    "cyber_instability_index",
    "d1_sensor_disagreement_rate",
    "resilience_degradation_rate",
    "rolling_mean_3_latency_ms",
    "rolling_mean_3_message_integrity_failure_rate",
    "rolling_mean_3_packet_loss_rate",
    "rolling_mean_3_sensor_disagreement_rate",
    "rolling_std_3_latency_ms",
    "rolling_std_3_message_integrity_failure_rate",
    "rolling_std_3_packet_loss_rate",
    "rolling_std_3_sensor_disagreement_rate",
    "rssi_dbm",
    "state_transition_magnitude",
}

feature_set = set(features)

cyber = cyber & feature_set
physical = physical & feature_set
derived = derived & feature_set

overlap = (cyber & physical) | (cyber & derived) | (physical & derived)
mapped = cyber | physical | derived
unmapped = feature_set - mapped

print("=" * 72)
print("PHASE 4.11 — THREE-MODALITY FEATURE MAP")
print("=" * 72)

print("Final features :", len(features))
print("Cyber          :", len(cyber))
print("Physical       :", len(physical))
print("Derived        :", len(derived))
print("Mapped         :", len(mapped))
print("Unmapped       :", len(unmapped))
print("Overlap        :", len(overlap))

if overlap:
    print("\nOVERLAPPING FEATURES:")
    for x in sorted(overlap):
        print("  -", x)

if unmapped:
    print("\nUNMAPPED FEATURES:")
    for x in sorted(unmapped):
        print("  -", x)

if overlap:
    raise RuntimeError("A feature belongs to multiple modalities")

if unmapped:
    raise RuntimeError("Every final predictor must have an explicit modality")

if len(mapped) != len(features):
    raise RuntimeError("Mapped feature count does not equal final feature count")

ordered = []

for index, feature in enumerate(features):
    if feature in cyber:
        modality = "cyber"
    elif feature in physical:
        modality = "physical"
    elif feature in derived:
        modality = "derived_temporal_resilience"
    else:
        raise RuntimeError(f"Unmapped feature: {feature}")

    ordered.append({
        "feature": feature,
        "original_index": index,
        "modality": modality
    })

result = {
    "phase": "4.11",
    "status": "PASS",
    "total_features": len(features),
    "cyber_feature_count": len(cyber),
    "physical_feature_count": len(physical),
    "derived_temporal_resilience_feature_count": len(derived),
    "cyber_features": sorted(cyber),
    "physical_features": sorted(physical),
    "derived_temporal_resilience_features": sorted(derived),
    "ordered_feature_map": ordered,
    "methodological_note": (
        "Features are explicitly separated into cyber/edge telemetry, "
        "physical/safety signals, and derived temporal/resilience signals. "
        "This avoids conflating binary/continuous treatment with modality "
        "and avoids forcing downstream derived features into a raw modality."
    )
}

OUT.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

print()
print("=" * 72)
print("THREE-MODALITY MAP PASS")
print("=" * 72)
print("Cyber          :", len(cyber))
print("Physical       :", len(physical))
print("Derived        :", len(derived))
print("Total          :", len(mapped))
print("Output         :", OUT)
