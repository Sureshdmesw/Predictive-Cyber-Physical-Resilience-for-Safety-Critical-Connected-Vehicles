import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ================================================================
# PATHS
# ================================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_10"
    / "phase5_10_cyber_physical_attack_scenarios.csv"
)

BASELINE_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_9"
    / "phase5_9_controlled_can_anomalies.csv"
)

CHECKPOINT = (
    ROOT
    / "models"
    / "v2"
    / "advanced"
    / "crss_2024_v2_temporal_transformer_controlled_best.pt"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_11"
)

EXP_DIR = (
    ROOT
    / "experiments"
    / "can_hil"
    / "phase5_11"
)

OUT_CSV = OUT_DIR / "phase5_11_predictive_detection.csv"
OUT_JSONL = OUT_DIR / "phase5_11_predictive_detection.jsonl"
OUT_SCHEMA = OUT_DIR / "phase5_11_predictive_detection_schema.json"
OUT_MANIFEST = EXP_DIR / "phase5_11_predictive_detection_manifest.json"


# ================================================================
# MODEL CONTRACT
# ================================================================

BINARY_FEATURES = [
    "authentication_failure_rate",
    "collision_event_flag",
    "connectivity_degradation_rate",
    "connectivity_drop_rate",
    "cyber_instability_index",
    "fire_explosion_flag",
    "message_integrity_failure_rate",
    "packet_loss_rate",
    "redundant_sensor_divergence",
    "replay_indicator_rate",
    "resilience_degradation_rate",
    "rolling_mean_3_message_integrity_failure_rate",
    "rolling_mean_3_packet_loss_rate",
    "rolling_mean_3_sensor_disagreement_rate",
    "rolling_std_3_message_integrity_failure_rate",
    "rolling_std_3_packet_loss_rate",
    "rolling_std_3_sensor_disagreement_rate",
    "rollover_flag",
    "safety_critical_ecu_ratio",
    "sensor_disagreement_rate",
    "sensor_plausibility_score",
    "sequence_counter_anomaly_rate",
    "speed_over_limit_flag",
    "state_transition_magnitude",
    "uds_request_rate",
    "unexpected_source_id_rate",
    "vehicle_damage_flag",
]

CONTINUOUS_FEATURES = [
    "VISION",
    "WEATHER",
    "can_bus_count",
    "can_bus_load_pct",
    "can_interarrival_jitter_ms",
    "can_interarrival_mean_ms",
    "can_message_rate",
    "can_message_rate_deviation",
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
    "d1_sensor_disagreement_rate",
    "d1_sequence_counter_anomaly_rate",
    "d1_telemetry_gap_duration_ms",
    "diagnostic_event_rate",
    "ecu_count",
    "ecu_state_transition_rate",
    "latency_jitter_ms",
    "latency_ms",
    "rolling_mean_3_latency_ms",
    "rolling_std_3_latency_ms",
    "rssi_dbm",
    "speed_limit_deviation",
    "speed_limit_deviation_abs",
    "telemetry_gap_duration_ms",
]

FEATURES = BINARY_FEATURES + CONTINUOUS_FEATURES

FEATURE_COUNT = 59
OBSERVATION_STEPS = 12

Y3_HORIZON = 15
Y6_HORIZON = 30

THRESHOLD = 0.50

EXPECTED_RECORDS_PER_TIMESTEP = 15
EXPECTED_TIMESTEPS = 600
EXPECTED_PREDICTIONS = 589

ATTACK_SCENARIOS = [
    "C01_SPEED_SENSOR_SPOOFING",
    "C02_BRAKE_ACCELERATOR_ATTACK",
    "C03_TORQUE_DYNAMICS_ATTACK",
    "C04_HV_POWER_ATTACK",
    "C05_BATTERY_THERMAL_ATTACK",
    "C06_HV_CONTACTOR_ATTACK",
    "C07_HEARTBEAT_INTEGRITY_ATTACK",
    "C08_SEQUENCE_GATEWAY_ATTACK",
    "C09_RESTRAINT_DOMAIN_ATTACK",
    "C10_MULTI_DOMAIN_ATTACK",
]


# ================================================================
# TRANSFORMER
# ================================================================

class TemporalTransformer(nn.Module):

    def __init__(
        self,
        input_dim=59,
        embedding_dim=128,
        num_layers=3,
        num_heads=8,
        ff_dim=256,
        dropout=0.1,
        max_len=12,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            embedding_dim,
        )

        self.position = nn.Parameter(
            torch.zeros(
                1,
                max_len,
                embedding_dim,
            )
        )

        self.input_norm = nn.LayerNorm(
            embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.attention_pool = nn.Linear(
            embedding_dim,
            1,
        )

        self.pool_norm = nn.LayerNorm(
            embedding_dim
        )

        self.output_norm = nn.LayerNorm(
            embedding_dim
        )

        self.y3_head = nn.Linear(
            embedding_dim,
            1,
        )

        self.y6_head = nn.Linear(
            embedding_dim,
            1,
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.input_norm(x)

        x = x + self.position[
            :,
            :x.size(1),
            :,
        ]

        x = self.encoder(x)

        scores = self.attention_pool(x)

        weights = torch.softmax(
            scores,
            dim=1,
        )

        pooled = torch.sum(
            weights * x,
            dim=1,
        )

        pooled = self.pool_norm(
            pooled
        )

        pooled = self.output_norm(
            pooled
        )

        y3 = self.y3_head(
            pooled
        ).squeeze(-1)

        y6 = self.y6_head(
            pooled
        ).squeeze(-1)

        return y3, y6


# ================================================================
# HELPERS
# ================================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except Exception:

        return default


def signal_value(signals, name, default=0.0):

    if name not in signals:
        return default

    return safe_float(
        signals.get(name),
        default,
    )


def parse_signals(value):

    if isinstance(value, dict):
        return value

    if value is None:
        return {}

    try:

        parsed = json.loads(
            str(value)
        )

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    return {}


def active_value(value):

    text = str(value).strip().lower()

    return text in {
        "1",
        "true",
        "yes",
        "y",
    }


def probability_to_level(probability):

    if probability < 0.05:
        return "NORMAL"

    if probability < 0.20:
        return "LOW"

    if probability < 0.50:
        return "MEDIUM"

    if probability < 0.80:
        return "HIGH"

    return "CRITICAL"


# ================================================================
# CAN RECORDS -> TIMESTEPS
# ================================================================

def build_timestep_records(scenario_df):

    data = scenario_df.reset_index(
        drop=True
    ).copy()

    if len(data) != 9000:

        raise ValueError(
            "Expected 9000 CAN records for one "
            f"scenario, received {len(data)}."
        )

    data["_record_order"] = np.arange(
        len(data)
    )

    data["_source_step"] = (
        data["_record_order"]
        // EXPECTED_RECORDS_PER_TIMESTEP
    )

    timestep_records = []

    for source_step, group in data.groupby(
        "_source_step",
        sort=True,
    ):

        if len(group) != EXPECTED_RECORDS_PER_TIMESTEP:

            raise ValueError(
                f"Source timestep {source_step} "
                f"contains {len(group)} CAN records; "
                f"expected {EXPECTED_RECORDS_PER_TIMESTEP}."
            )

        merged = {}

        timestamps = []

        active_flags = []

        for _, record in group.iterrows():

            parsed = parse_signals(
                record.get(
                    "signals_json",
                    "{}",
                )
            )

            for name, value in parsed.items():

                if name not in merged:

                    merged[name] = safe_float(
                        value
                    )

            timestamps.append(
                safe_float(
                    record.get(
                        "timestamp_s",
                        0.0,
                    )
                )
            )

            active_flags.append(
                active_value(
                    record.get(
                        "scenario_active",
                        0,
                    )
                )
            )

        timestamp_spread = (
            max(timestamps)
            - min(timestamps)
        )

        nominal_timestamp = (
            float(source_step)
            * 0.05
        )

        timestep_records.append(
            {
                "source_step": int(source_step),
                "timestamp_s": nominal_timestamp,
                "timestamp_spread_s": max(
                    0.0,
                    timestamp_spread,
                ),
                "signals": merged,
                "scenario_active": bool(
                    any(active_flags)
                ),
            }
        )

    if len(timestep_records) != EXPECTED_TIMESTEPS:

        raise ValueError(
            "Expected 600 source timesteps; "
            f"received {len(timestep_records)}."
        )

    return timestep_records


# ================================================================
# 59-FEATURE COMPATIBILITY ADAPTER
# ================================================================

def derive_features(
    signals,
    timestamp_spread_s=0.0,
):

    integrity = signal_value(
        signals,
        "IntegrityStatus",
        1.0,
    )

    gateway_integrity = signal_value(
        signals,
        "GatewayIntegrity",
        1.0,
    )

    sequence = signal_value(
        signals,
        "SequenceCounter",
        0.0,
    )

    restraint_sequence = signal_value(
        signals,
        "RestraintSequenceCounter",
        0.0,
    )

    heartbeat_values = [
        signal_value(
            signals,
            "VCUHeartbeat",
            1.0,
        ),
        signal_value(
            signals,
            "BMSHeartbeat",
            1.0,
        ),
        signal_value(
            signals,
            "InverterHeartbeat",
            1.0,
        ),
        signal_value(
            signals,
            "GatewayHeartbeat",
            1.0,
        ),
        signal_value(
            signals,
            "RCMHeartbeat",
            1.0,
        ),
        signal_value(
            signals,
            "OccupantSensorHeartbeat",
            1.0,
        ),
        signal_value(
            signals,
            "RestraintGatewayHeartbeat",
            1.0,
        ),
    ]

    heartbeat_failure = float(
        any(
            value < 0.5
            for value in heartbeat_values
        )
    )

    speed = signal_value(
        signals,
        "VehicleSpeed",
        0.0,
    )

    wheel_fl = signal_value(
        signals,
        "WheelSpeedFL",
        speed,
    )

    wheel_fr = signal_value(
        signals,
        "WheelSpeedFR",
        speed,
    )

    wheel_error = max(
        abs(wheel_fl - speed),
        abs(wheel_fr - speed),
    )

    sensor_disagreement = float(
        wheel_error > 2.0
    )

    torque_command = signal_value(
        signals,
        "TorqueCommand",
        0.0,
    )

    torque_feedback = signal_value(
        signals,
        "TorqueFeedback",
        torque_command,
    )

    diagnostic = signal_value(
        signals,
        "DiagnosticState",
        0.0,
    )

    restraint_diagnostic = signal_value(
        signals,
        "RestraintDiagnostic",
        0.0,
    )

    crash_event = signal_value(
        signals,
        "CrashEventStatus",
        0.0,
    )

    rollover = signal_value(
        signals,
        "RolloverStatus",
        0.0,
    )

    torque_error = abs(
        torque_feedback - torque_command
    )

    interval = 0.05

    jitter_ms = (
        max(
            0.0,
            timestamp_spread_s,
        )
        * 1000.0
    )

    rate = 1.0 / interval

    out = {}

    # ------------------------------------------------------------
    # 27 BINARY FEATURES
    # ------------------------------------------------------------

    out["authentication_failure_rate"] = 0.0

    out["collision_event_flag"] = float(
        crash_event > 0
    )

    out["connectivity_degradation_rate"] = (
        heartbeat_failure
    )

    out["connectivity_drop_rate"] = (
        heartbeat_failure
    )

    out["cyber_instability_index"] = (
        (
            (1.0 - integrity)
            + (1.0 - gateway_integrity)
            + heartbeat_failure
        )
        / 3.0
    )

    out["fire_explosion_flag"] = 0.0

    out["message_integrity_failure_rate"] = max(
        0.0,
        min(
            1.0,
            1.0 - integrity,
        ),
    )

    out["packet_loss_rate"] = (
        heartbeat_failure
    )

    out["redundant_sensor_divergence"] = (
        sensor_disagreement
    )

    out["replay_indicator_rate"] = 0.0

    out["resilience_degradation_rate"] = (
        (
            (1.0 - integrity)
            + (1.0 - gateway_integrity)
            + sensor_disagreement
        )
        / 3.0
    )

    out[
        "rolling_mean_3_message_integrity_failure_rate"
    ] = out[
        "message_integrity_failure_rate"
    ]

    out[
        "rolling_mean_3_packet_loss_rate"
    ] = out[
        "packet_loss_rate"
    ]

    out[
        "rolling_mean_3_sensor_disagreement_rate"
    ] = sensor_disagreement

    out[
        "rolling_std_3_message_integrity_failure_rate"
    ] = 0.0

    out[
        "rolling_std_3_packet_loss_rate"
    ] = 0.0

    out[
        "rolling_std_3_sensor_disagreement_rate"
    ] = 0.0

    out["rollover_flag"] = float(
        rollover > 0
    )

    out["safety_critical_ecu_ratio"] = (
        sum(
            value >= 0.5
            for value in heartbeat_values
        )
        / len(heartbeat_values)
    )

    out["sensor_disagreement_rate"] = (
        sensor_disagreement
    )

    out["sensor_plausibility_score"] = (
        1.0 - sensor_disagreement
    )

    out["sequence_counter_anomaly_rate"] = float(
        sequence < 0
        or restraint_sequence < 0
    )

    out["speed_over_limit_flag"] = float(
        speed > 130.0
    )

    out["state_transition_magnitude"] = (
        abs(diagnostic)
        + abs(restraint_diagnostic)
        + min(
            torque_error / 1000.0,
            1.0,
        )
    )

    out["uds_request_rate"] = 0.0

    out["unexpected_source_id_rate"] = float(
        gateway_integrity < 0.5
    )

    out["vehicle_damage_flag"] = float(
        crash_event > 0
        or rollover > 0
    )

    # ------------------------------------------------------------
    # 32 CONTINUOUS FEATURES
    # ------------------------------------------------------------

    out["VISION"] = 0.0
    out["WEATHER"] = 0.0

    out["can_bus_count"] = 1.0

    out["can_bus_load_pct"] = 0.0

    out["can_interarrival_jitter_ms"] = (
        jitter_ms
    )

    out["can_interarrival_mean_ms"] = (
        interval * 1000.0
    )

    out["can_message_rate"] = rate

    out["can_message_rate_deviation"] = abs(
        rate - 20.0
    )

    out["d1_authentication_failure_rate"] = 0.0

    out["d1_can_bus_load_pct"] = (
        out["can_bus_load_pct"]
    )

    out["d1_can_interarrival_jitter_ms"] = (
        out["can_interarrival_jitter_ms"]
    )

    out["d1_can_message_rate"] = rate

    out["d1_connectivity_drop_rate"] = (
        out["connectivity_drop_rate"]
    )

    out["d1_latency_jitter_ms"] = (
        out["can_interarrival_jitter_ms"]
    )

    out["d1_latency_ms"] = (
        out["can_interarrival_mean_ms"]
    )

    out["d1_message_integrity_failure_rate"] = (
        out["message_integrity_failure_rate"]
    )

    out["d1_packet_loss_rate"] = (
        out["packet_loss_rate"]
    )

    out["d1_replay_indicator_rate"] = 0.0

    out["d1_sensor_disagreement_rate"] = (
        out["sensor_disagreement_rate"]
    )

    out["d1_sequence_counter_anomaly_rate"] = (
        out["sequence_counter_anomaly_rate"]
    )

    out["d1_telemetry_gap_duration_ms"] = 0.0

    out["diagnostic_event_rate"] = float(
        diagnostic > 0
    )

    out["ecu_count"] = float(
        sum(
            value >= 0.5
            for value in heartbeat_values
        )
    )

    out["ecu_state_transition_rate"] = abs(
        diagnostic
    )

    out["latency_jitter_ms"] = (
        out["can_interarrival_jitter_ms"]
    )

    out["latency_ms"] = (
        out["can_interarrival_mean_ms"]
    )

    out["rolling_mean_3_latency_ms"] = (
        out["latency_ms"]
    )

    out["rolling_std_3_latency_ms"] = 0.0

    out["rssi_dbm"] = 0.0

    out["speed_limit_deviation"] = (
        speed - 130.0
    )

    out["speed_limit_deviation_abs"] = abs(
        out["speed_limit_deviation"]
    )

    out["telemetry_gap_duration_ms"] = 0.0

    return out


# ================================================================
# BUILD MODEL INPUT
# ================================================================

def build_feature_matrix(
    timestep_records,
):

    rows = []

    for record in timestep_records:

        features = derive_features(
            record["signals"],
            record["timestamp_spread_s"],
        )

        rows.append(
            [
                safe_float(
                    features.get(
                        name,
                        0.0,
                    )
                )
                for name in FEATURES
            ]
        )

    matrix = np.asarray(
        rows,
        dtype=np.float32,
    )

    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if matrix.shape != (
        EXPECTED_TIMESTEPS,
        FEATURE_COUNT,
    ):

        raise ValueError(
            "Unexpected feature matrix shape: "
            f"{matrix.shape}; expected "
            f"({EXPECTED_TIMESTEPS}, {FEATURE_COUNT})."
        )

    for index in range(
        len(BINARY_FEATURES)
    ):

        matrix[:, index] = np.clip(
            matrix[:, index],
            0.0,
            1.0,
        )

    return matrix


# ================================================================
# MODEL LOADING
# ================================================================

def load_model():

    model = TemporalTransformer(
        input_dim=FEATURE_COUNT,
        embedding_dim=128,
        num_layers=3,
        num_heads=8,
        ff_dim=256,
        dropout=0.1,
        max_len=OBSERVATION_STEPS,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model


# ================================================================
# RUN ONE STREAM
# ================================================================

def evaluate_stream(
    model,
    scenario_id,
    scenario_type,
    scenario_df,
):

    timestep_records = (
        build_timestep_records(
            scenario_df
        )
    )

    feature_matrix = (
        build_feature_matrix(
            timestep_records
        )
    )

    sequences = []
    metadata = []

    for end_index in range(
        OBSERVATION_STEPS - 1,
        EXPECTED_TIMESTEPS,
    ):

        start_index = (
            end_index
            - OBSERVATION_STEPS
            + 1
        )

        sequences.append(
            feature_matrix[
                start_index:end_index + 1
            ]
        )

        metadata.append(
            timestep_records[
                end_index
            ]
        )

    X = np.asarray(
        sequences,
        dtype=np.float32,
    )

    if X.shape != (
        EXPECTED_PREDICTIONS,
        OBSERVATION_STEPS,
        FEATURE_COUNT,
    ):

        raise ValueError(
            "Unexpected sequence shape: "
            f"{X.shape}; expected "
            f"({EXPECTED_PREDICTIONS}, "
            f"{OBSERVATION_STEPS}, "
            f"{FEATURE_COUNT})."
        )

    tensor = torch.from_numpy(X)

    with torch.no_grad():

        y3_logits, y6_logits = model(
            tensor
        )

        y3_probabilities = (
            torch.sigmoid(
                y3_logits
            )
            .cpu()
            .numpy()
        )

        y6_probabilities = (
            torch.sigmoid(
                y6_logits
            )
            .cpu()
            .numpy()
        )

    active_steps = [
        record["source_step"]
        for record in timestep_records
        if record["scenario_active"]
    ]

    if scenario_type == "ATTACK":

        if len(active_steps) != 20:

            raise ValueError(
                f"{scenario_id}: expected "
                "20 active attack timesteps; "
                f"received {len(active_steps)}."
            )

        attack_onset_step = min(
            active_steps
        )

        attack_end_step = max(
            active_steps
        )

        attack_onset_s = (
            attack_onset_step * 0.05
        )

        attack_end_s = (
            attack_end_step * 0.05
        )

    else:

        if len(active_steps) != 0:

            raise ValueError(
                "Baseline contains active "
                "attack timesteps."
            )

        attack_onset_step = None
        attack_end_step = None
        attack_onset_s = None
        attack_end_s = None

    first_y3 = None
    first_y6 = None

    first_pre_y3 = None
    first_pre_y6 = None

    first_post_y3 = None
    first_post_y6 = None

    for index, record in enumerate(
        metadata
    ):

        ts = record["timestamp_s"]

        y3 = y3_probabilities[index]
        y6 = y6_probabilities[index]

        if (
            first_y3 is None
            and y3 >= THRESHOLD
        ):
            first_y3 = ts

        if (
            first_y6 is None
            and y6 >= THRESHOLD
        ):
            first_y6 = ts

        if scenario_type == "ATTACK":

            if (
                first_pre_y3 is None
                and ts < attack_onset_s
                and y3 >= THRESHOLD
            ):
                first_pre_y3 = ts

            if (
                first_pre_y6 is None
                and ts < attack_onset_s
                and y6 >= THRESHOLD
            ):
                first_pre_y6 = ts

            if (
                first_post_y3 is None
                and ts >= attack_onset_s
                and y3 >= THRESHOLD
            ):
                first_post_y3 = ts

            if (
                first_post_y6 is None
                and ts >= attack_onset_s
                and y6 >= THRESHOLD
            ):
                first_post_y6 = ts

    if scenario_type == "ATTACK":

        predictive_y3 = first_pre_y3
        predictive_y6 = first_pre_y6

        if predictive_y3 is not None:

            y3_lead = (
                attack_onset_s
                - predictive_y3
            )

        else:

            y3_lead = None

        if predictive_y6 is not None:

            y6_lead = (
                attack_onset_s
                - predictive_y6
            )

        else:

            y6_lead = None

    else:

        predictive_y3 = None
        predictive_y6 = None
        y3_lead = None
        y6_lead = None

    results = []

    for index, record in enumerate(
        metadata
    ):

        ts = record["timestamp_s"]

        y3 = float(
            y3_probabilities[index]
        )

        y6 = float(
            y6_probabilities[index]
        )

        results.append(
            {
                "scenario_id": scenario_id,
                "scenario_type": scenario_type,
                "source_step": int(
                    record["source_step"]
                ),
                "timestamp_s": float(ts),

                "scenario_active": bool(
                    record["scenario_active"]
                ),

                "attack_onset_step": (
                    attack_onset_step
                ),

                "attack_end_step": (
                    attack_end_step
                ),

                "attack_onset_s": (
                    attack_onset_s
                ),

                "attack_end_s": (
                    attack_end_s
                ),

                "seconds_from_attack_onset": (
                    None
                    if attack_onset_s is None
                    else float(
                        ts - attack_onset_s
                    )
                ),

                "y3_probability": y3,
                "y6_probability": y6,

                "y3_risk_level": (
                    probability_to_level(y3)
                ),

                "y6_risk_level": (
                    probability_to_level(y6)
                ),

                "y3_threshold_0_50": bool(
                    y3 >= THRESHOLD
                ),

                "y6_threshold_0_50": bool(
                    y6 >= THRESHOLD
                ),

                "first_y3_threshold_crossing_s": (
                    first_y3
                ),

                "first_y6_threshold_crossing_s": (
                    first_y6
                ),

                "first_pre_attack_y3_detection_s": (
                    first_pre_y3
                ),

                "first_pre_attack_y6_detection_s": (
                    first_pre_y6
                ),

                "first_post_onset_y3_detection_s": (
                    first_post_y3
                ),

                "first_post_onset_y6_detection_s": (
                    first_post_y6
                ),

                "y3_predictive_lead_time_s": (
                    y3_lead
                ),

                "y6_predictive_lead_time_s": (
                    y6_lead
                ),

                "model_input_feature_count": (
                    FEATURE_COUNT
                ),

                "model_observation_window_steps": (
                    OBSERVATION_STEPS
                ),

                "model_observation_window_seconds": (
                    OBSERVATION_STEPS * 0.05
                ),

                "y3_horizon_seconds": (
                    Y3_HORIZON
                ),

                "y6_horizon_seconds": (
                    Y6_HORIZON
                ),

                "laboratory_generated": True,
                "physical_vehicle": False,
                "direct_actuation": False,
                "model_retraining": False,
                "real_world_cyberattack": False,
                "live_cortex_xdr": False,
            }
        )

    threshold_y3_count = int(
        np.sum(
            y3_probabilities >= THRESHOLD
        )
    )

    threshold_y6_count = int(
        np.sum(
            y6_probabilities >= THRESHOLD
        )
    )

    summary = {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,

        "input_records": int(
            len(scenario_df)
        ),

        "timesteps": int(
            len(timestep_records)
        ),

        "prediction_rows": int(
            len(results)
        ),

        "active_timesteps": int(
            len(active_steps)
        ),

        "max_y3_probability": float(
            np.max(y3_probabilities)
        ),

        "max_y6_probability": float(
            np.max(y6_probabilities)
        ),

        "mean_y3_probability": float(
            np.mean(y3_probabilities)
        ),

        "mean_y6_probability": float(
            np.mean(y6_probabilities)
        ),

        "y3_threshold_crossing_count": (
            threshold_y3_count
        ),

        "y6_threshold_crossing_count": (
            threshold_y6_count
        ),

        "y3_threshold_detected": bool(
            threshold_y3_count > 0
        ),

        "y6_threshold_detected": bool(
            threshold_y6_count > 0
        ),

        "first_y3_threshold_crossing_s": (
            first_y3
        ),

        "first_y6_threshold_crossing_s": (
            first_y6
        ),

        "first_pre_attack_y3_detection_s": (
            first_pre_y3
        ),

        "first_pre_attack_y6_detection_s": (
            first_pre_y6
        ),

        "first_post_onset_y3_detection_s": (
            first_post_y3
        ),

        "first_post_onset_y6_detection_s": (
            first_post_y6
        ),

        "y3_predictive_lead_time_s": (
            y3_lead
        ),

        "y6_predictive_lead_time_s": (
            y6_lead
        ),
    }

    return results, summary


# ================================================================
# JSON-SAFE CONVERSION
# ================================================================

def clean_json_value(value):

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):

        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        float,
    ):

        if not np.isfinite(value):
            return None

        return value

    return value


# ================================================================
# MAIN
# ================================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== PHASE 5.11 PREDICTIVE DETECTION ==="
    )

    # ------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Missing Phase 5.10 input: {INPUT_CSV}"
        )

    if not BASELINE_CSV.exists():

        raise FileNotFoundError(
            f"Missing Phase 5.9 baseline: {BASELINE_CSV}"
        )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Missing Transformer checkpoint: {CHECKPOINT}"
        )

    attack_df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    baseline_source_df = pd.read_csv(
        BASELINE_CSV,
        low_memory=False,
    )

    print(
        f"Phase 5.10 input rows: "
        f"{len(attack_df)}"
    )

    print(
        f"Phase 5.9 baseline rows: "
        f"{len(baseline_source_df)}"
    )

    required_columns = {
        "scenario_id",
        "scenario_active",
        "signals_json",
        "timestamp_s",
        "can_id",
        "message_name",
    }

    missing = (
        required_columns
        - set(attack_df.columns)
    )

    if missing:

        raise ValueError(
            "Phase 5.10 input is missing "
            f"required columns: {sorted(missing)}"
        )

    # ------------------------------------------------------------
    # NORMALIZE SCENARIO IDS
    # ------------------------------------------------------------

    attack_df["scenario_id"] = (
        attack_df["scenario_id"]
        .astype(str)
        .str.strip()
    )

    baseline_source_df["scenario_id"] = (
        baseline_source_df["scenario_id"]
        .astype(str)
        .str.strip()
    )

    available_scenarios = sorted(
        attack_df["scenario_id"]
        .dropna()
        .unique()
        .tolist()
    )

    scenarios = [
        scenario
        for scenario in ATTACK_SCENARIOS
        if scenario in available_scenarios
    ]

    print(
        f"Attack scenarios found: "
        f"{len(scenarios)}"
    )

    print(
        "Available scenario IDs:"
    )

    for scenario in available_scenarios:

        print(
            f"  {scenario}"
        )

    if len(scenarios) != 10:

        missing_scenarios = [
            scenario
            for scenario in ATTACK_SCENARIOS
            if scenario not in scenarios
        ]

        raise ValueError(
            "Phase 5.10 does not contain all "
            "10 expected cyber-physical scenarios. "
            f"Missing: {missing_scenarios}"
        )

    # ------------------------------------------------------------
    # ATTACK INPUT COUNT
    # ------------------------------------------------------------

    attack_rows_only = attack_df[
        attack_df["scenario_id"]
        != "S00_NORMAL_BASELINE"
    ].copy()

    if len(attack_rows_only) != 90000:

        raise ValueError(
            "Expected 90,000 attack-only "
            f"records; received {len(attack_rows_only)}."
        )

    # ------------------------------------------------------------
    # BASELINE
    # ------------------------------------------------------------

    baseline_ids = sorted(
        baseline_source_df["scenario_id"]
        .dropna()
        .unique()
        .tolist()
    )

    if (
        "S00_NORMAL_BASELINE"
        not in baseline_ids
    ):

        raise ValueError(
            "Phase 5.9 baseline file does not "
            "contain S00_NORMAL_BASELINE."
        )

    baseline_df = baseline_source_df[
        baseline_source_df["scenario_id"]
        == "S00_NORMAL_BASELINE"
    ].copy()

    print(
        f"Baseline records selected: "
        f"{len(baseline_df)}"
    )

    if len(baseline_df) != 9000:

        raise ValueError(
            "Expected 9,000 baseline records; "
            f"received {len(baseline_df)}."
        )

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model = load_model()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    checkpoint_hash = sha256_file(
        CHECKPOINT
    )

    print(
        f"Model parameters: "
        f"{parameter_count}"
    )

    print(
        f"Checkpoint SHA256: "
        f"{checkpoint_hash}"
    )

    if parameter_count != 407811:

        raise ValueError(
            "Unexpected Transformer parameter "
            f"count: {parameter_count}; "
            "expected 407811."
        )

    # ------------------------------------------------------------
    # EVALUATE BASELINE
    # ------------------------------------------------------------

    all_results = []

    summaries = []

    print(
        "Evaluating S00_NORMAL_BASELINE..."
    )

    baseline_results, baseline_summary = (
        evaluate_stream(
            model=model,
            scenario_id="S00_NORMAL_BASELINE",
            scenario_type="BASELINE",
            scenario_df=baseline_df,
        )
    )

    all_results.extend(
        baseline_results
    )

    summaries.append(
        baseline_summary
    )

    # ------------------------------------------------------------
    # EVALUATE 10 ATTACK STREAMS
    # ------------------------------------------------------------

    for scenario in scenarios:

        print(
            f"Evaluating {scenario}..."
        )

        scenario_df = attack_df[
            attack_df["scenario_id"]
            == scenario
        ].copy()

        if len(scenario_df) != 9000:

            raise ValueError(
                f"{scenario}: expected 9,000 "
                f"records; received {len(scenario_df)}."
            )

        results, summary = evaluate_stream(
            model=model,
            scenario_id=scenario,
            scenario_type="ATTACK",
            scenario_df=scenario_df,
        )

        all_results.extend(
            results
        )

        summaries.append(
            summary
        )

    # ------------------------------------------------------------
    # RESULT DATAFRAME
    # ------------------------------------------------------------

    result_df = pd.DataFrame(
        all_results
    )

    if result_df.empty:

        raise RuntimeError(
            "Predictive detection produced "
            "zero result rows."
        )

    expected_total_predictions = (
        EXPECTED_PREDICTIONS * 11
    )

    if len(result_df) != expected_total_predictions:

        raise RuntimeError(
            "Unexpected prediction row count: "
            f"{len(result_df)}; expected "
            f"{expected_total_predictions}."
        )

    # ------------------------------------------------------------
    # OUTPUT VALIDATION
    # ------------------------------------------------------------

    required_result_columns = {
        "scenario_id",
        "scenario_type",
        "source_step",
        "timestamp_s",
        "y3_probability",
        "y6_probability",
        "laboratory_generated",
        "physical_vehicle",
        "direct_actuation",
        "model_retraining",
        "real_world_cyberattack",
        "live_cortex_xdr",
    }

    missing_result = (
        required_result_columns
        - set(result_df.columns)
    )

    if missing_result:

        raise RuntimeError(
            "Prediction output is missing "
            f"required columns: {sorted(missing_result)}"
        )

    # ------------------------------------------------------------
    # SAVE CSV
    # ------------------------------------------------------------

    result_df.to_csv(
        OUT_CSV,
        index=False,
    )

    # ------------------------------------------------------------
    # SAVE JSONL
    # ------------------------------------------------------------

    with open(
        OUT_JSONL,
        "w",
        encoding="utf-8",
    ) as handle:

        for record in all_results:

            clean_record = {
                key: clean_json_value(value)
                for key, value in record.items()
            }

            handle.write(
                json.dumps(
                    clean_record,
                    separators=(
                        ",",
                        ":",
                    ),
                    allow_nan=False,
                )
                + "\n"
            )

    # ------------------------------------------------------------
    # VALIDATION CHECKS
    # ------------------------------------------------------------

    checks = {}

    checks[
        "input_phase5_10_exists"
    ] = INPUT_CSV.exists()

    checks[
        "baseline_phase5_9_exists"
    ] = BASELINE_CSV.exists()

    checks[
        "checkpoint_exists"
    ] = CHECKPOINT.exists()

    checks[
        "attack_input_rows_90000"
    ] = len(attack_rows_only) == 90000

    checks[
        "baseline_input_rows_9000"
    ] = len(baseline_df) == 9000

    checks[
        "attack_scenario_count_10"
    ] = len(scenarios) == 10

    checks[
        "feature_count_59"
    ] = len(FEATURES) == 59

    checks[
        "binary_count_27"
    ] = len(BINARY_FEATURES) == 27

    checks[
        "continuous_count_32"
    ] = len(CONTINUOUS_FEATURES) == 32

    checks[
        "parameter_count_407811"
    ] = parameter_count == 407811

    checks[
        "baseline_predictions_589"
    ] = (
        len(
            result_df[
                result_df["scenario_id"]
                == "S00_NORMAL_BASELINE"
            ]
        )
        == EXPECTED_PREDICTIONS
    )

    checks[
        "attack_predictions_589_each"
    ] = all(
        len(
            result_df[
                result_df["scenario_id"]
                == scenario
            ]
        )
        == EXPECTED_PREDICTIONS
        for scenario in scenarios
    )

    checks[
        "total_prediction_rows_6479"
    ] = (
        len(result_df)
        == EXPECTED_PREDICTIONS * 11
    )

    checks[
        "y3_finite"
    ] = bool(
        np.isfinite(
            result_df[
                "y3_probability"
            ].astype(float)
        ).all()
    )

    checks[
        "y6_finite"
    ] = bool(
        np.isfinite(
            result_df[
                "y6_probability"
            ].astype(float)
        ).all()
    )

    checks[
        "probabilities_in_range"
    ] = bool(
        result_df[
            "y3_probability"
        ].between(
            0.0,
            1.0,
        ).all()
        and
        result_df[
            "y6_probability"
        ].between(
            0.0,
            1.0,
        ).all()
    )

    checks[
        "all_attack_scenarios_present"
    ] = (
        set(
            result_df[
                result_df["scenario_type"]
                == "ATTACK"
            ]["scenario_id"]
        )
        == set(scenarios)
    )

    checks[
        "baseline_no_active_attack"
    ] = (
        baseline_summary[
            "active_timesteps"
        ]
        == 0
    )

    checks[
        "each_attack_has_20_active_steps"
    ] = all(
        summary[
            "active_timesteps"
        ]
        == 20
        for summary in summaries
        if summary[
            "scenario_type"
        ] == "ATTACK"
    )

    checks[
        "each_attack_has_9000_records"
    ] = all(
        summary[
            "input_records"
        ]
        == 9000
        for summary in summaries
        if summary[
            "scenario_type"
        ] == "ATTACK"
    )

    checks[
        "each_stream_has_600_timesteps"
    ] = all(
        summary[
            "timesteps"
        ]
        == EXPECTED_TIMESTEPS
        for summary in summaries
    )

    checks[
        "each_stream_has_589_predictions"
    ] = all(
        summary[
            "prediction_rows"
        ]
        == EXPECTED_PREDICTIONS
        for summary in summaries
    )

    checks[
        "no_model_retraining"
    ] = not bool(
        result_df[
            "model_retraining"
        ].any()
    )

    checks[
        "laboratory_only"
    ] = bool(
        result_df[
            "laboratory_generated"
        ].all()
        and not result_df[
            "physical_vehicle"
        ].any()
    )

    checks[
        "no_direct_actuation"
    ] = not bool(
        result_df[
            "direct_actuation"
        ].any()
    )

    checks[
        "no_real_world_attack_claim"
    ] = not bool(
        result_df[
            "real_world_cyberattack"
        ].any()
    )

    checks[
        "no_live_cortex_xdr"
    ] = not bool(
        result_df[
            "live_cortex_xdr"
        ].any()
    )

    # ------------------------------------------------------------
    # DISK RE-READ VALIDATION
    # ------------------------------------------------------------

    disk_df = pd.read_csv(
        OUT_CSV,
        low_memory=False,
    )

    checks[
        "disk_output_rows_6479"
    ] = (
        len(disk_df)
        == expected_total_predictions
    )

    disk_scenario_counts = (
        disk_df[
            "scenario_id"
        ]
        .value_counts()
        .to_dict()
    )

    checks[
        "disk_baseline_589"
    ] = (
        disk_scenario_counts.get(
            "S00_NORMAL_BASELINE",
            0,
        )
        == EXPECTED_PREDICTIONS
    )

    checks[
        "disk_each_attack_589"
    ] = all(
        disk_scenario_counts.get(
            scenario,
            0,
        )
        == EXPECTED_PREDICTIONS
        for scenario in scenarios
    )

    # ------------------------------------------------------------
    # OVERALL STATUS
    # ------------------------------------------------------------

    overall_pass = all(
        checks.values()
    )

    # ------------------------------------------------------------
    # SCHEMA
    # ------------------------------------------------------------

    schema = {

        "phase": "5.11",

        "status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),

        "purpose": (
            "Predictive detection evaluation "
            "using the frozen CRSS-trained "
            "Transformer against laboratory "
            "cyber-physical CAN/EV scenarios."
        ),

        "inputs": {

            "phase5_10_attack_input": str(
                INPUT_CSV
            ),

            "phase5_9_normal_baseline": str(
                BASELINE_CSV
            ),
        },

        "model": {

            "feature_count": FEATURE_COUNT,

            "binary_feature_count": (
                len(BINARY_FEATURES)
            ),

            "continuous_feature_count": (
                len(CONTINUOUS_FEATURES)
            ),

            "observation_steps": (
                OBSERVATION_STEPS
            ),

            "observation_window_seconds": (
                OBSERVATION_STEPS * 0.05
            ),

            "y3_horizon_seconds": (
                Y3_HORIZON
            ),

            "y6_horizon_seconds": (
                Y6_HORIZON
            ),

            "threshold": THRESHOLD,

            "embedding_dimension": 128,

            "transformer_layers": 3,

            "attention_heads": 8,

            "feedforward_dimension": 256,

            "dropout": 0.1,

            "parameters": parameter_count,

            "frozen": True,

            "checkpoint_sha256": (
                checkpoint_hash
            ),

            "normalization_boundary": (
                "Exact original training scaler "
                "parameters are not substituted "
                "or invented in Phase 5.11."
            ),
        },

        "dataset": {

            "phase5_10_total_records": int(
                len(attack_df)
            ),

            "phase5_10_attack_only_records": int(
                len(attack_rows_only)
            ),

            "phase5_9_baseline_records": int(
                len(baseline_df)
            ),

            "expected_attack_scenarios": 10,

            "expected_records_per_attack": 9000,

            "expected_prediction_rows_per_stream": (
                EXPECTED_PREDICTIONS
            ),
        },

        "streams": {

            "baseline": 1,

            "attack_scenarios": len(
                scenarios
            ),

            "total": (
                len(scenarios) + 1
            ),
        },

        "prediction_rows": int(
            len(result_df)
        ),

        "scenario_summary": summaries,

        "validation": checks,

        "boundaries": {

            "laboratory_generated": True,

            "physical_vehicle": False,

            "direct_actuation": False,

            "model_retraining": False,

            "real_world_cyberattack": False,

            "live_cortex_xdr": False,
        },
    }

    with open(
        OUT_SCHEMA,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            schema,
            handle,
            indent=2,
        )

    # ------------------------------------------------------------
    # MANIFEST
    # ------------------------------------------------------------

    manifest = {

        "phase": "5.11",

        "status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),

        "experiment": (
            "Frozen Transformer predictive "
            "detection over laboratory CAN/EV "
            "cyber-physical attack scenarios "
            "with normal-baseline comparison."
        ),

        "phase5_10_input_rows": int(
            len(attack_df)
        ),

        "phase5_10_attack_only_rows": int(
            len(attack_rows_only)
        ),

        "phase5_9_baseline_rows": int(
            len(baseline_df)
        ),

        "attack_scenario_count": int(
            len(scenarios)
        ),

        "total_stream_count": int(
            len(scenarios) + 1
        ),

        "prediction_rows": int(
            len(result_df)
        ),

        "checkpoint_sha256": (
            checkpoint_hash
        ),

        "model_parameters": int(
            parameter_count
        ),

        "threshold": THRESHOLD,

        "scenario_summary": summaries,

        "validation": checks,

        "outputs": {

            "csv": str(
                OUT_CSV
            ),

            "jsonl": str(
                OUT_JSONL
            ),

            "schema": str(
                OUT_SCHEMA
            ),

            "manifest": str(
                OUT_MANIFEST
            ),
        },

        "research_boundary": {

            "laboratory_generated": True,

            "physical_vehicle": False,

            "direct_actuation": False,

            "model_retraining": False,

            "real_world_cyberattack": False,

            "live_cortex_xdr": False,
        },
    }

    with open(
        OUT_MANIFEST,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
        )

    # ------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------

    print()
    print(
        "=== PHASE 5.11 RESULTS ==="
    )

    print(
        f"Attack scenarios evaluated: "
        f"{len(scenarios)}"
    )

    print(
        "Normal baseline evaluated: 1"
    )

    print(
        f"Phase 5.10 total input rows: "
        f"{len(attack_df)}"
    )

    print(
        f"Attack-only input rows: "
        f"{len(attack_rows_only)}"
    )

    print(
        f"Total prediction rows: "
        f"{len(result_df)}"
    )

    print(
        "Y3 maximum probability: "
        f"{result_df['y3_probability'].max():.6f}"
    )

    print(
        "Y6 maximum probability: "
        f"{result_df['y6_probability'].max():.6f}"
    )

    print()
    print(
        "Validation checks:"
    )

    for name, passed in checks.items():

        print(
            f"  {'PASS' if passed else 'FAIL'} "
            f"{name}"
        )

    print()
    print(
        f"CSV: {OUT_CSV}"
    )

    print(
        f"JSONL: {OUT_JSONL}"
    )

    print(
        f"Schema: {OUT_SCHEMA}"
    )

    print(
        f"Manifest: {OUT_MANIFEST}"
    )

    print()
    print(
        "STATUS: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )


if __name__ == "__main__":
    main()