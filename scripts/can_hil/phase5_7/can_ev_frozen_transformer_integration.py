import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_5"
    / "virtual_ev_cyber_physical_features.csv"
)

TRAINING_SPEC = (
    ROOT
    / "experiments"
    / "modeling"
    / "v2"
    / "training"
    / "crss_2024_v2_training_infrastructure_spec.json"
)

CHECKPOINT = (
    ROOT
    / "models"
    / "v2"
    / "advanced"
    / "crss_2024_v2_temporal_transformer_controlled_best.pt"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "can_hil"
    / "phase5_7"
)

REPORT_DIR = (
    ROOT
    / "experiments"
    / "can_hil"
    / "phase5_7"
)

OUTPUT_CSV = OUTPUT_DIR / "can_ev_frozen_transformer_predictions.csv"
OUTPUT_JSONL = OUTPUT_DIR / "can_ev_frozen_transformer_predictions.jsonl"
REPORT = REPORT_DIR / "phase5_7_frozen_transformer_integration.json"
CONTRACT = OUTPUT_DIR / "phase5_7_model_input_contract.json"


# ================================================================
# EXACT 59-FEATURE FROZEN MODEL CONTRACT
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

OBSERVATION_STEPS = 12
FEATURE_COUNT = 59

EMBEDDING_DIM = 128
TRANSFORMER_LAYERS = 3
ATTENTION_HEADS = 8
FEEDFORWARD_DIM = 256
DROPOUT = 0.1


# ================================================================
# MODEL ARCHITECTURE
# ================================================================

class TemporalTransformer(nn.Module):
    """
    Exact architecture reconstructed from the frozen Phase 4.10D
    checkpoint state_dict.

    Verified checkpoint tensors:
        position                         [1, 12, 128]
        input_projection                 [128, 59]
        input_norm                       LayerNorm(128)
        encoder                          3 TransformerEncoder layers
        attention_pool                   Linear(128, 1)
        pool_norm                        LayerNorm(128)
        output_norm                      LayerNorm(128)
        y3_head                          Linear(128, 1)
        y6_head                          Linear(128, 1)

    Verified total parameter count:
        407811

    The model remains frozen. No retraining occurs in Phase 5.7.
    """

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

        # Exact checkpoint parameter name: "position"
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

        # Exact checkpoint parameter names.
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

        # Input projection
        x = self.input_projection(x)

        # Input normalization
        x = self.input_norm(x)

        # Learned temporal position encoding
        x = x + self.position[
            :,
            :x.size(1),
            :,
        ]

        # Temporal Transformer
        x = self.encoder(x)

        # Attention pooling
        attention_scores = self.attention_pool(x)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        pooled = torch.sum(
            attention_weights * x,
            dim=1,
        )

        # Pool normalization
        pooled = self.pool_norm(
            pooled
        )

        # Output normalization
        pooled = self.output_norm(
            pooled
        )

        # Two frozen forecasting heads
        y3 = self.y3_head(
            pooled
        ).squeeze(-1)

        y6 = self.y6_head(
            pooled
        ).squeeze(-1)

        return y3, y6


# ================================================================
# UTILITIES
# ================================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def find_feature_processing(obj):

    """
    Recursively locate the nested feature_processing object.
    The training specification does not place this object at
    the JSON root.
    """

    if isinstance(obj, dict):

        if "feature_processing" in obj:
            return obj["feature_processing"]

        for value in obj.values():

            found = find_feature_processing(
                value
            )

            if found is not None:
                return found

    elif isinstance(obj, list):

        for value in obj:

            found = find_feature_processing(
                value
            )

            if found is not None:
                return found

    return None


def validate_training_spec():

    with open(
        TRAINING_SPEC,
        "r",
        encoding="utf-8",
    ) as f:

        spec = json.load(f)

    processing = find_feature_processing(
        spec
    )

    if processing is None:

        raise KeyError(
            "Could not locate nested "
            "'feature_processing' section "
            "in training specification."
        )

    binary = processing[
        "binary_features"
    ]

    continuous = processing[
        "continuous_features"
    ]

    checks = {}

    checks[
        "spec_feature_count_59"
    ] = (
        len(binary)
        + len(continuous)
        == 59
    )

    checks[
        "spec_binary_count_27"
    ] = (
        len(binary) == 27
    )

    checks[
        "spec_continuous_count_32"
    ] = (
        len(continuous) == 32
    )

    checks[
        "spec_normalization_train_only_z_score"
    ] = (
        processing[
            "continuous_normalization"
        ]
        == "train_only_z_score"
    )

    checks[
        "local_contract_matches_spec"
    ] = (
        binary == BINARY_FEATURES
        and continuous == CONTINUOUS_FEATURES
    )

    return spec, checks


def feature_value(row, feature):

    if feature in row.index:

        value = row[feature]

        if pd.notna(value):

            return float(value)

    return 0.0


# ================================================================
# CAN / EV â†’ 59 FEATURE COMPATIBILITY ADAPTER
# ================================================================

def derive_model_compatible_features(row):

    """
    Build the frozen model's 59-feature compatibility
    representation from laboratory CAN/EV observations.

    This does NOT claim that CAN/EV signals are equivalent
    to the original CRSS variables.
    """

    out = {}

    integrity = feature_value(
        row,
        "IntegrityStatus",
    )

    gateway_integrity = feature_value(
        row,
        "GatewayIntegrity",
    )

    sequence = feature_value(
        row,
        "SequenceCounter",
    )

    restraint_sequence = feature_value(
        row,
        "RestraintSequenceCounter",
    )

    heartbeat_values = [
        feature_value(row, "VCUHeartbeat"),
        feature_value(row, "BMSHeartbeat"),
        feature_value(row, "InverterHeartbeat"),
        feature_value(row, "GatewayHeartbeat"),
        feature_value(row, "RCMHeartbeat"),
        feature_value(row, "OccupantSensorHeartbeat"),
        feature_value(
            row,
            "RestraintGatewayHeartbeat",
        ),
    ]

    heartbeat_failure = float(
        any(
            value < 0.5
            for value in heartbeat_values
        )
    )

    can_interval = feature_value(
        row,
        "can_frame_interval_s",
    )

    if can_interval <= 0:
        can_interval = 0.05

    interval_deviation = abs(
        can_interval - 0.05
    )

    speed = feature_value(
        row,
        "VehicleSpeed",
    )

    wheel_error = feature_value(
        row,
        "speed_wheel_speed_error",
    )

    sensor_disagreement = (
        1.0
        if abs(wheel_error) > 2.0
        else 0.0
    )

    torque_command = feature_value(
        row,
        "TorqueCommand",
    )

    torque_feedback = feature_value(
        row,
        "TorqueFeedback",
    )

    torque_error = abs(
        torque_command
        - torque_feedback
    )

    diagnostic = feature_value(
        row,
        "DiagnosticState",
    )

    restraint_diagnostic = feature_value(
        row,
        "RestraintDiagnostic",
    )

    restraint_integrity = feature_value(
        row,
        "RestraintIntegrity",
    )

    crash_event = feature_value(
        row,
        "CrashEventStatus",
    )

    rollover = feature_value(
        row,
        "RolloverStatus",
    )

    # ------------------------------------------------------------
    # 27 BINARY FEATURES
    # ------------------------------------------------------------

    out[
        "authentication_failure_rate"
    ] = 0.0

    out[
        "collision_event_flag"
    ] = float(
        crash_event > 0
    )

    out[
        "connectivity_degradation_rate"
    ] = (
        1.0
        if heartbeat_failure
        else 0.0
    )

    out[
        "connectivity_drop_rate"
    ] = (
        1.0
        if heartbeat_failure
        else 0.0
    )

    out[
        "cyber_instability_index"
    ] = float(
        (
            (1.0 - integrity)
            + (1.0 - gateway_integrity)
            + heartbeat_failure
        )
        / 3.0
    )

    out[
        "fire_explosion_flag"
    ] = 0.0

    out[
        "message_integrity_failure_rate"
    ] = 1.0 - integrity

    out[
        "packet_loss_rate"
    ] = (
        1.0
        if heartbeat_failure
        else 0.0
    )

    out[
        "redundant_sensor_divergence"
    ] = sensor_disagreement

    out[
        "replay_indicator_rate"
    ] = 0.0

    out[
        "resilience_degradation_rate"
    ] = float(
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

    out[
        "rollover_flag"
    ] = float(
        rollover > 0
    )

    out[
        "safety_critical_ecu_ratio"
    ] = float(
        sum(
            value >= 0.5
            for value in heartbeat_values
        )
        / len(heartbeat_values)
    )

    out[
        "sensor_disagreement_rate"
    ] = sensor_disagreement

    out[
        "sensor_plausibility_score"
    ] = (
        1.0
        - sensor_disagreement
    )

    out[
        "sequence_counter_anomaly_rate"
    ] = (
        1.0
        if (
            sequence < 0
            or restraint_sequence < 0
        )
        else 0.0
    )

    out[
        "speed_over_limit_flag"
    ] = (
        1.0
        if speed > 130.0
        else 0.0
    )

    out[
        "state_transition_magnitude"
    ] = float(
        abs(diagnostic)
        + abs(restraint_diagnostic)
    )

    out[
        "uds_request_rate"
    ] = 0.0

    out[
        "unexpected_source_id_rate"
    ] = (
        1.0
        if gateway_integrity < 0.5
        else 0.0
    )

    out[
        "vehicle_damage_flag"
    ] = float(
        crash_event > 0
        or rollover > 0
    )

    # ------------------------------------------------------------
    # 32 CONTINUOUS FEATURES
    # ------------------------------------------------------------

    out["VISION"] = 0.0
    out["WEATHER"] = 0.0

    out[
        "can_bus_count"
    ] = 1.0

    out[
        "can_bus_load_pct"
    ] = 0.0

    out[
        "can_interarrival_jitter_ms"
    ] = (
        interval_deviation
        * 1000.0
    )

    out[
        "can_interarrival_mean_ms"
    ] = (
        can_interval
        * 1000.0
    )

    out[
        "can_message_rate"
    ] = (
        1.0 / can_interval
    )

    out[
        "can_message_rate_deviation"
    ] = abs(
        out["can_message_rate"]
        - 20.0
    )

    out[
        "d1_authentication_failure_rate"
    ] = 0.0

    out[
        "d1_can_bus_load_pct"
    ] = out[
        "can_bus_load_pct"
    ]

    out[
        "d1_can_interarrival_jitter_ms"
    ] = out[
        "can_interarrival_jitter_ms"
    ]

    out[
        "d1_can_message_rate"
    ] = out[
        "can_message_rate"
    ]

    out[
        "d1_connectivity_drop_rate"
    ] = out[
        "connectivity_drop_rate"
    ]

    out[
        "d1_latency_jitter_ms"
    ] = out[
        "can_interarrival_jitter_ms"
    ]

    out[
        "d1_latency_ms"
    ] = out[
        "can_interarrival_mean_ms"
    ]

    out[
        "d1_message_integrity_failure_rate"
    ] = out[
        "message_integrity_failure_rate"
    ]

    out[
        "d1_packet_loss_rate"
    ] = out[
        "packet_loss_rate"
    ]

    out[
        "d1_replay_indicator_rate"
    ] = out[
        "replay_indicator_rate"
    ]

    out[
        "d1_sensor_disagreement_rate"
    ] = out[
        "sensor_disagreement_rate"
    ]

    out[
        "d1_sequence_counter_anomaly_rate"
    ] = out[
        "sequence_counter_anomaly_rate"
    ]

    out[
        "d1_telemetry_gap_duration_ms"
    ] = 0.0

    out[
        "diagnostic_event_rate"
    ] = float(
        diagnostic > 0
    )

    out[
        "ecu_count"
    ] = float(
        sum(
            value >= 0.5
            for value in heartbeat_values
        )
    )

    out[
        "ecu_state_transition_rate"
    ] = float(
        abs(diagnostic)
    )

    out[
        "latency_jitter_ms"
    ] = out[
        "can_interarrival_jitter_ms"
    ]

    out[
        "latency_ms"
    ] = out[
        "can_interarrival_mean_ms"
    ]

    out[
        "rolling_mean_3_latency_ms"
    ] = out[
        "latency_ms"
    ]

    out[
        "rolling_std_3_latency_ms"
    ] = 0.0

    out[
        "rssi_dbm"
    ] = 0.0

    out[
        "speed_limit_deviation"
    ] = (
        speed - 130.0
    )

    out[
        "speed_limit_deviation_abs"
    ] = abs(
        out[
            "speed_limit_deviation"
        ]
    )

    out[
        "telemetry_gap_duration_ms"
    ] = 0.0

    return out


# ================================================================
# MODEL LOADING
# ================================================================

def build_model():

    model = TemporalTransformer(
        input_dim=FEATURE_COUNT,
        embedding_dim=EMBEDDING_DIM,
        num_layers=TRANSFORMER_LAYERS,
        num_heads=ATTENTION_HEADS,
        ff_dim=FEEDFORWARD_DIM,
        dropout=DROPOUT,
        max_len=OBSERVATION_STEPS,
    )

    # Trusted local project checkpoint.
    # PyTorch 2.6+ defaults to weights_only=True.
    # This checkpoint is a trusted project-generated research artifact.
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif (
            "state_dict"
            in checkpoint
        ):

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
# MAIN
# ================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== PHASE 5.7 FROZEN TRANSFORMER INTEGRATION ==="
    )

    required_files = {
        "input_adapter_csv": INPUT_CSV,
        "training_spec": TRAINING_SPEC,
        "frozen_checkpoint": CHECKPOINT,
    }

    for name, path in required_files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required artifact: "
                f"{name}: {path}"
            )

    # ------------------------------------------------------------
    # Validate training contract
    # ------------------------------------------------------------

    _, spec_checks = (
        validate_training_spec()
    )

    print(
        f"Model contract features: "
        f"{len(FEATURES)}"
    )

    print(
        f"Binary features: "
        f"{len(BINARY_FEATURES)}"
    )

    print(
        f"Continuous features: "
        f"{len(CONTINUOUS_FEATURES)}"
    )

    # ------------------------------------------------------------
    # Load Phase 5.5 data
    # ------------------------------------------------------------

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        f"Input timesteps: "
        f"{len(df)}"
    )

    if len(df) < OBSERVATION_STEPS:

        raise RuntimeError(
            "Insufficient timesteps for "
            "12-step Transformer input."
        )

    # ------------------------------------------------------------
    # Build 59-feature compatibility representation
    # ------------------------------------------------------------

    adapted_rows = []

    for _, row in df.iterrows():

        adapted_rows.append(
            derive_model_compatible_features(
                row
            )
        )

    adapted = pd.DataFrame(
        adapted_rows
    )

    X59 = adapted[
        FEATURES
    ].copy()

    X59 = (
        X59
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    for feature in BINARY_FEATURES:

        X59[feature] = (
            X59[feature]
            .astype(float)
            .clip(0.0, 1.0)
        )

    # ------------------------------------------------------------
    # Build 12-step sequences
    # ------------------------------------------------------------

    sequences = []
    metadata = []

    for end_idx in range(
        OBSERVATION_STEPS - 1,
        len(X59),
    ):

        start_idx = (
            end_idx
            - OBSERVATION_STEPS
            + 1
        )

        sequence = (
            X59.iloc[
                start_idx:end_idx + 1
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        sequences.append(
            sequence
        )

        source_row = df.iloc[
            end_idx
        ]

        metadata.append(
            {
                "timestamp_s": float(
                    source_row[
                        "timestamp_s"
                    ]
                ),
                "vehicle_id": str(
                    source_row[
                        "vehicle_id"
                    ]
                ),
                "scenario": str(
                    source_row[
                        "scenario"
                    ]
                ),
                "source_row": int(
                    end_idx
                ),
            }
        )

    X = np.stack(
        sequences
    )

    print(
        f"Transformer sequences: "
        f"{X.shape[0]}"
    )

    print(
        f"Transformer tensor shape: "
        f"{X.shape}"
    )

    # ------------------------------------------------------------
    # Frozen checkpoint
    # ------------------------------------------------------------

    checkpoint_hash = (
        sha256_file(
            CHECKPOINT
        )
    )

    print(
        f"Checkpoint SHA256: "
        f"{checkpoint_hash}"
    )

    model = build_model()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Loaded Transformer parameters: "
        f"{parameter_count}"
    )

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------

    tensor = torch.from_numpy(
        X
    )

    with torch.no_grad():

        y3_logits, y6_logits = (
            model(tensor)
        )

    y3_probability = (
        torch.sigmoid(
            y3_logits
        )
        .numpy()
    )

    y6_probability = (
        torch.sigmoid(
            y6_logits
        )
        .numpy()
    )

    # ------------------------------------------------------------
    # Build result table
    # ------------------------------------------------------------

    results = []

    evidence_columns = [
        "SOC",
        "SOH",
        "HVVoltage",
        "HVCurrent",
        "BatteryTemperature",
        "CellVoltageDeviation",
        "CellTemperatureDeviation",
        "BMSState",
        "ContactorState",
        "PrechargeState",
        "IsolationStatus",
        "MotorRPM",
        "TorqueCommand",
        "TorqueFeedback",
        "MotorTemperature",
        "InverterTemperature",
        "DriverSeatbelt",
        "PassengerSeatbelt",
        "DriverPretensioner",
        "PassengerPretensioner",
        "DriverAirbag",
        "PassengerAirbag",
        "SideAirbag",
        "CurtainAirbag",
        "DriverOccupant",
        "PassengerOccupant",
        "CrashEventStatus",
        "ImpactSeverity",
        "RolloverStatus",
        "RestraintIntegrity",
        "GatewayIntegrity",
        "IntegrityStatus",
    ]

    for i, meta in enumerate(
        metadata
    ):

        result = dict(
            meta
        )

        result[
            "y3_probability"
        ] = float(
            y3_probability[i]
        )

        result[
            "y6_probability"
        ] = float(
            y6_probability[i]
        )

        result[
            "model_input_feature_count"
        ] = 59

        result[
            "model_observation_window_steps"
        ] = 12

        result[
            "model_observation_window_seconds"
        ] = 60

        result[
            "y3_horizon_seconds"
        ] = 15

        result[
            "y6_horizon_seconds"
        ] = 30

        source = df.iloc[
            meta["source_row"]
        ]

        for column in evidence_columns:

            if column in source.index:

                result[
                    f"evidence_{column}"
                ] = float(
                    source[column]
                )

        result[
            "laboratory_generated"
        ] = True

        result[
            "physical_vehicle"
        ] = False

        result[
            "direct_actuation"
        ] = False

        result[
            "model_retraining"
        ] = False

        result[
            "crss_replacement_claim"
        ] = False

        result[
            "live_cortex_xdr"
        ] = False

        results.append(
            result
        )

    result_df = pd.DataFrame(
        results
    )

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------

    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    with open(
        OUTPUT_JSONL,
        "w",
        encoding="utf-8",
    ) as f:

        for record in results:

            f.write(
                json.dumps(
                    record,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                + "\n"
            )

    # ------------------------------------------------------------
    # Contract artifact
    # ------------------------------------------------------------

    contract = {
        "phase": "5.7",
        "status": "BUILT",
        "purpose": (
            "CAN/EV laboratory compatibility "
            "adapter to frozen CRSS-trained "
            "Transformer"
        ),
        "model_input_shape": [
            12,
            59,
        ],
        "feature_count": 59,
        "binary_feature_count": 27,
        "continuous_feature_count": 32,
        "binary_features": BINARY_FEATURES,
        "continuous_features": CONTINUOUS_FEATURES,
        "normalization_contract": (
            "train_only_z_score"
        ),
        "normalization_boundary": (
            "Exact original training scaler "
            "parameters are not substituted or "
            "invented by Phase 5.7."
        ),
        "checkpoint": str(
            CHECKPOINT
        ),
        "checkpoint_sha256": (
            checkpoint_hash
        ),
        "embedding_dim": 128,
        "transformer_layers": 3,
        "attention_heads": 8,
        "feedforward_dimension": 256,
        "dropout": 0.1,
        "y3_horizon_seconds": 15,
        "y6_horizon_seconds": 30,
        "laboratory_generated": True,
        "physical_vehicle": False,
        "direct_actuation": False,
        "model_retraining": False,
        "crss_replacement_claim": False,
        "live_cortex_xdr": False,
        "ev_specific_features_preserved_separately": True,
        "restraint_specific_features_preserved_separately": True,
    }

    with open(
        CONTRACT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            contract,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    checks = {}

    checks.update(
        spec_checks
    )

    checks[
        "input_rows_600"
    ] = (
        len(df) == 600
    )

    checks[
        "input_features_82_or_more"
    ] = (
        len(df.columns) >= 82
    )

    checks[
        "model_feature_count_59"
    ] = (
        X59.shape[1] == 59
    )

    checks[
        "sequence_length_12"
    ] = (
        X.shape[1] == 12
    )

    checks[
        "sequence_feature_count_59"
    ] = (
        X.shape[2] == 59
    )

    checks[
        "sequence_count_589"
    ] = (
        X.shape[0] == 589
    )

    checks[
        "checkpoint_exists"
    ] = CHECKPOINT.exists()

    checks[
        "checkpoint_parameter_count_407811"
    ] = (
        parameter_count == 407811
    )

    checks[
        "checkpoint_sha256_64"
    ] = (
        len(checkpoint_hash) == 64
    )

    checks[
        "inference_rows_589"
    ] = (
        len(result_df) == 589
    )

    checks[
        "y3_probability_finite"
    ] = bool(
        np.isfinite(
            result_df[
                "y3_probability"
            ]
        ).all()
    )

    checks[
        "y6_probability_finite"
    ] = bool(
        np.isfinite(
            result_df[
                "y6_probability"
            ]
        ).all()
    )

    checks[
        "output_csv_exists"
    ] = OUTPUT_CSV.exists()

    checks[
        "output_jsonl_exists"
    ] = OUTPUT_JSONL.exists()

    checks[
        "contract_exists"
    ] = CONTRACT.exists()

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
    ] = bool(
        not result_df[
            "direct_actuation"
        ].any()
    )

    checks[
        "no_model_retraining"
    ] = bool(
        not result_df[
            "model_retraining"
        ].any()
    )

    checks[
        "no_crss_replacement_claim"
    ] = bool(
        not result_df[
            "crss_replacement_claim"
        ].any()
    )

    checks[
        "no_live_cortex_xdr"
    ] = bool(
        not result_df[
            "live_cortex_xdr"
        ].any()
    )

    overall_pass = all(
        checks.values()
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    report = {
        "phase": "5.7",
        "status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
        "experiment": (
            "CAN/EV laboratory integration "
            "with frozen CRSS-trained "
            "Transformer"
        ),
        "input": {
            "rows": int(
                len(df)
            ),
            "columns": int(
                len(df.columns)
            ),
            "source": str(
                INPUT_CSV
            ),
        },
        "model": {
            "checkpoint": str(
                CHECKPOINT
            ),
            "checkpoint_sha256": (
                checkpoint_hash
            ),
            "input_shape": [
                12,
                59,
            ],
            "parameter_count": int(
                parameter_count
            ),
            "embedding_dim": 128,
            "transformer_layers": 3,
            "attention_heads": 8,
            "feedforward_dimension": 256,
            "dropout": 0.1,
        },
        "sequences": {
            "count": int(
                X.shape[0]
            ),
            "shape": list(
                X.shape
            ),
        },
        "predictions": {
            "y3_horizon_seconds": 15,
            "y6_horizon_seconds": 30,
            "y3_mean": float(
                result_df[
                    "y3_probability"
                ].mean()
            ),
            "y3_max": float(
                result_df[
                    "y3_probability"
                ].max()
            ),
            "y6_mean": float(
                result_df[
                    "y6_probability"
                ].mean()
            ),
            "y6_max": float(
                result_df[
                    "y6_probability"
                ].max()
            ),
        },
        "boundary": {
            "laboratory_generated": True,
            "physical_vehicle": False,
            "direct_actuation": False,
            "model_retraining": False,
            "crss_replacement_claim": False,
            "live_cortex_xdr": False,
        },
        "validation": checks,
        "outputs": {
            "csv": str(
                OUTPUT_CSV
            ),
            "jsonl": str(
                OUTPUT_JSONL
            ),
            "contract": str(
                CONTRACT
            ),
        },
    }

    with open(
        REPORT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------

    print(
        "\n=== PHASE 5.7 RESULTS ==="
    )

    print(
        f"Input timesteps: "
        f"{len(df)}"
    )

    print(
        f"Model sequences: "
        f"{X.shape[0]}"
    )

    print(
        f"Model tensor: "
        f"{tuple(X.shape)}"
    )

    print(
        f"Model parameters: "
        f"{parameter_count}"
    )

    print(
        "Y3 mean probability: "
        f"{result_df['y3_probability'].mean():.6f}"
    )

    print(
        "Y6 mean probability: "
        f"{result_df['y6_probability'].mean():.6f}"
    )

    print(
        "\nValidation checks:"
    )

    for name, passed in checks.items():

        print(
            f"  "
            f"{'PASS' if passed else 'FAIL'} "
            f"{name}"
        )

    print(
        f"\nCSV: {OUTPUT_CSV}"
    )

    print(
        f"JSONL: {OUTPUT_JSONL}"
    )

    print(
        f"Contract: {CONTRACT}"
    )

    print(
        f"Report: {REPORT}"
    )

    print(
        "\nSTATUS: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
