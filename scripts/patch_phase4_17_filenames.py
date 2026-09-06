from pathlib import Path

p = Path("scripts\crss_2024_phase4_17_error_analysis.py")
s = p.read_text(encoding="utf-8")

replacements = {
    '"scenario_test_v2_forecasting.npy"':
        '"scenario_ids_test_v2_forecasting.npy"',
    '"scenario_train_v2_forecasting.npy"':
        '"scenario_ids_train_v2_forecasting.npy"',
    '"prediction_origin_test_v2_forecasting.npy"':
        '"prediction_origins_test_v2_forecasting.npy"',
}

for old, new in replacements.items():
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")

print("PATCH: PASS")
print("Scenario file: scenario_ids_test_v2_forecasting.npy")
print("Origin file: prediction_origins_test_v2_forecasting.npy")
