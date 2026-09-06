from pathlib import Path

p = Path("scripts\crss_2024_phase4_17_error_analysis.py")
s = p.read_text(encoding="utf-8")

# Scenario IDs are authoritative string identifiers.
# Never cast them to int.
s = s.replace(
    '"scenario": int(scenario),',
    '"scenario": str(scenario),'
)

s = s.replace(
    '"scenario_id": int(scenario),',
    '"scenario_id": str(scenario),'
)

# Add explicit identifier validation after scenario loading.
marker = '''scenario_test = np.load(
    SEQ / "scenario_ids_test_v2_forecasting.npy",
    mmap_mode="r"
)
'''

if marker not in s:
    raise RuntimeError(
        "Could not locate scenario_test loading block."
    )

validation = marker + '''
# Scenario IDs are opaque identifiers and must remain strings.
if scenario_test.dtype.kind not in {"U", "S", "O"}:
    raise RuntimeError(
        f"Scenario IDs must be string/object identifiers, got dtype={scenario_test.dtype}"
    )

'''

if "Scenario IDs are opaque identifiers" not in s:
    s = s.replace(marker, validation, 1)

p.write_text(s, encoding="utf-8")

print("PATCH: PASS")
print("Scenario IDs are now treated as opaque strings.")
print("No int(scenario) conversion remains.")
