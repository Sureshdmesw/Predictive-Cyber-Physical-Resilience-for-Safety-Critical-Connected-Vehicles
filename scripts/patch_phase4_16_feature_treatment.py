from pathlib import Path

p = Path("scripts\crss_2024_phase4_16_generalization.py")
s = p.read_text(encoding="utf-8")

# Insert the exact Phase 4.7 feature-treatment indices immediately
# before temporal_summary(), if not already present.
marker = "def temporal_summary(X):"

insert = '''FINAL_FEATURE_INDICES = [
    i for i in range(63)
    if i not in {30, 35, 39, 62}
]

if len(FINAL_FEATURE_INDICES) != 59:
    raise RuntimeError(
        f"Expected 59 final features, got {len(FINAL_FEATURE_INDICES)}"
    )


'''

if "FINAL_FEATURE_INDICES = [" not in s:
    if marker not in s:
        raise RuntimeError("Could not locate temporal_summary().")
    s = s.replace(marker, insert + marker, 1)

# Make temporal_summary explicitly apply the exact 63 -> 59 treatment.
old = '''def temporal_summary(X):
    """
    EXACT Phase 4.9 representation.
'''

new = '''def temporal_summary(X):
    """
    EXACT Phase 4.9 representation.

    First applies the exact Phase 4.7 treatment:
    63 original features -> 59 final features.

    Then:
    59 features × 8 temporal statistics = 472 dimensions.
'''

if old not in s:
    raise RuntimeError("Expected temporal_summary header not found.")

s = s.replace(old, new, 1)

old_line = '''    X = np.asarray(X, dtype=np.float32)

    last = X[:, -1, :]
'''

new_line = '''    X = np.asarray(X, dtype=np.float32)

    # Apply the exact Phase 4.7 feature treatment used by Phase 4.9.
    # Removed constant features:
    #   30 gateway_present
    #   35 physical_state_completeness
    #   39 roadway_complexity_index
    #   62 vehicle_state_observation_count
    if X.shape[-1] == 63:
        X = X[:, :, FINAL_FEATURE_INDICES]
    elif X.shape[-1] != 59:
        raise ValueError(
            f"Expected 63 original or 59 treated features, got {X.shape[-1]}"
        )

    if X.shape[-1] != 59:
        raise RuntimeError(
            f"Feature treatment failed: expected 59, got {X.shape[-1]}"
        )

    last = X[:, -1, :]
'''

if old_line not in s:
    raise RuntimeError("Expected temporal_summary input block not found.")

s = s.replace(old_line, new_line, 1)

# Add a hard dimensionality assertion immediately after test representation.
old = '''X_test_repr = temporal_summary(X_test)

'''

new = '''X_test_repr = temporal_summary(X_test)

if X_test_repr.shape[1] != 472:
    raise RuntimeError(
        f"Phase 4.9 compatibility failure: expected 472 representation "
        f"features, got {X_test_repr.shape[1]}"
    )

'''

if old not in s:
    raise RuntimeError("Could not locate X_test_repr assignment.")

s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")

print("PATCH: PASS")
print("63 original features -> 59 treated features")
print("Temporal summary: 59 × 8 = 472")
print("Removed indices: [30, 35, 39, 62]")
