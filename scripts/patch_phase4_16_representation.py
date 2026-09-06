from pathlib import Path

p = Path("scripts/crss_2024_phase4_16_generalization.py")
s = p.read_text(encoding="utf-8")

old = '''def temporal_summary(X):
    X = np.asarray(X, dtype=np.float32)

    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    minimum = np.min(X, axis=1)
    maximum = np.max(X, axis=1)
    last = X[:, -1, :]
    delta = X[:, -1, :] - X[:, 0, :]

    # Two temporal statistics.
    median = np.median(X, axis=1)
    abs_delta = np.mean(np.abs(np.diff(X, axis=1)), axis=1)

    return np.concatenate(
        [
            mean,
            std,
            minimum,
            maximum,
            last,
            delta,
            median,
            abs_delta,
        ],
        axis=1,
    ).astype(np.float32)
'''

new = '''def temporal_summary(X):
    """
    EXACT Phase 4.9 representation.

    59 treated features × 8 statistics = 472 dimensions.

    Statistics, in the exact Phase 4.9 order:
      1. last
      2. mean
      3. std
      4. minimum
      5. maximum
      6. last_minus_first
      7. max_absolute_step_change
      8. last_step_change
    """
    X = np.asarray(X, dtype=np.float32)

    last = X[:, -1, :]
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    minimum = np.min(X, axis=1)
    maximum = np.max(X, axis=1)
    last_minus_first = X[:, -1, :] - X[:, 0, :]

    step_change = np.diff(X, axis=1)

    max_absolute_step_change = np.max(
        np.abs(step_change),
        axis=1,
    )

    last_step_change = X[:, -1, :] - X[:, -2, :]

    return np.concatenate(
        [
            last,
            mean,
            std,
            minimum,
            maximum,
            last_minus_first,
            max_absolute_step_change,
            last_step_change,
        ],
        axis=1,
    ).astype(np.float32)
'''

if old not in s:
    raise RuntimeError(
        "Expected incorrect temporal_summary block was not found. "
        "No file modification made."
    )

p.write_text(s.replace(old, new), encoding="utf-8")

print("PATCH: PASS")
print("Updated:", p)
print("Representation: exact Phase 4.9 8-statistic order")
print("Expected dimensions: 59 × 8 = 472")
