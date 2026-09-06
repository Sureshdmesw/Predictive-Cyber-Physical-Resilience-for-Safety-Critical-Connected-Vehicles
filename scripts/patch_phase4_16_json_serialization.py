from pathlib import Path

p = Path("scripts\crss_2024_phase4_16_generalization.py")
s = p.read_text(encoding="utf-8")

old = '''json.dumps(report, indent=2)'''

new = '''json.dumps(
    report,
    indent=2,
    default=lambda o: (
        bool(o) if type(o).__module__ == "numpy" and type(o).__name__ == "bool_"
        else int(o) if type(o).__module__ == "numpy" and type(o).__name__.startswith("int")
        else float(o) if type(o).__module__ == "numpy" and type(o).__name__.startswith("float")
        else o.tolist() if hasattr(o, "tolist") else str(o)
    )
)'''

if old not in s:
    raise RuntimeError("Could not locate the final json.dumps(report, indent=2) call.")

s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")

print("PATCH: PASS")
print("JSON serialization now handles NumPy scalar types.")
print("Model/evaluation logic unchanged.")
