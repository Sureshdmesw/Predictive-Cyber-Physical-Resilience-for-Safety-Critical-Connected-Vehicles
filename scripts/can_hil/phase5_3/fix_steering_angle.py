from pathlib import Path

p = Path(r"data\schemas\can_hil\ev\virtual_ev_complete.dbc")

s = p.read_text(encoding="utf-8")

old = 'SG_ SteeringAngle : 48|16@1- (0.1,-3276.8) [-3276.8|3276.7] "deg" ABS'
new = 'SG_ SteeringAngle : 48|16@1- (0.1,0) [-3276.8|3276.7] "deg" ABS'

if old not in s:
    print("Expected SteeringAngle definition not found.")
    raise SystemExit(1)

p.write_text(s.replace(old, new), encoding="utf-8")

print("SteeringAngle DBC definition corrected.")