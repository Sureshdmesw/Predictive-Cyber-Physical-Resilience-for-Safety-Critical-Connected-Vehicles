from pathlib import Path

p = Path(r"data\schemas\can_hil\ev\virtual_ev_complete.dbc")

s = p.read_text(encoding="utf-8")

replacements = {
    'SG_ LongitudinalAcceleration : 0|16@1- (0.01,-327.68) [-327.68|327.67] "g" RCM':
    'SG_ LongitudinalAcceleration : 0|16@1- (0.01,0) [-327.68|327.67] "g" RCM',

    'SG_ LateralAcceleration : 16|16@1- (0.01,-327.68) [-327.68|327.67] "g" RCM':
    'SG_ LateralAcceleration : 16|16@1- (0.01,0) [-327.68|327.67] "g" RCM',

    'SG_ YawRate : 32|16@1- (0.1,-3276.8) [-3276.8|3276.7] "deg/s" RCM':
    'SG_ YawRate : 32|16@1- (0.1,0) [-3276.8|3276.7] "deg/s" RCM',
}

for old, new in replacements.items():
    if old not in s:
        print("Definition not found:")
        print(old)
        raise SystemExit(1)

    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")

print("RESCUE_DYNAMICS signed-signal definitions corrected.")
print("LongitudinalAcceleration: offset 0")
print("LateralAcceleration: offset 0")
print("YawRate: offset 0")