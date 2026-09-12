from pathlib import Path
import re
import shutil
import py_compile
import subprocess
import sys

path = Path(r".\scripts\can_hil\phase5_16\connectivity_loss_recovery.py")
backup = Path(r".\scripts\can_hil\phase5_16\connectivity_loss_recovery.py.before_final_fix.bak")

print("\n=== PHASE 5.16 DEFINITIVE REPAIR ===")

# ------------------------------------------------------------
# Restore the clean backup created before the broken patch.
# ------------------------------------------------------------

if not backup.exists():
    print("ERROR: clean backup does not exist:")
    print(backup)
else:
    shutil.copy2(backup, path)
    print("PASS: clean backup restored")

    text = path.read_text(encoding="utf-8")

    # --------------------------------------------------------
    # FIX 1
    # Use the actual Phase 5.14 evidence window:
    #
    # 281-299 = CONNECTED
    # 300-305 = DISCONNECTED
    # 306-310 = RECOVERED
    # --------------------------------------------------------

    text, n1 = re.subn(
        r"DISCONNECT_START\s*=\s*300\s*\r?\n"
        r"DISCONNECT_END\s*=\s*330\s*\r?\n"
        r"EVENT_TIMESTEP\s*=\s*300",
        "DISCONNECT_START = 300\n"
        "DISCONNECT_END = 305\n"
        "EVENT_TIMESTEP = 300",
        text,
        count=1,
    )

    print(f"CONNECTIVITY CONSTANTS REPLACED: {n1}")

    # --------------------------------------------------------
    # FIX 2
    # Replace the entire recovery expression with the original
    # simple and safe definition:
    #
    # recovered = timestep > DISCONNECT_END
    # --------------------------------------------------------

    text, n2 = re.subn(
        r"(?ms)"
        r"    recovered\s*=\s*scenario_df\s*\["
        r".*?"
        r"\n\s*\]\s*\n"
        r"\s*event_rows\s*=",
        '    recovered = scenario_df[\n'
        '        scenario_df["timestep_index"] > DISCONNECT_END\n'
        '    ]\n\n'
        '    event_rows =',
        text,
        count=1,
    )

    print(f"RECOVERY EXPRESSION REPLACED: {n2}")

    # --------------------------------------------------------
    # FIX 3
    # Correct 31 -> 30 timestep validation.
    # --------------------------------------------------------

    text, n3 = re.subn(
        r'f"\{scenario_id\}_timesteps_31"\s*,\s*\n'
        r'\s*result\["timesteps"\]\s*==\s*31\s*,',
        'f"{scenario_id}_timesteps_30",\n'
        '            result["timesteps"] == 30,',
        text,
        count=1,
    )

    print(f"TIMESTEP VALIDATION REPLACED: {n3}")

    # --------------------------------------------------------
    # FIX 4
    # Remove every accidental RECOVERY_START reference.
    # We do NOT need this variable.
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*RECOVERY_START\s*=\s*306\s*\r?\n",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r'^\s*"recovery_start"\s*:\s*RECOVERY_START\s*,?\s*\r?\n',
        "",
        text,
        flags=re.MULTILINE,
    )

    text = text.replace("RECOVERY_START", "")

    # --------------------------------------------------------
    # FIX 5
    # Remove any literal PowerShell escape corruption.
    # --------------------------------------------------------

    text = text.replace("`r`n", "\n")

    # --------------------------------------------------------
    # Write repaired source.
    # --------------------------------------------------------

    path.write_text(text, encoding="utf-8")

    print("PASS: repaired source written")

    # ========================================================
    # STATIC VALIDATION
    # ========================================================

    source = path.read_text(encoding="utf-8")

    required = [
        "DISCONNECT_START = 300",
        "DISCONNECT_END = 305",
        "EVENT_TIMESTEP = 300",
        'result["timesteps"] == 30',
        'scenario_df["timestep_index"] > DISCONNECT_END',
    ]

    print("\n=== STATIC VALIDATION ===")

    for item in required:
        if item in source:
            print("PASS:", item)
        else:
            print("FAIL:", item)

    forbidden = [
        "DISCONNECT_END = 330",
        "timesteps_31",
        "RECOVERY_START",
        "`r`n",
    ]

    for item in forbidden:
        if item not in source:
            print("PASS: absent:", repr(item))
        else:
            print("FAIL: still present:", repr(item))

    # ========================================================
    # COMPILE VALIDATION
    # ========================================================

    print("\n=== PYTHON COMPILE ===")

    try:
        py_compile.compile(
            str(path),
            doraise=True,
        )
        print("PASS: Python compilation")
    except Exception as exc:
        print("FAIL: Python compilation")
        print(exc)

    # ========================================================
    # IMPORT / NAME VALIDATION
    #
    # Import the module without executing main().
    # This catches undefined module-level names.
    # ========================================================

    print("\n=== MODULE IMPORT VALIDATION ===")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "phase5_16_test",
            path,
        )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        print("PASS: module import")
        print("PASS: no module-level NameError")
    except Exception as exc:
        print("FAIL: module import")
        print(type(exc).__name__ + ":", exc)

    # ========================================================
    # PRE-FLIGHT DATA VALIDATION
    # ========================================================

    print("\n=== DATA PREFLIGHT ===")

    try:
        import pandas as pd

        input_csv = Path(module.INPUT_CSV)
        df = pd.read_csv(input_csv)

        print("ROWS:", len(df))
        print("SCENARIOS:", df["scenario_id"].nunique())
        print("MIN TIMESTEP:", df["timestep_index"].min())
        print("MAX TIMESTEP:", df["timestep_index"].max())
        print("UNIQUE TIMESTEPS:", df["timestep_index"].nunique())

        expected = (
            len(df) == 4650
            and df["scenario_id"].nunique() == 10
            and df["timestep_index"].nunique() == 30
            and df["timestep_index"].min() == 281
            and df["timestep_index"].max() == 310
        )

        if expected:
            print("PASS: input structure")
        else:
            print("FAIL: input structure")

    except Exception as exc:
        print("FAIL: data preflight")
        print(type(exc).__name__ + ":", exc)

    # ========================================================
    # FUNCTION-LEVEL VALIDATION
    #
    # Directly evaluate C01 before running main().
    # This catches recovery logic errors without touching output.
    # ========================================================

    print("\n=== C01 FUNCTION PREFLIGHT ===")

    try:
        result = module.evaluate_scenario(
            "C01_SPEED_SENSOR_SPOOFING",
            df,
        )

        print("records =", result["records"])
        print("timesteps =", result["timesteps"])
        print("event_records =", result["event_records"])
        print("disconnect_records =", result["disconnect_records"])
        print("recovery_records =", result["recovery_records"])

        checks = {
            "records == 465": result["records"] == 465,
            "timesteps == 30": result["timesteps"] == 30,
            "event_records > 0": result["event_records"] > 0,
            "disconnect_records > 0": result["disconnect_records"] > 0,
            "recovery_records > 0": result["recovery_records"] > 0,
        }

        for name, ok in checks.items():
            print(("PASS: " if ok else "FAIL: ") + name)

    except Exception as exc:
        print("FAIL: C01 function preflight")
        print(type(exc).__name__ + ":", exc)

    # ========================================================
    # ACTUAL PHASE RUN
    # ========================================================

    print("\n=== RUNNING PHASE 5.16 ===")

    completed = subprocess.run(
        [sys.executable, str(path)],
        text=True,
    )

    print("\n=== PHASE 5.16 PROCESS RETURN CODE ===")
    print(completed.returncode)

    if completed.returncode == 0:
        print("PASS: Phase 5.16 process completed")
    else:
        print("WARNING: Phase 5.16 process returned non-zero")
        print("PowerShell session remains open.")
