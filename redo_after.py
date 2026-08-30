"""k1 range at decimation=200, then both swing figures, then the QC feature dump."""
from sdc.scoring.knob_range import sweep, analyse
from sdc.scoring.tune_marks import JANCA_FIXED
print("=== JANCA k1 RANGE (dec=200) ===", flush=True)
print("janca fixed:", JANCA_FIXED, flush=True)
analyse("janca", sweep("janca"))
from sdc.scoring import swing
for key in ("P1_ANT2", "P1_ANT145"):
    print(f"\n=== SWING {key} ===", flush=True)
    r, g = swing.report(key)
    swing.figure(key, r, g)
print("=== ALL DONE ===")
