"""Re-run the rate sweeps invalidated by the config fixes (SCALE 100->70; sweep_rates.BARK
retyped constants; Janca band/decimation). Delphos is untouched -- its only knob is Spk_thr."""
from sdc.scoring.sweep_rates import run, BARK, JANCA_CFG
print("BARK:", BARK)
print("JANCA:", JANCA_CFG, flush=True)
for key in ("P1_ANT2", "P1_ANT145"):
    for det in ("janca", "barkmeier"):
        print(f"=== {key} / {det} ===", flush=True)
        run(key, det)
print("=== SWEEPS DONE ===")
