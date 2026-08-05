"""
run_sim_suite.py
----------------
Drive compare_spikes.py over every simulated run the scorer needs: one point per SNR level at
the tuned operating point, plus a threshold sweep per detector at a single mid SNR.

WHY SUBPROCESSES rather than a loop inside compare_spikes.py
  Each Delphos call allocates a multi-GB RAM balloon (pin_free_ram_gb) and starts its own
  parpool, and the Barkmeier arm holds a MATLAB engine. Running each point in a fresh process
  means no engine or balloon state leaks between operating points -- which matters, because
  Delphos's internal tiling is driven by FREE SYSTEM RAM and is therefore the one thing in
  this comparison that is not reproducible if the memory state drifts.

COST
  Janca and Barkmeier points are seconds. DELPHOS IS ~5 MIN PER UNCACHED POINT. The default
  job list is 6 op points + 15 sweep points, of which 6 + 4 involve an uncached Delphos call
  (the 5th sweep value is the operating point itself and hits the cache), so budget ~50 min
  the first time and seconds thereafter. --dry-run prints the plan without running anything.

    .venv\\Scripts\\python.exe run_sim_suite.py --dry-run
    .venv\\Scripts\\python.exe run_sim_suite.py            # SNR curve only (6 jobs)
    .venv\\Scripts\\python.exe run_sim_suite.py --sweep    # + the 15 threshold-sweep points

NOTHING HERE IS PLOTTING COST. Every figure in score_sim_detectors.py draws in under a second
from stored npz files. The entire runtime is the Delphos CLI: ~4-5 min per uncached call,
where "uncached" means a (file, window, Delphos-parameter) combination not seen before. A
Janca or Barkmeier sweep point still runs Delphos, but at unchanged parameters, so it hits
cache and costs seconds.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import sim_data

HERE = Path(__file__).resolve().parent
SIM_RUNS = HERE / "sim_runs"

MID_SNR = 5.0     # the sweep runs at ONE level; 5 sits in the middle of SNR_LIST and is where
                  # the detectors are separable but not saturated
# (detector, param, values). The value that equals compare_spikes' default is deliberately
# included so the sweep curve passes through the operating point the SNR curves used.
SWEEPS = [("Janca", "k1", [2.6, 3.0, 3.4, 3.8, 4.2]),
          # Barkmeier's default TAMP=1200 was tuned on real data; injected peaks here only
          # reach ~290 uV at SNR 5, so the low end is where it can see anything at all.
          # MEASURED: this SATURATES -- 1200->0.35, 800->0.56, 600->0.62, 400->0.67, 200->0.67
          # recall. A sixfold threshold drop buys 0.32 and stops, so amplitude is NOT what is
          # rejecting the template. The two sweeps below test what is.
          ("Barkmeier", "TAMP", [200, 400, 600, 800, 1200]),
          # The shape criteria, swept as matched left/right pairs. LD/RD are half-wave
          # DURATION thresholds (ms) and LS/RS are half-wave SLOPE thresholds -- these gate on
          # morphology, which is the one thing a synthetic Gaussian is least likely to satisfy.
          ("Barkmeier", "LD+RD", [2, 4, 6, 8, 12]),
          ("Barkmeier", "LS+RS", [0.5, 1.0, 2.0, 3.0, 5.0]),
          ("Delphos", "Spk_thr", [30, 40, 50, 70, 100])]

DELPHOS_MIN = 5.0   # rough per-uncached-point cost, for the estimate only


def jobs(only=None, sweep_detector=None):
    out = [{"snr": s, "point": "op", "override": None} for s in sim_data.SNR_LIST]
    for det, param, values in SWEEPS:
        if sweep_detector and det.lower() != sweep_detector.lower():
            continue
        for v in values:
            out.append({"snr": MID_SNR, "point": f"{det}-{param}-{v:g}",
                        "override": {"detector": det, "param": param, "value": v}})
    if only:
        out = [j for j in out if (j["override"] is None) == (only == "op")]
    return out


def npz_for(job, cfg):
    return (SIM_RUNS / f"sim_{cfg['tag']}_{sim_data.cfg_hash(cfg)}"
                       f"_snr{job['snr']:g}_{job['point']}.npz")


def run(job, cfg, log=print):
    # SIM_FORCE flips compare_spikes.SIMULATE for the child only, so the file on disk keeps
    # SIMULATE=False and a bare `python compare_spikes.py` still runs the real recording.
    env = {**os.environ, "SIM_FORCE": "1", "SIM_SNR": f"{job['snr']:g}",
           "SIM_POINT": job["point"], "SIM_OVERRIDE": json.dumps(job["override"] or {})}
    t0 = time.time()
    p = subprocess.run([sys.executable, "compare_spikes.py"], cwd=str(HERE), env=env)
    dt = time.time() - t0
    log(f"    -> exit {p.returncode} in {dt / 60:.1f} min")
    return p.returncode == 0


def main(argv):
    dry = "--dry-run" in argv
    # Default is the SNR curve only. The threshold sweeps are opt-in because they are 15 of the
    # 21 jobs and the Delphos ones are the expensive half; the SNR curve alone answers "how
    # good is each detector", the sweep only adds "where is its knee".
    only = "op" if "--sweep" not in argv else None
    sweep_detector = None
    if "--sweep" in argv:
        i = argv.index("--sweep") + 1
        if i < len(argv) and not argv[i].startswith("-"):
            sweep_detector = argv[i]      # e.g. --sweep Barkmeier
    if "--only" in argv:
        only = argv[argv.index("--only") + 1]

    cfg = sim_data.default_cfg()
    js = jobs(only, sweep_detector)
    todo = [j for j in js if not npz_for(j, cfg).is_file()]
    n_delphos = len(todo)   # every point pays a Delphos call unless its parameters are cached

    print(f"--- sim suite: '{cfg['tag']}' hash {sim_data.cfg_hash(cfg)}, "
          f"{cfg['n_chan']} ch x {cfg['dur_sec']:g}s ---")
    print(f"    {len(js)} jobs, {len(todo)} not yet on disk")
    print(f"    worst-case Delphos time ~{n_delphos * DELPHOS_MIN:.0f} min "
          f"(cached parameter values return instantly)")
    for j in js:
        mark = " " if npz_for(j, cfg).is_file() else "*"
        ov = f"  {j['override']['detector']}.{j['override']['param']}=" \
             f"{j['override']['value']:g}" if j["override"] else "  (operating point)"
        print(f"  {mark} SNR {j['snr']:>4g}  {j['point']:<26}{ov}")
    print("    * = will run")

    if dry:
        return 0

    print(f"\n[1/2] building sim recordings (idempotent) ...")
    for snr in sorted({j["snr"] for j in todo}):
        sim_data.ensure_sim_edf(cfg, snr)

    print(f"\n[2/2] running {len(todo)} detection jobs ...")
    failed = []
    for i, j in enumerate(todo, 1):
        print(f"\n  [{i}/{len(todo)}] SNR {j['snr']:g} {j['point']}")
        if not run(j, cfg):
            failed.append(j["point"])

    print(f"\n--- done: {len(todo) - len(failed)}/{len(todo)} succeeded ---")
    if failed:
        print(f"    FAILED: {', '.join(failed)}")
    print(f"    next:  .venv\\Scripts\\python.exe score_sim_detectors.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
