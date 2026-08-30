"""Add Delphos to the P1 artefact-matrix conditions that currently have only Janca+Barkmeier.

    .venv\\Scripts\\python.exe tools\\run_fill_delphos.py --list   # what would run, and why
    .venv\\Scripts\\python.exe tools\\run_fill_delphos.py          # run it

WHY THIS EXISTS. Between-detector spread is max/min over the detectors PRESENT, so a row with
two detectors and a row with three are not the same statistic and must never be read down the
same column. On the current matrix `mne p10` and `k450/grad OFF` are two-detector rows sitting
next to three-detector ones, and `mne p10` looks like the tightest condition at 145 Hz (1.17)
purely because Delphos -- the detector that disagrees most -- is missing from it. Filling these
in is not extra evidence, it is what makes the existing rows comparable at all.

This project has already been burned once by exactly this: a Janca k1xk3 grid was compared
against a 1-D Barkmeier grid and the admitted fractions were not comparable, and the conclusion
drawn from it had to be withdrawn.

CHEAP BY CONSTRUCTION. run_windows caches the per-window Janca/Barkmeier results, so a re-run
that adds Delphos re-uses them and pays only for the single whole-file Delphos call (~15 min).
See the RUN_DELPHOS note at the top of sdc/detect/run_windows.py.

RESUMABLE: a condition whose npz already contains `Delphos_idx` is skipped, so this can be
stopped and restarted freely.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# (recording, profile). Paired stim+pre throughout: every number in the matrix is a
# stim/baseline ratio, so a stim run whose baseline lacks Delphos still yields no Delphos row.
JOBS = [
    # --- 145 Hz. The abstract figure shows none / mne p10 / grad OFF / k450 g1000. `none` and
    # `k450g1000` are already complete, so mne p10 is the only gap that has to be filled.
    ("P1_stim", "mnebads10"),
    ("P1_pre", "mnebads10"),
    # --- 2 Hz. Same: mne p10 only. `none`, `dynr` and `dynrg1000` are already complete.
    ("P1_ANT2_stim", "mnebads10"),
    ("P1_ANT2_pre", "mnebads10"),
    # --- The grad-OFF row at the kStim the rest of the figure uses. `k150g0` is within 0.5% of
    # this ungated and identical to 3 decimals gated, so it stood in while the clock was tight;
    # running the real pair removes a substitution that would otherwise have to be explained in
    # the caption of a figure whose whole point is that conditions are compared like for like.
    ("P1_stim", "k450g0"),
    ("P1_pre", "k450g0"),
    # Restored: the 2 Hz stack shows masking WITH and WITHOUT the gradient, so that the
    # gradient's effect can be read against the 145 Hz panel where it does almost nothing
    # (1.07 -> 1.05) while at 2 Hz it is what reaches the pulse artefact.
    ("P1_ANT2_stim", "k450g0"),
    ("P1_ANT2_pre", "k450g0"),
]


def state(rec, profile):
    """'missing' | 'no-delphos' | 'done' -- read from the npz, never assumed from the name."""
    p = ROOT / "runs" / f"{rec}_qc{profile}.npz"
    if not p.is_file():
        return "missing"
    with np.load(p, allow_pickle=False) as z:
        return "done" if "Delphos_idx" in z.files else "no-delphos"


def main():
    logs = ROOT / "logs" / "fill_delphos"
    rows = [(r, p, state(r, p)) for r, p in JOBS]
    pending = [(r, p) for r, p, s in rows if s != "done"]

    print(f"{len(rows)} conditions, {len(pending)} pending "
          f"(~{15 * len(pending)} min at ~15 min/Delphos call)\n")
    for r, p, s in rows:
        print(f"  {r:<16}{p:<12}{s}")
    if "--list" in sys.argv or not pending:
        return

    logs.mkdir(parents=True, exist_ok=True)
    print(f"\nlogs -> {logs}\n")
    ok, failed, t_all = [], [], time.time()
    for i, (rec, prof) in enumerate(pending, 1):
        t0 = time.time()
        print(f"[{i}/{len(pending)}] {rec} {prof} ...", flush=True)
        log = logs / f"{rec}_qc{prof}.log"
        with open(log, "w", encoding="utf-8") as fh:
            res = subprocess.run(
                [sys.executable, "-m", "sdc.detect.run_windows"], cwd=str(ROOT),
                stdout=fh, stderr=subprocess.STDOUT,
                env={**os.environ, "RECORDING": rec, "QC_PROFILE": prof, "RUN_DELPHOS": "1"})
        mins = (time.time() - t0) / 60
        if res.returncode == 0 and state(rec, prof) == "done":
            ok.append((rec, prof))
            print(f"      done in {mins:.0f} min", flush=True)
        else:
            failed.append((rec, prof))
            print(f"      FAILED after {mins:.0f} min (exit {res.returncode}) -- {log}",
                  flush=True)

    print(f"\n{'=' * 60}\n{len(ok)} ok, {len(failed)} failed, "
          f"{(time.time() - t_all) / 60:.0f} min total")
    for rec, prof in failed:
        print(f"  FAILED  {rec} {prof}   {logs / f'{rec}_qc{prof}.log'}")


if __name__ == "__main__":
    main()
