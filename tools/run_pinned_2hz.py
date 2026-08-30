"""Re-detect the 2 Hz stack with Barkmeier's amplitude normaliser PINNED to one value.

    .venv\\Scripts\\python.exe tools\\run_pinned_2hz.py --list
    .venv\\Scripts\\python.exe tools\\run_pinned_2hz.py

WHAT THIS FIXES. mDetectSpike renormalises every 1-minute block to its own mean amplitude and
applies TAMP/LS/RS AFTER that, so the effective threshold tracks the recording. Barkmeier's
MEASUREMENT signal is band-limited to 1-35 Hz, and a 2 Hz pulse train's fundamental and low
harmonics sit INSIDE that band -- so continuous 2 Hz stimulation raises the normaliser from
16.8 to 25.4 uV and the detector runs a 51% higher threshold during stimulation than during its
own baseline. It then reports a spike-rate decrease that is partly just the threshold moving.
At 145 Hz the artefact is above the 35 Hz low-pass, is filtered out before the median is taken,
and the same quantity moves 1.3%: this is a low-frequency problem, not a stimulation problem.

WHY NOT BARK_SCALE. That knob back-solves one constant SCALE from a Python reconstruction of the
denominator and measurably OVERSHOOTS -- 0.516 -> 0.977, past Janca and Delphos. Not because the
reconstruction is wrong (it agrees with MATLAB to 4 s.f., 25.53 vs 25.53) but because ONE
CONSTANT CANNOT UNDO A PER-BLOCK NORMALISER: it corrects a second time what the per-block
renormalisation already partly corrected. Pinning replaces the normaliser outright instead.

ONE VALUE FOR EVERY RUN, STIM AND BASELINE ALIKE. Pinning only the stim file would leave the
baseline adapting per block and the stim file not -- a different comparison, not a corrected
one. The value is the median block denominator of the stim-free baseline, measured by MATLAB
itself (mDetectSpike's 4th output) rather than reconstructed.

DELPHOS IS NOT RE-RUN. `fixed_denom` reaches detect_barkmeier only; Janca and Delphos see an
identical signal and identical masking, so their detections are unchanged by construction. These
runs go out with RUN_DELPHOS=0 and the figure reads Barkmeier from the pinned run and Janca /
Delphos from the unpinned one. Janca IS re-run here, and its count matching the unpinned run is
the check that "unchanged by construction" is actually true -- see --list output after a run.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROFILES = ["none", "mnebads10", "k450g0", "k450g1000", "k150g150", "k450g150"]
STIM, BASE = "P1_ANT2_stim", "P1_ANT2_pre"
PROBE_TAG = "_denomprobe"          # keeps the probe out of the canonical baseline npz


def _run(rec, profile, env_extra, tag=""):
    logs = ROOT / "logs" / "pinned_2hz"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{rec}{tag}_qc{profile}.log"
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(
            [sys.executable, "-m", "sdc.detect.run_windows"], cwd=str(ROOT),
            stdout=fh, stderr=subprocess.STDOUT,
            env={**os.environ, "RECORDING": rec, "QC_PROFILE": profile,
                 "RUN_DELPHOS": "0", **env_extra})
    return p.returncode, log


def probe_denom():
    """MATLAB's own median block denominator for the stim-free baseline.

    Written under a RUN_TAG so it cannot overwrite the canonical baseline npz -- which carries
    Delphos, and a RUN_DELPHOS=0 re-run would silently drop it.
    """
    stem = f"{BASE}{PROBE_TAG}_qcnone"
    f = ROOT / "runs" / f"{stem}.npz"
    if not f.is_file():
        print(f"[probe] measuring the baseline normaliser -> {stem}.npz", flush=True)
        rc, log = _run(BASE, "none", {"RUN_TAG": PROBE_TAG}, tag=PROBE_TAG)
        if rc != 0 or not f.is_file():
            raise SystemExit(f"probe run failed (exit {rc}) -- see {log}")
    with np.load(f, allow_pickle=False) as z:
        if "bark_block_denom" not in z.files or not z["bark_block_denom"].size:
            raise SystemExit(
                f"{stem}.npz carries no bark_block_denom. It predates mDetectSpike's 4th "
                f"output -- delete it and re-run so the probe is measured, not assumed.")
        bd = np.asarray(z["bark_block_denom"], float)
    d = float(np.median(bd))
    print(f"[probe] {bd.size} blocks, median {d:.4f} uV "
          f"(range {bd.min():.2f}-{bd.max():.2f})", flush=True)
    return d


def jobs(d):
    suf = f"_bd{d:g}"
    return [(rec, p, suf) for rec in (BASE, STIM) for p in PROFILES]


def main():
    listing = "--list" in sys.argv
    d = probe_denom() if not listing else None
    if listing and (ROOT / "runs" / f"{BASE}{PROBE_TAG}_qcnone.npz").is_file():
        d = probe_denom()
    if d is None:
        print("no probe yet -- run without --list to measure it first")
        return

    todo = jobs(d)
    pending = [(r, p, s) for r, p, s in todo
               if not (ROOT / "runs" / f"{r}{s}_qc{p}.npz").is_file()]
    print(f"\nBARK_DENOM={d:.4f}  ->  runs/<rec>_bd{d:g}_qc<profile>.npz")
    print(f"{len(todo)} runs, {len(pending)} pending\n")
    for r, p, s in todo:
        done = (ROOT / "runs" / f"{r}{s}_qc{p}.npz").is_file()
        print(f"  {r:<16}{p:<12}{'[done]' if done else ''}")
    if listing or not pending:
        _check_janca(d)
        return

    t0 = time.time()
    for i, (rec, prof, _s) in enumerate(pending, 1):
        t = time.time()
        print(f"[{i}/{len(pending)}] {rec} {prof} pinned to {d:.3f} ...", flush=True)
        rc, log = _run(rec, prof, {"BARK_DENOM": f"{d:.6f}"})
        print(f"      {'done' if rc == 0 else f'FAILED (exit {rc}) -- {log}'} "
              f"in {(time.time() - t) / 60:.0f} min", flush=True)
    print(f"\n{(time.time() - t0) / 60:.0f} min total")
    _check_janca(d)


def _check_janca(d):
    """Janca must be IDENTICAL pinned vs unpinned -- fixed_denom reaches Barkmeier only.

    Printed rather than asserted: a mismatch would mean the pinned run differs from the unpinned
    one in something OTHER than the Barkmeier normaliser (a stale cache, a different mask), which
    is exactly the failure this project has hit before and which no amount of Barkmeier
    arithmetic would reveal.
    """
    suf = f"_bd{d:g}"
    print(f"\n{'run':<34}{'Janca pinned':>14}{'unpinned':>10}{'':>4}{'Barkmeier':>10}{'was':>9}")
    for rec in (BASE, STIM):
        for p in PROFILES:
            a = ROOT / "runs" / f"{rec}{suf}_qc{p}.npz"
            b = ROOT / "runs" / f"{rec}_qc{p}.npz"
            if not (a.is_file() and b.is_file()):
                continue
            with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
                ja, jb = za["Janca_idx"].size, zb["Janca_idx"].size
                ba, bb = za["Barkmeier_idx"].size, zb["Barkmeier_idx"].size
            flag = "" if ja == jb else "  <-- MISMATCH"
            print(f"  {rec + ' ' + p:<32}{ja:>14}{jb:>10}{flag:>4}{ba:>10}{bb:>9}")


if __name__ == "__main__":
    main()
