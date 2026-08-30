"""Run all three detectors over the AES cohort at one QC profile.

    set QC_PROFILE=final
    .venv\\Scripts\\python.exe tools\\run_cohort.py            # run it
    .venv\\Scripts\\python.exe tools\\run_cohort.py --list     # what would run, and why

RESUMABLE. A recording whose merged `runs/<rec>_qc<profile>.npz` already exists is skipped, so
the batch can be stopped and restarted without losing work. The per-window npz files underneath
are cached the same way, so even a recording interrupted half way resumes at the window it
reached rather than at the start.

WHICH RECORDINGS
  HF-int  (145 Hz, intermittent)  -- the stim file only. The OFF periods between stim blocks are
                                     the comparison, so the pre file is not needed for the
                                     primary contrast.
  LF-cont (2 or 7 Hz, continuous) -- the stim file AND its pre file. There is no OFF period
                                     inside the recording, so the pre file IS the comparison.

ORDER is smallest file first. A configuration error then surfaces in minutes on a 0.04 GB file
rather than after hours on a 5.9 GB one, and the cheap recordings are banked before anything
long is attempted.

A recording whose EDF is not on the local disk is reported and skipped, not guessed at. A trial
whose STIM file is missing is dropped whole -- running its pre file alone produces a baseline
with nothing to compare against.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sdc.detect.cohort import COHORT, arm            # noqa: E402
from sdc.detect.recordings import edf_path           # noqa: E402

PROFILE = os.environ.get("QC_PROFILE", "final")
SUFFIX = "" if PROFILE == "prod" else f"_qc{PROFILE}"
LOGS = ROOT / "logs" / f"cohort{SUFFIX}"


def jobs():
    """(size_gb, recording, arm, edf) smallest first, plus the skips and why."""
    out, skip = [], []
    for stem, c in COHORT.items():
        a = arm(stem)
        try:
            stim_p, _ = edf_path(f"{stem}_stim")
        except SystemExit as e:
            skip.append((stem, str(e)))
            continue
        if not Path(stim_p).is_file():
            # Drop the whole trial, pre file included -- see the module docstring.
            skip.append((stem, f"stim file not local: {Path(stim_p).name}"))
            continue
        recs = [f"{stem}_stim"] + ([f"{stem}_pre"] if a == "LF-cont" else [])
        for r in recs:
            p, _ = edf_path(r)
            if not Path(p).is_file():
                skip.append((r, f"not local: {Path(p).name}"))
                continue
            out.append((Path(p).stat().st_size / 1e9, r, a, Path(p).name))
    return sorted(out), skip


def done(rec):
    return (ROOT / "runs" / f"{rec}{SUFFIX}.npz").is_file()


def main():
    todo, skip = jobs()
    listing = "--list" in sys.argv
    pending = [j for j in todo if not done(j[1])]

    print(f"QC_PROFILE={PROFILE}  ->  runs/<rec>{SUFFIX}.npz")
    print(f"{len(todo)} recordings, {len(pending)} pending, "
          f"{sum(g for g, *_ in pending):.1f} GB to process\n")
    for gb, rec, a, fn in todo:
        print(f"  {rec:<20}{a:<9}{gb:6.2f} GB  {fn:<16}{'[done]' if done(rec) else ''}")
    if skip:
        print("\nskipped:")
        for what, why in skip:
            print(f"  {what:<20}{why}")
    if listing or not pending:
        return

    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"\nlogs -> {LOGS}\n")
    ok, failed, t_all = [], [], time.time()
    for i, (gb, rec, a, fn) in enumerate(pending, 1):
        t0 = time.time()
        print(f"[{i}/{len(pending)}] {rec}  ({gb:.2f} GB, {a})  ...", flush=True)
        log = LOGS / f"{rec}.log"
        with open(log, "w", encoding="utf-8") as fh:
            p = subprocess.run([sys.executable, "-m", "sdc.detect.run_windows"],
                               cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                               env={**os.environ, "RECORDING": rec, "QC_PROFILE": PROFILE})
        mins = (time.time() - t0) / 60
        if p.returncode == 0 and done(rec):
            ok.append(rec)
            print(f"      done in {mins:.0f} min", flush=True)
        else:
            failed.append(rec)
            print(f"      FAILED after {mins:.0f} min (exit {p.returncode}) -- {log}", flush=True)

    print(f"\n{'=' * 60}\n{len(ok)} ok, {len(failed)} failed, "
          f"{(time.time() - t_all) / 3600:.1f} h total")
    for r in failed:
        print(f"  FAILED  {r}   {LOGS / f'{r}.log'}")


if __name__ == "__main__":
    main()
