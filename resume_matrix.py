"""Emit the matrix runs that are still MISSING, so a restart does not repeat completed work.

The first attempt lost every kStim condition on a STIM recording: the relative-kStim baseline
dumps (qc_features/*_f*.npz) held 226 derived channel names against this run's 164 clinical
ones, and compare_spikes refuses to align a per-channel threshold across a montage change.
Baselines were unaffected -- kStim is dropped without a trial -- so roughly half the matrix
completed. Re-running all 46 would waste about two hours.
"""
import sys
from pathlib import Path

import numpy as np

N_CLINICAL = 164        # P1's clinical montage; a file with any other count is from before it

RUNS = Path(__file__).resolve().parent / "runs"
RECS = ["P1_stim", "P1_pre", "P1_ANT2_stim", "P1_ANT2_pre"]
TIER1 = ["mnebads10", "mnebads15", "mnebads75", "mnebads150", "k450g0"]
TIER2 = ["none", "k150g150", "k150g1000", "k450g1000", "k150g0", "e1"]
TIER3 = [("P1_ANT2_stim", "e1t10"), ("P1_ANT2_pre", "e1t10")]


def done(rec, p):
    """Complete means present AND on the clinical montage. Existence alone is not enough: 11
    files from the derived-montage matrix are still on disk under the very same names, and
    skipping those would leave 226-channel rows in a 164-channel comparison."""
    f = RUNS / f"{rec}_qc{p}.npz"
    if not f.is_file():
        return False
    try:
        return len(np.load(f, allow_pickle=False)["names"]) == N_CLINICAL
    except Exception:
        return False


def missing():
    out = []
    for rec in RECS:
        for p in TIER1:
            if not done(rec, p):
                out.append((rec, p, "0"))
    for rec in RECS:
        for p in TIER2:
            if not done(rec, p):
                out.append((rec, p, "1"))
    for rec, p in TIER3:
        if not done(rec, p):
            out.append((rec, p, "1"))
    return out


if __name__ == "__main__":
    rows = missing()
    if "--count" in sys.argv:
        done = 4 * (len(TIER1) + len(TIER2)) + len(TIER3) - len(rows)
        print(f"{len(rows)} missing, {done} already complete")
        for rec, p, d in rows:
            print(f"  {rec:<16}{p:<12}{'JBD' if d == '1' else 'JB'}")
    else:
        for rec, p, d in rows:
            print(f"{rec} {p} {d}")
