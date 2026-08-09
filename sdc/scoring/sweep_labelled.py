"""
sdc.scoring.sweep_labelled
--------------------------
Set each detector's operating point against the 852 expert-marked IEDs, instead of by matching
counts to another detector.

    .venv\\Scripts\\python.exe -m sdc.scoring.sweep_labelled janca barkmeier
    .venv\\Scripts\\python.exe -m sdc.scoring.sweep_labelled delphos      # the slow one

WHY THIS EXISTS
  `BARK["TAMP"] = 1200` is documented in compare_spikes as "tuned to match Janca's count". That
  makes Janca the definition of correct, which is not a defensible way to set another
  detector's sensitivity. The BIDS marks are the first reference here that is not one of the
  detectors, so every operating point should come from them.

THE OBJECTIVE IS RECALL AT A MATCHED DETECTION BUDGET, NOT F1
  Precision against these marks is a LOWER BOUND: the experts marked notable discharges in a
  3-minute window, not every transient, so an unmatched detection is "not on their list", not
  "wrong". Maximising F1 would therefore penalise real-but-unmarked detections, and penalise
  the most sensitive detector hardest -- it would optimise towards the annotation's sparsity.
  Instead: fix detections per channel-minute equal across detectors, and ask who finds the most
  expert marks inside that budget. The unreliable denominator drops out, and the three become
  comparable at equal cost.

  This module SWEEPS and SCORES. It deliberately does not pick a winner: the choice of budget
  is a judgement, so the output is the whole recall-vs-budget curve.

COST
  Janca and Barkmeier are seconds per point per subject. DELPHOS IS 1-2 MIN, so a 4-point grid
  over 25 subjects is ~3 hours -- run it separately and coarsely first. ONLY=<detector> keeps
  each sweep to the one arm being moved; without it every point would also pay for the other
  two, including a fresh MATLAB engine start for Barkmeier.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from sdc.common.paths import ROOT, RUNS
from sdc.scoring.bids_events import subjects, load_subject, truth_per_channel
from sdc.scoring.score_sim_detectors import event_scores

BIDS_ROOT = Path(r"C:\Users\amoo0039\Documents\ieeg_ieds_bids_final\ieeg_ieds_bids")
SWEEPS = RUNS / "sweeps"
TOL_S = 0.050

# One dial each: the detector's own sensitivity control. A second pass on the shape parameters
# is only worth it if these curves plateau -- a knob that does nothing on a flat curve tells you
# nothing about the knob.
GRIDS = {
    # k1 runs to 6.0 so the curve REACHES Barkmeier's default budget (2.08 det/chan-min).
    # Without the top of this range the two can only be compared where Janca happens to
    # sit, which is 2.7x Barkmeier's budget -- and that comparison measures the operating
    # points rather than the detectors.
    # Extended DOWN to k1=1.2 to find the RECALL CEILING. The question a monotone curve
    # cannot answer from its middle: does recall approach 1.0 if you accept enough detections,
    # or does it plateau? A plateau means expert marks that the detector cannot reach at ANY
    # threshold -- a statement about what it can see, not about where it is set.
    # Delphos was extended DOWN for the same reason, and later: at the original floor
    # (Spk_thr=8) its curve was still climbing, so its top recall was the end of the grid rather
    # than a ceiling. report_labelled now refuses to call an unplateaued curve a ceiling.
    "janca":     ("k1",      [1.2, 1.6, 2.0, 2.3, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.5, 6.0]),
    # 25/50/75 are a NEGATIVE CONTROL, not an attempt to go higher -- see CEILING below.
    "barkmeier": ("TAMP",    [25, 50, 75, 100, 200, 300, 400, 600, 800, 1000, 1200, 1500]),
    "delphos":   ("Spk_thr", [2, 4, 8, 15, 30, 50, 80, 120]),    # coarse: see COST
}
LABEL = {"janca": "Janca", "barkmeier": "Barkmeier", "delphos": "Delphos"}

# TAMP CANNOT TAKE BARKMEIER ANY HIGHER, and the grid above shows why rather than asserting it.
# mDetectSpike_coeffs.m:116 picks the CANDIDATE peaks with `-mean|x| - STDCoeff*std|x|`; TAMP
# (:148) only filters that pool. So once TAMP stops binding, lowering it cannot add a single
# detection -- and it stops binding around 200: TAMP 300 -> 200 -> 100 moves the rate 9.23 ->
# 9.87 -> 10.01 and recall 0.6735 -> 0.6751 -> 0.6762. A 3x change in the knob buys 0.8% more
# output. 25/50/75 stay in the grid to make that flat visible, and that is the only reason they
# are there -- an asserted saturation and a plotted one are not the same evidence.
#
# STDCoeff is the knob that sets the pool, and it is the direct analogue of Janca's k1 -- also a
# std multiplier, which is how Janca reaches 142 det/chan-min at k1=1.2. Swept SEPARATELY rather
# than merged into GRIDS["barkmeier"]: the operating points (TAMP=890 at 3.5 det/chan-min) and
# every LOSO/paired number are defined on the TAMP axis, and silently changing which knob that
# axis refers to would redefine them. This is a ceiling experiment and nothing else reads it.
CEILING = {
    "barkmeier": ("std_coeff", [1.5, 2.0, 2.5, 3.0]),   # default 4.0
}


def run_point(det, param, value, subs):
    """One grid point across every subject. Skips work already on disk."""
    for sub in subs:
        out = SWEEPS / f"bids_{sub}_{det}_{param}{value:g}.npz"
        if out.is_file():
            continue
        env = {**os.environ, "BIDS_SUBJECT": sub, "ONLY": det,
               "DET_OVERRIDE": json.dumps({"detector": det, "param": param, "value": value})}
        p = subprocess.run([sys.executable, "-m", "sdc.detect.compare_spikes"],
                           cwd=str(ROOT), env=env, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"    [fail] {sub} {param}={value:g}\n{p.stdout[-400:]}{p.stderr[-400:]}")


def score_point(det, param, value, subs):
    """Pooled recall and detection budget at one grid point.

    Budget is detections per CHANNEL-MINUTE, which is what makes points comparable across
    subjects with 8 to 75 channels and 121 to 291 seconds."""
    tp = n_true = n_det = 0
    chan_min = 0.0
    per_sub = []
    for sub in subs:
        f = SWEEPS / f"bids_{sub}_{det}_{param}{value:g}.npz"
        if not f.is_file():
            continue
        z = np.load(f, allow_pickle=False)
        names = [str(s) for s in z["names"]]
        fs, secs = float(z["fs"]), float(z["seconds"])
        d = load_subject(BIDS_ROOT, sub)
        if list(d["channels"]) != names:
            raise SystemExit(f"{sub}: channel order differs from channels.tsv")
        truth = truth_per_channel(d["times"], d["chan_lists"], names)
        key = LABEL[det]
        idx, ch = z[f"{key}_idx"], z[f"{key}_chan"]
        got = [np.sort(idx[ch == c] / fs) for c in range(len(names))]
        sc = event_scores(truth, [np.zeros(t.size) for t in truth], got, secs, tol_s=TOL_S)
        tp += sc["tp"]; n_true += sc["n_true"]; n_det += sc["n_det"]
        chan_min += len(names) * secs / 60.0
        per_sub.append(sc["recall"])
    if not per_sub:
        return None
    return {"value": value, "recall": tp / max(n_true, 1),
            "precision": tp / max(n_det, 1), "budget": n_det / chan_min,
            "tp": tp, "n_true": n_true, "n_det": n_det,
            "per_sub": np.array(per_sub)}


def sweep(det, subs, grid=None, tag=""):
    """`grid` overrides GRIDS[det] -- that is how the CEILING axis is driven without redefining
    the axis the operating points are measured on. `tag` keeps its curve in its own file."""
    param, values = grid or GRIDS[det]
    print(f"\n=== {LABEL[det]}: {param} over {values} on {len(subs)} subjects ===")
    rows = []
    for v in values:
        run_point(det, param, v, subs)
        r = score_point(det, param, v, subs)
        if r:
            rows.append(r)
            print(f"  {param}={v:<6g} recall {r['recall']:.3f}   "
                  f"budget {r['budget']:6.2f} det/chan-min   "
                  f"precision {r['precision']:.3f} (lower bound)")
    np.save(SWEEPS / f"curve_{det}{tag}.npy", np.array(
        [(r["value"], r["recall"], r["budget"], r["precision"]) for r in rows]))
    return rows


if __name__ == "__main__":
    # "<det>:ceiling" runs that detector's CEILING axis instead of its operating-point axis.
    args = [a.lower() for a in sys.argv[1:]] or ["janca", "barkmeier"]
    subs = subjects(BIDS_ROOT)
    SWEEPS.mkdir(parents=True, exist_ok=True)
    for a in args:
        d, _, mode = a.partition(":")
        if d not in GRIDS:
            raise SystemExit(f"unknown detector {d!r}; use one of {sorted(GRIDS)}")
        if mode == "ceiling":
            if d not in CEILING:
                raise SystemExit(f"no ceiling axis for {d!r}; have {sorted(CEILING)}")
            sweep(d, subs, grid=CEILING[d], tag="_ceiling")
        elif mode:
            raise SystemExit(f"unknown mode {mode!r}; use '<det>' or '<det>:ceiling'")
        else:
            sweep(d, subs)
    print(f"\ncurves in {SWEEPS}. Score with the 10/15 split + LOSO once all three exist.")
