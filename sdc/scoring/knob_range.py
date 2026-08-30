"""
sdc.scoring.knob_range
----------------------
Leave-one-patient-out over ONE knob, to produce an admissible RANGE rather than a point.

    .venv\\Scripts\\python.exe -m sdc.scoring.knob_range barkmeier
    .venv\\Scripts\\python.exe -m sdc.scoring.knob_range janca

One knob each, because each detector has exactly one that is meant to be tuned:
  janca      k1  -- "Coefficient value k1 was optimized using gold standard spikes" (Janca et
                 al.). Everything else in that paper is a fixed method choice, not a knob, which
                 is why winsize/buffering/polyspike_union measured as inert or bit-identical.
                 k3 is not in the paper at all (its threshold is mode+median); k3=0 IS the
                 published formula.
  barkmeier  TAMP -- the only threshold left free once SCALE is set to the paper's 70 uV. The
                 shape gates are dominated by it at matched recall (0/40 comparisons won).

Everything else is held at the values the pipeline actually ships, EXCEPT Janca's upper band
edge, which is 50 Hz rather than the paper's 60: 50 Hz mains would sit inside a 60 Hz edge on
this hardware. That is a data property, not a tuning choice, and costs ~0.009 marked macro F1.
Decimation stays at the paper's 200 Hz -- see tune_marks.JANCA_FIXED for why skipping it is not
worth 5x the compute.

THE ADMISSIBLE RULE: PAIRED DIFFERENCE, FIXED MARGIN
  For each setting take the per-patient difference d against the best setting, and admit it if
  mean(d) >= -ADMIT_MARGIN. Paired, because the between-patient LEVEL (P1 sits at 0.15-0.23,
  P5 at 0.29-0.42) is common to every setting and cancels in the difference; an unpaired rule
  using the SE of the level admitted 11/11 Barkmeier settings and so said nothing at all.

  IT IS A PRACTICAL MARGIN, NOT A SIGNIFICANCE TEST, AND THAT IS DELIBERATE. Three candidate
  rules were scored on the two real sweeps:
                                        k1              TAMP
    mean + 1*SE  >= 0            3.25 only [1/9]   900-1400 [5/11]
    mean + t2*SE >= 0 (a=0.05)   2.5-6     [6/9]   400-1600 [11/11]
    mean >= -0.01                3.25-3.65 [2/9]   900-1200 [4/11]
  The middle row is the honest significance test at n=3 and it admits everything -- the marked
  data genuinely cannot separate these settings, so any narrower range is a judgement rather
  than a statistic and should say so. The first row is bad in both directions at once: too
  narrow on k1 (it drops 3.65, only 0.010 behind) and too loose on TAMP (it admits 1400 at mean
  d -0.036 while rejecting 400 at -0.032, purely because 1400's SE is 0.045 -- it rewards
  noise). Adding the t-test to the margin changed nothing on either knob, so it is left out
  rather than kept as decoration.

  ADMIT_MARGIN = 0.01 is the size of the only held-out gain this project could demonstrate
  (+0.0098, Janca k1 at dec=200). Settings within it are ones there is no evidence to separate;
  beyond it, we would have noticed. SE(d) is still REPORTED beside every row so the uncertainty
  is visible -- it just does not gate.
"""
import sys

import numpy as np

from seeg import detect_spikes as bark
from seeg.spikes import DET_THRESHOLDS, STD_COEFF, FILTER_SPEC, TROUGH_SEARCH_MS, SCALE
from sdc.detect.janca_detect_spikes import detect_spikes as janca
from sdc.scoring.overnight import _score
from sdc.scoring.score_marks import done_blocks, RATERS, TOL
from sdc.scoring.tune_marks import JANCA_FIXED, BLOCK_REC, load_block


GRIDS = {"janca": (2.5, 3.0, 3.25, 3.65, 4.0, 4.5, 5.0, 5.5, 6.0),
         "barkmeier": (400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0,
                       1400.0, 1600.0)}
ADMIT_MARGIN = 0.01      # marked macro F1; see the docstring for why this and not an SE rule
KNOB = {"janca": "k1", "barkmeier": "TAMP"}
BASELINES = {"janca": (3.65,), "barkmeier": (800.0, 1000.0)}


def sweep(det, grid=None):
    grid = grid or GRIDS[det]
    data = [load_block(b) for b in [b for b in done_blocks(RATERS[0]) if b in BLOCK_REC]]
    rows = []
    for v in grid:
        per = {}
        for d in data:
            if det == "janca":
                out, _a, _b = janca(d["x"], d["fs"], k1=v, **JANCA_FIXED)
                ch = np.asarray(out["chan"], int)
                t = np.asarray(out["pos"], float)
                dd = {c: np.sort(t[ch == i]) for i, c in enumerate(d["chans"])}
            else:
                th = list(DET_THRESHOLDS)
                th[2] = v
                bark(d["rec"], None, post_mask_spikes=False, fill_bad_samples=False,
                     det_thresholds=th, std_coeff=STD_COEFF,
                     trough_search_ms=TROUGH_SEARCH_MS, filter_spec=FILTER_SPEC)
                dd = {c: np.sort(np.asarray(d["rec"]["info"]["DetectedSpikes"][i], float)
                                 / d["fs"]) for i, c in enumerate(d["chans"])}
            per[d["subj"]] = _score(dd, d, TOL)
        tp = sum(x["tp"] for x in per.values())
        fp = sum(x["fp"] for x in per.values())
        fn = sum(x["fn"] for x in per.values())
        rows.append({"v": v, "per": {k: x["marked_macro_f1"] for k, x in per.items()},
                     "f1": float(np.mean([x["marked_macro_f1"] for x in per.values()])),
                     "empty_fp": float(np.nanmean([x["empty_fp"] for x in per.values()])),
                     "prec": tp / max(tp + fp, 1), "rec": tp / max(tp + fn, 1)})
        print(f"  {KNOB[det]} {v:>7g}  F1m {rows[-1]['f1']:.3f}  "
              f"emptyFP {rows[-1]['empty_fp']:.2f}", flush=True)
    return rows


def analyse(det, rows):
    subs = sorted(rows[0]["per"])
    M = np.array([[r["per"][s] for s in subs] for r in rows])
    mean = M.mean(axis=1)
    b = int(np.argmax(mean))
    print(f"\nbest {KNOB[det]}={rows[b]['v']:g}, mean marked macro F1 {mean[b]:.3f}")
    print(f"\nPAIRED comparison vs best (between-patient level cancels):")
    print(f"{KNOB[det]:>8}{'F1m':>8}{'mean d':>9}{'SE d':>8}{'adm':>6}{'emptyFP':>9}"
          f"{'P':>7}{'R':>7}")
    adm = []
    for i, r in enumerate(rows):
        d = M[i] - M[b]
        se = float(d.std(ddof=1) / np.sqrt(len(subs)))
        ok = d.mean() >= -ADMIT_MARGIN
        adm.append(ok)
        print(f"{r['v']:>8g}{mean[i]:>8.3f}{d.mean():>9.3f}{se:>8.3f}"
              f"{('yes' if ok else 'no'):>6}{r['empty_fp']:>9.2f}{r['prec']:>7.3f}"
              f"{r['rec']:>7.3f}")
    keep = [r["v"] for r, o in zip(rows, adm) if o]
    print(f"\nADMISSIBLE {KNOB[det]} {min(keep):g}-{max(keep):g}  [{len(keep)}/{len(rows)}]")
    print(f"  emptyFP across it: " +
          ", ".join(f"{r['empty_fp']:.2f}" for r, o in zip(rows, adm) if o))

    print("\nper-patient optimum:")
    for s in subs:
        bb = max(rows, key=lambda r: r["per"][s])
        print(f"  {s}: best {KNOB[det]} {bb['v']:>7g}  F1m {bb['per'][s]:.3f}")

    for base in BASELINES[det]:
        d0 = [r for r in rows if r["v"] == base]
        if not d0:
            continue
        d0 = d0[0]
        gains = []
        print(f"\nLOPO vs fixed {KNOB[det]}={base:g} (F1m {d0['f1']:.3f}):")
        for h in subs:
            pick = max(rows, key=lambda r: np.mean([r["per"][s] for s in subs if s != h]))
            g = pick["per"][h] - d0["per"][h]
            gains.append(g)
            print(f"  hold {h}: tuned {pick['v']:>7g} -> {pick['per'][h]:.3f}"
                  f"   fixed -> {d0['per'][h]:.3f}   ({g:+.3f})")
        print(f"  mean held-out gain: {np.mean(gains):+.4f}")
    return [r for r, o in zip(rows, adm) if o]


if __name__ == "__main__":
    det = sys.argv[1] if len(sys.argv) > 1 else "barkmeier"
    if det == "janca":
        print(f"janca fixed: {JANCA_FIXED}")
    else:
        print(f"SCALE={SCALE}  DET_THRESHOLDS={DET_THRESHOLDS}  std_coeff={STD_COEFF}")
    analyse(det, sweep(det))
