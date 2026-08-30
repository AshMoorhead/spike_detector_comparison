"""
sdc.scoring.overnight
---------------------
Resumable parameter grids against the three marked baseline blocks (P1, P5, P8).

    .venv\\Scripts\\python.exe -m sdc.scoring.overnight janca
    .venv\\Scripts\\python.exe -m sdc.scoring.overnight barkmeier
    .venv\\Scripts\\python.exe -m sdc.scoring.overnight ant145

Each job appends ONE JSON line per setting to runs/sweeps/overnight_<job>.jsonl and skips
settings already present, so a crash costs only the setting in flight. Delete the file to start
over. The jobs are independent -- run them in any order, and a failure in one leaves the others
untouched.

WHY GRIDS AND NOT MORE ONE-AT-A-TIME SCREENS
  Everything measured so far moved a single parameter around its default, which cannot see
  interactions -- and there are known ones. `trough_search_ms` CAPS Ldur/Rdur, so it changes
  what the width gates can do. `std_coeff` (per-channel, relative) and `TAMP` (absolute) are two
  thresholds in series, so the second only bites on what the first admits. The one 2-D grid run
  so far did show the optimum sitting away from both marginal optima.

WHAT IS DELIBERATELY NOT SWEPT
  Janca: winsize/noverlap, buffering, polyspike_union_time -- all screened and inert
  (buffering was bit-identical at 300 vs 600).
  Janca k2 -- the "ambiguous" second tier (k2 != k1 runs a second lower-threshold pass whose
  detections are accepted conditionally). Off by default, and the module docstring flags the k2
  path as "ported for completeness", NOT covered by the verification that exercises the default
  path -- so any gain would need MATLAB cross-checking before it could be believed. Deliberately
  out of scope.
  Barkmeier BlockSize -- the length normalised at once; held at 1 min.

  SCALE IS NOT EXCLUDED AND MUST NOT BE. An earlier version of this file claimed SCALE was
  redundant with TAMP, on the reasoning that std_coeff thresholds fEEG at mean+k*std and is
  therefore scale-invariant. The candidate stage is indeed invariant, but the claim was tested
  and is false: SCALE=200/TAMP=1600 gives 2790 detections where SCALE=100/TAMP=800 gives 2619.
  `Lslope = Lamp/Ldur` scales with the signal too, so SCALE moves TAMP, LS AND RS together
  while leaving LD/RD (durations) alone -- compensating TAMP leaves the slope gates
  uncompensated. Measured at module defaults on the three marked blocks, SCALE 25 -> 400 spans
  531 -> 4714 detections, a wider range than TAMP 400-2000 reaches, and SCALE=70 gives
  precision 0.491 vs 0.404 at the default for the same F1. It is the strongest single
  operating-point knob the detector has.

REPORT HELD-OUT NUMBERS, NOT IN-SAMPLE ONES
  P1 carries 74% of the 2917 marks, so leave-one-patient-out is weak -- holding P1 out leaves
  757 marks. In-sample optima have already misled this project twice. Every row therefore
  stores its per-patient macro F1 so the LOPO summary can be recomputed without re-running.

METRICS ARE KEPT UNBLENDED
  `marked_macro_f1` (per-channel F1 over channels WITH marks) and `empty_fp_per_min` are stored
  separately. The blended macro F1 is ~30% "how many empty channels the rater happened to be
  shown", which is a property of the marking protocol, not the detector -- it once ranked a
  setting first that had the worst detection quality of those tested.
"""
import itertools
import json
import sys
import time

import numpy as np

from sdc.common.paths import RUNS
from sdc.common.spike_match import match
from sdc.scoring.score_marks import done_blocks, RATERS, TOL
from sdc.scoring.tune_marks import BARK_FIXED, JANCA_FIXED, BLOCK_REC, load_block

OUT = RUNS / "sweeps"

# ---- grids -------------------------------------------------------------------------------
JANCA = dict(k1=(2.0, 2.6, 3.0, 3.65, 4.5, 5.5), k3=(0.0, 0.5, 1.0, 2.0),
             band_high=(40.0, 50.0, 60.0))
BARK = dict(std_coeff=(2.0, 3.0, 4.0, 5.0, 6.0), TAMP=(600.0, 900.0, 1200.0, 1600.0, 2000.0),
            trough_search_ms=(20.0, 40.0, 60.0),
            filter_spec=((20.0, 50.0, 1.0, 35.0), (10.0, 60.0, 1.0, 35.0)))

# ---- the half-wave shape gates -------------------------------------------------------------
# `Lslope>LS && Rslope>RS && Lamp+Ramp>TAMP && Ldur>LD && Rdur>RD`, with Ldur/Rdur in ms and
# Lslope = Lamp/Ldur (scaled amplitude per ms). All four are MINIMUMS, so the earlier test that
# disabled them (-1e9 / -1) only probed LOOSENING -- it was never evidence that tightening does
# nothing, and it is not why they are here now.
#
# Measured distribution at TAMP=1200 / trough=40 with the gates open (n=2010 detections):
#     Ldur    p1 10.0   p5 16.5   p25 24.0   p50 35.0
#     Lslope  p1  2.6   p5  7.7   p25 18.9   p50 30.9
#     Rslope  p1  5.4   p5 12.8   p25 25.1   p50 39.3
# The defaults LD/RD=8 sit BELOW p1 of Ldur and LS/RS=3 at about p1 of Lslope: they are not weak
# gates, they are disconnected ones. Values below walk p1 -> p5 -> p25 -> p50 so the grid spans
# inert -> biting -> strangling instead of guessing.
#
# LS is tied to RS and LD to RD to keep this at 48 settings. That is a real simplification --
# spikes are asymmetric and Rslope runs ~25% above Lslope at every percentile, so a symmetric
# threshold bites the left half-wave first. Worth splitting only if the tied version moves
# anything.
#
# Held at std_coeff=4: the candidate stage feeds these gates, and letting it move too would make
# a change here unattributable. trough is included because it CAPS Ldur/Rdur -- at trough=20 a
# 24 ms minimum duration is unsatisfiable, which is the one interaction that must not be missed.
BARK_SHAPE = dict(slope=(3.0, 10.0, 20.0, 35.0), dur=(8.0, 16.0, 24.0),
                  trough_search_ms=(40.0, 60.0), TAMP=(900.0, 1200.0))


def _shape_expand(c):
    return dict(LS=c["slope"], RS=c["slope"], LD=c["dur"], RD=c["dur"],
                trough_search_ms=c["trough_search_ms"], TAMP=c["TAMP"])


def _shape_skip(c):
    """Ldur can never exceed the look-back window, so LD >= trough admits nothing."""
    return c["dur"] >= c["trough_search_ms"]


def _key(d):
    return json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in sorted(d.items())})


def _done(path):
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(_key(json.loads(line)["params"]))
    return out


def _score(dets, d, tol):
    """Pooled counts plus per-channel F1 over MARKED channels, and empty-channel FP rate."""
    tp = fp = fn = 0
    f_mk, emp = [], []
    for c, tt in dets.items():
        m = d["truth"][c]
        h = int(match(tt, m, tol)[1].sum()) if (tt.size and m.size) else 0
        if m.size:
            tp += h
            fp += tt.size - h
            fn += m.size - h
            den = 2 * h + (tt.size - h) + (m.size - h)
            f_mk.append(0.0 if den == 0 else 2 * h / den)
        else:
            fp += tt.size
            emp.append(tt.size / d["mins"])
    return dict(tp=tp, fp=fp, fn=fn, marked_macro_f1=float(np.mean(f_mk)) if f_mk else np.nan,
                empty_fp=float(np.mean(emp)) if emp else np.nan,
                n_det=sum(t.size for t in dets.values()))


def _detect(det, d, p):
    if det == "janca":
        from sdc.detect.janca_detect_spikes import detect_spikes as janca
        kw = {**JANCA_FIXED, **p}          # shared operating point; p overrides
        out, _a, _b = janca(d["x"], d["fs"], **kw)
        ch = np.asarray(out["chan"], int)
        t = np.asarray(out["pos"], float)
        return {c: np.sort(t[ch == i]) for i, c in enumerate(d["chans"])}
    from seeg import detect_spikes as bark
    q = {**BARK_FIXED, **p}
    bark(d["rec"], None, post_mask_spikes=False, fill_bad_samples=False,
         det_thresholds=[q["LS"], q["RS"], q["TAMP"], q["LD"], q["RD"]],
         std_coeff=q["std_coeff"], trough_search_ms=q["trough_search_ms"],
         filter_spec=q["filter_spec"])
    return {c: np.sort(np.asarray(d["rec"]["info"]["DetectedSpikes"][i], float) / d["fs"])
            for i, c in enumerate(d["chans"])}


def grid(det, spec, tag, tol=TOL, expand=None, skip=None):
    """`expand` maps a checkpointed combo to detector kwargs; `skip` drops degenerate cells."""
    path = OUT / f"overnight_{tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = _done(path)
    data = [load_block(b) for b in [b for b in done_blocks(RATERS[0]) if b in BLOCK_REC]]
    combos = [dict(zip(spec, v)) for v in itertools.product(*spec.values())]
    if skip is not None:
        combos = [c for c in combos if not skip(c)]
    todo = [c for c in combos if _key(c) not in seen]
    print(f"{tag}: {len(combos)} settings, {len(seen)} already done, {len(todo)} to run",
          flush=True)
    for i, p in enumerate(todo, 1):
        t0 = time.time()
        q = expand(p) if expand is not None else p
        per = {}
        for d in data:
            per[d["subj"]] = _score(_detect(det, d, q), d, tol)
        tp = sum(v["tp"] for v in per.values())
        fp = sum(v["fp"] for v in per.values())
        fn = sum(v["fn"] for v in per.values())
        row = {"params": {k: list(v) if isinstance(v, tuple) else v for k, v in p.items()},
               "prec": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
               "marked_macro_f1": float(np.mean([v["marked_macro_f1"] for v in per.values()])),
               "empty_fp": float(np.nanmean([v["empty_fp"] for v in per.values()])),
               "per_patient": {k: v["marked_macro_f1"] for k, v in per.items()},
               "secs": round(time.time() - t0, 1)}
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"  [{i}/{len(todo)}] {row['params']}  P {row['prec']:.3f} R {row['recall']:.3f}"
              f"  markedMacroF1 {row['marked_macro_f1']:.3f}  emptyFP {row['empty_fp']:.2f}"
              f"  ({row['secs']:.0f}s)", flush=True)
    print(f"[done] {path}")


def ant145():
    """Per-channel rates over each detector's grid on P1 145 Hz -- feeds swing_all_detectors."""
    from sdc.scoring.sweep_rates import run
    for d in ("janca", "barkmeier", "delphos"):
        out = RUNS / "sweeps" / f"rates_P1_ANT145_{d}.npz"
        if out.is_file():
            print(f"ant145/{d}: already done -> {out.name}", flush=True)
            continue
        print(f"ant145/{d}: starting", flush=True)
        run("P1_ANT145", d)


if __name__ == "__main__":
    job = sys.argv[1] if len(sys.argv) > 1 else "janca"
    if job == "janca":
        grid("janca", JANCA, "janca")
    elif job == "barkmeier":
        grid("barkmeier", BARK, "barkmeier")
    elif job == "barkmeier_shape":
        grid("barkmeier", BARK_SHAPE, "barkmeier_shape",
             expand=_shape_expand, skip=_shape_skip)
    elif job == "ant145":
        ant145()
    else:
        raise SystemExit(
            "job must be one of: janca, barkmeier, barkmeier_shape, ant145")
