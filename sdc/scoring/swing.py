"""
sdc.scoring.swing
-----------------
Threshold swing from the per-channel rate sweeps written by `sdc.scoring.sweep_rates`.

    .venv\\Scripts\\python.exe -m sdc.scoring.swing P1_ANT145
    .venv\\Scripts\\python.exe -m sdc.scoring.swing P1_ANT2

`sweep_rates` stores PER-CHANNEL stim and baseline rates for every setting, so every channel
gate here is applied post-hoc -- no gate choice is baked into the stored data and a new gate
costs no detector time.

THE THREE GATES, AND WHY THE ABSOLUTE ONE IS BUILT THE WAY IT IS
  all    every channel measurable in all detectors. The honest denominator.
  p67    relative: baseline rate above the 67th percentile. Keeps ~1/3 of channels at EVERY
         setting, so settings stay comparable within a detector -- but the surviving channels
         differ between detectors and between files, so it cannot compare across either.
  abs    absolute: baseline >= 10 det/chan-min.

  The obvious way to build `abs` is wrong, and the first version of this module had it wrong.
  Gating on each detector's OWN baseline at each setting makes the surviving population move
  with the threshold (Delphos kept 8 channels at the strict end and 206 at the loose end), so
  the "swing" partly measured the population changing under it. Worse, each detector scored a
  different channel set, so the detectors were not comparable either.

  Instead the gate is computed ONCE per recording, from the MEDIAN baseline rate across the
  three detectors at their production defaults, and that single channel list is then used for
  every detector and every setting. n is then identical everywhere -- the comparison is between
  thresholds and between detectors, never between populations. A channel must be measurable in
  all three detectors to be eligible at all, for the same reason.

MARKS-ADMISSIBLE RANGE (the solid bars)
  A threshold range is only interesting if the marked data would let you choose it. `marks()`
  scores every swept setting against the three marked stim-free blocks (P1, P5, P8) and keeps
  those whose mean PAIRED difference from the best is within knob_range.ADMIT_MARGIN (0.01
  marked macro F1). The solid bar is the effect range those admissible settings span; the pale
  bar is the full swept range.

  READ THE SOLID BAR WITHIN A DETECTOR, NEVER ACROSS ONE. It answers "how much does this
  detector's effect estimate depend on a threshold the marks cannot resolve?", which is
  well-posed. It does NOT rank detectors: Janca's admissible k1 spans 3.0-5.5, a factor of 1.83
  in knob units, while Barkmeier's TAMP spans 400-1500, a factor of 3.75, so an equal ADMITTED
  FRACTION covers twice as much parameter space for Barkmeier. There is no common axis for two
  different knobs, so a narrower bar is not evidence of a better-determined detector -- it can
  be evidence of a narrower grid.

  MARKED MACRO F1, precisely: per marked channel, F1 = 2TP/(2TP+FP+FN) at +-50 ms; averaged
  unweighted over channels WITH marks (a 200-mark channel counts the same as a 3-mark one);
  then averaged unweighted over the three patients (P1 counts the same as P5 and P8, though it
  holds 74% of the marks). Channels with no marks are excluded and scored as empty_fp instead.
  Observed values run 0.28-0.38, so the 0.01 margin is ~3% relative, not 1%. Delphos has no marks scoring
  (it has never been run against the marked blocks) so it shows a pale bar only, and is
  labelled as such rather than silently drawn as if constrained.
"""
import json
import sys

import numpy as np

from sdc.common.paths import RUNS, FIGURES

DETS = ("janca", "barkmeier", "delphos")
COL = {"janca": "#c8102e", "barkmeier": "#0072b2", "delphos": "#4a3aa7"}
KNOB = {"janca": "k1", "barkmeier": "TAMP", "delphos": "Spk_thr"}
# production operating points, used only to define the fixed absolute-gate channel set
DEFAULT = {"janca": (3.65, 0.0), "barkmeier": (1200.0, 0.0), "delphos": (50.0, 0.0)}
ABS_THRESH = 10.0


def load(key, det):
    p = RUNS / "sweeps" / f"rates_{key}_{det}.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    s = np.asarray(z["settings"], float)
    stim = np.asarray(z["stim"], float)
    base = np.asarray(z["base"], float)
    if det == "janca":
        # k3=0 ONLY. The sweep grid is 2-D (k1 x k3) but k3 is not a knob: it does not appear in
        # Janca et al. at all -- the published threshold is mode+median, i.e. k3=0 -- and the
        # k1-only decision was already taken. Leaving k3 in made the admissible SET
        # incomparable: at margin 0.01 it admitted 2/16 Janca settings against 3/6 Barkmeier
        # ones, 12% vs 50% of the grid, because varying k3 moves F1 enough to fall outside the
        # margin. That difference in admitted FRACTION, not any property of the detectors, was
        # producing Janca's apparently tighter effect band.
        keep = s[:, 1] == 0.0
        s, stim, base = s[keep], stim[keep], base[keep]
    return {"set": s, "stim": stim, "base": base,
            "names": [str(x) for x in z["names"]]}


def _default_row(d, det):
    want = DEFAULT[det]
    for i, s in enumerate(d["set"]):
        if abs(s[0] - want[0]) < 1e-9 and abs(s[1] - want[1]) < 1e-9:
            return i
    return int(np.argmin(np.abs(d["set"][:, 0] - want[0])))


def gates(data, ref_dets=("janca", "barkmeier")):
    """Channel masks shared by every detector and setting. Returns {name: mask}.

    `ref_dets` defines the channel set and defaults to Janca+Barkmeier because those are the two
    swept on EVERY file -- Delphos has no per-channel sweep on ANT 2 Hz. Building the reference
    from whatever happens to be present would give ANT 2 Hz and ANT 145 Hz different gates and
    silently break the cross-file comparison the absolute gate exists to make.
    """
    dets = [k for k in DETS if data.get(k) is not None]
    ref_use = [k for k in ref_dets if data.get(k) is not None] or dets
    n = data[dets[0]]["base"].shape[1]
    # eligible = finite and non-zero baseline in EVERY detector, at every setting
    ok = np.ones(n, bool)
    for k in dets:
        d = data[k]
        ok &= np.isfinite(d["base"]).all(axis=0) & np.isfinite(d["stim"]).all(axis=0)
        ok &= (d["base"] > 0).all(axis=0)
    # reference baseline: median across detectors at their production defaults
    ref = np.median(np.vstack([data[k]["base"][_default_row(data[k], k)] for k in ref_use]),
                    axis=0)
    p67 = np.zeros(n, bool)
    if ok.any():
        p67 = ok & (ref >= np.percentile(ref[ok], 67.0))
    return {"all": ok, "p67": p67, "abs": ok & (ref >= ABS_THRESH)}


def ratios(d, mask):
    out = []
    for i in range(d["set"].shape[0]):
        s, b = d["stim"][i], d["base"][i]
        m = mask & np.isfinite(s) & np.isfinite(b) & (b > 0)
        out.append(float(np.median(s[m] / b[m])) if m.sum() else np.nan)
    return np.array(out)


# ---------------------------------------------------------------- marks constraint
def marks(det, settings, cache=True):
    """Marked-channel macro F1 for each swept setting, against P1/P5/P8 baselines."""
    p = RUNS / "sweeps" / f"marks_swing_{det}.jsonl"
    have = {}
    if cache and p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                have[tuple(r["set"])] = r
    todo = [tuple(s) for s in settings.tolist() if tuple(s) not in have]
    if todo:
        from sdc.scoring.overnight import _detect, _score
        from sdc.scoring.score_marks import done_blocks, RATERS, TOL
        from sdc.scoring.tune_marks import BLOCK_REC, load_block
        data = [load_block(b) for b in [b for b in done_blocks(RATERS[0]) if b in BLOCK_REC]]
        for s in todo:
            # _detect already supplies band_low/decimation for janca
            q = ({"k1": s[0], "k3": s[1], "band_high": 50.0} if det == "janca"
                 else {"TAMP": s[0]})
            per = {d["subj"]: _score(_detect(det, d, q), d, TOL) for d in data}
            r = {"set": list(s),
                 "per": {k: v["marked_macro_f1"] for k, v in per.items()},
                 "f1": float(np.mean([v["marked_macro_f1"] for v in per.values()]))}
            have[s] = r
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")
            print(f"    marks {det} {s}: F1m {r['f1']:.3f}", flush=True)
    rows = [have[tuple(s)] for s in settings.tolist()]
    subs = sorted(rows[0]["per"])
    M = np.array([[r["per"][s] for s in subs] for r in rows])
    f1 = M.mean(axis=1)
    # PAIRED difference against the best setting, admitted on a fixed practical margin --
    # NOT on an SE, which at n=3 either admits everything (a real t-test does) or rewards noisy
    # settings (mean+1SE let Barkmeier's TAMP=1400 in at mean d -0.036 while rejecting 400 at
    # -0.032). sdc.scoring.knob_range.ADMIT_MARGIN is the single source for the threshold.
    from sdc.scoring.knob_range import ADMIT_MARGIN
    D = M - M[int(np.argmax(f1))]
    se = D.std(axis=1, ddof=1) / np.sqrt(len(subs))
    return f1, D.mean(axis=1) >= -ADMIT_MARGIN, float(np.mean(se))


def report(key, use_marks=True):
    data = {d: load(key, d) for d in DETS}
    missing = [d for d in DETS if data[d] is None]
    if missing:
        print(f"[warn] {key}: no sweep for {', '.join(missing)} -- omitted from the figure")
    g = gates(data)
    print(f"\n=== {key}: median per-channel stim/baseline ===")
    print(f"  channels: all n={g['all'].sum()}, p67 n={g['p67'].sum()}, "
          f">={ABS_THRESH:g}/min n={g['abs'].sum()}   (same set for every detector & setting)")
    res = {}
    for det in DETS:
        d = data[det]
        if d is None:
            continue
        res[det] = {"v": {k: ratios(d, m) for k, m in g.items()}, "adm": None}
        print(f"\n  {det.upper()}  ({d['set'].shape[0]} settings, {KNOB[det]} "
              f"{d['set'][0, 0]:g} -> {d['set'][-1, 0]:g})")
        if use_marks and det != "delphos":
            f1, adm, se = marks(det, d["set"])
            res[det]["adm"] = adm
            res[det]["f1"] = f1
            kept = ", ".join(format(s[0], "g") for s in d["set"][adm])
            print(f"    marks: best F1m {f1.max():.3f} (SE {se:.3f}), "
                  f"{adm.sum()}/{len(adm)} admissible [{KNOB[det]} = {kept}]")
        for gate, lab in (("all", "all channels"), ("p67", ">p67 baseline"),
                          ("abs", f">={ABS_THRESH:g} det/min")):
            v = res[det]["v"][gate]
            if not np.isfinite(v).any():
                continue
            line = (f"    {lab:<16} {np.nanmin(v):.3f}-{np.nanmax(v):.3f}"
                    f"   swing {np.nanmax(v) / np.nanmin(v):.2f}x")
            if res[det]["adm"] is not None and res[det]["adm"].any():
                a = v[res[det]["adm"]]
                line += (f"   | admissible {np.nanmin(a):.3f}-{np.nanmax(a):.3f}"
                         f" ({np.nanmax(a) / np.nanmin(a):.2f}x)")
            print(line)
    return res, g


def figure(key, res, g, fname=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gates_ = (("all", f"all channels (n={g['all'].sum()})"),
              ("p67", f">p67 baseline rate (n={g['p67'].sum()})"),
              ("abs", f">={ABS_THRESH:g} det/chan-min, fixed set (n={g['abs'].sum()})"))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharex=True)
    for ax, (gate, lab) in zip(axes, gates_):
        y, ticks, labels = 0, [], []
        for det in DETS:
            if det not in res:
                continue
            v = res[det]["v"][gate]
            f = v[np.isfinite(v)]
            if not f.size:
                continue
            ax.plot([f.min(), f.max()], [y, y], lw=9, color=COL[det], alpha=.25,
                    solid_capstyle="butt")
            adm = res[det]["adm"]
            if adm is not None and adm.any():
                a = v[adm & np.isfinite(v)]
                ax.plot([a.min(), a.max()], [y, y], lw=15, color=COL[det],
                        solid_capstyle="butt", zorder=3)
                labels.append(f"{det.capitalize()}\nadm {a.min():.2f}-{a.max():.2f}"
                              f" ({a.max() / a.min():.2f}x)")
            else:
                labels.append(f"{det.capitalize()}\n{f.min():.2f}-{f.max():.2f}"
                              f" ({f.max() / f.min():.2f}x)\n(no marks)")
            ax.plot(f, np.full(f.size, y), "o", ms=5, color="0.2", zorder=4)
            ticks.append(y)
            y -= 1
        ax.axvline(1.0, color="0.3", ls="--", lw=1.4)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylim(min(ticks) - 0.7 if ticks else -2.7, 0.7)
        ax.set_xlabel("median per-channel stim / baseline")
        ax.set_title(lab, fontsize=9.5, loc="left")
        ax.grid(axis="x", alpha=.3)
    fig.suptitle(f"{key}: how much does the detection threshold move the stim effect?\n"
                 "pale = full swept range, solid = settings the marked data admits; "
                 "dashed = no effect", fontsize=11)
    fig.tight_layout()
    out = FIGURES / "real" / "_tuning" / (fname or f"swing_all_detectors_{key}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    k = sys.argv[1] if len(sys.argv) > 1 else "P1_ANT145"
    r, g = report(k)
    figure(k, r, g)
