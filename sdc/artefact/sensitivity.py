"""
sdc.artefact.sensitivity
------------------------
How much does the stimulation result depend on the artefact mask -- and therefore how accurate
does the artefact detector actually need to be?

    .venv\\Scripts\\python.exe -m sdc.artefact.sensitivity

Reads runs/<rec>_qc<profile>.npz, one per rung of the ladder, produced by

    RECORDING=P1_stim QC_PROFILE=none .venv\\Scripts\\python.exe -m sdc.detect.run_windows

Each rung is a COMPLETE re-run, not a re-mask. That matters: with FILL_ALL the mask decides
which samples get AR-filled before any detector sees the array, and the filled array is what
gets written to the EDF Delphos reads. So the mask reaches the answer by four separate routes,
and only a full re-run exercises all of them:

    1. the signal      masked samples are replaced by spectrum-matched noise
    2. the detections  anything inside the dilated mask is dropped
    3. the denominator analysable seconds per channel, which sets every rate
    4. the channel set MIN_CLEAN_FRAC drops channels more than 80% masked

WHAT TO READ, AND IN WHAT ORDER
  The effect panel alone cannot answer the question, because two rungs can agree on the effect
  while disagreeing wildly on how much data survived. So every rung is reported as a triple --
  effect, masked fraction, channels retained -- and the useful comparison is the SHAPE across
  rungs rather than any single number:

    flat effect across rungs   the artefact detector barely matters here; its accuracy is not
                               what limits this result, and effort belongs elsewhere.
    effect moves with masking  it matters, and the direction says how. More masking pushing the
                               ratio toward 1 means the mask is eating the effect; more masking
                               pushing it away from 1 means artefact was diluting it.

ONE DETECTOR IS NOT SAFE TO READ HERE
  Janca and Barkmeier are deterministic, so a change between rungs is caused by the mask.
  Delphos is not: it tiles against available system RAM, and although `pin_free_ram_gb` pulls
  free RAM to a fixed value before each call, the pin can only lower free RAM and not raise it.
  The `prod` runs were also made earlier, under machine state nobody recorded. A change that
  appears in all three detectors is a mask effect; a Delphos-only change is confounded with
  tiling and needs the repeat-run determinism check before it means anything.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from seeg._style import RED, BLUE, MUTED, recessive

from sdc.common import cond
from sdc.common.paths import RUNS, figdir
from sdc.artefact.blocks import block_table, log_ratio, ci, permute, pair_changes

SUPTITLE_SINGLE = ("Detectors disagree on the sign of the stimulation effect\n"
                   "marked truth: suppression.  solid = own channels, dashed = fixed set")
VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}
DETS = ("Janca", "Barkmeier", "Delphos")

# Ordered by how much they mask, so the x axis is a real axis and not a set of labels.
LADDER = ["none", "loose", "prod", "strict", "vstrict"]
# The stimPowerThr-only sweep, ordered by how much it masks. Everything else is held at
# production, so movement along THIS ladder is attributable to one rule.
# "sp" is stimPowerThr and the number is its value -- sp200 means stimPowerThr=200. Terse, and
# kept only because the run files on disk already carry these names.
SP_LADDER = ["sp1000", "sp200", "prod", "sp10", "sp2"]
# The chosen operating point, shown ON the sp sweep so it can be read against the ladder rather
# than quoted on its own. It is NOT a rung of that sweep and must not be read as one: the sp
# rungs are ABSOLUTE stimPowerThr values, while `final` is kStim=450 RELATIVE to each channel's
# own baseline (and gradThr=4000, far looser than production's 400). It therefore masks a
# DIFFERENT set of samples, not merely a different amount -- which is the whole point of showing
# it here, because on P1 it masks the same fraction as sp200 and returns a very different ratio.
OFF_LADDER = ("final",)
RECS = ("P1_stim", "P5_stim")


def path_for(rec, profile):
    return RUNS / (f"{rec}.npz" if profile == "prod" else f"{rec}_qc{profile}.npz")


def load_rung(rec, profile, n_boot=1500):
    """Everything one rung contributes: the effect, its cost, and its provenance."""
    p = path_for(rec, profile)
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    # ABSENT and 'prod' are different things and must not be conflated -- the first version of
    # this check defaulted a missing key to 'prod' and then reported a mislabelled file, which
    # sent the investigation at the wrong target. A missing key means the run predates the
    # provenance field; that is only acceptable for the production rung, whose files do.
    if "qc_profile" in z.files:
        stored = str(z["qc_profile"])
        if stored != profile:
            raise SystemExit(f"{p.name} records qc_profile={stored!r} but its filename says "
                             f"{profile!r} -- a rung has been written to the wrong file.")
    elif profile != "prod":
        raise SystemExit(f"{p.name} carries no qc_profile, so it cannot be confirmed to be the "
                         f"{profile!r} rung. Re-merge it (run_windows.merge_windows) now that "
                         f"the field is written.")
    bp = block_table(z)
    ON, OFF = cond.select(z, "on"), cond.select(z, "off")

    out = {"n_chan": bp.n_chan, "n_block": bp.n_block,
           # masked fraction over the WHOLE recording, per condition, so the cost of a rung is
           # visible separately for the stim-ON time where artefact actually lives
           "masked_on": 1.0 - float(np.mean(ON.clean_sec) / max(ON.T, 1e-9)),
           "masked_off": 1.0 - float(np.mean(OFF.clean_sec) / max(OFF.T, 1e-9)),
           "det": {}}
    for d in DETS:
        if d not in bp.det:
            continue
        pt, lo, hi = ci(bp.det[d], bp.n_block, bp.n_chan, n_boot)
        _o, pv, _null, _ex = permute(bp.det[d], bp.n_block, bp.n_chan)
        v, _n = pair_changes(bp.det[d])
        v = v[np.isfinite(v)]
        out["det"][d] = {"ratio": 10 ** pt, "lo": 10 ** lo, "hi": 10 ** hi, "p": pv,
                         "per_pair": 10 ** v,
                         "d": abs(v.mean()) / v.std(ddof=1) if v.size > 2 else np.nan}
    return out


def report(recs=RECS, ladder=None):
    ladder = ladder or LADDER
    data = {}
    for rec in recs:
        for prof in ladder:
            r = load_rung(rec, prof)
            if r:
                data[(rec, prof)] = r
    if not data:
        raise SystemExit("no rungs found -- run the ladder first (see the module docstring).")

    for rec in recs:
        have = [p for p in ladder if (rec, p) in data]
        if not have:
            continue
        print(f"\n=== {rec}: {len(have)} rung(s) -- {', '.join(have)}")
        print(f"    {'profile':<9}{'masked ON':>11}{'masked OFF':>12}{'chan':>6}{'blk':>5}   "
              + "".join(f"{d:>26}" for d in DETS))
        for prof in have:
            r = data[(rec, prof)]
            cells = []
            for d in DETS:
                a = r["det"].get(d)
                cells.append("-" if a is None else
                             f"{a['ratio']:.3f}[{a['lo']:.2f},{a['hi']:.2f}] p{a['p']:.3f}")
            print(f"    {prof:<9}{r['masked_on']:>10.1%}{r['masked_off']:>12.1%}"
                  f"{r['n_chan']:>6}{r['n_block']:>5}   " + "".join(f"{c:>26}" for c in cells))
    return data


def figure(data, recs=RECS, ladder=None):
    ladder = ladder or LADDER
    fig, axes = plt.subplots(3, len(recs), figsize=(7.0 * len(recs), 10.0), squeeze=False,
                             sharex="col")
    for j, rec in enumerate(recs):
        have = [p for p in ladder if (rec, p) in data]
        if not have:
            continue
        # ORDER BY MEASURED MASKING, not by position in the hand-written list. The x axis claims
        # "left = no masking, right = most aggressive", and that was only true while every rung
        # came from one monotone sweep. `final` is off that sweep, so its place has to be
        # measured; sorting makes the axis label true by construction for anything added later.
        have.sort(key=lambda p: data[(rec, p)]["masked_on"])
        x = np.arange(len(have))

        # (row 0) the effect
        ax = axes[0][j]
        for d in DETS:
            y = [data[(rec, p)]["det"].get(d, {}).get("ratio", np.nan) for p in have]
            lo = [data[(rec, p)]["det"].get(d, {}).get("lo", np.nan) for p in have]
            hi = [data[(rec, p)]["det"].get(d, {}).get("hi", np.nan) for p in have]
            ax.plot(x, y, "-o", ms=6, lw=1.6, color=COLORS[d], label=d)
            ax.fill_between(x, lo, hi, color=COLORS[d], alpha=.12, lw=0)
        ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
        ax.set_yscale("log")
        ax.set_ylabel("ON/OFF ratio (log)\nshaded = 95% CI")
        ax.set_title(rec if not summary_panel else f"({'ab'[j]}) {rec}",
                     fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, ncol=3)

        # (row 1) what each rung cost -- the panel that stops row 0 being over-read
        ax = axes[1][j]
        ax.plot(x, [data[(rec, p)]["masked_on"] * 100 for p in have], "-o", ms=6, lw=1.8,
                color="#c2691f", label="masked during stim ON")
        ax.plot(x, [data[(rec, p)]["masked_off"] * 100 for p in have], "-s", ms=5, lw=1.4,
                color="#c2691f", alpha=.5, label="masked during stim OFF")
        ax.set_ylabel("% of time masked", color="#c2691f")
        ax.tick_params(axis="y", colors="#c2691f")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax2 = ax.twinx()
        ax2.plot(x, [data[(rec, p)]["n_chan"] for p in have], "-^", ms=6, lw=1.6,
                 color="0.25", label="channels retained")
        ax2.set_ylabel("channels retained", color="0.25")
        ax2.legend(frameon=False, fontsize=8, loc="upper right")

        # (row 2) the per-pair spread behind each point in row 0
        ax = axes[2][j]
        for i, p in enumerate(have):
            for k, d in enumerate(DETS):
                a = data[(rec, p)]["det"].get(d)
                if not a:
                    continue
                xs = i + (k - 1) * 0.22
                ax.scatter(np.full(a["per_pair"].size, xs)
                           + np.random.default_rng(0).uniform(-.05, .05, a["per_pair"].size),
                           a["per_pair"], s=16, color=COLORS[d], alpha=.45, edgecolor="none")
                ax.plot([xs - .09, xs + .09], [a["ratio"]] * 2, color=COLORS[d], lw=2.2)
        ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
        ax.set_yscale("log")
        ax.set_ylabel("per-pair ratios (log)")
        ax.set_xlabel("artefact profile  (left = no masking, right = most aggressive)")

        # Shade the off-ladder operating point(s) and label them differently. Sitting at the
        # right x position makes them comparable; looking identical would make them look like
        # another step of the same knob, which is exactly the misreading to avoid.
        for i, p in enumerate(have):
            if p in OFF_LADDER:
                for ax in axes[:, j]:
                    ax.axvspan(i - .42, i + .42, color="#2e7d32", alpha=.09, lw=0, zorder=0)
        for ax in axes[:, j]:
            ax.set_xticks(x)
            ax.set_xticklabels(have, fontsize=9)
            for lbl, p in zip(ax.get_xticklabels(), have):
                if p in OFF_LADDER:
                    lbl.set_color("#2e7d32")
                    lbl.set_fontweight("bold")
            recessive(ax)
            ax.grid(axis="y", alpha=.3)

    _has_off = any(p in OFF_LADDER for (_r, p) in data)
    fig.suptitle(
        "Sensitivity of the stimulation result to the artefact mask.\n"
        "Top: the effect. Middle: what each rung cost in time and channels. "
        "Bottom: the per-pair spread behind each point.\n"
        "A flat top row means the artefact detector's accuracy is not what limits this result."
        + ("\nRungs are ordered by MEASURED masked-ON fraction. The shaded green column is the "
           "chosen operating point ('final'): a RELATIVE per-channel threshold, so it masks a "
           "different SET of samples, not just a different amount." if _has_off else ""),
        fontsize=10)
    fig.tight_layout()
    _tag = "_sp" if ladder is SP_LADDER or set(ladder) & {"sp1000", "sp10"} else ""
    out = figdir("real") / f"artefact_sensitivity{_tag}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")


def channel_confound(recs=RECS, ladder=None, exclude=("vstrict",)):
    """Separate "the mask changed the channel SET" from "the mask changed the DATA".

    Every rung of the ladder does both at once: stricter masking removes contaminated time AND
    removes whole channels, and the second alone would move the answer even if the surviving
    data were untouched. As channels are removed the estimate necessarily drifts toward whatever
    the survivors say, so a ladder computed on each rung's own channel set cannot distinguish
    "artefact was removed" from "the noisy channels were removed".

    So each rung is recomputed twice:
      own      that rung's own surviving channels -- the ladder as reported
      common   the channels that survive EVERY rung, held fixed. Only the data differs, so any
               remaining movement is the mask acting on the signal and the denominators.

    `vstrict` is excluded from the common set by default: it leaves 7-8 channels, so including
    it would shrink the fixed set to almost nothing and answer a different, useless question.

    Also reports WHAT KIND of channel each rung keeps -- median OFF rate of survivors against
    those dropped, measured on the UNMASKED rung so that "how active is this channel" is not
    itself a function of the mask being tested.
    """
    ladder = ladder or LADDER
    print("\n=== channel-set confound: own vs fixed common channel set ===")
    for rec in recs:
        rungs = [p for p in ladder if p not in exclude and path_for(rec, p).is_file()]
        if len(rungs) < 2:
            continue
        tabs = {}
        for p in rungs:
            z = np.load(path_for(rec, p), allow_pickle=False)
            tabs[p] = block_table(z)
        common = np.ones_like(tabs[rungs[0]].keep)
        for p in rungs:
            common &= tabs[p].keep

        # A channel's intrinsic activity, taken from the UNMASKED rung. Using each rung's own
        # OFF rate would confound "which channels are busy" with "which channels this mask
        # happened to keep", which is the very thing being measured.
        base = tabs["none"] if "none" in tabs else tabs[rungs[0]]
        base_rate = np.full(base.keep.size, np.nan)
        a0 = base.det["Janca"]
        base_rate[base.keep] = a0["off_count"].sum(0) / np.maximum(a0["off_sec"].sum(0), 1e-9) * 60

        print(f"\n{rec}: {int(common.sum())} channels survive all of {', '.join(rungs)}")
        print(f"    {'profile':<9}{'chan':>6}{'kept med':>10}{'dropped med':>13}   "
              + "".join(f"{d + ' own/common':>26}" for d in DETS))
        for p in rungs:
            bt = tabs[p]
            kept = base_rate[bt.keep & ~np.isnan(base_rate)]
            drop_m = (~bt.keep) & ~np.isnan(base_rate)
            cells = []
            for d in DETS:
                own = 10 ** log_ratio(bt.det[d])
                sub = np.flatnonzero(common[bt.keep])
                com = 10 ** log_ratio({k: v[:, sub] for k, v in bt.det[d].items()})
                cells.append(f"{own:.3f} / {com:.3f}")
            print(f"    {p:<9}{bt.n_chan:>6}{np.median(kept):>10.2f}"
                  f"{(np.median(base_rate[drop_m]) if drop_m.any() else np.nan):>13.2f}   "
                  + "".join(f"{c:>26}" for c in cells))


def _rung_tables(rec, ladder, exclude):
    rungs = [p for p in ladder if p not in exclude and path_for(rec, p).is_file()]
    tabs = {p: block_table(np.load(path_for(rec, p), allow_pickle=False)) for p in rungs}
    common = np.ones_like(tabs[rungs[0]].keep)
    for p in rungs:
        common &= tabs[p].keep
    base = tabs.get("none", tabs[rungs[0]])
    rate = np.full(base.keep.size, np.nan)
    a0 = base.det["Janca"]
    rate[base.keep] = a0["off_count"].sum(0) / np.maximum(a0["off_sec"].sum(0), 1e-9) * 60
    return rungs, tabs, common, rate


def dropped_table(recs=RECS, ladder=None, exclude=("vstrict",)):
    """Full distribution of the intrinsic OFF rate for kept vs dropped channels, per rung.

    Medians alone cannot answer "is the mask removing the quiet channels or the busy ones",
    because two very different distributions share a median. The quartiles and the tails are
    what show whether masking is selecting on activity at all.
    """
    ladder = ladder or LADDER
    print("\n=== intrinsic OFF rate (det/min, measured on the UNMASKED rung) ===")
    for rec in recs:
        rungs, tabs, _common, rate = _rung_tables(rec, ladder, exclude)
        print(f"\n{rec}")
        print(f"    {'profile':<9}{'group':<9}{'n':>5}"
              + "".join(f"{q:>9}" for q in ("p10", "p25", "med", "p75", "p90")))
        for p in rungs:
            bt = tabs[p]
            for tag, m in (("kept", bt.keep & ~np.isnan(rate)),
                           ("dropped", (~bt.keep) & ~np.isnan(rate))):
                if not m.any():
                    continue
                q = np.percentile(rate[m], [10, 25, 50, 75, 90])
                print(f"    {p:<9}{tag:<9}{int(m.sum()):>5}"
                      + "".join(f"{v:>9.2f}" for v in q))


def confound_figure(recs=RECS, ladder=None, exclude=("vstrict",)):
    """Own channel set against a fixed one -- the figure that shows which is the real problem."""
    ladder = ladder or LADDER
    fig, axes = plt.subplots(2, len(recs), figsize=(7.0 * len(recs), 8.4), squeeze=False)
    for j, rec in enumerate(recs):
        rungs, tabs, common, rate = _rung_tables(rec, ladder, exclude)
        x = np.arange(len(rungs))

        ax = axes[0][j]
        for d in DETS:
            own, com = [], []
            for p in rungs:
                bt = tabs[p]
                own.append(10 ** log_ratio(bt.det[d]))
                sub = np.flatnonzero(common[bt.keep])
                com.append(10 ** log_ratio({k: v[:, sub] for k, v in bt.det[d].items()}))
            ax.plot(x, own, "-o", ms=7, lw=1.8, color=COLORS[d], label=f"{d} own channels")
            ax.plot(x, com, "--s", ms=5, lw=1.5, color=COLORS[d], alpha=.55,
                    label=f"{d} fixed {int(common.sum())} ch")
            # The gap between the two lines IS the channel-selection component; shading it
            # makes that the thing the eye measures rather than something to be inferred.
            ax.fill_between(x, own, com, color=COLORS[d], alpha=.10, lw=0)
        ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
        ax.set_yscale("log")
        ax.set_ylabel("ON/OFF ratio (log)")
        ax.set_title(f"({'ab'[j]}) {rec} -- solid = each rung's own channels,\n"
                     f"dashed = the {int(common.sum())} channels common to every rung. "
                     f"Shading = the channel-selection component.", fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=6.5, ncol=3)

        ax = axes[1][j]
        for i, p in enumerate(rungs):
            bt = tabs[p]
            for k, (tag, m, c) in enumerate((("kept", bt.keep & ~np.isnan(rate), "#2a7f45"),
                                             ("dropped", (~bt.keep) & ~np.isnan(rate), "#c2691f"))):
                if not m.any():
                    continue
                xs = i + (k - 0.5) * 0.34
                bx = ax.boxplot([rate[m]], positions=[xs], widths=0.28, showfliers=False,
                                patch_artist=True)
                bx["boxes"][0].set(facecolor=c, alpha=.30, edgecolor=c)
                for part in ("whiskers", "caps", "medians"):
                    for ln in bx[part]:
                        ln.set(color=c, lw=1.5)
        ax.set_yscale("log")
        ax.set_ylabel("intrinsic OFF rate (det/min, log)\ngreen = kept, orange = dropped")
        ax.set_xlabel("artefact profile")
        ax.set_title(f"({'cd'[j]}) are the dropped channels the quiet ones?",
                     fontsize=9, loc="left")

        for ax in axes[:, j]:
            ax.set_xticks(x)
            ax.set_xticklabels(rungs, fontsize=9)
            recessive(ax)
            ax.grid(axis="y", alpha=.3)

    fig.suptitle("Is the ladder measuring artefact, or measuring channel selection?\n"
                 "Where solid and dashed diverge, the movement is channels being removed -- "
                 "not artefact being removed from the data that remains.", fontsize=10)
    fig.tight_layout()
    _tag = "_sp" if ladder is SP_LADDER or set(ladder) & {"sp1000", "sp10"} else ""
    out = figdir("real") / f"artefact_channel_confound{_tag}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    _recs = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or RECS
    _lad = SP_LADDER if "--sp" in sys.argv else LADDER
    # The chosen operating point is shown alongside either ladder unless --no-final. Placed by
    # its measured masking (see figure()), not appended at the end, so the axis stays honest.
    if "--no-final" not in sys.argv:
        _lad = list(_lad) + [p for p in OFF_LADDER if p not in _lad]
    if "--channels" in sys.argv:
        channel_confound(_recs, _lad)
        dropped_table(_recs, _lad)
        confound_figure(_recs, _lad)
    else:
        figure(report(_recs, _lad), _recs, _lad)


def spread_figure(recs=RECS, ladder=None, exclude=("vstrict",),
                  fname="detector_spread_vs_artefact.png", summary_panel=True,
                  truth_below_one=False, extra=()):
    """Between-detector DISAGREEMENT along the artefact ladder -- the abstract's points 3 and 4.

    Panels (a),(b) are the per-recording ladders; the shaded band is the min-to-max across the
    three detectors at each rung, which IS the disagreement being claimed. Panel (c) reduces
    each rung to that one number so the two recordings can be compared directly.

    Two things the figure is built to keep visible, because both weaken the simple story:

    OWN vs FIXED CHANNELS. Solid lines use each rung's own surviving channels, dashed lines the
    channels surviving every rung. Stricter masking removes contaminated TIME and whole
    CHANNELS at once; only the dashed lines isolate the first. On P1 the spread improves 3.24x
    -> 1.52x on own channels but only 2.47x -> 1.62x on a fixed set, so most of the apparent
    benefit is excluding channels rather than cleaning what remains.

    P5 IS THE CONTROL AND IT DOES NOT IMPROVE. Its detectors already agree to 1.02x on a fixed
    channel set before any handling, and handling makes it slightly worse (1.10x). That is the
    evidence that the disagreement on P1 is artefact-driven rather than intrinsic to the
    detectors -- but it also means "artefact handling improves agreement" cannot be stated as a
    general result, only as one conditional on there being artefact.
    """
    ladder = ladder or LADDER
    ncol = len(recs) + (1 if summary_panel else 0)
    fig = plt.figure(figsize=(5.6 * ncol if ncol > 1 else 7.4, 5.2))
    gs = fig.add_gridspec(1, ncol, width_ratios=[1] * len(recs) + ([1.05] if summary_panel else []),
                          wspace=0.28)
    summary = {}
    # (a) and (b) SHARE the y axis deliberately. On its own scale P5's 1.35x band looks as
    # dramatic as P1's 3.24x; on a shared one it is visibly flat, which is the actual finding.
    axes_share = []

    for j, rec in enumerate(recs):
        rungs, tabs, common, _rate = _rung_tables(rec, ladder, exclude)
        x = np.arange(len(rungs))
        own = {d: [] for d in DETS}
        com = {d: [] for d in DETS}
        for d in DETS:
            for p in rungs:
                bt = tabs[p]
                own[d].append(10 ** log_ratio(bt.det[d]))
                sub = np.flatnonzero(common[bt.keep])
                com[d].append(10 ** log_ratio({k: v[:, sub] for k, v in bt.det[d].items()}))
        O = np.array([own[d] for d in DETS])
        C = np.array([com[d] for d in DETS])
        summary[rec] = {"rungs": rungs, "own": O.max(0) / O.min(0), "fixed": C.max(0) / C.min(0),
                        "n": [tabs[p].n_chan for p in rungs]}

        ax = fig.add_subplot(gs[0, j], sharey=axes_share[0] if axes_share else None)
        axes_share.append(ax)
        ax.fill_between(x, O.min(0), O.max(0), color="0.55", alpha=.18, zorder=0,
                        label="between-detector spread")
        for d in DETS:
            ax.plot(x, own[d], "o-", color=COLORS[d], lw=2.0, ms=6, label=f"{d} (own ch)")
            ax.plot(x, com[d], "s--", color=COLORS[d], lw=1.2, ms=4, alpha=.55,
                    label=f"{d} (fixed ch)")
        ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
        if truth_below_one:
            # Expert marking of this recording says stimulation SUPPRESSES spikes, so the true
            # ratio is below 1. Any detector above the line has the SIGN wrong, which is a
            # stronger statement than disagreement. Provisional until the marks are scored.
            ax.axhspan(ax.get_ylim()[0], 1.0, color="#2e7d32", alpha=.055, zorder=-2)
            ax.text(0.015, 0.02, "expert marks: suppression (ratio < 1)", color="#2e7d32",
                    transform=ax.transAxes, fontsize=8.5, va="bottom")
        for i in (0, len(rungs) - 1):
            ax.annotate(f"{O.max(0)[i] / O.min(0)[i]:.2f}x", (x[i], O.max(0)[i]),
                        xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=10, fontweight="bold", color="0.15")
        # Profiles that are NOT rungs of this ladder are drawn detached, past a separator.
        # finalv2 uses a RELATIVE per-channel stim threshold (kStim=450) where every rung here
        # uses an ABSOLUTE one, so it masks a different SET of samples, not a different amount.
        # Placing it in the sequence would invite reading it as "between prod and strict".
        xt = list(x)
        xl = [f"{p}\n({n} ch)" for p, n in zip(rungs, summary[rec]["n"])]
        for e_i, e in enumerate(extra):
            if not path_for(rec, e).is_file():
                continue
            bt = block_table(np.load(path_for(rec, e), allow_pickle=False))
            xe = len(rungs) + 0.6 + e_i * 1.6
            sub = np.flatnonzero(common[bt.keep])
            for d in DETS:
                # filled = this profile's own channels; open = the same fixed set the dashed
                # ladder lines use, so the channel-selection component is visible here too.
                ax.plot([xe], [10 ** log_ratio(bt.det[d])], "D", color=COLORS[d], ms=9,
                        mec="0.15", mew=.9, zorder=6)
                ax.plot([xe + 0.6],
                        [10 ** log_ratio({k: v[:, sub] for k, v in bt.det[d].items()})],
                        "D", mfc="none", color=COLORS[d], ms=9, mew=1.6, zorder=6)
            ax.axvline(len(rungs) - 0.35, color="0.5", ls=":", lw=1.2)
            xt += [xe, xe + 0.6]
            xl += [f"{e}\n({bt.n_chan} ch)", f"{e}\nfixed ({sub.size} ch)"]
        ax.set_yscale("log")
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=8)
        ax.set_ylabel("stim ON / OFF ratio (log)")
        ax.set_title(rec if not summary_panel else f"({'ab'[j]}) {rec}",
                     fontsize=10, loc="left")
        ax.grid(alpha=.25)
        if j == 0:
            ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=.92)

    if not summary_panel:
        for a in axes_share:
            a.set_xlabel("artefact handling")
        fig.suptitle(SUPTITLE_SINGLE, fontsize=11)
        fig.subplots_adjust(left=0.135, right=0.98, top=0.85, bottom=0.155)
        out = figdir("real") / fname
        fig.savefig(out, dpi=145)
        plt.close(fig)
        print(f"[saved] {out}")
        for rec, st in summary.items():
            print(f"  {rec}: own {st['own'][0]:.2f}x -> {st['own'][-1]:.2f}x   "
                  f"fixed {st['fixed'][0]:.2f}x -> {st['fixed'][-1]:.2f}x")
        return summary

    ax = fig.add_subplot(gs[0, len(recs)])
    for rec, st in summary.items():
        x = np.arange(len(st["rungs"]))
        ax.plot(x, st["own"], "o-", lw=2.2, ms=7, label=f"{rec} own channels")
        ax.plot(x, st["fixed"], "s--", lw=1.4, ms=5, alpha=.6, label=f"{rec} fixed channels")
    ax.axhline(1.0, color="0.25", ls=":", lw=1.4)
    ax.text(0.02, 1.02, "perfect agreement", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color="0.35")
    ax.set_xticks(np.arange(len(summary[recs[0]]["rungs"])))
    ax.set_xticklabels(summary[recs[0]]["rungs"], fontsize=9)
    ax.set_ylabel("max / min across the three detectors")
    ax.set_xlabel("artefact handling")
    ax.set_title("(c) disagreement, reduced to one number", fontsize=10, loc="left")
    ax.grid(alpha=.25)
    ax.legend(fontsize=7.5)

    fig.suptitle("Detector disagreement under stimulation, and what artefact handling does to it\n"
                 "P1 is contaminated and improves; P5 already agrees and does not. Dashed = "
                 "fixed channel set, which removes the channel-selection component. "
                 "(a) and (b) share a y axis.",
                 fontsize=11)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.82, bottom=0.14)
    out = figdir("real") / fname
    fig.savefig(out, dpi=145)
    plt.close(fig)
    print(f"[saved] {out}")
    for rec, st in summary.items():
        print(f"  {rec}: own {st['own'][0]:.2f}x -> {st['own'][-1]:.2f}x   "
              f"fixed {st['fixed'][0]:.2f}x -> {st['fixed'][-1]:.2f}x")
    return summary
