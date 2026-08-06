"""
stim_effect.py
--------------
What does each detector say stimulation did to the spike rate?

    .venv\\Scripts\\python.exe -m sdc.compare.stim_effect runs/P1_stim.npz

This exists because the three detectors do not agree on the SIGN of the effect: on P1, pooled
over the channels measurable in both conditions, the ON/OFF rate ratio is Janca 0.57,
Barkmeier 0.31, Delphos 0.90. That is not a small quantitative disagreement -- it is "spikes
more than halved" versus "nothing happened", from the same 600 seconds. A single figure that
puts the three side by side is the only honest way to present it.

Unlike `evaluate_detectors.py --COND`, this reads BOTH conditions at once, so it lives in its
own script rather than being a fourth view there.

Three things it is careful about, each of which would otherwise manufacture an effect:

  * ANALYSABLE TIME, not wall clock. Stim artefact pushes P1's masked fraction from 3.3% to
    15.1%, so dividing ON counts by 194 s charges every detector for time in which nothing
    could be detected. Denominators come from `clean_sec_on`/`clean_sec_off`, per channel.
  * THE SAME CHANNELS ON BOTH SIDES. 94 of 226 channels have ZERO analysable time during
    stim-ON. Comparing ON's 132 channels against OFF's 224 is not like for like, so everything
    here is restricted to the 132 measurable in both -- and the comparison is PAIRED, because
    a channel is its own best control.
  * SEGMENT STRUCTURE. The ON condition is three separated blocks, not one stretch. Panel (c)
    keeps them in time order against the OFF blocks between them, because "every ON block dips
    relative to its neighbours" and "the recording drifted downwards" look identical in a
    pooled mean and completely different in time.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from seeg._style import RED, BLUE, MUTED, GRID, recessive

from sdc.common import cond

from sdc.common.paths import ROOT as HERE   # repo root, not this file's dir --
                                            # see sdc/common/paths.py
NPZ = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs" / "P1_stim.npz"
if not NPZ.is_file():
    raise SystemExit(f"{NPZ} not found -- run compare_spikes.py first.")

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}
ON_SHADE = "#f0c419"
MIN_OFF_SPIKES = 3   # a channel needs a usable OFF baseline before its RATIO means anything;
                     # 1-vs-0 spikes is a ratio of 0 or infinity, not an effect size


# ----------------------------------------------------------------------
z = np.load(NPZ, allow_pickle=False)
names = [str(s) for s in z["names"]]
fs = float(z["fs"])
n_chan = len(names)
dets = [str(s) for s in z["detectors"]]

TMAX = float(os.environ.get("TMAX", 0)) or None
                       # TMAX=900 restricts to the first 900 s. Analysable time is
                       # recomputed exactly, not scaled -- see cond.select.
ON = cond.select(z, "on", tmax=TMAX)
OFF = cond.select(z, "off", tmax=TMAX)
if ON.clean_sec is None:
    raise SystemExit(f"{NPZ.name} has no clean_sec_on -- re-run compare_spikes.py so the "
                     f"rates can use analysable time rather than wall clock.")

BOTH = ON.measurable & OFF.measurable   # enough analysable time in BOTH conditions;
                                        # see cond.Selection.MIN_CLEAN_FRAC
print(f"{NPZ.name}: {ON.describe()}")
print(f"{' ' * len(NPZ.name)}  {OFF.describe()}")
print(f"[cond] {int(BOTH.sum())}/{n_chan} channels measurable in BOTH conditions "
      f"({int((~BOTH).sum())} lost entirely to the artefact mask in one of them)")

# counts per channel per condition, then rates over that channel's analysable seconds
rate_on, rate_off, cnt_off = {}, {}, {}
for d in dets:
    flag = z[f"{d}_on"].astype(bool)
    c_on = np.bincount(z[f"{d}_chan"][flag], minlength=n_chan)
    c_off = np.bincount(z[f"{d}_chan"][~flag], minlength=n_chan)
    rate_on[d], rate_off[d], cnt_off[d] = ON.rate(c_on), OFF.rate(c_off), c_off

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.32, wspace=0.26)


# ---------------------------------------------------------------- (a) paired scatter
# One point per channel: its OFF rate against its ON rate. A detector claiming stimulation
# suppressed spikes puts its cloud BELOW y=x. This is the whole disagreement in one row.
print(f"\n--- ON vs OFF, paired over the {int(BOTH.sum())} channels measurable in both ---")
print(f"{'detector':<11}{'ON Hz/ch':>10}{'OFF Hz/ch':>11}{'pooled':>9}{'median ch':>11}"
      f"{'n down':>9}{'Wilcoxon p':>13}{'->zero':>8}")
summary = {}
for i, d in enumerate(dets):
    ax = fig.add_subplot(gs[0, i])
    x, y = rate_off[d][BOTH], rate_on[d][BOTH]
    col = COLORS.get(d, MUTED)
    lo = max(min(x[x > 0].min(), y[y > 0].min()) * 0.6, 1e-4)
    hi = max(x.max(), y.max()) * 1.6
    ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1.0, zorder=1, label="no effect")
    ax.fill_between([lo, hi], [lo / 2, hi / 2], [lo, hi], color=GRID, alpha=.5, lw=0, zorder=0,
                    label="halved")
    ax.scatter(x, y, s=16, color=col, alpha=.7, edgecolor="none", zorder=2)

    ok = BOTH & (cnt_off[d] >= MIN_OFF_SPIKES)
    ratio_all = rate_on[d][ok] / rate_off[d][ok]
    # A channel that fired in OFF and NOT AT ALL in ON has no ratio -- log2(0) is -inf, and
    # clipping it to a big negative number is worse than dropping it: it dominates the box plot
    # with an arbitrary value chosen by the clip. It is still an effect, so it is COUNTED and
    # annotated on panel (b) rather than quietly discarded.
    n_zero = int((ratio_all == 0).sum())
    ratio = ratio_all[ratio_all > 0]
    pooled = float(np.sum(rate_on[d][BOTH]) / np.sum(rate_off[d][BOTH]))
    med = float(np.median(ratio_all))     # the zeros DO belong in the median -- it is a rank
    n_down = int((ratio_all < 1).sum())
    # Paired, on log ratio, so "no effect" is a symmetric null. Channels are not independent
    # (one brain, shared artefact), so read p as a consistency check on the sign, not as a
    # trial result.
    p = float(wilcoxon(np.log(ratio)).pvalue) if ratio.size > 5 else np.nan
    summary[d] = dict(pooled=pooled, med=med, ratio=ratio, n=ratio_all.size, p=p,
                      n_down=n_down, n_zero=n_zero)
    print(f"{d:<11}{np.mean(y):>10.4f}{np.mean(x):>11.4f}{pooled:>9.2f}{med:>11.2f}"
          f"{f'{n_down}/{ratio_all.size}':>9}{p:>13.1e}{n_zero:>8}")

    ax.plot([lo, hi], [lo * pooled, hi * pooled], color=col, lw=1.2, alpha=.85,
            label=f"pooled {pooled:.2f}x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("stim OFF rate (Hz)")
    if i == 0:
        ax.set_ylabel("stim ON rate (Hz)")
    ax.set_title(f"{d}   median channel {med:.2f}x", fontsize=10, loc="left", color=col)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    recessive(ax)


# ---------------------------------------------------------------- (b) effect size distribution
# The same thing as a distribution rather than a cloud: where does the typical channel sit?
# log2 so "halved" and "doubled" are the same distance from no-effect, which they are not on a
# linear ratio axis -- 0.5 and 2.0 look like 0.5 and 1.0 away from 1.
axb = fig.add_subplot(gs[1, 0])
for i, d in enumerate(dets):
    v = np.log2(summary[d]["ratio"])
    col = COLORS.get(d, MUTED)
    axb.boxplot([v], positions=[i], widths=.55, showfliers=False,
                medianprops=dict(color=col, lw=2), boxprops=dict(color=col),
                whiskerprops=dict(color=col), capprops=dict(color=col))
    axb.scatter(np.full(v.size, i) + np.linspace(-.16, .16, v.size), v, s=7, color=col,
                alpha=.35, edgecolor="none", zorder=3)
    nz = summary[d]["n_zero"]
    axb.annotate(f"+{nz} to zero" if nz else "none to zero", (i, 0.015),
                 xycoords=("data", "axes fraction"), ha="center", va="bottom",
                 fontsize=7.5, color=col)
axb.axhline(0, color=MUTED, ls="--", lw=1.0)
axb.text(.02, .97, "above 0 = more spikes during stim", transform=axb.transAxes, va="top",
         fontsize=8, color=MUTED)
axb.set_xticks(range(len(dets)))
axb.set_xticklabels(dets, fontsize=9)
axb.set_ylabel("log2(ON rate / OFF rate), per channel")
# The title used to assert "the detectors disagree on the SIGN". That was true of the 600 s
# slices under the old preprocessing and is no longer true of anything: once all three share a
# median-filtered input, P1 gives 0.39/0.30/0.27 and P5 0.57/0.56/0.54. A figure whose title
# states the opposite of its own contents is worse than an unlabelled one, so it now describes
# the axis and lets the boxes speak.
axb.set_title("(b) effect size per channel -- below 0 is suppression during stim",
              fontsize=9, loc="left")
recessive(axb)


# ---------------------------------------------------------------- (c) segment time course
# Pooled rate in each ON and OFF segment, in time order, normalised to that detector's own
# whole-window mean so three different absolute rates can share an axis. A real stimulation
# effect dips in EVERY shaded block and recovers between them; a drifting recording does not.
#
# Denominator here is the segment's WALL-CLOCK seconds, not analysable seconds -- the npz
# stores clean time per condition, not per segment. That is acceptable only because the mask on
# this file is close to all-or-nothing per channel: across the 132 channels used, the 10th
# percentile keeps 95.7% of the ON time. It would NOT be acceptable on a file where the mask is
# graded, so the caveat is printed rather than hidden.
axc = fig.add_subplot(gs[1, 1:])
# A rate from a 6 s sliver is noise, and P1's window opens with one before stim starts. Drop
# segments too short to carry an estimate rather than plotting a spike of pure sampling error.
MIN_SEG_SEC = 20.0
_all = sorted([(a, b, True) for a, b in ON.runs] + [(a, b, False) for a, b in OFF.runs])
segs = [sg for sg in _all if sg[1] - sg[0] >= MIN_SEG_SEC]
if len(segs) < len(_all):
    print(f"[note] panel (c) drops {len(_all) - len(segs)} segment(s) under {MIN_SEG_SEC:g}s: "
          + ", ".join(f"{a:.0f}-{b:.0f}s" for a, b, _ in _all if b - a < MIN_SEG_SEC))
mids = [0.5 * (a + b) for a, b, _ in segs]
for a, b, is_on in _all:
    if is_on:
        axc.axvspan(a, b, color=ON_SHADE, alpha=.22, lw=0, zorder=0)
for d in dets:
    t = z[f"{d}_idx"] / fs
    keep_ch = BOTH[z[f"{d}_chan"]]
    t = np.sort(t[keep_ch])
    vals = [(np.searchsorted(t, b) - np.searchsorted(t, a)) / (b - a) for a, b, _ in segs]
    vals = np.array(vals) / np.mean(vals)
    axc.plot(mids, vals, "-o", color=COLORS.get(d, MUTED), lw=1.6, ms=6, label=d)
axc.axhline(1.0, color=MUTED, ls="--", lw=1.0)
axc.set_xlim(0, float(z["seconds"]))
axc.set_xlabel("time (s) -- shaded = stim ON")
axc.set_ylabel("segment rate / that detector's mean")
axc.set_title("(c) does every ON block dip, or is the recording just drifting?",
              fontsize=9, loc="left")
axc.legend(frameon=False, fontsize=8, ncol=len(dets))
recessive(axc)

rec = str(z["rec_id"]) if "rec_id" in z.files else NPZ.stem
fig.suptitle(f"What each detector says stimulation did | {rec}, {int(z['stim_hz'])} Hz, "
             f"{ON.T:.0f}s ON vs {OFF.T:.0f}s OFF, "
             f"{int(BOTH.sum())} of {n_chan} channels measurable in both\n"
             + "   ".join(f"{d} {summary[d]['pooled']:.2f}x" for d in dets), fontsize=11)

OUT = HERE / "figures" / "real" / NPZ.stem
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "stim_effect.png", dpi=130, bbox_inches="tight")
print(f"\n[saved] {OUT / 'stim_effect.png'}")
print("[note] panel (c) uses wall-clock seconds per SEGMENT (the npz stores analysable time "
      "per condition, not per segment); valid here because the mask is near all-or-nothing "
      "per channel -- 10th pct of the channels used keeps 95.7% of the ON time.")
plt.close(fig)
