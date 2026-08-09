"""
sdc.compare.tuned_vs_default
----------------------------
What did re-tuning the operating points actually DO to P1 and P5?

    .venv\\Scripts\\python.exe -m sdc.compare.tuned_vs_default

Reads runs/<rec>.npz and runs/<rec>_tuned.npz only -- no detector is re-run.

THE TWO SETTINGS
  default   Janca k1=3.65 (its published default), Delphos Spk_thr=50 (its agreed operating
            point), and Barkmeier TAMP=1200 -- which is NOT a default. It was hand-set to match
            Janca's detection count on P1 (compare_spikes.py:269). seeg's own default is 800 and
            the paper's is 600.
  tuned     the BIDS operating points: k1=4.482, TAMP=890, Spk_thr=46.2, each interpolated to
            give 3.5 det/chan-min pooled over 25 subjects and 852 expert-marked IEDs.

  So this is not "defaults vs tuned" so much as "anchored to Janca vs anchored to expert marks",
  and the panels are arranged to show that the two anchors disagree in a structured way rather
  than by a scale factor.

WHY A COUNT CHANGE IS NOT THE INTERESTING NUMBER
  Panel (a) is the easy question and (c) is the real one. A detector can lose 38% of its
  detections by dropping the weakest 38%, or by returning a substantially different set at a
  similar size. Only matching the two runs event by event distinguishes those, and they have
  completely different implications for every agreement number computed downstream.

  It found a difference between the detectors that the counts hide entirely:

    Janca      keeps 58-61%, adds ~1%   -> a clean NESTED SUBSET. Raising k1 dropped the
                                           weakest detections and did nothing else.
    Barkmeier  keeps 98-99%, adds 25-44% -> a clean nested SUPERSET, the same in reverse.
    Delphos    keeps 89-93%, adds 14-18% -> NOT nested. Its count rose 8%, but it discarded
                                           ~10% of what it previously found and replaced it.

  Delphos moving a threshold is therefore not the monotone operation it is for the other two,
  and its ~10% churn is larger than the ~4% run-to-run nondeterminism already documented in
  Delphos.md. How much is threshold and how much is the RAM-dependent internal tiling cannot be
  separated from this figure -- that needs the repeat-run determinism check on sim data.
"""
import numpy as np
import matplotlib.pyplot as plt

from seeg._style import RED, BLUE, MUTED, GRID, recessive

from sdc.common import cond
from sdc.common.paths import ROOT, figdir
from sdc.common.spike_match import match

RUNS = ROOT / "runs"
TOL_S = 0.050            # the same match radius as every other comparison in this repo
TARGET_RATE = 3.5        # what the tuned points were fitted to, on the BIDS cohort

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}
RECS = [("P1_pre", "P1\nbaseline"), ("P1_stim", "P1\nstim"),
        ("P5_pre", "P5\nbaseline"), ("P5_stim", "P5\nstim")]


def load(stem, tag, label="all"):
    z = np.load(RUNS / f"{stem}{tag}.npz", allow_pickle=False)
    fs, n = float(z["fs"]), len(z["names"])
    sel = cond.select(z, label)
    out = {}
    for d in [str(s) for s in z["detectors"]]:
        keep = sel.keep(d)
        idx, ch = z[f"{d}_idx"][keep], z[f"{d}_chan"][keep]
        spikes = [np.sort(idx[ch == c] / fs) for c in range(n)]
        out[d] = dict(spikes=spikes, n=int(keep.sum()),
                      rate=sel.rate([s.size for s in spikes]) * 60.0)   # det/chan-MINUTE
    return out, [str(s) for s in z["names"]]


def overlap(a, b):
    """Per-channel match between the two settings, pooled.

    Returns (retained, added): the fraction of DEFAULT detections that survive into the tuned
    run, and the fraction of TUNED detections that were not in the default run. These are two
    different denominators on purpose -- with counts moving by -40% to +74% a single symmetric
    'agreement' number would be dominated by whichever run is larger."""
    m = na = nb = 0
    for sa, sb in zip(a, b):
        na += sa.size
        nb += sb.size
        if sa.size and sb.size:
            m += int(match(sa, sb, TOL_S)[0].sum())
    return (m / na if na else np.nan), ((nb - m) / nb if nb else np.nan)


# ----------------------------------------------------------------------
D, T, NAMES = {}, {}, {}
for stem, _ in RECS:
    D[stem], NAMES[stem] = load(stem, "")
    T[stem], _n = load(stem, "_tuned")
    if _n != NAMES[stem]:
        raise SystemExit(f"{stem}: channel names differ between the two runs; not comparable")

DETS = list(D[RECS[0][0]])
x = np.arange(len(RECS))
w = 0.26

fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
(ax1, ax2), (ax3, ax4) = axes


# ---- (a) how many detections moved --------------------------------------------------------
print("(a) detection count, default -> tuned")
print(f"{'recording':<10}{'detector':<11}{'default':>9}{'tuned':>9}{'change':>9}")
for j, d in enumerate(DETS):
    ys = [100 * (T[s][d]["n"] - D[s][d]["n"]) / max(D[s][d]["n"], 1) for s, _ in RECS]
    ax1.bar(x + (j - 1) * w, ys, w * 0.9, color=COLORS[d], label=d)
    for s, _ in RECS:
        print(f"{s:<10}{d:<11}{D[s][d]['n']:>9}{T[s][d]['n']:>9}"
              f"{100 * (T[s][d]['n'] - D[s][d]['n']) / max(D[s][d]['n'], 1):>+8.0f}%")
ax1.axhline(0, color="0.3", lw=1.0)
ax1.set_ylabel("change in detection count (%)")
ax1.set_title("(a) Janca and Barkmeier move in OPPOSITE directions -- which is what dropping\n"
              "the count-matching predicts, not a side effect of it", fontsize=9, loc="left")
ax1.legend(frameon=False, fontsize=8, ncol=3)


# ---- (b) did the tuning transfer? ---------------------------------------------------------
# The tuned points were fitted to give TARGET_RATE on the BIDS cohort. Whether they give it
# HERE is the whole question of transfer, and the answer is no.
print(f"\n(b) median per-channel rate (det/chan-min); target was {TARGET_RATE:g}")
print(f"{'recording':<10}{'detector':<11}{'default':>9}{'tuned':>9}")
for j, d in enumerate(DETS):
    lo = [np.nanmedian(D[s][d]["rate"][np.isfinite(D[s][d]["rate"])]) for s, _ in RECS]
    hi = [np.nanmedian(T[s][d]["rate"][np.isfinite(T[s][d]["rate"])]) for s, _ in RECS]
    xs = x + (j - 1) * w
    ax2.vlines(xs, lo, hi, color=COLORS[d], lw=1.6, zorder=2)
    ax2.scatter(xs, lo, s=34, facecolor="white", edgecolor=COLORS[d], lw=1.6, zorder=3)
    ax2.scatter(xs, hi, s=40, color=COLORS[d], zorder=3, label=d)
    for s, l, h in zip([s for s, _ in RECS], lo, hi):
        print(f"{s:<10}{d:<11}{l:>9.2f}{h:>9.2f}")
ax2.axhline(TARGET_RATE, color="0.35", ls="--", lw=1.2)
ax2.annotate(f"the rate the tuned points were fitted to ({TARGET_RATE:g})",
             (len(RECS) - 0.45, TARGET_RATE + 0.12), fontsize=7.5, color="0.35", ha="right")
ax2.set_ylabel("median per-channel rate (det/chan-min)")
ax2.set_title("(b) open = default, filled = tuned. Fitted to 3.5 on the BIDS cohort, and it\n"
              "does NOT land there here -- a cohort fit is a cohort statement", fontsize=9,
              loc="left")
ax2.legend(frameon=False, fontsize=8, ncol=3)


# ---- (c) is it the SAME detections? --------------------------------------------------------
print(f"\n(c) event-level overlap between the settings (tol {TOL_S * 1000:g} ms)")
print(f"{'recording':<10}{'detector':<11}{'kept':>8}{'new':>8}")
for j, d in enumerate(DETS):
    kept, new = zip(*[overlap(D[s][d]["spikes"], T[s][d]["spikes"]) for s, _ in RECS])
    xs = x + (j - 1) * w
    ax3.bar(xs, np.array(kept) * 100, w * 0.9, color=COLORS[d], label=f"{d}: kept")
    ax3.bar(xs, -np.array(new) * 100, w * 0.9, color=COLORS[d], alpha=.42,
            label=f"{d}: new")
    for s, k, nw in zip([s for s, _ in RECS], kept, new):
        print(f"{s:<10}{d:<11}{k:>7.0%}{nw:>8.0%}")
ax3.axhline(0, color="0.3", lw=1.0)
ax3.set_ylabel("% of default kept (up)   /   % of tuned that is new (down)")
ax3.set_title("(c) Janca and Barkmeier NEST -- pure subset / pure superset, ~1% new. Delphos\n"
              "does NOT: it swaps ~10% of its detections for different ones", fontsize=9,
              loc="left")
ax3.legend(frameon=False, fontsize=7, ncol=3)


# ---- (d) does the stim finding move? -------------------------------------------------------
# Finding 8. The three detectors disagreed about the SIGN of the stimulation effect at the old
# operating points. If that disagreement were an artefact of where the thresholds sat, re-tuning
# would move it. It does not.
print("\n(d) stim ON/OFF ratio of median per-channel rate (finding 8)")
print(f"{'recording':<10}{'detector':<11}{'default':>9}{'tuned':>9}")
stim = [s for s, _ in RECS if s.endswith("_stim")]
xs2 = np.arange(len(stim))
for j, d in enumerate(DETS):
    lo, hi = [], []
    for s in stim:
        vals = []
        for tag in ("", "_tuned"):
            on = load(s, tag, "on")[0][d]["rate"]
            off = load(s, tag, "off")[0][d]["rate"]
            on, off = np.nanmedian(on[np.isfinite(on)]), np.nanmedian(off[np.isfinite(off)])
            vals.append(on / off if off else np.nan)
        lo.append(vals[0]); hi.append(vals[1])
        print(f"{s:<10}{d:<11}{vals[0]:>9.2f}{vals[1]:>9.2f}")
    xx = xs2 + (j - 1) * w
    ax4.vlines(xx, lo, hi, color=COLORS[d], lw=1.6, zorder=2)
    ax4.scatter(xx, lo, s=34, facecolor="white", edgecolor=COLORS[d], lw=1.6, zorder=3)
    ax4.scatter(xx, hi, s=40, color=COLORS[d], zorder=3, label=d)
ax4.axhline(1.0, color="0.35", ls="--", lw=1.2)
ax4.annotate("no effect", (len(stim) - 0.5, 1.02), fontsize=7.5, color="0.35", ha="right")
ax4.set_xticks(xs2)
ax4.set_xticklabels([s.replace("_stim", " stim") for s in stim], fontsize=8)
ax4.set_ylabel("stim ON / stim OFF rate")
ax4.set_title("(d) finding 8 survives the re-tune: the three still disagree about the SIGN,\n"
              "so that disagreement is not an artefact of where the thresholds sat",
              fontsize=9, loc="left")
ax4.legend(frameon=False, fontsize=8, ncol=3)

for ax in (ax1, ax2, ax3):
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in RECS], fontsize=8)
for ax in axes.ravel():
    ax.grid(axis="y", alpha=.3)
    recessive(ax)

fig.suptitle("Re-anchoring the operating points from Janca's count to 852 expert marks:  "
             "k1 3.65->4.48,  TAMP 1200->890,  Spk_thr 50->46.2", fontsize=11)
fig.tight_layout()
out = figdir("real") / "tuned_vs_default.png"
fig.savefig(out, dpi=130)
plt.close(fig)
print(f"\n[saved] {out}")
