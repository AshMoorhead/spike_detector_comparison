"""
sdc.artefact.nullcheck
----------------------
Which ON/OFF estimator should we believe? Answered against data where the true answer is known.

    .venv\\Scripts\\python.exe -m sdc.artefact.nullcheck

THE IDEA
  P1_pre and P5_pre contain NO stimulation. Impose the corresponding stim file's ON/OFF block
  pattern on one of them and the correct answer is exactly "no effect" -- every estimator
  should return its null (1.0 for ratios, 0.0 for differences). Anything else it returns is
  its own bias, measured on real seEG with real channel heterogeneity, real artefact masking
  and real detector behaviour. A simulation could not test any of that.

  The pattern is imposed by circularly shifting the stim recording's own per-second ON mask to
  a random phase and cropping it to the baseline's length. Circular shifting preserves the
  exact duty cycle and the exact block-length distribution, so the pseudo-split differs from
  the real one only in WHERE the blocks fall -- which is the one thing that has to be arbitrary
  for this to be a null.

THREE THINGS IT MEASURES, and the third is the one that matters
  bias      median estimate across random splits, minus the null. A systematic offset.
  spread    2.5-97.5% of those estimates. How much an estimator moves on pure noise.
  FPR       how often the estimator's OWN 95% bootstrap CI excludes the null. This is a
            calibration check: an estimator claiming 95% intervals should be wrong 5% of the
            time. If it is wrong 30% of the time, its confidence intervals are fiction and
            every significance claim made with it -- including the ones in this project --
            is overstated.

  Bias and spread are properties of the point estimate; FPR is a property of the whole
  inferential apparatus, and it is possible to have a beautifully unbiased estimator with a
  badly miscalibrated interval. That combination would be the worst case, because it looks
  right in every summary table.

CAVEAT: a baseline recording still has structure -- drift, sleep-stage changes, arousals --
so a "false positive" here may be a real change in the brain that happens to align with the
imposed block pattern. That does not weaken the test: those same slow changes are present in
the stim recordings too, so an estimator that mistakes them for stimulation there will mistake
them for stimulation here. That is precisely what we want counted.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from seeg._style import RED, BLUE, MUTED, recessive

from sdc.common.paths import RUNS, figdir
from sdc.artefact import ratio_metrics as rm

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}

PAIRS = [("P1_pre", "P1_stim"), ("P5_pre", "P5_stim")]
N_SPLIT = 200         # random pseudo-ON placements
N_BOOT = 250          # bootstrap draws inside each split, for the calibration check
SEED = 0


def pseudo_masks(stim_rec, n_sec, n_split=N_SPLIT, seed=SEED):
    """`n_split` per-second ON masks of length `n_sec`, from a stim recording's own pattern.

    Circular shift then crop: duty cycle and block lengths are preserved exactly, only the
    phase is random. Tiles first if the baseline is longer than the stim recording, so the
    pattern never runs out.
    """
    z = np.load(RUNS / f"{stim_rec}.npz", allow_pickle=False)
    if "on_per_sec" not in z.files:
        raise SystemExit(f"{stim_rec} has no on_per_sec -- re-run compare_spikes.py.")
    pat = np.asarray(z["on_per_sec"], bool)
    if pat.size < n_sec:
        pat = np.tile(pat, int(np.ceil(n_sec / pat.size)))
    rng = np.random.default_rng(seed)
    offs = rng.integers(0, pat.size, size=n_split)
    return [np.roll(pat, int(o))[:n_sec] for o in offs], float(pat.mean())


def run_null(pre_rec, stim_rec, n_split=N_SPLIT, n_boot=N_BOOT):
    """Every estimator x every detector across `n_split` pseudo-splits of a baseline file."""
    z = np.load(RUNS / f"{pre_rec}.npz", allow_pickle=False)
    n_sec = int(z["clean_per_sec"].shape[0])
    masks, duty = pseudo_masks(stim_rec, n_sec, n_split)

    print(f"\n=== {pre_rec}: {n_sec}s of stim-FREE recording, "
          f"{n_split} pseudo-splits using {stim_rec}'s block pattern (duty {duty:.0%})")

    acc = {}          # (estimator, detector) -> {"pts": [...], "reject": [...]}
    n_used = []
    for k, m in enumerate(masks):
        p = rm.paired(z, on_sec_mask=m)
        if p.n < 10:
            continue
        n_used.append(p.n)
        dr = rm.draws(p.n, n_boot, seed=1000 + k)
        for est in rm.ESTIMATORS:
            for d in p.det:
                pt, lo, hi = rm.ci(est, p.det[d], dr)
                a = acc.setdefault((est.name, d), {"pts": [], "reject": [], "width": []})
                a["pts"].append(pt)
                a["reject"].append(not (lo < est.null < hi))
                # On the log scale for ratios, so a width is a fold-range and is directly
                # comparable to the across-split spread computed the same way.
                a["width"].append(np.log10(hi / lo) if est.null == 1.0 else hi - lo)

    print(f"    {len(n_used)} usable splits, {int(np.median(n_used))} channels each (median)")
    print(f"\n    {'estimator':<21}{'detector':<11}{'null':>7}{'median':>9}"
          f"{'bias':>9}{'2.5-97.5%':>20}{'FPR':>7}")
    for est in rm.ESTIMATORS:
        for d in COLORS:
            a = acc.get((est.name, d))
            if not a:
                continue
            pts = np.array(a["pts"], float)
            pts = pts[np.isfinite(pts)]
            med = float(np.median(pts))
            bias = med / est.null if est.null else med - est.null
            lo, hi = np.percentile(pts, [2.5, 97.5])
            fpr = float(np.mean(a["reject"]))
            warn = "  <-- MISCALIBRATED" if fpr > 0.15 else ("  <-- high" if fpr > 0.10 else "")
            print(f"    {est.name:<21}{d:<11}{est.null:>7.2g}{med:>9.3f}"
                  f"{bias:>9.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}{fpr:>7.0%}{warn}")

    # ---- where the uncertainty actually comes from ------------------------------------
    # Both quantities are measured on the SAME splits of the SAME recording, so this is a
    # like-for-like decomposition rather than an inference across two different files:
    #   channel   median width of the within-split bootstrap CI  -- between-channel variance
    #   total     spread of the point estimate ACROSS splits     -- everything
    # Treating them as independent, var(total) = var(channel) + var(temporal), so the share
    # of variance the channel bootstrap can even see is (channel/total)^2.
    print(f"\n    WHERE THE UNCERTAINTY COMES FROM (95% widths, log10 for ratios)")
    print(f"    {'estimator':<21}{'detector':<11}{'channel':>9}{'total':>9}{'total/chan':>12}"
          f"{'% var seen by':>15}")
    for est in rm.ESTIMATORS:
        for d in COLORS:
            a = acc.get((est.name, d))
            if not a:
                continue
            pts = np.array(a["pts"], float)
            pts = pts[np.isfinite(pts)]
            if est.null == 1.0:
                pts = np.log10(pts[pts > 0])
            wl, wh = np.percentile(pts, [2.5, 97.5])
            total = float(wh - wl)
            chan = float(np.median(a["width"]))
            share = min(1.0, (chan / total) ** 2) if total > 0 else np.nan
            print(f"    {est.name:<21}{d:<11}{chan:>9.3f}{total:>9.3f}"
                  f"{total / max(chan, 1e-9):>11.2f}x{share:>14.0%}")
    return acc


def calibrated(pre_rec, stim_rec, acc):
    """The observed stimulation effect, judged against the stim-free null distribution.

    This is the inference to actually use, and it exists because `run_null` showed the
    bootstrap-over-channels CI is not usable: it rejects a true null 46-100% of the time
    rather than 5%. The cause is that resampling channels can only see BETWEEN-CHANNEL
    variability, while the dominant noise here is TEMPORAL -- which particular minutes of
    recording happened to fall inside an ON block. Every channel shares those same minutes, so
    their errors are correlated and no amount of channel resampling can measure it. Adding
    channels does not help; only more ON blocks would.

    So the reference distribution is taken empirically instead: the same estimator, the same
    detectors, the same masking, the same block structure, on a recording where nothing
    happened. The p-value is the two-sided fraction of that distribution at least as extreme
    as what the stim recording gave.

    Its own limitation, stated rather than hidden: the null comes from a DIFFERENT recording
    (the baseline), so it carries that recording's own drift and channel count rather than the
    stim recording's. It is a calibration reference, not a permutation of the actual data.
    """
    p = rm.load(stim_rec)
    dr = rm.draws(p.n)
    print(f"\n=== {stim_rec} judged against the {pre_rec} null "
          f"({p.n} channels; naive CI shown only to size how wrong it is)")
    print(f"    {'estimator':<21}{'detector':<11}{'observed':>9}{'null 2.5-97.5%':>22}"
          f"{'p':>8}   {'naive bootstrap CI':<22}")
    for est in rm.ESTIMATORS:
        for d in COLORS:
            a = acc.get((est.name, d))
            if not a or d not in p.det:
                continue
            null = np.array(a["pts"], float)
            null = null[np.isfinite(null)]
            obs, blo, bhi = rm.ci(est, p.det[d], dr)
            pv = 2 * min(float((null <= obs).mean()), float((null >= obs).mean()))
            pv = min(pv, 1.0)
            lo, hi = np.percentile(null, [2.5, 97.5])
            naive_sig = "*" if not (blo < est.null < bhi) else " "
            print(f"    {est.name:<21}{d:<11}{obs:>9.3f}{f'[{lo:.3f}, {hi:.3f}]':>22}"
                  f"{pv:>8.3f}{'  SIG' if pv < 0.05 else '  ns '}"
                  f"   [{blo:.3f}, {bhi:.3f}]{naive_sig}")


def figure(results):
    """One panel per estimator: the null distribution of each detector's estimate."""
    ests = rm.ESTIMATORS
    fig, axes = plt.subplots(len(PAIRS), len(ests),
                             figsize=(3.0 * len(ests), 3.4 * len(PAIRS)), squeeze=False)
    for i, (pre, _stim) in enumerate(PAIRS):
        for j, est in enumerate(ests):
            ax = axes[i][j]
            for d, c in COLORS.items():
                a = results[pre].get((est.name, d))
                if not a:
                    continue
                pts = np.array(a["pts"], float)
                pts = pts[np.isfinite(pts)]
                if est.null == 1.0:
                    pts = np.log10(pts[pts > 0])
                ax.hist(pts, bins=28, histtype="step", lw=1.5, color=c, label=d)
            ax.axvline(0.0, color="0.25", ls="--", lw=1.2)
            recessive(ax)
            ax.grid(alpha=.25)
            ax.set_yticks([])
            if i == 0:
                ax.set_title(est.name, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{pre}\n(no stimulation)", fontsize=9)
            ax.set_xlabel("log10 ratio" if est.null == 1.0 else est.unit, fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("Every estimator on STIM-FREE data with a stimulation block pattern imposed. "
                 "The true answer is the dashed line.\nSpread = how much the estimator moves "
                 "on noise alone; offset from the line = its bias.", fontsize=10)
    fig.tight_layout()
    out = figdir("real") / "estimator_null.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    pairs = PAIRS if len(sys.argv) < 3 else [(sys.argv[1], sys.argv[2])]
    res = {}
    for pre, stim in pairs:
        res[pre] = run_null(pre, stim)
        calibrated(pre, stim, res[pre])
    if len(res) == len(PAIRS):
        figure(res)
