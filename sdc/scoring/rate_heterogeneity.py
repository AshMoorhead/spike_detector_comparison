"""
sdc.scoring.rate_heterogeneity
------------------------------
Do the detectors TRACK real between-patient differences in spike rate, or flatten them?

    .venv\\Scripts\\python.exe -m sdc.scoring.rate_heterogeneity

THE QUESTION THIS ANSWERS, AND THE ASSUMPTION IT REPLACES
  Tuning picks one parameter per detector so the POOLED output is 3.5 det/chan-min over 25
  subjects, and operating_points (b) then reports a 7-11x spread in what individual subjects
  actually give. That spread has been treated as the COST of a single global threshold.

  It is only a cost if the subjects really do have similar rates. They do not: the expert marks
  themselves span 48x (p10-p90) and 654x end to end. So the right question is not "how much
  does the achieved rate vary" but "does it vary the RIGHT amount, in the RIGHT direction".

  Under the assumption that between-patient rate differences are real, a detector should track
  them 1:1 -- twice the spikes, twice the detections. The slope of log(achieved detection rate)
  on log(expert mark rate) measures exactly that. 1.0 = tracks fully. 0.0 = ignores the patient
  entirely and emits at its own fixed rate.

WHY THIS FLIPS THE CONCLUSION ABOUT PER-PATIENT TUNING
  If the spread were noise, normalising each patient to a common rate would be the fix. It is
  not noise -- it is an UNDER-response, 7-11x against a true 48x -- so normalising per patient
  would drive the slope toward 0 and delete real biology. The failure mode is the opposite of
  the one the single-threshold framing suggests.

THE CONFOUND, STATED PLAINLY
  "Expert mark rate" is marks per channel-minute, and marks reflect both the true spike rate AND
  how exhaustively that recording was marked. Four subjects have fewer than 10 events. A subject
  with 3 marks may be quiet or may be sparsely annotated, and nothing in the dataset separates
  those. So 48x is a measurement of the MARKS, and the slope inherits that. It is still the
  right comparison BETWEEN detectors, because all three are scored against the same marks.
"""
import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from seeg._style import RED, BLUE, MUTED, recessive

from sdc.common.paths import figdir
from sdc.scoring.bids_events import subjects, load_subject
from sdc.scoring.pick_operating_point import (matrix, budget_recall, value_for_budget,
                                              per_subject_at)
from sdc.scoring.sweep_labelled import BIDS_ROOT, GRIDS, LABEL, SWEEPS

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}
MATCH_RATE = 3.5
N_BOOT = 2000
BOOT_SEED = 0


def expert_rate(subs):
    """Expert marks per channel-minute, per subject. Seconds comes from a sweep npz rather than
    the BIDS sidecar -- whatever the EDF says wins, which is the rule everywhere else here."""
    out = []
    for s in subs:
        f = sorted(glob.glob(str(SWEEPS / f"bids_{s}_janca_*.npz")))
        if not f:
            raise SystemExit(f"no sweep run for {s}; run sweep_labelled first")
        secs = float(np.load(f[0], allow_pickle=False)["seconds"])
        d = load_subject(BIDS_ROOT, s)
        out.append(d["n_events"] / (len(d["channels"]) * secs / 60.0))
    return np.array(out)


def _draws(n):
    """ONE fixed set of resamples, shared by every detector. That is what makes the slope
    DIFFERENCES paired: comparing two marginal CIs asks whether each slope is separately
    pinned down, which with 25 subjects it is not, and would hide a difference that is
    consistent draw by draw. Same distinction as labelled_report (a) vs (d)."""
    return np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n))


def slopes(x, y, draws):
    """log-log slope and its bootstrap distribution over SUBJECTS (the unit of the claim)."""
    lx, ly = np.log10(x), np.log10(y)
    return float(np.polyfit(lx, ly, 1)[0]), \
        np.array([np.polyfit(lx[d], ly[d], 1)[0] for d in draws])


def main():
    subs = subjects(BIDS_ROOT)
    dets = [d for d in GRIDS if (SWEEPS / f"curve_{d}.npy").is_file()]
    M = {d: matrix(d, subs) for d in dets}
    allrows = np.arange(len(subs))
    truth = expert_rate(subs)

    lo, hi = np.percentile(truth, [10, 90])
    print(f"EXPERT marks/chan-min: p10-p90 {lo:.3f}-{hi:.3f} ({hi / lo:.0f}x), "
          f"min-max {truth.min():.3f}-{truth.max():.3f}")
    print(f"{'detector':<11}{'achieved span':>15}{'slope':>8}{'  95% CI':>16}"
          f"{'  rho':>7}{'  micro':>8}{'  macro':>8}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    rows = {}
    # Every detector is fitted on the SAME subjects with the SAME resamples, so the differences
    # below are paired. `keep` fixes the subject set too -- a subject dropped for one detector
    # (zero detections at the tuned point) must be dropped for all, or the slopes are fitted on
    # different cohorts and are not comparable at all.
    rates_all = {}
    for d in dets:
        values, tp, nt, nd, cm = M[d]
        b, r = budget_recall(values, tp, nt, nd, cm, allrows)
        v = value_for_budget(values, b, MATCH_RATE)
        rates_all[d] = per_subject_at(values, tp, nt, nd, cm, v)[1]
    keep = (truth > 0) & np.all([rates_all[d] > 0 for d in dets], axis=0)
    print(f"[fit] {int(keep.sum())}/{len(subs)} subjects usable for the slope "
          f"(nonzero expert marks and nonzero detections for all three)")
    draws = _draws(int(keep.sum()))

    for d in dets:
        values, tp, nt, nd, cm = M[d]
        b, r = budget_recall(values, tp, nt, nd, cm, allrows)
        v = value_for_budget(values, b, MATCH_RATE)
        rec, rate = per_subject_at(values, tp, nt, nd, cm, v)
        sl, dist = slopes(truth[keep], rate[keep], draws)
        cl, ch = np.percentile(dist, [2.5, 97.5])
        rl, rh = np.percentile(rate, [10, 90])
        rho = spearmanr(truth, rate).statistic
        micro = float(np.interp(v, np.sort(values), r[np.argsort(values)]))
        rows[d] = dict(rate=rate, rec=rec, slope=sl, ci=(cl, ch), rho=rho, dist=dist,
                       micro=micro, macro=float(rec.mean()))
        print(f"{LABEL[d]:<11}{f'{rl:.2f}-{rh:.2f}':>15}{sl:>8.2f}"
              f"{f'{cl:+.2f} to {ch:+.2f}':>16}{rho:>7.2f}{micro:>8.3f}{rec.mean():>8.3f}")

        c = COLORS[LABEL[d]]
        ax.scatter(truth[keep], rate[keep], s=26, color=c, alpha=.7, edgecolor="none",
                   label=LABEL[d])
        xs = np.array([truth[keep].min(), truth[keep].max()])
        a, bb = np.polyfit(np.log10(truth[keep]), np.log10(rate[keep]), 1)
        ax.plot(xs, 10 ** (a * np.log10(xs) + bb), color=c, lw=1.6)

    # 1:1 reference anchored at the medians -- what "tracks the patient exactly" would look like.
    xs = np.array([truth[keep].min(), truth[keep].max()])
    anchor = np.median([np.median(rows[d]["rate"][keep]) for d in dets]) / np.median(truth[keep])
    ax.plot(xs, anchor * xs, color="0.35", ls="--", lw=1.3)
    ax.annotate("slope 1 = tracks the patient exactly", (xs[1], anchor * xs[1]),
                fontsize=7.5, color="0.35", ha="right", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("expert marks per channel-minute (log)")
    ax.set_ylabel("achieved detection rate at the tuned point (log)")
    ax.set_title("(a) one dot = one subject. Flatter than the dashed line means the detector\n"
                 "UNDER-responds to how busy the patient actually is", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    recessive(ax)

    # ---- (b) the slopes, with CIs ---------------------------------------------------------
    ax = axes[1]
    for j, d in enumerate(dets):
        c = COLORS[LABEL[d]]
        cl, ch = rows[d]["ci"]
        ax.plot([j, j], [cl, ch], color=c, lw=2.4)
        ax.plot([j], [rows[d]["slope"]], "o", ms=9, color=c)
    ax.axhline(1.0, color="0.35", ls="--", lw=1.2)
    ax.annotate("tracks fully", (len(dets) - .5, 1.02), fontsize=7.5, color="0.35", ha="right")
    ax.axhline(0.0, color=MUTED, lw=1.0)
    ax.annotate("ignores the patient", (len(dets) - .5, 0.02), fontsize=7.5, color=MUTED,
                ha="right")
    ax.set_xticks(range(len(dets)))
    ax.set_xticklabels([LABEL[d] for d in dets], fontsize=9)
    ax.set_ylabel("slope of log(detected) on log(expert)")
    # PAIRED differences, printed rather than eyeballed off the overlapping CIs above. All three
    # intervals overlap heavily, so the marginal CIs cannot rank the detectors; whether the
    # difference is consistent draw by draw is a different question and the only one that can be
    # answered with 25 subjects.
    print(f"\nPAIRED slope differences (same subjects, same {N_BOOT} resamples)")
    print(f"{'pair':<24}{'diff':>8}{'  95% CI':>18}{'  consistent':>13}")
    parts = []
    for i, da in enumerate(dets):
        for db in dets[i + 1:]:
            dd = rows[da]["dist"] - rows[db]["dist"]
            cl2, ch2 = np.percentile(dd, [2.5, 97.5])
            frac = float((dd > 0).mean())
            print(f"{LABEL[da][:4]} - {LABEL[db][:4]:<17}{rows[da]['slope'] - rows[db]['slope']:>+8.2f}"
                  f"{f'{cl2:+.2f} to {ch2:+.2f}':>18}{f'{frac:.0%}':>13}")
            parts.append(f"{LABEL[da][:4]}-{LABEL[db][:4]} {frac:.0%}")
    ax.set_ylim(-0.22, None)
    ax.annotate("paired, fraction of resamples with a positive difference:\n" + ",  ".join(parts),
                (0.5, 0.02), xycoords="axes fraction", fontsize=7, color=MUTED, ha="center")
    ax.set_title("(b) all three compress real differences 2.5-4x. The CIs OVERLAP, so these\n"
                 "cannot be ranked from this panel alone -- see the paired test below",
                 fontsize=9, loc="left")
    recessive(ax)

    # ---- (c) does the estimator choice change the answer? ----------------------------------
    # Under "rates really differ", the pooled (micro) recall is the wrong average: it weights
    # each subject by its event count, so the busy ones set the number. Macro gives every
    # subject one vote. Shown because it is the obvious objection, and the answer is that it
    # barely matters HERE -- which is worth knowing rather than assuming.
    ax = axes[2]
    w = 0.36
    xs = np.arange(len(dets))
    ax.bar(xs - w / 2, [rows[d]["micro"] for d in dets], w,
           color=[COLORS[LABEL[d]] for d in dets], label="micro (pooled)")
    ax.bar(xs + w / 2, [rows[d]["macro"] for d in dets], w, alpha=.45,
           color=[COLORS[LABEL[d]] for d in dets], label="macro (per subject)")
    for j, d in enumerate(dets):
        ax.annotate(f"{rows[d]['macro'] - rows[d]['micro']:+.3f}",
                    (j, max(rows[d]["micro"], rows[d]["macro"]) + .012), fontsize=7.5,
                    ha="center", color=COLORS[LABEL[d]])
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[d] for d in dets], fontsize=9)
    ax.set_ylabel(f"recall at {MATCH_RATE:g} det/chan-min")
    ax.set_title("(c) solid = pooled (busy subjects dominate), faded = one vote per subject.\n"
                 "The ranking does not change, so the headline numbers survive either way",
                 fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    recessive(ax)

    fig.suptitle("Do the detectors track how busy a patient really is?   "
                 f"expert marks span {hi / lo:.0f}x across 25 subjects; the detectors span "
                 f"7-11x", fontsize=11)
    fig.tight_layout()
    out = figdir("labelled") / "rate_heterogeneity.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
