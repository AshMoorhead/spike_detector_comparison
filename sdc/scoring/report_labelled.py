"""
sdc.scoring.report_labelled
---------------------------
The labelled-benchmark result: four panels, each answering a question the others cannot.

    .venv\\Scripts\\python.exe -m sdc.scoring.report_labelled

  (a) recall vs DETECTION RATE, out to the CEILING. Compare vertically, at equal output. The
      plateau is the point: where the curve flattens, extra detections buy no extra recall, so
      whatever is still missed is unreachable at ANY threshold -- a statement about what the
      detector can SEE rather than where it is set. A curve that was still climbing when its
      sweep ended gets an ARROW instead of a knee ring, and no ceiling is reported for it: the
      first version of this panel ringed the last grid point of all three and read out
      "32% / 29% unreachable" for Barkmeier and Delphos, which was a fact about the grid.
  (b) LOSO parameter stability -- IQR and far-fold count, not max-min span.
  (c) per-subject recall, all three at the SAME detection rate.
  (d) PAIRED per-subject differences at that rate, one sub-row per pair. Each bar is COLOURED
      BY THE WINNER, so the direction never has to be decoded from an "A - B" label. The pooled
      gap is a mean; this shows how OFTEN each detector wins and by how much, which is the
      difference between "better" and "better on average because of four subjects".

  plus runs/sweeps/summary.csv, so numbers quoted elsewhere have a source that is not a commit
  message.

WHAT IS NOT HERE
  A held-out-recall bar chart: fitting on 10 subjects and reporting recall on 15 just re-reads
  panel (a). The per-subject ACHIEVED-RATE panel moved to operating_points.py, where it belongs
  -- it is a statement about operating points rather than about the benchmark.

TERMINOLOGY: "detection rate" = detections per channel-minute, the OUTPUT rate. Distinct from
"sensitivity", which here means recall against ground truth. They are the two axes of the same
curve, and conflating them is how a 26-point gap got reported that was really 5-8.
"""
import csv
import itertools

import numpy as np
import matplotlib.pyplot as plt

from seeg._style import RED, BLUE, MUTED, recessive

from sdc.common.paths import figdir
from sdc.scoring.bids_events import subjects
from sdc.scoring.pick_operating_point import (matrix, budget_recall, value_for_budget,
                                              per_subject_at)
from sdc.scoring.sweep_labelled import BIDS_ROOT, GRIDS, LABEL, SWEEPS

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}
MATCH_RATE = 3.5      # det/chan-min: mid-range, and where all three curves overlap
PLATEAU_TOL = 0.005   # recall gain per DOUBLING of rate below which the curve counts as flat


def plateau(values, b, r):
    """The knee of the curve, the recall there, and whether the curve ACTUALLY flattened.

    Defined on recall gain per DOUBLING of detection rate, not per grid step: the grids are not
    evenly spaced, so a per-step criterion would measure the grid rather than the detector.

    The third return value is the one that stops this being misread. If the last grid point is
    still gaining recall, the knee lands ON the end of the sweep and the top recall is simply
    where we ran out of grid -- not a ceiling, and not evidence of anything unreachable. Janca's
    grid was extended to k1=1.2 for exactly this reason; the other two have not been."""
    o = np.argsort(b)
    bb, rr = b[o], r[o]
    for k in range(len(bb) - 1, 0, -1):
        if bb[k - 1] <= 0:
            continue
        gain = (rr[k] - rr[k - 1]) / max(np.log2(bb[k] / bb[k - 1]), 1e-9)
        if gain > PLATEAU_TOL:
            return bb[k], rr[-1], k < len(bb) - 1
    return bb[0], rr[-1], True


def main():
    subs = subjects(BIDS_ROOT)
    dets = [d for d in GRIDS if (SWEEPS / f"curve_{d}.npy").is_file()]
    M = {d: matrix(d, subs) for d in dets}
    allrows = np.arange(len(subs))
    pairs = list(itertools.combinations(dets, 2))

    with open(SWEEPS / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["detector", "param", "value", "det_per_chan_min", "recall",
                    "precision_lower_bound", "tp", "n_true", "n_det"])
        for d in dets:
            values, tp, nt, nd, cm = M[d]
            b, r = budget_recall(values, tp, nt, nd, cm, allrows)
            for k, v in enumerate(values):
                w.writerow([LABEL[d], GRIDS[d][0], f"{v:g}", f"{b[k]:.3f}", f"{r[k]:.4f}",
                            f"{tp[:, k].sum() / max(nd[:, k].sum(), 1):.4f}",
                            int(tp[:, k].sum()), int(nt[:, k].sum()), int(nd[:, k].sum())])
    print(f"[saved] {SWEEPS / 'summary.csv'}")

    fig = plt.figure(figsize=(15.5, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.24)

    # ---- (a) recall vs detection rate, to the ceiling ------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    print("\nrecall CEILING -- where extra detections stop buying recall")
    print(f"{'detector':<11}{'top recall':>12}{'knee rate':>11}{'unreachable':>13}"
          f"{'  plateau reached?':>19}")
    for d in dets:
        values, tp, nt, nd, cm = M[d]
        b, r = budget_recall(values, tp, nt, nd, cm, allrows)
        o = np.argsort(b)
        knee, top, flat = plateau(values, b, r)
        # Every complete grid point is drawn -- the curve is the evidence, so truncating it or
        # decorating the end with an arrow both amount to editorialising over the data.
        ax.plot(b[o], r[o], "-o", ms=4.5, lw=1.9, color=COLORS[LABEL[d]],
                label=LABEL[d] + ("" if flat else "  (still rising at the end of its sweep)"))
        # The RING is the only mark, and it appears only where the curve genuinely flattened.
        # No ring means no ceiling was measured -- not a ceiling drawn at the last grid point.
        if flat:
            ax.plot([knee], [np.interp(knee, b[o], r[o])], "o", ms=13, mfc="none",
                    mec=COLORS[LABEL[d]], mew=1.8)
        print(f"{LABEL[d]:<11}{top:>12.3f}{knee:>11.1f}"
              f"{f'{1 - top:.0%}' if flat else 'n/a':>13}{'yes' if flat else 'NO':>19}")
    ax.axvline(MATCH_RATE, color="0.4", ls="--", lw=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("detection rate (detections per channel-minute, log)")
    ax.set_ylabel("recall vs expert marks")
    ax.set_title("(a) recall vs output, every grid point swept. RING = a real knee: beyond it\n"
                 "extra detections buy no recall, so what is missed is unreachable at any "
                 "threshold", fontsize=9, loc="left")
    # Which knob each curve is swept on. It belongs on the panel because a knee is a statement
    # about THAT KNOB, not about the detector: Barkmeier's TAMP saturates (mDetectSpike:148 only
    # filters a pool that STDCoeff at :116 has already fixed), so its flat top is where TAMP
    # stops binding rather than where Barkmeier stops seeing. See CEILING in sweep_labelled.
    ax.text(.01, .015, "swept: " + ",  ".join(f"{LABEL[d]} {GRIDS[d][0]}" for d in dets),
            transform=ax.transAxes, fontsize=7, color=MUTED)
    ax.legend(frameon=False, fontsize=9)
    recessive(ax)

    # ---- (b) LOSO parameter stability -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    print(f"\nLOSO PARAMETER stability at rate {MATCH_RATE:g}")
    print(f"{'detector':<11}{'median':>9}{'IQR':>8}{'beyond 5%':>12}")
    for j, d in enumerate(dets):
        values, tp, nt, nd, cm = M[d]
        picks = []
        for i in range(len(subs)):
            rows = np.array([k for k in range(len(subs)) if k != i])
            b, _ = budget_recall(values, tp, nt, nd, cm, rows)
            v = value_for_budget(values, b, MATCH_RATE)
            if np.isfinite(v):
                picks.append(v)
        picks = np.array(picks)
        if not picks.size:
            continue
        rel = 100 * (picks - np.median(picks)) / np.median(picks)
        ax.scatter(np.full(rel.size, j) + np.linspace(-.17, .17, rel.size), rel, s=24,
                   color=COLORS[LABEL[d]], alpha=.75, edgecolor="none", zorder=3)
        ax.plot([j - .3, j + .3], [0, 0], color=COLORS[LABEL[d]], lw=2.5)
        # IQR and far-fold count, NOT max-min: the span once made Delphos read as "21-27%
        # unstable" when its typical fold is within 1.5%. Same ordering on every statistic,
        # magnitude overstated by an order of magnitude.
        iqr = float(np.subtract(*np.percentile(rel, [75, 25])))
        far = int((np.abs(rel) > 5).sum())
        ax.annotate("IQR {:.1f}%\n{}/{} beyond 5%\nmed {:.3g}".format(
                        iqr, far, rel.size, float(np.median(picks))),
                    (j, -13.2), fontsize=7.5, ha="center", va="bottom",
                    color=COLORS[LABEL[d]])
        print(f"{LABEL[d]:<11}{np.median(picks):>9.3g}{iqr:>7.1f}%{f'{far}/{rel.size}':>12}")
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_ylim(-15, None)
    ax.set_xticks(range(len(dets)))
    ax.set_xticklabels([LABEL[d] for d in dets], fontsize=9)
    ax.set_ylabel("chosen parameter, % from median")
    ax.set_title("(b) PARAMETER stability across folds -- not recall.\n"
                 "Leave-one-subject-out, 25 folds", fontsize=9, loc="left")
    recessive(ax)

    # ---- per-subject recall at the matched rate, shared by (c) and (d) --------------------
    rec_at = {}
    for d in dets:
        values, tp, nt, nd, cm = M[d]
        b, _ = budget_recall(values, tp, nt, nd, cm, allrows)
        v = value_for_budget(values, b, MATCH_RATE)
        rec_at[LABEL[d]] = (per_subject_at(values, tp, nt, nd, cm, v)[0]
                            if np.isfinite(v) else np.full(len(subs), np.nan))
    difficulty = np.nanmean(np.array([rec_at[LABEL[d]] for d in dets]), axis=0)
    order = np.argsort(difficulty)          # hardest first; (c) and (d) share this ordering,
                                            # so a subject sits at the same x in both

    # ---- (c) per-subject recall -----------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for d in dets:
        ax.plot(np.arange(len(subs)), rec_at[LABEL[d]][order], "-o", ms=4, lw=1.3,
                color=COLORS[LABEL[d]], label=LABEL[d])
    ax.set_xticks(np.arange(len(subs)))
    ax.set_xticklabels([subs[i].replace("sub-", "") for i in order], fontsize=6.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("subject, hardest first")
    ax.set_ylabel("recall vs expert marks")
    ax.set_title("(c) per subject, all three at the SAME detection rate "
                 "({:g} det/chan-min).\nThe spread is BETWEEN-SUBJECT variation in the data, "
                 "not detector instability".format(MATCH_RATE), fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    recessive(ax)

    # ---- (d) paired differences, coloured by winner ---------------------------------------
    sub = gs[1, 1].subgridspec(len(pairs), 1, hspace=0.5)
    print(f"\nPAIRED per-subject differences at rate {MATCH_RATE:g}")
    print(f"{'pair':<24}{'wins':>12}{'median':>9}{'IQR':>20}")
    for i, (da, db) in enumerate(pairs):
        pa, pb = LABEL[da], LABEL[db]
        axp = fig.add_subplot(sub[i])
        diff = (rec_at[pa] - rec_at[pb])[order]
        # colour == winner, so sign and colour carry the same information and neither has to be
        # decoded from a label. The first version put all three pairs on one axis with an
        # "A - B" legend, which made "who won" genuinely unreadable.
        cols = [COLORS[pa] if v > 0 else COLORS[pb] for v in diff]
        axp.bar(np.arange(len(subs)), diff, 0.72, color=cols)
        axp.axhline(0, color="0.3", lw=1.0)
        lim = float(np.nanmax(np.abs(diff))) * 1.35 or 0.1
        axp.set_ylim(-lim, lim)
        axp.set_xticks([])
        axp.tick_params(labelsize=6)
        w = int(np.nansum(diff > 0))
        axp.text(0.004, 0.90, f"{pa} better  ({w}/{len(subs)})", transform=axp.transAxes,
                 fontsize=7.5, color=COLORS[pa], va="top", fontweight="bold")
        axp.text(0.004, 0.10, f"{pb} better  ({len(subs) - w}/{len(subs)})",
                 transform=axp.transAxes, fontsize=7.5, color=COLORS[pb], va="bottom",
                 fontweight="bold")
        if i == 0:
            axp.set_title("(d) PAIRED: same subjects, same detection rate. Bar height = "
                          "margin,\ncolour = winner. A pooled mean cannot say whether a lead "
                          "is consistent", fontsize=9, loc="left")
        if i == len(pairs) - 1:
            axp.set_xlabel("subject, hardest first (same order as (c))", fontsize=8)
        q1, q3 = np.nanpercentile(diff, [25, 75])
        print("{:<24}{:>12}{:>+9.3f}{:>20}".format(
            f"{pa[:4]} vs {pb[:4]}", f"{w}/{len(subs)}", float(np.nanmedian(diff)),
            f"{q1:+.3f} to {q3:+.3f}"))
        recessive(axp)

    fig.suptitle("Three detectors against 852 expert-marked IEDs, 25 subjects", fontsize=11)
    out = figdir("labelled") / "labelled_report.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
