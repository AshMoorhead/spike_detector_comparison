"""
sdc.scoring.report_labelled
---------------------------
The labelled-benchmark result as figures and a machine-readable table.

    .venv\\Scripts\\python.exe -m sdc.scoring.report_labelled

Four outputs, all from the sweep already on disk -- nothing here re-runs a detector:

  (a) recall vs DETECTION RATE, per detector, with the per-subject spread behind it. Compare
      VERTICALLY: at equal output, who finds the most expert marks.
  (b) LOSO fold distribution of the chosen parameter. This is the figure that carries the
      stability result; until now it existed only as three numbers in a log.
  (c) per-subject recall at a MATCHED detection rate. The earlier per-subject panel compared
      detectors at their defaults, i.e. at three different output rates, so it partly plotted
      the operating points rather than the detectors.
  (d) runs/sweeps/summary.csv -- every (detector, value) with rate, recall, precision, so the
      numbers quoted in the README have a source that is not a commit message.

TERMINOLOGY: "detection rate" = detections per channel-minute, the OUTPUT rate. Kept distinct
from "sensitivity", which in this repo means recall against ground truth. They are the two axes
of the same curve and conflating them is how the 26-point gap got reported.
"""
import csv

import numpy as np
import matplotlib.pyplot as plt

from seeg._style import RED, BLUE, MUTED, GRID, recessive

from sdc.common.paths import RUNS, figdir
from sdc.scoring.bids_events import subjects
from sdc.scoring.pick_operating_point import (matrix, budget_recall, value_for_budget,
                                              stratified_split, N_TRAIN, TARGETS)
from sdc.scoring.sweep_labelled import BIDS_ROOT, GRIDS, LABEL, SWEEPS

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}
MATCH_RATE = 3.5          # det/chan-min for the per-subject panel; mid-range and the only
                          # region where all three curves overlap


def main():
    subs = subjects(BIDS_ROOT)
    dets = [d for d in GRIDS if (SWEEPS / f"curve_{d}.npy").is_file()]
    M = {d: matrix(d, subs) for d in dets}
    allrows = np.arange(len(subs))
    tr, te = stratified_split(subs, N_TRAIN)

    # ---- (d) the table -------------------------------------------------------------------
    out_csv = SWEEPS / "summary.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["detector", "param", "value", "det_per_chan_min", "recall",
                    "precision_lower_bound", "tp", "n_true", "n_det"])
        for d in dets:
            values, tp, nt, nd, cm = M[d]
            b, r = budget_recall(values, tp, nt, nd, cm, allrows)
            for k, v in enumerate(values):
                w.writerow([LABEL[d], GRIDS[d][0], f"{v:g}", f"{b[k]:.3f}", f"{r[k]:.4f}",
                            f"{tp[:,k].sum()/max(nd[:,k].sum(),1):.4f}",
                            int(tp[:, k].sum()), int(nt[:, k].sum()), int(nd[:, k].sum())])
    print(f"[saved] {out_csv}")

    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.32, wspace=0.26)

    # ---- (a) recall vs detection rate, with per-subject spread ---------------------------
    ax = fig.add_subplot(gs[0, :2])
    for d in dets:
        values, tp, nt, nd, cm = M[d]
        b, r = budget_recall(values, tp, nt, nd, cm, allrows)
        o = np.argsort(b)
        # per-subject spread at each grid point: IQR, so one subject cannot define the band
        lo, hi = [], []
        for k in range(len(values)):
            rs = tp[:, k] / np.maximum(nt[:, k], 1)
            lo.append(np.percentile(rs, 25)); hi.append(np.percentile(rs, 75))
        ax.fill_between(b[o], np.array(lo)[o], np.array(hi)[o],
                        color=COLORS[LABEL[d]], alpha=.13, lw=0)
        ax.plot(b[o], r[o], "-o", ms=5, lw=1.9, color=COLORS[LABEL[d]], label=LABEL[d])
    ax.axvline(MATCH_RATE, color="0.35", ls="--", lw=1.2)
    ax.annotate(f"{MATCH_RATE:g}", (MATCH_RATE, ax.get_ylim()[0]), fontsize=8, color="0.35",
                ha="center", va="bottom")
    ax.set_xlabel("detection rate (detections per channel-minute)")
    ax.set_ylabel("recall vs expert marks")
    ax.set_title("(a) pooled recall, band = per-subject IQR. Compare VERTICALLY, at equal "
                 "detection rate", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=9)
    recessive(ax)

    # ---- (b) LOSO fold distribution ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
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
        rel = 100 * (picks - np.median(picks)) / np.median(picks)   # % from the median, so the
        ax.scatter(np.full(rel.size, j) + np.linspace(-.16, .16, rel.size), rel,  # three dials
                   s=22, color=COLORS[LABEL[d]], alpha=.75, edgecolor="none")     # share an axis
        ax.plot([j - .28, j + .28], [0, 0], color=COLORS[LABEL[d]], lw=2.5)
        ax.annotate(f"{picks.max()/picks.min()-1:+.0%} span\nmed {np.median(picks):.3g}",
                    (j, rel.max() + 1.5), fontsize=7.5, ha="center", color=COLORS[LABEL[d]])
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_xticks(range(len(dets)))
    ax.set_xticklabels([LABEL[d] for d in dets], fontsize=9)
    ax.set_ylabel(f"chosen parameter, % from median")
    ax.set_title(f"(b) LOSO stability at rate {MATCH_RATE:g}\n25 folds, one per left-out subject",
                 fontsize=9, loc="left")
    recessive(ax)

    # ---- (c) per-subject recall at a MATCHED detection rate ------------------------------
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(subs))
    for d in dets:
        values, tp, nt, nd, cm = M[d]
        b_all, _ = budget_recall(values, tp, nt, nd, cm, allrows)
        v = value_for_budget(values, b_all, MATCH_RATE)
        o = np.argsort(values)
        ys = []
        for i in range(len(subs)):
            rs = tp[i] / np.maximum(nt[i], 1)
            ys.append(np.interp(v, values[o], rs[o]))
        ax.plot(x, ys, "-o", ms=4, lw=1.3, color=COLORS[LABEL[d]], label=LABEL[d])
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("sub-", "") for s in subs], fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("subject")
    ax.set_ylabel("recall vs expert marks")
    ax.set_title(f"(c) per subject, all three at the SAME detection rate ({MATCH_RATE:g} "
                 f"det/chan-min) -- the earlier per-subject panel compared them at their "
                 f"defaults, i.e. at three different rates", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=3)
    recessive(ax)

    fig.suptitle("Detector performance against 852 expert-marked IEDs, 25 subjects",
                 fontsize=11)
    out = figdir("labelled") / "labelled_report.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
