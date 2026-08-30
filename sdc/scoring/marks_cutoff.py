"""
sdc.scoring.marks_cutoff
------------------------
Per-channel precision and recall against the expert marks, for Janca and Barkmeier, pooled
across the three marked baseline blocks -- and what a channel RATE CUT-OFF would buy.

    .venv\\Scripts\\python.exe -m sdc.scoring.marks_cutoff

WHY THIS RATHER THAN marks_summary.png
  That figure shows three detectors x four blocks x two metrics at once, which is enough to see
  that something happens but not enough to act on. This one drops to the two detectors under
  consideration, pools the blocks (the pattern replicates across all three, so keeping them
  apart costs clarity and buys nothing), and puts the decision variable -- the channel's own
  mark rate -- on the x-axis of every panel.

THE DECISION IT SUPPORTS
  Precision collapses on low-rate channels for every detector, so the useful question is where
  to stop trusting a channel. Panel (c) answers it directly: it sweeps a cut-off on the
  channel's mark rate and shows what the retained set looks like at each one, so the cut can be
  chosen against a stated precision target rather than by eye.

  Note what the cut-off is ON. Here it is the EXPERT mark rate, because that is the only
  unbiased measure of how active a channel really is. In production there are no marks and the
  cut has to use the detector's own baseline rate -- so panel (d) plots one against the other,
  and the correlation there is what licenses substituting one for the other.

CAVEAT THAT LIMITS EVERY NUMBER HERE
  Three blocks, ~2900 marks, and 74% of them on six channels. The low-rate end of every panel is
  a handful of channels carrying a handful of marks, so the collapse is well measured but the
  exact cut-off is not. Treat the shape as the finding and the specific threshold as indicative.
"""
import numpy as np

from sdc.common.paths import figdir
from sdc.scoring.score_marks import collect, DETS

SHOW = ("Janca", "Barkmeier")
COLORS = {"Janca": "#c8102e", "Barkmeier": "#0072b2", "Delphos": "#4a3aa7"}


def gather(tol=0.050):
    """One row per (block, channel): mark rate, and per-detector precision/recall/counts."""
    rows = []
    for s in collect(tol):
        if "strict" in s["rater"]:
            continue                     # one pass per block, or channels are counted twice
        mins = (s["t1"] - s["t0"]) / 60.0
        for r in s["rows"]:
            rec = {"subj": s["subj"], "chan": r["chan"], "n_mark": r["n_mark"],
                   "mark_rate": r["n_mark"] / mins, "mins": mins}
            for d in DETS:
                rec[d] = {"prec": r[d]["prec"], "recall": r[d]["recall"],
                          "n_det": r[d]["n_det"], "det_rate": r[d]["n_det"] / mins,
                          "hit": r[d]["hit"]}
            rows.append(rec)
    return rows


def cutoff_table(rows, cuts=(0, 1, 2, 5, 10, 20, 40)):
    """Pooled precision/recall/F1 over the channels ABOVE each mark-rate cut-off."""
    out = []
    for c in cuts:
        keep = [r for r in rows if r["mark_rate"] >= c]
        e = {"cut": c, "n_chan": len(keep),
             "n_mark": sum(r["n_mark"] for r in keep),
             "frac_marks": (sum(r["n_mark"] for r in keep)
                            / max(sum(r["n_mark"] for r in rows), 1))}
        for d in SHOW:
            tp = sum(r[d]["hit"] for r in keep)
            nd = sum(r[d]["n_det"] for r in keep)
            nm = sum(r["n_mark"] for r in keep)
            p = tp / nd if nd else np.nan
            rc = tp / nm if nm else np.nan
            e[d] = {"prec": p, "recall": rc,
                    "f1": 0.0 if not (p > 0 and rc > 0) else 2 * p * rc / (p + rc)}
        out.append(e)
    return out


def figure(tol=0.050, outdir=None):
    import matplotlib.pyplot as plt

    rows = gather(tol)
    marked = [r for r in rows if r["n_mark"] > 0]
    empty = [r for r in rows if r["n_mark"] == 0]
    cuts = cutoff_table(marked)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # (a) precision vs channel activity -- the collapse
    ax = axes[0][0]
    for d in SHOW:
        x = [r["mark_rate"] for r in marked]
        y = [r[d]["prec"] for r in marked]
        ax.scatter(x, y, s=34, color=COLORS[d], alpha=.65, edgecolor="none", label=d)
    ax.axhline(0.5, color="0.5", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("expert marks per minute on that channel")
    ax.set_ylabel("precision")
    ax.set_title("(a) Precision collapses on quiet channels", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.3, which="both")

    # (b) recall, for contrast -- it does NOT collapse the same way
    ax = axes[0][1]
    for d in SHOW:
        ax.scatter([r["mark_rate"] for r in marked], [r[d]["recall"] for r in marked],
                   s=34, color=COLORS[d], alpha=.65, edgecolor="none", label=d)
    ax.axhline(0.5, color="0.5", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("expert marks per minute on that channel")
    ax.set_ylabel("recall")
    ax.set_title("(b) Recall does not -- so the problem is FALSE POSITIVES",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.3, which="both")

    # (c) what a cut-off buys, and what it costs
    ax = axes[1][0]
    xs = [c["cut"] for c in cuts]
    for d in SHOW:
        ax.plot(xs, [c[d]["prec"] for c in cuts], "-o", ms=6, color=COLORS[d],
                label=f"{d} precision")
        ax.plot(xs, [c[d]["recall"] for c in cuts], "--s", ms=5, color=COLORS[d],
                alpha=.55, label=f"{d} recall")
    ax2 = ax.twinx()
    ax2.plot(xs, [c["n_chan"] for c in cuts], ":", lw=2, color="0.4")
    ax2.set_ylabel("channels retained", color="0.4")
    ax.set_xlabel("cut-off: keep channels with >= this many marks/min")
    ax.set_ylabel("pooled precision / recall")
    ax.set_ylim(0, 1)
    ax.set_title("(c) Choosing the cut-off (dotted grey = channels kept)",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower right")
    ax.grid(alpha=.3)

    # (d) can the detector's own rate stand in for the mark rate?
    ax = axes[1][1]
    for d in SHOW:
        x = [r["mark_rate"] for r in marked]
        y = [r[d]["det_rate"] for r in marked]
        ax.scatter(x, y, s=34, color=COLORS[d], alpha=.65, edgecolor="none")
        ax.scatter([0.15] * len(empty), [r[d]["det_rate"] for r in empty], s=34,
                   color=COLORS[d], alpha=.65, marker="x")
        from scipy.stats import spearmanr
        rho = spearmanr(x, y)[0]
        ax.plot([], [], "o", color=COLORS[d], label=f"{d}  rho={rho:+.2f}")
    lim = [0.1, 120]
    ax.plot(lim, lim, color="0.4", ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("expert marks/min  (x at left = channels with NO marks)")
    ax.set_ylabel("detector rate/min")
    ax.set_title("(d) In production there are no marks -- can the detector's\n"
                 "own rate substitute? Dashed = equality", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=.3, which="both")

    fig.suptitle(f"Per-channel accuracy against expert marks, {len(marked)} marked + "
                 f"{len(empty)} empty channels over 3 baseline blocks (P1, P5, P8)",
                 fontsize=12)
    fig.tight_layout()
    out = (figdir("real") if outdir is None else outdir) / "marks_cutoff.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    print(f"{'cut':>5}{'chan':>6}{'marks':>7}{'%marks':>8}"
          + "".join(f"{d[:4] + ' prec':>11}{d[:4] + ' rec':>10}{d[:4] + ' F1':>9}"
                    for d in SHOW))
    for c in cuts:
        print(f"{c['cut']:>5}{c['n_chan']:>6}{c['n_mark']:>7}{c['frac_marks']:>8.0%}"
              + "".join(f"{c[d]['prec']:>11.3f}{c[d]['recall']:>10.3f}{c[d]['f1']:>9.3f}"
                        for d in SHOW))
    print(f"\n[saved] {out}")
    return rows, cuts


if __name__ == "__main__":
    figure()
