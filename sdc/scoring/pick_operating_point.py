"""
sdc.scoring.pick_operating_point
--------------------------------
Choose each detector's operating point on TRAINING subjects, measure it on HELD-OUT ones.

    .venv\\Scripts\\python.exe -m sdc.scoring.pick_operating_point

WHAT IS ACTUALLY BEING FITTED, which is subtler than it looks
  The sweep curves are MONOTONE -- lower threshold, more detections, higher recall, no interior
  optimum. So "the best parameter" does not exist on its own; what exists is a parameter that
  hits a chosen DETECTION BUDGET. The budget is a judgement (how many detections per
  channel-minute are you willing to review), and the parameter follows from it.

  So the fitted quantity is the budget -> parameter mapping, and the held-out question is
  whether that mapping TRANSFERS: does the value chosen on 10 subjects still land near the
  target budget, and deliver comparable recall, on 15 subjects it never saw?

  This matters more than it sounds. If the mapping does not transfer, then every published
  "recommended threshold" is a statement about the tuning cohort, not the detector.

DESIGN
  * 10 train / 15 test. One scalar is being fitted per detector, so training data is cheap and
    test data is not -- the estimate to protect is the held-out one.
  * Stratified by EVENT COUNT and CHANNEL COUNT, both properties of the data. NOT by recall:
    that is a detector's output and stratifying on it would leak the answer into the split.
  * The split is drawn ONCE from a fixed seed. Re-drawing it after seeing a disappointing test
    number is the classic way this goes wrong.
  * LOSO on top, which is free -- the expensive (subject x parameter) matrix is already
    computed, so both designs are just aggregations of it. LOSO answers a different question:
    is the chosen parameter STABLE, or did one split get lucky?
"""
import numpy as np
import matplotlib.pyplot as plt

from seeg._style import RED, BLUE, MUTED, GRID, recessive

from sdc.common.paths import RUNS, figdir
from sdc.scoring.bids_events import subjects, load_subject, truth_per_channel
from sdc.scoring.score_sim_detectors import event_scores
from sdc.scoring.sweep_labelled import BIDS_ROOT, GRIDS, LABEL, SWEEPS, TOL_S

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}
N_TRAIN = 10
SEED = 0
TARGETS = [2.5, 3.5, 4.5]      # detection rates to report, det per channel-minute
MATCH_TARGET = 3.5             # the one panel (b) fixes a parameter at; same as report_labelled


def matrix(det, subs):
    """(subject x parameter) -> tp, n_true, n_det, chan_min. Everything else is an aggregation
    of this, which is why the split and LOSO cost nothing extra.

    INCOMPLETE GRID POINTS ARE DROPPED, not zero-filled. A value scored on 2 of 25 subjects
    would otherwise land on the pooled curve as a real point -- with its numerator from two
    subjects and its denominator from twenty-five, so an arbitrarily low rate at an arbitrary
    recall. Reading a sweep while it is still running is the obvious way to hit this, and it
    produced a Barkmeier point at 0.5 det/chan-min and recall 0.86 that looked like a result."""
    param, values = GRIDS[det]
    tp = np.zeros((len(subs), len(values)))
    nt = np.zeros_like(tp); nd = np.zeros_like(tp); cm = np.zeros(len(subs))
    have = np.zeros(tp.shape, bool)
    for i, sub in enumerate(subs):
        d = load_subject(BIDS_ROOT, sub)
        for k, v in enumerate(values):
            f = SWEEPS / f"bids_{sub}_{det}_{param}{v:g}.npz"
            if not f.is_file():
                continue
            have[i, k] = True
            z = np.load(f, allow_pickle=False)
            names = [str(s) for s in z["names"]]
            fs, secs = float(z["fs"]), float(z["seconds"])
            truth = truth_per_channel(d["times"], d["chan_lists"], names)
            idx, ch = z[f"{LABEL[det]}_idx"], z[f"{LABEL[det]}_chan"]
            got = [np.sort(idx[ch == c] / fs) for c in range(len(names))]
            sc = event_scores(truth, [np.zeros(t.size) for t in truth], got, secs, tol_s=TOL_S)
            tp[i, k], nt[i, k], nd[i, k] = sc["tp"], sc["n_true"], sc["n_det"]
            cm[i] = len(names) * secs / 60.0
    keep = have.all(axis=0)
    if not keep.all():
        miss = [f"{param}={v:g} ({have[:, k].sum()}/{len(subs)})"
                for k, v in enumerate(values) if not keep[k]]
        print(f"  [{LABEL[det]}] dropping {len(miss)} incomplete grid point(s): "
              f"{', '.join(miss)}")
    v = np.array(values, float)[keep]
    return v, tp[:, keep], nt[:, keep], nd[:, keep], cm


def budget_recall(values, tp, nt, nd, cm, rows):
    """Pooled budget and recall over a set of subjects, per parameter value."""
    b = nd[rows].sum(axis=0) / max(cm[rows].sum(), 1e-9)
    r = tp[rows].sum(axis=0) / np.maximum(nt[rows].sum(axis=0), 1)
    return b, r


def value_for_budget(values, b, target):
    """The parameter that hits `target` on these subjects. Budget falls as the threshold
    rises, so interpolate on the reversed axis."""
    o = np.argsort(b)
    if not (b[o].min() <= target <= b[o].max()):
        return np.nan
    return float(np.interp(target, b[o], values[o]))


def per_subject_at(values, tp, nt, nd, cm, v):
    """Per-subject recall and ACHIEVED detection rate at one fixed parameter `v`.

    Lives here rather than in report_labelled because it is the per-subject counterpart of
    `budget_recall`, which pools -- and pooling is exactly what hides the spread in panel (b)."""
    o = np.argsort(values)
    rec, rate = [], []
    for i in range(tp.shape[0]):
        rec.append(np.interp(v, values[o], (tp[i] / np.maximum(nt[i], 1))[o]))
        rate.append(np.interp(v, values[o], (nd[i] / max(cm[i], 1e-9))[o]))
    return np.array(rec), np.array(rate)


def stratified_split(subs, n_train, seed=SEED):
    """Split stratified on EVENT COUNT and CHANNEL COUNT -- properties of the data, never of a
    detector's performance."""
    info = [(s, load_subject(BIDS_ROOT, s)) for s in subs]
    key = np.array([[d["n_events"], len(d["channels"])] for _, d in info], float)
    rank = np.lexsort((key[:, 1], key[:, 0]))
    rng = np.random.default_rng(seed)
    train = []
    for blk in np.array_split(rank, n_train):          # one pick per stratum
        train.append(int(rng.choice(blk)))
    test = [i for i in range(len(subs)) if i not in train]
    return np.array(sorted(train)), np.array(test)


def main():
    subs = subjects(BIDS_ROOT)
    dets = [d for d in GRIDS if (SWEEPS / f"curve_{d}.npy").is_file()]
    M = {d: matrix(d, subs) for d in dets}
    tr, te = stratified_split(subs, N_TRAIN)
    print(f"{len(subs)} subjects -> {len(tr)} train / {len(te)} test, stratified on "
          f"(n_events, n_channels), seed {SEED}")
    print(f"  train: {', '.join(subs[i] for i in tr)}")

    print(f"\n--- fit the budget->parameter map on TRAIN, measure on TEST ---")
    print(f"{'target':>7}{'detector':>11}{'param(train)':>14}{'budget(test)':>14}"
          f"{'recall(test)':>14}{'recall(train)':>15}")
    for target in TARGETS:
        for d in dets:
            values, tp, nt, nd, cm = M[d]
            b_tr, r_tr = budget_recall(values, tp, nt, nd, cm, tr)
            b_te, r_te = budget_recall(values, tp, nt, nd, cm, te)
            v = value_for_budget(values, b_tr, target)
            if not np.isfinite(v):
                continue
            o = np.argsort(values)
            print(f"{target:>7.1f}{LABEL[d]:>11}{v:>14.3g}"
                  f"{np.interp(v, values[o], b_te[o]):>14.2f}"
                  f"{np.interp(v, values[o], r_te[o]):>14.3f}"
                  f"{np.interp(v, values[o], r_tr[o]):>15.3f}")

    print(f"\n--- LOSO stability: is the chosen parameter the same across folds? ---")
    for target in TARGETS:
        for d in dets:
            values, tp, nt, nd, cm = M[d]
            picks = []
            for i in range(len(subs)):
                rows = np.array([k for k in range(len(subs)) if k != i])
                b, _ = budget_recall(values, tp, nt, nd, cm, rows)
                v = value_for_budget(values, b, target)
                if np.isfinite(v):
                    picks.append(v)
            picks = np.array(picks)
            if picks.size:
                print(f"  budget {target:.1f}  {LABEL[d]:<10} "
                      f"median {np.median(picks):.3g}  "
                      f"range {picks.min():.3g}-{picks.max():.3g}  "
                      f"spread {100*(picks.max()-picks.min())/max(np.median(picks),1e-9):.0f}%")

    # ---- figure -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax = axes[0]
    for d in dets:
        values, tp, nt, nd, cm = M[d]
        b, r = budget_recall(values, tp, nt, nd, cm, np.arange(len(subs)))
        o = np.argsort(b)
        ax.plot(b[o], r[o], "-o", ms=5, lw=1.8, color=COLORS[LABEL[d]], label=LABEL[d])
    # THIN lines, not thick bands. These are single target rates, and drawn at lw=5 they read as
    # confidence intervals -- which is exactly how the first reader took them. Nothing here has a
    # width: each is one rate you read UP from, and where a curve crosses it is that detector's
    # recall at equal output.
    for t in TARGETS:
        ax.axvline(t, color="0.55", ls="--", lw=1.0, zorder=0)
        ax.annotate(f"{t:g}", (t, 0.012), xycoords=("data", "axes fraction"),
                    fontsize=8, color="0.4", ha="center", va="bottom")
    ax.annotate("target rates", (TARGETS[-1] + 0.35, 0.012), xycoords=("data", "axes fraction"),
                fontsize=8, color="0.4", ha="left", va="bottom")
    # CLIPPED, deliberately: Janca's grid runs out to 142 det/chan-min and on a linear axis that
    # tail squeezes all three targets into the first 5% of the panel. This panel is about where
    # the targets sit; the ceiling question is labelled_report (a), which plots every point on a
    # log axis. Said on the panel rather than only here -- a clipped axis that does not announce
    # itself is indistinguishable from a curve that ends.
    ax.set_xlim(0, 15)
    ax.text(.99, .10, "axis clipped to the target region -- full curves in labelled_report (a)",
            transform=ax.transAxes, fontsize=7, color=MUTED, ha="right")
    ax.set_xlabel("detection rate (detections per channel-minute)")
    ax.set_ylabel("recall vs expert marks")
    ax.set_title("(a) all 25 subjects -- compare VERTICALLY, at equal detection rate",
                 fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=9)
    recessive(ax)

    # ---- (b) ONE fixed parameter -> what rate does each subject actually give? --------------
    # This is the cost of a single published threshold. Panel (a) is pooled, so it shows the
    # target being hit exactly by construction; the target is hit on the COHORT and on almost
    # no individual subject. A held-out-recall bar chart used to sit here and only re-read (a).
    ax = axes[1]
    target = MATCH_TARGET
    print(f"\n--- one fixed parameter at rate {target:g}: per-subject ACHIEVED rate ---")
    print(f"{'detector':<11}{'param':>9}{'median':>9}{'p10-p90':>16}{'span':>8}")
    floor = np.inf
    for j, d in enumerate(dets):
        values, tp, nt, nd, cm = M[d]
        b, _ = budget_recall(values, tp, nt, nd, cm, np.arange(len(subs)))
        v = value_for_budget(values, b, target)
        if not np.isfinite(v):
            continue
        rates = per_subject_at(values, tp, nt, nd, cm, v)[1]
        c = COLORS[LABEL[d]]
        ax.scatter(np.full(rates.size, j) + np.linspace(-.19, .19, rates.size), rates,
                   s=26, color=c, alpha=.75, edgecolor="none", zorder=3)
        ax.plot([j - .32, j + .32], [np.median(rates)] * 2, color=c, lw=2.5, zorder=4)
        lo, hi = np.percentile(rates, [10, 90])
        ax.annotate(f"{GRIDS[d][0]}={v:.3g}\np10-p90 spans {hi / max(lo, 1e-9):.1f}x",
                    (j, 0.02), xycoords=("data", "axes fraction"), fontsize=7.5,
                    ha="center", va="bottom", color=c)
        floor = min(floor, float(rates[rates > 0].min()))
        print(f"{LABEL[d]:<11}{v:>9.3g}{np.median(rates):>9.2f}"
              f"{f'{lo:.2f}-{hi:.2f}':>16}{f'{hi / max(lo, 1e-9):.1f}x':>8}")
    ax.axhline(target, color="0.4", ls="--", lw=1.1)
    ax.set_yscale("log")
    if np.isfinite(floor):
        ax.set_ylim(bottom=floor / 3.0)      # headroom under the lowest dot for the annotations
    ax.set_xticks(range(len(dets)))
    ax.set_xticklabels([LABEL[d] for d in dets], fontsize=9)
    ax.set_ylabel("detections per channel-minute on that subject (log)")
    ax.set_title(f"(b) ONE parameter, set to give {target:g} det/chan-min over the cohort.\n"
                 "Dashes = target. Each dot is a subject: the cohort hits it, nobody does",
                 fontsize=9, loc="left")
    recessive(ax)
    fig.suptitle("Operating points set against 852 expert-marked IEDs, not against each other",
                 fontsize=11)
    fig.tight_layout()
    out = figdir("labelled") / "operating_points.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
