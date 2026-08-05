"""
sdc.scoring.score_labelled
--------------------------
Score the three detectors against EXPERT-MARKED IEDs.

    .venv\\Scripts\\python.exe -m sdc.scoring.score_labelled

Everything else in this repo measures AGREEMENT -- do the detectors find the same things -- or
accuracy against spikes we synthesised ourselves and therefore already understood. This is the
first thing that answers "which one is right" against marks a neurologist made.

INPUTS
  runs/bids_<sub>.npz   one per subject, written by compare_spikes.py in BIDS mode. Same keys
                        as any other run, so nothing here re-runs a detector.
  the BIDS derivatives  read through sdc.scoring.bids_events (see there for why the truth is
                        the interpretation file and not events.tsv).

WHAT IS AND IS NOT COMPARABLE HERE
  * Recall is the number to trust. The expert marked what they marked; if a detector missed it,
    it missed it.
  * PRECISION IS A LOWER BOUND, NOT A MEASUREMENT, and this is the single most important caveat
    on this page. The experts marked interictal discharges they considered notable in a 3-minute
    sleep window -- they did not exhaustively mark every epileptiform transient, and nothing in
    the dataset claims they did. So an unmatched detection is "not on the expert's list", which
    is not the same as "wrong". Treat precision as a ceiling on how bad a detector could be,
    compare it BETWEEN detectors, and never quote it as a false-positive rate.
  * Between-detector comparison is safe on both, because all three are scored against the same
    marks with the same matcher.

Per-subject counts are small (10-61 events), so every rate carries a Wilson interval and the
across-subject summary reports the SPREAD rather than pooling -- 25 subjects with wildly
different event counts would otherwise be dominated by the few busy ones.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from seeg._style import RED, BLUE, MUTED, GRID, recessive

from sdc.common.paths import RUNS, figdir
from sdc.scoring.bids_events import subjects, load_subject, truth_per_channel
from sdc.scoring.score_sim_detectors import event_scores, wilson

BIDS_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(r"C:\Users\amoo0039\Documents\ieeg_ieds_bids_final\ieeg_ieds_bids")

TOL_MS = 50.0          # match radius, the same one the sim scorer and the agreement code use
VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}


def load_scored(bids_root=BIDS_ROOT, tol_ms=TOL_MS):
    """Score every subject that has BOTH a run npz and expert marks.

    Returns a list of per-subject dicts: {"subject", "n_events", "n_chan", "seconds",
    "scores": {detector: event_scores(...)}}."""
    out = []
    for sub in subjects(bids_root):
        npz = RUNS / f"bids_{sub}.npz"
        if not npz.is_file():
            continue
        z = np.load(npz, allow_pickle=False)
        names = [str(s) for s in z["names"]]
        fs, seconds = float(z["fs"]), float(z["seconds"])
        dets = [str(s) for s in z["detectors"]]

        d = load_subject(bids_root, sub)
        if list(d["channels"]) != names:
            raise SystemExit(
                f"{sub}: channel order in bids_{sub}.npz does not match channels.tsv.\n"
                f"  npz[0:3]={names[:3]}  tsv[0:3]={d['channels'][:3]}\n"
                f"Truth is built positionally, so a mismatch would score every detection "
                f"against the wrong channel -- refusing to continue.")
        truth = truth_per_channel(d["times"], d["chan_lists"], names, fs=None)   # seconds
        # event_scores wants an amplitude per true spike for its P(detected)-vs-amplitude
        # curve. The experts recorded no amplitude, so pass zeros and ignore that output --
        # the curve is a sim-only diagnostic, not something to fake here.
        amp = [np.zeros(t.size) for t in truth]

        scores = {}
        for det in dets:
            idx, ch = z[f"{det}_idx"], z[f"{det}_chan"]
            per = [np.sort(idx[ch == c] / fs) for c in range(len(names))]
            scores[det] = event_scores(truth, amp, per, seconds, tol_s=tol_ms / 1000.0)
        out.append({"subject": sub, "n_events": d["n_events"], "n_chan": len(names),
                    "seconds": seconds, "scores": scores, "dets": dets})
    return out


def report(scored):
    if not scored:
        raise SystemExit(
            "No scored subjects. Produce the runs first:\n"
            "    for s in sub-01 ...: BIDS_SUBJECT=$s python -m sdc.detect.compare_spikes")
    dets = scored[0]["dets"]
    n_ev = sum(s["n_events"] for s in scored)
    print(f"--- {len(scored)} subjects, {n_ev} expert-marked IEDs, +-{TOL_MS:g} ms match ---")
    print("PRECISION IS A LOWER BOUND: experts marked notable discharges, not every transient.")
    print(f"\n{'detector':<11}{'recall':>18}{'precision':>18}{'med |dt|':>10}{'bias':>9}")
    pooled = {}
    for d in dets:
        tp = sum(s["scores"][d]["tp"] for s in scored)
        nt = sum(s["scores"][d]["n_true"] for s in scored)
        nd = sum(s["scores"][d]["n_det"] for s in scored)
        rl, rh = wilson(tp, max(nt, 1))
        pl, ph = wilson(tp, max(nd, 1))
        offs = [s["scores"][d]["med_off_ms"] for s in scored
                if np.isfinite(s["scores"][d]["med_off_ms"])]
        bias = [s["scores"][d]["bias_ms"] for s in scored
                if np.isfinite(s["scores"][d]["bias_ms"])]
        pooled[d] = dict(recall=tp / max(nt, 1), precision=tp / max(nd, 1),
                         rl=rl, rh=rh, pl=pl, ph=ph,
                         med_off=np.median(offs) if offs else np.nan,
                         bias=np.median(bias) if bias else np.nan, tp=tp, nt=nt, nd=nd)
        p = pooled[d]
        print(f"{d:<11}{p['recall']:>8.3f} [{rl:.2f}-{rh:.2f}]"
              f"{p['precision']:>8.3f} [{pl:.2f}-{ph:.2f}]"
              f"{p['med_off']:>9.1f}ms{p['bias']:>8.1f}ms")
    return pooled


def figure(scored, pooled):
    dets = scored[0]["dets"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    subs = [s["subject"].replace("sub-", "") for s in scored]
    x = np.arange(len(scored))

    # (a) per-subject recall. One line per detector; the spread across subjects IS the result,
    # because 25 patients with different implants is the only generalisation on offer.
    ax = axes[0]
    for d in dets:
        y = [s["scores"][d]["recall"] for s in scored]
        ax.plot(x, y, "-o", ms=4, lw=1.3, color=COLORS.get(d, MUTED), label=d)
    ax.set_xticks(x)
    ax.set_xticklabels(subs, fontsize=6, rotation=90)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("recall vs expert marks")
    ax.set_xlabel("subject")
    ax.set_title("(a) per subject -- the spread is the result", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)

    # (b) pooled recall/precision with Wilson intervals
    ax = axes[1]
    w = 0.35
    for i, d in enumerate(dets):
        p = pooled[d]
        ax.bar(i - w / 2, p["recall"], w, color=COLORS.get(d, MUTED), alpha=.85)
        ax.plot([i - w / 2] * 2, [p["rl"], p["rh"]], color="0.2", lw=1.4)
        ax.bar(i + w / 2, p["precision"], w, color=COLORS.get(d, MUTED), alpha=.40)
        ax.plot([i + w / 2] * 2, [p["pl"], p["ph"]], color="0.2", lw=1.4)
    ax.set_xticks(range(len(dets)))
    ax.set_xticklabels(dets, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("solid = recall,  faded = precision")
    ax.set_title("(b) pooled, Wilson 95% -- precision is a LOWER BOUND", fontsize=9, loc="left")

    # (c) timing. The sim and the real agreement data both say Barkmeier marks late; expert
    # marks are an independent third reference for that.
    ax = axes[2]
    for i, d in enumerate(dets):
        v = [s["scores"][d]["bias_ms"] for s in scored
             if np.isfinite(s["scores"][d]["bias_ms"])]
        ax.scatter(np.full(len(v), i) + np.linspace(-.15, .15, len(v)), v, s=14,
                   color=COLORS.get(d, MUTED), alpha=.6, edgecolor="none")
        if v:
            ax.plot([i - .25, i + .25], [np.median(v)] * 2, color=COLORS.get(d, MUTED), lw=2.5)
    ax.axhline(0, color=MUTED, ls="--", lw=1.0)
    ax.set_xticks(range(len(dets)))
    ax.set_xticklabels(dets, fontsize=9)
    ax.set_ylabel("detection - expert mark (ms)")
    ax.set_title("(c) timing bias against a human", fontsize=9, loc="left")

    for a in axes:
        a.grid(axis="y", alpha=.3)
        recessive(a)
    fig.suptitle(f"Scored against {sum(s['n_events'] for s in scored)} expert-marked IEDs, "
                 f"{len(scored)} subjects, unmontaged, no preprocessing", fontsize=11)
    fig.tight_layout()
    out = figdir("labelled") / "score_labelled.png"
    fig.savefig(out, dpi=130)
    print(f"\n[saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    _scored = load_scored()
    figure(_scored, report(_scored))
