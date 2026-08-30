"""
sdc.compare.ez_ranking
----------------------
Do the detectors' busiest channels land on the clinically defined EZ?

    .venv\\Scripts\\python.exe -m sdc.compare.ez_ranking [runs/P1_pre.npz ...]

`rate_comparators.py` asks how HIGH the rate is on EZ contacts. This asks a different and more
clinically direct question: if you ranked the implant by spike rate and took the top n, how many
of them would be EZ -- and which EZ contacts would you have MISSED?

That second half is the point. A detector that finds the EZ but also ranks three EZ contacts
below rank 100 is telling you something a summary rate cannot: its localisation is incomplete,
and the specific contacts it loses are checkable against the raw signal.

WHAT THE NUMBERS MEAN, AND DO NOT
  The EZ list is a CLINICAL judgement recorded in the trials JSON. Agreeing with it is evidence
  a detector is finding real epileptic tissue, not proof -- and disagreeing is not proof of
  error, since interictal spikes and the seizure onset zone are related but not identical.
  Chance is reported alongside every count, because "7 of the top 21" means nothing until you
  know that 2 would happen by luck.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from seeg._style import RED, BLUE, MUTED, GRID, recessive

from sdc.common import cond
from sdc.common.paths import RUNS, figdir

META = Path(r"C:\Users\amoo0039\Documents\local\data_meta\stim_trials.json")
VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}
LOW_RANK = 3.0        # an EZ contact ranked worse than LOW_RANK x |EZ| is called out by name
LOG_GAIN = False      # panel (a) x-axis. LINEAR by default: it is a cumulative-gain curve, and
                      # on a linear axis CHANCE IS A STRAIGHT DIAGONAL, which is what makes
                      # "above chance" readable at a glance. On log, chance is a curve and the
                      # comparison gets harder -- the thing the panel exists for.
LOG_RANK = True       # panel (b) x-axis. LOG, because it is a rank plot and the question is
                      # "is this contact near the top": rank 5 vs 20 matters, 120 vs 150 does
                      # not. On linear, ranks 1-21 of 226 occupy the leftmost 9% of the panel.


def load(stem, label="all"):
    z = np.load(RUNS / f"{stem}.npz", allow_pickle=False)
    names = [str(s) for s in z["names"]]
    sel = cond.select(z, label)
    dets = [str(s) for s in z["detectors"]]
    rates = {d: sel.rate(np.bincount(z[f"{d}_chan"][sel.keep(d)], minlength=len(names)))
             for d in dets}
    meta = json.loads(META.read_text())
    ez = [c for c in (next(p for p in meta if p["patient_id"] == int(z["patient"])).get("EZ")
                      or []) if c in names]
    return names, dets, rates, np.array([names.index(c) for c in ez], int), sel


def ranks(rate, measurable):
    """Rank 1 = busiest. Unmeasurable channels are ranked last, not dropped, so every EZ
    contact has a rank and a missing one cannot silently vanish from the count."""
    r = np.where(measurable, np.nan_to_num(rate, nan=-1.0), -1.0)
    order = np.argsort(-r, kind="stable")
    out = np.empty(len(r), int)
    out[order] = np.arange(1, len(r) + 1)
    return out


def report(stem, label="all"):
    names, dets, rates, ez, sel = load(stem, label)
    n_chan, n_ez = len(names), ez.size
    print(f"\n=== {stem} ({label}) -- {n_ez} EZ contacts of {n_chan} channels ===")
    chance = n_ez * n_ez / n_chan
    print(f"top-{n_ez} by rate: how many are EZ?   (chance = {chance:.1f})")
    out = {}
    for d in dets:
        rk = ranks(rates[d], sel.measurable if sel.measurable is not None
                   else np.ones(n_chan, bool))
        hit = int((rk[ez] <= n_ez).sum())
        out[d] = rk
        print(f"  {d:<11} {hit:2d}/{n_ez}   ({hit/max(chance,1e-9):.1f}x chance)   "
              f"median EZ rank {int(np.median(rk[ez])):3d}   worst {int(rk[ez].max()):3d}")
    print(f"\n  EZ contacts ranked worse than {LOW_RANK:g}x|EZ| (={int(LOW_RANK*n_ez)}) "
          f"by ANY detector:")
    flagged = False
    for k, c in enumerate(ez):
        rr = {d: int(out[d][c]) for d in dets}
        if max(rr.values()) > LOW_RANK * n_ez:
            flagged = True
            print(f"    {names[c]:<12}" + "  ".join(f"{d[:4]} {rr[d]:>3d}" for d in dets))
    if not flagged:
        print("    none")
    return names, dets, out, ez, n_chan


def figure(recs):
    fig, axes = plt.subplots(2, len(recs), figsize=(7.0 * len(recs), 8.4), squeeze=False)
    for col, (stem, (names, dets, rk, ez, n_chan)) in enumerate(recs.items()):
        n_ez = ez.size
        # (a) cumulative gain: of the top k channels, how many are EZ?
        ax = axes[0][col]
        ks = np.arange(1, n_chan + 1)
        for d in dets:
            found = np.cumsum(np.isin(np.argsort(rk[d], kind="stable"), ez))
            ax.plot(ks, found, lw=1.8, color=COLORS.get(d, MUTED), label=d)
        ax.plot(ks, ks * n_ez / n_chan, ls="--", lw=1.2, color=MUTED, label="chance")
        ax.axvline(n_ez, color=GRID, lw=6, alpha=.6, zorder=0)
        ax.annotate(f"top {n_ez}\n(= |EZ|)", (n_ez, n_ez * 0.55), fontsize=7.5, color="0.35",
                    ha="left")
        if LOG_GAIN:
            ax.set_xscale("log")
        ax.set_xlabel("top k channels by rate")
        ax.set_ylabel("EZ contacts found")
        ax.set_title(f"({'ab'[col]}) {stem}: does the ranking find the EZ?",
                     fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=8)
        recessive(ax)

        # (b) every EZ contact's rank, per detector. Low = found.
        ax = axes[1][col]
        order = np.argsort([np.median([rk[d][c] for d in dets]) for c in ez])
        y = np.arange(n_ez)
        for j, d in enumerate(dets):
            ax.scatter([rk[d][ez[i]] for i in order], y + (j - 1) * 0.22, s=26,
                       color=COLORS.get(d, MUTED), label=d, zorder=3)
        ax.axvline(n_ez, color="0.4", ls="--", lw=1.2)
        ax.annotate(f"rank {n_ez}", (n_ez, -1.2), fontsize=7, color="0.4", ha="center")
        if LOG_RANK:
            ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels([names[ez[i]] for i in order], fontsize=7)
        ax.set_ylim(-1.6, n_ez)
        ax.set_xlabel("rank by spike rate (1 = busiest)")
        ax.set_title("EZ contacts, ranked -- further right = the detector missed it",
                     fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        recessive(ax)
    fig.suptitle("Do the busiest channels land on the clinically defined EZ?  "
                 "(EZ list is from the trials JSON, independent of every detector)",
                 fontsize=11)
    fig.tight_layout()
    # named from the stems: the same figure over different runs is a different result, and one
    # fixed filename means the last invocation silently replaces the previous patient's answer
    out = figdir("real") / ("ez_ranking_" + "_".join(recs) + ".png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    stems = [Path(a).stem for a in sys.argv[1:]] or ["P1_pre", "P5_pre"]
    recs = {}
    for st in stems:
        names, dets, rk, ez, n_chan = report(st)
        recs[st] = (names, dets, rk, ez, n_chan)
    figure(recs)
