"""
sdc.compare.channel_rates
-------------------------
Per-channel spike rate as a bar chart, one panel per detector.

    .venv\\Scripts\\python.exe -m sdc.compare.channel_rates [P1_pre_qcfinalv2] [rate|name]

All three panels share ONE channel order so the bars line up vertically -- sorting each panel by
its own detector's rate would put a different channel at each x position and make the panels
uncomparable, which is the one thing this figure exists to allow.

The default order is by mean rate across detectors. `name` keeps electrode order instead, which
is what to use when the question is about a shank rather than about the rate distribution.

EZ contacts are coloured, and channels carrying expert marks are ticked underneath, because the
interesting part of this plot is the LOW-rate tail: that is where precision collapses (see
`sdc.scoring.bids_per_channel`), so it matters whether a short bar is a quiet real channel or
noise being counted.
"""
import json
import sys
from pathlib import Path

import numpy as np

from sdc.common import cond
from sdc.common.paths import RUNS, figdir

META = Path(r"C:\Users\amoo0039\Documents\local\data_meta\stim_trials.json")
COLORS = {"Janca": "#c8102e", "Barkmeier": "#0072b2", "Delphos": "#4a3aa7"}
EZ_COLOR = "#f0a202"


def rates(stem, label="all"):
    z = np.load(RUNS / f"{stem}.npz", allow_pickle=False)
    names = [str(s) for s in z["names"]]
    sel = cond.select(z, label)
    dets = [str(s) for s in z["detectors"]]
    # cond.rate is Hz; every other per-channel number in this repo is det/chan-min
    r = {d: 60.0 * sel.rate(np.bincount(z[f"{d}_chan"][sel.keep(d)], minlength=len(names)))
         for d in dets}
    try:
        meta = json.loads(META.read_text())
        ez = set((next(p for p in meta if p["patient_id"] == int(z["patient"])).get("EZ")) or [])
    except Exception:                                            # noqa: BLE001
        ez = set()
    return names, dets, r, ez


def marked_channels(subject):
    """Channels carrying expert marks, so the low-rate tail can be read against ground truth."""
    try:
        from sdc.scoring.score_marks import collect
        out = set()
        for s in collect():
            if s["subj"] == subject:
                out |= {r["chan"] for r in s["rows"] if r["n_mark"] > 0}
        return out
    except Exception:                                            # noqa: BLE001
        return set()


def figure(stem="P1_pre_qcfinalv2", order="rate", label="all", outdir=None,
           width_per_ch=0.20, tick_pt=7.0):
    import matplotlib.pyplot as plt

    names, dets, r, ez = rates(stem, label)
    n = len(names)
    stack = np.vstack([r[d] for d in dets])
    mean = np.where(np.isnan(stack).all(0), np.nan,          # unmeasurable everywhere -> nan,
                    np.nanmean(np.where(np.isnan(stack), 0, stack), axis=0))   # not a warning
    idx = np.argsort(-np.nan_to_num(mean)) if order == "rate" else np.argsort(names)
    subj = f"P{int(np.load(RUNS / f'{stem}.npz', allow_pickle=False)['patient'])}"
    marks = marked_channels(subj)

    # width_per_ch has to leave room for a rotated tick label: below ~1.6x the font's cap height
    # in inches the channel names start colliding, which defeats the point of naming them
    fig, axes = plt.subplots(len(dets), 1, sharex=True,
                             figsize=(max(14.0, n * width_per_ch), 10.5))
    x = np.arange(n)
    top = np.nanmax([np.nanmax(r[d]) for d in dets])
    for ax, d in zip(np.atleast_1d(axes), dets):
        v = np.nan_to_num(r[d])[idx]
        cols = [EZ_COLOR if names[i] in ez else COLORS.get(d, "0.4") for i in idx]
        ax.bar(x, v, width=.78, color=cols, linewidth=0)
        ax.set_ylim(0, top * 1.05)
        ax.set_ylabel(f"{d}\ndet / chan-min")
        ax.grid(axis="y", alpha=.3)
        ax.margins(x=.004)
        med = np.median(v[v > 0]) if (v > 0).any() else 0
        ax.axhline(med, color="0.35", ls=":", lw=1.0)
        ax.annotate(f"median of non-zero = {med:.1f}", (n * .995, med), fontsize=7.5,
                    color="0.35", ha="right", va="bottom")
    ax = np.atleast_1d(axes)[-1]
    ax.set_xticks(x)
    ax.set_xticklabels([names[i] for i in idx], rotation=90, fontsize=tick_pt)
    for t, i in zip(ax.get_xticklabels(), idx):
        if names[i] in ez:
            t.set_color(EZ_COLOR)
            t.set_fontweight("bold")
        elif names[i] in marks:
            t.set_color("#1a7f37")
    ax.set_xlabel(f"channel, ordered by {'mean rate across detectors' if order == 'rate' else 'name'}"
                  f"   (orange = EZ" + (", green = has expert marks)" if marks else ")"))
    fig.suptitle(f"{stem} ({label}): per-channel spike rate", fontsize=12)
    fig.tight_layout()
    out = (Path(outdir) if outdir else figdir("real")) / f"channel_rates_{stem}_{order}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"{stem}: {n} channels, {len(ez & set(names))} EZ, {len(marks)} with expert marks")
    for d in dets:
        v = np.nan_to_num(r[d])
        print(f"  {d:<11} median {np.median(v):6.2f}   p90 {np.percentile(v, 90):6.2f}   "
              f"max {v.max():7.2f} det/chan-min   zero-rate channels {(v == 0).sum():3d}/{n}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    stem = sys.argv[1] if len(sys.argv) > 1 else "P1_pre_qcfinalv2"
    figure(stem, sys.argv[2] if len(sys.argv) > 2 else "rate")
