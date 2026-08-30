r"""
sdc.compare.raster
------------------
Detection raster for one recording, optionally restricted to a channel subset.

    .venv\Scripts\python.exe -m sdc.compare.raster P1_ANT2_stim_qcfinalv2 [top20]

WHY THIS EXISTS ALONGSIDE run_windows.draw_raster
  `draw_raster` reads the module-level RECORDING/RUN_TAG that the env sets for a detector run, and
  always draws every channel. This one takes an explicit stem and an explicit channel set, which
  is what looking at a GATED subset needs -- and with a few dozen channels every channel can be
  labelled instead of one tick in twenty, so a claim about a specific contact is checkable.

THE `top20` GATE
  Top 20% of channels by their rate in the STIM-FREE pre recording, averaged across detectors.

  Measured on the pre file so the gate cannot select on the effect being estimated, and averaged
  across detectors because the raster shares one channel axis -- a per-detector gate (which is
  what `rate_gate_box` uses, correctly, for a per-detector statistic) would draw three different
  channel sets on top of each other and the rows would not correspond.

  This is the subset where the three detectors stop disagreeing on the 2 Hz file: 3.24x spread
  across all 189 channels, 1.19x at p80 (`sdc.artefact.exposure.rate_gate_box`). The raster is
  how to check that the agreement is real rather than arithmetic -- if the retained channels show
  three detectors marking the same events at the same times, it is real.
"""
import sys
from pathlib import Path

import numpy as np

from sdc.common.paths import RUNS, ROOT

COLORS = {"Janca": "#c8102e", "Barkmeier": "#0072b2", "Delphos": "#4a3aa7"}


def _rate(z, names):
    mins = np.asarray(z["clean_per_sec"], float).sum(axis=0) / float(z["fs"]) / 60.0
    dets = [d for d in COLORS if f"{d}_idx" in z.files]
    per = np.vstack([np.where(mins > 0,
                              np.bincount(z[f"{d}_chan"], minlength=len(names)) /
                              np.maximum(mins, 1e-9), np.nan) for d in dets])
    return np.where(np.isnan(per).all(0), np.nan,                    # unmeasurable -> nan,
                    np.nanmean(np.where(np.isnan(per), 0, per), axis=0))   # not a warning


def top_channels(pre_stem, stim_stem=None, q=80.0):
    """Channel names at or above the q-th percentile of mean-across-detector baseline rate.

    The percentile is taken over the channels measurable in BOTH files, not over every channel
    in the pre file. `rate_gate_box` computes its gate on that same population, and a percentile
    of a different denominator is a different gate -- on this recording, 45 channels instead of
    the 38-39 the reported p80 result refers to."""
    z = np.load(RUNS / f"{pre_stem}.npz", allow_pickle=False)
    names = np.array([str(x) for x in z["names"]])
    r = _rate(z, names)
    ok = np.isfinite(r) & (r > 0)
    if stim_stem:
        zs = np.load(RUNS / f"{stim_stem}.npz", allow_pickle=False)
        sn = [str(x) for x in zs["names"]]
        rs = _rate(zs, sn)
        fin = {c for c, v in zip(sn, np.isfinite(rs)) if v}
        ok &= np.array([c in fin for c in names])
    return sorted(names[ok & (r >= np.nanpercentile(r[ok], q))]), r, names, ok


def figure(stem, chans=None, label="", outdir=None, out_name=None):
    import matplotlib.pyplot as plt
    from seeg._style import recessive

    z = np.load(RUNS / f"{stem}.npz", allow_pickle=False)
    names = [str(s) for s in z["names"]]
    fs, T = float(z["fs"]), float(z["seconds"])
    keep = [i for i, c in enumerate(names) if c in set(chans)] if chans else list(range(len(names)))
    dets = [d for d in ("Janca", "Barkmeier", "Delphos") if f"{d}_idx" in z.files]
    per = {d: [np.sort(z[f"{d}_idx"][z[f"{d}_chan"] == k] / fs) for k in keep] for d in dets}
    nk = len(keep)
    # one shared order, from the FIRST detector, so a row is the same channel in every panel
    order = np.argsort([-per[dets[0]][i].size for i in range(nk)])

    edges = np.arange(0, T + 1, 1.0)
    fig, axes = plt.subplots(len(dets) + 1, 1, sharex=True,
                             figsize=(17, 4 + max(2.4, nk * 0.085) * len(dets)),
                             gridspec_kw={"height_ratios": [1] + [3] * len(dets)})
    axr = axes[0]
    for d in dets:
        allt = np.concatenate([p for p in per[d] if p.size]) if any(p.size for p in per[d]) \
            else np.zeros(0)
        axr.plot(edges[:-1] + .5, np.histogram(allt, bins=edges)[0], color=COLORS[d], lw=1.0,
                 label=f"{d} ({sum(p.size for p in per[d])})")
    axr.set_ylabel("pop. rate\n(spikes/s)")
    axr.legend(loc="upper right", frameon=False, fontsize=8, ncol=len(dets))
    recessive(axr)

    on = z["on_runs"] / fs if "on_runs" in z.files else np.zeros((0, 2))
    for a0, a1 in np.atleast_2d(on) if on.size else []:
        for ax in axes:
            ax.axvspan(a0, a1, color="#f0c419", alpha=.16, lw=0, zorder=0)
    if on.size:
        axr.text(.004, .96, "shaded = stim ON", transform=axr.transAxes, va="top",
                 fontsize=8, color="#8a6d00")

    for ax, d in zip(axes[1:], dets):
        ax.eventplot([per[d][order[i]] for i in range(nk)], colors=COLORS[d],
                     lineoffsets=np.arange(nk), linelengths=.8,
                     linewidths=.8 if nk <= 60 else .4)
        ax.set_title(f"{d} (n={sum(p.size for p in per[d])})", loc="left", fontsize=10,
                     color=COLORS[d])
        ax.set_ylabel("channel (busiest -> top)")
        # every channel gets a label when there are few enough for it to be legible
        yt = np.arange(nk) if nk <= 60 else np.arange(0, nk, max(nk // 20, 1))
        ax.set_yticks(yt)
        ax.set_yticklabels([names[keep[order[i]]] for i in yt], fontsize=6.5 if nk <= 60 else 6)
        ax.set_ylim(nk, -1)
        recessive(ax)
    axes[-1].set_xlim(0, T)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{stem}{('  [' + label + ']') if label else ''}: {T:.0f}s ({T / 60:.1f} min), "
                 f"{nk} of {len(names)} bipolar channels at {fs:g} Hz", fontsize=11)
    fig.tight_layout()
    out = (Path(outdir) if outdir else ROOT / "figures" / "real" / stem) / \
        (out_name or "compare_raster.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    stem = sys.argv[1] if len(sys.argv) > 1 else "P1_ANT2_stim_qcfinalv2"
    if len(sys.argv) > 2 and sys.argv[2] == "top20":
        pre = sys.argv[3] if len(sys.argv) > 3 else "P1_ANT2_pre_qcfinalv2"
        ch, r, nm, ok = top_channels(pre, stem)
        print(f"top20 gate on {pre}: {len(ch)} of {int(ok.sum())} measurable "
              f"({len(nm)} total) channels, "
              f"baseline rate >= {np.nanpercentile(r[ok], 80):.2f} det/chan-min")
        print("  " + " ".join(ch))
        figure(stem, ch, label="top 20% by baseline rate",
               outdir=ROOT / "figures" / "real" / f"{stem}_top20")
    else:
        figure(stem)
