"""
sdc.scoring.bids_per_channel
----------------------------
Does a detector's output rate TRACK a channel's true activity, or does it emit at its own rate
regardless? Tested per channel, on 25 BIDS subjects, against our own three marked blocks.

    .venv\\Scripts\\python.exe -m sdc.scoring.bids_per_channel

THE HYPOTHESIS
  On our marks, Janca and Delphos fire 4-9 detections per channel-minute on channels the rater
  viewed and found EMPTY, while Barkmeier fires 0.1-0.5. If that is a normalisation problem --
  a per-channel threshold referenced to the channel's own amplitude, with no absolute floor --
  then on a quiet channel the threshold descends into the noise and the detector emits anyway.
  The signature is a detection rate that is FLAT across channels of very different true rates.

WHY NOT JUST COUNT FALSE POSITIVES AGAIN
  Because at their published defaults the three sit at 5.49 / 3.22 / 2.08 det/chan-min. A raw
  count on quiet channels would rank them by operating point, not by normalisation, and Janca
  would look worst for the trivial reason that it emits most everywhere. Both statistics below
  are therefore SHAPE statistics -- invariant to multiplying a detector's output by a constant:

    A  rho(detection rate, expert mark rate) across channels within a subject.
       1.0 = tracks channel activity fully.  0.0 = ignores it.
    B  median detection rate on ZERO-mark channels / median on MARKED channels.
       0.0 = silent where there is nothing.  1.0 = same rate whether or not there are spikes.

  Neither changes if a detector is turned uniformly up or down, so they isolate the thing the
  hypothesis is actually about.

THE CAVEAT THAT LIMITS B ON BIDS, AND WHY A SURVIVES IT
  The BIDS experts marked discharges they considered notable in a ~3 min sleep window; they did
  not exhaustively mark every channel, so a BIDS channel with zero marks may be quiet OR may be
  unannotated. B on BIDS is therefore an upper bound on how clean a detector is, not a
  measurement -- exactly the same limitation `score_labelled` records for precision.

  Statistic A is far less exposed, because it is computed over the MARKED channels' rank order:
  it asks whether a detector emits more on channels the expert marked more, which does not
  depend on the unmarked channels being truly empty. And both are compared BETWEEN detectors on
  identical channels, which is safe under any annotation policy.

  Our own blocks have no such caveat -- every channel shown was marked exhaustively, empty ones
  included -- so they anchor the BIDS result rather than merely repeating it.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from sdc.common.paths import RUNS, figdir
from sdc.common.spike_match import match
from sdc.scoring.bids_events import subjects, load_subject, truth_per_channel

BIDS_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(r"C:\Users\amoo0039\Documents\ieeg_ieds_bids_final\ieeg_ieds_bids")
DETS = ("Janca", "Barkmeier", "Delphos")
COLORS = {"Janca": "#c8102e", "Barkmeier": "#0072b2", "Delphos": "#4a3aa7"}
TOL = 0.050
MIN_MARKED = 4          # subjects with fewer marked channels cannot support a rank correlation


def _stats(chans, mins, marks, dets):
    """Statistics A and B from per-channel (minutes, mark times, detection times)."""
    mins = np.asarray(mins, float)
    mr = np.array([m.size for m in marks], float) / np.maximum(mins, 1e-9)
    out = {"chan": list(chans), "mins": mins, "mark_rate": mr,
           "n_mark": np.array([m.size for m in marks]), "n_marked_ch": int((mr > 0).sum())}
    for d, per in dets.items():
        dr = np.array([t.size for t in per], float) / np.maximum(mins, 1e-9)
        hit = np.array([int(match(t, m, TOL)[1].sum()) if (t.size and m.size) else 0
                        for t, m in zip(per, marks)], float)
        good = np.isfinite(dr) & np.isfinite(mr)
        rho = spearmanr(mr[good], dr[good])[0] if good.sum() >= MIN_MARKED else np.nan
        z, nz = mr == 0, mr > 0
        ratio = (np.median(dr[z]) / np.median(dr[nz])
                 if z.sum() and nz.sum() and np.median(dr[nz]) > 0 else np.nan)
        out[d] = {"det_rate": dr, "hit": hit, "rho": rho, "ratio": ratio,
                  "quiet_rate": np.median(dr[z]) if z.sum() else np.nan,
                  "busy_rate": np.median(dr[nz]) if nz.sum() else np.nan,
                  "n_zero_ch": int(z.sum())}
    return out


def bids_subject(sub, root=BIDS_ROOT):
    d = load_subject(root, sub)
    z = np.load(RUNS / f"bids_{sub}.npz", allow_pickle=False)
    names = [str(x) for x in z["names"]]
    fs = float(z["fs"])
    mins = np.asarray(z["clean_sec_off"], float) / 60.0
    truth = truth_per_channel(d["times"], d["chan_lists"], d["channels"])
    tmap = dict(zip(d["channels"], truth))
    marks = [np.asarray(tmap.get(c, []), float) for c in names]
    dets = {dd: [np.sort(z[f"{dd}_idx"][z[f"{dd}_chan"] == i] / fs) for i in range(len(names))]
            for dd in DETS}
    s = _stats(names, mins, marks, dets)
    s["label"] = sub
    s["source"] = "BIDS"
    return s


def ours():
    """The same statistics on our exhaustively marked blocks."""
    from sdc.scoring.score_marks import collect, _tag
    out = []
    for st in collect(TOL):
        rows = st["rows"]
        mins = [(st["t1"] - st["t0"]) / 60.0] * len(rows)
        # score_marks already matched; rebuild only what _stats needs
        mr = np.array([r["n_mark"] for r in rows], float) / mins[0]
        s = {"chan": [r["chan"] for r in rows], "mins": np.asarray(mins),
             "mark_rate": mr, "n_mark": np.array([r["n_mark"] for r in rows]),
             "n_marked_ch": int((mr > 0).sum()), "label": _tag(st), "source": "ours"}
        for d in DETS:
            dr = np.array([r[d]["n_det"] for r in rows], float) / mins[0]
            z, nz = mr == 0, mr > 0
            s[d] = {"det_rate": dr, "hit": np.array([r[d]["hit"] for r in rows], float),
                    "rho": spearmanr(mr, dr)[0],
                    "ratio": np.median(dr[z]) / np.median(dr[nz]) if np.median(dr[nz]) > 0
                    else np.nan,
                    "quiet_rate": np.median(dr[z]), "busy_rate": np.median(dr[nz]),
                    "n_zero_ch": int(z.sum())}
        out.append(s)
    return out


def collect_all(root=BIDS_ROOT):
    sets = []
    for sub in subjects(root):
        if not (RUNS / f"bids_{sub}.npz").is_file():
            continue
        try:
            sets.append(bids_subject(sub, root))
        except Exception as e:                                   # noqa: BLE001
            print(f"  [skip] {sub}: {e}")
    return sets, ours()


def figure(root=BIDS_ROOT, outdir=None):
    import matplotlib.pyplot as plt

    bids, mine = collect_all(root)
    usable = [s for s in bids if s["n_marked_ch"] >= MIN_MARKED]
    top = sorted(usable, key=lambda s: -s["n_marked_ch"])[:2]

    print(f"BIDS: {len(bids)} subjects, {len(usable)} with >={MIN_MARKED} marked channels\n")
    print(f"  {'set':<12}{'ch':>4}{'mkd':>5}{'zero':>5}   " +
          "  ".join(f"{d[:4]:>4} rho  B" for d in DETS))
    for s in usable + mine:
        cells = [f"{s[d]['rho']:>+5.2f} {s[d]['ratio']:>4.2f}" for d in DETS]
        print(f"  {s['label']:<12}{len(s['chan']):>4}{s['n_marked_ch']:>5}"
              f"{len(s['chan']) - s['n_marked_ch']:>5}   " + "  ".join(cells))

    print("\n  A: rho(detection rate, expert mark rate) across channels -- median [IQR]")
    for grp, name in ((usable, f"BIDS n={len(usable)}"), (mine, f"ours n={len(mine)}")):
        for d in DETS:
            v = np.array([s[d]["rho"] for s in grp], float)
            v = v[np.isfinite(v)]
            print(f"    {name:<12}{d:<10}{np.median(v):>+6.2f}  "
                  f"[{np.percentile(v, 25):+.2f}, {np.percentile(v, 75):+.2f}]")
    print("\n  B: median det rate on zero-mark ch / on marked ch -- median [IQR]")
    for grp, name in ((usable, f"BIDS n={len(usable)}"), (mine, f"ours n={len(mine)}")):
        for d in DETS:
            v = np.array([s[d]["ratio"] for s in grp], float)
            v = v[np.isfinite(v)]
            print(f"    {name:<12}{d:<10}{np.median(v):>6.2f}  "
                  f"[{np.percentile(v, 25):.2f}, {np.percentile(v, 75):.2f}]   n={v.size}")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))

    # (a)(b) the two BIDS subjects with the most marked channels, per-channel scatter
    for ax, s in zip((axes[0, 0], axes[0, 1]), top):
        mr = s["mark_rate"]
        for d in DETS:
            dr = s[d]["det_rate"]
            m = mr > 0
            ax.plot(mr[m], dr[m], "o", ms=7, color=COLORS[d], alpha=.75,
                    label=f"{d}  rho={s[d]['rho']:+.2f}")
            if (~m).any():                       # zero-mark channels, parked at the left edge
                ax.plot(np.full((~m).sum(), 0.06), dr[~m], "o", ms=7, color=COLORS[d],
                        alpha=.75, mfc="none", mew=1.5)
        ax.axvline(0.11, color="0.7", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("expert marks / chan-min   (open markers, left of dotted = ZERO marks)")
        ax.set_ylabel("detections / chan-min")
        ax.set_title(f"{s['label']}  ({s['n_marked_ch']} marked, "
                     f"{len(s['chan']) - s['n_marked_ch']} unmarked channels)",
                     fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.grid(alpha=.3)

    def _dist(ax, key, ylab, title, ref=None):
        for j, d in enumerate(DETS):
            for k, (grp, jit, mfc) in enumerate(((usable, -0.16, COLORS[d]),
                                                 (mine, 0.16, "none"))):
                v = np.array([s[d][key] for s in grp], float)
                v = v[np.isfinite(v)]
                if not v.size:
                    continue
                x = np.full(v.size, j + jit) + np.random.default_rng(0).normal(0, .035, v.size)
                ax.plot(x, v, "o", ms=6, color=COLORS[d], mfc=mfc, mew=1.4, alpha=.75,
                        label=("BIDS" if k == 0 else "ours") if j == 0 else None)
                ax.hlines(np.median(v), j + jit - .12, j + jit + .12, color="0.2", lw=2.2)
        if ref is not None:
            ax.axhline(ref, color="0.4", ls="--", lw=1.1)
        ax.set_xticks(range(len(DETS)))
        ax.set_xticklabels(DETS)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=.3)

    _dist(axes[1, 0], "rho", "rho(det rate, mark rate)",
          "A. Does output track channel activity?\n(1 = tracks fully, 0 = ignores the channel)",
          ref=0.0)
    axes[1, 0].set_ylim(-1.05, 1.05)
    _dist(axes[1, 1], "ratio", "quiet / busy detection rate",
          "B. Rate on zero-mark channels, relative to marked ones\n"
          "(0 = silent where there is nothing, 1 = emits the same either way)", ref=1.0)
    axes[1, 1].set_yscale("symlog", linthresh=0.1)

    fig.suptitle("Per-channel normalisation: does the detector know the channel is quiet?",
                 fontsize=12)
    fig.tight_layout()
    out = (Path(outdir) if outdir else figdir("real")) / "bids_per_channel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")
    return bids, mine


if __name__ == "__main__":
    figure()
