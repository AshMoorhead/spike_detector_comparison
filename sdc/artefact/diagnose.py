"""
sdc.artefact.diagnose
---------------------
Two claims about why the block-paired result differs from the whole-condition one, tested
directly instead of inferred from a handful of per-block numbers.

    .venv\\Scripts\\python.exe -m sdc.artefact.diagnose

CLAIM 1 -- "the recording drifts, and that is what the whole-condition contrast was picking up"
  The evidence originally offered for this was six per-block OFF rates from one detector with
  no interval on any of them, and the largest of the six was block 0 -- the ONE block whose
  matched OFF window is taken AFTER the ON block rather than before, because stimulation
  starts at t=6 s and there is no room in front of it. So the single point anchoring the trend
  was also the single structurally different point, and it is equally consistent with
  post-stimulation rebound. That is not evidence of drift.

  Tested properly here on the PRE recordings, which contain no stimulation at all. Any time
  trend there is drift and nothing else, with no stim effect and no block structure to confuse
  it. Reported as a least-squares slope on the MEDIAN per-channel rate, with a bootstrap
  interval over channels -- see `summarise` and `fit` for why that pairing and not another.

CLAIM 2 -- "P5's effect lived in the channels the block design drops"
  The decomposition behind this WAS a direct measurement -- same detections, same estimator,
  only the channel set changed, and Janca moved 0.658 -> 0.923. What was never checked is the
  explanation: that those channels are the quiet ones. Checked here by comparing the dropped
  and kept channels on their OFF rate and on their own ON/OFF ratio.

WHY BOTH MATTER: they are the difference between "stimulation does nothing and the old result
was a time confound" and "stimulation does something that the block design is too blunt or too
channel-hungry to see". Those call for opposite next steps.
"""
import numpy as np
import matplotlib.pyplot as plt

from seeg._style import RED, BLUE, MUTED, recessive

from sdc.common import cond
from sdc.common.paths import RUNS, figdir
from sdc.artefact import ratio_metrics as rm
from sdc.artefact.blocks import block_table

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}
BIN_SEC = 60.0
CASCADE = {}          # rec -> (implanted, survive artefact, enough counts, block-paired)
N_BOOT = 2000
SEED = 0


def binned_rates(rec, bin_sec=BIN_SEC):
    """Median per-channel rate in fixed time bins, over ANALYSABLE time.

    Returns (bin_centres, {detector: per_channel_rate[bin, chan]}, on_frac).
    Channels are restricted to those measurable overall, so a bin's summary is taken over a
    stable channel set rather than a different one in every bin.
    """
    z = np.load(RUNS / f"{rec}.npz", allow_pickle=False)
    fs, n = float(z["fs"]), len(z["names"])
    cps = z["clean_per_sec"]
    n_sec = cps.shape[0]
    edges = np.arange(0, n_sec + 1, int(bin_sec))
    if edges[-1] < n_sec:
        edges = np.append(edges, n_sec)
    nb = edges.size - 1

    sel = cond.select(z, "all")
    keep = sel.measurable if sel.measurable is not None else np.ones(n, bool)

    clean = np.array([cps[edges[b]:edges[b + 1]].sum(axis=0) / fs for b in range(nb)])
    out = {}
    for d in [str(s) for s in z["detectors"]]:
        t, c = z[f"{d}_idx"] / fs, z[f"{d}_chan"]
        cnt = np.zeros((nb, n))
        for b in range(nb):
            m = (t >= edges[b]) & (t < edges[b + 1])
            cnt[b] = np.bincount(c[m], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cnt / clean * 60.0
        r[~np.isfinite(r)] = np.nan
        out[d] = r[:, keep]

    on_frac = np.zeros(nb)
    if "on_per_sec" in z.files:
        ops = np.asarray(z["on_per_sec"], bool)
        for b in range(nb):
            seg = ops[edges[b]:min(edges[b + 1], ops.size)]
            on_frac[b] = seg.mean() if seg.size else 0.0
    return 0.5 * (edges[:-1] + edges[1:]), out, on_frac


def fit(x, y):
    """Ordinary least squares. Returns (slope, intercept).

    Theil-Sen (median of all pairwise slopes) was used here first, on the reasoning that it
    resists outlying bins. On THIS data it fails outright: the bin summary used to be a median
    over per-channel rates, i.e. a median over small integer counts, so it lands on 1.00, 2.00,
    3.00 and most pairwise slopes are exactly zero -- whereupon the median of those slopes is
    also exactly zero. Every slope it reported was 0.000 with intervals like [0.000, 0.000].

    The real fault was the bin summary, not the fit. Averaging per-channel rates instead of
    taking their median gives a continuous quantity, and once the quantisation is gone there is
    nothing left for a robust fit to protect against that a plain least-squares line does not
    handle -- so this is now just a mean-based fit, which is also the thing anyone reading the
    figure will assume the line is.
    """
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return np.nan, np.nan
    s, c = np.polyfit(x, y, 1)
    return float(s), float(c)


def summarise(r):
    """One number per bin: the MEDIAN per-channel rate.

    Consistent with every other rate figure in the project (rate_comparators, the block
    estimator), and the right choice on a distribution this skewed -- a handful of very busy
    contacts would otherwise set the level.

    IT QUANTISES, and the figure shows it: a 60 s bin gives each channel an integer count, so
    a per-channel rate is an integer per minute and their median lands on an integer or a half.
    Panels (c)/(d) band horizontally because of this, not because the brain does.

    That is survivable for the LINE but was fatal for the first slope estimator used here.
    Theil-Sen takes the median of all pairwise slopes; with quantised y most pairs give exactly
    zero, so it returned exactly 0.000 for every recording. Least squares has no such failure --
    it uses the values themselves rather than their ordering -- which is why `fit` is OLS and
    why the median can be kept.
    """
    return np.nanmedian(r, axis=1)


def drift(recs=("P1_pre", "P5_pre", "P1_stim", "P5_stim")):
    """Time trend in spike rate, per detector. On the PRE files there is no stimulation."""
    rng = np.random.default_rng(SEED)
    res = {}
    print(f"\n=== TIME TREND in MEAN per-channel rate ({BIN_SEC:g}s bins, least squares)")
    print(f"    slope in det/min per HOUR, 95% interval bootstrapped over channels")
    print(f"\n    {'recording':<11}{'detector':<11}{'bins':>6}{'slope/hr':>11}"
          f"{'95% CI':>22}  {'fitted start->end':<20}")
    for rec in recs:
        t, per_det, on_frac = binned_rates(rec)
        res[rec] = (t, per_det, on_frac)
        for d, r in per_det.items():
            y = summarise(r)
            s, c = fit(t, y)
            b = []
            for _ in range(400):
                cols = rng.integers(0, r.shape[1], r.shape[1])
                b.append(fit(t, summarise(r[:, cols]))[0] * 3600.0)
            lo, hi = np.nanpercentile(b, [2.5, 97.5])
            flag = "" if (lo < 0 < hi) else "  *"
            print(f"    {rec:<11}{d:<11}{len(y):>6}{s * 3600:>11.3f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>22}"
                  f"  {c + s * t[0]:.2f} -> {c + s * t[-1]:.2f} /min{flag}")
    return res


def dropped_channels(rec="P5_stim"):
    """What are the channels the block-paired design drops, and WHY?

    Two gates do the dropping and they mean opposite things:
      no_time    the dilated artefact mask left under 20% of the matched window analysable.
                 This channel was LOST TO ARTEFACT.
      no_count   fewer than 3 OFF detections in the matched window. The channel is simply
                 quiet, and would have been dropped on a perfectly clean recording too.
    They overlap, so the counts below are reported as artefact-only / quiet-only / both rather
    than forced into one bucket each.
    """
    z = np.load(RUNS / f"{rec}.npz", allow_pickle=False)
    p, bp = rm.load(rec), block_table(z)
    inblk = bp.keep[p.keep]                      # of p's channels, which survive block pairing
    no_time, no_count = bp.no_time[p.keep], bp.no_count[p.keep]

    # The cascade, recorded for the figure title. `rm.load` has ALREADY applied the artefact
    # gate, so by the time this function runs the artefact casualties are gone and every
    # remaining drop is a count. Reporting only the visible drops would say "artefact costs us
    # nothing", which is the reverse of what the first step shows.
    n_impl = len(z["names"])
    ON, OFF = cond.select(z, "on"), cond.select(z, "off")
    n_art = int((ON.measurable & OFF.measurable).sum())
    CASCADE[rec] = (n_impl, n_art, p.n, bp.n_chan)
    art_only = no_time & ~no_count
    quiet_only = no_count & ~no_time
    both = no_time & no_count

    print(f"\n=== {rec}: {p.n} channels whole-condition, {int(inblk.sum())} survive block "
          f"pairing, {int((~inblk).sum())} dropped")
    print(f"    of the {int((~inblk).sum())} dropped: {int(art_only.sum())} lost to ARTEFACT "
          f"only, {int(quiet_only.sum())} too QUIET only, {int(both.sum())} both")
    print(f"\n    {'detector':<11}{'group':<14}{'n':>4}{'OFF rate':>10}{'ON rate':>10}"
          f"{'ratio':>8}")
    out = {}
    for d in p.det:
        a = p.det[d]
        r = rm._ratio(a, "clip")
        for tag, m in (("kept", inblk), ("dropped: artefact", no_time),
                       ("dropped: quiet", quiet_only)):
            if not m.any():
                continue
            print(f"    {d:<11}{tag:<14}{int(m.sum()):>4}"
                  f"{np.median(a['off_rate'][m]):>10.3f}{np.median(a['on_rate'][m]):>10.3f}"
                  f"{np.median(r[m]):>8.3f}")
        out[d] = (r, inblk, a, no_time, no_count)
    return out


def figure(res, drop):
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 12.0))

    # ---- (a,b) one continuous clock: baseline THEN stim -----------------------------------
    # The baseline recording happened BEFORE the stimulation recording, so drawing both from
    # t=0 asserts they are contemporaneous, which is exactly the assumption being questioned.
    # The EDF start timestamps are anonymised to 2001-01-01 00:00:00 in all four files, so the
    # true gap between them is unrecoverable -- the baseline is therefore abutted directly onto
    # the start of the stim file and the unknown gap is marked rather than invented.
    for k, (pre, stim) in enumerate((("P1_pre", "P1_stim"), ("P5_pre", "P5_stim"))):
        ax = axes[0][k]
        t0, _, _ = res[pre]
        pre_len = t0[-1] + BIN_SEC / 2
        for rec, off, ls in ((pre, -pre_len, "--"), (stim, 0.0, "-")):
            t, per_det, on_frac = res[rec]
            for d, r in per_det.items():
                ax.plot((t + off) / 60.0, summarise(r), ls, lw=1.3,
                        color=COLORS.get(d, MUTED),
                        label=f"{d}" if rec == stim else None)
            for i, f in enumerate(on_frac):
                if f > 0.5:
                    ax.axvspan((t[i] + off - BIN_SEC / 2) / 60, (t[i] + off + BIN_SEC / 2) / 60,
                               color="#f0c419", alpha=.20, lw=0)
        ax.axvline(0, color="0.25", lw=1.4)
        # Bottom of the axes, not the top: x=0 falls mid-panel on P5 and the note collided
        # with the legend there.
        ax.annotate("baseline ends | stim file begins\n(true gap unknown -- EDF timestamps "
                    "anonymised in all four files)",
                    (0, 0.02), xycoords=("data", "axes fraction"), fontsize=6.8,
                    color="0.35", ha="center", va="bottom")
        recessive(ax); ax.grid(alpha=.3)
        ax.set_xlabel("time (min)   -- negative = baseline recording")
        ax.set_ylabel("median rate (det/min)")
        ax.set_title(f"({'ab'[k]}) {pre[:2]}  dashed = baseline (no stim), solid = stim file, "
                     f"shading = stim ON", fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=7, ncol=3)

    # ---- (c,d) baseline only, with the fitted line labelled -------------------------------
    for k, rec in enumerate(("P1_pre", "P5_pre")):
        ax = axes[1][k]
        t, per_det, _ = res[rec]
        # Labels are staggered along the line rather than all pinned to its right-hand end --
        # the three fits converge there on P5 and the text overlapped itself.
        for i, (d, r) in enumerate(per_det.items()):
            c = COLORS.get(d, MUTED)
            y = summarise(r)
            ax.scatter(t / 60.0, y, s=16, color=c, alpha=.55)
            s, b = fit(t, y)
            ax.plot(t / 60.0, b + s * t, lw=1.8, color=c)
            xa = t[0] + (0.42 + 0.24 * i) * (t[-1] - t[0])
            ax.annotate(f"{d}: {b + s * t[0]:.2f} -> {b + s * t[-1]:.2f} /min "
                        f"({s * 3600:+.2f}/hr)",
                        (xa / 60.0, b + s * xa), fontsize=7.2, color=c,
                        ha="center", va="bottom", xytext=(0, 5),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                                  alpha=.75))
        recessive(ax); ax.grid(alpha=.3)
        ax.set_xlabel("time (min)"); ax.set_ylabel("median rate (det/min)")
        ax.set_title(f"({'cd'[k]}) {rec} -- NO stimulation anywhere in this panel",
                     fontsize=9, loc="left")

    # ---- (e,f) dropped vs kept channels ---------------------------------------------------
    for k, (rec, dd) in enumerate(drop.items()):
        ax = axes[2][k]
        hi = 0.0
        for d, (r, inblk, a, no_time, no_count) in dd.items():
            c = COLORS.get(d, MUTED)
            hi = max(hi, float(np.nanpercentile(a["off_rate"], 98)))
            y = np.clip(r, 0.05, 20)
            # Artefact-lost channels get a halo behind them so they stand out from the merely
            # quiet ones -- "did artefact cost us this channel" is the question the panel is
            # here to answer, and a bare marker shape does not carry it.
            if no_time.any():
                ax.scatter(a["off_rate"][no_time], y[no_time], s=150, marker="o",
                           facecolor="#f0c419", alpha=.35, edgecolor="none", zorder=1)
            for tag, m, mk in (("kept", inblk, "o"),
                               ("dropped: too quiet", (~inblk) & ~no_time, "x"),
                               ("dropped: LOST TO ARTEFACT", no_time, "P")):
                if m.any():
                    ax.scatter(a["off_rate"][m], y[m], s=22 if mk != "P" else 34, marker=mk,
                               facecolor="none" if mk == "o" else c, edgecolor=c,
                               alpha=.75, lw=1.1, zorder=3,
                               label=f"{tag}" if d == "Janca" else None)
        ax.axhline(1.0, color="0.35", ls="--", lw=1.1)
        # Linear x as asked. The 98th percentile caps it so one very busy channel does not
        # squash every other point into the left-hand margin; y stays log because a ratio is
        # multiplicative and 0.5 must look as far from 1 as 2.0 does.
        ax.set_xlim(0, hi)
        ax.set_yscale("log")
        recessive(ax); ax.grid(alpha=.3)
        ax.set_xlabel("OFF rate (det/min, linear, clipped at p98)")
        ax.set_ylabel("ON/OFF ratio (log)")
        # The artefact-lost channels are NOT on this panel -- they have no usable rate, so
        # they were removed before it was drawn. Without the cascade in the title the crosses
        # read as "the artefact casualties", which is the opposite of the truth.
        casc = CASCADE.get(rec)
        ax.set_title(f"({'ef'[k]}) {rec}: why block pairing drops a channel"
                     + (f"\n{casc[0]} implanted -> {casc[1]} survive ARTEFACT "
                        f"(-{casc[0] - casc[1]}, not plotted) -> {casc[2]} have enough "
                        f"detections -> {casc[3]} paired" if casc else ""),
                     fontsize=8.5, loc="left")
        ax.legend(frameon=False, fontsize=7)

    fig.suptitle("(a,b) baseline placed before the stim file on one clock.  "
                 "(c,d) baseline alone, fitted rate labelled.  "
                 "(e,f) are the channels block-pairing drops different?", fontsize=11)
    fig.tight_layout()
    out = figdir("real") / "diagnose_drift_channels.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    _res = drift()
    _drop = {r: dropped_channels(r) for r in ("P1_stim", "P5_stim")}
    figure(_res, _drop)
