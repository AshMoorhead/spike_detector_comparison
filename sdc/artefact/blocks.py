"""
sdc.artefact.blocks
-------------------
The block-paired ON/OFF design. This is the primary estimator; everything in
`ratio_metrics.py` that compares all-of-ON against all-of-OFF is superseded by it.

    .venv\\Scripts\\python.exe -m sdc.artefact.blocks
    .venv\\Scripts\\python.exe -m sdc.artefact.blocks --null      # calibration check

WHY THE WHOLE-CONDITION DESIGN HAD TO GO
  Spike rate drifts over a night -- sleep stage, arousals, medication. Comparing every ON
  minute against every OFF minute puts that drift inside the comparison, where it has to be
  argued away statistically afterwards. Two things went wrong as a result:

    * Judging estimators by how rarely they confused drift with stimulation would have
      selected the estimator LEAST able to see change of any kind. The blunt one wins that
      contest: on P5 the matched-count estimator returns exactly 0.000 for all three
      detectors, and would have looked admirably specific while measuring nothing.
    * The confidence intervals were badly wrong. Bootstrapping CHANNELS assumes channels are
      the thing that varies; they are not. All channels share the same minutes, so their
      errors are correlated, and a channel bootstrap is structurally blind to the dominant
      noise. Measured on stim-free data it rejected a true null 46-100% of the time against a
      nominal 5% (nullcheck.py).

  Both have the same cure, and it is a design change rather than a statistical one: compare
  each ON block only against the OFF time IMMEDIATELY ADJACENT to it. Slow drift cancels
  inside each local contrast instead of being modelled, and the unit that actually varies --
  the block -- becomes the unit that is resampled.

WHY COUNTS ARE POOLED ACROSS BLOCKS RATHER THAN AVERAGED PER BLOCK
  A block is ~65 s and a typical channel fires at 0.5-5 /min, so a single block-channel cell
  holds a handful of detections and often zero. A per-cell ratio would be mostly noise and
  mostly undefined. So the point estimate pools counts over whichever blocks are in play,
  then forms one ratio per channel; the block structure enters through the RESAMPLING, which
  is where it belongs and where sparsity does not bite.

THE PRICE, STATED PLAINLY
  There are 6 matched block pairs on P1 and 11 on P5. Resampling 6 things gives a wide, coarse
  interval, and no amount of channel count fixes that -- 118 channels observed over 6 blocks
  carry roughly 6 blocks' worth of information about a block-varying effect. The intervals
  here are much wider than the ones this project reported before. They are also the first ones
  that are right, and the width is the real message: blocks, not channels, are the scarce
  resource, which is worth knowing before trading channels away to artefact masking.
"""
import sys

from pathlib import Path

import numpy as np

from sdc.common import cond
from sdc.common.paths import RUNS
from sdc.artefact.ratio_metrics import MIN_OFF_RATE, _matched_windows, _runs_from_mask

N_BOOT = 4000
SEED = 0

AGG = "mean"         # how per-channel log ratios are combined into one number per block.
                     #
                     # "mean" is the GEOMETRIC MEAN of the ratios -- 10 ** mean(log10 r) -- and
                     # it is the primary. The log is what makes a mean legitimate: 0.5 and 2.0
                     # are -0.301 and +0.301 in log space and cancel to exactly 1.0, whereas an
                     # arithmetic mean of the raw ratios gives 1.25 and invents an increase.
                     #
                     # It beats the median on every count that matters here:
                     #   effect/SD   1.47 vs 1.15 (Janca P1), 1.48 vs 1.19 (Barkmeier P1) --
                     #               roughly the difference between needing 7 blocks and 11
                     #   no quantisation  a median over ~100 channels with small integer counts
                     #               lands on discrete values; P5's per-pair medians pin to
                     #               exactly 1.000 and 1.414 and cannot resolve anything smaller
                     #   same estimate   P1 0.611 median vs 0.638 geometric mean, so the gain
                     #               is in precision rather than in moving the answer
                     #
                     # "median" remains available and is worth reporting alongside wherever the
                     # per-channel distribution is skewed enough for the two to diverge -- on
                     # P5 they differ by ~9% (1.000 vs 0.908 for Barkmeier), which is the mean
                     # following the tail and is a real caveat, not a rounding difference.


class BlockPaired:
    """Per-(block, channel) counts and analysable time on duration-matched adjacent windows.

    Arrays are (n_block, n_chan_kept). `on_w[b]` and `off_w[b]` span the same number of
    seconds and sit next to each other in the recording.
    """

    def __init__(self, rec, names, keep, on_w, off_w, det, T_on):
        self.rec, self.names, self.keep = rec, names, keep
        self.on_w, self.off_w, self.det, self.T_on = on_w, off_w, det, T_on

    @property
    def n_block(self):
        return self.on_w.shape[0]

    @property
    def n_chan(self):
        return int(self.keep.sum())


def _win_counts(t, c, wins, n):
    """Detections per (window, channel)."""
    out = np.zeros((len(wins), n))
    for i, (t0, t1) in enumerate(wins):
        sel = (t >= t0) & (t < t1)
        out[i] = np.bincount(c[sel], minlength=n)
    return out


def _win_clean(cps, wins, fs):
    """Analysable seconds per (window, channel), from the stored per-second clean counts."""
    out = np.zeros((len(wins), cps.shape[1]))
    for i, (t0, t1) in enumerate(wins):
        a, b = int(np.floor(t0)), min(int(np.ceil(t1)), cps.shape[0])
        if b > a:
            out[i] = cps[a:b].sum(axis=0) / fs
    return out


def baseline_rates(pre_rec, names):
    """Per-channel rate (det/min) from a patient's stim-free BASELINE recording, aligned to
    `names`. Returns {detector: array}.

    Preferred over gating on the stim file's own OFF periods. Those sit between stimulation
    blocks and may carry carryover, so conditioning on them conditions -- weakly, but really --
    on the data being analysed. The baseline is a separate recording made before any
    stimulation, so a gate built from it cannot select on the outcome at all.
    """
    z = np.load(RUNS / f"{pre_rec}.npz", allow_pickle=False)
    pre_names = [str(s) for s in z["names"]]
    if pre_names != list(names):
        raise SystemExit(f"{pre_rec}: channel names differ from the stim run, so a baseline "
                         f"gate cannot be aligned by position.")
    sel = cond.select(z, "all")
    n = len(pre_names)
    out = {}
    for d in [str(s) for s in z["detectors"]]:
        c = np.bincount(z[f"{d}_chan"][sel.keep(d)], minlength=n)
        r = sel.rate(c) * 60.0
        out[d] = np.where(np.isfinite(r), r, 0.0)     # unmeasurable in baseline -> fails a gate
    return out


def block_table(z, on_sec_mask=None, min_off_rate=MIN_OFF_RATE, min_rate=0.0,
                gate_rates=None, tmax=None, prefer="before", chans=None,
                off_full=False):
    """Build a BlockPaired from an open run npz.

    `on_sec_mask` imposes an arbitrary per-second ON/OFF split in place of the recording's
    own, which is how the stim-free calibration check is built -- the null then travels this
    exact code path rather than an approximation of it.

    `tmax` keeps only pairs lying ENTIRELY before it, in seconds. Slicing this way is exact
    rather than approximate: every count and every analysable-time figure in this design is
    already computed inside the matched windows, so dropping whole pairs needs no other
    truncation. Its use is P1, whose recording stops mid-block and leaves an 8 s sixth ON run
    that the per-pair estimator would otherwise weight equally with the 64-66 s blocks.
    """
    if "clean_per_sec" not in z.files:
        raise SystemExit("block pairing needs `clean_per_sec`; re-merge this run.")
    names = [str(s) for s in z["names"]]
    dets = [str(s) for s in z["detectors"]]
    n, fs = len(names), float(z["fs"])
    cps = z["clean_per_sec"]

    if on_sec_mask is None:
        ON, OFF = cond.select(z, "on"), cond.select(z, "off")
        on_runs, off_runs, T_on = ON.runs, OFF.runs, ON.T
    else:
        m = np.asarray(on_sec_mask, bool)[:cps.shape[0]]
        on_runs, off_runs, T_on = _runs_from_mask(m), _runs_from_mask(~m), float(m.sum())

    on_w, off_w = _matched_windows(on_runs, off_runs, prefer=prefer,
                                   off_full=off_full)
    if on_w.shape[0] == 0:
        raise SystemExit("no ON block could be matched to an equal-duration adjacent OFF "
                         "window -- the condition structure does not support this design.")

    on_sec = _win_clean(cps, on_w, fs)
    off_sec = _win_clean(cps, off_w, fs)

    # Same usability gates as the whole-condition path, applied to the matched windows only:
    # a channel must have been observed in both, and must have a baseline to be relative to.
    # The two gates are recorded SEPARATELY because they mean completely different things --
    # one says the artefact mask ate the channel, the other says the channel is simply quiet --
    # and a figure that lumps them together cannot answer "did artefact cost us this channel?".
    tot_on, tot_off = on_sec.sum(axis=0), off_sec.sum(axis=0)
    span = float(np.diff(on_w, axis=1).sum())
    enough_time = (tot_on >= cond.Selection.MIN_CLEAN_FRAC * span) & \
                  (tot_off >= cond.Selection.MIN_CLEAN_FRAC * span)
    keep = enough_time.copy()
    if chans is not None:
        want = set(chans)
        missing = want - set(names)
        if missing:
            raise SystemExit(f"channels not in this recording: {sorted(missing)[:6]}")
        keep &= np.array([nm in want for nm in names], bool)
    enough_count = np.ones(n, bool)

    raw = {}
    for d in dets:
        t, c = z[f"{d}_idx"] / fs, z[f"{d}_chan"]
        oc, fc = _win_counts(t, c, on_w, n), _win_counts(t, c, off_w, n)
        raw[d] = (oc, fc)
        # The SAME rate bar the whole-condition path uses, so a channel is not dropped
        # here merely because the matched windows are shorter than the full OFF condition.
        with np.errstate(invalid="ignore", divide="ignore"):
            ok_rate = (fc.sum(axis=0) / np.maximum(off_sec.sum(axis=0), 1e-9) * 60.0) >= min_off_rate
        enough_count &= ok_rate
        keep &= ok_rate
        if min_rate > 0:
            # Never gate on the ON rate: a channel that stimulation silenced would fail such a
            # gate and be removed for having shown the effect. `gate_rates` (the separate
            # baseline recording) is the safest source; the stim file's own OFF periods are the
            # fallback when no baseline run is available.
            if gate_rates is not None:
                keep &= gate_rates[d] >= min_rate
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    keep &= (fc.sum(axis=0) /
                             np.maximum(off_sec.sum(axis=0), 1e-9) * 60.0) >= min_rate

    if tmax is not None:
        sel = np.array([max(a[1], b[1]) <= float(tmax)
                        for a, b in zip(on_w, off_w)], bool)
        if not sel.any():
            raise SystemExit(f"tmax={tmax:g}s leaves no complete pair.")
        on_w, off_w = on_w[sel], off_w[sel]
        on_sec, off_sec = on_sec[sel], off_sec[sel]
        raw = {d: (raw[d][0][sel], raw[d][1][sel]) for d in dets}

    det = {d: {"on_count": raw[d][0][:, keep], "off_count": raw[d][1][:, keep],
               "on_sec": on_sec[:, keep], "off_sec": off_sec[:, keep]}
           for d in dets}
    bp = BlockPaired(str(z["rec_id"]) if "rec_id" in z.files else "?", names, keep,
                     on_w, off_w, det, T_on)
    bp.min_rate = float(min_rate)
    # Why each dropped channel was dropped. `no_time` is the artefact mask; `no_count` is the
    # channel being too quiet. They overlap, so a figure should show the overlap explicitly
    # rather than assigning each channel to one bucket.
    bp.no_time = ~enough_time
    bp.no_count = ~enough_count
    bp.clean_frac_on = tot_on / max(span, 1e-9)
    bp.clean_frac_off = tot_off / max(span, 1e-9)
    return bp


def rate_gate(recs=(("P1_stim", "P1_pre"), ("P5_stim", "P5_pre")),
              grid=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0), source="baseline"):
    """Does requiring a minimum baseline rate make the result more consistent?

    The motivation is visible in diagnose.py panels (e)/(f): channels with a low rate carry
    ON/OFF ratios scattered over two orders of magnitude, while busy channels cluster tightly.
    A quiet channel's ratio is a ratio of two small integers, so it is mostly quantisation
    noise -- 1 detection against 3 reads as 0.33 and means almost nothing.

    `source="baseline"` gates on the patient's stim-free recording; `source="off"` gates on the
    stim file's own OFF periods. Baseline is preferred -- see `baseline_rates`.

    THREE QUANTITIES, AND THEY ARE NOT THE SAME THING
      CI          uncertainty in where the median sits. Shrinks with more blocks/channels.
                  This is the shaded band in the top row.
      channel     the interquartile range of the per-channel ratios. How much channels
      spread      DISAGREE with each other -- a property of the implant, not of sample size,
                  so it does not shrink with more data.
      n           channels the gate leaves behind.

    A gate that tightens channel spread while collapsing n has not made the measurement more
    consistent; it has made it narrower. Both are drawn on the same axes for that reason.
    """
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, MUTED, recessive
    from sdc.common.paths import figdir

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    dets = ("Janca", "Barkmeier", "Delphos")
    res = {}

    print(f"\n=== MINIMUM RATE GATE, source = {source.upper()} (block-paired) ===")
    for rec, pre in recs:
        z = np.load(RUNS / f"{rec}.npz", allow_pickle=False)
        gr = baseline_rates(pre, [str(s) for s in z["names"]]) if source == "baseline" else None
        print(f"\n{rec}   (gate from {pre if source == 'baseline' else 'its own OFF blocks'})")
        print(f"    {'min rate':>9}{'n chan':>8}   " +
              "".join(f"{d:>24}" for d in dets))
        rows = []
        for mr in grid:
            bp = block_table(z, min_rate=mr, gate_rates=gr)
            if bp.n_chan < 8:
                print(f"    {mr:>9.1f}{bp.n_chan:>8}   (too few channels left to estimate)")
                continue
            cells, vals = [], {}
            for d in dets:
                pt, lo, hi = ci(bp.det[d], bp.n_block, bp.n_chan, 1500)
                v = per_channel_log_ratio(bp.det[d])
                iqr = float(np.diff(np.percentile(v, [25, 75]))[0]) if v.size > 3 else np.nan
                vals[d] = (10 ** pt, 10 ** lo, 10 ** hi, 10 ** iqr)
                cells.append(f"{10 ** pt:>5.2f}[{10 ** lo:.2f},{10 ** hi:.2f}] s{10 ** iqr:.1f}")
            rows.append((mr, bp.n_chan, vals))
            print(f"    {mr:>9.1f}{bp.n_chan:>8}   " + "".join(f"{c:>24}" for c in cells))
        res[rec] = rows


    fig, axes = plt.subplots(2, len(recs), figsize=(6.8 * len(recs), 8.6), squeeze=False,
                             sharex=True)
    for j, (rec, _pre) in enumerate(recs):
        rows = res[rec]
        x = np.array([r[0] for r in rows])
        ax = axes[0][j]
        for d in dets:
            pt = np.array([r[2][d][0] for r in rows])
            lo = np.array([r[2][d][1] for r in rows])
            hi = np.array([r[2][d][2] for r in rows])
            ax.plot(x, pt, "-o", ms=4, lw=1.5, color=colors[d], label=d)
            ax.fill_between(x, lo, hi, color=colors[d], alpha=.13, lw=0)
        ax.axhline(1.0, color="0.3", ls="--", lw=1.2)
        ax.set_yscale("log")
        ax.set_ylabel("ON/OFF ratio (log)\nshaded = 95% CI on the MEDIAN")
        ax.set_title(f"({'ab'[j]}) {rec}", fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8)
        recessive(ax); ax.grid(alpha=.3)

        # Channel spread and surviving channels share one axis pair: a gate that tightens the
        # spread while collapsing n has narrowed the measurement rather than improved it, and
        # separating the two panels would let either be read without the other.
        ax = axes[1][j]
        for d in dets:
            ax.plot(x, [r[2][d][3] for r in rows], "-o", ms=4, lw=1.5, color=colors[d],
                    label=f"{d} channel IQR")
        ax.axhline(1.0, color="0.55", ls=":", lw=1.2)
        ax.set_yscale("log")
        ax.set_ylabel("spread ACROSS CHANNELS\n(IQR of per-channel ratio, fold)")
        ax.set_xlabel(f"minimum {'baseline' if source == 'baseline' else 'OFF'} rate required "
                      f"of a channel (det/min)")
        recessive(ax); ax.grid(alpha=.3)
        ax2 = ax.twinx()
        ax2.plot(x, [r[1] for r in rows], "-s", ms=5, lw=1.6, color="#c2691f",
                 label="channels left")
        ax2.set_ylabel("channels surviving the gate", color="#c2691f")
        ax2.tick_params(axis="y", colors="#c2691f")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper center", ncol=2)

    src = ("each patient's own stim-free BASELINE recording" if source == "baseline"
           else "the stim file's own OFF blocks")
    fig.suptitle(f"Minimum-rate gate, applied from {src}.\n"
                 "Top: the effect and its uncertainty.  Bottom: how much CHANNELS disagree "
                 "with each other (colour) against the channels the gate costs (orange).",
                 fontsize=10)
    fig.tight_layout()
    out = figdir("real") / f"min_rate_gate_{source}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")
    return res


# ----------------------------------------------------------------------
# Estimator + cluster bootstrap
# ----------------------------------------------------------------------
def per_channel_log_ratio(a, blocks=None, chans=None):
    """One log10(ON rate / OFF rate) per surviving channel, counts pooled over `blocks`.

    The vector behind the estimate. Its MEDIAN is the effect; its SPREAD is how much channels
    disagree with each other, which is a property of the implant rather than of how much data
    was collected and so does not shrink as blocks or channels are added. Those two are easy to
    conflate on a figure, which is why they are returned by different functions here.
    """
    b = slice(None) if blocks is None else blocks
    oc, fc = a["on_count"][b], a["off_count"][b]
    os_, fs_ = a["on_sec"][b], a["off_sec"][b]
    if chans is not None:
        oc, fc, os_, fs_ = oc[:, chans], fc[:, chans], os_[:, chans], fs_[:, chans]

    on_t, off_t = os_.sum(axis=0), fs_.sum(axis=0)
    on_c, off_c = oc.sum(axis=0), fc.sum(axis=0)
    ok = (on_t > 0) & (off_t > 0) & (off_c > 0)
    if not ok.any():
        return np.zeros(0)
    # Half-detection continuity correction on the ON side, over that channel's own observed
    # time -- keeps a channel that stimulation silenced completely, which is evidence and not
    # a nuisance value to be deleted.
    num = np.where(on_c[ok] > 0, on_c[ok], 0.5) / on_t[ok]
    den = off_c[ok] / off_t[ok]
    return np.log10(num / den)


def log_ratio(a, blocks=None, chans=None, agg=AGG):
    """One log10(ON/OFF) per block set, combining channels by `agg` (see AGG).

    nan if a resample has no usable channel -- the caller drops those draws rather than
    substituting a value for a resample that carries no information."""
    v = per_channel_log_ratio(a, blocks, chans)
    if not v.size:
        return np.nan
    return float(np.mean(v) if agg == "mean" else np.median(v))


def pair_changes(a, min_chan=5, agg=AGG):
    """ONE change per matched ON/OFF pair: the median per-channel log10 ratio within that pair.

    A different estimator from `log_ratio`, not a rearrangement of it:

      log_ratio      pool counts over all blocks -> one ratio per channel -> combine. Every
                     block contributes in proportion to how many detections it holds, so a
                     long or busy block carries more of the answer than a short quiet one.
      pair_changes   each pair is reduced to one number FIRST, then those numbers are combined.
                     Every pair counts once regardless of size, and -- the reason to want it --
                     the block-to-block spread becomes visible instead of being averaged away
                     inside the estimate.

    Channels are still paired inside each pair, so this is a within-block, within-channel
    contrast: the tightest comparison available in this data.

    THE COST is sparsity. A 64 s block at ~3 det/min gives a channel about 3 detections, and a
    channel with none in the OFF half of that pair has no ratio there and drops out for that
    pair only. `min_chan` refuses to report a pair that fewer than that many channels survived,
    because a median over 2 channels is not a measurement. Pairs that fail return nan and are
    excluded rather than being filled in.
    """
    n_block = a["on_count"].shape[0]
    out = np.full(n_block, np.nan)
    n_used = np.zeros(n_block, int)
    for b in range(n_block):
        v = per_channel_log_ratio(a, blocks=[b])
        n_used[b] = v.size
        if v.size >= min_chan:
            out[b] = float(np.mean(v) if agg == "mean" else np.median(v))
    return out, n_used


def ci(a, n_block, n_chan, n_boot=N_BOOT, seed=SEED, q=(2.5, 97.5)):
    """Cluster bootstrap: resample BLOCKS and channels together.

    Blocks are resampled because they are what varies -- a block is a few minutes of a night,
    and the next block is a different few minutes. Channels are resampled as well so the
    interval covers both sources rather than pretending the implant is fixed. Block variance
    dominates, which is the entire finding.
    """
    rng = np.random.default_rng(seed)
    pt = log_ratio(a)
    bb = rng.integers(0, n_block, size=(n_boot, n_block))
    cc = rng.integers(0, n_chan, size=(n_boot, n_chan))
    v = np.array([log_ratio(a, bb[i], cc[i]) for i in range(n_boot)])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return pt, np.nan, np.nan
    return pt, float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))


def permute(a, n_block, n_chan, max_perm=4096):
    """Exact within-pair sign-flip test: p-value, and the permutation null.

    With 6 block pairs on P1 and 11 on P5, a cluster bootstrap is resampling far too few
    clusters to be trusted -- the calibration check bears that out, sitting at 19-37% on
    splits that yield only 4 pairs. A permutation test has no such problem: it makes no
    distributional assumption and its validity does not depend on the number of clusters,
    only its RESOLUTION does. Six pairs give 2^6 = 64 sign assignments, so the smallest
    reachable two-sided p is about 1/32 -- coarse, but exact rather than optimistic.

    The null being tested is that ON and OFF are exchangeable WITHIN a pair. Adjacency has
    already removed slow drift, which is what makes that null plausible here and would not
    have made it plausible for a whole-condition contrast.
    """
    swaps = np.arange(2 ** n_block) if 2 ** n_block <= max_perm else \
        np.random.default_rng(SEED).integers(0, 2 ** n_block, size=max_perm)
    obs = log_ratio(a)
    null = np.empty(len(swaps))
    for i, s in enumerate(swaps):
        flip = ((s >> np.arange(n_block)) & 1).astype(bool)
        b = dict(a)
        if flip.any():
            oc, fc = a["on_count"].copy(), a["off_count"].copy()
            os_, fs_ = a["on_sec"].copy(), a["off_sec"].copy()
            oc[flip], fc[flip] = a["off_count"][flip], a["on_count"][flip]
            os_[flip], fs_[flip] = a["off_sec"][flip], a["on_sec"][flip]
            b = {"on_count": oc, "off_count": fc, "on_sec": os_, "off_sec": fs_}
        null[i] = log_ratio(b)
    null = null[np.isfinite(null)]
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean()) if null.size else np.nan
    return obs, p, null, len(swaps) == 2 ** n_block


def report(rec, n_boot=N_BOOT):
    z = np.load(RUNS / f"{rec}.npz", allow_pickle=False)
    bp = block_table(z)
    span = float(np.diff(bp.on_w, axis=1).sum())
    print(f"\n=== {rec}: {bp.n_block} matched block pairs, {span:.0f}s per side, "
          f"{bp.n_chan} channels")
    print(f"    block lengths: "
          f"{', '.join(f'{d:.0f}s' for d in np.diff(bp.on_w, axis=1).ravel())}")
    print(f"\n    {'detector':<11}{'ratio':>8}{'95% CI (blocks+channels)':>28}"
          f"{'perm p':>9}")
    out = {}
    for d in bp.det:
        pt, lo, hi = ci(bp.det[d], bp.n_block, bp.n_chan, n_boot)
        _o, p, _null, exact = permute(bp.det[d], bp.n_block, bp.n_chan)
        out[d] = (10 ** pt, 10 ** lo, 10 ** hi, p)
        print(f"    {d:<11}{10 ** pt:>8.3f}"
              f"{f'[{10 ** lo:.3f}, {10 ** hi:.3f}]':>28}{p:>9.3f}"
              f"{'  SIG' if p < 0.05 else '  ns'}")
    print(f"    permutation: {'exact, all' if exact else 'sampled'} "
          f"{2 ** bp.n_block if exact else 4096} sign assignments "
          f"(smallest reachable p ~ {2 / 2 ** bp.n_block:.3f})")
    return out


def segment_course(z, keep, min_seg_sec=20.0):
    """Mean per-channel rate in every ON and OFF segment, in time order.

    The same view as stim_effect.py panel (c) -- "does every ON block dip, or is the recording
    just drifting?" -- but on the block-paired channel set, so it sits directly above the
    per-pair changes and describes the same channels.

    One improvement on the original: analysable seconds are taken PER SEGMENT from
    `clean_per_sec`, rather than wall-clock seconds. stim_effect.py had to use wall clock
    because the npz stores clean time per CONDITION only, and printed a caveat saying so; the
    per-second array makes the caveat unnecessary.

    Returns (mid_times_sec, is_on, {detector: normalised rate per segment}).
    """
    fs = float(z["fs"])
    cps = z["clean_per_sec"]
    ON, OFF = cond.select(z, "on"), cond.select(z, "off")
    segs = sorted([(a, b, True) for a, b in ON.runs] + [(a, b, False) for a, b in OFF.runs])
    # A rate from a few-second sliver is sampling noise, not a measurement; P1's window opens
    # with one before stimulation starts.
    segs = [s for s in segs if s[1] - s[0] >= min_seg_sec]

    n = len(z["names"])
    out = {}
    for d in [str(s) for s in z["detectors"]]:
        t, c = z[f"{d}_idx"] / fs, z[f"{d}_chan"]
        vals = []
        for a, b, _ in segs:
            m = (t >= a) & (t < b)
            cnt = np.bincount(c[m], minlength=n)[keep]
            sec = cps[int(a):int(b)].sum(axis=0)[keep] / fs
            ok = sec > 0
            vals.append(np.mean(cnt[ok] / sec[ok]) * 60.0 if ok.any() else np.nan)
        v = np.array(vals, float)
        out[d] = v / np.nanmean(v)
    return (np.array([0.5 * (a + b) for a, b, _ in segs]),
            np.array([s[2] for s in segs], bool), out, segs)


def report_pairs(recs=("P1_stim", "P5_stim"), min_chan=5, tag="", outdir=None,
                 prefer="before", tmax=None, chans=None, absolute=True,
                 off_full=True):
    """Per-pair changes: the table, a sign test, and a figure of every pair.

    The sign test asks only whether pairs fall on one side of no-change more often than a coin
    would. It throws away the size of each change, which makes it weak -- but it assumes nothing
    at all about the distribution, and with 6 pairs there is not enough data to justify assuming
    anything. It is also the natural test for this design, because the design has already
    reduced the experiment to a handful of paired observations.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import binomtest
    from seeg._style import RED, BLUE, recessive
    from sdc.common.paths import figdir

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    dets = ("Janca", "Barkmeier", "Delphos")
    store = {}

    fig, axes = plt.subplots(2, len(recs), figsize=(7.4 * len(recs), 8.6), squeeze=False,
                             sharex="col")
    for j, rec in enumerate(recs):
        z = np.load(RUNS / f"{rec}{tag}.npz", allow_pickle=False)
        bp = block_table(z, prefer=prefer, tmax=tmax, chans=chans, off_full=off_full)

        # ---- top: the segment time course, for context -----------------------------------
        # A stimulation effect dips in EVERY shaded block and recovers between them; a drifting
        # recording slides through them without noticing. The panel below turns each of those
        # dips into one number, so this is the raw material for it.
        axt = axes[0][j]
        # ABSOLUTE rate by default. segment_course divides each detector by its OWN mean over
        # all segments, which makes the top panel unreadable against the bottom one: a block can
        # sit ABOVE that mean (so it looks elevated) while its ratio to the ADJACENT OFF window
        # is below 1, or vice versa. P1's third pair does exactly that -- it is the busiest ON
        # block AND has the quietest OFF window, so it reads as a rise in one panel and a fall
        # in the other. On an absolute axis the two panels describe the same quantity.
        if absolute:
            from sdc.artefact.exposure import _segment_rates
            mids, is_on, course = _segment_rates(z, bp.keep)
            # shading from the recording's own ON runs; _segment_rates does not return spans
            segs = [(a, b, True) for a, b in cond.select(z, "on").runs]
        else:
            mids, is_on, course, segs = segment_course(z, bp.keep)
        for a, b, on in segs:
            if on:
                axt.axvspan(a / 60.0, b / 60.0, color="#f0c419", alpha=.22, lw=0, zorder=0)
        for d in dets:
            axt.plot(mids / 60.0, course[d], "-o", ms=5, lw=1.5, color=colors[d], label=d)
        if not absolute:
            axt.axhline(1.0, color="0.4", ls="--", lw=1.0)
        axt.set_ylabel("detections / min / channel" if absolute
                       else "segment rate / that detector's mean")
        axt.set_title(f"({'ac'[j]}) {rec} -- every segment in time order, shaded = stim ON",
                      fontsize=9, loc="left")
        axt.legend(frameon=False, fontsize=8, ncol=3)
        recessive(axt)
        axt.grid(alpha=.3)

        ax = axes[1][j]
        print(f"\n=== {rec}{tag}: change per matched ON/OFF pair ({bp.n_block} pairs, "
              f"{bp.n_chan} channels)")
        print(f"    {'detector':<11}{'pairs used':>11}{'median':>9}{'mean':>8}"
              f"{'pair range':>18}{'below 1':>9}{'sign p':>9}   pooled")
        for i, d in enumerate(dets):
            v, n_used = pair_changes(bp.det[d], min_chan)
            ok = np.isfinite(v)
            store[(rec, d)] = (v, n_used)
            if ok.sum() < 2:
                print(f"    {d:<11}{ok.sum():>11}   too few usable pairs")
                continue
            vv = v[ok]
            n_below = int((vv < 0).sum())
            p = binomtest(n_below, len(vv), 0.5).pvalue
            pooled = 10 ** log_ratio(bp.det[d])
            print(f"    {d:<11}{ok.sum():>11}{10 ** np.median(vv):>9.3f}"
                  f"{10 ** np.mean(vv):>8.3f}"
                  f"{f'{10 ** vv.min():.2f}-{10 ** vv.max():.2f}':>18}"
                  f"{f'{n_below}/{len(vv)}':>9}{p:>9.3f}   {pooled:.3f}")

            # x = each pair's position in the recording, so a trend across the night is
            # visible rather than being hidden behind a block index.
            xs = bp.on_w[ok, 0] / 60.0
            ax.plot(xs, 10 ** vv, "o", ms=7, color=colors[d], alpha=.8, label=d)
            ax.axhline(10 ** np.median(vv), color=colors[d], lw=1.4, ls="-", alpha=.55)
        ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
        ax.set_yscale("log")
        ax.set_xlabel("time of the ON block (min into the recording)")
        ax.set_ylabel("ON/OFF ratio for that pair (log)")
        ax.set_title(f"({'bd'[j]}) {rec} -- one point per matched pair, "
                     f"line = median across pairs", fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=8)
        recessive(ax)
        ax.grid(alpha=.3)

    fig.suptitle("Top: every segment in time order (stim_effect panel c, on the block-paired "
                 "channel set and using analysable time per segment).\n"
                 "Bottom: each shaded block above reduced to ONE number against the OFF time "
                 "beside it. The scatter is the block-to-block variability a pooled estimate "
                 f"hides.{f'   [{tag.lstrip(chr(95)).upper()} operating points]' if tag else ''}",
                 fontsize=10)
    fig.tight_layout()
    # `outdir` puts one recording's figure in its own folder, beside the stim_effect
    # figure for the same run. Without it every per-recording call writes the same
    # filename and the last one silently wins.
    if outdir is not None:
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / "pair_changes.png"
    else:
        out = figdir("real") / f"pair_changes{tag}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")
    return store


def pair_scales(a, on_w, min_chan=5):
    """Each matched pair's change expressed three ways. Returns {scale: per-pair array}.

      diff        median over channels of (ON rate - OFF rate), det/min
      ratio       median over channels of (ON rate / OFF rate)
      geo         10 ** mean over channels of log10(ON/OFF) -- the geometric mean

    `ratio` and a median of LOG ratios are the same number, because the median commutes with
    any monotone transform; only the axis differs. `geo` is the one that genuinely changes the
    answer, because a mean does not commute -- and it is the more powerful of the two here
    (effect/SD 1.47 against 1.15 for Janca on P1), at the cost of following outliers when the
    per-channel distribution is skewed.
    """
    n_block = a["on_count"].shape[0]
    out = {k: np.full(n_block, np.nan) for k in ("diff", "ratio", "geo")}
    for b in range(n_block):
        on_t, off_t = a["on_sec"][b], a["off_sec"][b]
        on_c, off_c = a["on_count"][b], a["off_count"][b]
        ok = (on_t > 0) & (off_t > 0) & (off_c > 0)
        if ok.sum() < min_chan:
            continue
        on_r, off_r = on_c[ok] / on_t[ok] * 60.0, off_c[ok] / off_t[ok] * 60.0
        # Half-detection floor on the ON side over that block's own duration, so a channel
        # silenced during the block stays in rather than making the ratio undefined.
        span = max(on_w[b][1] - on_w[b][0], 1e-9)
        on_r = np.where(on_c[ok] > 0, on_r, 0.5 / span * 60.0)
        out["diff"][b] = float(np.median(on_r - off_r))
        out["ratio"][b] = float(np.median(on_r / off_r))
        out["geo"][b] = float(10 ** np.mean(np.log10(on_r / off_r)))
    return out


def binned_course(z, keep, bin_sec=60.0):
    """Mean per-channel rate in fixed bins across the whole recording.

    The MEAN across channels, not the median. At one-minute bins a channel's count is a small
    integer, so a median across channels snaps to whole numbers and a time course drawn from it
    steps rather than moves -- the same quantisation that makes P5's per-pair medians land on
    exactly 1.000. The mean has no such problem and, being a mean of equal-length bins, is
    exactly the pooled rate over those bins.

    Returns (bin_centres_sec, {detector: rate}, on_fraction_per_bin).
    """
    fs = float(z["fs"])
    cps = z["clean_per_sec"]
    edges = np.arange(0, cps.shape[0] + 1, int(bin_sec))
    if edges[-1] < cps.shape[0]:
        edges = np.append(edges, cps.shape[0])
    n = len(z["names"])

    out = {}
    for d in [str(s) for s in z["detectors"]]:
        t, c = z[f"{d}_idx"] / fs, z[f"{d}_chan"]
        mu, md = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (t >= a) & (t < b)
            cnt = np.bincount(c[m], minlength=n)[keep]
            sec = cps[a:b].sum(axis=0)[keep] / fs
            ok = sec > 0
            r = cnt[ok] / sec[ok] * 60.0 if ok.any() else np.array([np.nan])
            mu.append(np.mean(r))
            md.append(np.median(r))
        out[d] = {"mean": np.array(mu, float), "median": np.array(md, float)}

    on_frac = np.zeros(edges.size - 1)
    if "on_per_sec" in z.files:
        ops = np.asarray(z["on_per_sec"], bool)
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            seg = ops[a:min(b, ops.size)]
            on_frac[i] = seg.mean() if seg.size else 0.0
    return 0.5 * (edges[:-1] + edges[1:]), out, on_frac


def block_binned_course(z, keep, bin_sec=60.0):
    """Bins that RESPECT the block boundaries: each ON run and each OFF run is split into
    roughly `bin_sec` pieces, so no bin ever straddles the ON/OFF edge.

    This is the difference between this figure and a fixed-grid one, and it is not cosmetic.
    P1's ON blocks are 64 s against a 60 s grid, so on a fixed grid the block edge walks through
    the bins and every bin mixes ON with OFF -- diluting the very contrast being drawn. Splitting
    the runs instead gives exactly one point for a ~1 min ON block and three for a ~3 min OFF,
    each one pure.

    Returns (centres_sec, {detector: rate}, is_on) with one entry per bin.
    """
    fs = float(z["fs"])
    cps = np.asarray(z["clean_per_sec"])
    n = len(z["names"])
    on = np.asarray(z["on_per_sec"], bool) if "on_per_sec" in z.files else \
        np.zeros(cps.shape[0], bool)
    m = min(on.size, cps.shape[0])
    on = on[:m]

    edges, flags = [], []
    for a, b in _runs_from_mask(on):                      # ON runs
        k = max(1, int(round((b - a) / bin_sec)))
        cut = np.linspace(a, b, k + 1)
        edges += [(cut[i], cut[i + 1]) for i in range(k)]
        flags += [True] * k
    for a, b in _runs_from_mask(~on):                     # OFF runs
        k = max(1, int(round((b - a) / bin_sec)))
        cut = np.linspace(a, b, k + 1)
        edges += [(cut[i], cut[i + 1]) for i in range(k)]
        flags += [False] * k
    order = np.argsort([e[0] for e in edges])
    edges = [edges[i] for i in order]
    flags = np.array([flags[i] for i in order], bool)

    out = {}
    for d in [str(s) for s in z["detectors"]]:
        t, c = z[f"{d}_idx"] / fs, z[f"{d}_chan"]
        r, md = [], []
        for a, b in edges:
            sel = (t >= a) & (t < b)
            cnt = np.bincount(c[sel], minlength=n)[keep]
            sec = cps[int(np.floor(a)):int(np.ceil(b))].sum(axis=0)[keep] / fs
            ok = sec > 0
            rr = cnt[ok] / sec[ok] * 60.0 if ok.any() else np.array([np.nan])
            r.append(np.mean(rr))
            md.append(np.median(rr))
        out[d] = {"mean": np.array(r, float), "median": np.array(md, float)}
    return np.array([0.5 * (a + b) for a, b in edges]), out, flags


def relative_course(recs=None, bin_sec=60.0, tag="_qcfinal", baseline="auto", min_chan=5):
    """Rate in ~1 min bins as a MULTIPLE of the baseline median rate, one panel per trial.

    The binning is the recording's own structure at 1 min resolution: a ~1 min ON block becomes
    one point and the ~3 min OFF between blocks becomes three, so the shape of the recovery
    between blocks is visible instead of being collapsed into a single OFF number.

    WHAT "BASELINE" MEANS, AND WHY IT DIFFERS BY ARM
      pre  the stim-free recording, median across ITS 1 min bins. The honest baseline: a separate
           recording, so normalising by it cannot condition on the data being analysed.
      off  the median across the stim file's own OFF bins. Weaker -- OFF sits between stimulation
           blocks and may carry carryover -- but it is the ONLY option for a trial whose pre file
           has not been through the detectors.

    LF-continuous trials have no OFF period at all, so they can only ever use `pre`.
    HF-intermittent trials currently have no pre RUN, so in practice they use `off`. Which one
    was used is printed and drawn on every panel, because the two are not interchangeable and a
    figure mixing them silently would be misread.

    Normaliser and series are the same statistic -- median across bins of the mean-over-channels
    rate -- so the ratio is of like with like.
    """
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, recessive
    from sdc.common.paths import figdir
    from sdc.detect.cohort import COHORT, arm

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}

    if recs is None:                       # every trial whose stim run is on disk
        recs = [s for s in COHORT if (RUNS / f"{s}_stim{tag}.npz").is_file()]

    panels = []
    for stem in recs:
        z = np.load(RUNS / f"{stem}_stim{tag}.npz", allow_pickle=False)
        a = arm(stem)
        # Channels: the block-paired set when there are blocks, otherwise anything measurable.
        try:
            keep = block_table(z).keep if a == "HF-int" else None
        except Exception:
            keep = None
        # COUNT the selection, do not measure it. bp.keep is a BOOLEAN MASK, so np.size is the
        # number of channels in the recording (226 on P1) rather than the number selected (116)
        # -- which both defeats the min_chan guard and reports a channel count that is wrong in
        # the direction that flatters the figure.
        n_sel = (int(np.count_nonzero(keep)) if keep is not None
                 and np.asarray(keep).dtype == bool else (0 if keep is None else np.size(keep)))
        if keep is None or n_sel < min_chan:
            keep = np.flatnonzero(np.asarray(z["clean_per_sec"]).sum(axis=0) > 0)
            n_sel = keep.size

        # Block-aligned bins: a bin never straddles an ON/OFF edge. on_frac is then 1 or 0.
        mids, course, _ison = block_binned_course(z, keep, bin_sec)
        on_frac = _ison.astype(float)

        use, norm = None, {}
        # LOCAL is the default wherever there are OFF bins, and it is the reason pair_changes
        # shows an effect this figure was hiding. A single global baseline leaves the recording's
        # own drift inside the trace: measured on the OFF bins ALONE, the reference moves 2.2x
        # over P1, 3.2x over P1_Pulv145 and 1.5x over P11_Pulv145 -- comparable to or larger than
        # the ~0.5-0.7 effect being looked for, so it buries it. Dividing each bin by the OFF
        # level interpolated between neighbouring OFF bins cancels drift the same way pairing
        # each ON block with its ADJACENT OFF does, but keeps 1 min resolution.
        # Cost, stated plainly: OFF bins now sit at ~1 by construction, so this figure shows the
        # ON deflection and says nothing about how OFF itself behaves.
        if baseline in ("local", "auto") and on_frac.size:
            off_b = on_frac < 0.5
            if off_b.sum() >= 2:
                local = {}
                for d in course:
                    y = course[d]["mean"]
                    ref = np.interp(mids, mids[off_b], y[off_b])       # drift, from OFF only
                    with np.errstate(invalid="ignore", divide="ignore"):
                        local[d] = np.where(ref > 0, y / ref, np.nan)
                panels.append((stem, a, mids, {d: {"rel": local[d]} for d in local},
                               on_frac, {d: 1.0 for d in local}, "local", n_sel))
                continue

        pre_p = RUNS / f"{stem}_pre{tag}.npz"
        want_pre = baseline in ("pre", "auto")
        if want_pre and pre_p.is_file():
            zp = np.load(pre_p, allow_pickle=False)
            if [str(s) for s in zp["names"]] == [str(s) for s in z["names"]]:
                pk = np.flatnonzero(np.asarray(zp["clean_per_sec"]).sum(axis=0) > 0)
                _m, pc, _f = binned_course(zp, pk, bin_sec)
                norm = {d: float(np.nanmedian(pc[d]["mean"])) for d in pc}
                use = "pre"
        if use is None and baseline in ("off", "auto") and on_frac.size:
            off = on_frac < 0.5
            if off.any():
                norm = {d: float(np.nanmedian(course[d]["mean"][off])) for d in course}
                use = "off"
        if use is None:
            print(f"  [skip] {stem}: no usable baseline "
                  f"({'no OFF bins' if a == 'LF-cont' else 'no OFF'}, and no pre run on disk)")
            continue
        panels.append((stem, a, mids, course, on_frac, norm, use, n_sel))

    if not panels:
        raise SystemExit("no trial had a usable baseline; nothing to draw.")

    ncol = 2
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.0 * ncol, 3.4 * nrow),
                             squeeze=False, sharey=True)
    print(f"\n{'trial':<15}{'arm':<9}{'base':<6}{'chan':>5}   baseline median rate (det/min/ch)")
    for k, (stem, a, mids, course, on_frac, norm, use, nch) in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        for i, f in enumerate(on_frac):
            if f > 0.5:
                ax.axvspan((mids[i] - bin_sec / 2) / 60.0, (mids[i] + bin_sec / 2) / 60.0,
                           color="#f0c419", alpha=.22, lw=0, zorder=0)
        for d in ("Janca", "Barkmeier", "Delphos"):
            if d not in course:
                continue
            if "rel" in course[d]:                      # already divided by the local OFF level
                y = course[d]["rel"]
            else:
                if not np.isfinite(norm.get(d, np.nan)) or norm[d] <= 0:
                    continue
                y = course[d]["mean"] / norm[d]
            ax.plot(mids / 60.0, y, "-o", ms=3.2, lw=1.4, color=colors[d], label=d)
        ax.axhline(1.0, color="0.25", ls="--", lw=1.2)
        _what = {"local": "  (local OFF level, drift removed)",
                 "pre": "  (pre file)", "off": "  (this file OFF median)"}[use]
        ax.set_title(f"{stem}  [{a}]  baseline = {use}{_what},  {nch} ch",
                     fontsize=9, loc="left")
        ax.set_xlabel("time (min)")
        ax.set_ylabel(f"rate / baseline median\n({bin_sec:g}s bins)", fontsize=8)
        if k == 0:
            ax.legend(frameon=False, fontsize=8, ncol=3)
        recessive(ax)
        ax.grid(alpha=.3)
        if use == "local":
            _on = on_frac > 0.5
            print(f"{stem:<15}{a:<9}{use:<6}{nch:>5}   " + "  ".join(
                f"{d[:4]} {np.nanmedian(course[d]['rel'][_on]):.2f}"
                for d in sorted(course) if _on.any()) + "   (median ON bin, vs local OFF)")
        else:
            print(f"{stem:<15}{a:<9}{use:<6}{nch:>5}   " +
                  "  ".join(f"{d[:4]} {norm.get(d, float('nan')):.2f}" for d in sorted(norm)))
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle(
        f"Spike rate in {bin_sec:g}s bins, as a multiple of the baseline median rate.\n"
        "Shaded = stim ON. Dashed line = baseline. LINEAR y: a halving and a doubling are 0.5 and\n"
        "1.0 from the line, so decreases are visually compressed against increases.\n"
        "PANELS ARE NOT ALL NORMALISED THE SAME WAY -- 'pre' is a separate stim-free recording, "
        "'off' is this file's own OFF bins and may carry carryover; each panel says which.",
        fontsize=10)
    fig.tight_layout()
    out = figdir("real") / f"stim_effect_relative{tag}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")
    return panels


def scale_comparison(recs=("P1_stim", "P5_stim"), min_chan=5, bin_sec=60.0):
    """The same pairs on three scales, in the pair_changes time-course style.

        .venv\\Scripts\\python.exe -m sdc.artefact.blocks --scales
    """
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, recessive
    from sdc.common.paths import figdir

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    dets = ("Janca", "Barkmeier", "Delphos")
    rows = [("diff", "ON - OFF rate (det/min)", 0.0, "linear"),
            ("ratio", "ON / OFF  (median over channels)", 1.0, "log"),
            ("geo", "ON / OFF  (geometric mean over channels)", 1.0, "log")]

    fig, axes = plt.subplots(len(rows) + 1, len(recs),
                             figsize=(7.2 * len(recs), 3.1 * (len(rows) + 1)),
                             squeeze=False, sharex="col")
    for j, rec in enumerate(recs):
        z = np.load(RUNS / f"{rec}.npz", allow_pickle=False)
        bp = block_table(z)

        # ---- row 0: the raw rate the contrasts below are built from ----------------------
        axr = axes[0][j]
        mids, course, on_frac = binned_course(z, bp.keep, bin_sec)
        for i, f in enumerate(on_frac):
            if f > 0.5:
                axr.axvspan((mids[i] - bin_sec / 2) / 60.0, (mids[i] + bin_sec / 2) / 60.0,
                            color="#f0c419", alpha=.22, lw=0, zorder=0)
        for d in dets:
            axr.plot(mids / 60.0, course[d]["mean"], "-", lw=1.5, color=colors[d],
                     label=f"{d} mean")
            # The median is drawn faint and stepped on purpose: at 60 s bins a channel's count
            # is a small integer, so the median across channels can only land on whole numbers
            # and the trace moves in steps. That is the estimator quantising, not the brain.
            axr.plot(mids / 60.0, course[d]["median"], drawstyle="steps-mid", ls="--", lw=1.1,
                     color=colors[d], alpha=.45, label=f"{d} median")
        recessive(axr)
        axr.grid(alpha=.3)
        axr.set_ylabel(f"rate in {bin_sec:g}s bins (det/min/channel)\n"
                       f"solid = mean over channels, faint = median", fontsize=8)
        axr.set_title(f"({'ab'[j]}) {rec} -- {bp.n_block} pairs, {bp.n_chan} channels. "
                      f"Shaded = stim ON", fontsize=10, loc="left")
        axr.legend(frameon=False, fontsize=7, ncol=3)

        print(f"\n=== {rec}: {bp.n_block} pairs, {bp.n_chan} channels")
        print(f"    {'detector':<11}{'diff (det/min)':>16}{'ratio':>9}{'geo mean':>10}")
        for d in dets:
            sc = pair_scales(bp.det[d], bp.on_w, min_chan)
            print(f"    {d:<11}"
                  + "".join(f"{np.nanmedian(sc[k]):>{w}.3f}"
                            for k, w in (("diff", 16), ("ratio", 9), ("geo", 10))))
            for i, (key, _lab, null, _sc) in enumerate(rows):
                ax = axes[i + 1][j]
                v = sc[key]
                ok = np.isfinite(v)
                ax.plot(bp.on_w[ok, 0] / 60.0, v[ok], "-o", ms=6, lw=1.4,
                        color=colors[d], alpha=.85, label=d)
                ax.axhline(np.nanmedian(v), color=colors[d], lw=1.2, alpha=.4)
        for i, (key, lab, null, scale) in enumerate(rows):
            ax = axes[i + 1][j]
            ax.axhline(null, color="0.25", ls="--", lw=1.3)
            ax.set_yscale(scale)
            ax.set_ylabel(lab, fontsize=8.5)
            recessive(ax)
            ax.grid(alpha=.3)
            if i == len(rows) - 1:
                ax.set_xlabel("time (min into the recording)")

    fig.suptitle("The same matched pairs on three scales. Faint line = that detector's median "
                 "across pairs, dashed = no change.\nMiddle and a median of LOG ratios are the "
                 "SAME number (the median commutes); only the geometric mean differs.",
                 fontsize=10)
    fig.tight_layout()
    out = figdir("real") / "pair_scales.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")


def compare_estimators(recs=("P1_stim", "P5_stim"), tags=("", "_tuned"), min_chan=5,
                       panel_tag=None, outdir=None, prefer="before", tmax=None,
                       scale="log", chans=None, off_full=True):
    """Pooled-over-blocks against per-pair-then-combined, and WHY they differ.

    The two estimators use the same detections, the same channels and the same matched windows.
    They differ only in when the blocks are combined:

      pooled     counts are summed across blocks first, so a block contributes in proportion to
                 how many detections it holds. A long or busy block can carry the answer.
      per-pair   each pair is reduced to one ratio first, so every pair gets one vote whatever
                 its size.

    Neither is right in general. Pooling is more efficient if the blocks are really measuring
    the same thing; per-pair is more robust if they are not, and is the only one of the two
    whose spread means anything, because it has a value per block to be spread over. The gap
    between them is therefore a diagnostic in its own right: if it is large, the estimate is
    resting on a few high-count blocks, which is worth knowing before quoting either number.
    """
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, recessive
    from sdc.common.paths import figdir

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    dets = ("Janca", "Barkmeier", "Delphos")
    summary = []
    # `scale` applies to every ratio axis here. Linear is easier to read off but it is NOT
    # symmetric for ratios -- a halving sits 0.5 below 1 and a doubling 1.0 above -- so a
    # decrease looks smaller than the equivalent increase. Said on the figure rather than left
    # for the reader to notice.

    # Which run the mechanism panels (a, b) draw. It used to be hardwired to tags[0], so calling
    # this with a new profile produced a figure whose box-and-points panels still showed the OLD
    # run -- identical pixels to the previous figure under a filename claiming otherwise. Only
    # panel (c) ever varied. Default to the LAST tag, which is the one a caller is introducing.
    panel_tag = tags[-1] if panel_tag is None else panel_tag
    if panel_tag not in tags:
        raise SystemExit(f"panel_tag {panel_tag!r} is not among tags {tags!r}")

    fig, axes = plt.subplots(1, len(recs) + 1, figsize=(6.2 * (len(recs) + 1), 5.4))
    rng = np.random.default_rng(SEED)
    for j, rec in enumerate(recs):
        ax = axes[j]
        for tag in tags:
            z = np.load(RUNS / f"{rec}{tag}.npz", allow_pickle=False)
            bp = block_table(z, prefer=prefer, tmax=tmax, chans=chans, off_full=off_full)
            for i, d in enumerate(dets):
                v, _ = pair_changes(bp.det[d], min_chan)
                ok = np.isfinite(v)
                pooled = log_ratio(bp.det[d])
                per_pair = float(np.median(v[ok]))
                summary.append((rec, tag, d, 10 ** pooled, 10 ** per_pair))
                if tag != panel_tag:
                    continue                     # the mechanism panel shows one run at a time

                r = 10 ** v[ok]
                bx = ax.boxplot([r], positions=[i], widths=0.5, showfliers=False,
                                patch_artist=True, zorder=2)
                bx["boxes"][0].set(facecolor=colors[d], alpha=.20, edgecolor=colors[d])
                for part in ("whiskers", "caps", "medians"):
                    for ln in bx[part]:
                        ln.set(color=colors[d], lw=1.6)

                # Marker area tracks the block's detection count, which is exactly the weight
                # pooling gives it and the box does not. A pooled marker sitting away from the
                # box median should have the big points on that same side -- that is the whole
                # explanation for any gap between the two estimators.
                w = (bp.det[d]["on_count"] + bp.det[d]["off_count"]).sum(axis=1)[ok]
                ax.scatter(i + rng.uniform(-.16, .16, r.size), r,
                           s=25 + 200 * w / max(w.max(), 1), color=colors[d], alpha=.45,
                           edgecolor="none", zorder=3)
                ax.scatter([i], [10 ** pooled], marker="D", s=95, facecolor="white",
                           edgecolor=colors[d], lw=2.2, zorder=4)
        ax.axhline(1.0, color="0.25", ls="--", lw=1.2)      # no-change
        ax.set_yscale(scale)
        ax.set_xticks(range(len(dets)))
        ax.set_xticklabels(dets, fontsize=9)
        ax.set_ylabel(f"ON/OFF ratio ({scale})")
        ax.set_title(f"({'ab'[j]}) {rec}{panel_tag} [{panel_tag.lstrip('_') or 'prod'}]: "
                     f"box + points = the per-pair values,\n"
                     f"white diamond = pooled.  Point size = that block's detection count",
                     fontsize=9, loc="left")
        recessive(ax); ax.grid(axis="y", alpha=.3)

    ax = axes[-1]
    for rec, tag, d, pooled, per_pair in summary:
        ax.scatter(pooled, per_pair, s=95, color=colors[d],
                   marker="o" if rec.startswith("P1") else "^",
                   facecolor=colors[d] if tag == "" else "none",
                   edgecolor=colors[d], lw=1.8, alpha=.9)
    lim = [0.4, 1.7]
    ax.plot(lim, lim, color="0.35", ls="--", lw=1.2)     # where the two estimators agree
    ax.axhline(1.0, color="0.75", ls=":", lw=1.0)
    ax.axvline(1.0, color="0.75", ls=":", lw=1.0)
    ax.set_xscale(scale); ax.set_yscale(scale)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("pooled across blocks")
    ax.set_ylabel("per-pair, then combined")
    _fill_lbl = tags[0].lstrip("_") or "prod"
    _open_lbl = ", ".join(t.lstrip("_") for t in tags[1:]) or "-"
    ax.set_title("(c) the two estimators against each other\n"
                 f"circle = P1, triangle = P5;  filled = {_fill_lbl}, open = {_open_lbl}",
                 fontsize=9, loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[d], label=d) for d in dets]
    ax.legend(handles=handles, frameon=False, fontsize=8)
    recessive(ax); ax.grid(alpha=.3)

    print(f"\n{'recording':<10}{'tag':<8}{'detector':<11}{'pooled':>9}{'per-pair':>10}"
          f"{'ratio':>8}")
    for rec, tag, d, pooled, per_pair in summary:
        print(f"{rec:<10}{tag or '(default)':<8}{d:<11}{pooled:>9.3f}{per_pair:>10.3f}"
              f"{per_pair / pooled:>8.2f}")

    _sc = ("LINEAR axes: a halving sits 0.5 below 1 and a doubling 1.0 above, so decreases look "
           "smaller than the equivalent increase." if scale == "linear" else
           "Log axes, so 0.5x and 2x are equally far from no-change.")
    fig.suptitle("Pooled-across-blocks vs per-pair-then-combined: same detections, same "
                 "channels, same windows -- only WHEN the blocks are combined differs.\n"
                 + _sc, fontsize=11)
    fig.tight_layout()
    # Name the figure after the run tags it compares. It used to be a fixed name, so plotting a
    # second profile silently overwrote the first and the two were indistinguishable on disk.
    _sfx = "".join(t for t in tags if t)
    # Per-recording folder when asked; otherwise the shared, tag-named figure.
    if outdir is not None:
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / "pooled_vs_pairwise.png"
    else:
        out = figdir("real") / f"pooled_vs_pairwise{_sfx}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")
    return summary


def null_check(pre_rec, stim_rec, n_split=200, n_boot=400, seed=SEED):
    """Does the block-paired interval actually cover 95% on stim-free data?

    Same construction as nullcheck.py -- impose the stim recording's block pattern at random
    phases on a baseline recording, where the truth is "no effect" -- but through the
    block-paired design and the cluster bootstrap. The number to look at is FPR: the
    whole-condition design gave 46-100% here against a nominal 5%.
    """
    from sdc.artefact.nullcheck import pseudo_masks

    z = np.load(RUNS / f"{pre_rec}.npz", allow_pickle=False)
    masks, duty = pseudo_masks(stim_rec, int(z["clean_per_sec"].shape[0]), n_split, seed)
    acc, nb = {}, []
    for k, m in enumerate(masks):
        try:
            bp = block_table(z, on_sec_mask=m)
        except SystemExit:
            continue
        if bp.n_block < 2 or bp.n_chan < 10:
            continue
        nb.append(bp.n_block)
        for d in bp.det:
            pt, lo, hi = ci(bp.det[d], bp.n_block, bp.n_chan, n_boot, seed=1000 + k)
            a = acc.setdefault(d, {"pts": [], "reject": []})
            a["pts"].append(pt)
            a["reject"].append(not (lo < 0 < hi))

    print(f"\n=== CALIBRATION: {pre_rec} (no stimulation), {len(nb)} pseudo-splits using "
          f"{stim_rec}'s pattern")
    print(f"    {int(np.median(nb))} block pairs per split (median), duty {duty:.0%}")
    print(f"\n    {'detector':<11}{'median':>9}{'2.5-97.5% of estimates':>26}"
          f"{'FPR':>7}   (nominal 5%)")
    for d, a in acc.items():
        pts = np.array(a["pts"], float)
        pts = pts[np.isfinite(pts)]
        lo, hi = np.percentile(pts, [2.5, 97.5])
        fpr = float(np.mean(a["reject"]))
        verdict = "  OK" if fpr <= 0.10 else ("  high" if fpr <= 0.20 else "  MISCALIBRATED")
        print(f"    {d:<11}{10 ** float(np.median(pts)):>9.3f}"
              f"{f'[{10 ** lo:.3f}, {10 ** hi:.3f}]':>26}{fpr:>7.0%}{verdict}")
    return acc


if __name__ == "__main__":
    if "--null" in sys.argv:
        for pre, stim in (("P1_pre", "P1_stim"), ("P5_pre", "P5_stim")):
            null_check(pre, stim)
    elif "--rate-gate" in sys.argv:
        rate_gate()
    elif "--scales" in sys.argv:
        scale_comparison()
    else:
        for r in [a for a in sys.argv[1:] if not a.startswith("--")] or ["P1_stim", "P5_stim"]:
            report(r)
