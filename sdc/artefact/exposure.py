"""
sdc.artefact.exposure
---------------------
How strongly does each detector fire on what the artefact mask catches?

    .venv\\Scripts\\python.exe -m sdc.artefact.exposure

THE MEASUREMENT
  Every run stores the detections the mask REJECTED (`{Det}_idx_masked`) beside the ones it
  kept, so both sides are countable against the time each occupies:

      in-mask  rate = rejected detections / channel-minutes the mask removed
      outside  rate = kept detections     / channel-minutes the mask left

  Their ratio is the enrichment. A detector indifferent to whatever the mask is catching scores
  ~1. Above 1 it is preferentially firing on it; below 1 it is avoiding it.

WHY THIS AND NOT "% OF DETECTIONS MASKED"
  A percentage confounds the detector's behaviour with how much time the mask removed -- a trial
  with 90% of its ON time masked will show a high percentage for every detector. Normalising by
  the time on each side removes that, so what is left is a property of the detector.

WHAT IT IS NOT
  It says a detector fires where the mask fires, not that either is right. The mask is not ground
  truth. Read the enrichment as "how exposed is this detector to whatever the mask selects",
  which is exactly the quantity that decides how much a detector's answer depends on the
  threshold -- and it is measured at ONE operating point, with no sweep.
"""
from pathlib import Path

import numpy as np

from sdc.common.paths import RUNS, figdir

DETS = ("Janca", "Barkmeier", "Delphos")
# P1's four trials: two at 145 Hz and two at 2 Hz, one ANT and one Pulvinar of each. That
# crossing is the point -- it separates a detector property from a target or a frequency.
RECS = ("P1", "P1_Pulv145", "P1_ANT2", "P1_Pulv2")
LABEL = {"P1": "P1 ANT 145 Hz", "P1_Pulv145": "P1 Pulv 145 Hz",
         "P1_ANT2": "P1 ANT 2 Hz", "P1_Pulv2": "P1 Pulv 2 Hz"}


def measure(stem, tag="_qcfinal"):
    """(in-mask rate, outside rate) in det/min/channel, per detector, plus the time split."""
    z = np.load(RUNS / f"{stem}_stim{tag}.npz", allow_pickle=False)
    if f"{DETS[0]}_idx_masked" not in z.files:
        raise SystemExit(f"{stem}: this run predates the mask-rejected arrays; re-run it.")
    fs, T = float(z["fs"]), float(z["seconds"])
    cps = np.asarray(z["clean_per_sec"], float)
    clean = cps.sum() / fs / 60.0                 # channel-minutes the mask LEFT
    total = cps.shape[1] * T / 60.0
    masked = max(total - clean, 1e-9)             # channel-minutes the mask REMOVED
    out = {}
    for d in DETS:
        if f"{d}_idx" not in z.files:
            continue
        out[d] = (z[f"{d}_idx_masked"].size / masked, z[f"{d}_idx"].size / clean)
    return out, clean, masked


def figure(recs=RECS, tag="_qcfinal"):
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, recessive

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    data, split = {}, {}
    for s in recs:
        data[s], cl, mk = measure(s, tag)
        split[s] = (cl, mk)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4))

    # ---- (a) the two rates side by side -------------------------------------------------
    ax = axes[0]
    x = np.arange(len(recs))
    # Six bars per trial (3 detectors x inside/outside) laid out on even slots across 0.86 of
    # the spacing. The previous version placed detector groups at fixed +-0.27 offsets with bars
    # 0.19 wide, so adjacent groups overlapped and the trial groups ran into each other.
    n_slot = len(DETS) * 2
    span = 0.86
    slot = span / n_slot
    for i, d in enumerate(DETS):
        ins = [data[s][d][0] for s in recs]
        outs = [data[s][d][1] for s in recs]
        p_in = x - span / 2 + slot * (2 * i) + slot / 2
        p_out = x - span / 2 + slot * (2 * i + 1) + slot / 2
        ax.bar(p_in, ins, slot * 0.86, color=colors[d], alpha=.95,
               label=f"{d} inside mask")
        ax.bar(p_out, outs, slot * 0.86, color=colors[d], alpha=.35,
               label=f"{d} outside")
    for xi in x[:-1]:                      # divider between trials, so the groups read apart
        ax.axvline(xi + 0.5, color="0.85", lw=0.8, zorder=0)
    ax.set_xlim(-0.6, len(recs) - 0.4)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(s, s) for s in recs], fontsize=9)
    ax.set_ylabel("detections / min / channel  (log)")
    ax.set_title("(a) rate INSIDE the masked time (solid) vs OUTSIDE it (faint).\n"
                 "Same mask, same recording, same channels -- only the detector differs.",
                 fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    recessive(ax)
    ax.grid(axis="y", alpha=.3)

    # ---- (b) the ratio, which is the finding --------------------------------------------
    ax = axes[1]
    for i, d in enumerate(DETS):
        y = [data[s][d][0] / max(data[s][d][1], 1e-12) for s in recs]
        ax.plot(x, y, "-o", ms=8, lw=1.8, color=colors[d], label=d)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}x", (xi, yi), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=colors[d])
    ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(s, s) for s in recs], fontsize=9)
    ax.set_ylabel("enrichment: in-mask rate / outside rate  (log)")
    ax.set_title("(b) ENRICHMENT. 1.0 = fires the same either side, so is indifferent to what\n"
                 "the mask catches. Above = drawn to it. Below = avoids it.", fontsize=9,
                 loc="left")
    ax.legend(frameon=False, fontsize=8)
    recessive(ax)
    ax.grid(axis="y", alpha=.3)

    print(f"{'trial':<16}{'masked':>9}{'clean':>9}   " +
          "  ".join(f"{d[:4]:>16}" for d in DETS))
    print(f"{'':<16}{'ch-min':>9}{'ch-min':>9}   " +
          "  ".join(f"{'in / out = x':>16}" for _ in DETS))
    for s in recs:
        cl, mk = split[s]
        cells = []
        for d in DETS:
            i_, o_ = data[s][d]
            cells.append(f"{i_:6.1f}/{o_:5.2f}={i_/max(o_,1e-12):5.1f}x")
        print(f"{LABEL.get(s, s):<16}{mk:>9.0f}{cl:>9.0f}   " + "  ".join(cells))

    fig.suptitle(
        "Detector exposure to the artefact mask, at ONE operating point (QC profile 'final').\n"
        "The mask is not ground truth -- this measures how much each detector's answer is "
        "hostage to where the threshold is put,\nwhich is why the same threshold change moves "
        "one detector and not another.", fontsize=10)
    fig.tight_layout()
    out = figdir("real") / f"detector_exposure{tag}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")
    return data


if __name__ == "__main__":
    figure()



def _continuous(z):
    """True if the recording has no OFF period at all (continuously stimulated).

    LF trials stimulate throughout, so every ON/OFF construction in this module is undefined for
    them: there is no adjacent OFF to pair against and no OFF channels to require. They are
    handled by chunking the whole recording and referring it to the stim-free pre file instead,
    which is the same estimator with a different reference rather than a different measurement.
    """
    return "sec_off" in z.files and float(z["sec_off"]) <= 0.0



class _ChunkBlocks:
    """block_table's interface for a CONTINUOUS recording: fixed chunks instead of ON blocks."""

    def __init__(self, keep, det, n_block):
        self.keep, self.det, self.n_block = keep, det, n_block

    @property
    def n_chan(self):
        return int(self.keep.sum())


def _chunk_blocks(z, chans=None, blk=64.0):
    """Per-(chunk, channel) counts and analysable seconds over a continuously-stimulated file."""
    fs = float(z["fs"])
    cps = np.asarray(z["clean_per_sec"], float)
    names = np.array([str(x) for x in z["names"]])
    n = len(names)
    keep = cps.sum(axis=0) > 0
    if chans is not None:
        want = set(chans)
        keep &= np.array([nm in want for nm in names], bool)
    ki = np.flatnonzero(keep)
    edges = np.arange(0, cps.shape[0] + 1, int(blk))
    det = {}
    for d in [str(x) for x in z["detectors"]]:
        t, c = z[f"{d}_idx"] / fs, z[f"{d}_chan"]
        oc, os_ = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (t >= a) & (t < b)
            oc.append(np.bincount(c[m], minlength=n)[ki])
            os_.append(cps[a:b].sum(axis=0)[ki] / fs)
        det[d] = {"on_count": np.array(oc, float), "on_sec": np.array(os_, float)}
    return _ChunkBlocks(keep, det, len(edges) - 1)


def _segment_rates(z, keep, min_seg_sec=20.0):
    """ABSOLUTE mean-over-channels rate (det/min/channel) for every ON and OFF segment.

    segment_course returns each detector's segments divided by that detector's own mean, which
    is right for "does every block dip" but useless as a numerator against an external baseline
    -- the values sit near 1 by construction whatever the rate is. This returns the rate itself.
    """
    from sdc.common import cond
    fs = float(z["fs"])
    cps = np.asarray(z["clean_per_sec"], float)
    n = len(z["names"])
    ON, OFF = cond.select(z, "on"), cond.select(z, "off")
    segs = sorted([(a, b, True) for a, b in ON.runs] + [(a, b, False) for a, b in OFF.runs])
    segs = [x for x in segs if x[1] - x[0] >= min_seg_sec]
    out = {}
    for d in [str(x) for x in z["detectors"]]:
        t, c = z[f"{d}_idx"] / fs, z[f"{d}_chan"]
        r = []
        for a, b, _o in segs:
            m = (t >= a) & (t < b)
            cnt = np.bincount(c[m], minlength=n)[keep]
            sec = cps[int(a):int(b)].sum(axis=0)[keep] / fs
            ok = sec > 0
            r.append(np.mean(cnt[ok] / sec[ok]) * 60.0 if ok.any() else np.nan)
        out[d] = np.array(r, float)
    return (np.array([0.5 * (a + b) for a, b, _ in segs]),
            np.array([x[2] for x in segs], bool), out)


def _on_bounds(z):
    """ON-run start/stop seconds from a stim run, as two lists."""
    if "on_runs" not in z.files:
        return [], []
    r = np.asarray(z["on_runs"], float) / float(z["fs"])
    return list(r[:, 0]), list(r[:, 1])


def _chunk_rates(z, keep, chunk_sec):
    """Median per-chunk rate (det/min/channel) over a recording with no ON/OFF structure.

    Used for the stim-free baseline. Chunking to the STIM file's block length matters: a rate
    measured over one 652 s span and a rate measured over a 64 s block are not the same
    statistic once the train is bursty, so the reference has to be built on the same timescale
    as the thing it normalises.
    """
    fs = float(z["fs"])
    cps = np.asarray(z["clean_per_sec"], float)
    n = len(z["names"])
    edges = np.arange(0, cps.shape[0] + 1, max(int(chunk_sec), 1))
    out = {}
    for d in [str(x) for x in z["detectors"]]:
        t, c = z[f"{d}_idx"] / fs, z[f"{d}_chan"]
        r = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (t >= a) & (t < b)
            cnt = np.bincount(c[m], minlength=n)[keep]
            sec = cps[a:b].sum(axis=0)[keep] / fs
            ok = sec > 0
            if ok.any():
                r.append(np.mean(cnt[ok] / sec[ok]) * 60.0)
        out[d] = float(np.nanmedian(r)) if r else float("nan")
    return out


# --------------------------------------------------------------------------------------
def period_course(recs=("P1_stim",), tag="_qcfinal", prefer="after", bin_sec=60.0,
                  baseline="auto", outdir=None, pre_stem=None, chans=None,
                  off_full=True, show_baseline=True, paired=True, fname=None, tmax=None):
    """The stim recording in ~1 min bins that RESPECT the ON/OFF transitions, with the
    stim-free baseline recording drawn on the same axis to its left.

    BINS OBEY THE BLOCK EDGES. Each ON run and each OFF run is split into roughly `bin_sec`
    pieces, so no bin ever straddles a transition -- a ~1 min ON block gives one point and a
    ~3 min OFF gives three, each one pure. On a fixed grid the block edge walks through the bins
    (P1's blocks are 64 s against a 60 s grid) and every bin mixes the two, diluting the contrast
    the figure exists to show.

    THE BASELINE RECORDING IS DRAWN, NOT JUST DIVIDED BY. Showing only a horizontal line at 1
    hides how much the baseline itself moves from minute to minute -- which is the natural yard-
    stick for whether an ON dip is large. It sits left of the break at t=0, in its own bins of
    the same length, so like is compared with like.
    """
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, recessive
    from sdc.artefact.blocks import block_table, block_binned_course

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    panels = []
    for stem in recs:
        z = np.load(RUNS / f"{stem}{tag}.npz", allow_pickle=False)
        if paired:
            keep = block_table(z, prefer=prefer, chans=chans, off_full=off_full).keep
        else:
            # POOLED channel selection: every channel measurable in this recording, without the
            # block-paired gates. Keeps this figure independent of the paired estimator, so an
            # artefact argument built on it does not presuppose the pairing -- the two can then
            # be presented as separate steps rather than one compound change.
            # Measurable in BOTH conditions. "Any clean time anywhere" is not enough: on
            # finalv2, 224 channels have some analysable time but only 139 have any during
            # stim ON, because stim_dilation condemns whole ON blocks per channel. Under that
            # rule the ON points were averaged over 139 channels and the OFF points over 224 --
            # the two halves of the same panel describing different channel sets, which is the
            # unpaired error that flipped Barkmeier's P5 ratio from 1.113 to 0.824.
            _cps = np.asarray(z["clean_per_sec"], float)
            _on = np.asarray(z["on_per_sec"], bool)
            _m = min(_on.size, _cps.shape[0])
            _k_on = _cps[:_m][_on[:_m]].sum(axis=0) > 0
            _k_off = _cps[:_m][~_on[:_m]].sum(axis=0) > 0
            keep = _k_on if _continuous(z) else (_k_on & _k_off)
            if chans is not None:
                want = set(chans)
                keep &= np.array([str(x) in want for x in z["names"]], bool)
        mids, course, ison = block_binned_course(z, keep, bin_sec)
        if tmax is not None:
            # Drop bins beginning after `tmax`. Used to cut a terminal ON burst whose artefact
            # dominates a detector: on P1 145 Hz the 970-1034 s block drives most of Delphos's
            # elevation, so leaving it in reports an artefact level as an effect size.
            # The BASELINE is untouched -- it is a separate recording and has no such burst.
            _sel = np.asarray(mids, float) <= float(tmax)
            mids, ison = np.asarray(mids)[_sel], np.asarray(ison)[_sel]
            course = {d: {k2: (np.asarray(v2)[_sel]
                               if np.ndim(v2) and np.asarray(v2).shape[:1] == _sel.shape else v2)
                          for k2, v2 in dd.items()} for d, dd in course.items()}

        pre = (RUNS / f"{pre_stem}.npz" if pre_stem else
               RUNS / f"{stem.replace('_stim', '_pre')}{tag}.npz")
        pre_mids, pre_course, norm, use = None, None, {}, None
        if baseline in ("pre", "auto") and pre.is_file():
            zp = np.load(pre, allow_pickle=False)
            if [str(x) for x in zp["names"]] == [str(x) for x in z["names"]]:
                pre_course = _chunk_series(zp, keep, bin_sec)
                pre_mids = (np.arange(len(next(iter(pre_course.values())))) + 0.5) * bin_sec
                norm = {d: float(np.nanmedian(v)) for d, v in pre_course.items()}
                use = "pre"
        if use is None:
            norm = {d: float(np.nanmedian(course[d]["mean"][~ison])) for d in course}
            use = "off"
        panels.append((stem, mids, ison, course, pre_mids, pre_course, norm, use,
                       int(keep.sum())))

    fig, axes = plt.subplots(len(panels), 1, figsize=(14.0, 4.6 * len(panels)), squeeze=False)
    for k, (stem, mids, ison, course, pre_mids, pre_course, norm, use, nch) in enumerate(panels):
        ax = axes[k][0]
        # baseline to the LEFT of 0, stim to the right, with a visible break between them
        off = 0.0
        # show_baseline=False keeps the pre file as the NORMALISER but stops drawing it, so the
        # x axis is the stim recording alone and the panel sits beside pooled_vs_pairwise on the
        # same time span. The dashed line at 1 still means "the stim-free baseline".
        if not show_baseline:
            pre_mids = pre_course = None
        if pre_mids is not None:
            off = (pre_mids[-1] + bin_sec) / 60.0
            ax.axvspan(-off - .3, 0, color="0.93", lw=0, zorder=0)
            ax.text(-off / 2, 0.02, "stim-free baseline", transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=8, color="0.35")
        ax.axvline(0, color="0.4", lw=1.4)
        for t, o in zip(mids, ison):
            if o:
                ax.axvspan((t - bin_sec / 2) / 60.0, (t + bin_sec / 2) / 60.0,
                           color="#f0c419", alpha=.20, lw=0, zorder=0)
        for d in ("Janca", "Barkmeier", "Delphos"):
            if d not in course or not np.isfinite(norm.get(d, np.nan)) or norm[d] <= 0:
                continue
            if pre_course is not None and d in pre_course:
                yb = pre_course[d] / norm[d]
                ax.plot(pre_mids / 60.0 - off, yb, "-", lw=1.2, color=colors[d], alpha=.45)
                ax.scatter(pre_mids / 60.0 - off, yb, s=30, facecolor="white",
                           edgecolor=colors[d], lw=1.3, zorder=3)
            y = course[d]["mean"] / norm[d]
            ax.plot(mids / 60.0, y, "-", lw=1.3, color=colors[d], alpha=.7, zorder=2)
            # Median over channels alongside the mean. The mean is rate-weighted, so it follows
            # the busiest contacts; the median is the typical channel. Where they diverge, a
            # minority of channels is carrying the trace -- which is the whole 2 Hz story.
            if "median" in course[d]:
                ax.plot(mids / 60.0, course[d]["median"] / norm[d], ":", lw=1.6,
                        color=colors[d], alpha=.85, zorder=2)
            ax.scatter(mids[ison] / 60.0, y[ison], s=62, color=colors[d], zorder=4,
                       label=f"{d} stim ON")
            ax.scatter(mids[~ison] / 60.0, y[~ison], s=40, facecolor="white", zorder=4,
                       edgecolor=colors[d], lw=1.5, label=f"{d} OFF")
        ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
        ax.set_xlabel("time (min)   |   0 = start of the stim recording")
        ax.set_ylabel(f"rate / baseline median\n({bin_sec:g}s bins, block-aligned)")
        ax.set_title(f"{stem}{tag}: baseline = {use}"
                     f"{'  (pre file, drawn left of 0)' if use == 'pre' else '  (own OFF bins)'},"
                     f"  {nch} channels", fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=7, ncol=3)
        recessive(ax)
        ax.grid(alpha=.3)
        cells = []
        for d in sorted(course):
            if not np.isfinite(norm.get(d, np.nan)) or norm[d] <= 0:
                continue
            y = course[d]["mean"] / norm[d]
            cells.append(f"{d[:4]} ON {np.nanmedian(y[ison]):.2f} OFF {np.nanmedian(y[~ison]):.2f}")
        print(f"{stem:<12} baseline={use:<4} {nch:>4} ch   " + "  ".join(cells))

    fig.suptitle("Every ~1 min bin, aligned to the ON/OFF transitions so no bin straddles one.\n"
                 "Filled = stim ON, open = OFF, open on grey = the stim-free baseline recording. "
                 "Dashed = its median.", fontsize=10)
    fig.tight_layout()
    # Named after the CONDITION, not a fixed filename. Every call used to write the same
    # period_vs_baseline.png, so running it for a second artefact condition silently replaced
    # the first and the two could not be compared -- the failure this project keeps hitting.
    out = (Path(outdir) if outdir else figdir("real")) / (fname or "period_vs_baseline.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")
    return panels


def _chunk_series(z, keep, chunk_sec):
    """Per-chunk rate SERIES (not just its median) for a recording with no ON/OFF structure."""
    fs = float(z["fs"])
    cps = np.asarray(z["clean_per_sec"], float)
    n = len(z["names"])
    edges = np.arange(0, cps.shape[0] + 1, max(int(chunk_sec), 1))
    out = {}
    for d in [str(x) for x in z["detectors"]]:
        t, c = z[f"{d}_idx"] / fs, z[f"{d}_chan"]
        r = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (t >= a) & (t < b)
            cnt = np.bincount(c[m], minlength=n)[keep]
            sec = cps[a:b].sum(axis=0)[keep] / fs
            ok = sec > 0
            r.append(np.mean(cnt[ok] / sec[ok]) * 60.0 if ok.any() else np.nan)
        out[d] = np.array(r, float)
    return out


# --------------------------------------------------------------------------------------
def baseline_estimators(stim_stem="P1_stim", tag="_qcfinal", pre_stem=None, prefer="after",
                        chans=None, outdir=None, scale="linear", agg="mean",
                        off_full=True):
    """Pooled vs per-block, for the ON / stim-free BASELINE ratio.

    The same question compare_estimators asks of the ON/OFF contrast -- does it matter WHEN the
    blocks are combined? -- but against the pre recording instead of the adjacent OFF window.

      pooled     counts summed over ON blocks first, then one ratio per channel against that
                 channel's baseline rate. A long or busy block carries more of the answer.
      per-block  each ON block reduced to one number against the baseline first, so every block
                 gets one vote whatever its size. Only this one has a spread to report.

    WHY IT IS WORTH HAVING SEPARATELY FROM THE ON/OFF VERSION
      ON/OFF cancels drift by construction but can only see the CONTRAST -- an ON dip and an OFF
      rise are the same number. Against a fixed external baseline both sides are visible, at the
      cost of no longer being drift-free. Reporting the pair is the honest way round that: if the
      two agree, drift is not driving the result.

    Channels must be measurable in BOTH recordings, and names are checked rather than assumed --
    a positional match would silently compare different contacts.
    """
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, recessive
    from sdc.artefact.blocks import block_table

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    zs = np.load(RUNS / f"{stim_stem}{tag}.npz", allow_pickle=False)
    zp = np.load(RUNS / f"{pre_stem or stim_stem.replace('_stim', '_pre') + tag}.npz",
                 allow_pickle=False)
    if [str(x) for x in zp["names"]] != [str(x) for x in zs["names"]]:
        raise SystemExit("stim and baseline runs have different channel names.")

    # A continuously-stimulated file has no ON blocks to pair, so chunk the whole recording
    # into blocks of the length the HF trials use; everything below needs only per-block counts
    # and analysable seconds.
    bp = (_chunk_blocks(zs, chans=chans, blk=64.0) if _continuous(zs)
          else block_table(zs, prefer=prefer, chans=chans, off_full=off_full))
    names = np.array([str(x) for x in zs["names"]])
    kept = names[bp.keep]
    n = len(names)
    fs_p = float(zp["fs"])
    cps_p = np.asarray(zp["clean_per_sec"], float)
    idx = [list(names).index(c) for c in kept]

    out, summary = {}, []
    for d in [str(x) for x in zs["detectors"]]:
        if d not in bp.det:
            continue
        base_c = np.bincount(zp[f"{d}_chan"], minlength=n)[idx].astype(float)
        base_t = cps_p.sum(axis=0)[idx] / fs_p / 60.0          # channel-minutes
        ok = (base_t > 0) & (base_c > 0)
        if not ok.any():
            continue
        base_r = base_c[ok] / base_t[ok]

        oc = bp.det[d]["on_count"][:, ok]
        ot = bp.det[d]["on_sec"][:, ok] / 60.0

        # pooled: blocks summed FIRST, then one ratio per channel
        pc, pt = oc.sum(axis=0), ot.sum(axis=0)
        good = (pt > 0) & (base_r > 0)
        lg = np.log10(np.where(pc[good] > 0, pc[good], 0.5) / pt[good] / base_r[good])
        pooled = 10 ** (lg.mean() if agg == "mean" else np.median(lg))

        # per-block: each block reduced to one number FIRST
        per = []
        for b in range(oc.shape[0]):
            g = (ot[b] > 0) & (base_r > 0)
            if not g.any():
                continue
            l = np.log10(np.where(oc[b][g] > 0, oc[b][g], 0.5) / ot[b][g] / base_r[g])
            per.append(10 ** (l.mean() if agg == "mean" else np.median(l)))
        per = np.array(per, float)
        out[d] = {"pooled": pooled, "per": per,
                  "combined": 10 ** np.median(np.log10(per)) if per.size else np.nan}
        summary.append((d, pooled, out[d]["combined"], per))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    for i, (d, pooled, comb, per) in enumerate(summary):
        bx = ax.boxplot([per], positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                        zorder=2)
        bx["boxes"][0].set(facecolor=colors[d], alpha=.20, edgecolor=colors[d])
        for part in ("whiskers", "caps", "medians"):
            for ln in bx[part]:
                ln.set(color=colors[d], lw=1.6)
        rng = np.random.default_rng(0)
        ax.scatter(i + rng.uniform(-.12, .12, per.size), per, s=42, color=colors[d],
                   alpha=.6, edgecolor="none", zorder=3)
        ax.scatter([i], [pooled], marker="D", s=95, facecolor="white", edgecolor=colors[d],
                   lw=2.2, zorder=4)
    ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
    ax.set_yscale(scale)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels([d for d, *_ in summary], fontsize=9)
    ax.set_ylabel(f"ON block / baseline ({scale})")
    ax.set_title(f"(a) {stim_stem}{tag}, {int(bp.keep.sum())} channels, {len(summary and per)} "
                 f"ON blocks\nbox + points = per-block; white diamond = pooled",
                 fontsize=9, loc="left")
    recessive(ax); ax.grid(axis="y", alpha=.3)

    ax = axes[1]
    for d, pooled, comb, per in summary:
        ax.scatter([pooled], [comb], s=110, color=colors[d], label=d, zorder=3)
    lim = [min(0.1, *[min(s[1], s[2]) for s in summary]) * 0.9,
           max(1.2, *[max(s[1], s[2]) for s in summary]) * 1.1]
    ax.plot(lim, lim, color="0.35", ls="--", lw=1.2)
    ax.axhline(1.0, color="0.75", ls=":", lw=1.0)
    ax.axvline(1.0, color="0.75", ls=":", lw=1.0)
    ax.set_xscale(scale); ax.set_yscale(scale)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("pooled over ON blocks")
    ax.set_ylabel("per-block, then combined")
    ax.set_title("(b) the two estimators against each other", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    recessive(ax); ax.grid(alpha=.3)

    print(f"{stim_stem}{tag}  vs baseline   {int(bp.keep.sum())} channels")
    print(f"  {'detector':<11}{'pooled':>9}{'per-block':>11}{'ratio':>8}   per-block values")
    for d, pooled, comb, per in summary:
        print(f"  {d:<11}{pooled:>9.3f}{comb:>11.3f}{comb / pooled:>8.2f}   "
              + " ".join(f"{x:.2f}" for x in per))

    fig.suptitle("Pooled vs per-block for the ON / stim-free-BASELINE ratio.\n"
                 "Unlike the ON/OFF version this is not drift-free -- but it can tell an ON dip "
                 "from an OFF rise, which ON/OFF cannot.", fontsize=11)
    fig.tight_layout()
    out_p = (Path(outdir) if outdir else figdir("real")) / "pooled_vs_pairwise_baseline.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out_p}")
    return out


# --------------------------------------------------------------------------------------
def effect_per_channel(stim_stem="P1_stim", tag="_qcfinal", pre_stem=None, prefer="after",
                       chans=None, outdir=None, label="EZ"):
    """stim_effect's panel (b) -- per-channel effect size -- for BOTH contrasts side by side.

    Left  ON vs its adjacent OFF window. Drift-free by construction, but an ON dip and an OFF
          rise are indistinguishable in it.
    Right ON vs the stim-free BASELINE recording. Sees both sides, but is not drift-free.

    Agreement between the two is the useful part: drift would have to move the baseline
    comparison by exactly the amount needed to mimic the paired one, so when they land in the
    same place drift is not what is producing the effect.

    log2 like the original panel, so "halved" and "doubled" sit the same distance from no-effect
    -- on a linear ratio axis they are 0.5 and 1.0 away, which reads as an asymmetry that is not
    there. Channels with zero ON detections are counted rather than dropped: a channel
    stimulation silenced completely is the strongest evidence there is, and deleting it removes
    exactly that.
    """
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, MUTED, recessive
    from sdc.artefact.blocks import block_table, per_channel_log_ratio

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    L2 = np.log10(2.0)
    zs = np.load(RUNS / f"{stim_stem}{tag}.npz", allow_pickle=False)
    zp = np.load(RUNS / f"{pre_stem or stim_stem.replace('_stim', '_pre') + tag}.npz",
                 allow_pickle=False)
    if [str(x) for x in zp["names"]] != [str(x) for x in zs["names"]]:
        raise SystemExit("stim and baseline runs have different channel names.")

    # A continuously-stimulated file has no ON blocks to pair, so chunk the whole recording
    # into blocks of the length the HF trials use; everything below needs only per-block counts
    # and analysable seconds.
    bp = (_chunk_blocks(zs, chans=chans, blk=64.0) if _continuous(zs)
          else block_table(zs, prefer=prefer, chans=chans, off_full=off_full))
    names = np.array([str(x) for x in zs["names"]])
    kept = names[bp.keep]
    idx = [list(names).index(c) for c in kept]
    n = len(names)
    cps_p = np.asarray(zp["clean_per_sec"], float)
    fs_p = float(zp["fs"])

    dets = [d for d in ("Janca", "Barkmeier", "Delphos") if d in bp.det]
    data = {}
    for d in dets:
        onoff = per_channel_log_ratio(bp.det[d]) / L2            # log2, one per channel

        base_c = np.bincount(zp[f"{d}_chan"], minlength=n)[idx].astype(float)
        base_t = cps_p.sum(axis=0)[idx] / fs_p / 60.0
        oc = bp.det[d]["on_count"].sum(axis=0)
        ot = bp.det[d]["on_sec"].sum(axis=0) / 60.0
        ok = (ot > 0) & (base_t > 0) & (base_c > 0)
        onbase = np.log10(np.where(oc[ok] > 0, oc[ok], 0.5) / ot[ok]
                          / (base_c[ok] / base_t[ok])) / L2
        data[d] = {"onoff": onoff, "onbase": onbase,
                   "n_zero": int((oc[ok] == 0).sum())}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    for ax, key, ttl in ((axes[0], "onoff", "ON / adjacent OFF   (drift-free)"),
                         (axes[1], "onbase", "ON / stim-free baseline   (sees both sides)")):
        for i, d in enumerate(dets):
            v = data[d][key]
            v = v[np.isfinite(v)]
            col = colors[d]
            ax.boxplot([v], positions=[i], widths=.55, showfliers=False,
                       medianprops=dict(color=col, lw=2), boxprops=dict(color=col),
                       whiskerprops=dict(color=col), capprops=dict(color=col))
            ax.scatter(np.full(v.size, i) + np.linspace(-.18, .18, v.size), v, s=22,
                       color=col, alpha=.55, edgecolor="none", zorder=3)
            ax.annotate(f"med {2 ** np.median(v):.2f}x  n={v.size}", (i, 0.015),
                        xycoords=("data", "axes fraction"), ha="center", va="bottom",
                        fontsize=7.5, color=col)
            if key == "onbase" and data[d]["n_zero"]:
                ax.annotate(f"+{data[d]['n_zero']} silenced", (i, 0.075),
                            xycoords=("data", "axes fraction"), ha="center", va="bottom",
                            fontsize=7, color=col)
        ax.axhline(0, color=MUTED, ls="--", lw=1.1)
        ax.set_xticks(range(len(dets)))
        ax.set_xticklabels(dets, fontsize=9)
        ax.set_title(ttl, fontsize=9, loc="left")
        recessive(ax)
        ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("log2(rate ratio), per channel\nbelow 0 = suppression during stim")
    axes[0].text(.02, .97, "above 0 = more spikes during stim", transform=axes[0].transAxes,
                 va="top", fontsize=8, color=MUTED)

    print(f"{stim_stem}{tag}  [{label}]  {int(bp.keep.sum())} channels, {bp.n_block} ON blocks")
    print(f"  {'detector':<11}{'ON/OFF med':>12}{'ON/base med':>13}{'silenced':>10}")
    for d in dets:
        print(f"  {d:<11}{2 ** np.median(data[d]['onoff']):>12.3f}"
              f"{2 ** np.median(data[d]['onbase']):>13.3f}{data[d]['n_zero']:>10}")

    fig.suptitle(f"{stim_stem}{tag} -- per-channel effect size, {label} channels "
                 f"({int(bp.keep.sum())}).  Both contrasts, same channels, same blocks.\n"
                 "They have opposite weaknesses, so agreement between them is the point.",
                 fontsize=11)
    fig.tight_layout()
    out = (Path(outdir) if outdir else figdir("real")) / f"effect_per_channel_{label}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")
    return data


# --------------------------------------------------------------------------------------
def pair_course(rec="P1_stim", tag="_qcfinalv2_t970", prefer="after", chans=None,
                outdir=None, scale="linear", label="", off_full=True):
    """stim_effect panel (c), but plotting the NUMBERS pooled_vs_pairwise reports.

    Panel (c) normalises each segment by that detector's own mean across all segments -- ON and
    OFF together -- so its heights are not comparable with any ratio quoted elsewhere, and the
    ON blocks are part of the baseline they are being measured against. Here each ON block is
    divided by ITS OWN matched OFF window, so a marker's height IS that pair's ratio: exactly
    the points in pooled_vs_pairwise's box, in time order instead of collapsed into a box.

    The two horizontal lines are the two estimators from that figure -- pooled (all blocks
    combined first) and per-pair (each block reduced first, then combined). Where they separate,
    the gap is being driven by whichever blocks sit furthest from the per-pair line.
    """
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, recessive
    from sdc.artefact.blocks import block_table, pair_changes, log_ratio

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    z = np.load(RUNS / f"{rec}{tag}.npz", allow_pickle=False)
    bp = block_table(z, prefer=prefer, chans=chans, off_full=off_full)
    dets = [d for d in ("Janca", "Barkmeier", "Delphos") if d in bp.det]
    mid = np.array([0.5 * (a + b) for a, b in bp.on_w]) / 60.0

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for a, b in bp.on_w:
        ax.axvspan(a / 60.0, b / 60.0, color="#f0c419", alpha=.20, lw=0, zorder=0)
    for d in dets:
        v, _ = pair_changes(bp.det[d])
        y = 10 ** v
        ok = np.isfinite(y)
        ax.plot(mid[ok], y[ok], "-o", ms=9, lw=1.5, color=colors[d], label=d, zorder=3)
        ax.axhline(10 ** log_ratio(bp.det[d]), color=colors[d], ls=":", lw=1.6, alpha=.8)
        ax.axhline(10 ** np.median(v[ok]), color=colors[d], ls="-", lw=1.2, alpha=.35)
    ax.axhline(1.0, color="0.25", ls="--", lw=1.4)
    ax.set_yscale(scale)
    ax.set_xlabel("time (min) -- shaded = the ON block, plotted at its own midpoint")
    ax.set_ylabel(f"ON block / its OWN matched OFF window ({scale})")
    ax.set_title(f"{rec}{tag}{('  [' + label + ']') if label else ''} -- "
                 f"{bp.n_block} pairs, {bp.n_chan} channels.  Dotted = pooled, "
                 f"faint solid = per-pair median.", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=3)
    recessive(ax)
    ax.grid(axis="y", alpha=.3)

    print(f"{rec}{tag}{('  ' + label) if label else ''}   {bp.n_block} pairs, {bp.n_chan} ch")
    for d in dets:
        v, _ = pair_changes(bp.det[d])
        y = 10 ** v[np.isfinite(v)]
        print(f"  {d:<11}pooled {10 ** log_ratio(bp.det[d]):.3f}  per-pair {np.median(y):.3f}   "
              + " ".join(f"{x:.2f}" for x in y))

    fig.tight_layout()
    out = (Path(outdir) if outdir else figdir("real")) / "pair_course.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")
    return out


# --------------------------------------------------------------------------------------
def baseline_effect_box(rec="P1_stim", tag="_qcfinalv2_t970", pre_stem=None, chans=None,
                        outdir=None, label="", fname="baseline_effect_box"):
    """stim_effect panel (b), but against the stim-free BASELINE instead of the OFF periods.

    One value per channel: log2(its ON rate / its own rate in the pre recording). Box plus every
    channel as a point, per detector. Zero means "stimulation changed nothing on this channel".

    WHY AGAINST THE BASELINE
      Panel (b) uses ON/OFF, which cannot separate an ON dip from an OFF rise and is blind to
      anything that moves both. The pre file is a separate recording, so it fixes the zero point
      externally -- the three detectors then sit on one axis whose meaning does not depend on any
      of them, which is what makes their spread readable as agreement.

    CHANNEL SELECTION is pooled (measurable in BOTH conditions and in the baseline), not the
    block-paired set, so this figure stands on its own without presupposing the pairing. Pass
    `chans` to force a common set across runs -- otherwise a masked run and an unmasked one are
    summarising different channels and the comparison is partly selection.

    Channels with zero ON detections are counted, not dropped: a channel stimulation silenced is
    the strongest evidence there is, and deleting it removes exactly that.
    """
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, MUTED, recessive

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    L2 = np.log10(2.0)
    zs = np.load(RUNS / f"{rec}{tag}.npz", allow_pickle=False)
    zp = np.load(RUNS / f"{pre_stem}.npz", allow_pickle=False)
    names = np.array([str(x) for x in zs["names"]])
    if [str(x) for x in zp["names"]] != list(names):
        raise SystemExit("stim and baseline runs have different channel names.")

    fs = float(zs["fs"])
    cps = np.asarray(zs["clean_per_sec"], float)
    on = np.asarray(zs["on_per_sec"], bool)
    m = min(on.size, cps.shape[0])
    on_sec = cps[:m][on[:m]].sum(axis=0) / fs / 60.0          # channel-minutes ON
    off_sec = cps[:m][~on[:m]].sum(axis=0) / fs / 60.0
    keep = (on_sec > 0) & (off_sec > 0) if not _continuous(zs) else (on_sec > 0)
    if chans is not None:
        want = set(chans)
        keep &= np.array([n in want for n in names], bool)
    idx = np.flatnonzero(keep)

    cpp = np.asarray(zp["clean_per_sec"], float)
    base_min = cpp.sum(axis=0) / float(zp["fs"]) / 60.0
    n = len(names)
    dets = [d for d in ("Janca", "Barkmeier", "Delphos") if f"{d}_idx" in zs.files]

    data = {}
    for d in dets:
        t = zs[f"{d}_idx"] / fs
        sel = on[np.clip(t.astype(int), 0, on.size - 1)]
        on_c = np.bincount(zs[f"{d}_chan"][sel], minlength=n)[idx].astype(float)
        b_c = np.bincount(zp[f"{d}_chan"], minlength=n)[idx].astype(float)
        b_t = base_min[idx]
        ok = (b_c > 0) & (b_t > 0) & (on_sec[idx] > 0)
        v = np.log10(np.where(on_c[ok] > 0, on_c[ok], 0.5) / on_sec[idx][ok]
                     / (b_c[ok] / b_t[ok])) / L2
        data[d] = {"v": v, "n_zero": int((on_c[ok] == 0).sum())}

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for i, d in enumerate(dets):
        v = data[d]["v"]
        v = v[np.isfinite(v)]
        col = colors[d]
        ax.boxplot([v], positions=[i], widths=.55, showfliers=False,
                   medianprops=dict(color=col, lw=2.2), boxprops=dict(color=col),
                   whiskerprops=dict(color=col), capprops=dict(color=col))
        ax.scatter(np.full(v.size, i) + np.linspace(-.19, .19, v.size), v, s=14,
                   color=col, alpha=.45, edgecolor="none", zorder=3)
        ax.annotate(f"{2 ** np.median(v):.2f}x", (i, 0.015), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=9, color=col)
        if data[d]["n_zero"]:
            ax.annotate(f"+{data[d]['n_zero']} silenced", (i, 0.075),
                        xycoords=("data", "axes fraction"), ha="center", va="bottom",
                        fontsize=7.5, color=col)
    ax.axhline(0, color=MUTED, ls="--", lw=1.2)
    ax.text(.02, .97, "above 0 = MORE spikes during stim", transform=ax.transAxes, va="top",
            fontsize=8, color=MUTED)
    ax.set_xticks(range(len(dets)))
    ax.set_xticklabels(dets, fontsize=10)
    ax.set_ylabel("log2(ON rate / stim-free baseline rate), per channel")
    ax.set_title(f"{rec}{tag}{('  [' + label + ']') if label else ''}\n"
                 f"{idx.size} channels, pooled selection.  Zero = the stim-free recording.",
                 fontsize=10, loc="left")
    recessive(ax)
    ax.grid(axis="y", alpha=.3)

    med = {d: 2 ** float(np.median(data[d]["v"])) for d in dets}
    print(f"{rec}{tag}{('  ' + label) if label else ''}   {idx.size} ch   " +
          "  ".join(f"{d[:4]} {med[d]:.3f}" for d in dets) +
          f"   spread {max(med.values()) / min(med.values()):.2f}x")

    fig.tight_layout()
    # `fname` because the name was hardcoded: calling this for a second variant into the same
    # folder silently replaced the first, which is how the control version of this figure was
    # lost once already.
    out = (Path(outdir) if outdir else figdir("real")) / f"{fname}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")
    return data


# --------------------------------------------------------------------------------------
def rate_gate_box(rec="P1_ANT2_stim", tag="_qcfinalv2", pre_stem="P1_ANT2_pre_qcfinalv2",
                  gates=(None, 50, 67, 80), outdir=None, label="", fname="rate_gate_box"):
    """How the stim/baseline effect changes as low-rate channels are excluded.

    A ratio needs a denominator. A channel firing 0.1/min in the baseline gives a ratio built on
    a handful of detections, and on this file those channels are what drive Delphos to 1.78 --
    its stim/baseline correlates -0.61 with the channel's baseline rate, reading 3.49x on the
    quietest third and 0.92x on the busiest. `block_table` already refuses such channels for the
    HF arm (MIN_OFF_RATE); this asks the same question of the LF arm, where nothing does.

    The gate is a PERCENTILE OF THE BASELINE RATE, per detector -- a channel-quality criterion
    measured on the stim-free recording, so it cannot select on the effect being measured.

    Row (b) carries the absolute rates so the gate's cost is visible: a subset that agrees
    beautifully because it kept only the busiest ten channels is not a result.
    """
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, MUTED, recessive

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    L2 = np.log10(2.0)
    zs = np.load(RUNS / f"{rec}{tag}.npz", allow_pickle=False)
    zp = np.load(RUNS / f"{pre_stem}.npz", allow_pickle=False)
    names = np.array([str(x) for x in zs["names"]])
    n = len(names)
    fs = float(zs["fs"])
    stim_min = np.asarray(zs["clean_per_sec"], float).sum(axis=0) / fs / 60.0
    base_min = np.asarray(zp["clean_per_sec"], float).sum(axis=0) / float(zp["fs"]) / 60.0
    dets = [d for d in ("Janca", "Barkmeier", "Delphos") if f"{d}_idx" in zs.files]

    rates = {}
    for d in dets:
        sc = np.bincount(zs[f"{d}_chan"], minlength=n).astype(float)
        bc = np.bincount(zp[f"{d}_chan"], minlength=n).astype(float)
        rs = np.where(stim_min > 0, sc / np.maximum(stim_min, 1e-9), np.nan)
        rb = np.where(base_min > 0, bc / np.maximum(base_min, 1e-9), np.nan)
        rates[d] = (rs, rb)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 9.0), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
    labels, summary = [], []
    for gi, q in enumerate(gates):
        labels.append("all" if q is None else f">p{q}")
        for di, d in enumerate(dets):
            rs, rb = rates[d]
            ok = np.isfinite(rs) & np.isfinite(rb) & (rb > 0)
            sel = ok if q is None else ok & (rb >= np.nanpercentile(rb[ok], q))
            v = np.log10(np.where(rs[sel] > 0, rs[sel], 0.5 / np.maximum(stim_min[sel], 1e-9))
                         / rb[sel]) / L2
            v = v[np.isfinite(v)]
            x = gi + (di - 1) * 0.28
            col = colors[d]
            bx = axes[0].boxplot([v], positions=[x], widths=.22, showfliers=False,
                                 patch_artist=True, zorder=2)
            bx["boxes"][0].set(facecolor=col, alpha=.20, edgecolor=col)
            for part in ("whiskers", "caps", "medians"):
                for ln in bx[part]:
                    ln.set(color=col, lw=1.6)
            rng = np.random.default_rng(0)
            axes[0].scatter(x + rng.uniform(-.08, .08, v.size), v, s=11, color=col,
                            alpha=.40, edgecolor="none", zorder=3)
            axes[1].plot([x], [np.nanmedian(rb[sel])], "s", ms=8, color=col, alpha=.45)
            axes[1].plot([x], [np.nanmedian(rs[sel])], "o", ms=8, color=col)
            # The median is the number every conclusion is drawn from, and reading it off a
            # box edge is guesswork. Printed in BOTH units: log2 at the median line, because
            # that is the axis, and the ratio along the bottom, because that is what gets
            # quoted -- and 2^-0.5 = 0.71 is not something to do in your head.
            _m = float(np.median(v))
            axes[0].annotate(f"{_m:+.2f}", (x, _m), ha="center", va="bottom", fontsize=7.5,
                             color=col, fontweight="bold", zorder=6,
                             xytext=(0, 4), textcoords="offset points")
            axes[0].annotate(f"{2 ** _m:.2f}x", (x, 0.012), xycoords=("data", "axes fraction"),
                             ha="center", va="bottom", fontsize=7.5, color=col)
            summary.append((labels[-1], d, int(sel.sum()), 2 ** float(np.median(v)),
                            float(np.nanmedian(rb[sel])), float(np.nanmedian(rs[sel]))))
    axes[0].axhline(0, color=MUTED, ls="--", lw=1.2)
    axes[0].set_ylabel("log2(stim rate / baseline rate), per channel")
    axes[0].set_title(f"{rec}{tag}{('  [' + label + ']') if label else ''} -- excluding "
                      f"low-rate channels by BASELINE-rate percentile.\n"
                      "Zero = the stim-free recording. Box + one point per channel; "
                      "bold = median log2, bottom = the same as a ratio.",
                      fontsize=10, loc="left")
    h = [plt.Line2D([], [], color=colors[d], lw=3, label=d) for d in dets]
    axes[0].legend(handles=h, frameon=False, fontsize=9, ncol=3)
    recessive(axes[0]); axes[0].grid(axis="y", alpha=.3)

    axes[1].set_ylabel("median channel rate (det/min)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_xlabel("baseline-rate gate  (channels kept)")
    axes[1].set_title("(b) the absolute rates behind it -- squares = baseline, circles = stim.\n"
                      "A gate that agrees only because it kept the busiest few is not a result.",
                      fontsize=9, loc="left")
    recessive(axes[1]); axes[1].grid(axis="y", alpha=.3)
    for gi, lab in enumerate(labels):
        nn = [s[2] for s in summary if s[0] == lab][0]
        axes[1].annotate(f"n={nn}", (gi, 0.02), xycoords=("data", "axes fraction"),
                         ha="center", fontsize=8, color="0.35")

    print(f"{'gate':<7}{'det':<11}{'n':>5}{'stim/base':>11}{'base/min':>10}{'stim/min':>10}")
    for lab, d, nn, r, rb, rs in summary:
        print(f"{lab:<7}{d:<11}{nn:>5}{r:>11.3f}{rb:>10.2f}{rs:>10.2f}")

    fig.tight_layout()
    out = (Path(outdir) if outdir else figdir("real")) / f"{fname}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")


# --------------------------------------------------------------------------------------
_ALPHAS = (.34, .16, .05)
_LSTYLES = ("-", "--", ":")


def baseline_effect_compare(rec="P1_ANT2_stim",
                            variants=(("_qcfinalv2", "P1_ANT2_pre_qcfinalv2", "median 5 (2.5 ms)"),
                                      ("_med11_qcfinalv2", "P1_ANT2_pre_med11_qcfinalv2",
                                       "median 11 (5.5 ms)")),
                            chans=None, outdir=None, label="", fname="baseline_effect_compare"):
    """baseline_effect_box for two or more preprocessing variants, side by side.

    Each variant is a stim run WITH ITS OWN matching baseline: the median kernel changes detection
    counts substantially (Delphos +70% between no-median and med5), so comparing a med11 stim
    file against a med5 baseline would put the filter change straight into the ratio and read as
    an artefact result.

    Channels are the set measurable in EVERY run, so the boxes summarise the same contacts and a
    difference between them cannot be channel selection.
    """
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, MUTED, recessive

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    L2 = np.log10(2.0)
    loaded, sets = [], []
    for tag, pre, lab in variants:
        zs = np.load(RUNS / f"{rec}{tag}.npz", allow_pickle=False)
        zp = np.load(RUNS / f"{pre}.npz", allow_pickle=False)
        names = np.array([str(x) for x in zs["names"]])
        sm = np.asarray(zs["clean_per_sec"], float).sum(axis=0) / float(zs["fs"]) / 60.0
        bm = np.asarray(zp["clean_per_sec"], float).sum(axis=0) / float(zp["fs"]) / 60.0
        loaded.append((lab, zs, zp, names, sm, bm))
        sets.append(set(names[(sm > 0) & (bm > 0)]))
    common = set.intersection(*sets)
    if chans is not None:
        common &= set(chans)
    common = sorted(common)

    dets = [d for d in ("Janca", "Barkmeier", "Delphos") if f"{d}_idx" in loaded[0][1].files]
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    rng = np.random.default_rng(0)
    rows = []
    for di, d in enumerate(dets):
        for vi, (lab, zs, zp, names, sm, bm) in enumerate(loaded):
            n = len(names)
            idx = np.array([list(names).index(c) for c in common])
            sc = np.bincount(zs[f"{d}_chan"], minlength=n)[idx].astype(float)
            bc = np.bincount(zp[f"{d}_chan"], minlength=n)[idx].astype(float)
            ok = (bc > 0) & (bm[idx] > 0) & (sm[idx] > 0)
            v = np.log10(np.where(sc[ok] > 0, sc[ok], 0.5) / sm[idx][ok]
                         / (bc[ok] / bm[idx][ok])) / L2
            v = v[np.isfinite(v)]
            # slots must fit inside the 1.0 gap between detectors: at the old fixed
            # 0.32 spacing four variants spanned 1.28 and Janca's last box landed on
            # Barkmeier's first
            _step = 0.78 / max(len(loaded), 1)
            x = di + (vi - (len(loaded) - 1) / 2) * _step
            col = colors[d]
            bx = ax.boxplot([v], positions=[x], widths=_step * .82, showfliers=False,
                            patch_artist=True, zorder=2)
            # one style PER VARIANT, not first-vs-rest: with three variants the old
            # solid/dashed pair made the middle box indistinguishable from the last
            bx["boxes"][0].set(facecolor=col, alpha=_ALPHAS[vi % len(_ALPHAS)],
                               edgecolor=col, lw=1.4,
                               ls=_LSTYLES[vi % len(_LSTYLES)])
            for part in ("whiskers", "caps", "medians"):
                for ln in bx[part]:
                    ln.set(color=col, lw=1.7)
            ax.scatter(x + rng.uniform(-_step * .28, _step * .28, v.size), v, s=13, color=col,
                       alpha=(.45, .30, .22)[vi % 3], edgecolor="none", zorder=3)
            ax.annotate(f"{2 ** np.median(v):.2f}x", (x, 0.015),
                        xycoords=("data", "axes fraction"), ha="center", va="bottom",
                        fontsize=8, color=col)
            rows.append((d, lab, int(v.size), 2 ** float(np.median(v)),
                         int(zs[f"{d}_idx"].size)))
    ax.axhline(0, color=MUTED, ls="--", lw=1.2)
    ax.text(.02, .97, "above 0 = MORE spikes during stim", transform=ax.transAxes,
            va="top", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(dets)))
    ax.set_xticklabels(dets, fontsize=10)
    ax.set_ylabel("log2(stim rate / its own stim-free baseline), per channel")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="0.45",
                                     alpha=_ALPHAS[i % len(_ALPHAS)], edgecolor="0.25",
                                     ls=_LSTYLES[i % len(_LSTYLES)], label=lo[0])
                       for i, lo in enumerate(loaded)],
              frameon=False, fontsize=8.5, loc="upper right",   # the per-box "N.NNx"
              # annotations sit along the bottom axis, so lower right collides with them
              title="left to right, per detector", title_fontsize=8)
    ax.set_title(f"{rec}{('  [' + label + ']') if label else ''} -- {len(common)} common "
                 f"channels, each variant against ITS OWN baseline.",
                 fontsize=10, loc="left")
    recessive(ax)
    ax.grid(axis="y", alpha=.3)

    print(f"{'det':<11}{'variant':<22}{'ch':>4}{'stim/base':>11}{'detections':>12}")
    for d, lab, nn, r, nd in rows:
        print(f"{d:<11}{lab:<22}{nn:>4}{r:>11.3f}{nd:>12}")

    fig.tight_layout()
    out = (Path(outdir) if outdir else figdir("real")) / f"{fname}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")


# --------------------------------------------------------------------------------------
def _per_chunk(z, keep, blk=64.0):
    """(counts, minutes) per (chunk, channel) -- the unit a per-channel bootstrap resamples."""
    fs = float(z["fs"])
    cps = np.asarray(z["clean_per_sec"], float)
    n = len(z["names"])
    ki = np.flatnonzero(keep)
    edges = np.arange(0, cps.shape[0] + 1, int(blk))
    out = {}
    for d in [str(x) for x in z["detectors"]]:
        t, c = z[f"{d}_idx"] / fs, z[f"{d}_chan"]
        cc, mm = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (t >= a) & (t < b)
            cc.append(np.bincount(c[m], minlength=n)[ki])
            mm.append(cps[a:b].sum(axis=0)[ki] / fs / 60.0)
        out[d] = (np.array(cc, float), np.array(mm, float))
    return out


def channel_confidence(rec="P1_ANT2_stim", tag="_qcfinalv2", pre_stem="P1_ANT2_pre_qcfinalv2",
                       blk=64.0, n_boot=800, outdir=None, label="", seed=0, chans=None,
                       conf_cut=1.5):
    """Per-channel effect WITH a confidence interval, and what determines the confidence.

    THE STATISTIC
      effect  log2(stim rate / that channel's own stim-free baseline rate)
      CI      resample the ~64 s BLOCKS with replacement, independently on each side, and take
              the 2.5/97.5 percentiles. Width in log2 units IS the confidence measure: narrow
              means trustworthy, wide means the channel lacks the counts to support a ratio.

    WHY BLOCKS RATHER THAN A POISSON INTERVAL
      sqrt(1/n_stim + 1/n_base) is the obvious closed form, but real spike trains here are
      overdispersed -- README finding 7 measures Fano ~3.5 -- so Poisson understates the width
      by roughly sqrt(3.5) ~ 1.9x. Resampling blocks captures that empirically, and uses the
      same resampling unit as every other interval in this project.

    Panel (b) is the point of it: CI width against baseline rate. If those collapse onto one
    line then "low confidence" and "low rate" are the same statement, and a rate gate and a
    confidence gate do the same job -- worth knowing before choosing between them.
    """
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, MUTED, recessive

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    L2 = np.log10(2.0)
    zs = np.load(RUNS / f"{rec}{tag}.npz", allow_pickle=False)
    zp = np.load(RUNS / f"{pre_stem}.npz", allow_pickle=False)
    names = np.array([str(x) for x in zs["names"]])
    if [str(x) for x in zp["names"]] != list(names):
        raise SystemExit("stim and baseline runs have different channel names.")
    keep = (np.asarray(zs["clean_per_sec"], float).sum(axis=0) > 0) & \
           (np.asarray(zp["clean_per_sec"], float).sum(axis=0) > 0)
    if chans is not None:
        want = set(chans)
        keep &= np.array([nm in want for nm in names], bool)
    kn = names[keep]
    S, B = _per_chunk(zs, keep, blk), _per_chunk(zp, keep, blk)
    rng = np.random.default_rng(seed)

    def ratio(ci, mi, cj, mj):
        with np.errstate(divide="ignore", invalid="ignore"):
            ti, tj = mi.sum(axis=0), mj.sum(axis=0)
            rs = ci.sum(axis=0) / np.maximum(ti, 1e-9)
            rb = cj.sum(axis=0) / np.maximum(tj, 1e-9)
            num = np.where(rs > 0, rs, 0.5 / np.maximum(ti, 1e-9))
            return np.log10(num / np.where(rb > 0, rb, np.nan)) / L2

    res = {}
    for d in [x for x in ("Janca", "Barkmeier", "Delphos") if x in S and x in B]:
        sc, sm = S[d]
        bc, bm = B[d]
        pt = ratio(sc, sm, bc, bm)
        draws = np.empty((n_boot, len(kn)))
        for k in range(n_boot):
            i = rng.integers(0, sc.shape[0], sc.shape[0])
            j = rng.integers(0, bc.shape[0], bc.shape[0])
            draws[k] = ratio(sc[i], sm[i], bc[j], bm[j])
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
        res[d] = {"pt": pt, "lo": lo, "hi": hi, "w": hi - lo,
                  "base": bc.sum(axis=0) / np.maximum(bm.sum(axis=0), 1e-9),
                  "names": kn}

    dets = list(res)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))
    ax = axes[0]
    for i, d in enumerate(dets):
        r = res[d]
        ok = np.isfinite(r["pt"]) & np.isfinite(r["w"])
        v, w = r["pt"][ok], r["w"][ok]
        col = colors[d]
        bx = ax.boxplot([v], positions=[i - 0.21], widths=.30, showfliers=False,
                        patch_artist=True, zorder=2)
        bx["boxes"][0].set(facecolor=col, alpha=.14, edgecolor=col, lw=1.3)
        for part in ("whiskers", "caps", "medians"):
            for ln in bx[part]:
                ln.set(color=col, lw=2.0)
        hi95 = np.nanpercentile(w, 95)
        conf = np.clip(1.0 - (w - np.nanmin(w)) / max(hi95 - np.nanmin(w), 1e-9), 0.05, 1.0)
        ax.scatter(i - 0.21 + np.linspace(-.10, .10, v.size), v, s=6 + 46 * conf ** 2,
                   color=col, alpha=0.42, edgecolor="none", zorder=3)
        # The SAME channels restricted to a usable interval, drawn beside the full set so the
        # cost and the effect of the gate are read together. `conf_cut` is the CI WIDTH in log2
        # units, so a smaller number is a stricter gate: 1.5 means the interval spans no more
        # than a factor of 2**1.5 = 2.8.
        keepc = w <= conf_cut
        vc = v[keepc]
        if vc.size:
            bx2 = ax.boxplot([vc], positions=[i + 0.21], widths=.30, showfliers=False,
                             patch_artist=True, zorder=2)
            bx2["boxes"][0].set(facecolor=col, alpha=.40, edgecolor=col, lw=1.4)
            for part in ("whiskers", "caps", "medians"):
                for ln in bx2[part]:
                    ln.set(color=col, lw=2.0)
            ax.scatter(i + 0.21 + np.linspace(-.10, .10, vc.size), vc,
                       s=6 + 46 * conf[keepc] ** 2, color=col, alpha=.7, edgecolor="none",
                       zorder=3)
            ax.annotate(f"all\n{2 ** np.median(v):.2f}x\nn={v.size}", (i - 0.21, 0.012),
                        xycoords=("data", "axes fraction"), ha="center", va="bottom",
                        fontsize=7.5, color=col, alpha=.75)
            ax.annotate(f"CI<={conf_cut:g}\n{2 ** np.median(vc):.2f}x\nn={vc.size}",
                        (i + 0.21, 0.012), xycoords=("data", "axes fraction"), ha="center",
                        va="bottom", fontsize=7.5, color=col, fontweight="bold")
    ax.axhline(0, color=MUTED, ls="--", lw=1.2)
    for xi in range(len(dets) - 1):
        ax.axvline(xi + 0.5, color="0.9", lw=0.8, zorder=0)
    ax.set_xlim(-0.55, len(dets) - 0.45)
    ax.set_xticks(range(len(dets)))
    ax.set_xticklabels(dets, fontsize=10)
    ax.set_ylabel("log2(stim / its own baseline), per channel")
    ax.set_title(f"(a) {rec}{tag} {label} -- {len(kn)} channels.\n"
                 "LEFT of each pair = all channels; RIGHT (solid box) = only those with a "
                 f"CI width <= {conf_cut:g}.\nPoint size = confidence.", fontsize=9,
                 loc="left")
    recessive(ax)
    ax.grid(axis="y", alpha=.3)

    ax = axes[1]
    for d in dets:
        r = res[d]
        ok = np.isfinite(r["w"]) & (r["base"] > 0)
        ax.scatter(r["base"][ok], r["w"][ok], s=16, color=colors[d], alpha=.5,
                   edgecolor="none", label=d)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(conf_cut, color="0.25", ls="--", lw=1.4)
    ax.annotate(f"CI width = {conf_cut:g}  (kept below this)", (0.98, conf_cut),
                xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=8,
                color="0.3")
    ax.set_xlabel("baseline rate (det/min) -- how busy the channel is")
    ax.set_ylabel("95% CI width (log2) -- lower = more confident")
    ax.set_title("(b) is confidence just channel rate?\n"
                 "If these collapse onto one line, a rate gate and a confidence gate are the "
                 "same gate.", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    recessive(ax)
    ax.grid(alpha=.3)

    print(f"{'det':<11}{'ch':>5}{'med CI width':>14}{'all':>8}{'CI<=cut':>9}{'n':>6}"
          f"{'rho(rate,width)':>17}")
    for d in dets:
        r = res[d]
        ok = np.isfinite(r["w"]) & (r["base"] > 0) & np.isfinite(r["pt"])
        rho = np.corrcoef(np.log10(r["base"][ok]), np.log10(r["w"][ok]))[0, 1]
        tight = ok & (r["w"] <= conf_cut)
        print(f"{d:<11}{int(ok.sum()):>5}{np.nanmedian(r['w'][ok]):>14.2f}"
              f"{2 ** np.median(r['pt'][ok]):>8.2f}"
              f"{(2 ** np.median(r['pt'][tight])) if tight.any() else float('nan'):>9.2f}"
              f"{int(tight.sum()):>6}{rho:>17.2f}")

    fig.tight_layout()
    out = (Path(outdir) if outdir else figdir("real")) / "channel_confidence.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")
    return res
