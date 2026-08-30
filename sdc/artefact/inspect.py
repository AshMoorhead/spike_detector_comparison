"""
sdc.artefact.inspect
--------------------
Look at the epochs the artefact mask is least sure about, and decide by eye whether it is right.

    .venv\\Scripts\\python.exe -m sdc.artefact.inspect                       # rank + table
    .venv\\Scripts\\python.exe -m sdc.artefact.inspect --view                # open the viewer
    .venv\\Scripts\\python.exe -m sdc.artefact.inspect P5_stim --k-stim 80

WHY RANKED, NOT SCROLLED
  P1_stim is 1097 s x 226 channels -- roughly 124 000 channel-epochs. Paging through them in
  order is useless, and the middle of a stim block is useless twice over: everyone agrees it is
  contaminated, so looking at it cannot change a threshold. The only epochs that carry
  information are the ones sitting CLOSE TO THE CURRENT THRESHOLD, where a small change in K
  flips the verdict. This module ranks by |log10(feature / threshold)| and shows those first.

  Each row prints the K that would flip it, so "what would I have to believe to call this one
  differently" is readable off the screen rather than inferred.

RELATIVE THRESHOLDS
  `pStim` and `grad` are compared against each channel's OWN baseline, measured on the stim-free
  `_pre` recording -- both scale with signal amplitude, so an absolute threshold cannot transfer
  between patients, amplifiers or implant depths. `dynR` is deliberately NOT relative: it is a
  dropout test against the ADC least-significant-bit, and a dead channel is dead whatever its
  normal amplitude.

WHAT THE VIEWER DRAWS
  The RAW 2 kHz trace -- the signal the mask actually sees. The mask is computed on the raw and
  applied to the median-filtered array, so judging it on the filtered signal would be judging it
  on evidence it never had.
"""
import os
import sys

import numpy as np

from sdc.common.paths import ROOT, RUNS
from sdc.artefact.qc_features import FEAT

# Baseline percentile per channel. THE MEDIAN, and this matters more than it looks.
#
# p95 was used first, reasoning that the threshold should sit outside the channel's ordinary
# range. That conflates two jobs: the BASELINE should be a robust estimate of the channel's
# typical level, and K supplies the headroom. Doing both at once makes the normaliser hostage
# to outliers, and on P1 it broke completely -- the H shaft has a handful of epochs in the
# stim-free file carrying huge 145 Hz power, so its p95 baseline came out at 5e4-8e4 against a
# median of ~0.3. Those channels then had a threshold nothing could exceed and were never
# flagged, at any K, despite being obviously contaminated on the trace.
#
# The median is stable across channels (0.28-0.41 on every channel checked), which is what a
# normaliser has to be.
BASE_PCT = 50.0
K_STIM = 150.0        # pStim multiple of baseline. Production's absolute 50 is ~150x the
                      # baseline median, so this is the like-for-like starting point.
K_GRAD = 8.0          # gradRatio multiple of baseline
N_SHOW = 40


def load(rec, tag=""):
    """Features for a stim recording plus the per-channel baseline from its `_pre` partner.

    `tag` selects a frequency-specific dump. A pStim baseline is only meaningful against the
    frequency it was measured at, so the 2 Hz trial has its own files (`..._2hz.npz`) and its
    own pre recording -- baseline_1, not the baseline the 145 Hz trial uses.
    """
    z = np.load(FEAT / f"{rec}{tag}.npz", allow_pickle=False)
    pre_name = rec.replace("_stim", "_pre")
    zp = np.load(FEAT / f"{pre_name}{tag}.npz", allow_pickle=False)
    if list(zp["names"]) != list(z["names"]):
        raise SystemExit(f"{pre_name} channel names differ from {rec}; baseline cannot align.")
    base = {"pStim": np.percentile(zp["pStimAll"], BASE_PCT, axis=0),
            "grad": np.percentile(zp["feat"][0], BASE_PCT, axis=0)}
    return z, zp, base


def rank(rec, k_stim=K_STIM, k_grad=K_GRAD, n=N_SHOW, on_only=True, tag=""):
    """The channel-epochs closest to the decision boundary, most ambiguous first.

    Distance is measured in log units so that "twice the threshold" and "half the threshold" are
    equally far from it -- these are ratio quantities and a linear distance would rank the busy
    channels as always-ambiguous.
    """
    z, _zp, base = load(rec, tag)
    names = [str(s) for s in z["names"]]
    grad, _pstim_on, _dynr = z["feat"]
    pstim, is_on, t = z["pStimAll"], z["isOn"], z["t"]

    thr_s = k_stim * np.maximum(base["pStim"], 1e-12)      # (chan,)
    thr_g = k_grad * np.maximum(base["grad"], 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_s = np.abs(np.log10(pstim / thr_s[None, :]))
        d_g = np.abs(np.log10(grad / thr_g[None, :]))
    # Whichever rule this epoch is nearer to deciding on is the one worth looking at.
    dist = np.fmin(np.nan_to_num(d_s, nan=np.inf), np.nan_to_num(d_g, nan=np.inf))
    if on_only:
        dist = np.where(is_on[:, None], dist, np.inf)      # stim-ON is where the mask bites

    flat = np.argsort(dist, axis=None)[:n]
    ep, ch = np.unravel_index(flat, dist.shape)
    rows = []
    for e, c in zip(ep, ch):
        # The K that would flip THIS epoch, per rule -- the number that makes the decision
        # legible: "call this artefact and you are choosing K_stim below 91".
        flip_s = pstim[e, c] / max(base["pStim"][c], 1e-12)
        flip_g = grad[e, c] / max(base["grad"][c], 1e-12)
        rows.append(dict(t=float(t[e]), chan=names[c], ci=int(c), ei=int(e),
                         on=bool(is_on[e]),
                         pstim=float(pstim[e, c]), grad=float(grad[e, c]),
                         flip_stim=float(flip_s), flip_grad=float(flip_g),
                         stim_fires=bool(pstim[e, c] > thr_s[c]),
                         grad_fires=bool(grad[e, c] > thr_g[c])))
    return rows


def table(rec, **kw):
    rows = rank(rec, **kw)
    k_stim = kw.get("k_stim", K_STIM)
    k_grad = kw.get("k_grad", K_GRAD)
    print(f"\n=== {rec}: {len(rows)} most ambiguous stim-ON channel-epochs "
          f"(K_stim={k_stim:g}, K_grad={k_grad:g}, baseline = p{BASE_PCT:g} of the pre file)")
    print(f"    {'t (s)':>8}{'channel':>12}{'verdict':>10}"
          f"{'pStim/base':>12}{'grad/base':>11}   flips if K_stim / K_grad crosses")
    for r in rows:
        v = ",".join([n for n, f in (("stim", r["stim_fires"]), ("grad", r["grad_fires"])) if f]) \
            or "clean"
        print(f"    {r['t']:>8.0f}{r['chan']:>12}{v:>10}"
              f"{r['flip_stim']:>12.1f}{r['flip_grad']:>11.1f}"
              f"      {r['flip_stim']:.0f} / {r['flip_grad']:.1f}")
    print(f"\n    Rows are sorted by how near they sit to the boundary, so the top of this list "
          f"is\n    where your judgement changes the outcome. The last column is the K at which "
          f"each\n    epoch flips -- pick a K above it to call the epoch clean, below it to mask "
          f"it.")
    return rows


def _runs(m):
    """(start, stop) index pairs of each contiguous True run in a 1-D bool array."""
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def fill_gaps(bad, max_gap):
    """Mark clean runs of `max_gap` epochs or fewer that are flanked by bad on BOTH sides.

    artefact.py's own version only ever closes a gap of exactly ONE, and it does so with a
    sequential neighbour test -- so `B C C B` survives it: for the first C the right neighbour
    is clean, for the second the left one is. Two-epoch gaps therefore pass through untouched
    today, which is the opposite of what a gap-filling rule is for. This closes runs up to
    `max_gap`, and leaves runs at the very start or end of the recording alone because those
    are flanked on one side only and there is no evidence about the other.
    """
    out = bad.copy()
    for c in range(bad.shape[1]):
        for a, b in _runs(~bad[:, c]):
            if (b - a) <= max_gap and a > 0 and b < bad.shape[0]:
                out[a:b, c] = True
    return out


def mask_at(rec, k_stim=K_STIM, k_grad=K_GRAD, dyn_mult=3.0,
            fill_gap=2, dilate_thr=0.5, grad_abs=None, tag=""):
    """The per-(epoch, channel) verdict at a given K, plus a reason string for each cell.

    Applies the three RULES and then the two POST-RULES that artefact.py applies after them,
    because the real mask includes both and judging the rules alone means judging something
    narrower than what actually reaches the detectors:

      isolated       a clean run of <= `fill_gap` epochs between two bad ones is marked. One or
                     two clean epochs inside a contaminated stretch are far more likely to be a
                     threshold that just missed than a genuine recovery.
      stim_dilation  if more than `dilate_thr` of an ON block is already bad on a channel, the
                     WHOLE block is marked for that channel. The argument is that partial
                     contamination of a stimulation block means the channel was compromised for
                     its duration, whatever the per-epoch numbers say.

    Both are tagged distinctly so the viewer can show what the rules caught and what the
    post-rules added. Set `fill_gap=0, dilate_thr=0` to see the rules alone.
    """
    z, _zp, base = load(rec, tag)
    grad, _p_on, dynr = z["feat"]
    pstim, is_on = z["pStimAll"], z["isOn"]
    thr_s = k_stim * np.maximum(base["pStim"], 1e-12)

    ss = np.zeros(grad.shape, bool)
    ss[is_on] = pstim[is_on] > thr_s[None, :]
    if grad_abs is not None:
        # ABSOLUTE uV/sample, as production has always used. Relative is the wrong shape for
        # this rule's main job: movement artefact is a large gradient in absolute terms, and
        # dividing by a high-amplitude channel's own baseline scales the threshold right past
        # it. pStim is different -- there the quantity of interest genuinely is "how much more
        # than this channel normally shows at the stim frequency".
        lf = grad > float(grad_abs)
    else:
        lf = grad > (k_grad * np.maximum(base["grad"], 1e-12))[None, :]
    low = dynr < (dyn_mult * float(z["lsb"]))

    reason = np.empty(grad.shape, object)
    reason[:] = ""
    for tag, m in (("lf_artefact", lf), ("stim_spec", ss), ("low_dyn", low)):
        for e, c in zip(*np.nonzero(m)):
            reason[e, c] = tag if not reason[e, c] else reason[e, c] + "," + tag
    bad = lf | ss | low

    if fill_gap > 0:
        filled = fill_gaps(bad, fill_gap)
        for e, c in zip(*np.nonzero(filled & ~bad)):
            reason[e, c] = "isolated"
        bad = filled

    if dilate_thr > 0:
        # ON blocks are contiguous runs of isOn -- the same bins detect_stim wrote, recovered
        # from the per-epoch flag rather than re-read, since that is all the dump carries.
        for a, b in _runs(is_on):
            n_in = b - a
            if n_in == 0:
                continue
            frac = bad[a:b].sum(axis=0) / n_in
            for c in np.flatnonzero(frac > dilate_thr):
                new = a + np.flatnonzero(~bad[a:b, c])
                for e in new:
                    reason[e, c] = "stim_dilation"
                bad[a:b, c] = True
    return z, bad, reason


def view(rec="P1_stim", t0=None, duration=40.0, n_chan=18, k_stim=K_STIM, k_grad=K_GRAD,
         chans=None, save=None, interactive=True, med_kernel=5, factor=2,
         fill_gap=2, dilate_thr=0.5, grad_abs=None, tag="", run_tag=None):
    """Draw the RAW trace with the mask shaded, at a chosen K.

    Channels default to a SPREAD across the contamination range in this window -- the most
    flagged, the least, and a sample between -- because a screen of uniformly bad or uniformly
    clean channels shows nothing about where the threshold sits.
    """
    import matplotlib
    # Default to writing a PNG rather than opening a window. A GUI viewer is the right tool
    # when a person is driving it, but it is useless for producing something to look at later,
    # and under tkagg the window also picks up stray clicks (the viewer binds click-to-hide).
    if not interactive:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from seeg import read_edf_header, load_edf_segment, derive_montage, apply_montage
    from seeg.eeg_viewer import view as eeg_view
    from seeg.preprocess import decimate_recording
    from sdc.common.paths import figdir

    z, bad, reason = mask_at(rec, k_stim, k_grad, fill_gap=fill_gap,
                             dilate_thr=dilate_thr, grad_abs=grad_abs, tag=tag)
    names = [str(s) for s in z["names"]]
    t_ep, is_on = z["t"], z["isOn"]

    if t0 is None:                      # default to the first stim-ON block
        t0 = float(t_ep[is_on][0]) - 4.0
    t1 = t0 + duration
    sel_ep = (t_ep >= t0) & (t_ep < t1)
    if not sel_ep.any():
        raise SystemExit(f"no epochs between {t0:.0f} and {t1:.0f}s")

    if chans is None:
        frac = bad[sel_ep].mean(axis=0)          # how flagged each channel is in this window
        order = np.argsort(-frac)
        pick = np.unique(np.concatenate([order[:n_chan // 3],
                                         order[len(order) // 2 - n_chan // 6:
                                               len(order) // 2 + n_chan // 6],
                                         order[-(n_chan // 3):]]))
        chans = [names[i] for i in pick]

    # Resolved through the recordings table, NOT read off a previous run. Two reasons: the
    # MERGED npz records the PREPROCESSED edf (median-filtered, decimated, AR-filled) because
    # that is what Delphos read, and shading the mask over the filtered signal would show the
    # verdict against evidence the rule never saw; and requiring a detector run to have
    # happened first makes it impossible to look at a recording before processing it, which is
    # exactly when looking is most useful.
    from sdc.detect.recordings import edf_path
    raw_edf = str(edf_path(rec)[0])
    hdr = read_edf_header(raw_edf)
    r0, r1 = max(1, int(t0)), int(t1)
    rec_d = load_edf_segment(raw_edf, hdr, r0, r1)
    # derive_montage takes the LABEL LIST, not the header.
    rec_d = apply_montage(rec_d, derive_montage(rec_d["info"]["SelectedSignals"]), verbose=False)
    disp = list(rec_d["info"]["SelectedSignals"])
    keep = [c for c in chans if c in disp]
    idx = [disp.index(c) for c in keep]
    fs = float(rec_d["info"]["SampleRate"])

    sub_raw = {"data": rec_d["data"][:, idx],
               "info": {**rec_d["info"], "SelectedSignals": keep,
                        "NumSelectedSignals": len(keep)}}
    # The PREPROCESSED signal, recomputed here from this same raw segment rather than read from
    # the merged EDF on disk: that file was AR-filled against the PRODUCTION mask, so overlaying
    # it while viewing some other K would put two different masks on one screen. Recomputing
    # gives median + anti-alias + decimate with no fill, which is the honest comparison -- what
    # the filter does to the signal, separate from what the mask does.
    dec = decimate_recording(sub_raw, factor=factor, med_kernel=med_kernel, keep_raw=True)
    fs_d = float(dec["info"]["SampleRate"])
    sub = {"data": dec["data"], "info": dict(dec["info"]),
           "raw": {"data": sub_raw["data"], "fs": fs}}
    ep_i = np.flatnonzero(sel_ep)
    ch_i = [names.index(c) for c in keep]
    # Shading is indexed against recording["data"], which is now the DECIMATED array, so the
    # epoch grid has to be in decimated samples.
    starts = np.round((t_ep[ep_i] - 1.0 - r0) * fs_d).astype(int)   # t is the epoch CENTRE
    qc = {"epoch": {"starts": starts,
                    "epochSamp": int(round(2.0 * fs_d)),
                    "nEpoch": len(ep_i),
                    "isOn": is_on[ep_i],
                    "bad": bad[np.ix_(ep_i, ch_i)],
                    "reason": reason[np.ix_(ep_i, ch_i)]}}

    n_bad = int(qc["epoch"]["bad"].sum())
    _gd = f"grad_abs={grad_abs:g} uV/sample" if grad_abs is not None else f"K_grad={k_grad:g}"
    print(f"[view] {rec}  t={r0}-{r1}s  {len(keep)} channels  K_stim={k_stim:g}  {_gd}")
    print(f"[view] fill_gap={fill_gap} dilate_thr={dilate_thr:g} "
          f"(set both to 0 to see the rules alone)")
    print(f"[view] {n_bad}/{qc['epoch']['bad'].size} channel-epochs flagged "
          f"({n_bad / max(qc['epoch']['bad'].size, 1):.0%}); "
          f"{int(is_on[ep_i].sum())}/{len(ep_i)} epochs are stim-ON")
    print(f"[view] black = preprocessed {fs_d:g} Hz (median {med_kernel}, /{factor}); "
          f"grey = RAW {fs:g} Hz, which is what the mask is computed on")
    print(f"[view] toggle the grey backdrop with the 'raw' checkbox in the left margin")

    # Detections overlaid, one marker set per detector, from a finished run. The viewer takes
    # {label: per-channel index arrays} indexed against recording["data"] -- which here is the
    # DECIMATED array starting at r0 -- so absolute detection samples are shifted and rescaled
    # onto that grid. Without this the mask can be judged but not what the detectors did with it,
    # which is the actual question when looking at a stim block.
    spikes = None
    if run_tag is not None:
        zr = np.load(RUNS / f"{rec}{run_tag}.npz", allow_pickle=False)
        r_names = [str(x) for x in zr["names"]]
        fs_run = float(zr["fs"])
        spikes = {}
        for d in [str(x) for x in zr["detectors"]]:
            ti, ci = zr[f"{d}_idx"] / fs_run, zr[f"{d}_chan"]
            per = []
            for c in keep:
                j = r_names.index(c) if c in r_names else -1
                if j < 0:
                    per.append(np.zeros(0, int))
                    continue
                m = (ci == j) & (ti >= r0) & (ti < r1)
                per.append(np.round((ti[m] - r0) * fs_d).astype(int))
            spikes[d] = per
        print(f"[view] detections from {rec}{run_tag}.npz: " +
              ", ".join(f"{d} {sum(len(x) for x in v)}" for d, v in spikes.items()))

    v = eeg_view(sub, qc, spikes=spikes, t0=0.0, duration=float(r1 - r0), block=False)
    if interactive:
        v.show(block=True)
        return v
    out = save or (figdir("real") / f"mask_view_{rec}_K{k_stim:g}.png")
    v.fig.set_size_inches(16, 9)
    v.fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(v.fig)
    print(f"[saved] {out}")
    return out


def threshold_figure(recs=("P1_stim", "P5_stim"), k_marks=(10, 50, 150, 500, 1500)):
    """Where should K sit? The distribution the threshold has to cut, drawn.

    A threshold is only well-placed if it falls somewhere the distribution is SPARSE -- put it on
    a dense part and a small change in K flips many epochs at once, which is exactly the
    instability the ladder kept showing. The ranked table hinted at this (several epochs pinned
    at pStim/base = 150.0); this shows whether there is a valley to aim for.

    OFF epochs are drawn as the reference. The baseline comes from the stim-free pre file, so an
    OFF epoch in the stim recording should sit near 1 -- and where the ON distribution separates
    from it is the contamination the mask is meant to catch.
    """
    import matplotlib.pyplot as plt
    from seeg._style import recessive
    from sdc.common.paths import figdir

    fig, axes = plt.subplots(2, len(recs), figsize=(7.0 * len(recs), 8.0), squeeze=False)
    for j, rec in enumerate(recs):
        z, _zp, base = load(rec)
        pstim, is_on = z["pStimAll"], z["isOn"]
        ratio = pstim / np.maximum(base["pStim"], 1e-12)[None, :]
        on, off = ratio[is_on].ravel(), ratio[~is_on].ravel()
        on, off = on[np.isfinite(on) & (on > 0)], off[np.isfinite(off) & (off > 0)]

        # SURVIVAL curves, not histograms. Every candidate K sits far out in the tail, where a
        # density plot renders as zero -- the first version of this figure showed two tall modes
        # below ratio 1 and nothing at all in the region the threshold actually occupies.
        # P(ratio > x) puts the tail on the y axis where it can be read.
        ax = axes[0][j]
        xs = np.logspace(-3, 4, 300)
        s_on = np.array([(on > x).mean() for x in xs])
        s_off = np.array([(off > x).mean() for x in xs])
        ax.plot(xs, s_on, lw=2.2, color="#c2691f", label=f"stim-ON ({on.size} ch-epochs)")
        ax.plot(xs, s_off, lw=1.8, color="0.35", label=f"stim-OFF ({off.size})")
        # The gap between them IS the contamination: OFF is what this rule flags when there is
        # no stimulation, so anything above that line at a given K is what stimulation added.
        ax.fill_between(xs, s_off, s_on, where=(s_on > s_off), color="#c2691f", alpha=.18, lw=0)
        for k in k_marks:
            ax.axvline(k, color="0.2", ls="--", lw=1.0)
            ax.annotate(f"K={k:g}", (k, 0.98), xycoords=("data", "axes fraction"),
                        rotation=90, fontsize=7, color="0.3", ha="right", va="top")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.2)
        ax.set_ylabel("P(ratio > x)  -- fraction of channel-epochs above")
        ax.set_xlabel("pStim / that channel's pre-file baseline (log)")
        ax.set_title(f"({'ab'[j]}) {rec} -- shaded = what stimulation ADDED over the OFF rate",
                     fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=8)
        recessive(ax); ax.grid(alpha=.3)

        # How fast does the mask grow as K moves? A steep segment means the threshold is sitting
        # on a dense part of the distribution and the result will be sensitive to it.
        ax = axes[1][j]
        ks = np.logspace(0, 4, 200)
        frac = [(on > k).mean() for k in ks]
        ax.plot(ks, np.array(frac) * 100, lw=2.0, color="#c2691f", label="stim-ON masked")
        ax.plot(ks, [(off > k).mean() * 100 for k in ks], lw=1.6, color="0.35",
                label="stim-OFF masked (false positives)")
        ax.legend(frameon=False, fontsize=8)
        for k in k_marks:
            ax.axvline(k, color="0.2", ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("K (pStim as a multiple of baseline)")
        ax.set_ylabel("% of stim-ON channel-epochs masked")
        ax.set_title(f"({'cd'[j]}) flat stretches are where K is safe to sit",
                     fontsize=9, loc="left")
        recessive(ax); ax.grid(alpha=.3)

    fig.suptitle(f"Choosing K for the relative pStim rule. Baseline = p{BASE_PCT:g} of each "
                 f"channel's band power in the stim-free pre file.\nTop: the ON and OFF "
                 f"distributions K must separate. Bottom: how fast the mask grows with K.",
                 fontsize=10)
    fig.tight_layout()
    out = figdir("real") / "artefact_threshold_choice.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


def feature_choice_figure(recs=(("P1_stim", ""), ("P5_stim", ""), ("P1_2hz_stim", "_2hz")),
                          k_marks=(10, 50, 150, 500, 1500),
                          grad_marks=(60, 100, 150, 200, 400, 700),
                          dyn_marks=(1, 3, 6, 10), chosen=(450, 1000, 3)):
    """All three rules on one page, each drawn as the curve you pick a threshold off.

    WHAT THE SHAPE MEANS
      Each panel is a survival curve: the fraction of channel-epochs a given threshold would
      flag. A SIGMOID is the bad case -- on log-x that is what a single population looks like,
      so there is no natural boundary and any threshold is arbitrary. What you want is a
      STAIRCASE: flat, drop, flat SHELF, drop. The shelf means two separated populations, its
      height is the contaminated fraction, and its width is how much the threshold can move
      without changing the answer.

    THE REFERENCE IS THE PRE FILE, NOT WITHIN-FILE OFF
      Earlier versions compared stim-ON against stim-OFF epochs of the same recording. That
      cannot work for a continuously-stimulated file -- P1's 2 Hz trial has no OFF epochs at
      all -- and the pre recording is the better reference anyway: it contains no stimulation
      by construction, where inter-block OFF may carry carryover. Every row is therefore
      "this recording's stimulated epochs" against "its own paired pre file".

    pStim is RELATIVE (feature / that channel's pre-file median) because the quantity of
    interest genuinely is "more than this channel normally shows at the stim frequency". grad
    is ABSOLUTE in uV/sample: movement artefact is a large gradient in absolute terms, and
    dividing by a high-amplitude channel's own baseline scales the threshold straight past it.
    dynR is absolute in lsb, and its rule fires BELOW threshold, so its panel is a CDF.
    """
    import matplotlib.pyplot as plt
    from seeg._style import recessive
    from sdc.common.paths import figdir

    fig, axes = plt.subplots(len(recs), 3, figsize=(16.5, 4.4 * len(recs)), squeeze=False)
    for i, (rec, tag) in enumerate(recs):
        z, zp, base = load(rec, tag)
        is_on, lsb = z["isOn"], float(z["lsb"])
        grad, _p_on, dynr = z["feat"]
        pgrad, _pp, pdynr = zp["feat"]
        sel = is_on if is_on.any() else np.ones(is_on.size, bool)

        panels = [
            ("pStim  (relative)",
             (z["pStimAll"] / np.maximum(base["pStim"], 1e-12)[None, :])[sel],
             zp["pStimAll"] / np.maximum(base["pStim"], 1e-12)[None, :],
             k_marks, "above", np.logspace(-2, 10, 400), "pStim / channel baseline (K_stim)"),
            ("grad  (ABSOLUTE)", grad[sel], pgrad,
             grad_marks, "above", np.logspace(0, 5, 400), "max|diff|  (uV/sample)"),
            ("dynR  (absolute)", dynr[sel] / lsb, pdynr / lsb,
             dyn_marks, "below", np.linspace(0, 25, 300), "dynamic range / lsb"),
        ]
        for j, (name, mat, ref, marks, side, xs, xlab) in enumerate(panels):
            ax = axes[i][j]
            a, b = mat.ravel(), ref.ravel()
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            f = (lambda v, x: (v > x).mean()) if side == "above" else (lambda v, x: (v < x).mean())
            f_on = np.array([f(a, x) for x in xs])
            f_ref = np.array([f(b, x) for x in xs])
            ax.plot(xs, f_on, lw=2.2, color="#c2691f", label="stimulated")
            ax.plot(xs, f_ref, lw=1.8, color="0.35", label="pre file (no stim)")
            ax.fill_between(xs, f_ref, f_on, where=(f_on > f_ref), color="#c2691f",
                            alpha=.18, lw=0)
            for k in marks:
                ax.axvline(k, color="0.2", ls="--", lw=0.9)
                ax.annotate(f"{k:g}", (k, 0.98), xycoords=("data", "axes fraction"),
                            rotation=90, fontsize=6.5, color="0.35", ha="right", va="top")
            # The chosen operating point, drawn to dominate the candidate marks. It was picked
            # by looking at traces, not off these curves, so the figure's job here is to show
            # what that choice costs rather than to justify it.
            if chosen is not None:
                ch = chosen[j]
                ax.axvline(ch, color="#1b6ca8", lw=2.6, alpha=.85, zorder=5)
                hit = f(a, ch)
                ref_hit = f(b, ch)
                ax.annotate(f"CHOSEN {ch:g}\n{hit:.0%} stim / {ref_hit:.0%} pre",
                            (ch, 0.60), xycoords=("data", "axes fraction"), fontsize=7.5,
                            color="#1b6ca8", ha="left", va="center", fontweight="bold",
                            xytext=(6, 0), textcoords="offset points")
            if side == "above":
                ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel(xlab)
            if j == 0:
                ax.set_ylabel(f"{rec}\nfraction flagged")
            ax.set_title(f"({'abcdefghi'[i * 3 + j]}) {name}", fontsize=9, loc="left")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)
            recessive(ax); ax.grid(alpha=.3)

    fig.suptitle("Threshold selection for all three rules, against each recording's own "
                 "stim-free pre file.\nShaded = what stimulation ADDED. A flat SHELF means two "
                 "separated populations and a safe threshold; a smooth sigmoid means one "
                 "population and an arbitrary one.\n"
                 f"Blue = the chosen operating point: K_stim={chosen[0]:g}, "
                 f"grad_abs={chosen[1]:g} uV/sample, dynFloorMult={chosen[2]:g}.",
                 fontsize=10)
    fig.tight_layout()
    out = figdir("real") / "artefact_feature_choice.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


def view_run(rec="P1_ANT2_stim", run_tag="_pb5_qcfinalv2", t0=None, duration=30.0,
             n_chan=14, chans=None, pulse_blank_ms=None, gate_peak_uv=1e4,
             gate_rail_frac=0.05, med_kernel=5, factor=2, interactive=True, save=None):
    """View EXACTLY what a finished run fed its detectors: blanked trace, its own mask, its
    own detections.

    `view()` above answers a different question -- it recomputes a mask at a chosen K so a
    THRESHOLD can be judged, and it draws the raw signal because that is what the rules saw.
    Neither is right for checking a finished run. Two things it gets wrong there:

      * it decimates the RAW segment, so a run that pulse-blanked before decimating is drawn
        UNBLANKED -- the one thing you are trying to look at is absent;
      * it shades a mask rebuilt from qc_features/<rec>_f<hz>.npz, which is dumped per
        RECORDING with no variant suffix, so it is the control's mask however the run was
        configured.

    Here the mask comes from the run's own `clean_per_sec` (0 clean samples in a second = that
    second was masked), which is the mask that actually reached the detectors rather than a
    reconstruction of it, and the trace is blanked the same way the run blanked it.

    THE GREY BACKDROP STAYS UNBLANKED ON PURPOSE. Black is what the detectors saw; grey is what
    was really there. Blanking both would hide the very thing being checked -- whether the blank
    landed on the artefact and whether anything survived it.
    """
    import matplotlib
    if not interactive:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from seeg import read_edf_header, load_edf_segment, derive_montage, apply_montage
    from seeg.artefact import detect_pulses, blank_pulses, pulse_channel_gate
    from seeg.eeg_viewer import view as eeg_view
    from seeg.preprocess import decimate_recording
    from seeg.stim import get_stim_channel, _stim_column
    from sdc.common.paths import figdir
    from sdc.detect.recordings import edf_path, load_patient_montage

    z = np.load(RUNS / f"{rec}{run_tag}.npz", allow_pickle=False)
    names = [str(s) for s in z["names"]]
    fs_run = float(z["fs"])
    cps = np.asarray(z["clean_per_sec"], float)
    if t0 is None:
        on = np.asarray(z["on_runs"], float) / fs_run if "on_runs" in z.files else None
        t0 = float(on[0][0]) if on is not None and on.size else 0.0
    r0, r1 = max(1, int(t0)), int(t0 + duration)

    if chans is None:                      # spread over how masked each channel is here
        frac = 1.0 - cps[r0:r1].mean(axis=0) / fs_run
        order = np.argsort(-frac)
        pick = np.unique(np.concatenate([order[:n_chan // 2], order[-(n_chan - n_chan // 2):]]))
        chans = [names[i] for i in pick]

    edf, trial = edf_path(rec)
    hdr = read_edf_header(str(edf))
    rd = load_edf_segment(str(edf), hdr, r0, r1)
    # The CLINICAL montage, matching the run. This used derive_montage unconditionally, which
    # pairs every consecutive contact and keeps all 226 rather than the clinician's 164 -- so a
    # channel named in the run could be absent here, and `keep` below drops silently whatever it
    # cannot find. Same defect that invalidated every run in this project once already.
    mont = load_patient_montage(rec.split("_")[0], allow_derived=True)
    rd = apply_montage(rd, mont if mont is not None else
                       derive_montage(rd["info"]["SelectedSignals"]), verbose=False)
    fs = float(rd["info"]["SampleRate"])
    orig = np.array(rd["data"], float, copy=True)      # the grey backdrop, never blanked

    if pulse_blank_ms is None:
        pulse_blank_ms = float(z["pulse_blank_ms"]) if "pulse_blank_ms" in z.files else 0.0
    # The FILL comes from the run too, not a default. This was hardcoded to "auto", which fills
    # any run over 4 samples with AR-synthesised noise -- so a 15 ms blank at 2 kHz (30 samples)
    # was drawn entirely AR-filled while the run itself used interp. AR fill invents plausible
    # segments at amplitudes unrelated to the neighbours, which reads as square waves in the
    # trace, and it is the fill that sent Delphos from 1.88 to 10.69 in the pulse-blanking work.
    # The picture was of a run that was never made.
    pulse_fill = str(z["pulse_fill"]) if "pulse_fill" in z.files else "interp"
    pinfo = None
    if pulse_blank_ms and trial is not None:
        rd["info"]["stim_trial"] = trial
        pm, pinfo = detect_pulses(_stim_column(rd, get_stim_channel(rd, verbose=False)), fs,
                                  stim_hz=float(trial["stim_frequency"]))
        elig, _ = pulse_channel_gate(rd, pm, max_peak_uv=gate_peak_uv,
                                     max_rail_frac=gate_rail_frac)
        rd, binfo = blank_pulses(rd, pm, method=pulse_fill, blank_ms=pulse_blank_ms,
                                 channels=elig)
        print(f"[view_run] blanked {pulse_blank_ms:g} ms, fill={pulse_fill!r}, on "
              f"{binfo['n_channels_blanked']}/{len(elig)} channels "
              f"({pinfo['n_pulses']} pulses in this segment)")

    disp = list(rd["info"]["SelectedSignals"])
    keep = [c for c in chans if c in disp]
    # SAY SO when a requested channel is not drawn. Dropping it silently means asking for six
    # channels, being shown four, and reading the four as the answer.
    if len(keep) < len(chans):
        print(f"[view_run] NOT DRAWN (absent from the montage): "
              f"{', '.join(c for c in chans if c not in disp)}")
    if not keep:
        raise SystemExit(f"None of {chans} exist in this montage. Available e.g.: {disp[:8]}")
    idx = [disp.index(c) for c in keep]
    sub_raw = {"data": rd["data"][:, idx],
               "info": {**rd["info"], "SelectedSignals": keep,
                        "NumSelectedSignals": len(keep)}}
    dec = decimate_recording(sub_raw, factor=factor, med_kernel=med_kernel, keep_raw=False)
    fs_d = float(dec["info"]["SampleRate"])
    sub = {"data": dec["data"], "info": dict(dec["info"]),
           "raw": {"data": orig[:, idx], "fs": fs}}

    # mask straight from the run: a second with no clean samples was masked for that channel
    ch_i = [names.index(c) for c in keep]
    # load_edf_segment takes 1-BASED record numbers, so records r0..r1 begin at wall-clock
    # second r0-1. Everything drawn on this segment must be offset by that, not by r0 --
    # `view()` above does it with an explicit `- 1.0` and this function did not, so both the
    # mask and every detection were drawn ONE SECOND EARLY against the trace.
    t_off = r0 - 1
    secs = np.arange(r0, r1)
    bad = cps[np.ix_(secs, ch_i)] < 0.5 * fs_run
    qc = {"epoch": {"starts": np.round((secs - t_off) * fs_d).astype(int),
                    "epochSamp": int(round(fs_d)), "nEpoch": len(secs),
                    "isOn": np.ones(len(secs), bool),
                    "bad": bad,
                    "reason": np.where(bad, "run mask", "")}}

    spikes = {}
    for d in [str(x) for x in z["detectors"]]:
        ti, ci = z[f"{d}_idx"] / fs_run, z[f"{d}_chan"]
        per = []
        for c in keep:
            j = names.index(c)
            m = (ci == j) & (ti >= t_off) & (ti < r1)
            per.append(np.round((ti[m] - t_off) * fs_d).astype(int))
        spikes[d] = per

    print(f"[view_run] {rec}{run_tag}  t={r0}-{r1}s  {len(keep)} channels")
    print(f"[view_run] mask: {bad.mean():.1%} of channel-seconds here")
    print(f"[view_run] detections: " +
          ", ".join(f"{d} {sum(len(x) for x in v)}" for d, v in spikes.items()))
    print(f"[view_run] black = blanked+median{med_kernel}+/{factor:g} at {fs_d:g} Hz "
          f"(what the detectors saw); grey = UNBLANKED raw at {fs:g} Hz")

    v = eeg_view(sub, qc, spikes=spikes, t0=0.0, duration=float(r1 - r0), block=False)
    if interactive:
        v.show(block=True)
        return v
    out = save or (figdir("real") / f"view_run_{rec}{run_tag}_{r0}s.png")
    v.fig.set_size_inches(18, 10)
    v.fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(v.fig)
    print(f"[saved] {out}")
    return out
