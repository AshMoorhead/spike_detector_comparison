"""
sdc.scoring.labelled_effect
---------------------------
The stimulation effect measured from the RATER'S MARKS, drawn the same two ways the detector
figures draw it.

    .venv\\Scripts\\python.exe -m sdc.scoring.labelled_effect

WHY THIS EXISTS
  Everywhere else the ON/baseline ratio is a DETECTOR's answer, and the artefact work is an
  argument about which detector to believe. P1 now has both conditions marked exhaustively on
  the SAME 22 channels -- `sub-P1_task-baseline` win-000/001 and `sub-P1_task-ANT145Hz`
  win-000/001, all four `done` -- so for the first time the ratio can be computed from the
  marks themselves. That is the number the detectors are trying to recover, so it is drawn
  here as a fourth series beside them rather than quoted as a scalar somewhere else.

  Two figures, matching `sdc.artefact.exposure`:
    pooled_vs_pairwise_baseline.png   does the answer depend on WHEN blocks are combined?
    period_vs_baseline.png            the same recording binned in time, baseline drawn beside it.

THE FULL RUN, NOT `_t970`
  The example figures were built on `P1_stim_t970`, which stops at 964 s. The second marked
  window runs 737-1057 s, so `_t970` throws away the 970-1034 ON block entirely and clips
  another. On the full run three ON blocks are 100% marked and a fourth is 86%; on `_t970`
  only two survive. Nothing about the marks favours the truncation, so it is not used.

THE RATER IS HELD TO THE DETECTORS' TIME BASE
  Detections in the runs are stored POST-mask (`{d}_idx` sits entirely inside clean seconds;
  `{d}_idx_masked` holds what QC removed), and their rates use `clean_per_sec` as the
  denominator. The rater, though, scanned the RAW data and marked spikes in stretches QC later
  condemned. Counting those marks against a clean-seconds denominator would inflate the rater's
  rate purely because the exposures differ.

  So marks falling in masked seconds are DROPPED and the rater's denominator is clean seconds
  too -- identical channels, identical seconds, identical blocks as the detectors, so a
  difference between the rater and a detector is the detector being wrong rather than the two
  measuring different recordings. The raw-time rater rates are printed alongside as a check
  that the gating is not itself moving the answer.

WHAT THE MARKS CANNOT DO
  The rater covered 640 s of the stim recording in two windows, not all 1097 s. Bins and blocks
  outside those windows have no rater value and are left empty rather than interpolated; a block
  is used only if `MIN_MARKED_FRAC` of it was actually marked. The detector series in these
  figures are restricted to exactly the same windows, channels and blocks, so every series on
  the page describes the same slice of recording.
"""
import collections
import csv
import json
from pathlib import Path

import numpy as np

from sdc.common.paths import RUNS, figdir
from sdc.artefact.ratio_metrics import _runs_from_mask

LABELS = Path(r"C:\Users\amoo0039\Documents\label-SEEG-data")
RATER = "rater-AM"
RKEY = "Rater-AM"
DETS = ("Janca", "Barkmeier", "Delphos")

STIM_STEM = "P1_stim_qcfinalv2"
PRE_STEM = "P1_pre_qcfinalv2"
STIM_BLOCKS = ("sub-P1_task-ANT145Hz_grp-sample_win-000",
               "sub-P1_task-ANT145Hz_grp-sample_win-001")
PRE_BLOCKS = ("sub-P1_task-baseline_grp-sample_win-000",
              "sub-P1_task-baseline_grp-sample_win-001")

MIN_MARKED_FRAC = 0.5   # a block/bin needs this much marked coverage to contribute
MIN_BIN_FRAC = 0.5      # ...and must itself be at least this fraction of BIN_SEC long.
                        # P1's ON/OFF mask opens with a 6 s OFF sliver before the first ON
                        # block; split as its own bin it is a 6 s estimate of a ~1/min rate,
                        # which lands near zero and drags the whole normalised trace down.
BIN_SEC = 60.0
AGG = "mean"            # geometric mean across channels, as sdc.artefact.blocks.AGG

COLORS = {RKEY: "#1b7f3b", "Janca": "#c8102e", "Barkmeier": "#0072b2", "Delphos": "#4a3aa7"}
SERIES = (RKEY,) + DETS


# ---------------------------------------------------------------------------------------
# marks + blocks
# ---------------------------------------------------------------------------------------
def _blocks():
    d = json.loads((LABELS / "blocks.json").read_text(encoding="utf-8"))["blocks"]
    return d if isinstance(d, dict) else {b["block_id"]: b for b in d}


def _done(rater=RATER):
    p = LABELS / "annotations" / rater / "progress.json"
    return {k for k, v in json.loads(p.read_text(encoding="utf-8"))["blocks"].items()
            if v.get("status") == "done"}


def _marks(rater=RATER):
    """{block_id: {channel: sorted absolute times}}. `time_in_sec` is recording time on the
    same axis as the run npz -- not an offset from the block start."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(LABELS / "annotations" / rater / "marks.tsv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[r["block_id"]][r["channel"]].append(float(r["time_in_sec"]))
    return {b: {c: np.sort(v) for c, v in d.items()} for b, d in out.items()}


def _windows(block_ids, blocks):
    """[(t0, t1)] for the named blocks, merged where they abut."""
    w = sorted((float(blocks[b]["t_start"]),
                float(blocks[b]["t_start"]) + float(blocks[b]["t_dur"])) for b in block_ids)
    merged = [list(w[0])]
    for a, b in w[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(x) for x in merged]


def _channels(block_ids, blocks):
    sets = [[c["name"] for c in blocks[b]["channels"]] for b in block_ids]
    if any(s != sets[0] for s in sets[1:]):
        raise SystemExit("blocks do not share one channel list")
    return sets[0]


def _overlap(a, b, windows):
    return sum(max(0.0, min(b, w1) - max(a, w0)) for w0, w1 in windows)


# ---------------------------------------------------------------------------------------
# one recording, reduced to (times, channel) per series
# ---------------------------------------------------------------------------------------
class Rec:
    """A run plus the rater's marks on it, both reduced to the marked windows, the 22 marked
    channels, and the seconds QC kept."""

    def __init__(self, stem, block_ids, chans, blocks, marks):
        self.stem = stem
        self.z = z = np.load(RUNS / f"{stem}.npz", allow_pickle=False)
        self.fs = fs = float(z["fs"])
        self.cps = np.asarray(z["clean_per_sec"], float)
        self.nsec = self.cps.shape[0]
        self.windows = _windows(block_ids, blocks)
        self.chans = chans

        names = [str(x) for x in z["names"]]
        missing = [c for c in chans if c not in names]
        if missing:
            raise SystemExit(f"{stem}: marked channels absent from run: {missing}")
        self.idx = np.array([names.index(c) for c in chans], int)   # run column per marked chan
        self.cps = self.cps[:, self.idx]                            # [nsec, 22], QC-kept samples

        on = (np.asarray(z["on_per_sec"], bool) if "on_per_sec" in z.files
              else np.zeros(self.nsec, bool))
        self.on = on[:self.nsec]

        self.times, self.raw_times = {}, {}
        for d in DETS:
            if d not in [str(x) for x in z["detectors"]]:
                continue
            t = z[f"{d}_idx"] / fs
            c = z[f"{d}_chan"]
            pos = {v: k for k, v in enumerate(self.idx)}
            keep = np.array([x in pos for x in c], bool)
            local = np.array([pos[x] for x in c[keep]], int) if keep.any() else np.zeros(0, int)
            self.times[d] = self._clip(t[keep], local)

        mk_t, mk_c = [], []
        for b in block_ids:
            for ch, ts in marks.get(b, {}).items():
                if ch not in chans:
                    continue
                mk_t.append(ts)
                mk_c.append(np.full(ts.size, chans.index(ch), int))
        mt = np.concatenate(mk_t) if mk_t else np.zeros(0)
        mc = np.concatenate(mk_c) if mk_c else np.zeros(0, int)
        self.raw_times[RKEY] = self._clip(mt, mc, clean_gate=False)
        self.times[RKEY] = self._clip(mt, mc)

    def _clip(self, t, c, clean_gate=True):
        """Drop events outside the marked windows and (by default) outside QC-kept seconds."""
        if t.size == 0:
            return np.zeros(0), np.zeros(0, int)
        keep = np.zeros(t.size, bool)
        for w0, w1 in self.windows:
            keep |= (t >= w0) & (t < w1)
        keep &= (t >= 0) & (t < self.nsec)
        if clean_gate:
            s = np.clip(t.astype(int), 0, self.nsec - 1)
            keep &= self.cps[s, c] > 0
        return t[keep], c[keep]

    def exposure(self, a, b):
        """Clean channel-MINUTES per marked channel over [a, b), marked windows only."""
        lo, hi = int(np.floor(a)), int(np.ceil(b))
        lo, hi = max(0, lo), min(self.nsec, hi)
        if hi <= lo:
            return np.zeros(len(self.chans))
        sec = np.arange(lo, hi)
        inw = np.zeros(sec.size, bool)
        for w0, w1 in self.windows:
            inw |= (sec >= np.floor(w0)) & (sec < np.ceil(w1))
        if not inw.any():
            return np.zeros(len(self.chans))
        return self.cps[sec[inw]].sum(axis=0) / self.fs / 60.0

    def counts(self, a, b, series, raw=False):
        t, c = (self.raw_times if raw else self.times).get(series, (np.zeros(0), np.zeros(0, int)))
        sel = (t >= a) & (t < b)
        return np.bincount(c[sel], minlength=len(self.chans)).astype(float)

    def marked_sec(self, a, b):
        return _overlap(a, b, self.windows)

    def blocks_on(self, want_on=True):
        """ON (or OFF) runs that are marked enough to use: [(a, b, marked_sec)]."""
        out = []
        for a, b in _runs_from_mask(self.on if want_on else ~self.on):
            m = self.marked_sec(a, b)
            if b > a and m / (b - a) >= MIN_MARKED_FRAC:
                out.append((float(a), float(b), m))
        return out


def load():
    blocks, marks, done = _blocks(), _marks(), _done()
    for b in STIM_BLOCKS + PRE_BLOCKS:
        if b not in done:
            raise SystemExit(f"block {b} is not marked `done`")
    chans = _channels(STIM_BLOCKS, blocks)
    if _channels(PRE_BLOCKS, blocks) != chans:
        raise SystemExit("baseline and stim blocks were marked on different channels")
    stim = Rec(STIM_STEM, STIM_BLOCKS, chans, blocks, marks)
    pre = Rec(PRE_STEM, PRE_BLOCKS, chans, blocks, marks)
    if [str(x) for x in pre.z["names"]] != [str(x) for x in stim.z["names"]]:
        raise SystemExit("stim and baseline runs have different channel names")
    return stim, pre, chans


def _geo(logs):
    return 10 ** (np.mean(logs) if AGG == "mean" else np.median(logs))


# ---------------------------------------------------------------------------------------
# figure 1 -- pooled vs per-block, ON / stim-free baseline
# ---------------------------------------------------------------------------------------
def pooled_vs_pairwise_baseline(outdir=None, scale="log"):
    # log, not linear: these are ratios, the estimator already combines them in log10, and one
    # Delphos block at 2.14 against a rater at 0.22 spans an order of magnitude -- on a linear
    # axis that single point flattens every box worth reading.
    import matplotlib.pyplot as plt
    from seeg._style import recessive

    stim, pre, chans = load()
    on_blocks = stim.blocks_on(True)
    a0, b0 = pre.windows[0][0], pre.windows[-1][1]
    base_t = pre.exposure(a0, b0)

    summary, raw_note = [], {}
    for s in SERIES:
        if s not in stim.times:
            continue
        base_c = pre.counts(a0, b0, s)
        ok = (base_t > 0) & (base_c > 0)
        if not ok.any():
            continue
        base_r = base_c[ok] / base_t[ok]

        oc = np.array([stim.counts(a, b, s)[ok] for a, b, _ in on_blocks])
        ot = np.array([stim.exposure(a, b)[ok] for a, b, _ in on_blocks])

        pc, pt = oc.sum(axis=0), ot.sum(axis=0)
        good = pt > 0
        pooled = _geo(np.log10(np.where(pc[good] > 0, pc[good], 0.5) / pt[good] / base_r[good]))

        per = []
        for k in range(oc.shape[0]):
            g = ot[k] > 0
            if not g.any():
                continue
            per.append(_geo(np.log10(np.where(oc[k][g] > 0, oc[k][g], 0.5)
                                     / ot[k][g] / base_r[g])))
        per = np.array(per, float)
        comb = 10 ** np.median(np.log10(per)) if per.size else np.nan
        summary.append((s, pooled, comb, per))

        if s == RKEY:   # the same estimate on RAW marked time, as a gating check
            bt = np.array([pre.marked_sec(a0, b0) / 60.0] * len(chans))[ok]
            br = pre.counts(a0, b0, s, raw=True)[ok] / bt
            rc = np.array([stim.counts(a, b, s, raw=True)[ok] for a, b, _ in on_blocks]).sum(0)
            rt = np.array([stim.marked_sec(a, b) / 60.0 for a, b, _ in on_blocks]).sum()
            g = br > 0
            raw_note["pooled"] = _geo(np.log10(np.where(rc[g] > 0, rc[g], 0.5) / rt / br[g]))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, (s, pooled, comb, per) in enumerate(summary):
        if per.size:
            bx = ax.boxplot([per], positions=[i], widths=0.5, showfliers=False,
                            patch_artist=True, zorder=2)
            bx["boxes"][0].set(facecolor=COLORS[s], alpha=.20, edgecolor=COLORS[s])
            for part in ("whiskers", "caps", "medians"):
                for ln in bx[part]:
                    ln.set(color=COLORS[s], lw=1.6)
            ax.scatter(i + rng.uniform(-.12, .12, per.size), per, s=42, color=COLORS[s],
                       alpha=.6, edgecolor="none", zorder=3)
        ax.scatter([i], [pooled], marker="D", s=95, facecolor="white", edgecolor=COLORS[s],
                   lw=2.2, zorder=4)
    ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
    ax.set_yscale(scale)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels([s for s, *_ in summary], fontsize=9)
    ax.set_ylabel(f"ON block / stim-free baseline ({scale})")
    ax.set_title(f"(a) P1 ANT 145 Hz on the {len(chans)} MARKED channels, "
                 f"{len(on_blocks)} marked ON blocks\n"
                 f"box + points = per-block; white diamond = pooled", fontsize=9, loc="left")
    recessive(ax)
    ax.grid(axis="y", alpha=.3)

    ax = axes[1]
    for s, pooled, comb, per in summary:
        ax.scatter([pooled], [comb], s=110, color=COLORS[s], label=s, zorder=3)
    vals = [v for _, p, c, _ in summary for v in (p, c) if np.isfinite(v)]
    lim = [min(0.1, min(vals)) * 0.9, max(1.2, max(vals)) * 1.1]
    ax.plot(lim, lim, color="0.35", ls="--", lw=1.2)
    ax.axhline(1.0, color="0.75", ls=":", lw=1.0)
    ax.axvline(1.0, color="0.75", ls=":", lw=1.0)
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("pooled over ON blocks")
    ax.set_ylabel("per-block, then combined")
    ax.set_title("(b) the two estimators against each other", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    recessive(ax)
    ax.grid(alpha=.3)

    print(f"\nP1 ANT145 vs stim-free baseline -- {len(chans)} marked channels, "
          f"{len(on_blocks)} ON blocks")
    for a, b, m in on_blocks:
        print(f"    ON {a:6.0f}-{b:6.0f}s   marked {m:5.0f}s of {b - a:4.0f}s"
              f"  ({m / (b - a) * 100:5.1f}%)")
    print(f"  {'series':<12}{'pooled':>9}{'per-block':>11}{'ratio':>8}   per-block values")
    for s, pooled, comb, per in summary:
        print(f"  {s:<12}{pooled:>9.3f}{comb:>11.3f}{comb / pooled:>8.2f}   "
              + " ".join(f"{x:.2f}" for x in per))
    if raw_note:
        print(f"  [check] rater on RAW marked time (no QC gate), pooled = "
              f"{raw_note['pooled']:.3f}")

    fig.suptitle("Pooled vs per-block for the ON / stim-free-BASELINE ratio, computed from the "
                 "RATER'S MARKS and from each detector\n"
                 "on the same 22 channels, same marked windows, same QC-kept seconds. "
                 "Green is the truth the detectors are trying to recover.", fontsize=10)
    fig.tight_layout()
    out = (Path(outdir) if outdir else figdir("labelled", "P1_ANT145")) / \
        "pooled_vs_pairwise_baseline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")
    return summary


# ---------------------------------------------------------------------------------------
# figure 2 -- the recording in block-aligned bins, baseline drawn beside it
# ---------------------------------------------------------------------------------------
def _binned(rec, bin_sec=BIN_SEC, respect_on=True):
    """Block-aligned bins, marked windows only.
    Returns (edges, mids, is_on, {series: rates/min})."""
    edges, flags = [], []
    if respect_on:
        runs = [(a, b, True) for a, b in _runs_from_mask(rec.on)] + \
               [(a, b, False) for a, b in _runs_from_mask(~rec.on)]
    else:
        runs = [(w0, w1, False) for w0, w1 in rec.windows]
    for a, b, f in runs:
        k = max(1, int(round((b - a) / bin_sec)))
        cut = np.linspace(a, b, k + 1)
        for i in range(k):
            lo, hi = cut[i], cut[i + 1]
            if (hi - lo) < MIN_BIN_FRAC * bin_sec:
                continue
            if rec.marked_sec(lo, hi) / (hi - lo) >= MIN_MARKED_FRAC:
                edges.append((lo, hi))
                flags.append(f)
    order = np.argsort([e[0] for e in edges])
    edges = [edges[i] for i in order]
    flags = np.array([flags[i] for i in order], bool)
    mids = np.array([(a + b) / 2 for a, b in edges], float)

    out = {}
    for s in SERIES:
        if s not in rec.times:
            continue
        mean, med = [], []
        for a, b in edges:
            cnt = rec.counts(a, b, s)
            t = rec.exposure(a, b)
            ok = t > 0
            r = cnt[ok] / t[ok] if ok.any() else np.array([np.nan])
            mean.append(np.mean(r))
            med.append(np.median(r))
        out[s] = {"mean": np.array(mean, float), "median": np.array(med, float)}
    return edges, mids, flags, out


def _gapped(edges, mids, y, tol=1.0):
    """x, y with a NaN inserted wherever consecutive bins are separated by unmarked time, so
    the connecting line BREAKS instead of drawing straight across a stretch nobody looked at."""
    xs, ys = [], []
    for i in range(len(mids)):
        if i and edges[i][0] - edges[i - 1][1] > tol:
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(mids[i])
        ys.append(y[i])
    return np.array(xs, float), np.array(ys, float)


def period_vs_baseline(outdir=None, bin_sec=BIN_SEC):
    import matplotlib.pyplot as plt
    from seeg._style import recessive

    stim, pre, chans = load()
    edges, mids, ison, course = _binned(stim, bin_sec, respect_on=True)
    pre_edges, pre_mids, _, pre_course = _binned(pre, bin_sec, respect_on=False)
    pre_mids = pre_mids - pre.windows[0][0]          # baseline own axis, starting at 0
    norm = {s: float(np.nanmedian(v["mean"])) for s, v in pre_course.items()}

    fig, ax = plt.subplots(figsize=(14.0, 5.4))
    off = (pre_mids[-1] + bin_sec / 2) / 60.0
    ax.axvspan(-off - .3, 0, color="0.93", lw=0, zorder=0)
    ax.text(-off / 2, 0.02, "stim-free baseline", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=8, color="0.35")
    ax.axvline(0, color="0.4", lw=1.4)
    for t, o in zip(mids, ison):
        if o:
            ax.axvspan((t - bin_sec / 2) / 60.0, (t + bin_sec / 2) / 60.0,
                       color="#f0c419", alpha=.20, lw=0, zorder=0)
    # the unmarked stretches, so the gaps in every series are visibly a coverage limit
    prev = 0.0
    for w0, w1 in stim.windows + [(stim.nsec, stim.nsec)]:
        if w0 > prev:
            ax.axvspan(prev / 60.0, w0 / 60.0, facecolor="0.85", alpha=.55, lw=0, zorder=0,
                       hatch="///", edgecolor="0.7")
        prev = w1

    for s in SERIES:
        if s not in course or not np.isfinite(norm.get(s, np.nan)) or norm[s] <= 0:
            continue
        yb = pre_course[s]["mean"] / norm[s]
        ax.plot(pre_mids / 60.0 - off, yb, "-", lw=1.2, color=COLORS[s], alpha=.45)
        ax.scatter(pre_mids / 60.0 - off, yb, s=30, facecolor="white", edgecolor=COLORS[s],
                   lw=1.3, zorder=3)
        y = course[s]["mean"] / norm[s]
        lw = 2.1 if s == RKEY else 1.3
        gx, gy = _gapped(edges, mids, y)
        ax.plot(gx / 60.0, gy, "-", lw=lw, color=COLORS[s], alpha=.8, zorder=2)
        _, gm = _gapped(edges, mids, course[s]["median"] / norm[s])
        ax.plot(gx / 60.0, gm, ":", lw=1.6, color=COLORS[s], alpha=.85, zorder=2)
        ax.scatter(mids[ison] / 60.0, y[ison], s=72 if s == RKEY else 62, color=COLORS[s],
                   zorder=4, label=f"{s} stim ON")
        ax.scatter(mids[~ison] / 60.0, y[~ison], s=44, facecolor="white", zorder=4,
                   edgecolor=COLORS[s], lw=1.5, label=f"{s} OFF")

    ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
    ax.set_xlabel("time (min)   |   0 = start of the stim recording   |   "
                  "hatched = not marked")
    ax.set_ylabel(f"rate / baseline median\n({bin_sec:g}s bins, block-aligned)")
    ax.set_title(f"P1 ANT 145 Hz, {len(chans)} marked channels: rater's marks (green, thick) "
                 f"against the three detectors", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=4)
    recessive(ax)
    ax.grid(alpha=.3)

    print(f"\nblock-aligned {bin_sec:g}s bins, marked coverage >= {MIN_MARKED_FRAC:.0%}"
          f"   {int(ison.sum())} ON bins, {int((~ison).sum())} OFF bins, "
          f"{len(pre_mids)} baseline bins")
    print(f"  {'series':<12}{'ON median':>11}{'OFF median':>12}{'ON/OFF':>9}")
    for s in SERIES:
        if s not in course:
            continue
        y = course[s]["mean"] / norm[s]
        on_m, off_m = np.nanmedian(y[ison]), np.nanmedian(y[~ison])
        print(f"  {s:<12}{on_m:>11.3f}{off_m:>12.3f}{on_m / off_m:>9.3f}")

    fig.suptitle("Every ~1 min bin, aligned to the ON/OFF transitions so no bin straddles one, "
                 "restricted to the marked windows.\n"
                 "Filled = stim ON, open = OFF, open on grey = the stim-free baseline "
                 "recording. Dotted = median across channels.", fontsize=10)
    fig.tight_layout()
    out = (Path(outdir) if outdir else figdir("labelled", "P1_ANT145")) / \
        "period_vs_baseline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")
    return mids, ison, course


# ---------------------------------------------------------------------------------------
def _selfcheck():
    """Cheap invariants -- these are the ways this module could be quietly wrong."""
    stim, pre, chans = load()
    assert len(chans) == 22, chans
    # every event kept must lie inside a marked window AND inside a QC-kept second
    for rec, nm in ((stim, "stim"), (pre, "pre")):
        for s, (t, c) in rec.times.items():
            inw = np.zeros(t.size, bool)
            for w0, w1 in rec.windows:
                inw |= (t >= w0) & (t < w1)
            assert inw.all(), f"{nm}/{s}: {(~inw).sum()} events outside marked windows"
            sec = np.clip(t.astype(int), 0, rec.nsec - 1)
            assert (rec.cps[sec, c] > 0).all(), f"{nm}/{s}: events in masked seconds"
    # the QC gate must drop marks, not invent them
    for rec, nm in ((stim, "stim"), (pre, "pre")):
        assert rec.times[RKEY][0].size <= rec.raw_times[RKEY][0].size, nm
    # exposure must never exceed the marked seconds of the interval
    for a, b, m in stim.blocks_on(True):
        assert stim.exposure(a, b).max() <= m / 60.0 + 1e-6, (a, b, m)
    # marks and detections are on the same axis: the rater's counts per channel must match a
    # direct pass over the tsv, restricted the same way
    n_raw = sum(v.size for b in STIM_BLOCKS for k, v in _marks().get(b, {}).items()
                if k in chans)
    assert stim.raw_times[RKEY][0].size == n_raw, (stim.raw_times[RKEY][0].size, n_raw)
    print(f"[selfcheck] ok -- {len(chans)} channels; stim marks kept "
          f"{stim.times[RKEY][0].size}/{n_raw} after the QC gate; "
          f"pre marks kept {pre.times[RKEY][0].size}/{pre.raw_times[RKEY][0].size}")


if __name__ == "__main__":
    _selfcheck()
    pooled_vs_pairwise_baseline()
    period_vs_baseline()
