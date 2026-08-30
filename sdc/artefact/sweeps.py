"""
sdc.artefact.sweeps
-------------------
The two P1 sweeps around the chosen operating point 'final'.

    .venv\\Scripts\\python.exe -m sdc.artefact.sweeps

A  pStim K on P1's 145 Hz file, grad held at 4000. "Is 0.61 a plateau or a point on a slope?"
B  grad on P1's 2 Hz file, K held at 450. At 2 Hz pStim only measures +-5 Hz around the
   FUNDAMENTAL, while the artefact is a harmonic comb (2, 4, 6 ... Hz -- 99.9% of the 0.5-80 Hz
   power on P1's stim channel), so pStim is blind to it and grad is the only rule doing anything.

BOTH COLUMNS SHOW THE SAME QUANTITY -- a rate ratio, read the same way, on the same log axis.
The 2 Hz file is CONTINUOUS and has no internal OFF, so each ~64 s stim block is referred to the
stim-free PRE recording chunked to the same length, instead of to an adjacent OFF window. Same
statistic, different reference; the reference is named on the panel rather than left implicit.

BLOCKS ARE THE RESAMPLING UNIT IN BOTH COLUMNS. The HF interval comes from the block-paired
bootstrap; the LF interval bootstraps the stim blocks. Neither resamples channels, for the
reason in blocks.py: channels share the same minutes, so a channel bootstrap is blind to the
noise that actually dominates.

CAVEAT ON THE LF REFERENCE: the pre run is held at 'final' while the stim side varies, so the LF
column shows the knob's effect on the STIM side only. Matching pre runs at each rung would make
it symmetric and would cost about 25 minutes.

Delphos is included but its ~5-10% run-to-run nondeterminism (RAM-dependent tiling) is the
same size as the movement a stability sweep looks for, so Janca and Barkmeier carry any argument
about flatness and Delphos is read as corroboration only.
"""
from pathlib import Path

import numpy as np

from sdc.common.paths import RUNS, figdir

RED, BLUE, VIOLET = "#c0392b", "#2471a3", "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}
# Delphos added after the sweep completed: its per-window Janca/Barkmeier results were
# cached and unaffected, so only the merge stage was repeated. Read its line with the
# caveat that it is nondeterministic at 5-10% from RAM tiling -- movement smaller than
# that across rungs is not evidence of anything.
DETS = ("Janca", "Barkmeier", "Delphos")

# Run STEMS, not profile names: 'final' was renamed to 'finalv2' when the profile split, and
# every rung is trimmed to the same 6-970 s span as the slide figures so the sweep and they
# describe one recording.
SWEEP_A = [("P1_stim_qck225_t970", 225), ("P1_stim_qck300_t970", 300),
           ("P1_stim_qcfinalv2_t970", 450), ("P1_stim_qck675_t970", 675),
           ("P1_stim_qck1000_t970", 1000)]
# 'final' IS grad=1000 since the operating point moved, so it is the g1000 rung rather than a
# separate one -- listing both put the same run on the axis twice, once mislabelled 4000. The
# old 4000 point no longer exists: that run was overwritten when 'final' was re-run.
SWEEP_B = [("g500", 500), ("g750", 750), ("g1000", 1000), ("g1250", 1250),
           ("g1500", 1500), ("g2000", 2000), ("g3000", 3000), ("g6000", 6000)]
MARK_B = 1000


def _masked_on(z):
    """Fraction of stim-ON channel-time the mask removed."""
    fs = float(z["fs"])
    cps = np.asarray(z["clean_per_sec"], float)
    on = np.asarray(z["on_per_sec"], bool)
    m = min(on.size, cps.shape[0])
    denom = cps.shape[1] * on[:m].sum() * fs
    return 1.0 - cps[:m][on[:m]].sum() / max(denom, 1e-9)


def _block_rates(z, keep, blk):
    """Mean-over-channels rate (det/min/channel) in consecutive `blk`-second blocks."""
    fs = float(z["fs"])
    cps = np.asarray(z["clean_per_sec"], float)
    n = len(z["names"])
    edges = np.arange(0, cps.shape[0] + 1, int(blk))
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


def _collect():
    from sdc.artefact.blocks import block_table, ci, pair_changes, log_ratio

    cols = []

    # ---------------- A: HF, each ON BLOCK against the stim-free pre --------------------
    # Was ON vs its adjacent OFF. Changed so BOTH columns use the same estimator -- block over
    # stim-free baseline -- which fixes the zero point externally instead of at whatever the OFF
    # periods happen to be doing, and makes the two columns readable on one axis.
    #
    # The baseline is held at ONE run (P1_pre_qcfinalv2) across every K rung. That is correct
    # rather than lazy: a baseline recording has no stimulation, so compare_spikes drops the
    # relative stim threshold entirely for it and K cannot change it. Only gradThr could, and
    # grad is inert on this 145 Hz pair -- turning it off moves masked-ON by 0.1 points.
    pre_a = RUNS / "P1_pre_qcfinalv2.npz"
    rungs = []
    for stem, K in SWEEP_A:
        f = RUNS / f"{stem}.npz"
        if not (f.is_file() and pre_a.is_file()):
            continue
        z = np.load(f, allow_pickle=False)
        zp = np.load(pre_a, allow_pickle=False)
        if [str(x) for x in zp["names"]] != [str(x) for x in z["names"]]:
            continue
        bp = block_table(z, prefer="after", off_full=True)
        keep = np.flatnonzero(bp.keep)
        blk = float(np.median([b - a for a, b in bp.on_w]))
        rp = _block_rates(zp, keep, blk)
        e = {}
        rng = np.random.default_rng(0)
        for d in DETS:
            if d not in bp.det or d not in rp:
                continue
            base = float(np.nanmedian(rp[d]))
            oc, ot = bp.det[d]["on_count"], bp.det[d]["on_sec"] / 60.0
            v = []
            for b in range(oc.shape[0]):
                g = ot[b] > 0
                if g.any():
                    v.append(np.mean(oc[b][g] / ot[b][g]) / max(base, 1e-12))
            v = np.array([x for x in v if np.isfinite(x) and x > 0], float)
            lg = np.log10(v)
            bs = [10 ** lg[rng.integers(0, lg.size, lg.size)].mean() for _ in range(2000)]
            e[d] = {"ratio": 10 ** lg.mean(),
                    "lo": float(np.percentile(bs, 2.5)),
                    "hi": float(np.percentile(bs, 97.5)),
                    "pooled": float(np.nanmean([np.mean(oc[b][ot[b] > 0] / ot[b][ot[b] > 0])
                                                for b in range(oc.shape[0])]) / max(base, 1e-12)),
                    "vals": v}
        rungs.append({"label": str(K), "mark": K == 450, "det": e,
                      "masked_on": _masked_on(z) * 100, "n_chan": bp.n_chan})
    cols.append({"title": "(a) SWEEP A -- P1 ANT 145 Hz, pStim K   (grad=4000, dyn=3 held)",
                 "xlabel": "K_pStim   (multiple of each channel's own baseline)",
                 "unit": "ON block / stim-free baseline", "rungs": rungs})

    # ---------------- B: LF, each stim block against the stim-free pre ------------------
    pre = RUNS / "P1_ANT2_pre_qcfinalv2.npz"
    rungs = []
    for prof, g in SWEEP_B:
        f = RUNS / f"P1_ANT2_stim_qc{prof}.npz"
        if not (f.is_file() and pre.is_file()):
            continue
        z = np.load(f, allow_pickle=False)
        zp = np.load(pre, allow_pickle=False)
        if [str(x) for x in zp["names"]] != [str(x) for x in z["names"]]:
            continue
        cs = np.asarray(z["clean_per_sec"], float).sum(axis=0)
        cp = np.asarray(zp["clean_per_sec"], float).sum(axis=0)
        keep = np.flatnonzero((cs > 0) & (cp > 0))     # measurable on BOTH sides
        blk = 64.0                                     # the HF file's block length
        rs, rp = _block_rates(z, keep, blk), _block_rates(zp, keep, blk)
        e = {}
        rng = np.random.default_rng(0)
        for d in DETS:
            if d not in rs or d not in rp:
                continue
            base = float(np.nanmedian(rp[d]))
            v = rs[d] / max(base, 1e-12)
            v = v[np.isfinite(v) & (v > 0)]
            lg = np.log10(v)
            bs = [10 ** lg[rng.integers(0, lg.size, lg.size)].mean() for _ in range(2000)]
            e[d] = {"ratio": 10 ** lg.mean(),
                    "lo": float(np.percentile(bs, 2.5)),
                    "hi": float(np.percentile(bs, 97.5)),
                    "pooled": float(np.nanmean(rs[d]) / max(base, 1e-12)),
                    "vals": v}
        rungs.append({"label": str(g), "mark": g == MARK_B, "det": e,
                      "masked_on": _masked_on(z) * 100, "n_chan": int(keep.size)})
    cols.append({"title": "(b) SWEEP B -- P1 ANT 2 Hz, grad_abs   (K=450, dyn=3 held)",
                 "xlabel": "grad_abs   (uV per sample)",
                 "unit": "stim block / pre-file median", "rungs": rungs})
    return cols


def figure():
    import matplotlib.pyplot as plt
    from seeg._style import recessive

    cols = _collect()
    fig, axes = plt.subplots(3, len(cols), figsize=(8.6 * len(cols), 11.5), squeeze=False,
                             sharex="col")
    rng = np.random.default_rng(0)

    for j, col in enumerate(cols):
        rungs = col["rungs"]
        x = np.arange(len(rungs))

        # ---- row 0: the effect ----------------------------------------------------
        ax = axes[0][j]
        for d in DETS:
            y = [r["det"].get(d, {}).get("ratio", np.nan) for r in rungs]
            lo = [r["det"].get(d, {}).get("lo", np.nan) for r in rungs]
            hi = [r["det"].get(d, {}).get("hi", np.nan) for r in rungs]
            ax.plot(x, y, "-o", ms=6, lw=1.6, color=COLORS[d], label=d)
            ax.fill_between(x, lo, hi, color=COLORS[d], alpha=.12, lw=0)
        ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
        ax.set_yscale("linear")
        ax.set_ylabel(f"rate ratio (linear)\n{col['unit']}\nshaded = 95% CI")
        ax.set_title(col["title"], fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, ncol=3)

        # ---- row 1: what the rung cost ---------------------------------------------
        ax = axes[1][j]
        ax.plot(x, [r["masked_on"] for r in rungs], "-o", ms=6, lw=1.8, color="#c2691f",
                label="masked during stim ON")
        ax.set_ylabel("% of stim-ON time masked", color="#c2691f")
        ax.tick_params(axis="y", colors="#c2691f")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax2 = ax.twinx()
        ax2.plot(x, [r["n_chan"] for r in rungs], "-^", ms=6, lw=1.6, color="0.25",
                 label="channels retained")
        ax2.set_ylabel("channels retained", color="0.25")
        ax2.legend(frameon=False, fontsize=8, loc="upper right")

        # ---- row 2: the spread behind row 0, as box + strip -------------------------
        ax = axes[2][j]
        for i, r in enumerate(rungs):
            for k, d in enumerate(DETS):
                a = r["det"].get(d)
                if not a or not len(a["vals"]):
                    continue
                xs = i + (k - 0.5) * 0.30
                bx = ax.boxplot([a["vals"]], positions=[xs], widths=0.22, showfliers=False,
                                patch_artist=True, zorder=2)
                bx["boxes"][0].set(facecolor=COLORS[d], alpha=.20, edgecolor=COLORS[d])
                for part in ("whiskers", "caps", "medians"):
                    for ln in bx[part]:
                        ln.set(color=COLORS[d], lw=1.5)
                ax.scatter(xs + rng.uniform(-.07, .07, len(a["vals"])), a["vals"], s=22,
                           color=COLORS[d], alpha=.55, edgecolor="none", zorder=3)
                ax.scatter([xs], [a["pooled"]], marker="D", s=60, facecolor="white",
                           edgecolor=COLORS[d], lw=1.8, zorder=4)
        ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
        ax.set_yscale("linear")
        ax.set_ylabel("per-block ratios (linear)\nbox + points; white diamond = pooled")
        ax.set_xlabel(col["xlabel"])

        for ax in axes[:, j]:
            ax.set_xticks(x)
            ax.set_xticklabels([r["label"] for r in rungs], fontsize=9)
            for i, r in enumerate(rungs):
                if r["mark"]:
                    ax.axvspan(i - .45, i + .45, color="#2e7d32", alpha=.08, lw=0, zorder=0)
            for lbl, r in zip(ax.get_xticklabels(), rungs):
                if r["mark"]:
                    lbl.set_color("#2e7d32")
                    lbl.set_fontweight("bold")
            recessive(ax)
            ax.grid(axis="y", alpha=.3)

    for col in cols:
        print(f"\n{col['title']}   [{col['unit']}]")
        print(f"{'rung':>7}{'maskON':>8}{'chan':>6}   " + "  ".join(f"{d:>26}" for d in DETS))
        for r in col["rungs"]:
            cells = []
            for d in DETS:
                a = r["det"].get(d)
                cells.append((f"{a['ratio']:.3f} [{a['lo']:.2f}, {a['hi']:.2f}] n={len(a['vals'])}"
                              if a else "-").rjust(26))
            print(f"{r['label']:>7}{r['masked_on']:>7.1f}%{r['n_chan']:>6}   " + "  ".join(cells))

    fig.suptitle(
        "P1 sensitivity to the two artefact knobs, one at a time, around 'final' (shaded green).\n"
        "BOTH columns are a rate ratio on the same log axis. HF has an internal OFF; LF is "
        "continuous, so each stim block is referred to the stim-free pre file.\n"
        "Row 2 is the spread the point estimate hides -- one point per block.", fontsize=11)
    fig.tight_layout()
    out = figdir("real") / "p1_sweeps_around_final.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    figure()


# --------------------------------------------------------------------------------------
def grad_confound(rungs=((300, "g300"), (1000, "g1000"), (4000, "g4000")),
                  rec="P1_ANT2_stim", pre="P1_ANT2_pre_qcfinalv2", outdir=None):
    """The channel-confound question, asked of the grad ladder on the 2 Hz file.

    Every rung does two things at once: it masks contaminated TIME and it removes whole
    CHANNELS. The second alone moves the estimate even if the surviving data were untouched,
    because the answer drifts toward whatever the survivors say. So each rung is computed twice:

      own      that rung's own surviving channels -- the ladder as normally reported
      common   the channels surviving EVERY rung, held fixed. Only the data differs, so any
               movement left is the mask acting on the signal rather than on the channel list.

    This is the same construction as artefact_channel_confound.png, which showed that most of
    the OLD absolute-pStim ladder's swing (P1 Janca 1.997 -> 0.429) was channel selection rather
    than artefact removal -- 180 channels down to 7. The relative pStim threshold fixed that at
    145 Hz; grad at 2 Hz is absolute, so the question has to be asked again here.

    The 2 Hz file is continuously stimulated, so the effect is each ~64 s block against the
    stim-free pre recording rather than against an internal OFF.
    """
    import matplotlib.pyplot as plt
    from seeg._style import recessive
    from sdc.artefact.exposure import _chunk_blocks

    zp = np.load(RUNS / f"{pre}.npz", allow_pickle=False)
    names = np.array([str(x) for x in zp["names"]])
    have, sets = [], []
    for g, prof in rungs:
        f = RUNS / f"{rec}_qc{prof}.npz"
        if not f.is_file():
            print(f"  [skip] {prof}: no run")
            continue
        z = np.load(f, allow_pickle=False)
        bp = _chunk_blocks(z, blk=64.0)
        have.append((g, prof, z, bp))
        sets.append(set(names[bp.keep]))
    if len(have) < 2:
        raise SystemExit("need at least two rungs")
    common = sorted(set.intersection(*sets))
    print(f"\n{rec}: {len(have)} rungs, common channel set {len(common)}")

    def effect(z, bp, chans=None):
        keep = np.flatnonzero(bp.keep)
        if chans is not None:
            want = set(chans)
            sel = [i for i, k in enumerate(keep) if names[k] in want]
        else:
            sel = list(range(len(keep)))
        rp = _block_rates(zp, keep[sel], 64.0)
        out = {}
        for d in DETS:
            if d not in bp.det or d not in rp:
                continue
            base = float(np.nanmedian(rp[d]))
            oc = bp.det[d]["on_count"][:, sel]
            ot = bp.det[d]["on_sec"][:, sel] / 60.0
            v = []
            for b in range(oc.shape[0]):
                ok = ot[b] > 0
                if ok.any():
                    v.append(np.mean(oc[b][ok] / ot[b][ok]) / max(base, 1e-12))
            v = np.array([x for x in v if np.isfinite(x) and x > 0], float)
            out[d] = 10 ** np.log10(v).mean() if v.size else np.nan
        return out

    own = [effect(z, bp) for _g, _p, z, bp in have]
    com = [effect(z, bp, common) for _g, _p, z, bp in have]
    gs = [g for g, *_ in have]
    x = np.arange(len(gs))

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    for d in DETS:
        yo = [o.get(d, np.nan) for o in own]
        yc = [c.get(d, np.nan) for c in com]
        ax.plot(x, yo, "-o", ms=7, lw=1.8, color=COLORS[d], label=f"{d} own channels")
        ax.plot(x, yc, "--s", ms=6, lw=1.5, color=COLORS[d], alpha=.6, mfc="white",
                label=f"{d} common {len(common)}")
    ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
    ax.set_ylabel("stim block / stim-free baseline")
    ax.set_title(f"{rec}: is the grad ladder moving the DATA or the CHANNEL LIST?\n"
                 "solid = each rung's own channels; dashed = the set surviving every rung.\n"
                 "Where they diverge, the movement is channel selection.",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    recessive(ax); ax.grid(axis="y", alpha=.3)

    ax = axes[1]
    ax.plot(x, [_masked_on(z) * 100 for _g, _p, z, _b in have], "-o", ms=7, lw=1.8,
            color="#c2691f", label="masked during stim")
    ax.set_ylabel("% of stim time masked", color="#c2691f")
    ax.tick_params(axis="y", colors="#c2691f")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax2 = ax.twinx()
    ax2.plot(x, [int(bp.keep.sum()) for _g, _p, _z, bp in have], "-^", ms=7, lw=1.6,
             color="0.25", label="channels retained")
    ax2.set_ylabel("channels retained", color="0.25")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gs])
    ax.set_xlabel("grad_abs   (uV per sample)")
    recessive(ax); ax.grid(axis="y", alpha=.3)

    print(f"{'grad':>6}{'maskON':>9}{'chan':>6}   " +
          "  ".join(f"{d[:4] + ' own/common':>20}" for d in DETS))
    for i, (g, _p, z, bp) in enumerate(have):
        cells = [f"{own[i].get(d, float('nan')):.3f} / {com[i].get(d, float('nan')):.3f}"
                 for d in DETS]
        print(f"{g:>6}{_masked_on(z) * 100:>8.1f}%{int(bp.keep.sum()):>6}   " +
              "  ".join(f"{c:>20}" for c in cells))

    fig.tight_layout()
    out = (figdir("real") if outdir is None else Path(outdir)) / "grad_channel_confound.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")
