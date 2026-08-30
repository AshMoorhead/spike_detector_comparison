"""
sdc.artefact.matrix_report
--------------------------
Read-out for the artefact-handling comparison (plans/polymorphic-wiggling-breeze.md).

    .venv\\Scripts\\python.exe -m sdc.artefact.matrix_report          # the 2x2 working view
    .venv\\Scripts\\python.exe -c "from sdc.artefact.matrix_report import figure_headline; \\
                                   figure_headline(interval='ci')"   # the abstract figure

THREE FIGURES, THREE JOBS. `figure_2x2` and `figure_simple` are working views over every
condition on disk; `figure_headline` is the abstract figure and shows only the conditions that
differ in KIND -- nothing, whole-channel rejection, epoch-level rejection (+ peri-pulse at 2 Hz).
`figure_selection` isolates the channel-selection component on its own.

ALL THREE DETECTORS, ON THE SAME CHANNELS. Every row uses `row_mask`: the intersection of
channels every detector can measure, so between-detector spread is a difference between
detectors and not between three montages. A row missing a detector is flagged, because spread is
max/min over the detectors PRESENT and a two-detector row is not the same statistic as a
three-detector one -- `mne p10` once read as the tightest condition in the panel purely because
Delphos was absent from it.

STIMULATION-ON TIME ONLY, against the whole stim-free baseline. See
`rate_confound.per_channel_stems` for why: reading "all" diluted the intermittent 145 Hz trial
with 765 s of unstimulated recording and left the continuous 2 Hz trial untouched, so the two
panels were not the same measurement.

COST IS REPORTED BESIDE EFFECT, always. A condition that agrees by condemning most of the
recording has not succeeded; every column therefore carries the channels it retained. The
artefact ladder taught this the hard way -- most of its apparent improvement turned out to be
excluding channels rather than cleaning what remained, which is what the fixed-channel-set
squares in `figure_headline` exist to separate.

WHAT THE FIGURES SHOW, as of the current runs (all channels, stim-ON vs paired baseline):

    145 Hz   no rejection    J 1.029  B 0.510  D 1.222   spread 2.40  [122 ch]
             whole-channel   J 0.848  B 0.396  D 0.999   spread 2.52  [ 91 ch]
             epoch-level     J 0.328  B 0.400  D 0.553   spread 1.69  [ 71 ch]
      2 Hz   no rejection    J 1.038  B 1.025  D 1.869   spread 1.82  [128 ch]
             whole-channel   J 0.985  B 1.073  D 1.780   spread 1.81  [123 ch]
             epoch-level     J 0.856  B 1.015  D 0.981   spread 1.19  [ 69 ch]
             + peri-pulse    J 0.849  B 0.995  D 0.965   spread 1.17  [ 69 ch]

Barkmeier is PINNED on the 2 Hz panel (see seeg.spikes.scale_denominator); the published-scaling
version is drawn translucent beside it on the first column only. Epoch-level rejection is the
only condition bringing all three within 0.25 in ratio, on both trials.
"""
import numpy as np

from sdc.common import cond
from sdc.common.paths import RUNS, figdir
from sdc.artefact.blocks import block_table, log_ratio

DETS = ("Janca", "Barkmeier")
DETS_ALL = ("Janca", "Barkmeier", "Delphos")
ORDER = ["none", "mnebads10", "mnebads15", "mnebads75", "mnebads150",
         "k150g150", "k150g1000", "k450g1000", "k150g0", "k450g0",
         # Gradient-only rules, no stim-spectral term. These exist on the 2 Hz trial only and
         # are silently skipped on the 145 Hz one (rows_gated drops any condition whose paired
         # runs are absent), which is why one ORDER can serve both recordings.
         "dynr", "dynrg1000",
         "e1", "e1t10"]
CONDITIONS = {"P1_stim": list(ORDER), "P1_ANT2_stim": list(ORDER)}
BASELINE_OF = {"P1_stim": "P1_pre", "P1_ANT2_stim": "P1_ANT2_pre"}
# det/chan-min on the BASELINE recording, for the gated column. 6 keeps ~20-30 channels on
# P1_stim; 10 kept only 8, which is not an analysable montage. The gate is NON-MONOTONIC below
# ~4: gating at 1-3 det/min made between-detector spread WORSE than no gate (4.51 / 4.09 / 3.03
# against 3.84 ungated), because it strips channels where Barkmeier ran high relative to
# baseline while keeping the quiet ones where artefact dominates the ratio.
GATE_RATE = 6.0
LABEL = {"none": "A  none", "mnebads10": "B  mne p10 (48ch)", "mnebads15": "B  mne p15 (32ch)",
         "mnebads75": "B  mne p75 (14ch)", "mnebads150": "B  mne p150 (8ch)",
         "k150g150": "C  k150/g150", "k150g1000": "C  k150/g1000",
         "k450g1000": "C  k450/g1000 (finalv2)", "k150g0": "C  k150/grad OFF",
         "k450g0": "C  k450/grad OFF", "e1": "E1 periodicity z>5",
         "e1t10": "E1 periodicity z>10",
         "dynr": "D  dynR only (no stim rule)", "dynrg1000": "D  dynR + g1000, no stim rule"}
# P1_ANT2_stim is CONTINUOUS stimulation: 477 ON epochs and no OFF, so `cond.select(z,"off")`
# selects nothing and the within-file ratio every other recording uses does not exist. The only
# available contrast is the stim file against its paired stim-free PRE file, processed under the
# SAME profile -- otherwise the ratio compares two different maskings rather than two conditions.
CONTINUOUS = {"P1_ANT2_stim": "P1_ANT2_pre"}


def _rows(rec):
    out = []
    for p in CONDITIONS[rec]:
        f = RUNS / f"{rec}_qc{p}.npz"
        if not f.is_file():
            print(f"  [missing] {f.name}")
            continue
        z = np.load(f, allow_pickle=False)
        bt = block_table(z)
        ON, OFF = cond.select(z, "on"), cond.select(z, "off")
        r = {"profile": p, "n_chan": bt.n_chan,
             "masked_on": 1.0 - float(np.mean(ON.clean_sec) / max(ON.T, 1e-9)),
             "masked_off": 1.0 - float(np.mean(OFF.clean_sec) / max(OFF.T, 1e-9))}
        for d in DETS:
            r[d] = 10 ** log_ratio(bt.det[d]) if d in bt.det else np.nan
        v = [r[d] for d in DETS if np.isfinite(r[d])]
        r["spread"] = (max(v) / min(v)) if len(v) > 1 and min(v) > 0 else np.nan
        r["both_below_1"] = bool(len(v) > 1 and max(v) < 1.0)
        out.append(r)
    return out


def report():
    all_rows = {}
    for rec in CONDITIONS:
        if rec in CONTINUOUS:
            # No OFF period exists, so the within-file ratio is undefined -- not an error, a
            # property of the trial. report_continuous() handles these against their pre file.
            continue
        rows = _rows(rec)
        all_rows[rec] = rows
        if not rows:
            continue
        print(f"\n=== {rec} ===")
        print(f"  {'condition':<26}{'chan':>6}{'mask ON':>9}{'mask OFF':>10}"
              + "".join(f"{d:>11}" for d in DETS) + f"{'spread':>9}{'both<1':>8}")
        for r in rows:
            print(f"  {LABEL.get(r['profile'], r['profile']):<26}{r['n_chan']:>6}"
                  f"{r['masked_on']:>8.1%}{r['masked_off']:>10.1%}"
                  + "".join(f"{r[d]:>11.3f}" for d in DETS)
                  + f"{r['spread']:>9.2f}{'YES' if r['both_below_1'] else '-':>8}")
        win = [r for r in rows if r["both_below_1"]]
        print(f"  -> {len(win)}/{len(rows)} conditions put BOTH detectors below 1.0"
              + (f": {', '.join(LABEL.get(r['profile'], r['profile']) for r in win)}"
                 if win else ""))
    return all_rows


def figure(all_rows, fname="artefact_matrix_P1.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [r for r in CONDITIONS if all_rows.get(r)]
    fig, axes = plt.subplots(1, len(recs), figsize=(7.6 * len(recs), 5.4), squeeze=False)
    for j, rec in enumerate(recs):
        rows = all_rows[rec]
        ax = axes[0][j]
        y = np.arange(len(rows))
        for d, c, m in zip(DETS, ("#c0392b", "#0072b2"), ("o", "s")):
            ax.plot([r[d] for r in rows], y, m, ms=9, color=c, label=d)
        for i, r in enumerate(rows):
            v = [r[d] for d in DETS if np.isfinite(r[d])]
            if len(v) > 1:
                ax.plot([min(v), max(v)], [i, i], color="0.6", lw=1.4, zorder=0)
        ax.axvline(1.0, color="0.25", ls="--", lw=1.4)
        ax.axvspan(ax.get_xlim()[0], 1.0, color="#2e7d32", alpha=.05, zorder=-2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{LABEL.get(r['profile'], r['profile'])}\n"
                            f"{r['n_chan']} ch, {r['masked_on']:.0%} masked"
                            for r in rows], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("stim ON / OFF ratio")
        ax.set_title(rec, fontsize=10, loc="left")
        ax.grid(axis="x", alpha=.25)
        if j == 0:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Artefact handling vs detector agreement and direction\n"
                 "green = the direction expert marking supports (suppression); "
                 "grey bar = between-detector spread", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = figdir("real") / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n[saved] {out}")


def _rows_vs_baseline(rec, base_rec):
    from sdc.artefact.blocks import baseline_rates
    from sdc.common import cond as _cond

    out = []
    for p in CONDITIONS[rec]:
        fs_, fb = RUNS / f"{rec}_qc{p}.npz", RUNS / f"{base_rec}_qc{p}.npz"
        if not fs_.is_file() or not fb.is_file():
            print(f"  [missing] {fs_.name if not fs_.is_file() else fb.name}")
            continue
        z = np.load(fs_, allow_pickle=False)
        names = [str(s) for s in z["names"]]
        sel = _cond.select(z, "all")
        r = {"profile": p, "n_chan": len(names)}
        try:
            base = baseline_rates(f"{base_rec}_qc{p}", names)
        except SystemExit as e:
            print(f"  [skip {p}] {e}")
            continue
        for d in DETS:
            if d not in [str(s) for s in z["detectors"]]:
                r[d] = np.nan
                continue
            c = np.bincount(z[f"{d}_chan"][sel.keep(d)], minlength=len(names))
            stim_rate = sel.rate(c) * 60.0
            b = base.get(d)
            # Only channels MEASURABLE IN BOTH: a rule that drops a channel in one recording and
            # not the other would otherwise contribute a ratio against a missing denominator.
            ok = np.isfinite(stim_rate) & np.isfinite(b) & (b > 0) & (stim_rate > 0)
            r[d] = float(np.median(stim_rate[ok] / b[ok])) if ok.sum() else np.nan
            r[f"{d}_n"] = int(ok.sum())
        v = [r[d] for d in DETS if np.isfinite(r[d])]
        r["spread"] = (max(v) / min(v)) if len(v) > 1 and min(v) > 0 else np.nan
        r["both_below_1"] = bool(len(v) > 1 and max(v) < 1.0)
        r["masked_on"] = r["masked_off"] = np.nan
        out.append(r)
    return out


def report_continuous():
    all_rows = {}
    for rec, base in CONTINUOUS.items():
        rows = _rows_vs_baseline(rec, base)
        if not rows:
            continue
        all_rows[rec] = rows
        print(f"\n=== {rec} vs {base} (continuous stim: median per-channel stim/baseline) ===")
        print(f"  {'condition':<26}{'chan':>6}" + "".join(f"{d:>11}" for d in DETS)
              + f"{'n both':>8}{'spread':>9}{'both<1':>8}")
        for r in rows:
            print(f"  {LABEL.get(r['profile'], r['profile']):<26}{r['n_chan']:>6}"
                  + "".join(f"{r[d]:>11.3f}" for d in DETS)
                  + f"{r.get(DETS[0] + '_n', 0):>8}{r['spread']:>9.2f}"
                  + f"{'YES' if r['both_below_1'] else '-':>8}")
        win = [r for r in rows if r["both_below_1"]]
        print(f"  -> {len(win)}/{len(rows)} put BOTH detectors below 1.0"
              + (f": {', '.join(LABEL.get(r['profile'], r['profile']) for r in win)}"
                 if win else ""))
    return all_rows


if __name__ == "__main__":
    figure_2x2()


def rows_gated(rec, base_rec, gate_rate=None):
    """Every condition for one recording, as stim/baseline ratios, ungated and gated.

    Both recordings are now read the SAME way -- stim file against its paired baseline file --
    rather than ON/OFF within the stim file. The 145 Hz trial does have OFF periods, but they
    sit between stimulation blocks and may carry carryover, so conditioning on them conditions
    weakly on the data being analysed. The 2 Hz trial is continuous and has no OFF at all. One
    contrast for both removes an inconsistency as well as a confound.

    THE GATE IS BUILT FROM THE BASELINE, never from the stim file: a channel that stimulation
    silenced would fail a stim-side rate gate and be dropped for having shown the effect.
    """
    from sdc.artefact.blocks import baseline_rates
    from sdc.common import cond as _cond

    gate_rate = GATE_RATE if gate_rate is None else gate_rate
    out = []
    for p in CONDITIONS[rec]:
        fs_, fb = RUNS / f"{rec}_qc{p}.npz", RUNS / f"{base_rec}_qc{p}.npz"
        if not fs_.is_file() or not fb.is_file():
            continue
        z = np.load(fs_, allow_pickle=False)
        names = [str(s) for s in z["names"]]
        # STIM-ON ONLY -- see rate_confound.per_channel_stems for why. "all" diluted the
        # intermittent 145 Hz trial with 765 s of unstimulated time and left the continuous
        # 2 Hz trial untouched, so the two rows of this figure were not the same measurement.
        sel = _cond.select(z, "on")
        try:
            base = baseline_rates(f"{base_rec}_qc{p}", names)
        except SystemExit:
            continue
        r = {"profile": p, "n_chan": len(names)}
        dets = [str(s) for s in z["detectors"]]
        for d in DETS_ALL:
            if d not in dets:
                r[d] = r[f"{d}_gated"] = np.nan
                continue
            c = np.bincount(z[f"{d}_chan"][sel.keep(d)], minlength=len(names))
            stim = sel.rate(c) * 60.0
            b = base[d]
            ok = np.isfinite(stim) & np.isfinite(b) & (b > 0) & (stim > 0)
            g = ok & (b >= gate_rate)
            r[d] = float(np.median(stim[ok] / b[ok])) if ok.sum() else np.nan
            r[f"{d}_gated"] = float(np.median(stim[g] / b[g])) if g.sum() else np.nan
            r[f"{d}_n"], r[f"{d}_ngated"] = int(ok.sum()), int(g.sum())
        out.append(r)
    return out


# ---- the abstract figure -----------------------------------------------------------------
# Four conditions, not twelve. The mne rungs above 10 and the k150/k450 pairs are working views:
# they answer "does this knob matter" (it does not, mostly) and then have no further job. These
# four are the ones that differ in KIND -- nothing, whole-channel rejection, epoch masking
# without the gradient rule, epoch masking with it.
# Three conditions that differ in KIND: nothing, whole-channel rejection, epoch masking.
# The kStim+dynR and gradient-1000 rungs are dropped -- on a FIXED channel set they are
# within 0.5 of each other at 145 Hz (1.68 / 1.55) while the tight gradient reaches 1.20,
# and at 2 Hz none of them move anything at all. They were ladder rungs, not conditions.
SIMPLE = ["none", "mnebads10", "k450g150"]
SIMPLE_LABEL = {"none": "no artefact handling", "mnebads10": "bad-channel rejection (MNE)",
                "k450g0": "epoch masking (kStim + dynR)",
                "k450g1000": "epoch masking + gradient (1000)",
                # The TIGHT gradient rung. kStim is 150 here rather than 450 only because that
                # is the pairing that already carries Delphos on all four recordings, and kStim
                # is near-inert once a gradient rule is on (k150 vs k450 at g1000 differ by
                # <0.5% ungated and are identical to 3 dp gated). Re-run as k450g150 if that
                # ever has to be defended rather than asserted.
                "k450g150": "epoch masking + gradient (150)"}
# X LIMITS ARE COMPUTED, NOT CHOSEN. The two trials differ by more than a decade in how far the
# worst channels run, so one shared axis leaves the 145 Hz panel a cluster of bars near 1. But
# hand-set limits are how a figure starts lying: a first attempt at (0.45, 2.2) for 145 Hz would
# have silently clipped the mne-p10 gated Janca IQR, which runs 0.70-4.64 -- the single widest
# thing in that panel and the reason that condition is not the winner it looks like.
#
# So the axis is fitted to the data with one guarantee: EVERY median and IQR is inside it.
# Whiskers may run off, and when they do an arrow is drawn at the edge, so a clipped bar reads
# as clipped rather than as a bar that ends there.
XPAD_LO, XPAD_HI = 0.85, 1.18
# Rows that are not plain profiles, appended per recording. Each is (label, stim_stem, base_stem).
# The BASELINE STEM IS THE UNMODIFIED PROFILE RUN in both cases, deliberately:
#   * peri-pulse rejection has nothing to reject in a stim-free file (pulse_reject says so);
#   * the scale correction makes the STIM file adopt the BASELINE's scale factor, so correcting
#     the baseline too would defeat the entire point.
EXTRA_ROWS = {
    "P1_ANT2_stim": [
        # Barkmeier comes from the PINNED pair here too, so this row differs from the one
        # above it by peri-pulse rejection alone. Reading it against an unpinned Barkmeier
        # would make the pin and the rejection change together and neither attributable.
        # A peri-pulse-ONLY row was built and removed. It is not wrong -- rejecting a
        # -5/+15 ms window with no epoch masking reached Janca 0.744 / Delphos 0.774 while
        # keeping 65 channels against grad-150's 45 -- but it is a fourth kind of intervention
        # in a panel that argues about three. The runs are on disk
        # (P1_ANT2_stim[_bd16.7902]_prm5p15_qcnone) if it is wanted back.
        # ...and stacked on the gradient. It was previously stacked on grad-1000, which is no
        # longer a row, so it was comparing against a condition not in the figure. On grad-150
        # it removes 2.5% / 2.9% -- AT CHANCE, i.e. the gradient has already taken every
        # pulse-locked detection and this step is redundant once it is on.
        ("+ peri-pulse rejection (-5/+15 ms)",
         "P1_ANT2_stim_prm5p15_qck450g150", "P1_ANT2_pre_qck450g150",
         "P1_ANT2_stim_bd16.7902_prm5p15_qck450g150", "P1_ANT2_pre_bd16.7902_qck450g150"),
        # The old BARK_SCALE demonstration row is REMOVED. It showed the mechanism by
        # overshooting (0.516 -> 0.977); the pinned rows show it correctly, and keeping both
        # would put two different corrections of the same confound in one panel.
    ],
}
# The cumulative stack the abstract figure argues: each row is the one above plus one rule.
# BOTH trials show masking with AND without the gradient. That pair is the comparison: at
# 145 Hz the gradient changes almost nothing (spread 1.07 -> 1.05) while at 2 Hz it is what
# reaches the pulse artefact. Showing the rung on only one trial would leave that asymmetry
# as an assertion in the caption instead of two rows the reader can compare.
SIMPLE_BY_REC = {}   # both trials use SIMPLE: the with/without-gradient pair is
                     # shown on BOTH so the contrast between them can be read.
# Recordings where a Barkmeier-pinned variant should be shown when one exists on disk. 145 Hz
# is deliberately NOT here: its normaliser moves 1.3% between stim and baseline, so a pinned
# row there would be a duplicate of the row above it and would imply a correction was needed.
PIN_REC = {"P1_ANT2_stim"}


def pinned_stem(rec, profile):
    """The Barkmeier-PINNED run for this recording+profile, or None.

    Found by glob rather than by a hardcoded number: the pin value is whatever MATLAB measured
    on the baseline (`tools/run_pinned_2hz.py`), and hardcoding it here would go stale silently
    the first time that probe is re-measured.
    """
    # The `_bd<value>` token must be the LAST thing before `_qc`. A bare `*_bd*_qc<p>` glob also
    # matches derived runs that carry a further variant in between -- `..._bd16.79_prm5p15_qc...`
    # -- so the plain pinned run and its pulse-rejected child both matched, len(hits)==2, and the
    # row silently fell back to UNPINNED Barkmeier while its neighbours were pinned.
    import re
    pat = re.compile(rf"^{re.escape(rec)}_bd[0-9.]+_qc{re.escape(profile)}$")
    hits = sorted(f.stem for f in RUNS.glob(f"{rec}_bd*_qc{profile}.npz") if pat.match(f.stem))
    return hits[0] if len(hits) == 1 else None


def _simple_rows(rec, base, conditions=None):
    """Row specs for one recording: dicts with label/stim/base and optional Barkmeier overrides.

    `bark_stim`/`bark_base` let ONE row take Barkmeier from a different pair of runs than Janca
    and Delphos. That is how the pinned-normaliser rows work: `fixed_denom` reaches Barkmeier
    only, so the pinned runs are Barkmeier+Janca (no Delphos, which would cost 15 min a run to
    reproduce something guaranteed identical). Janca is re-run and checked against the unpinned
    count -- see tools/run_pinned_2hz._check_janca -- so the substitution is verified, not
    assumed.
    """
    out = []
    for p in (conditions or SIMPLE_BY_REC.get(rec, SIMPLE)):
        s, b = f"{rec}_qc{p}", f"{base}_qc{p}"
        if not ((RUNS / f"{s}.npz").is_file() and (RUNS / f"{b}.npz").is_file()):
            continue
        ps, pb = pinned_stem(rec, p), pinned_stem(base, p)
        if ps and pb and rec in PIN_REC:
            # ONE column, with Barkmeier drawn TWICE on the first condition: solid = the pinned
            # normaliser, translucent = as published. Previously these were two whole columns,
            # which spent a third of the panel restating a single detector's threshold
            # convention and invited reading it as a step in the artefact ladder, which it is
            # not. `bark_alt` carries the published pair; every other condition shows the
            # pinned Barkmeier only.
            first = p == (conditions or SIMPLE_BY_REC.get(rec, SIMPLE))[0]
            out.append(dict(label=SIMPLE_LABEL.get(p, p), stim=s, base=b,
                            bark_stim=ps, bark_base=pb,
                            bark_alt=((s, b) if first else None)))
        else:
            out.append(dict(label=SIMPLE_LABEL.get(p, p), stim=s, base=b))
    for row in EXTRA_ROWS.get(rec, []):
        lab, s, b = row[0], row[1], row[2]
        bs, bb = (row[3], row[4]) if len(row) >= 5 else (None, None)
        if not ((RUNS / f"{s}.npz").is_file() and (RUNS / f"{b}.npz").is_file()):
            continue
        if bs and not ((RUNS / f"{bs}.npz").is_file() and (RUNS / f"{bb}.npz").is_file()):
            bs = bb = None          # pinned pair not built yet -- fall back, and say so
            print(f"[simple] {lab}: pinned Barkmeier pair missing, using unpinned")
        out.append(dict(label=lab, stim=s, base=b, bark_stim=bs, bark_base=bb))
    return out


def _row_channels(spec, tmax=None):
    """{detector: (stim_rate, baseline_rate)} for a row spec, with Barkmeier swapped if asked."""
    from sdc.artefact.rate_confound import per_channel_stems

    o, _ = per_channel_stems(spec["stim"], spec["base"], tmax=tmax)
    if spec.get("bark_stim"):
        ob, _ = per_channel_stems(spec["bark_stim"], spec["bark_base"], tmax=tmax)
        if "Barkmeier" in ob:
            o = {**o, "Barkmeier": ob["Barkmeier"]}
    return o


def figure_simple(gate_rate=None, fname="artefact_matrix_P1_simple.png", conditions=None):
    """The four conditions, every channel drawn -- box + strip, not a median point.

    WHY THE DISTRIBUTION AND NOT THE MEDIAN. The 2x2 view plots one median per detector per
    condition, and three medians close together read as "the detectors agree". They do not:
    at 2 Hz the per-channel ratios span more than three orders of magnitude, and a handful of
    noise-dominated channels (P1's H shaft: baseline 0.14-1.5 det/chan-min, stim up to 114)
    carry ratios of 80-13000 that no median can show. The claim this figure has to support is
    about the SPREAD collapsing, so the spread has to be on the page.

    The box is the per-channel ratio distribution (median, IQR, 5-95 whiskers); the strip behind
    it is every channel. A condition succeeds by pulling the whole distribution together and
    below 1, not by moving its median.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    from sdc.artefact.rate_confound import per_channel_stems

    gate_rate = GATE_RATE if gate_rate is None else gate_rate
    recs = [(r, BASELINE_OF[r]) for r in BASELINE_OF]
    rows = {r: _simple_rows(r, b, conditions) for r, b in recs}
    n_row = max(len(v) for v in rows.values())
    fig, axes = plt.subplots(len(recs), 2,
                             figsize=(14.0, 1.6 + 0.62 * n_row * len(recs)), squeeze=False)
    colors = {"Janca": "#c0392b", "Barkmeier": "#0072b2", "Delphos": "#4a3aa7"}
    off = {"Janca": +0.24, "Barkmeier": 0.0, "Delphos": -0.24}

    # Pass 1: the axis limits, from every IQR that will be drawn on this recording's two panels.
    # Both columns share them so the gated column is read as a subset of the same axis rather
    # than as a differently-scaled picture.
    xlim = {}
    for rec, base in recs:
        q = []
        for spec in rows[rec]:
            try:
                o = _row_channels(spec)
            except (FileNotFoundError, SystemExit, KeyError):
                continue
            for d in DETS_ALL:
                if d not in o:
                    continue
                s, b = o[d]
                for g in (False, True):
                    m = np.isfinite(s) & np.isfinite(b) & (b > 0) & (s > 0)
                    if g:
                        m &= b >= gate_rate
                    if m.sum() >= 3:
                        q += list(np.percentile(s[m] / b[m], [25, 75]))
        xlim[rec] = ((min(q) * XPAD_LO, max(q) * XPAD_HI) if q else (0.1, 10.0))

    for i, (rec, base) in enumerate(recs):
        lo_x, hi_x = xlim[rec]
        for j, gated in enumerate((False, True)):
            ax = axes[i][j]
            present = []
            for k, spec in enumerate(rows[rec]):
                lab = spec["label"]
                try:
                    o = _row_channels(spec)
                except (FileNotFoundError, SystemExit, KeyError):
                    continue
                n_here = 0
                for d in DETS_ALL:
                    if d not in o:
                        continue
                    s, b = o[d]
                    m = np.isfinite(s) & np.isfinite(b) & (b > 0) & (s > 0)
                    if gated:
                        m &= b >= gate_rate
                    if m.sum() < 3:
                        continue
                    v = s[m] / b[m]
                    n_here = max(n_here, int(m.sum()))
                    y = k + off[d]
                    # A LETTER-VALUE BAR, not a box with 160 dots behind it. The dots were the
                    # problem: at 2 Hz the per-channel ratios span four decades, so the strip
                    # became a smear across the whole axis and buried the one comparison the
                    # figure exists to make -- whether the three detectors' distributions sit on
                    # top of each other. Thin line = 10-90th percentile, thick = IQR, dot =
                    # median. Same information, one twentieth of the ink.
                    p10, p25, p50, p75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
                    ax.plot([max(p10, lo_x), min(p90, hi_x)], [y, y], lw=1.1, color=colors[d],
                            solid_capstyle="butt", zorder=2)
                    # Mark a whisker that runs off the axis, so the bar is not read as ending
                    # where the panel does.
                    for val, edge, mk in ((p10, lo_x, "<"), (p90, hi_x, ">")):
                        if (val < lo_x) if mk == "<" else (val > hi_x):
                            ax.plot([edge], [y], mk, ms=4.0, color=colors[d], zorder=5,
                                    clip_on=False)
                    ax.plot([p25, p75], [y, y], lw=4.2, color=colors[d], solid_capstyle="butt",
                            alpha=.55, zorder=3)
                    ax.plot([p50], [y], "o", ms=6.0, color=colors[d], mec="white", mew=1.0,
                            zorder=4)
                present.append((k, f"{lab}  [{n_here}ch]"))
            ax.axvline(1.0, color="0.25", ls="--", lw=1.3, zorder=1)
            ax.set_xscale("log")
            # Explicit ticks in plain numbers. matplotlib's default log locator puts minor
            # labels at 2/3/4/6 x 10^n, which on these narrow ranges collide into unreadable
            # runs like "3x10^0 4x10^0" -- and the reader needs to compare a ratio against 1,
            # which is a job for "0.5 / 1 / 2", not for scientific notation.
            ax.xaxis.set_major_locator(mticker.FixedLocator(
                [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 8, 16, 32]))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{x:g}" if x >= 0.1 else ""))
            ax.set_yticks([k for k, _ in present])
            ax.set_yticklabels([t for _, t in present], fontsize=8.5)
            ax.set_ylim(-0.6, (max(k for k, _ in present) if present else 0) + 0.6)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.25)
            ax.set_xlim(lo_x, hi_x)
            if i == 0:
                ax.set_title("all channels" if not gated
                             else f"baseline rate >= {gate_rate:g} det/chan-min", fontsize=10)
            if i == len(recs) - 1:
                ax.set_xlabel("per-channel  stim-ON / baseline  (log)")
            if j == 1:
                ax.set_ylabel(rec, fontsize=9)
                ax.yaxis.set_label_position("right")
    handles = [plt.Line2D([], [], marker="o", ls="-", color=colors[d], label=d)
               for d in DETS_ALL]
    # Above the panels, not inside one: in-axes it landed on the bottom row of P1_stim and hid
    # the condition the figure is arguing for.
    fig.legend(handles=handles, fontsize=9, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.935), frameon=False)
    fig.suptitle("Artefact handling: STIMULATION-ON rate vs the stim-free baseline\n"
                 "dot = median, thick bar = IQR, thin line = 10-90th percentile across channels; "
                 "left of the dashed line is suppression.", fontsize=11)
    fig.tight_layout(rect=(0.02, 0, 1, 0.90))
    fig.subplots_adjust(wspace=0.54)
    out = figdir("real") / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=145)
    plt.close(fig)
    print(f"[saved] {out}")


def figure_2x2(gate_rate=None, fname="artefact_matrix_P1_2x2.png", note=None):
    """Rows = stim frequency, columns = all channels vs baseline-rate-gated channels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gate_rate = GATE_RATE if gate_rate is None else gate_rate
    data = {r: rows_gated(r, BASELINE_OF[r], gate_rate) for r in BASELINE_OF}
    data = {k: v for k, v in data.items() if v}
    if not data:
        print("[2x2] no paired runs found yet")
        return
    recs = list(data)
    # Wide enough for the row labels: they carry the condition, the channel count AND the
    # detector-count warning, and at 13.5 in the longest were clipped off the left edge.
    fig, axes = plt.subplots(len(recs), 2, figsize=(15.5, 3.2 + 0.42 * max(
        len(v) for v in data.values()) * len(recs)), squeeze=False, sharex="col")
    cols = [("", "all channels measurable in both"),
            ("_gated", f"baseline rate >= {gate_rate:g} det/chan-min")]
    for i, rec in enumerate(recs):
        rows = data[rec]
        for j, (suf, ctitle) in enumerate(cols):
            ax = axes[i][j]
            y = np.arange(len(rows))
            for d, c, m in zip(DETS_ALL, ("#c0392b", "#0072b2", "#4a3aa7"), ("o", "s", "D")):
                v = [r.get(d + suf, np.nan) for r in rows]
                if np.isfinite(v).any():
                    ax.plot(v, y, m, ms=7, color=c, label=d)
            for k, r in enumerate(rows):
                v = [r.get(d + suf, np.nan) for d in DETS_ALL]
                v = [x for x in v if np.isfinite(x)]
                if len(v) > 1:
                    ax.plot([min(v), max(v)], [k, k], color="0.6", lw=1.3, zorder=0)
            ax.axvline(1.0, color="0.25", ls="--", lw=1.3)
            ax.set_yticks(y)
            # Channel count on EVERY row. Without it the gated column looks free, when in fact
            # it buys agreement by discarding most of the montage -- the single most misleading
            # thing this figure could do.
            nkey = "_ngated" if suf else "_n"
            # FLAG INCOMPLETE ROWS. Spread is max/min over the detectors PRESENT, so a row that
            # is missing Delphos -- the detector that disagrees most on both trials -- draws a
            # SHORT grey bar and reads as the tightest condition in the panel. That is the
            # single most misleading thing this figure can do, and it is invisible unless said:
            # at 145 Hz `mne p10` shows 1.17 against `none`'s 1.33 purely by absence.
            # tools/run_fill_delphos.py fills these in.
            lab = []
            for r in rows:
                n_det = sum(np.isfinite(r.get(d + suf, np.nan)) for d in DETS_ALL)
                lab.append(f'{LABEL.get(r["profile"], r["profile"])}  '
                           f'[{r.get("Janca" + nkey, 0)}ch]'
                           + ("" if n_det >= 3 else f"  ({n_det} det only)"))
            ax.set_yticklabels(lab, fontsize=7.5)
            for t, r in zip(ax.get_yticklabels(), rows):
                if sum(np.isfinite(r.get(d + suf, np.nan)) for d in DETS_ALL) < 3:
                    t.set_color("#b03a2e")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.25)
            if i == 0:
                ax.set_title(ctitle, fontsize=9.5)
            if j == 1:
                ax.set_ylabel(rec, fontsize=9)
                ax.yaxis.set_label_position("right")
            if i == len(recs) - 1:
                ax.set_xlabel("median per-channel  stim-ON / baseline")
            if i == 0 and j == 0:
                ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Artefact handling: stim/baseline ratio by condition\n"
                 + (note or "clinical montage; both recordings read against their paired "
                            "stim-free baseline. Grey bar = between-detector spread."),
                 fontsize=11)
    # left=0.055: tight_layout sizes the axes from the tick labels of the LEFT column only, so
    # the right column's equally long labels overhang into the left column's axes and the first
    # character of each is clipped off the canvas. Reserving the margin explicitly fixes both.
    fig.tight_layout(rect=(0.055, 0, 1, 0.93))
    # BOTH columns carry row labels, because the channel count differs between them -- that is
    # the whole point of the gated column -- so the gap has to hold a full label, not a tick.
    fig.subplots_adjust(wspace=0.34)
    out = figdir("real") / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=145)
    plt.close(fig)
    print(f"[saved] {out}")


# ---- the channel-selection component -------------------------------------------------------
SHORT = {"no artefact handling": "none", "bad-channel rejection (MNE)": "bad-channel",
         "epoch masking + gradient (150)": "masking\n+ grad 150",
         "+ peri-pulse rejection (-5/+15 ms)": "+ peri-pulse"}


def _short(label):
    base = label.split("  [")[0]
    tag = label.split("  [")[1].rstrip("]") if "  [" in label else ""
    return SHORT.get(base, base) + (f"\n[{tag}]" if tag else "")


def figure_selection(fname="artefact_selection_component_P1.png"):
    """Is a condition removing ARTEFACT, or removing CHANNELS? One panel per trial.

    Solid = each condition scored on its OWN surviving channels, which is what the main figure
    shows and what a reader assumes. Dashed = every condition scored on the channels that
    survive EVERY condition, so the channel set is held constant and only the data can move.
    The shading between them is the CHANNEL-SELECTION COMPONENT: the part of a condition's
    apparent effect that comes from which channels it kept rather than from what it cleaned.

    Built because that component is large and trial-dependent. On P1's 2 Hz trial, holding the
    channel set fixed removes essentially the whole effect -- every condition lands between
    1.74 and 1.84 between-detector spread, against 1.26-1.64 on their own channels. On the
    145 Hz trial it does not: the tight gradient still reaches 1.20 on a fixed 19-channel set
    against 2.05 for no handling. Same rules, opposite conclusions, and only this comparison
    separates them.

    ALL channels measurable in both recordings, NOT the rate-gated subset -- the gate is itself
    a channel selection, and stacking it here would confound the very thing being measured.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    recs = [(r, BASELINE_OF[r]) for r in BASELINE_OF]
    fig, axes = plt.subplots(1, len(recs), figsize=(7.4 * len(recs), 5.6), squeeze=False)
    colors = {"Janca": "#c0392b", "Barkmeier": "#0072b2", "Delphos": "#4a3aa7"}

    for j, (rec, base) in enumerate(recs):
        ax = axes[0][j]
        # The [published] row is EXCLUDED here. It differs from its neighbour in Barkmeier's
        # threshold convention, not in artefact handling, so joining it to the ladder with a
        # line would draw a step that is not a step -- and this figure is about what channel
        # selection does along the ladder, which that row says nothing about.
        specs = [s_ for s_ in _simple_rows(rec, base) if "[published]" not in s_["label"]]
        chans = {}
        for spec in specs:
            o = _row_channels(spec)
            m = np.ones(len(o["Janca"][0]), bool)
            for d in DETS_ALL:
                if d in o:
                    s, b = o[d]
                    m &= np.isfinite(s) & np.isfinite(b) & (b > 0) & (s > 0)
            chans[spec["label"]] = (o, m)
        common = np.ones(len(next(iter(chans.values()))[1]), bool)
        for _o, m in chans.values():
            common &= m

        x = np.arange(len(specs))
        for d in DETS_ALL:
            own, fix = [], []
            for spec in specs:
                o, m = chans[spec["label"]]
                if d not in o:
                    own.append(np.nan); fix.append(np.nan); continue
                s, b = o[d]
                own.append(float(np.median(s[m] / b[m])) if m.sum() else np.nan)
                fix.append(float(np.median(s[common] / b[common])) if common.sum() else np.nan)
            own, fix = np.asarray(own, float), np.asarray(fix, float)
            ax.plot(x, own, "-o", ms=7, lw=2.0, color=colors[d], label=f"{d} own channels")
            ax.plot(x, fix, "--s", ms=6, lw=1.6, color=colors[d], alpha=.65,
                    label=f"{d} fixed {int(common.sum())} ch")
            ok = np.isfinite(own) & np.isfinite(fix)
            ax.fill_between(x[ok], own[ok], fix[ok], color=colors[d], alpha=.13, lw=0)

        ax.axhline(1.0, color="0.25", ls="--", lw=1.3)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.FixedLocator([0.2, 0.3, 0.5, 0.75, 1, 1.5, 2]))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{_short(s['label'])}\n[{int(chans[s['label']][1].sum())}ch]"
                            for s in specs], fontsize=8)
        ax.set_ylabel("stim-ON / baseline  (median per channel, log)")
        ax.set_title(f"({'ab'[j]}) {rec}", fontsize=10, loc="left")
        ax.grid(axis="y", which="major", alpha=.38)
        ax.grid(axis="y", which="minor", alpha=.15, lw=.6)
        ax.legend(fontsize=7, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), frameon=False)

    fig.suptitle("Is the condition removing artefact, or removing channels?\n"
                 "Solid = each condition's own channels; dashed = the channels common to every "
                 "condition. Shading is the channel-selection component.", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    out = figdir("real") / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=145)
    plt.close(fig)
    print(f"[saved] {out}")


def row_mask(o, gate_rate=None):
    """Channels usable by EVERY detector in this row -- the intersection, not the union.

    WHY THE INTERSECTION. Each detector fires on a different set of channels: at 145 Hz under
    grad-150, Janca is measurable on 88, Delphos on 94, Barkmeier on 60. Taking each detector's
    median over its OWN set means the three numbers in a row describe three different montages,
    so part of what is read as "between-detector spread" is a between-channel-set difference.
    The figures reported max() of those counts as the row's channel count, which also disagreed
    with the selection-component figure's intersection (94 vs 56 on that row) -- the same defect
    showing up as a cosmetic mismatch.

    KNOWN BIAS, NOT FIXED HERE: requiring stim_rate > 0 conditions on the outcome. A channel
    that stimulation silenced completely is dropped for having shown the effect most strongly,
    which biases every ratio in this project upward. It predates this function and is applied
    identically to every condition, so it cannot explain a DIFFERENCE between conditions -- but
    it does mean the absolute ratios are conservative.
    """
    m = None
    for d in DETS_ALL:
        if d not in o:
            continue
        s, b = o[d]
        # NO `s > 0` TEST. Rates carry a Haldane continuity correction (rate_confound.HALDANE),
        # so a channel with zero detections during stimulation lands at its own detection floor
        # instead of at 0 -- and stays IN. Requiring s > 0 dropped exactly the channels showing
        # complete suppression: 15 of Barkmeier's 123 on P1 145 Hz, the largest effects in the
        # data. `b > 0` is likewise unnecessary now, but harmless and kept as a guard.
        ok = np.isfinite(s) & np.isfinite(b) & (b > 0)
        if gate_rate is not None:
            ok &= b >= gate_rate
        m = ok if m is None else (m & ok)
    return m if m is not None else np.zeros(0, bool)


HEAD_LABEL = {"no artefact handling": "no artefact\nrejection",
              "bad-channel rejection (MNE)": "whole-channel\nrejection",
              # "rejection" throughout, for consistency with the two rows above. Note these two
              # reject TIME (epochs, peri-pulse windows) while the rows above reject CHANNELS --
              # the axis label and channel counts carry that distinction, not these labels.
              "epoch masking + gradient (150)": "epoch-level\nrejection",
              "+ peri-pulse rejection (-5/+15 ms)": "+ peri-pulse\nrejection"}
HEAD_TAG = {"published": "Barkmeier default", "pinned": "Barkmeier scaling fixed"}
PANEL = {"P1_stim": "ANT 145 Hz", "P1_ANT2_stim": "ANT 2 Hz"}
# Detectors are ANONYMISED in the figure. The results text names them, so the mapping has
# to be stated in the caption or the detector-specific claims cannot be checked against it.
DET_DISPLAY = {"Janca": "Detector 1", "Barkmeier": "Detector 2", "Delphos": "Detector 3"}
# A column counts as CONVERGED when the three medians span <= this, in ratio units.
CONVERGE_TOL = 0.25


def _head_label(label):
    base = label.split("  [")[0]
    tag = label.split("  [")[1].rstrip("]") if "  [" in label else ""
    out = HEAD_LABEL.get(base, base)
    return out + (f"\n({HEAD_TAG.get(tag, tag)})" if tag else "")


def _intervals(v, mode, rng):
    """(thin_lo, thin_hi, thick_lo, thick_hi, median) for one bar.

    TWO DIFFERENT QUESTIONS, and the bar height answers whichever is chosen:

    "pct"  thick = IQR, thin = 10-90th percentile. DISPERSION ACROSS CHANNELS -- how much
           channels differ from each other. It does not shrink with more channels, because it
           is not uncertainty; it is heterogeneity, and it is the honest picture of how varied
           the response is.

    "ci"   thick = 95% bootstrap CI of the MEDIAN, thin = IQR. PRECISION OF THE CENTRE -- how
           well the median is pinned down. Roughly sqrt(n) narrower, so bars stop overlapping
           and "are these two detectors different" becomes readable. It says nothing about
           whether individual channels agree.

    Do not read a "ci" bar as the range of channel responses: on P1 145 Hz epoch masking, Janca's
    IQR is 0.22-0.49 while its CI is 0.30-0.37. Same data, and the second is not a tighter
    measurement of the first.
    """
    p10, p25, p50, p75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
    if mode == "pct":
        return p10, p90, p25, p75, p50
    bs = np.median(rng.choice(v, (2000, v.size), replace=True), axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return p25, p75, lo, hi, p50


YMAX = 5.0        # y-axis cap; bars past it get an arrow. See the note at set_ylim below.


def figure_headline(fname="artefact_headline_P1.png", gate_rate=None, tmax=None,
                    interval="ci"):
    """ONE figure carrying both claims: the spread ACROSS channels, and how much of the
    movement between conditions is channel selection rather than cleaning.

    Filled bar + dot  = the per-channel distribution on that condition's OWN channels
                        (median, IQR, 10-90th percentile). This is the spread.
    Open marker       = the same detector's median on the FIXED channel set common to every
                        condition. The gap to the filled dot is the CHANNEL-SELECTION
                        COMPONENT: movement from which channels survived, not from what was
                        cleaned out of the ones that remain.

    All three detectors in a column use the SAME channels (row_mask), so the spread between
    them is a difference between detectors and not between montages -- see row_mask.

    WHY EVEN "no artefact rejection" IS NOT THE FULL MONTAGE. The common set is limited by the
    most conservative detector. On P1 145 Hz, Janca and Delphos are measurable on 164/163 of
    164 channels but Barkmeier on only 108: it is silent at baseline on 41 channels, and on 15
    more it detects at baseline and NOTHING during stimulation -- total suppression, dropped by
    the stim_rate > 0 requirement. So 107, not 164, and the missing channels are not random.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    from sdc.artefact.rate_confound import per_channel_stems

    recs = [(r, BASELINE_OF[r]) for r in BASELINE_OF]
    rows = {r: _simple_rows(r, b) for r, b in recs}
    widths = [max(len(rows[r]), 1) for r, _ in recs]
    fig, axes = plt.subplots(1, len(recs), figsize=(6.4 + 2.15 * sum(widths), 11.6),
                             squeeze=False, gridspec_kw={"width_ratios": widths})
    colors = {"Janca": "#c0392b", "Barkmeier": "#0072b2", "Delphos": "#4a3aa7"}
    # Janca and Delphos pushed out to make room for the published-Barkmeier bar, which sits
    # just LEFT of the corrected one (ALT_DX). It was at +0.20, i.e. nearly on top of Delphos at
    # +0.24, so the two Barkmeier variants read as bracketing Delphos rather than as a pair.
    off = {"Janca": -0.17, "Barkmeier": +0.015, "Delphos": +0.17}
    ALT_DX = -0.10
    rng = np.random.default_rng(0)      # seeded: the bootstrap must not move between runs
    panel_state = []                    # (ax, span, bars) per panel, for the shared y limits

    for i, (rec, base) in enumerate(recs):
        ax = axes[0][i]
        specs = rows[rec]
        chans = {}
        for spec in specs:
            o = _row_channels(spec, tmax=tmax)
            chans[spec["label"]] = (o, row_mask(o, gate_rate))
        common = None
        for _o, m in chans.values():
            common = m.copy() if common is None else (common & m)

        span = []          # p10/p90 of the SOLID bars, to set this panel's y limits
        col_med = {}       # {column: [median per detector]}, for the converged-column cue
        bars = []          # (x, p10, p90, colour, is_alt) so clipped bars can be marked
        for k, spec in enumerate(specs):
            o, m = chans[spec["label"]]
            for d in DETS_ALL:
                if d not in o or m.sum() < 3:
                    continue
                s, b = o[d]
                v = s[m] / b[m]
                x = k + off[d]
                t_lo, t_hi, k_lo, k_hi, p50 = _intervals(v, interval, rng)
                span += [t_lo, t_hi]
                bars.append((x, t_lo, t_hi, colors[d], False))
                ax.plot([x, x], [t_lo, t_hi], lw=2.4, color=colors[d], zorder=2)
                ax.plot([x, x], [k_lo, k_hi], lw=14.0, color=colors[d], alpha=.5, zorder=3)
                ax.plot([x], [p50], "o", ms=10.5, color=colors[d], mec="white", mew=1.6,
                        zorder=5)
                col_med.setdefault(k, []).append(p50)
                if common is not None and common.sum() >= 3:
                    f50 = float(np.median(s[common] / b[common]))
                    ax.plot([x, x], [p50, f50], lw=1.6, color=colors[d], ls=":", zorder=4)
                    ax.plot([x], [f50], "s", ms=10.5, mfc="white", mec=colors[d], mew=2.2,
                            zorder=6)
            # Barkmeier AS PUBLISHED, translucent, beside its corrected self -- same channels,
            # same masking, only the amplitude normaliser differs. Drawn on one condition only.
            if spec.get("bark_alt"):
                ob, _ = per_channel_stems(*spec["bark_alt"], tmax=tmax)
                if "Barkmeier" in ob and m.sum() >= 3:
                    sb, bb = ob["Barkmeier"]
                    va = sb[m] / bb[m]
                    va = va[np.isfinite(va)]
                    if va.size >= 3:
                        xa = k + off["Barkmeier"] + ALT_DX
                        a_lo, a_hi, j_lo, j_hi, q50 = _intervals(va, interval, rng)
                        c = colors["Barkmeier"]
                        bars.append((xa, a_lo, a_hi, c, True))
                        ax.plot([xa, xa], [a_lo, a_hi], lw=2.0, color=c, alpha=.30, zorder=2)
                        ax.plot([xa, xa], [j_lo, j_hi], lw=9.0, color=c, alpha=.20, zorder=3)
                        ax.plot([xa], [q50], "o", ms=8.5, color=c, alpha=.40, mec="white",
                                mew=1.0, zorder=5)
        ax.axhline(1.0, color="0.25", ls="--", lw=1.3, zorder=1)
        ax.set_yscale("log")
        # Scaled to the SOLID bars. The translucent published-Barkmeier bar runs down to 0.07 on
        # the 2 Hz panel and, left in the autoscale, compressed every real bar into the middle
        # third of the axis to accommodate one reference series. It clips instead, with a marker.
        # THE ANSWER, MARKED. The left-hand columns are the problem and the right-hand ones
        # are the result, but nothing in the geometry says so -- the eye lands on the widest,
        # messiest bars. Columns where the three detectors agree to within CONVERGE_TOL are
        # shaded and labelled with the converged estimate. Chosen from the data rather than
        # hardcoded, so the cue cannot end up on the wrong column after a condition changes.
        # THE FIRST converged column only, not every one that qualifies. Later columns also
        # meet the tolerance -- peri-pulse rejection does on the 2 Hz panel -- but they add
        # nothing to the claim, and shading them pooled their medians into the annotation:
        # the 2 Hz figure read 0.97x because it averaged epoch-level with peri-pulse rather
        # than reporting the column it points at.
        conv = sorted(k for k, v in col_med.items()
                      if len(v) >= 3 and (max(v) - min(v)) <= CONVERGE_TOL)
        if conv:
            k = conv[0]
            ax.axvspan(k - 0.46, k + 0.46, color="#2e7d32", alpha=.07, zorder=-2)
            ax.text(k, 0.975, f"converged to\n{np.median(col_med[k]):.1f}x baseline",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=15, color="#2e7d32", fontweight="bold", linespacing=1.4)
        # y limits are set AFTER the loop, from the first panel, and shared -- see below.
        panel_state.append((ax, span, bars))
        # POWERS OF TWO. The axis is log because this is a ratio -- halving and doubling must
        # look like equal steps -- but ticks at 0.1/0.2/0.3/0.5/0.75/1/1.5/2/3/5 are unevenly
        # spaced on it, so the eye cannot use them to judge distance. Each step here is x2, so
        # tick spacing on the page is uniform and "half the baseline rate" is one gridline down.
        ax.yaxis.set_major_locator(mticker.FixedLocator([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]))
        ax.yaxis.set_minor_locator(mticker.FixedLocator(
            [0.09, 0.18, 0.35, 0.7, 1.4, 2.8, 5.6]))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda t, _: f"{t:g}"))
        ax.set_xticks(np.arange(len(specs)))
        ax.set_xticklabels(
            [f'{_head_label(sp["label"])}\n[{int(chans[sp["label"]][1].sum())} ch]'
             for sp in specs], fontsize=15.5)
        ax.set_xlim(-0.62, len(specs) - 0.38)
        ax.tick_params(axis="y", labelsize=16.5)
        ax.grid(axis="y", which="major", alpha=.38)
        ax.grid(axis="y", which="minor", alpha=.15, lw=.6)
        ax.set_title(f"{chr(97 + i)}) {PANEL.get(rec, rec)}", fontsize=27.5, loc="left")
        # channel count as a SEPARATE right-hand title: it is provenance, not a heading, and at
        # the panel-label size it competed with the panel label for the reader's attention.
        ax.set_title(f"fixed set (squares) = {int(common.sum()) if common is not None else 0} chans",
                     fontsize=17.5, loc="right", color="0.35")
        if i == 0:
            ax.set_ylabel("per-channel stim-ON/baseline ratio", fontsize=17.5)

    # ONE y RANGE FOR BOTH PANELS, taken from the FIRST (145 Hz). Per-panel autoscaling made
    # the two axes silently different -- 145 Hz ran to 0.125 while 2 Hz stopped at 0.5 -- so a
    # bar at the same height in the two panels meant two different ratios and the panels could
    # not be compared by eye, which is most of what a two-panel figure is for. Capped at YMAX;
    # anything outside gets an arrow rather than ending silently at the frame.
    _spans = [sp for _a, sp, _b in panel_state if sp]
    if _spans:
        lo, hi = min(_spans[0]) * 0.80, min(max(_spans[0]) * 1.25, YMAX)
        for ax_, _sp, bars_ in panel_state:
            ax_.set_ylim(lo, hi)
            for xb, b10, b90, cb, is_alt in bars_:
                a = .5 if is_alt else .9
                if b90 > hi:
                    ax_.plot(xb, hi, "^", ms=8, color=cb, alpha=a, clip_on=False, zorder=7)
                if b10 < lo:
                    ax_.plot(xb, lo, "v", ms=8, color=cb, alpha=a, clip_on=False, zorder=7)

    handles = [plt.Line2D([], [], marker="o", ms=10, lw=2.4, ls="-", color=colors[d],
                          label=DET_DISPLAY.get(d, d))
               for d in DETS_ALL]
    handles += [plt.Line2D([], [], marker="o", ms=10, lw=2.4, ls="-", color=colors["Barkmeier"], alpha=.35,
                           label=DET_DISPLAY["Barkmeier"] + ", scaling as published"),
                plt.Line2D([], [], marker="s", ls="", mfc="white", mec="0.3", mew=2.4,
                           ms=12, color="0.3", label="median on the fixed channel set")]
    # SERIES on the left, GLYPH MEANINGS on the right. Putting both in one legend row made a
    # four-column strip of 8.5pt text under the title, and folding the glyph key into the
    # suptitle made the title itself two dense lines. They answer different questions -- which
    # detector, and what the mark means -- so they are separated.
    fig.legend(handles=handles, fontsize=18, ncol=3, loc="upper left",
               bbox_to_anchor=(0.055, 0.985), frameon=False)
    _what = ("bar = 95% CI of the median\nline = IQR across channels" if interval == "ci"
             else "bar = IQR\nline = 10-90th percentile across channels")
    # The open square is a LEGEND entry, not a corner note -- it names a series, whereas this
    # corner text only says what the bar and line geometry mean.
    fig.text(0.985, 0.985, _what,
             ha="right", va="top", fontsize=17, linespacing=1.6, color="0.25")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = figdir("real") / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=145)
    plt.close(fig)
    print(f"[saved] {out}")
