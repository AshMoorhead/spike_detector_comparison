"""
sdc.artefact.rate_confound
--------------------------
WHY does channel gating buy agreement at 2 Hz but not at 145 Hz?

    .venv\\Scripts\\python.exe -m sdc.artefact.rate_confound

The 2x2 matrix shows gating to `baseline rate >= 6 det/chan-min` collapsing between-detector
spread on the 2 Hz trial (4.29 -> 1.64) while making it slightly WORSE on the 145 Hz trial
(1.33 -> 1.42). Read on its own that looks like a property of the gate, and the natural reading
-- "continuous stimulation needs channel selection" -- names the trial rather than a mechanism.

This module tests the mechanism instead. If gating works by removing channels where residual
artefact dominates the ratio, then disagreement must be CONCENTRATED IN LOW-RATE CHANNELS,
because a quiet channel has few genuine spikes for a fixed count of artefact-driven false
positives to be divided by. If instead disagreement is spread evenly across activity, gating
cannot help however many channels it discards, and any improvement it shows is noise.

Measured on P1, condition `none` (no artefact handling at all), by baseline-rate quartile:

    145 Hz   Q1 1.21  Q2 1.39  Q3 1.31  Q4 1.31     flat -- and small at every activity level
      2 Hz   Q1 7.18  Q2 4.85  Q3 4.99  Q4 1.82     steep -- disagreement lives in quiet channels

READ THE MAGNITUDES, NOT ONLY THE SHAPES. The 145 Hz gate curve is not flat -- it drifts from
1.33 down to ~1.10 across gate 0-20 -- so "gating does nothing at 145 Hz" would be false. What
is true is that it starts at 1.33, RISES to ~1.54 around gate 1-4 before falling, and never has
much to remove: the whole range it moves through is smaller than the amount 2 Hz sheds in its
first two gate steps. A gate that must be raised non-monotonically to help is not isolating a
contaminated population; it is reshuffling which channels dominate the median. (The same
non-monotonicity is why `matrix_report.GATE_RATE` is 6 and not 1-3.)

So the answer is NOT "because it is continuous". It is because at 2 Hz the artefact survives
rejection -- `pStim` integrates +-5 Hz around the fundamental and a 2 Hz train is a harmonic
comb, so the rule is structurally blind -- and surviving artefact lands hardest where there is
least true signal to dilute it. Continuity is why the rejection fails (there is no clean time
within the file to normalise against); the rate dependence is what makes gating the remedy.
The prediction that follows, and that this module does not yet test: fix the 2 Hz rejection and
the gate should stop mattering there too.

The direction of each detector's error is visible in the same table and is NOT shared: at 2 Hz
Q1, Janca reads 2.63 and Delphos 3.68 while Barkmeier reads 0.51. Two detectors inflate on the
pulse artefact and one deflates -- see seeg.spikes.scale_denominator for why Barkmeier deflates.
At 145 Hz Q1 all three inflate together (1.41 / 1.31 / 1.59), which is a different failure: a
shared bias that agreement cannot detect, and which gating would not fix either.
"""
import numpy as np

from sdc.common.paths import RUNS, figdir
from sdc.common import cond as _cond
from sdc.artefact.blocks import baseline_rates

DETS = ("Janca", "Barkmeier", "Delphos")
COLOR = {"Janca": "#c0392b", "Barkmeier": "#0072b2", "Delphos": "#4a3aa7"}
MARK = {"Janca": "o", "Barkmeier": "s", "Delphos": "D"}
# (stim, baseline, label). One low-frequency continuous trial and one high-frequency
# intermittent one -- the whole comparison is between these two regimes.
TRIALS = (("P1_stim", "P1_pre", "145 Hz, intermittent"),
          ("P1_ANT2_stim", "P1_ANT2_pre", "2 Hz, continuous"))
GATE_MARK = 6.0


def per_channel(rec, base, profile):
    """{detector: (stim_rate, baseline_rate)} per channel, det/chan-min. Rates, never counts.

    Counts are activity-weighted and a handful of busy channels dominate them; every claim in
    this module is about the DISTRIBUTION across channels, so it must be built from per-channel
    rates or it is measuring something else entirely.
    """
    return per_channel_stems(f"{rec}_qc{profile}", f"{base}_qc{profile}")


def per_channel_stems(stim_stem, base_stem, tmax=None):
    """Same, addressed by full run STEM rather than recording+profile.

    Needed because the interesting rows are no longer all plain profiles: pulse-rejected runs
    (`..._prm5p15_qcnone`) and the Barkmeier-scale-corrected run (`..._bscale_qcdynrg1000`) carry
    their variant in the middle of the stem, and a `<rec>_qc<profile>` template cannot name them.
    """
    z = np.load(RUNS / f"{stim_stem}.npz", allow_pickle=False)
    names = [str(s) for s in z["names"]]
    # STIMULATION-ON TIME ONLY, against the whole stim-free baseline.
    #
    # This used to be "all", and that made the two trials incomparable while looking uniform.
    # P1's 145 Hz trial is INTERMITTENT -- 332 s ON out of 1097 -- so "all" averaged the
    # stimulated period with 765 s of unstimulated time, pulling every detector toward 1.0 and
    # toward each other. It reported between-detector spread of 1.05 under epoch masking where
    # the stimulated time alone gives 2.16, and 1.42 unmasked where ON alone gives 4.01. The
    # 2 Hz trial is CONTINUOUS, so its "all" was already all-ON and its numbers never had the
    # dilution -- the two panels were being read differently while the code claimed otherwise.
    #
    # Selecting ON is the single rule that is correct for both: on a continuous trial ON is the
    # whole file, so this is a no-op there, and on an intermittent one it removes the dilution.
    # It also stops epoch masking flattering itself -- masking removes 35% of ON time against
    # 1.7% of OFF time at 145 Hz, so a whole-file ratio is computed on a stim file whose
    # stimulated portion has been preferentially deleted.
    #
    # The BASELINE stays whole-file: it is stim-free, so it has no ON period to select and
    # `baseline_rates` reads all of it.
    # `tmax` truncates the STIM recording only -- the baseline is a separate file with no
    # stimulation in it, so there is nothing there to cut and cutting it would just shrink the
    # denominator's sample. Used to drop a terminal ON burst whose artefact dominates a
    # detector; see exposure.period_course's tmax note.
    sel = _cond.select(z, "on", tmax=tmax)
    if not np.any(np.asarray(z["on_per_sec"], bool)):
        raise SystemExit(
            f"{stim_stem}: no stimulation-ON seconds, so an ON-only rate is undefined. "
            f"A baseline recording cannot be used as the stim side of this comparison.")
    b = _baseline_rates_corrected(base_stem, names)
    have = [str(s) for s in z["detectors"]]
    out = {}
    for d in DETS:
        if d not in have:
            continue
        c = np.bincount(z[f"{d}_chan"][sel.keep(d)], minlength=len(names))
        out[d] = (sel.rate(c + HALDANE) * 60.0, b[d])
    return out, names


# Continuity correction, added to BOTH counts before forming a rate ratio (Haldane-Anscombe).
#
# WHY IT IS HERE. Consumers used to require stim_rate > 0, which silently DROPPED every channel
# where a detector fired at baseline and not once during stimulation -- i.e. complete
# suppression, the largest effect in the data. On P1 145 Hz that was 15 of Barkmeier's 123
# measurable channels, and because the channel set is intersected across detectors it pulled the
# whole comparison down to 107 channels from 164. Excluding the strongest responders biases
# every ratio upward, and it does so worst for the most conservative detector.
#
# The alternative to a correction is a ratio of 0, which is honest but cannot be plotted on a
# log axis, cannot be averaged sensibly, and treats "no detections in 332 s" as infinitely
# stronger evidence than "one detection". +0.5 makes a zero-numerator channel land at the
# DETECTION FLOOR for its own exposure -- the smallest effect the recording could have resolved
# -- which is the conservative reading and keeps the channel in.
#
# It is applied to both numerator and denominator so it cannot bias the ratio in either
# direction; on channels with many detections it is negligible (0.5 against hundreds).
HALDANE = 0.5


def _baseline_rates_corrected(base_stem, names):
    """Baseline det/chan-min with the same continuity correction as the stim side.

    Not `blocks.baseline_rates`: that one is shared with the gating code, applies no correction,
    and maps unmeasurable channels to 0.0 so they fail a rate gate. Here an unmeasurable channel
    must stay NaN -- it has no baseline, so it has no ratio, which is different from having a
    low one.
    """
    zb = np.load(RUNS / f"{base_stem}.npz", allow_pickle=False)
    if [str(s) for s in zb["names"]] != list(names):
        raise SystemExit(f"{base_stem}: channel names differ from the stim run.")
    selb = _cond.select(zb, "all")
    have = [str(s) for s in zb["detectors"]]
    out = {}
    for d in DETS:
        if d not in have:
            continue
        c = np.bincount(zb[f"{d}_chan"][selb.keep(d)], minlength=len(names)).astype(float)
        r = selb.rate(c + HALDANE) * 60.0
        # A CHANNEL WITH NO BASELINE DETECTIONS HAS NO DENOMINATOR, so it is NaN -- not a
        # floor. The correction has to be one-sided here, and this is why: on the numerator a
        # zero means "the effect was total", which is a measurement; on the denominator a zero
        # means "this detector never found anything here", which is not a baseline to compare
        # against. Flooring it instead manufactures huge ratios out of nothing -- letting the
        # 41 channels where Barkmeier is silent at baseline into P1's 145 Hz `none` row moved
        # its median from 0.542 to 1.290 and inverted the direction of the result.
        out[d] = np.where(c > 0, r, np.nan)
    return out


def _spread_at(o, gate):
    """Between-detector spread over channels whose BASELINE rate clears `gate`, and the count.

    The gate is read off the BASELINE recording, never the stim one: a channel that stimulation
    genuinely silenced would fail a stim-side gate and be dropped for having shown the effect.
    """
    vals = []
    n = 0
    for d in DETS:
        if d not in o:
            continue
        s, b = o[d]
        g = np.isfinite(s) & np.isfinite(b) & (b > 0) & (s > 0) & (b >= gate)
        if not g.sum():
            return np.nan, 0
        vals.append(float(np.median(s[g] / b[g])))
        n = max(n, int(g.sum()))
    if len(vals) < 2 or min(vals) <= 0:
        return np.nan, n
    return max(vals) / min(vals), n


def report(profile="none"):
    for rec, base, lab in TRIALS:
        o, _ = per_channel(rec, base, profile)
        jb = o["Janca"][1]
        ok = np.isfinite(jb) & (jb > 0)
        qs = np.percentile(jb[ok], [0, 25, 50, 75, 100])
        print(f"=== {lab}  ({rec}, condition {profile!r}) ===")
        print(f"  {'baseline-rate bin':<22}{'nch':>5}"
              + "".join(f"{d:>11}" for d in DETS) + f"{'spread':>9}")
        for i in range(4):
            lo, hi = qs[i], qs[i + 1]
            m = ok & (jb >= lo) & ((jb <= hi) if i == 3 else (jb < hi))
            row = []
            for d in DETS:
                if d not in o:
                    row.append(np.nan)
                    continue
                s, b = o[d]
                g = m & np.isfinite(s) & np.isfinite(b) & (b > 0) & (s > 0)
                row.append(float(np.median(s[g] / b[g])) if g.sum() else np.nan)
            v = [x for x in row if np.isfinite(x)]
            sp = max(v) / min(v) if len(v) > 1 and min(v) > 0 else np.nan
            print(f"  Q{i + 1} [{lo:5.1f},{hi:7.1f}]{int(m.sum()):>7}"
                  + "".join(f"{x:>11.3f}" for x in row) + f"{sp:>9.2f}")
        print()


def figure(profile="none", fname="rate_confound_P1.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), squeeze=False)
    gates = np.linspace(0, 20, 81)
    for i, (rec, base, lab) in enumerate(TRIALS):
        o, _ = per_channel(rec, base, profile)

        ax = axes[i][0]
        for d in DETS:
            if d not in o:
                continue
            s, b = o[d]
            g = np.isfinite(s) & np.isfinite(b) & (b > 0) & (s > 0)
            ax.scatter(b[g], s[g] / b[g], s=15, alpha=.55, color=COLOR[d],
                       marker=MARK[d], linewidths=0, label=d)
        ax.axhline(1.0, color="0.25", ls="--", lw=1.2)
        ax.axvline(GATE_MARK, color="#e07b39", ls=":", lw=1.6)
        ax.text(GATE_MARK, ax.get_ylim()[1], f" gate {GATE_MARK:g}", fontsize=7.5,
                color="#e07b39", va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("baseline rate (det/chan-min)")
        ax.set_ylabel("stim / baseline")
        ax.set_title(f"{lab} — per channel", fontsize=10, loc="left")
        ax.grid(alpha=.25)
        if i == 0:
            ax.legend(fontsize=8, loc="lower left")

        ax = axes[i][1]
        sp, nn = zip(*[_spread_at(o, g) for g in gates])
        sp, nn = np.asarray(sp, float), np.asarray(nn, float)
        ax.plot(gates, sp, lw=2.0, color="#2c3e50")
        ax.axvline(GATE_MARK, color="#e07b39", ls=":", lw=1.6)
        ax.axhline(1.0, color="0.6", ls="--", lw=1.0)
        ax.set_xlabel("channel gate: baseline rate >= (det/chan-min)")
        ax.set_ylabel("between-detector spread", color="#2c3e50")
        ax.set_title(f"{lab} — what gating buys, and what it costs", fontsize=10, loc="left")
        ax.grid(alpha=.25)
        # Channels retained on the same x axis. Without it the left-hand fall in spread looks
        # free, when most of it is paid for by discarding the montage.
        ax2 = ax.twinx()
        ax2.plot(gates, nn, lw=1.4, ls="--", color="#7f8c8d")
        ax2.set_ylabel("channels retained", color="#7f8c8d")
        ax2.set_ylim(0, max(nn.max(), 1) * 1.15)
        # Anchor the curve at two readable points. The claim is about HOW MUCH disagreement the
        # gate removes, and a y axis auto-scaled per panel hides that the two panels differ by
        # an order of magnitude in what there was to remove.
        for gx in (0.0, GATE_MARK):
            s0, n0 = _spread_at(o, gx)
            if np.isfinite(s0):
                ax.annotate(f"{s0:.2f}x\n{n0} ch", (gx, s0), textcoords="offset points",
                            xytext=(6, 8), fontsize=8, color="#2c3e50")

    fig.suptitle(
        "Is channel gating needed because stimulation is continuous, or because artefact "
        "survives?\n"
        f"P1, condition {profile!r} (no artefact handling). At 2 Hz disagreement is concentrated "
        "in quiet channels and the gate removes it; at 145 Hz there is little to remove at any "
        "activity level.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = figdir("real") / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=145)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    report()
    figure()
