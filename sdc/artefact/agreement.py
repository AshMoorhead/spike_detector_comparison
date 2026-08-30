"""
sdc.artefact.agreement
----------------------
How many channels does a confident ON/OFF answer actually need?

    .venv\\Scripts\\python.exe -m sdc.artefact.agreement

WHY THIS RUNS BEFORE ANY MASK IS TOUCHED
  Every artefact threshold trades contamination against exclusion: mask harder and fewer
  channels survive. That trade is only decidable once "how many channels is enough" is a
  number rather than an opinion, because otherwise there is a trivial winner -- mask until
  only pristine channels remain, watch the detectors agree, and report the agreement. That
  path ends at a ratio of 1.0 with no data left, which is the null dressed as a result.

  So: subsample channels, and measure directly how the confidence interval behaves as the
  channel count falls. The output is the floor the mask ladder has to respect.

  P1_stim currently has 118 channels usable by all three detectors and P5_stim has 109, out
  of 226 and 183 implanted. Masking has ALREADY cost roughly half the implant in both. This
  says whether what is left is comfortable or marginal.

METHOD
  For each n: draw many channel subsets of size n WITHOUT replacement (a real implant has no
  duplicate channels), bootstrap WITHIN each subset to get that subset's CI, then report
  across subsets:

    width       median CI width. On log scale for ratios, so an interval is a fold-range and
                n is comparable between detectors with very different rates.
    power       fraction of subsets whose CI excludes the null. This is the operational
                number -- "with n channels, how often would we have called this effect?"

  Nested resampling is the honest way to ask this: the outer draw is "a smaller implant", the
  inner one is "the uncertainty that implant would have had". Collapsing them into a single
  bootstrap would answer a different and easier question.

CAVEAT worth keeping in view: the subsets are drawn from the channels that SURVIVED masking,
so this measures the precision available from n typical surviving channels. Masking harder
does not remove a random n -- it removes the contaminated ones first, which are also often
the ones nearest the pathology. So the curve is optimistic about aggressive masking, and the
number it gives is a floor, not a target.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from seeg._style import RED, BLUE, MUTED, recessive

from sdc.common.paths import figdir
from sdc.artefact import ratio_metrics as rm

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}

N_SUBSET = 300        # channel subsets per n
N_BOOT = 400          # bootstrap draws within each subset
GRID = [8, 12, 16, 24, 32, 48, 64, 80, 100, 120]
SEED = 0


def per_channel(a, kind, zero="clip"):
    """The per-channel quantity whose MEDIAN is the estimator.

    Reducing an estimator to "median of this vector" is what makes the nested resampling
    affordable -- everything below is then fancy indexing plus np.median, and no estimator
    function is called inside the inner loop at all.
    """
    if kind == "log_ratio":
        on, off = a["on_rate"].copy(), a["off_rate"]
        z = a["on_count"] == 0
        if zero == "clip":
            on[z] = 0.5 / np.maximum(a["on_sec"][z], 1e-9) * 60.0
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.log10(on / off)
        return q[np.isfinite(q)]
    if kind == "rate_diff":
        return a["on_rate"] - a["off_rate"]
    raise ValueError(kind)


def curve(q, grid=GRID, n_subset=N_SUBSET, n_boot=N_BOOT, seed=SEED):
    """(n, median CI width, P(CI excludes 0)) for the median of `q`.

    `q` is on a scale whose null is 0 -- log ratio or rate difference -- so one code path
    covers both and "excludes the null" is one comparison.
    """
    rng = np.random.default_rng(seed)
    N = q.size
    ns, widths, power = [], [], []
    for n in [g for g in grid if g <= N]:
        sub = np.array([rng.choice(N, size=n, replace=False) for _ in range(n_subset)])
        qs = q[sub]                                            # (n_subset, n)
        bi = rng.integers(0, n, size=(n_subset, n_boot, n))
        boots = np.median(qs[np.arange(n_subset)[:, None, None], bi], axis=2)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=1)
        ns.append(n)
        widths.append(float(np.median(hi - lo)))
        power.append(float(np.mean((lo > 0) | (hi < 0))))
    return np.array(ns), np.array(widths), np.array(power)


def run(recs=("P1_stim", "P5_stim"), kind="log_ratio"):
    """How the BETWEEN-CHANNEL component of the uncertainty scales with channel count.

    This deliberately reports width only, not power. A power curve would have to be built on
    the channel bootstrap, and nullcheck.py showed that interval is miscalibrated -- it fires
    on stim-free data far more often than its nominal 5% -- because it cannot see the
    temporal component at all. Plotting power off it would put a confident-looking number on
    a yardstick known to be wrong.

    Width still means something on its own: it is a real measurement of how much of the
    uncertainty comes from having a finite number of channels, and comparing it against the
    across-split spread in nullcheck.py is what shows that channels are the SMALL part.
    """
    fig, axes = plt.subplots(1, len(recs), figsize=(6.2 * len(recs), 4.4), squeeze=False)
    print(f"\n=== between-channel CI width vs channel count  [{kind}] ===")
    out_w = {}
    for j, rec in enumerate(recs):
        ax = axes[0][j]
        p = rm.load(rec)
        print(f"\n{rec}: {p.n} channels currently usable by all three detectors")
        for d in p.det:
            q = per_channel(p.det[d], kind)
            n, w, _pw = curve(q)
            c = COLORS.get(d, MUTED)
            ax.plot(n, w, "-o", ms=4, lw=1.4, color=c, label=d)
            out_w[(rec, d)] = (n, w)
            print(f"   {d:<10} ratio {10 ** float(np.median(q)):.3f}   "
                  f"width at n={n[-1]}: {w[-1]:.3f} log10 = {10 ** w[-1]:.2f}x")

        ax.axvline(p.n, color="0.55", ls=":", lw=1.2)
        recessive(ax)
        ax.grid(alpha=.3)
        # axes-fraction, not data coordinates: the y range is set by whichever detector has
        # the widest CI and an annotation pinned to a data value lands somewhere different
        # on every recording.
        ax.annotate(f"all {p.n} surviving", (p.n, 0.98),
                    xycoords=("data", "axes fraction"), fontsize=7.5,
                    color="0.4", ha="right", va="top", rotation=90)
        ax.set_title(f"({'ab'[j]}) {rec}", fontsize=10, loc="left")
        ax.set_ylabel("median 95% CI width (log10 ratio)")
        ax.set_xlabel("channels retained after artefact masking")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Between-channel uncertainty vs channel count. Dotted line = what survives "
                 "masking today.\nThis is only the channel component -- the temporal one "
                 "(which minutes fell in an ON block) is larger and is not reduced by "
                 "adding channels.", fontsize=10)
    fig.tight_layout()
    out = figdir("real") / f"channels_needed_{kind}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")
    return out_w


if __name__ == "__main__":
    run(recs=tuple(sys.argv[1:]) or ("P1_stim", "P5_stim"))
