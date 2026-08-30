"""
compare_recordings.py
---------------------
Does each finding hold across CONDITION (stim ON vs OFF) and across PATIENT?

    .venv\\Scripts\\python.exe -m sdc.compare.compare_recordings

Reads `runs/*.npz` ONLY -- no detector is ever re-run, so this is seconds even though the runs
behind it cost ~20 minutes of Delphos.

THE QUESTION IS "IS THE SHAPE THE SAME", NOT "ARE THE VALUES THE SAME"
  P1 has 226 bipolar channels and P5 has 183, with no correspondence between them: different
  implants, different names, different brains. So nothing here is ever compared per channel.
  Every quantity is a per-RECORDING summary, and what is being asked of it is whether the
  ORDERING between detectors survives. Absolute rates differ between patients and that is
  expected, not a finding.

  For the same reason each panel draws its own reference. A rank correlation is judged against
  that recording's reliability ceiling, not against 1.0, because a short stim-ON condition has
  far fewer spikes and therefore a lower ceiling -- an unadjusted rho would read as "the
  detectors agree less under stimulation" when the truth is "there was less data".

COLUMNS
  Six: each patient's baseline, plus their stim file split into ON and OFF. The stim file's
  whole-window number is deliberately NOT a column -- it is a mixture of two conditions in
  whatever ratio that protocol happened to use, so it means nothing on its own.
"""
import itertools
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from seeg._style import RED, BLUE, MUTED, GRID, recessive

from sdc.common import cond
from sdc.common.spike_match import match

from sdc.common.paths import ROOT as HERE   # repo root, not this file's dir --
                                            # see sdc/common/paths.py
RUNS = HERE / "runs"
OUT = HERE / "figures" / "real"

# (npz stem, COND) -> column label. Order is the x-axis order.
# RUN_TAG=_tuned compares the runs made at the BIDS-tuned operating points instead of the
# defaults. Both sets live side by side on disk, and the figure name carries the tag so
# neither can overwrite the other.
TAG = os.environ.get("RUN_TAG", "")
COLUMNS = [("P1_pre", "all", "P1\nbaseline"),
           ("P1_stim", "off", "P1 stim\nOFF"),
           ("P1_stim", "on", "P1 stim\nON"),
           ("P5_pre", "all", "P5\nbaseline"),
           ("P5_stim", "off", "P5 stim\nOFF"),
           ("P5_stim", "on", "P5 stim\nON")]

SELF_BLOCK_SEC = 30.0    # interleaved blocks for the reliability ceiling (evaluate_detectors)
CV_BIN_SEC = 60.0        # Barkmeier's own processing block (spike_statistics Q1b)
TOL_MS = 50.0            # matching tolerance for the timing panel

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}
PAIR_COLORS = {("Janca", "Delphos"): VIOLET, ("Janca", "Barkmeier"): BLUE,
               ("Barkmeier", "Delphos"): "#7a7a7a"}


def load(stem, label):
    """One column: per-channel rates, per-detector spike times, and the selection."""
    z = np.load(RUNS / f"{stem}.npz", allow_pickle=False)
    fs = float(z["fs"])
    n_chan = len(z["names"])
    dets = [str(s) for s in z["detectors"]]
    sel = cond.select(z, label)
    spikes, rates = {}, {}
    for d in dets:
        keep = sel.keep(d)
        idx, ch = z[f"{d}_idx"][keep], z[f"{d}_chan"][keep]
        spikes[d] = [np.sort(idx[ch == c] / fs) for c in range(n_chan)]
        rates[d] = sel.rate([s.size for s in spikes[d]])
    measurable = np.all([np.isfinite(rates[d]) for d in dets], axis=0)
    return dict(z=z, fs=fs, n_chan=n_chan, dets=dets, sel=sel, spikes=spikes, rates=rates,
                measurable=measurable)


def self_rho(col, d):
    """A detector's rank agreement with its own second opinion -- the ceiling for any pair."""
    if col["sel"].bins(SELF_BLOCK_SEC).shape[0] < 4:
        return np.nan
    cnt = np.array([col["sel"].bin_counts(s, SELF_BLOCK_SEC) for s in col["spikes"][d]])
    a, b = cnt[:, 0::2].sum(axis=1), cnt[:, 1::2].sum(axis=1)
    keep = (a + b) > 0
    return spearmanr(a[keep], b[keep]).statistic if keep.sum() >= 3 else np.nan


def pair_rho(col, a, b):
    keep = col["measurable"] & (col["rates"][a] > 0) & (col["rates"][b] > 0)
    if keep.sum() < 3:
        return np.nan
    return spearmanr(col["rates"][a][keep], col["rates"][b][keep]).statistic


def block_cv(col, d):
    """CV of counts in fixed CV_BIN_SEC bins, as a multiple of the Poisson expectation.

    Reported as CV/Poisson, not raw CV: a lower-rate detector has more counting noise, so raw
    CVs are not comparable ACROSS detectors -- which is the entire point of this figure."""
    t = [s for s in col["spikes"][d] if s.size]
    if not t:
        return np.nan
    cnt = col["sel"].bin_counts(np.sort(np.concatenate(t)), CV_BIN_SEC)
    if cnt.size < 4 or cnt.mean() <= 0:
        return np.nan
    return float(cnt.std(ddof=1) / cnt.mean() * np.sqrt(cnt.mean()))


def median_dt(col, a, b):
    """Median |timing offset| between matched detections, pooled over channels."""
    tol = TOL_MS / 1000.0
    offs = []
    for c in range(col["n_chan"]):
        sa, sb = col["spikes"][a][c], col["spikes"][b][c]
        if sa.size and sb.size:
            _, _, o = match(sa, sb, tol)
            if len(o):
                offs.append(np.abs(o))
    return float(np.median(np.concatenate(offs)) * 1000) if offs else np.nan


# ----------------------------------------------------------------------
cols, labels = [], []
for stem, label, disp in COLUMNS:
    if not (RUNS / f"{stem}{TAG}.npz").is_file():
        print(f"[skip] runs/{stem}{TAG}.npz not found")
        continue
    try:
        cols.append(load(stem + TAG, label))
    except SystemExit as e:                # e.g. COND=on on a baseline
        print(f"[skip] {stem} {label}: {e}")
        continue
    labels.append(disp)
    print(f"[load] {stem:<9} {label:<4} {cols[-1]['sel'].describe()}")
if len(cols) < 2:
    raise SystemExit("need at least two recordings -- run compare_spikes.py for more.")

DETS = cols[0]["dets"]
PAIRS = list(itertools.combinations(DETS, 2))
x = np.arange(len(cols))

fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
(ax1, ax2), (ax3, ax4) = axes


# ---- 1. rank agreement, against each column's own ceiling ------------------------------
# Finding 1. Plotted as a FRACTION of the achievable ceiling so a short ON condition is not
# penalised for having fewer spikes.
print(f"\n--- finding 1: rank agreement (rho, and % of that column's ceiling) ---")
ceil = [{d: self_rho(c, d) for d in DETS} for c in cols]
for pa, pb in PAIRS:
    ys, raw = [], []
    for c, ce in zip(cols, ceil):
        r = pair_rho(c, pa, pb)
        k = np.sqrt(max(ce[pa], 0) * max(ce[pb], 0))
        raw.append(r)
        ys.append(r / k if np.isfinite(k) and k > 0 else np.nan)
    ax1.plot(x, ys, "-o", lw=1.8, ms=7, color=PAIR_COLORS.get((pa, pb), MUTED),
             label=f"{pa[:4]}-{pb[:4]}")
    print(f"  {pa[:4]}-{pb[:4]:<10} " + "  ".join(
        f"{r:+.2f}({y:.0%})" if np.isfinite(r) else "  n/a  " for r, y in zip(raw, ys)))
ax1.axhline(0, color=MUTED, lw=1.0)
ax1.set_ylabel("rank agreement / achievable ceiling")
ax1.set_title("(1) Does Janca-Delphos lead everywhere?", fontsize=10, loc="left")
ax1.legend(frameon=False, fontsize=8, ncol=3)


# ---- 2. self-consistency -----------------------------------------------------------------
# Finding 2. A detector cannot agree with another better than it agrees with itself, so a low
# bar in panel 1 next to a high bar here is real disagreement rather than noise.
print(f"\n--- finding 2: self-consistency (interleaved {SELF_BLOCK_SEC:g}s blocks) ---")
for d in DETS:
    ys = [ce[d] for ce in ceil]
    ax2.plot(x, ys, "-o", lw=1.8, ms=7, color=COLORS.get(d, MUTED), label=d)
    print(f"  {d:<11} " + "  ".join(f"{v:.3f}" if np.isfinite(v) else " n/a " for v in ys))
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("rank agreement with itself")
ax2.set_title("(2) Is the disagreement systematic, or just noise?", fontsize=10, loc="left")
ax2.legend(frameon=False, fontsize=8, ncol=3)


# ---- 3. block-to-block activity tracking --------------------------------------------------
print(f"\n--- finding 3: per-{CV_BIN_SEC:g}s count CV, as a multiple of Poisson ---")
for d in DETS:
    ys = [block_cv(c, d) for c in cols]
    ax3.plot(x, ys, "-o", lw=1.8, ms=7, color=COLORS.get(d, MUTED), label=d)
    print(f"  {d:<11} " + "  ".join(f"{v:5.1f}x" if np.isfinite(v) else "  n/a" for v in ys))
ax3.axhline(1.0, color=MUTED, ls="--", lw=1.0, label="Poisson")
ax3.set_ylabel(f"CV of {CV_BIN_SEC:g}s counts / Poisson CV")
ax3.set_title("(3) Does the detector TRACK the recording? (low = flat)", fontsize=10, loc="left")
ax3.legend(frameon=False, fontsize=8, ncol=4)
ax3.text(.01, .02, "columns with <4 whole bins in the condition are omitted",
         transform=ax3.transAxes, fontsize=7, color=MUTED)


# ---- 4. timing ---------------------------------------------------------------------------
print(f"\n--- finding 4: median |dt| between matched detections (ms, tol {TOL_MS:g} ms) ---")
for pa, pb in PAIRS:
    ys = [median_dt(c, pa, pb) for c in cols]
    ax4.plot(x, ys, "-o", lw=1.8, ms=7, color=PAIR_COLORS.get((pa, pb), MUTED),
             label=f"{pa[:4]}-{pb[:4]}")
    print(f"  {pa[:4]}-{pb[:4]:<10} " + "  ".join(
        f"{v:5.1f}" if np.isfinite(v) else "  n/a" for v in ys))
ax4.set_ylabel(f"median |dt| (ms)")
ax4.set_title("(4) Does Barkmeier's lag persist?", fontsize=10, loc="left")
ax4.legend(frameon=False, fontsize=8, ncol=3)

for ax in axes.ravel():
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=.3)
    for i in range(1, len(cols)):
        ax.axvline(i - 0.5, color=GRID, lw=0.8, zorder=0)
    recessive(ax)

fig.suptitle("Does each finding hold across condition and patient?   "
             "(per-recording summaries only -- no channel correspondence between patients)",
             fontsize=11)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / f"compare_recordings{TAG}.png", dpi=130)
print(f"\n[saved] {OUT / 'compare_recordings.png'}")
plt.close(fig)
