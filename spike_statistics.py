"""
spike_statistics.py
-------------------
Two questions the rate/rank views cannot answer, both from `detections.npz` alone:

Q1  DO THE DETECTORS ALTER THE STATISTICAL STRUCTURE OF THE DETECTED SPIKES?
    Not "do they find the same events" but "does the point process they produce have the same
    shape". Three structural measures, per detector:
      * ISI distribution (log axis) + coefficient of variation. CV = 1 is Poisson; CV > 1 is
        clustered/bursty; a hard floor in the ISI histogram is a merge rule, not physiology.
        All three detectors share one floor -- compare_spikes.MERGE_MS, read from the npz and
        drawn on the plot -- so the floor position is a setting, not a finding. What IS a
        finding is the SHAPE just above it: at a 20 ms floor the fraction of intervals under
        50 ms was Janca 15.1%, Barkmeier 2.8%, Delphos 1.6%, because Delphos detects one TF
        blob where Janca marks every polyspike component.
      * burst fraction -- share of spikes with a neighbour within BURST_MS on the same channel.
      * synchrony -- share of spikes with a spike on a DIFFERENT channel within SYNC_MS. A
        detector triggered by a shared artefact looks far more synchronous than one that is not.
    If these differ, per-channel rates from two detectors are not interchangeable even after
    the totals are matched, because they are counting differently-shaped processes.

Q2  WHAT BIN WIDTH IS NEEDED FOR A REASONABLE RATE ESTIMATE?
    Estimated empirically by SPLIT-HALF RELIABILITY: bin the segment, give alternate bins to
    half A and half B, and correlate the per-channel rates from A against B. Two halves of the
    same recording estimating the same channel's rate is the best case -- if they disagree at a
    given bin width, that width cannot support a rate estimate. Reported against the Poisson
    expectation (relative error 1/sqrt(rate*width)), which is the floor: real spike trains are
    burstier than Poisson, so they need LONGER bins than the analytic curve suggests.

    A rate is a per-CHANNEL quantity, so the answer is per-channel too: a 0.8 Hz channel needs
    far less time than a 0.05 Hz one. The curve is therefore reported for rate tiers.

    .venv\\Scripts\\python.exe spike_statistics.py

Reads only detections.npz -- run compare_spikes.py first, ideally on a long segment (600 s+):
with a 60 s window the widest usable bin is a few seconds and the answer is uninformative.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from seeg._style import RED, BLUE, MUTED, GRID, recessive

import cond

HERE = Path(__file__).resolve().parent
# Which run to evaluate. Defaults to the P1 baseline; pass any npz to switch:
#     python spike_statistics.py runs/P1_stim.npz
#     python spike_statistics.py sim_runs/sim_ar16_<hash>_snr8_op.npz
# Figures go to figures/<real|sim>/<run>/ -- one FOLDER per recording, plain filenames
# inside. Routing is read from the npz itself ("simulated" key), not guessed from the
# path, so a sim run can never be mistaken for -- or overwrite -- patient data.
#
# On a stim recording, COND restricts everything below to the stim-ON (or stim-OFF) blocks:
#     COND=on python spike_statistics.py runs/P1_stim.npz
# That subset is GAPPY, and every measure here is gap-sensitive: an ISI measured across an OFF
# block is an artefact of the split, and a bin straddling a boundary is part ON and part OFF.
# cond.py handles both -- nothing below may use np.diff or np.arange on the time axis directly.
NPZ = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs" / "P1_pre.npz"
if not NPZ.is_file():
    raise SystemExit(f"{NPZ} not found -- run compare_spikes.py first.")
_z0 = np.load(NPZ, allow_pickle=False)
TAG = cond.select(_z0).suffix   # ""/"_on"/"_off" -- so a condition split does not overwrite
                                # the all-window figures sitting in the same folder
OUT = HERE / "figures" / ("sim" if "simulated" in _z0.files else "real") / NPZ.stem
OUT.mkdir(parents=True, exist_ok=True)

BURST_MS = 200.0      # Q1: same-channel neighbour distance counted as "in a burst"
SYNC_MS = 50.0        # Q1: cross-channel co-occurrence window
SYNC_SHUFFLES = 3     # Q1: circular-shift repeats for the chance-synchrony baseline
MIN_RATE = 0.02       # Q2: ignore near-silent channels -- no bin width rescues 1 spike/10 min
RATE_TIERS = [(0.02, 0.1, "quiet 0.02-0.1 Hz"), (0.1, 0.3, "mid 0.1-0.3 Hz"),
              (0.3, 99.0, "busy >0.3 Hz")]
TARGET_ERR = 0.20     # Q2: relative error of a single-bin rate estimate we call "reasonable"
ISI_LOG = True        # log x-axis for the ISI histogram. ISIs span 4 decades (the merge floor
                      # to tens of seconds), so on a LINEAR axis everything piles into the
                      # leftmost sliver and neither the floor nor the burst structure is
                      # visible. Set False for a linear axis, which is the honest view of where
                      # the mass actually sits -- and note the two look very different because
                      # a density plotted on a log axis is NOT the density of log(ISI): the
                      # apparent mode moves. ISI_LINEAR_MAX_MS caps the linear view.
ISI_LINEAR_MAX_MS = 3000.0

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}

# ----------------------------------------------------------------------
z = np.load(NPZ, allow_pickle=False)
names = [str(s) for s in z["names"]]
fs = float(z["fs"])
detectors = [str(s) for s in z["detectors"]]
n_chan = len(names)
SEL = cond.select(z)
T = SEL.T           # seconds IN THE CONDITION, not the window length
COND_TAG = "" if SEL.label == "all" else f"  [stim {SEL.label.upper()} only]"
spikes = {d: [np.sort(z[f"{d}_idx"][SEL.keep(d)][z[f"{d}_chan"][SEL.keep(d)] == c] / fs)
              for c in range(n_chan)]
          for d in detectors}
rates = {d: SEL.rate([s.size for s in spikes[d]]) for d in detectors}   # analysable time
print(f"{NPZ.name}: {n_chan} channels, {T:g}s, detectors {detectors}")
print(f"[cond] {SEL.describe()}")


# ----------------------------------------------------------------------
# Q1 -- structure of the point process
# ----------------------------------------------------------------------
def _sync_fraction(det):
    """Share of spikes with a spike on a DIFFERENT channel within +-SYNC_MS.

    Counted as (neighbours in window) - (same-channel neighbours in window) > 0, both from
    searchsorted, so no per-spike Python loop."""
    keep = [s for s in det if s.size]
    if not keep:
        return np.nan
    times = np.concatenate(keep)
    chans = np.concatenate([np.full(s.size, c) for c, s in enumerate(det) if s.size])
    o = np.argsort(times)
    times, chans = times[o], chans[o]
    w = SYNC_MS / 1000
    n_all = (np.searchsorted(times, times + w, "right")
             - np.searchsorted(times, times - w, "left") - 1)      # -1 drops self
    n_same = np.zeros(times.size, int)
    for c in np.unique(chans):
        m = chans == c
        t = times[m]
        n_same[m] = (np.searchsorted(t, t + w, "right")
                     - np.searchsorted(t, t - w, "left") - 1)
    return float((n_all - n_same > 0).mean())


def _structure(det, rng):
    """(ISIs in ms, CV, burst fraction, observed synchrony, CHANCE synchrony).

    The chance level matters more than the observed value: at ~24 spikes/s pooled over 226
    channels, a +-50 ms window catches a couple of unrelated spikes on its own, so a raw
    synchrony of 90% can be entirely coincidence. Chance is estimated by circularly shifting
    each channel independently -- this destroys cross-channel timing while preserving every
    channel's own rate AND its within-channel burst structure."""
    # SEL.isis, not np.diff: an interval running from the last spike before an OFF block to
    # the first one after it is an artefact of the condition split, and it lands in the tail
    # where it inflates the CV rather than looking obviously wrong.
    isi = np.concatenate([SEL.isis(s) for s in det if s.size > 1] or [np.zeros(0)]) * 1000
    cv = isi.std() / isi.mean() if isi.size else np.nan
    n_tot = sum(s.size for s in det)
    n_burst = sum(int((SEL.isis(s) < BURST_MS / 1000).sum()) for s in det if s.size > 1)

    obs = _sync_fraction(det)
    chance = np.mean([_sync_fraction([np.sort((s + rng.uniform(0, T)) % T) for s in det])
                      for _ in range(SYNC_SHUFFLES)])
    return isi, cv, n_burst / max(n_tot, 1), obs, chance

print(f"\n--- Q1: structure of each detector's point process ---")
print(f"{'detector':<11}{'spikes':>8}{'ISI CV':>9}{'median ISI':>12}{'min ISI':>9}"
      f"{f'<{BURST_MS:g}ms':>9}{'sync':>7}{'chance':>8}{'excess':>8}")
struct = {}
rng = np.random.default_rng(0)
for d in detectors:
    isi, cv, burst, sync, chance = _structure(spikes[d], rng)
    struct[d] = isi
    print(f"{d:<11}{sum(s.size for s in spikes[d]):>8}{cv:>9.2f}{np.median(isi):>11.0f}ms"
          f"{isi.min():>7.0f}ms{burst:>9.0%}{sync:>7.0%}{chance:>8.0%}"
          f"{sync - chance:>+8.0%}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
bins = (np.logspace(np.log10(2), np.log10(max(T * 1000 / 4, 1e4)), 60) if ISI_LOG
        else np.linspace(0, ISI_LINEAR_MAX_MS, 60))
for d in detectors:
    axes[0].hist(struct[d], bins=bins, histtype="step", lw=1.4, density=True,
                 color=COLORS.get(d, MUTED), label=f"{d} (CV {struct[d].std()/struct[d].mean():.2f})")
# Annotate the merge ACTUALLY APPLIED, read from the npz. This used to be hardcoded at
# "Janca 120 ms polyspike union", which was wrong twice over once MERGE_MS was retied: the
# value had moved, and the rule is no longer Janca-specific -- all three detectors get the
# same floor (Janca via its internal union, the other two via _merge_close).
_merge = float(z["merge_ms"]) if "merge_ms" in z.files else None
if _merge:
    axes[0].axvline(_merge, color=MUTED, ls="--", lw=1.0)
    axes[0].annotate(f"{_merge:g} ms shared merge\n(all three detectors)",
                     (_merge, axes[0].get_ylim()[1] * 0.6), fontsize=7, color=MUTED, ha="left")
else:
    print("[warn] no merge_ms in the npz (written before provenance was added) -- "
          "the ISI floor is not annotated; re-run compare_spikes.py to restore it")
if ISI_LOG:
    axes[0].set_xscale("log")
axes[0].set_xlabel("inter-spike interval within a channel (ms)")
axes[0].set_ylabel("density")
axes[0].set_title("ISI distribution -- a floor is a merge rule, not physiology",
                  fontsize=9, loc="left")
axes[0].legend(frameon=False, fontsize=8)

# Fano factor vs bin width: 1 = Poisson, >1 = clustered. Structure, not rate.
# The width list is filtered UP FRONT, not skipped inside the loop: `fano` is plotted against
# `widths` and a `continue` would silently shorten one of them. Under COND=on the binding
# constraint is the SEGMENT length, not T -- a 300 s bin does not fit in a 60 s ON block even
# when the ON total is 300 s.
widths = np.array([w for w in (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300)
                   if w <= T / 8 and SEL.bins(w).shape[0] >= 2])
for d in detectors:
    fano = []
    for w in widths:
        cnt = np.array([SEL.bin_counts(s, w) for s in spikes[d]])
        act = cnt[rates[d] > MIN_RATE]
        m = act.mean(axis=1)
        fano.append(np.median(act.var(axis=1) / np.where(m > 0, m, np.nan)))
    axes[1].plot(widths, fano, "o-", color=COLORS.get(d, MUTED), lw=1.3, label=d)
axes[1].axhline(1.0, color=MUTED, ls="--", lw=1.0, label="Poisson (Fano = 1)")
axes[1].set_xscale("log")
axes[1].set_xlabel("bin width (s)")
axes[1].set_ylabel("median Fano factor (var/mean)")
axes[1].set_title("clustering vs timescale", fontsize=9, loc="left")
axes[1].legend(frameon=False, fontsize=8)
for a in axes:
    recessive(a)
fig.suptitle(f"Q1: does the detector change the statistical structure? | {T:g}s, "
             f"{n_chan} channels{COND_TAG}", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / f"eval_spike_structure{TAG}.png", dpi=130)
print(f"[saved] eval_spike_structure{TAG}.png")


# ----------------------------------------------------------------------
# Q1b -- does each detector TRACK the recording's activity block by block?
#
# BLOCK_SEC defaults to 60 s because that is Barkmeier's processing block
# (seeg.spikes.BLOCK_SIZE_MIN = 1 minute). mDetectSpike.m:291 recomputes its peak threshold
# from EACH BLOCK'S OWN DATA -- `thresh = -mean(|fEEG|) - STDCoeff*std(|fEEG|)` -- so a block
# containing more spikes raises its own bar. If that matters, its per-block count should be
# FLAT while detectors with a per-channel background model rise and fall with the recording.
#
# Raw CV is not comparable across detectors: a lower-rate detector has more counting noise.
# The Poisson band (1/sqrt(N)) is drawn so "flat" is judged against sampling noise, and the
# printed ratio is CV / Poisson-CV.
# ----------------------------------------------------------------------
BLOCK_SEC = 60.0
_blk = SEL.bins(BLOCK_SEC)
n_blk = _blk.shape[0]
if n_blk >= 4:

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    print(f"\n--- Q1b: per-{BLOCK_SEC:g}s activity tracking ({n_blk} blocks) ---")
    print(f"{'detector':<11}{'mean/block':>11}{'CV':>8}{'CV/Poisson':>12}   per-block counts")
    for d in detectors:
        cnt = SEL.bin_counts(np.sort(np.concatenate([s for s in spikes[d] if s.size])),
                             BLOCK_SEC)
        cv = cnt.std(ddof=1) / cnt.mean() if cnt.mean() else np.nan
        ratio = cv * np.sqrt(cnt.mean()) if cnt.mean() else np.nan
        col = COLORS.get(d, MUTED)
        axes[0].plot(np.arange(1, n_blk + 1), cnt, "-o", color=col, lw=1.6, ms=5, label=d)
        axes[1].plot(np.arange(1, n_blk + 1), cnt / cnt.mean(), "-o", color=col, lw=1.6, ms=5,
                     label=f"{d}  CV {cv:.3f} ({ratio:.1f}x Poisson)")
        print(f"{d:<11}{cnt.mean():>11.0f}{cv:>8.3f}{ratio:>12.1f}   "
              + " ".join(f"{v}" for v in cnt))
    # sampling-noise band for the busiest detector, as the most forgiving reference
    _n = max(SEL.bin_counts(np.sort(np.concatenate([s for s in spikes[d] if s.size])),
                            BLOCK_SEC).mean() for d in detectors)
    axes[1].axhspan(1 - 1 / np.sqrt(_n), 1 + 1 / np.sqrt(_n), color=GRID, alpha=.5, lw=0,
                    zorder=0, label="Poisson counting noise")
    axes[1].axhline(1.0, color=MUTED, lw=0.8, ls="--")
    axes[0].set_ylabel("detections in block")
    axes[1].set_ylabel("detections / that detector's mean")
    axes[0].set_title(f"Per-{BLOCK_SEC:g}s detections", fontsize=9, loc="left")
    axes[1].set_title("Same, normalised -- does the detector TRACK the recording?",
                      fontsize=9, loc="left")
    for a in axes:
        a.set_xlabel(f"block ({BLOCK_SEC:g} s each)")
        a.grid(alpha=.3)
        a.legend(frameon=False, fontsize=8)
        recessive(a)
    fig.suptitle(f"Block-to-block activity tracking | {T:g}s, {n_chan} channels"
                 f"{COND_TAG}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"eval_block_stability{TAG}.png", dpi=130)
    print(f"[saved] eval_block_stability{TAG}.png")


# ----------------------------------------------------------------------
# Q2 -- how long must a bin be to estimate a rate?
# ----------------------------------------------------------------------
print(f"\n--- Q2: how far is a single {'w'}-second estimate from the channel's true rate? ---")
print(f"    (each bin is ONE estimate from exactly w seconds, scored against that channel's"
      f" long-run rate; target: median error <= {TARGET_ERR:.0%})")
# NOTE ON AN EARLIER VERSION: this used to correlate odd-numbered bins against even-numbered
# bins. That is not a precision measurement -- both halves use T/2 seconds NO MATTER WHAT w
# IS, so widening the bin cannot improve it; the curve only fell as the two halves drifted
# apart in time. It measured stationarity. Here each estimate uses exactly w seconds, so the
# curve answers the question actually asked.
fig, axes = plt.subplots(1, len(RATE_TIERS), figsize=(5.0 * len(RATE_TIERS), 4.4),
                         squeeze=False, sharey=True)
widths_q2 = np.array([w for w in (1, 2, 5, 10, 20, 30, 60, 120, 300)
                      if w <= T / 3 and SEL.bins(w).shape[0] >= 2])
for ax, (lo_r, hi_r, tier) in zip(axes[0], RATE_TIERS):
    for d in detectors:
        sel = (rates[d] >= lo_r) & (rates[d] < hi_r)
        if sel.sum() < 3:
            continue
        err = []
        for w in widths_q2:
            cnt = np.array([SEL.bin_counts(s, w) for s in spikes[d]])[sel]
            ref = rates[d][sel][:, None]                     # long-run rate per channel
            err.append(np.median(np.abs(cnt / w - ref) / ref))
        ax.plot(widths_q2, err, "o-", lw=1.3, color=COLORS.get(d, MUTED),
                label=f"{d} (n={sel.sum()})")
        ok = [w for w, x in zip(widths_q2, err) if np.isfinite(x) and x <= TARGET_ERR]
        print(f"  {tier:<20} {d:<10} median error <={TARGET_ERR:.0%} from "
              f"{(str(int(min(ok))) + ' s') if ok else f'not reached by {widths_q2.max():g} s'}")
    # Poisson floor for the tier's midpoint rate: relative error = 1/sqrt(rate*w)
    mid = np.mean([lo_r, min(hi_r, 1.0)])
    ax.plot(widths_q2, 1 / np.sqrt(mid * widths_q2), color=MUTED, ls=":", lw=1.2,
            label=f"Poisson floor @ {mid:.2f} Hz")
    ax.axhline(TARGET_ERR, color=MUTED, ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bin width (s)")
    ax.set_title(tier, fontsize=9, loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    recessive(ax)
axes[0][0].set_ylabel("median |estimate - long-run rate| / rate")
fig.suptitle(f"Q2: bin width needed for a per-channel rate estimate{COND_TAG} | {T:g}s "
             f"(dashed = {TARGET_ERR:.0%} error; dotted = Poisson floor)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / f"eval_bin_width{TAG}.png", dpi=130)
print(f"[saved] eval_bin_width{TAG}.png")

# Analytic reference: a Poisson channel needs rate*width counts for a given relative error.
print(f"\n  Poisson floor (relative error 1/sqrt(rate*width)) -- best case, real trains are worse:")
for r in (0.05, 0.1, 0.3, 0.8):
    print(f"    {r:>4.2f} Hz channel: +-20% needs {25 / r:6.0f} s per bin, "
          f"+-10% needs {100 / r:6.0f} s")
