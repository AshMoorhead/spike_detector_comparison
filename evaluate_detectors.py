"""
evaluate_detectors.py
---------------------
Evaluation plots for the detector comparison. Reads ONLY `detections.npz` (written by
compare_spikes.py) -- so redrawing a figure never re-runs a detector, and never pays the
~5 min Delphos call.

    .venv\\Scripts\\python.exe compare_spikes.py      # once, to produce detections.npz
    .venv\\Scripts\\python.exe evaluate_detectors.py  # as often as you like

Three views, each answering a different question about whether the detectors agree on WHERE
the epileptic activity is (not on individual events -- compare_spikes.py's Jaccard covers
that, and it is brutal towards detectors that resolve bursts differently):

  1. eval_rate_scatter.png -- per-channel mean rate over the whole segment, detector vs
     detector, against y=x. Answers "same rate, or a constant gain, or neither?".
  2. eval_rank_scatter.png -- the same but on channel RANK (1 = spikiest), with Spearman rho
     and top-k overlap. Answers "is the spikiest channel the same channel?", which is what
     actually matters if these rates are used to localise.
  3. eval_binned_rates.png -- per-channel distribution of BIN_SEC-second bin rates, one panel
     per detector, channels in a shared order (biggest first by ORDER_BY). Answers "is a
     channel's rate steady or driven by one burst?" and shows how that differs per detector.

The rate/rank views are burst-insensitive by construction: a detector that merges a polyspike
run into one event loses a few counts, but the channel ORDERING survives -- which is why rank
is the fairer basis for comparison than raw counts.
"""
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

from seeg._style import RED, BLUE, MUTED, GRID, recessive

HERE = Path(__file__).resolve().parent
NPZ = HERE / "detections.npz"

BIN_SEC = 20.0        # bin width for view 3; see the guard below -- 60 s / 20 s = 3 bins only
TOP_K = 10            # "is the spikiest channel the same channel" -- overlap of the top K
LABEL_EVERY = 8       # x-tick density in view 3 (226 channels will not all fit)
LOG_AXES = False      # view 1 on log-log; useful when most channels sit near zero
# View 2 (rank) only.
# ACTIVE_ONLY drops channels with no detections: no rate, so no rank. Cheap and safe -- on the
# 600 s P1 baseline that is 2 channels for Janca/Delphos and 27 for Barkmeier.
# DROP_TIES is OFF by default, and the measurement is why. Spearman already handles ties
# correctly (average ranks); dropping them only helps against one huge tied block, which this
# data does not have. What it does instead: 226 channels hold only ~115 DISTINCT counts (11
# channels at exactly 1 spike, 10 at 5, ...), so requiring a unique count in BOTH detectors
# cut 199 active channels to 46 -- and those 46 are the sparse high-count tail, because that
# is where counts stop colliding. That is survivorship bias towards the busiest channels, not
# a cleaner subset. Turn it on only if a single tie block is genuinely dominating rho.
RANK_ACTIVE_ONLY = True
RANK_DROP_TIES = False

VIOLET = "#4a3aa7"    # matches compare_spikes.py
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}


# ----------------------------------------------------------------------
# Load (output only -- nothing here imports compare_spikes)
# ----------------------------------------------------------------------
if not NPZ.is_file():
    raise SystemExit(f"{NPZ.name} not found -- run compare_spikes.py first.")
z = np.load(NPZ, allow_pickle=False)
names = [str(s) for s in z["names"]]
fs = float(z["fs"])
T = float(z["seconds"])
detectors = [str(s) for s in z["detectors"]]
n_chan = len(names)

# per detector: list of per-channel spike-time arrays (seconds from window start)
spikes = {}
for d in detectors:
    idx, chan = z[f"{d}_idx"], z[f"{d}_chan"]
    spikes[d] = [np.sort(idx[chan == c] / fs) for c in range(n_chan)]

rates = {d: np.array([s.size for s in spikes[d]]) / T for d in detectors}   # Hz per channel
print(f"{NPZ.name}: {n_chan} channels, {T:g}s at {fs:g} Hz, detectors {detectors}")
for d in detectors:
    print(f"  {d:<10} total {sum(s.size for s in spikes[d]):5d}  "
          f"mean/channel {rates[d].mean():.3f} Hz  busiest {names[int(rates[d].argmax())]} "
          f"({rates[d].max():.2f} Hz)")

PAIRS = list(itertools.combinations(detectors, 2))
if not PAIRS:
    raise SystemExit("Need at least two detectors in detections.npz to compare.")


def _pair_fig(title, w=5.2):
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(w * len(PAIRS), w + 0.4), squeeze=False)
    fig.suptitle(title, fontsize=11)
    return fig, axes[0]


def _identity(ax, lo, hi):
    """y=x plus a shaded 2x band -- deviation from the LINE is bias, deviation from the
    BAND is a channel the two detectors disagree about by more than a factor of two."""
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, ls="--", zorder=1, label="y = x")
    ax.fill_between([lo, hi], [lo / 2, hi / 2], [lo * 2, hi * 2], color=GRID, alpha=0.45,
                    lw=0, zorder=0, label="within 2x")


# ----------------------------------------------------------------------
# 1. per-channel mean rate, detector vs detector
# ----------------------------------------------------------------------
fig, axes = _pair_fig(f"Per-channel mean spike rate over {T:g}s ({n_chan} bipolar channels)")
for ax, (a, b) in zip(axes, PAIRS):
    x, y = rates[a], rates[b]
    hi = max(x.max(), y.max()) * 1.08
    lo = max(min(x[x > 0].min(), y[y > 0].min()) * 0.7, 1e-3) if LOG_AXES else 0.0
    _identity(ax, max(lo, 1e-3) if LOG_AXES else 0.0, hi)
    ax.scatter(x, y, s=14, color=COLORS.get(b, MUTED), edgecolor="none", alpha=0.75, zorder=2)
    # slope through the origin = the overall gain between the two detectors
    gain = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) else np.nan
    rho = spearmanr(x, y).statistic
    r = np.corrcoef(x, y)[0, 1]
    ax.plot([0, hi], [0, gain * hi], color=COLORS.get(b, MUTED), lw=1.0, alpha=0.8,
            label=f"fit through 0: {gain:.2f}x")
    if LOG_AXES:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{a} rate (Hz)")
    ax.set_ylabel(f"{b} rate (Hz)")
    ax.set_title(f"{a} vs {b}   Pearson r={r:.2f}, Spearman rho={rho:.2f}",
                 fontsize=9, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    recessive(ax)
fig.tight_layout()
fig.savefig(HERE / "eval_rate_scatter.png", dpi=130)
print("[saved] eval_rate_scatter.png")


# ----------------------------------------------------------------------
# 2. channel RANK agreement (1 = spikiest)
# ----------------------------------------------------------------------
def _rankable(a, b):
    """Channels whose ORDER the data can actually support, for the pair (a, b).

    Two exclusions, both about ties rather than about being uninteresting:
      * silent channels -- every detector's zeros collapse into one averaged rank, so a block
        of ~40% of channels lands on a single arbitrary value in both axes;
      * channels still tied inside the surviving subset -- at these rates a 1-count difference
        IS the rank, so tied counts carry no ordering information.
    Ties are judged AFTER the activity cut (a value tied only with a dropped channel is not
    tied in the subset), and dropping tied channels cannot create new ties, so one pass."""
    keep = np.ones(n_chan, bool)
    if RANK_ACTIVE_ONLY:
        keep &= (rates[a] > 0) & (rates[b] > 0)
    if RANK_DROP_TIES:
        for d in (a, b):
            v = rates[d][keep]
            vals, cnt = np.unique(v, return_counts=True)
            tied = set(vals[cnt > 1].tolist())
            keep[keep] = [x not in tied for x in v]
    return keep


fig, axes = _pair_fig(f"Channel rank by spike rate (1 = spikiest), ties and silent "
                      f"channels excluded")
for ax, (a, b) in zip(axes, PAIRS):
    keep = _rankable(a, b)
    n_keep = int(keep.sum())
    if n_keep < 25:
        # Short segments leave almost nothing rankable: over 60 s most channels have 0-3
        # spikes, which is one big tie block. rho on a handful of channels is noise.
        print(f"[warn] {a} vs {b}: only {n_keep} channels survive the tie/activity cut -- "
              f"rho on that few is not interpretable; use a longer segment.")
    if n_keep < 3:
        ax.set_title(f"{a} vs {b}: only {n_keep} rankable channels", fontsize=9, loc="left")
        recessive(ax)
        continue
    # Ranks are recomputed WITHIN the subset -- global ranks would leave gaps and stretch the
    # axis over channels that are no longer plotted.
    x = rankdata(-rates[a][keep], method="average")
    y = rankdata(-rates[b][keep], method="average")
    rho = spearmanr(rates[a][keep], rates[b][keep]).statistic
    rho_all = spearmanr(rates[a], rates[b]).statistic
    kept_idx = np.where(keep)[0]
    top_a = set(kept_idx[np.argsort(-rates[a][keep])[:TOP_K]])
    shared = len(top_a & set(kept_idx[np.argsort(-rates[b][keep])[:TOP_K]]))
    _identity(ax, 1, n_keep)
    ax.scatter(x, y, s=16, color=COLORS.get(b, MUTED), edgecolor="none", alpha=0.8, zorder=2)
    hot = np.argsort(-rates[a][keep])[:TOP_K]
    ax.scatter(x[hot], y[hot], s=54, facecolor="none", edgecolor=COLORS.get(a, MUTED),
               lw=1.3, zorder=3, label=f"{a} top {TOP_K} ({shared}/{TOP_K} shared)")
    ax.set_xlim(0, n_keep + 1)
    ax.set_ylim(0, n_keep + 1)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{a} rank (of {n_keep})")
    ax.set_ylabel(f"{b} rank (of {n_keep})")
    ax.set_title(f"{a} vs {b}   rho={rho:.2f}  (all {n_chan} ch: {rho_all:.2f})",
                 fontsize=9, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    recessive(ax)
fig.tight_layout()
fig.savefig(HERE / "eval_rank_scatter.png", dpi=130)
print("[saved] eval_rank_scatter.png")

# --- reliability ceiling: how well does each detector agree with ITSELF? ---
# A detector cannot rank-correlate with another better than it correlates with its own second
# opinion. Split the segment into SELF_BLOCK_SEC blocks and give alternate blocks to two
# halves: same recording, same channels, same detector, so anything below 1.0 is that
# detector's own sampling noise. Interleaved (not first-half/second-half) so slow drift in the
# recording is not charged to the detector.
# The classic attenuation bound then applies: rho_ab <= sqrt(rho_aa * rho_bb).
SELF_BLOCK_SEC = 30.0


def _self_rho(d):
    nb = int(T // SELF_BLOCK_SEC)
    if nb < 4:
        return np.nan
    e = np.arange(nb + 1) * SELF_BLOCK_SEC
    cnt = np.array([np.histogram(s, bins=e)[0] for s in spikes[d]])
    a, b = cnt[:, 0::2].sum(axis=1), cnt[:, 1::2].sum(axis=1)
    keep = (a + b) > 0
    return spearmanr(a[keep], b[keep]).statistic


self_rho = {d: _self_rho(d) for d in detectors}
print(f"--- reliability ceiling (rank agreement of each detector with ITSELF, "
      f"interleaved {SELF_BLOCK_SEC:g}s blocks) ---")
for d in detectors:
    print(f"  {d:<10} self-rho = {self_rho[d]:.3f}")

print(f"--- rank agreement (Spearman rho on per-channel rate) ---")
for a, b in PAIRS:
    keep = _rankable(a, b)
    rho_all = spearmanr(rates[a], rates[b]).statistic
    kept_idx = np.where(keep)[0]
    if keep.sum() >= 3:
        rho = spearmanr(rates[a][keep], rates[b][keep]).statistic
        shared = len(set(kept_idx[np.argsort(-rates[a][keep])[:TOP_K]]) &
                     set(kept_idx[np.argsort(-rates[b][keep])[:TOP_K]]))
        ceiling = np.sqrt(max(self_rho[a], 0) * max(self_rho[b], 0))
        print(f"  {a} vs {b}: rho={rho:.3f} on {keep.sum():3d} rankable ch "
              f"(all {n_chan}: {rho_all:.3f})   top-{TOP_K} overlap {shared}/{TOP_K}"
              f"   ceiling {ceiling:.3f} -> {rho / ceiling:.0%} of achievable"
              if np.isfinite(ceiling) and ceiling > 0 else
              f"  {a} vs {b}: rho={rho:.3f} on {keep.sum():3d} rankable ch "
              f"(all {n_chan}: {rho_all:.3f})   top-{TOP_K} overlap {shared}/{TOP_K}")
    else:
        print(f"  {a} vs {b}: only {keep.sum()} rankable channels (all {n_chan}: {rho_all:.3f})")


# --- the same thing as a figure: what each detector CAN achieve, and what it does ---
# Left: self-rho, the ceiling each detector sets for itself. Right: each pair's observed rho
# drawn against that pair's ceiling, so a short bar next to a tall ceiling reads as "real
# disagreement" and a short bar next to a short ceiling reads as "too noisy to tell".
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                               gridspec_kw={"width_ratios": [1, 1.35]})
axL.bar(range(len(detectors)), [self_rho[d] for d in detectors],
        color=[COLORS.get(d, MUTED) for d in detectors], width=0.6)
for i, d in enumerate(detectors):
    axL.text(i, self_rho[d] + 0.02, f"{self_rho[d]:.3f}", ha="center", fontsize=9)
axL.set_xticks(range(len(detectors)))
axL.set_xticklabels(detectors)
axL.set_ylim(0, 1.08)
axL.set_ylabel("rank agreement with itself (Spearman rho)")
axL.set_title(f"reliability: interleaved {SELF_BLOCK_SEC:g}s blocks, same detector",
              fontsize=9, loc="left")

pair_lbl, obs, ceil = [], [], []
for a, b in PAIRS:
    keep = _rankable(a, b)
    if keep.sum() < 3:
        continue
    pair_lbl.append(f"{a[:4]}\nvs {b[:4]}")
    obs.append(spearmanr(rates[a][keep], rates[b][keep]).statistic)
    ceil.append(np.sqrt(max(self_rho[a], 0) * max(self_rho[b], 0)))
xs = np.arange(len(pair_lbl))
axR.bar(xs, ceil, width=0.62, color=GRID, label="ceiling sqrt(self_a x self_b)")
axR.bar(xs, obs, width=0.62, color=MUTED, label="observed rho")
for i, (o, c) in enumerate(zip(obs, ceil)):
    axR.text(i, max(o, 0) + 0.03, f"{o:.3f}\n{o / c:.0%} of ceiling", ha="center", fontsize=8)
axR.set_xticks(xs)
axR.set_xticklabels(pair_lbl)
axR.set_ylim(0, 1.08)
axR.set_ylabel("Spearman rho")
axR.set_title("agreement vs what the pair could possibly achieve", fontsize=9, loc="left")
axR.legend(frameon=False, fontsize=8, loc="upper right")
for a_ in (axL, axR):
    recessive(a_)
fig.suptitle(f"Is the disagreement real, or just noise? | {T:g}s, {n_chan} channels",
             fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "eval_reliability.png", dpi=130)
print("[saved] eval_reliability.png")


# ----------------------------------------------------------------------
# 3. per-channel distribution of binned rates
# ----------------------------------------------------------------------
n_bins = int(np.floor(T / BIN_SEC))
if n_bins < 1:                    # BIN_SEC wider than the whole segment
    raise SystemExit(f"BIN_SEC ({BIN_SEC:g}s) exceeds the {T:g}s segment -- nothing to bin.")
if n_bins < 8:
    print(f"[warn] {T:g}s / {BIN_SEC:g}s = {n_bins} bins per channel -- a box drawn from "
          f"{n_bins} points shows almost nothing. Lower BIN_SEC, or raise SECONDS in "
          f"compare_spikes.py (Delphos cost grows with segment length).")
edges = np.arange(0, n_bins + 1) * BIN_SEC
binned = {d: np.array([np.histogram(s, bins=edges)[0] / BIN_SEC for s in spikes[d]])
          for d in detectors}                                   # [n_chan x n_bins] in Hz

ORDER_BY = detectors[0]         # shared channel order: biggest first by this detector
order = np.argsort(-rates[ORDER_BY])
ymax = max(b.max() for b in binned.values()) * 1.05 or 1.0

fig, axes = plt.subplots(len(detectors), 1, figsize=(max(12, n_chan * 0.075),
                                                     2.8 * len(detectors) + 1.2),
                         sharex=True, squeeze=False)
for ax, d in zip(axes[:, 0], detectors):
    data = [binned[d][c] for c in order]
    bp = ax.boxplot(data, widths=0.62, showfliers=True, showmeans=True,
                    patch_artist=False,
                    medianprops=dict(color=COLORS.get(d, MUTED), lw=1.1),
                    boxprops=dict(color=COLORS.get(d, MUTED), lw=0.7),
                    whiskerprops=dict(color=MUTED, lw=0.6),
                    capprops=dict(color=MUTED, lw=0.6),
                    flierprops=dict(marker=".", markersize=2.5, markerfacecolor=RED,
                                    markeredgecolor="none"),
                    meanprops=dict(marker="x", markersize=3, markeredgecolor=RED, lw=0.8))
    ax.set_ylabel(f"{d}\nrate (Hz)", color=COLORS.get(d, MUTED), fontsize=9)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=GRID, lw=0.5)
    recessive(ax)
ticks = np.arange(0, n_chan, LABEL_EVERY)
axes[-1, 0].set_xticks(ticks + 1)
axes[-1, 0].set_xticklabels([names[order[i]] for i in ticks], rotation=90, fontsize=5)
axes[-1, 0].set_xlabel(f"channel (ordered by {ORDER_BY} mean rate, biggest first)")
fig.suptitle(f"Per-channel {BIN_SEC:g}s-bin rate distribution "
             f"(box = median/IQR, x = mean, dots = outliers) | {n_bins} bins over {T:g}s",
             fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "eval_binned_rates.png", dpi=130)
print("[saved] eval_binned_rates.png")
