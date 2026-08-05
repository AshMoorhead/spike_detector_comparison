"""
stim_artefact_check.py
----------------------
A ROUGH check on whether a detector's stim-ON behaviour is driven by residual stimulation
artefact rather than by brain activity.

    .venv\\Scripts\\python.exe -m sdc.compare.stim_artefact_check runs/P1_stim.npz

The question it exists for: `stim_effect.py` shows Janca and Barkmeier saying spikes more than
halved during stimulation while Delphos says nothing happened. One explanation is physiology.
The other is that 145 Hz artefact survives into Delphos's 8-512 Hz detection band and keeps its
blob count up, while Janca's 10-60 Hz envelope and Barkmeier's 20-50 Hz shape gates never see
it. Those two predict different things, and the difference is cheap to measure.

THE TEST, in one line: per channel, does the detector's ON/OFF rate ratio track how much extra
145 Hz power that channel picked up?

  * A detector reporting real neural change should show NO relationship -- its ratio is set by
    the brain, and the artefact is just something the QC masked.
  * A detector being triggered by the artefact should show a POSITIVE one -- the channels that
    got the most stim contamination are the ones where its count held up.

DELIBERATELY ROUGH. It samples a handful of one-second windows rather than reading the whole
recording, uses a fixed band around the stimulation frequency, and reports a rank correlation.
It is a triage tool: it can say "look harder at Delphos", it cannot quantify an effect. Two
specific things it does NOT control for:

  * The channels worst hit by artefact are exactly the ones the QC mask removed entirely (94 of
    226 on P1), so this necessarily measures the SURVIVORS -- the moderately contaminated ones.
    That biases towards finding nothing, so a positive result here is the meaningful direction.
  * Stim-band power and proximity to the stimulating contact are the same thing, and so is
    whatever the stimulation actually does to nearby tissue. A correlation is consistent with
    artefact, it does not prove it.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import spearmanr

from seeg import read_edf_header, load_edf_segment, derive_montage, apply_montage
from seeg._style import RED, BLUE, MUTED, GRID, recessive

from sdc.common import cond

from sdc.common.paths import ROOT as HERE   # repo root, not this file's dir --
                                            # see sdc/common/paths.py
NPZ = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs" / "P1_stim.npz"

N_WIN = 10          # one-second windows sampled per condition. Ten is enough for a band-power
                    # ratio spanning orders of magnitude, and keeps this under a minute.
HALF_BW = 6.0       # +-Hz around the stim frequency counted as "stim band"
REF_BANDS = [(90.0, 130.0), (160.0, 200.0)]   # neighbouring bands, as the per-channel reference
                    # for how much broadband power that channel has -- without this, a channel
                    # that is simply noisier looks contaminated

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}


def band_power(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    return pxx[m].mean(axis=0)


def sample_windows(edf, hdr, runs, n_win, names):
    """Mean PSD per channel over n_win one-second windows spread evenly across `runs`.

    Evenly spaced, not random: this has to give the same answer twice, and a rough test whose
    number moves between runs is worse than no test."""
    fs = float(hdr["SampleRate"])
    starts = []
    total = float(np.diff(runs, axis=1).sum())
    for a, b in runs:
        k = max(int(round(n_win * (b - a) / total)), 1)
        # skip the first and last second of each block: the ramp in and out is not steady state
        lo, hi = a + 1.0, b - 2.0
        if hi <= lo:
            continue
        starts += list(np.linspace(lo, hi, k))
    acc, f_ref = None, None
    for t0 in starts:
        r0 = int(np.floor(t0)) + 1                     # 1-based inclusive records
        rec = load_edf_segment(edf, hdr, r0, r0 + 1)
        rec = apply_montage(rec, derive_montage(rec["info"]["SelectedSignals"], verbose=False),
                            verbose=False)
        if list(rec["info"]["SelectedSignals"]) != names:
            raise SystemExit("EDF channel order does not match the npz -- re-run compare_spikes.")
        f, pxx = welch(rec["data"], fs=fs, nperseg=int(fs), axis=0)
        acc = pxx if acc is None else acc + pxx
        f_ref = f
    return f_ref, acc / len(starts), len(starts)


# ----------------------------------------------------------------------
z = np.load(NPZ, allow_pickle=False)
names = [str(s) for s in z["names"]]
fs_det = float(z["fs"])
n_chan = len(names)
dets = [str(s) for s in z["detectors"]]
stim_hz = float(z["stim_hz"])
if not np.isfinite(stim_hz):
    raise SystemExit(f"{NPZ.name} is not a stim recording (stim_hz is nan).")

ON, OFF = cond.select(z, "on"), cond.select(z, "off")
BOTH = (ON.clean_sec > 0) & (OFF.clean_sec > 0)
edf = str(z["edf"])
hdr = read_edf_header(edf)
print(f"{NPZ.name}: {stim_hz:g} Hz stim, {ON.T:.0f}s ON / {OFF.T:.0f}s OFF, "
      f"{int(BOTH.sum())}/{n_chan} channels measurable in both")
print(f"sampling {N_WIN} x 1 s windows per condition from {Path(edf).name} ...")

f, psd_on, n_on = sample_windows(edf, hdr, ON.runs, N_WIN, names)
_, psd_off, n_off = sample_windows(edf, hdr, OFF.runs, N_WIN, names)
print(f"  {n_on} ON windows, {n_off} OFF windows at {hdr['SampleRate']:g} Hz")

# stim-band power RELATIVE to the neighbouring bands, per channel, per condition. The relative
# form is what makes ON and OFF comparable: overall gain differences divide out.
def rel_stim(psd):
    s = band_power(f, psd, stim_hz - HALF_BW, stim_hz + HALF_BW)
    ref = np.mean([band_power(f, psd, lo, hi) for lo, hi in REF_BANDS], axis=0)
    return s / np.where(ref > 0, ref, np.nan)


contam = np.log2(rel_stim(psd_on) / rel_stim(psd_off))   # extra stim-band power, in doublings
# ALSO the broadband change, because panel (a) shows the ON spectrum lifted at EVERY frequency,
# not only at the harmonics -- and `contam` divides that out by construction. Barkmeier's
# threshold is `-mean(|fEEG|) - 4*std(|fEEG|)` over the block, so broadband power is the
# quantity that moves it; the narrowband measure would miss that entirely.
BROAD = (20.0, 200.0)
broad = np.log2(band_power(f, psd_on, *BROAD) / band_power(f, psd_off, *BROAD))
_pk = f[(f > 100) & (f < 600)][np.argsort(
    -np.nanmedian(psd_on[(f > 100) & (f < 600)][:, BOTH], axis=1))[:4]]
print(f"\ntop spectral peaks 100-600 Hz during stim: {np.sort(_pk)} Hz "
      f"(expect {stim_hz:g} and harmonics)")
print(f"\nresidual {stim_hz:g} Hz power, ON vs OFF (log2, relative to neighbouring bands):")
print(f"  median {np.nanmedian(contam[BOTH]):+.2f}   "
      f"90th pct {np.nanpercentile(contam[BOTH], 90):+.2f}   "
      f"max {np.nanmax(contam[BOTH]):+.2f} ({names[int(np.nanargmax(np.where(BOTH, contam, -np.inf)))]})")

# per-detector ON/OFF rate ratio, same construction as stim_effect.py
ratio = {}
for d in dets:
    flag = z[f"{d}_on"].astype(bool)
    r_on = ON.rate(np.bincount(z[f"{d}_chan"][flag], minlength=n_chan))
    r_off = OFF.rate(np.bincount(z[f"{d}_chan"][~flag], minlength=n_chan))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio[d] = np.where(r_off > 0, r_on / r_off, np.nan)

fig, axes = plt.subplots(1, 1 + len(dets), figsize=(4.6 * (1 + len(dets)), 4.2))

# ---- (a) the artefact itself: mean spectrum ON vs OFF ----------------------------------
ax = axes[0]
ax.loglog(f, np.nanmedian(psd_off[:, BOTH], axis=1), color=MUTED, lw=1.3, label="stim OFF")
ax.loglog(f, np.nanmedian(psd_on[:, BOTH], axis=1), color="#c8102e", lw=1.3, label="stim ON")
for h in (stim_hz, 2 * stim_hz, 3 * stim_hz):
    if h < f.max():
        ax.axvline(h, color=GRID, lw=6, alpha=.55, zorder=0)
ax.axvspan(8, 512, color=VIOLET, alpha=.06, lw=0, zorder=0)
ax.text(20, ax.get_ylim()[0] * 1.5, "Delphos band 8-512 Hz", fontsize=7, color=VIOLET)
ax.set_xlim(1, min(f.max(), 600))
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("median PSD across channels")
ax.set_title(f"(a) does {stim_hz:g} Hz survive into the detection bands?", fontsize=9, loc="left")
ax.legend(frameon=False, fontsize=8)
recessive(ax)

# ---- (b..) does each detector's ON/OFF ratio track contamination? ----------------------
print(f"\n--- rank correlation: extra {stim_hz:g} Hz power vs the detector's ON/OFF rate ratio ---")
print("    (near zero = the detector is not being driven by the artefact)")
for ax, d in zip(axes[1:], dets):
    ok = BOTH & np.isfinite(contam) & np.isfinite(ratio[d])
    x, y = contam[ok], ratio[d][ok]
    rho = spearmanr(x, y).statistic
    p = spearmanr(x, y).pvalue
    okb = BOTH & np.isfinite(broad) & np.isfinite(ratio[d])
    rho_b = spearmanr(broad[okb], ratio[d][okb]).statistic
    col = COLORS.get(d, MUTED)
    ax.scatter(x, y, s=18, color=col, alpha=.7, edgecolor="none")
    ax.axhline(1.0, color=MUTED, ls="--", lw=1.0)
    ax.axvline(0.0, color=MUTED, ls=":", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel(f"extra {stim_hz:g} Hz power during stim (log2)")
    ax.set_ylabel("ON / OFF rate" if d == dets[0] else "")
    ax.set_title(f"({'bcd'[dets.index(d)]}) {d}   rho = {rho:+.2f} (p {p:.2g})"
                 f"   broadband rho = {rho_b:+.2f}", fontsize=9, loc="left", color=col)
    recessive(ax)
    print(f"  {d:<10} {stim_hz:g}Hz band rho = {rho:+.3f} (p={p:.2g})   "
          f"broadband 20-200Hz rho = {rho_b:+.3f}   (n={int(ok.sum())} channels)")

fig.suptitle(f"Is the stim-ON result artefact? | {str(z['rec_id'])}, {stim_hz:g} Hz, "
             f"{n_on}+{n_off} one-second windows -- ROUGH TRIAGE, not a measurement", fontsize=11)
fig.tight_layout()
OUT = HERE / "figures" / "real" / NPZ.stem
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "stim_artefact_check.png", dpi=130)
print(f"\n[saved] {OUT / 'stim_artefact_check.png'}")
print("[read it as] a positive rho means that detector held its count up on exactly the "
      "channels that got the most stim contamination. Near zero means it did not.")
plt.close(fig)
