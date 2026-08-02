"""
compare_spikes.py
-----------------
Compare three spike detectors over the SAME window of the same recording:
  * Janca     -- this repo's janca_detect_spikes.py.
  * Barkmeier -- python_pipeline seeg.detect_spikes (MATLAB mDetectSpike via the engine).
  * Delphos   -- the Delphos CLI (Roehri et al.), via delphos_detect_spikes.py.

All run at their OWN defaults. The script always does one baseline run (print + raster +
population rate). A parameter SWEEP is provided but OFF by default -- edit the SWEEP_* config
and set RUN_SWEEP=True to run it yourself.

FAIR-COMPARISON CHOICES:
  #3 same sample rate  -- signal decimated to DETECT_FS (400 Hz); Janca and Barkmeier both
     run there. Janca uses dec=0 (internal decimation OFF). At DETECT_FS != 200 Janca's
     bandpass falls back from Chebyshev-II to Butterworth -- exactly what v24 MATLAB does;
     faithful. (Use DETECT_FS=200 to get Janca's native cheby2 path.)
  #4 symmetric masking -- one dilated artefact mask (DILATE_MS) applied to ALL outputs
     (MASK_ARTEFACTS); Barkmeier's own post-mask is off (post_mask_spikes=False). Detector
     disagreement about how spiky an ARTEFACT is is not a result, so it is excluded for all.
  #5 shared polyspike rule -- MERGE_MS collapses marks closer than that into one, for every
     detector. The detectors' native burst handling differs wildly (Janca merges within
     120 ms by construction; Delphos can emit marks one sample apart), and without a common
     rule that book-keeping difference is scored as disagreement. Per-stage attrition is
     printed so you can see what each detector lost to the mask and to the merge.

  DELPHOS IS THE EXCEPTION TO #3, deliberately. It is a compiled black box that reads the
  raw file itself, at the file's own 2 kHz, and derives its OWN bipolar montage -- so it
  cannot be fed the decimated, pre-montaged array the other two see. Its wide detection band
  (8-512 Hz) would also be gutted by a 200 Hz Nyquist. It is therefore run at its validated
  operating point (Delphos.md) on the same EDF window, and its detections -- marker
  positions are in absolute file seconds -- are mapped onto the common 400 Hz time axis and
  onto pipeline channel names. Treat it as a REFERENCE detector compared on rates/counts,
  not as a like-for-like third arm: its RAM-dependent internal tiling means even it does not
  reproduce itself bit-exactly between operating points.

DETECTOR SETTINGS
  Janca (envelope detector): thresholds the Hilbert-envelope of the band-passed signal
    against a per-channel lognormal model of the background. Knobs:
      k1  main threshold multiplier -- detect where envelope > k1*(mode+median) of the
          background. HIGHER k1 = fewer/only-bigger spikes. This is THE sensitivity dial.
      fl,fh  bandpass (Hz) defining "fast" -- default 10-60. Spikes live here.
      k2  second (ambiguous) threshold; if < k1, low-tier spikes are kept only when another
          channel has an obvious one. Off by default (k2=k1).
      k3  sk*(mean-mode) threshold shift; 0 = off.
      pt  polyspike-union time (s) -- maxima closer than this merge into one event (~refractory).
    NOTE Janca has NO amplitude-in-µV or half-width/duration criterion: it gates on a
    STATISTICAL envelope threshold only. Barkmeier gates on shape (slope+amplitude+duration).
    That difference -- not a threshold -- is why the two find different events.
  Barkmeier (shape detector): DetThresholds = [LS RS TAMP LD RD] (mDetectSpike_coeffs.m:27):
      LS,RS   left/right half-wave slope thresholds
      TAMP    summed half-wave amplitude threshold (Lamp+Ramp) -- the AMPLITUDE / HALF-HEIGHTS
      LD,RD   left/right half-wave duration thresholds (ms) -- the HALF-WIDTHS
      std_coeff, trough_search_ms, filter_spec -- peak-threshold scale, trough window, passband.
  Delphos (time-frequency detector): thresholds normalised power in a whitened TF plane.
      Spk_thr       normalised TF power threshold (40 here, vs the CLI default 80 -> more
                    sensitive). This is its sensitivity dial.
      Spk_time_thr  time-width ratio (estimated/theoretical dirac) BELOW which a blob is a
                    spike -- i.e. "sharp enough", the TF analogue of a half-width criterion.
      freq_band     8-512 Hz, so it sees fast components the 400 Hz arms cannot.
    Delphos gates on neither shape (Barkmeier) nor a per-channel background model (Janca)
    but on TF blob sharpness -- a third, independent notion of "spike".

Barkmeier needs a local MATLAB R2026a install; if the engine can't start it falls back to
Janca-only. Delphos needs MATLAB Runtime 9.5 (R2018b) -- a separate install -- plus ~5 min
per call, so it is cached on disk and any failure degrades to "no Delphos panel".

Run with the local venv:
    .venv\\Scripts\\Activate.ps1      then      python compare_spikes.py
    (or, without activating:   .venv\\Scripts\\python.exe compare_spikes.py)
"""
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from seeg import (read_edf_header, load_edf_segment, derive_montage, apply_montage,
                  make_cfg_artefact, windowed_artefact_detector, decimate_recording, view)
from seeg import detect_spikes as detect_barkmeier
from seeg._style import RED, BLUE, MUTED, MEDIAN, recessive
from janca_detect_spikes import detect_spikes as detect_janca
from delphos_detect_spikes import detect_spikes as detect_delphos

EDF = r"C:\Users\amoo0039\Documents\local\P1\baseline.edf"
SECONDS = 600          # window length (records are 1 s)
DETECT_FS = 400.0     # common rate the Janca/Barkmeier arms run at (2000 Hz -> /5)
DILATE_MS = 60        # symmetric artefact-exclusion radius applied to every detector
MASK_ARTEFACTS = True # drop detections inside the dilated artefact mask (all detectors alike)
MERGE_MS = 20.0       # shared polyspike rule: marks closer than this collapse to one. 20 ms is
                      # a refractory-like floor that actually bites -- at DETECT_FS=400 one
                      # sample is 2.5 ms, so anything below that only removes exact duplicates.
                      # Janca is unaffected (it already unions within 120 ms), so this mostly
                      # brings Barkmeier and Delphos onto Janca's footing.
TOL_MS = 50           # agreement tolerance
INTERACTIVE = False   # open the scroll/zoom viewer after the baseline run
RUN_DELPHOS = True    # False -> skip the Delphos arm entirely (2-panel raster, as before)

VIOLET = "#4a3aa7"    # Delphos; from the QC palette -- third hue, no red/green pairing

# ---- detector defaults (reverted) ----
# dec=0 keeps DETECT_FS. `pt` (polyspike_union_time) is DELIBERATELY moved off its 0.12 s
# default and tied to MERGE_MS: it is Janca's INTERNAL union, so a post-hoc MERGE_MS cannot
# undo it, and leaving it at 0.12 while the others merge at MERGE_MS means Janca is still
# collapsing bursts nobody else collapses. Tying them makes MERGE_MS one dial for all three.
# `pt` only sets a merge span (janca_detect_spikes.py:269-271, :370) -- it does not change the
# threshold or the envelope -- so this suppresses merging without altering what is detected.
# Set JANCA = dict(dec=0) to restore Janca's published default and accept the asymmetry.
JANCA = dict(dec=0, pt=MERGE_MS / 1000.0)
BARK = dict(LS=3.0, RS=3.0, TAMP=1200.0, LD=8, RD=8,  # TAMP=1200 tuned to match Janca's count
            std_coeff=4.0, trough_search_ms=40.0, filter_spec=(20.0, 50.0, 1.0, 35.0))
# Delphos runs on the RAW EDF at its agreed operating point (Delphos.md): Spk_thr 40,
# 8-512 Hz, its own bipolar montage, RAM pinned to 12 GB so the internal tiling -- hence the
# detections -- stays in one regime. Keep pin/chunk FIXED across everything you compare.
DELPHOS = dict(pin_free_ram_gb=12, Spk_thr=50, Spk_time_thr=1.25, chunk_sec=None)   # exe path etc: delphos_detect_spikes.DEFAULTS
# ~5 min/call -> memoised by file+window+params. Anchored to THIS FILE, not the cwd: a miss
# costs 5 minutes, so the cache must not quietly move when you run the script from elsewhere.
DELPHOS_CACHE = Path(__file__).resolve().parent / ".delphos_cache"
# Detections dumped for evaluate_detectors.py (the evaluation plots read ONLY this file).
DETECTIONS_NPZ = Path(__file__).resolve().parent / "detections.npz"

# ---- SWEEP: you drive this. Set RUN_SWEEP=True, pick a detector/param/grid, run the script. ----
# The swept detector is scored against SWEEP_REF's DEFAULT run -- that is how Barkmeier's
# TAMP was pulled down to Janca's count, and the same move works for Delphos's Spk_thr.
RUN_SWEEP = False
SWEEP_DETECTOR = "Delphos"                # "Janca" | "Barkmeier" | "Delphos" (case-insensitive)
SWEEP_REF = "Janca"                       # scored against this detector's default run
SWEEP_PARAM = "Spk_time_thr"                   # janca:     k1 fl fh pt k3
                                          # barkmeier: LS RS TAMP LD RD std_coeff trough_search_ms
                                          # delphos:   Spk_thr Spk_time_thr freq_band_start/end
SWEEP_VALUES = [1.1, 1.3, 1.5]     # Delphos TF-power threshold; 40 = our default, 80 = CLI's
# ready-made grids (copy into SWEEP_PARAM / SWEEP_VALUES):
#   amplitude / summed half-heights : "TAMP", [400, 600, 800, 1000, 1200, 1500]
#   left  half-width  (ms)          : "LD",   [4, 6, 8, 10, 12, 15]
#   right half-width  (ms)          : "RD",   [4, 6, 8, 10, 12, 15]
#   Barkmeier slope                 : "LS",   [3, 5, 7, 9, 11]      (and/or "RS")
#   Janca threshold                 : "k1",   [2.6, 3.0, 3.4, 3.8, 4.2]
#   Janca high-cut (Hz)             : "fh",   [40, 50, 60, 80]
#   Delphos TF-power threshold      : "Spk_thr",      [40, 60, 80, 100, 120]   <- the sensitivity dial
#   Delphos sharpness (width ratio) : "Spk_time_thr", [1.1, 1.2, 1.3, 1.4, 1.5]
# COST: Janca/Barkmeier points are seconds. DELPHOS IS ~5 MIN PER UNCACHED POINT (each call
# pays a fresh parpool startup), so a 5-point grid is ~25 min the first time. Points are
# cached by parameter value, so re-running the same grid is instant.

# ----------------------------------------------------------------------
hdr = read_edf_header(EDF)
FACTOR = int(round(hdr["SampleRate"] / DETECT_FS))
rec = load_edf_segment(EDF, hdr, start_rec=1, stop_rec=SECONDS)

rec = apply_montage(rec, derive_montage(rec["info"]["SelectedSignals"]))
dec = decimate_recording(rec, factor=FACTOR)
fs = dec["info"]["SampleRate"]
names = dec["info"]["SelectedSignals"]
n_chan = len(names)

qc = windowed_artefact_detector(dec, make_cfg_artefact(hdr["lsb"], trial=None))  # baseline: no stim rule
_d = int(round(DILATE_MS / 1000 * fs))
dmask = binary_dilation(qc["sampleMask"], structure=np.ones((2 * _d + 1, 1), bool)) if _d else qc["sampleMask"]


_SAMPLE_MS = 1000.0 / fs
if MERGE_MS and MERGE_MS < _SAMPLE_MS:
    print(f"[warn] MERGE_MS={MERGE_MS:g} ms is finer than one sample at {fs:g} Hz "
          f"({_SAMPLE_MS:.1f} ms), so it can only remove exact duplicates. Detections live on "
          f"the sample grid: the closest two distinct spikes can be is {_SAMPLE_MS:.1f} ms. "
          f"Raise MERGE_MS above that to actually merge anything.")


def _merge_close(idx, min_gap):
    """Collapse detections closer than `min_gap` samples into the FIRST of each run.

    Chained, not pairwise: three marks 1 sample apart become one, not two. That matches what
    a polyspike-union rule does (Janca's `_detection_union`), so applying this to every
    detector puts them on the same footing."""
    if idx.size < 2 or min_gap <= 0:
        return idx
    keep = np.ones(idx.size, bool)
    last = idx[0]
    for i in range(1, idx.size):
        if idx[i] - last < min_gap:
            keep[i] = False
        else:
            last = idx[i]
    return idx[keep]


def _finalise(per_chan, label=None):
    """The SAME post-processing for every detector, in order: clip -> artefact mask -> merge.

    Both steps exist to stop the comparison being decided by something other than "where are
    the spikes":
      * MASK_ARTEFACTS -- one dilated mask (DILATE_MS) shared by all detectors, so a detector
        that fires freely on artefact does not get scored on it. Barkmeier's own post-mask
        stays off (post_mask_spikes=False) so this is the only masking applied.
      * MERGE_MS -- a shared polyspike rule. Without it the detectors are counting differently
        (Janca merges within 120 ms by construction, Delphos can emit marks one sample apart),
        which shows up as disagreement that is really a difference in book-keeping.
    Prints the attrition per stage when `label` is given -- how many spikes a detector loses to
    the mask is itself a result (it says how artefact-prone that detector is)."""
    gap = MERGE_MS / 1000.0 * fs
    n_raw = n_masked = n_merged = 0
    out = []
    for c in range(n_chan):
        idx = np.unique(np.asarray(per_chan[c], int))
        idx = idx[(idx >= 0) & (idx < dmask.shape[0])]
        n_raw += idx.size
        if MASK_ARTEFACTS:
            idx = idx[~dmask[idx, c]]
        n_masked += idx.size
        idx = _merge_close(idx, gap)
        n_merged += idx.size
        out.append(idx)
    if label:
        print(f"  [{label}] {n_raw} detected"
              f" -> artefact mask {'ON' if MASK_ARTEFACTS else 'OFF'}: "
              f"-{n_raw - n_masked} ({(n_raw - n_masked) / max(n_raw, 1):.0%})"
              f" -> merge <{MERGE_MS:g} ms: -{n_masked - n_merged}"
              f" -> {n_merged} kept")
    return out


def run_janca(data, label=None, **override):
    """Janca -> per-channel sample indices at `fs`. Override any Janca setting by keyword."""
    out, _disch, _info = detect_janca(data, fs, **{**JANCA, **override})
    return _finalise([np.round(out["pos"][out["chan"] == c] * fs).astype(int)
                      for c in range(n_chan)], label)


def run_bark(recording, label=None, **override):
    """Barkmeier -> per-channel sample indices. Override LS/RS/TAMP/LD/RD/std_coeff/... by keyword."""
    p = {**BARK, **override}
    detect_barkmeier(recording, qc, post_mask_spikes=False,
                     det_thresholds=[p["LS"], p["RS"], p["TAMP"], p["LD"], p["RD"]],
                     std_coeff=p["std_coeff"], trough_search_ms=p["trough_search_ms"],
                     filter_spec=p["filter_spec"])
    return _finalise([np.asarray(s, int) for s in recording["info"]["DetectedSpikes"]], label)


def run_delphos(label=None, **override):
    """Delphos -> per-channel sample indices on the common `fs` axis, same channel order.

    Runs on the RAW EDF over the SAME wall-clock window as everything else (start_rec=1 with
    1 s records => file seconds [0, SECONDS)), because the CLI reads the file itself. Marker
    positions are absolute file seconds, so passing `fs` converts them straight onto the
    decimated axis; `_finalise` then applies the same mask and merge as the other two."""
    return _finalise(detect_delphos(EDF, names, fs, start_sec=0.0, duration_sec=SECONDS,
                                    cache_dir=DELPHOS_CACHE, **{**DELPHOS, **override}), label)


def _match(a, b, tol):
    """Greedy one-to-one nearest match on sorted sample indices. offset = a - b."""
    a, b = np.sort(a), np.sort(b)
    i = j = matched = 0
    offs = []
    while i < a.size and j < b.size:
        d = int(a[i]) - int(b[j])
        if abs(d) <= tol:
            matched += 1
            offs.append(d)
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return matched, offs


def compare(a, b, tol_ms=TOL_MS):
    """Aggregate agreement metrics between two per-channel spike sets."""
    tol = int(round(tol_ms / 1000 * fs))
    matched, offs = 0, []
    for c in range(n_chan):
        m, o = _match(a[c], b[c], tol)
        matched += m
        offs += o
    na = sum(x.size for x in a)
    nb = sum(x.size for x in b)
    return dict(na=na, nb=nb, matched=matched,
                jaccard=matched / (na + nb - matched) if (na + nb - matched) else 0.0,
                med_off_ms=float(np.median(np.abs(offs)) / fs * 1000) if offs else float("nan"))


# --- baseline run: every detector at its defaults ---
# A detector that cannot run is dropped from `dets` (and from the figure) rather than faked
# with zeros -- an empty panel would read as "found nothing", which is a different claim.
print(f"--- post-processing (identical for every detector) ---")
janca = run_janca(dec["data"], label="Janca")
dets = [("Janca", janca, RED)]

try:
    bark = run_bark(dec, label="Barkmeier")
    dets.append(("Barkmeier", bark, BLUE))
except Exception as e:            # no MATLAB / engine failure -> skip that arm
    print(f"[warn] Barkmeier unavailable ({type(e).__name__}: {e}); skipping that arm.")
    bark = [np.zeros(0, int) for _ in range(n_chan)]

if RUN_DELPHOS:
    try:
        delphos = run_delphos(label="Delphos")
        dets.append(("Delphos", delphos, VIOLET))
    except Exception as e:        # no MATLAB Runtime 9.5 / CLI failure -> skip that arm
        print(f"[warn] Delphos unavailable ({type(e).__name__}: {e}); skipping that arm.")

counts = {name: sum(x.size for x in det) for name, det, _ in dets}
n_j, n_b = counts.get("Janca", 0), counts.get("Barkmeier", 0)

# Dump the detections so evaluate_detectors.py can work from the OUTPUT alone -- no re-running
# this script (and no 5 min Delphos call) just to redraw an evaluation plot. Ragged per-channel
# arrays are flattened to (index, channel) pairs; anchored to this file, like the cache.
_dump = {"names": np.array(names), "fs": fs, "seconds": SECONDS, "edf": EDF,
         "detectors": np.array([n for n, _, _ in dets])}
for name, det, _ in dets:
    _dump[f"{name}_idx"] = np.concatenate(det)
    _dump[f"{name}_chan"] = np.concatenate([np.full(d.size, c, int) for c, d in enumerate(det)])
np.savez(DETECTIONS_NPZ, **_dump)
print(f"[saved] {DETECTIONS_NPZ.name}")
print(f"{' | '.join(f'{k} {v}' for k, v in counts.items())} "
      f"| {n_chan} bipolar ch, {SECONDS}s at {fs:g} Hz (defaults)")

# Pairwise agreement over every pair of detectors that ran.
pairs = {}
print(f"--- pairwise agreement within {TOL_MS} ms ---")
for i in range(len(dets)):
    for k in range(i + 1, len(dets)):
        (na_, a, _), (nb_, b, _) = dets[i], dets[k]
        mm = compare(a, b)
        pairs[(na_, nb_)] = mm
        print(f"  {na_} vs {nb_}: matched {mm['matched']}  "
              f"| {na_}-only {mm['na'] - mm['matched']} ({(mm['na']-mm['matched'])/max(mm['na'],1):.0%})"
              f"  | {nb_}-only {mm['nb'] - mm['matched']} ({(mm['nb']-mm['matched'])/max(mm['nb'],1):.0%})"
              f"  | Jaccard {mm['jaccard']:.0%}  | median |dt| {mm['med_off_ms']:.1f} ms")

m = pairs.get(("Janca", "Barkmeier"), dict(matched=0, jaccard=0.0))
matched, jaccard = m["matched"], m["jaccard"]


# --- spike raster + population rate, channels sorted busiest-first (top) ---
T = dec["data"].shape[0] / fs
order = np.argsort([sum(det[c].size for _, det, _ in dets) for c in range(n_chan)])[::-1]
edges = np.arange(0, T + 1.0, 1.0)
centres = (edges[:-1] + edges[1:]) / 2
yt = np.linspace(0, n_chan - 1, 15).astype(int)

fig, axes = plt.subplots(len(dets) + 1, 1, sharex=True, figsize=(16, 5 + 3 * len(dets)),
                         gridspec_kw={"height_ratios": [1] + [3] * len(dets)})
axr, raster_axes = axes[0], axes[1:]
for name, det, color in dets:
    all_t = np.concatenate([det[c] for c in range(n_chan)]) / fs
    axr.plot(centres, np.histogram(all_t, bins=edges)[0], color=color, lw=1.3,
             label=f"{name} ({counts[name]})")
axr.set_ylabel("pop. rate\n(spikes/s)")
axr.legend(loc="upper right", frameon=False, fontsize=8, ncol=len(dets))
recessive(axr)
for ax, (name, det, color) in zip(raster_axes, dets):
    ax.eventplot([det[order[i]] / fs for i in range(n_chan)], colors=color,
                 lineoffsets=np.arange(n_chan), linelengths=0.8, linewidths=0.5)
    ax.set_title(f"{name} (n={counts[name]})", loc="left", fontsize=10, color=color)
    ax.set_ylabel("channel (busiest -> top)")
    ax.grid(axis="x", color="#e1e0d9", lw=0.5)
    ax.set_yticks(yt)
    ax.set_yticklabels([names[order[i]] for i in yt], fontsize=7)
    ax.set_ylim(n_chan, -1)
    recessive(ax)
raster_axes[-1].set_xlim(0, T)
raster_axes[-1].set_xlabel("time (s)")
rate_note = f"{fs:g} Hz" + (f"; Delphos on the raw {hdr['SampleRate']:g} Hz file"
                            if any(n == "Delphos" for n, _, _ in dets) else "")
fig.suptitle(f"{' vs '.join(n for n, _, _ in dets)} at their defaults | "
             f"{n_chan} bipolar ch, {SECONDS}s ({rate_note})\n"
             + "  ".join(f"{a[:4]}/{b[:4]} Jaccard {mm['jaccard']:.0%}"
                         for (a, b), mm in pairs.items())
             + f"  (within {TOL_MS} ms)", fontsize=10)
fig.tight_layout()
fig.savefig("compare_raster.png", dpi=130)
print("[saved] compare_raster.png")


# --- parameter sweep (opt-in: set RUN_SWEEP=True and edit SWEEP_* above) ---
# One runner per detector, keyed lower-case. Names are matched case-insensitively because
# SWEEP_DETECTOR is written the way it reads in a title ("Delphos"), not the way it was
# spelled in an if-branch -- the old code compared against a bare "janca" and silently swept
# the wrong detector when the config said "Janca".
RUNNERS = {"janca": lambda **o: run_janca(dec["data"], **o),
           "barkmeier": lambda **o: run_bark(dec, **o),
           "delphos": lambda **o: run_delphos(**o)}
COLORS = {"janca": RED, "barkmeier": BLUE, "delphos": VIOLET}
LABELS = {"janca": "Janca", "barkmeier": "Barkmeier", "delphos": "Delphos"}


def run_sweep(detector, param, values, ref=SWEEP_REF):
    """Sweep one setting of `detector`; `ref` stays at its default run.
    Returns list of (value, swept_count, matched, jaccard).

    Delphos points cost ~5 min each when uncached, so each one prints its own timing. Its
    detections also move with the RAM pin, so DELPHOS[pin_free_ram_gb] must stay fixed across
    the grid or the sweep confounds the parameter with the operating point -- it is held
    fixed here because only `param` is overridden."""
    key, ref_key = detector.lower(), ref.lower()
    baseline = {name.lower(): det for name, det, _ in dets}
    if key not in RUNNERS:
        raise ValueError(f"unknown SWEEP_DETECTOR {detector!r}; pick from {sorted(RUNNERS)}")
    if ref_key not in baseline:
        raise ValueError(f"SWEEP_REF {ref!r} did not run this session "
                         f"(available: {sorted(baseline)})")
    if key == "barkmeier" and param not in BARK:
        raise ValueError(f"unknown Barkmeier parameter {param!r}; pick from {sorted(BARK)}")
    # Janca validates its own kwargs (TypeError) and Delphos rejects unknown settings in
    # delphos_detect_spikes.detect_spikes, so a typo fails fast rather than sweeping a
    # constant -- which would otherwise look like "this knob does nothing".

    fixed = baseline[ref_key]
    print(f"--- sweep {LABELS[key]}.{param} vs {LABELS[ref_key]} default "
          f"(n={sum(x.size for x in fixed)}) ---")
    if key == "delphos":
        print(f"    {len(values)} Delphos points, ~5 min each unless already cached")
    rows = []
    for v in values:
        t0 = time.perf_counter()
        swept = RUNNERS[key](**{param: v})
        mm = compare(swept, fixed)
        n_sw = sum(x.size for x in swept)
        rows.append((v, n_sw, mm["matched"], mm["jaccard"]))
        print(f"  {param}={v}: {LABELS[key]} n={n_sw:4d}  matched={mm['matched']:4d}  "
              f"Jaccard={mm['jaccard']:.0%}  [{time.perf_counter() - t0:.0f}s]")
    if len(rows) > 1 and len({r[1] for r in rows}) == 1:
        print(f"  [warn] count is identical at every {param} -- is that knob reaching the "
              f"detector?")
    return rows


def plot_sweep(detector, param, rows, ref=SWEEP_REF):
    key, ref_key = detector.lower(), ref.lower()
    vals = [r[0] for r in rows]
    n_sw = [r[1] for r in rows]
    jac = [r[3] * 100 for r in rows]
    ref_n = counts[LABELS[ref_key]]
    ref_lbl, c = LABELS[ref_key], COLORS[key]
    figs, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(vals, n_sw, "o-", color=c, label=f"{LABELS[key]} count")
    ax1.axhline(ref_n, color=MUTED, lw=1.2, ls="--", label=f"{ref_lbl} default ({ref_n})")
    ax1.set_xlabel(f"{LABELS[key]}: {param}")
    ax1.set_ylabel("detections", color=c)
    ax2 = ax1.twinx()
    ax2.plot(vals, jac, "s-", color=MEDIAN, label="Jaccard")
    ax2.set_ylabel(f"Jaccard vs {ref_lbl} (%)", color=MEDIAN)
    ax1.legend(loc="best", frameon=False, fontsize=8)
    ax1.set_title(f"{LABELS[key]} {param} sweep @ {fs:g} Hz"
                  + (f" (Delphos on the raw {hdr['SampleRate']:g} Hz file)"
                     if key == "delphos" else ""))
    for a in (ax1, ax2):
        recessive(a)
    figs.tight_layout()
    out = f"sweep_{LABELS[key]}_{param}.png"
    figs.savefig(out, dpi=130)
    print(f"[saved] {out}")


if RUN_SWEEP:
    plot_sweep(SWEEP_DETECTOR, SWEEP_PARAM, run_sweep(SWEEP_DETECTOR, SWEEP_PARAM, SWEEP_VALUES))


# --- interactive explorer ---
if INTERACTIVE:
    view(dec, qc, spikes={f"{name} ({counts[name]})": det for name, det, _ in dets},
         chans_per_page=15, t0=0, duration=30)
