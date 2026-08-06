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
    .venv\\Scripts\\Activate.ps1      then      python -m sdc.detect.compare_spikes
    (or, without activating:   .venv\\Scripts\\python.exe -m sdc.detect.compare_spikes)
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from seeg import (read_edf_header, load_edf_segment, derive_montage, apply_montage,
                  make_cfg_artefact, windowed_artefact_detector, decimate_recording, view,
                  load_trials, get_patient, get_trial, resolve_file, detect_stim,
                  dilate_mask, merge_close)
from seeg import detect_spikes as detect_barkmeier
from seeg.spikes import fill_mask_ar
# Import the detector's constants rather than retyping them. These were previously copied as
# literals here, which meant an upstream change would not reach this comparison -- and the
# copies agreed only by coincidence.
from seeg.spikes import FILTER_SPEC, STD_COEFF, TROUGH_SEARCH_MS
from seeg._style import RED, BLUE, MUTED, MEDIAN, recessive
from sdc.detect.janca_detect_spikes import detect_spikes as detect_janca
from sdc.detect.delphos_detect_spikes import detect_spikes as detect_delphos
from sdc.common.spike_match import match as spike_match_fn
from sdc.common.paths import ROOT as _ROOT

# ---- which recording ----------------------------------------------------------------
# Recordings are named by (patient, trial, condition) and RESOLVED through the pipeline's own
# identity chain: load_trials -> get_patient -> get_trial -> resolve_file. Nothing here
# hardcodes an EDF path. `resolve_file` returns (stem, trial_or_None), and that None for 'pre'
# is the pipeline-wide signal for "no stim" that make_cfg_artefact consumes -- so the stim
# rules switch themselves on and off correctly just by naming the condition.
BASE_DIR = Path(r"C:\Users\amoo0039\Documents\local")
META_PATH = BASE_DIR / "data_meta" / "stim_trials.json"
RECORDINGS = {
    "P1_pre":  dict(patient=1, trial_index=1, file_type="pre"),
    "P1_stim": dict(patient=1, trial_index=1, file_type="stim"),
    "P5_pre":  dict(patient=5, trial_index=1, file_type="pre"),
    "P5_stim": dict(patient=5, trial_index=1, file_type="stim"),
}
RECORDING = os.environ.get("RECORDING", "P1_pre")   # driven per job, like SIM_SNR

SECONDS = 600          # window length (records are 1 s). Kept fixed across recordings so the
                       # comparison unit is identical; the files are 652-3742 s.
DETECT_FS = 1000.0    # common rate ALL THREE arms run at (2000 Hz -> /2).
                      # Was 400 Hz. Raised because Delphos now reads the preprocessed signal
                      # too (see PREP_DELPHOS): at 400 Hz its 8-512 Hz detection band would be
                      # cut at the 200 Hz Nyquist, which is most of its range. 1000 Hz keeps it
                      # nearly intact while still halving the data.
MED_KERNEL = int(os.environ.get('MED_KERNEL', 5))
                      # decimate_recording's median filter, in NATIVE samples. 5 = 2.5 ms, the
                      # pipeline default. It used to be forced to 1 (off) here, because it is a
                      # NONLINEAR peak-shaving filter that Janca and Barkmeier saw and Delphos
                      # -- a compiled binary reading the raw file -- could not. That asymmetry
                      # is now closed the other way: the preprocessed signal is WRITTEN BACK TO
                      # AN EDF and Delphos reads that, so all three see the same array and the
                      # pipeline's own default can be restored.
PREP_DELPHOS = os.environ.get('PREP_DELPHOS', '1') == '1'
                      # write the post-montage, median-filtered, decimated signal to an EDF and
                      # point Delphos at it, instead of the raw file. This is the ONLY way to
                      # give Delphos the same input as the other two -- it is a compiled binary
                      # that reads a file, so it cannot be handed an array.
                      # Delphos derives its OWN bipolar montage by default; since this EDF is
                      # already bipolar it must be told `bipolar: False`, exactly as the sim
                      # path does. Get that wrong and it pairs the pairs, silently.
                      # Set 0 to restore the old behaviour (Delphos on the raw file).
FILL_ALL = os.environ.get('FILL_ALL', '1') == '1'
                      # AR-fill the masked samples ONCE, in the shared preprocessed array,
                      # before ANY detector sees it -- including before the EDF Delphos reads.
                      #
                      # WHY. Masking removes detections INSIDE artefact after the fact, but the
                      # artefact still reached the detector and biased whatever it estimates
                      # from the data: Barkmeier's per-block mean/std, Janca's per-channel
                      # background, Delphos's TF whitening. Measured on P1: on the L/B/H/T
                      # shafts, Delphos runs at 0.059 Hz/ch in the stim-free baseline and
                      # 0.181 in the stim recording's OFF blocks -- 3.1x, on data with no
                      # stimulation in it -- while the rest of the implant sits still
                      # (0.077 -> 0.071). The artefact corrupts the threshold for the whole
                      # window, so detections in clean periods are wrong.
                      #
                      # seeg.fill_mask_ar fits a 20th-order AR model per channel to the CLEAN
                      # samples and substitutes median + spectrum-matched coloured noise. Its
                      # own docstring gives this exact rationale for Barkmeier; it applies
                      # verbatim to the other two.
                      #
                      # NOT EXCISION. Cutting the masked spans and re-joining would leave a
                      # step discontinuity at every join, and Delphos detects sharp edges by
                      # construction -- the synthetic template had exactly this bug, and fixing
                      # it moved Delphos's SNR-12 precision 0.810 -> 0.997. Filling avoids
                      # manufacturing thousands of them.
                      #
                      # Filled against the DILATED mask -- a DELIBERATE DEPARTURE from
                      # seeg.detect_spikes, which fills the undilated one. Two reasons:
                      #  * artefact does not stop at the flagged samples. A low-dynamic-range
                      #    stretch has a step at each end and ringing either side, and the mask
                      #    is 2 s epoch-resolution so those edges routinely fall just outside
                      #    it. Leaving them in puts the sharpest features in the recording into
                      #    the estimator -- exactly what this is meant to prevent, and exactly
                      #    what Delphos fires on.
                      #  * consistency. Detections are REJECTED inside the dilated mask and
                      #    clean_sec (hence the 80% channel gate) is computed from it. Treating
                      #    a region as too contaminated to accept a detection from, but clean
                      #    enough to fit the normaliser on, is incoherent.
                      # The error is asymmetric: filling a little too much costs some real
                      # signal replaced by spectrum-matched noise; filling too little leaves
                      # the artefact in every threshold.
FILL_BAD_SAMPLES = os.environ.get('FILL_BAD', '0') == '1'
                      # Barkmeier's OWN fill, inside seeg.detect_spikes. Stays OFF: with
                      # FILL_ALL the array is already filled, and filling twice would refit the
                      # AR model on a signal that is partly synthetic.
                      # seeg.detect_spikes defaults this TRUE, which AR-fills masked
                      # regions for Barkmeier ONLY -- Janca and Delphos see the real artefact.
                      # It matters even though those regions are masked afterwards, because
                      # Barkmeier's block normalisers (SCALE, std_coeff) are computed over the
                      # whole block INCLUDING the fill. False keeps all three symmetric.
QC_NATIVE = os.environ.get('QC_NATIVE', '1') == '1'
                      # run windowed_artefact_detector at the file's own 2 kHz rather than on
                      # the decimated array. gradThr is a PER-SAMPLE threshold with no rate
                      # term (artefact.py: max|diff| > gradThr), so at 400 Hz it is exactly
                      # FACTOR times stricter in uV/ms than the 2 kHz it was calibrated for.
                      # Running native also keeps stim_bins and the QC epochs in ONE sample
                      # space, and sidesteps asking how much of a 145 Hz stim band survives
                      # the anti-alias filter. Costs nothing: decimate_recording already keeps
                      # the native array in dec["raw"].
                      #
                      # KEEP THIS ON NOW THAT MED_KERNEL=5. Two reasons, both load-bearing:
                      #  1. dec["raw"] is the UNFILTERED native array, so gradThr=400 is still
                      #     the number it was calibrated against. Running QC on the decimated,
                      #     median-filtered signal instead would need a re-tuned threshold that
                      #     nobody has measured yet -- an assumption this avoids entirely.
                      #  2. decimate_recording copies info WITHOUT rescaling stim_bins, which
                      #     detect_stim wrote in NATIVE samples. QC on the decimated array
                      #     would compare epoch centres (decimated) against stim_bins (native)
                      #     and mislabel every isOn by FACTOR. Silent, and it would land
                      #     squarely on the ON/OFF results.
                      # The mask therefore describes the RAW recording, which is the right
                      # thing: an epoch that was bad is bad regardless of what we filtered
                      # afterwards. It may over-mask slightly where the median already removed
                      # an impulse -- conservative, and shared by all three detectors alike.
DILATE_MS = TROUGH_SEARCH_MS + 20   # artefact-exclusion radius, applied to EVERY detector.
                      # DERIVED, not a literal: this is seeg.detect_spikes' own default
                      # (its 40 ms trough search + a 20 ms buffer). Writing 60 here would
                      # silently stop tracking trough_search_ms the moment it is swept.
MASK_ARTEFACTS = True # drop detections inside the dilated artefact mask (all detectors alike)
MERGE_MS = 100.0      # shared polyspike rule: marks closer than this collapse to one.
                      # A cutoff this size puts the comparison at EVENT level not COMPONENT level,
                      # and that choice is forced by Delphos: it detects time-frequency BLOBS,
                      # so a polyspike run inside one blob is one detection and it has no
                      # sub-blob events to expose. Merging can move Janca and Barkmeier toward
                      # Delphos; nothing moves Delphos the other way. At 20 ms the three were
                      # counting different things -- measured on the real 600 s baseline, the
                      # fraction of inter-detection intervals under 50 ms was Janca 15.1%,
                      # Barkmeier 2.8%, Delphos 1.6%, which is a counting convention and not a
                      # difference in what was found.
                      # 100 ms is the current working value. It is safe for the SIMULATION
                      # (true refractory 200 ms with a point mass ON it, so nothing real can
                      # be merged) and sits just under Janca's published 120 ms default, which
                      # is the only clinician-chosen prior available for this question.
                      # IT IS NOT SETTLED FOR REAL DATA -- polyspikes there are a continuum.
                      # Work through figures/real/polyspike_*.png before defending it.
                      # NOTE re-running real data at this value DESTROYS the 20-300 ms pairs
                      # that review needs; archive/detections_merge20.npz keeps a copy.
                      # NOTE at DETECT_FS=400 one sample is 2.5 ms, so anything below that
                      # would only remove exact duplicates.
                      # Barkmeier and Delphos get this from `_merge_close`; Janca gets it from
                      # its own union via `_janca_pt`, which is NOT MERGE_MS/1000 -- read that
                      # docstring before changing either. Verify with the min inter-detection
                      # interval in detections.npz: it must be the same for all three.
TOL_MS = 50           # agreement tolerance
INTERACTIVE = False   # open the scroll/zoom viewer after the baseline run
RUN_DELPHOS = os.environ.get("RUN_DELPHOS", "1") == "1"
                      # False -> skip the Delphos arm entirely (2-panel raster, as before).
                      # run_windows.py sets RUN_DELPHOS=0 for every window and runs Delphos
                      # ONCE over the assembled whole-file EDF instead: it already tiles
                      # internally on free RAM, so windowing it would add process overhead
                      # AND re-whiten its time-frequency plane far more often than needed,
                      # making its normalisation depend on OUR window size.

VIOLET = "#4a3aa7"    # Delphos; from the QC palette -- third hue, no red/green pairing

# ---- detector defaults (reverted) ----
# dec=0 keeps DETECT_FS. `pt` (polyspike_union_time) is DELIBERATELY moved off its 0.12 s
# default and tied to MERGE_MS: it is Janca's INTERNAL union, so a post-hoc MERGE_MS cannot
# undo it, and leaving it at 0.12 while the others merge at MERGE_MS means Janca is still
# collapsing bursts nobody else collapses. Tying them makes MERGE_MS one dial for all three.
# `pt` only sets a merge span (janca_detect_spikes.py:269-271, :370) -- it does not change the
# threshold or the envelope -- so this suppresses merging without altering what is detected.
# It is NOT set here: `pt` needs `fs`, and it needs COMPENSATING (see _janca_pt below) --
# pt=MERGE_MS/1000 does not give Janca a MERGE_MS floor.
# Set JANCA = dict(dec=0) and JANCA_PT = None to restore Janca's published default (0.12 s)
# and accept the asymmetry.
JANCA = dict(dec=0, fl=10.0, fh=50.0)   # band narrowed from the 10-60 default
BARK = dict(LS=3.0, RS=3.0, TAMP=1200.0, LD=8, RD=8,
            # 1200 is the REAL-DATA operating point, tuned to match Janca's count. Do not move
            # it for a simulation experiment: it was briefly dropped to 400 for a sim sweep and
            # the next real run silently came back with Barkmeier at 28351 instead of 14420 --
            # a 2x change that had nothing to do with the merge being tested. Sim-specific
            # operating points belong in run_sim_suite.SWEEPS (via SIM_OVERRIDE), which is
            # exactly what that mechanism is for.
            # TAMP is NOT in uV:
            # mDetectSpike.m:284 rescales each block by 100/median(mean|EEG|) first, which
            # measured 7.02 on the real baseline, so 1200 there means ~171 uV of summed
            # half-heights. 400 is the value make_sim_detector_test_data.m itself used at
            # build time. Swept alongside, so the curve is visible rather than assumed.
            # These three are seeg.spikes' own defaults, IMPORTED not retyped. They were
            # previously written as literals identical to the upstream values, so an upstream
            # change would not have reached this comparison -- and DILATE_MS above depends on
            # trough_search_ms agreeing.
            std_coeff=STD_COEFF, trough_search_ms=TROUGH_SEARCH_MS, filter_spec=FILTER_SPEC)
# Delphos runs on the RAW EDF at its agreed operating point (Delphos.md): Spk_thr 40,
# 8-512 Hz, its own bipolar montage, RAM pinned to 12 GB so the internal tiling -- hence the
# detections -- stays in one regime. Keep pin/chunk FIXED across everything you compare.
DELPHOS = dict(pin_free_ram_gb=12, Spk_thr=50, Spk_time_thr=1.25, chunk_sec=None)   # exe path etc: delphos_detect_spikes.DEFAULTS
# ~5 min/call -> memoised by file+window+params. Anchored to THIS FILE, not the cwd: a miss
# costs 5 minutes, so the cache must not quietly move when you run the script from elsewhere.
DELPHOS_CACHE = _ROOT / ".delphos_cache"
# Detections dumped for evaluate_detectors.py (the evaluation plots read ONLY this file).
DETECTIONS_NPZ = _ROOT / "detections.npz"

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

# ---- SIMULATION: run the SAME three detectors on synthetic data with KNOWN spike times. ----
# Everything below this config block is shared with the real-data path on purpose: the whole
# value of the sim is that the detectors, the artefact mask, the 20 ms merge and the npz dump
# are byte-for-byte the ones used on the patient recording. See sim_data.py.
# ---- LABELLED BENCHMARK: the same three detectors on data with EXPERT-MARKED spikes -------
# BIDS_SUBJECT=sub-01 runs one subject of the BIDS iEEG sleep dataset (25 subjects, 852
# expert-marked IEDs). This is the only arm scored against a human rather than against the
# other detectors or against spikes we generated ourselves.
#
# THE DATA IS RUN EXACTLY AS IT IS: no median, no anti-alias, no decimation, NO MONTAGE.
# Unmontaged is the important one -- the expert marks name monopolar contacts, so leaving it
# unmontaged makes the channels BE the labelled channels. Montaging would force a
# contact->pair mapping, and whether to credit one pair or both would silently set recall.
# The cost is that it is scalp-referenced, so a discharge is common-mode across nearby
# contacts: a caveat on the absolute numbers, not on the ranking between detectors.
BIDS_SUBJECT = os.environ.get("BIDS_SUBJECT", "")
BIDS_ROOT = Path(r"C:\Users\amoo0039\Documents\ieeg_ieds_bids_final\ieeg_ieds_bids")

SIMULATE = False       # True -> build/load a sim EDF instead of the patient EDF
SIMULATE = SIMULATE or os.environ.get("SIM_FORCE") == "1"   # run_sim_suite.py flips it per job
SIM_SNR = float(os.environ.get("SIM_SNR", 8.0))          # which SNR level to run
SIM_POINT = os.environ.get("SIM_POINT", "op")            # run label -> npz filename
SIM_OVERRIDE = json.loads(os.environ.get("SIM_OVERRIDE", "") or "{}")   # {detector,param,value}
SIM_CLEAN_MASK = True  # zero the QC mask (the sim contains no artefacts -- see below)
SIM_RUNS = _ROOT / "sim_runs"
# The env vars just DEFAULT to the constants above, so nothing changes when they are unset.
# They exist so run_sim_suite.py can drive 6 SNR levels + 15 sweep points without editing this
# file 21 times.

# ----------------------------------------------------------------------
_sim = None
if BIDS_SUBJECT:
    EDF = str(BIDS_ROOT / BIDS_SUBJECT / "ieeg" / f"{BIDS_SUBJECT}_task-sleep_ieeg.edf")
    if not Path(EDF).is_file():
        raise SystemExit(f"{EDF} not found")
    _bhdr = read_edf_header(EDF)
    SECONDS = int(_bhdr["NumDataRecords"] * _bhdr["DataRecordDuration"])
    STOP_REC = SECONDS
    DETECT_FS = float(_bhdr["SampleRate"])   # its OWN rate -> FACTOR 1 -> no decimation
    MED_KERNEL = 1                           # and no median. Run as recorded.
    TRIAL = None                             # sleep recording, no stimulation
    DELPHOS = {**DELPHOS, "bipolar": False}  # already the channels the labels name
    REC_META = dict(rec_id=f"bids_{BIDS_SUBJECT}", patient=-1, condition="bids",
                    stim_hz=float("nan"))
    DETECTIONS_NPZ = _ROOT / "runs" / f"bids_{BIDS_SUBJECT}.npz"
    print(f"--- {BIDS_SUBJECT}: {SECONDS}s at {DETECT_FS:g} Hz, "
          f"{_bhdr['NumSignals']} channels, unmontaged ---")
elif SIMULATE:
    from sdc.detect import sim_data
    _sim = sim_data.ensure_sim_edf(snr=SIM_SNR)
    EDF = str(_sim["edf"])
    if SECONDS != _sim["cfg"]["dur_sec"]:
        raise SystemExit(f"SECONDS={SECONDS} but the sim is {_sim['cfg']['dur_sec']}s long; "
                         f"they must match or the window is not the recording.")
    # SIM1..SIM16 are ALREADY bipolar-like (the AR models were fitted on real bipolar
    # channels), so Delphos must not build a montage on top of them either.
    DELPHOS = {**DELPHOS, "bipolar": False}
    SIM_RUNS.mkdir(parents=True, exist_ok=True)
    # The generator config hash is in the name for the same reason it is in the EDF name: a
    # changed stimulus must never silently reuse -- or be scored alongside -- runs made on the
    # previous one.
    DETECTIONS_NPZ = (SIM_RUNS / f"sim_{_sim['cfg']['tag']}_{sim_data.cfg_hash(_sim['cfg'])}"
                                 f"_snr{SIM_SNR:g}_{SIM_POINT}.npz")

# BIDS mode has already set TRIAL / REC_META / EDF / DETECTIONS_NPZ above and must not fall
# through here. It did once: the block below re-resolved EDF from RECORDINGS, so a
# BIDS_SUBJECT run silently analysed P1 instead -- 181 s of the wrong recording, written over
# a control run. The variant naming protects against two CONFIGS colliding; it cannot protect
# against a MODE falling through to the default recording. Hence the explicit guard.
if not BIDS_SUBJECT:
    TRIAL = None        # the stim trial record, or None for baseline/'pre'
    REC_META = dict(rec_id="sim" if SIMULATE else RECORDING, patient=-1, condition="sim",
                    stim_hz=float("nan"))
if not SIMULATE and not BIDS_SUBJECT:
    _cfg = RECORDINGS[RECORDING]
    _entry = get_trial(get_patient(load_trials(META_PATH), _cfg["patient"]),
                       _cfg["trial_index"])
    _stem, TRIAL = resolve_file(_entry, _cfg["file_type"])
    EDF = str(BASE_DIR / f"P{_cfg['patient']}" / f"{_stem}.edf")
    REC_META = dict(rec_id=RECORDING, patient=_cfg["patient"], condition=_cfg["file_type"],
                    stim_hz=float(TRIAL["stim_frequency"]) if TRIAL else float("nan"))
    # The filename carries any DEVIATION from the canonical config, and nothing else. A run at
    # the canonical settings is plain `<rec_id>.npz` -- so downstream defaults keep working --
    # while a variant gets a suffix and CANNOT overwrite it.
    # This is not hypothetical tidiness: a MED_KERNEL=1 control run silently replaced the
    # canonical MED_KERNEL=5 result minutes after it was produced, and only a hand copy saved
    # it. Same failure as the TAMP=400 incident noted above -- a differently-configured run
    # sitting in the file where the canonical one is expected.
    _variant = "".join([
        "" if MED_KERNEL == 5 else f"_med{MED_KERNEL}",
        "" if DETECT_FS == 1000.0 else f"_{DETECT_FS:g}Hz",
        "" if PREP_DELPHOS else "_rawdelphos",
        "" if FILL_ALL else "_nofill",
        "" if QC_NATIVE else "_qcdec",
        "_fillbad" if FILL_BAD_SAMPLES else "",
        "" if MERGE_MS == 100.0 else f"_merge{MERGE_MS:g}",
    ])
    DETECTIONS_NPZ = _ROOT / "runs" / f"{RECORDING}{_variant}.npz"
    if _variant:
        print(f"[config] non-canonical run -> {DETECTIONS_NPZ.name}")
    print(f"--- {RECORDING}: P{_cfg['patient']} trial {_cfg['trial_index']} "
          f"{_cfg['file_type']} -> {_stem}.edf"
          + (f"  ({TRIAL['target']} {TRIAL['stim_frequency']}Hz"
             f"{', intermittent' if TRIAL.get('intermittent') else ''})" if TRIAL else "") + " ---")

hdr = read_edf_header(EDF)
FACTOR = int(round(hdr["SampleRate"] / DETECT_FS))

# WHICH RECORDS. Default is the first SECONDS of the file, which is what every result up to
# this point used. run_windows.py drives START_REC/STOP_REC to walk a whole recording in
# RAM-sized windows -- the files are 652-3742 s and P5_stim alone is 12 GB at native rate, so
# a whole-file load is not an option. Records are 1 s and the bounds are 1-based inclusive,
# matching seeg.edf.window_bounds and load_edf_segment.
START_REC = int(os.environ.get("START_REC", 1))
STOP_REC = int(os.environ.get("STOP_REC", SECONDS))
SECONDS = STOP_REC - START_REC + 1          # everything downstream measures THIS window
WINDOW_TAG = os.environ.get("WINDOW_TAG", "")   # set by the driver; keeps per-window outputs
                                                # from overwriting each other
DUMP_DEC = os.environ.get("DUMP_DEC", "")
                      # path to save the decimated (post-median, post-montage) array to.
                      # The driver concatenates these across windows and writes ONE EDF
                      # for Delphos. Saved WHOLE, not trimmed: the driver owns the
                      # overlap arithmetic, so this stays a plain "process these
                      # records" script with no notion of windowing semantics.
if WINDOW_TAG:
    DETECTIONS_NPZ = DETECTIONS_NPZ.with_name(f"{DETECTIONS_NPZ.stem}{WINDOW_TAG}.npz")
    print(f"[window] records {START_REC}-{STOP_REC} ({SECONDS}s) -> {DETECTIONS_NPZ.name}")
rec = load_edf_segment(EDF, hdr, start_rec=START_REC, stop_rec=STOP_REC)

if not SIMULATE and not BIDS_SUBJECT:
    # derive_montage's contact regex matches SIM1..SIM16 and WOULD build 15 pairs
    # SIM1_SIM2, SIM2_SIM3, ... Re-montaging already-bipolar sim channels would smear every
    # ground-truth spike across two pairs with opposite polarity, so the sim skips this.
    rec = apply_montage(rec, derive_montage(rec["info"]["SelectedSignals"]))

# Stim detection MUST come after the montage (stim_channels are post-montage PAIR names) and
# before the QC. make_cfg_artefact(lsb, trial) on its own does NOTHING: the stim-spectral rule
# is gated on `is_on[e] AND have_stim`, and is_on comes from info["stim_bins"], which only
# detect_stim writes. Skip this and the stim rule silently never fires.
if TRIAL is not None:
    rec["info"]["stim_trial"] = TRIAL
    rec = detect_stim(rec)

dec = decimate_recording(rec, factor=FACTOR, med_kernel=MED_KERNEL)
fs = dec["info"]["SampleRate"]
names = dec["info"]["SelectedSignals"]
n_chan = len(names)

# QC at the file's own rate (see QC_NATIVE). decimate_recording already kept the native array
# in dec["raw"], so this costs nothing extra -- and it keeps stim_bins and the QC epochs in the
# SAME sample space, which is the difference between isOn being right and being wrong by FACTOR.
if QC_NATIVE:
    _qc_rec = {"data": dec["raw"]["data"],
               "info": {**rec["info"], "SampleRate": dec["raw"]["fs"],
                        "NSamples": dec["raw"]["data"].shape[0]}}
else:
    _qc_rec = dec
_qc_fs = _qc_rec["info"]["SampleRate"]
qc = windowed_artefact_detector(_qc_rec, make_cfg_artefact(hdr["lsb"], trial=TRIAL))
if TRIAL is not None:
    _tags = {t for r in qc["epoch"]["reason"].ravel() if r for t in r.split(",")}
    _on = qc["epoch"]["isOn"]
    print(f"[stim] {_on.sum()}/{_on.size} epochs ON; reason tags {sorted(_tags)}")
    if "stim_spec" not in _tags:
        print("[warn] stim_spec never fired on a stim recording -- check detect_stim ran and "
              "that stim_frequency is above the LF band.")
if SIMULATE and SIM_CLEAN_MASK:
    # The sim injects NO artefacts, so anything the QC flags is a false positive -- and a
    # masked epoch silently deletes TRUE spikes, which reads as a detector missing them.
    # Overwrite sampleMask in place rather than faking a qc dict: view() needs qc["epoch"],
    # and an all-clear mask also makes fill_mask_ar a no-op, so Barkmeier's input is exactly
    # the synthesised signal. The printed fraction is a free QC-on-known-clean-data check.
    _masked = float(np.mean(qc["sampleMask"]))
    print(f"[sim] QC would mask {_masked:.4%} of synthetic samples that contain no artefacts "
          f"-- zeroing it. (>0.1% is a finding about windowed_artefact_detector, not about "
          f"any detector; rerun with SIM_CLEAN_MASK=False to see what it costs.)")
    qc["sampleMask"] = np.zeros_like(qc["sampleMask"])
    _SIM_MASK_ZEROED = True
def _fold_to_detection_grid(mask):
    """Native-rate (n_samp, n_chan) bool -> the DETECT_FS grid.

    reshape-and-`any`, NOT striding: a decimated sample is bad if ANY of the FACTOR native
    samples behind it was bad. Striding would keep one native point in five and quietly drop
    most of the artefact."""
    n_det = dec["data"].shape[0]
    if mask.shape[0] == n_det:
        return mask
    n_keep = (mask.shape[0] // FACTOR) * FACTOR
    out = mask[:n_keep].reshape(-1, FACTOR, mask.shape[1]).any(axis=1)
    if out.shape[0] < n_det:                        # trailing partial group
        out = np.vstack([out, mask[n_keep:].any(axis=0, keepdims=True)])
    return out[:n_det]


# Dilate at whatever rate the QC ran at -- finer is better, so dilate BEFORE folding -- using
# seeg's own helper, so the exclusion radius has ONE definition shared with detect_spikes.
dmask = _fold_to_detection_grid(dilate_mask(qc["sampleMask"], _qc_fs, DILATE_MS))

# seeg.detect_spikes validates sampleMask against the recording it is given, and it is given
# the DECIMATED array -- so anything consuming `dec` needs the mask on that grid. Keep the
# native qc for the mask/ON-OFF arithmetic above and hand this one to the detector and viewer.
QC_DET = dict(qc)
QC_DET["sampleMask"] = _fold_to_detection_grid(qc["sampleMask"])
if QC_NATIVE and "epoch" in qc and "starts" in qc["epoch"]:
    _ep = dict(qc["epoch"])
    _ep["starts"] = (np.asarray(qc["epoch"]["starts"]) // FACTOR).astype(int)
    if "epochSamp" in _ep:
        _ep["epochSamp"] = int(round(_ep["epochSamp"] / FACTOR))
    QC_DET["epoch"] = _ep
if QC_NATIVE:
    print(f"[qc] ran at {_qc_fs:g} Hz, mask folded to {dec['data'].shape[0]} samples "
          f"at {fs:g} Hz ({dmask.mean():.2%} masked after dilation)")

# ---- AR-fill the masked samples, ONCE, for every detector (see FILL_ALL) --------------
if FILL_ALL and np.any(dmask):
    _before = float(np.mean(np.abs(dec["data"])))
    dec["data"] = fill_mask_ar({"data": dec["data"], "info": dec["info"]},
                               {"sampleMask": dmask})["data"]
    print(f"[fill] AR-filled {dmask.mean():.2%} of samples (dilated mask); "
          f"mean |x| {_before:.1f} -> {np.mean(np.abs(dec['data'])):.1f} uV")

# Per-sample stim ON/OFF on the DETECTION grid, expanded from the QC epochs. Stored rather than
# used to pre-split, because a Delphos call is ~5 min: the split definition has to be
# changeable downstream without re-running anything.
ON_MASK = np.zeros(dec["data"].shape[0], bool)
if TRIAL is not None:
    _starts = qc["epoch"]["starts"]
    _elen = int(round(qc["epoch"]["epochSamp"])) if "epochSamp" in qc["epoch"] else \
        int(round(2.0 * _qc_fs))
    _scale = fs / _qc_fs                       # QC-rate sample -> detection-rate sample
    for _s, _isOn in zip(_starts, qc["epoch"]["isOn"]):
        if _isOn:
            _a = int(round(_s * _scale))
            ON_MASK[_a:_a + int(round(_elen * _scale))] = True
    print(f"[stim] {ON_MASK.mean():.1%} of the scored window is stim-ON "
          f"({ON_MASK.sum() / fs:.0f}s ON, {(~ON_MASK).sum() / fs:.0f}s OFF)")

# The ON SEGMENT BOUNDARIES, not just the per-sample flag. A per-detection ON/OFF label is
# enough to count spikes but not to measure them: an ISI spanning an OFF block, or a rate bin
# straddling a boundary, both need to know WHERE the blocks are. Stored as [start, stop)
# detection-grid samples; cond.py turns them into the time base every downstream figure uses.
if ON_MASK.any():
    _edge = np.flatnonzero(np.diff(ON_MASK.astype(np.int8))) + 1   # every ON<->OFF transition
    _b = np.concatenate([np.array([0] if ON_MASK[0] else [], np.int64), _edge,
                         np.array([ON_MASK.size] if ON_MASK[-1] else [], np.int64)])
    ON_RUNS = _b.reshape(-1, 2).astype(np.int64)   # pairs: a leading/trailing ON block
else:                                              # contributes its own open/close edge
    ON_RUNS = np.zeros((0, 2), np.int64)


# ----------------------------------------------------------------------
# The preprocessed EDF -- the only way to give Delphos the same input as the other two.
# ----------------------------------------------------------------------
# Delphos is a compiled binary that reads a file, so it cannot be handed `dec["data"]`. Writing
# that array back out as an EDF is what finally closes the median-filter asymmetry that forced
# MED_KERNEL=1 for every run before this one.
#
# write_edf/verify_edf are sim_data's, reused rather than reimplemented: verify_edf round-trips
# the file back through BOTH readers the pipeline uses and raises on any mismatch. That check is
# the whole reason to reuse it -- a silently wrong EDF would cost a 5-minute Delphos call to
# discover, and would look like a Delphos result rather than a writer bug.
PREP_EDF = None
if PREP_DELPHOS and RUN_DELPHOS and not SIMULATE and not BIDS_SUBJECT:
    from sdc.detect.sim_data import write_edf, verify_edf
    _prep_dir = _ROOT / "prep_edf"
    _prep_dir.mkdir(parents=True, exist_ok=True)
    # Name carries everything that changes the CONTENT, so a stale file can never be picked up
    # after a config change -- and so Delphos's cache (keyed on path + size) cannot collide.
    PREP_EDF = str(_prep_dir / f"{REC_META['rec_id']}{WINDOW_TAG}"
                   f"_med{MED_KERNEL}_{fs:g}Hz{'_fill' if FILL_ALL else '_nofill'}"
                   f"_{START_REC}-{STOP_REC}.edf")
    if Path(PREP_EDF).is_file():
        print(f"[prep] reusing {Path(PREP_EDF).name}")
    else:
        print(f"[prep] writing {Path(PREP_EDF).name} "
              f"({dec['data'].shape[0]} x {dec['data'].shape[1]} at {fs:g} Hz) ...")
        write_edf(PREP_EDF, dec["data"], list(names), fs)
        verify_edf(PREP_EDF, dec["data"], list(names), fs)
        print(f"[prep] verified: round-trips through both pipeline readers")


if DUMP_DEC:
    Path(DUMP_DEC).parent.mkdir(parents=True, exist_ok=True)
    # Per-SECOND clean-sample counts per channel, so the driver can compute analysable time
    # over an arbitrary interior EXACTLY rather than scaling a whole-window fraction. Interiors
    # always fall on whole seconds, so 1 s granularity loses nothing. ~110 kB per window
    # against ~55 MB for the mask itself, which is why it is counts and not the mask.
    _sps = int(round(fs))
    _n_sec = dmask.shape[0] // _sps
    _clean = (~dmask[:_n_sec * _sps]).reshape(_n_sec, _sps, -1).sum(axis=1).astype(np.uint16)
    np.save(str(Path(DUMP_DEC).with_suffix("")) + "_clean.npy", _clean)
    _on = (ON_MASK[:_n_sec * _sps].reshape(_n_sec, _sps).any(axis=1)
           if ON_MASK.any() else np.zeros(_n_sec, bool))
    np.save(str(Path(DUMP_DEC).with_suffix("")) + "_on.npy", _on)
    # The DILATED ARTEFACT MASK itself, packed to 1 bit per sample. Needed because
    # Delphos is run by the driver over the assembled file and must go through the SAME
    # mask as the other two -- without this it is the only unmasked arm, which is the
    # exact asymmetry the shared mask exists to remove.
    np.save(str(Path(DUMP_DEC).with_suffix("")) + "_mask.npy",
            np.packbits(dmask, axis=0))
    np.save(str(Path(DUMP_DEC).with_suffix("")) + "_maskshape.npy",
            np.array(dmask.shape, np.int64))
    # float32: the EDF this ends up in is int16, so float64 would be 2x the disk and
    # the bytes would be discarded by write_edf's quantisation anyway.
    np.save(DUMP_DEC, dec["data"].astype(np.float32))
    print(f"[dump] {Path(DUMP_DEC).name}  {dec['data'].shape} at {fs:g} Hz")

_SAMPLE_MS = 1000.0 / fs
if MERGE_MS and MERGE_MS < _SAMPLE_MS:
    print(f"[warn] MERGE_MS={MERGE_MS:g} ms is finer than one sample at {fs:g} Hz "
          f"({_SAMPLE_MS:.1f} ms), so it can only remove exact duplicates. Detections live on "
          f"the sample grid: the closest two distinct spikes can be is {_SAMPLE_MS:.1f} ms. "
          f"Raise MERGE_MS above that to actually merge anything.")


# _merge_close now lives upstream as seeg.merge_close -- it is the rule three
# detectors have to agree on, so it belongs with the pipeline rather than in a
# caller. Behaviour is unchanged (verified over 300 random cases).


def _janca_pt(merge_ms, fs):
    """Janca `pt` (s) that gives its INTERNAL union the same floor `_merge_close` gives
    everyone else. Returns None when there is nothing to merge.

    pt = MERGE_MS/1000 is the obvious choice and it is WRONG by two samples, because the two
    rules are not the same kind of rule:
      * `_merge_close` is a strict inequality -- merges d < gap, so the closest survivor is
        `ceil(gap)` samples apart.
      * `_detection_union` (janca_detect_spikes.py:301-315, faithful to
        spike_detector_hilbert_v24.m:786-791) is a morphological CLOSING. It takes
        `us = ceil(pt*fs)`, bumps it to the next ODD value if even (+1, needed for a
        symmetric 'same' convolution), and a closing with a length-L element fills gaps of up
        to L samples -- so it merges d <= L, one further than a `< L` test would.
    At MERGE_MS=20, fs=400 that is gap=8 -> us=8 -> bumped to 9 -> merges d<=9, closest
    survivor 10 samples = 25.0 ms, against 20.0 ms for Barkmeier and Delphos. Measured, not
    inferred: min inter-detection interval in detections.npz was 10 / 8 / 8 samples.

    So ask for L = ceil(gap)-1 instead. Only ODD L is reachable (the bump), so an even target
    rounds DOWN -- under-merging, which `_merge_close` then finishes, rather than over-merging,
    which nothing can undo."""
    gap = merge_ms / 1000.0 * fs
    L = int(np.ceil(gap)) - 1
    if L % 2 == 0:
        L -= 1                      # only odd lengths survive the even-bump
    if L < 1:
        return 0.0                  # us=0 -> bumped to 1 -> closing is the identity
    # HALF a sample below L/fs, not L/fs: the union takes ceil(pt*fs), and pt=7/400 is not
    # exact in binary -- 0.0175*400 = 7.000000000000001, so ceil gives 8, the even-bump makes
    # it 9, and the 25 ms floor comes straight back. Landing mid-sample makes the ceil immune
    # to which side of L the product falls on.
    pt = (L - 0.5) / fs
    if L != int(np.ceil(gap)) - 1:
        print(f"[note] Janca pt set to {pt*1000:.1f} ms (union span {L} samples): the closing "
              f"can only use odd spans, so it merges <= {L} samples and _merge_close removes "
              f"the rest. Same {merge_ms:g} ms floor, reached in two steps.")
    return pt


JANCA_PT = _janca_pt(MERGE_MS, fs)


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
        idx = merge_close(idx, gap)
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
    """Janca -> per-channel sample indices at `fs`. Override any Janca setting by keyword.

    `pt` comes from JANCA_PT (compensated -- see `_janca_pt`) unless JANCA or an explicit
    keyword sets it, so sweeping `pt` still works."""
    p = {**JANCA, **override}
    if "pt" not in p and "polyspike_union_time" not in p and JANCA_PT is not None:
        p["pt"] = JANCA_PT
    out, _disch, _info = detect_janca(data, fs, **p)
    return _finalise([np.round(out["pos"][out["chan"] == c] * fs).astype(int)
                      for c in range(n_chan)], label)


def run_bark(recording, label=None, **override):
    """Barkmeier -> per-channel sample indices. Override LS/RS/TAMP/LD/RD/std_coeff/... by keyword."""
    p = {**BARK, **override}
    detect_barkmeier(recording, QC_DET, post_mask_spikes=False,
                     # FILL_BAD_SAMPLES: seeg defaults this True, which AR-fills masked regions
                     # for Barkmeier only. Its block normalisers are computed over the whole
                     # block INCLUDING the fill, so it is not neutralised by the later mask.
                     fill_bad_samples=FILL_BAD_SAMPLES,
                     det_thresholds=[p["LS"], p["RS"], p["TAMP"], p["LD"], p["RD"]],
                     std_coeff=p["std_coeff"], trough_search_ms=p["trough_search_ms"],
                     filter_spec=p["filter_spec"])
    return _finalise([np.asarray(s, int) for s in recording["info"]["DetectedSpikes"]], label)


def run_delphos(label=None, **override):
    """Delphos -> per-channel sample indices on the common `fs` axis, same channel order.

    Reads a FILE, not an array -- it is a compiled binary. With PREP_DELPHOS that file is the
    preprocessed EDF written by `_write_prep_edf()`, so it finally sees the same median-filtered,
    decimated, post-montage signal as Janca and Barkmeier. Without it, the raw EDF, and the
    median-filter asymmetry is back.

    Marker positions are absolute file seconds either way, so passing `fs` converts them onto
    the detection axis; `_finalise` then applies the same mask and merge as the other two."""
    # PREP_EDF is None when no preprocessed file was written -- BIDS mode runs the
    # recording exactly as it is, so Delphos reads the original. Guarding on the PATH,
    # not just the flag: with the flag alone this passed None and lost the arm.
    src = PREP_EDF if (PREP_DELPHOS and PREP_EDF) else EDF
    cfg = {**DELPHOS, **override}
    if PREP_DELPHOS:
        # The preprocessed EDF is ALREADY bipolar. Delphos montages whatever it is given
        # (bipolar=True by default), so leaving this on would pair the pairs -- R_8_R_9 with
        # R_9_R_10 -- and the label match would collapse. Same override the sim path uses.
        cfg["bipolar"] = False
    return _finalise(detect_delphos(src, names, fs, start_sec=0.0, duration_sec=SECONDS,
                                    cache_dir=DELPHOS_CACHE, **cfg), label)


def _match(a, b, tol):
    """Greedy one-to-one nearest match on sample indices. offset = a - b.

    Thin wrapper over spike_match.match, which is the SAME algorithm the ground-truth scorer
    uses. Keeping one implementation is deliberate: if agreement and accuracy were measured by
    two copies of this loop, a drift in either would make their numbers incomparable."""
    mask_a, _mask_b, offs = spike_match_fn(a, b, tol)
    return int(mask_a.sum()), offs.tolist()


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


def _dump_detections(dets, path, extra=None):
    """Dump the detections so the evaluation scripts can work from the OUTPUT alone -- no
    re-running this script (and no 5 min Delphos call) just to redraw a plot. Ragged
    per-channel arrays are flattened to (index, channel) pairs.

    Everything written must be a plain dtype: the readers use allow_pickle=False."""
    dump = {"names": np.array(names), "fs": fs, "seconds": SECONDS, "edf": EDF,
            "detectors": np.array([n for n, _, _ in dets])}
    for name, det, _ in dets:
        dump[f"{name}_idx"] = np.concatenate(det) if det else np.zeros(0, int)
        dump[f"{name}_chan"] = (np.concatenate([np.full(d.size, c, int)
                                                for c, d in enumerate(det)]) if det
                                else np.zeros(0, int))
    # Post-processing provenance goes in EVERY dump, real or simulated. Without it the
    # downstream plots have to hardcode the settings, and they go stale silently the moment a
    # constant here moves -- spike_statistics.py annotated its ISI histogram with a fixed
    # "Janca 120 ms polyspike union" long after MERGE_MS had been retied and applied to all
    # three detectors.
    dump.update({"merge_ms": float(MERGE_MS), "dilate_ms": float(DILATE_MS),
                 "tol_ms": float(TOL_MS), "detect_fs": float(DETECT_FS),
                 "mask_artefacts": np.int64(bool(MASK_ARTEFACTS)),
                 "janca_pt_ms": float(JANCA_PT * 1000) if JANCA_PT is not None else float("nan"),
                 # preprocessing toggles -- these MOVE the numbers, so a run that does not
                 # record them cannot be compared with one that does
                 "qc_native": np.int64(bool(QC_NATIVE)), "med_kernel": np.int64(MED_KERNEL),
                 "fill_all": np.int64(bool(FILL_ALL)),
                 # WHICH FILE DELPHOS READ. Before this existed, a run with the median filter
                 # on and one with it off were indistinguishable in the npz, and they are not
                 # comparable -- Delphos saw a different signal in each.
                 "delphos_input": "preprocessed" if (PREP_DELPHOS and not SIMULATE) else "raw",
                 "fill_bad_samples": np.int64(bool(FILL_BAD_SAMPLES)),
                 "mask_frac": float(np.mean(dmask)),
                 # recording identity -- absent entirely before, which is why a second
                 # recording used to overwrite the first
                 "rec_id": str(REC_META["rec_id"]), "patient": np.int64(REC_META["patient"]),
                 "condition": str(REC_META["condition"]),
                 "stim_hz": float(REC_META["stim_hz"]),
                 "sec_on": float(ON_MASK.sum() / fs),
                 "sec_off": float((~ON_MASK).sum() / fs),
                 "start_rec": np.int64(START_REC), "stop_rec": np.int64(STOP_REC),
                 "on_runs": ON_RUNS,
                 # ANALYSABLE seconds per channel, per condition -- wall-clock seconds minus
                 # the dilated artefact mask. This is the denominator a rate actually needs and
                 # the two differ a lot here: stim artefact pushed P1's masked fraction from
                 # 3.3% at baseline to 15.1%, and it is NOT spread evenly, so dividing stim-ON
                 # counts by 194 s charges every detector for time in which nothing could be
                 # detected -- and charges the channels nearest the stim contacts hardest.
                 # Per CHANNEL because the mask is per channel and a rate is a per-channel
                 # quantity; summing it first would average that away.
                 "clean_sec_on": (~dmask[ON_MASK]).sum(axis=0) / fs,
                 "clean_sec_off": (~dmask[~ON_MASK]).sum(axis=0) / fs})
    # per-detection stim-ON flag, parallel to {Det}_idx
    for name, det, _ in dets:
        idx = dump[f"{name}_idx"]
        dump[f"{name}_on"] = ON_MASK[idx] if idx.size else np.zeros(0, bool)
    dump.update(extra or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **dump)
    print(f"[saved] {Path(path).name}")


def _sim_extra(point):
    """Ground truth + provenance for a sim run, merged into the npz. The scorer groups runs by
    these STORED fields, never by parsing the filename."""
    z = np.load(_sim["truth"], allow_pickle=False)
    d, p, v = (SIM_OVERRIDE.get("detector", ""), SIM_OVERRIDE.get("param", ""),
               SIM_OVERRIDE.get("value", float("nan")))
    return {k: z[k] for k in ("truth_idx", "truth_chan", "truth_amp", "truth_fs", "snr",
                              "noise_std", "rates_per_min", "inband_snr", "inband_dets",
                              "template", "template_peak", "sim_tag", "sim_cfg_json",
                              "sim_cfg_hash")} | {
        "simulated": np.int64(1),
        "run_kind": "sweep" if SIM_OVERRIDE else "op",
        "run_point": str(point),
        "sweep_detector": str(d), "sweep_param": str(p), "sweep_value": float(v),
        "merge_ms": float(MERGE_MS), "dilate_ms": float(DILATE_MS),
        "tol_ms": float(TOL_MS), "detect_fs": float(DETECT_FS),
        "mask_artefacts": np.int64(bool(MASK_ARTEFACTS)),
        "clean_mask": np.int64(bool(SIM_CLEAN_MASK))}


# --- baseline run: every detector at its defaults ---
# A detector that cannot run is dropped from `dets` (and from the figure) rather than faked
# with zeros -- an empty panel would read as "found nothing", which is a different claim.
if SIM_OVERRIDE:
    # One knob moved off its default for this run, so the sweep can be scored against ground
    # truth. Applied to the module-level dict so the sweep machinery and the runners agree.
    _t = {"janca": JANCA, "barkmeier": BARK, "delphos": DELPHOS}
    _key = str(SIM_OVERRIDE["detector"]).lower()
    if _key not in _t:
        raise SystemExit(f"SIM_OVERRIDE detector {SIM_OVERRIDE['detector']!r} unknown; "
                         f"use one of {sorted(_t)}")
    # "LD+RD" sets both to the same value: the left/right half-wave criteria are a matched
    # pair, and moving one alone measures asymmetry rather than the criterion's strictness.
    for _p in str(SIM_OVERRIDE["param"]).split("+"):
        _t[_key][_p] = SIM_OVERRIDE["value"]
    print(f"[sim] operating point: {SIM_OVERRIDE['detector']}.{SIM_OVERRIDE['param']} = "
          f"{SIM_OVERRIDE['value']:g}")

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

_dump_detections(dets, DETECTIONS_NPZ, _sim_extra(SIM_POINT) if SIMULATE else None)
print(f"{' | '.join(f'{k} {v}' for k, v in counts.items())} "
      f"| {n_chan} bipolar ch, {SECONDS}s at {fs:g} Hz (defaults)")
if SIMULATE:
    _n_true = int(np.load(_sim["truth"], allow_pickle=False)["truth_idx"].size)
    print(f"[sim] SNR {SIM_SNR:g}: {_n_true} TRUE spikes injected -- "
          f"{' | '.join(f'{k} {v / max(_n_true, 1):.2f}x' for k, v in counts.items())}")

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
# Shade the stim-ON blocks across every panel. Without this a stim raster just has an
# unexplained change in density partway along, and the reader has no way to tell whether that
# is the stimulation or the patient.
for _a0, _a1 in (ON_RUNS / fs if ON_RUNS.size else []):
    for _ax in axes:
        _ax.axvspan(_a0, _a1, color="#f0c419", alpha=0.16, lw=0, zorder=0)
if ON_RUNS.size:
    axr.text(0.005, 0.97, "shaded = stim ON", transform=axr.transAxes, va="top",
             fontsize=8, color="#8a6d00")
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
# Anchored to THIS FILE, not the cwd; tagged and moved to sim_out/ in sim mode. An untagged
# raster meant every simulated run overwrote the real-data figure, so the file on disk was
# whichever sim job finished last -- a SWEEP point with one detector deliberately desensitised.
# SWEEP POINTS GET NO RASTER AT ALL: a raster of one detector at an off-default threshold
# answers no question, and writing one per point produced 30 near-identical figures.
_HERE = _ROOT
if SIMULATE and SIM_OVERRIDE:
    _RASTER = None
elif SIMULATE:
    _RASTER = _HERE / "figures" / "sim" / DETECTIONS_NPZ.stem / "compare_raster.png"
else:
    _RASTER = _HERE / "figures" / "real" / REC_META["rec_id"] / "compare_raster.png"
if WINDOW_TAG:
    # A per-window raster is a FRAGMENT, and every window writes to the same path -- so what
    # survives on disk is whichever window finished last, mislabelled as the recording. That is
    # worse than no figure: figures/real/P1_pre/compare_raster.png silently became 112 s of a
    # 652 s file. run_windows draws one raster for the assembled recording instead.
    _RASTER = None
if _RASTER is not None:
    _RASTER.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_RASTER, dpi=130)
    print(f"[saved] {_RASTER.name}")
plt.close(fig)


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
    view(dec, QC_DET, spikes={f"{name} ({counts[name]})": det for name, det, _ in dets},
         chans_per_page=15, t0=0, duration=30)
