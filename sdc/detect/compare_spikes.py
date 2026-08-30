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
from sdc.common.invariants import check_run

# ---- which recording ----------------------------------------------------------------
# Recordings are named by (patient, trial, condition) and RESOLVED through the pipeline's own
# identity chain: load_trials -> get_patient -> get_trial -> resolve_file. Nothing here
# hardcodes an EDF path. `resolve_file` returns (stem, trial_or_None), and that None for 'pre'
# is the pipeline-wide signal for "no stim" that make_cfg_artefact consumes -- so the stim
# rules switch themselves on and off correctly just by naming the condition.
BASE_DIR = Path(r"C:\Users\amoo0039\Documents\local")
META_PATH = BASE_DIR / "data_meta" / "stim_trials.json"
from sdc.detect.recordings import (RECORDINGS, edf_path, load_patient_montage,
                                   montage_path)   # noqa: E402  -- shared with
                                                  # run_windows.py, which used to keep its own
                                                  # copy. edf_path is the single resolver.
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
PULSE_BLANK_MS = float(os.environ.get('PULSE_BLANK_MS', 0) or 0)
                      # >0 turns on stimulation-PULSE blanking, an optional stage that runs on
                      # the RAW native signal before decimation. It must be before the median
                      # filter, not after: med_kernel=5 at 2 kHz already removes a 1-2 sample
                      # impulse, so a blank applied to the decimated array would be a no-op.
PULSE_MAX_PEAK_UV = float(os.environ.get('PULSE_MAX_PEAK_UV', 1e4))
PULSE_MAX_RAIL_FRAC = float(os.environ.get('PULSE_MAX_RAIL_FRAC', 0.05))
PULSE_FILL = os.environ.get('PULSE_FILL', 'auto')
                      # how blanked samples are repaired: 'auto' (interp for runs <= 4 samples,
                      # AR beyond), 'interp', or 'ar'. At 5 ms / 9 samples 'auto' means AR, and
                      # AR fill inserts synthesised noise that does not join the surrounding
                      # samples -- two discontinuities per blank. Delphos detects edges: with
                      # AR fill 81.9% of its detections landed within +-5 ms of a pulse
                      # (2% by chance) and its ratio went 1.88 -> 10.69.
_PB_SUF = ("" if not PULSE_BLANK_MS else
           f"_pb{PULSE_BLANK_MS:g}{ {'auto': '', 'interp': 'i', 'ar': 'a'}[PULSE_FILL] }")
                      # ONE definition, used for the npz name AND the prep EDF name. Blanking
                      # rewrites samples without changing their count, so a differently-filled
                      # file is the same SIZE -- and Delphos caches on path + size.
                      # the recoverability gate. Channels above the peak, or railing on more
                      # than that fraction of pulses, are NOT blanked and therefore keep QC's
                      # verdict on their untouched signal -- every rule in
                      # windowed_artefact_detector is per-channel, so no second QC pass is
                      # needed to achieve that. See seeg.artefact.pulse_channel_gate.
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
QC_PROFILE = os.environ.get('QC_PROFILE', 'prod')
                      # WHICH ARTEFACT MASK. The point of the ladder is to find out how much
                      # the stimulation result depends on the artefact detector at all, and
                      # therefore how accurate that detector actually has to be. Every rung is
                      # a complete end-to-end run: the mask changes, so the AR fill changes,
                      # so the array every detector sees changes, so the preprocessed EDF
                      # Delphos reads changes. That is the real pipeline behaviour and the
                      # thing worth measuring -- a post-hoc re-mask would miss all of it.
                      #
                      # Values are overrides onto make_cfg_artefact's output; `dynFloorMult`
                      # is in multiples of lsb, because an absolute microvolt floor is not
                      # comparable between recordings.
QC_PROFILES = {
    # production: gradThr 400 uV/sample, stimPowerThr 50, floor 3*lsb, dilation 0.50
    'prod':    {},
    # masking OFF entirely -- the control that says what the detector is worth. gradThr=0 and
    # a zero floor disable those two rules by artefact.py's own convention; the stim-power bar
    # is put out of reach rather than removed so the feature is still computed and stored.
    'none':    dict(gradThr=0.0, stimPowerThr=1e12, dynFloorMult=0.0, stimDilationThr=0.0),
    'loose':   dict(gradThr=2000.0, stimPowerThr=200.0, dynFloorMult=0.0, stimDilationThr=0.0),
    'strict':  dict(gradThr=150.0, stimPowerThr=10.0, stimDilationThr=0.25),
    'vstrict': dict(gradThr=50.0, stimPowerThr=2.0, stimDilationThr=0.10),
    # ---- artefact-handling comparison (plans/polymorphic-wiggling-breeze.md) ---------------
    # Condition C: the CURRENT windowed detector on a 2x2 of its two live knobs, with dynR
    # held at 3*lsb. kStim is RELATIVE (K x the paired baseline's per-channel band power);
    # gradThr is ABSOLUTE uV/sample. k450g1000 is finalv2 by another name and is kept as a
    # separate entry so the four cells are read as one grid rather than three plus a special
    # case -- and because finalv2's stored runs are at Janca dec=0, which this is not.
    'k150g150':   dict(kStim=150.0, gradThr=150.0,  dynFloorMult=3.0),
    'k150g1000':  dict(kStim=150.0, gradThr=1000.0, dynFloorMult=3.0),
    'k450g150':   dict(kStim=450.0, gradThr=150.0,  dynFloorMult=3.0),
    'k450g1000':  dict(kStim=450.0, gradThr=1000.0, dynFloorMult=3.0),
    # Condition B: MNE's annotate_amplitude, whole-channel rejection only, no epoch masking.
    # Dispatched by NAME (see _qc_run below) rather than by cfg, because it is a different
    # detector and not a different threshold. The number is `peak` in uV/sample: despite the
    # function name it is a sustained consecutive-sample GRADIENT test, so these are the same
    # two rungs as gradThr above and B reads directly against C.
    # 75 and 150, not 150 and 1000: at 1000 the rule drops nothing at all on the 2 Hz recording
    # (max p99 |diff| there is 2,749 uV/sample against P1_stim's 193,006), so that arm would
    # have been condition A under another name. Both live rungs are now tighter than gradThr's
    # production value, which is the direction the evidence points.
    # gradThr OFF, isolating kStim. The 2x2 above showed kStim barely moves anything while
    # gradThr moves it a lot (k150->k450 at g150: 0.442 -> 0.457; g150->g1000 at k150:
    # 0.442 -> 0.860), so these say whether kStim contributes at all once grad is removed.
    'k150g0': dict(kStim=150.0, gradThr=0.0, dynFloorMult=3.0),
    'k450g0': dict(kStim=450.0, gradThr=0.0, dynFloorMult=3.0),
    # E1: kStim + dynR + PERIODICITY, no gradThr. periodicity_index returns NaN above 10 Hz, so
    # on the 145 Hz file this is exactly k150g0 and should reproduce it -- that equality is a
    # free check that the periodicity gate is really inert at high stim frequency.
    # Threshold 5 from sdc.artefact.periodicity_check: flags 60% of 2 Hz stim channel-epochs
    # against 2.7% of the stim-free baseline recording.
    'e1': dict(kStim=150.0, gradThr=0.0, dynFloorMult=3.0, periodicityThr=5.0),
    # Second periodicity rung. On the 2 Hz file thr 5 flags 53.5% of stim
    # channel-epochs and 2.4% of the stim-free baseline; thr 10 is stricter still.
    # Identical to e1 on any 145 Hz or baseline file, where periodicity is inert.
    'e1t10': dict(kStim=150.0, gradThr=0.0, dynFloorMult=3.0, periodicityThr=10.0),
    # Under the CLINICAL montage the clinician has already removed the worst 62 of 226 contacts,
    # so these drop far fewer than the same thresholds did on the derived set (14 vs 22 at 75).
    # On P1_stim: 10 -> ~35, 15 -> 32, 25 -> 21, 75 -> 14, 150 -> 8 of 164.
    # dynR ONLY: no stim-spectral rule, no gradient, no dilation. The floor you would always
    # want on -- it removes epochs whose dynamic range is under 3*lsb, where a spike is
    # physically impossible, and that is 18-24% of Janca's BASELINE detections. Used as the
    # reference for the pulse-blanking rows so the comparison isolates blanking rather than
    # confounding it with epoch masking, while still not crediting blanking for flat-epoch
    # false positives it did nothing about.
    'dynr': dict(gradThr=0.0, stimPowerThr=1e12, dynFloorMult=3.0, stimDilationThr=0.0),
    # dynR + a LOOSE gradient, no stim-spectral rule. Pairs with pulse blanking: blanking
    # removes the pulses it can reach, and grad=1000 catches the residual too large to blank.
    # gradThr=1000 is inert at 145 Hz (within 0.4% of off) but removes 37% of Janca's
    # detections at 2 Hz -- the low-frequency pulses are big enough to clear it, the
    # high-frequency ones are not. So this profile is a 2 Hz instrument by construction.
    'dynrg1000': dict(gradThr=1000.0, stimPowerThr=1e12, dynFloorMult=3.0, stimDilationThr=0.0),
    'mnebads10':  {},
    'mnebads15':  {},
    'mnebads25':  {},
    'mnebads75':  {},
    'mnebads150': {},
    # ---- stimPowerThr sweep: everything else HELD AT PRODUCTION ----------------------------
    # The five profiles above move three rules at once, and the QC-feature attribution showed
    # that is wasted effort: stim_spec does essentially all the masking (on P5 the "any rule"
    # share equals it almost exactly), lf_artefact is nearly a subset of it below vstrict, and
    # low_dyn never varies because dynFloorMult stays at 3.0 throughout. So the only knob worth
    # a ladder is this one, and moving it alone makes the result attributable.
    #
    # Points chosen from the dumped features to step the masked ON fraction evenly rather than
    # by guessing thresholds (P1 / P5 share of stim-ON channel-epochs):
    #   1000 -> 35% / 15%   the stim rule effectively OFF; isolates gradThr + dropout
    #    200 -> 37% / 21%
    #     50 -> 41% / 35%   = production, already on disk
    #     10 -> 58% / 58%
    #      2 -> 80% / 83%
    'sp1000': dict(stimPowerThr=1000.0),
    'sp200':  dict(stimPowerThr=200.0),
    'sp10':   dict(stimPowerThr=10.0),
    'sp2':    dict(stimPowerThr=2.0),

    # ---- THE FINALISED OPERATING POINT ----------------------------------------------------
    # Chosen by viewing traces on three files -- P1 ANT 145 Hz, P5 ANT 145 Hz, P1 ANT 2 Hz --
    # not read off a curve. None of the three features shows a clean two-population shelf, so
    # there is no data-driven optimum to find; the curves say what a choice COSTS, and the
    # trace says whether it is right.
    #
    #   kStim 450    RELATIVE, x each channel's median band power in its paired pre file.
    #                Nearly useless below ~60 Hz -- at 2 Hz the +-5 Hz band is delta and the
    #                stimulated and pre distributions almost coincide (5.5% flagged vs 26.7%
    #                at 145 Hz) -- so on low-frequency trials grad carries the mask alone.
    #   gradThr 4000 ABSOLUTE uV/sample. Deliberately NOT relative: movement artefact is a
    #                large gradient in absolute terms, and dividing by a high-amplitude
    #                channel's own baseline scales the threshold straight past it.
    #   floor 3*lsb  unchanged. A dropout test against the quantiser is a device-level fact,
    #                and a dead channel is dead whatever its normal amplitude.
    #
    # Against production this is about the same on P1 (39% vs 41% of stim-ON masked), looser on
    # P5 (20% vs 35%), and far looser on the 2 Hz file (10% vs 92%) -- production's absolute
    # gradThr=400 masks almost the whole of a recording whose median gradient is 142 uV/sample.
    'final':  dict(kStim=450.0, gradThr=4000.0, dynFloorMult=3.0),

    # finalv2: 'final' with gradThr LOWERED 4000 -> 1000, after the 2 Hz grad sweep. A SEPARATE
    # NAME rather than a redefinition, because redefining one silently turned the runs on disk
    # into a mixture -- four at 1000 and fourteen at 4000, every one of them stamped
    # qc_profile='final' -- and put the same run on the sweep axis twice, once mislabelled.
    #
    # Janca's stim/baseline ratio at 2 Hz is flat at 0.84-0.89 for grad <= 1500 and climbs to
    # 1.14 by 6000; Barkmeier, artefact-immune here (0.0x enrichment at every rung), sits at
    # 0.89-0.90 throughout. Cost: 189 channels vs 204, 17.5% of ON time masked vs 10.5%.
    # Nearly a no-op at 145 Hz, where grad is inert -- turning it OFF moves masked-ON by 0.1
    # points and it uniquely flags 0.12% of ON channel-epochs.
    #
    # NOT SETTLED: Delphos reads 1.42-2.64 at 2 Hz at EVERY grad rung, so no threshold in this
    # family reconciles it with the other two. See sweeps.py.
    'finalv2': dict(kStim=450.0, gradThr=1000.0, dynFloorMult=3.0),

    # --- SWEEPS AROUND 'final'. One knob moves; everything else stays at 'final' exactly, so a
    # difference between rungs is attributable to that knob. 'final' itself changed TWO things
    # at once against production (pStim absolute-50 -> relative-450, gradThr 400 -> 4000), which
    # is why its result cannot be compared with the old absolute ladder to decide whether the
    # relative scheme or the grad change was responsible.
    #
    # A: pStim on P1's 145 Hz file. Masked-ON at each rung, measured from the stored QC features
    #    before running anything: 42.0 / 40.9 / 38.8(final) / 36.3 / 34.9 %.
    'k225':   dict(kStim=225.0, gradThr=4000.0, dynFloorMult=3.0),
    'k300':   dict(kStim=300.0, gradThr=4000.0, dynFloorMult=3.0),
    'k675':   dict(kStim=675.0, gradThr=4000.0, dynFloorMult=3.0),
    'k1000':  dict(kStim=1000.0, gradThr=4000.0, dynFloorMult=3.0),

    # B: grad on P1's 2 Hz file, where pStim is inert -- the +-5 Hz band at 2 Hz IS delta, so
    #    grad is the only rule doing anything. Masked-ON: 12.0 / 11.4 / 10.5(final) / 8.6 %, with
    #    the knee between 1000 and 2000, which is why the rungs are not symmetric about 4000.
    'g1500':  dict(kStim=450.0, gradThr=1500.0, dynFloorMult=3.0),
    'g2000':  dict(kStim=450.0, gradThr=2000.0, dynFloorMult=3.0),
    'g3000':  dict(kStim=450.0, gradThr=3000.0, dynFloorMult=3.0),
    'g6000':  dict(kStim=450.0, gradThr=6000.0, dynFloorMult=3.0),

    # 300 is chosen to sit just BELOW U13_U14's median stim gradient (320 uV/sample), the worked
    # example of a quiet channel that carries visible 2 Hz artefact through both rules and makes
    # Delphos read 8.8x its baseline rate. 4000 is the pre-finalv2 operating point, kept so the
    # ladder spans the decision rather than sitting to one side of it.
    'g300':   dict(kStim=450.0, gradThr=300.0, dynFloorMult=3.0),
    'g4000':  dict(kStim=450.0, gradThr=4000.0, dynFloorMult=3.0),

    # The LOW end of grad on the 2 Hz file. 'final' at 4000 sits between p90 (2566) and p95
    # (15447) of the stim-ON gradRatio distribution, so it catches only the top ~8% -- and at
    # 2 Hz grad is the ONLY rule doing anything, because pStim measures +-5 Hz around the
    # fundamental while the artefact is a harmonic comb. Unlike the HF pStim sweep, this region
    # of the distribution is DENSE: 4000 -> 1000 moves masked-ON 10.5% -> 17.5% and costs 14
    # channels, so these rungs buy masking at a real price and the cost has to be read alongside.
    'g500':   dict(kStim=450.0, gradThr=500.0, dynFloorMult=3.0),
    'g750':   dict(kStim=450.0, gradThr=750.0, dynFloorMult=3.0),
    'g1000':  dict(kStim=450.0, gradThr=1000.0, dynFloorMult=3.0),
    'g1250':  dict(kStim=450.0, gradThr=1250.0, dynFloorMult=3.0),
}
if QC_PROFILE not in QC_PROFILES:
    raise SystemExit(f"QC_PROFILE={QC_PROFILE!r}; expected one of {sorted(QC_PROFILES)}")

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
BASELINE_SEC = float(os.environ.get('BASELINE_SEC', 300))
                      # How much of the PRE file sets the per-channel relative stim threshold,
                      # taken from its END. 300 s is a comparability choice, not a precision one:
                      # the threshold is stable to ~15% at any window from 60 s up, and a 15%
                      # threshold error moves under 1% of channel-epochs, well inside the plateau
                      # a 4.4x sweep of K traced. Capping matters because the cohort's baselines
                      # span 43-4162 s, so an uncapped rule gives trials thresholds built on
                      # wildly different amounts of data AND different amounts of drift.
MASK_ARTEFACTS = True # drop detections inside the dilated artefact mask (all detectors alike)
REJECTED = {}         # {detector: [per-channel indices the mask removed]}, filled by _finalise.
                      # Module-level rather than returned so the three call sites and every
                      # downstream reader of _finalise keep their existing signatures.
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
ONLY = os.environ.get("ONLY", "").lower()
                      # "janca" | "barkmeier" | "delphos" -> run ONLY that arm. A parameter
                      # sweep moves one detector's knob, so running the other two at every grid
                      # point is pure waste -- and for Barkmeier each run pays a fresh MATLAB
                      # engine start. 25 subjects x 6 points x 2 idle detectors is hours.
BARK_SCALE = float(os.environ.get("BARK_SCALE", 0) or 0) or None
                      # Barkmeier's block target amplitude. None keeps seeg.spikes.SCALE (70).
                      # WHY THIS IS SETTABLE: `scale = SCALE / median(mean|EEG|)` is computed per
                      # block per RECORDING, and TAMP is absolute and applied AFTER scaling -- so
                      # a condition that raises the block amplitude silently RAISES the effective
                      # threshold. Measured on P1 ANT 2 Hz: the denominator is 17.64 uV at
                      # baseline and 25.06 uV during stimulation, moving the effective TAMP from
                      # 302 to 430 uV, so Barkmeier is 42% more conservative during stim for
                      # reasons that have nothing to do with spiking. At 145 Hz epoch masking
                      # restores the denominator (18.96 vs 18.88) and the effect vanishes.
                      # Passing SCALE * (denom_stim / denom_base) for the stim file reproduces
                      # the baseline's scale factor and makes the threshold condition-invariant.
                      # This is a deliberate deviation from Barkmeier et al., who normalise per
                      # block for single-recording detection and never compare conditions.
                      #
                      # SUPERSEDED BY BARK_DENOM BELOW -- prefer that. This knob back-solves one
                      # constant SCALE from a PYTHON RECONSTRUCTION of the denominator, and it
                      # measurably overshoots: on P1 ANT 2 Hz it took Barkmeier's gated ratio
                      # 0.516 -> 0.977, past Janca (0.742) and Delphos (0.771), i.e. it swapped
                      # the sign of the error instead of removing it. Two reasons, both fixed by
                      # BARK_DENOM: the reconstruction does not reproduce mDetectSpike's
                      # artifactChans handling, and one constant cannot undo a normaliser that
                      # is recomputed every block -- it corrects a second time what the
                      # per-block renormalisation already partly corrected.
BARK_DENOM = os.environ.get("BARK_DENOM", "") or None
_BD_SUF = ("" if not BARK_DENOM else
           "_bdauto" if BARK_DENOM == "auto" else f"_bd{float(BARK_DENOM):g}")
                      # Pin Barkmeier's per-block amplitude normaliser, so a stim recording and
                      # its baseline are detected at the SAME absolute threshold. A float, or
                      # 'auto' to take it from the paired baseline run's stored block
                      # denominators. This is the RIGHT version of BARK_SCALE: the value comes
                      # from mDetectSpike itself (4th output, added for this) rather than from a
                      # reconstruction, and it replaces the per-block normaliser outright rather
                      # than trying to cancel it with a constant.
                      #
                      # BOTH RECORDINGS MUST GET THE SAME VALUE, baseline included. Pinning only
                      # the stim file leaves the baseline adapting per block and the stim file
                      # not, which is a different comparison, not a corrected one.
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
# dec=200 is the PAPER's resampling ("to maintain filter characteristics constant") and is now
# the repo-wide Janca operating point -- see sdc.scoring.tune_marks.JANCA_FIXED, which every
# tuning and sweep module imports. It was dec=0 here, so this script alone ran a Janca nothing
# else did. Measured against the marked blocks, dec=200 costs 0.022 marked macro F1 but is 5x
# faster and gives visibly tighter across-patient agreement (paired SE roughly halves), which is
# what matters for a pipeline that must generalise to unmarked patients.
# fh=50, not the paper's 60: 50 Hz mains sits inside a 60 Hz upper edge on this hardware.
JANCA = dict(dec=200.0, fl=10.0, fh=50.0)
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
# {detector, param, value} -- move ONE knob off its default for this run. Not sim-specific
# despite the old name: it mutates the module-level detector dicts before the runners, so it
# works in every mode. DET_OVERRIDE is the name to use; SIM_OVERRIDE stays as an alias because
# run_sim_suite.py drives it.
SIM_OVERRIDE = json.loads(os.environ.get("DET_OVERRIDE",
                                         os.environ.get("SIM_OVERRIDE", "")) or "{}")
# DET_TUNE sets SEVERAL detectors at once: {"janca":{"k1":4.48}, "barkmeier":{"TAMP":890}, ...}.
# DET_OVERRIDE moves one knob for a sweep point; this applies a whole operating point, which is
# what the tuned-vs-default comparison needs.
DET_TUNE = json.loads(os.environ.get("DET_TUNE", "") or "{}")
RUN_TAG = os.environ.get("RUN_TAG", "")   # appended to the npz name. Without it a tuned run
                                          # overwrites the default run it is meant to be
                                          # compared against -- which has happened three times
                                          # in this project, hence the explicit knob.
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
    if SIM_OVERRIDE:
        # A sweep point is NOT the operating-point run and must never land on its filename.
        _sw = _ROOT / "runs" / "sweeps"
        _sw.mkdir(parents=True, exist_ok=True)
        DETECTIONS_NPZ = _sw / (f"bids_{BIDS_SUBJECT}_{SIM_OVERRIDE['detector']}"
                                f"_{SIM_OVERRIDE['param']}{SIM_OVERRIDE['value']:g}.npz")
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
    # ONE resolver, shared with run_windows.plan() and qc_features.dump(). This block used to
    # run the trial_index chain itself, which is only meaningful against the JSON the index came
    # from: the cohort is defined against AES_trials.json, so resolving it through the local
    # stim_trials.json gave two trials the WRONG recording outright and two baselines a bare
    # '.edf'. Four copies of this chain existed and each had to be found separately.
    _cfg = RECORDINGS[RECORDING]
    _path, TRIAL = edf_path(RECORDING)
    _stem = _path.stem
    EDF = str(_path)
    REC_META = dict(rec_id=RECORDING, patient=_cfg["patient"], condition=_cfg["file_type"],
                    stim_hz=float(TRIAL["stim_frequency"]) if TRIAL else float("nan"))
    # The filename carries any DEVIATION from the canonical config, and nothing else. A run at
    # the canonical settings is plain `<rec_id>.npz` -- so downstream defaults keep working --
    # while a variant gets a suffix and CANNOT overwrite it.
    # This is not hypothetical tidiness: a MED_KERNEL=1 control run silently replaced the
    # canonical MED_KERNEL=5 result minutes after it was produced, and only a hand copy saved
    # it. Same failure as the TAMP=400 incident noted above -- a differently-configured run
    # sitting in the file where the canonical one is expected.
    _variant = RUN_TAG + "".join([
        "" if MED_KERNEL == 5 else f"_med{MED_KERNEL}",
        "" if DETECT_FS == 1000.0 else f"_{DETECT_FS:g}Hz",
        "" if PREP_DELPHOS else "_rawdelphos",
        "" if FILL_ALL else "_nofill",
        "" if QC_NATIVE else "_qcdec",
        "_fillbad" if FILL_BAD_SAMPLES else "",
        "" if MERGE_MS == 100.0 else f"_merge{MERGE_MS:g}",
        # BEFORE the qc suffix: run_windows._VAR_SUF appends _QC_SUF last, and its
        # comment requires this list to mirror that order exactly. Getting it wrong
        # writes windows as _qcX_pbY while merge_windows looks for _pbY_qcX.
        _PB_SUF,
        # A PINNED Barkmeier normaliser is a different detector configuration and must not
        # share a filename with an unpinned one. It is added automatically rather than left to
        # RUN_TAG, because BARK_SCALE was left to RUN_TAG and that is exactly how a run gets
        # written into the canonical file by someone who forgot the tag.
        _BD_SUF,
        "" if QC_PROFILE == "prod" else f"_qc{QC_PROFILE}",
    ])
    DETECTIONS_NPZ = _ROOT / "runs" / f"{RECORDING}{_variant}.npz"
    if _variant:
        print(f"[config] non-canonical run -> {DETECTIONS_NPZ.name}")
    print(f"--- {RECORDING}: P{_cfg['patient']} "
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
    # CLINICAL montage where one exists, and a hard failure where it does not. This line called
    # derive_montage unconditionally for the whole life of the project, so every run used 226
    # consecutively-paired contacts instead of the clinician's 164 -- silently, because a
    # derived montage always succeeds. See recordings.load_patient_montage.
    _mont = load_patient_montage(_cfg["patient"],
                                 allow_derived=os.environ.get("MONTAGE", "") == "derived")
    if _mont is None:
        _mont = derive_montage(rec["info"]["SelectedSignals"])
        print(f"[montage] DERIVED: {len(_mont)} pairs")
    else:
        print(f"[montage] clinical {montage_path(_cfg['patient']).name}: {len(_mont)} pairs")
    rec = apply_montage(rec, _mont)

# Stim detection MUST come after the montage (stim_channels are post-montage PAIR names) and
# before the QC. make_cfg_artefact(lsb, trial) on its own does NOTHING: the stim-spectral rule
# is gated on `is_on[e] AND have_stim`, and is_on comes from info["stim_bins"], which only
# detect_stim writes. Skip this and the stim rule silently never fires.
if TRIAL is not None:
    rec["info"]["stim_trial"] = TRIAL
    rec = detect_stim(rec)

PULSE_INFO = {}
_RAW_FOR_QC = None
                      # The UNBLANKED native array, kept so windowed_artefact_detector scores
                      # the signal as it really was. QC MUST NOT SEE THE BLANKED SIGNAL: its
                      # lf_artefact rule fires on max|diff|, blanking removes exactly that
                      # gradient, and the 50-105 ms decay a 5 ms blank cannot reach is left
                      # behind -- so blanking first does not clean a contaminated channel, it
                      # stops QC noticing one. Measured on P1 ANT 2 Hz: blanking before QC
                      # admitted 21 channels QC had excluded outright, and Delphos then fired
                      # 44.8/chan-min on them against 8.7 on the rest.
if PULSE_BLANK_MS and TRIAL is not None:
    from seeg import detect_pulses, blank_pulses, pulse_channel_gate
    from seeg.stim import get_stim_channel, _stim_column
    _sch = get_stim_channel(rec, verbose=False)
    _pm, _pi = detect_pulses(_stim_column(rec, _sch), rec["info"]["SampleRate"],
                             stim_hz=float(TRIAL["stim_frequency"]))
    _elig, _tbl = pulse_channel_gate(rec, _pm, max_peak_uv=PULSE_MAX_PEAK_UV,
                                     max_rail_frac=PULSE_MAX_RAIL_FRAC)
    # blank_pulses returns a NEW array, so this reference keeps the pre-blank signal alive
    # for QC at no extra memory or compute.
    _RAW_FOR_QC = np.asarray(rec["data"], float)
    rec, _bi = blank_pulses(rec, _pm, method=PULSE_FILL, blank_ms=PULSE_BLANK_MS,
                            channels=_elig)
    PULSE_INFO = {"stim_channel": _sch, **{k: _pi[k] for k in
                  ("n_pulses", "threshold", "median_isi_ms", "median_width_ms",
                   "frac_isi_on_period")},
                  "n_eligible": int(_elig.sum()), "n_gated_out": int((~_elig).sum()),
                  **{k: v for k, v in _bi.items() if k != "method"}}
    print(f"[pulse] {_sch}: {_pi['n_pulses']} pulses, ISI {_pi['median_isi_ms']:.1f} ms "
          f"(expected {1000 / float(TRIAL['stim_frequency']):.1f}), "
          f"on-period {_pi['frac_isi_on_period']:.3f}")
    print(f"[pulse] gate {PULSE_MAX_PEAK_UV:g} uV / {PULSE_MAX_RAIL_FRAC:.0%} railed -> "
          f"{_elig.sum()}/{_elig.size} channels blanked at {PULSE_BLANK_MS:g} ms "
          f"({_bi['blanked_frac_of_blanked_channels']:.2%} of each); "
          f"gated out: {', '.join(t['ch'] for t in _tbl if not t['eligible']) or 'none'}")
    print("[pulse] QC will score the UNBLANKED signal; blanking only cleans what QC passes")
elif PULSE_BLANK_MS:
    print("[pulse] PULSE_BLANK_MS set but this is not a stim trial -- skipping")

dec = decimate_recording(rec, factor=FACTOR, med_kernel=MED_KERNEL)
fs = dec["info"]["SampleRate"]
names = dec["info"]["SelectedSignals"]
n_chan = len(names)

# QC at the file's own rate (see QC_NATIVE). decimate_recording already kept the native array
# in dec["raw"], so this costs nothing extra -- and it keeps stim_bins and the QC epochs in the
# SAME sample space, which is the difference between isOn being right and being wrong by FACTOR.
if QC_NATIVE:
    # dec["raw"] is whatever was decimated, which under pulse blanking is the BLANKED array --
    # see _RAW_FOR_QC above for why QC must not be given that.
    _qc_src = _RAW_FOR_QC if _RAW_FOR_QC is not None else dec["raw"]["data"]
    _qc_rec = {"data": _qc_src,
               "info": {**rec["info"], "SampleRate": dec["raw"]["fs"],
                        "NSamples": _qc_src.shape[0]}}
else:
    _qc_rec = dec
_qc_fs = _qc_rec["info"]["SampleRate"]
QC_CFG = make_cfg_artefact(hdr["lsb"], trial=TRIAL)
if QC_PROFILES[QC_PROFILE]:
    _ov = dict(QC_PROFILES[QC_PROFILE])
    if "dynFloorMult" in _ov:                       # multiples of lsb -> absolute uV
        _ov["minDynRangeFloor"] = _ov.pop("dynFloorMult") * hdr["lsb"]
    # A RELATIVE stim-power threshold: K times each channel's own median band power in the
    # paired stim-free pre recording. artefact.py compares `ps > cfg["stimPowerThr"]` where
    # `ps` is (n_chan,), so handing it a (n_chan,) ARRAY is a per-channel threshold and needs
    # no change to the pipeline at all.
    #
    # Why relative: an absolute bar cannot transfer between patients, amplifiers or implant
    # depths, and this cohort spans 8 patients. Why the MEDIAN and not a high percentile: a
    # percentile makes the normaliser hostage to a few contaminated baseline epochs -- P1's H
    # shaft has a p95 five orders of magnitude above its median, and under a p95 baseline those
    # channels were never flagged at any K despite being obviously contaminated.
    if "kStim" in _ov and TRIAL is None:
        # A baseline recording. `make_cfg_artefact(trial=None)` leaves stimHz None, which
        # disables the stim-spectral rule outright -- so there is no threshold for a relative
        # one to scale, and no frequency to measure a baseline AT. Dropping the knob is the
        # whole correct behaviour here. Without this the LF-continuous pre files (which must be
        # run, because a continuous trial has no OFF period and the pre file IS the comparison)
        # looked for `<rec>_f0.npz` and aborted.
        _ov.pop("kStim")
        print("[qc] baseline recording: no stimulation, so the relative stim threshold is not "
              "applied (the stim-spectral rule is off).")
    if "kStim" in _ov:
        _k = _ov.pop("kStim")
        _pre_rec = RECORDING.replace("_stim", "_pre")
        _fhz = float(TRIAL["stim_frequency"])
        _fp = _ROOT / "qc_features" / f"{_pre_rec}_f{_fhz:g}.npz"
        if not _fp.is_file():
            raise SystemExit(
                f"{_fp.name} not found. A relative stim threshold needs the paired pre "
                f"recording's band power AT THIS TRIAL'S FREQUENCY ({_fhz:g} Hz) -- a baseline "
                f"measured at another frequency is a different quantity. Dump it first:\n"
                f"    QC_STIM_HZ={_fhz:g} python -c \"from sdc.artefact.qc_features import "
                f"dump; dump('{_pre_rec}', freq={_fhz:g})\"")
        _zb = np.load(_fp, allow_pickle=False)
        if [str(s) for s in _zb["names"]] != list(names):
            raise SystemExit(f"{_fp.name} channel names differ from this run; a per-channel "
                             f"threshold cannot be aligned.")
        # CHECK THE CONTENTS, NOT THE NAME. A dump named `_f145` held 2 Hz band power for 11 of
        # 17 baselines (QC_STIM_HZ was pinned by the first dump in the process), and nothing
        # downstream could tell -- the threshold was simply wrong and the runs looked fine.
        if "stim_hz_used" not in _zb.files:
            raise SystemExit(
                f"{_fp.name} predates the stim_hz_used field, so the frequency it was measured "
                f"at cannot be confirmed and its filename is not evidence. Re-dump it:\n"
                f"    python -c \"from sdc.artefact.qc_features import dump; "
                f"dump('{_pre_rec}', freq={_fhz:g})\"")
        _mhz = float(_zb["stim_hz_used"])
        if abs(_mhz - _fhz) > 1e-6:
            raise SystemExit(
                f"{_fp.name} was measured at {_mhz:g} Hz but this trial stimulates at {_fhz:g} Hz. "
                f"Band power at another frequency is a different quantity and cannot threshold "
                f"this recording. Re-dump the baseline at {_fhz:g} Hz.")
        # THE LAST `BASELINE_SEC` OF THE PRE FILE, not the whole thing. Band power drifts, and
        # the baseline always precedes the stim recording, so its final window is the closest in
        # time to what this threshold has to predict. Measured against the stim file's own
        # stim-OFF band power, the last 300 s more than halves the mismatch on P5 (median
        # |log2| 0.202 whole-file -> 0.092) and changes nothing on P1, whose baseline is only
        # 650 s so the two windows nearly coincide. More baseline is NOT better: two disjoint
        # 600 s windows of the same file disagree more than two 60 s ones, because the limit is
        # non-stationarity rather than sample size.
        _pa = _zb["pStimAll"]
        _t = np.asarray(_zb["t"], float) if "t" in _zb.files else None
        _ep = float(np.median(np.diff(_t))) if _t is not None and _t.size > 1 else 2.0
        _nk = max(int(round(BASELINE_SEC / max(_ep, 1e-6))), 1)
        _used = _pa[-_nk:] if _pa.shape[0] > _nk else _pa
        _ov["stimPowerThr"] = _k * np.maximum(np.median(_used, axis=0), 1e-12)
        print(f"[qc] relative stim threshold: K={_k:g} x per-channel baseline from "
              f"{_fp.name}, last {_used.shape[0] * _ep:.0f}s of {_pa.shape[0] * _ep:.0f}s "
              f"(median {np.median(_ov['stimPowerThr']):.3g})")
    QC_CFG.update(_ov)
    print(f"[qc] profile {QC_PROFILE!r}: " +
          ", ".join(f"{k}={np.median(QC_CFG[k]):g}" if np.ndim(QC_CFG[k]) else
                    f"{k}={QC_CFG[k]:g}" for k in sorted(_ov)))
def _qc_run(rec, cfg, profile):
    """Dispatch on the profile NAME: some conditions are a different detector, not a threshold.

    `mnebadsN` -> MNE's annotate_amplitude at peak=N uV/sample, whole-channel rejection only.

    The list is READ, not recomputed. A bad channel is a property of the recording, not of a
    60 s window, and `annotate_amplitude`'s `bad_percent` is evaluated against whatever span it
    is handed -- so computing it per window would let a channel come and go, which is epoch
    masking at window granularity wearing a bad-channel label. `sdc.artefact.mne_bads_check.dump`
    writes the whole-recording list once; this reads it by NAME.

    min_duration in that dump is 2 ms, NOT MNE's 5 ms default: at 5 ms a channel swinging
    +-150 mV is not flagged at any threshold above ~400 uV/sample, because pulsatile artefact has
    huge gradients at pulse edges and small ones between and so never stays above threshold for
    10 consecutive samples. Measured on P1_stim -- see seeg.artefact_mne.mne_bad_channels.
    """
    if not profile.startswith("mnebads"):
        return windowed_artefact_detector(rec, cfg)

    import json
    from seeg.artefact_mne import _empty_qc, _finalise

    peak = float(profile[len("mnebads"):])
    fp = _ROOT / "runs" / "mne_bads" / f"{RECORDING}_p{peak:g}.json"
    if not fp.is_file():
        raise SystemExit(
            f"{fp.name} not found. Condition {profile!r} needs the WHOLE-RECORDING bad-channel "
            f"list, which must be computed once before windowing. Dump it first:\n"
            f"    python -c \"from sdc.artefact.mne_bads_check import dump; "
            f"dump(('{RECORDING}',))\"")
    meta = json.loads(fp.read_text(encoding="utf-8"))
    names_here = [str(n) for n in rec["info"]["SelectedSignals"]]
    bads = [b for b in meta["bads"]]
    missing = [b for b in bads if b not in names_here]
    if missing:
        raise SystemExit(f"{fp.name} names {missing[:5]} which this run does not have; the "
                         f"montage differs between the dump and this run.")
    idx = [names_here.index(b) for b in bads]
    qc = _empty_qc(rec, cfg)
    if idx:
        qc["epoch"]["bad"][:, idx] = True
        for c in idx:
            qc["epoch"]["reason"][:, c] = "mne_bad_channel"
    qc["features"]["mne_bads"] = np.array(bads, dtype=object)
    qc["features"]["mne_bad_idx"] = np.asarray(idx, int)
    print(f"[qc] mne_bads peak={peak:g} uV/sample (min_duration={meta['min_duration']*1000:g} ms)"
          f": {len(idx)}/{len(names_here)} channels dropped, from {fp.name}")
    return _finalise(qc, rec, cfg)


qc = _qc_run(_qc_rec, QC_CFG, QC_PROFILE)
# ---- QC-ONLY: dump the features and stop, without running a single detector -------------
# The three rules are thresholded inside windowed_artefact_detector and only their VERDICT
# survives, so "which rule did the masking" cannot be answered from a finished run. The
# features themselves are threshold-independent, so dumping them once makes the composition of
# the mask at ANY threshold pure arithmetic -- no detector, no MATLAB, no Delphos, minutes
# instead of hours. QC_FEATURES=<path> writes and exits.
_QC_FEATURES = os.environ.get("QC_FEATURES", "")
if _QC_FEATURES:
    if QC_PROFILE.startswith("mnebads"):
        raise SystemExit(
            f"QC_FEATURES dumps dynR/pStim/gradRatio, which only the windowed detector "
            f"computes -- {QC_PROFILE!r} is MNE's annotate_amplitude and produces a channel "
            f"list instead. Dump the features under a windowed profile (they are "
            f"threshold-independent, so one dump serves every rung), or inspect this "
            f"condition with sdc.artefact.mne_bads_check.")
    _f = qc["features"]
    # pStim as stored is nan outside stim-ON epochs, because the rule is only evaluated there.
    # That makes a BASELINE impossible -- and a baseline is exactly what a relative threshold
    # needs. bandpower_stim is public, so recompute it on EVERY epoch here. On a stim-free
    # recording (TRIAL is None, stimHz unset) QC_STIM_HZ supplies the frequency to measure at,
    # which is how the pre files provide the null distribution for the stim band.
    from seeg.artefact import bandpower_stim as _bps
    _sh = float(os.environ.get("QC_STIM_HZ", 0) or 0) or float(QC_CFG.get("stimHz") or 0)
    _es = int(qc["epoch"]["epochSamp"])
    _xq = _qc_rec["data"]
    if _sh > 0:
        _p_all = np.array([_bps(_xq[s:s + _es, :], _qc_fs, _sh, QC_CFG["stimBW_Hz"])
                           for s in qc["epoch"]["starts"]], dtype=np.float32)
    else:
        _p_all = np.full_like(_f["gradRatio"], np.nan, dtype=np.float32)
    Path(_QC_FEATURES).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _QC_FEATURES,
        dynR=_f["dynR"].astype(np.float32),
        pStim=_f["pStim"].astype(np.float32),
        gradRatio=_f["gradRatio"].astype(np.float32),
        pStimAll=_p_all, stim_hz_used=np.float64(_sh),
        starts=np.asarray(qc["epoch"]["starts"], np.int64),
        isOn=np.asarray(qc["epoch"]["isOn"], bool),
        epoch_samp=np.int64(qc["epoch"]["epochSamp"]),
        qc_fs=np.float64(_qc_fs), lsb=np.float64(hdr["lsb"]),
        names=np.array(list(names)), rec_id=str(RECORDING),
        start_rec=np.int64(START_REC), stop_rec=np.int64(STOP_REC))
    print(f"[qc-only] wrote {Path(_QC_FEATURES).name}: "
          f"{_f['dynR'].shape[0]} epochs x {_f['dynR'].shape[1]} channels")
    raise SystemExit(0)

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

# ---- Barkmeier's normaliser, recorded for every run -----------------------------------
# mDetectSpike divides SCALE by median(mean|EEG|) per block and applies TAMP/LS/RS AFTER that,
# so this number IS Barkmeier's effective threshold up to a constant. It is printed on every
# run because it is the one detector parameter that the RECORDING changes underneath us: two
# conditions can differ in Barkmeier rate purely because masking moved this, with no threshold
# having been touched. Printing it makes that visible instead of inferable.
#
# Computed on the array the detector is actually handed -- post median-5, post decimation, POST
# AR-FILL -- and with NO sample mask, because mDetectSpike never receives one (seeg.spikes
# line ~327 fills the masked samples instead). So epoch masking reaches the normaliser only
# through the fill, which is exactly the mechanism this print exposes.
try:
    from seeg.spikes import scale_denominator as _scale_denom
    _denom = _scale_denom(dec)
    print(f"[bark] scale denominator median(mean|EEG|) = {_denom:.2f} uV  "
          f"-> scale {70.0 / _denom:.3f}, effective TAMP {1200.0 * _denom / 70.0:.0f} uV")
except Exception as _e:                                  # never let a diagnostic kill a run
    _denom = float("nan")
    print(f"[bark] scale denominator unavailable: {_e}")

if os.environ.get("SCALE_ONLY") == "1":
    # Measure the normaliser across QC profiles without paying for any detector.
    print(f"[scale-only] {RECORDING} {QC_PROFILE} denom={_denom:.4f}")
    raise SystemExit(0)

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
                   f"{'' if QC_PROFILE == 'prod' else '_qc' + QC_PROFILE}"
                   # blanking rewrites samples without changing their count, so a blanked file
                   # is the same SIZE as an unblanked one -- and Delphos's cache keys on
                   # path + size, so an unsuffixed name returns the unblanked result silently
                   f"{_PB_SUF}"
                   f"_{START_REC}-{STOP_REC}.edf")
    _reuse = Path(PREP_EDF).is_file()
    if _reuse:
        # The cache key is the FILENAME, which encodes the profile and variant but NOT the
        # montage -- so a prep EDF written under a different montage is served silently, and
        # Delphos then runs on a different channel set from Janca and Barkmeier while the npz
        # records one list of names. Found for real: P1_stim_..._qcnone.edf held 226 derived
        # pairs while this run used the clinical 164. Check the count, do not trust the name.
        try:
            _n_have = int(read_edf_header(PREP_EDF)["NumSignals"])
        except Exception as _e:                       # unreadable -> rewrite rather than guess
            print(f"[prep] {Path(PREP_EDF).name} unreadable ({_e}); rewriting")
            _n_have, _reuse = -1, False
        if _reuse and _n_have != len(names):
            print(f"[prep] {Path(PREP_EDF).name} has {_n_have} channels but this run has "
                  f"{len(names)} -- montage changed; rewriting")
            _reuse = False
    if _reuse:
        print(f"[prep] reusing {Path(PREP_EDF).name} ({len(names)} channels)")
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


# The rate the UNION actually runs at, which is the decimated one: _spike_detector calls
# _decimate and REASSIGNS fs before `_detection_union(..., pt * fs)`. Passing DETECT_FS here
# while Janca decimates to 200 Hz would compute pt for a 5x higher rate, shrinking the internal
# union to a fifth of the intended floor -- silently, since the result is still a valid merge.
JANCA_PT = _janca_pt(MERGE_MS, float(JANCA["dec"]) or fs)


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
    out, rej = [], []
    for c in range(n_chan):
        idx = np.unique(np.asarray(per_chan[c], int))
        idx = idx[(idx >= 0) & (idx < dmask.shape[0])]
        n_raw += idx.size
        if MASK_ARTEFACTS:
            # Keep what the mask threw away. Storing only the survivors makes the stored run a
            # one-way door: a STRICTER mask can be applied afterwards by dropping more, but a
            # LOOSER one cannot, because the detections it would readmit are already gone. That
            # asymmetry is what forced a full re-run per rung of the threshold ladder. These
            # go to the npz under `{Det}_idx_masked` and are NOT part of any result -- the
            # kept arrays keep their exact previous meaning.
            rej.append(merge_close(idx[dmask[idx, c]], gap))
            idx = idx[~dmask[idx, c]]
        else:
            rej.append(np.zeros(0, int))
        n_masked += idx.size
        idx = merge_close(idx, gap)
        n_merged += idx.size
        out.append(idx)
    if label:
        REJECTED[label] = rej
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


def _resolve_bark_denom():
    """BARK_DENOM -> a float, or None. 'auto' reads the PAIRED BASELINE run for this profile.

    Auto deliberately fails loudly rather than falling back to per-block normalisation: a run
    that silently reverted would be indistinguishable in the npz from one that was pinned, and
    it would sit in a figure column captioned as corrected.
    """
    if not BARK_DENOM:
        return None
    if BARK_DENOM != "auto":
        return float(BARK_DENOM)
    base_rec = RECORDING.replace("_stim", "_pre")
    f = _ROOT / "runs" / f"{base_rec}_qc{QC_PROFILE}.npz"
    if not f.is_file():
        raise SystemExit(
            f"BARK_DENOM=auto needs the paired baseline run {f.name}, which does not exist.\n"
            f"  Run the baseline FIRST (unpinned), then the stim file with BARK_DENOM=auto.")
    with np.load(f, allow_pickle=False) as _z:
        if "bark_block_denom" not in _z.files or not _z["bark_block_denom"].size:
            raise SystemExit(
                f"{f.name} predates the block-denominator output and does not carry one.\n"
                f"  Re-run the baseline once (no BARK_DENOM) to record it, then retry.")
        d = float(np.median(_z["bark_block_denom"]))
    print(f"[bark] BARK_DENOM=auto -> {d:.3f} uV, median block denominator of {f.name}")
    return d


BARK_DENOM_VALUE = _resolve_bark_denom()
BARK_OUT = {}          # receives {"block_denom": ...} from the detector, stored in the npz


def run_bark(recording, label=None, **override):
    """Barkmeier -> per-channel sample indices. Override LS/RS/TAMP/LD/RD/std_coeff/... by keyword."""
    p = {**BARK, **override}
    detect_barkmeier(recording, QC_DET, post_mask_spikes=False, out=BARK_OUT,
                     fixed_denom=BARK_DENOM_VALUE,
                     # FILL_BAD_SAMPLES: seeg defaults this True, which AR-fills masked regions
                     # for Barkmeier only. Its block normalisers are computed over the whole
                     # block INCLUDING the fill, so it is not neutralised by the later mask.
                     fill_bad_samples=FILL_BAD_SAMPLES,
                     det_thresholds=[p["LS"], p["RS"], p["TAMP"], p["LD"], p["RD"]],
                     std_coeff=p["std_coeff"], trough_search_ms=p["trough_search_ms"],
                     filter_spec=p["filter_spec"], scale=BARK_SCALE)
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
        # The detections the artefact mask removed, same (index, channel) flattening. Present
        # so a LOOSER mask can be evaluated after the fact; see the note in _finalise.
        r = REJECTED.get(name)
        dump[f"{name}_idx_masked"] = (np.concatenate(r) if r and any(x.size for x in r)
                                      else np.zeros(0, int))
        dump[f"{name}_chan_masked"] = (np.concatenate([np.full(x.size, c, int)
                                                       for c, x in enumerate(r)])
                                       if r and any(x.size for x in r) else np.zeros(0, int))
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
                 "qc_profile": str(QC_PROFILE),
                 "fill_all": np.int64(bool(FILL_ALL)),
                 "baseline_sec": float(BASELINE_SEC),
                 # Barkmeier's per-block amplitude normaliser, AS MDETECTSPIKE COMPUTED IT, plus
                 # the value it was pinned to (nan = not pinned, i.e. published behaviour). Both
                 # stored because they are what makes two runs' Barkmeier rates comparable or
                 # not, and because BARK_DENOM=auto reads the first of these back off the paired
                 # baseline run. A run with neither key predates the 4th MATLAB output.
                 "bark_block_denom": np.asarray(BARK_OUT.get("block_denom", []), float),
                 "bark_fixed_denom": float(BARK_DENOM_VALUE if BARK_DENOM_VALUE
                                           else float("nan")),
                 "bark_scale": float(BARK_SCALE if BARK_SCALE else float("nan")),
                 "pulse_blank_ms": float(PULSE_BLANK_MS),
                 "pulse_fill": str(PULSE_FILL),
                 "pulse_max_peak_uv": float(PULSE_MAX_PEAK_UV),
                 "pulse_n": np.int64(PULSE_INFO.get("n_pulses", 0)),
                 "pulse_thr": float(PULSE_INFO.get("threshold", float("nan"))),
                 "pulse_isi_ms": float(PULSE_INFO.get("median_isi_ms", float("nan"))),
                 "pulse_n_gated_out": np.int64(PULSE_INFO.get("n_gated_out", 0)),
                 "pulse_blank_frac": float(PULSE_INFO.get("blanked_frac", 0.0)),
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
        midx = dump.get(f"{name}_idx_masked", np.zeros(0, int))
        dump[f"{name}_on_masked"] = ON_MASK[midx] if midx.size else np.zeros(0, bool)
    dump.update(extra or {})
    # Structural checks BEFORE the write: a wrong number here becomes a result, and the
    # cross-detector disagreement this project leans on cannot see errors in shared code.
    check_run(dump, n_samp=dmask.shape[0], fs=fs)
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
if DET_TUNE:
    # A whole operating point, applied before the runners so the sweep machinery and the
    # detectors agree. Printed in full: a run at non-default settings that does not say so in
    # its own log is the thing this project keeps being bitten by.
    _t = {"janca": JANCA, "barkmeier": BARK, "delphos": DELPHOS}
    for _d, _kv in DET_TUNE.items():
        if _d.lower() not in _t:
            raise SystemExit(f"DET_TUNE detector {_d!r} unknown; use one of {sorted(_t)}")
        for _p, _v in _kv.items():
            _t[_d.lower()][_p] = _v
        print(f"[tune] {_d}: " + ", ".join(f"{k}={v:g}" for k, v in _kv.items()))

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
# ONLY=<detector> runs one arm. A dropped arm is ABSENT from `dets`, never zero-filled: an
# empty panel reads as "found nothing", which is a different claim from "not run".
dets = []
janca = [np.zeros(0, int) for _ in range(n_chan)]
if ONLY in ("", "janca"):
    janca = run_janca(dec["data"], label="Janca")
    dets.append(("Janca", janca, RED))

bark = [np.zeros(0, int) for _ in range(n_chan)]
if ONLY in ("", "barkmeier"):
    try:
        bark = run_bark(dec, label="Barkmeier")
        dets.append(("Barkmeier", bark, BLUE))
    except Exception as e:        # no MATLAB / engine failure
        # SKIPPING IS ONLY SAFE ON A MACHINE WITH NO MATLAB AT ALL. A TRANSIENT failure -- the
        # licence server being briefly unreachable ("Unable to launch MVM server", error 5001)
        # -- is different in kind: the batch keeps going, this window is written and CACHED with
        # Janca only, and every later re-run reuses it. The merged file then reports a Barkmeier
        # rate missing one whole window, which on a 5-window recording is a ~20% undercount that
        # looks entirely plausible. That happened, and only a hand check caught it.
        #
        # So a failure is fatal by default and opt-out only. run_windows also refuses to merge
        # windows with mismatched detector sets, but that is the second line of defence: it
        # cannot repair the cached window, it can only tell you to delete it.
        if os.environ.get("ALLOW_MISSING_BARKMEIER") != "1":
            raise SystemExit(
                f"Barkmeier failed: {type(e).__name__}: {e}\n"
                f"  REFUSING to write a run without it -- a cached window missing one detector "
                f"is silent and permanent.\n"
                f"  If this is a licence hiccup, just re-run; completed windows are reused.\n"
                f"  On a machine with no MATLAB at all, set ALLOW_MISSING_BARKMEIER=1.")
        print(f"[warn] Barkmeier unavailable ({type(e).__name__}: {e}); skipping that arm "
              f"(ALLOW_MISSING_BARKMEIER=1).")
        bark = [np.zeros(0, int) for _ in range(n_chan)]

if RUN_DELPHOS and ONLY in ("", "delphos"):
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
