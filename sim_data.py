"""
sim_data.py
-----------
Synthetic SEEG with KNOWN spike times, so the three detectors can be scored against ground
truth instead of only against each other. Python port of
seeg_analysis/spike_detector_comparison/make_sim_detector_test_data.m.

WHY THIS EXISTS
  compare_spikes.py can only measure AGREEMENT. On the real 600 s P1 baseline that hit a wall:
  Barkmeier is the MOST self-consistent detector (self-rho 0.944 vs Janca 0.874, Delphos 0.778)
  yet agrees LEAST with the other two -- so its disagreement is systematic, not noise, and no
  amount of cross-detector correlation can say which one is right. Ground truth is the only
  arbiter.

WHAT IS SIMULATED
  * Background: per-channel AR(16) models fitted (Burg) on 60 s of real bipolar SEEG from P1,
    shipped as sim_noise_model.mat (164 models @ 2000 Hz). White noise driven through the
    all-pole filter reproduces each channel's spectrum and amplitude to within ~2%.
  * Spikes: a parametric transient (sharp Gaussian peak + slow after-wave), unit-peak, scaled
    to `snr * channel_noise_std`. ADDITIVE.
  * Times: clamped-exponential renewal, independent per channel, rates ramped 0..MAX_RATE_MIN
    across channels so channel 1 is a zero-rate control.
  * NO artefacts. The MATLAB injects none, and compare_spikes.py zeroes the QC mask in sim mode
    (see SIM_CLEAN_MASK) because any mask on clean data is a QC false positive that would delete
    true spikes and silently depress recall.

IS THE TEMPLATE FAIR TO ALL THREE DETECTORS?
  Worth checking, because the obvious statistic gives the wrong answer. At the MATLAB default
  SHARPNESS=2.5 the transient is a sigma = 10 ms Gaussian whose ENERGY splits 81% below 8 Hz /
  18% in 8-30 Hz / 0.7% in 30-80 Hz / 0.04% above 80 Hz. Delphos scores time-frequency blob
  sharpness over 8-512 Hz, so that reads as "Delphos is handed 19% of the signal and nothing
  in the band that makes a blob look sharp" -- i.e. any Delphos floor would be a property of
  the stimulus, not the detector.

  That inference is wrong, and inband_snr() is what shows it: the AR background is steeply
  1/f, so Delphos's band contains very little NOISE either. Measured at the default config,
  median in-band SNR (max|bandpassed spike| / std(bandpassed noise)) is

      nominal SNR      1      2      3      5      8     12
      Janca         2.84   5.68   8.52  14.20  22.73  34.09
      Barkmeier     1.59   3.17   4.76   7.93  12.69  19.04
      Delphos       2.59   5.18   7.76  12.94  20.70  31.05

  Every arm has usable signal across the whole sweep, and Delphos sits within 10% of Janca.
  So a Delphos floor on this data WOULD be a detector result. Do not take that on trust --
  main() reprints this table on every build, and band_energy() is kept so the misleading
  statistic and the correct one can be read side by side.

  SHARPNESS is still the knob if you want a stimulus control. Raising it moves energy up in
  frequency but NARROWS the peak, so it helps Delphos and hurts the two 400 Hz arms:
  at sharpness 12 the medians at nominal SNR 5 are Janca 13.6 / Barkmeier 10.8 / Delphos 20.5.
  (A cusp-shaped peak was tried and dropped: at matched FWHM it is uniformly WORSE in band for
  all three, because unit-peak normalisation shrinks its total energy.)

  Barkmeier's TAMP=1200 (tuned on real data) is a different matter: it is a summed half-height
  threshold in uV, while injected peaks here are 12-57 uV at SNR 1 and 60-286 uV at SNR 5. The
  MATLAB generator used TAMP=400 for exactly this reason. Expect Barkmeier to see nothing at
  low SNR until you sweep it -- an operating-point fact, not a sensitivity result.

OUTPUT (per SNR level), under sim_data/:
    sim_<tag>_snr<k>_<cfghash>.edf         real EDF, 2000 Hz, 1 s records, int16
    sim_<tag>_snr<k>_<cfghash>.truth.npz   ground truth + provenance

  The config hash is in the FILENAME on purpose. delphos_detect_spikes.py keys its 5-min cache
  on (resolved path, file SIZE, window, params) -- and every SNR level produces a byte-identical
  FILE SIZE, so the path is the only discriminator. Hashing the whole generator config into the
  name means changing SHARPNESS, SEED, N_CHAN, anything, forces a new cache key instead of
  silently serving detections computed on different data.

Run with the local venv:
    .venv\\Scripts\\python.exe sim_data.py            # build every SNR level + the diagnostic
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, lfilter

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MODEL = Path(r"C:\Users\amoo0039\Documents\seeg_analysis\spike_detector_comparison"
             r"\sim_noise_model.mat")
OUT_DIR = HERE / "sim_data"     # the EDFs (large, gitignored, regenerable)
FIG_DIR = HERE / "figures" / "sim"   # every figure this module draws is simulated

TAG = "ar16"           # names the sim SET; bump it when you build a variant to keep both
N_CHAN = 16            # MATLAB parity. NOTE evaluate_detectors.py warns below 25 rankable
                       # channels and its TOP_K=10 overlap is near-vacuous at 16 -- the rank
                       # views are the accepted cost of this width; rate/reliability are fine.
LABEL_STUB = "SIM"     # channels SIM1..SIMn
DUR_SEC = 600          # MUST equal compare_spikes.SECONDS. Integer seconds (1 s EDF records).
MAX_RATE_MIN = 30.0    # rates ramp linspace(0, MAX_RATE_MIN, N_CHAN) -> ch 1 is a 0-rate control
MIN_ISI_MS = 200.0     # refractory floor on the renewal process

SPIKE_DUR_MS = 50.0    # sharp-transient width
SHARPNESS = 2.5        # larger -> narrower peak (sigma = dur/(2*sharpness)), which trades the
                       # 400 Hz arms against Delphos -- see the in-band SNR note in the docstring
SLOW_WAVE = True       # add the after-wave at all
# THREE-COMPONENT MORPHOLOGY, replacing the MATLAB's sharp-plus-opposite-trough. The shape is
#     sharp transient  ->  dip BELOW baseline  ->  slow-wave mound in the SAME direction
# which is what this dataset actually looks like. Widths are FWHM in ms (a duration is what
# gets specified clinically); amplitudes are fractions of the sharp peak.
#
# This is not cosmetic: Barkmeier measures its right half-wave as peak minus the minimum within
# trough_search_ms (40 ms), so where the undershoot sits and how deep it is directly set Ramp
# and Rdur -- the features it gates on. Set UNDERSHOOT_AMP=0 and SLOW_WAVE_AMP=-0.4 with
# SLOW_WAVE_MS=283, SLOW_WAVE_DELAY_MS=110 to get back to the MATLAB waveform.
UNDERSHOOT_AMP = 0.25       # depth of the post-transient dip, as a fraction of the sharp peak
UNDERSHOOT_MS = 45.0        # its FWHM
UNDERSHOOT_DELAY_MS = 38.0  # its centre, ms after the sharp peak
SLOW_WAVE_AMP = 0.50        # mound height as a fraction of the sharp peak
SLOW_WAVE_MS = 80.0         # mound FWHM
SLOW_WAVE_DELAY_MS = 120.0  # mound centre, ms after the sharp peak
# Realised at fs=2000 with the defaults above: peak +1.00, trough -0.23 at +37 ms,
# mound +0.52 at +120 ms with FWHM 78 ms, total template 400 ms.
TEMPLATE = "gaussian"  # only shape kept; a matched-FWHM cusp measured worse in every band
TAPER_MS = 60.0        # DEPARTS FROM THE MATLAB, deliberately -- set 0 for exact parity.
                       # make_spike_template.m truncates the template at dur+250 ms while the
                       # slow wave is still at -0.19 of peak, so EVERY injected spike ends in a
                       # step discontinuity back to baseline (30 uV at SNR 8 = 1.5x the noise
                       # std). That step is a genuine sharp edge, and it is detected: Delphos's
                       # unmatched marks sat at +286 ms, exactly the template end, costing it
                       # ~150 of 2552 marks at SNR 8. It penalises the SHARPNESS detector
                       # specifically, which is precisely the comparison being run. A raised-
                       # cosine ramp over the last TAPER_MS (and the short lead-in) removes it.

EQUALISE_CHANNEL_SNR = False
                       # Only meaningful with FIXED_PEAK_UV set.
                       # True  -> each channel's noise is scaled so EVERY channel sits at
                       #          exactly the nominal SNR. Clean x-axis, but it erases the one
                       #          thing worth testing below.
                       # False -> ALL channels are scaled by ONE factor, so the AR pool's
                       #          natural 12-58 uV spread survives and channels sit at
                       #          genuinely DIFFERENT SNRs within a single recording (about
                       #          0.34x to 1.67x the nominal).
                       # WHY THAT MATTERS. Barkmeier normalises by a SINGLE GLOBAL SCALAR per
                       # 1-minute block -- mDetectSpike.m:284,
                       #     scale = SCALE / median(mean(abs(EEG_1-35Hz)))
                       # taken as the MEDIAN ACROSS CHANNELS -- whereas Janca fits a background
                       # model PER CHANNEL and Delphos normalises its TF plane per channel. So
                       # on a recording with unequal channels, Barkmeier's effective threshold
                       # is too high on quiet channels and too low on noisy ones, while the
                       # other two adapt. With EQUALISE_CHANNEL_SNR=True every channel is
                       # identical and that asymmetry is invisible; with False it is directly
                       # measurable as per-channel recall vs channel noise, inside one run.
                       # (Real P1 channels span 7-44 uV, so this is a real-data effect.)
FIXED_PEAK_UV = 143.0  # None -> MATLAB behaviour: amp = snr * channel noise std, so the spike
                       # grows with SNR and every channel gets a different amplitude.
                       # A NUMBER -> every spike on every channel has exactly this peak (uV)
                       # and the SNR sweep is achieved by SCALING THE NOISE instead. Two
                       # reasons this is the more informative design:
                       #   1. It separates "can the detector see a spike of THIS size" from
                       #      "does the spike happen to grow with the noise". With the spike
                       #      held still, a recall curve is a pure function of the background.
                       #   2. Combined with EQUALISE_CHANNEL_SNR=False it puts channels at
                       #      different SNRs inside one recording, which is what exposes
                       #      Barkmeier's global normaliser (see above).
                       # NOT a test of absolute vs relative thresholding -- an earlier version
                       # of this comment claimed that and was WRONG. TAMP is NOT in uV: it is
                       # compared against half-wave amplitudes measured AFTER
                       # mDetectSpike.m:284-285 rescales the block by
                       # 100/median(mean(abs(EEG))). Measured on real P1 that factor is 7.02,
                       # so TAMP=1200 means 171 uV of summed half-heights -- which sits right
                       # on the real median spike (197 uV peak-to-peak). TAMP therefore ADAPTS
                       # to the recording amplitude like the other two, and the absolute uV
                       # value of FIXED_PEAK_UV is arbitrary. 143 uV is chosen only because it
                       # is the real P1 median, so the plots carry readable numbers.
                       # The noise is scaled PER CHANNEL so every channel sits at exactly the
                       # target SNR -- otherwise a single global multiplier leaves the 16
                       # channels spread over 24x-117x because the AR pool spans 12-58 uV, and
                       # each SNR level would be a mixture rather than a point.
                       # With 143 uV and EQUALISE_CHANNEL_SNR=False the recording is close
                       # to real P1 at nominal SNR ~8 (real median spike 143 uV against a
                       # median channel noise of 18 uV).
AMP_LOG_SD = 0.61      # per-spike amplitude spread: amp = median_amp * lognormal(0, AMP_LOG_SD).
                       # 0 -> every spike identical (the MATLAB behaviour, and what made the
                       # individual traces in sim_preview differ only by background noise).
                       # 0.61 reproduces the p90/p10 ratio of 4.8 measured on real P1 spikes;
                       # the median is left exactly on target, so FIXED_PEAK_UV / SNR still
                       # mean what they say. NOTE the real tail is HEAVIER than lognormal --
                       # real p99/median is 7.3 against 4.2 for this sigma -- so the very
                       # largest spikes are under-represented.
                       # This is also what turns sim_pdetect_vs_amp.png from 16 discrete
                       # points into a genuine psychometric curve.
SNR_LIST = (1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 12.0)   # peak amplitude / channel noise std
                       # 6.0 sits in the steepest part of Barkmeier's recall curve (0.35 at
                       # SNR 5 -> 0.65 at SNR 8) and is the level to inspect the detailed
                       # per-run figures on. NOTE SNR_LIST is deliberately NOT in default_cfg:
                       # adding a level must not change the config hash and invalidate every
                       # EDF and Delphos cache entry already built.
SEED = 1000
PAIRED_ACROSS_SNR = True   # DEPARTS FROM MATLAB (which reseeds per level, rng(1000+si)).
                           # True -> noise and spike times do not depend on the SNR index, so
                           # the six recordings differ ONLY in injected amplitude and the SNR
                           # curve carries no re-draw variance. False restores MATLAB behaviour.
PHYS_HEADROOM = 1.1        # EDF physical range = ceil(headroom * max|x|), so peaks never clip
PREVIEW_SNRS = (3.0, 6.0, 12.0)   # levels main() draws a trace preview for; () to skip
# WHAT THESE AMPLITUDES CORRESPOND TO, measured on the real P1 baseline (600 s, 226 bipolar
# channels, at 20731 Janca detections):
#     channel noise scale (MAD*1.4826) : median 18.4 uV   (p10-p90  7.4 - 43.7)
#     spike peak deviation             : median  143 uV   (p10-p90   47 -  374, p99 1040)
#     EFFECTIVE SNR = peak / noise     : median  5.9      (p10-p90  3.2 -  15.3, p99 49.7)
# So SNR 6 reproduces the MEDIAN real spike almost exactly, and the AR noise is already right
# (sim 12-58 uV vs real 7-44). Raising SNR further does not make the sim more realistic -- at
# SNR 12 every spike sits near the real p73.
# WHAT IS STILL WRONG: `amp = snr * noise_std` is CONSTANT per channel, so the sim has no
# amplitude distribution at all, while real spikes on one channel span 20x (47 -> 1040 uV).
# Fixing that means amp = snr * noise_std * lognormal(0, ~0.61), which reproduces the measured
# p90/p10 ratio of 4.8 and would turn sim_pdetect_vs_amp.png into a real psychometric curve
# instead of 16 discrete points. NOTE the measurement is taken at Janca's detections, so it is
# truncated from below -- the p10 partly reflects Janca's threshold, not the true floor.

# Passbands the in-band SNR diagnostic reports, one per detector arm.
DETECTOR_BANDS = {"Janca": (10.0, 60.0),      # janca_detect_spikes default fl/fh
                  "Barkmeier": (20.0, 50.0),  # compare_spikes BARK filter_spec
                  "Delphos": (8.0, 512.0)}    # delphos_detect_spikes freq_band_start/end

_KIND_NOISE, _KIND_TIMES, _KIND_AMP = 0, 1, 2


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def mround(x):
    """MATLAB round: half AWAY FROM ZERO. numpy rounds half to even, which would put 2.5 -> 2
    and shift ground-truth samples by one relative to the MATLAB generator."""
    x = np.asarray(x, float)
    return np.floor(np.abs(x) + 0.5) * np.sign(x)


def default_cfg(**override):
    """The generator config as a plain dict -- this is what gets hashed into the filename and
    stored in the npz, so every field here must be JSON-serialisable and must actually affect
    the data."""
    cfg = dict(tag=TAG, n_chan=N_CHAN, label_stub=LABEL_STUB, dur_sec=DUR_SEC,
               max_rate_min=MAX_RATE_MIN, min_isi_ms=MIN_ISI_MS, spike_dur_ms=SPIKE_DUR_MS,
               sharpness=SHARPNESS, slow_wave=SLOW_WAVE,
               slow_wave_amp=SLOW_WAVE_AMP, slow_wave_ms=SLOW_WAVE_MS,
               slow_wave_delay_ms=SLOW_WAVE_DELAY_MS,
               undershoot_amp=UNDERSHOOT_AMP, undershoot_ms=UNDERSHOOT_MS,
               undershoot_delay_ms=UNDERSHOOT_DELAY_MS,
               template=TEMPLATE, taper_ms=TAPER_MS,
               fixed_peak_uv=FIXED_PEAK_UV, amp_log_sd=AMP_LOG_SD,
               equalise_channel_snr=EQUALISE_CHANNEL_SNR,
               seed=SEED, paired_across_snr=PAIRED_ACROSS_SNR, phys_headroom=PHYS_HEADROOM,
               model=str(MODEL))
    unknown = set(override) - set(cfg)
    if unknown:
        raise TypeError(f"unknown sim setting(s) {sorted(unknown)}; valid: {sorted(cfg)}")
    cfg.update(override)
    return cfg


def cfg_hash(cfg):
    """8 hex chars over the whole config. Short enough to read in a filename, wide enough that
    a collision between two configs you actually try is not a thing."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def edf_name(cfg, snr):
    return f"sim_{cfg['tag']}_snr{snr:g}_{cfg_hash(cfg)}.edf"


def _rng(cfg, snr, kind, ch):
    """Per-(level, kind, channel) stream. Independent substreams mean adding a channel or an
    SNR level cannot shift any OTHER channel's realisation -- which a single shared stream
    would do, silently invalidating every comparison you had already made."""
    level = 0 if cfg["paired_across_snr"] else int(round(float(snr) * 1000))
    return np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), level, kind, ch]))


# ----------------------------------------------------------------------
# noise model
# ----------------------------------------------------------------------
def load_noise_model(path=None):
    """Read sim_noise_model.mat (fit by fit_sim_noise_model.m: Burg AR(16) on 60 s of P1
    bipolar SEEG at 2000 Hz). `resid_std` is the DRIVE std for synthesis, `source_std` is the
    real channel's std -- the synthesised output lands near the latter."""
    path = Path(path or MODEL)
    if not path.is_file():
        raise FileNotFoundError(
            f"noise model not found: {path}\nIt lives in the seeg_analysis repo at "
            f"spike_detector_comparison/sim_noise_model.mat; point MODEL at your copy.")
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    models = [{"label": str(r.label), "a": np.asarray(r.a, float).ravel(),
               "resid_std": float(r.resid_std if hasattr(r, "resid_std") else r.residStd),
               "source_std": float(r.source_std if hasattr(r, "source_std") else r.sourceStd)}
              for r in np.atleast_1d(m["arModels"])]
    return {"fs": float(m["fs"]), "ar_order": int(m["arOrder"]),
            "method": str(m["arMethod"]), "models": models}


def ar_background(model, n_samp, rng):
    """One channel of background: white noise through the all-pole AR filter, zero initial
    conditions (so the MATLAB's start-of-record transient is reproduced, not papered over)."""
    drive = model["resid_std"] * rng.standard_normal(n_samp)
    return lfilter([1.0], model["a"], drive)


# ----------------------------------------------------------------------
# spike template
# ----------------------------------------------------------------------
def make_template(fs, dur_ms=None, sharpness=None, slow_wave=None, shape=None,
                  taper_ms=None, sw_amp=None, sw_ms=None, sw_delay_ms=None,
                  us_amp=None, us_ms=None, us_delay_ms=None):
    """Unit-peak epileptiform transient + the 0-BASED index of its peak.

    Reproduces make_spike_template.m, with ONE deliberate difference: the ends are tapered to
    zero (see TAPER_MS). With taper_ms=0 it is the MATLAB exactly -- at fs=2000, dur 50 ms,
    sharpness 2.5, slow wave on: n=600, peak at index 30, minimum -0.542580517 at +110 ms,
    and a -0.1885 step at the final sample.

    The taper is applied BEFORE the unit-peak normalisation and its ramps are placed clear of
    both the sharp peak and the slow-wave trough, so the morphology the detectors key on is
    unchanged -- only the truncation artefact goes.
    """
    dur_ms = SPIKE_DUR_MS if dur_ms is None else float(dur_ms)
    sharpness = SHARPNESS if sharpness is None else float(sharpness)
    slow_wave = SLOW_WAVE if slow_wave is None else bool(slow_wave)
    shape = (TEMPLATE if shape is None else shape).lower()
    taper_ms = TAPER_MS if taper_ms is None else float(taper_ms)
    sw_amp = SLOW_WAVE_AMP if sw_amp is None else float(sw_amp)
    sw_ms = SLOW_WAVE_MS if sw_ms is None else float(sw_ms)
    sw_delay_ms = SLOW_WAVE_DELAY_MS if sw_delay_ms is None else float(sw_delay_ms)
    us_amp = UNDERSHOOT_AMP if us_amp is None else float(us_amp)
    us_ms = UNDERSHOOT_MS if us_ms is None else float(us_ms)
    us_delay_ms = UNDERSHOOT_DELAY_MS if us_delay_ms is None else float(us_delay_ms)

    sigma = (dur_ms / 1000.0) / (2.0 * max(sharpness, 0.5))
    # long enough for the mound to decay back to baseline before the taper starts, or the
    # taper would clip the slow wave rather than just the tail
    total_ms = (dur_ms + sw_delay_ms + 2.5 * sw_ms) if slow_wave else dur_ms + 40.0
    n = int(mround(total_ms / 1000.0 * fs))
    # peak sits `0.30 * dur_ms` into the template so a leading baseline exists. (The MATLAB
    # comment says "~30% into the template"; the CODE is 30% of dur_ms, which at these settings
    # is 5% of the template. The code is what was used, so the code is what is ported.)
    lead0 = int(mround(0.30 * dur_ms / 1000.0 * fs))
    # When tapering, PREPEND real baseline rather than ramping over the existing lead-in. That
    # lead-in is only 15 ms and is already on the spike's rising edge -- windowing it would
    # distort exactly the feature Barkmeier's left-slope/left-duration criteria gate on, which
    # is the comparison this simulation exists to make.
    pad = int(mround(0.5 * taper_ms / 1000.0 * fs)) if taper_ms > 0 else 0
    n += pad
    t = (np.arange(n) - (lead0 + pad)) / fs

    if shape != "gaussian":
        raise ValueError(f"unknown template shape {shape!r}; only 'gaussian' exists (a "
                         f"matched-FWHM cusp was measured and dropped -- see the module "
                         f"docstring). Use `sharpness` to move the spectrum instead.")
    sharp = np.exp(-(t ** 2) / (2.0 * sigma ** 2))

    tmpl = sharp
    if slow_wave:
        # THREE components, not the MATLAB's two: sharp transient -> UNDERSHOOT below baseline
        # -> a distinct slow-wave mound in the same direction as the spike. Widths are given as
        # FWHM in ms and converted here (FWHM = 2*sqrt(2 ln2)*sigma = 2.3548*sigma), because a
        # duration is what gets specified clinically and a sigma is not.
        _F = 2.0 * math.sqrt(2.0 * math.log(2.0))
        tmpl = (sharp
                - us_amp * np.exp(-((t - us_delay_ms / 1000.0) ** 2)
                                  / (2.0 * (us_ms / 1000.0 / _F) ** 2))
                + sw_amp * np.exp(-((t - sw_delay_ms / 1000.0) ** 2)
                                  / (2.0 * (sw_ms / 1000.0 / _F) ** 2)))

    if taper_ms > 0:
        trail = min(int(mround(taper_ms / 1000.0 * fs)), n - pad - 1)
        w = np.ones(n)
        if pad > 0:      # ramps over PREPENDED baseline only -- the spike is never touched
            w[:pad] = 0.5 * (1 - np.cos(np.pi * np.arange(pad) / pad))
        if trail > 0:
            w[n - trail:] = 0.5 * (1 + np.cos(np.pi * np.arange(1, trail + 1) / trail))
        tmpl = tmpl * w

    tmpl = tmpl / np.max(np.abs(tmpl))
    return tmpl, int(np.argmax(np.abs(tmpl)))


# ----------------------------------------------------------------------
# spike times
# ----------------------------------------------------------------------
def draw_spike_times(rate_per_min, dur_sec, fs, min_isi_ms, guard, n_samp, rng):
    """Clamped-exponential renewal process -> 0-based peak sample indices.

    `max(exponential, min_isi)` is CENSORING, not rejection or shifting: it puts a point mass at
    exactly min_isi and makes the realised rate slightly below nominal. That is what the MATLAB
    does and what the detectors were compared against, so it is reproduced rather than fixed.
    """
    lam = float(rate_per_min) / 60.0
    if lam <= 0:
        return np.zeros(0, np.int64)
    min_isi = float(min_isi_ms) / 1000.0

    times, t = [], 0.0
    while True:
        # 1 - random() is in (0, 1], so -log() is finite. numpy's random() CAN return 0.0
        # (MATLAB's rand cannot), which would give an infinite ISI and truncate the train.
        isi = max(-math.log(1.0 - rng.random()) / lam, min_isi)
        t += isi
        if t > dur_sec:
            break
        times.append(t)
    if not times:
        return np.zeros(0, np.int64)

    pk = mround(np.asarray(times) * fs).astype(np.int64)   # MATLAB's +1 dropped: 0-based
    return pk[(pk >= guard) & (pk < n_samp - guard)]


# ----------------------------------------------------------------------
# synthesis
# ----------------------------------------------------------------------
def synthesise(cfg, snr, model=None):
    """Build one recording. Returns data + ground truth, all 0-based, samples at the model's fs."""
    nm = model or load_noise_model(cfg["model"])
    fs = nm["fs"]
    pool = nm["models"]
    n_chan = int(cfg["n_chan"])
    n_samp = int(round(cfg["dur_sec"] * fs))
    if n_samp % int(fs):
        raise ValueError(f"dur_sec*fs must be a whole number of 1 s records, got {n_samp}")

    tmpl, peak_off = make_template(fs, cfg["spike_dur_ms"], cfg["sharpness"],
                                   cfg["slow_wave"], cfg["template"], cfg["taper_ms"],
                                   cfg["slow_wave_amp"], cfg["slow_wave_ms"],
                                   cfg["slow_wave_delay_ms"], cfg["undershoot_amp"],
                                   cfg["undershoot_ms"], cfg["undershoot_delay_ms"])
    tlen = tmpl.size
    guard = tlen + 1                       # the whole template always fits inside the record
    rates = np.linspace(0.0, float(cfg["max_rate_min"]), n_chan)
    labels = [f"{cfg['label_stub']}{k + 1}" for k in range(n_chan)]

    x = np.zeros((n_samp, n_chan))
    noise_std = np.zeros(n_chan)
    truth_idx, truth_chan, truth_amp = [], [], []

    fixed = cfg.get("fixed_peak_uv")
    equalise = bool(cfg.get("equalise_channel_snr", True))

    # PASS 1 -- background only. The scaling decision needs every channel's natural level
    # first, because with equalise=False the whole set is scaled by ONE factor derived from
    # the median channel, which is what preserves the pool's 12-58 uV spread.
    sd0 = np.zeros(n_chan)
    for ch in range(n_chan):
        noise = ar_background(pool[ch % len(pool)], n_samp, _rng(cfg, snr, _KIND_NOISE, ch))
        sd0[ch] = float(np.std(noise, ddof=1))       # PURE noise, before any spike is added
        x[:, ch] = noise
    if fixed:
        gain = (float(fixed) / (float(snr) * sd0) if equalise
                else np.full(n_chan, float(fixed) / (float(snr) * float(np.median(sd0)))))
        x *= gain                                     # amplitude only; AR spectra unchanged
    else:
        gain = np.ones(n_chan)

    # PASS 2 -- inject
    for ch in range(n_chan):
        amp = float(fixed) if fixed else float(snr) * sd0[ch]
        noise_std[ch] = sd0[ch] * gain[ch]

        pk = draw_spike_times(rates[ch], cfg["dur_sec"], fs, cfg["min_isi_ms"],
                              guard, n_samp, _rng(cfg, snr, _KIND_TIMES, ch))
        # per-spike amplitude, lognormal about `amp` so the MEDIAN stays exactly on target
        log_sd = float(cfg.get("amp_log_sd") or 0.0)
        amps_k = (amp * np.exp(_rng(cfg, snr, _KIND_AMP, ch).normal(0.0, log_sd, pk.size))
                  if log_sd > 0 else np.full(pk.size, amp))
        for p, a in zip(pk, amps_k):
            s0 = int(p) - peak_off
            x[s0:s0 + tlen, ch] += a * tmpl        # additive; consecutive tails DO overlap
                                                   # (min ISI 200 ms < 300 ms template)
        truth_idx.append(pk)
        truth_chan.append(np.full(pk.size, ch, np.int64))
        truth_amp.append(amps_k)

    return {"data": x, "fs": fs, "labels": labels, "n_chan": n_chan, "n_samp": n_samp,
            "truth_idx": np.concatenate(truth_idx) if truth_idx else np.zeros(0, np.int64),
            "truth_chan": np.concatenate(truth_chan) if truth_chan else np.zeros(0, np.int64),
            "truth_amp": np.concatenate(truth_amp) if truth_amp else np.zeros(0),
            "noise_std": noise_std, "rates_per_min": rates,
            "template": tmpl, "template_peak": peak_off, "snr": float(snr)}


# ----------------------------------------------------------------------
# EDF writer
# ----------------------------------------------------------------------
def _fld(value, width, name):
    s = str(value)
    if len(s) > width:
        raise ValueError(f"EDF field {name!r} needs {len(s)} chars, only {width} available: {s!r}")
    return s.ljust(width).encode("ascii")


def write_edf(path, x, labels, fs, headroom=None):
    """Write [n_samp x n_chan] microvolts to a plain EDF: 1 s data records, int16 LE.

    Hand-rolled rather than adding pyedflib/edfio, for the same reason seeg/edf.py hand-parses
    the header -- it is a fixed-width ASCII layout and this is four numbers plus a matrix.
    verify_edf() then proves the round trip, so a writer bug can never be misdiagnosed as a
    Delphos bug.

    Two details that would otherwise corrupt silently:
      * physical ranges are INTEGER uV (ceil of the headroom'd peak). "%d" always fits the
        8-char field and round-trips exactly through both readers; a float like 1234.5678 does
        not, and truncation there is invisible until it isn't. Costs <1% of dynamic range.
      * the digital range is ASYMMETRIC (-32768..32767) against a symmetric physical range, so
        the mapping is the EDF affine, NOT x/pmax*32767. The -0.5 term is that asymmetry; drop
        it and every channel gains a half-LSB DC offset.
    """
    headroom = PHYS_HEADROOM if headroom is None else float(headroom)
    x = np.asarray(x, float)
    n_samp, n_chan = x.shape
    fs_i = int(round(fs))
    if abs(fs - fs_i) > 1e-9:
        raise ValueError(f"EDF needs an integer sample rate, got {fs}")
    if n_samp % fs_i:
        raise ValueError(f"n_samp ({n_samp}) must be a whole number of 1 s records at {fs_i} Hz")
    if len(labels) != n_chan:
        raise ValueError(f"{len(labels)} labels for {n_chan} channels")
    if len(set(labels)) != n_chan:
        raise ValueError("EDF channel labels must be unique")

    n_rec = n_samp // fs_i
    pmax = np.maximum(np.ceil(headroom * np.max(np.abs(x), axis=0)), 1.0)   # floor 1: an
    # all-zero channel must not produce a zero-width physical range (division by zero on read)
    gain = 65535.0 / (2.0 * pmax)                       # (dig_max - dig_min) / (phys span)
    dig = np.clip(np.round(x * gain - 0.5), -32768, 32767).astype("<i2")

    hdr = bytearray()
    hdr += _fld("0", 8, "version")
    hdr += _fld("X X X X", 80, "patient")
    hdr += _fld("Startdate 01-JAN-2001 X X sim_data.py", 80, "recording")
    hdr += _fld("01.01.01", 8, "startdate")
    hdr += _fld("00.00.00", 8, "starttime")
    hdr += _fld(256 * (1 + n_chan), 8, "header_bytes")
    hdr += _fld("", 44, "reserved")                     # blank: plain EDF, not EDF+C
    hdr += _fld(n_rec, 8, "n_records")
    hdr += _fld(1, 8, "record_duration")
    hdr += _fld(n_chan, 4, "n_signals")

    for i, lab in enumerate(labels):
        hdr += _fld(lab, 16, f"label[{i}]")
    hdr += b"".join(_fld("", 80, "transducer") for _ in range(n_chan))
    hdr += b"".join(_fld("uV", 8, "dimension") for _ in range(n_chan))
    hdr += b"".join(_fld(f"{-int(p)}", 8, "phys_min") for p in pmax)
    hdr += b"".join(_fld(f"{int(p)}", 8, "phys_max") for p in pmax)
    hdr += b"".join(_fld(-32768, 8, "dig_min") for _ in range(n_chan))
    hdr += b"".join(_fld(32767, 8, "dig_max") for _ in range(n_chan))
    hdr += b"".join(_fld("", 80, "prefilter") for _ in range(n_chan))
    hdr += b"".join(_fld(fs_i, 8, "samples_per_record") for _ in range(n_chan))
    hdr += b"".join(_fld("", 32, "reserved") for _ in range(n_chan))
    assert len(hdr) == 256 * (1 + n_chan), f"header is {len(hdr)} bytes"

    # records are channel-major within each second: [ch0 x fs][ch1 x fs]...
    body = dig.reshape(n_rec, fs_i, n_chan).transpose(0, 2, 1)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(bytes(hdr))
        f.write(np.ascontiguousarray(body, dtype="<i2").tobytes())
    return path


def verify_edf(path, x, labels, fs):
    """Round-trip the file back through BOTH readers the pipeline uses and raise on any
    mismatch. Runs on every write (~0.5 s for 38 MB) -- cheap insurance against spending an
    hour of Delphos time on a file that says something other than what we synthesised."""
    from seeg import read_edf_header, load_edf_segment
    import mne

    path = Path(path)
    x = np.asarray(x, float)
    n_samp, n_chan = x.shape
    fs_i = int(round(fs))
    n_rec = n_samp // fs_i

    expect = 256 * (1 + n_chan) + n_rec * n_chan * fs_i * 2
    actual = path.stat().st_size
    if actual != expect:
        raise ValueError(f"{path.name}: {actual} bytes on disk, expected {expect}")

    hdr = read_edf_header(path)
    checks = [("SampleRate", hdr["SampleRate"], float(fs_i)),
              ("NumDataRecords", hdr["NumDataRecords"], n_rec),
              ("DataRecordDuration", hdr["DataRecordDuration"], 1.0),
              ("NSamplesTotal", hdr["NSamplesTotal"], n_samp)]
    for name, got, want in checks:
        if float(got) != float(want):
            raise ValueError(f"{path.name}: header {name} = {got}, expected {want}")
    if list(hdr["SignalLabels"]) != list(labels):
        raise ValueError(f"{path.name}: header labels {hdr['SignalLabels']} != {labels}")

    ch_names = mne.io.read_raw_edf(path, preload=False, verbose="error").ch_names
    if list(ch_names) != list(labels):
        raise ValueError(f"{path.name}: MNE channel ORDER {ch_names} != {labels}")

    y = load_edf_segment(path, hdr, start_rec=1, stop_rec=n_rec)["data"]
    if y.shape != x.shape:
        raise ValueError(f"{path.name}: read back {y.shape}, wrote {x.shape}")
    lsb = np.abs(hdr["PhysicalMax"] - hdr["PhysicalMin"]) / (hdr["DigitalMax"] - hdr["DigitalMin"])
    err = np.max(np.abs(y - x), axis=0)
    bad = np.flatnonzero(err > lsb * 1.001)
    if bad.size:
        raise ValueError(f"{path.name}: channels {bad.tolist()} differ by more than 1 LSB "
                         f"(max {err[bad].max():.4g} uV vs lsb {lsb[bad].max():.4g})")
    return {"ok": True, "max_err_lsb": float(np.max(err / lsb)), "lsb": lsb, "bytes": actual}


# ----------------------------------------------------------------------
# stimulus diagnostic
# ----------------------------------------------------------------------
def inband_snr(cfg, snr, bands=None, model=None, n_probe=None):
    """Per-detector-passband SNR of the injected transient. READ THIS BEFORE SPENDING DELPHOS TIME.

    The nominal `snr` is broadband peak / broadband noise std. What actually decides whether a
    detector fires is how much of the transient survives ITS filter. For a Gaussian template
    almost all the energy is below 8 Hz, so Delphos's 8-512 Hz band sees a small fraction of a
    spike that the nominal SNR calls large. Returns
      {"dets": [...], "snr": [n_chan x n_det]}   with snr = max|bp(spike)| / std(bp(noise)).
    """
    bands = bands or DETECTOR_BANDS
    nm = model or load_noise_model(cfg["model"])
    fs = nm["fs"]
    n_chan = int(cfg["n_chan"]) if n_probe is None else int(n_probe)
    tmpl, _ = make_template(fs, cfg["spike_dur_ms"], cfg["sharpness"],
                            cfg["slow_wave"], cfg["template"], cfg["taper_ms"],
                                   cfg["slow_wave_amp"], cfg["slow_wave_ms"],
                                   cfg["slow_wave_delay_ms"], cfg["undershoot_amp"],
                                   cfg["undershoot_ms"], cfg["undershoot_delay_ms"])
    n_probe_samp = int(round(20.0 * fs))              # 20 s of noise is plenty for a std
    pad = np.zeros(n_probe_samp)
    pad[:tmpl.size] = tmpl

    names = list(bands)
    out = np.zeros((n_chan, len(names)))
    fixed = cfg.get("fixed_peak_uv")
    equalise = bool(cfg.get("equalise_channel_snr", True))
    # mirror synthesise() exactly, including the two-pass global scaling -- otherwise this
    # diagnostic reports an equalised recording while the EDF on disk has a channel spread
    raw = [ar_background(nm["models"][ch % len(nm["models"])], n_probe_samp,
                         _rng(cfg, snr, _KIND_NOISE, ch)) for ch in range(n_chan)]
    sd0 = np.array([float(np.std(r, ddof=1)) for r in raw])
    if fixed:
        gain = (float(fixed) / (float(snr) * sd0) if equalise
                else np.full(n_chan, float(fixed) / (float(snr) * float(np.median(sd0)))))
    else:
        gain = np.ones(n_chan)

    for ch in range(n_chan):
        noise = raw[ch] * gain[ch]
        amp = float(fixed) if fixed else float(snr) * sd0[ch]
        for j, name in enumerate(names):
            lo, hi = bands[name]
            hi = min(hi, 0.99 * fs / 2.0)
            b, a = butter(4, [lo / (fs / 2.0), hi / (fs / 2.0)], btype="band")
            sig = filtfilt(b, a, amp * pad)
            bg = filtfilt(b, a, noise)
            out[ch, j] = np.max(np.abs(sig)) / np.std(bg, ddof=1)
    return {"dets": names, "snr": out}


def band_energy(cfg, edges=(0, 8, 30, 80, 512), model=None):
    """Fraction of the template's energy in each band -- the one-line version of why Delphos
    may be blind to this stimulus."""
    nm = model or load_noise_model(cfg["model"])
    fs = nm["fs"]
    tmpl, _ = make_template(fs, cfg["spike_dur_ms"], cfg["sharpness"],
                            cfg["slow_wave"], cfg["template"], cfg["taper_ms"],
                                   cfg["slow_wave_amp"], cfg["slow_wave_ms"],
                                   cfg["slow_wave_delay_ms"], cfg["undershoot_amp"],
                                   cfg["undershoot_ms"], cfg["undershoot_delay_ms"])
    n = 1 << int(np.ceil(np.log2(tmpl.size * 8)))
    p = np.abs(np.fft.rfft(tmpl, n)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    total = p.sum()
    return {f"{lo:g}-{hi:g} Hz": float(p[(f >= lo) & (f < hi)].sum() / total)
            for lo, hi in zip(edges, edges[1:])}


# ----------------------------------------------------------------------
# build / cache
# ----------------------------------------------------------------------
def ensure_sim_edf(cfg=None, snr=8.0, force=False, out_dir=None, log=print):
    """Idempotent build of one SNR level. Returns {"edf", "truth", "cfg", "snr", "rebuilt"}.

    A rebuild is skipped when both files exist AND the sidecar's config hash matches, so
    re-running the suite is free. Because the hash is in the filename, a changed config never
    overwrites an existing recording -- it makes a new one, which is also what stops Delphos's
    (path, size)-keyed cache from serving detections computed on different data.
    """
    cfg = cfg or default_cfg()
    out_dir = Path(out_dir or OUT_DIR)
    edf = out_dir / edf_name(cfg, snr)
    truth = edf.with_suffix(".truth.npz")
    want = cfg_hash(cfg)

    if edf.is_file() and truth.is_file() and not force:
        z = np.load(truth, allow_pickle=False)
        have = str(z["sim_cfg_hash"])
        if have == want:
            return {"edf": edf, "truth": truth, "cfg": cfg, "snr": float(snr), "rebuilt": False}
        raise RuntimeError(
            f"{truth.name} was built with config hash {have}, current config hashes to {want}. "
            f"The filename should have changed -- this means the sidecar was hand-edited or a "
            f"build was interrupted. Delete it, or pass force=True to overwrite.")

    log(f"[sim] building {edf.name} ...")
    nm = load_noise_model(cfg["model"])
    s = synthesise(cfg, snr, model=nm)
    write_edf(edf, s["data"], s["labels"], s["fs"], cfg["phys_headroom"])
    v = verify_edf(edf, s["data"], s["labels"], s["fs"])
    ib = inband_snr(cfg, snr, model=nm)

    np.savez(truth,
             truth_idx=s["truth_idx"].astype(np.int64),
             truth_chan=s["truth_chan"].astype(np.int64),
             truth_amp=s["truth_amp"].astype(float),
             truth_fs=float(s["fs"]),
             names=np.array(s["labels"], dtype="U"),
             snr=float(snr),
             noise_std=s["noise_std"],
             rates_per_min=s["rates_per_min"],
             inband_snr=ib["snr"],
             inband_dets=np.array(ib["dets"], dtype="U"),
             template=s["template"],
             template_peak=np.int64(s["template_peak"]),
             sim_tag=str(cfg["tag"]),
             sim_cfg_json=json.dumps(cfg, sort_keys=True),
             sim_cfg_hash=want)
    log(f"[sim] {edf.name}: {s['truth_idx'].size} spikes over {cfg['dur_sec']:g}s x "
        f"{cfg['n_chan']} ch, round trip max {v['max_err_lsb']:.2f} LSB, "
        f"{v['bytes'] / 1e6:.0f} MB")
    return {"edf": edf, "truth": truth, "cfg": cfg, "snr": float(snr), "rebuilt": True}


# ----------------------------------------------------------------------
def plot_preview(cfg=None, snr=6.0, chans=None, t0=None, dur_sec=12.0, out=None, log=print):
    """Look at the simulated data. Traces with the TRUE spike times marked, plus a zoom on the
    injected waveform against the noise it is buried in.

    Reads the EDF back rather than replotting `synthesise()` output, so what is drawn is what
    the detectors actually saw -- including the int16 quantisation of the round trip. Truth
    indices are at the file's own rate, so they align without resampling.
    """
    import matplotlib.pyplot as plt
    from seeg import read_edf_header, load_edf_segment
    from seeg._style import MUTED, RED, recessive

    cfg = cfg or default_cfg()
    built = ensure_sim_edf(cfg, snr, log=log)
    z = np.load(built["truth"], allow_pickle=False)
    fs = float(z["truth_fs"])
    labels = [str(s) for s in z["names"]]
    n_chan = len(labels)
    rates = z["rates_per_min"]
    truth = [z["truth_idx"][z["truth_chan"] == c] for c in range(n_chan)]
    # read the amplitudes actually injected rather than recomputing snr * noise_std -- with
    # FIXED_PEAK_UV / AMP_LOG_SD that formula no longer holds
    amps = np.array([np.median(z["truth_amp"][z["truth_chan"] == c]) if (z["truth_chan"] == c).any()
                     else 0.0 for c in range(n_chan)])
    amp_spread = [z["truth_amp"][z["truth_chan"] == c] for c in range(n_chan)]

    if chans is None:      # a silent control, two mid rates, and the busiest
        chans = sorted({0, n_chan // 3, 2 * n_chan // 3, n_chan - 1})
    busiest = chans[-1]
    if t0 is None:         # start somewhere with a few spikes to actually look at
        t0 = max(truth[busiest][3] / fs - 1.0, 0.0) if truth[busiest].size > 3 else 10.0

    hdr = read_edf_header(built["edf"])
    r0 = int(t0) + 1
    x = load_edf_segment(built["edf"], hdr, r0, min(int(t0 + dur_sec) + 1,
                                                    hdr["NumDataRecords"]))["data"]
    tt = (r0 - 1) + np.arange(x.shape[0]) / fs

    fig = plt.figure(figsize=(15, 4 + 1.5 * len(chans)))
    gs = fig.add_gridspec(len(chans) + 2, 3, height_ratios=[1] * len(chans) + [0.35, 1.6])

    # Per-channel scaling, in units of THAT channel's noise. A shared uV scale makes the
    # low-noise channels a flat line (the AR pool spans 12-58 uV), and scaling by noise is
    # also the honest comparison: it shows each spike at the prominence its own detector sees.
    for k, c in enumerate(chans):
        ax = fig.add_subplot(gs[k, :])
        sd_c = float(z["noise_std"][c])
        # The axis MUST clear the injected peak (snr * sd), not just the noise. At +-5 sd the
        # upstroke was cut off from SNR 6 up while the slow wave (0.54x as tall) stayed inside
        # it, so raising the SNR appeared to grow only the negative lobe -- a pure axis
        # artefact on data whose peak:trough ratio is fixed at 1.843 for every SNR.
        half = max(5.0 * sd_c, 1.25 * amps[c]) if rates[c] > 0 else 5.0 * sd_c
        ax.plot(tt, x[:, c], lw=0.7, color=MUTED)
        pk = truth[c] / fs
        pk = pk[(pk >= tt[0]) & (pk <= tt[-1])]
        if pk.size:
            ax.plot(pk, np.full(pk.size, half * 0.86), "v", ms=5, color=RED, clip_on=False)
        ax.set_ylim(-half, half)
        ax.set_xlim(tt[0], tt[-1])
        ax.set_ylabel(f"{labels[c]}\n{rates[c]:.0f}/min", fontsize=8)
        ax.set_yticks([-sd_c, 0, sd_c])
        ax.set_yticklabels(["-1sd", "0", "+1sd"], fontsize=6)
        if k < len(chans) - 1:
            ax.set_xticklabels([])
        sp = amp_spread[c]
        rng_txt = (f", p10-p90 {np.percentile(sp, 10):.0f}-{np.percentile(sp, 90):.0f}"
                   if sp.size > 10 and cfg.get("amp_log_sd") else "")
        note = (f"silent control  (noise sd {sd_c:.0f} uV)" if rates[c] == 0 else
                f"median peak {amps[c]:.0f} uV{rng_txt}  |  noise sd {sd_c:.0f} uV  "
                f"({amps[c] / sd_c:.1f}x)"
                f"{'   [no spike in this window]' if pk.size == 0 else ''}")
        ax.text(0.995, 0.04, note, transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
        recessive(ax)
    ax.set_xlabel("time (s)")

    # --- what one injected spike actually looks like against its own noise ---
    tmpl, peak = make_template(fs, cfg["spike_dur_ms"], cfg["sharpness"], cfg["slow_wave"],
                               cfg["template"], cfg["taper_ms"])
    w = int(0.25 * fs)
    ax2 = fig.add_subplot(gs[-1, 0])
    idx = truth[busiest]
    idx = idx[(idx > w) & (idx < int(hdr["NSamplesTotal"]) - w)][:200]
    full = load_edf_segment(built["edf"], hdr, 1, min(300, hdr["NumDataRecords"]))["data"][:, busiest]
    keep = idx[idx < full.size - w]
    segs = np.stack([full[i - w:i + w] for i in keep]) if keep.size else np.zeros((0, 2 * w))
    tz = (np.arange(-w, w)) / fs * 1000
    for s in segs[:25]:
        ax2.plot(tz, s, lw=0.5, color=MUTED, alpha=0.35)
    if segs.size:
        ax2.plot(tz, segs.mean(0), lw=2.0, color=RED, label=f"mean of {keep.size}")
    ax2.set_title(f"{labels[busiest]}: individual spikes (grey) vs their mean", fontsize=9)
    ax2.set_xlabel("ms from true peak")
    ax2.set_ylabel("uV")
    ax2.legend(frameon=False, fontsize=8)
    recessive(ax2)

    ax3 = fig.add_subplot(gs[-1, 1])
    clean = np.zeros(2 * w)
    s0 = w - peak
    n_fit = min(tmpl.size, clean.size - s0)      # the template outlasts the +-250 ms window
    clean[s0:s0 + n_fit] = amps[busiest] * tmpl[:n_fit]
    ax3.plot(tz, clean, lw=1.6, color=RED)
    ax3.axhline(0, lw=0.6, color=MUTED)
    for sd in (1, -1):
        ax3.axhline(sd * amps[busiest] / snr, lw=0.8, ls=":", color=MUTED)
    ax3.text(tz[5], amps[busiest] / snr, " +1 noise sd", fontsize=7, color=MUTED, va="bottom")
    ax3.set_title(f"the injected waveform alone, SNR {snr:g}", fontsize=9)
    ax3.set_xlabel("ms from true peak")
    recessive(ax3)

    ax4 = fig.add_subplot(gs[-1, 2])
    tms = np.arange(tmpl.size) / fs * 1000 - peak / fs * 1000
    if cfg.get("fixed_peak_uv"):
        # The spike is CONSTANT in this mode -- plotting it per SNR would be seven identical
        # curves. What moves is the noise, so draw the spike once against the +-1 sd band at
        # each level. That is the actual experiment: an absolute threshold (Barkmeier's TAMP)
        # stays put while the background rises to meet it.
        a = float(amps[busiest])
        ax4.plot(tms, a * tmpl, lw=1.8, color=RED, zorder=5, label=f"spike ({a:.0f} uV)")
        base = float(z["noise_std"][busiest]) * float(snr)      # = fixed peak, by construction
        for k, sv in enumerate(SNR_LIST):
            nsd = base / sv
            ax4.fill_between(tms, -nsd, nsd, color=MUTED, alpha=0.10, lw=0, zorder=1)
            ax4.text(tms[-1], nsd, f" SNR {sv:g}: +-{nsd:.0f} uV", fontsize=6, color=MUTED,
                     va="center", ha="left")
        ax4.set_xlim(tms[0], tms[-1] * 1.45)
        ax4.set_title("spike is FIXED; noise band grows as SNR falls", fontsize=9)
        ax4.legend(frameon=False, fontsize=7, loc="upper right")
    else:
        for sv in SNR_LIST:
            a = z["noise_std"][busiest] * sv
            ax4.plot(tms, a * tmpl, lw=1.2, alpha=0.85, label=f"SNR {sv:g} ({a:.0f} uV)")
        ax4.set_title("amplitude across the SNR sweep", fontsize=9)
        ax4.legend(frameon=False, fontsize=7)
    ax4.axhline(0, lw=0.6, color=MUTED)
    ax4.set_xlabel("ms from true peak")
    recessive(ax4)

    fig.suptitle(f"Simulated data, SNR {snr:g} -- red markers are the TRUE spike times "
                 f"({int(z['truth_idx'].size)} injected over {cfg['dur_sec']:g}s x "
                 f"{n_chan} ch)", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = Path(out or FIG_DIR / f"sim_preview_snr{snr:g}.png")
    fig.savefig(out, dpi=130)
    log(f"[saved] {out.name}")
    return out


def main():
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, MUTED

    cfg = default_cfg()
    print(f"--- sim set '{cfg['tag']}'  hash {cfg_hash(cfg)} ---")
    print(f"    {cfg['n_chan']} ch x {cfg['dur_sec']:g}s, template={cfg['template']} "
          f"dur={cfg['spike_dur_ms']:g}ms sharpness={cfg['sharpness']:g}")

    be = band_energy(cfg)
    print("\nTEMPLATE ENERGY BY BAND -- the MISLEADING statistic, printed so you can see why:")
    print("    " + "   ".join(f"{k}: {v:.2%}" for k, v in be.items()))
    print("    Reading this alone says Delphos (8-512 Hz) gets a sliver of the spike. It does")
    print("    -- but the AR background is steeply 1/f, so it gets a sliver of the noise too.")

    built = [ensure_sim_edf(cfg, snr) for snr in SNR_LIST]

    z = np.load(built[0]["truth"], allow_pickle=False)
    dets = [str(d) for d in z["inband_dets"]]
    print(f"\nIN-BAND SNR (median over channels) -- the statistic that decides detectability:")
    print(f"    max|bandpassed spike| / std(bandpassed noise), per detector passband")
    print(f"    {'nominal':>8}  " + "  ".join(f"{d:>10}" for d in dets))
    curves = {d: [] for d in dets}
    for b in built:
        zz = np.load(b["truth"], allow_pickle=False)
        med = np.median(zz["inband_snr"], axis=0)
        for d, v in zip(dets, med):
            curves[d].append(v)
        print(f"    {b['snr']:>8.0f}  " + "  ".join(f"{v:>10.2f}" for v in med))

    # Gate: an arm whose in-band SNR never clears ~1 cannot be expected to detect anything, and
    # a floor measured under that condition says nothing about the detector.
    floored = [d for d in dets if curves[d][-1] < 1.5]
    if floored:
        print(f"\n[!] {', '.join(floored)}: in-band SNR is still below 1.5 at the HIGHEST "
              f"nominal SNR.\n    Any detection floor you measure would be a property of the "
              f"STIMULUS, not the detector.\n    Adjust SHARPNESS / SPIKE_DUR_MS and rebuild "
              f"BEFORE spending ~50 min of Delphos time.")
    else:
        lo = min(curves[d][0] for d in dets)
        hi = ", ".join(f"{d} {curves[d][-1]:.1f}" for d in dets)
        print(f"\n[ok] every arm clears in-band SNR 1.5 at the top level ({hi}), worst case "
              f"{lo:.2f} at\n     nominal SNR {SNR_LIST[0]:g}. A detection floor measured here "
              f"IS a detector result.")

    colors = {"Janca": RED, "Barkmeier": BLUE, "Delphos": "#4a3aa7"}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for d in dets:
        ax.plot(SNR_LIST, curves[d], "-o", color=colors.get(d, MUTED), lw=1.6, label=d)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1)
    ax.annotate("in-band SNR = 1", (SNR_LIST[0], 1.0), textcoords="offset points",
                xytext=(4, 4), fontsize=8, color=MUTED)
    ax.set_xlabel("nominal SNR (broadband peak / noise std)")
    ax.set_ylabel("in-band SNR")
    ax.set_yscale("log")
    ax.set_title(f"What each detector's passband actually sees\n"
                 f"template={cfg['template']}, dur={cfg['spike_dur_ms']:g} ms, "
                 f"sharpness={cfg['sharpness']:g}", fontsize=10)
    ax.legend(frameon=False)
    ax.grid(alpha=.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "sim_inband_snr.png"
    fig.savefig(out, dpi=130)
    print(f"\n[saved] {out.name}")

    # Look at the data. A summary table cannot tell you the spikes are the right SIZE, and at
    # the amplitudes measured on real P1 (median effective SNR 5.9) they are NOT obviously
    # bigger than the background -- which is the point, and is only visible in a trace.
    for s in (PREVIEW_SNRS if PREVIEW_SNRS else ()):
        if s in SNR_LIST:
            plot_preview(cfg, s)


if __name__ == "__main__":
    main()
