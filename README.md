# spike_detector_comparison

Three interictal spike detectors run over the **same window** of the same recording, through
**identical post-processing**, and scored two ways: against each other on real SEEG, and
against known ground truth on synthetic data.

| detector | how it decides | implementation |
|---|---|---|
| **Janča** | statistical threshold on the Hilbert envelope of a band-passed signal, against a per-channel background model | `janca_detect_spikes.py` — pure Python port of `spike_detector_hilbert_v24.m` |
| **Barkmeier** | waveform *shape* — half-wave slopes, summed amplitude, half-widths | `seeg.detect_spikes` → `mDetectSpike.m` via the MATLAB engine |
| **Delphos** | sharpness of blobs in a whitened time-frequency plane | `delphos_detect_spikes.py` → compiled `Delphos_cmd_line.exe` |

They disagree because they use three unrelated definitions of "spike", not because one is
mistuned. The point of the repo is to characterise *how* they disagree and *why*.

---

## Data flow

```
   trials JSON ─► (patient, trial, condition) ─► resolve_file ─► EDF ─┐
                                                                     ├─► compare_spikes.py
   sim_data.py ──────────────────────────────────► synthetic EDF ────┘   (~5 min per
                                                                         uncached Delphos)
                          │                                    │
                    runs/<id>.npz                        sim_runs/*.npz
                          │                                    │
        ┌─────────────────┼──────────────────┐                 └─► score_sim_detectors.py
        ▼                 ▼                  ▼
  evaluate_detectors  spike_statistics  polyspike_review
```

**The npz is the only interface.** Every evaluation script reads it and **never re-runs a
detector** — that is deliberate, because a Delphos call costs ~5 minutes and redrawing a figure
must not cost anything. Anything an evaluation needs has to be *in* the npz.

---

## Scripts

| file | reads | writes | cost |
|---|---|---|---|
| `sdc/detect/compare_spikes.py` | an EDF (via the trials JSON) | `runs/<id>.npz`, `figures/real/<id>/compare_raster.png` | ~5 min (Delphos), seconds if cached |
| `sdc/compare/evaluate_detectors.py` | an npz | 4 × `figures/<real\|sim>/<run>/eval_*.png` | seconds |
| `sdc/compare/spike_statistics.py` | an npz | 3 × `figures/<real\|sim>/<run>/eval_*.png` | seconds |
| `sdc/compare/stim_effect.py` | a **stim** npz | `figures/real/<run>/stim_effect.png` | seconds |
| `sdc/compare/stim_artefact_check.py` | a **stim** npz + its EDF | `figures/real/<run>/stim_artefact_check.png` | ~1 min |
| `sdc/tools/block_size_test.py` | an npz + its EDF | `figures/real/<run>/block_size_test.png` | ~10 min (MATLAB x4) |
| `sdc/compare/compare_recordings.py` | all of `runs/*.npz` | `figures/real/compare_recordings.png` | seconds |
| `sdc/tools/polyspike_review.py` | the ARCHIVED 20 ms npz + the EDF | 6 × `figures/real/polyspike_review/*.png` | ~1 min |
| `sdc/detect/sim_data.py` | `sim_noise_model.mat` | `sim_data/*.edf` + `.truth.npz`, preview figures | ~1 min, 38 MB/level |
| `sdc/detect/run_sim_suite.py` | — | `sim_runs/*.npz` (drives `compare_spikes.py` per job) | ~40 min uncached |
| `sdc/scoring/score_sim_detectors.py` | `sim_runs/*.npz` | 6 × `figures/sim/_summary/sim_*.png` | seconds |
| `sdc/common/spike_match.py` | — | — | the one greedy matcher, shared by agreement *and* accuracy |
| `sdc/common/cond.py` | — | — | stim ON/OFF subsetting + the gap-aware time base it forces |
| `janca_detect_spikes.py`, `delphos_detect_spikes.py` | — | — | detector wrappers |
| `tests/test_*.py` | — | — | plain scripts, no pytest; `<10 s` |

Run everything **from the repo root** as a module: `.venv\Scripts\python.exe -m sdc.<group>.<name>`.
Layout: `sdc/detect` produces detections, `sdc/compare` compares detectors against each
other, `sdc/scoring` scores them against ground truth, `sdc/tools` are one-off
investigations, `sdc/common` is shared. Data directories resolve through
`sdc/common/paths.py`, never from a module's own location.
The evaluation scripts take an optional npz path: `python -m sdc.compare.evaluate_detectors sim_runs/<x>.npz`
sends the figures to `figures/sim/<run>/` instead of `figures/real/<run>/`.

`compare_spikes.py` picks its recording from `$RECORDING` (`P1_pre`, `P1_stim`, …), one npz each.

### Restricting to stim ON or OFF

On an intermittent-stim recording, `$COND` restricts an evaluation to the stim-ON or stim-OFF
blocks. Figures gain an `_on` / `_off` suffix, so a split never overwrites the whole-window ones:

```
COND=on  python -m sdc.compare.spike_statistics   runs/P1_stim.npz
COND=off python -m sdc.compare.evaluate_detectors runs/P1_stim.npz
```

**This is not just a filter on the detections, and that is the whole reason `cond.py` exists.**
The ON subset is a handful of separated blocks, not a shorter recording, so three things go
wrong quietly — each still produces a number, just the wrong one:

| | what breaks | why it is not obvious |
|---|---|---|
| rate | the denominator is the ON seconds, not `seconds` | off by exactly the duty cycle — 3.1× on P1 |
| ISI | an interval from the last spike before an OFF block to the first one after it | it is *large*, so it lands in the tail and inflates the CV instead of looking wrong |
| any binned estimate | a bin straddling a boundary is part ON, part OFF | it is systematically **low**, which reads as a real drop in rate |

So `cond.Selection` works in **segments**: `isis()` never diffs across a boundary, and `bins()`
only returns bins lying *entirely* inside one segment (a bin that would be cut short is dropped,
not shortened, so every bin is still the same amount of time). The boundaries come from the
`on_runs` key, written by `compare_spikes.py`; a run made before that key existed raises rather
than guessing. There is deliberately **no "concatenate the ON blocks" mode** — it would make all
of the above work by construction, and every interval it invented would be a lie about a gap
that really happened.

`COND=on` on a baseline recording is an error, not an empty plot.

---

## Config that matters, and why

### Preprocessing: all three detectors on one signal (current canonical config)

`MED_KERNEL=5` + anti-alias + decimate to **`DETECT_FS=1000` Hz**, written to an EDF that
Delphos reads (`PREP_DELPHOS=1`). Before this, the three arms saw three *different* signals:

| config | Janca | Barkmeier | Delphos | counts (P1_pre, 600 s) |
|---|---|---|---|---|
| old baseline | 400 Hz | 400 Hz | **2 kHz, raw file** | 15934 / 11891 / 16199 |
| median off, preprocessed | 1 kHz | 1 kHz | 1 kHz | 17749 / 11941 / 15552 |
| **median 5, preprocessed** | 1 kHz | 1 kHz | 1 kHz | **17270 / 12006 / 13310** |

**The old "400 Hz" applied to Janca and Barkmeier only.** Delphos is a compiled binary that
opens the EDF itself, so it read the raw 2 kHz file with nothing done to it -- full 8-512 Hz
band while the other two were capped at a 200 Hz Nyquist. Its 16199 -> 15552 is therefore *not*
a rate increase; it is losing everything above 500 Hz **and** reading our montage instead of
building its own. That this cost only 4% says little of its detection mass lived above 500 Hz.

Note `med_kernel=1` disables **only** the median -- the anti-alias FIR and the downsample still
run, so "400 Hz, median off" was genuinely 400 Hz.

**What the preprocessing does, measured** (`sdc/tools/lost_to_median.py`, `run_delta.py`):

| change | Janca | Barkmeier | Delphos |
|---|---|---|---|
| 400 -> 1000 Hz (median off both) | **+11.4%** | +0.4% | n/a (2 kHz -> 1 kHz) |
| median 1 -> 5 (all at 1 kHz) | -2.7% | +0.5% | **-14.4%** |

The median hits Delphos ~5x harder than anything else, and that is not a coincidence: a 2.5 ms
median is a **sharpness remover** and sharpness is what Delphos detects. Of the 4288 detections
it drops, **85% are Delphos-only marks on a ~1 ms, 3.4-MAD impulse** sitting on an otherwise
unremarkable background -- recording glitches. The remaining 15% are real, corroborated
spike-and-slow-waves that *also* carried such an impulse: Delphos was triggered by the glitch
riding on the discharge, so removing it loses the whole event. Net: ~3655 glitch detections
removed for ~630 real ones, so the filter stays on. At kept detections the filter removes
0.67 MAD -- identical to random time points, i.e. nothing. **A median(5) at 2 kHz cannot touch
anything wider than ~1.25 ms, and an IED is 20-70 ms, so no real discharge can be erased by
it** -- only a sharp artefact riding on one.

Janca's +11.4% is the opposite case and looks like genuine sensitivity: those 2608 additions
are sharp-transient-plus-slow-wave, corroborated at 32.5% against its own 69.5% baseline (vs
14.8% for Delphos's glitches). At 1 kHz a 20 ms transient is 20 samples rather than 8, so
envelope maxima that were smeared below threshold now cross it. The labelled BIDS data is what
settles whether they are real.


**`MERGE_MS = 100`** (`compare_spikes.py`) — the shared polyspike rule; marks closer than this
collapse to one, for all three detectors. This is the single most consequential setting.

- It exists because the detectors *count* differently: Delphos detects a TF **blob**, so a
  polyspike run inside one blob is one detection and it has no sub-blob events to expose,
  while Janča marks every component. At a 20 ms floor the fraction of inter-detection intervals
  under 50 ms was Janča **15.1%**, Barkmeier 2.8%, Delphos 1.6% — a counting convention, not a
  difference in what was found.
- Merging can only move Janča and Barkmeier *toward* Delphos; nothing moves Delphos the other
  way. So the comparison is at **event** level, not component level, by necessity.
- **100 ms is not settled for real data.** It was chosen as a working value under Janča's
  published 120 ms default. Work through `figures/real/polyspike_review/` before defending it.

**`JANCA_PT`** is *derived* from `MERGE_MS`, not set to `MERGE_MS/1000`. Janča's internal union
is a morphological closing that merges `d <= L` and rounds odd, so the naive value gives a
25 ms floor when 20 ms was asked for. `_janca_pt()` compensates. Verify with the minimum
inter-detection interval in the npz: **it must be identical for all three detectors.**

**`BARK["TAMP"] = 1200` is NOT in µV.** `mDetectSpike.m:284` rescales each block by
`100 / median(mean(abs(EEG)))` first. Measured on real P1 that factor is **7.02**, so
TAMP=1200 means ≈**171 µV** of summed half-heights — which sits right on the real median spike
(197 µV peak-to-peak). Do not move it for a simulation experiment; sim-specific operating
points belong in `run_sim_suite.SWEEPS`.

**`DETECT_FS = 400`** — Janča and Barkmeier run here; Delphos runs on the raw 2 kHz file
because it is a compiled binary that reads the file itself. **`TOL_MS = 50`** is the agreement
tolerance. **`DELPHOS["pin_free_ram_gb"] = 12`** pins Delphos's internal tiling, which is
RAM-dependent and therefore not bit-reproducible between operating points.

---

## Outputs

One **folder per run**, routed by whether the npz says it is simulated:

```
figures/real/P1_pre/            per-recording evaluation (compare_raster + 7 eval_*)
figures/real/polyspike_review/  merge-cutoff review, from the archived 20 ms detections
figures/real/_sweeps/           historical parameter sweeps
figures/sim/_summary/           aggregates ACROSS sim runs (sim_*)
figures/sim/<sim run>/          per-SNR-level evaluation
```

Filenames are plain inside each folder -- the folder carries the identity, so `P5_stim`
cannot overwrite `P1_pre`. Routing reads the `simulated` key from the npz rather than guessing
from the path.

| figure | answers |
|---|---|
| `compare_raster.png` | where each detector fired, population rate over time |
| `eval_rate_scatter.png` | per-channel rate, detector vs detector, against y=x |
| `eval_rank_scatter.png` | is the spikiest channel the same channel? |
| `eval_reliability.png` | self-consistency, and observed agreement against its ceiling |
| `eval_binned_rates.png` | is a channel's rate steady or driven by one burst? |
| `eval_spike_structure.png` | ISI distribution + Fano vs timescale (`_LINEAR` = linear x-axis) |
| `eval_block_stability.png` | does the detector **track** the recording minute to minute? |
| `eval_bin_width.png` | how long must a bin be for a usable rate estimate? |
| `stim_effect.png` | what each detector says stimulation did — paired ON vs OFF per channel, effect size, and whether every ON block dips |
| `stim_artefact_check.png` | rough triage: does each detector's ON/OFF ratio track how much stim power that channel picked up? |
| `polyspike_*.png` | real polyspike candidates by inter-mark gap, for choosing `MERGE_MS` |
| `sim_metrics_vs_snr.png` | recall / precision / F1 / FP-rate vs SNR against ground truth |
| `sim_sweep_curves.png` | does each Barkmeier knob do anything? |
| `sim_per_channel.png` | per-channel performance vs channel noise |
| `sim_preview_snr*.png` | what the simulated data actually looks like |
| `sim_inband_snr.png` | what each detector's passband sees of the template |

---

## Findings

All on P1 baseline (`runs/P1_pre.npz`), 600 s, 226 bipolar channels, unless stated.
Counts: Janča 15934 | Barkmeier 11891 | Delphos 16199.

### Now measured on WHOLE recordings (652 / 1097 / 2107 / 3742 s)

All four run end to end via `sdc/detect/run_windows.py`. Janca and Barkmeier run per 240-300 s
window; Delphos gets the assembled file in one pass. Counts:

| | seconds | Janca | Barkmeier | Delphos |
|---|---|---|---|---|
| P1_pre | 652 | 18503 | 13159 | 14079 |
| P1_stim | 1097 | 25611 | 17266 | 31811 |
| P5_pre | 2107 | 31375 | 19940 | 24973 |
| P5_stim | 3742 | 64773 | 42048 | 39542 |

**All four findings hold in all six columns** (`figures/real/compare_recordings.png`):

| finding | range across the six columns |
|---|---|
| 1 — Janca-Delphos leads | **0.75-0.90**, at 83-96% of ceiling. Barkmeier pairs never exceed 48% |
| 2 — disagreement is systematic | self-rho 0.78-0.99, far above any cross-detector pairing. *Which* detector is most self-consistent is still recording-specific: Barkmeier on P1_pre, Janca on P5 |
| 3 — Barkmeier tracks activity least | **lowest in all six**: 4.3-6.5x Poisson vs Janca 6.7-12.7x and Delphos 7.4-24.3x. The two exceptions seen on 600 s slices were small-sample artefacts -- full files give 10-62 blocks instead of 4 |
| 4 — Barkmeier marks late | Janca-Barkmeier 13-16 ms vs Janca-Delphos 4-5 ms, everywhere. At 400 Hz this read as exactly 15.0 ms because one sample WAS 2.5 ms; at 1 kHz the spread is visible and the effect is unchanged |

**Four recordings now exist**: `P1_pre`, `P1_stim`, `P5_pre`, `P5_stim` — two patients, each
with a 600 s baseline and a 600 s ANT 145 Hz intermittent-stim file. P5 has 183 bipolar
channels against P1's 226, and no channel correspondence between them, so cross-patient
comparison is on summary statistics only, never per channel.

### What replicated on the second patient

| finding | P1 | P5 | verdict |
|---|---|---|---|
| 1 — Janča–Delphos rank agreement leads | ρ 0.737 vs 0.403 / 0.146 | ρ **0.831** vs **−0.036** / **−0.142** | **holds, and harder.** Janča–Delphos is stable at 0.83–0.87 across every recording and condition. Barkmeier on P5 has *no* relationship with either — and 0/10 top-10 overlap, with self-ρ 0.919, so it is reliably ranking channels, just ranking different ones |
| 3 — Barkmeier tracks activity least | CV 0.129 vs 0.189 / 0.371 | CV **0.154** vs 0.301 / 0.417 | **holds, and more clearly.** On P5 the per-minute counts show Janča and Delphos swinging together from ~300 to ~900 while Barkmeier sits inside 371–653 throughout |
| 8 — stimulation suppresses spikes | 0.57 / 0.31 / 0.90 | **1.08 / 1.01 / 1.21** | **does not replicate.** P5 shows no effect at all. Finding 8 is a property of the P1 recording, not of stimulation or of the detectors |

Finding 8's collapse is the important one, and it is consistent with finding 9: P5's residual
145 Hz contamination is *higher* than P1's (median +3.11 log2 against +0.92) yet its broadband
correlations are all near zero (−0.01 / −0.12 / −0.13, against P1's +0.34 / −0.19 / +0.34).
Whatever produced P1's ON/OFF split, more artefact is not sufficient to reproduce it.

Q1b under stim-ON now runs on P5 (two ~129 s blocks give the four 60 s bins P1's three ~65 s
blocks could not), but four bins make a CV estimate too noisy to rank detectors — reported, not
relied on.

**These numbers post-date the preprocessing fixes** (native-rate QC, `med_kernel=1`,
`fill_bad_samples=False`). Earlier figures in the history used 400 Hz QC with the median filter
on and Barkmeier's input AR-filled; the staged deltas were Janča 15918→15934, Barkmeier
12777→11891, Delphos 17193→16199. Anything quoting the old triple is pre-fix.

They also post-date the **analysable-time denominator**: a per-channel rate divides by that
channel's unmasked seconds (`clean_sec_*`), not by the window length. On the baseline this moves
almost nothing (1 channel of 226 is fully masked, ρ 0.740→0.737); on a stim recording it is the
difference between a real result and an artefact, see below.

| # | finding | evidence |
|---|---|---|
| 0 | **Per-channel MEAN rate confounds pathology with implant coverage** | P5/P1 baseline rate is 0.67 / 0.91 / 0.65 over all channels but **0.30 / 0.41 / 0.34** over the EZ — the whole-implant average was *masking* a real patient difference, and for Barkmeier hid it completely (0.91 reads as "the same patient"). Use median ± MAD over a channel set defined independently of the detectors — `rate_comparators.png` |
| 1 | Janča–Delphos rank agreement is high; Barkmeier agrees with neither | ρ **0.737** / 0.403 / 0.146 — `eval_rank_scatter.png`. Holds in **both** stim conditions: OFF 0.826/0.497/0.365, ON 0.634/0.337/0.069 |
| 2 | Barkmeier is the most *self*-consistent yet agrees least, so its disagreement is systematic, not noise | self-ρ **0.952** vs 0.907 / 0.837 — `eval_reliability.png` |
| 3 | **Barkmeier tracks activity least** | per-minute CV **0.129** (4.4× Poisson) vs Janča 0.189 (7.5×) and Delphos 0.371 (14.9×) — `eval_block_stability.png`. Weaker than the pre-fix numbers (0.065 / 0.197 / 0.408) and **not yet reproduced on a second recording**: in P1_stim's OFF blocks Barkmeier is 0.143 against Janča's 0.141, though from only 4 blocks |
| 8 | **Stimulation suppresses spikes, and all three detectors now agree** | ON/OFF rate ratio over analysable time, whole recordings: P1 0.39/0.30/0.27, P5 0.57/0.56/0.54 — `stim_effect.png`. On the 600 s slices with the old preprocessing this read as a *disagreement about the sign* (P1 0.57/0.31/**0.90**, P5 1.08/1.01/1.21). Two things changed together, so attribute with care: full recordings give 6 and 11 ON blocks against 3 and 2, AND all three now share a median-filtered input |
| 9 | **Delphos's apparent immunity to stimulation was stim artefact** | Its P1 ON/OFF ratio moved **0.90 → 0.27** when the median filter reached its input. The chain is established, not inferred: stimulation produces sharp sub-millisecond impulses; Delphos fires on them because it detects time-frequency sharpness; a 2.5 ms median removes them; Delphos stops firing and its ON count collapses to match the other two. **The detector that looked immune to stimulation was mostly counting stim glitches.** Mechanism measured independently in `lost_to_median.py` — of the detections the median removes, 85% sit on a ~1 ms, 3.4-MAD impulse over an unremarkable background, against 0.67 MAD (i.e. nothing) at the detections it keeps. Caveat: P1's Delphos Wilcoxon p is 0.75 (median channel 0.49, 155/223 down), so the pooled 0.27 is carried by busy channels rather than being a broad per-channel effect; P5's is 6e-12 |
| 10 | **Barkmeier is selective, not insensitive** | It detects the least of the three everywhere, but its detections are far more concentrated on the clinically defined EZ. EZ median ÷ all-channel median: **8.4** (P1) and **3.8** (P5), against 3.5 / 1.5 for Janča and 2.6 / 1.4 for Delphos — 2.4–3.2× the others, consistent across both patients — `rate_comparators.png`. Its low count is a property of *what* it fires on, not uniform under-detection. This is the first result here that favours Barkmeier, and it is the one that matters for localisation. **Not yet adjudicated**: EZ concentration is measured against a clinical label, so it says Barkmeier agrees with the clinicians more, not that it is more accurate. The labelled BIDS data (§`score_labelled.py`) is what separates selectivity from missed spikes |
| 4 | **Barkmeier marks ~40 ms late** | sim +39→+42 ms, stable across SNR; real median \|Δt\| 15 ms for both Barkmeier pairs vs 5 ms Janča–Delphos |
| 5 | Delphos merges polyspikes by construction | sub-50 ms ISI 15.1% / 2.8% / 1.6% at a 20 ms floor (see `archive/detections_merge20.npz`) |
| 6 | On synthetic ground truth at event level, Janča and Delphos are close; Barkmeier's deficit is recall, not false positives | at 150 ms merge, SNR 12: 0.949/0.961, 0.931/0.995, **0.551/1.000** |
| 7 | Real-data Fano peaks (~3.5) are physiology, not counting policy | ground truth measures Fano ≈ 1 at every width and all three detectors reproduce it |

### Finding 3 is now causal, not correlational — and the direction was the surprise

`block_size_test.py` re-runs Barkmeier across `block_size_min` and measures CV in FIXED 60 s
bins, so the y-axis means the same thing at every x.

| block_size_min | P1_pre total | CV/Poisson | P1_stim total | CV/Poisson | ON/OFF |
|---|---|---|---|---|---|
| 0.25 | 11944 | **2.7×** | 9952 | 10.2× | **0.47** |
| 0.5 | 12175 | 3.8× | 10065 | 10.8× | 0.36 |
| 1 (default) | 11891 | 4.4× | 9405 | 13.3× | 0.25 |
| 2 | 11807 | **6.5×** | 8853 | **17.6×** | **0.22** |

**The prediction recorded before running was that a SHORTER block would make Barkmeier track
MORE. That was backwards.** CV rises monotonically *with* block size, on both recordings, while
the total count barely moves (±3% on the baseline; ±12% on stim, so the control is weaker
there). The correct reading: a short block adapts its threshold to that block's own activity —
a busy 15 s window raises its own bar and loses detections, a quiet one lowers it and gains
them — which **equalises** counts across blocks. A long block shares one threshold across more
time, so variation inside it is never compensated and survives into the counts.

So the per-block threshold is a flattener and shorter blocks flatten harder. Two consequences:

- **Finding 3 is confirmed causally.** Changing one normalisation window moves Barkmeier from
  2.7× to 6.5× Poisson, covering most of the distance to Janča's 7.5×. Its flatness is the
  block size, not what it detects — and that makes it fixable, not just diagnosable.
- **Finding 8's Barkmeier number is largely a block-size artefact.** The stim ON/OFF ratio
  moves from 0.22 to 0.47 — more than 2× — purely by changing that window. The 0.31 reported
  above is not a measurement of what stimulation did to spikes.

Caveat on provenance: every non-default point depends on the stride bugfix below being correct.
It is verified by the exact-reproduction guard at `block_size_min=1` (11891 on P1_pre, 9405 on
P1_stim) plus reasoning, not by an independent implementation.

**Finding 9's Barkmeier sign is finding 3's mechanism, arriving independently.** `mDetectSpike.m:291`
computes `thresh = -mean(|fEEG|) - 4*std(|fEEG|)` from each block's own data, so a channel with
more broadband power during stim raises its own bar and loses detections — which is exactly the
negative correlation measured. The same mechanism was inferred on the baseline from a completely
different axis (per-minute count stability), and on the simulation from a rate/recall correlation.

**Findings 3 and 4 both trace to `mDetectSpike.m`:**

- `:291` — `thresh = -mean(|fEEG|) - 4*std(|fEEG|)`, computed **per 1-minute block from that
  block's own data**. More spikes → higher bar. Measured on the sim: the threshold rises
  **2.65×** from the 2/min channel to the 30/min channel with identical spikes on both, and
  per-channel recall correlates **−0.959** with rate (noise controlled) against −0.44 and −0.25
  for the others. This also explains why no Barkmeier knob helps — `TAMP`, `LD`, `RD`, `LS`,
  `RS` are all applied *downstream* of a peak that never crossed threshold.
- `:300` — the reported time is `spikeI = max(EEG[newPeakI-20ms : newPeakI])`, a **20 ms
  look-back** from the 20–50 Hz negative lobe. Once that lobe sits more than 20 ms after the
  true peak, the peak is unreachable. (`:332` assembles the index; the look-back is `:300`.)

**Which file.** Line numbers above are the reference original,
`seeg_analysis/shared_utils/mDetectSpike.m`. What actually executes is
`python_pipeline/matlab/mDetectSpike_coeffs.m`, a derived variant where the same statements are
at `:97`, `:104` and `:113`. Both were checked; they agree.

**`block_size_min` was broken for every value except the default**, in the original and in the
derived copy alike. The block *stride* was hardcoded `(CurrentBlock-1)*60*Fs` while the block
*length* is `BlockSize*60*Fs` and the spike-index offset is `(CurrentBlock-1)*BlockSize*60*Fs`.
The three only agree at `BlockSize == 1`. Below 1 the last blocks index past the end of the
recording and MATLAB throws; above 1 the blocks silently **overlap**, the tail of the recording
is never reached, and the reported indices are offset as if the stride had scaled. Fixed in
`mDetectSpike_coeffs.m` by scaling the stride; `block_size_test.py` requires `block_size_min=1`
to reproduce the stored count exactly, which is the proof the fix is a no-op at the default.
`shared_utils/mDetectSpike.m` still has it.

---

## Corrections — things that were wrong, so they are not re-derived

- **The simulated template had a step discontinuity.** It was truncated at −0.19 of peak, so
  every injected spike ended in a sharp edge that Delphos correctly detected. This made Delphos
  look imprecise. Fixing it moved its SNR-12 precision **0.810 → 0.997** and reversed the
  conclusion about which detector is cleanest.
- **`TAMP` is not in µV** (see above). Two measurements that looked contradictory were in
  different units.
- **The +40 ms lag is not `trough_search_ms`.** `mDetectSpike.m:332` returns a *positive* peak
  from a 20 ms look-back; `trough_search_ms` only feeds `Lamp`/`Ramp`/`Ldur`/`Rdur`.
- **The stim disagreement is not Delphos-specific.** The first reading of finding 8 was that
  Delphos's flat ON/OFF ratio came from 145 Hz artefact surviving into its 8–512 Hz band while
  the other two never saw it. Measured, that is wrong in two ways: Janča tracks contamination
  just as strongly (+0.34 broadband, against Delphos's +0.34), and Barkmeier tracks it in the
  *opposite* direction. The stim spectrum is lifted at **every** frequency, not only at the
  145/290/435/580 Hz harmonics, so a narrowband story was never going to be the whole one.
- **There is no Fano "cliff at 60 s".** The raw curve turns down at 10–20 s, and that is an
  estimator artefact — a rate-matched Poisson control measures 0.80 at 10 bins and 0.70 at 5.
  Bias-corrected, Barkmeier matches Janča to 40 s. The strong evidence for its flat activity
  tracking is `eval_block_stability.png`, not the Fano tail.
- **Window-based specificity and ROC-AUC saturate** — the imbalance is ~39:1 at 100 ms bins, so
  everything scores ≥0.97 and ~0.99 regardless of quality. The event-based table is the
  headline; the window accounting exists only because specificity needs a defined true negative.

---

## Known limits and open items

- **`MERGE_MS` is unsettled for real data.** See `polyspike_*.png`. The sim currently runs at
  150 ms while the real data is at 100, so **the sim is not a matched calibration until it is
  re-run at the same value.**
- **Barkmeier's simulated numbers are morphology-dependent.** Its timing reference moved 55 ms
  when only the template's after-wave changed, and every shape knob is inert on a smooth
  Gaussian. Treat its sim results as statements about waveform matching, not clinical
  sensitivity.
- **The `block_size_min` test has not been run.** Finding 3's mechanism is supported by a
  correlational figure; changing Barkmeier's block size is the falsifying experiment.
- **FIXED: the two input asymmetries.** `fill_bad_samples=False` and `med_kernel=1` now give
  all three detectors the same array. The AR fill had been inflating Barkmeier by ~6% (12646 →
  11891); the median filter had been manufacturing ~2300 Janča detections inside artefact.
- **FIXED: QC now runs at the native 2 kHz.** The 400 Hz QC was *under*-masking, and not for
  the reason first assumed: `gradThr` is per-sample, so for content below the new Nyquist both
  the threshold and `max|diff|` scale identically and the rate cancels. What does not cancel is
  that decimation *deletes* everything above 200 Hz — exactly the sharp artefact the rule
  exists to catch. Delphos moved most (mask attrition 6% → 13%) because it detects in 8–512 Hz.
- **Stim-ON is measured on 132 of 226 channels.** The artefact mask leaves **94 channels with
  zero analysable time** during P1's stim blocks (whole shafts, B among them). They are excluded
  rather than counted as 0 Hz, so any ON/OFF comparison must be restricted to the channels
  measurable in *both* conditions — comparing ON's 132 against OFF's 224 is not like for like.
- **Q1b cannot run on P1's stim-ON.** It needs ≥4 blocks of 60 s and the ON condition is three
  ~65 s segments, so `eval_block_stability_on.png` is not drawn and finding 3's strongest
  intended test — does stim artefact inflate Barkmeier's per-block threshold further — is still
  unrun. A longer stim file, or a second patient, is the way to get it.
- **`polyspike_review.py` has no COND support.** It needs the archived 20 ms npz and a hardcoded
  baseline EDF path, so reviewing polyspikes under stimulation is a separate job.
- `archive/` holds superseded runs: `detections_merge20.npz` (the under-merged real detections
  that `polyspike_review.py` needs) and five earlier simulated generations.

---

## Requirements

- venv at `.venv`; `numpy`, `scipy`, `matplotlib`, `mne`, and `seeg-pipeline` (editable, from
  `python_pipeline`).
- **Barkmeier** needs a local MATLAB install with the engine. Without it that arm is skipped
  and the comparison degrades to two detectors.
- **Delphos** needs **MATLAB Runtime 9.13 (R2022b)** — *not* the 9.5/R2018b the bundled
  `readme.txt` claims. Failure signature is exit −1 with no output at all. Results are memoised
  in `.delphos_cache/` keyed on (resolved path, file size, window, parameters); a call is
  ~5 min, so treat that cache as valuable. It is **not bit-reproducible** — Delphos tiles
  internally based on free RAM, hence `pin_free_ram_gb`.
- `sim_data/` (~250 MB of synthetic EDFs) is gitignored and regenerable in ~1 min; the config
  hash in each filename means a changed generator can never silently reuse an old cache entry.
