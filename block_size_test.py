"""
block_size_test.py
------------------
The falsifying experiment for finding 3: is Barkmeier's behaviour caused by the per-block
threshold in `mDetectSpike.m:291`?

    .venv\\Scripts\\python.exe block_size_test.py runs/P1_pre.npz
    .venv\\Scripts\\python.exe block_size_test.py runs/P1_stim.npz

WHY THIS AND NOT MORE CORRELATION
  Three independent lines already point at the same mechanism, and all three are correlational:
    * baseline    -- Barkmeier's per-minute count CV is the lowest of the three (0.129 vs 0.189
                     and 0.371), i.e. it does not track the recording's own activity.
    * simulation  -- its threshold rises 2.65x from a 2/min channel to a 30/min channel with
                     IDENTICAL spikes on both, and per-channel recall correlates -0.959 with
                     rate once noise is controlled.
    * stimulation -- its ON/OFF rate ratio correlates -0.19 with that channel's extra broadband
                     power, while Janca and Delphos are both +0.34.
  Every one is consistent with `thresh = -mean(|fEEG|) - STDCoeff*std(|fEEG|)` computed from
  each block's OWN data, so a block containing more signal raises its own bar. None of them
  MANIPULATES it. `block_size_min` is exposed all the way through seeg.detect_spikes, so it can
  be, and the prediction is sharp enough to be wrong.

THE PREDICTION
  Shrinking the block makes the threshold LOCAL: a 10 s block containing a burst raises its bar
  for 10 s instead of 60. So as block_size_min falls:
    1. per-block count CV should RISE towards Janca's -- the detector starts tracking activity;
    2. on a stim recording, the ON/OFF ratio should FALL further -- each ON block now sets its
       own inflated threshold rather than sharing one with neighbouring OFF time;
    3. total count should move relatively little -- this is about WHERE detections fall in time,
       not how many there are.
  If (1) and (2) both sit flat across a 8x range of block size, the mechanism is wrong and
  findings 3 and 9 need rewriting. That is the point.

  CV here is always measured in FIXED 60 s bins regardless of block_size_min, so the y-axis
  means the same thing at every x. Binning at the block size would confound the manipulation
  with the measurement.

WHAT IT RUNS
  Barkmeier ONLY. Janca and Delphos are read from the existing npz as fixed reference lines --
  they have no block structure, so re-running them would burn ~5 min of Delphos for identical
  numbers. Preprocessing is reproduced from compare_spikes.py; the guard below checks that by
  requiring the block_size_min=1 point to match the stored Barkmeier count EXACTLY. If that
  check fails, the load paths have diverged and every number here is uninterpretable.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from seeg import (read_edf_header, load_edf_segment, derive_montage, apply_montage,
                  decimate_recording, windowed_artefact_detector, make_cfg_artefact,
                  detect_spikes as detect_barkmeier, dilate_mask, merge_close,
                  load_trials, get_patient, get_trial, resolve_file, detect_stim)
from seeg.spikes import FILTER_SPEC, STD_COEFF, TROUGH_SEARCH_MS
from seeg._style import RED, BLUE, MUTED, GRID, recessive

import cond

HERE = Path(__file__).resolve().parent
NPZ = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs" / "P1_pre.npz"
if not NPZ.is_file():
    raise SystemExit(f"{NPZ} not found -- run compare_spikes.py first.")

BLOCK_SIZES = [0.25, 0.5, 1.0, 2.0]     # minutes. 1.0 is seeg's default and the reproduction
                                        # check; 0.25 is 15 s, near the floor -- _run_matlab
                                        # refuses a recording shorter than half a block.
CV_BIN_SEC = 60.0                       # FIXED, independent of block_size_min (see above)

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}

# ---- these MUST track compare_spikes.py. The reproduction guard below is what enforces it. ----
BASE_DIR = Path(r"C:\Users\amoo0039\Documents\local")
META_PATH = BASE_DIR / "data_meta" / "stim_trials.json"
DETECT_FS = 400.0
MED_KERNEL = 1
FILL_BAD_SAMPLES = False
QC_NATIVE = True
DILATE_MS = TROUGH_SEARCH_MS + 20
MERGE_MS = 100.0
BARK = dict(LS=3.0, RS=3.0, TAMP=1200.0, LD=8, RD=8,
            std_coeff=STD_COEFF, trough_search_ms=TROUGH_SEARCH_MS, filter_spec=FILTER_SPEC)


# ----------------------------------------------------------------------
z = np.load(NPZ, allow_pickle=False)
names_ref = [str(s) for s in z["names"]]
fs_ref = float(z["fs"])
SECONDS = int(z["seconds"])
PATIENT = int(z["patient"])
CONDITION = str(z["condition"])
REC_ID = str(z["rec_id"])
BARK_REF = int(z["Barkmeier_idx"].size)
for k, v in (("med_kernel", MED_KERNEL), ("fill_bad_samples", int(FILL_BAD_SAMPLES)),
             ("qc_native", int(QC_NATIVE)), ("merge_ms", MERGE_MS)):
    if not np.isclose(float(z[k]), float(v)):
        raise SystemExit(f"{NPZ.name} was made with {k}={z[k]}, this script uses {v}. "
                         f"Re-run compare_spikes.py or fix the constant -- the block-size "
                         f"curve is only interpretable against a matching baseline.")

trials = load_trials(META_PATH)
pat = get_patient(trials, PATIENT)
trial = get_trial(pat, 1)
stem, TRIAL = resolve_file(trial, CONDITION)   # (stem, trial_or_None); None for 'pre' is the
                                               # pipeline-wide "no stim" signal
EDF = str(BASE_DIR / f"P{PATIENT}" / f"{stem}.edf")
print(f"--- block_size_min sweep on {REC_ID}: {Path(EDF).name}, {SECONDS}s ---")

hdr = read_edf_header(EDF)
FACTOR = int(round(hdr["SampleRate"] / DETECT_FS))
rec = load_edf_segment(EDF, hdr, start_rec=1, stop_rec=SECONDS)
rec = apply_montage(rec, derive_montage(rec["info"]["SelectedSignals"]))
if TRIAL is not None:
    rec["info"]["stim_trial"] = TRIAL
    rec = detect_stim(rec)
dec = decimate_recording(rec, factor=FACTOR, med_kernel=MED_KERNEL)
fs = dec["info"]["SampleRate"]
names = dec["info"]["SelectedSignals"]
n_chan = len(names)
if list(names) != names_ref:
    raise SystemExit("channel order differs from the npz -- refusing to compare.")

_qc_rec = ({"data": dec["raw"]["data"],
            "info": {**rec["info"], "SampleRate": dec["raw"]["fs"],
                     "NSamples": dec["raw"]["data"].shape[0]}} if QC_NATIVE else dec)
_qc_fs = _qc_rec["info"]["SampleRate"]
qc = windowed_artefact_detector(_qc_rec, make_cfg_artefact(hdr["lsb"], trial=TRIAL))


def _fold(mask):
    n_det = dec["data"].shape[0]
    if mask.shape[0] == n_det:
        return mask
    n_keep = (mask.shape[0] // FACTOR) * FACTOR
    out = mask[:n_keep].reshape(-1, FACTOR, mask.shape[1]).any(axis=1)
    if out.shape[0] < n_det:
        out = np.vstack([out, mask[n_keep:].any(axis=0, keepdims=True)])
    return out[:n_det]


dmask = _fold(dilate_mask(qc["sampleMask"], _qc_fs, DILATE_MS))
QC_DET = dict(qc)
QC_DET["sampleMask"] = _fold(qc["sampleMask"])
if QC_NATIVE and "epoch" in qc and "starts" in qc["epoch"]:
    _ep = dict(qc["epoch"])
    _ep["starts"] = (np.asarray(qc["epoch"]["starts"]) // FACTOR).astype(int)
    if "epochSamp" in _ep:
        _ep["epochSamp"] = int(round(_ep["epochSamp"] / FACTOR))
    QC_DET["epoch"] = _ep

ON_MASK = np.zeros(dec["data"].shape[0], bool)
if TRIAL is not None:
    _elen = int(round(qc["epoch"]["epochSamp"]))
    _scale = fs / _qc_fs
    for _s, _isOn in zip(qc["epoch"]["starts"], qc["epoch"]["isOn"]):
        if _isOn:
            _a = int(round(_s * _scale))
            ON_MASK[_a:_a + int(round(_elen * _scale))] = True
print(f"[qc] {dmask.mean():.2%} masked after dilation; "
      f"{ON_MASK.mean():.1%} of the window is stim-ON")


def finalise(per_chan):
    """compare_spikes._finalise, minus the printing: clip -> artefact mask -> merge."""
    gap = MERGE_MS / 1000.0 * fs
    out = []
    for c in range(n_chan):
        idx = np.unique(np.asarray(per_chan[c], int))
        idx = idx[(idx >= 0) & (idx < dmask.shape[0])]
        idx = idx[~dmask[idx, c]]
        out.append(merge_close(idx, gap))
    return out


# ----------------------------------------------------------------------
print(f"\n{'block_size_min':>15}{'total':>9}{'CV@60s':>9}{'CV/Poisson':>12}"
      + (f"{'ON/OFF':>9}" if TRIAL is not None else ""))
rows = []
for bs in BLOCK_SIZES:
    detect_barkmeier(dec, QC_DET, post_mask_spikes=False,
                     fill_bad_samples=FILL_BAD_SAMPLES,
                     det_thresholds=[BARK["LS"], BARK["RS"], BARK["TAMP"],
                                     BARK["LD"], BARK["RD"]],
                     std_coeff=BARK["std_coeff"], trough_search_ms=BARK["trough_search_ms"],
                     filter_spec=BARK["filter_spec"], block_size_min=bs, verbose=False)
    per = finalise([np.asarray(s, int) for s in dec["info"]["DetectedSpikes"]])
    allt = np.sort(np.concatenate([p for p in per if p.size]) / fs)
    edges = np.arange(int(SECONDS // CV_BIN_SEC) + 1) * CV_BIN_SEC
    cnt = np.histogram(allt, bins=edges)[0]
    cv = cnt.std(ddof=1) / cnt.mean()
    total = int(sum(p.size for p in per))
    row = dict(bs=bs, total=total, cv=float(cv), ratio=float(cv * np.sqrt(cnt.mean())))
    if TRIAL is not None:
        flat = np.concatenate([p for p in per if p.size])
        on = ON_MASK[flat]
        sec_on, sec_off = ON_MASK.sum() / fs, (~ON_MASK).sum() / fs
        row["on_off"] = float((on.sum() / sec_on) / ((~on).sum() / sec_off))
    rows.append(row)
    print(f"{bs:>15g}{total:>9}{cv:>9.3f}{row['ratio']:>12.1f}"
          + (f"{row['on_off']:>9.2f}" if TRIAL is not None else ""))

# THE GUARD. block_size_min=1.0 is seeg's default, so it must reproduce the stored run exactly.
# Anything else means this script's preprocessing has drifted from compare_spikes.py and the
# whole curve is measuring that drift instead of the block size.
_ref = next(r for r in rows if r["bs"] == 1.0)
if _ref["total"] != BARK_REF:
    raise SystemExit(f"REPRODUCTION FAILED: block_size_min=1.0 gives {_ref['total']} but "
                     f"{NPZ.name} stored {BARK_REF}. The preprocessing here has diverged from "
                     f"compare_spikes.py -- fix that before reading the curve.")
print(f"[ok] block_size_min=1.0 reproduces the stored Barkmeier count exactly ({BARK_REF})")

# reference lines from the existing run -- Janca and Delphos have no block structure
sel = cond.select(z, "all")
ref_cv = {}
for d in ("Janca", "Delphos"):
    if f"{d}_idx" not in z.files:
        continue
    t = np.sort(z[f"{d}_idx"] / fs_ref)
    edges = np.arange(int(SECONDS // CV_BIN_SEC) + 1) * CV_BIN_SEC
    c = np.histogram(t, bins=edges)[0]
    ref_cv[d] = float(c.std(ddof=1) / c.mean())

n_panel = 3 if TRIAL is not None else 2
fig, axes = plt.subplots(1, n_panel, figsize=(5.2 * n_panel, 4.3))
bs = [r["bs"] for r in rows]

ax = axes[0]
ax.plot(bs, [r["cv"] for r in rows], "-o", color=COLORS["Barkmeier"], lw=1.8, ms=7,
        label="Barkmeier")
for d, v in ref_cv.items():
    ax.axhline(v, color=COLORS[d], ls="--", lw=1.2, label=f"{d} (no block structure)")
ax.axvline(1.0, color=MUTED, ls=":", lw=1.0)
ax.annotate("seeg default", (1.0, ax.get_ylim()[1]), fontsize=7, color=MUTED, rotation=90,
            va="top", ha="right")
ax.set_xscale("log")
ax.set_xticks(bs)
ax.set_xticklabels([f"{b:g}" for b in bs])
ax.set_xlabel("block_size_min (minutes)")
ax.set_ylabel(f"CV of counts in fixed {CV_BIN_SEC:g}s bins")
ax.set_title("(a) does a shorter block make it TRACK?", fontsize=9, loc="left")
ax.legend(frameon=False, fontsize=8)
recessive(ax)

ax = axes[1]
ax.plot(bs, [r["total"] for r in rows], "-o", color=COLORS["Barkmeier"], lw=1.8, ms=7)
ax.axhline(BARK_REF, color=MUTED, ls=":", lw=1.0)
ax.set_xscale("log")
ax.set_xticks(bs)
ax.set_xticklabels([f"{b:g}" for b in bs])
ax.set_xlabel("block_size_min (minutes)")
ax.set_ylabel("total detections")
ax.set_title("(b) control: is it just detecting more?", fontsize=9, loc="left")
recessive(ax)

if TRIAL is not None:
    ax = axes[2]
    ax.plot(bs, [r["on_off"] for r in rows], "-o", color=COLORS["Barkmeier"], lw=1.8, ms=7)
    ax.axhline(1.0, color=MUTED, ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xticks(bs)
    ax.set_xticklabels([f"{b:g}" for b in bs])
    ax.set_xlabel("block_size_min (minutes)")
    ax.set_ylabel("stim ON / OFF rate")
    ax.set_title("(c) does a shorter block deepen the stim deficit?", fontsize=9, loc="left")
    recessive(ax)

fig.suptitle(f"Is Barkmeier's behaviour the per-block threshold? | {REC_ID}, {SECONDS}s, "
             f"{n_chan} channels  (Barkmeier re-run at each block size; others fixed)",
             fontsize=11)
fig.tight_layout()
OUT = HERE / "figures" / "real" / NPZ.stem
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "block_size_test.png", dpi=130)
print(f"\n[saved] {OUT / 'block_size_test.png'}")
plt.close(fig)
