"""
sdc.scoring.tune_marks
----------------------
Tune Janca against OUR OWN exhaustively marked blocks, on a MACRO objective.

    .venv\\Scripts\\python.exe -m sdc.scoring.tune_marks

WHY NOT THE EXISTING TUNING
  Both configurations in the project so far optimise a POOLED quantity: match pooled
  det/chan-min at 3.5, then read pooled recall. On P1's block 74% of the marks sit on six
  channels, so a pooled objective is a six-channel objective -- a detector can fire 7 det/min
  on every quiet channel in the implant and pay nothing for it. That is exactly the failure the
  marks exposed: on channels the rater viewed and found EMPTY, Janca fires 6.9 / 5.2 / 3.8
  det/chan-min on P1 / P5 / P8 against Barkmeier's 0.1 / 0.3 / 0.5.

  So the objective here is MACRO F1 -- per-channel F1, averaged over channels, every channel
  weighted alike. Empty channels enter it with F1 = 0 whenever the detector fires on them,
  which is what makes silence on an empty channel worth something.

WHY k3, WHICH HAS NEVER BEEN SWEPT
  Janca's threshold is

      prah = k1 * (ln_mode + ln_median) - k3 * (ln_mean - ln_mode)

  and every term is a lognormal fit of THAT CHANNEL's own envelope. It is purely relative with
  no absolute floor, which is why a quiet channel gets a threshold down in its own noise.

  `k1` scales the whole expression, so it moves every channel together and CANNOT change
  across-channel behaviour -- which is why every previous sweep, all of them on k1, left the
  quiet-channel problem exactly where it was. `k3` is the only term that responds to the SHAPE
  of the envelope distribution: on a busy channel ln_mean >> ln_mode so it lowers the threshold,
  on a quiet channel mean == mode so it does nothing. That is the one dial that can widen the
  busy/quiet contrast. Default 0.0, never moved.

VALIDATION
  Three patients, so leave-one-patient-out is possible and is the only number worth quoting:
  fit on two, report on the third. A macro-F1 read on the block it was fitted to is not a
  result.

  NO CEILING IS AVAILABLE. P1's two passes were marked at deliberately different
  sensitivities, so their 0.758 overlap at +-50 ms is not a reliability estimate and bounds
  nothing. Judge a tuning by held-out macro F1, not against that number.

WHAT IS DELIBERATELY NOT TUNED HERE
  The marked blocks are all STIM-FREE baselines. Threshold behaviour on quiet channels is a
  property of the detector, and tuning it on clean data keeps it uncontaminated by artefact.
  The P1 ANT 2 Hz marks answer a different question -- whether the channels that become busy
  under stimulation contain real spikes -- and must not enter this objective, or thresholds get
  fitted to artefact.
"""
import itertools
import json

import numpy as np

from seeg import read_edf_header, load_edf_segment, derive_montage, apply_montage
from seeg.preprocess import decimate_recording

from sdc.common.paths import figdir, ROOT
from sdc.common.spike_match import match
from sdc.detect.janca_detect_spikes import detect_spikes as janca
from sdc.detect.recordings import edf_path
from sdc.scoring.score_marks import LABELS, _blocks, _marks, done_blocks, RATERS, TOL

DETECT_FS = 1000.0
MED_KERNEL = 5
# block -> the recording id that edf_path resolves. All stim-free.
BLOCK_REC = {"sub-P1_task-baseline_grp-sample_win-000": "P1_pre",
             "sub-P5_task-baseline_grp-sample_win-002": "P5_pre",
             "sub-P8_task-baseline_grp-sample_win-002": "P8_ANT145_pre"}


def load_block(block):
    """Decimated array + the channels the viewer showed + the rater's marks, for one block."""
    meta = _blocks()[block]
    t0, dur = float(meta["t_start"]), float(meta["t_dur"])
    edf, _ = edf_path(BLOCK_REC[block])
    hdr = read_edf_header(edf)
    rec = load_edf_segment(edf, hdr, int(t0) + 1, int(t0 + dur))
    rec = apply_montage(rec, derive_montage(rec["info"]["SelectedSignals"]), verbose=False)
    fs0 = rec["info"]["SampleRate"]
    dec = decimate_recording(rec, factor=int(round(fs0 / DETECT_FS)), med_kernel=MED_KERNEL,
                             keep_raw=False)
    names = list(dec["info"]["SelectedSignals"])
    shown = [c["name"] for c in meta["channels"] if c["name"] in names]
    idx = [names.index(c) for c in shown]
    mk = _marks("rater-AM", block)
    truth = {c: np.asarray(mk.get(c, []), float) - t0 for c in shown}   # block-relative seconds
    x = np.asarray(dec["data"])[:, idx]
    # Barkmeier takes a RECORDING, not an array, and writes into info["DetectedSpikes"].
    sub = {"data": x, "info": {**dec["info"], "SelectedSignals": shown,
                               "NumSelectedSignals": len(shown), "NSamples": x.shape[0]}}
    return {"block": block, "subj": meta["subject"], "x": x, "rec": sub,
            "fs": float(dec["info"]["SampleRate"]), "chans": shown, "truth": truth,
            "mins": dur / 60.0}


def score(det_per_chan, truth, mins, tol=TOL):
    """Macro F1 over channels, plus the diagnostics the objective exists to move."""
    f1s, empty_fp, busy_r, quiet_r = [], [], [], []
    tp_all = fp_all = fn_all = 0
    for c, t in det_per_chan.items():
        m = truth[c]
        hit = int(match(t, m, tol)[1].sum()) if (t.size and m.size) else 0
        tp, fp, fn = hit, t.size - hit, m.size - hit
        tp_all += tp
        fp_all += fp
        fn_all += fn
        if m.size:
            f1s.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
            busy_r.append(t.size / mins)
        else:
            f1s.append(0.0 if t.size else 1.0)     # silence on an empty channel is correct
            empty_fp.append(t.size / mins)
            quiet_r.append(t.size / mins)
    return {"macro_f1": float(np.mean(f1s)),
            "pooled_f1": 0.0 if (2 * tp_all + fp_all + fn_all) == 0
            else 2 * tp_all / (2 * tp_all + fp_all + fn_all),
            "recall": tp_all / max(tp_all + fn_all, 1),
            "prec": tp_all / max(tp_all + fp_all, 1),
            "empty_fp_per_min": float(np.mean(empty_fp)) if empty_fp else np.nan,
            "quiet_busy": (float(np.mean(quiet_r)) / float(np.mean(busy_r))
                           if busy_r and np.mean(busy_r) > 0 else np.nan),
            "det_per_chan_min": (sum(t.size for t in det_per_chan.values())
                                 / (len(det_per_chan) * mins))}


# The three block-normalisation knobs are IMPORTED from seeg.spikes, not retyped.
# compare_spikes.BARK does the same and says why: "previously written as literals identical to
# the upstream values, so an upstream change would not have reached this comparison". Retyping
# them here reintroduced exactly that bug -- the literals said std_coeff=3 / trough=20 /
# band 10-60 against the real 4 / 40 / 20-50, so every Barkmeier sweep ran against a baseline
# nothing uses, and a ladder rung that merely restored the true default was reported as a
# +0.06 F1 improvement. compare_spikes itself is not importable here: it is a runnable script
# that loads EDFs and runs detectors at import time.
from seeg.spikes import STD_COEFF, TROUGH_SEARCH_MS, FILTER_SPEC   # noqa: E402

# LS/RS/TAMP/LD/RD are literals in compare_spikes.BARK too, so they are literals here.
BARK_FIXED = dict(LS=3.0, RS=3.0, TAMP=1200.0, LD=8, RD=8,
                  std_coeff=STD_COEFF, trough_search_ms=TROUGH_SEARCH_MS,
                  filter_spec=FILTER_SPEC)

# The Janca operating point, defined ONCE and imported by overnight / knob_range / sweep_rates.
# Three separate copies of the Barkmeier constants drifted apart in this repo before this was
# written down, so it lives here beside BARK_FIXED rather than being repeated per caller.
#
#   band_high=50   50 Hz mains sits inside the paper's 60 Hz upper edge on this hardware.
#                  Costs ~0.009 marked macro F1 against 60. This is a data property.
#   decimation=200 the paper's resampling, and NOT what the line-noise argument is about:
#                  at 200 Hz the Nyquist is 100 Hz, so a 10-50 band is untouched by it.
#                  Skipping it measured +0.022 marked macro F1 -- inside the noise band that
#                  every knob in this project sits in -- for 5x the compute, because the
#                  Chebyshev filtering, Hilbert envelope and per-segment MLE all run on five
#                  times the samples (393 s vs ~80 s per setting per file). Not worth it on a
#                  pipeline that must run over many patients and hours.
JANCA_FIXED = dict(k3=0.0, band_low=10.0, band_high=50.0, decimation=200.0)


_DELPHOS_EDF = {}


def _block_edf(d):
    """Write this block's decimated array to its own EDF, once, for Delphos.

    Delphos reads a FILE. It could be pointed at the existing full-recording prep EDF with
    start_sec/duration_sec, but that file was AR-FILLED against the production mask -- so
    Delphos would see a different signal from the one Janca and Barkmeier are scored on here.
    Writing the same array the other two receive keeps all three on identical input, which is
    the whole point of scoring them against one set of marks.
    """
    from sdc.detect.sim_data import write_edf
    if d["block"] not in _DELPHOS_EDF:
        out = ROOT / "prep_edf" / f"_marks_{d['block']}.edf"
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            write_edf(str(out), d["x"], list(d["chans"]), d["fs"])
        _DELPHOS_EDF[d["block"]] = str(out)
    return _DELPHOS_EDF[d["block"]]


def _detect(det, d, a, b):
    """Per-channel detection times in block-relative SECONDS, for either detector.

    `a`/`b` are the two swept values: (k1, k3) for Janca, (TAMP, unused) for Barkmeier.
    """
    if det == "janca":
        out, _disch, _info = janca(d["x"], d["fs"], k1=a, k3=b)
        ch = np.asarray(out["chan"], int)
        t = np.asarray(out["pos"], float)
        return {c: np.sort(t[ch == i]) for i, c in enumerate(d["chans"])}
    if det == "delphos":
        from sdc.detect.delphos_detect_spikes import detect_spikes as delph
        # bipolar=False: the written EDF is already bipolar pairs, and Delphos would otherwise
        # re-montage them and collapse the label match.
        per = delph(_block_edf(d), list(d["chans"]), d["fs"], start_sec=0.0,
                    duration_sec=-1.0, Spk_thr=a, bipolar=False, pin_free_ram_gb=12)
        return {c: np.sort(np.asarray(per[i], float) / d["fs"])
                for i, c in enumerate(d["chans"])}
    from seeg import detect_spikes as bark
    bark(d["rec"], None, post_mask_spikes=False, fill_bad_samples=False,
         det_thresholds=[BARK_FIXED["LS"], BARK_FIXED["RS"], a,
                         BARK_FIXED["LD"], BARK_FIXED["RD"]],
         std_coeff=BARK_FIXED["std_coeff"],
         trough_search_ms=BARK_FIXED["trough_search_ms"],
         filter_spec=BARK_FIXED["filter_spec"])
    per = d["rec"]["info"]["DetectedSpikes"]
    return {c: np.sort(np.asarray(per[i], float) / d["fs"]) for i, c in enumerate(d["chans"])}


def run(blocks=None, k1s=(3.0, 3.65, 4.5, 5.5), k3s=(0.0, 0.5, 1.0, 2.0), tol=TOL,
        det="janca"):
    if det == "barkmeier":
        k1s, k3s = (400.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0), (0.0,)
    elif det == "delphos":
        k1s, k3s = (15.0, 30.0, 50.0, 80.0, 120.0), (0.0,)
    blocks = blocks or [b for b in done_blocks(RATERS[0]) if b in BLOCK_REC]
    data = [load_block(b) for b in blocks]
    for d in data:
        print(f"  {d['block']}  {d['x'].shape[0] / d['fs']:.0f}s  {len(d['chans'])} channels, "
              f"{sum(v.size for v in d['truth'].values())} marks")

    rows = []
    for k1, k3 in itertools.product(k1s, k3s):
        per_block = {}
        for d in data:
            dets = _detect(det, d, k1, k3)
            per_block[d["subj"]] = score(dets, d["truth"], d["mins"], tol)
        rows.append({"k1": k1, "k3": k3, "by": per_block,
                     "macro_f1": float(np.mean([v["macro_f1"] for v in per_block.values()])),
                     "empty_fp": float(np.nanmean([v["empty_fp_per_min"]
                                                   for v in per_block.values()])),
                     "rate": float(np.mean([v["det_per_chan_min"] for v in per_block.values()]))})
    return rows, [d["subj"] for d in data]


if __name__ == "__main__":
    import sys
    _det = sys.argv[1] if len(sys.argv) > 1 else "janca"
    print(f"detector: {_det}")
    rows, subs = run(det=_det)
    print(f"\n{'k1':>5}{'k3':>5}{'macroF1':>9}{'pooledF1':>10}{'recall':>8}{'prec':>7}"
          f"{'emptyFP/min':>13}{'det/ch-min':>12}")
    for r in sorted(rows, key=lambda r: -r["macro_f1"]):
        p = r["by"]
        print(f"{r['k1']:>7.6g}{r['k3']:>5.2f}{r['macro_f1']:>9.3f}"
              f"{np.mean([v['pooled_f1'] for v in p.values()]):>10.3f}"
              f"{np.mean([v['recall'] for v in p.values()]):>8.3f}"
              f"{np.mean([v['prec'] for v in p.values()]):>7.3f}"
              f"{r['empty_fp']:>13.2f}{r['rate']:>12.2f}")

    print(f"\nLEAVE-ONE-PATIENT-OUT (fit macro F1 on two, report on the third)")
    print(f"  {'held out':<10}{'chosen k1/k3':>14}{'macroF1 held-out':>18}"
          f"{'default k1=3.65,k3=0':>22}")
    for s in subs:
        fit = [r for r in rows]
        best = max(fit, key=lambda r: np.mean([v["macro_f1"] for k, v in r["by"].items()
                                               if k != s]))
        dflt = [r for r in rows if r["k1"] == 3.65 and r["k3"] == 0.0]
        d0 = dflt[0]["by"][s]["macro_f1"] if dflt else float("nan")
        print(f"  {s:<10}{best['k1']:>7.2f}/{best['k3']:<6.2f}"
              f"{best['by'][s]['macro_f1']:>18.3f}{d0:>22.3f}")
