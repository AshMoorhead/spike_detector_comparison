"""
sdc.scoring.sweep_rates
-----------------------
Per-channel stim and baseline rates at every threshold setting, saved for later gating.

    .venv\\Scripts\\python.exe -m sdc.scoring.sweep_rates P1_ANT2 janca

WHY THIS EXISTS RATHER THAN A SCRIPT THAT PRINTS RATIOS
  The first version of this sweep stored only the summary -- median ratio over all channels and
  over the p67 gate. Asking a different question afterwards ("what about an ABSOLUTE rate gate
  at 10 det/chan-min?") then required re-running every setting, which for Delphos is ~12 min
  each. Saving the per-CHANNEL rates instead makes every gate a post-hoc numpy operation, and
  the gate is exactly the knob still under discussion.

WHY AN ABSOLUTE GATE IS THE RIGHT SHAPE
  A percentile keeps a third of whatever is there, however contaminated it is. Stimulation adds
  a roughly CONSTANT rate to a detector (measured: +1.86 det/chan-min for Delphos on P1 ANT
  2 Hz), so the inflation of a channel's ratio is `added / baseline` -- a property of that
  channel's absolute rate, not of its rank. On P1 ANT 2 Hz the p67 cut lands at 4.78 det/min,
  where the addition is still 39% of the signal; 10 det/min puts it under 20%. An absolute cut
  is also comparable across patients, which a percentile is not.

OUTPUT
  runs/sweeps/rates_<rec>_<det>.npz with `settings` (n_set, 2), `stim` and `base`
  (n_set, n_chan), and `names`. Rates are per CLEAN channel-minute, masked with the production
  QC mask at one-second resolution.
"""
import sys
import time

import numpy as np

from seeg import read_edf_header, load_edf_segment, derive_montage, apply_montage
from seeg.preprocess import decimate_recording
from seeg.spikes import DET_THRESHOLDS, STD_COEFF, TROUGH_SEARCH_MS, FILTER_SPEC

from sdc.common.paths import RUNS, ROOT
from sdc.detect.recordings import edf_path

GRIDS = {"janca": [(k1, k3) for k1 in (3.0, 3.65, 4.5, 5.5) for k3 in (0.0, 0.5, 1.0, 2.0)],
         "barkmeier": [(t, 0.0) for t in (400.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0)],
         "delphos": [(s, 0.0) for s in (15.0, 30.0, 50.0, 80.0, 120.0)]}
# recording key -> (stim rec id, pre rec id, stim run stem, pre run stem)
PAIRS = {"P1_ANT2": ("P1_ANT2_stim", "P1_ANT2_pre",
                     "P1_ANT2_stim_qcfinalv2", "P1_ANT2_pre_qcfinalv2"),
         "P1_ANT145": ("P1_stim", "P1_pre", "P1_stim_qcfinalv2", "P1_pre_qcfinalv2")}
# IMPORTED, never retyped. The previous literal here (std_coeff=3, trough=20, band 10-60)
# matched neither seeg.spikes (4 / 40 / 20-50) nor sdc's own BARK_FIXED, so the Barkmeier
# effect curves in the swing figures were computed at a configuration used nowhere else --
# and, worse, at a DIFFERENT one from the marks scoring those figures draw their admissible
# bars from. Taking BARK_FIXED wholesale keeps both halves of the swing figure on one config
# and keeps this repo on its deliberate matched-rate operating point (LS/RS/LD/RD 3/3/8/8,
# mirroring compare_spikes.BARK) rather than silently switching to the shipped 7/7/10/10.
from sdc.scoring.tune_marks import BARK_FIXED, JANCA_FIXED   # noqa: E402

BARK = {k: v for k, v in BARK_FIXED.items() if k != "TAMP"}   # TAMP is the swept axis
JANCA_CFG = {k: v for k, v in JANCA_FIXED.items() if k != "k3"}  # k3 is swept


def _prep_edf(run_stem):
    """The preprocessed EDF Delphos read for that run -- same signal the other two saw."""
    rec = run_stem.split("_qc")[0]
    prof = run_stem.split("_qc")[1]
    return ROOT / "prep_edf" / f"{rec}_full_med5_1000Hz_fill_qc{prof}.edf"


def _load(rec_id, run_stem):
    edf, _ = edf_path(rec_id)
    h = read_edf_header(edf)
    r = load_edf_segment(edf, h, 1, int(h["NumDataRecords"]))
    r = apply_montage(r, derive_montage(r["info"]["SelectedSignals"]), verbose=False)
    d = decimate_recording(r, factor=int(round(r["info"]["SampleRate"] / 1000.0)),
                           med_kernel=5, keep_raw=False)
    z = np.load(RUNS / f"{run_stem}.npz", allow_pickle=False)
    cps = np.asarray(z["clean_per_sec"], float)
    fs_run = float(z["fs"])
    return {"rec": d, "x": np.asarray(d["data"]), "fs": float(d["info"]["SampleRate"]),
            "names": [str(x) for x in z["names"]], "clean": cps > 0.5 * fs_run,
            "mins": cps.sum(axis=0) / fs_run / 60.0, "prep": str(_prep_edf(run_stem))}


def _rates(per_chan, P):
    c = np.zeros(len(P["mins"]))
    for i, idx in enumerate(per_chan):
        t = np.asarray(idx, float)
        if not t.size:
            continue
        sec = np.clip((t / P["fs"]).astype(int), 0, P["clean"].shape[0] - 1)
        c[i] = int(P["clean"][sec, i].sum())
    return np.where(P["mins"] > 0, c / np.maximum(P["mins"], 1e-9), np.nan)


def _detect(det, P, a):
    if det == "janca":
        from sdc.detect.janca_detect_spikes import detect_spikes as janca
        # From tune_marks.JANCA_FIXED, so the swing curves and the marks scoring that supplies
        # their admissible bars cannot run on different Jancas -- which is what happened while
        # this was left at the port's own defaults.
        out, _d, _i = janca(P["x"], P["fs"], k1=a[0], k3=a[1], **JANCA_CFG)
        ch = np.asarray(out["chan"], int)
        t = np.asarray(out["pos"], float) * P["fs"]
        return [t[ch == i] for i in range(len(P["names"]))]
    if det == "barkmeier":
        from seeg import detect_spikes as bark
        bark(P["rec"], None, post_mask_spikes=False, fill_bad_samples=False,
             det_thresholds=[BARK["LS"], BARK["RS"], a[0], BARK["LD"], BARK["RD"]],
             std_coeff=BARK["std_coeff"], trough_search_ms=BARK["trough_search_ms"],
             filter_spec=BARK["filter_spec"])
        return P["rec"]["info"]["DetectedSpikes"]
    from sdc.detect.delphos_detect_spikes import detect_spikes as delph
    # bipolar=False: the prep EDF is already bipolar pairs.
    return delph(P["prep"], P["names"], P["fs"], start_sec=0.0, duration_sec=-1.0,
                 Spk_thr=a[0], bipolar=False, pin_free_ram_gb=12)


def run(key, det):
    s_id, b_id, s_run, b_run = PAIRS[key]
    S, B = _load(s_id, s_run), _load(b_id, b_run)
    grid = GRIDS[det]
    stim = np.full((len(grid), len(S["names"])), np.nan)
    base = np.full((len(grid), len(B["names"])), np.nan)
    print(f"{key} / {det}: {len(grid)} settings   stim {S['x'].shape}  pre {B['x'].shape}")
    for i, a in enumerate(grid):
        t0 = time.time()
        stim[i] = _rates(_detect(det, S, a), S)
        base[i] = _rates(_detect(det, B, a), B)
        ok = np.isfinite(stim[i]) & np.isfinite(base[i]) & (base[i] > 0) & (stim[i] > 0)
        print(f"  {a[0]:>7g}/{a[1]:<4g} n={ok.sum():>4} "
              f"med ratio {np.median(stim[i][ok] / base[i][ok]):.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    out = RUNS / "sweeps" / f"rates_{key}_{det}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, settings=np.array(grid, float), stim=stim, base=base,
                        names=np.array(S["names"]))
    print(f"[saved] {out}")
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "P1_ANT2",
        sys.argv[2] if len(sys.argv) > 2 else "janca")
