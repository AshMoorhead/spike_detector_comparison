"""
sdc.detect.run_windows
----------------------
Run the comparison over a WHOLE recording, in RAM-sized overlapping windows.

    RECORDING=P1_pre .venv\\Scripts\\python.exe -m sdc.detect.run_windows

WHY WINDOWS
  The files are 652-3742 s. P5_stim alone is 12 GB at native rate before the montage copy, so
  a whole-file load is not an option. `seeg.edf.window_bounds` already solves this for the
  pipeline -- RAM-budgeted windows with one block of overlap, plus interior bounds for trimming
  that overlap back off -- so this reuses it rather than inventing a second scheme.

DIVISION OF LABOUR, and why it is this way round
  * Janca, Barkmeier and the QC run PER WINDOW, as subprocesses of compare_spikes.py. Each is
    given a record range and knows nothing about windowing; this driver owns all the overlap
    arithmetic. That keeps compare_spikes a plain "process these records" script, which is also
    what makes a single-window run and a windowed run the same code path.
  * DELPHOS RUNS ONCE, over the whole assembled file. It already tiles internally against free
    RAM, so per-window calls would add process overhead AND re-whiten its time-frequency plane
    far more often than it needs to -- its normalisation would end up depending on our window
    size, which is precisely the kind of hidden coupling this comparison exists to avoid.
    Each window therefore dumps its decimated array; this driver concatenates the interiors,
    writes ONE preprocessed EDF, and hands that to Delphos.

WHAT WINDOWING COSTS, stated rather than hidden
  Janca's per-channel background model and Barkmeier's per-block threshold are both computed
  over whatever span they are handed, so both now see a window rather than a whole recording.
  Delphos does not, by the arrangement above. This is a real asymmetry and it is the price of
  fitting 12 GB recordings in memory; it is recorded in the npz as `window_sec` so a run can
  never be compared against one made at a different window size without noticing.

  Detections are kept only from each window's INTERIOR, so the overlap never double-counts.
  MERGE_MS is then re-applied across the joins, because two detections either side of an
  interior boundary were merged by neither window.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from seeg import read_edf_header, load_trials, get_patient, get_trial, resolve_file, merge_close
from seeg.edf import window_bounds

from sdc.common.paths import ROOT, RUNS
from sdc.common.invariants import check_run

# Delphos is ~60% of a run's wall time and is nondeterministic at 5-10% from RAM tiling, so a
# threshold sweep is often better answered by the two DETERMINISTIC detectors first. This was
# honoured for the per-window jobs but not by the merge stage, which called Delphos regardless.
# Skipping it also skips the assembled EDF, which exists only to feed Delphos. The per-window
# Janca/Barkmeier results are cached either way, so adding Delphos afterwards re-runs only the
# merge, not the sweep.
MERGE_DELPHOS = os.environ.get("RUN_DELPHOS", "1") == "1"

RECORDING = os.environ.get("RECORDING", "P1_pre")
QC_PROFILE = os.environ.get("QC_PROFILE", "prod")
_QC_SUF = "" if QC_PROFILE == "prod" else f"_qc{QC_PROFILE}"
                      # The artefact profile has to reach every path this driver touches, for
                      # exactly the reason spelled out below for RUN_TAG. It is folded into
                      # this module's RUN_TAG rather than exported, because compare_spikes
                      # appends the same suffix itself from the inherited QC_PROFILE -- so the
                      # child and the parent independently build the identical filename, and
                      # adding it to the child's environment as well would double it.
# MED_KERNEL and FILL_ALL change the CONTENT of every window file and every decimated array,
# and compare_spikes already puts them in its own filename. This driver did not, so the child
# wrote `P1_stim_med1_nofill_qcnone_w001.npz` while the driver looked for
# `P1_stim_qcnone_w001.npz` -- and, worse, `dec_dir` was keyed on the QC profile alone, so a
# MEDIAN-FILTERED decimated array would have been silently reused for an unfiltered run. The
# order here mirrors compare_spikes' `_variant` exactly so the two names agree.
_MED = int(os.environ.get("MED_KERNEL", 5))
_FILL = os.environ.get("FILL_ALL", "1") == "1"
_PB = float(os.environ.get("PULSE_BLANK_MS", 0) or 0)
_PBF = {"auto": "", "interp": "i", "ar": "a"}[os.environ.get("PULSE_FILL", "auto")]
_BD = os.environ.get("BARK_DENOM", "") or None
_BD_SUF = ("" if not _BD else "_bdauto" if _BD == "auto" else f"_bd{float(_BD):g}")
_VAR_SUF = (("" if _MED == 5 else f"_med{_MED}") + ("" if _FILL else "_nofill")
            + ("" if not _PB else f"_pb{_PB:g}{_PBF}") + _BD_SUF)
RUN_TAG = os.environ.get("RUN_TAG", "") + _VAR_SUF + _QC_SUF
                      # Threaded through EVERY filename this driver reads or writes. It was
                      # not, and the failure was silent in the worst way: compare_spikes
                      # honoured the tag when writing per-window files, so a tuned run computed
                      # tuned windows -- and then merge_windows read the UNTAGGED (default)
                      # windows and wrote the untagged merged file. The output equalled the
                      # default exactly, which reads as "the tuning made no difference".
# DELPHOS_PIN_GB overrides the RAM balloon. 12 is the tuned default and MUST stay the default:
# Delphos's detections move with the pin, so two runs compared against each other have to share
# it. It is overridable because 12 starves Delphos outright on long recordings -- it exited 0
# without writing on the three files over ~4000 s (P5_ANT7, P5_Pulv7, P4_ANT7), which is an OOM.
# Raising it for those trials means re-running BOTH sides of their comparison at the new value,
# because all three are LF-continuous and are compared against their pre file. The value used is
# written into the merged npz as `delphos_pin_gb` so a mismatched pair is detectable afterwards
# rather than being invisible.
DELPHOS_CFG = dict(pin_free_ram_gb=int(os.environ.get("DELPHOS_PIN_GB", "12")),
                   Spk_thr=50, Spk_time_thr=1.25)
_tune = json.loads(os.environ.get("DET_TUNE", "") or "{}").get("delphos", {})
DELPHOS_CFG.update({k: v for k, v in _tune.items()})
                      # Delphos runs HERE, not in compare_spikes, so DET_TUNE has to be applied
                      # here too. Spk_thr was hardcoded to 50, which meant a tuned Delphos value
                      # was accepted, printed, and then ignored.
MEM_BUDGET = float(os.environ.get("MEM_BUDGET", 1e9))   # bytes; window_bounds' own default
BLOCK_SEC = int(os.environ.get("BLOCK_SEC", 60))        # the pipeline's block size, not the
                                                        # 120 s the interactive script uses
BASE_DIR = Path(r"C:\Users\amoo0039\Documents\local")
META_PATH = BASE_DIR / "data_meta" / "stim_trials.json"
from sdc.detect.recordings import RECORDINGS      # noqa: E402  -- one shared definition; this
                                                  # table used to be duplicated in both modules
                                                  # and they drifted the moment a recording was
                                                  # added to only one of them.


def plan(rec_id):
    """Every window for this recording: (start_rec, stop_rec, interior_t0, interior_t1) in
    1-based inclusive records and absolute seconds."""
    # One resolver for every caller. This used to resolve independently through the LOCAL
    # stim_trials.json by trial_index, which is wrong for the AES cohort: the two JSONs list
    # different trials in different orders, so P3's 7 Hz trial resolved a 145 Hz file, P8's ANT
    # trial resolved a Pulv one, and two baselines resolved to a bare ".edf" because the local
    # JSON leaves those fields empty. edf_path() keys on the FILENAME the cohort table carries.
    from sdc.detect.recordings import edf_path
    edf, _entry = edf_path(rec_id)
    hdr = read_edf_header(str(edf))
    fs = float(hdr["SampleRate"])
    total_sec = int(hdr["NumDataRecords"] * hdr["DataRecordDuration"])

    out, n = [], 1
    while True:
        try:
            w = window_bounds(hdr, n, mem_budget=MEM_BUDGET, block_sec=BLOCK_SEC)
        except ValueError:
            break                       # past the end of the recording
        start_rec = int(np.ceil(w["start"] / fs))
        stop_rec = min(int(np.floor(w["stop"] / fs)), total_sec)
        if stop_rec < start_rec:
            break
        # Interior, in ABSOLUTE seconds. window_bounds gives it relative to the window in
        # 1-based samples; everything downstream is easier in seconds from file start.
        t0 = (w["start"] - 1 + w["interior_start"] - 1) / fs
        t1 = (w["start"] - 1 + w["interior_stop"]) / fs
        if n == 1:
            t0 = 0.0
        out.append(dict(n=n, start_rec=start_rec, stop_rec=stop_rec,
                        t0=float(t0), t1=float(min(t1, total_sec))))
        if stop_rec >= total_sec:
            break
        n += 1
    out[-1]["t1"] = float(total_sec)     # the last window owns everything to the end
    return str(edf), hdr, total_sec, out


def main():
    edf, hdr, total_sec, windows = plan(RECORDING)
    print(f"--- {RECORDING}: {Path(edf).name}, {total_sec}s "
          f"({total_sec/60:.1f} min) in {len(windows)} window(s) ---")
    for w in windows[:3]:
        print(f"    w{w['n']:03d} records {w['start_rec']}-{w['stop_rec']}  "
              f"interior {w['t0']:.0f}-{w['t1']:.0f}s")
    if len(windows) > 3:
        print(f"    ... {len(windows)-3} more")

    # Per-profile: the dumped arrays are AR-filled against the mask, so two profiles produce
    # DIFFERENT decimated signal under the same window number.
    dec_dir = ROOT / "prep_edf" / f"_{RECORDING}{_VAR_SUF}{_QC_SUF}_windows"
    dec_dir.mkdir(parents=True, exist_ok=True)
    for w in windows:
        tag = f"_w{w['n']:03d}"
        env = {**os.environ, "RECORDING": RECORDING,
               "START_REC": str(w["start_rec"]), "STOP_REC": str(w["stop_rec"]),
               "WINDOW_TAG": tag,
               "RUN_DELPHOS": "0",                       # once, at the end, over the whole file
               "DUMP_DEC": str(dec_dir / f"dec{tag}.npy")}
        print(f"\n=== window {w['n']}/{len(windows)} ===")
        p = subprocess.run([sys.executable, "-m", "sdc.detect.compare_spikes"],
                           cwd=str(ROOT), env=env)
        if p.returncode != 0:
            raise SystemExit(f"window {w['n']} failed (exit {p.returncode})")
    print(f"\nall {len(windows)} windows done; assemble with merge_windows()")




def _assemble_mask(dec_dir, windows, fs, n_chan):
    """The dilated artefact mask for the whole file, from the per-window packed dumps."""
    out = []
    for w in windows:
        stem = dec_dir / f"dec_w{w['n']:03d}"
        shp = np.load(str(stem) + "_maskshape.npy")
        m = np.unpackbits(np.load(str(stem) + "_mask.npy"), axis=0)[:shp[0]].astype(bool)
        i0 = int(round((w["t0"] - (w["start_rec"] - 1)) * fs))
        i1 = min(int(round((w["t1"] - (w["start_rec"] - 1)) * fs)), m.shape[0])
        out.append(m[i0:i1])
    return np.concatenate(out, axis=0)


def merge_windows(rec_id=None, delete_parts=True):
    """Assemble the per-window outputs into one run, and run Delphos ONCE over the whole file.

    Order matters: the EDF has to exist before Delphos can read it, and Delphos's detections
    then go through the same artefact mask and MERGE_MS as the other two, exactly as in a
    single-window run.
    """
    from sdc.detect.sim_data import write_edf, verify_edf
    from sdc.detect.delphos_detect_spikes import detect_spikes as detect_delphos

    rec_id = rec_id or RECORDING
    edf, hdr, total_sec, windows = plan(rec_id)
    dec_dir = ROOT / "prep_edf" / f"_{rec_id}{_VAR_SUF}{_QC_SUF}_windows"
    parts = [np.load(RUNS / f"{rec_id}{RUN_TAG}_w{w['n']:03d}.npz", allow_pickle=False)
             for w in windows]
    z0 = parts[0]
    fs = float(z0["fs"])
    names = [str(s) for s in z0["names"]]
    n_chan = len(names)
    dets = [str(s) for s in z0["detectors"] if str(s) != "Delphos"]

    # EVERY WINDOW MUST CARRY THE SAME DETECTORS. `dets` is read from window 1 alone, and the
    # per-window npz files are CACHED and reused on a re-run -- so a window that lost a detector
    # is permanent until deleted, and silent. This is not hypothetical: a MATLAB licence failure
    # mid-batch ("Unable to launch MVM server", error 5001) made compare_spikes print
    # "[warn] Barkmeier unavailable" and carry on, writing one window with Janca only. Merging
    # that produces a Barkmeier rate missing an entire window of the recording -- a ~20% undercount
    # on a 5-window file, in the right ballpark to be believed.
    _want = set(dets)
    for w, p in zip(windows, parts):
        _have = {str(s) for s in p["detectors"] if str(s) != "Delphos"}
        if _have != _want:
            f = RUNS / f"{rec_id}{RUN_TAG}_w{w['n']:03d}.npz"
            raise SystemExit(
                f"window {w['n']:03d} has detectors {sorted(_have)} but window 001 has "
                f"{sorted(_want)}.\n"
                f"  A cached window is incomplete -- most likely a detector failed for that "
                f"window only (check its log for 'unavailable').\n"
                f"  DELETE it and re-run; the other windows are reused, so this is cheap:\n"
                f"    del \"{f}\"")

    # ---- detections: interior only, shifted to absolute file time -------------------------
    per = {d: [[] for _ in range(n_chan)] for d in dets}
    # The mask-rejected detections travel alongside, through the identical interior filter and
    # merge, so `{Det}_idx_masked` in the merged file means exactly what `{Det}_idx` means minus
    # having survived the mask. Needed to evaluate a LOOSER mask without re-running detectors.
    rej = {d: [[] for _ in range(n_chan)] for d in dets}
    for w, z in zip(windows, parts):
        off = w["start_rec"] - 1                       # window start, seconds from file start
        for d in dets:
            for src, acc in ((f"{d}_idx", per[d]), (f"{d}_idx_masked", rej[d])):
                if src not in z.files:                 # run made before the masked arrays existed
                    continue
                t = z[src] / fs + off                  # absolute seconds
                c = z[src.replace("_idx", "_chan")]
                keep = (t >= w["t0"]) & (t < w["t1"])  # INTERIOR only -- overlap never counted twice
                for tt, cc in zip(t[keep], c[keep]):
                    acc[cc].append(tt)

    # ---- the whole-file preprocessed EDF ---------------------------------------------------
    chunks = []
    for w in windows:
        a = np.load(dec_dir / f"dec_w{w['n']:03d}.npy", mmap_mode="r")
        i0 = int(round((w["t0"] - (w["start_rec"] - 1)) * fs))
        i1 = min(int(round((w["t1"] - (w["start_rec"] - 1)) * fs)), a.shape[0])
        chunks.append(np.asarray(a[i0:i1]))
    full = np.concatenate(chunks, axis=0)
    del chunks
    expect = int(round(total_sec * fs))
    if abs(full.shape[0] - expect) > fs:               # a second of slack for record rounding
        raise SystemExit(f"assembled {full.shape[0]} samples, expected ~{expect} "
                         f"({total_sec}s at {fs:g} Hz) -- interior arithmetic is wrong.")
    # THE FILENAME MUST CARRY EVERY SETTING THAT CHANGES THE CONTENT. Two bugs, both silent,
    # both hit at once when FILL_ALL was introduced:
    #   * this used to skip the write when a file of that name existed, so a re-run after a
    #     config change read the PREVIOUS run's signal;
    #   * Delphos's on-disk cache is keyed on {path, SIZE, cfg}, and filling does not change
    #     the size -- so even a rewritten file would have returned the stale cached result.
    # Delphos came back byte-identical (14079) after the fill and looked unaffected by it.
    # Distinct names defeat both: the file is rewritten and the cache misses.
    _v = f"_med{int(z0['med_kernel'])}_{fs:g}Hz"
    _v += "_fill" if ("fill_all" not in z0.files or int(z0["fill_all"])) else "_nofill"
    # Read from the npz, not from the environment: the assembled EDF must be named after the
    # profile the WINDOWS were actually computed at, which is the same argument the med_kernel
    # and fill flags above are read from z0 for.
    _prof = str(z0["qc_profile"]) if "qc_profile" in z0.files else "prod"
    _v += "" if _prof == "prod" else f"_qc{_prof}"
    # Pulse blanking changes the SAMPLES without changing their number, so a blanked file is
    # byte-different but exactly the same SIZE as the unblanked one -- the precise case the
    # comment above says the name has to defeat. Omitting it here handed Delphos the stale
    # unblanked EDF and it returned a bit-identical cached result for every blanked run.
    _pb = float(z0["pulse_blank_ms"]) if "pulse_blank_ms" in z0.files else 0.0
    _pbf = str(z0["pulse_fill"]) if "pulse_fill" in z0.files else "auto"
    _v += "" if not _pb else f"_pb{_pb:g}{ {'auto': '', 'interp': 'i', 'ar': 'a'}[_pbf] }"
    prep = ROOT / "prep_edf" / f"{rec_id}_full{_v}.edf"
    if not MERGE_DELPHOS:
        print(f"[prep] RUN_DELPHOS=0 -- not writing {prep.name} (it exists only for Delphos)")
    elif not prep.is_file():
        print(f"[prep] writing {prep.name}  {full.shape} ...")
        write_edf(str(prep), full, names, fs)
        verify_edf(str(prep), full, names, fs)
        print("[prep] verified")
    n_samp = full.shape[0]
    del full

    # ---- analysable time and stim ON, summed over interiors --------------------------------
    clean = np.zeros(n_chan)
    on_sec = np.zeros(0, bool)
    on_all = []
    clean_secs = []
    for w in windows:
        cl = np.load(dec_dir / f"dec_w{w['n']:03d}_clean.npy")
        on = np.load(dec_dir / f"dec_w{w['n']:03d}_on.npy")
        s0 = int(round(w["t0"])) - (w["start_rec"] - 1)
        s1 = min(int(round(w["t1"])) - (w["start_rec"] - 1), cl.shape[0])
        clean += cl[s0:s1].sum(axis=0) / fs
        clean_secs.append(cl[s0:s1])
        on_all.append(on[s0:s1])
    on_sec = np.concatenate(on_all) if on_all else np.zeros(0, bool)

    # ---- Delphos, ONCE, over the assembled file --------------------------------------------
    if not MERGE_DELPHOS:
        print("[delphos] SKIPPED (RUN_DELPHOS=0): Janca and Barkmeier only")
        dl = None
    else:
        print(f"[delphos] one call over the whole {total_sec}s file")
        dl = detect_delphos(str(prep), names, fs, start_sec=0.0, duration_sec=float(total_sec),
                        cache_dir=ROOT / ".delphos_cache", bipolar=False, **DELPHOS_CFG)
    # SAME artefact mask as the other two. Janca and Barkmeier were masked inside their
    # per-window runs; Delphos is run here, so it must be masked here. Skipping this once made
    # Delphos the only unmasked arm and inflated it by ~20%, which read as a Delphos result.
    dmask = _assemble_mask(dec_dir, windows, fs, n_chan)
    kept, dropped = [], []
    n_raw = 0
    for c, x in enumerate(dl if dl is not None else []):
        idx = np.unique(np.asarray(x, int))
        idx = idx[(idx >= 0) & (idx < dmask.shape[0])]
        n_raw += idx.size
        kept.append(list(idx[~dmask[idx, c]] / fs))
        dropped.append(list(idx[dmask[idx, c]] / fs))     # kept for the same reason as above
    print(f"  [Delphos] {n_raw} detected -> artefact mask: "
          f"-{n_raw - sum(len(k) for k in kept)} -> {sum(len(k) for k in kept)} kept")
    if dl is not None:
        per["Delphos"] = kept
        rej["Delphos"] = dropped

    # ---- one merge across the joins ---------------------------------------------------------
    merge_ms = float(z0["merge_ms"])

    def _merge_all(src):
        return {d: [merge_close(np.unique(np.round(np.asarray(v, float) * fs).astype(int)),
                                merge_ms / 1000.0 * fs) if v else np.zeros(0, int)
                    for v in src[d]] for d in list(src)}

    out = _merge_all(per)
    out_rej = _merge_all(rej)
    for d in out:
        print(f"  {d:<10} {sum(len(x) for x in out[d])} kept"
              f"  (+{sum(len(x) for x in out_rej.get(d, []))} masked, stored separately)")
    return dict(rec_id=rec_id, fs=fs, names=names, seconds=total_sec, prep=str(prep),
                per=out, rej=out_rej, clean_sec=clean, on_sec=on_sec, n_samp=n_samp,
                windows=windows,
                clean_per_sec=(np.concatenate(clean_secs) if clean_secs
                               else np.zeros((0, n_chan), np.uint16)))


def write_merged(m):
    """One npz for the whole recording, keyed like any single-window run so every downstream
    script reads it unchanged."""
    n_chan = len(m["names"])
    dump = {"names": np.array(m["names"]), "fs": np.float64(m["fs"]),
            "seconds": np.int64(m["seconds"]), "edf": m["prep"],
            "detectors": np.array([d for d in ("Janca", "Barkmeier", "Delphos")
                                   if d in m["per"]])}
    for d in [str(x) for x in dump["detectors"]]:
        # Kept and mask-rejected, written the same way. The `_masked` arrays are what lets a
        # looser mask be tried later; they are NOT part of any result and no reader that ignores
        # them changes behaviour. Note this is a whitelist-style writer -- a new key that is not
        # named here never reaches the merged file, which is how `qc_profile` and `pStimAll`
        # were both silently lost after being added correctly to the per-window dump.
        for suf, src in (("", m["per"]), ("_masked", m.get("rej", {}))):
            idx = src.get(d, [])
            has = any(len(x) for x in idx)
            dump[f"{d}_idx{suf}"] = np.concatenate(idx) if has else np.zeros(0, int)
            dump[f"{d}_chan{suf}"] = (np.concatenate([np.full(len(x), c, int)
                                                      for c, x in enumerate(idx)])
                                      if has else np.zeros(0, int))
    z0 = np.load(RUNS / f"{m['rec_id']}{RUN_TAG}_w001.npz", allow_pickle=False)
    for k in ("merge_ms", "dilate_ms", "tol_ms", "detect_fs", "mask_artefacts", "janca_pt_ms",
              "qc_profile",   # which artefact mask produced this run. Without it the merged
                              # file cannot be told apart from a production run except by its
                              # filename, and a filename is not provenance.
              "qc_native", "med_kernel", "fill_bad_samples", "rec_id", "patient", "condition",
              "stim_hz", "delphos_input",
              # Pulse blanking and the baseline window. These were dumped per WINDOW but not
              # carried here, so the merged file -- the one every figure reads -- had no record
              # of them. sdc.artefact.inspect.view_run reads pulse_blank_ms from the merged npz
              # to decide whether to blank the trace it draws, and silently drew an UNBLANKED
              # trace labelled "blanked" because the key was missing.
              "pulse_blank_ms", "pulse_fill", "pulse_max_peak_uv", "pulse_n", "pulse_thr",
              "pulse_isi_ms", "pulse_n_gated_out", "pulse_blank_frac", "baseline_sec",
              # Barkmeier's normaliser: what it was PINNED to (nan = published behaviour) and
              # the legacy BARK_SCALE knob. Both are scalar and constant across windows.
              "bark_fixed_denom", "bark_scale"):
        if k in z0.files:
            dump[k] = z0[k]
    # ...and the MEASURED per-block denominators, which are NOT constant across windows: each
    # window reports its own blocks, so the merged record is their concatenation. Taken from
    # every window rather than from w001, because the median over one window is not the median
    # over the recording -- and that median is what tools/run_pinned_2hz.py reads back to pin
    # the whole comparison to.
    _bd = []
    for _i in range(1, len(m["windows"]) + 1):
        _p = RUNS / f"{m['rec_id']}{RUN_TAG}_w{_i:03d}.npz"
        if not _p.is_file():
            continue
        with np.load(_p, allow_pickle=False) as _zw:
            if "bark_block_denom" in _zw.files and _zw["bark_block_denom"].size:
                _bd.append(np.asarray(_zw["bark_block_denom"], float).ravel())
    dump["bark_block_denom"] = np.concatenate(_bd) if _bd else np.zeros(0, float)
    # windowing provenance -- a run made at a different window size is NOT comparable, because
    # Janca's background model and Barkmeier's block threshold are both computed over the span
    # they are handed.
    # PER-SECOND analysable samples per channel, and the per-second stim flag. Without these,
    # restricting an analysis to a sub-range (e.g. "the first 900 s, before the artefact at the
    # end") can only SCALE the whole-recording clean time -- which assumes masking is uniform in
    # time, and the entire reason for wanting a sub-range is that it is not.
    # The Delphos operating point this run used. Not cosmetic: the detections move with it, so a
    # run compared against one made at a different pin is confounded, and without this the only
    # evidence of which pin produced a file is the shell that launched it.
    dump["delphos_pin_gb"] = np.int64(DELPHOS_CFG.get("pin_free_ram_gb") or 0)
    dump["clean_per_sec"] = m["clean_per_sec"]
    dump["on_per_sec"] = m["on_sec"]
    dump["window_sec"] = np.int64(m["windows"][0]["stop_rec"] - m["windows"][0]["start_rec"] + 1)
    dump["n_windows"] = np.int64(len(m["windows"]))
    on = m["on_sec"]
    dump["clean_sec_on"] = m["clean_sec"] * 0.0 if not on.any() else None   # filled below
    # ON/OFF: per-second resolution is enough -- the QC epochs are 2 s anyway.
    if on.any():
        # EXACT, from the per-second counts. This used to be `clean_sec * on.mean()` -- the
        # total analysable time split by the ON/OFF duty cycle -- which assumes each channel's
        # masking is spread uniformly in time. It is not: stim artefact is concentrated in the
        # ON blocks by definition, so a channel heavily masked during stim was still credited
        # with ~30% of its whole-recording clean time as "ON clean time" (up to 231 s of
        # fiction on P1_stim). Its ON rate was then divided by a denominator far too large,
        # UNDERSTATING ON rates and OVERSTATING suppression -- i.e. biasing finding 8 in the
        # direction it was reporting.
        _cps = m["clean_per_sec"]
        _n = min(_cps.shape[0], on.size)
        dump["clean_sec_on"] = _cps[:_n][on[:_n]].sum(axis=0) / m["fs"]
        dump["clean_sec_off"] = _cps[:_n][~on[:_n]].sum(axis=0) / m["fs"]
        edges = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
        b = np.concatenate([np.array([0] if on[0] else [], np.int64), edges,
                            np.array([on.size] if on[-1] else [], np.int64)])
        dump["on_runs"] = (b.reshape(-1, 2) * int(m["fs"])).astype(np.int64)
        dump["sec_on"] = np.float64(on.sum())
        dump["sec_off"] = np.float64((~on).sum())
        for d in [str(x) for x in dump["detectors"]]:
            for suf in ("", "_masked"):
                v = dump[f"{d}_idx{suf}"]
                t = (v / m["fs"]).astype(int)
                dump[f"{d}_on{suf}"] = (on[np.clip(t, 0, on.size - 1)] if v.size
                                        else np.zeros(0, bool))
    else:
        dump["clean_sec_on"] = np.zeros(n_chan)
        dump["clean_sec_off"] = m["clean_sec"]
        dump["on_runs"] = np.zeros((0, 2), np.int64)
        dump["sec_on"] = np.float64(0.0)
        dump["sec_off"] = np.float64(m["seconds"])
        # dump["detectors"], NOT a hardcoded triple. The stim branch above already does this;
        # this branch did not, so a BASELINE run with RUN_DELPHOS=0 died with
        # KeyError: 'Delphos_idx'. Stim runs were unaffected because they take the other branch,
        # which is why it survived until a baseline was first run without Delphos.
        for d in [str(x) for x in dump["detectors"]]:
            dump[f"{d}_on"] = np.zeros(dump[f"{d}_idx"].size, bool)
    out = RUNS / f"{m['rec_id']}{RUN_TAG}.npz"
    check_run(dump, n_samp=m["n_samp"], fs=m["fs"])
    np.savez(out, **{k: v for k, v in dump.items() if v is not None})
    print(f"[saved] {out.name}   "
          + "  ".join(f"{d} {dump[f'{d}_idx'].size}"
                        for d in [str(x) for x in dump["detectors"]]))
    return out


if __name__ == "__main__":
    main()
    write_merged(merge_windows())


def draw_raster(rec_id=None):
    """One raster for the WHOLE assembled recording.

    Separate from compare_spikes' raster, which is suppressed under WINDOW_TAG because a
    per-window figure is a fragment written to the recording's path."""
    import matplotlib.pyplot as plt
    from seeg._style import RED, BLUE, MUTED, recessive
    rec_id = rec_id or RECORDING
    z = np.load(RUNS / f"{rec_id}{RUN_TAG}.npz", allow_pickle=False)
    names = [str(s) for s in z["names"]]
    fs, T = float(z["fs"]), float(z["seconds"])
    n_chan = len(names)
    dets = [(d, c) for d, c in (("Janca", RED), ("Barkmeier", BLUE), ("Delphos", "#4a3aa7"))
            if f"{d}_idx" in z.files]
    per = {d: [np.sort(z[f"{d}_idx"][z[f"{d}_chan"] == k] / fs) for k in range(n_chan)]
           for d, _ in dets}
    order = np.argsort([-per[dets[0][0]][k].size for k in range(n_chan)])
    edges = np.arange(0, T + 1, 1.0)
    centres = edges[:-1] + 0.5

    fig, axes = plt.subplots(len(dets) + 1, 1, sharex=True,
                             figsize=(16, 5 + 3 * len(dets)),
                             gridspec_kw={"height_ratios": [1] + [3] * len(dets)})
    axr = axes[0]
    for d, col in dets:
        allt = np.concatenate([p for p in per[d] if p.size]) if any(p.size for p in per[d]) \
            else np.zeros(0)
        axr.plot(centres, np.histogram(allt, bins=edges)[0], color=col, lw=1.0,
                 label=f"{d} ({sum(p.size for p in per[d])})")
    axr.set_ylabel("pop. rate\n(spikes/s)")
    axr.legend(loc="upper right", frameon=False, fontsize=8, ncol=len(dets))
    recessive(axr)
    on = z["on_runs"] / fs if "on_runs" in z.files else np.zeros((0, 2))
    for a0, a1 in on:
        for ax in axes:
            ax.axvspan(a0, a1, color="#f0c419", alpha=.16, lw=0, zorder=0)
    if on.size:
        axr.text(.004, .96, "shaded = stim ON", transform=axr.transAxes, va="top",
                 fontsize=8, color="#8a6d00")
    for ax, (d, col) in zip(axes[1:], dets):
        ax.eventplot([per[d][order[i]] for i in range(n_chan)], colors=col,
                     lineoffsets=np.arange(n_chan), linelengths=0.8, linewidths=0.4)
        ax.set_title(f"{d} (n={sum(p.size for p in per[d])})", loc="left", fontsize=10,
                     color=col)
        ax.set_ylabel("channel (busiest -> top)")
        yt = np.arange(0, n_chan, max(n_chan // 20, 1))
        ax.set_yticks(yt)
        ax.set_yticklabels([names[order[i]] for i in yt], fontsize=6)
        ax.set_ylim(n_chan, -1)
        recessive(ax)
    axes[-1].set_xlim(0, T)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{rec_id}: whole recording, {T:.0f}s ({T/60:.1f} min), {n_chan} bipolar ch "
                 f"at {fs:g} Hz  |  {int(z['n_windows'])} windows of "
                 f"{int(z['window_sec'])}s for Janca/Barkmeier, Delphos in one pass",
                 fontsize=11)
    fig.tight_layout()
    out = ROOT / "figures" / "real" / rec_id / f"compare_raster{RUN_TAG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")
