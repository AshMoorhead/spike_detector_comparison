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
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from seeg import read_edf_header, load_trials, get_patient, get_trial, resolve_file, merge_close
from seeg.edf import window_bounds

from sdc.common.paths import ROOT, RUNS
from sdc.common.invariants import check_run

RECORDING = os.environ.get("RECORDING", "P1_pre")
MEM_BUDGET = float(os.environ.get("MEM_BUDGET", 1e9))   # bytes; window_bounds' own default
BLOCK_SEC = int(os.environ.get("BLOCK_SEC", 60))        # the pipeline's block size, not the
                                                        # 120 s the interactive script uses
BASE_DIR = Path(r"C:\Users\amoo0039\Documents\local")
META_PATH = BASE_DIR / "data_meta" / "stim_trials.json"
RECORDINGS = {
    "P1_pre":  dict(patient=1, trial_index=1, file_type="pre"),
    "P1_stim": dict(patient=1, trial_index=1, file_type="stim"),
    "P5_pre":  dict(patient=5, trial_index=1, file_type="pre"),
    "P5_stim": dict(patient=5, trial_index=1, file_type="stim"),
}


def plan(rec_id):
    """Every window for this recording: (start_rec, stop_rec, interior_t0, interior_t1) in
    1-based inclusive records and absolute seconds."""
    cfg = RECORDINGS[rec_id]
    trials = load_trials(META_PATH)
    entry = get_trial(get_patient(trials, cfg["patient"]), cfg["trial_index"])
    stem, _trial = resolve_file(entry, cfg["file_type"])
    edf = BASE_DIR / f"P{cfg['patient']}" / f"{stem}.edf"
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

    dec_dir = ROOT / "prep_edf" / f"_{RECORDING}_windows"
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
    dec_dir = ROOT / "prep_edf" / f"_{rec_id}_windows"
    parts = [np.load(RUNS / f"{rec_id}_w{w['n']:03d}.npz", allow_pickle=False) for w in windows]
    z0 = parts[0]
    fs = float(z0["fs"])
    names = [str(s) for s in z0["names"]]
    n_chan = len(names)
    dets = [str(s) for s in z0["detectors"] if str(s) != "Delphos"]

    # ---- detections: interior only, shifted to absolute file time -------------------------
    per = {d: [[] for _ in range(n_chan)] for d in dets}
    for w, z in zip(windows, parts):
        off = w["start_rec"] - 1                       # window start, seconds from file start
        for d in dets:
            t = z[f"{d}_idx"] / fs + off               # absolute seconds
            c = z[f"{d}_chan"]
            keep = (t >= w["t0"]) & (t < w["t1"])      # INTERIOR only -- overlap never counted twice
            for tt, cc in zip(t[keep], c[keep]):
                per[d][cc].append(tt)

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
    prep = ROOT / "prep_edf" / f"{rec_id}_full{_v}.edf"
    if not prep.is_file():
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
    print(f"[delphos] one call over the whole {total_sec}s file")
    dl = detect_delphos(str(prep), names, fs, start_sec=0.0, duration_sec=float(total_sec),
                        cache_dir=ROOT / ".delphos_cache", bipolar=False,
                        pin_free_ram_gb=12, Spk_thr=50, Spk_time_thr=1.25)
    # SAME artefact mask as the other two. Janca and Barkmeier were masked inside their
    # per-window runs; Delphos is run here, so it must be masked here. Skipping this once made
    # Delphos the only unmasked arm and inflated it by ~20%, which read as a Delphos result.
    dmask = _assemble_mask(dec_dir, windows, fs, n_chan)
    kept = []
    n_raw = 0
    for c, x in enumerate(dl):
        idx = np.unique(np.asarray(x, int))
        idx = idx[(idx >= 0) & (idx < dmask.shape[0])]
        n_raw += idx.size
        kept.append(list(idx[~dmask[idx, c]] / fs))
    print(f"  [Delphos] {n_raw} detected -> artefact mask: "
          f"-{n_raw - sum(len(k) for k in kept)} -> {sum(len(k) for k in kept)} kept")
    per["Delphos"] = kept

    # ---- one merge across the joins ---------------------------------------------------------
    merge_ms = float(z0["merge_ms"])
    out = {}
    for d in list(per):
        idx = [merge_close(np.unique(np.round(np.asarray(v, float) * fs).astype(int)),
                           merge_ms / 1000.0 * fs) if v else np.zeros(0, int) for v in per[d]]
        out[d] = idx
        print(f"  {d:<10} {sum(len(x) for x in idx)}")
    return dict(rec_id=rec_id, fs=fs, names=names, seconds=total_sec, prep=str(prep),
                per=out, clean_sec=clean, on_sec=on_sec, n_samp=n_samp, windows=windows,
                clean_per_sec=(np.concatenate(clean_secs) if clean_secs
                               else np.zeros((0, n_chan), np.uint16)))


def write_merged(m):
    """One npz for the whole recording, keyed like any single-window run so every downstream
    script reads it unchanged."""
    n_chan = len(m["names"])
    dump = {"names": np.array(m["names"]), "fs": np.float64(m["fs"]),
            "seconds": np.int64(m["seconds"]), "edf": m["prep"],
            "detectors": np.array(["Janca", "Barkmeier", "Delphos"])}
    for d in ("Janca", "Barkmeier", "Delphos"):
        idx = m["per"][d]
        dump[f"{d}_idx"] = (np.concatenate(idx) if any(len(x) for x in idx)
                            else np.zeros(0, int))
        dump[f"{d}_chan"] = (np.concatenate([np.full(len(x), c, int)
                                             for c, x in enumerate(idx)])
                             if any(len(x) for x in idx) else np.zeros(0, int))
    z0 = np.load(RUNS / f"{m['rec_id']}_w001.npz", allow_pickle=False)
    for k in ("merge_ms", "dilate_ms", "tol_ms", "detect_fs", "mask_artefacts", "janca_pt_ms",
              "qc_native", "med_kernel", "fill_bad_samples", "rec_id", "patient", "condition",
              "stim_hz", "delphos_input"):
        if k in z0.files:
            dump[k] = z0[k]
    # windowing provenance -- a run made at a different window size is NOT comparable, because
    # Janca's background model and Barkmeier's block threshold are both computed over the span
    # they are handed.
    # PER-SECOND analysable samples per channel, and the per-second stim flag. Without these,
    # restricting an analysis to a sub-range (e.g. "the first 900 s, before the artefact at the
    # end") can only SCALE the whole-recording clean time -- which assumes masking is uniform in
    # time, and the entire reason for wanting a sub-range is that it is not.
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
        for d in ("Janca", "Barkmeier", "Delphos"):
            t = (dump[f"{d}_idx"] / m["fs"]).astype(int)
            dump[f"{d}_on"] = on[np.clip(t, 0, on.size - 1)] if dump[f"{d}_idx"].size                 else np.zeros(0, bool)
    else:
        dump["clean_sec_on"] = np.zeros(n_chan)
        dump["clean_sec_off"] = m["clean_sec"]
        dump["on_runs"] = np.zeros((0, 2), np.int64)
        dump["sec_on"] = np.float64(0.0)
        dump["sec_off"] = np.float64(m["seconds"])
        for d in ("Janca", "Barkmeier", "Delphos"):
            dump[f"{d}_on"] = np.zeros(dump[f"{d}_idx"].size, bool)
    out = RUNS / f"{m['rec_id']}.npz"
    check_run(dump, n_samp=m["n_samp"], fs=m["fs"])
    np.savez(out, **{k: v for k, v in dump.items() if v is not None})
    print(f"[saved] {out.name}   "
          + "  ".join(f"{d} {dump[f'{d}_idx'].size}" for d in ("Janca", "Barkmeier", "Delphos")))
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
    z = np.load(RUNS / f"{rec_id}.npz", allow_pickle=False)
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
    out = ROOT / "figures" / "real" / rec_id / "compare_raster.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")
