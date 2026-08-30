"""
sdc.tools.trim_run
------------------
Cut a finished run npz at a wall-clock time, producing a run that every downstream script reads
as an ordinary (shorter) recording.

    .venv\\Scripts\\python.exe -m sdc.tools.trim_run P1_stim_qcfinal 970

WHY TRIM RATHER THAN RE-RUN
  Nothing in a run npz depends on the recording's length: detections are absolute sample indices,
  analysable time is per second and per channel, and the ON mask is per second. So a truncation
  is exact -- the same numbers a re-run over the shorter span would have produced -- for
  everything except the detectors' own internal models, which were fitted per WINDOW anyway
  (Janca's background, Barkmeier's block threshold; see run_windows). Those are unchanged by
  this, which is the one thing a physical re-run would do differently and is noted here rather
  than hidden.

WHY IT IS NEEDED ON P1
  P1's recording stops mid-block: the sixth ON run is an 8 s sliver, and the fifth (970-1034 s)
  has no room for a duration-matched OFF after it. The block-paired estimator can drop those by
  itself, but the descriptive figures -- rasters, per-bin rates, stim_effect -- cannot, so they
  describe a slightly different recording from the one the estimates come from. Trimming at 970 s
  puts every figure on the same span.

EVERY TIME-INDEXED FIELD IS RECOMPUTED, not scaled. `clean_sec_on/off` in particular are summed
from `clean_per_sec` over the retained seconds; scaling them by the retained fraction would
assume masking is uniform in time, and it is concentrated in the ON blocks by construction --
the exact bug that once put 231 s of fiction into P1_stim's ON clean time.
"""
import sys

import numpy as np

from sdc.common.invariants import check_run
from sdc.common.paths import RUNS


def trim(stem, t_end, out_stem=None, t_start=0.0):
    z = np.load(RUNS / f"{stem}.npz", allow_pickle=False)
    fs = float(z["fs"])
    t_end, t_start = float(t_end), float(t_start)
    # `t_start` drops a short leading run as well as a short trailing one. P1's 145 Hz file opens
    # with SIX SECONDS of OFF before the first stim block -- too short to be a rate measurement,
    # but long enough to appear as its own segment in every descriptive figure and contribute a
    # point built on 6 s of data.
    s0 = int(np.floor(t_start))
    n_sec = int(np.floor(t_end)) - s0
    a_samp, n_samp = int(round(t_start * fs)), int(round((t_end - t_start) * fs))
    dets = [str(d) for d in z["detectors"]]
    out = {}

    for k in z.files:
        v = z[k]
        if k.endswith(("_idx", "_chan", "_on", "_idx_masked", "_chan_masked", "_on_masked")):
            continue                                   # handled per detector below
        if k in ("clean_per_sec", "on_per_sec"):
            out[k] = v[s0:s0 + n_sec]
        elif k in ("seconds",):
            out[k] = np.int64(n_sec)
        elif k in ("clean_sec_on", "clean_sec_off", "sec_on", "sec_off", "on_runs"):
            continue                                   # recomputed below
        else:
            out[k] = v

    # detections: keep both the kept and the mask-rejected sets, with their channel and ON flags
    for d in dets:
        for suf in ("", "_masked"):
            ik, ck, ok = f"{d}_idx{suf}", f"{d}_chan{suf}", f"{d}_on{suf}"
            if ik not in z.files:
                continue
            idx = z[ik]
            m = (idx >= a_samp) & (idx < a_samp + n_samp)
            out[ik] = idx[m] - a_samp          # re-base onto the trimmed timeline
            out[ck] = z[ck][m] if ck in z.files else np.zeros(m.sum(), int)
            if ok in z.files:
                out[ok] = z[ok][m]

    on = np.asarray(out["on_per_sec"], bool) if "on_per_sec" in out else np.zeros(n_sec, bool)
    cps = np.asarray(out["clean_per_sec"], float)
    n_chan = cps.shape[1]
    if on.any():
        out["clean_sec_on"] = cps[on].sum(axis=0) / fs
        out["clean_sec_off"] = cps[~on].sum(axis=0) / fs
        out["sec_on"] = np.float64(on.sum())
        out["sec_off"] = np.float64((~on).sum())
        e = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
        b = np.concatenate([np.array([0] if on[0] else [], np.int64), e,
                            np.array([on.size] if on[-1] else [], np.int64)])
        out["on_runs"] = (b.reshape(-1, 2) * int(fs)).astype(np.int64)
    else:
        out["clean_sec_on"] = np.zeros(n_chan)
        out["clean_sec_off"] = cps.sum(axis=0) / fs
        out["sec_on"] = np.float64(0.0)
        out["sec_off"] = np.float64(n_sec)

    check_run(out, n_samp=n_samp, fs=fs)
    dst = RUNS / f"{out_stem or (stem + f'_t{n_sec}')}.npz"
    np.savez_compressed(dst, **out)

    print(f"{stem}.npz -> {dst.name}")
    print(f"  {float(z['seconds']):.0f}s -> {n_sec}s (from {s0}s);  ON {float(z['sec_on']):.0f} -> "
          f"{float(out['sec_on']):.0f}s;  OFF {float(z['sec_off']):.0f} -> "
          f"{float(out['sec_off']):.0f}s")
    print(f"  ON runs {len(np.atleast_2d(z['on_runs']))} -> {len(out['on_runs'])}")
    for d in dets:
        print(f"  {d:<11}{z[f'{d}_idx'].size:>7} -> {out[f'{d}_idx'].size:<7}"
              f"  (masked {z.get(f'{d}_idx_masked', np.zeros(0)).size if hasattr(z, 'get') else 0})"
              if False else
              f"  {d:<11}{z[f'{d}_idx'].size:>7} -> {out[f'{d}_idx'].size}")
    return dst


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m sdc.tools.trim_run <run_stem> <t_end_seconds>")
    trim(sys.argv[1], float(sys.argv[2]))
