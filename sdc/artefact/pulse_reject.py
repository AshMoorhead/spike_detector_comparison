"""
sdc.artefact.pulse_reject
-------------------------
Reject DETECTIONS that fall within +-tol of a stimulation pulse, and write a derived run.

    .venv\\Scripts\\python.exe -m sdc.artefact.pulse_reject P1_ANT2_stim_pb5i_qcfinalv2 10

A different intervention from `blank_pulses`, and a more honest one where the artefact cannot be
removed. Blanking edits the SIGNAL and hopes the detectors then behave; this edits nothing and
simply declines to trust anything the detectors report in the window where a pulse makes them
untrustworthy. Nothing is invented, so nothing new can be detected -- which is exactly the
failure AR fill produced (Delphos 1.88 -> 10.69 by firing on its own repair).

THE TIME MUST COME OUT OF THE DENOMINATOR. Rejecting +-10 ms around a 2 Hz pulse train discards
4% of the recording. Drop the detections without dropping the time and every rate falls by 4%
for free, and every stim/baseline ratio with it -- an effect manufactured by the bookkeeping.
So `clean_per_sec` is reduced by the rejected span too.

  Exactly how much clean time each channel loses depends on the OVERLAP between the rejection
  windows and the QC mask, and the run npz stores only per-second clean COUNTS, not the
  per-sample mask. So the loss is apportioned proportionally: a second that is 60% clean loses
  60% of that second's rejected samples. Exact in expectation, and the error is bounded by how
  strongly the QC mask correlates with pulse timing within a single second.

THE BASELINE IS NOT TOUCHED. It contains no stimulation, so there is nothing to reject; the
comparison stays rate against rate, not count against count.
"""
import sys

import numpy as np

from seeg import (read_edf_header, load_edf_segment, derive_montage, apply_montage,
                  detect_stim, detect_pulses)
from seeg.stim import get_stim_channel, _stim_column

from sdc.common.paths import RUNS
from sdc.detect.recordings import edf_path

DETS = ("Janca", "Barkmeier", "Delphos")


def pulse_times(rec_id):
    """Pulse onsets in seconds, from the raw stim channel of the recording itself."""
    edf, trial = edf_path(rec_id)
    if trial is None:
        return np.zeros(0), None
    hdr = read_edf_header(edf)
    r = load_edf_segment(edf, hdr, 1, int(hdr["NumDataRecords"]))
    r = apply_montage(r, derive_montage(r["info"]["SelectedSignals"]), verbose=False)
    r["info"]["stim_trial"] = trial
    r = detect_stim(r, verbose=False)
    fs = r["info"]["SampleRate"]
    _, info = detect_pulses(_stim_column(r, get_stim_channel(r, verbose=False)), fs,
                            stim_hz=float(trial["stim_frequency"]))
    return info["onsets"] / fs, info


def reject(stem, tol_ms=10.0, rec_id=None, out_stem=None, pre_ms=None, post_ms=None):
    """Write <stem> with peri-pulse detections removed and the time removed with them.

    ASYMMETRIC when pre_ms/post_ms are given: the window runs from `pre_ms` BEFORE each pulse to
    `post_ms` AFTER it. Symmetric +-tol_ms otherwise.

    The stimulus-evoked response is not symmetric about the pulse -- it peaks around 100 ms
    after and there is nothing to reject before -- so a symmetric window either misses the
    evoked response or discards an equal span of clean pre-pulse signal for nothing. Rejecting
    -20/+100 ms deliberately removes the evoked potential as well as the artefact, which is a
    choice about what counts as a spike, not only about what counts as noise.
    """
    z = np.load(RUNS / f"{stem}.npz", allow_pickle=False)
    rec_id = rec_id or str(z["rec_id"])
    fs = float(z["fs"])
    ons, info = pulse_times(rec_id)
    if not ons.size:
        raise SystemExit(f"{rec_id} has no stim trial -- nothing to reject")
    pre = (tol_ms if pre_ms is None else pre_ms) / 1000.0
    post = (tol_ms if post_ms is None else post_ms) / 1000.0
    tag = f"{tol_ms:g}" if pre_ms is None and post_ms is None else f"m{pre * 1e3:g}p{post * 1e3:g}"

    def near(t):
        # lag to the PRECEDING pulse, so the window can be asymmetric about it
        j = np.clip(np.searchsorted(ons, t) - 1, 0, ons.size - 1)
        lag = t - ons[j]
        after = (lag >= 0) & (lag <= post)
        j2 = np.clip(j + 1, 0, ons.size - 1)
        before = (ons[j2] - t >= 0) & (ons[j2] - t <= pre)
        return after | before

    out = {k: z[k] for k in z.files}
    n_before, n_after = {}, {}
    for d in DETS:
        if f"{d}_idx" not in z.files:
            continue
        t = z[f"{d}_idx"] / fs
        keep = ~near(t)
        n_before[d], n_after[d] = t.size, int(keep.sum())
        for suf in ("_idx", "_chan", "_on"):
            k = f"{d}{suf}"
            if k in z.files and z[k].shape[0] == t.size:
                out[k] = z[k][keep]

    # per-second rejected span, then apportioned by how clean that second already was
    cps = np.asarray(z["clean_per_sec"], float)
    n_sec = cps.shape[0]
    edges = np.arange(n_sec + 1)
    lo, hi = np.clip(ons - pre, 0, n_sec), np.clip(ons + post, 0, n_sec)
    cov = np.zeros(n_sec)
    for a, b in zip(lo, hi):                       # windows are short and disjoint at 2 Hz
        i0, i1 = int(a), min(int(b), n_sec - 1)
        if i0 == i1:
            cov[i0] += b - a
        else:
            cov[i0] += edges[i0 + 1] - a
            cov[i1] += b - edges[i1]
    frac = np.clip(cov, 0, 1.0)                    # fraction of each second rejected
    out["clean_per_sec"] = cps * (1.0 - frac)[:, None]
    out["pulse_reject_ms"] = np.float64(max(pre, post) * 1000.0)
    out["pulse_reject_pre_ms"] = np.float64(pre * 1000.0)
    out["pulse_reject_post_ms"] = np.float64(post * 1000.0)
    out["pulse_reject_n"] = np.int64(ons.size)

    out_stem = out_stem or f"{stem.split('_qc')[0]}_pr{tag}_qc{stem.split('_qc')[1]}"
    np.savez_compressed(RUNS / f"{out_stem}.npz", **out)
    lost = float(frac.mean())
    print(f"[reject] {stem} -> {out_stem}   -{pre * 1e3:g}/+{post * 1e3:g} ms "
          f"around {ons.size} pulses")
    print(f"[reject] clean time removed: {lost:.2%} "
          f"(chance for a {1000 / np.median(np.diff(ons)) if ons.size > 1 else 0:.0f} Hz train)")
    for d in n_before:
        print(f"  {d:<11}{n_before[d]:>7} -> {n_after[d]:>7} "
              f"({1 - n_after[d] / max(n_before[d], 1):>6.1%} rejected, "
              f"{lost:.1%} expected if unrelated to the pulse)")
    return out_stem


if __name__ == "__main__":
    stem = sys.argv[1] if len(sys.argv) > 1 else "P1_ANT2_stim_pb5i_qcfinalv2"
    if len(sys.argv) > 3:                       # stem PRE POST -> asymmetric
        reject(stem, pre_ms=float(sys.argv[2]), post_ms=float(sys.argv[3]))
    else:
        reject(stem, float(sys.argv[2]) if len(sys.argv) > 2 else 10.0)
