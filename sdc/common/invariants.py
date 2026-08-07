"""
sdc.common.invariants
---------------------
Cheap structural checks run at the npz-write boundary, where a wrong number becomes a result.

    from sdc.common.invariants import check_run
    check_run(dump, n_samp=..., fs=...)          # raises on violation

WHY THESE EXIST
  The main diagnostic used throughout this project is DISAGREEMENT BETWEEN DETECTORS, which is
  powerful but structurally blind: it cannot see an error in code all three depend on. Every
  late bug here was of exactly that kind, and each produced a plausible number rather than a
  crash:

    * clean_sec_on/off were computed by SCALING total clean time by the ON/OFF duty cycle,
      crediting channels masked during stimulation with up to 231 s of analysable time they
      never had -- which understated ON rates and overstated suppression, i.e. biased a finding
      in the direction that finding was reporting.
    * Delphos was left UNMASKED in the window merge while the other two were masked, inflating
      it ~20%. It read as a Delphos result.
    * A rate denominator divided 8 detections by 3.88 s of analysable time and reported the
      channel as the busiest of 226; its rank by raw count was 215th.

  The first two are caught by the checks below. They are deliberately structural -- arithmetic
  identities and bounds, not thresholds on plausibility -- because a check that needs tuning is
  another thing to get wrong.

WHAT THEY ARE NOT
  Not a correctness proof. They cannot tell you the detector is right, only that the book-keeping
  around it is self-consistent. Keep the frozen-output regression for the rest.
"""
import numpy as np


class InvariantError(AssertionError):
    """A structural check failed. The run is not trustworthy -- do not write it."""


def _fail(msg):
    raise InvariantError(msg)


def check_run(dump, n_samp=None, fs=None, tol=1e-6):
    """Validate a run dict before it is written. Missing keys are skipped, not assumed.

    dump    the mapping about to go to np.savez
    n_samp  samples on the detection grid, if known -- enables the index-bounds check
    """
    dets = [str(d) for d in np.asarray(dump.get("detectors", []))]
    seconds = float(dump["seconds"]) if "seconds" in dump else None
    fs = float(dump.get("fs", fs) or 0) or None

    # ---- detection indices lie inside the recording ------------------------------------
    for d in dets:
        idx = np.asarray(dump.get(f"{d}_idx", np.zeros(0, int)))
        chan = np.asarray(dump.get(f"{d}_chan", np.zeros(0, int)))
        if idx.size != chan.size:
            _fail(f"{d}: {idx.size} indices but {chan.size} channel labels")
        if idx.size and idx.min() < 0:
            _fail(f"{d}: negative sample index {idx.min()}")
        if n_samp is not None and idx.size and idx.max() >= n_samp:
            _fail(f"{d}: index {idx.max()} beyond the recording ({n_samp} samples). "
                  f"An arm run on a different array, or a window offset not applied.")
        n_chan = len(np.asarray(dump.get("names", [])))
        if n_chan and chan.size and (chan.min() < 0 or chan.max() >= n_chan):
            _fail(f"{d}: channel {chan.max()} outside 0..{n_chan-1}")

    # ---- analysable time cannot exceed the recording -----------------------------------
    for key in ("clean_sec_on", "clean_sec_off"):
        if key in dump and seconds is not None:
            v = np.asarray(dump[key], float)
            if v.size and (v.min() < -tol or v.max() > seconds + 1.0):
                _fail(f"{key}: {v.max():.1f}s analysable in a {seconds:.0f}s recording "
                      f"(min {v.min():.1f})")

    # ---- the two conditions PARTITION the recording ------------------------------------
    # This is the one that catches a scaled -- rather than measured -- split. Scaling satisfies
    # the bound above and fails here only if it is done per channel, so the check is the sum of
    # the durations, which is exact under any correct construction.
    if {"sec_on", "sec_off"} <= set(dump) and seconds is not None:
        tot = float(dump["sec_on"]) + float(dump["sec_off"])
        if abs(tot - seconds) > 1.0:
            _fail(f"sec_on + sec_off = {tot:.1f}s but the recording is {seconds:.0f}s")
    if {"clean_sec_on", "clean_sec_off", "clean_per_sec"} <= set(dump) and fs:
        exact = np.asarray(dump["clean_per_sec"], float).sum(axis=0) / fs
        got = np.asarray(dump["clean_sec_on"], float) + np.asarray(dump["clean_sec_off"], float)
        if got.size and np.abs(got - exact).max() > 1.0:
            _fail(f"clean_sec_on + clean_sec_off differs from the per-second counts by up to "
                  f"{np.abs(got - exact).max():.1f}s -- the split is SCALED, not measured.")

    # ---- the ON flag partitions each detector's detections ------------------------------
    for d in dets:
        if f"{d}_on" not in dump:
            continue
        on = np.asarray(dump[f"{d}_on"], bool)
        idx = np.asarray(dump.get(f"{d}_idx", np.zeros(0, int)))
        if on.size != idx.size:
            _fail(f"{d}_on has {on.size} flags for {idx.size} detections")

    # ---- stim segments lie inside the recording and do not overlap ----------------------
    if "on_runs" in dump:
        r = np.asarray(dump["on_runs"], float)
        if r.size:
            if r.ndim != 2 or r.shape[1] != 2:
                _fail(f"on_runs must be (n, 2), got {r.shape}")
            if (r[:, 1] <= r[:, 0]).any():
                _fail("on_runs contains an empty or reversed segment")
            if n_samp is not None and r.max() > n_samp + 1:
                _fail(f"on_runs reaches sample {r.max():.0f} beyond the recording ({n_samp})")
            if len(r) > 1 and (r[1:, 0] < r[:-1, 1]).any():
                _fail("on_runs segments overlap")
    return True
