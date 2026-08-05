"""
cond.py
-------
Restrict a run to stim-ON or stim-OFF, and give the downstream scripts a time base that is
honest about the fact that the result is GAPPY.

    COND=on  python spike_statistics.py runs/P1_stim.npz
    COND=off python evaluate_detectors.py runs/P1_stim.npz
    COND=all python evaluate_detectors.py runs/P1_stim.npz     # default, one segment

Selecting a condition is not just a mask on the detections. An intermittent stim protocol cuts
the 600 s window into alternating blocks, so the ON subset is a handful of separated segments,
not a shorter recording. Three things break if that is ignored, and all three break QUIETLY --
they produce a number, just the wrong one:

  * RATE. The denominator is the ON seconds, not 600. Off by 1/duty-cycle otherwise, which for
    a 1-in-5 protocol is a 5x error in the headline quantity.
  * ISI. An interval measured from the last spike before an OFF gap to the first one after it
    spans the gap and is pure artefact of the split. It is also large, so it lands in the tail
    and inflates the CV rather than being obviously wrong.
  * ANY BINNED ESTIMATE (split-half reliability, Fano, block stability). A bin straddling a
    boundary is part ON and part OFF, so it measures neither, and it is systematically LOW
    because only part of it is in the condition being counted.

So the unit here is the SEGMENT, not the window. `runs` are the contiguous stretches of the
selected condition; `isis` diffs within a segment and never across; `bins` only returns bins
that fit ENTIRELY inside one segment (and reports what that discarded). What is deliberately
NOT offered is a "concatenate the ON blocks into a continuous recording" mode: it would make
every one of the above work by construction, and every cross-boundary interval it invented
would be a lie about a gap that really happened.

`on_runs` is written by compare_spikes.py. Runs made before it existed (and every stim-free
baseline) fall back to a single segment covering the whole window, which is exactly right for
COND=all on a recording with no stim.
"""
import os

import numpy as np


class Selection:
    """The selected condition: which detections survive, and over what stretches of time.

    Attributes
      label    "all" | "on" | "off"
      T        total seconds in the condition -- the rate denominator
      runs     (n, 2) float array of [t0, t1) segment bounds, seconds from window start
      suffix   "" for all, "_on"/"_off" otherwise -- for figure filenames
    """

    def __init__(self, label, runs, T, keep, clean_sec=None):
        self.label, self.runs, self.T, self._keep = label, runs, float(T), keep
        self.suffix = "" if label == "all" else f"_{label}"
        # ANALYSABLE seconds per channel: T minus that channel's dilated artefact mask. On a
        # stim recording this is the difference between "the detectors went quiet" and "the
        # artefact mask covered the time they would have fired in". None on runs predating the
        # key; then `rate` falls back to wall-clock and says so.
        self.clean_sec = None if clean_sec is None else np.asarray(clean_sec, float)

    def rate(self, counts):
        """Per-channel rate in Hz, over ANALYSABLE time where that is known.

        `counts` is one count per channel. Channels with no analysable time give nan, not inf:
        a channel that was masked out entirely has no rate, and zero is the wrong answer
        because it reads as "silent" rather than "not measured"."""
        counts = np.asarray(counts, float)
        if self.clean_sec is None:
            return counts / self.T
        return np.divide(counts, self.clean_sec,
                         out=np.full(counts.shape, np.nan), where=self.clean_sec > 0)

    def keep(self, det):
        """Bool mask over the flat {det}_idx array: is this detection in the condition?"""
        return self._keep[det]

    # -- gap-aware primitives -------------------------------------------------------------
    def isis(self, t):
        """Intervals within `t` (sorted seconds), never crossing a segment boundary.

        A spike is assigned to the segment containing it, and diffs are taken per segment.
        Spikes outside every segment (there should be none after `keep`) are dropped."""
        t = np.asarray(t, float)
        if t.size < 2 or self.runs.shape[0] == 0:
            return np.zeros(0)
        if self.runs.shape[0] == 1:
            return np.diff(t)
        # segment id per spike: searchsorted on the flattened bounds gives an ODD index for a
        # time inside a segment (t0 <= x < t1) and an even one for a time in a gap.
        edges = self.runs.reshape(-1)
        pos = np.searchsorted(edges, t, side="right")
        inside = (pos % 2) == 1
        seg, tt = pos[inside] // 2, t[inside]
        if tt.size < 2:
            return np.zeros(0)
        d = np.diff(tt)
        return d[np.diff(seg) == 0]

    def bins(self, width):
        """(m, 2) bin bounds of `width` seconds, each lying entirely inside ONE segment.

        Partial bins at the end of a segment are dropped rather than shortened, so every bin is
        an estimate over the same amount of time -- which is the whole premise of comparing bin
        counts to each other."""
        out = []
        for t0, t1 in self.runs:
            n = int((t1 - t0) // width)
            if n:
                starts = t0 + width * np.arange(n)
                out.append(np.column_stack([starts, starts + width]))
        return np.vstack(out) if out else np.zeros((0, 2))

    def bin_counts(self, t, width):
        """Counts of `t` (sorted seconds) in each bin from `bins(width)`."""
        b = self.bins(width)
        if b.shape[0] == 0:
            return np.zeros(0, int)
        t = np.asarray(t, float)
        return (np.searchsorted(t, b[:, 1]) - np.searchsorted(t, b[:, 0])).astype(int)

    def describe(self):
        n = self.runs.shape[0]
        if self.label == "all" and n == 1:
            return f"whole window, {self.T:g}s"
        lens = np.diff(self.runs, axis=1).ravel()
        return (f"{self.label.upper()}: {self.T:.0f}s over {n} segment(s), "
                f"each {lens.min():.0f}-{lens.max():.0f}s")


def select(z, label=None):
    """Build a Selection from an open npz. `label` defaults to $COND, defaulting to 'all'."""
    label = (label or os.environ.get("COND", "all")).lower()
    if label not in ("all", "on", "off"):
        raise SystemExit(f"COND={label!r} -- expected all, on or off.")
    fs, T_all = float(z["fs"]), float(z["seconds"])
    detectors = [str(s) for s in z["detectors"]]

    if label == "all":
        runs = np.array([[0.0, T_all]])
        keep = {d: np.ones(z[f"{d}_idx"].size, bool) for d in detectors}
        clean = (z["clean_sec_on"] + z["clean_sec_off"]) if "clean_sec_on" in z.files else None
        return Selection(label, runs, T_all, keep, clean)

    # A condition split needs a stim recording. Saying so beats drawing empty axes.
    sec_on = float(z["sec_on"]) if "sec_on" in z.files else 0.0
    if sec_on <= 0:
        rec = str(z["rec_id"]) if "rec_id" in z.files else "this run"
        raise SystemExit(f"COND={label} but {rec} has no stim-ON time (sec_on=0). "
                         f"Only COND=all applies to a baseline recording.")

    if "on_runs" in z.files:
        on = np.asarray(z["on_runs"], float) / fs          # samples -> seconds
    else:
        raise SystemExit(
            "This run has no 'on_runs' key, so the ON segment boundaries are unknown and the "
            "gap-aware ISI/bin handling cannot be done. Re-run compare_spikes.py "
            f"(RECORDING={str(z['rec_id']) if 'rec_id' in z.files else '?'}) -- the Delphos "
            "call is cached, so it is fast.")
    runs = on if label == "on" else _complement(on, T_all)
    T = float(np.diff(runs, axis=1).sum()) if runs.size else 0.0
    if runs.shape[0] == 0:
        raise SystemExit(f"COND={label} selects no time in this run.")

    keep = {}
    for d in detectors:
        flag = z[f"{d}_on"].astype(bool)
        keep[d] = flag if label == "on" else ~flag
    clean = z[f"clean_sec_{label}"] if f"clean_sec_{label}" in z.files else None
    return Selection(label, runs, T, keep, clean)


def _complement(runs, T):
    """The gaps between `runs` inside [0, T) -- i.e. stim-OFF given the ON segments."""
    out, prev = [], 0.0
    for t0, t1 in runs:
        if t0 > prev:
            out.append((prev, t0))
        prev = max(prev, t1)
    if prev < T:
        out.append((prev, T))
    return np.array(out, float).reshape(-1, 2)
