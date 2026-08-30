"""
sdc.artefact.ratio_metrics
--------------------------
The paired ON/OFF core. Every artefact module builds on this, and every number in the
stimulation analysis comes out of it.

    .venv\\Scripts\\python.exe -m sdc.artefact.ratio_metrics            # the Q1 comparison

TWO RULES, BOTH LEARNED THE HARD WAY
  1. EVERYTHING IS PAIRED. A channel is only ever compared against itself. Barkmeier on P5
     reads 1.113 by ratio-of-medians and 0.824 paired -- opposite sides of "no effect", from
     the same detections, purely because the unpaired form takes the median over a different
     set of channels on each side. 94 of 226 channels on P1_stim have ZERO analysable ON
     time, so the two sides are never the same implant unless you make them.
  2. ONE CHANNEL SET FOR ALL THREE DETECTORS. Built by intersecting their usability masks, so
     a Janca-vs-Delphos difference is a difference in what they detected and not in which
     channels survived their own gates. `paired()` enforces this; nothing else has to.

The bootstrap resamples CHANNELS, and one fixed draw matrix is shared across every estimator
and every detector (`draws()`), which makes each of those comparisons paired too. Resampling
detections instead would ignore that a channel's detections are not independent of each other.

WHY BOTH RATIOS AND DIFFERENCES
  A channel going 0.10 -> 0.05 /min and one going 10 -> 5 /min share a ratio of 0.5 and differ
  a hundredfold in difference. Ratios weight quiet and busy channels equally; differences
  weight busy channels. Which one is the clinical quantity is not a question this module can
  settle, so it reports both and every estimator carries the null value it should return when
  nothing happened (1.0 for ratios, 0.0 for differences).

THE ZERO PROBLEM, AND WHY 'DROP' IS THE WRONG DEFAULT
  A channel with zero ON detections makes every ratio estimator undefined -- log10(0) on P5 is
  where this was found. Dropping those channels is NOT neutral: a channel silenced completely
  by stimulation is the strongest single piece of evidence that stimulation did something, and
  dropping it throws away exactly that, biasing the result toward "no effect". So the default
  is `zero="clip"`, a half-detection continuity correction, and the number of affected
  channels is always reported. `zero="drop"` exists to show what that choice costs.

  A channel too quiet in OFF is a different case and is dropped (MIN_OFF_RATE): with no
  baseline there is nothing to be relative to, and 0-vs-0 is not evidence of anything.
"""
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

from sdc.common import cond
from sdc.common.paths import RUNS

MIN_OFF_RATE = 0.2   # det/MIN a channel needs in OFF before its ratio means anything.
                     #
                     # A RATE, not a count. It was an absolute 3 detections, which is not a
                     # property of the channel at all -- it is a property of how long you
                     # happened to look at it. The block-paired design sees only the matched
                     # OFF windows (332 s on P1 against the full condition's 765 s, 840 s
                     # against 2902 s on P5), so the same "3 detections" silently became a
                     # 2.4x and 3.5x stricter bar there, dropping 20 and 31 extra channels for
                     # a reason having nothing to do with the channels. As a rate the two paths
                     # ask the same question and their channel counts now track each other
                     # (132/132 at 0, 100/98 at 0.5, against 118/98 under the old count rule).
                     #
                     # 0.2 AND NOT 1.0, deliberately. This gate exists to remove channels whose
                     # ratio is not defined well enough to mean anything -- it is NOT a quality
                     # filter, and it must not become one, because it conditions on the stim
                     # recording's own OFF periods. Raising it to 1.0 costs over half of P5's
                     # channels (111 -> 50) and walks its result from 0.94/1.00/0.82 to
                     # 1.42/1.30/1.55, which is selection doing the work, not stimulation.
                     # The estimate is flat from 0 to 0.5 and only moves above 0.75, so the
                     # danger is above this value, not below it -- and on P1 the matched window
                     # is 5.53 min, so achievable rates are quantised at 0.18/min and 0.2 means
                     # exactly ">= 2 detections". Reads as "one detection every five minutes":
                     # a definedness bar, which is all this is for.
                     # For deliberate quality selection use blocks.rate_gate(source=
                     # "baseline"), which is built from a separate recording and cannot select
                     # on the outcome.
N_BOOT = 4000
BOOT_SEED = 0


# ----------------------------------------------------------------------
# Paired extraction
# ----------------------------------------------------------------------
@dataclass
class Paired:
    """Per-channel ON/OFF rates and counts, on the one channel set every detector can use.

    Arrays are per KEPT channel, in the recording's own channel order. `rate` fields are
    det/MINUTE over ANALYSABLE time (wall clock minus that channel's dilated artefact mask),
    which is the only denominator that means anything on a stim recording.
    """
    rec: str
    names: list
    keep: np.ndarray                 # bool over all channels
    on_sec: np.ndarray               # analysable ON seconds, per kept channel
    off_sec: np.ndarray
    det: dict                        # {detector: {on_rate, off_rate, on_count, off_count}}
    n_zero_on: dict                  # {detector: how many kept channels had 0 ON detections}
    T_on: float
    T_off: float

    @property
    def n(self):
        return int(self.keep.sum())

    def data(self, d):
        return self.det[d]

    def take(self, idx):
        """A Paired restricted to channel positions `idx` -- used by the bootstrap and by the
        channel-subsampling curve. Scalars are carried through unchanged."""
        return Paired(self.rec, self.names, self.keep, self.on_sec[idx], self.off_sec[idx],
                      {d: {k: v[idx] for k, v in a.items()} for d, a in self.det.items()},
                      self.n_zero_on, self.T_on, self.T_off)


def _matched_windows(on_runs, off_runs, prefer="before", off_full=False):
    """Pair every ON block with an equal-duration OFF window adjacent to it.

    `prefer` fixes WHICH SIDE the OFF window is taken from, "before" or "after". A block with no
    room on that side is DROPPED, not matched on the other one. Falling back mixed the two
    directions within one file, which is a confound rather than a convenience:
    on P1 the first ON block starts at 6 s, so no 66 s OFF exists before it and under the old
    fallback it alone was matched FORWARD while the other five went backward. That one pair
    carries the file's strongest suppression (0.30/0.40/0.41), and dropping it moves Janca from
    0.858 to 1.033 -- across no-change. The side has to be chosen, and honoured.

    Raw counts are only interpretable when both sides span the same time, and on these files
    they do not by a wide margin (ON is 332s vs OFF 765s on P1). Rather than dividing the bias
    away -- which turns a count back into a rate and defeats the point of having a count
    estimator at all -- each ON block is matched against the OFF time immediately next to it.

    Adjacency is doing a second job: a recording that drifts over an hour would show up as an
    effect in any estimator that compares the whole of ON against the whole of OFF, and cannot
    do so here, because every comparison is local in time.

    Returns (on_windows, off_windows) as (m, 2) second arrays, or empty if nothing matched.
    """
    def inside(t0, t1):
        return any(a <= t0 and t1 <= b for a, b in off_runs)

    on_w, off_w = [], []
    for t0, t1 in on_runs:
        d = t1 - t0
        if off_full:
            # THE WHOLE adjacent OFF run, not a duration-matched slice of it. Legitimate here
            # because every estimator downstream is a RATE (count / analysable seconds), so the
            # two sides need not span equal time -- matching was only ever required for the
            # count-based estimator. It buys a much less noisy denominator: on P1 the OFF runs
            # are ~176 s against a 64 s ON block, and Delphos's rate differs 2.7x between the
            # first 64 s of an OFF period and the next 64 s, so a matched slice makes the
            # answer depend on WHICH minute is used.
            tol = 2.0
            if prefer == "after":
                cand = next(((a, b) for a, b in off_runs if abs(a - t1) <= tol), None)
            else:
                cand = next(((a, b) for a, b in off_runs if abs(b - t0) <= tol), None)
        else:
            cand = (t1, t1 + d) if prefer == "after" else (t0 - d, t0)
            if not (cand[0] >= 0 and inside(*cand)):
                cand = None
        if cand is not None:
            on_w.append((t0, t1))
            off_w.append(cand)
    return (np.array(on_w, float).reshape(-1, 2), np.array(off_w, float).reshape(-1, 2))


def _counts_in(times, chans, wins, n):
    """Detections per channel falling inside any of `wins`."""
    out = np.zeros(n)
    for t0, t1 in wins:
        sel = (times >= t0) & (times < t1)
        out += np.bincount(chans[sel], minlength=n)
    return out


def _runs_from_mask(m):
    """Contiguous True stretches of a per-second bool mask, as (k, 2) second bounds."""
    e = np.flatnonzero(np.diff(m.astype(np.int8))) + 1
    b = np.concatenate([[0] if m[0] else [], e, [m.size] if m[-1] else []]).astype(float)
    return b.reshape(-1, 2)


def paired(z, tmax=None, min_off_rate=MIN_OFF_RATE, on_sec_mask=None, prefer="before"):
    """Build a Paired from an open run npz.

    The kept channel set is the intersection over detectors of: measurable in ON, measurable
    in OFF, and an OFF rate of at least `min_off_rate`. Identical for all three by
    construction --
    see rule 2 in the module docstring.

    `on_sec_mask` overrides the recording's own stim structure with an arbitrary per-second
    ON/OFF split. That is what builds the stim-free null in `nullcheck.py`: impose a stim
    file's block pattern on a baseline recording, where the true answer is known to be "no
    effect", and see what each estimator returns. Everything downstream -- rates, analysable
    time, matched blocks -- is then computed identically to a real split, which is the point:
    the null has to travel the same code path as the result or it tests nothing.
    """
    names = [str(s) for s in z["names"]]
    dets = [str(s) for s in z["detectors"]]
    n, fs = len(names), float(z["fs"])

    if on_sec_mask is None:
        ON, OFF = cond.select(z, "on", tmax=tmax), cond.select(z, "off", tmax=tmax)
        if ON.clean_sec is None:
            raise SystemExit(f"{z['rec_id']}: no clean_sec_on -- re-run compare_spikes.py so "
                             f"rates use analysable time rather than wall clock.")
        on_clean, off_clean = ON.clean_sec, OFF.clean_sec
        T_on, T_off, on_runs, off_runs = ON.T, OFF.T, ON.runs, OFF.runs
        measurable = ON.measurable & OFF.measurable
        sel = {d: (ON.keep(d), OFF.keep(d)) for d in dets}
    else:
        if "clean_per_sec" not in z.files:
            raise SystemExit("an imposed ON/OFF split needs `clean_per_sec`, which this run "
                             "predates -- re-merge it rather than scaling clean time.")
        cps = z["clean_per_sec"]
        m = np.asarray(on_sec_mask, bool)[:cps.shape[0]]
        on_clean, off_clean = cps[m].sum(axis=0) / fs, cps[~m].sum(axis=0) / fs
        T_on, T_off = float(m.sum()), float((~m).sum())
        on_runs, off_runs = _runs_from_mask(m), _runs_from_mask(~m)
        # The same 20%-of-condition gate cond.Selection applies, so an imposed split is held
        # to the identical usability standard as a real one.
        measurable = (on_clean >= cond.Selection.MIN_CLEAN_FRAC * T_on) & \
                     (off_clean >= cond.Selection.MIN_CLEAN_FRAC * T_off)
        sel = {}
        for d in dets:
            s = np.clip((z[f"{d}_idx"] / fs).astype(int), 0, m.size - 1)
            sel[d] = (m[s], ~m[s])

    counts = {}
    keep = measurable.copy()
    for d in dets:
        on_c = np.bincount(z[f"{d}_chan"][sel[d][0]], minlength=n)
        off_c = np.bincount(z[f"{d}_chan"][sel[d][1]], minlength=n)
        counts[d] = (on_c, off_c)
        # A RATE over that channel's own analysable OFF time, so the bar does not move with
        # window length. See MIN_OFF_RATE.
        with np.errstate(invalid="ignore", divide="ignore"):
            keep &= (off_c / np.maximum(off_clean, 1e-9) * 60.0) >= min_off_rate

    on_w, off_w = _matched_windows(on_runs, off_runs, prefer=prefer)

    det, n_zero = {}, {}
    for d in dets:
        on_c, off_c = counts[d][0][keep], counts[d][1][keep]
        t, c = z[f"{d}_idx"] / fs, z[f"{d}_chan"]
        if tmax:
            within = t < float(tmax)
            t, c = t[within], c[within]
        mo = _counts_in(t, c, on_w, n)[keep]
        mf = _counts_in(t, c, off_w, n)[keep]
        # on_sec/off_sec ride along inside each detector's dict so that the bootstrap, which
        # indexes every array in the dict with one channel draw, keeps a channel's analysable
        # time attached to its own counts. An estimator needing time (the zero correction)
        # then cannot silently pair the wrong channel's denominator.
        det[d] = {"on_count": on_c.astype(float), "off_count": off_c.astype(float),
                  "on_rate": on_c / on_clean[keep] * 60.0,
                  "off_rate": off_c / off_clean[keep] * 60.0,
                  "on_sec": on_clean[keep], "off_sec": off_clean[keep],
                  "on_count_m": mo, "off_count_m": mf}
        n_zero[d] = int((on_c == 0).sum())

    p = Paired(str(z["rec_id"]) if "rec_id" in z.files else "?", names, keep,
               on_clean[keep], off_clean[keep], det, n_zero, T_on, T_off)
    p.matched_sec = float(np.diff(on_w, axis=1).sum()) if on_w.size else 0.0
    p.n_matched = int(on_w.shape[0])
    p.n_on_blocks = int(on_runs.shape[0])
    return p


def load(rec, tag="", **kw):
    z = np.load(RUNS / f"{rec}{tag}.npz", allow_pickle=False)
    return paired(z, **kw)


# ----------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------
@dataclass
class Estimator:
    name: str
    fn: Callable
    null: float          # the value it returns when stimulation did nothing
    unit: str
    note: str

    def __call__(self, a, zero="clip"):
        return self.fn(a, zero)


def _ratio(a, zero):
    """Per-channel ON/OFF rate ratio, with the zero-ON channels handled explicitly."""
    on, off = a["on_rate"].copy(), a["off_rate"]
    z = a["on_count"] == 0
    if zero == "drop":
        ok = ~z
        return (on[ok] / off[ok])[off[ok] > 0]
    if zero == "clip":
        # Half a detection over THAT CHANNEL'S OWN analysable time -- the standard continuity
        # correction. It keeps the channel in (a channel silenced by stimulation is evidence,
        # not a nuisance) and scales with how long the channel was actually observed, so a
        # briefly-observed channel is not handed an implausibly low rate.
        on[z] = 0.5 / np.maximum(a["on_sec"][z], 1e-9) * 60.0
        with np.errstate(invalid="ignore", divide="ignore"):
            r = on / off
        return r[np.isfinite(r)]
    raise ValueError(f"zero={zero!r}; expected 'clip' or 'drop'")


def _median_ratio(a, zero):
    r = _ratio(a, zero)
    return float(np.median(r)) if r.size else np.nan


def _hodges_lehmann_log(a, zero):
    """HL location of the per-channel log10 ratio: median of all pairwise means.

    More efficient than the median for near-symmetric data and far more robust than the mean,
    which is what a log ratio needs -- the distribution is roughly symmetric in log space and
    badly skewed in linear space."""
    r = _ratio(a, zero)
    r = r[r > 0]
    if r.size == 0:
        return np.nan
    x = np.log10(r)
    s = np.add.outer(x, x)[np.triu_indices(x.size)] / 2.0
    return float(10 ** np.median(s))


def _rate_diff(a, zero):
    """Paired rate difference in det/min. No zero problem -- 0 is a perfectly good rate."""
    return float(np.median(a["on_rate"] - a["off_rate"]))


def _count_diff(a, zero):
    """Paired raw count difference over the WHOLE of each condition, ON minus OFF.

    Kept deliberately naive and reported anyway, because it is the number anyone would reach
    for first. It is also wrong: ON and OFF span very different amounts of time (2.4x on P1,
    3.5x on P5), so it measures duration far more than it measures stimulation. Its job here
    is to show how large that error is, next to the matched version that fixes it.
    """
    return float(np.median(a["on_count"] - a["off_count"]))


def _count_diff_matched(a, zero):
    """Paired count difference on duration-matched, time-adjacent blocks. See
    `_matched_windows` -- equal time either side, and local in time so drift cannot pose as
    an effect. This is the count estimator to actually use."""
    return float(np.median(a["on_count_m"] - a["off_count_m"]))


def _pooled_ratio(a, zero):
    """Pool counts and analysable time across channels, then divide once.

    Included as the counter-example: pooling makes the busiest few channels the whole answer,
    and those are exactly the channels nearest the pathology -- so it is the estimator most
    likely to be measuring one shaft and reporting it as the implant."""
    on = a["on_count"].sum() / max(a["on_sec"].sum(), 1e-9)
    off = a["off_count"].sum() / max(a["off_sec"].sum(), 1e-9)
    return float(on / off) if off > 0 else np.nan


ESTIMATORS = [
    Estimator("median ratio", _median_ratio, 1.0, "ON/OFF",
              "scale-free; equal weight to quiet and busy channels"),
    Estimator("HL log ratio", _hodges_lehmann_log, 1.0, "ON/OFF",
              "as above, better-behaved location estimate in log space"),
    Estimator("rate difference", _rate_diff, 0.0, "det/min",
              "no zero problem; weights busy channels; clinician-readable units"),
    Estimator("count diff (matched)", _count_diff_matched, 0.0, "det",
              "equal time either side, blocks adjacent in time -- the count to use"),
    Estimator("count diff (naive)", _count_diff, 0.0, "det",
              "NOT time-corrected; shown to size the duration bias it suffers from"),
    Estimator("pooled ratio", _pooled_ratio, 1.0, "ON/OFF",
              "UNPAIRED -- dominated by the busiest channels; the counter-example"),
]


# ----------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------
def draws(n, n_boot=N_BOOT, seed=BOOT_SEED):
    """ONE resample matrix, shared by every estimator and every detector, so that all of
    those comparisons are paired rather than each carrying its own independent noise."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def ci(est, a, dr, zero="clip", q=(2.5, 97.5)):
    """(point, lo, hi) for one estimator on one detector's arrays."""
    point = est(a, zero)
    b = np.array([est({k: v[i] for k, v in a.items()}, zero) for i in dr])
    b = b[np.isfinite(b)]
    if b.size == 0:
        return point, np.nan, np.nan
    return point, float(np.percentile(b, q[0])), float(np.percentile(b, q[1]))


def contrast(est, a, b, dr, zero="clip"):
    """Paired difference between two detectors on the SAME channels and the SAME resamples.

    Returned on the estimator's own scale: a ratio of ratios for ratio estimators, a plain
    difference for difference estimators. Non-overlapping marginal CIs imply a real
    difference, but overlapping ones do not imply the absence of one -- so the comparison is
    done directly rather than being eyeballed off two intervals.
    """
    def f(d1, d2):
        v1, v2 = est(d1, zero), est(d2, zero)
        return v1 / v2 if est.null == 1.0 else v1 - v2

    point = f(a, b)
    s = np.array([f({k: v[i] for k, v in a.items()}, {k: v[i] for k, v in b.items()})
                  for i in dr])
    s = s[np.isfinite(s)]
    lo, hi = (np.percentile(s, [2.5, 97.5]) if s.size else (np.nan, np.nan))
    return point, float(lo), float(hi), not (lo < est.null < hi)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def report(p, zero="clip", n_boot=N_BOOT):
    """Every estimator x every detector for one recording, with CIs and contrasts."""
    dr = draws(p.n, n_boot)
    print(f"\n=== {p.rec}: {p.n} channels usable by ALL detectors "
          f"(ON {p.T_on:.0f}s / OFF {p.T_off:.0f}s wall clock)")
    print(f"    analysable time per channel: ON {np.median(p.on_sec):.0f}s  "
          f"OFF {np.median(p.off_sec):.0f}s (medians)  "
          f"-> the NAIVE count difference carries a "
          f"{np.median(p.off_sec) / max(np.median(p.on_sec), 1e-9):.2f}x duration bias against ON")
    print(f"    duration-matched blocks: {p.n_matched}/{p.n_on_blocks} ON blocks matched, "
          f"{p.matched_sec:.0f}s per side")
    zeros = {d: v for d, v in p.n_zero_on.items() if v}
    if zeros:
        print(f"    zero-ON channels (zero={zero!r}): " +
              ", ".join(f"{d} {v}" for d, v in zeros.items()))

    dets = list(p.det)
    for est in ESTIMATORS:
        print(f"\n  {est.name:<18} [{est.unit}, null={est.null:g}]  {est.note}")
        vals = {}
        for d in dets:
            pt, lo, hi = ci(est, p.det[d], dr, zero)
            vals[d] = pt
            flag = "" if (lo < est.null < hi) else "  *"
            print(f"      {d:<10} {pt:>9.3f}  [{lo:>7.3f}, {hi:>7.3f}]{flag}")
        for i, d1 in enumerate(dets):
            for d2 in dets[i + 1:]:
                pt, lo, hi, sig = contrast(est, p.det[d1], p.det[d2], dr, zero)
                print(f"      {d1:>9} vs {d2:<10} {pt:>7.3f} [{lo:.3f}, {hi:.3f}]"
                      f"  {'DIFFER' if sig else 'ns'}")
    return dr


if __name__ == "__main__":
    recs = sys.argv[1:] or ["P1_stim", "P5_stim"]
    for r in recs:
        report(load(r, tag=os.environ.get("RUN_TAG", "")))
