"""
polyspike_review.py
-------------------
Pull REAL polyspike candidates out of the recording and draw them with all three detectors'
marks overlaid, so the merge cutoff (compare_spikes.MERGE_MS) can be chosen by looking rather
than by optimising a number.

WHY THIS EXISTS, AND WHY IT IS NOT A TEST SUITE
  Moving the comparison to EVENT level means picking a refractory below which two marks are
  one event. On the simulation that is trivial -- the generator enforces a 200 ms floor, so
  anything under ~180 ms is provably safe. On real data there is no floor: polyspike
  components, repetitive discharges and separate spikes form a continuum, and no statistic
  can tell you where one event stops and the next begins. That is a judgement about the
  physiology, so this script's only job is to put the right examples in front of you.

  It also cannot be answered by agreement between detectors, because they disagree BY
  CONSTRUCTION: Delphos detects a time-frequency blob, so a polyspike run inside one blob is
  one detection and it has no sub-blob events to expose. Janca finds envelope maxima and marks
  every component. Measured on the 600 s P1 baseline at MERGE_MS=20, the fraction of
  inter-detection intervals under 50 ms was Janca 15.1%, Barkmeier 2.8%, Delphos 1.6% -- a
  counting convention, not a difference in what was found.

HOW EXAMPLES ARE CHOSEN (not at random -- that would mostly show isolated spikes)
  1. Candidate = any place where SOME detector has >= 2 marks within WINDOW_MS on one channel.
  2. Stratified by the gap between the closest pair, into GAP_BANDS. The cutoff decision lives
     on that axis, so each band needs examples; a random sample would be dominated by whatever
     gap is commonest and tell you nothing about the edges.
  3. Within a band, ranked by DISAGREEMENT -- the spread in how many marks the detectors place.
     Where all three agree there is nothing to decide.

WHAT TO ASK OF EACH PANEL
  Is this ONE epileptiform event with internal structure, or TWO discharges? Your answers
  across the bands set the cutoff. The companion figure (absorption curve) then tells you what
  that choice costs each detector.

Drawn at the file's NATIVE rate, not the 400 Hz detection axis: polyspike components live in
the fast band and decimation blurs exactly the thing being judged.

    .venv\\Scripts\\python.exe -m sdc.tools.polyspike_review
    .venv\\Scripts\\python.exe -m sdc.tools.polyspike_review 12      # 12 examples per band
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from seeg import read_edf_header, load_edf_segment, derive_montage, apply_montage
from seeg._style import RED, BLUE, MUTED, recessive

from sdc.common import cond

from sdc.common.paths import ROOT as HERE   # repo root, not this file's dir --
                                            # see sdc/common/paths.py
# Needs UNDER-merged detections (see the guard in main()): the live runs are at
# MERGE_MS=100, which has already collapsed the 20-300 ms pairs this reviews.
NPZ = (Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].endswith(".npz")
       else HERE / "archive" / "detections_merge20.npz")
# COND=on / COND=off restricts to the stim blocks of a stim recording, the same switch the
# evaluation scripts take. The candidate search then runs on the subset, and a pair straddling
# an ON/OFF boundary can never seed a candidate because that gap is not a real interval:
#     COND=on .venv\Scripts\python.exe -m sdc.tools.polyspike_review runs/P1_stim.npz 9
SUB = ""                 # filled from $COND at load time; "" for the whole window
OUT = HERE / "figures" / "real" / "polyspike_review"
                         # re-pointed per run in load_detections(), so a stim recording's
                         # examples never overwrite the baseline's

# The EDF comes from the npz, NOT from a constant here. Hardcoding the baseline path meant a
# stim run would have been drawn against baseline traces with no error anywhere -- the marks
# would land on the wrong signal and simply look wrong.
EDF = None
SECONDS = 600            # overwritten from the npz

WINDOW_MS = 800.0        # two marks closer than this on one channel make a candidate;
                         # must exceed the widest GAP_BAND or those pairs never seed
GAP_BANDS = [(20, 50), (50, 80), (80, 120), (120, 200), (200, 300),
             # The 300-500 ms region is NOT about the merge cutoff -- nothing sane merges
             # there. It is here because the ISI distribution has a secondary bump at
             # 350-450 ms in all three detectors (Janca 9.8%/9.0%, Barkmeier 9.8%/9.3%,
             # Delphos 8.4%/7.5%) after a dip at 200-300. ~2.5 Hz is repetitive-discharge
             # rate, so these are the candidates for "is that bump real rhythmic activity or
             # a detector artefact".
             (300, 400), (400, 500),
             # 500-700 is a STIM-ON band. In P1's ON blocks all three detectors grow a second
             # ISI peak there that is absent from the baseline and from the same file's OFF
             # blocks: Barkmeier 3.7% and 3.8% in the 600-650 and 650-700 ms bins against 0.9%
             # either side, Janca 3.2/3.1, Delphos 2.9/2.7. ~1.5 Hz. Whether that is a stim
             # rhythm or the detectors re-triggering on something periodic is exactly what
             # these traces are for.
             (500, 600), (600, 700)]
PER_BAND = 6             # examples drawn per band (override with argv[1])
MAX_MARKS = 8            # a 400 ms window with more marks than this from ONE detector
                         # is a train or an artefact, not a polyspike run -- see pick()
PAD_MS = 600.0           # trace shown either side of the candidate's midpoint; must
                         # leave headroom around the widest GAP_BAND or a 700 ms pair
                         # sits on the axis edge with no context on either side
CUTOFF_GRID = np.arange(20, 305, 5)     # for the absorption curve

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}


# ----------------------------------------------------------------------
def load_detections():
    global EDF, SECONDS, OUT, SUB
    if not NPZ.is_file():
        raise SystemExit(f"{NPZ.name} not found -- run compare_spikes.py first.")
    z = np.load(NPZ, allow_pickle=False)
    names = [str(s) for s in z["names"]]
    fs = float(z["fs"])
    dets = [str(s) for s in z["detectors"]]
    sel = cond.select(z)
    per = {}
    for d in dets:
        keep = sel.keep(d)
        idx, ch = z[f"{d}_idx"][keep], z[f"{d}_chan"][keep]
        per[d] = [np.sort(idx[ch == c] / fs) for c in range(len(names))]
    EDF = str(z["edf"])
    SECONDS = float(z["seconds"])
    SUB = sel.suffix
    # Per-run folder: the baseline's polyspike panels and a stim recording's are different
    # questions and must not share filenames.
    OUT = HERE / "figures" / "real" / NPZ.stem / "polyspike_review"
    return names, fs, dets, per, SECONDS, sel


def find_candidates(per, dets, n_chan, window_s, sel):
    """Every (channel, time) where some detector places >= 2 marks inside `window_s`.

    Returns one row per candidate: the closest pair's gap, the mark count per detector, and
    the midpoint. Overlapping candidates on a channel are collapsed to the earliest, so one
    long polyspike run yields one example rather than a dozen shifted copies."""
    rows = []
    for c in range(n_chan):
        seeds = []
        for d in dets:
            s = per[d][c]
            if s.size < 2:
                continue
            # sel.isis, not np.diff: under COND=on the pair straddling an OFF block is 180 s
            # apart in real time and would never qualify anyway, but the indices would still
            # be misaligned. Diffing per segment keeps gap and position consistent.
            gaps = sel.isis(s)
            if gaps.size != s.size - 1:          # a segment boundary fell inside this channel
                keepable = np.flatnonzero(np.diff(s) <= window_s)
                gaps_all = np.diff(s)
                for i in keepable:
                    if _same_segment(sel, s[i], s[i + 1]):
                        seeds.append((s[i], gaps_all[i]))
                continue
            for i in np.flatnonzero(gaps <= window_s):
                seeds.append((s[i], s[i + 1] - s[i]))
        if not seeds:
            continue
        seeds.sort()
        last_t = -np.inf
        for t0, gap in seeds:
            if t0 - last_t < window_s:        # same burst, already represented
                continue
            last_t = t0
            lo, hi = t0 - window_s, t0 + 2 * window_s
            counts = {d: int(((per[d][c] >= lo) & (per[d][c] <= hi)).sum()) for d in dets}
            if max(counts.values()) < 2:
                continue
            rows.append({"chan": c, "t": float(t0), "gap_ms": float(gap * 1000),
                         "counts": counts,
                         "spread": max(counts.values()) - min(counts.values()),
                         "mid": float(t0 + gap / 2)})
    return rows


def _same_segment(sel, ta, tb):
    """Are two times inside the SAME contiguous segment of the selected condition?"""
    e = sel.runs.reshape(-1)
    pa, pb = np.searchsorted(e, ta, "right"), np.searchsorted(e, tb, "right")
    return pa == pb and pa % 2 == 1


def _shaft(name):
    """Leading letters of a bipolar pair name -> the electrode shaft (B1_B2 -> B, R_8_R_9 -> R)."""
    import re as _re
    m = _re.match(r"^([A-Za-z]+)", name.replace("_", ""))
    return m.group(1)[0] if m else "?"


def pick(rows, bands, per_band, names=None):
    """Stratify by gap band, then take HALF by disagreement and HALF as typical cases.

    Ranking purely by disagreement was tried first and is a trap: the cases where detectors
    disagree most are ARTEFACTS, not polyspikes. The first run surfaced a ~50 Hz train of
    identical transients (Janca 18 marks, Barkmeier 1, Delphos 0), a step discontinuity, and
    a channel full of sharp downward spikes -- all of which survived MASK_ARTEFACTS. Artefacts
    maximise disagreement precisely because they are not spikes, so that ranking selects
    against the thing being judged.

    Two guards:
      * MAX_MARKS -- a 400 ms window holding more than this many marks from one detector is a
        train, not a polyspike run; whatever the right cutoff is, it is not decided by these.
      * half the slots go to the MEDIAN mark count in the band, so the panel shows what is
        typical alongside what is contentious.
    """
    def _take(ranked, k, names):
        """Greedy pick of k, spreading over ELECTRODE SHAFTS then channels.

        Without this the panels cluster on whatever few channels rank highest, which reads as
        'this only happens on lead B' when the truth is the opposite -- measured on the
        400-500 ms band, B was the RAREST shaft (17 candidates) and R the commonest (284),
        yet an unconstrained ranking picked B twice. A shaft-diverse sample is the only way
        the figure answers 'where does this happen'."""
        picked, seen_shaft, seen_chan = [], set(), set()
        for pass_no in (0, 1):                 # pass 0: new shafts only; pass 1: fill up
            for r in ranked:
                if len(picked) >= k:
                    break
                sh = _shaft(names[r["chan"]])
                if r["chan"] in seen_chan or (pass_no == 0 and sh in seen_shaft):
                    continue
                picked.append(r)
                seen_shaft.add(sh)
                seen_chan.add(r["chan"])
        return picked

    out = {}
    for lo, hi in bands:
        inb = [r for r in rows if lo <= r["gap_ms"] < hi
               and max(r["counts"].values()) <= MAX_MARKS]
        if not inb:
            out[(lo, hi)] = []
            continue
        n_dis = per_band // 2
        by_dis = _take(sorted(inb, key=lambda r: (-r["spread"], r["gap_ms"])), n_dis, names)
        chosen = {id(r) for r in by_dis}
        used = {r["chan"] for r in by_dis}
        med = np.median([max(r["counts"].values()) for r in inb])
        rest = sorted((r for r in inb if id(r) not in chosen and r["chan"] not in used),
                      key=lambda r: (abs(max(r["counts"].values()) - med), r["spread"]))
        out[(lo, hi)] = by_dis + _take(rest, per_band - len(by_dis), names)
    return out


# ----------------------------------------------------------------------
def draw_band(band, rows, load_window, fs_raw, names, dets, per, n_cols=3):
    if not rows:
        print(f"  {band[0]}-{band[1]} ms: no candidates")
        return None
    n = len(rows)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 2.9 * n_rows),
                             squeeze=False)
    pad = PAD_MS / 1000.0
    for ax, r in zip(axes.ravel(), rows):
        c = r["chan"]
        win, win_t0 = load_window(r["mid"])
        t0, t1 = r["mid"] - pad, r["mid"] + pad
        i0 = max(int(round((t0 - win_t0) * fs_raw)), 0)
        i1 = min(int(round((t1 - win_t0) * fs_raw)), win.shape[0])
        seg = win[i0:i1, c]
        tt = win_t0 + np.arange(i0, i1) / fs_raw
        ax.plot(tt, seg, lw=0.8, color="0.35")
        span = np.percentile(np.abs(seg - np.median(seg)), 99.5) * 2.6
        base = float(np.median(seg))
        for k, d in enumerate(dets):
            m = per[d][c]
            m = m[(m >= tt[0]) & (m <= tt[-1])]
            y = base + span * (0.62 - 0.12 * k)
            if m.size:
                ax.plot(m, np.full(m.size, y), "v", ms=6, color=COLORS.get(d, MUTED),
                        clip_on=False)
            ax.text(tt[0], y, f"{d[:4]} {r['counts'][d]} ", ha="right", va="center",
                    fontsize=6, color=COLORS.get(d, MUTED))
        ax.set_ylim(base - span, base + span)
        ax.set_xlim(tt[0], tt[-1])
        ax.set_yticks([])
        ax.set_title(f"{names[c]}  t={r['mid']:.2f}s   closest pair {r['gap_ms']:.0f} ms",
                     fontsize=8, loc="left")
        ax.set_xlabel("time (s)", fontsize=7)
        ax.tick_params(labelsize=6)
        recessive(ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    _c = "" if not SUB else f"  [stim {SUB[1:].upper()} only]"
    fig.suptitle(f"Real polyspike candidates, closest pair {band[0]}-{band[1]} ms   "
                 f"(ONE event with structure, or TWO discharges?){_c}", fontsize=11)
    fig.tight_layout()
    out = OUT / f"polyspike_{band[0]:03d}_{band[1]:03d}ms{SUB}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  {band[0]}-{band[1]} ms: {n} examples -> {out.name}")
    return out


def draw_absorption(per, dets, seconds, SEL_G):
    """What each candidate cutoff COSTS: the share of marks it absorbs, per detector.

    This is the other half of the decision -- the panels say what is physiologically one
    event, this says how much of each detector's output a given rule removes."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for d in dets:
        isi = np.concatenate([SEL_G.isis(s) for s in per[d] if s.size > 1]) * 1000
        n_marks = sum(s.size for s in per[d])
        absorbed = [float((isi < g).sum()) / n_marks for g in CUTOFF_GRID]
        ax.plot(CUTOFF_GRID, np.array(absorbed) * 100, lw=1.8, color=COLORS.get(d, MUTED),
                label=f"{d} ({n_marks} marks)")
        ax2.plot(CUTOFF_GRID, [(n_marks - a * n_marks) / (seconds / 60) for a in absorbed],
                 lw=1.8, color=COLORS.get(d, MUTED), label=d)
    for a in (ax, ax2):
        for v, lab in ((120, "Janca default"), (150, "sim value")):
            a.axvline(v, color=MUTED, ls=":", lw=1)
            a.annotate(lab, (v, a.get_ylim()[1]), fontsize=7, color=MUTED, rotation=90,
                       va="top", ha="right")
        a.set_xlabel("candidate cutoff (ms)")
        a.grid(alpha=.3)
        a.legend(frameon=False, fontsize=8)
        recessive(a)
    ax.set_ylabel("% of marks absorbed by the merge")
    ax.set_title("Cost of the cutoff, per detector", fontsize=10, loc="left")
    ax2.set_ylabel("surviving marks per minute (all channels)")
    ax2.set_title("What survives", fontsize=10, loc="left")
    fig.suptitle("Choosing MERGE_MS: what each cutoff removes", fontsize=11)
    fig.tight_layout()
    out = OUT / f"polyspike_absorption{SUB}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  absorption curve -> {out.name}")


# ----------------------------------------------------------------------
def main():
    _args = [a for a in sys.argv[1:] if not a.endswith(".npz")]
    per_band = int(_args[0]) if _args else PER_BAND
    names, fs, dets, per, seconds, sel = load_detections()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{NPZ.name}: {len(names)} channels, {seconds:g}s, detectors {dets}")
    print(f"[cond] {sel.describe()}   traces from {Path(EDF).name}")

    # A band entirely below the run's merge floor cannot be populated -- those pairs were
    # collapsed before this ever saw them. Skip such bands rather than exiting: the SLOW bands
    # (300-500 ms) sit well above any sane cutoff and are perfectly valid on a merged run,
    # even though the sub-100 ms bands need archive/detections_merge20.npz.
    min_isi = min(float(np.min(sel.isis(s))) * 1000
                  for d in dets for s in per[d] if sel.isis(s).size)
    usable = [b for b in GAP_BANDS if b[1] > min_isi]
    dropped = [b for b in GAP_BANDS if b[1] <= min_isi]
    print(f"min inter-detection interval {min_isi:.0f} ms")
    if dropped:
        print("[note] " + ", ".join(f"{lo}-{hi}" for lo, hi in dropped)
              + " ms cannot be populated at this merge floor -- those pairs were merged away.\n"
                "       Use archive/detections_merge20.npz for the sub-100 ms bands.")
    if not usable:
        raise SystemExit(f"every band sits below the {min_isi:.0f} ms merge floor; nothing to show.")

    rows = find_candidates(per, dets, len(names), WINDOW_MS / 1000.0, sel)
    print(f"\n{len(rows)} polyspike candidates (>=2 marks within {WINDOW_MS:g} ms on a channel)")
    for lo, hi in usable:
        n = sum(lo <= r["gap_ms"] < hi for r in rows)
        print(f"  closest pair {lo:>3}-{hi:<3} ms : {n:>5}")

    # Load only the seconds actually drawn. The full 600 s x 226 ch at 2 kHz is ~2.2 GB before
    # the montage copy, which is not worth holding for ~30 one-second excerpts (and would
    # fight a concurrent Delphos run for memory).
    hdr = read_edf_header(EDF)
    fs_raw = float(hdr["SampleRate"])
    print(f"\ndrawing at {fs_raw:g} Hz (detections are on the {fs:g} Hz axis), "
          f"loading only the windows shown")

    def load_window(t_mid):
        """Montaged excerpt around t_mid at the file's native rate -> (data, window t0)."""
        pad = PAD_MS / 1000.0
        r0 = max(int(np.floor(t_mid - pad)), 0) + 1          # 1-based inclusive records
        r1 = min(int(np.ceil(t_mid + pad)) + 1, int(seconds))
        rec = load_edf_segment(EDF, hdr, r0, r1)
        rec = apply_montage(rec, derive_montage(rec["info"]["SelectedSignals"], verbose=False),
                            verbose=False)
        if list(rec["info"]["SelectedSignals"]) != names:
            raise SystemExit("channel order in the EDF does not match detections.npz -- "
                             "re-run compare_spikes.py so the two agree.")
        return rec["data"], float(r0 - 1)

    for band, chosen in pick(rows, usable, per_band, names).items():
        draw_band(band, chosen, load_window, fs_raw, names, dets, per)
    draw_absorption(per, dets, seconds, sel)
    print(f"\nAll figures in {OUT}. Look at the bands in order and decide where a pair stops\n"
          f"being one event; the absorption curve says what that costs.")


if __name__ == "__main__":
    main()
