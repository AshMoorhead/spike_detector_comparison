"""
sdc.scoring.score_marks
-----------------------
Score the detectors against OUR OWN expert marks, per channel, per block, per rater.

    .venv\\Scripts\\python.exe -m sdc.scoring.score_marks

WHY PRECISION MEANS SOMETHING HERE AND DID NOT ON BIDS
  The BIDS annotators marked notable discharges, so an unmatched detection was "not on their list"
  rather than wrong, and precision could only be compared BETWEEN detectors. These blocks were
  marked EXHAUSTIVELY -- every discharge the rater saw, on every channel the viewer showed,
  including channels where the answer was "nothing here". So an unmatched detection really is a
  false positive and precision is a rate rather than a lower bound.

  Which is why the channel list comes from `blocks.json` and not from the marks file. A channel
  the rater viewed and found empty contributes only false positives; a channel that was never
  shown contributes nothing. Those two look identical in a marks file and mean opposite things,
  and dropping the empty ones would flatter every detector. `blocks.json` is what the viewer
  actually presented, so it is the only thing that can tell them apart.

ONLY `done` BLOCKS
  progress.json gates this. Scoring a partially marked block counts real discharges the rater had
  not reached yet as false positives.

THE STRICT PASS IS EXCLUDED
  `rater-AM-strict` re-marked P1 from scratch at a deliberately LOWER sensitivity, skipping
  marginal spikes. It is a different target, not a second attempt at this one, so it is not
  scored here and not used for tuning. Its overlap with the lenient pass is not a reliability
  coefficient and does not bound what a detector can achieve -- an earlier version of this
  module said it did, and that number was quoted as a ceiling in several places.

  Establishing a real ceiling would need two passes at the SAME criterion, or a second rater.
  Neither exists yet, so no ceiling is quoted anywhere in this repo.

"""
import collections
import csv
import json
from pathlib import Path

import numpy as np

from sdc.common.paths import RUNS, figdir
from sdc.common.spike_match import match

LABELS = Path(r"C:\Users\amoo0039\Documents\label-SEEG-data")
DETS = ("Janca", "Barkmeier", "Delphos")
TOL = 0.050
RATERS = ("rater-AM",)
                # Lenient only. `rater-AM-strict` exists on disk but is EXCLUDED from scoring
                # and tuning: it was re-marked from scratch at a deliberately lower sensitivity,
                # so it is a different target rather than a repeat of this one. Averaging or
                # comparing the two conflates "where the bar for a spike sits" with "did the
                # detector find it", and an earlier version of this module mistook their overlap
                # for a reliability ceiling.

# subject -> run stem. The block's `edf` field names the same recording each of these was built on.
STEMS = {"P1": "P1_pre_qcfinalv2", "P5": "P5_pre_qcfinalv2", "P8": "P8_ANT145_pre_qcfinalv2"}
COLORS = {"Janca": "#c8102e", "Barkmeier": "#0072b2", "Delphos": "#4a3aa7"}
MARKERS = {"P1": "o", "P5": "s", "P8": "^"}


def _blocks():
    d = json.loads((LABELS / "blocks.json").read_text(encoding="utf-8"))["blocks"]
    return d if isinstance(d, dict) else {b["block_id"]: b for b in d}


def _marks(rater, block):
    out = collections.defaultdict(list)
    with open(LABELS / "annotations" / rater / "marks.tsv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["block_id"] == block:
                out[r["channel"]].append(float(r["time_in_sec"]))
    return {c: np.sort(v) for c, v in out.items()}


def done_blocks(rater):
    p = LABELS / "annotations" / rater / "progress.json"
    j = json.loads(p.read_text(encoding="utf-8"))["blocks"]
    return [k for k, v in j.items() if v.get("status") == "done"]


def score_block(rater, block, tol=TOL):
    """Per-channel hits/recall/precision for one rater's completed block."""
    meta = _blocks()[block]
    t0 = float(meta["t_start"])
    t1 = t0 + float(meta["t_dur"])
    subj = meta["subject"]
    stem = STEMS[subj]
    marks = _marks(rater, block)

    z = np.load(RUNS / f"{stem}.npz", allow_pickle=False)
    names = [str(x) for x in z["names"]]
    fs = float(z["fs"])
    cps = np.asarray(z["clean_per_sec"], float)

    rows = []
    for ch in meta["channels"]:                      # every channel the viewer SHOWED
        c = ch["name"]
        mk = marks.get(c, np.zeros(0))
        if c not in names:
            print(f"  [skip] {c} not in {stem}")
            continue
        i = names.index(c)
        # detections are stored post-mask; if QC removed part of the window the rater still saw
        # and marked the raw data there, so recall would be penalised for samples never scanned
        clean = cps[int(t0):int(t1), i].sum() / fs
        rec = {"chan": c, "n_mark": mk.size, "stratum": ch.get("stratum"),
               "region": ch.get("region"), "clean_frac": clean / (t1 - t0),
               "mark_rate": mk.size / ((t1 - t0) / 60.0)}
        for d in DETS:
            t = z[f"{d}_idx"][z[f"{d}_chan"] == i] / fs
            t = np.sort(t[(t >= t0) & (t < t1)])
            hit = int(match(t, mk, tol)[1].sum()) if (mk.size and t.size) else 0
            rec[d] = {"n_det": t.size, "hit": hit,
                      "recall": hit / mk.size if mk.size else np.nan,
                      "prec": hit / t.size if t.size else np.nan,
                      "fp_rate": (t.size - hit) / ((t1 - t0) / 60.0)}
        rows.append(rec)
    rows.sort(key=lambda r: -r["n_mark"])
    return {"rater": rater, "block": block, "subj": subj, "stem": stem,
            "t0": t0, "t1": t1, "rows": rows}


def collect(tol=TOL):
    out = []
    for r in RATERS:
        for b in sorted(done_blocks(r)):
            out.append(score_block(r, b, tol))
    return out


def _tag(s):
    return s["subj"]


def report(sets):
    for s in sets:
        rows = s["rows"]
        mins = (s["t1"] - s["t0"]) / 60.0
        empt = [r for r in rows if r["n_mark"] == 0]
        print(f"\n{'=' * 96}\n{s['block']}   rater {s['rater']}\n"
              f"{s['stem']}  {s['t0']:.0f}-{s['t1']:.0f}s ({mins:.1f} min)  "
              f"{len(rows)} channels shown, {len(empt)} viewed-and-empty\n{'=' * 96}")
        print(f"  {'channel':<14}{'strat':<6}{'marks':>6}{'cln':>6}   " +
              "  ".join(f"{d[:4]:>6} rec/prec" for d in DETS))
        for r in rows:
            cells = []
            for d in DETS:
                rc, pc = r[d]["recall"], r[d]["prec"]
                cells.append(f"{'  -  ' if not np.isfinite(rc) else f'{rc:.2f}'}/"
                             f"{'  -  ' if not np.isfinite(pc) else f'{pc:.2f}'}")
            print(f"  {r['chan']:<14}{str(r['stratum'] or '')[:5]:<6}{r['n_mark']:>6}"
                  f"{r['clean_frac']:>6.2f}   " + "  ".join(f"{c:>16}" for c in cells))
        tm = sum(r["n_mark"] for r in rows)
        line = []
        for d in DETS:
            h = sum(r[d]["hit"] for r in rows)
            n = sum(r[d]["n_det"] for r in rows)
            line.append(f"{h / max(tm, 1):.3f}/{n and h / n:.3f}".rjust(16))
        print(f"\n  {'POOLED':<14}{'':<6}{tm:>6}{'':>6}   " + "  ".join(line))
        if empt:
            print(f"  viewed-and-empty channels ({len(empt) * mins:.0f} chan-min, no real spikes):")
            for d in DETS:
                fp = sum(r[d]["n_det"] for r in empt)
                print(f"    {d:<10}{fp:>5} FP  = {fp / (len(empt) * mins):>5.1f}/chan-min")


def figure(tol=TOL, outdir=None):
    import matplotlib.pyplot as plt

    sets = collect(tol)
    report(sets)
    out = Path(outdir) if outdir else figdir("real")
    out.mkdir(parents=True, exist_ok=True)

    # ---- per-block detail -------------------------------------------------------------------
    for s in sets:
        rows = s["rows"]
        x = np.arange(len(rows))
        fig, axes = plt.subplots(3, 1, figsize=(max(9.0, .58 * len(rows) + 3), 10.0), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 2, 1.3]})
        for ax, key, name in ((axes[0], "recall", "recall (sensitivity)"),
                              (axes[1], "prec", "precision")):
            for d in DETS:
                ax.plot(x, [r[d][key] for r in rows], "-o", ms=6, lw=1.4,
                        color=COLORS[d], label=d)
            ax.axhline(0.5, color="0.5", ls=":", lw=1.0)
            ax.set_ylim(-0.03, 1.03)
            ax.set_ylabel(name)
            ax.grid(axis="y", alpha=.3)
        axes[0].legend(frameon=False, fontsize=9, ncol=3)
        for i, r in enumerate(rows):
            if r["n_mark"] == 0:
                for ax in axes:
                    ax.axvspan(i - .5, i + .5, color="0.92", lw=0, zorder=0)
        axes[0].set_title(
            f"{s['block']}   rater {s['rater']}\n{s['stem']}, {s['t0']:.0f}-{s['t1']:.0f}s, "
            f"+-{tol * 1000:.0f} ms match.  Shaded = viewed and empty "
            f"(recall undefined, all detections are FPs).", fontsize=10, loc="left")

        ax = axes[2]
        ax.bar(x, [r["n_mark"] for r in rows], color="0.55", width=.6)
        for d in DETS:
            ax.plot(x, [r[d]["n_det"] for r in rows], "-o", ms=4, lw=1.2,
                    color=COLORS[d], alpha=.8)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylabel("count in window\n(bars = marks, lines = detections)")
        ax.set_xticks(x)
        ax.set_xticklabels([r["chan"] for r in rows], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("channel, ordered by number of expert marks")
        ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        f = out / f"marks_per_channel_{_tag(s)}.png"
        fig.savefig(f, dpi=130)
        plt.close(fig)
        print(f"[saved] {f}")

    # ---- cross-block summary: does the trend replicate? -------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5))
    for ax, key, name in ((axes[0, 0], "recall", "recall"), (axes[0, 1], "prec", "precision")):
        for s in sets:
            for d in DETS:
                xs = [r["mark_rate"] for r in s["rows"] if r["n_mark"] > 0]
                ys = [r[d][key] for r in s["rows"] if r["n_mark"] > 0]
                ax.plot(xs, ys, MARKERS[s["subj"]], ms=7, color=COLORS[d], alpha=.55,
                        mew=1.4, ls="none")
        ax.set_xscale("log")
        ax.set_xlabel("expert marks per minute on that channel")
        ax.set_ylabel(name)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=.3)
        ax.set_title(f"{name} vs channel activity", fontsize=10, loc="left")
    hand = [plt.Line2D([], [], color=COLORS[d], marker="o", ls="none", label=d) for d in DETS]
    hand += [plt.Line2D([], [], color="0.35", marker=MARKERS[p], ls="none", label=p)
             for p in ("P1", "P5", "P8")]
    axes[0, 0].legend(handles=hand, frameon=False, fontsize=8, ncol=2, loc="upper left")

    # A block scored by two raters must use ONE empty set, or the bar moves when only the labels
    # changed. The intersection is the conservative choice: empty by every rater who looked.
    by_block = collections.defaultdict(list)
    for s in sets:
        by_block[s["block"]].append(s)
    ax = axes[1, 0]
    blocks = sorted(by_block, key=lambda b: by_block[b][0]["subj"])
    tags = [by_block[b][0]["subj"] for b in blocks]
    w = 0.26
    for j, d in enumerate(DETS):
        v = []
        for b in blocks:
            ss = by_block[b]
            empt = set.intersection(*({r["chan"] for r in s["rows"] if r["n_mark"] == 0}
                                      for s in ss))
            s0 = ss[0]
            m = (s0["t1"] - s0["t0"]) / 60.0
            n = sum(r[d]["n_det"] for r in s0["rows"] if r["chan"] in empt)
            v.append(n / (len(empt) * m) if empt else np.nan)
        ax.bar(np.arange(len(blocks)) + (j - 1) * w, v, w, color=COLORS[d], label=d)
    ax.set_xticks(np.arange(len(blocks)))
    ax.set_xticklabels([f"{t}\n({len(set.intersection(*({r['chan'] for r in s['rows'] if r['n_mark'] == 0} for s in by_block[b])))} ch)"
                        for t, b in zip(tags, blocks)])
    ax.set_ylabel("false positives / chan-min")
    ax.set_title("On channels every rater viewed and found EMPTY\n"
                 "(every detection here is wrong)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.3)

    ax = axes[1, 1]
    for s in sets:
        for d in DETS:
            tm = sum(r["n_mark"] for r in s["rows"])
            h = sum(r[d]["hit"] for r in s["rows"])
            n = sum(r[d]["n_det"] for r in s["rows"])
            ax.plot(h / max(tm, 1), h / max(n, 1), MARKERS[s["subj"]], ms=12, color=COLORS[d],
                    mew=1.8)
            ax.annotate(_tag(s), (h / max(tm, 1), h / max(n, 1)), fontsize=7,
                        xytext=(6, -3), textcoords="offset points", color=COLORS[d])
    for f1 in (0.3, 0.4, 0.5, 0.6):                      # iso-F1 contours
        r = np.linspace(f1 / (2 - f1) + 1e-3, 1, 100)
        ax.plot(r, f1 * r / (2 * r - f1), color="0.8", lw=.8, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("pooled recall")
    ax.set_ylabel("pooled precision")
    ax.set_title("Pooled per block (grey = iso-F1 0.3/0.4/0.5/0.6)", fontsize=10, loc="left")
    ax.grid(alpha=.3)

    fig.suptitle(f"Detectors vs expert marks, {len(sets)} completed blocks, "
                 f"+-{tol * 1000:.0f} ms match", fontsize=12)
    fig.tight_layout()
    f = out / "marks_summary.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    print(f"[saved] {f}")
    return sets


if __name__ == "__main__":
    figure()
