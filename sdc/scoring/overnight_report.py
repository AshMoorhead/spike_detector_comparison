"""
sdc.scoring.overnight_report
----------------------------
Read-out for the sweeps written by `sdc.scoring.overnight`.

    .venv\\Scripts\\python.exe -m sdc.scoring.overnight_report

Reads whatever runs/sweeps/overnight_*.jsonl files exist, so it can be run mid-sweep.

THE NUMBER THAT MATTERS IS THE HELD-OUT ONE
  `lopo()` does what a person would do with the grid -- pick the best setting -- but on two
  patients and scores it on the third. That is the only estimate of what tuning would buy on a
  patient you have not marked. The in-sample best is reported beside it purely to show the gap;
  on Janca it was +0.009 in-sample and -0.006 held out.

THE SHAPE-GATE CONTROL
  Tightening LS/RS/LD/RD raises precision (0.500 -> 0.706 at the corners), but so does simply
  raising TAMP. The question is whether the morphology gates carry information the amplitude
  threshold does not, and the test is whether the shape points sit ABOVE the TAMP-only frontier
  at matched recall -- not whether they beat the default. `frontier_figure` plots both:
    - the `barkmeier` grid at default gates, which traces what amplitude alone can do
    - the `barkmeier_shape` grid, which adds morphology on top
  Shape points on the amplitude curve mean the gates are an expensive way to raise a threshold.
"""
import json
from pathlib import Path

import numpy as np

from sdc.common.paths import RUNS, FIGURES

OUT = RUNS / "sweeps"


def load(tag):
    p = OUT / f"overnight_{tag}.jsonl"
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def lopo(rows, default=None):
    """Tune on all-but-one patient, score on the held-out one. Returns (per-fold, mean gain)."""
    subs = sorted(rows[0]["per_patient"])
    out, gains = [], []
    for h in subs:
        best = max(rows, key=lambda r: np.mean([r["per_patient"][s] for s in subs if s != h]))
        v = best["per_patient"][h]
        dv = default["per_patient"][h] if default else np.nan
        out.append((h, best["params"], v, dv))
        gains.append(v - dv)
    return out, float(np.nanmean(gains))


def _fmt(p):
    return " ".join(f"{k}={v}" for k, v in p.items())


def report(tag, default_params=None):
    rows = load(tag)
    if not rows:
        print(f"\n### {tag}: no results yet")
        return
    print(f"\n### {tag}  ({len(rows)} settings)")
    d = None
    if default_params:
        m = [r for r in rows if r["params"] == default_params]
        d = m[0] if m else None
    print("  top 5 in-sample (marked macro F1):")
    for r in sorted(rows, key=lambda r: -r["marked_macro_f1"])[:5]:
        print(f"    {r['marked_macro_f1']:.3f}  P={r['prec']:.3f} R={r['recall']:.3f}"
              f" emptyFP={r['empty_fp']:.2f}   {_fmt(r['params'])}")
    if d:
        print(f"  default: {d['marked_macro_f1']:.3f}  P={d['prec']:.3f} R={d['recall']:.3f}"
              f" emptyFP={d['empty_fp']:.2f}")
        folds, gain = lopo(rows, d)
        print("  leave-one-patient-out:")
        for h, p, v, dv in folds:
            print(f"    hold {h}: {v:.3f} vs default {dv:.3f}  ({v - dv:+.3f})   {_fmt(p)}")
        print(f"    mean held-out gain: {gain:+.4f}"
              f"   <- {'tuning helps' if gain > 0.01 else 'no reliable gain'}")
    # best setting per empty-FP budget: the 'quiet channels stay quiet' objective
    print("  best marked macro F1 within an empty-FP budget:")
    for cap in (0.1, 0.5, 1.0, 2.0, 5.0):
        ok = [r for r in rows if r["empty_fp"] <= cap]
        if ok:
            b = max(ok, key=lambda r: r["marked_macro_f1"])
            print(f"    <={cap:>4} FP/min: {b['marked_macro_f1']:.3f}"
                  f"  P={b['prec']:.3f} R={b['recall']:.3f}   {_fmt(b['params'])}")


def frontier_figure(fname="shape_vs_amplitude.png"):
    """Do the shape gates beat raising TAMP at matched recall?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amp, shp = load("barkmeier"), load("barkmeier_shape")
    if not amp or not shp:
        print("[frontier] need both barkmeier and barkmeier_shape; skipping")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (yk, ylab) in zip(axes, [("prec", "precision"),
                                     ("empty_fp", "empty-channel FP / min")]):
        ax.scatter([r["recall"] for r in amp], [r[yk] for r in amp], s=18, c="#888",
                   label="amplitude only (std_coeff x TAMP x trough x band)")
        ax.scatter([r["recall"] for r in shp], [r[yk] for r in shp], s=26, c="#c0392b",
                   marker="^", label="+ shape gates (LS/RS/LD/RD)")
        # upper envelope of the amplitude-only cloud, binned on recall
        rr = np.array([r["recall"] for r in amp])
        vv = np.array([r[yk] for r in amp])
        bins = np.linspace(rr.min(), rr.max(), 12)
        ix = np.digitize(rr, bins)
        bx, by = [], []
        for b in range(1, len(bins) + 1):
            m = ix == b
            if m.sum():
                bx.append(rr[m].mean())
                by.append(vv[m].max() if yk == "prec" else vv[m].min())
        ax.plot(bx, by, "k--", lw=1.5, label="amplitude-only frontier")
        ax.set_xlabel("recall (pooled)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=.3)
    axes[1].set_yscale("symlog", linthresh=0.1)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Do Barkmeier's shape gates carry information amplitude does not?\n"
                 "red above the dashed line = yes; on the line = an expensive threshold",
                 fontsize=10)
    fig.tight_layout()
    p = FIGURES / "tuning" / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[frontier] {p}")


if __name__ == "__main__":
    report("janca", {"k1": 3.65, "k3": 0.0, "band_high": 50.0})
    report("barkmeier", {"std_coeff": 4.0, "TAMP": 1200.0, "trough_search_ms": 40.0,
                         "filter_spec": [20.0, 50.0, 1.0, 35.0]})
    report("barkmeier_shape", {"slope": 3.0, "dur": 8.0, "trough_search_ms": 40.0,
                               "TAMP": 1200.0})
    frontier_figure()
