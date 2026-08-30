"""
sdc.scoring.tune_grid
---------------------
Screen detector parameters against the marked blocks, then plot precision/recall.

    .venv\\Scripts\\python.exe -m sdc.scoring.tune_grid janca
    .venv\\Scripts\\python.exe -m sdc.scoring.tune_grid barkmeier

SCREENING BEFORE GRIDDING
  A full factorial over Janca's 17 parameters is not affordable and mostly not informative --
  most of them change filtering or timing rather than sensitivity. So this moves ONE parameter
  at a time away from the default and reports what happens. A knob that does nothing here
  cannot do anything in combination either, so the screen is what decides where a 2-D grid is
  worth spending.

WHAT IS BEING OPTIMISED, AND WHY IT IS NOT POOLED F1
  Pooled precision/recall is dominated by the few busy channels -- 74% of the marks sit on six
  of them. Macro F1 (per-channel F1, averaged over channels) is reported alongside, and empty
  channels count: silence on a channel the rater viewed and found empty scores 1, firing on it
  scores 0. That is the failure the pooled numbers cannot see.

THERE IS NO MEASURED CEILING HERE
  P1 was marked twice, but at DELIBERATELY DIFFERENT SENSITIVITIES -- the strict pass skips
  marginal spikes on purpose. Their 0.758 overlap at +-50 ms is therefore not intra-rater
  reliability and does not bound what a detector can reach; an earlier version of this note
  claimed it did. A real ceiling needs two passes at the SAME criterion, or a second rater.

  What IS usable: the overlap rises to 0.896 at +-100 ms, and that 14-point step is timing
  jitter on events both passes marked -- so +-50 ms is tight relative to how precisely a human
  places a mark, whatever the criterion was.
"""
import sys

import numpy as np

from sdc.common.paths import ROOT
from sdc.scoring.tune_marks import BARK_FIXED, _detect, load_block, score, BLOCK_REC
from sdc.scoring.score_marks import done_blocks, RATERS, TOL

OUT = ROOT / "figures" / "real" / "_tuning"
COLORS = {"janca": "#c8102e", "barkmeier": "#0072b2"}

# One-at-a-time screens. (label, kwargs) -- the first entry of each list is the current default.
SCREENS = {
    "janca": [
        ("repo default", {}),
        ("module default (fh=60, dec=200)", {"band_high": 60.0, "decimation": 200.0}),
        # The k1 ladder must reach the LOOSE end, not just the strict one. A screen that only
        # moves k1 upward measures the precision side of the trade and mistakes the recall it
        # happens to have for a ceiling. The BIDS sweep reaches recall 0.833 at k1=1.2, so the
        # real question is where recall PLATEAUS -- a plateau means marks the detector cannot
        # reach at ANY threshold, which is a statement about what it can see rather than about
        # where it is set.
        ("k1=1.2", {"k1": 1.2}),
        ("k1=1.6", {"k1": 1.6}),
        ("k1=2.0", {"k1": 2.0}),
        ("k1=2.6", {"k1": 2.6}),
        ("k1=3.0", {"k1": 3.0}),
        ("k1=4.5", {"k1": 4.5}),
        ("k1=5.5", {"k1": 5.5}),
        ("k3=0.5", {"k3": 0.5}),
        # winsize/noverlap: the background model window. Short windows give a noisy lognormal
        # fit on a quiet channel, the threshold dips, and it fires -- the mechanism behind the
        # empty-channel false positives. Defaults are 5*fs / 4*fs.
        ("winsize=10s", {"winsize": 10000, "noverlap": 8000}),
        ("winsize=20s", {"winsize": 20000, "noverlap": 16000}),
        ("winsize=2.5s", {"winsize": 2500, "noverlap": 2000}),
        ("band 10-60", {"band_high": 60.0}),
        ("band 10-80", {"band_high": 80.0}),
        ("band 10-40", {"band_high": 40.0}),
        ("band 20-50", {"band_low": 20.0}),
        ("pt=0.05", {"polyspike_union_time": 0.05}),
        ("pt=0.20", {"polyspike_union_time": 0.20}),
        ("buffering=600", {"buffering": 600.0}),
    ],
    "barkmeier": [
        # Baseline is now the REAL production config (std_coeff=4, trough=40, band 20-50),
        # imported via BARK_FIXED. An earlier version of this screen retyped those as
        # 3 / 20 / 10-60, so "trough=40 beats the default" was really "the default beats a
        # value nothing uses".
        ("production default", {}),
        ("TAMP=400", {"TAMP": 400.0}),
        ("TAMP=800", {"TAMP": 800.0}),
        ("TAMP=1600", {"TAMP": 1600.0}),
        ("TAMP=2000", {"TAMP": 2000.0}),
        ("TAMP=2600", {"TAMP": 2600.0}),
        # std_coeff is the PER-CHANNEL adaptive stage: thresh = -mean|fEEG| - k*std|fEEG|.
        # Same relative structure as Janca's threshold, so it is the knob most likely to
        # drive the quiet-channel behaviour -- and it had only ever been sampled at +-1.
        ("std_coeff=1", {"std_coeff": 1.0}),
        ("std_coeff=2", {"std_coeff": 2.0}),
        ("std_coeff=3", {"std_coeff": 3.0}),
        ("std_coeff=5", {"std_coeff": 5.0}),
        ("std_coeff=6", {"std_coeff": 6.0}),
        ("std_coeff=8", {"std_coeff": 8.0}),
        # trough_search_ms caps Ldur/Rdur at its own value, which is why LD/RD are inert.
        ("trough=20ms", {"trough_search_ms": 20.0}),
        ("trough=60ms", {"trough_search_ms": 60.0}),
        ("trough=80ms", {"trough_search_ms": 80.0}),
        # Lowering to zero is the only NON-CIRCULAR test of a gate: a survivor distribution
        # cannot show what was rejected, since everything in it passed.
        ("ALL shape gates off", {"LD": 0, "RD": 0, "LS": 0.0, "RS": 0.0}),
        ("slope LS/RS=8", {"LS": 8.0, "RS": 8.0}),
        ("width LD/RD=30", {"LD": 30, "RD": 30}),
        ("band 10-60", {"filter_spec": (10.0, 60.0, 1.0, 35.0)}),
    ],
}


def _run_one(det, data, kw, tol=TOL):
    """Score one parameter set on every block; returns per-block and pooled numbers."""
    per, chan_rows = {}, []
    for d in data:
        if det == "janca":
            from sdc.detect.janca_detect_spikes import detect_spikes as janca
            # THE REPO'S OPERATING POINT, not the module defaults: compare_spikes runs
            # JANCA = dict(dec=0, fl=10.0, fh=50.0). Screening around the module defaults
            # (fh=60, dec=200) measures a configuration that is never used -- and silently,
            # because "band 10-60" then comes back bit-identical to "default".
            p = {"k1": 3.65, "k3": 0.0, "band_low": 10.0, "band_high": 50.0,
                 "decimation": 0, **kw}
            out, _a, _b = janca(d["x"], d["fs"], **p)
            ch = np.asarray(out["chan"], int)
            t = np.asarray(out["pos"], float)
            dets = {c: np.sort(t[ch == i]) for i, c in enumerate(d["chans"])}
        else:
            from seeg import detect_spikes as bark
            p = {**BARK_FIXED, "TAMP": 1200.0, **kw}
            bark(d["rec"], None, post_mask_spikes=False, fill_bad_samples=False,
                 det_thresholds=[p["LS"], p["RS"], p["TAMP"], p["LD"], p["RD"]],
                 std_coeff=p["std_coeff"], trough_search_ms=p["trough_search_ms"],
                 filter_spec=p["filter_spec"])
            dets = {c: np.sort(np.asarray(d["rec"]["info"]["DetectedSpikes"][i], float) / d["fs"])
                    for i, c in enumerate(d["chans"])}
        per[d["subj"]] = score(dets, d["truth"], d["mins"], tol)
        from sdc.common.spike_match import match
        for c, tt in dets.items():
            m = d["truth"][c]
            hit = int(match(tt, m, tol)[1].sum()) if (tt.size and m.size) else 0
            chan_rows.append({"subj": d["subj"], "chan": c, "n_mark": m.size,
                              "mark_rate": m.size / d["mins"], "n_det": tt.size, "hit": hit})
    tp = sum(r["hit"] for r in chan_rows)
    nd = sum(r["n_det"] for r in chan_rows)
    nm = sum(r["n_mark"] for r in chan_rows)
    # MARKED-ONLY macro F1, kept separate from the empty-channel FP rate. The blended macro F1
    # is ~30% "how many empty channels the rater happened to be shown", which is a property of
    # the marking protocol rather than of the detector -- on this set it made a configuration
    # with the WORST detection quality look like the best overall.
    f_mk = []
    for r in chan_rows:
        if r["n_mark"]:
            den = 2 * r["hit"] + (r["n_det"] - r["hit"]) + (r["n_mark"] - r["hit"])
            f_mk.append(0.0 if den == 0 else 2 * r["hit"] / den)
    return {"prec": tp / nd if nd else np.nan, "recall": tp / nm if nm else np.nan,
            "marked_macro_f1": float(np.mean(f_mk)) if f_mk else np.nan,
            "macro_f1": float(np.mean([v["macro_f1"] for v in per.values()])),
            "empty_fp": float(np.nanmean([v["empty_fp_per_min"] for v in per.values()])),
            "rate": float(np.mean([v["det_per_chan_min"] for v in per.values()])),
            "per": per, "chans": chan_rows}


def run(det="janca", tol=TOL):
    blocks = [b for b in done_blocks(RATERS[0]) if b in BLOCK_REC]
    data = [load_block(b) for b in blocks]
    res = []
    for lab, kw in SCREENS[det]:
        r = _run_one(det, data, kw, tol)
        r["label"], r["kw"] = lab, kw
        res.append(r)
        print(f"  {lab:<22} P {r['prec']:.3f}  R {r['recall']:.3f}  "
              f"F1 {2 * r['prec'] * r['recall'] / max(r['prec'] + r['recall'], 1e-9):.3f}  "
              f"markedMacroF1 {r['marked_macro_f1']:.3f}  "f"emptyFP {r['empty_fp']:.2f}/min  "
              f"rate {r['rate']:.1f}", flush=True)
    return res


def figure(det="janca", tol=TOL):
    import matplotlib.pyplot as plt
    res = run(det, tol)
    OUT.mkdir(parents=True, exist_ok=True)
    col = COLORS[det]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))
    ax = axes[0]
    for f1 in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        r = np.linspace(f1 / (2 - f1) + 1e-3, 1, 200)
        ax.plot(r, f1 * r / (2 * r - f1), color="0.85", lw=.9, zorder=0)
        ax.annotate(f"F1={f1:g}", (0.98, f1 * 0.98 / (2 * 0.98 - f1)), fontsize=6.5,
                    color="0.6", ha="right")
    for i, r in enumerate(res):
        first = i == 0
        ax.scatter(r["recall"], r["prec"], s=150 if first else 70,
                   color="0.15" if first else col, marker="*" if first else "o",
                   zorder=4, edgecolor="white", linewidth=.8)
        ax.annotate(r["label"], (r["recall"], r["prec"]), fontsize=7,
                    xytext=(6, 4), textcoords="offset points", color="0.25")
    ax.set_xlabel("pooled recall")
    ax.set_ylabel("pooled precision")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.set_title("(a) Pooled, 3 marked blocks (star = current default)",
                 fontsize=10, loc="left")
    ax.grid(alpha=.3)

    ax = axes[1]
    for i, r in enumerate(res):
        first = i == 0
        ax.scatter(r["empty_fp"], r["marked_macro_f1"], s=150 if first else 70,
                   color="0.15" if first else col, marker="*" if first else "o",
                   zorder=4, edgecolor="white", linewidth=.8)
        ax.annotate(r["label"], (r["empty_fp"], r["marked_macro_f1"]), fontsize=7,
                    xytext=(6, 4), textcoords="offset points", color="0.25")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("false positives / chan-min on channels the rater found EMPTY")
    ax.set_ylabel("macro F1 over channels WITH marks")
    ax.set_title("(b) The two numbers, unblended: detection quality vs "
                 "firing on empty channels (up and left is better)",
                 fontsize=10, loc="left")
    ax.grid(alpha=.3)

    fig.suptitle(f"{det.capitalize()}: one-at-a-time parameter screen against expert marks "
                 f"(P1, P5, P8 baselines)", fontsize=12)
    fig.tight_layout()
    f = OUT / f"param_screen_{det}.png"
    fig.savefig(f, dpi=140)
    plt.close(fig)
    print(f"[saved] {f}")
    return res


if __name__ == "__main__":
    figure(sys.argv[1] if len(sys.argv) > 1 else "janca")
