"""
score_sim_detectors.py
----------------------
Score the three detectors against KNOWN spike times. Python port of
seeg_analysis/spike_detector_comparison/score_sim_detectors.m, plus the specificity/ROC/AUC
the MATLAB never had.

Reads ONLY sim_runs/*.npz -- never imports compare_spikes (that module is a script: importing
it would run a 600 s load and a ~5 min Delphos call as a side effect) and never re-runs a
detector. Everything here is arithmetic on stored detections.

TWO ACCOUNTINGS, because "sensitivity" means two different things and the difference matters:

  EVENT-BASED (ports the MATLAB). Greedy one-to-one match inside +/-TOL_MS, per channel,
  pooled across channels. recall = TP/n_true, precision = TP/n_detected, F1. There is no
  "true negative" here at all -- a detector that stays silent is not credited with anything,
  which is why the MATLAB reports precision instead of specificity. This is the headline.

  WINDOW-BASED (new, needed for specificity and AUC). Tile each channel into BIN_MS bins and
  score it as binary classification. This DOES define a TN, so specificity and an ROC exist --
  but read them with the imbalance in mind. It is printed on every run: at the default sim
  (2400 true spikes, 100 ms bins over 600 s x 16 ch = 96 000 bins) the ratio is ~39 negatives
  per positive, so a detector emitting fewer than ~2800 false-positive bins already scores
  specificity >= 0.97 and any ROC-AUC lands near 0.99 REGARDLESS OF WHETHER IT IS ANY GOOD.
  Window accounting also charges sub-bin jitter twice -- once as FP, once as FN -- so
  `fp_adjacent` is reported to show how much of FP is book-keeping rather than a spurious
  detection.

AUC IS PARTIAL, and labelled as such. A 5-point threshold sweep does not span FPR in [0,1];
anchoring the curve at (0,0) and (1,1) to get a "full" AUC is extrapolation, not measurement.
What is reported is the trapezoid over the SWEPT range, normalised by that range, with every
point annotated with the threshold that produced it.

    .venv\\Scripts\\python.exe score_sim_detectors.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from seeg._style import RED, BLUE, MUTED, recessive
from spike_match import match

HERE = Path(__file__).resolve().parent
SIM_RUNS = HERE / "sim_runs"   # inputs: one npz per (SNR, operating point), tracked in git
OUT = HERE / "figures" / "sim" / "_summary"   # aggregates ACROSS sim runs, so it sits
                               # beside the per-run folders rather than inside one

TOL_MS = 50.0          # event-based match radius; the MATLAB's w_ms
BIN_MS = 100.0         # window-based bin width (= 2 x TOL_MS, so a matched spike lands in
                       # its own bin or the neighbour, not three bins away)
EDGE_GUARD_SEC = 1.0   # ignore truth AND detections within this of either end -- the AR noise
                       # starts from zero initial conditions and every detector has filter
                       # transients there, so the edges score nobody fairly
N_AMP_BINS = 12        # MATLAB parity (score_sim_detectors.m section 3)

VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
def load_runs(directory=SIM_RUNS, pattern="*.npz"):
    """Every simulated run in `directory`, as dicts of spike TIMES IN SECONDS.

    Grouping keys come from the STORED fields, never from parsing the filename -- a renamed
    file must not silently become a different experiment."""
    directory = Path(directory)
    if not directory.is_dir():
        raise SystemExit(f"{directory} not found -- run compare_spikes.py with SIMULATE=True "
                         f"(or run_sim_suite.py) first.")
    runs = []
    for p in sorted(directory.glob(pattern)):
        z = np.load(p, allow_pickle=False)
        if "simulated" not in z.files:
            print(f"[skip] {p.name}: not a simulated run (no ground truth)")
            continue
        n_chan = len(z["names"])
        fs, truth_fs = float(z["fs"]), float(z["truth_fs"])
        seconds = float(z["seconds"])
        lo, hi = EDGE_GUARD_SEC, seconds - EDGE_GUARD_SEC

        def per_chan(idx, chan, rate):
            t = np.asarray(idx, float) / rate
            keep = (t >= lo) & (t <= hi)
            t, c = t[keep], np.asarray(chan)[keep]
            return [np.sort(t[c == k]) for k in range(n_chan)]

        truth = per_chan(z["truth_idx"], z["truth_chan"], truth_fs)
        amp_all = np.asarray(z["truth_amp"], float)
        t_all = np.asarray(z["truth_idx"], float) / truth_fs
        keep = (t_all >= lo) & (t_all <= hi)
        c_all = np.asarray(z["truth_chan"])[keep]
        amp = [amp_all[keep][c_all == k] for k in range(n_chan)]

        runs.append({
            "path": p, "names": [str(s) for s in z["names"]], "n_chan": n_chan,
            "fs": fs, "seconds": seconds, "scored_sec": max(hi - lo, 0.0),
            "snr": float(z["snr"]), "run_kind": str(z["run_kind"]),
            "point": str(z["run_point"]) if "run_point" in z.files else "",
            "sweep_detector": str(z["sweep_detector"]), "sweep_param": str(z["sweep_param"]),
            "sweep_value": float(z["sweep_value"]),
            "detectors": [str(s) for s in z["detectors"]],
            "truth": truth, "truth_amp": amp,
            "rates_per_min": np.asarray(z["rates_per_min"], float),
            "noise_std": np.asarray(z["noise_std"], float),
            "inband": np.asarray(z["inband_snr"], float),
            "inband_dets": [str(s) for s in z["inband_dets"]],
            "cfg": json.loads(str(z["sim_cfg_json"])),
            "det": {d: per_chan(z[f"{d}_idx"], z[f"{d}_chan"], fs) for d in
                    (str(s) for s in z["detectors"])}})
    if not runs:
        raise SystemExit(f"no simulated runs found in {directory}")

    # Runs built from DIFFERENT generator configs are different experiments and must never be
    # pooled into one curve. Keep the largest set and say exactly what was dropped.
    by_hash = {}
    for r in runs:
        by_hash.setdefault(str(np.load(r["path"], allow_pickle=False)["sim_cfg_hash"]),
                           []).append(r)
    if len(by_hash) > 1:
        keep = max(by_hash, key=lambda h: len(by_hash[h]))
        print(f"[warn] {directory.name} holds runs from {len(by_hash)} different generator "
              f"configs. Scoring only '{keep}' ({len(by_hash[keep])} runs); ignoring "
              + ", ".join(f"'{h}' ({len(v)})" for h, v in by_hash.items() if h != keep)
              + ".\n       Delete the stale ones, or move them aside, to silence this.")
        runs = by_hash[keep]
    return runs


# ----------------------------------------------------------------------
# accounting 1: events (the MATLAB)
# ----------------------------------------------------------------------
def event_scores(truth, truth_amp, det, seconds, tol_s=None):
    """Greedy +/-tol match per channel, counts pooled across channels.

    Returns pooled TP/FP/FN, recall/precision/F1, FP per channel-minute, the median absolute
    timing offset, per-channel rates, and the per-true-spike matched flag (amplitude-aligned)
    that feeds the P(detected) curve."""
    tol_s = (TOL_MS / 1000.0) if tol_s is None else tol_s
    n_chan = len(truth)
    tp = n_true = n_det = 0
    offs, hits, amps, ch_recall, true_rate, det_rate = [], [], [], [], [], []
    for c in range(n_chan):
        mt, _md, o = match(truth[c], det[c], tol_s)
        tp += int(mt.sum())
        n_true += truth[c].size
        n_det += det[c].size
        offs.append(-o)          # match returns truth - det; report DETECTION - TRUTH
        hits.append(mt.astype(float))
        amps.append(truth_amp[c])
        ch_recall.append(mt.mean() if mt.size else np.nan)
        true_rate.append(truth[c].size / seconds * 60.0)
        det_rate.append(det[c].size / seconds * 60.0)

    offs = np.concatenate(offs) if offs else np.zeros(0)
    recall = tp / max(n_true, 1)
    precision = tp / max(n_det, 1)
    return {"tp": tp, "fp": n_det - tp, "fn": n_true - tp, "n_true": n_true, "n_det": n_det,
            "recall": recall, "precision": precision,
            "f1": 2 * precision * recall / max(precision + recall, np.finfo(float).eps),
            "fp_per_chan_min": (n_det - tp) / (n_chan * seconds / 60.0),
            "med_off_ms": float(np.median(np.abs(offs)) * 1000) if offs.size else np.nan,
            # SIGNED median matters more than the absolute one: a detector that marks a
            # different point of the waveform shows up here as a systematic lag, and that lag
            # is then charged as lost sensitivity by any window-based accounting.
            # Measured on this sim: Janca +0.5 ms, Delphos +2.5 ms, Barkmeier +39 to +42 ms.
            "bias_ms": float(np.median(offs) * 1000) if offs.size else np.nan,
            "hit": np.concatenate(hits) if hits else np.zeros(0),
            "amp": np.concatenate(amps) if amps else np.zeros(0),
            "chan_recall": np.array(ch_recall), "true_rate": np.array(true_rate),
            "det_rate": np.array(det_rate)}


# ----------------------------------------------------------------------
# accounting 2: windows (specificity / ROC)
# ----------------------------------------------------------------------
def window_scores(truth, det, seconds, bin_s=None):
    """Tile each channel into `bin_s` bins and score as binary classification.

    `fp_adjacent` counts false-positive bins that TOUCH a positive bin: those are almost
    always one detection landing on the wrong side of a bin edge, charged here as both an FP
    and an FN. Reporting it separately keeps a jitter artefact from reading as a spurious
    detection."""
    bin_s = (BIN_MS / 1000.0) if bin_s is None else bin_s
    n_chan = len(truth)
    n_bins = int(np.floor(seconds / bin_s))
    edges = np.arange(n_bins + 1) * bin_s
    tp = fp = fn = tn = fp_adj = 0
    for c in range(n_chan):
        # times are absolute file seconds clipped to [EDGE_GUARD, seconds-EDGE_GUARD]; shift
        # so bin 0 starts at the first scored sample
        pos = np.histogram(truth[c] - EDGE_GUARD_SEC, bins=edges)[0] > 0
        hit = np.histogram(det[c] - EDGE_GUARD_SEC, bins=edges)[0] > 0
        tp += int((pos & hit).sum())
        fn += int((pos & ~hit).sum())
        fp += int((~pos & hit).sum())
        tn += int((~pos & ~hit).sum())
        near = pos.copy()
        near[1:] |= pos[:-1]
        near[:-1] |= pos[1:]
        fp_adj += int((~pos & hit & near).sum())
    total = tp + fp + fn + tn
    assert total == n_chan * n_bins, f"window bookkeeping lost bins: {total} != {n_chan*n_bins}"
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "fp_adjacent": fp_adj,
            "n_bins": n_bins * n_chan, "n_pos": tp + fn, "n_neg": fp + tn,
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "fpr": fp / max(tn + fp, 1)}


# ----------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """Wilson score interval -- correct near p=0 and p=1, where the normal approximation puts
    the bound outside [0, 1]. Avoids a scikit-learn/statsmodels dependency for four lines."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def pdetect_vs_amp(amp, hit, n_bins=N_AMP_BINS):
    """P(detected) in equal-width amplitude bins (MATLAB section 3: half-open, last bin closed)."""
    amp, hit = np.asarray(amp, float), np.asarray(hit, float)
    if amp.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, int), np.zeros(0), np.zeros(0)
    edges = np.linspace(amp.min(), amp.max(), n_bins + 1)
    ctr, p, n, lo, hi = [], [], [], [], []
    for b in range(n_bins):
        # half-open [lo, hi), except the LAST bin which is closed -- otherwise the single
        # largest spike falls outside every bin (MATLAB's `b==nBins` special case)
        last = b == n_bins - 1
        inb = (amp >= edges[b]) & ((amp <= edges[b + 1]) if last else (amp < edges[b + 1]))
        ctr.append(0.5 * (edges[b] + edges[b + 1]))
        k, m = int(hit[inb].sum()), int(inb.sum())
        n.append(m)
        p.append(k / m if m else np.nan)
        a, c = wilson(k, m)
        lo.append(a)
        hi.append(c)
    return np.array(ctr), np.array(p), np.array(n), np.array(lo), np.array(hi)


def partial_auc(x, y):
    """Trapezoid over the SWEPT range, normalised by that range. Not a [0,1] AUC and must
    never be presented as one -- see the module docstring."""
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    if x.size < 2 or x[-1] <= x[0]:
        return np.nan
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def step_ap(recall, precision):
    """Average precision as the step sum sum((r_i - r_{i-1}) * p_i), anchored at r=0."""
    o = np.argsort(recall)
    r, p = np.asarray(recall)[o], np.asarray(precision)[o]
    return float(np.sum(np.diff(np.concatenate([[0.0], r])) * p))


# ----------------------------------------------------------------------
def score_run(run):
    """Both accountings for every detector in one run."""
    out = {}
    for d in run["detectors"]:
        ev = event_scores(run["truth"], run["truth_amp"], run["det"][d], run["scored_sec"])
        wi = window_scores(run["truth"], run["det"][d], run["scored_sec"])
        out[d] = {"event": ev, "window": wi}
    return out


def report(runs):
    ops = sorted([r for r in runs if r["run_kind"] == "op"], key=lambda r: r["snr"])
    sweeps = [r for r in runs if r["run_kind"] == "sweep"]
    scored = {id(r): score_run(r) for r in runs}
    dets = sorted({d for r in runs for d in r["detectors"]},
                  key=lambda d: list(COLORS).index(d) if d in COLORS else 99)

    if not ops:
        raise SystemExit("no `op` runs found -- nothing to plot against SNR.")
    cfg = ops[0]["cfg"]
    print(f"--- ground-truth scoring: sim '{cfg['tag']}', {cfg['n_chan']} ch x "
          f"{cfg['dur_sec']:g}s, template sharpness {cfg['sharpness']:g} ---")
    print(f"    scored window excludes {EDGE_GUARD_SEC:g}s at each end "
          f"({ops[0]['scored_sec']:g}s of {ops[0]['seconds']:g}s kept)")

    # --- the imbalance, measured rather than asserted -------------------
    w0 = scored[id(ops[-1])][dets[0]]["window"]
    ratio = w0["n_neg"] / max(w0["n_pos"], 1)
    print(f"\n[imbalance] {BIN_MS:g} ms bins: {w0['n_pos']} positive vs {w0['n_neg']} negative "
          f"= {ratio:.0f}:1.")
    print(f"    A detector can emit {int(0.03 * w0['n_neg'])} false-positive bins and still "
          f"score specificity 0.97.")
    print(f"    Read specificity and ROC-AUC with that in mind; the event-based table below "
          f"is the headline.")

    # --- event-based table ----------------------------------------------
    print(f"\nEVENT-BASED (greedy +/-{TOL_MS:g} ms one-to-one match, pooled over channels)")
    print(f"{'SNR':>5} {'detector':<10} {'nTrue':>6} {'nDet':>7} {'TP':>6} {'recall':>7} "
          f"{'prec':>6} {'F1':>6} {'FP/ch/min':>10} {'|off|':>8} {'bias':>9}")
    for r in ops:
        for d in r["detectors"]:
            e = scored[id(r)][d]["event"]
            print(f"{r['snr']:>5.0f} {d:<10} {e['n_true']:>6} {e['n_det']:>7} {e['tp']:>6} "
                  f"{e['recall']:>7.3f} {e['precision']:>6.3f} {e['f1']:>6.3f} "
                  f"{e['fp_per_chan_min']:>10.2f} {e['med_off_ms']:>7.1f}ms "
                  f"{e['bias_ms']:>+8.1f}ms")

    print(f"\nWINDOW-BASED ({BIN_MS:g} ms bins)")
    print(f"{'SNR':>5} {'detector':<10} {'sens':>6} {'spec':>7} {'FPR':>7} {'FP':>7} "
          f"{'of which adjacent':>18}")
    for r in ops:
        for d in r["detectors"]:
            w = scored[id(r)][d]["window"]
            adj = w["fp_adjacent"] / max(w["fp"], 1)
            print(f"{r['snr']:>5.0f} {d:<10} {w['sensitivity']:>6.3f} {w['specificity']:>7.4f} "
                  f"{w['fpr']:>7.4f} {w['fp']:>7} {adj:>17.0%}")

    # --- timing-convention gate ------------------------------------------
    # A detector that marks a DIFFERENT POINT of the waveform is not less sensitive, it is
    # differently labelled -- but every window-based metric charges the lag as lost
    # sensitivity, and a Jaccard at the same tolerance charges it as disagreement.
    # Only a LARGE bias is a labelling-convention problem. High FP-adjacency on its own is
    # expected and harmless -- it just means a detector's few extra marks cluster near real
    # spikes (duplicates), which is a different thing entirely and must not be reported with
    # the same words.
    for r in ops[-1:]:
        for d in r["detectors"]:
            e, w = scored[id(r)][d]["event"], scored[id(r)][d]["window"]
            bias, adj = e["bias_ms"], w["fp_adjacent"] / max(w["fp"], 1)
            if abs(bias) <= 0.2 * TOL_MS:
                continue
            # Barkmeier's mechanism, from the source rather than inferred. mDetectSpike.m:332
            # returns chanPeaks(:,2) = `spikeI`, which is
            #     [spikeV spikeI] = max(EEG(newPeakI-20*c : newPeakI, Chan))
            # i.e. the POSITIVE peak inside a 20 ms LOOK-BACK from `newPeakI`, the minimum of
            # the 20-50 Hz filtered signal. So the reported time can only ever land in
            # [newPeakI-20 ms, newPeakI]: once that negative lobe falls more than 20 ms after
            # the true peak, the true peak is outside the search window and cannot be
            # reported. `trough_search_ms` is NOT involved -- it only feeds Lamp/Ramp/Ldur/Rdur.
            # (An earlier version of this message blamed trough_search_ms; that was wrong.)
            why = ("\n    mDetectSpike.m:332 reports `spikeI` = max over a 20 ms LOOK-BACK from "
                   "the 20-50 Hz\n    negative lobe, so once that lobe sits >20 ms after the "
                   "true peak the peak itself\n    is out of reach. The lag is set by waveform "
                   "shape, not amplitude -- it is stable\n    across SNR here, and moved 55 ms "
                   "when only the template's after-wave was changed."
                   ) if d == "Barkmeier" else ""
            print(f"\n[!] {d} has a SYSTEMATIC timing bias of {bias:+.1f} ms at SNR "
                  f"{r['snr']:g} ({adj:.0%} of its window\n    false positives are adjacent "
                  f"to a true bin, i.e. the same spike in the next bin).{why}\n    Its window "
                  f"sensitivity ({w['sensitivity']:.3f}) understates it; the event figure "
                  f"({e['recall']:.3f}) is\n    the fair one. A bias this size also inflates "
                  f"disagreement in ANY comparison run at a\n    {TOL_MS:g} ms tolerance, real "
                  f"data included.")

    # --- gates ----------------------------------------------------------
    top = ops[-1]
    for d in top["detectors"]:
        e = scored[id(top)][d]["event"]
        if e["recall"] < 0.2:
            ib = top["inband"][:, top["inband_dets"].index(d)] if d in top["inband_dets"] \
                else None
            extra = f" Its in-band SNR here is {np.median(ib):.1f}." if ib is not None else ""
            print(f"\n[!] {d} recall is only {e['recall']:.2f} at the HIGHEST SNR "
                  f"({top['snr']:g}).{extra}")
            if ib is not None and np.median(ib) > 1.5:
                print(f"    That is comfortably detectable signal, so this is a DETECTOR "
                      f"result, not a stimulus artefact.")
            else:
                print(f"    The stimulus may be out of this detector's band -- adjust "
                      f"SHARPNESS in sim_data.py before concluding anything.")
        if e["n_det"] == 0 and all(scored[id(r)][d]["event"]["n_det"] == 0 for r in ops):
            print(f"\n[!] {d} detected NOTHING at any SNR. Injected peaks span "
                  f"{top['truth_amp'][-1].max() if top['truth_amp'][-1].size else 0:.0f} uV "
                  f"downwards; check its amplitude threshold (Barkmeier TAMP=1200 is tuned on "
                  f"real data and exceeds this range) and use the sweep.")

    _fig_metrics(ops, scored, dets)
    _fig_pdetect(ops, scored, dets)
    _fig_rate_scatter(ops, scored, dets)
    _fig_per_channel_noise(ops, scored, dets)
    if sweeps:
        _fig_sweep_curves(sweeps, scored, ops)
        _fig_roc_pr(sweeps, scored)
    else:
        print(f"\n[note] no sweep runs in {SIM_RUNS.name}, so no ROC/PR figure. Run "
              f"run_sim_suite.py to produce them.")


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def _fig_metrics(ops, scored, dets):
    snrs = [r["snr"] for r in ops]
    panels = [("recall", "Recall (sensitivity)", "event"),
              ("precision", "Precision", "event"),
              ("f1", "F1", "event"),
              ("fp_per_chan_min", "False positives / channel-min", "event")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, (key, title, acc) in zip(axes, panels):
        for d in dets:
            y = [scored[id(r)][d][acc][key] if d in r["detectors"] else np.nan for r in ops]
            ax.plot(snrs, y, "-o", color=COLORS.get(d, MUTED), lw=1.6, label=d)
        if key == "recall":   # window accounting on the same axis, so the gap is visible
            for d in dets:
                y = [scored[id(r)][d]["window"]["sensitivity"] if d in r["detectors"] else np.nan
                     for r in ops]
                ax.plot(snrs, y, "--", color=COLORS.get(d, MUTED), lw=1.1, alpha=.7)
            ax.plot([], [], "--", color=MUTED, lw=1.1, label=f"window ({BIN_MS:g} ms bins)")
        ax.set_xlabel("SNR (peak / noise std)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.3)
        if key != "fp_per_chan_min":
            ax.set_ylim(0, 1.02)
        else:
            ax.set_yscale("symlog", linthresh=0.1)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Detector accuracy against ground truth", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "sim_metrics_vs_snr.png", dpi=130)
    print(f"\n[saved] sim_metrics_vs_snr.png")


def _fig_pdetect(ops, scored, dets):
    fig, ax = plt.subplots(figsize=(7.6, 5))
    for d in dets:
        amp = np.concatenate([scored[id(r)][d]["event"]["amp"] for r in ops
                              if d in r["detectors"]] or [np.zeros(0)])
        hit = np.concatenate([scored[id(r)][d]["event"]["hit"] for r in ops
                              if d in r["detectors"]] or [np.zeros(0)])
        if amp.size == 0:
            continue
        ctr, p, n, lo, hi = pdetect_vs_amp(amp, hit)
        ok = n > 0
        ax.plot(ctr[ok], p[ok], "-o", color=COLORS.get(d, MUTED), lw=1.5, ms=4, label=d)
        ax.fill_between(ctr[ok], lo[ok], hi[ok], color=COLORS.get(d, MUTED), alpha=.15, lw=0)
    ax.set_xlabel("True spike peak amplitude (uV)")
    ax.set_ylabel("P(detected)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
    ax.legend(frameon=False)
    ax.set_title(f"Detection probability vs injected amplitude\n"
                 f"pooled over the SNR sweep, {N_AMP_BINS} equal-width bins, Wilson 95% CI",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "sim_pdetect_vs_amp.png", dpi=130)
    print(f"[saved] sim_pdetect_vs_amp.png")


def _fig_rate_scatter(ops, scored, dets):
    fig, ax = plt.subplots(figsize=(6.4, 6))
    mx = 0.0
    for d in dets:
        x = np.concatenate([scored[id(r)][d]["event"]["true_rate"] for r in ops
                            if d in r["detectors"]] or [np.zeros(0)])
        y = np.concatenate([scored[id(r)][d]["event"]["det_rate"] for r in ops
                            if d in r["detectors"]] or [np.zeros(0)])
        if x.size == 0:
            continue
        ax.scatter(x, y, 20, color=COLORS.get(d, MUTED), alpha=.65, edgecolors="none", label=d)
        mx = max(mx, x.max(), y.max())
    ax.plot([0, mx], [0, mx], "--", color=MUTED, lw=1)
    ax.annotate("y = x", (mx * .82, mx * .86), color=MUTED, fontsize=9)
    ax.set_xlabel("True rate (spikes/min)")
    ax.set_ylabel("Detected rate (spikes/min)")
    ax.set_title("Per-channel rate recovery, pooled over the SNR sweep", fontsize=10)
    ax.grid(alpha=.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "sim_rate_scatter.png", dpi=130)
    print(f"[saved] sim_rate_scatter.png")


def _fig_per_channel_noise(ops, scored, dets):
    """Per-channel performance against that channel's own noise level.

    THE QUESTION THIS ANSWERS. Barkmeier normalises each 1-minute block by a SINGLE GLOBAL
    scalar (mDetectSpike.m:284, `scale = SCALE / median(mean(abs(EEG)))`, the median taken
    ACROSS CHANNELS), then tests half-wave amplitudes against a fixed TAMP. Janca fits a
    background model PER CHANNEL and Delphos normalises its TF plane per channel. So when
    channels differ in noise -- real P1 spans 7-44 uV -- Barkmeier applies effectively the
    same uV bar everywhere while the other two move theirs with the local background.

    Prediction if that matters: on channels noisier than the median, Barkmeier's bar is low
    RELATIVE to the local noise, so its false-positive rate should climb with channel noise
    faster than the other two. Spearman rho of (channel noise, per-channel FP rate) is the
    test; it is printed alongside so the figure is not read by eye alone.

    Only meaningful when the channels actually differ -- with EQUALISE_CHANNEL_SNR=True every
    channel has identical noise and this is a scatter of one x value.
    """
    from scipy.stats import spearmanr

    def _partial(x, y, z):
        """Spearman rho(x, y) with z partialled out.

        REQUIRED here, not optional. Channel noise comes from the AR pool in channel order
        while the rate ramp also runs with channel index, so the two are entangled
        (rho ~ -0.47 on the shipped 16-channel set). Raw rho(noise, recall) reported
        Barkmeier at +0.268 -- an apparent ADVANTAGE on noisy channels -- which is entirely
        that confound; the partial is -0.016."""
        a, b, c = (spearmanr(x, y).statistic, spearmanr(z, y).statistic,
                   spearmanr(x, z).statistic)
        den = np.sqrt(max((1 - b ** 2) * (1 - c ** 2), 1e-12))
        return (a - b * c) / den

    run = ops[len(ops) // 2]        # a mid-SNR level: not saturated, not floored
    ns = run["noise_std"]
    if ns.max() / max(ns.min(), 1e-9) < 1.2:
        print(f"\n[note] channels differ in noise by only {ns.max()/ns.min():.2f}x "
              f"(EQUALISE_CHANNEL_SNR=True?), so the per-channel normalisation view is "
              f"uninformative; skipping sim_per_channel.png")
        return

    fig, (ax_r, ax_f) = plt.subplots(1, 2, figsize=(13, 5.2))
    print(f"\nPER-CHANNEL vs CHANNEL NOISE at nominal SNR {run['snr']:g} "
          f"(channels span {ns.min():.0f}-{ns.max():.0f} uV)")
    print(f"  (rho is PARTIAL, channel rate controlled -- noise and rate are confounded at "
          f"rho {spearmanr(ns, run['rates_per_min']).statistic:+.2f})")
    print(f"  {'detector':<11}{'rho(noise, recall)':>20}{'rho(noise, FP/min)':>20}")
    for d in dets:
        if d not in run["detectors"]:
            continue
        e = scored[id(run)][d]["event"]
        act = run["rates_per_min"] > 0            # the 0-rate control has no recall
        rec = e["chan_recall"]
        # per-channel false positives, expressed per minute so channels are comparable
        fp_min = np.array([
            (run["det"][d][c].size - int(match(run["truth"][c], run["det"][d][c],
                                               TOL_MS / 1000.0)[0].sum()))
            / (run["scored_sec"] / 60.0) for c in range(run["n_chan"])])
        rt = run["rates_per_min"]
        rr = _partial(ns[act], rec[act], rt[act])
        rf = _partial(ns, fp_min, rt)
        print(f"  {d:<11}{rr:>20.3f}{rf:>20.3f}")
        col = COLORS.get(d, MUTED)
        ax_r.plot(ns[act], rec[act], "o", color=col, ms=6, alpha=.8, label=f"{d}  rho {rr:+.2f}")
        ax_f.plot(ns, fp_min, "o", color=col, ms=6, alpha=.8, label=f"{d}  rho {rf:+.2f}")
    for ax, lab in ((ax_r, "per-channel recall"), (ax_f, "per-channel false positives / min")):
        ax.axvline(np.median(ns), color=MUTED, ls="--", lw=1)
        ax.annotate("median channel\n(sets Barkmeier's global scale)", (np.median(ns), 0),
                    textcoords="offset points", xytext=(6, 12), fontsize=7, color=MUTED)
        ax.set_xlabel("channel noise sd (uV)  -- spike is 143 uV on every channel")
        ax.set_ylabel(lab)
        ax.grid(alpha=.3)
        ax.legend(frameon=False, fontsize=8)
    ax_r.set_ylim(-0.02, 1.02)
    fig.suptitle(f"Per-channel performance vs channel noise (spike size FIXED across channels)"
                 f"   (spike fixed, nominal SNR {run['snr']:g})", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "sim_per_channel.png", dpi=130)
    print(f"[saved] sim_per_channel.png")


def _fig_sweep_curves(sweeps, scored, ops):
    """Recall and precision against the SWEPT PARAMETER VALUE, one panel per knob.

    sim_roc_pr.png shows the same runs parametrically (FPR vs TPR, recall vs precision) with
    the threshold annotated on each point. That is the right shape for comparing operating
    points, but it cannot show whether a knob DOES ANYTHING -- a flat sweep collapses to a
    cluster of overlapping dots there. This figure puts the parameter on the x-axis, which is
    the only way to read 'moving this changes nothing'.

    A reference line marks each detector's recall at the operating point used for the SNR
    curves, so the sweep can be read against the run everything else is based on.
    """
    by = {}
    for r in sweeps:
        d = r["sweep_detector"]
        if d in r["detectors"]:
            by.setdefault((d, r["sweep_param"]), []).append(r)
    if not by:
        return
    keys = sorted(by)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.0 * len(keys), 4.6), squeeze=False)
    print(f"\nSWEEP CURVES (recall / precision vs the knob itself)")
    for ax, (d, param) in zip(axes[0], keys):
        rows = sorted(by[(d, param)], key=lambda r: r["sweep_value"])
        vals = [r["sweep_value"] for r in rows]
        rec = [scored[id(r)][d]["event"]["recall"] for r in rows]
        pre = [scored[id(r)][d]["event"]["precision"] for r in rows]
        col = COLORS.get(d, MUTED)
        ax.plot(vals, rec, "-o", color=col, lw=1.8, ms=6, label="recall")
        ax.plot(vals, pre, "--s", color=col, lw=1.4, ms=5, alpha=.65, label="precision")
        snr = rows[0]["snr"]
        same = [r for r in ops if r["snr"] == snr and d in r["detectors"]]
        if same:
            ax.axhline(scored[id(same[0])][d]["event"]["recall"], color=MUTED, ls=":", lw=1)
            ax.annotate("recall at the operating point", (vals[0],
                        scored[id(same[0])][d]["event"]["recall"]), fontsize=6, color=MUTED,
                        va="bottom")
        span = max(rec) - min(rec)
        ax.set_title(f"{d}.{param}   (recall spans {span:.02f} over "
                     f"{min(vals):g}-{max(vals):g})", fontsize=9, loc="left")
        ax.set_xlabel(f"{param}")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.3)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        recessive(ax)
        print(f"  {d}.{param:<9} recall " + " ".join(f"{v:g}->{x:.2f}" for v, x in zip(vals, rec))
              + f"   [span {span:.2f}]")
    axes[0][0].set_ylabel("event-based score")
    fig.suptitle(f"Does the knob do anything? Sweeps at nominal SNR "
                 f"{sweeps[0]['snr']:g}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "sim_sweep_curves.png", dpi=130)
    plt.close(fig)
    print(f"[saved] sim_sweep_curves.png")


def _fig_roc_pr(sweeps, scored):
    # Keyed on (detector, PARAM) -- grouping by detector alone silently concatenated
    # Barkmeier's TAMP, LD+RD and LS+RS sweeps into one nonsense curve whose x-axis ran
    # 0.5 -> 1200 across three different units.
    by_det = {}
    for r in sweeps:
        d = r["sweep_detector"]
        if d in r["detectors"]:
            by_det.setdefault((d, r["sweep_param"]), []).append(r)
    if not by_det:
        return

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.4))
    print(f"\nSWEPT OPERATING POINTS (partial estimates over the swept range only)")
    for (d, param), rows in by_det.items():
        rows = sorted(rows, key=lambda r: r["sweep_value"])
        col = COLORS.get(d, MUTED)
        fpr = [scored[id(r)][d]["window"]["fpr"] for r in rows]
        tpr = [scored[id(r)][d]["window"]["sensitivity"] for r in rows]
        rec = [scored[id(r)][d]["event"]["recall"] for r in rows]
        pre = [scored[id(r)][d]["event"]["precision"] for r in rows]
        vals = [r["sweep_value"] for r in rows]

        pauc, ap = partial_auc(fpr, tpr), step_ap(rec, pre)
        print(f"  {d}.{param} at SNR {rows[0]['snr']:g}: "
              + ", ".join(f"{v:g}->(rec {rr:.2f}, prec {pp:.2f})"
                          for v, rr, pp in zip(vals, rec, pre)))
        print(f"      partial AUC {pauc:.3f} over FPR {min(fpr):.4f}-{max(fpr):.4f}"
              f"  |  step-AP {ap:.3f} over recall {min(rec):.2f}-{max(rec):.2f}")

        # same detector, several knobs -> same colour, different dash, so the curves stay
        # attributable without inventing a fourth hue
        ls = ["-", "--", ":", "-."][sorted(p for _d, p in by_det if _d == d).index(param) % 4]
        ax_roc.plot(fpr, tpr, marker="o", ls=ls, color=col, lw=1.6, ms=5,
                    label=f"{d}.{param}  pAUC {pauc:.3f}")
        ax_pr.plot(rec, pre, marker="o", ls=ls, color=col, lw=1.6, ms=5,
                   label=f"{d}.{param}  AP {ap:.3f}")
        for x, y, v in zip(fpr, tpr, vals):
            ax_roc.annotate(f"{v:g}", (x, y), textcoords="offset points", xytext=(5, -3),
                            fontsize=7, color=col)
        for x, y, v in zip(rec, pre, vals):
            ax_pr.annotate(f"{v:g}", (x, y), textcoords="offset points", xytext=(5, -3),
                           fontsize=7, color=col)

    ax_roc.set_xlabel(f"False positive rate ({BIN_MS:g} ms bins)")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC over the swept range\nNOT a [0,1] AUC -- see the label", fontsize=10)
    ax_pr.set_xlabel("Recall (event-based)")
    ax_pr.set_ylabel("Precision (event-based)")
    ax_pr.set_title("Precision-recall over the swept range", fontsize=10)
    for ax in (ax_roc, ax_pr):
        ax.grid(alpha=.3)
        ax.legend(frameon=False, fontsize=8)
        ax.set_ylim(0, 1.02)
    fig.suptitle("Threshold sweep against ground truth (5 points per detector, one SNR)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "sim_roc_pr.png", dpi=130)
    print(f"[saved] sim_roc_pr.png")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else SIM_RUNS
    report(load_runs(directory))
