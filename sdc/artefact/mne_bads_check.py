"""
sdc.artefact.mne_bads_check
---------------------------
Check (d) of plans/polymorphic-wiggling-breeze.md: WHICH channels does condition B drop, and is
it dropping them for being bad or for being busy?

    .venv\\Scripts\\python.exe -m sdc.artefact.mne_bads_check

A bad-channel rule that removes the most active channels is selecting on activity, not quality
-- the same confound that made the artefact ladder look better than it was, where most of the
apparent improvement turned out to be excluding channels rather than cleaning data. So every
dropped channel is printed next to its BASELINE spike rate, measured on the paired pre
recording, which no artefact rule in this comparison touches.

`annotate_amplitude(peak=...)` is a sustained consecutive-sample GRADIENT test in uV/sample, not
an amplitude test -- see seeg.artefact_mne.mne_bad_channels. Thresholds here are therefore swept
on the gradThr scale (150 and 1000 uV/sample are the two rungs the windowed conditions use), so
B can be read directly against C.
"""
import numpy as np

from seeg import read_edf_header, load_edf_segment, derive_montage, apply_montage
from seeg.artefact_mne import mne_bad_channels

from sdc.common.paths import RUNS, figdir
from sdc.detect.recordings import edf_path, load_patient_montage

# uV/sample. The two gradThr rungs the windowed conditions use, plus a looser and a tighter
# bracket so the sweep shows whether the choice sits on a plateau or a slope.
PEAKS = (75.0, 150.0, 400.0, 1000.0, 2500.0)
PAIRS = {"P1_stim": "P1_pre_qcfinalv2", "P1_ANT2_stim": "P1_ANT2_pre_qcfinalv2",
         "P1_ANT2_pre": "P1_ANT2_pre_qcfinalv2"}
CFG = {"epochLengthSec": 2, "epochStepSec": 2, "stimHz": None}


def _load_native(rec_id, patient=None):
    """The montaged NATIVE-rate array, i.e. the signal QC actually scores (pre-median).

    Uses the CLINICAL montage, same as compare_spikes. It must: the bad-channel lists written
    here are read back by name at run time, so a different channel set here would either fail
    the name check or -- worse -- silently name channels the run does not have.
    """
    edf, _ = edf_path(rec_id)
    h = read_edf_header(edf)
    r = load_edf_segment(edf, h, 1, int(h["NumDataRecords"]))
    pat = patient or rec_id.split("_")[0]
    mont = load_patient_montage(pat) or derive_montage(r["info"]["SelectedSignals"])
    return apply_montage(r, mont, verbose=False)


def _baseline_rate(run_stem, names):
    """Janca detections per clean channel-minute on the paired baseline run."""
    p = RUNS / f"{run_stem}.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    zn = [str(s) for s in z["names"]]
    cps = np.asarray(z["clean_per_sec"], float)
    fs = float(z["fs"])
    mins = cps.sum(axis=0) / fs / 60.0
    chan = np.asarray(z["Janca_chan"], int)
    cnt = np.bincount(chan, minlength=len(zn)).astype(float)
    rate = np.where(mins > 0, cnt / np.maximum(mins, 1e-9), np.nan)
    lut = dict(zip(zn, rate))
    return np.array([lut.get(n, np.nan) for n in names])


def run(recs=tuple(PAIRS)):
    for rec in recs:
        d = _load_native(rec)
        names = [str(n) for n in d["info"]["SelectedSignals"]]
        x = np.asarray(d["data"], float)
        base = _baseline_rate(PAIRS[rec], names)
        # each channel's own sustained gradient, for reference against the thresholds
        g = np.percentile(np.abs(np.diff(x, axis=0)), 99.0, axis=0)

        print(f"\n=== {rec}  ({len(names)} ch, fs {d['info']['SampleRate']:g} Hz) ===")
        print(f"  p99 |diff| across channels: median {np.median(g):.0f}, "
              f"range {g.min():.0f}-{g.max():.0f} uV/sample")
        print(f"  {'peak uV/samp':>13}{'dropped':>9}{'%':>7}   "
              f"{'median base rate kept':>22}{'dropped':>10}")
        for pk in PEAKS:
            qc = mne_bad_channels(d, CFG, peak_uv=pk)
            idx = np.asarray(qc["features"]["mne_bad_idx"], int)
            drop = np.zeros(len(names), bool)
            drop[idx] = True
            kb = np.nanmedian(base[~drop]) if base is not None and (~drop).any() else np.nan
            db = np.nanmedian(base[drop]) if base is not None and drop.any() else np.nan
            print(f"  {pk:>13.0f}{drop.sum():>9}{drop.mean() * 100:>6.1f}%"
                  f"{kb:>22.2f}{db:>10.2f}")
            if drop.any() and drop.sum() <= 12:
                print("        " + ", ".join(np.array(names)[drop]))
        if base is not None:
            print("  -> if the dropped median rate EXCEEDS the kept median, the rule is "
                  "selecting on activity")


if __name__ == "__main__":
    run()


def traces(rec="P1_stim", peak_uv=150.0, secs=10.0, t0=None, fname=None):
    """Eyeball two dropped and two retained channels: are the dropped ones actually bad?

    The threshold sweep above shows WHAT gets dropped and that it is not an activity confound.
    It cannot show whether the dropped channels are genuinely unusable, which is the thing that
    decides whether condition B is defensible. Only looking does that.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _load_native(rec)
    names = np.array([str(n) for n in d["info"]["SelectedSignals"]])
    fs = float(d["info"]["SampleRate"])
    x = np.asarray(d["data"], float)
    qc = mne_bad_channels(d, CFG, peak_uv=peak_uv)
    drop = np.zeros(len(names), bool)
    drop[np.asarray(qc["features"]["mne_bad_idx"], int)] = True
    if not drop.any():
        print(f"[traces] nothing dropped at peak={peak_uv:g}")
        return
    g = np.percentile(np.abs(np.diff(x, axis=0)), 99.0, axis=0)
    # worst two dropped, and two kept sitting at the median gradient (typical, not cherry-picked)
    show = list(np.flatnonzero(drop)[np.argsort(-g[drop])][:2])
    kept = np.flatnonzero(~drop)
    show += list(kept[np.argsort(np.abs(g[kept] - np.median(g[kept])))][:2])

    n = int(secs * fs)
    s0 = int((t0 if t0 is not None else (x.shape[0] / fs) * 0.5) * fs)
    s0 = max(0, min(s0, x.shape[0] - n))
    t = np.arange(n) / fs

    fig, axes = plt.subplots(len(show), 1, figsize=(13, 2.1 * len(show)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), show):
        ax.plot(t, x[s0:s0 + n, c], lw=.6,
                color="#c0392b" if drop[c] else "#2c3e50")
        ax.set_ylabel(f"{names[c]}\n{'DROPPED' if drop[c] else 'kept'}", fontsize=8)
        ax.text(.995, .93, f"p99 |diff| {g[c]:.0f} uV/sample", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="0.35")
        ax.grid(alpha=.25)
    np.atleast_1d(axes)[-1].set_xlabel("s")
    fig.suptitle(f"{rec}: channels dropped by annotate_amplitude at peak={peak_uv:g} uV/sample\n"
                 f"red = dropped, dark = retained (typical gradient)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = figdir("real") / (fname or f"mne_bads_traces_{rec}_p{peak_uv:g}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[saved] {out}")


def traces_ab(rec="P1_stim", peaks=(150.0, 1000.0), min_duration=0.002, secs=12.0,
              t0=None, n_kept=4, fname=None):
    """Every channel B1 and B2 drop, overlaid on the raw, colour-coded by which one drops it.

    Channels here span three orders of magnitude (150,000 uV on a stimulating contact against
    ~100 uV on a good one), so raw stacking is unreadable and a shared axis is worse. Each trace
    is therefore divided by its OWN robust scale (MAD) and offset -- shape is comparable, height
    is not, and the true gradient is printed per row so the discarded information is still on
    the figure.

    `min_duration` defaults to 2 ms, not MNE's 5 ms: at 5 ms a channel swinging +-150 mV is not
    flagged at any threshold above ~400 uV/sample, because pulsatile artefact has huge gradients
    at pulse edges and small ones between, so it never stays above threshold for 10 consecutive
    samples. See seeg.artefact_mne.mne_bad_channels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _load_native(rec)
    names = np.array([str(n) for n in d["info"]["SelectedSignals"]])
    fs = float(d["info"]["SampleRate"])
    x = np.asarray(d["data"], float)
    g = np.percentile(np.abs(np.diff(x, axis=0)), 99.0, axis=0)

    drops = []
    for pk in peaks:
        q = mne_bad_channels(d, CFG, peak_uv=pk, min_duration=min_duration)
        m = np.zeros(len(names), bool)
        m[np.asarray(q["features"]["mne_bad_idx"], int)] = True
        drops.append(m)
    loose, tight = drops[0], drops[-1]        # peaks[0] is the lower (more permissive) threshold

    both = loose & tight
    only_loose = loose & ~tight
    kept = np.flatnonzero(~loose)
    ref = kept[np.argsort(np.abs(g[kept] - np.median(g[kept])))][:n_kept]
    order = list(np.flatnonzero(both)) + list(np.flatnonzero(only_loose)) + list(ref)
    if not order:
        print("[traces_ab] nothing dropped")
        return

    n = int(secs * fs)
    s0 = int((t0 if t0 is not None else (x.shape[0] / fs) * 0.5) * fs)
    s0 = max(0, min(s0, x.shape[0] - n))
    t = np.arange(n) / fs

    fig, ax = plt.subplots(figsize=(14, 0.42 * len(order) + 2.4))
    for row, c in enumerate(order):
        seg = x[s0:s0 + n, c]
        scale = np.median(np.abs(seg - np.median(seg))) or 1.0
        col, lab = ("#8e1b1b", "B1+B2") if both[c] else \
                   (("#e07b39", "B1 only") if only_loose[c] else ("#2c3e50", "kept"))
        ax.plot(t, seg / (6 * scale) - row, lw=.5, color=col)
        ax.text(-0.006, -row, f"{names[c]}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7, color=col)
        ax.text(1.004, -row, f"{g[c]:,.0f}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.5, color="0.45")
    ax.set_yticks([])
    ax.set_xlabel("s")
    ax.set_xlim(t[0], t[-1])
    ax.text(1.004, 1.012, "p99 |diff|\nuV/sample", transform=ax.transAxes, fontsize=6.5,
            color="0.45", va="bottom")
    for col, lab, k in (("#8e1b1b", f"dropped by both (peak {peaks[-1]:g})", int(both.sum())),
                        ("#e07b39", f"dropped by peak {peaks[0]:g} only", int(only_loose.sum())),
                        ("#2c3e50", "retained (typical gradient)", len(ref))):
        ax.plot([], [], color=col, lw=2, label=f"{lab}  n={k}")
    ax.legend(fontsize=7.5, loc="lower left", ncol=3)
    fig.suptitle(f"{rec}: channels dropped by annotate_amplitude, min_duration={min_duration*1000:g} ms\n"
                 f"each trace scaled by its own MAD -- shape comparable, amplitude is not",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = figdir("real") / (fname or f"mne_bads_ab_{rec}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[saved] {out}  (both {both.sum()}, B1-only {only_loose.sum()})")


def dump(recs=tuple(PAIRS), peaks=(150.0, 1000.0), min_duration=0.002):
    """Write the WHOLE-RECORDING bad-channel list, so the run does not recompute it per window.

    Condition B is meant to be "drop bad channels", and a bad channel is a property of the
    recording, not of a 60 s window. Left to compute per window, `bad_percent` is evaluated
    against the window instead, so a channel bad in one window and fine in the next comes and
    goes -- which is epoch masking at window granularity wearing a bad-channel label.

    Computing it once here also makes the list inspectable and stable, and costs one load per
    recording rather than one per window.

    Written to runs/mne_bads/<rec>_p<peak:g>.json; compare_spikes reads it by NAME and refuses
    to run if it is absent, mirroring how the relative kStim baseline is handled.
    """
    import json

    out_dir = RUNS / "mne_bads"
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in recs:
        d = _load_native(rec)
        names = [str(n) for n in d["info"]["SelectedSignals"]]
        for pk in peaks:
            qc = mne_bad_channels(d, CFG, peak_uv=pk, min_duration=min_duration)
            bads = [str(b) for b in qc["features"]["mne_bads"]]
            p = out_dir / f"{rec}_p{pk:g}.json"
            p.write_text(json.dumps({
                "recording": rec, "peak_uv": pk, "min_duration": min_duration,
                "n_chan": len(names), "bads": bads,
                "note": "annotate_amplitude peak is a sustained consecutive-sample gradient "
                        "in uV/sample, not a peak-to-peak amplitude",
            }, indent=1), encoding="utf-8")
            print(f"[dump] {p.name}: {len(bads)}/{len(names)} bad -> "
                  + (", ".join(bads) if len(bads) <= 14 else f"{bads[:14]} ..."))
