"""
sdc.artefact.periodicity_check
------------------------------
Check (a) of plans/polymorphic-wiggling-breeze.md: does the periodicity index separate stim-ON
from stim-OFF on the REAL 2 Hz recording, and where should its threshold go?

    .venv\\Scripts\\python.exe -m sdc.artefact.periodicity_check

The synthetic tests in seeg.artefact.periodicity_index establish that the statistic is
arithmetically right (background z ~ 1.2, a clear train 10-15, a 1 Hz heartbeat scored at 2 Hz
2.5). They cannot establish that real 2 Hz stimulation artefact looks like a train to it, or
where a threshold belongs on real data. Only this does.

THE THRESHOLD IS READ OFF THIS FIGURE, NOT GUESSED. If the ON and OFF distributions overlap,
the rule does not work on this recording and E1 should not include it -- which is a result, and
a cheap one, since no detector has to run to find out.

ON/OFF comes from `on_per_sec` in the existing run npz, i.e. the pipeline's OWN stim
classification, rather than re-deriving it with detect_stim. Same labels the mask used.
"""
import numpy as np

from seeg.artefact import periodicity_index, PERIODICITY_MAX_HZ

from sdc.common.paths import RUNS, figdir
from sdc.artefact.mne_bads_check import _load_native

REC = "P1_ANT2_stim"
BASELINE = "P1_ANT2_pre"
RUN_STEM = "P1_ANT2_stim_qcnone"      # any profile: on_per_sec does not depend on the mask
STIM_HZ = 2.0
EPOCH_SEC = 2.0
CANDIDATES = (3.0, 5.0, 8.0)


def _epoch_z(rec_id, stim_hz, epoch_sec=EPOCH_SEC, on_per_sec=None):
    """Per-epoch periodicity z for every channel. Returns (z [n_ep, n_chan], is_on, names, d)."""
    d = _load_native(rec_id)
    fs = float(d["info"]["SampleRate"])
    x = np.asarray(d["data"], float)
    names = [str(n) for n in d["info"]["SelectedSignals"]]
    n_ep_samp = int(round(epoch_sec * fs))
    starts = np.arange(0, x.shape[0] - n_ep_samp + 1, n_ep_samp)
    z = np.full((starts.size, x.shape[1]), np.nan)
    for e, s in enumerate(starts):
        z[e, :] = periodicity_index(x[s:s + n_ep_samp, :], fs, stim_hz)[0]
    is_on = np.zeros(starts.size, bool)
    if on_per_sec is not None:
        sec = (starts / fs).astype(int)
        mid = np.clip(sec + int(epoch_sec // 2), 0, len(on_per_sec) - 1)
        is_on = np.asarray(on_per_sec, float)[mid] > 0.5
    return z, is_on, names, d


def run(fname="periodicity_check_P1_ANT2.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zrun = np.load(RUNS / f"{RUN_STEM}.npz", allow_pickle=False)
    on_per_sec = np.asarray(zrun["on_per_sec"], float)
    if on_per_sec.ndim > 1:
        on_per_sec = on_per_sec.mean(axis=1)

    print(f"[periodicity] {REC} at {STIM_HZ:g} Hz, {EPOCH_SEC:g} s epochs")
    z, is_on, names, d = _epoch_z(REC, STIM_HZ, on_per_sec=on_per_sec)
    print(f"  {z.shape[0]} epochs ({int(is_on.sum())} ON / {int((~is_on).sum())} OFF), "
          f"{z.shape[1]} channels")
    # The baseline recording is the strongest possible negative control: same patient, same
    # electrodes, no stimulator running at all. Anything the rule flags there is a false alarm.
    zb, _ib, _nb, _db = _epoch_z(BASELINE, STIM_HZ)
    print(f"  baseline {BASELINE}: {zb.shape[0]} epochs")

    on = z[is_on, :].ravel()
    off = z[~is_on, :].ravel()
    base = zb.ravel()
    on, off, base = (v[np.isfinite(v)] for v in (on, off, base))
    print(f"  {'':<10}{'n':>9}{'median':>9}{'p90':>9}{'p99':>9}")
    for nm, v in (("stim ON", on), ("stim OFF", off), ("baseline", base)):
        if v.size:
            print(f"  {nm:<10}{v.size:>9}{np.median(v):>9.2f}"
                  f"{np.percentile(v, 90):>9.2f}{np.percentile(v, 99):>9.2f}")
    for thr in CANDIDATES:
        print(f"  thr {thr:>4.1f}: flags {100 * (on > thr).mean():>5.1f}% of ON, "
              f"{100 * (off > thr).mean():>5.1f}% of OFF, "
              f"{100 * (base > thr).mean():>5.1f}% of baseline")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    # log bins: z spans 0.1 to >2000 on this recording, so a linear axis compresses the
    # entire baseline distribution into the first bin and shows nothing.
    bins = np.logspace(-1, np.log10(max(np.percentile(on, 99.9) if on.size else 10, 20)), 70)
    for nm, v, c in (("stim ON", on, "#c0392b"), ("stim OFF", off, "#2c3e50"),
                     (f"baseline ({BASELINE})", base, "#7f8c8d")):
        if v.size:
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8, color=c, label=nm)
    for thr in CANDIDATES:
        ax.axvline(thr, color="#e07b39", ls=":", lw=1.3)
        ax.text(thr, ax.get_ylim()[1] * .92, f" {thr:g}", fontsize=7.5, color="#e07b39")
    ax.set_xscale("log")
    ax.set_xlabel("periodicity z at the 2 Hz lag (log)")
    ax.set_ylabel("density")
    ax.set_title("(a) does stim separate from the baseline recording?",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    ax = axes[1]
    # Across-CHANNEL summary per epoch, not three hand-picked channels: the top channels sit
    # at z ~ 3000 and would hide everything else, and the question is how much of the montage
    # is affected, not how bad the worst channel is.
    t = np.arange(z.shape[0]) * EPOCH_SEC
    ax.fill_between(t, 0, 1, where=is_on, transform=ax.get_xaxis_transform(),
                    color="#c0392b", alpha=.08, step="mid", label="stim ON")
    for q, st in ((90, "-"), (50, "-"), (10, ":")):
        ax.plot(t, np.nanpercentile(z, q, axis=1), st, lw=1.4, label=f"p{q} across channels")
    ax.set_yscale("log")
    for thr in CANDIDATES:
        ax.axhline(thr, color="#e07b39", ls=":", lw=1.2)
    ax.set_xlabel("s")
    ax.set_ylabel("periodicity z")
    ax.set_title("(b) spread across the montage, per epoch", fontsize=10, loc="left")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=.25)

    ax = axes[2]
    thrs = np.linspace(1, 15, 60)
    for nm, v, c in (("stim ON", on, "#c0392b"), ("stim OFF", off, "#2c3e50"),
                     ("baseline", base, "#7f8c8d")):
        if v.size:
            ax.plot(thrs, [100 * (v > th).mean() for th in thrs], lw=1.8, color=c, label=nm)
    for thr in CANDIDATES:
        ax.axvline(thr, color="#e07b39", ls=":", lw=1.2)
    ax.set_xlabel("threshold")
    ax.set_ylabel("% of channel-epochs flagged")
    ax.set_title("(c) what each threshold costs", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    # This trial is CONTINUOUS stimulation -- 477 stim epochs, no OFF -- so there is no
    # within-file ON/OFF contrast and the stim-free PRE recording is the only control.
    fig.suptitle(f"Periodicity index: {REC} ({STIM_HZ:g} Hz, continuous) vs its stim-free "
                 f"baseline\nclinical montage, {EPOCH_SEC:g} s epochs -- "
                 f"the threshold is read off this figure", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = figdir("real") / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[saved] {out}")
    return z, is_on, names


if __name__ == "__main__":
    run()
