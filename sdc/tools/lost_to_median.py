"""
sdc.tools.lost_to_median
------------------------
The median filter costs Delphos ~14% of its detections. Are those real spikes or artefact?

    .venv\\Scripts\\python.exe -m sdc.tools.lost_to_median

WHY THIS MATTERS
  Applying median(5) to all three detectors is what finally makes their inputs identical, and
  "identical input" sounds unarguable. But it is not neutral: a 2.5 ms median is a SHARPNESS
  REMOVER and sharpness is exactly what Delphos detects, so the fair-looking change lands
  almost entirely on one arm. Whether that is a fix or a mutilation depends on what it removed,
  and the only way to know is to look.

WHAT IT COMPARES
  Two runs of the SAME recording differing only in MED_KERNEL, both at 1000 Hz with all three
  detectors on the preprocessed EDF:
      runs/P1_pre_med1.npz   median off
      runs/P1_pre.npz        median on (canonical)
  A med-off detection with no med-on partner within TOL_MS is LOST.

THE ARGUMENT, AND ITS WEAK POINT
  Corroboration by Janca or Barkmeier is evidence a detection was real. Measured on P1: 73.5%
  of KEPT detections are corroborated against 14.8% of LOST ones. That is a 5x difference and
  it points one way -- but it is NOT proof, because both other detectors band-pass at 10-60 and
  20-50 Hz and would be blind to genuinely fast activity by construction. A detector cannot
  corroborate what it cannot see. Hence the traces: `median_kills` draws the raw 2 kHz signal
  under the median-filtered one so the removed component is visible directly.

  What to ask of each panel: is there a spike-shaped deflection 20-70 ms wide that the filter
  flattened (bad -- the filter is destroying signal), or is the raw trace carrying a spike or
  two of single-sample noise on an otherwise unremarkable background (fine -- that is what a
  median filter is for)? A median of width 5 at 2 kHz cannot remove anything wider than about
  1.25 ms, so a genuine IED CANNOT be erased by it -- only a sharp feature riding on one can.
"""
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d, decimate

from seeg import read_edf_header, load_edf_segment, derive_montage, apply_montage
from seeg._style import RED, BLUE, MUTED, recessive

from sdc.common.paths import RUNS, figdir
from sdc.common.spike_match import match

A_NPZ = None          # set in __main__; the med-OFF npz the marks are drawn from
MED_OFF = RUNS / "P1_pre_med1.npz"
MED_ON = RUNS / "P1_pre.npz"
TOL_MS = 50.0
PAD_MS = 150.0        # trace either side of the mark -- an IED is 20-70 ms, so this shows the
                      # whole discharge plus enough background to judge it
PER_ROW = 3
VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}


def _per_channel(z, det, n_chan):
    i, c = z[f"{det}_idx"], z[f"{det}_chan"]
    return [np.sort(i[c == k]) for k in range(n_chan)]


def classify(a, b, tol):
    """Split med-off Delphos detections into lost/kept, and tag each by corroboration.

    Returns a list of dicts: {chan, idx, lost, corroborated}."""
    n = len(a["names"])
    D1, D5 = _per_channel(a, "Delphos", n), _per_channel(b, "Delphos", n)
    J, B = _per_channel(a, "Janca", n), _per_channel(a, "Barkmeier", n)
    rows = []
    for k in range(n):
        m1, _, _ = match(D1[k], D5[k], tol)
        if not D1[k].size:
            continue
        cj, _, _ = match(D1[k], J[k], tol)
        cb, _, _ = match(D1[k], B[k], tol)
        corr = cj | cb
        for j, s in enumerate(D1[k]):
            rows.append({"chan": k, "idx": int(s), "lost": not bool(m1[j]),
                         "corroborated": bool(corr[j])})
    return rows


def _shaft(name):
    """Leading letters of a pair name -> the electrode shaft (B1_B2 -> B, R_8_R_9 -> R)."""
    m = re.match(r"^([A-Za-z]+)", name.replace("_", ""))
    return m.group(1)[0] if m else "?"


def draw_dropped(rows, names, edf, fs_det, out, n_panels=12, n_cols=3, pad_ms=250.0,
                 corroborated=None):
    """The DROPPED detections, drawn like polyspike_review: one panel each, all three
    detectors' marks overlaid, raw under filtered.

    Shaft-diverse selection, for the reason polyspike_review learned the hard way: ranking by
    any single quantity clusters the panels onto two or three electrodes and the figure then
    reads as "this only happens on lead X" when it does not."""
    hdr = read_edf_header(edf)
    fs_raw = float(hdr["SampleRate"])
    factor = int(round(fs_raw / fs_det))
    lost = [r for r in rows if r["lost"]]
    if corroborated is not None:
        # The two groups are different animals and must not be mixed in one figure: sorting
        # corroborated-first once produced twelve panels of textbook spike-and-slow-wave and
        # made the 85%% majority invisible.
        lost = [r for r in lost if r["corroborated"] == corroborated]

    picked, seen_shaft, seen_chan = [], set(), set()
    for pass_no in (0, 1):                      # pass 0: a new shaft each time; pass 1: fill up
        for r in sorted(lost, key=lambda r: r["idx"]):
            if len(picked) >= n_panels:
                break
            sh = _shaft(names[r["chan"]])
            if r["chan"] in seen_chan or (pass_no == 0 and sh in seen_shaft):
                continue
            picked.append(r); seen_shaft.add(sh); seen_chan.add(r["chan"])

    n_rows = int(np.ceil(len(picked) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 2.9 * n_rows), squeeze=False)
    pad = pad_ms / 1000.0
    det_sets = {d: _per_channel(A_NPZ, d, len(names)) for d in ("Janca", "Barkmeier", "Delphos")}
    for ax, r in zip(axes.ravel(), picked):
        c = r["chan"]
        t_mid = r["idx"] / fs_det
        r0 = max(int(np.floor(t_mid - pad)), 0) + 1
        r1 = int(np.ceil(t_mid + pad)) + 1
        rec = load_edf_segment(edf, hdr, r0, r1)
        rec = apply_montage(rec, derive_montage(rec["info"]["SelectedSignals"], verbose=False),
                            verbose=False)
        x = rec["data"][:, c]
        t_raw = (r0 - 1) + np.arange(x.size) / fs_raw
        med = medfilt2d(x[:, None], kernel_size=(5, 1))[:, 0]
        dec = decimate(med[:, None], factor, ftype="fir", axis=0)[:, 0]
        t_dec = (r0 - 1) + np.arange(dec.size) / fs_det
        m = (t_raw >= t_mid - pad) & (t_raw <= t_mid + pad)
        md = (t_dec >= t_mid - pad) & (t_dec <= t_mid + pad)
        ax.plot(t_raw[m], x[m], lw=0.7, color="0.66")
        ax.plot(t_dec[md], dec[md], lw=1.2, color="0.12")

        span = np.percentile(np.abs(x[m] - np.median(x[m])), 99.5) * 3.0
        base = float(np.median(x[m]))
        for k, d in enumerate(("Janca", "Barkmeier", "Delphos")):
            t = det_sets[d][c] / fs_det
            t = t[(t >= t_mid - pad) & (t <= t_mid + pad)]
            y = base + span * (0.80 - 0.13 * k)
            if t.size:
                ax.plot(t, np.full(t.size, y), "v", ms=6, color=COLORS[d], clip_on=False)
            ax.text(t_mid - pad, y, f"{d[:4]} ", ha="right", va="center", fontsize=6,
                    color=COLORS[d])
        ax.axvline(t_mid, color=VIOLET, lw=1.0, ls="--", alpha=.75)
        ax.set_ylim(base - span, base + span)
        ax.set_xlim(t_mid - pad, t_mid + pad)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        ax.set_title(f"{names[c]}  t={t_mid:.2f}s"
                     + ("   (another detector agrees)" if r["corroborated"] else ""),
                     fontsize=8, loc="left")
        ax.set_xlabel("time (s)", fontsize=7)
        recessive(ax)
    for ax in axes.ravel()[len(picked):]:
        ax.axis("off")
    _tag = ("" if corroborated is None else
            "  --  ANOTHER DETECTOR AGREES (real discharges)" if corroborated else
            "  --  Delphos-only (the 85% majority)")
    fig.suptitle("Delphos detections DROPPED by the median filter" + _tag
                 + "\ngrey = raw 2 kHz, black = median(5)+decimate; "
                   "dashed = the dropped mark", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    for p in (MED_OFF, MED_ON):
        if not p.is_file():
            raise SystemExit(f"{p} not found -- need both a MED_KERNEL=1 and a MED_KERNEL=5 run.")
    a, b = np.load(MED_OFF, allow_pickle=False), np.load(MED_ON, allow_pickle=False)
    fs = float(a["fs"])
    names = [str(s) for s in a["names"]]
    rows = classify(a, b, int(round(TOL_MS / 1000 * fs)))
    lost = [r for r in rows if r["lost"]]
    kept = [r for r in rows if not r["lost"]]
    lc = sum(r["corroborated"] for r in lost)
    kc = sum(r["corroborated"] for r in kept)
    print(f"Delphos med-OFF {len(rows)} -> med-ON {int(b['Delphos_idx'].size)}")
    print(f"  lost {len(lost)}  kept {len(kept)}")
    print(f"  corroborated by Janca/Barkmeier: kept {kc/max(len(kept),1):.1%}, "
          f"lost {lc/max(len(lost),1):.1%}")
    print("  (corroboration is EVIDENCE, not proof: both others band-pass at 10-60 / 20-50 Hz\n"
          "   and cannot corroborate activity they are built not to see.)")
    A_NPZ = a                      # module scope already; the marks are drawn from the med-OFF run
    for _c, _name in ((False, "uncorroborated"), (True, "corroborated")):
        draw_dropped(rows, names, str(a["edf"]), fs,
                     figdir("real", "P1_pre") / f"lost_to_median_{_name}.png",
                     corroborated=_c)


def draw_with_zoom(rows, names, edf, fs_det, out, n_each=4, pad_ms=250.0, zoom_ms=15.0):
    """Both classes in ONE figure, each panel carrying a ZOOM on the removed component.

    The zoom is the point. At full scale the thing the median filter actually removes is
    invisible: it is ~3 MAD on a discharge that is 10-20 MAD, and about 1 ms wide against a
    500 ms panel. Without it the corroborated panels look like untouched spikes and the whole
    argument -- that Delphos was triggered by an impulse riding on a real discharge -- is
    invisible in the very figure meant to show it.
    """
    hdr = read_edf_header(edf)
    fs_raw = float(hdr["SampleRate"])
    factor = int(round(fs_raw / fs_det))

    def pick(corr, k):
        out_, seen = [], set()
        for r in sorted([r for r in rows if r["lost"] and r["corroborated"] == corr],
                        key=lambda r: r["idx"]):
            if _shaft(names[r["chan"]]) in seen or r["chan"] in {p["chan"] for p in out_}:
                continue
            seen.add(_shaft(names[r["chan"]]))
            out_.append(r)
            if len(out_) >= k:
                break
        return out_

    blocks = [("DROPPED, Delphos-only  (85% -- artefact)", pick(False, n_each)),
              ("DROPPED, another detector agrees  (15% -- real discharge)", pick(True, n_each))]
    fig, axes = plt.subplots(2, n_each, figsize=(4.4 * n_each, 7.4), squeeze=False)
    pad, zoom = pad_ms / 1000.0, zoom_ms / 1000.0
    for row, (title, chosen) in enumerate(blocks):
        for col, r in enumerate(chosen):
            ax = axes[row][col]
            c = r["chan"]
            t_mid = r["idx"] / fs_det
            r0 = max(int(np.floor(t_mid - pad)), 0) + 1
            rec = load_edf_segment(edf, hdr, r0, int(np.ceil(t_mid + pad)) + 1)
            rec = apply_montage(rec, derive_montage(rec["info"]["SelectedSignals"],
                                                    verbose=False), verbose=False)
            x = rec["data"][:, c]
            t_raw = (r0 - 1) + np.arange(x.size) / fs_raw
            med = medfilt2d(x[:, None], kernel_size=(5, 1))[:, 0]
            dec = decimate(med[:, None], factor, ftype="fir", axis=0)[:, 0]
            t_dec = (r0 - 1) + np.arange(dec.size) / fs_det
            m = (t_raw >= t_mid - pad) & (t_raw <= t_mid + pad)
            md = (t_dec >= t_mid - pad) & (t_dec <= t_mid + pad)
            ax.plot(t_raw[m], x[m], lw=0.7, color="0.66")
            ax.plot(t_dec[md], dec[md], lw=1.1, color="0.12")
            span = np.percentile(np.abs(x[m] - np.median(x[m])), 99.5) * 3.2
            base = float(np.median(x[m]))
            ax.axvline(t_mid, color=VIOLET, lw=1.0, ls="--", alpha=.7)
            ax.set_ylim(base - span, base + span)
            ax.set_xlim(t_mid - pad, t_mid + pad)
            ax.set_yticks([])
            ax.tick_params(labelsize=6)
            ax.set_xlabel("time (s)", fontsize=7)
            agree = [d for d in ("Janca", "Barkmeier")
                     if np.any(np.abs(_per_channel(A_NPZ, d, len(names))[c] / fs_det - t_mid)
                               <= TOL_MS / 1000.0)]
            ax.set_title(f"{names[c]}  t={t_mid:.2f}s"
                         + (f"   [{'+'.join(a[:4] for a in agree)}]" if agree else ""),
                         fontsize=8, loc="left")
            recessive(ax)

            # ZOOM: +-zoom_ms, raw against filtered, so the removed component is visible
            iz = ax.inset_axes([0.60, 0.62, 0.38, 0.36])
            mz = (t_raw >= t_mid - zoom) & (t_raw <= t_mid + zoom)
            mzd = (t_dec >= t_mid - zoom) & (t_dec <= t_mid + zoom)
            iz.plot((t_raw[mz] - t_mid) * 1000, x[mz], lw=0.9, color="0.55")
            iz.plot((t_dec[mzd] - t_mid) * 1000, dec[mzd], lw=1.3, color="#c8102e")
            iz.set_xlim(-zoom_ms, zoom_ms)
            iz.set_xticks([-zoom_ms, 0, zoom_ms])
            iz.tick_params(labelsize=5.5, pad=1)
            iz.set_yticks([])
            for s in iz.spines.values():
                s.set_linewidth(0.6)
            iz.set_title(f"zoom +-{zoom_ms:g} ms", fontsize=6, pad=2)
            if col == 0:
                ax.set_ylabel(title, fontsize=8)
    fig.suptitle("What the median filter removes, and from what   "
                 "(grey/black = full view; inset = raw vs filtered, +-15 ms)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")
