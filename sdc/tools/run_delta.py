"""
sdc.tools.run_delta
-------------------
What changed between two runs, for one detector, drawn so you can judge it.

    python -m sdc.tools.run_delta <before.npz> <after.npz> <Detector> [added|removed]

    # what the 400 -> 1000 Hz change ADDED to Janca
    python -m sdc.tools.run_delta runs/P1_pre_400Hz.npz runs/P1_pre_med1.npz Janca added
    # what the median filter REMOVED from Delphos
    python -m sdc.tools.run_delta runs/P1_pre_med1.npz runs/P1_pre.npz Delphos removed

GENERALISED FROM lost_to_median.py, which answered exactly this question for one detector and
one config change and then immediately needed to answer it for another. The comparison is
always the same shape: match the two runs per channel, take the unmatched detections on one
side, and draw them with the raw trace under the processed one.

TWO THINGS IT DOES THAT MATTER
  * MATCHES IN SECONDS, NOT SAMPLES. The runs being compared often differ in DETECT_FS -- that
    is frequently the change under test -- and matching raw indices across different sample
    rates silently compares 400 Hz sample 1000 with 1000 Hz sample 1000, i.e. 2.5 s against
    1.0 s. Everything here is converted to seconds first.
  * SPLITS BY CORROBORATION. Whether the other two detectors also marked an event is the only
    cheap evidence available about whether it was real. It is EVIDENCE, NOT PROOF: Janca and
    Barkmeier band-pass at 10-60 and 20-50 Hz, so they cannot corroborate genuinely fast
    activity, and a Delphos-only detection is not automatically wrong. Sorting a figure without
    splitting on this once produced twelve panels of textbook spike-and-slow-wave and hid the
    85% majority that looked nothing like it.
"""
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d, decimate

from seeg import read_edf_header, load_edf_segment, derive_montage, apply_montage
from seeg._style import RED, BLUE, MUTED, recessive

from sdc.common.paths import figdir
from sdc.common.spike_match import match

TOL_MS = 50.0
PAD_MS = 250.0
ZOOM_MS = 15.0
VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}


def _shaft(name):
    m = re.match(r"^([A-Za-z]+)", str(name).replace("_", ""))
    return m.group(1)[0] if m else "?"


def per_channel_sec(z, det, n_chan):
    """Detections as SECONDS per channel -- never raw indices, see the module docstring."""
    fs = float(z["fs"])
    i, c = z[f"{det}_idx"], z[f"{det}_chan"]
    return [np.sort(i[c == k] / fs) for k in range(n_chan)]


def delta(before, after, det, tol_s=TOL_MS / 1000.0):
    """Detections in `after` with no `before` partner (added), and vice versa (removed).

    Corroboration is taken from the run the detection LIVES IN -- an added detection is judged
    against the other detectors in `after`, a removed one against `before`. Judging both against
    one run would ask whether a detector agreed with an event it was never shown."""
    names = [str(s) for s in after["names"]]
    n = len(names)
    A = {d: per_channel_sec(before, d, n) for d in [str(s) for s in before["detectors"]]}
    B = {d: per_channel_sec(after, d, n) for d in [str(s) for s in after["detectors"]]}
    others = [d for d in B if d != det]
    added, removed = [], []
    for k in range(n):
        ma, mb, _ = match(A[det][k], B[det][k], tol_s)
        for t in B[det][k][~mb]:
            corr = any(np.any(np.abs(B[o][k] - t) <= tol_s) for o in others)
            added.append({"chan": k, "t": float(t), "corroborated": corr})
        for t in A[det][k][~ma]:
            corr = any(np.any(np.abs(A[o][k] - t) <= tol_s) for o in others)
            removed.append({"chan": k, "t": float(t), "corroborated": corr})
    return added, removed


def draw(rows, names, edf, z_after, out, title, n_each=4, pad_ms=PAD_MS, zoom_ms=ZOOM_MS):
    """Two rows -- uncorroborated then corroborated -- each panel with a +-zoom_ms inset.

    The inset is not decoration. The component that separates these classes is ~1 ms wide and a
    few MAD tall on a discharge of 10-20 MAD, so at full scale it is invisible and the figure
    argues nothing."""
    hdr = read_edf_header(edf)
    fs_raw = float(hdr["SampleRate"])
    fs_det = float(z_after["fs"])
    med_k = int(z_after["med_kernel"]) | 1
    factor = max(int(round(fs_raw / fs_det)), 1)

    def pick(corr, k):
        out_, seen_shaft, seen_chan = [], set(), set()
        for pass_no in (0, 1):
            for r in sorted([r for r in rows if r["corroborated"] == corr], key=lambda r: r["t"]):
                if len(out_) >= k:
                    break
                sh = _shaft(names[r["chan"]])
                if r["chan"] in seen_chan or (pass_no == 0 and sh in seen_shaft):
                    continue
                out_.append(r); seen_shaft.add(sh); seen_chan.add(r["chan"])
        return out_

    blocks = [("no other detector agrees", pick(False, n_each)),
              ("another detector agrees", pick(True, n_each))]
    fig, axes = plt.subplots(2, n_each, figsize=(4.4 * n_each, 7.4), squeeze=False)
    pad, zoom = pad_ms / 1000.0, zoom_ms / 1000.0
    for row, (label, chosen) in enumerate(blocks):
        for col in range(n_each):
            ax = axes[row][col]
            if col >= len(chosen):
                ax.axis("off")
                continue
            r = chosen[col]
            c, t_mid = r["chan"], r["t"]
            r0 = max(int(np.floor(t_mid - pad)), 0) + 1
            rec = load_edf_segment(edf, hdr, r0, int(np.ceil(t_mid + pad)) + 1)
            rec = apply_montage(rec, derive_montage(rec["info"]["SelectedSignals"],
                                                    verbose=False), verbose=False)
            x = rec["data"][:, c]
            t_raw = (r0 - 1) + np.arange(x.size) / fs_raw
            proc = medfilt2d(x[:, None], kernel_size=(med_k, 1))[:, 0]
            dec = (decimate(proc[:, None], factor, ftype="fir", axis=0)[:, 0]
                   if factor > 1 else proc)
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
            ax.set_title(f"{names[c]}  t={t_mid:.2f}s", fontsize=8, loc="left")
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            recessive(ax)

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
    fig.suptitle(title + "\ngrey = raw, black = what the detector saw; "
                         "inset = the same +-15 ms, red = processed", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__.strip().split("\n\n")[1])
    p_before, p_after, det = sys.argv[1], sys.argv[2], sys.argv[3]
    which = sys.argv[4] if len(sys.argv) > 4 else "added"
    zb, za = (np.load(p, allow_pickle=False) for p in (p_before, p_after))
    names = [str(s) for s in za["names"]]
    added, removed = delta(zb, za, det)
    rows = added if which == "added" else removed
    tag = "ADDED to" if which == "added" else "REMOVED from"
    nb = int(zb[f"{det}_idx"].size); na = int(za[f"{det}_idx"].size)
    corr = sum(r["corroborated"] for r in rows)
    print(f"{det}: {nb} -> {na}   added {len(added)}  removed {len(removed)}")
    print(f"  {which}: {len(rows)}, corroborated by another detector {corr}/{len(rows)} "
          f"= {corr/max(len(rows),1):.1%}")
    stem = f"{det.lower()}_{which}_{Path(p_before).stem}_to_{Path(p_after).stem}.png" \
        if False else f"{det.lower()}_{which}.png"
    draw(rows, names, str(za["edf"]), za,
         figdir("real", "P1_pre") / stem,
         f"{len(rows)} detections {tag} {det}   "
         f"({Path(p_before).name} -> {Path(p_after).name})")
