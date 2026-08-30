"""
sdc.compare.rate_comparators
----------------------------
Per-channel spike rate as a DISTRIBUTION, over all channels and over the EZ contacts.

    .venv\\Scripts\\python.exe -m sdc.compare.rate_comparators

WHY NOT THE MEAN OVER ALL CHANNELS
  That is what every rate figure here has used, and it answers a worse question than it looks.
  Two problems, both of which push in the same direction:

    * It is dominated by near-silent channels. Most contacts in any implant see almost nothing,
      so the mean mostly measures how many of those there are.
    * It conflates HOW EPILEPTIC THE PATIENT IS with WHAT FRACTION OF THEIR CONTACTS SIT IN
      EPILEPTIC TISSUE. A focal implant concentrated on the seizure onset zone gives a higher
      per-channel mean than a broad one at identical underlying pathology. P1 has 226 channels
      and P5 has 183, different implants in different lobes, so this is not hypothetical --
      it is exactly the confound in "is P5 quieter, or just differently covered".

  Median + MAD over the distribution fixes the first problem. Restricting to the EZ fixes the
  second, and it is the only comparator here defined INDEPENDENTLY OF THE DETECTORS: the EZ
  contact list is clinical, and comes from the trials JSON rather than from anything measured.

ONE CHANNEL SET PER PATIENT, SHARED BY EVERY CONDITION AND EVERY DETECTOR
  This figure used to take each condition's median over whatever channels survived in THAT
  condition. On P1 that is 223 channels in stim OFF and 132 in stim ON, because the artefact
  mask removes the contaminated ones -- so the two medians described different implants and the
  difference between them was partly just that. It mattered: Barkmeier read 1.89 -> 2.77 that
  way, a 47% INCREASE under stimulation, against 0.59 -- a 41% decrease -- once the same
  channels are used on both sides. A sign flip, from nothing but the channel set.

  So the mask is now intersected across baseline, stim OFF and stim ON, and across all three
  detectors, and every median on the page is over that one common set. The medians are still
  medians; what changed is what they are medians OF. The cost is printed, because intersecting
  discards channels and that should be visible rather than assumed small.

WHAT IT CANNOT TELL YOU
  Whether the EZ list is right. It is a clinical judgement, so a detector that disagrees with it
  is not thereby wrong. And the EZ is ~20 channels of ~200, so its median carries real sampling
  noise -- the MAD is drawn for that reason and should be read, not skipped.

  It is also still a LEVEL comparator, not an effect estimate. Pairing the channels removes the
  worst confound but not the others: stim ON and stim OFF are different minutes of the night,
  and a rate that drifts will show up here as a condition difference. For the ON/OFF effect
  itself use sdc.artefact.blocks, which contrasts each ON block against the OFF time beside it.
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from seeg._style import RED, BLUE, MUTED, GRID, recessive

from sdc.common import cond
from sdc.common.paths import RUNS, figdir

META = Path(r"C:\Users\amoo0039\Documents\local\data_meta\stim_trials.json")
# (patient, [(run stem, COND, condition label), ...]). Grouped so the PATIENT comparison --
# the question this figure exists for -- is adjacent on the x axis for each detector, rather
# than six recordings in a row with P1 and P5 four positions apart.
PATIENTS = [("P1", [("P1_pre", "all", "baseline"),
                    ("P1_stim", "off", "stim OFF"),
                    ("P1_stim", "on", "stim ON")]),
            ("P5", [("P5_pre", "all", "baseline"),
                    ("P5_stim", "off", "stim OFF"),
                    ("P5_stim", "on", "stim ON")])]
COND_COLOR = {"baseline": "#3a3a3a", "stim OFF": "#4a7fb5", "stim ON": "#e0a80c"}
VIOLET = "#4a3aa7"
COLORS = {"Janca": RED, "Barkmeier": BLUE, "Delphos": VIOLET}


def ez_channels(patient_id):
    """Clinically-defined epileptogenic-zone contacts for a patient, as montage pair names."""
    meta = json.loads(META.read_text())
    p = [x for x in meta if x["patient_id"] == patient_id][0]
    return list(p.get("EZ") or [])


RUN_TAG = os.environ.get("RUN_TAG", "")
# "" = the canonical runs at each detector's default operating point; "_tuned" = the BIDS
# operating points (k1 4.482, TAMP 890, Spk_thr 46.2), fitted to 3.5 det/chan-min against 852
# expert-marked IEDs. Kept as a suffix rather than a replacement so both result sets stay on
# disk and a figure can never be silently the wrong one -- the tag goes in the filename too.


def load(stem, label):
    z = np.load(RUNS / f"{stem}{RUN_TAG}.npz", allow_pickle=False)
    names = [str(s) for s in z["names"]]
    sel = cond.select(z, label)
    dets = [str(s) for s in z["detectors"]]
    rates = {}
    for d in dets:
        keep = sel.keep(d)
        cnt = np.bincount(z[f"{d}_chan"][keep], minlength=len(names))
        rates[d] = sel.rate(cnt) * 60.0          # per MINUTE -- per-second numbers are all
                                                 # 0.0x and unreadable on a shared axis
    ez = ez_channels(int(z["patient"]))
    missing = [c for c in ez if c not in names]
    if missing:
        # A silently dropped EZ contact biases the EZ median, so this is loud rather than a
        # filter. Contact-number gaps in the montage are the usual cause.
        raise SystemExit(f"{stem}: EZ contacts absent from the montage: {missing}")
    ez_idx = np.array([names.index(c) for c in ez], int)
    return dict(names=names, dets=dets, rates=rates, sel=sel, ez=ez_idx,
                patient=int(z["patient"]))


def mad(v):
    """Median absolute deviation, scaled to be comparable with a standard deviation."""
    v = v[np.isfinite(v)]
    return float(np.median(np.abs(v - np.median(v))) * 1.4826) if v.size else np.nan


def main():
    data = {}
    for pat, specs in PATIENTS:
        for stem, label, cond_lab in specs:
            if not (RUNS / f"{stem}{RUN_TAG}.npz").is_file():
                continue          # must test the SAME file load() will open, tag included
            data[(pat, cond_lab)] = load(stem, label)
    if not data:
        raise SystemExit("no runs found")
    dets = next(iter(data.values()))["dets"]
    conds = ["baseline", "stim OFF", "stim ON"]

    # ---- one channel set per patient -----------------------------------------------------
    # Intersected over every condition AND every detector, so that each patient's three boxes
    # describe the same contacts. Without this the stim-ON box is drawn over the subset of the
    # implant that survived the artefact mask, and is compared against a stim-OFF box drawn
    # over nearly all of it.
    common, cost = {}, {}
    for pat, specs in PATIENTS:
        keys = [(pat, cl) for _, _, cl in specs if (pat, cl) in data]
        if not keys:
            continue
        names0 = data[keys[0]]["names"]
        m = np.ones(len(names0), bool)
        per_cond = {}
        for k in keys:
            if data[k]["names"] != names0:
                raise SystemExit(
                    f"{pat}: channel names differ between {keys[0][1]!r} and {k[1]!r}; the "
                    f"conditions cannot be paired by position.")
            c_m = np.ones(len(names0), bool)
            for d in dets:
                c_m &= np.isfinite(data[k]["rates"][d])
            per_cond[k[1]] = int(c_m.sum())
            m &= c_m
        common[pat] = m
        cost[pat] = (per_cond, int(m.sum()), len(names0))

    print("channels per patient after intersecting conditions x detectors")
    for pat, (per_cond, n_common, n_all) in cost.items():
        print(f"  {pat}: " + ", ".join(f"{k} {v}" for k, v in per_cond.items())
              + f"  ->  common {n_common} of {n_all} implanted")
    print()

    def vals(key, det, scope):
        c = data[key]
        m = common[key[0]].copy()
        if scope == "EZ":
            ez = np.zeros(m.size, bool)
            ez[c["ez"]] = True
            m &= ez
        v = c["rates"][det][m]
        return v[np.isfinite(v)]

    print("per-channel rate, spikes/min   (median +- MAD)   -- PAIRED channel set\n")
    for scope in ("all", "EZ"):
        print(f"  -- {scope} channels --")
        print(f"{'':<12}" + "".join(f"{d:>22}" for d in dets))
        for pat, _ in PATIENTS:
            for cl in conds:
                if (pat, cl) not in data:
                    continue
                line = f"{pat} {cl:<9}"
                for d in dets:
                    v = vals((pat, cl), d, scope)
                    line += f"{np.median(v):>10.2f} +-{mad(v):<8.2f}"
                print(line)
        # the ratio this figure exists to show
        print(f"{'P5/P1 baseline':<12}" + "".join(
            f"{np.median(vals(('P5','baseline'), d, scope)) / np.median(vals(('P1','baseline'), d, scope)):>22.2f}"
            for d in dets) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9), sharex=True)
    GAP = 0.9                     # extra space between the two patient blocks
    width = 0.26
    centres, tick_lab = [], []
    for ax, scope in zip(axes, ("all", "EZ")):
        for pi, (pat, _) in enumerate(PATIENTS):
            for j, d in enumerate(dets):
                base = pi * (len(dets) + GAP) + j
                if ax is axes[0]:
                    centres.append(base)
                    tick_lab.append(d)
                for k, cl in enumerate(conds):
                    if (pat, cl) not in data:
                        continue
                    v = vals((pat, cl), d, scope)
                    v = v[v > 0]                       # log axis has no place for a zero
                    if not v.size:
                        continue
                    pos = base + (k - 1) * width
                    bp = ax.boxplot([v], positions=[pos], widths=width * 0.8,
                                    showfliers=False, patch_artist=True)
                    col = COND_COLOR[cl]
                    bp["boxes"][0].set(facecolor=col, alpha=.3, edgecolor=col)
                    for part in ("whiskers", "caps", "medians"):
                        for ln in bp[part]:
                            ln.set(color=col, lw=1.5)
                    x = np.full(v.size, pos) + np.linspace(-width * .28, width * .28, v.size)
                    ax.scatter(x, v, s=5, color=col, alpha=.3, edgecolor="none", zorder=3)
        # separator between the two patient blocks
        ax.axvline(len(dets) - 1 + GAP / 2, color="0.35", lw=1.2, ls="--", alpha=.8)
        ax.set_yscale("log")
        ax.set_ylabel(f"{scope} channels\nspikes / min / channel")
        ax.grid(axis="y", alpha=.3)
        recessive(ax)
    for pi, (pat, _) in enumerate(PATIENTS):
        mid = pi * (len(dets) + GAP) + (len(dets) - 1) / 2
        axes[0].text(mid, 1.04, pat, transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=13, fontweight="bold")
    axes[0].set_title("(a) all channels in the common set -- the SAME contacts in all three "
                      "conditions", fontsize=9, loc="left")
    axes[1].set_title("(b) EZ contacts within that common set -- clinically defined, "
                      "independent of any detector", fontsize=9, loc="left")
    axes[1].set_xticks(centres)
    axes[1].set_xticklabels(tick_lab, fontsize=9)
    handles = [plt.Line2D([], [], color=COND_COLOR[c], lw=3, label=c) for c in conds]
    axes[0].legend(handles=handles, frameon=False, fontsize=9, ncol=3, loc="upper right")
    # The channel counts belong in the title rather than a caption: they are what makes this
    # page comparable to itself, and the previous version's headline numbers were wrong
    # precisely because they were computed over different sets without saying so.
    fig.suptitle("Per-channel rate by detector and condition, over ONE channel set per "
                 f"patient ({', '.join(f'{p} {cost[p][1]}/{cost[p][2]}' for p in cost)} "
                 "contacts, common to baseline / stim OFF / stim ON and to all 3 detectors)\n"
                 "Levels only -- ON and OFF are different minutes of the night, so for the "
                 "stimulation EFFECT see sdc.artefact.blocks", fontsize=10)
    fig.tight_layout()
    out = figdir("real") / f"rate_comparators{RUN_TAG}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
