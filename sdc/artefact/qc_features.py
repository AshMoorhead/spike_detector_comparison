"""
sdc.artefact.qc_features
------------------------
Which artefact RULE is doing the masking -- pStim, dynR or gradRatio?

    .venv\\Scripts\\python.exe -m sdc.artefact.qc_features --dump     # once per recording
    .venv\\Scripts\\python.exe -m sdc.artefact.qc_features            # the attribution

WHY THIS IS CHEAP
  `windowed_artefact_detector` thresholds three features and keeps only the verdict, so a
  finished run cannot say which rule fired. But the FEATURES are threshold-independent: dump
  them once and the composition of the mask at any threshold is arithmetic. No detector runs,
  no MATLAB, no Delphos -- minutes rather than the hour a ladder rung costs.

  `--dump` re-reads the EDF in the same RAM-budgeted windows the detector runs use, calls
  compare_spikes with QC_FEATURES set (which writes the features and exits before any detector),
  and concatenates the per-window results onto one global epoch grid.

WHAT THE ATTRIBUTION CAN AND CANNOT SAY
  It gives the share of masked channel-epochs each rule is responsible for, and how much they
  overlap -- so "gradThr is doing all the work and stimPowerThr almost none" becomes a measured
  statement rather than a guess. It does NOT give each rule's effect on the ON/OFF ratio; that
  needs a re-run per rule, and the point of this pass is to say which rules are worth spending
  those hours on.

  Note the rules are not symmetric in scope. `stim_spec` is evaluated on stim-ON epochs only
  (pStim is nan elsewhere by construction), so its share of the whole recording understates its
  role during stimulation, which is where it matters. Both are reported.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from sdc.common.paths import ROOT, RUNS, figdir
from sdc.detect.run_windows import plan

FEAT = ROOT / "qc_features"
RECS = ("P1_stim", "P5_stim")
RULES = ("lf_artefact", "stim_spec", "low_dyn")

# The same ladder compare_spikes exposes, restated here so the attribution can be evaluated at
# every rung without importing the detector module.
PROFILES = {
    "none":    dict(gradThr=0.0, stimPowerThr=1e12, dynFloorMult=0.0),
    "loose":   dict(gradThr=2000.0, stimPowerThr=200.0, dynFloorMult=0.0),
    "prod":    dict(gradThr=400.0, stimPowerThr=50.0, dynFloorMult=3.0),
    "strict":  dict(gradThr=150.0, stimPowerThr=10.0, dynFloorMult=3.0),
    "vstrict": dict(gradThr=50.0, stimPowerThr=2.0, dynFloorMult=3.0),
}


def dump(rec, tag="", freq=None):
    """Run the QC-only pass over the whole recording, window by window, and concatenate.

    `tag` is appended to every filename. Use it to keep dumps made at different stimulation
    frequencies apart: pStim at 2 Hz and pStim at 145 Hz are different measurements of the same
    channel, and a single filename cannot hold both without one silently overwriting the other.
    """
    if freq is not None:
        # Frequency in the name, not a hand-written tag: with 55 trials spanning 2-145 Hz a
        # mislabelled baseline is a silent wrong answer, and `_2hz`-style tags rely on whoever
        # runs it remembering to pass the right one.
        tag = f"_f{float(freq):g}"
        # ASSIGN, do not setdefault. setdefault only takes effect if the variable is unset, so
        # dumping several recordings from ONE process pinned QC_STIM_HZ to the FIRST call's
        # frequency and measured every later recording at it -- while still naming each file
        # after its own. Confirmed on P5_Pulv7_pre: the stored `_f7` file was bit-identical to a
        # deliberate 2 Hz measurement, because the batch happened to start with a 2 Hz trial.
        # The frequency is also written into the npz (see compare_spikes' QC_FEATURES block) so
        # a file's contents can be checked against its name instead of trusted.
        os.environ["QC_STIM_HZ"] = f"{float(freq):g}"
    FEAT.mkdir(parents=True, exist_ok=True)
    _edf, _hdr, total, windows = plan(rec)
    parts = []
    for w in windows:
        out = FEAT / f"_{rec}{tag}_w{w['n']:03d}.npz"
        if not out.is_file():
            env = {**os.environ, "RECORDING": rec, "START_REC": str(w["start_rec"]),
                   "STOP_REC": str(w["stop_rec"]), "QC_FEATURES": str(out)}
            print(f"  window {w['n']}/{len(windows)} ({w['start_rec']}-{w['stop_rec']}s)")
            p = subprocess.run([sys.executable, "-m", "sdc.detect.compare_spikes"],
                               cwd=str(ROOT), env=env)
            if p.returncode != 0:
                raise SystemExit(f"window {w['n']} failed")
        parts.append((w, np.load(out, allow_pickle=False)))

    # Epochs are 2 s on a grid anchored at each window's start, and windows overlap. Keep only
    # epochs whose CENTRE falls in the window's interior, which is the same rule the detector
    # runs use for detections -- otherwise the overlap is counted twice.
    # Same cache, same failure mode as the pStimAll check below, different field: per-window
    # files are keyed on filename only, so a dump made under a DIFFERENT MONTAGE is reused and
    # concatenated silently. That produced a 226-channel baseline for a 164-channel run, and
    # compare_spikes then refused every relative-kStim condition on a stim recording.
    _nch = {z["names"].shape[0] if "names" in z.files else z["gradRatio"].shape[1]
            for _w, z in parts}
    if len(_nch) > 1:
        raise SystemExit(f"windows disagree on channel count {sorted(_nch)} -- delete "
                         f"qc_features/_{rec}{tag}_w*.npz and re-run --dump.")
    stale = [w["n"] for w, z in parts if "pStimAll" not in z.files]
    if stale:
        raise SystemExit(
            f"windows {stale} predate the pStimAll field. They were reused from cache, which is "
            f"why the concatenated file silently lacked a baseline. Delete "
            f"qc_features/_{rec}{tag}_w*.npz and re-run --dump.")

    keep_f, keep_on, keep_t, keep_pa = [], [], [], []
    for w, z in parts:
        fs = float(z["qc_fs"])
        t = w["start_rec"] + (z["starts"] + z["epoch_samp"] / 2) / fs     # absolute seconds
        m = (t >= w["t0"]) & (t < w["t1"])
        keep_f.append(np.stack([z["gradRatio"][m], z["pStim"][m], z["dynR"][m]]))
        keep_on.append(z["isOn"][m])
        keep_t.append(t[m])
        # Band power on EVERY epoch, not just stim-ON. This is what a relative threshold needs
        # and the reason the pre files are dumped at all -- they have no stimulation, so their
        # pStimAll is the null distribution for the stim band.
        keep_pa.append(z["pStimAll"][m])
    # The frequency pStimAll was ACTUALLY measured at. The per-window files carried this all
    # along but the concatenation dropped it, so the only evidence of a dump's frequency was its
    # filename -- and a filename is not a measurement. That is exactly how 11 baselines came to
    # be named `_f145`/`_f7` while holding 2 Hz band power. Carried through and checked on read.
    hz = {float(z["stim_hz_used"]) for _w, z in parts if "stim_hz_used" in z.files}
    if len(hz) > 1:
        raise SystemExit(f"{rec}{tag}: windows disagree on the measured frequency {sorted(hz)} "
                         f"-- some were reused from a cache made at another frequency. Delete "
                         f"qc_features/_{rec}{tag}_w*.npz and re-dump.")
    measured = hz.pop() if hz else float("nan")
    if freq is not None and np.isfinite(measured) and abs(measured - float(freq)) > 1e-6:
        raise SystemExit(f"{rec}{tag}: asked for {float(freq):g} Hz but the windows were measured "
                         f"at {measured:g} Hz. Stale cache, or QC_STIM_HZ set in the environment.")

    order = np.argsort(np.concatenate(keep_t))
    out = FEAT / f"{rec}{tag}.npz"
    np.savez_compressed(out,
                        feat=np.concatenate(keep_f, axis=1)[:, order],   # (3, epoch, chan)
                        isOn=np.concatenate(keep_on)[order],
                        pStimAll=np.concatenate(keep_pa, axis=0)[order],
                        t=np.concatenate(keep_t)[order],
                        stim_hz_used=np.float64(measured),
                        lsb=parts[0][1]["lsb"], names=parts[0][1]["names"])
    z = np.load(out, allow_pickle=False)
    print(f"[dump] {out.name}: {z['feat'].shape[1]} epochs x {z['feat'].shape[2]} channels, "
          f"{int(z['isOn'].sum())} ON")
    return out


def fires(feat, is_on, lsb, cfg):
    """Per-rule boolean masks over [epoch, channel], exactly as artefact.py evaluates them."""
    grad, pstim, dynr = feat
    lf = (grad > cfg["gradThr"]) if cfg["gradThr"] > 0 else np.zeros(grad.shape, bool)
    # stim_spec is ON-epochs only, and pStim is nan off them -- so the comparison has to be
    # guarded rather than relying on nan>x being False, which is true but easy to break later.
    ss = np.zeros(grad.shape, bool)
    with np.errstate(invalid="ignore"):
        ss[is_on] = np.nan_to_num(pstim[is_on], nan=-1.0) > cfg["stimPowerThr"]
    low = dynr < (cfg["dynFloorMult"] * lsb)
    return {"lf_artefact": lf, "stim_spec": ss, "low_dyn": low}


def attribute(recs=RECS):
    for rec in recs:
        f = FEAT / f"{rec}.npz"
        if not f.is_file():
            print(f"[skip] {f.name} not dumped yet -- run with --dump")
            continue
        z = np.load(f, allow_pickle=False)
        feat, is_on, lsb = z["feat"], z["isOn"], float(z["lsb"])
        n_all, n_on = feat[0].size, int(is_on.sum()) * feat.shape[2]
        print(f"\n=== {rec}: {feat.shape[1]} epochs x {feat.shape[2]} channels "
              f"({is_on.mean():.0%} of epochs are stim-ON)")
        print(f"    {'profile':<9}{'any rule':>10}"
              + "".join(f"{r:>14}" for r in RULES)
              + f"{'grad only':>11}{'stim only':>11}{'  (share of ALL channel-epochs)'}")
        for name, cfg in PROFILES.items():
            fr = fires(feat, is_on, lsb, cfg)
            any_ = fr["lf_artefact"] | fr["stim_spec"] | fr["low_dyn"]
            only_g = fr["lf_artefact"] & ~fr["stim_spec"] & ~fr["low_dyn"]
            only_s = fr["stim_spec"] & ~fr["lf_artefact"] & ~fr["low_dyn"]
            print(f"    {name:<9}{any_.mean():>9.1%}"
                  + "".join(f"{fr[r].mean():>14.1%}" for r in RULES)
                  + f"{only_g.mean():>11.1%}{only_s.mean():>11.1%}")
        # The same thing restricted to stim-ON epochs, where the mask actually bites: the
        # whole-recording share above understates stim_spec by the duty cycle.
        print(f"\n    -- restricted to stim-ON epochs only --")
        print(f"    {'profile':<9}{'any rule':>10}" + "".join(f"{r:>14}" for r in RULES))
        for name, cfg in PROFILES.items():
            fr = {k: v[is_on] for k, v in fires(feat, is_on, lsb, cfg).items()}
            any_ = fr["lf_artefact"] | fr["stim_spec"] | fr["low_dyn"]
            print(f"    {name:<9}{any_.mean():>9.1%}"
                  + "".join(f"{fr[r].mean():>14.1%}" for r in RULES))
        del n_all, n_on


if __name__ == "__main__":
    _recs = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or RECS
    if "--dump" in sys.argv:
        for r in _recs:
            print(f"\n### {r}")
            dump(r)
    attribute(_recs)
