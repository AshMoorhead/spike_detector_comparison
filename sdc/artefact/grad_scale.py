"""
sdc.artefact.grad_scale
-----------------------
Does the QC's absolute gradient threshold condemn patients unequally because of recording gain?

    .venv\\Scripts\\python.exe -m sdc.artefact.grad_scale

`windowed_artefact_detector` flags an epoch when `max|diff(x)| > gradThr`, in signal units per
sample, with no normalisation of any kind -- not per channel, not per recording. (It is stored
as `gradRatio`, inherited from the MATLAB, but it is not a ratio.)

WHY THAT MIGHT BE A PROBLEM, AND WHY IT MIGHT NOT
  Barkmeier's block scaling exists precisely because native amplitude varies between
  recordings: measured on the three marked baselines, the median channel amplitude is 25.85 uV
  for P1 against 11.77 and 11.20 for P5 and P8 -- a 2.3x spread. A fixed absolute gradient
  threshold therefore meets a systematically larger signal on P1.

  But it does not follow that the threshold is wrong. STIMULATION artefact -- the thing this QC
  mostly exists to catch -- injects a voltage that does NOT scale with the channel's background,
  so an absolute bar is the right shape for it. The question is only whether the between-PATIENT
  gain difference distorts how much each patient gets condemned.

WHAT THIS MEASURES
  For each recording: the gradRatio distribution, the fraction of channel-epochs the LF rule
  condemns at gradThr, and the same two after dividing gradRatio by that recording's own median
  amplitude. If the condemned fraction tracks amplitude before normalisation and stops tracking
  it after, there is something to fix; if it does not track it in the first place, the absolute
  threshold is already fine and no change is warranted.

  The normalisation deliberately mirrors Barkmeier's: ONE scalar per recording, applied to all
  channels. A per-channel version would import Janca's failure mode -- a relative threshold
  fires on quiet channels, which is why Janca produces 5.12 empty-channel FP/min against
  Barkmeier's 0.58.
"""
import numpy as np

from sdc.common.paths import RUNS

# native median channel amplitude, uV, from the Barkmeier measurement chain (scale_denom.py):
# butter(2,1Hz,high) -> butter(4,35Hz,low), median over channels of mean|x|, per 1-min block.
AMPLITUDE = {"P1_pre": 25.85, "P5_pre": 11.77, "P8_ANT145_pre": 11.20}
GRAD_THR = 1000.0          # QC_PROFILES['finalv2']
REF = float(np.median(list(AMPLITUDE.values())))


def load(rec):
    p = RUNS / f"qcfeat_{rec}.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    return np.asarray(z["gradRatio"], float)


def main():
    rows = []
    for rec, amp in AMPLITUDE.items():
        g = load(rec)
        if g is None:
            print(f"[warn] {rec}: no qcfeat npz")
            continue
        v = g[np.isfinite(g)]
        # absolute rule, as shipped
        frac = float((v > GRAD_THR).mean())
        # one scalar per recording, mirroring Barkmeier's block scaling
        frac_n = float((v * REF / amp > GRAD_THR).mean())
        rows.append((rec, amp, np.median(v), np.percentile(v, 95), frac, frac_n))
    if not rows:
        return
    print(f"gradThr = {GRAD_THR:g} uV/sample   reference amplitude = {REF:.2f} uV\n")
    print(f"{'recording':<16}{'amp uV':>8}{'med grad':>10}{'p95 grad':>10}"
          f"{'condemned':>11}{'normalised':>12}")
    for rec, amp, med, p95, f, fn in rows:
        print(f"{rec:<16}{amp:>8.2f}{med:>10.1f}{p95:>10.1f}{f * 100:>10.2f}%{fn * 100:>11.2f}%")

    amps = np.array([r[1] for r in rows])
    meds = np.array([r[2] for r in rows])
    fr = np.array([r[4] for r in rows])
    frn = np.array([r[5] for r in rows])
    print(f"\nspread across recordings (max/min):")
    print(f"  native amplitude   {amps.max() / amps.min():.2f}x")
    print(f"  median gradient    {meds.max() / meds.min():.2f}x"
          f"   <- if this tracks amplitude, gain is driving the gradient")
    if fr.min() > 0:
        print(f"  condemned fraction {fr.max() / fr.min():.2f}x  ->  "
              f"{frn.max() / max(frn.min(), 1e-12):.2f}x after normalising")
    else:
        print(f"  condemned fraction {fr.min() * 100:.3f}%-{fr.max() * 100:.3f}% "
              f"(a zero makes the ratio undefined)")
        print(f"  normalised         {frn.min() * 100:.3f}%-{frn.max() * 100:.3f}%")
    print("\nread it as: normalising is worth doing only if the condemned fraction tracks "
          "amplitude\nand the normalised column is visibly more even than the raw one.")


if __name__ == "__main__":
    main()
