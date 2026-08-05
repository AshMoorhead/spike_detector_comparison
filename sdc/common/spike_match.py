"""
spike_match.py
--------------
The one greedy spike matcher, shared by the agreement code (compare_spikes.py) and the
ground-truth scorer (score_sim_detectors.py).

WHY ITS OWN MODULE
  MATLAB's match_mask (score_sim_detectors.m) and compare_spikes._match are the same algorithm
  written twice. The scorer cannot import compare_spikes -- that module is a script with no
  main(), so importing it would run a 600 s EDF load and a ~5 min Delphos call as an import
  side effect. Copying the matcher instead is exactly the kind of drift the shared 20 ms merge
  rule exists to prevent: the moment the two copies disagree, "detector agreement" and
  "detector accuracy" stop being measured the same way and nothing downstream is comparable.

SEMANTICS (ported from match_mask, which is in turn derived from spikeCoincidenceSimilarity.m)
  * Symmetric tolerance: |a - b| <= tol counts as a match. No forward/backward asymmetry --
    many spike benchmarks allow more lag than lead; this one deliberately does not.
  * Strictly ONE-TO-ONE: on a match BOTH pointers advance, so one true spike consumes exactly
    one detection. Duplicate detections around a single true spike are not scored as extra
    hits -- they fall through unmatched and cost precision, which is the point.
  * GREEDY IN TIME, not optimal: the first detection inside the window wins even if a later one
    is closer, and it can consume a detection that would have been the only match for the next
    true spike. With tol=50 ms against a 200 ms refractory floor this essentially never fires,
    but it is the MATLAB's behaviour and is reproduced rather than improved.
"""
import numpy as np


def match(a, b, tol):
    """Greedy one-to-one match between two spike-time arrays.

    a, b  : array-like, same units (samples or seconds). Need not be sorted.
    tol   : match radius in those units, inclusive.

    Returns (mask_a, mask_b, offsets):
      mask_a  bool[len(a)]  True where that element of `a` found a partner, in INPUT order
      mask_b  bool[len(b)]  ditto for `b`
      offsets float[n_matched]  a - b for each matched pair, in ascending time order
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    mask_a = np.zeros(a.size, bool)
    mask_b = np.zeros(b.size, bool)
    if a.size == 0 or b.size == 0:
        return mask_a, mask_b, np.zeros(0, float)

    # stable sort so ties keep input order, and the inverse maps masks back (match_mask's
    # `mask(ord) = m`) -- callers zip these against per-spike amplitudes, so order matters.
    oa = np.argsort(a, kind="stable")
    ob = np.argsort(b, kind="stable")
    sa, sb = a[oa], b[ob]

    hit_a = np.zeros(sa.size, bool)
    hit_b = np.zeros(sb.size, bool)
    offs = []
    i = j = 0
    while i < sa.size and j < sb.size:
        d = sa[i] - sb[j]
        if abs(d) <= tol:
            hit_a[i] = hit_b[j] = True
            offs.append(float(d))
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1

    mask_a[oa] = hit_a
    mask_b[ob] = hit_b
    return mask_a, mask_b, np.asarray(offs, float)
