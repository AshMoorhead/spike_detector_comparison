"""
test_score_sim.py
-----------------
Checks the matcher and both scoring accountings on hand-built cases with known answers, so a
recall number can be trusted without a 50 min Delphos run behind it.

The load-bearing one is test_match_equivalence: compare_spikes.py measures AGREEMENT and
score_sim_detectors.py measures ACCURACY, and they must use the same matcher or their numbers
stop being comparable. That test pins spike_match.match against an inlined copy of the
algorithm compare_spikes used before the extraction.

    .venv\\Scripts\\python.exe -m tests.test_score_sim
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np

from sdc.scoring import score_sim_detectors as ss
from sdc.common.spike_match import match


# ----------------------------------------------------------------------
def test_match_basic():
    ma, mb, off = match([10.0, 20.0], [10.02, 30.0], 0.05)
    assert ma.tolist() == [True, False] and mb.tolist() == [True, False]
    assert abs(off[0] - (-0.02)) < 1e-12

    # empty on either side
    for a, b in ([[], [1.0]], [[1.0], []], [[], []]):
        ma, mb, off = match(a, b, 1.0)
        assert not ma.any() and not mb.any() and off.size == 0
    print("  match: basic pairing, offsets, empty inputs")


def test_match_is_one_to_one():
    """Two true spikes, ONE detection inside tolerance of both -> exactly one TP. A
    many-to-one matcher would score 2 and inflate recall."""
    ma, mb, _ = match([10.00, 10.04], [10.02], 0.05)
    assert ma.sum() == 1 and mb.sum() == 1, (ma, mb)

    # and the mirror: two detections around one true spike -> one TP, one unmatched FP
    ma, mb, _ = match([10.02], [10.00, 10.04], 0.05)
    assert ma.sum() == 1 and mb.sum() == 1
    print("  match: strictly one-to-one in both directions")


def test_match_preserves_input_order():
    """Masks must come back aligned to the INPUT arrays -- the scorer zips them against
    per-spike amplitudes, so a sorted-order mask would mislabel every amplitude."""
    a = np.array([30.0, 10.0, 20.0])                 # deliberately unsorted
    ma, _mb, _ = match(a, [10.01, 30.01], 0.05)      # 30 and 10 match; 20 does not
    assert ma.tolist() == [True, True, False], ma.tolist()
    # the same information in sorted order would be [True, False, True] -- the difference is
    # exactly the bug this guards, since the caller zips ma against per-spike amplitudes
    assert ma[np.argsort(a)].tolist() == [True, False, True]
    print("  match: masks returned in input order, not sorted order")


def test_match_equivalence():
    """spike_match.match == the algorithm compare_spikes._match used before the extraction."""
    def legacy(a, b, tol):
        a, b = np.sort(a), np.sort(b)
        i = j = matched = 0
        while i < a.size and j < b.size:
            d = float(a[i]) - float(b[j])
            if abs(d) <= tol:
                matched += 1
                i += 1
                j += 1
            elif d < 0:
                i += 1
            else:
                j += 1
        return matched

    rng = np.random.default_rng(0)
    for _ in range(200):
        a = np.sort(rng.uniform(0, 100, rng.integers(0, 40)))
        b = np.sort(rng.uniform(0, 100, rng.integers(0, 40)))
        tol = float(rng.uniform(0.05, 2.0))
        ma, _mb, _ = match(a, b, tol)
        assert int(ma.sum()) == legacy(a, b, tol), (a, b, tol)
    print("  match: identical to the pre-extraction compare_spikes._match over 200 cases")


# ----------------------------------------------------------------------
def test_event_scores():
    """3 true, 3 detected, 2 matched -> recall 2/3, precision 2/3."""
    truth = [np.array([1.0, 5.0]), np.array([2.0])]
    amp = [np.array([100.0, 200.0]), np.array([50.0])]
    det = [np.array([1.01, 9.0]), np.array([2.02])]     # 2 TP, 1 FP, 1 FN
    e = ss.event_scores(truth, amp, det, 60.0)
    assert (e["tp"], e["fp"], e["fn"], e["n_true"], e["n_det"]) == (2, 1, 1, 3, 3), e
    assert abs(e["recall"] - 2 / 3) < 1e-12 and abs(e["precision"] - 2 / 3) < 1e-12
    assert abs(e["f1"] - 2 / 3) < 1e-12
    assert abs(e["fp_per_chan_min"] - 1 / (2 * 1.0)) < 1e-12, e["fp_per_chan_min"]
    assert e["hit"].tolist() == [1.0, 0.0, 1.0], e["hit"]
    assert e["amp"].tolist() == [100.0, 200.0, 50.0], "hit flags must stay amplitude-aligned"
    assert abs(e["med_off_ms"] - 15.0) < 1e-9, e["med_off_ms"]   # |−10| and |−20| ms
    print(f"  event scores: TP2/FP1/FN1, recall {e['recall']:.3f}, "
          f"median |offset| {e['med_off_ms']:.0f} ms")


def test_event_scores_channel_isolation():
    """A detection on the wrong channel is never a hit."""
    truth = [np.array([1.0]), np.zeros(0)]
    det = [np.zeros(0), np.array([1.0])]
    e = ss.event_scores(truth, [np.array([10.0]), np.zeros(0)], det, 60.0)
    assert (e["tp"], e["fp"], e["fn"]) == (0, 1, 1), e
    print("  event scores: cross-channel detections are not credited")


def test_window_scores():
    """Bins are 100 ms over a 1 s x 2 ch record -> 20 bins total, exactly accounted for."""
    ss_guard = ss.EDGE_GUARD_SEC
    ss.EDGE_GUARD_SEC = 0.0
    try:
        truth = [np.array([0.05, 0.55]), np.zeros(0)]     # bins 0 and 5 on ch 0
        # ch0: 0.06 -> bin 0 (TP). 0.62 -> bin 6, i.e. the SAME spike as truth 0.55 landing one
        # bin late: window accounting charges it as FP(bin 6) AND FN(bin 5), which is the
        # double-counting fp_adjacent exists to expose. ch1: 0.25 -> a genuinely spurious FP.
        det = [np.array([0.06, 0.62]), np.array([0.25])]
        w = ss.window_scores(truth, det, 1.0, bin_s=0.1)
        assert w["n_bins"] == 20, w["n_bins"]
        assert (w["tp"], w["fn"], w["fp"]) == (1, 1, 2), w
        assert w["tn"] == 20 - 1 - 1 - 2
        assert w["n_pos"] == 2 and w["n_neg"] == 18
        assert abs(w["sensitivity"] - 0.5) < 1e-12
        assert abs(w["specificity"] - 16 / 18) < 1e-12
        assert w["fp_adjacent"] == 1, w["fp_adjacent"]
        print(f"  window scores: TP1/FN1/FP2/TN16 of 20 bins, "
              f"{w['fp_adjacent']}/{w['fp']} FP adjacent to a true bin")
    finally:
        ss.EDGE_GUARD_SEC = ss_guard


def test_window_totals_always_close():
    """The assert inside window_scores is the real guard; make sure it fires on real shapes."""
    rng = np.random.default_rng(2)
    ss_guard = ss.EDGE_GUARD_SEC
    ss.EDGE_GUARD_SEC = 0.0
    try:
        for _ in range(20):
            truth = [np.sort(rng.uniform(0, 30, rng.integers(0, 20))) for _ in range(4)]
            det = [np.sort(rng.uniform(0, 30, rng.integers(0, 40))) for _ in range(4)]
            w = ss.window_scores(truth, det, 30.0, bin_s=0.1)
            assert w["tp"] + w["fp"] + w["fn"] + w["tn"] == w["n_bins"]
    finally:
        ss.EDGE_GUARD_SEC = ss_guard
    print("  window scores: TP+FP+FN+TN == n_bins over 20 random cases")


# ----------------------------------------------------------------------
def test_pdetect_bins():
    """MATLAB's closed last bin: every spike lands in exactly one bin, including the max."""
    amp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    hit = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    ctr, p, n, lo, hi = ss.pdetect_vs_amp(amp, hit, n_bins=4)
    assert n.sum() == amp.size, f"{n.sum()} of {amp.size} spikes binned"
    assert p[0] == 0.0 and p[-1] == 1.0, p
    assert np.all((lo <= np.nan_to_num(p, nan=0.5)) & (np.nan_to_num(p, nan=0.5) <= hi))
    print(f"  P(detected) bins: {n.tolist()} spikes per bin, all {amp.size} accounted for")


def test_wilson():
    lo, hi = ss.wilson(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.4, (lo, hi)
    lo, hi = ss.wilson(10, 10)
    assert hi == 1.0 and 0.6 < lo < 1.0, (lo, hi)
    lo, hi = ss.wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.25
    print("  Wilson CI: stays inside [0,1] at p=0 and p=1")


def test_partial_auc_and_ap():
    # a perfect step: TPR 1 everywhere -> normalised partial AUC 1
    assert abs(ss.partial_auc([0.0, 0.1, 0.2], [1.0, 1.0, 1.0]) - 1.0) < 1e-12
    # a diagonal over the swept range -> 0.5
    assert abs(ss.partial_auc([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]) - 0.5) < 1e-12
    assert np.isnan(ss.partial_auc([0.1], [0.5])), "one point cannot make a curve"
    assert abs(ss.step_ap([0.5], [0.8]) - 0.4) < 1e-12
    assert abs(ss.step_ap([0.5, 1.0], [1.0, 0.5]) - (0.5 * 1.0 + 0.5 * 0.5)) < 1e-12
    print("  partial AUC / step-AP: correct on analytic cases, NaN on a single point")


# ----------------------------------------------------------------------
def test_npz_roundtrip(tmp):
    """A sim npz written in compare_spikes' schema must load with allow_pickle=False and come
    back as seconds on the right channels."""
    from sdc.detect import sim_data as sd
    fs, truth_fs, seconds, n_chan = 400.0, 2000.0, 20.0, 3
    truth_idx = np.array([4000, 12000, 30000], np.int64)      # 2, 6, 15 s @ 2000 Hz
    truth_chan = np.array([0, 1, 1], np.int64)
    det_idx = np.array([800, 2401], np.int64)                 # 2.0, 6.0025 s @ 400 Hz
    det_chan = np.array([0, 1], np.int64)
    cfg = sd.default_cfg(n_chan=n_chan, dur_sec=int(seconds))

    p = tmp / "sim_toy_snr5_op.npz"
    np.savez(p, names=np.array([f"SIM{k+1}" for k in range(n_chan)], dtype="U"),
             fs=fs, seconds=np.int64(seconds), edf="toy.edf",
             detectors=np.array(["Janca"], dtype="U"),
             Janca_idx=det_idx, Janca_chan=det_chan,
             simulated=np.int64(1), truth_idx=truth_idx, truth_chan=truth_chan,
             truth_amp=np.array([10.0, 20.0, 30.0]), truth_fs=truth_fs, snr=5.0,
             noise_std=np.ones(n_chan), rates_per_min=np.linspace(0, 30, n_chan),
             inband_snr=np.ones((n_chan, 1)), inband_dets=np.array(["Janca"], dtype="U"),
             template=np.zeros(4), template_peak=np.int64(0),
             sim_tag="toy", sim_cfg_json=__import__("json").dumps(cfg, sort_keys=True),
             sim_cfg_hash=sd.cfg_hash(cfg),
             run_kind="op", run_point="op", sweep_detector="", sweep_param="",
             sweep_value=float("nan"), merge_ms=20.0, dilate_ms=60.0, tol_ms=50.0,
             detect_fs=400.0, mask_artefacts=np.int64(1), clean_mask=np.int64(1))

    runs = ss.load_runs(tmp)
    assert len(runs) == 1
    r = runs[0]
    assert r["run_kind"] == "op" and r["snr"] == 5.0 and r["n_chan"] == n_chan
    assert np.allclose(r["truth"][0], [2.0]) and np.allclose(r["truth"][1], [6.0, 15.0])
    assert np.allclose(r["det"]["Janca"][0], [2.0])
    e = ss.event_scores(r["truth"], r["truth_amp"], r["det"]["Janca"], r["scored_sec"])
    assert (e["tp"], e["fn"], e["fp"]) == (2, 1, 0), e
    print(f"  npz round trip: loaded {len(runs)} run, {e['tp']} TP / {e['fn']} FN, "
          f"scored window {r['scored_sec']:g}s of {r['seconds']:g}s")


def test_edge_guard_drops_both_sides(tmp):
    """The guard must drop TRUTH as well as detections -- dropping only detections would
    manufacture false negatives at both ends."""
    from sdc.detect import sim_data as sd
    cfg = sd.default_cfg(n_chan=1, dur_sec=20)
    p = tmp / "sim_edge_snr5_op.npz"
    np.savez(p, names=np.array(["SIM1"], dtype="U"), fs=400.0, seconds=np.int64(20),
             edf="toy.edf", detectors=np.array(["Janca"], dtype="U"),
             Janca_idx=np.array([100, 4000], np.int64),      # 0.25 s (guarded) and 10 s
             Janca_chan=np.array([0, 0], np.int64),
             simulated=np.int64(1),
             truth_idx=np.array([500, 20000], np.int64),     # 0.25 s (guarded) and 10 s
             truth_chan=np.array([0, 0], np.int64),
             truth_amp=np.array([10.0, 10.0]), truth_fs=2000.0, snr=5.0,
             noise_std=np.ones(1), rates_per_min=np.zeros(1),
             inband_snr=np.ones((1, 1)), inband_dets=np.array(["Janca"], dtype="U"),
             template=np.zeros(4), template_peak=np.int64(0), sim_tag="toy",
             sim_cfg_json=__import__("json").dumps(cfg, sort_keys=True),
             sim_cfg_hash=sd.cfg_hash(cfg), run_kind="op", run_point="op",
             sweep_detector="", sweep_param="", sweep_value=float("nan"),
             merge_ms=20.0, dilate_ms=60.0, tol_ms=50.0, detect_fs=400.0,
             mask_artefacts=np.int64(1), clean_mask=np.int64(1))
    r = [x for x in ss.load_runs(tmp) if x["path"].name.startswith("sim_edge")][0]
    assert r["truth"][0].size == 1 and r["det"]["Janca"][0].size == 1
    e = ss.event_scores(r["truth"], r["truth_amp"], r["det"]["Janca"], r["scored_sec"])
    assert (e["tp"], e["fp"], e["fn"]) == (1, 0, 0), e
    print(f"  edge guard: dropped the {ss.EDGE_GUARD_SEC:g}s-edge spike from BOTH sides -> "
          f"recall {e['recall']:.0f}, not 0.5")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="score_sim_test_"))
    try:
        print("test_score_sim.py")
        test_match_basic()
        test_match_is_one_to_one()
        test_match_preserves_input_order()
        test_match_equivalence()
        test_event_scores()
        test_event_scores_channel_isolation()
        test_window_scores()
        test_window_totals_always_close()
        test_pdetect_bins()
        test_wilson()
        test_partial_auc_and_ap()
        test_npz_roundtrip(tmp)
        test_edge_guard_drops_both_sides(tmp)
        print("ALL PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
