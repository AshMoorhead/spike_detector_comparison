"""
test_cond.py -- pins the gap-aware arithmetic in cond.py.

These are the three things that break silently when an intermittent-stim run is subset, so they
are the three things worth a test: the rate denominator, the ISI that spans a gap, and the bin
that straddles a boundary.

    .venv\\Scripts\\python.exe test_cond.py
"""
import contextlib

import numpy as np

import cond


@contextlib.contextmanager
def raises(exc, match):
    """Local stand-in for pytest.raises -- this repo's tests run as plain scripts."""
    try:
        yield
    except exc as e:
        assert match in str(e), f"expected {match!r} in {e!r}"
    else:
        raise AssertionError(f"expected {exc.__name__} containing {match!r}")


def _sel(runs, T=100.0, label="on"):
    return cond.Selection(label, np.asarray(runs, float), np.diff(runs, axis=1).sum(),
                          {"D": np.ones(0, bool)})


# ---------------------------------------------------------------- ISI across gaps
def test_isi_never_crosses_a_gap():
    # two ON blocks; the 10->30 interval spans the OFF gap and must NOT appear
    s = _sel([[0, 10], [30, 40]])
    t = np.array([1.0, 3.0, 9.0, 31.0, 35.0])
    got = s.isis(t)
    assert np.allclose(np.sort(got), [2.0, 4.0, 6.0])
    assert 22.0 not in got and not np.any(got > 10)


def test_isi_single_segment_is_plain_diff():
    s = _sel([[0, 100]], label="all")
    t = np.array([1.0, 2.5, 9.0])
    assert np.allclose(s.isis(t), np.diff(t))


def test_isi_drops_spikes_in_the_gap():
    # a spike at 20 s is in the OFF gap: it should not anchor an interval at all
    s = _sel([[0, 10], [30, 40]])
    assert s.isis(np.array([5.0, 20.0, 35.0])).size == 0


def test_isi_boundary_is_half_open():
    # t1 is exclusive, t0 inclusive -- a spike exactly on t1 belongs to the gap
    s = _sel([[0, 10], [30, 40]])
    assert s.isis(np.array([9.0, 10.0])).size == 0          # 10.0 is outside
    assert np.allclose(s.isis(np.array([30.0, 32.0])), [2.0])   # 30.0 is inside


def test_isi_too_few_spikes():
    s = _sel([[0, 10], [30, 40]])
    assert s.isis(np.array([5.0])).size == 0
    assert s.isis(np.zeros(0)).size == 0


# ---------------------------------------------------------------- bins inside segments
def test_bins_stay_inside_one_segment():
    s = _sel([[0, 25], [30, 40]])
    b = s.bins(10.0)
    # 25 s segment gives two whole bins (the 20-25 remainder is dropped), 10 s gives one
    assert b.shape == (3, 2)
    assert np.allclose(b, [[0, 10], [10, 20], [30, 40]])


def test_bins_are_all_the_same_width():
    s = _sel([[0, 25], [30, 47]])
    b = s.bins(10.0)
    assert np.allclose(np.diff(b, axis=1), 10.0)


def test_bins_none_fit():
    s = _sel([[0, 5], [30, 33]])
    assert s.bins(10.0).shape == (0, 2)


def test_bin_counts_ignore_the_gap():
    s = _sel([[0, 10], [30, 40]])
    t = np.array([1.0, 2.0, 20.0, 31.0])       # 20.0 sits in the OFF gap
    assert np.array_equal(s.bin_counts(t, 10.0), [2, 1])


# ---------------------------------------------------------------- complement (= OFF)
def test_complement_interleaves():
    off = cond._complement(np.array([[10.0, 20.0], [50.0, 60.0]]), 100.0)
    assert np.allclose(off, [[0, 10], [20, 50], [60, 100]])


def test_complement_flush_ends():
    off = cond._complement(np.array([[0.0, 20.0], [80.0, 100.0]]), 100.0)
    assert np.allclose(off, [[20, 80]])


def test_complement_of_everything_is_empty():
    assert cond._complement(np.array([[0.0, 100.0]]), 100.0).shape == (0, 2)


def test_on_and_off_durations_sum_to_the_window():
    on = np.array([[10.0, 20.0], [50.0, 65.0]])
    off = cond._complement(on, 100.0)
    assert np.isclose(np.diff(on, axis=1).sum() + np.diff(off, axis=1).sum(), 100.0)


# ---------------------------------------------------------------- select()
def _npz(**kw):
    base = dict(fs=np.float64(400.0), seconds=np.int64(100),
                detectors=np.array(["D"]),
                D_idx=np.array([400, 8000, 12400]),        # 1 s, 20 s, 31 s
                D_on=np.array([True, False, True]),
                sec_on=np.float64(20.0), rec_id=np.str_("X"),
                on_runs=np.array([[0, 4000], [12000, 16000]]))   # 0-10 s, 30-40 s
    base.update(kw)

    class Z:
        files = list(base)

        def __getitem__(self, k):
            return base[k]

        def __contains__(self, k):
            return k in base
    return Z()


def test_select_on_uses_the_on_denominator():
    s = cond.select(_npz(), "on")
    assert s.T == 20.0 and s.suffix == "_on"
    assert np.array_equal(s.keep("D"), [True, False, True])


def test_select_off_is_the_complement():
    s = cond.select(_npz(), "off")
    assert s.T == 80.0
    assert np.array_equal(s.keep("D"), [False, True, False])


def test_select_all_is_the_whole_window():
    s = cond.select(_npz(), "all")
    assert s.T == 100.0 and s.suffix == "" and s.keep("D").all()


def test_select_on_refuses_a_baseline():
    with raises(SystemExit, "no stim-ON"):
        cond.select(_npz(sec_on=np.float64(0.0)), "on")


def test_select_demands_on_runs():
    z = _npz()
    z.files.remove("on_runs")
    with raises(SystemExit, "on_runs"):
        cond.select(z, "on")


def test_select_rejects_a_typo():
    with raises(SystemExit, "expected all"):
        cond.select(_npz(), "ON_ONLY")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("test_cond.py")
    test_isi_never_crosses_a_gap()
    test_isi_single_segment_is_plain_diff()
    test_isi_drops_spikes_in_the_gap()
    test_isi_boundary_is_half_open()
    test_isi_too_few_spikes()
    test_bins_stay_inside_one_segment()
    test_bins_are_all_the_same_width()
    test_bins_none_fit()
    test_bin_counts_ignore_the_gap()
    test_complement_interleaves()
    test_complement_flush_ends()
    test_complement_of_everything_is_empty()
    test_on_and_off_durations_sum_to_the_window()
    test_select_on_uses_the_on_denominator()
    test_select_off_is_the_complement()
    test_select_all_is_the_whole_window()
    test_select_on_refuses_a_baseline()
    test_select_demands_on_runs()
    test_select_rejects_a_typo()
    print("ALL PASS")
