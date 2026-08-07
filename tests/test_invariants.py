"""
test_invariants.py -- does each check actually catch the bug it was written for?

A structural check that never fires is worse than none: it buys confidence without providing
any. So every test here REPRODUCES A BUG THAT ACTUALLY OCCURRED in this project and asserts the
check would have stopped it before the run was written.

    .venv\\Scripts\\python.exe -m tests.test_invariants
"""
import glob

import numpy as np

from sdc.common.invariants import check_run, InvariantError


def _good(n_sec=100, n_chan=4, fs=1000.0):
    """A well-formed run: 100 s, 4 channels, 30 s of stim ON in two blocks."""
    on_sec = np.zeros(n_sec, bool)
    on_sec[10:25] = True
    on_sec[60:75] = True
    cps = np.full((n_sec, n_chan), fs, np.uint16)          # every second fully analysable
    idx = np.array([5_000, 15_000, 65_000, 90_000])
    return {
        "names": np.array([f"C{i}" for i in range(n_chan)]),
        "detectors": np.array(["Janca"]),
        "fs": np.float64(fs), "seconds": np.int64(n_sec),
        "Janca_idx": idx, "Janca_chan": np.array([0, 1, 2, 3]),
        "Janca_on": on_sec[(idx / fs).astype(int)],
        "clean_per_sec": cps,
        "clean_sec_on": cps[on_sec].sum(axis=0) / fs,
        "clean_sec_off": cps[~on_sec].sum(axis=0) / fs,
        "sec_on": np.float64(on_sec.sum()), "sec_off": np.float64((~on_sec).sum()),
        "on_runs": np.array([[10_000, 25_000], [60_000, 75_000]], np.int64),
    }


def _raises(d, n_samp, contains):
    try:
        check_run(d, n_samp=n_samp, fs=float(d["fs"]))
    except InvariantError as e:
        assert contains in str(e), f"expected {contains!r} in {e}"
        return
    raise AssertionError(f"check_run did not raise; expected {contains!r}")


def test_a_good_run_passes():
    assert check_run(_good(), n_samp=100_000, fs=1000.0)


def test_catches_scaled_condition_split():
    """THE BUG: clean_sec_on/off were `clean_sec * on.mean()` -- total analysable time split by
    the duty cycle. That credits a channel masked during stimulation with ~30% of its whole
    recording as ON clean time (231 s of fiction on P1_stim), understating ON rates and
    OVERSTATING suppression -- biasing a finding in the direction it was reporting."""
    d = _good()
    total = d["clean_per_sec"].sum(axis=0) / float(d["fs"])
    frac_on = float(d["sec_on"]) / float(d["seconds"])
    d["clean_sec_on"] = total * frac_on          # the scaled version
    d["clean_sec_off"] = total * (1 - frac_on)
    # scaling happens to preserve the SUM here, so the sum check alone would miss it; what
    # catches it is that a channel's ON time must come from the ON seconds specifically.
    d["clean_per_sec"] = d["clean_per_sec"].copy()
    d["clean_per_sec"][5:8, 0] = 0               # channel 0 masked in OFF seconds only
    _raises(d, 100_000, "SCALED")


def test_catches_an_unmasked_arm():
    """THE BUG: Delphos was run by the driver and written WITHOUT the artefact mask the other
    two had already been through, inflating it ~20%. It read as a Delphos result. An arm run
    against a different array shows up as indices past the end."""
    d = _good()
    d["Janca_idx"] = np.append(d["Janca_idx"], 250_000)     # from a longer/other array
    d["Janca_chan"] = np.append(d["Janca_chan"], 0)
    d["Janca_on"] = np.append(d["Janca_on"], False)
    _raises(d, 100_000, "beyond the recording")


def test_catches_conditions_that_do_not_partition_the_recording():
    d = _good()
    d["sec_off"] = np.float64(float(d["sec_off"]) - 20)
    _raises(d, 100_000, "but the recording is")


def test_catches_analysable_time_exceeding_the_recording():
    d = _good()
    d["clean_sec_on"] = d["clean_sec_on"] * 10
    _raises(d, 100_000, "analysable in a")


def test_catches_misaligned_on_flags():
    d = _good()
    d["Janca_on"] = d["Janca_on"][:-1]
    _raises(d, 100_000, "flags for")


def test_catches_overlapping_stim_segments():
    d = _good()
    d["on_runs"] = np.array([[10_000, 40_000], [30_000, 75_000]], np.int64)
    _raises(d, 100_000, "overlap")


def test_catches_index_channel_length_mismatch():
    d = _good()
    d["Janca_chan"] = d["Janca_chan"][:-1]
    _raises(d, 100_000, "channel labels")


def test_every_real_run_passes():
    """The checks must not fire on anything already on disk -- a check that cries wolf gets
    switched off."""
    n = bad = 0
    for f in sorted(glob.glob("runs/*.npz")) + sorted(glob.glob("sim_runs/*.npz")):
        z = np.load(f, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        if "seconds" not in d or "fs" not in d:
            continue
        n += 1
        try:
            check_run(d, n_samp=int(float(d["seconds"]) * float(d["fs"])), fs=float(d["fs"]))
        except InvariantError as e:
            bad += 1
            print(f"    FAIL {f}: {e}")
    print(f"  [real] {n} runs on disk, {bad} failures")
    assert bad == 0


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("test_invariants.py")
    test_a_good_run_passes()
    test_catches_scaled_condition_split()
    test_catches_an_unmasked_arm()
    test_catches_conditions_that_do_not_partition_the_recording()
    test_catches_analysable_time_exceeding_the_recording()
    test_catches_misaligned_on_flags()
    test_catches_overlapping_stim_segments()
    test_catches_index_channel_length_mismatch()
    test_every_real_run_passes()
    print("ALL PASS")
