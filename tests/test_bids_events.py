"""
test_bids_events.py -- pins the reader for the expert-labelled BIDS dataset.

This is ground truth: every mistake here becomes a recall or precision number that looks like a
statement about a detector. The three that would be silent are the ones tested hardest -- the
seconds->sample conversion, a labelled contact that is not in the channel list, and reading the
wrong one of the two events files.

Runs against the real dataset if it is present and skips cleanly if not, so it stays useful on
a machine without the data.

    .venv\\Scripts\\python.exe -m tests.test_bids_events
"""
import csv
import tempfile
from pathlib import Path

import numpy as np

from sdc.scoring import bids_events as be

BIDS = Path(r"C:\Users\amoo0039\Documents\ieeg_ieds_bids_final\ieeg_ieds_bids")


# ---------------------------------------------------------------- synthetic fixture
def _make(tmp, sub="sub-01", chans=("A1", "A2", "B1"), events=None):
    """Minimal BIDS tree: channels.tsv + the derivatives interpretation file."""
    events = events if events is not None else [(1.5, "Rt sharp", "A1 A2"),
                                                (0.25, "Rt sharp", "B1")]
    d = tmp / sub / "ieeg"
    d.mkdir(parents=True, exist_ok=True)
    (tmp / "derivatives").mkdir(exist_ok=True)
    (d / f"{sub}_task-sleep_ieeg.edf").write_bytes(b"")     # presence is all `subjects()` needs
    with open(d / f"{sub}_task-sleep_channels.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["name", "type"])
        for c in chans:
            w.writerow([c, "SEEG"])
    with open(tmp / "derivatives" / f"{sub}_task-sleep_events_interpretation.tsv",
              "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["time_in_sec", "annotation", "chans"])
        for t, a, c in events:
            w.writerow([t, a, c])
    return tmp


def test_events_are_sorted_by_time(tmp):
    _make(tmp)
    t, ch, _ = be.load_events(tmp, "sub-01", ["A1", "A2", "B1"])
    assert list(t) == [0.25, 1.5]
    assert ch[0] == ["B1"] and ch[1] == ["A1", "A2"]      # channel lists travel WITH the sort


def test_truth_per_channel_shape_and_placement(tmp):
    _make(tmp)
    chans = ["A1", "A2", "B1"]
    d = be.load_subject(tmp, "sub-01")
    per = be.truth_per_channel(d["times"], d["chan_lists"], chans)
    assert len(per) == len(chans)                          # one entry per channel, always
    assert np.allclose(per[0], [1.5]) and np.allclose(per[1], [1.5])
    assert np.allclose(per[2], [0.25])


def test_channel_with_no_marks_is_empty_not_missing(tmp):
    _make(tmp, chans=("A1", "A2", "B1", "C9"))
    d = be.load_subject(tmp, "sub-01")
    per = be.truth_per_channel(d["times"], d["chan_lists"], d["channels"])
    assert per[3].size == 0 and len(per) == 4              # zips against a detector's output


def test_seconds_to_samples_rounds_not_floors(tmp):
    # 0.8025 s at 1000 Hz is 802.5 -- floor would put it a full sample early, and at a 50 ms
    # match tolerance a systematic half-sample bias is not visible until it matters.
    _make(tmp, events=[(0.8029, "x", "A1")])
    d = be.load_subject(tmp, "sub-01")
    per = be.truth_per_channel(d["times"], d["chan_lists"], d["channels"], fs=1000.0)
    assert per[0][0] == 803, per[0]
    assert per[0].dtype.kind == "i"                        # indices, not floats


def test_unknown_contact_raises_rather_than_dropping(tmp):
    # Dropping a labelled contact deletes a true positive: recall falls and precision RISES,
    # which reads as a better detector. It must be loud.
    _make(tmp, events=[(1.0, "x", "A1 ZZ9")])
    try:
        be.load_events(tmp, "sub-01", ["A1", "A2", "B1"], strict=True)
    except ValueError as e:
        assert "ZZ9" in str(e)
    else:
        raise AssertionError("expected a ValueError naming the unknown contact")


def test_non_strict_drops_but_keeps_the_event(tmp):
    _make(tmp, events=[(1.0, "x", "A1 ZZ9")])
    t, ch, _ = be.load_events(tmp, "sub-01", ["A1", "A2", "B1"], strict=False)
    assert len(t) == 1 and ch[0] == ["A1"]


def test_subjects_needs_both_files(tmp):
    _make(tmp)
    assert be.subjects(tmp) == ["sub-01"]
    (tmp / "derivatives" / "sub-01_task-sleep_events_interpretation.tsv").unlink()
    assert be.subjects(tmp) == []                          # no truth -> not scoreable


# ---------------------------------------------------------------- against the real dataset
def test_real_dataset():
    if not BIDS.is_dir():
        print("  [skip] BIDS dataset not present")
        return
    subs = be.subjects(BIDS)
    assert len(subs) == 25, subs
    total = sum(be.load_subject(BIDS, s)["n_events"] for s in subs)
    # The dataset README states 852 annotated IEDs. If this ever disagrees, either the reader
    # is reading the wrong file or the release changed -- both worth stopping for.
    assert total == 852, f"expected 852 IEDs across the release, read {total}"

    d = be.load_subject(BIDS, "sub-01")
    per = be.truth_per_channel(d["times"], d["chan_lists"], d["channels"], fs=1000.0)
    i = d["channels"].index("RMH1")
    assert list(per[i][:3]) == [802, 5614, 5906]           # the first three rows of the TSV
    # strict=True across every subject is the real check that no name reconciliation is needed
    for s in subs:
        be.load_subject(BIDS, s, strict=True)
    print(f"  [real] {len(subs)} subjects, {total} IEDs, all contacts resolve")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("test_bids_events.py")
    with tempfile.TemporaryDirectory() as td:
        for fn in (test_events_are_sorted_by_time, test_truth_per_channel_shape_and_placement,
                   test_channel_with_no_marks_is_empty_not_missing,
                   test_seconds_to_samples_rounds_not_floors,
                   test_unknown_contact_raises_rather_than_dropping,
                   test_non_strict_drops_but_keeps_the_event, test_subjects_needs_both_files):
            sub = Path(td) / fn.__name__
            sub.mkdir()
            fn(sub)
    test_real_dataset()
    print("ALL PASS")
