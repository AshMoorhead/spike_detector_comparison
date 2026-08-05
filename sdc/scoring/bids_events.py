"""
sdc.scoring.bids_events
-----------------------
Read the expert IED annotations from the BIDS iEEG sleep dataset.

    from sdc.scoring.bids_events import subjects, load_subject
    for sub in subjects(BIDS_ROOT):
        rec = load_subject(BIDS_ROOT, sub)

25 subjects, 852 expert-marked interictal epileptiform discharges, ~181-245 s each,
8-52 channels, recorded on a Blackrock system through sleep and referenced to a central
scalp electrode. Roehri-style detectors have never been scored against real expert marks in
this repo -- only against each other and against synthetic ground truth.

WHICH FILE THE TRUTH IS IN
  NOT `sub-XX/ieeg/sub-XX_task-sleep_events.tsv`. That one carries `trial_type` values like
  "Rt sharp", i.e. a HEMISPHERE, so scoring against it could only ever be done per side.
  `derivatives/sub-XX_task-sleep_events_interpretation.tsv` carries the same events plus a
  `chans` column -- a space-separated list of the contacts each discharge appears on:

      time_in_sec   annotation   chans
      0.802         Rt sharp     RMH1 RA1 RA3

  So the truth is PER CHANNEL, which is what the detectors produce and what spike_match needs.
  Verified across subjects 01/05/12/25: every name in `chans` appears in that subject's
  `channels.tsv`, so no name reconciliation is required -- and `strict=True` below keeps it
  that way rather than silently dropping a contact if a future subject disagrees.

WHY THIS DATA IS NOT PREPROCESSED
  The comparison's own recordings get median-filtered and decimated so all three detectors
  share an input. This dataset is left exactly as it is -- no median, no anti-alias, no
  decimation, and NO MONTAGE. Leaving it unmontaged is the important one: the labels name
  monopolar contacts, so unmontaged channels ARE the labelled channels. Bipolar would force a
  contact->pair mapping, and the choice of whether to credit one pair or both would silently
  set recall.

  A NOTE ON WHAT THAT COSTS: it is scalp-referenced, so a discharge is common-mode across
  nearby contacts and the detectors see something unlike the bipolar signal they were tuned
  on. That is a real caveat on the absolute numbers, not on the ranking between detectors.
"""
import csv
from pathlib import Path

import numpy as np


def subjects(root):
    """Sorted subject ids ('sub-01', ...) that have BOTH an EDF and an interpretation file."""
    root = Path(root)
    out = []
    for d in sorted(root.glob("sub-*")):
        if not d.is_dir():
            continue
        sub = d.name
        if (d / "ieeg" / f"{sub}_task-sleep_ieeg.edf").is_file() and \
           (root / "derivatives" / f"{sub}_task-sleep_events_interpretation.tsv").is_file():
            out.append(sub)
    return out


def _tsv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def load_channels(root, sub):
    """Channel names in FILE ORDER, plus the per-channel metadata rows.

    File order matters: the detectors index channels positionally, so truth must be built
    against the same order the EDF reader returns."""
    rows = _tsv(Path(root) / sub / "ieeg" / f"{sub}_task-sleep_channels.tsv")
    return [r["name"] for r in rows], rows


def load_events(root, sub, channels=None, strict=True):
    """Expert IEDs for one subject.

    Returns (times, chan_lists, annotations):
      times        (n,) float seconds from file start, sorted
      chan_lists   list of n lists of channel NAMES the expert marked that discharge on
      annotations  list of n raw annotation strings ("Rt sharp", "mesial left phg", ...)

    `strict` raises if a labelled contact is absent from `channels`, rather than dropping it.
    A silently dropped contact removes a true positive and inflates precision, so it must be
    loud."""
    rows = _tsv(Path(root) / "derivatives" / f"{sub}_task-sleep_events_interpretation.tsv")
    times, chans, annot = [], [], []
    unknown = set()
    for r in rows:
        names = (r.get("chans") or "").split()
        if channels is not None:
            missing = [n for n in names if n not in channels]
            unknown.update(missing)
            names = [n for n in names if n in channels]
        times.append(float(r["time_in_sec"]))
        chans.append(names)
        annot.append((r.get("annotation") or "").strip())
    if unknown and strict:
        raise ValueError(f"{sub}: labelled contacts absent from channels.tsv: "
                         f"{sorted(unknown)}. Dropping them would delete true positives and "
                         f"inflate precision -- reconcile the names first.")
    order = np.argsort(times)
    return (np.asarray(times, float)[order],
            [chans[i] for i in order],
            [annot[i] for i in order])


def truth_per_channel(times, chan_lists, channels, fs=None):
    """Expert marks as one array per channel, in `channels` order.

    With `fs`, values are 0-based SAMPLE indices (round, not floor -- a mark at 0.8025 s
    belongs to the nearer sample); without it, seconds. Channels with no marks get an empty
    array, so the result always has len(channels) entries and can be zipped straight against a
    detector's output."""
    idx = {c: i for i, c in enumerate(channels)}
    per = [[] for _ in channels]
    for t, names in zip(times, chan_lists):
        for n in names:
            if n in idx:
                per[idx[n]].append(t)
    if fs is None:
        return [np.asarray(sorted(p), float) for p in per]
    return [np.round(np.asarray(sorted(p), float) * fs).astype(int) for p in per]


def load_subject(root, sub, strict=True):
    """Everything needed to score one subject: paths, channel order, and per-channel truth."""
    root = Path(root)
    channels, chan_rows = load_channels(root, sub)
    times, chan_lists, annot = load_events(root, sub, channels, strict=strict)
    return {
        "subject": sub,
        "edf": str(root / sub / "ieeg" / f"{sub}_task-sleep_ieeg.edf"),
        "channels": channels,
        "channel_rows": chan_rows,
        "times": times,              # seconds, sorted
        "chan_lists": chan_lists,    # names per event
        "annotations": annot,
        "n_events": len(times),
        "labelled_channels": sorted({n for names in chan_lists for n in names}),
    }
