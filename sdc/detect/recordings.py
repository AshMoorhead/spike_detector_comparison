"""
sdc.detect.recordings
---------------------
The recording table, on its own, as the single definition.

WHY IT IS ITS OWN MODULE
  compare_spikes.py is a SCRIPT -- importing it runs the whole comparison -- so run_windows.py
  could not import the table from it and kept a second copy instead. The two then drifted:
  registering the 2 Hz trial in one left the other raising KeyError on the same name. Putting
  the table here lets both share one definition without either importing the other's side
  effects.

A recording name maps to (patient, trial_index, file_type). Everything downstream resolves the
actual EDF through the pipeline's own identity chain -- load_trials -> get_patient -> get_trial
-> resolve_file -- so no path is ever hardcoded here. `trial_index` is 1-BASED, matching
get_trial.
"""

RECORDINGS = {
    # P1/P5 trial 1 is ANT at 145 Hz
    "P1_pre":  dict(patient=1, trial_index=1, file_type="pre"),
    "P1_stim": dict(patient=1, trial_index=1, file_type="stim"),
    "P5_pre":  dict(patient=5, trial_index=1, file_type="pre"),
    "P5_stim": dict(patient=5, trial_index=1, file_type="stim"),

    # P1 trial 2 is ANT at 2 Hz. Its own pre file is baseline_1, NOT the baseline the 145 Hz
    # trial uses -- so it carries its own paired baseline and must not borrow P1_pre's. The
    # stimulation frequency also changes what `pStim` measures: at 2 Hz the +-5 Hz band sits
    # inside delta, so a feature dump made for this trial is not comparable with a 145 Hz one
    # and must be tagged (see qc_features.dump's `tag`).
    "P1_2hz_pre":  dict(patient=1, trial_index=2, file_type="pre"),
    "P1_2hz_stim": dict(patient=1, trial_index=2, file_type="stim"),
}

# The AES cohort -- 17 trials, 34 recordings -- merged in. Generated from AES_trials.json, so
# these resolve by FILENAME in edf_path() rather than by trial_index; see the note there.
from sdc.detect.cohort import RECORDINGS as _COHORT_RECORDINGS   # noqa: E402
for _k, _v in _COHORT_RECORDINGS.items():
    RECORDINGS.setdefault(_k, _v)

from pathlib import Path                                      # noqa: E402

BASE_DIR = Path(r"C:\Users\amoo0039\Documents\local")
META_PATH = BASE_DIR / "data_meta" / "stim_trials.json"

# The cohort's own trials file, and the authority for every cohort recording. Kept as a LOCAL
# copy (refreshed from the S: original) so a long batch does not depend on the network drive
# staying up, and so a half-read file cannot silently change a threshold mid-run.
AES_META = BASE_DIR / "data_meta" / "trials_json" / "AES_trials.json"


def _aes_trial(c):
    """The genuine trial record behind a cohort entry, matched on patient + stim FILENAME.

    Returning the real record rather than a hand-built one matters: `seeg.stim` needs
    `stim_channels`, and `make_cfg_artefact` needs `stim_frequency`. A synthesised dict carrying
    only the three fields cohort.py happens to store would satisfy the type and then fail, or
    worse silently disable the stim rule, at the point of use.
    """
    import json

    for pat in json.load(open(AES_META, encoding="utf-8-sig")):
        if str(pat["patient_id"]).lstrip("Pp") != str(c["patient"]):
            continue
        for t in pat["trials"]:
            if t["filename"] == c["stim"]:
                return t
    raise SystemExit(f"no trial in {AES_META.name} for patient {c['patient']} "
                     f"file {c['stim']!r} -- cohort.py and the JSON have diverged; "
                     f"regenerate with tools/gen_cohort.py")


def edf_path(rec):
    """The RAW EDF for a recording name, and its trial entry.

    COHORT ENTRIES RESOLVE BY FILENAME, NOT BY trial_index. `trial_index` is only meaningful
    against the JSON it was read from, and the cohort is defined against AES_trials.json while
    this module's META_PATH points at the local stim_trials.json -- which lists different trials
    in a different order. Checked: 2 of the 17 cohort trials would have resolved the WRONG
    recording that way (P3's 7 Hz trial -> a 145 Hz file, P8's ANT trial -> a Pulv file), and
    both would have produced entirely plausible numbers. So the cohort table carries the actual
    stim/baseline filenames and they are used directly.

    Legacy entries (P1_pre, P1_stim, ...) keep the trial_index chain, which is correct for them
    because they were defined against the local JSON in the first place.

    The second return value is the TRIAL RECORD, or None for a '_pre' file -- matching
    `resolve_file`'s contract, where a baseline has no stim trial and a None there is what
    disables the stim-spectral rule downstream.
    """
    from sdc.detect.cohort import COHORT

    for stem, c in COHORT.items():
        for suffix, key in (("_stim", "stim"), ("_pre", "baseline")):
            if rec == f"{stem}{suffix}":
                fn = c[key]
                if not fn:
                    raise SystemExit(f"{rec}: no {key} file is named for this trial.")
                p = BASE_DIR / f"P{c['patient']}" / f"{fn}.edf"
                if not p.is_file():                      # P10 has 'Baseline_3.edf'
                    alt = [x for x in p.parent.glob("*.edf") if x.stem.lower() == fn.lower()]
                    if alt:
                        p = alt[0]
                return p, (_aes_trial(c) if key == "stim" else None)

    from seeg import load_trials, get_patient, get_trial, resolve_file

    cfg = RECORDINGS[rec]
    trial = get_trial(get_patient(load_trials(META_PATH), cfg["patient"]), cfg["trial_index"])
    stem, entry = resolve_file(trial, cfg["file_type"])
    return BASE_DIR / f"P{cfg['patient']}" / f"{stem}.edf", entry


def ez_channels(patient, key="EZ"):
    """The patient's epileptogenic-zone bipolar channels, from AES_trials.json.

    `key` also accepts "THC". These are recorded per PATIENT rather than per trial, so the same
    set applies to every recording from that implant. Returned as names because that is what
    survives a montage change; an index list would silently point elsewhere.
    """
    import json

    for pat in json.load(open(AES_META, encoding="utf-8-sig")):
        if str(pat["patient_id"]).lstrip("Pp") == str(patient):
            return [str(c).strip().replace("​", "") for c in pat.get(key) or []]
    raise SystemExit(f"patient {patient} not in {AES_META.name}")


def montage_path(patient):
    """The clinician-defined montage CSV for a patient: <BASE_DIR>/<patient>/<patient>_montage.csv"""
    # `patient` in RECORDINGS is the bare integer (1), while the directory is "P1" -- the same
    # f"P{...}" convention edf_path uses two functions above. Passing "P1" already works too.
    tag = str(patient) if str(patient).startswith("P") else f"P{patient}"
    return BASE_DIR / tag / f"{tag}_montage.csv"


def load_patient_montage(patient, allow_derived=False):
    """The clinical montage, or a loud failure. Returns rows, or None when derived is allowed.

    THERE IS NO SILENT FALLBACK, deliberately. `compare_spikes` called `derive_montage`
    unconditionally for the whole life of this project, so every run was made on 226
    consecutively-paired contacts rather than the clinician's 164 -- and nothing said so. The
    derived montage pairs (n, n+1) on each shaft and keeps everything; the clinical one is a
    curated subset that drops contacts judged unusable (62 of 226 on P1). Those are different
    experiments, and the difference was invisible.

    `allow_derived=True` (MONTAGE=derived) opts back in explicitly, for a patient whose montage
    file does not exist yet.
    """
    from seeg.montage import read_montage

    p = montage_path(patient)
    if p.is_file():
        return read_montage(p)
    if allow_derived:
        print(f"[montage] {p.name} not found and MONTAGE=derived -- deriving bipolar pairs "
              f"from the EDF labels. Channel set will NOT match the clinical montage.")
        return None
    raise SystemExit(
        f"No montage for {patient}: expected {p}\n"
        f"  A derived montage can be used instead -- it pairs consecutive contacts on each "
        f"shaft and keeps every contact, so it is a SUPERSET of the clinical one (226 vs 164 "
        f"pairs on P1) and includes contacts the clinician excluded.\n"
        f"  To use it deliberately, set MONTAGE=derived.")
