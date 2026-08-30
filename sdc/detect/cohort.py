"""
sdc.detect.cohort
------------------
The AES cohort: 17 trials from 7 patients, generated from AES_trials.json.

SELECTION  target in {ANT, Pulv} and either
             145 Hz INTERMITTENT  -> 9 trials, ON/OFF contrast available
             2 or 7 Hz CONTINUOUS -> 8 trials, no OFF period, compared against the pre
                                     file instead
           LF-intermittent and HF-continuous are excluded so that band and recording
           mode are not half-confounded; as it stands band IS mode, which is a property
           of the data rather than a choice.

NAMING    P<n>_<target><freq> with _stim / _pre suffixes. P1 and P5 keep their bare
          names for ANT-145Hz: those two already have runs, figures and cached Delphos
          output under them, and renaming would orphan all of it.

REGENERATE with tools/gen_cohort.py when the trials JSON changes. Do not hand-edit --
the JSON is the authority and this file is a snapshot of it.
"""

COHORT = {
    "P1_ANT2": dict(patient=1, trial_index=2, target="ANT", freq=2,
                intermittent=False, stim="ANT_2Hz", baseline="baseline_1"),
    "P1": dict(patient=1, trial_index=1, target="ANT", freq=145,
                intermittent=True, stim="ANT_145Hz", baseline="baseline"),
    "P1_Pulv2": dict(patient=1, trial_index=3, target="Pulv", freq=2,
                intermittent=False, stim="Pulv_2Hz", baseline="baseline_2"),
    "P1_Pulv145": dict(patient=1, trial_index=4, target="Pulv", freq=145,
                intermittent=True, stim="Pulv_145Hz", baseline="baseline_3"),
    "P3_Pulv7": dict(patient=3, trial_index=3, target="Pulv", freq=7,
                intermittent=False, stim="Pulv_7Hz", baseline="baseline_3"),
    "P3_Pulv145": dict(patient=3, trial_index=2, target="Pulv", freq=145,
                intermittent=True, stim="Pulv_145Hz", baseline="baseline_2"),
    "P4_ANT7": dict(patient=4, trial_index=1, target="ANT", freq=7,
                intermittent=False, stim="ANT_7Hz", baseline="baseline_0"),
    "P4_ANT145": dict(patient=4, trial_index=2, target="ANT", freq=145,
                intermittent=True, stim="ANT_145Hz", baseline="baseline_1"),
    "P5_ANT7": dict(patient=5, trial_index=2, target="ANT", freq=7,
                intermittent=False, stim="ANT_7Hz", baseline="baseline_1"),
    "P5": dict(patient=5, trial_index=1, target="ANT", freq=145,
                intermittent=True, stim="ANT_145Hz", baseline="baseline"),
    "P5_Pulv7": dict(patient=5, trial_index=3, target="Pulv", freq=7,
                intermittent=False, stim="Pulv_7Hz", baseline="baseline_2"),
    "P5_Pulv145": dict(patient=5, trial_index=4, target="Pulv", freq=145,
                intermittent=True, stim="Pulv_145Hz", baseline="baseline_2"),
    "P8_ANT145": dict(patient=8, trial_index=3, target="ANT", freq=145,
                intermittent=True, stim="ANT_145Hz", baseline="baseline_10"),
    "P8_Pulv7": dict(patient=8, trial_index=1, target="Pulv", freq=7,
                intermittent=False, stim="Pulv_7Hz", baseline="baseline"),
    "P10_ANT145": dict(patient=10, trial_index=2, target="ANT", freq=145,
                intermittent=True, stim="ANT_145Hz", baseline="baseline_2"),
    "P11_Pulv7": dict(patient=11, trial_index=1, target="Pulv", freq=7,
                intermittent=False, stim="Pulv_7Hz", baseline="baseline_2"),
    "P11_Pulv145": dict(patient=11, trial_index=2, target="Pulv", freq=145,
                intermittent=True, stim="Pulv_145Hz", baseline="baseline_4"),
}


# The recordings table the detector scripts consume: one _stim and one _pre per trial.
RECORDINGS = {}
for _k, _v in COHORT.items():
    RECORDINGS[f"{_k}_stim"] = dict(patient=_v["patient"], trial_index=_v["trial_index"], file_type="stim")
    RECORDINGS[f"{_k}_pre"] = dict(patient=_v["patient"], trial_index=_v["trial_index"], file_type="pre")


def arm(stem):
    """"HF-int" (ON/OFF contrast available) or "LF-cont" (compare against the pre file)."""
    return "HF-int" if COHORT[stem]["intermittent"] else "LF-cont"
