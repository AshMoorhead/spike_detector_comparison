import json
from pathlib import Path

META = r'S:\MNHS-CCS-NEURO\mn-alf-epilepsy-surg\AshM\Trial-stim\data_meta\trials_json\AES_trials.json'
LF, TARG = {2, 7}, {'ANT', 'Pulv', 'PuM'}
# P1/P5 ANT-145Hz already have runs, figures and in-flight jobs under these names. Renaming
# them would orphan all of that for no gain, so they stay and everything else is systematic.
LEGACY = {(1, 'ANT', 145.0): 'P1', (5, 'ANT', 145.0): 'P5'}

rows = []
for pat in json.load(open(META, encoding='utf-8-sig')):
    pid = pat['patient_id']
    for i, t in enumerate(pat.get('trials', [])):
        f, tg, it = float(t.get('stim_frequency') or 0), t.get('target'), t.get('intermittent', True)
        if tg not in TARG:
            continue
        if not ((f == 145 and it) or (f in LF and not it)):
            continue
        stem = LEGACY.get((pid, tg, f), f'P{pid}_{tg}{f:g}')
        rows.append(dict(stem=stem, pid=pid, idx=i + 1, target=tg, freq=f,
                         intermittent=it, stim=t['filename'], base=t.get('baseline')))

rows.sort(key=lambda r: (r['pid'], r['target'], r['freq']))
out = ['"""', 'sdc.detect.cohort',
       '------------------',
       'The AES cohort: 17 trials from 7 patients, generated from AES_trials.json.',
       '',
       'SELECTION  target in {ANT, Pulv} and either',
       '             145 Hz INTERMITTENT  -> 9 trials, ON/OFF contrast available',
       '             2 or 7 Hz CONTINUOUS -> 8 trials, no OFF period, compared against the pre',
       '                                     file instead',
       '           LF-intermittent and HF-continuous are excluded so that band and recording',
       '           mode are not half-confounded; as it stands band IS mode, which is a property',
       '           of the data rather than a choice.',
       '',
       'NAMING    P<n>_<target><freq> with _stim / _pre suffixes. P1 and P5 keep their bare',
       '          names for ANT-145Hz: those two already have runs, figures and cached Delphos',
       '          output under them, and renaming would orphan all of it.',
       '',
       'REGENERATE with tools/gen_cohort.py when the trials JSON changes. Do not hand-edit --',
       'the JSON is the authority and this file is a snapshot of it.',
       '"""', '',
       'COHORT = {']
for r in rows:
    out.append(f'    "{r["stem"]}": dict(patient={r["pid"]}, trial_index={r["idx"]}, '
               f'target="{r["target"]}", freq={r["freq"]:g},')
    out.append(f'        {"":<8}intermittent={r["intermittent"]}, '
               f'stim="{r["stim"]}", baseline="{r["base"]}"),')
out += ['}', '', '',
        '# The recordings table the detector scripts consume: one _stim and one _pre per trial.',
        'RECORDINGS = {}',
        'for _k, _v in COHORT.items():',
        '    RECORDINGS[f"{_k}_stim"] = dict(patient=_v["patient"], '
        'trial_index=_v["trial_index"], file_type="stim")',
        '    RECORDINGS[f"{_k}_pre"] = dict(patient=_v["patient"], '
        'trial_index=_v["trial_index"], file_type="pre")',
        '', '',
        'def arm(stem):',
        '    """"HF-int" (ON/OFF contrast available) or "LF-cont" (compare against the pre file)."""',
        '    return "HF-int" if COHORT[stem]["intermittent"] else "LF-cont"',
        '']
Path('sdc/detect/cohort.py').write_text('\n'.join(out), encoding='utf-8')
print(f'wrote sdc/detect/cohort.py with {len(rows)} trials')
