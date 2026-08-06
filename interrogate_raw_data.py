"""
interrogate_raw_data.py
-----------------------
Look at the actual signal behind a set of detections, with every detector's marks overlaid.

EDIT THE CONFIG BLOCK BELOW AND RUN IT. That is the whole interface.

    .venv\\Scripts\\python.exe interrogate_raw_data.py

Everything else in this repo turns detections into numbers. This turns them back into a trace,
because a rate of 0.8/s on one detector and 0.02/s on another is a question, not an answer --
and the answer is nearly always visible in ten seconds of signal.

    <- ->  scroll (shift: a whole page)      up/down  gain       [ ]  timebase
    n / p  next / previous channel page      click    hide a channel
    r      reset                             s        save a PNG

Needs an interactive matplotlib backend. From a plain terminal that is automatic. In the VS
Code interactive window run `%matplotlib tk` FIRST, or the figure is a dead PNG and the keys
do nothing.

WHICH SIGNAL YOU ARE LOOKING AT
  Default is the file the run npz points to, which for a windowed run is the PREPROCESSED EDF
  -- post-montage, median-filtered, decimated. That is deliberately the default because it is
  what the detectors actually saw, so what is on screen is what they responded to.
  Set RAW = True to load the original recording at its native rate instead and montage it here.
  Use that to ask "was this in the data, or did the preprocessing make it?".
"""
# %% ------------------------------------------------------------------ CONFIG: edit this
NPZ = "runs/P1_stim.npz"       # any file in runs/
CHANNELS = None # ["L6_L7", "L5_L6", "L1_L2", "B1_B2", "B9_B10", "H14_H15", "T11_T12", "O6_O7"]
                               # channel names, or None for the busiest DISAGREEMENT channels
T0 = 900.0                     # seconds from the start of the recording
DURATION = 300.0                # seconds on screen
GAIN = 1.0                     # vertical zoom; also adjustable live with up/down
RAW = False                    # False: what the detectors saw. True: the original EDF.
N_AUTO = 50                    # how many channels CHANNELS=None picks

# %% ------------------------------------------------------------------
from pathlib import Path

import numpy as np

from seeg import (read_edf_header, load_edf_segment, derive_montage, apply_montage, view,
                  load_trials, get_patient, get_trial, resolve_file)

from sdc.common import cond
from sdc.common.paths import ROOT

BASE_DIR = Path(r"C:\Users\amoo0039\Documents\local")
META_PATH = BASE_DIR / "data_meta" / "stim_trials.json"
RECORDINGS = {"P1_pre": (1, "pre"), "P1_stim": (1, "stim"),
              "P5_pre": (5, "pre"), "P5_stim": (5, "stim")}


def pick_disagreement(z, names, n):
    """The n channels where the detectors disagree most, by max/min of their rates.

    A default that puts something worth looking at on screen: agreement is not interesting and
    a plain 'busiest' list just shows the same few epileptic channels every time."""
    sel = cond.select(z, "all")
    r = []
    for d in [str(s) for s in z["detectors"]]:
        cnt = np.bincount(z[f"{d}_chan"][sel.keep(d)], minlength=len(names))
        r.append(sel.rate(cnt))
    r = np.array(r)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.nanmax(r, axis=0) / np.maximum(np.nanmin(r, axis=0), 1e-9)
    spread[~np.isfinite(r).all(axis=0)] = -1        # unmeasurable channels are not candidates
    spread[np.nanmax(r, axis=0) < 0.05] = -1        # nor near-silent ones: a 0/0.001 ratio is
                                                    # arithmetic, not disagreement
    return [names[i] for i in np.argsort(-spread)[:n]]


z = np.load(ROOT / NPZ, allow_pickle=False)
names = [str(s) for s in z["names"]]
fs_det = float(z["fs"])
dets = [str(s) for s in z["detectors"]]
chans = CHANNELS or pick_disagreement(z, names, N_AUTO)
missing = [c for c in chans if c not in names]
if missing:
    raise SystemExit(f"not in this recording: {missing}\nfirst few available: {names[:8]}")
idx = [names.index(c) for c in chans]

# ---- the signal ------------------------------------------------------------------------
if RAW:
    pid, ftype = RECORDINGS[str(z["rec_id"])]
    stem, _ = resolve_file(get_trial(get_patient(load_trials(META_PATH), pid), 1), ftype)
    edf = str(BASE_DIR / f"P{pid}" / f"{stem}.edf")
else:
    edf = str(z["edf"])
hdr = read_edf_header(edf)
r0 = max(int(T0), 0) + 1                                   # records are 1 s, 1-based inclusive
r1 = min(int(np.ceil(T0 + DURATION)) + 1, int(hdr["NumDataRecords"]))
rec = load_edf_segment(edf, hdr, r0, r1)
if RAW:
    rec = apply_montage(rec, derive_montage(rec["info"]["SelectedSignals"]))
print(f"{Path(edf).name}: {rec['data'].shape[0]} samples x {rec['data'].shape[1]} ch "
      f"at {rec['info']['SampleRate']:g} Hz   ({'RAW' if RAW else 'what the detectors saw'})")

# Keep only the requested channels, in the requested order.
all_names = list(rec["info"]["SelectedSignals"])
sub = [all_names.index(c) for c in chans if c in all_names]
if len(sub) != len(chans):
    raise SystemExit(f"missing from {Path(edf).name}: "
                     f"{[c for c in chans if c not in all_names]}")
rec = {"data": rec["data"][:, sub],
       "info": {**rec["info"], "SelectedSignals": chans, "NumSelectedSignals": len(chans)}}

# ---- detections, in the same channel order and on the VIEWER's clock ---------------------
# The viewer's x axis starts at 0 for the loaded segment, so absolute detection times need the
# segment start subtracted. Getting this wrong shifts every mark by T0 and makes a detector
# look like it is firing on nothing.
t_off = r0 - 1
spikes = {}
for d in dets:
    per = []
    for c in idx:
        t = z[f"{d}_idx"][z[f"{d}_chan"] == c] / fs_det - t_off
        t = t[(t >= 0) & (t <= rec["data"].shape[0] / rec["info"]["SampleRate"])]
        per.append(np.round(t * rec["info"]["SampleRate"]).astype(int))
    spikes[d] = per
    print(f"  {d:<10} {sum(len(p) for p in per):4d} marks on these channels in this window")

view(rec, spikes=spikes, chans_per_page=min(len(chans), 25),
     t0=0.0, duration=DURATION, gain=GAIN)
