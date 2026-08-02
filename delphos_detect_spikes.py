"""
delphos_detect_spikes.py
------------------------
Python port of seeg_analysis/src/extraction/{run_delphos,markersToInfo}.m -- run the
Delphos CLI spike detector (Roehri et al. 2016/2017/2018) and return spikes aligned to
the pipeline's bipolar channel order.

Delphos is a compiled MATLAB app driven over the command line. It reads the RAW file
(any Fieldtrip-readable format), builds its OWN bipolar montage (`bipolar True`) and
writes `<basename>_delphos.mat` into `output_dir`. We parse `results.markers` (keeping
only "Spike" markers), whose `.position` is in seconds of ABSOLUTE file time, so chunked
runs merge by plain concatenation -- no per-chunk offset.

LABEL SPACE. Delphos emits `C'1-C'2`; the pipeline uses `C_1_C_2` (montage.py coerces the
apostrophe, then joins the pair with `_`). Reconcile by mapping BOTH `'` and `-` to `_` on
both sides -- exactly what run_delphos.m does.

RAM / REPRODUCIBILITY. Delphos tiles each call internally via parpool, and picks the tile
count from *available system RAM*; the tiling shifts detections near tile boundaries. So a
run is only comparable to another at the SAME operating point. We reproduce run_delphos.m's
"balloon": allocate and touch memory to pull free RAM down to `pin_free_ram_gb` for the
duration of the (blocking) call. That lands within ~100 MB, not exactly -- good enough for
rate-level comparison, not bit-exact. See spike_detector_comparison/Delphos.md for the full
characterisation and the agreed operating point (pin 12 GB, chunk cap 30 min).

NOT PORTED: per-spike features. `results.detection_charac` is a MATLAB *table*, which
scipy.io.loadmat cannot decode (it comes back as an opaque object). run_delphos.m's
`SpikeFeatures` therefore has no counterpart here; spike times only.

REQUIREMENT. `Delphos_cmd_line.exe` needs **MATLAB Runtime 9.13 (R2022b)** -- a separate
install from MATLAB itself. Note `delphos_command_line/readme.txt` says 9.5 (R2018b) and is
WRONG (stale vendor readme): installing 9.5 leaves the exe still dead, and running it
interactively names 9.13. Get it from mathworks.com/products/compiler/matlab-runtime.html
(pick R2022b, Windows 64-bit). Without the right runtime the exe exits -1 printing nothing
even with no arguments -- that case is called out explicitly in the error raised below.
Every runtime found under `mcr_root` is prepended to the CHILD's PATH (`_runtime_env`), so a
shell that predates the install still works without being restarted.

Deps: numpy, scipy (both already required by the comparison script). Windows only (the
RAM pin uses GlobalMemoryStatusEx; without it the run still works, just unpinned).
"""
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# Defaults mirror run_delphos.m's apply_defaults() -- our agreed parameters (Delphos.md).
DEFAULTS = dict(
    delphos_exe=r"C:\Users\amoo0039\Documents\seeg_analysis\delphos_command_line\Delphos_cmd_line.exe",
    detection_type="Spk",     # spikes only; markers are filtered to "Spike" regardless
    freq_band_start=8,
    freq_band_end=512,
    Osc_time_thr=1.4,
    Spk_thr=40,               # lower than the CLI default 80 -> more sensitive
    Spk_time_thr=1.3,
    bipolar=True,             # Delphos applies its own bipolar montage
    chunk_sec=None,           # None -> one call for the whole span
    pin_free_ram_gb=12,       # 0/None -> no pin. Machine-specific; see Delphos.md
    keep_output=False,        # True -> leave the temp output dir on disk
    mcr_root=r"C:\Program Files\MATLAB\MATLAB Runtime",   # searched for runtime\win64; see _runtime_env
)

_CHAN_KEY = "__delphos_channels__"   # reserved npz key: the full label set (see _to_indices)


# ----------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------
def normalise_label(label):
    """Delphos `C'1-C'2` -> pipeline `C_1_C_2`: map both `'` (and the curly variant)
    and `-` to `_`. Applied to BOTH label spaces before matching."""
    s = str(label).strip()
    for ch in ("'", "\u2019", "-"):
        s = s.replace(ch, "_")
    return s


# ----------------------------------------------------------------------
# .mat parsing (port of markersToInfo.m, spike times only)
# ----------------------------------------------------------------------
def _as_list(x):
    """squeeze_me collapses 1-element cells/structs to scalars; re-expand to a flat list."""
    if x is None:
        return []
    arr = np.atleast_1d(np.asarray(x, dtype=object).ravel())
    return [v for v in arr]


def _marker_channels(chs, labels):
    """Marker `.channels` -> label strings. Delphos writes a cell of label strings, but
    markersToInfo.m also tolerates indices/masks, so keep that tolerance."""
    out = []
    for c in _as_list(chs):
        if isinstance(c, (bytes, str, np.str_)):
            out.append(str(c))
        elif np.isscalar(c) and not isinstance(c, np.bool_):
            i = int(c) - 1                       # MATLAB 1-based
            if 0 <= i < len(labels):
                out.append(labels[i])
    return out


def read_delphos_mat(mat_path):
    """Read `<base>_delphos.mat` -> (spikes, channels).

    spikes   : {normalised label: sorted spike times (s, absolute file time)}
    channels : every normalised label Delphos derived (`results.labels`) -- a SUPERSET of
               the montage, and the thing to check label matching against, since a channel
               can legitimately be present with zero spikes.

    Only `label == "Spike"` markers are kept, which is what makes a
    `detection_type {'Spk','Osc'}` run safe to request (markersToInfo.m:60)."""
    S = loadmat(mat_path, squeeze_me=True, struct_as_record=False, mat_dtype=False)
    if "results" not in S:
        raise ValueError(f"{mat_path}: no `results` variable (keys: {sorted(S)}).")
    results = S["results"]

    labels = [str(v) for v in _as_list(getattr(results, "labels", None))]
    spikes = {}
    for m in _as_list(getattr(results, "markers", None)):
        if str(getattr(m, "label", "")) != "Spike":
            continue
        pos = getattr(m, "position", None)
        if pos is None or not np.isfinite(pos):
            continue
        for lab in _marker_channels(getattr(m, "channels", None), labels):
            spikes.setdefault(normalise_label(lab), []).append(float(pos))

    return ({k: np.unique(np.asarray(v, float)) for k, v in spikes.items()},
            [normalise_label(v) for v in labels])


# ----------------------------------------------------------------------
# Running the CLI
# ----------------------------------------------------------------------
class _RamBalloon:
    """Hold system free RAM near `target_gb` for the duration of a `with` block.

    Port of run_delphos.m's balloon. Delphos reads free RAM at ITS OWN launch -- after ours
    -- so this fixes our allocation, never the other processes' jitter (~100 MB). That is
    the documented limit of the approach, not a bug here."""

    def __init__(self, target_gb, log):
        self.target_gb = target_gb
        self.log = log
        self.buf = None

    @staticmethod
    def available_bytes():
        """Windows GlobalMemoryStatusEx -> ullAvailPhys. None off Windows."""
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return None
            return int(st.ullAvailPhys)
        except (AttributeError, OSError):
            return None

    def __enter__(self):
        if not self.target_gb:
            return self
        target = int(self.target_gb * 2 ** 30)
        avail = self.available_bytes()
        if avail is None:
            self.log("RAM pin: cannot read system memory; running without pin.")
            return self
        if avail <= target:
            self.log(f"RAM pin: free RAM ({avail / 2**30:.1f} GB) already <= target "
                     f"({self.target_gb:.1f} GB); cannot pin up -- running as-is.")
            return self
        try:
            self.buf = np.empty(avail - target, dtype=np.uint8)
            self.buf[::4096] = 1                 # touch every page so it is committed
        except MemoryError:
            self.buf = None
            self.log("RAM pin: allocation failed; running without pin.")
            return self
        self.log(f"RAM pin: Available {avail / 2**30:.1f} -> "
                 f"{(self.available_bytes() or 0) / 2**30:.1f} GB "
                 f"(target {self.target_gb:.1f}).")
        return self

    def __exit__(self, *exc):
        self.buf = None
        return False


def _runtime_env(cfg, log):
    """Child env with every installed MATLAB Runtime's runtime\\win64 prepended to PATH.

    The installer adds that directory to the MACHINE PATH, but already-running shells (and
    the editor that spawned them) keep their stale copy -- so the exe dies with the
    missing-runtime signature until everything is restarted. Putting the directory on the
    CHILD's PATH removes that trap entirely. Prepending several versions is safe: the exe
    loads its runtime by versioned filename (mclmcrrt9_13.dll), so only the matching one
    can answer."""
    env = os.environ.copy()
    root = Path(cfg["mcr_root"]) if cfg.get("mcr_root") else None
    if root is None or not root.is_dir():
        return env
    dirs = [str(d) for d in sorted(root.glob("*/runtime/win64")) if d.is_dir()]
    if not dirs:
        log(f"[delphos] no MATLAB Runtime found under {root}; relying on PATH as-is.")
        return env
    env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
    return env


def _run_one_chunk(edf_path, start_sec, dur_sec, cfg, log):
    """Run Delphos on one [start, start+dur] window; return (spikes, channels)."""
    out_dir = Path(tempfile.mkdtemp(prefix=f"delphos_{int(round(start_sec))}_"))
    try:
        cmd = [str(cfg["delphos_exe"]),
               "input_file", str(edf_path),
               "detection_type", str(cfg["detection_type"]),
               "freq_band_start", f"{cfg['freq_band_start']:g}",
               "freq_band_end", f"{cfg['freq_band_end']:g}",
               "Osc_time_thr", f"{cfg['Osc_time_thr']:g}",
               "Spk_thr", f"{cfg['Spk_thr']:g}",
               "Spk_time_thr", f"{cfg['Spk_time_thr']:g}",
               "start", f"{start_sec:g}",
               "duration", f"{dur_sec:g}",
               "bipolar", "True" if cfg["bipolar"] else "False",
               "output_dir", str(out_dir)]

        env = _runtime_env(cfg, log)
        with _RamBalloon(cfg["pin_free_ram_gb"], log):
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            hint = ""
            if not (proc.stdout or proc.stderr):
                # With its runtime missing the exe dies before printing anything -- even with
                # no args. It needs 9.13 (R2022b); the bundled readme.txt saying 9.5/R2018b is
                # stale, and 9.5 alone does NOT satisfy it.
                found = [d.name for d in sorted(Path(cfg["mcr_root"]).glob("*"))] \
                    if cfg.get("mcr_root") and Path(cfg["mcr_root"]).is_dir() else []
                hint = ("\nNo output at all: the exe cannot find its MATLAB Runtime. It needs "
                        "9.13 (R2022b) -- a separate install from MATLAB itself, and NOT the "
                        "9.5/R2018b that delphos_command_line/readme.txt claims.\n"
                        f"Runtimes found under {cfg.get('mcr_root')}: {found or 'none'} "
                        "(all of these were put on the child's PATH already, so this is a "
                        "missing INSTALL, not a stale shell). Double-click\n"
                        f'    "{cfg["delphos_exe"]}"   (bare, no arguments)\n'
                        "and it will name the version it wants.")
            raise RuntimeError(f"Delphos returned status {proc.returncode} for "
                               f"start={start_sec:g}.{hint}\n{proc.stdout}\n{proc.stderr}")

        mats = sorted(out_dir.glob("*_delphos.mat"))
        if not mats:
            # status 0 but no output is the OOM signature: Delphos exits without writing.
            raise RuntimeError(
                f"No *_delphos.mat in {out_dir} (Delphos exited 0 without writing -- "
                f"likely OOM; raise or disable pin_free_ram_gb).\n"
                f"--- Delphos stdout ---\n{proc.stdout}")
        return read_delphos_mat(mats[0])
    finally:
        if cfg["keep_output"]:
            log(f"Delphos output kept: {out_dir}")
        else:
            shutil.rmtree(out_dir, ignore_errors=True)


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def detect_spikes(edf_path, labels, fs, start_sec=0.0, duration_sec=-1.0,
                  file_dur_sec=None, cache_dir=None, log=print, **cfg):
    """Run Delphos over [start_sec, start_sec+duration_sec] of `edf_path`.

    labels        : pipeline bipolar labels ("C_1_C_2" style) -- the output channel order.
    fs            : rate the returned SAMPLE INDICES are expressed at. Delphos marker
                    positions are in seconds, so this need not be the EDF's rate: pass the
                    common comparison rate and indices land on that axis directly.
    start_sec /   : window in the file. duration_sec < 0 -> to end of file (needs
    duration_sec    `file_dur_sec`, or it is read from the EDF header).
    cache_dir     : if set, memoise the run there keyed by file+window+parameters. A 60 s
                    call costs ~5 min (parpool startup dominates), so re-running the
                    comparison script without a cache is painful.
    **cfg         : any DEFAULTS key (delphos_exe, Spk_thr, chunk_sec, pin_free_ram_gb, ...)

    Returns a list of `len(labels)` int arrays of 0-BASED sample indices at `fs`, relative
    to `start_sec` -- ready to sit alongside the other detectors' output. Pipeline labels
    with no Delphos counterpart get an empty array.
    """
    # Reject unknown knobs rather than absorbing them: a typo'd sweep parameter would
    # otherwise run the full grid at identical settings and read as "this knob does nothing".
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown Delphos setting(s) {sorted(unknown)}; "
                        f"valid: {sorted(DEFAULTS)}")
    cfg = {**DEFAULTS, **cfg}
    edf_path = Path(edf_path)
    if not Path(cfg["delphos_exe"]).is_file():
        raise FileNotFoundError(f"Delphos exe not found: {cfg['delphos_exe']}")
    if not edf_path.is_file():
        raise FileNotFoundError(f"Input file not found: {edf_path}")

    # --- window + chunk boundaries (run_delphos.m:38-66) ---
    if duration_sec is not None and duration_sec >= 0:
        span = float(duration_sec)
    else:
        if file_dur_sec is None:
            from seeg import read_edf_header
            hdr = read_edf_header(edf_path)
            file_dur_sec = hdr["NumDataRecords"] * hdr["DataRecordDuration"]
        span = float(file_dur_sec) - float(start_sec)
    if span <= 0:
        raise ValueError("Nothing to process (start_sec beyond file end).")

    chunk = cfg["chunk_sec"]
    if not chunk or chunk <= 0:
        windows = [(float(start_sec), span)]
    else:
        n = max(1, int(np.ceil(span / chunk)))
        windows = [(start_sec + i * chunk, min(chunk, start_sec + span - (start_sec + i * chunk)))
                   for i in range(n)]
        windows = [w for w in windows if w[1] > 1e-6]

    # --- cache key: everything that changes the detections ---
    cache_path = None
    if cache_dir is not None:
        # Only settings that can change the DETECTIONS belong in the key. `keep_output` and
        # `mcr_root` are plumbing -- including them would silently invalidate every cached
        # run (5 min each) the next time an unrelated option is added to DEFAULTS.
        key = json.dumps({"file": str(edf_path.resolve()),
                          "size": edf_path.stat().st_size,
                          "windows": windows,
                          "cfg": {k: v for k, v in sorted(cfg.items())
                                  if k not in ("keep_output", "mcr_root")}},
                         default=str, sort_keys=True)
        cache_path = Path(cache_dir) / f"delphos_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"
        if cache_path.is_file():
            z = np.load(cache_path, allow_pickle=False)
            chans = [str(c) for c in z[_CHAN_KEY]]
            times = {str(k): z[k] for k in z.files if k != _CHAN_KEY}
            log(f"[delphos] cache hit: {cache_path.name}")
            return _to_indices(times, chans, labels, fs, start_sec, log)

    # --- run, merging chunks (positions are absolute -> plain concatenation) ---
    times, chans = {}, []
    for i, (s, d) in enumerate(windows, 1):
        log(f"[delphos] chunk {i}/{len(windows)}: start {s:.0f} s, dur {d:.0f} s")
        chunk_times, chunk_chans = _run_one_chunk(edf_path, s, d, cfg, log)
        for lab, t in chunk_times.items():
            times[lab] = np.concatenate([times[lab], t]) if lab in times else t
        chans += [c for c in chunk_chans if c not in chans]
    times = {k: np.unique(v) for k, v in times.items()}

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **{_CHAN_KEY: np.array(chans, dtype="U")}, **times)
        log(f"[delphos] cached -> {cache_path.name}")

    return _to_indices(times, chans, labels, fs, start_sec, log)


def _to_indices(times, chans, labels, fs, start_sec, log):
    """{normalised Delphos label: absolute times (s)} -> per-pipeline-channel sample indices.
    Delphos-only pairs (it derives a superset of the montage) are dropped; pipeline labels
    Delphos never derived get an empty array and are counted as unmatched."""
    known, out, matched = set(chans), [], 0
    for lab in labels:
        key = normalise_label(lab)
        matched += key in known
        t = times.get(key)
        out.append(np.zeros(0, int) if t is None
                   else np.unique(np.round((t - start_sec) * fs).astype(int)))
    frac = matched / max(len(labels), 1)
    msg = (f"[delphos] matched {matched}/{len(labels)} pipeline labels ({frac:.0%}) "
           f"against {len(known)} Delphos pairs")
    log(msg if frac >= 0.8 else msg + "  <-- LOW: check label normalisation")
    return out
