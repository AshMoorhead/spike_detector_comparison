"""
janca_detect_spikes.py
----------------------
Pure-Python reimplementation of the Janca et al. Hilbert-envelope spike detector,
ported from `spike_detector_hilbert_v24.m` (the canonical ISARG version).

Reference:
  Janca R. et al. "Detection of Interictal Epileptiform Discharges Using Signal
  Envelope Distribution Modelling: Application to Epileptic and Non-Epileptic
  Intracranial Recordings." Brain Topography 28.1 (2015): 172-183.
  DOI: 10.1007/s10548-014-0379-1

WHY v24 (not v25): the two .m files carry the same internal function name and the
same algorithm. v25 is an older copy with local "John" edits that (a) revert the
ambiguous-spike acceptance test to a single-sample `ovious_M(idx-0.01*fs)` lookup
(with the `ovious_M` typo), and (b) guard the cross-buffer discharge concatenation
with `if ~isempty(discharges.MV)`, which -- because discharges.MV starts as [] --
means discharges are NEVER accumulated across buffers. v24 has the corrected
symmetric +/- discharge_tol window and always accumulates. v24 is the one to trust.

PARITY NOTE (same caveat as seeg/spikes.py): this is NOT bit-exact with MATLAB.
`scipy.signal.resample_poly` and MATLAB's `resample(...,100)` use different FIR
designs, and filtfilt edge handling differs slightly. Spike *counts and positions*
match to within a sample or two on clean data; exact equality is not achievable and
not the goal. For a decimation-free comparison pass `dec=0`.

DEFAULT PATH VERIFIED: the default settings (dec=200, beta=Inf i.e. beta detector
off, k1==k2 i.e. no ambiguous tier, cheby2 filtering) are the path exercised by the
smoke test at the bottom (`python -m sdc.detect.janca_detect_spikes`). The beta detector and the
ambiguous/k2 tier are ported for completeness but are off by default.

INDEXING: `out['pos']` and `discharges['MP']` are in SECONDS of real time (relative
to the start of the input `d`). `out['chan']` is 0-BASED (MATLAB returns 1-based;
converted here to match the rest of the Python pipeline, e.g. seeg/spikes.py).

Deps: numpy, scipy.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import (
    butter, filtfilt, hilbert, resample_poly, cheb2ord, cheby2, freqz,
)
from scipy.interpolate import CubicSpline


# ----------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------
_DEFAULTS = dict(
    band_low=10.0,      # -fl
    band_high=60.0,     # -fh
    k1=3.65,            # -k1  obvious-spike threshold
    k2=None,            # -k2  ambiguous threshold (None -> equals k1, tier disabled)
    k3=0.0,             # -k3
    buffering=300.0,    # -buf  seconds
    main_hum_freq=50.0,  # -h
    beta=np.inf,        # -b    Inf -> beta/mu detector off
    beta_win=20.0,      # -bw
    beta_ar=12,         # -br
    f_type=1,           # -ft   1 cheby2, 2 butter, 3 fir
    discharge_tol=0.005,  # -dt  seconds
    polyspike_union_time=0.12,  # -pt seconds
    decimation=200.0,   # -dec  Hz (0 -> disable, keep original fs)
    ti_switch=1,        # -ti   1 envelope max, 2 abs-amplitude max
    winsize=None,       # -w    samples (None -> 5*fs)
    noverlap=None,      # -n    samples or fraction (None -> 4*fs)
)

# maps the MATLAB '-xx' settings-string flags onto the kwarg names above.
_FLAG_MAP = {
    "fl": "band_low", "fh": "band_high", "k1": "k1", "k2": "k2", "k3": "k3",
    "w": "winsize", "n": "noverlap", "buf": "buffering", "h": "main_hum_freq",
    "b": "beta", "bw": "beta_win", "br": "beta_ar", "ft": "f_type",
    "dt": "discharge_tol", "pt": "polyspike_union_time", "dec": "decimation",
    "ti": "ti_switch",
}


def _parse_settings(settings, overrides):
    """Merge defaults, a MATLAB-style settings string ('-k1 3.0 -fh 80'), and kwargs.

    Override kwargs accept either the full name (`decimation`, `band_high`) or the
    MATLAB short flag (`dec`, `fh`). Unknown keys raise, so a typo can't silently
    no-op the way a bare `dec=0` would if it were dropped.
    """
    cfg = dict(_DEFAULTS)
    if settings:
        toks = settings.replace("  ", " ").split()
        i = 0
        while i < len(toks) - 1:
            if toks[i].startswith("-"):
                flag = toks[i][1:]
                if flag in _FLAG_MAP:
                    cfg[_FLAG_MAP[flag]] = float(toks[i + 1])
                i += 2
            else:
                i += 1
    for key, val in overrides.items():
        if val is None:
            continue
        name = key if key in _DEFAULTS else _FLAG_MAP.get(key)
        if name is None:
            raise TypeError(f"unknown setting {key!r}; expected one of "
                            f"{sorted(_DEFAULTS)} or a MATLAB flag {sorted(_FLAG_MAP)}")
        cfg[name] = val
    return cfg


# ----------------------------------------------------------------------
# filters (ported subfunctions)
# ----------------------------------------------------------------------
def _filt_50hz(d, fs, hum_fs, band_high):
    """Notch out mains hum and its harmonics up to 1.1*band_high (filt50Hz.m)."""
    f0 = np.arange(hum_fs, fs / 2.0, hum_fs)
    f0 = f0[f0 <= 1.1 * band_high]
    R, r = 1.0, 0.985
    for f in f0:
        b = [1.0, -2 * R * np.cos(2 * np.pi * f / fs), R * R]
        a = [1.0, -2 * r * np.cos(2 * np.pi * f / fs), r * r]
        d = filtfilt(b, a, d, axis=0)
    return d


def _bandpass(d, fs, band_low, band_high, f_type, decimation):
    """Bandpass filtering (filtering.m). Default cheby2; falls back to butter if
    the signal was not decimated to 200 Hz (matches the MATLAB guard)."""
    if decimation != 200 and f_type == 1:
        f_type = 2  # cheby2 design is only validated at fs=200 in the original

    nyq = fs / 2.0
    if f_type == 1:  # cheby2
        Wp, Ws = band_high / nyq, band_high / nyq + 0.1
        n, Ws = cheb2ord(Wp, Ws, 6, 60)
        bl, al = cheby2(n, 60, Ws)
        Wp, Ws = band_low / nyq, band_low / nyq - 0.05
        n, Ws = cheb2ord(Wp, Ws, 6, 60)
        bh, ah = cheby2(n, 60, Ws, "high")
    elif f_type == 2:  # butterworth
        Wp, Ws = band_high / nyq, min(band_high / nyq + 0.1, 1.0)
        n, Ws = butter_ord(Wp, Ws)
        bl, al = butter(n, Ws)
        Wp, Ws = band_low / nyq, band_low / nyq - 0.05
        Ws = Ws if Ws > 0 else 0.1
        n, Ws = butter_ord(Wp, Ws)
        bh, ah = butter(n, Ws, "high")
    else:  # FIR
        from scipy.signal import firwin
        bl, al = firwin(int(fs / 2) + 1, band_high / nyq), 1.0
        bh, ah = firwin(int(fs / 2) + 1, band_low / nyq, pass_zero=False), 1.0

    # stability sanity check, as in the MATLAB
    for b, a in ((bl, al), (bh, ah)):
        _, h = freqz(b, a, worN=10 * int(fs))
        if np.max(np.abs(h)) > 1.001:
            raise RuntimeError("bandpass filter is probably unstable")

    d = filtfilt(bh, ah, d, axis=0)
    if band_high == nyq:
        return d
    d = filtfilt(bl, al, d, axis=0)
    return d


def butter_ord(Wp, Ws, Rp=6, Rs=60):
    from scipy.signal import buttord
    return buttord(Wp, Ws, Rp, Rs)


# ----------------------------------------------------------------------
# decimation (staged, like the MATLAB resample loop)
# ----------------------------------------------------------------------
def _decimate(d, fs, decimation):
    """Staged resample from fs to `decimation` Hz (spike_detector.m downsampling).

    Returns (d_dec, fs_dec, rfactor). rfactor = original_fs / decimation.
    """
    rfactor = fs / decimation
    if not (rfactor > 1 or decimation != fs):
        return d, fs, rfactor

    n_it = int(np.ceil(np.log10(rfactor)))
    n_it = max(n_it, 1)
    for i in range(n_it):
        fs_out = decimation if i == n_it - 1 else round(fs / rfactor ** (1.0 / n_it))
        up, down = int(round(fs_out)), int(round(fs))
        g = np.gcd(up, down)
        d = resample_poly(d, up // g, down // g, axis=0)
        fs = fs_out
    return d, float(fs), rfactor


# ----------------------------------------------------------------------
# beta / mu activity detector (beta_detect.m) -- OFF by default (beta=Inf)
# ----------------------------------------------------------------------
def _lpc(x, order):
    """LPC coefficients via autocorrelation + Levinson-Durbin (MATLAB lpc)."""
    x = np.asarray(x, float)
    r = np.correlate(x, x, "full")[len(x) - 1:len(x) + order]
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    if e == 0:
        return a
    for i in range(1, order + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / e
        a[1:i + 1] += k * np.concatenate(([0.0], a[1:i][::-1] if i > 1 else []))[:i]
        a[i] = k
        e *= (1 - k * k)
        if e <= 0:
            break
    return a


def _beta_detect(d, fs, beta, winsize_s, beta_ar):
    """Flag mu/beta-rhythm segments (autoregressive spectral peak in (beta, 25) Hz)."""
    n, nch = d.shape
    winsize = int(round(winsize_s * fs))
    noverlap = int(round(0.5 * winsize))
    step = winsize - noverlap
    if step < 1 or winsize >= n:
        index = np.array([0])
        winsize = n
    else:
        index = np.arange(0, n - winsize + 1, step)
    bb, aa = butter(4, 2 * 30.0 / fs)

    M = np.zeros((n, nch), bool)
    for ch in range(nch):
        flags = np.zeros(len(index), bool)
        for i, st in enumerate(index):
            seg = filtfilt(bb, aa, d[st:st + winsize, ch])
            a = _lpc(seg - seg.mean(), beta_ar)
            w, h = freqz(1.0, a, worN=512, fs=fs)
            h = np.abs(h)
            # local maxima frequencies
            dsign = np.sign(np.diff(np.concatenate(([0.0], h))))
            peaks = w[np.diff(np.concatenate(([0.0], dsign))) < 0]
            flags[i] = np.any((peaks < 25) & (peaks > beta))
        xp = np.concatenate([index, [n - 1]]).astype(float)
        fp = np.concatenate([flags, [flags[-1]]]).astype(float)
        M[:, ch] = np.interp(np.arange(n), xp, fp) >= 0.5
    return M


# ----------------------------------------------------------------------
# core per-channel detection
# ----------------------------------------------------------------------
def _local_maxima_detection(envelope, prah, fs, pt, ti_switch, d_decim):
    """local_maxima_detection.m: reduce threshold-crossing runs to spike markers."""
    n = envelope.shape[0]
    cross = (envelope > prah).astype(int)
    starts = np.flatnonzero(np.diff(np.concatenate(([0], cross))) > 0)
    stops = np.flatnonzero(np.diff(np.concatenate((cross, [0]))) < 0)

    peak_signal = np.abs(d_decim) if ti_switch == 2 else envelope

    marker = np.zeros(n, bool)
    for p1, p2 in zip(starts, stops):  # inclusive [p1, p2]
        if p2 - p1 > 2:
            seg = peak_signal[p1:p2 + 1]
            ssign = np.sign(np.diff(seg))
            locmax = np.flatnonzero(np.diff(np.concatenate(([0], ssign))) < 0)
            marker[p1 + locmax] = True
        else:
            seg = peak_signal[p1:p2 + 1]
            marker[p1 + int(np.argmax(seg))] = True

    # union of local maxima that are close together (< pt seconds apart)
    pointer = np.flatnonzero(marker)
    span = int(np.ceil(pt * fs))
    state_prev = False
    start = 0
    for pk in pointer:
        hi = min(pk + 1 + span, n)
        seg = marker[pk + 1:hi]
        if state_prev:
            if seg.sum() > 0:
                state_prev = True
            else:
                state_prev = False
                marker[start:pk + 1] = True
        else:
            if seg.sum() > 0:
                state_prev = True
                start = pk

    # within each merged run keep only local maxima that have a gradient both sides
    starts = np.flatnonzero(np.diff(np.concatenate(([0], marker.astype(int)))) > 0)
    stops = np.flatnonzero(np.diff(np.concatenate((marker.astype(int), [0]))) < 0)
    for p1, p2 in zip(starts, stops):
        if p2 - p1 > 1:
            lmax = pointer[(pointer >= p1) & (pointer <= p2)]
            marker[p1:p2 + 1] = False
            vals = envelope[lmax]
            keep = np.diff(np.sign(np.diff(np.concatenate(([0], vals, [0])))) < 0) > 0
            marker[lmax[keep]] = True
    return marker


def _detection_union(marker, envelope, union_samples):
    """detection_union.m: morphological close, then one marker (envelope max) per run."""
    us = int(np.ceil(union_samples))
    if us % 2 == 0:
        us += 1
    mask = np.ones(us)
    m = np.convolve(marker.astype(float), mask, "same") > 0            # dilation
    m = ~(np.convolve((~m).astype(float), mask, "same") > 0)           # erosion

    out = np.zeros(marker.shape[0], bool)
    starts = np.flatnonzero(np.diff(np.concatenate(([0], m.astype(int)))) > 0)
    stops = np.flatnonzero(np.diff(np.concatenate((m.astype(int), [0]))) < 0)
    for p1, p2 in zip(starts, stops):
        out[p1 + int(np.argmax(envelope[p1:p2 + 1]))] = True
    return out


def _one_channel_detect(d, fs, index, winsize, k1, k2, k3, pt, ti_switch, d_decim):
    """one_channel_detect.m: envelope, lognormal-background thresholds, markers."""
    envelope = np.abs(hilbert(d))

    # per-window MLE of log-envelope (lognormal) params
    phat = np.zeros((len(index), 2))
    for k, st in enumerate(index):
        seg = envelope[st:st + winsize]
        seg = seg[seg > 0]
        lg = np.log(seg)
        phat[k, 0] = lg.mean()
        phat[k, 1] = lg.std(ddof=1)

    r = envelope.shape[0] / len(index)
    n_average = winsize / fs
    L = int(round(n_average * fs / r))
    if L > 1 and phat.shape[0] > 3 * L:
        b = np.ones(L) / L
        phat = filtfilt(b, [1.0], phat, axis=0)

    # interpolate the two params onto a per-sample "background" curve
    if phat.shape[0] > 1:
        xs = np.asarray(index) + round(winsize / 2)
        xq = np.arange(index[0], index[-1] + 1) + round(winsize / 2)
        pi0 = CubicSpline(xs, phat[:, 0])(xq)
        pi1 = CubicSpline(xs, phat[:, 1])(xq)
        head = int(np.floor(winsize / 2))
        tail = envelope.shape[0] - (len(pi0) + head)
        tail = max(tail, 0)
        phat_int = np.empty((envelope.shape[0], 2))
        phat_int[:, 0] = np.concatenate([np.full(head, pi0[0]), pi0, np.full(tail, pi0[-1])])[:envelope.shape[0]]
        phat_int[:, 1] = np.concatenate([np.full(head, pi1[0]), pi1, np.full(tail, pi1[-1])])[:envelope.shape[0]]
    else:
        phat_int = np.ones((d.shape[0], 2)) * phat

    mu, sig = phat_int[:, 0], phat_int[:, 1]
    ln_mode = np.exp(mu - sig ** 2)
    ln_median = np.exp(mu)
    ln_mean = np.exp(mu + sig ** 2 / 2)

    prah = np.empty((envelope.shape[0], 2))
    prah[:, 0] = k1 * (ln_mode + ln_median) - k3 * (ln_mean - ln_mode)
    if k2 != k1:
        prah[:, 1] = k2 * (ln_mode + ln_median) - k3 * (ln_mean - ln_mode)
    else:
        prah[:, 1] = prah[:, 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        env_cdf = 0.5 + 0.5 * _erf((np.log(envelope) - mu) / np.sqrt(2 * sig ** 2))
        env_pdf = np.exp(-0.5 * ((np.log(envelope) - mu) / sig) ** 2) / (envelope * sig * np.sqrt(2 * np.pi))

    markers_high = _local_maxima_detection(envelope, prah[:, 0], fs, pt, ti_switch, d_decim)
    markers_high = _detection_union(markers_high, envelope, pt * fs)
    if k2 != k1:
        markers_low = _local_maxima_detection(envelope, prah[:, 1], fs, pt, ti_switch, d_decim)
        markers_low = _detection_union(markers_low, envelope, pt * fs)
    else:
        markers_low = markers_high

    return envelope, markers_high, markers_low, prah, env_cdf, env_pdf


def _erf(x):
    from scipy.special import erf
    return erf(x)


# ----------------------------------------------------------------------
# one buffer of signal -> out + discharges (spike_detector.m)
# ----------------------------------------------------------------------
def _spike_detector(d, fs, winsize, noverlap, cfg):
    dec = cfg["decimation"]
    k1 = cfg["k1"]
    k2 = cfg["k2"] if cfg["k2"] is not None else k1
    k3 = cfg["k3"]
    pt = cfg["polyspike_union_time"]
    dt = cfg["discharge_tol"]
    ti = int(cfg["ti_switch"])
    beta = cfg["beta"]
    beta_win = cfg["beta_win"]

    # --- decimate ---
    rfactor = fs / dec
    if rfactor > 1 or dec != fs:
        ws_s, nov_s = winsize / fs, noverlap / fs
        d, fs, rfactor = _decimate(d, fs, dec)
        winsize = int(round(ws_s * fs))
        noverlap = int(round(nov_s * fs))

    fs = int(round(fs))
    n, nch = d.shape

    # --- segmentation start indices (0-based) ---
    if noverlap < 1:
        step = int(round(winsize * (1 - noverlap)))
    else:
        step = int(winsize - noverlap)
    step = max(step, 1)
    index = np.arange(0, n - winsize + 1, step)
    if len(index) == 0:
        index = np.array([0])

    # --- mains + high-pass, keep the "clean raw" (d_decim) ---
    d = _filt_50hz(d, fs, cfg["main_hum_freq"], cfg["band_high"])
    bb, aa = butter(2, 2 * 1.0 / fs, "high")
    d_decim = filtfilt(bb, aa, d, axis=0)

    M_beta = None
    if beta < fs / 2 and beta_win > 0:
        M_beta = _beta_detect(d, fs, beta, beta_win, cfg["beta_ar"])

    # --- bandpass ---
    d = _bandpass(d, fs, cfg["band_low"], cfg["band_high"], int(cfg["f_type"]), dec)

    # --- per channel ---
    envelope = np.zeros((n, nch))
    markers_high = np.zeros((n, nch), bool)
    markers_low = np.zeros((n, nch), bool)
    background = np.zeros((n, nch, 1 if k1 == k2 else 2))
    env_cdf = np.zeros((n, nch))
    env_pdf = np.zeros((n, nch))
    for ch in range(nch):
        if np.all(d[:, ch] == 0):
            continue
        e, mh, ml, prah, ec, ep = _one_channel_detect(
            d[:, ch], fs, index, winsize, k1, k2, k3, pt, ti, d_decim[:, ch])
        envelope[:, ch] = e
        markers_high[:, ch] = mh
        markers_low[:, ch] = ml
        background[:, ch, :] = prah[:, :background.shape[2]]
        env_cdf[:, ch] = ec
        env_pdf[:, ch] = ep

    # --- edges are unreliable (filter response) ---
    markers_high[:fs, :] = markers_high[-fs:, :] = False
    markers_low[:fs, :] = markers_low[-fs:, :] = False
    tail = int(np.ceil(dt * fs + 1))
    markers_high[-tail:, :] = False
    markers_low[-tail:, :] = False

    if M_beta is not None:
        markers_high[M_beta] = False
        markers_low[M_beta] = False

    obvious_M = markers_high.sum(axis=1) > 0

    # --- obvious spikes ---
    pos, dur, chan, con, weight, pdf = [], [], [], [], [], []
    t_dur = 0.005
    for ch in range(nch):
        idx = np.flatnonzero(markers_high[:, ch])
        if idx.size:
            pos.append(idx / fs)
            dur.append(np.full(idx.size, t_dur))
            chan.append(np.full(idx.size, ch))
            con.append(np.ones(idx.size))
            weight.append(env_cdf[idx, ch])
            pdf.append(env_pdf[idx, ch])

    # --- ambiguous spikes (only when k2 != k1) ---
    if k2 != k1:
        for ch in range(nch):
            idx = np.flatnonzero(markers_low[:, ch])
            idx = idx[~markers_high[idx, ch]]
            for i in idx:
                lo = int(round(i - dt * fs))
                hi = int(round(i + dt * fs))
                lo, hi = max(lo, 0), min(hi, n - 1)
                if obvious_M[lo:hi + 1].sum() > 0:
                    pos.append(np.array([i / fs]))
                    dur.append(np.array([t_dur]))
                    chan.append(np.array([ch]))
                    con.append(np.array([0.5]))
                    weight.append(np.array([env_cdf[i, ch]]))
                    pdf.append(np.array([env_pdf[i, ch]]))

    def _cat(parts):
        return np.concatenate(parts) if parts else np.zeros(0)

    out = dict(pos=_cat(pos), dur=_cat(dur), chan=_cat(chan).astype(int),
               con=_cat(con), weight=_cat(weight), pdf=_cat(pdf))

    discharges = _build_discharges(out, d, d_decim, envelope, background, env_cdf,
                                   env_pdf, fs, dt, k1, ti)
    return d_decim, envelope, background, discharges, out, env_pdf, rfactor, fs


def _build_discharges(out, d, d_decim, envelope, background, env_cdf, env_pdf,
                      fs, dt, k1, ti):
    """Group per-channel spikes into multichannel events (the discharges.* struct)."""
    n, nch = d.shape
    M = np.zeros((n, nch))
    for k in range(out["pos"].size):
        a = out["pos"][k] * fs
        rows = np.round(np.arange(a, a + dt * fs + 1e-9)).astype(int)
        rows = rows[(rows >= 0) & (rows < n)]
        M[rows, out["chan"][k]] = out["con"][k]

    active = (M.sum(axis=1) > 0).astype(int)
    starts = np.flatnonzero(np.diff(np.concatenate(([0], active))) > 0)
    stops = np.flatnonzero(np.diff(np.concatenate((active, [0]))) < 0)

    keys = ("MV", "MA", "MP", "MD", "MW", "MPDF", "MRAW")
    if starts.size == 0:
        return {k: np.zeros((0, nch)) for k in keys}

    rows_out = {k: [] for k in keys}
    for p1, p2 in zip(starts, stops):
        seg_M = M[p1:p2 + 1, :]
        mv = seg_M.max(axis=0)

        if ti == 1:
            seg = envelope[p1:p2 + 1, :] - background[p1:p2 + 1, :, 0] / k1
        else:
            lo = max(p1 - int(round(fs * 10e-3)), 0)
            hi = min(p2 + int(round(fs * 10e-3)), n - 1)
            seg = envelope[lo:hi + 1, :] - background[lo:hi + 1, :, 0] / k1
        ma = np.abs(seg).max(axis=0)

        raw = d_decim[p1:p2 + 1, :]
        poz = np.argmax(np.abs(raw), axis=0)
        mraw = np.abs(raw)[poz, np.arange(nch)] * np.sign(raw[poz, np.arange(nch)])

        mw = env_cdf[p1:p2 + 1, :].max(axis=0)
        mpdf = (env_pdf[p1:p2 + 1, :] * (seg_M > 0)).max(axis=0)

        mp = np.full(nch, np.nan)
        for ch in range(nch):
            hits = np.flatnonzero(seg_M[:, ch] > 0)
            if hits.size:
                mp[ch] = (hits[-1] + p1) / fs   # last (max-row) hit, column-major MATLAB
        md = np.full(nch, (p2 - p1) / fs)

        for key, val in zip(keys, (mv, ma, mp, md, mw, mpdf, mraw)):
            rows_out[key].append(val)

    return {k: np.vstack(v) for k, v in rows_out.items()}


# ----------------------------------------------------------------------
# public entry point (spike_detector_hilbert_v24.m top level)
# ----------------------------------------------------------------------
def detect_spikes(d, fs, settings=None, **overrides):
    """Detect interictal epileptiform discharges (Janca et al., v24 port).

    Parameters
    ----------
    d : array (n_samples, n_channels)   raw SEEG/iEEG, time-major. 1-D is treated
        as a single channel.
    fs : float                          sampling rate of `d` (Hz).
    settings : str, optional            MATLAB-style flags, e.g. "-k1 3.0 -fh 80".
    **overrides                         keyword equivalents (band_low, band_high, k1,
        k2, k3, decimation, beta, ...). See _DEFAULTS. Passing None keeps the default.

    Returns
    -------
    out : dict of arrays (one row per detected spike)
        pos    : position in SECONDS (real time from start of d)
        dur    : duration in seconds (fixed 5 ms)
        chan   : channel index, 0-BASED
        con    : 1.0 obvious, 0.5 ambiguous
        weight : envelope lognormal CDF at the spike (significance)
        pdf    : envelope lognormal PDF at the spike
    discharges : dict of (n_events x n_channels) arrays -- multichannel events
        (MV, MA, MP, MD, MW, MPDF, MRAW). MP is in seconds (NaN where a channel did
        not participate). Empty (0 x n_chan) arrays if nothing was detected.
    info : dict  (fs_decimated, and, for the single-buffer case, the decimated
        signal / envelope / background curves for plotting; None for those when the
        recording spanned multiple buffers).

    Example
    -------
    >>> out, disch, info = detect_spikes(x, 2048.0)          # defaults (dec 200 Hz)
    >>> # per-channel 0-based sample indices at the ORIGINAL fs:
    >>> per_chan = [np.round(out['pos'][out['chan'] == c] * fs).astype(int)
    ...             for c in range(x.shape[1])]
    """
    cfg = _parse_settings(settings, overrides)

    d = np.asarray(d, float)
    if d.ndim == 1:
        d = d[:, None]
    n_orig, nch = d.shape

    winsize = int(cfg["winsize"]) if cfg["winsize"] else int(round(5 * fs))
    noverlap = cfg["noverlap"] if cfg["noverlap"] else 4 * fs

    buffering = cfg["buffering"]
    if buffering / (winsize / fs) < 10:
        buffering = (winsize / fs) * 10
    if buffering > n_orig / fs:
        buffering = n_orig / fs
    if cfg["decimation"] == 0:
        cfg["decimation"] = fs
    if cfg["band_high"] > cfg["decimation"]:
        raise ValueError("band_high (-fh) exceeds the decimated Nyquist (decimation)")

    # --- buffer boundaries (0-based, with two-sided 3*winsize overlap) ---
    N_seg = max(int(np.floor(n_orig / (buffering * fs))), 1)
    T_seg = int(round(n_orig / N_seg / fs))
    starts = np.arange(0, n_orig, T_seg * fs)
    if starts.size > 1:
        starts[1:] = starts[1:] - 3 * winsize
        stops = starts + T_seg * fs + 2 * (3 * winsize) - 1
        stops[0] = stops[0] - 3 * winsize
        stops[-1] = n_orig - 1
        if stops[-1] - starts[-1] < T_seg * fs:
            starts = starts[:-1]
            stops = np.delete(stops, -2)
    else:
        stops = np.array([n_orig - 1])
    starts = starts.astype(int)
    stops = stops.astype(int)

    agg = {k: [] for k in ("pos", "dur", "chan", "con", "weight", "pdf")}
    dagg = {k: [] for k in ("MV", "MA", "MP", "MD", "MW", "MPDF", "MRAW")}
    single_buffer = len(stops) == 1
    info = {"fs_decimated": None, "d_decim": None, "envelope": None, "background": None}

    for i in range(len(stops)):
        sub = d[starts[i]:stops[i] + 1, :]
        d_dec, env, bg, disch, sout, _epdf, _rf, fs_dec = _spike_detector(
            sub, fs, winsize, noverlap, cfg)
        info["fs_decimated"] = fs_dec

        left = 3 * winsize / fs if i > 0 else 0.0
        seg_len = (stops[i] - starts[i]) / fs
        right = seg_len - (3 * winsize / fs if i < len(stops) - 1 else 0.0)
        offset = starts[i] / fs

        if sout["pos"].size:
            keep = (sout["pos"] >= left) & (sout["pos"] <= right)
            agg["pos"].append(sout["pos"][keep] + offset)
            for k in ("dur", "chan", "con", "weight", "pdf"):
                agg[k].append(sout[k][keep])

        if disch["MP"].shape[0]:
            mpmin = np.nanmin(disch["MP"], axis=1)
            keepd = (mpmin >= left) & (mpmin <= right)
            for k in dagg:
                block = disch[k][keepd]
                if k == "MP":
                    block = block + offset
                dagg[k].append(block)

        if single_buffer:
            info["d_decim"] = d_dec
            info["envelope"] = env
            info["background"] = bg

    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in agg.items()}
    out["chan"] = out["chan"].astype(int)
    discharges = {k: (np.vstack(v) if v else np.zeros((0, nch))) for k, v in dagg.items()}

    # drop detections in the first/last 2 s (filter-response artefacts)
    dur_s = n_orig / fs
    m = (out["pos"] > 2) & (out["pos"] < dur_s - 2)
    out = {k: v[m] for k, v in out.items()}
    if discharges["MP"].shape[0]:
        mpmin = np.nanmin(discharges["MP"], axis=1)
        md = (mpmin > 2) & (mpmin < dur_s - 2)
        discharges = {k: v[md] for k, v in discharges.items()}

    return out, discharges, info


# ----------------------------------------------------------------------
# smoke test (verifies the default path end-to-end)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fs = 512.0
    T = 30.0
    n = int(T * fs)
    t = np.arange(n) / fs
    nch = 3
    x = rng.standard_normal((n, nch)) * 20.0            # background noise (uV-ish)
    x += 15 * np.sin(2 * np.pi * 50 * t)[:, None]       # mains hum on every channel

    # inject sharp biphasic spikes on channels 0 and 2 at known times
    spike_times = [5.0, 9.3, 14.7, 20.1, 25.5]
    def _spike(width_s=0.03, amp=350.0):
        w = int(width_s * fs)
        tt = np.linspace(-3, 3, 2 * w)
        return amp * (-tt * np.exp(-tt ** 2))
    wav = _spike()
    for ts in spike_times:
        c = int(ts * fs)
        seg = slice(c - len(wav) // 2, c - len(wav) // 2 + len(wav))
        x[seg, 0] += wav
        x[seg, 2] += 0.8 * wav

    out, disch, info = detect_spikes(x, fs)

    print(f"decimated fs         : {info['fs_decimated']} Hz")
    print(f"total spikes detected: {out['pos'].size}")
    print(f"multichannel events  : {disch['MP'].shape[0]}")
    for c in range(nch):
        pc = np.sort(out["pos"][out["chan"] == c])
        print(f"  channel {c}: {pc.size:2d} spikes  {np.round(pc, 2).tolist()}")
    print(f"injected spike times : {spike_times}")
    # sanity: every injected spike (that is >2s from the edges) should land on ch0
    hit = [any(abs(out['pos'][out['chan'] == 0] - ts) < 0.1) for ts in spike_times]
    print(f"injected recovered on ch0: {hit}")
