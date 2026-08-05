"""
test_sim_data.py
----------------
Checks the synthetic-data generator and the hand-rolled EDF writer without building a 38 MB
recording or touching Delphos.

Worth keeping as a tool. Two of these are load-bearing for every number the scorer produces:
  * test_edf_roundtrip / test_edf_affine -- we WRITE the EDF that Delphos reads. If the writer
    is wrong, Delphos looks wrong, and that misdiagnosis costs ~5 min per attempt to chase.
  * test_truth_alignment -- ground truth is generated at 2000 Hz and scored against detections
    on the 400 Hz decimated axis. If decimate_recording ever shifts timing, every recall
    number silently drops and nothing else would tell you.

    .venv\\Scripts\\python.exe test_sim_data.py
"""
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

import sim_data as sd


# ----------------------------------------------------------------------
def test_mround():
    """MATLAB rounds half AWAY FROM ZERO; numpy rounds half to EVEN. Ground-truth sample
    indices come out of this, so the difference is a one-sample offset on every tie."""
    got = sd.mround([0.5, 1.5, 2.5, 3.5, -0.5, -2.5]).tolist()
    assert got == [1.0, 2.0, 3.0, 4.0, -1.0, -3.0], got
    assert np.round([0.5, 1.5, 2.5]).tolist() == [0.0, 2.0, 2.0], "numpy changed its tie rule"
    print("  mround: half away from zero, unlike np.round")


def test_template_matlab_parity():
    """taper_ms=0 reproduces make_spike_template.m exactly, step discontinuity and all."""
    # the MATLAB used sigma=0.120 s directly; make_template takes FWHM, so convert
    # exactly rather than rounding, or parity misses by ~1e-5
    LEGACY = dict(taper_ms=0, sw_amp=-0.4, us_amp=0.0, sw_delay_ms=110.0,
                  sw_ms=0.120 * 2 * math.sqrt(2 * math.log(2)) * 1000)
    t, pk = sd.make_template(2000.0, 50.0, 2.5, True, "gaussian", **LEGACY)
    assert pk == 30, pk
    assert abs(np.max(np.abs(t)) - 1.0) < 1e-12
    assert abs(t[pk] - 1.0) < 1e-12, "peak must be the POSITIVE sharp component"
    lo = int(np.argmin(t))
    assert abs(t[lo] - (-0.542580517)) < 1e-6, t[lo]
    assert abs((lo - pk) / 2000.0 - 0.110) < 1e-3, "slow-wave trough should sit at +110 ms"

    t2, _ = sd.make_template(2000.0, 50.0, 2.5, False, "gaussian", **LEGACY)
    assert t2.min() >= 0.0, "no slow wave -> no negative lobe"
    print(f"  template (MATLAB parity): peak@30, trough {t[lo]:.6f} at +110 ms")


def test_template_taper_preserves_the_spike():
    """The taper must remove the end step WITHOUT touching anything a detector gates on.

    The rising edge is Barkmeier's left-slope/left-duration feature and the trough is the slow
    wave; if the window clipped either, every Barkmeier number would move for a reason that has
    nothing to do with Barkmeier. Hence the front PAD -- ramping over the existing 15 ms lead-in
    distorted the rising edge by 0.375, which is why it is not done that way."""
    old, po = sd.make_template(2000.0, taper_ms=0)
    new, pn = sd.make_template(2000.0, taper_ms=60.0)
    assert abs(new[0]) < 1e-12 and abs(new[-1]) < 1e-12, "both ends must be exactly zero"
    assert abs(new[pn] - 1.0) < 1e-12, "peak still unit"
    # compare peak-aligned over the spike and the after-wave, out to +150 ms. Written on the
    # TIME axis, not on argmin, so it holds for either after-wave polarity.
    w = int(0.150 * 2000)
    # lead clamped to what both templates actually have: the undershoot pulls the argmax a
    # sample earlier than lead0, so a hard -30 ran off the front of the untapered one
    lead = min(25, po, pn)
    a, b = old[po - lead:po + w], new[pn - lead:pn + w]
    assert a.size == b.size > 0, (a.size, b.size)
    assert np.max(np.abs(b - a)) < 1e-12, np.max(np.abs(b - a))
    print(f"  template (tapered): n={new.size} peak@{pn}, ends exactly 0, "
          f"spike bit-identical from -15 ms through the trough")


def test_three_component_morphology():
    """sharp transient -> dip BELOW baseline -> same-direction mound.

    Pinned because these three numbers ARE the stimulus: Barkmeier's right half-wave is
    peak-minus-min within trough_search_ms (40 ms), so the undershoot's depth and latency set
    Ramp and Rdur directly -- the features it gates on."""
    t, pk = sd.make_template(2000.0)
    assert abs(t[pk] - 1.0) < 1e-12 and abs(t[0]) < 1e-12 and abs(t[-1]) < 1e-12

    w = int(0.09 * 2000)
    lo = int(np.argmin(t[pk:pk + w])) + pk
    assert t[lo] < -0.10, f"post-transient minimum must go BELOW baseline, got {t[lo]:+.3f}"
    assert 25 <= (lo - pk) / 2 <= 55, f"undershoot at +{(lo - pk) / 2:.0f} ms"

    aft = t[pk + int(0.06 * 2000):]
    mo = int(np.argmax(aft)) + pk + int(0.06 * 2000)
    assert 0.40 < t[mo] < 0.62, f"mound should be ~50% of the peak, got {t[mo]:+.3f}"
    idx = np.flatnonzero(t > t[mo] / 2)
    seg = idx[idx > lo]
    fwhm = (seg[-1] - seg[0]) / 2
    assert 65 <= fwhm <= 95, f"mound FWHM {fwhm:.0f} ms, wanted ~80"
    print(f"  morphology: peak +1.00 -> trough {t[lo]:+.3f} at +{(lo-pk)/2:.0f} ms -> "
          f"mound {t[mo]:+.3f} at +{(mo-pk)/2:.0f} ms, FWHM {fwhm:.0f} ms")


def test_stimulus_is_fair_to_all_arms():
    """The template must give EVERY detector usable signal, or a measured floor says nothing
    about the detector.

    Note the trap this guards: only 0.04% of the template's ENERGY sits above 80 Hz, which
    reads as "Delphos (8-512 Hz) is blind to this". It isn't -- the AR background is steeply
    1/f, so that band holds very little noise either, and Delphos's in-band SNR lands within
    ~10% of Janca's. Assert on in-band SNR, never on the energy fraction."""
    cfg = sd.default_cfg()
    nm = sd.load_noise_model()
    worst = sd.inband_snr(cfg, min(sd.SNR_LIST), model=nm)
    best = sd.inband_snr(cfg, max(sd.SNR_LIST), model=nm)
    med_w = dict(zip(worst["dets"], np.median(worst["snr"], axis=0)))
    med_b = dict(zip(best["dets"], np.median(best["snr"], axis=0)))
    for d, v in med_b.items():
        assert v > 1.5, f"{d} in-band SNR only {v:.2f} at the top level -- stimulus problem"
    spread = max(med_b.values()) / min(med_b.values())
    assert spread < 3.0, f"arms differ {spread:.1f}x in-band; the template favours one of them"
    assert sd.band_energy(cfg)["80-512 Hz"] < 0.01, "the energy fraction really is tiny"
    print("  stimulus fairness: in-band SNR at nominal "
          f"{min(sd.SNR_LIST):g}/{max(sd.SNR_LIST):g} = "
          + ", ".join(f"{d} {med_w[d]:.1f}/{med_b[d]:.1f}" for d in med_b)
          + f" (spread {spread:.2f}x)")


def test_draw_spike_times():
    fs, dur, guard, n = 2000.0, 600.0, 601, 1_200_000
    assert sd.draw_spike_times(0.0, dur, fs, 200.0, guard, n,
                               np.random.default_rng(0)).size == 0, "rate 0 must be silent"

    pk = sd.draw_spike_times(30.0, dur, fs, 200.0, guard, n, np.random.default_rng(1))
    assert pk.min() >= guard and pk.max() < n - guard, "template must fit inside the record"
    isi = np.diff(pk) / fs
    assert isi.min() >= 0.200 - 1.5 / fs, f"refractory floor violated: {isi.min() * 1000:.2f} ms"
    # censoring puts a point mass AT the floor, so it must actually be reached
    assert (isi < 0.2005).sum() > 0, "clamped-exponential should pile up at min_isi"
    # 30/min over 600 s = 300 nominal; censoring pulls it slightly down
    assert 240 <= pk.size <= 320, pk.size
    same = sd.draw_spike_times(30.0, dur, fs, 200.0, guard, n, np.random.default_rng(1))
    assert np.array_equal(pk, same), "same seed must give the same train"
    print(f"  spike times: {pk.size} spikes, min ISI {isi.min() * 1000:.1f} ms, "
          f"{(isi < 0.2005).sum()} at the floor")


def test_ar_background():
    nm = sd.load_noise_model()
    assert nm["fs"] == 2000.0 and nm["ar_order"] == 16 and len(nm["models"]) == 164
    m = nm["models"][0]
    x = sd.ar_background(m, 200_000, np.random.default_rng(3))
    ratio = np.std(x, ddof=1) / m["source_std"]
    assert 0.85 < ratio < 1.15, f"synth std is {ratio:.3f}x the real channel's"

    # the all-pole recursion, written out, must equal what lfilter did
    rng = np.random.default_rng(4)
    drive = m["resid_std"] * rng.standard_normal(50)
    ref = np.zeros(50)
    a = m["a"]
    for i in range(50):
        acc = drive[i]
        for k in range(1, a.size):
            if i - k >= 0:
                acc -= a[k] * ref[i - k]
        ref[i] = acc
    got = sd.ar_background(m, 50, np.random.default_rng(4))
    assert np.allclose(got, ref, atol=1e-9), np.max(np.abs(got - ref))
    print(f"  AR(16): std {ratio:.3f}x source_std, matches an explicit 17-tap recursion")


# ----------------------------------------------------------------------
def test_edf_roundtrip(tmp):
    """Header, label ORDER, <=1 LSB reconstruction, exact file size."""
    fs, n_sec, n_ch = 500, 3, 4
    rng = np.random.default_rng(5)
    x = rng.normal(0, 40, (fs * n_sec, n_ch))
    x[:, 0] *= 20          # widely different per-channel ranges
    x[:, 3] = 0.0          # an all-zero channel must not give a zero-width physical range
    labels = ["SIM1", "SIM2", "SIM3", "SIM4"]

    p = sd.write_edf(tmp / "rt.edf", x, labels, fs)
    v = sd.verify_edf(p, x, labels, fs)          # raises on any mismatch
    assert v["max_err_lsb"] <= 1.0, v["max_err_lsb"]
    assert p.stat().st_size == 256 * (1 + n_ch) + n_sec * n_ch * fs * 2
    print(f"  EDF round trip: max {v['max_err_lsb']:.3f} LSB, "
          f"lsb {v['lsb'].min():.4g}-{v['lsb'].max():.4g} uV, {v['bytes']} bytes")


def test_edf_affine(tmp):
    """The digital range is asymmetric (-32768..32767) against a symmetric physical range.
    Using x/pmax*32767 instead of the EDF affine gives every channel a half-LSB DC offset --
    invisible in a plot, wrong in a threshold."""
    from seeg import read_edf_header, load_edf_segment
    fs = 100
    x = np.linspace(-90.0, 90.0, fs * 2).reshape(-1, 1)
    p = sd.write_edf(tmp / "affine.edf", x, ["SIM1"], fs)
    hdr = read_edf_header(p)
    y = load_edf_segment(p, hdr, 1, 2)["data"]

    pmax = hdr["PhysicalMax"][0]
    assert pmax == float(int(pmax)), f"physical range must be integer, got {pmax}"
    gain = (hdr["DigitalMax"][0] - hdr["DigitalMin"][0]) / (pmax - hdr["PhysicalMin"][0])
    want = np.clip(np.round(x[:, 0] * gain - 0.5), -32768, 32767)
    back = (want - hdr["DigitalMin"][0]) * (pmax - hdr["PhysicalMin"][0]) / \
           (hdr["DigitalMax"][0] - hdr["DigitalMin"][0]) + hdr["PhysicalMin"][0]
    assert np.allclose(y[:, 0], back, atol=1e-6), np.max(np.abs(y[:, 0] - back))
    bias = float(np.mean(y[:, 0] - x[:, 0]))
    lsb = (pmax - hdr["PhysicalMin"][0]) / 65535
    assert abs(bias) < 0.05 * lsb, f"DC bias {bias:.5g} uV is {bias / lsb:.2f} LSB"
    print(f"  EDF affine: mean reconstruction bias {bias / lsb:+.3f} LSB (pmax={pmax:g} uV)")


def test_edf_rejects_bad_input(tmp):
    x = np.zeros((250, 2))
    for bad, why in [(dict(labels=["A"]), "label count"),
                     (dict(labels=["A", "A"]), "duplicate labels"),
                     (dict(fs=100), "non-integer records")]:
        kw = dict(labels=["A", "B"], fs=500)
        kw.update(bad)
        try:
            sd.write_edf(tmp / "bad.edf", x, kw["labels"], kw["fs"])
        except ValueError:
            continue
        raise AssertionError(f"write_edf accepted {why}")
    print("  EDF writer rejects mismatched labels, duplicates and ragged records")


# ----------------------------------------------------------------------
def test_truth_alignment():
    """THE load-bearing assumption: truth at 2000 Hz maps onto the 400 Hz detection axis as
    idx/5, with no shift from medfilt2d(5) + the anti-alias FIR."""
    from seeg import decimate_recording
    fs, factor = 2000.0, 5
    tmpl, peak = sd.make_template(fs)
    n = int(30 * fs)
    x = np.zeros((n, 1))
    truth = np.array([4000, 9001, 15002, 23003, 27004])   # deliberately not multiples of 5
    for t in truth:
        x[t - peak:t - peak + tmpl.size, 0] += 100.0 * tmpl

    dec = decimate_recording({"data": x, "info": {"SampleRate": fs, "NSamples": n,
                                                  "SelectedSignals": ["SIM1"]}},
                             factor=factor)
    y = dec["data"][:, 0]
    errs, amps = [], []
    for t in truth:
        c = int(round(t / factor))
        w = y[c - 8:c + 9]
        errs.append(int(np.argmax(w)) - 8)
        amps.append(w.max() / 100.0)
    assert max(abs(e) for e in errs) <= 1, f"peak offsets {errs} samples at 400 Hz"
    assert min(amps) > 0.98, f"peak amplitudes {amps}"
    print(f"  truth alignment: offsets {errs} samples, amplitude "
          f"{min(amps):.5f}-{max(amps):.5f}x after 2000->400 Hz")


def test_cfg_hash_changes():
    base = sd.default_cfg()
    assert sd.edf_name(base, 8) == sd.edf_name(sd.default_cfg(), 8), "hash must be stable"
    for k, v in [("sharpness", 8.0), ("seed", 1), ("n_chan", 32), ("undershoot_amp", 0.0)]:
        other = sd.default_cfg(**{k: v})
        assert sd.edf_name(other, 8) != sd.edf_name(base, 8), f"{k} did not change the name"
    assert sd.edf_name(base, 8) != sd.edf_name(base, 12), "SNR must be in the name"
    try:
        sd.default_cfg(sharpnes=8.0)
    except TypeError:
        pass
    else:
        raise AssertionError("default_cfg accepted a misspelt setting")
    print(f"  cfg hash: {sd.cfg_hash(base)} for the current config, changes with every knob")


def test_fixed_peak_and_amplitude_spread():
    """FIXED_PEAK_UV holds the spike constant and moves the NOISE; AMP_LOG_SD spreads the
    per-spike amplitude while leaving the MEDIAN exactly on target."""
    cfg = sd.default_cfg(n_chan=3, dur_sec=300, fixed_peak_uv=1400.0, amp_log_sd=0.61,
                         equalise_channel_snr=True)
    out = {}
    for snr in (1.0, 6.0, 12.0):
        s = sd.synthesise(cfg, snr)
        out[snr] = (np.median(s["truth_amp"]), s["noise_std"])
        # every channel must land on the SAME realised SNR -- that is the point of scaling
        # the noise per channel rather than applying one global multiplier
        assert np.allclose(s["noise_std"], s["noise_std"][0], rtol=1e-9), s["noise_std"]
        assert abs(1400.0 / s["noise_std"][0] - snr) < 0.02 * snr, (snr, s["noise_std"][0])
    # the spike does NOT change with SNR; the noise does, inversely
    meds = [out[k][0] for k in out]
    assert max(meds) / min(meds) < 1.15, f"spike size drifted across SNR: {meds}"
    assert out[1.0][1][0] > 10 * out[12.0][1][0], "noise should scale as 1/snr"

    # median on target, spread matching the real p90/p10 of 4.8
    big = sd.synthesise(sd.default_cfg(n_chan=4, dur_sec=600, fixed_peak_uv=1400.0,
                                       amp_log_sd=0.61, equalise_channel_snr=True),
                        6.0)["truth_amp"]
    assert abs(np.median(big) / 1400.0 - 1.0) < 0.10, np.median(big)
    ratio = np.percentile(big, 90) / np.percentile(big, 10)
    assert 3.5 < ratio < 6.5, f"p90/p10 = {ratio:.2f}, expected ~4.8"

    # and amp_log_sd=0 must still give identical spikes (MATLAB behaviour)
    flat = sd.synthesise(sd.default_cfg(n_chan=2, dur_sec=120, fixed_peak_uv=1400.0,
                                        amp_log_sd=0.0, equalise_channel_snr=True),
                         6.0)["truth_amp"]
    assert np.allclose(flat, 1400.0), "amp_log_sd=0 must switch the spread off"
    print(f"  fixed peak: median {np.median(big):.0f} uV (target 1400), p90/p10 {ratio:.2f}, "
          f"noise {out[1.0][1][0]:.0f} uV at SNR 1 -> {out[12.0][1][0]:.0f} uV at SNR 12")


def test_channel_snr_spread_is_preserved():
    """EQUALISE_CHANNEL_SNR=False must keep the AR pool's natural amplitude spread, so channels
    sit at DIFFERENT SNRs inside one recording.

    That spread is the whole point: Barkmeier normalises by ONE global scalar per block
    (mDetectSpike.m:284, median across channels), while Janca and Delphos normalise per
    channel. With every channel identical the asymmetry cannot be measured at all."""
    cfg = sd.default_cfg(n_chan=16, dur_sec=120, fixed_peak_uv=143.0,
                         equalise_channel_snr=False)
    s = sd.synthesise(cfg, 8.0)
    ns = s["noise_std"]
    ch_snr = 143.0 / ns
    assert ch_snr.max() / ch_snr.min() > 3.0, f"channels too uniform: {ch_snr.min():.1f}-{ch_snr.max():.1f}"
    # the MEDIAN channel must still land on the nominal SNR -- that is what the axis means
    assert abs(np.median(ch_snr) / 8.0 - 1.0) < 0.12, np.median(ch_snr)
    # and the equalised mode must still collapse it
    eq = sd.synthesise(sd.default_cfg(n_chan=16, dur_sec=120, fixed_peak_uv=143.0,
                                      equalise_channel_snr=True), 8.0)["noise_std"]
    assert np.allclose(eq, eq[0], rtol=1e-9), "equalise=True must give identical channels"
    print(f"  channel SNR spread: {ch_snr.min():.1f}-{ch_snr.max():.1f} "
          f"(median {np.median(ch_snr):.1f}, nominal 8), noise {ns.min():.0f}-{ns.max():.0f} uV")


def test_synthesise_small():
    """End to end on a 20 s / 4 ch toy in the LEGACY amplitude mode: amp = snr * noise std."""
    # legacy/MATLAB amplitude convention, pinned explicitly so the module defaults can move
    cfg = sd.default_cfg(n_chan=4, dur_sec=20, max_rate_min=60.0,
                         fixed_peak_uv=None, amp_log_sd=0.0)
    s = sd.synthesise(cfg, snr=5.0)
    assert s["data"].shape == (40_000, 4)
    assert (s["truth_chan"] == 0).sum() == 0, "channel 0 is the zero-rate control"
    for ch in range(1, 4):
        amp = s["truth_amp"][s["truth_chan"] == ch]
        assert np.allclose(amp, 5.0 * s["noise_std"][ch]), "amp = snr * pre-injection noise std"
    counts = [(s["truth_chan"] == c).sum() for c in range(4)]
    assert counts[1] < counts[3], f"rate ramp not monotone: {counts}"
    # the injected peak must actually be there, at the recorded index
    ch = 3
    idx = s["truth_idx"][s["truth_chan"] == ch]
    peak_vals = s["data"][idx, ch]
    assert np.median(peak_vals) > 2.0 * s["noise_std"][ch], np.median(peak_vals)
    print(f"  synthesise: counts per channel {counts}, "
          f"median peak {np.median(peak_vals) / s['noise_std'][ch]:.1f}x noise std")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="sim_data_test_"))
    try:
        print("test_sim_data.py")
        test_mround()
        test_template_matlab_parity()
        test_template_taper_preserves_the_spike()
        test_three_component_morphology()
        test_stimulus_is_fair_to_all_arms()
        test_draw_spike_times()
        test_ar_background()
        test_edf_roundtrip(tmp)
        test_edf_affine(tmp)
        test_edf_rejects_bad_input(tmp)
        test_truth_alignment()
        test_cfg_hash_changes()
        test_fixed_peak_and_amplitude_spread()
        test_channel_snr_spread_is_preserved()
        test_synthesise_small()
        print("ALL PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
