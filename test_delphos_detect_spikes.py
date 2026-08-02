"""
test_delphos_detect_spikes.py
-----------------------------
Checks everything in delphos_detect_spikes.py that does NOT need the Delphos exe: the
.mat parse, label normalisation, chunk merging, the on-disk cache, and the mapping onto
pipeline channels/sample indices. The CLI call itself is stubbed out.

Worth keeping as a tool: Delphos costs ~5 min per real call and needs MATLAB Runtime 9.5,
so this is the only fast way to tell "my parsing/merging is broken" from "Delphos moved".

    .venv\\Scripts\\python.exe test_delphos_detect_spikes.py
"""
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import savemat

import delphos_detect_spikes as dd

FS = 400.0


def _write_results_mat(path, labels, markers):
    """Write a `results` struct shaped like Delphos's own output.

    markers: list of (label, position_sec, [channel labels]) -- field names are fixed by the
    AnyWave mex file (Readme_Delphos_cmd_line.txt), so keep all of them present."""
    rec = np.zeros((len(markers),), dtype=[("label", object), ("value", object),
                                           ("position", object), ("duration", object),
                                           ("channels", object), ("color", object)])
    for i, (lab, pos, chans) in enumerate(markers):
        rec[i]["label"] = lab
        rec[i]["value"] = 0.0
        rec[i]["position"] = float(pos)
        rec[i]["duration"] = 0.0
        rec[i]["channels"] = np.array(chans, dtype=object).reshape(1, -1)
        rec[i]["color"] = "#f05523"
    savemat(path, {"results": {"markers": rec,
                               "labels": np.array(labels, dtype=object).reshape(-1, 1),
                               "n_Spk": sum(m[0] == "Spike" for m in markers)}})


def test_normalise_label():
    assert dd.normalise_label("C'1-C'2") == "C_1_C_2"      # Delphos -> pipeline space
    assert dd.normalise_label("B1-B2") == "B1_B2"
    assert dd.normalise_label("C’1-C’2") == "C_1_C_2"   # curly apostrophe
    print("ok  normalise_label")


def test_read_mat(tmp):
    mat = tmp / "x_delphos.mat"
    _write_results_mat(
        mat,
        labels=["B1-B2", "C'1-C'2", "H1-H2"],
        markers=[("Spike", 1.25, ["B1-B2"]),
                 ("Spike", 2.50, ["C'1-C'2", "B1-B2"]),   # one marker, two channels
                 ("Ripple", 3.00, ["B1-B2"]),             # dropped: not a Spike
                 ("Spike", 1.25, ["B1-B2"])],             # duplicate -> collapsed
    )
    spikes, chans = dd.read_delphos_mat(mat)
    assert chans == ["B1_B2", "C_1_C_2", "H1_H2"], chans
    assert np.allclose(spikes["B1_B2"], [1.25, 2.50]), spikes["B1_B2"]
    assert np.allclose(spikes["C_1_C_2"], [2.50]), spikes["C_1_C_2"]
    assert "H1_H2" not in spikes                          # present as a channel, no spikes
    print("ok  read_delphos_mat (Spike-only, multi-channel markers, dedup)")


def test_single_marker_mat(tmp):
    """squeeze_me collapses a 1-element struct array to a scalar -- the classic parse trap."""
    mat = tmp / "one_delphos.mat"
    _write_results_mat(mat, labels=["B1-B2"], markers=[("Spike", 0.5, ["B1-B2"])])
    spikes, chans = dd.read_delphos_mat(mat)
    assert chans == ["B1_B2"] and np.allclose(spikes["B1_B2"], [0.5])
    print("ok  read_delphos_mat (single marker / single channel)")


def test_to_indices():
    """Absolute seconds -> 0-based sample indices at FS, relative to the window start."""
    times = {"B1_B2": np.array([10.0, 10.5]), "C_1_C_2": np.array([12.0])}
    chans = ["B1_B2", "C_1_C_2", "H1_H2"]
    out = dd._to_indices(times, chans, ["B1_B2", "H1_H2", "Z9_Z10"], FS, 10.0, print)
    assert np.array_equal(out[0], [0, 200]), out[0]        # 10.0 s -> 0, 10.5 s -> 200
    assert out[1].size == 0 and out[2].size == 0           # no spikes / not a Delphos pair
    print("ok  _to_indices (window offset, rate conversion, unmatched channels)")


def test_chunk_merge_and_cache(tmp, monkey_edf):
    """Chunked run: absolute positions merge by plain concatenation, and the cache round-trips.
    The CLI is stubbed, so this exercises our plumbing only."""
    calls = []

    def fake_chunk(edf_path, start_sec, dur_sec, cfg, log):
        calls.append((start_sec, dur_sec))
        # one spike 0.5 s into each chunk, on the same channel, in ABSOLUTE file time
        return {"B1_B2": np.array([start_sec + 0.5])}, ["B1_B2", "H1_H2"]

    real, dd._run_one_chunk = dd._run_one_chunk, fake_chunk
    try:
        kw = dict(fs=FS, start_sec=0.0, duration_sec=60.0, chunk_sec=25.0,
                  cache_dir=tmp / "cache", log=lambda s: None)
        out = dd.detect_spikes(monkey_edf, ["B1_B2", "H1_H2"], **kw)
        assert calls == [(0.0, 25.0), (25.0, 25.0), (50.0, 10.0)], calls
        assert np.array_equal(out[0], [200, 10200, 20200]), out[0]   # 0.5, 25.5, 50.5 s

        calls.clear()
        cached = dd.detect_spikes(monkey_edf, ["B1_B2", "H1_H2"], **kw)
        assert calls == [], "cache hit should not re-run Delphos"
        assert np.array_equal(cached[0], out[0]) and cached[1].size == 0
    finally:
        dd._run_one_chunk = real
    print("ok  chunk merge + on-disk cache")


def test_runtime_env(tmp):
    """Every installed runtime's runtime\\win64 goes on the CHILD's PATH, so a shell opened
    before the runtime was installed still works."""
    root = tmp / "mcr"
    for v in ("R2022b", "v95"):
        (root / v / "runtime" / "win64").mkdir(parents=True)
    env = dd._runtime_env({"mcr_root": root}, print)
    head = env["PATH"].split(os.pathsep)[:2]
    assert head == [str(root / "R2022b" / "runtime" / "win64"),
                    str(root / "v95" / "runtime" / "win64")], head
    assert dd._runtime_env({"mcr_root": tmp / "nope"}, print)["PATH"] == os.environ["PATH"]
    print("ok  _runtime_env (PATH injection, missing root)")


def test_missing_runtime_message(tmp, monkey_edf):
    """A silent non-zero exit is the missing-MATLAB-Runtime signature; say so."""
    class _P:
        returncode, stdout, stderr = -1, "", ""

    real, dd.subprocess.run = dd.subprocess.run, lambda *a, **k: _P()
    try:
        dd.detect_spikes(monkey_edf, ["B1_B2"], fs=FS, start_sec=0.0, duration_sec=5.0,
                         pin_free_ram_gb=0, log=lambda s: None)
    except RuntimeError as e:
        assert "MATLAB Runtime" in str(e), e
        print("ok  missing-runtime error message")
    else:
        raise AssertionError("expected a RuntimeError")
    finally:
        dd.subprocess.run = real


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="delphos_test_"))
    edf = tmp / "fake.edf"           # existence is all detect_spikes checks before running
    edf.write_bytes(b"")
    try:
        test_normalise_label()
        test_read_mat(tmp)
        test_single_marker_mat(tmp)
        test_to_indices()
        test_chunk_merge_and_cache(tmp, edf)
        test_runtime_env(tmp)
        test_missing_runtime_message(tmp, edf)
        print("\nAll checks passed (the Delphos CLI itself is NOT exercised here).")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
