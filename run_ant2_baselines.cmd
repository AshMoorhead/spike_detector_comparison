@echo off
REM Matched baselines for the ANT2 half of the artefact matrix.
REM
REM P1_ANT2_stim is CONTINUOUS stimulation -- 477 ON epochs, 0 OFF -- so the within-file ON/OFF
REM ratio every other recording uses does not exist. The effect there can only be stim-file rate
REM against the paired stim-free PRE file, which means the PRE file has to be processed under
REM the SAME artefact profile or the ratio compares two different maskings.
REM
REM On a baseline `kStim` is dropped (make_cfg_artefact leaves stimHz None), so k150g150 and
REM k450g150 produce identical output, as do k150g1000 and k450g1000. Both are still run so each
REM stim condition has a same-named partner -- the few wasted minutes are cheaper than a mapping
REM that has to be remembered.
set PY=.venv\Scripts\python.exe

echo === dumping mne bad channels for the baseline ===
%PY% -c "from sdc.artefact.mne_bads_check import dump; dump(recs=('P1_ANT2_pre',), peaks=(150.0,))"

for %%P in (none mnebads150 k150g150 k150g1000 k450g150 k450g1000) do (
  echo === P1_ANT2_pre / %%P ===
  cmd /c "set RECORDING=P1_ANT2_pre&& set QC_PROFILE=%%P&& set RUN_DELPHOS=0&& %PY% -m sdc.detect.run_windows"
)
echo === ANT2 BASELINES DONE ===
