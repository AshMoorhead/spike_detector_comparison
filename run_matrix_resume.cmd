@echo off
REM Resume the v2 matrix. Two things had to be fixed first, both the same failure mode:
REM   1. qc_features per-window dumps are cached on filename only, so 226-channel derived-montage
REM      baselines were reused for a 164-channel run. compare_spikes then refused every relative
REM      kStim condition on a stim recording. Cache cleared; a channel-count guard added.
REM   2. 11 run npz files from the derived-montage matrix still sit under the same names.
REM      resume_matrix.py now treats a file as complete only if it has 164 channels.
set PY=.venv\Scripts\python.exe

echo ############ regenerating relative-kStim baselines ############
%PY% -c "from sdc.artefact.qc_features import dump; dump('P1_pre', freq=145)"
%PY% -c "from sdc.artefact.qc_features import dump; dump('P1_ANT2_pre', freq=2)"

echo ############ remaining matrix runs ############
for /f "tokens=1,2,3" %%a in ('%PY% resume_matrix.py') do (
  echo === %%a / %%b  [delphos=%%c] ===
  cmd /c "set RECORDING=%%a&& set QC_PROFILE=%%b&& set RUN_DELPHOS=%%c&& %PY% -m sdc.detect.run_windows"
)
echo === MATRIX RESUME DONE ===
