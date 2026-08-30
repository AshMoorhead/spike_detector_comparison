@echo off
REM Dump windowed_artefact_detector features (gradRatio, dynR, pStim) for the three labelled
REM baselines. QC_FEATURES=<path> makes compare_spikes write the features and exit before any
REM detector runs, so this is minutes rather than hours.
set PY=.venv\Scripts\python.exe
for %%R in (P1_pre P5_pre P8_ANT145_pre) do (
  echo === %%R ===
  cmd /c "set RECORDING=%%R&& set QC_PROFILE=finalv2&& set QC_FEATURES=runs\qcfeat_%%R.npz&& %PY% -m sdc.detect.compare_spikes"
)
echo === QC FEATURES DONE ===
