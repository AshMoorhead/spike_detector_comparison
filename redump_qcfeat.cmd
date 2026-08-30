@echo off
REM Regenerate the relative-kStim baselines under the CLINICAL montage. The stored dumps hold
REM 226 derived channel names; compare_spikes compares them against this run's 164 and refuses
REM to align -- correctly, but it means every kStim condition on a STIM recording failed.
REM Baselines were unaffected because kStim is dropped without a trial.
set PY=.venv\Scripts\python.exe
echo === P1_pre at 145 Hz ===
%PY% -c "from sdc.artefact.qc_features import dump; dump('P1_pre', freq=145)"
echo === P1_ANT2_pre at 2 Hz ===
%PY% -c "from sdc.artefact.qc_features import dump; dump('P1_ANT2_pre', freq=2)"
echo === QC FEATURE DUMPS DONE ===
