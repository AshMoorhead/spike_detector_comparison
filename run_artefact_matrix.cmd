@echo off
REM Phase 2 of plans/polymorphic-wiggling-breeze.md: artefact-handling comparison.
REM
REM 7 conditions x 2 recordings = 14 runs, Janca + Barkmeier only (RUN_DELPHOS=0).
REM   A            none          no artefact handling at all -- the control
REM   B1/B2        mnebads75     MNE annotate_amplitude, whole-channel rejection, peak in
REM                mnebads150    {75, 150} uV/sample at min_duration=2 ms. The list is computed
REM                              ONCE per recording by sdc.artefact.mne_bads_check.dump and read
REM                              here -- a bad channel is a property of the recording, not of a
REM                              240 s window. Run that dump before this file.
REM   C1..C4       k{150,450}g{150,1000}   the current windowed detector, 2x2 over its two
REM                              live knobs, dynR held at 3*lsb. k450g1000 IS finalv2.
REM
REM Each condition writes runs\<rec>_qc<profile>.npz (plus per-window files), so nothing
REM overwrites anything and a crash costs only the run in flight -- re-running this file
REM redoes completed conditions, so comment out what has finished if you restart.
REM
REM Delphos is off for this phase and revisited separately. Janca is dec=200 as of today,
REM so these runs are NOT comparable with the stored qcfinalv2 ones (which are dec=0).

set PY=.venv\Scripts\python.exe
set RUN_DELPHOS=0

REM 13 runs, not 14. On P1_ANT2_stim the peak=75 and peak=150 bad-channel lists are IDENTICAL
REM (the same 7 channels), so the same mask would produce the same output twice -- mnebads75 is
REM therefore run on P1_stim only, where the two differ (22 vs 16 channels). That the 2 Hz file
REM is threshold-insensitive across 75-150 and empty at 1000 is a result, not a gap.

for %%P in (none mnebads75 mnebads150 k150g150 k150g1000 k450g150 k450g1000) do (
  echo === P1_stim / %%P ===
  cmd /c "set RECORDING=P1_stim&& set QC_PROFILE=%%P&& set RUN_DELPHOS=0&& %PY% -m sdc.detect.run_windows"
)

for %%P in (none mnebads150 k150g150 k150g1000 k450g150 k450g1000) do (
  echo === P1_ANT2_stim / %%P ===
  cmd /c "set RECORDING=P1_ANT2_stim&& set QC_PROFILE=%%P&& set RUN_DELPHOS=0&& %PY% -m sdc.detect.run_windows"
)
echo === ARTEFACT MATRIX DONE ===
