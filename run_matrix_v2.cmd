@echo off
REM Artefact matrix v2 -- CLINICAL montage (164 pairs, not the derived 226). Every recording is
REM run against its paired baseline, so the 145 Hz file reads as stim-vs-baseline rather than
REM ON-vs-OFF (its OFF periods sit between stim blocks and may carry carryover).
REM
REM ALL PREVIOUS RUNS USED THE DERIVED MONTAGE AND ARE SUPERSEDED.
REM
REM WHAT IS ACTIVE ON A BASELINE FILE: only gradThr and dynR. kStim is dropped (no trial, so no
REM frequency to measure a threshold at), periodicity needs have_stim, and stim-dilation needs
REM stim bins. So k150g0 == k450g0 == e1 == e1t10 on any pre file -- all four reduce to dynR
REM alone. They are still run under their own names so every stim condition has a same-named
REM partner for the ratio, and the equality is a free check that the drop paths work.
REM
REM TIERS, fast first, so a short night still leaves a complete Janca/Barkmeier matrix.
REM   tier 1   5 conditions x 4 recordings = 20 runs, Janca + Barkmeier      (~1.5 h)
REM   tier 2   6 conditions x 4 recordings = 24 runs, + Delphos              (~3.5 h)
REM   tier 3   e1t10 on the 2 Hz pair only = 2 runs, + Delphos               (~0.3 h)
REM Total 46 runs, ~5.3 h against an 8 h window.
REM
REM e1t10 is 2 Hz only: periodicity is inert above 10 Hz and on baselines, so at 145 Hz it would
REM reproduce e1 exactly and waste two Delphos runs to prove nothing.

set PY=.venv\Scripts\python.exe

echo ############ TIER 1: Janca + Barkmeier ############
for %%R in (P1_stim P1_pre P1_ANT2_stim P1_ANT2_pre) do (
  for %%P in (mnebads10 mnebads15 mnebads75 mnebads150 k450g0) do (
    echo === %%R / %%P  [JB] ===
    cmd /c "set RECORDING=%%R&& set QC_PROFILE=%%P&& set RUN_DELPHOS=0&& %PY% -m sdc.detect.run_windows"
  )
)

echo ############ TIER 2: + Delphos ############
for %%R in (P1_stim P1_pre P1_ANT2_stim P1_ANT2_pre) do (
  for %%P in (none k150g150 k150g1000 k450g1000 k150g0 e1) do (
    echo === %%R / %%P  [JBD] ===
    cmd /c "set RECORDING=%%R&& set QC_PROFILE=%%P&& set RUN_DELPHOS=1&& %PY% -m sdc.detect.run_windows"
  )
)

echo ############ TIER 3: second periodicity rung, 2 Hz only ############
for %%R in (P1_ANT2_stim P1_ANT2_pre) do (
  echo === %%R / e1t10  [JBD] ===
  cmd /c "set RECORDING=%%R&& set QC_PROFILE=e1t10&& set RUN_DELPHOS=1&& %PY% -m sdc.detect.run_windows"
)
echo === MATRIX V2 DONE ===
