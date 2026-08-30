@echo off
REM Pulse blanking PLUS a loose gradient rule on the 2 Hz recording.
REM
REM Blanking reaches the pulses it can, but the largest excursions survive it -- visible in the
REM raw trace. gradThr=1000 catches those: it is inert at 145 Hz (within 0.4% of off) yet
REM removes 37% of Janca's detections at 2 Hz, because only the low-frequency pulses are large
REM enough to clear it.
REM
REM Reference row is dynrg1000 with no blanking, so the blanking rows are attributable.
set PY=.venv\Scripts\python.exe

for %%R in (P1_ANT2_stim P1_ANT2_pre) do (
  echo === %%R / dynrg1000, no blanking ===
  cmd /c "set RECORDING=%%R&& set QC_PROFILE=dynrg1000&& set RUN_DELPHOS=1&& %PY% -m sdc.detect.run_windows"
  for %%B in (5 15) do (
    echo === %%R / dynrg1000 + pb%%Bi ===
    cmd /c "set RECORDING=%%R&& set QC_PROFILE=dynrg1000&& set PULSE_BLANK_MS=%%B&& set PULSE_FILL=interp&& set RUN_DELPHOS=1&& %PY% -m sdc.detect.run_windows"
  )
)
echo === BLANK+GRAD DONE ===
