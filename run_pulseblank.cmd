@echo off
REM Pulse blanking on the 2 Hz recording, against a dynR-only reference.
REM
REM QC is dynR ONLY (no kStim, no grad, no dilation) so the rows differ by BLANKING alone --
REM while still removing flat epochs, where a spike is physically impossible and which account
REM for 18-24% of Janca's baseline detections.
REM
REM interp fill only. `auto` fills anything over 4 samples with AR-synthesised noise, and 5 ms
REM at 2 kHz is 10 samples -- that variant sent Delphos from 1.88 to 10.69 by detecting its own
REM repair, because a synthesised patch is an edge and Delphos's whitened plane finds edges.
REM
REM Blanking is a NO-OP on the pre file (`if PULSE_BLANK_MS and TRIAL is not None`), so the
REM three baselines must come out identical -- a free check, and it lets the report pair rows
REM by filename without a special case.
set PY=.venv\Scripts\python.exe

for %%R in (P1_ANT2_stim P1_ANT2_pre) do (
  echo === %%R / dynr, no blanking ===
  cmd /c "set RECORDING=%%R&& set QC_PROFILE=dynr&& set RUN_DELPHOS=1&& %PY% -m sdc.detect.run_windows"
  for %%B in (5 15) do (
    echo === %%R / dynr + pb%%Bi ===
    cmd /c "set RECORDING=%%R&& set QC_PROFILE=dynr&& set PULSE_BLANK_MS=%%B&& set PULSE_FILL=interp&& set RUN_DELPHOS=1&& %PY% -m sdc.detect.run_windows"
  )
)
echo === PULSE BLANKING DONE ===
