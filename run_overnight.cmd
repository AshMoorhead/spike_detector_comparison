@echo off
REM Resumable overnight sweeps. Each job checkpoints per setting to
REM runs\sweeps\overnight_<job>.jsonl and skips what is already there, so re-running this
REM file after a crash resumes rather than restarting. Jobs are independent: a failure in one
REM does not stop the others (no && chaining, and errorlevel is not checked deliberately).
REM Cheapest and most informative first, so an overnight that dies early still yields the grids.
set PY=.venv\Scripts\python.exe

echo === janca  k1 x k3 x band  (72) ===
%PY% -m sdc.scoring.overnight janca

echo === barkmeier  std_coeff x TAMP x trough x band  (150) ===
%PY% -m sdc.scoring.overnight barkmeier

echo === barkmeier shape gates  slope x dur x trough x TAMP  (48) ===
%PY% -m sdc.scoring.overnight barkmeier_shape

echo === P1 ANT 145 Hz per-channel rates (janca, barkmeier, delphos) ===
%PY% -m sdc.scoring.overnight ant145

echo === ALL JOBS ATTEMPTED ===
