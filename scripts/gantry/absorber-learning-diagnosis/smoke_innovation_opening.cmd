@echo off
rem Pre-flight smoke runs of the server script, both array arms, end to end on the CPU (detached).
rem THESIS_SMOKE=1: 2 s of every record, 4 updates, one validation, 2-iteration polish, diagnostics, saving.
rem INNOV_OPEN=3 so the opening ends (and the optimizer factor switches) inside the 4 smoke updates.
rem Pass = each log ends with "=== EXIT 0 ===" and contains "[innovation opening] phase ended".
cd /d "%~dp0"
set THESIS_NOISE=noisy
set THESIS_START=detuned
set THESIS_NA=2
set THESIS_SEED=1
set THESIS_NF=0.1
set THESIS_SMOKE=1
set THESIS_DEVICE=cpu
set INNOV_OPEN=3
set INNOV_M=40
set INNOV_FACTOR=100
set THESIS_ARM=u
call run.cmd gantry_innovation_opening.py > outputs\smoke_innov_u.log 2>&1
set THESIS_ARM=obc
call run.cmd gantry_innovation_opening.py > outputs\smoke_innov_obc.log 2>&1
echo SMOKE DONE > outputs\smoke_innov.done
