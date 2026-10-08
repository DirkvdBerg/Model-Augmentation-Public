@echo off
rem Gate D chain: waits for the gantry seeds chain, then the nonlinear benchmark (9 runs).
cd /d "%~dp0"
:wait
if not exist outputs\chain_seeds.done (timeout /t 60 /nobreak >nul & goto wait)
call ..\run.cmd overnight-clean-method\gate_d.py 800 > outputs\gate_d.log 2>&1
cd /d "%~dp0"
echo CHAIN DONE > outputs\chain_gate_d.done
