@echo off
rem Sequential jobs after E1, one at a time, each to completion (detached via Start-Process).
cd /d "%~dp0"
set U_LR_A=1e-3
call run.cmd g2_steal.py f_eeoe.pt > outputs\g2_eeoe.log 2>&1
call run.cmd u_probe.py 600 64 u1ctl600 > outputs\u_u1ctl600.log 2>&1
call run.cmd g2_steal.py u_u1ctl600.pt > outputs\g2_u1ctl600.log 2>&1
echo CHAIN DONE > outputs\chain_steal.done
