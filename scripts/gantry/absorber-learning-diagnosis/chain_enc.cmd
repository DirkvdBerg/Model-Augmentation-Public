@echo off
rem After chain_steal.cmd: encoder probe E2, then steal checks on E2 and (fixed combinations line) on E1.
cd /d "%~dp0"
:wait
if not exist outputs\chain_steal.done (timeout /t 30 /nobreak >nul & goto wait)
set U_LR_A=1e-3
set F_EE_ROWS=added
set F_LR_ENC_A=1e-3
call run.cmd f_eeoe.py 600 64 enc > outputs\f_enc.log 2>&1
set F_EE_ROWS=
set F_LR_ENC_A=
call run.cmd g2_steal.py f_enc.pt > outputs\g2_enc.log 2>&1
call run.cmd g2_steal.py f_eeoe.pt > outputs\g2_eeoe_v2.log 2>&1
echo CHAIN DONE > outputs\chain_enc.done
