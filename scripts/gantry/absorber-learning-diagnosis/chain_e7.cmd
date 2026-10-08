@echo off
rem Overnight chain 3 (after chain 2): E7 = E5 continued on the pure thesis loss; steal + AC1.
cd /d "%~dp0"
:wait
if not exist outputs\chain_e6.done (timeout /t 30 /nobreak >nul & goto wait)
set F_START=f2_e5.pt
set F_HOLD=0
set F_END=0
call run.cmd f2_open.py 600 64 e7 > outputs\f2_e7.log 2>&1
set F_START=
set F_HOLD=
set F_END=
call run.cmd g2_steal.py f2_e7.pt > outputs\g2_e7.log 2>&1
call run.cmd ac_band.py f2_e7.pt > outputs\ac_band_e7.log 2>&1
echo CHAIN DONE > outputs\chain_e7.done
