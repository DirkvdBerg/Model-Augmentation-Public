@echo off
rem Overnight chain 4 (after chain 3): E8 (E5, seed 3), E9 (seed 2, innovation phase 500); steal + AC1 on each.
cd /d "%~dp0"
:wait
if not exist outputs\chain_e7.done (timeout /t 30 /nobreak >nul & goto wait)
set F_EE_FROM=300
set F_HOLD=600
set F_END=750
set F_SEED=3
call run.cmd f2_open.py 900 64 e8 > outputs\f2_e8.log 2>&1
call run.cmd g2_steal.py f2_e8.pt > outputs\g2_e8.log 2>&1
call run.cmd ac_band.py f2_e8.pt > outputs\ac_band_e8.log 2>&1
set F_EE_FROM=500
set F_HOLD=800
set F_END=950
set F_SEED=2
call run.cmd f2_open.py 1100 64 e9 > outputs\f2_e9.log 2>&1
set F_EE_FROM=
set F_HOLD=
set F_END=
set F_SEED=
call run.cmd g2_steal.py f2_e9.pt > outputs\g2_e9.log 2>&1
call run.cmd ac_band.py f2_e9.pt > outputs\ac_band_e9.log 2>&1
echo CHAIN DONE > outputs\chain_e8.done
