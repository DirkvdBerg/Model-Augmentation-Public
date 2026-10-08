@echo off
rem Overnight chain 2: AC1 band check on control, E3, E4, E5; then E6 (E5, seed 2); steal + AC1 on E6.
cd /d "%~dp0"
call run.cmd ac_band.py u_u1ctl600.pt f_inn.pt f2_e4.pt f2_e5.pt > outputs\ac_band1.log 2>&1
set F_EE_FROM=300
set F_HOLD=600
set F_END=750
set F_SEED=2
call run.cmd f2_open.py 900 64 e6 > outputs\f2_e6.log 2>&1
set F_EE_FROM=
set F_HOLD=
set F_END=
set F_SEED=
call run.cmd g2_steal.py f2_e6.pt > outputs\g2_e6.log 2>&1
call run.cmd ac_band.py f2_e6.pt > outputs\ac_band_e6.log 2>&1
echo CHAIN DONE > outputs\chain_e6.done
