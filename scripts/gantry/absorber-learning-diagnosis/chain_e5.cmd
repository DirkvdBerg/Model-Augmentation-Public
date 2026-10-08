@echo off
rem Overnight chain: steal check on E4, E5 (ordered opening), steal check on E5.
cd /d "%~dp0"
call run.cmd g2_steal.py f2_e4.pt > outputs\g2_e4.log 2>&1
set F_EE_FROM=300
set F_HOLD=600
set F_END=750
call run.cmd f2_open.py 900 64 e5 > outputs\f2_e5.log 2>&1
set F_EE_FROM=
set F_HOLD=
set F_END=
call run.cmd g2_steal.py f2_e5.pt > outputs\g2_e5.log 2>&1
echo CHAIN DONE > outputs\chain_e5.done
