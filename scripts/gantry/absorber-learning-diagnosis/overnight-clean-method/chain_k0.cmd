@echo off
rem Cubic-absorber Gate 1: plain S-DP seed 0 (K0) on the cubic test set, then its held-out evaluation.
cd /d "%~dp0"
set P_DATA=cubic
set P_SEED=0
set P_MODE=control
set P_CAL=stratified
call ..\run.cmd overnight-clean-method\ps2.py 600 k0 > outputs\ps2_k0.log 2>&1
cd /d "%~dp0"
call ..\run.cmd overnight-clean-method\evalc.py ps2_k0.pt > outputs\evalc_k0.log 2>&1
cd /d "%~dp0"
echo CHAIN DONE > outputs\chain_k0.done
