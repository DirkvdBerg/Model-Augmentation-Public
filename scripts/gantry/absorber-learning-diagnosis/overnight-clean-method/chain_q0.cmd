@echo off
rem Cubic-absorber Q0: PS2 seed 0 (fall-back rule on), then held-out evaluation paired with K0.
cd /d "%~dp0"
set P_DATA=cubic
set P_SEED=0
set P_MODE=cand
set P_HEAD=mlp
set P_CAL=stratified
set P_FALLBACK=1
call ..\run.cmd overnight-clean-method\ps2.py 600 q0 > outputs\ps2_q0.log 2>&1
cd /d "%~dp0"
call ..\run.cmd overnight-clean-method\evalc.py ps2_q0.pt ps2_k0.pt ps2_q0.pt=ps2_k0.pt > outputs\evalc_q0.log 2>&1
cd /d "%~dp0"
echo CHAIN DONE > outputs\chain_q0.done
