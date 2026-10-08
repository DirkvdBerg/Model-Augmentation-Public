@echo off
rem PS2 chain 1b: P1b (candidate, seed 0, MLP head, stratified calibration), then held-out evaluation against C0.
cd /d "%~dp0"
set P_SEED=0
set P_MODE=cand
set P_HEAD=mlp
set P_CAL=stratified
call ..\run.cmd overnight-clean-method\ps2.py 600 p1b > outputs\ps2_p1b.log 2>&1
cd /d "%~dp0"
call ..\run.cmd overnight-clean-method\evalc.py ps2_p1b.pt u_u1ctl600.pt ps2_p1b.pt=u_u1ctl600.pt > outputs\evalc_p1b.log 2>&1
cd /d "%~dp0"
echo CHAIN DONE > outputs\chain_p1b.done
