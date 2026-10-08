@echo off
rem PS2 chain 2: paired seeds 2 and 3 (candidate and control each), then held-out evaluation of the pairs.
cd /d "%~dp0"
set P_HEAD=mlp
set P_CAL=stratified
set P_SEED=2
set P_MODE=cand
call ..\run.cmd overnight-clean-method\ps2.py 600 p2 > outputs\ps2_p2.log 2>&1
cd /d "%~dp0"
set P_MODE=control
call ..\run.cmd overnight-clean-method\ps2.py 600 c2 > outputs\ps2_c2.log 2>&1
cd /d "%~dp0"
set P_SEED=3
set P_MODE=cand
call ..\run.cmd overnight-clean-method\ps2.py 600 p3 > outputs\ps2_p3.log 2>&1
cd /d "%~dp0"
set P_MODE=control
call ..\run.cmd overnight-clean-method\ps2.py 600 c3 > outputs\ps2_c3.log 2>&1
cd /d "%~dp0"
call ..\run.cmd overnight-clean-method\evalc.py ps2_p2.pt ps2_c2.pt ps2_p3.pt ps2_c3.pt ps2_p2.pt=ps2_c2.pt ps2_p3.pt=ps2_c3.pt > outputs\evalc_seeds.log 2>&1
cd /d "%~dp0"
echo CHAIN DONE > outputs\chain_seeds.done
