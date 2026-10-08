@echo off
rem PS2 overnight chain 1: P1 (candidate, seed 0, MLP head), then held-out evaluation against C0 (u_u1ctl600).
cd /d "%~dp0"
set P_SEED=0
set P_MODE=cand
set P_HEAD=mlp
call ..\run.cmd overnight-clean-method\ps2.py 600 p1 > outputs\ps2_p1.log 2>&1
cd /d "%~dp0"
call ..\run.cmd overnight-clean-method\evalc.py ps2_p1.pt u_u1ctl600.pt ps2_p1.pt=u_u1ctl600.pt > outputs\evalc_p1.log 2>&1
cd /d "%~dp0"
echo CHAIN DONE > outputs\chain_p1.done
