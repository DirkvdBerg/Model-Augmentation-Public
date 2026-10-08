@echo off
rem After the cubic test chain: D4 rollout consistency on the linear-set and cubic-set checkpoints (no training).
cd /d "%~dp0"
:wait
if not exist outputs\chain_cubic_test.done (timeout /t 60 /nobreak >nul & goto wait)
call ..\run.cmd overnight-clean-method\d4_rollout_consistency.py ps2_p1b.pt u_u1ctl600.pt ps2_p2.pt ps2_c2.pt ps2_p3.pt ps2_c3.pt > outputs\d4_linear.log 2>&1
cd /d "%~dp0"
set P_DATA=cubic
call ..\run.cmd overnight-clean-method\d4_rollout_consistency.py ps2_q0.pt ps2_k0.pt ps2_q2.pt ps2_k2.pt ps2_q3.pt ps2_k3.pt > outputs\d4_cubic.log 2>&1
cd /d "%~dp0"
echo CHAIN DONE > outputs\chain_d4.done
