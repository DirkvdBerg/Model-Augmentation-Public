@echo off
rem Cubic-absorber seeds 2 and 3: PS2 (fall-back on) and plain S-DP each, then paired held-out evaluation.
cd /d "%~dp0"
set P_DATA=cubic
set P_HEAD=mlp
set P_CAL=stratified
set P_FALLBACK=1
for %%S in (2 3) do (
  set P_SEED=%%S
  set P_MODE=cand
  call ..\run.cmd overnight-clean-method\ps2.py 600 q%%S > outputs\ps2_q%%S.log 2>&1
  cd /d "%~dp0"
  set P_MODE=control
  call ..\run.cmd overnight-clean-method\ps2.py 600 k%%S > outputs\ps2_k%%S.log 2>&1
  cd /d "%~dp0"
)
call ..\run.cmd overnight-clean-method\evalc.py ps2_q2.pt ps2_k2.pt ps2_q3.pt ps2_k3.pt ps2_q2.pt=ps2_k2.pt ps2_q3.pt=ps2_k3.pt > outputs\evalc_cubic_seeds.log 2>&1
cd /d "%~dp0"
echo CHAIN DONE > outputs\chain_cubic_seeds.done
