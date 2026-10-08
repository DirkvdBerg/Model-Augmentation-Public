@echo off
rem After the cubic seeds chain: generate 4 cubic test records (MATLAB replica), then evaluate the 6 checkpoints on the test split.
cd /d "%~dp0"
:wait
if not exist outputs\chain_cubic_seeds.done (timeout /t 60 /nobreak >nul & goto wait)
"C:\Program Files\MATLAB\R2025a\bin\matlab.exe" -nojvm -batch "run('%~dp0..\cubic-absorber\gen_cubic_test.m'); fprintf('%%s%%s\n','@@CUBIC_','DONE@@');" -logfile "%~dp0..\cubic-absorber\gen_cubic_test_matlab.log"
cd /d "%~dp0"
set P_DATA=cubic
set E_SPLIT=test
call ..\run.cmd overnight-clean-method\evalc.py ps2_q0.pt ps2_k0.pt ps2_q2.pt ps2_k2.pt ps2_q3.pt ps2_k3.pt ps2_q0.pt=ps2_k0.pt ps2_q2.pt=ps2_k2.pt ps2_q3.pt=ps2_k3.pt > outputs\evalc_cubic_test.log 2>&1
cd /d "%~dp0"
echo CHAIN DONE > outputs\chain_cubic_test.done
