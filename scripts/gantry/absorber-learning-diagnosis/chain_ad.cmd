@echo off
rem Overnight chain 5 (after chain 4): AD1 full-record free-run validation of the opening checkpoints.
cd /d "%~dp0"
:wait
if not exist outputs\chain_e8.done (timeout /t 30 /nobreak >nul & goto wait)
call run.cmd ad_valfree.py u_u1ctl600.pt f2_e5.pt f2_e6.pt f2_e7.pt f2_e8.pt f2_e9.pt > outputs\ad_valfree.log 2>&1
echo CHAIN DONE > outputs\chain_ad.done
