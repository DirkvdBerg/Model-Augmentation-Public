@echo off
rem One train_bb_clmap.py run, detached from the Claude session; all arguments passed through.
rem   start: Start-Process cmd.exe -ArgumentList '/c ""tools\run_one.cmd" --version ... > "<log>" 2>&1"'
set E=C:\Users\20203253\AppData\Local\anaconda3\envs\GraduationProject
set PATH=%E%;%E%\Library\mingw-w64\bin;%E%\Library\usr\bin;%E%\Library\bin;%E%\Scripts;%PATH%
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
cd /d "%~dp0.."
echo === RUN %* ===
"%E%\python.exe" -u train_bb_clmap.py %*
echo === RUN DONE (exit %ERRORLEVEL%) ===
