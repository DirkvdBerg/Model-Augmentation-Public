@echo off
rem Section-10 references (ref_scores.py), detached. Log: outputs\ref\ref.log
set E=C:\Users\20203253\AppData\Local\anaconda3\envs\GraduationProject
set PATH=%E%;%E%\Library\mingw-w64\bin;%E%\Library\usr\bin;%E%\Library\bin;%E%\Scripts;%PATH%
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
cd /d "%~dp0.."
"%E%\python.exe" -u ref_scores.py
echo === REF DONE ===
