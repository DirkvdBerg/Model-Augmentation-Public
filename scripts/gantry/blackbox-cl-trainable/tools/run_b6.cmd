@echo off
rem Option B run b6 (noise-free, seed 3), rerun after b6 in batch_r3 was terminated externally
rem (no traceback) at update 316. Detached; log: outputs\b\b6_r4.log
set E=C:\Users\20203253\AppData\Local\anaconda3\envs\GraduationProject
set PATH=%E%;%E%\Library\mingw-w64\bin;%E%\Library\usr\bin;%E%\Library\bin;%E%\Scripts;%PATH%
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
cd /d "%~dp0.."
echo === RUN noisefree seed 3 ===
"%E%\python.exe" -u train_bb_clmap.py --version noisefree --seed 3 --epochs 10 --out outputs\b\noisefree_s3_r4
echo === EXIT CODE %ERRORLEVEL% ===
