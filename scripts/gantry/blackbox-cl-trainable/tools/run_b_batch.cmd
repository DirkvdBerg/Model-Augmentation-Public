@echo off
rem Option B runs b2 to b6 (DECISIONS.md BB2-018), one after another, detached from the Claude
rem session so a session exit does not kill them. Log: outputs\b\batch_r3.log
set E=C:\Users\20203253\AppData\Local\anaconda3\envs\GraduationProject
set PATH=%E%;%E%\Library\mingw-w64\bin;%E%\Library\usr\bin;%E%\Library\bin;%E%\Scripts;%PATH%
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
cd /d "%~dp0.."
echo === RUN noisy seed 2 ===
"%E%\python.exe" -u train_bb_clmap.py --version noisy --seed 2 --epochs 10 --out outputs\b\noisy_s2_r3
echo === RUN noisy seed 3 ===
"%E%\python.exe" -u train_bb_clmap.py --version noisy --seed 3 --epochs 10 --out outputs\b\noisy_s3_r3
echo === RUN noisefree seed 1 ===
"%E%\python.exe" -u train_bb_clmap.py --version noisefree --seed 1 --epochs 10 --out outputs\b\noisefree_s1_r3
echo === RUN noisefree seed 2 ===
"%E%\python.exe" -u train_bb_clmap.py --version noisefree --seed 2 --epochs 10 --out outputs\b\noisefree_s2_r3
echo === RUN noisefree seed 3 ===
"%E%\python.exe" -u train_bb_clmap.py --version noisefree --seed 3 --epochs 10 --out outputs\b\noisefree_s3_r3
echo === BATCH DONE ===
