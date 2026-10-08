@echo off
rem Proof runs p1-p5 (DECISIONS.md BB2-026), one after another, detached. Log: outputs\p\proof.log
set E=C:\Users\20203253\AppData\Local\anaconda3\envs\GraduationProject
set PATH=%E%;%E%\Library\mingw-w64\bin;%E%\Library\usr\bin;%E%\Library\bin;%E%\Scripts;%PATH%
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
cd /d "%~dp0.."
set A=--epochs 12 --target e --in64 --dr --lr 1e-2 --its-per-val 399 --batch 256 --stride 25
echo === RUN noisy seed 2 ===
"%E%\python.exe" -u train_bb_clmap.py --version noisy --seed 2 %A% --out outputs\p\noisy_s2
echo === RUN noisy seed 3 ===
"%E%\python.exe" -u train_bb_clmap.py --version noisy --seed 3 %A% --out outputs\p\noisy_s3
echo === RUN noisefree seed 1 ===
"%E%\python.exe" -u train_bb_clmap.py --version noisefree --seed 1 %A% --out outputs\p\noisefree_s1
echo === RUN noisefree seed 2 ===
"%E%\python.exe" -u train_bb_clmap.py --version noisefree --seed 2 %A% --out outputs\p\noisefree_s2
echo === RUN noisefree seed 3 ===
"%E%\python.exe" -u train_bb_clmap.py --version noisefree --seed 3 %A% --out outputs\p\noisefree_s3
echo === BATCH DONE ===
