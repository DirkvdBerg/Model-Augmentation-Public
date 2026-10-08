@echo off
rem Detached launcher for one probe: run.cmd <script.py> [args...]; env switches are inherited.
rem Launched with Start-Process so a Claude session exit does not kill it.
set E=C:\Users\20203253\AppData\Local\anaconda3\envs\GraduationProject
set PATH=%E%;%E%\Library\mingw-w64\bin;%E%\Library\usr\bin;%E%\Library\bin;%E%\Scripts;%PATH%
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
"%E%\python.exe" -u %*
echo === EXIT %ERRORLEVEL% ===
