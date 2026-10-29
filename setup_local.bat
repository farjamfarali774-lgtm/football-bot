@echo off
rem Start setup_local.py so you can type the token (it is hidden while typing).
setlocal
cd /d "%~dp0"
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY set "PY=C:\Users\farjam\AppData\Local\Python\pythoncore-3.14-64\python.exe"
"%PY%" setup_local.py
echo.
pause
