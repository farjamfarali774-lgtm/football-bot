@echo off
rem One-click wrapper: makes the bot start at every Windows login (no admin needed).
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_startup.ps1"
echo.
pause
