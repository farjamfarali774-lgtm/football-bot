@echo off
rem Undo install_startup.bat: remove the login entry and stop the bot.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_startup.ps1" -Uninstall
echo.
pause
