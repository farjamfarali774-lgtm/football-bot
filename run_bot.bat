@echo off
rem Runs the Telegram bot forever, restarting it if it ever crashes.
rem Logs go to logs\bot.log
setlocal
cd /d "%~dp0"
if not exist logs mkdir logs

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY set "PY=C:\Users\farjam\AppData\Local\Python\pythoncore-3.14-64\python.exe"

if not exist ".env" (
  echo No .env found - run setup_local.bat first.
  exit /b 1
)

:loop
echo. >> logs\bot.log
echo [%date% %time%] starting bot >> logs\bot.log
"%PY%" bot.py >> logs\bot.log 2>&1
echo [%date% %time%] bot stopped, restarting in 10s >> logs\bot.log
timeout /t 10 /nobreak >nul
goto loop
