# Makes the bot start at every Windows login, without admin rights.
#
#     powershell -ExecutionPolicy Bypass -File install_startup.ps1
#     powershell -ExecutionPolicy Bypass -File install_startup.ps1 -Uninstall
#
# It writes a per-user Run entry (HKCU), so no elevation is needed. The bot runs
# through run_bot_hidden.vbs, so no console window stays open.

param([switch]$Uninstall)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$name = "FootballBot"

if ($Uninstall) {
    Remove-ItemProperty -Path $key -Name $name -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*bot.py*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Host "Removed the login entry and stopped the bot."
    exit 0
}

$vbs = Join-Path $here "run_bot_hidden.vbs"
if (-not (Test-Path $vbs)) {
    Write-Host "run_bot_hidden.vbs is missing next to this script."
    exit 1
}

$value = "wscript.exe `"$vbs`""
New-ItemProperty -Path $key -Name $name -Value $value -PropertyType String -Force | Out-Null
Write-Host "Registered login entry:"
Write-Host "  $value"
Write-Host ""
Write-Host "Starting the bot now (it runs in the background, nothing will appear)."

if (-not (Test-Path (Join-Path $here ".env"))) {
    Write-Host ""
    Write-Host "NOTE: .env is missing, so the bot cannot start yet."
    Write-Host "      Run setup_local.bat once to enter the token."
    exit 0
}

Start-Process -FilePath "wscript.exe" -ArgumentList "`"$vbs`"" -WindowStyle Hidden
Write-Host "Started. Logs: logs\bot.log"
