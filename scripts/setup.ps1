# One-time setup on the Windows demo laptop (PowerShell, from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "Python launcher 'py' not found. Install Python 3.12 from python.org (tick 'py launcher')." }
if (-not (Test-Path .venv)) {
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
    if ($LASTEXITCODE -ne 0) { throw "Python >= 3.11 required (3.12 recommended)." }
    py -3 -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
git lfs pull
if (-not (Test-Path config\demo.toml)) {
    Copy-Item config\demo.example.toml config\demo.toml
    Write-Host "Created config\demo.toml - enter the real DevEUIs before the demo." -ForegroundColor Yellow
}
Write-Host "`nAllow the gateway to reach the laptop (run once in an ADMIN PowerShell):" -ForegroundColor Yellow
Write-Host '  New-NetFirewallRule -DisplayName "LoRaWAN demonstrator 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Domain'
Write-Host "`nSetup done. Start with: scripts\start_demo.ps1"
