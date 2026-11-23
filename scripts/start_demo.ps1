# Start the demonstrator and open the dashboard.
#   powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Test-Path .venv)) { throw "Run scripts\setup.ps1 first." }
if (-not (Test-Path data\models\manifest.json)) { Write-Warning "No trained models in data\models - replay/AI panels will be empty. See README (Train the models)." }
$cfg = if (Test-Path config\demo.toml) { "config\demo.toml" } else { "config\demo.example.toml" }
Start-Job -ScriptBlock { Start-Sleep 3; Start-Process "http://localhost:8000/" } | Out-Null
.\.venv\Scripts\python.exe -m demonstrator --config $cfg
