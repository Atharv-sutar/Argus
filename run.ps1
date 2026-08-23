# Argus Surveillance System PowerShell Launcher

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[ARGUS] Virtual environment not found. Creating .venv..." -ForegroundColor Yellow
    py -3.12 -m venv .venv
    & (Join-Path $PSScriptRoot ".venv\Scripts\pip.exe") install -e .
}

& $venvPython -m src.app.main $args
