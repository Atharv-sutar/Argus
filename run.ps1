# ============================================================
# Argus Surveillance Operations Center — PowerShell Launcher
# ============================================================

$repoDir = $PSScriptRoot
Set-Location $repoDir

$venvPython = Join-Path $repoDir ".venv\Scripts\python.exe"

# 1. Locate available Python interpreter
$pythonCmd = $null
if (Test-Path $venvPython) {
    $pythonCmd = $venvPython
} else {
    $pyCmdObj = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmdObj) {
        $pythonCmd = $pyCmdObj.Source
    } else {
        $pyCmdObj = Get-Command py -ErrorAction SilentlyContinue
        if ($pyCmdObj) {
            $pythonCmd = $pyCmdObj.Source
        }
    }
}

# 2. If Python is completely missing, attempt automatic installation via winget
if (-not $pythonCmd) {
    Write-Host "[ARGUS] Python was not found in system PATH." -ForegroundColor Yellow
    Write-Host "[ARGUS] Installing Python 3.12 via Windows Package Manager (winget)..." -ForegroundColor Cyan
    try {
        winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
        $pyCmdObj = Get-Command python -ErrorAction SilentlyContinue
        if ($pyCmdObj) {
            $pythonCmd = $pyCmdObj.Source
        }
    } catch {
        Write-Host "[CRITICAL] Could not install Python automatically. Please install Python 3.11+ from https://www.python.org/downloads/" -ForegroundColor Red
        exit 1
    }
}

# 3. Launch self-healing bootstrapper
$bootstrapScript = Join-Path $repoDir "scripts\bootstrap.py"
& $pythonCmd $bootstrapScript $args
