# ============================================================
# Argus Surveillance Operations Center - PowerShell Launcher
# ============================================================

$repoDir = $PSScriptRoot
Set-Location $repoDir

function Test-WorkingPython {
    param([string]$ExePath)
    if (-not $ExePath) { return $false }
    try {
        & $ExePath -c "import sys" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

# 1. Locate a working Python interpreter (.venv first, then PATH)
$venvPython = Join-Path $repoDir ".venv\Scripts\python.exe"
$pythonCmd = $null

if ((Test-Path $venvPython) -and (Test-WorkingPython $venvPython)) {
    $pythonCmd = $venvPython
} else {
    foreach ($candidate in @("python", "py")) {
        $cmdObj = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmdObj -and (Test-WorkingPython $cmdObj.Source)) {
            $pythonCmd = $cmdObj.Source
            break
        }
    }
}

# 2. If Python is missing, attempt automatic installation via winget
if (-not $pythonCmd) {
    Write-Host "[ARGUS] Python was not found in system PATH." -ForegroundColor Yellow
    Write-Host "[ARGUS] Installing Python 3.12 via Windows Package Manager (winget)..." -ForegroundColor Cyan
    try {
        winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
        $installedPy312 = Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"
        if ((Test-Path $installedPy312) -and (Test-WorkingPython $installedPy312)) {
            $pythonCmd = $installedPy312
        } else {
            $cmdObj = Get-Command python -ErrorAction SilentlyContinue
            if ($cmdObj -and (Test-WorkingPython $cmdObj.Source)) {
                $pythonCmd = $cmdObj.Source
            }
        }
    } catch {
        Write-Host "[CRITICAL] Could not install Python automatically. Please install Python 3.11+ from https://www.python.org/downloads/" -ForegroundColor Red
        exit 1
    }
}

if (-not $pythonCmd) {
    Write-Host "[CRITICAL] No working Python interpreter found. Please install Python 3.11+ and re-run." -ForegroundColor Red
    exit 1
}

# 3. Launch self-healing bootstrapper with proper argument splatting
$bootstrapScript = Join-Path $repoDir "scripts\bootstrap.py"
& $pythonCmd $bootstrapScript @args
exit $LASTEXITCODE
