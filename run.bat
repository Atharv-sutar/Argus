@echo off
REM ============================================================
REM Argus Surveillance Operations Center - Self-Healing Launcher
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM 1. Locate a working Python interpreter (.venv first, then PATH)
set "PYTHON_CMD="
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" --version >nul 2>&1
    if !ERRORLEVEL! EQU 0 set "PYTHON_CMD=.venv\Scripts\python.exe"
)

if not defined PYTHON_CMD (
    where python >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        python -c "import sys" >nul 2>&1
        if !ERRORLEVEL! EQU 0 set "PYTHON_CMD=python"
    )
)

if not defined PYTHON_CMD (
    where py >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        py -c "import sys" >nul 2>&1
        if !ERRORLEVEL! EQU 0 set "PYTHON_CMD=py"
    )
)

REM 2. If Python is missing, attempt automatic installation via winget
if not defined PYTHON_CMD (
    echo [ARGUS] Python was not found in system PATH.
    echo [ARGUS] Installing Python 3.12 via Windows Package Manager ^(winget^)...
    winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
    if !ERRORLEVEL! NEQ 0 (
        echo [CRITICAL] Automatic Python installation failed.
        echo Please manually install Python 3.11 or 3.12 from https://www.python.org/downloads/
        if not defined ARGUS_NO_PAUSE pause
        exit /b 1
    )
    if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
        set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python312\python.exe"
    ) else (
        set "PYTHON_CMD=python"
    )
)

REM 3. Execute self-healing bootstrapper (verifies venv, installs missing packages, launches Argus)
"%PYTHON_CMD%" scripts\bootstrap.py %*
set "EXIT_CODE=%ERRORLEVEL%"

if !EXIT_CODE! NEQ 0 (
    echo.
    echo [ARGUS] Process exited with code !EXIT_CODE!.
    if not defined ARGUS_NO_PAUSE pause
)
exit /b !EXIT_CODE!
