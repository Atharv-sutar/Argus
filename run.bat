@echo off
REM Argus Surveillance System Launcher

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ARGUS] Virtual environment not found. Setting up .venv...
    py -3.12 -m venv .venv
    .venv\Scripts\pip install -r requirements.txt
)

.venv\Scripts\python.exe -m src.app.main %*
