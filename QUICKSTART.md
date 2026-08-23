# Argus Surveillance — Quickstart Guide

This guide explains how to set up and run the Argus single-camera surveillance pipeline.

---

## 1. Fast Start (Recommended)

Run the launcher script corresponding to your terminal. It will automatically use the project's virtual environment:

### PowerShell:
```powershell
.\run.ps1
```

### Windows Command Prompt (CMD):
```cmd
run.bat
```

---

## 2. Manual Setup & Running

If you prefer to run commands manually:

### Step 1: Activate Virtual Environment
```powershell
# In PowerShell:
.\.venv\Scripts\Activate.ps1

# Or in CMD:
.venv\Scripts\activate.bat
```

### Step 2: Run Application
```bash
python -m src.app.main
```

---

## 3. Common Usage Examples

### A. Run with Webcam (Default)
Opens the default webcam (device `0`) and shows real-time person detection & tracking:
```powershell
.\run.ps1
```

### B. Run on a Video File
```powershell
.\run.ps1 --source "path\to\video.mp4"
```

### C. Run Headless Simulation (No camera/GPU needed)
Runs a synthetic test feed in headless mode:
```powershell
.\run.ps1 --synthetic --no-gui
```

### D. Specify Hardware Acceleration
Force GPU or CPU execution:
```powershell
.\run.ps1 --device cuda
# or
.\run.ps1 --device cpu
```

---

## 4. Running Automated Tests

Run the full pytest suite:
```powershell
.\.venv\Scripts\pytest.exe
```

---

## 5. Controls
- Press **`q`** or **`Esc`** in the video window to stop the application.
