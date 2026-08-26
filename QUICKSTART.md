# Argus Multi-Camera Surveillance — Quickstart Guide

This guide explains how to set up and run the Argus Multi-Camera Surveillance Operations Center.

---

## 1. Fast Start (Recommended)

Run the launcher script corresponding to your terminal. It will automatically launch the surveillance pipeline and open the web dashboard in your default browser at **`http://127.0.0.1:8765`**:

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

## 3. Web Dashboard Features

- **Multi-Camera Live Matrix**: View live video streams across all configured cameras simultaneously.
- **Click-to-Focus & Target Lock**: Click any camera tile to focus; click on a tracked person in that feed to lock focus and accumulate their appearance gallery.
- **Vertical Target Gallery (Right Side)**: Dedicated real-time scrollable column displaying all locked appearances (MANUAL and AUTO angles).
- **Progressive Multi-Camera Search**: When the target is lost, adjacent cameras automatically activate at Radius 1 &rarr; Radius 2 &rarr; Radius 3 (max radius = 3).

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
