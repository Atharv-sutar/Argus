# Pre-Release & Quality Assurance Workflow

Run this workflow prior to completing major milestones or releases.

## Checklist

### 1. Automated Test Suite
Run the test suites across core, multi-camera, target, and tracking modules:
```powershell
.\.venv\Scripts\python -m pytest tests/unit/core/ tests/unit/multi_camera/ tests/unit/target/ tests/unit/tracking/ -v
```

### 2. Frontend Syntax & Static Assets
Verify JavaScript syntax across web interface scripts:
```powershell
node -c src/multi_camera/static/js/app.js
node -c src/multi_camera/static/js/api.js
node -c src/multi_camera/static/js/graph.js
node -c src/multi_camera/static/js/camera.js
node -c src/multi_camera/static/js/inspector.js
```

### 3. Git Working Tree Cleanliness
Ensure no accidental SQLite databases, log dumps, or unversioned videos are staged:
```powershell
git status
```

### 4. Topology Configuration Integrity
Verify default camera graph is valid:
```powershell
.\.venv\Scripts\python -m pytest tests/unit/multi_camera/test_camera_graph.py
```

### 5. Final Push
Execute clean push:
```powershell
git add -A
git commit -m "chore(release): pre-release quality checks verified"
git push
```

