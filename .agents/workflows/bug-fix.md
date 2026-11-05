# Bug Fix Workflow

This workflow guides the reproducible identification, tracking, resolution, and verification of defects.

## Step 1: Register in Volatile Bug Tracker
1. If the bug is verified or reported, log it in `BUGS.md` in the project root:
   - Bug ID / Title
   - Symptom & error log
   - Affected module
   - Status: `[OPEN]`

## Step 2: Root Cause Analysis & Reproduction
1. Formulate a minimal reproduction script or automated unit test reproducing the failure.
2. Inspect the exact execution path without modifying unrelated code.
3. Check for previous regressions or related past fixes.

## Step 3: Implement Surgical Fix
1. Apply the smallest complete fix resolving the root cause.
2. Guard against edge cases (e.g. `None` frames, missing camera handles, disconnected SSE streams).
3. Ensure no memory leaks or unbounded queues are created.

## Step 4: Verify Fix & Regression Testing
1. Run the reproduction test to verify it passes.
2. Run the full component test suite:
   ```powershell
   .\.venv\Scripts\python -m pytest tests/unit/
   ```

## Step 5: Close Bug & Auto Push
1. Remove or mark the resolved bug entry as cleared in `BUGS.md`.
2. Commit and push changes:
   ```powershell
   git add -A
   git commit -m "fix(<component>): resolve <issue description>"
   git push
   ```

