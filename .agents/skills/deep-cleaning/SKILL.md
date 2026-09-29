---
name: deep-cleaning
description: A systematic workflow to deep clean the entire project file by file, line by line, removing obsolete code while preserving in-progress and future work.
---

# Deep Cleaning Workflow

When the user asks you to "clean the project", "remove unnecessary things", or run the deep cleaning skill, follow this rigorous file-by-file process.

## 1. Setup the Progress Log
- Check for the existence of `CLEANING_PROGRESS.md` in the project root. If it doesn't exist, create it.
- This file acts as a volatile progress log to track exactly what files and lines have been visited. 
- Example format:
  ```markdown
  # Cleaning Progress
  - [x] src/core/types.py (Completed)
  - [ ] src/multi_camera/camera_graph.py (In Progress - Line 150)
  - [ ] src/target/manager.py (Pending)
  ```
- Use `list_dir` or similar tools to get a full list of project files and populate the pending list.

## 2. Iterative Cleaning Process
- Iterate through each file in the `CLEANING_PROGRESS.md` list one by one.
- Read the file content carefully.
- Identify and remove:
  - Obsolete code, unused imports, unused variables, and dead functions.
  - Wasted logic or unnecessary comments.
- **DO NOT REMOVE:**
  - Code that is currently being implemented.
  - Placeholders, interfaces, or structures clearly marked or needed for future features.
- Apply surgical changes using code editing tools.

## 3. Log Updates
- Continuously update `CLEANING_PROGRESS.md` with your exact progress (file name and last checked line number).
- **CRITICAL:** If you are nearing token limits or need to stop, ensure `CLEANING_PROGRESS.md` accurately reflects the exact line you stopped at. This ensures the next agent can take over directly without going through previously cleaned code.

## 4. Completion
- Once all files in the project are marked as completed, delete the `CLEANING_PROGRESS.md` file.
- Push your final changes to GitHub as per the project rules.
- Announce the completion of the deep cleaning process to the user.
