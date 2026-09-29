# Bug Tracking Rules

- **Volatile Bug Register:** Every bug or problem found during development MUST be registered in a volatile space, specifically a file named `BUGS.md` in the root directory. 
- **Bug Resolution:** Once a bug is successfully fixed, it MUST be removed from `BUGS.md`. The file should only contain active, unresolved issues.
- **Regression Prevention:** You MUST ensure that problems that have been fixed NEVER occur again (e.g., camera FPS drops, topology issues, memory leaks). When modifying code, review the context of past fixes and ensure your new changes do not undo them or reintroduce old bugs.
