# Git Workflow Rules

## Purpose
Ensure that work is properly tracked, safe, and proactively synchronized with the remote GitHub repository without manual user intervention.

## 1. Automatic Pushes (Mandatory)
- After every major step, milestone, new feature, or significant bugfix is completed, you **MUST** automatically commit and push the changes to GitHub.
- Do NOT wait for the user to explicitly ask you to push.
- When running in an environment where network access requires sandbox elevation, execute the push using the sanctioned bypass parameter (`BypassSandbox: true`).

## 2. Standard Command Sequence
```powershell
git add -A
git commit -m "<concise descriptive summary of changes>"
git push
```

## 3. Commit Message Standards
- Use concise, imperative commit messages (e.g., `fix(matrix): resolve dual webcam detection and backend alignment`).
- If resolving a specific bug or issue, reference the issue number or bug description in the message (e.g., `fix(reid): prevent gallery contamination on cross-camera handoff (#42)`).

## 4. Focused Commits
- Keep commits cohesive and focused on the logical task at hand.
- Do not bundle unrelated refactorings or file renames into a feature/bugfix commit.
- Never commit runtime test databases (`*.db`, `*.sqlite`), temporary video dumps, or unversioned model artifacts.
