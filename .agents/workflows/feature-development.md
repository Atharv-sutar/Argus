# Feature Development Workflow

This workflow guides the development of new features in the Argus surveillance codebase.

## Step 1: Requirements & Architectural Alignment
1. Inspect only the relevant files.
2. Review architecture rules in `.agents/rules/project-core.md`.
3. Verify that changes maintain modular boundaries (`core`, `camera`, `detection`, `tracking`, `reid`, `target`, `multi_camera`, `pipeline`, `visualization`).
4. Ensure hardware constraints are respected (GTX 1650 4GB VRAM ceiling).

## Step 2: Planning & Approval
1. Use `/plan` to draft an implementation plan with success criteria, proposed file diffs, and verification steps.
2. Surface any tradeoffs or design ambiguities to the user before writing code.

## Step 3: Surgical Implementation
1. Modify only the required files.
2. Adhere to typed dataclasses and interfaces.
3. Keep existing comments, docstrings, and invariants intact.
4. Avoid unneeded dependencies or heavy frameworks.

## Step 4: Verification & Automated Testing
1. Run relevant unit and integration test suites:
   ```powershell
   .\.venv\Scripts\python -m pytest tests/unit/<component>/ -v
   ```
2. Verify visual UI changes in browser or with syntax checks (`node -c <file>.js`).

## Step 5: Automatic Git Push
1. Inspect `git status` and `git diff`.
2. Commit with concise descriptive message:
   ```powershell
   git add -A
   git commit -m "feat(<component>): <description>"
   git push
   ```

