# Bug Tracking Rules

## 1. Volatile Bug Register (`BUGS.md`)
- Every unresolved bug, regression, or failure identified during development or testing **MUST** be recorded in `BUGS.md` in the project root.
- Structure each entry with:
  - **Bug ID / Title**: Short summary.
  - **Symptom**: Observable failure or error log.
  - **Affected Module**: File or component responsible.
  - **Status**: `[OPEN]` or `[INVESTIGATING]`.

## 2. Bug Resolution
- Once a bug is verified and resolved with passing tests, remove its entry from `BUGS.md`.
- `BUGS.md` should only contain active, open issues. If all known issues are resolved, `BUGS.md` should clearly state that there are no active bugs.

## 3. Regression Prevention
- Review past fixes before implementing non-trivial architectural changes.
- Ensure that past critical issues (e.g., dual webcam DirectShow indexing, memory leaks in video loops, ReID gallery contamination, SSE disconnection hangs) are not re-introduced.
- Write or update a regression test in `tests/unit/` whenever fixing a recurring or complex defect.
