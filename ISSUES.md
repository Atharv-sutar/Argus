# ISSUES.md — Project Core Rules: Identified Problems & Fixes

> **Audience:** This file is written for an AI agent (or human) resolving inconsistencies in `PROJECT_CORE_RULES.md` before it is treated as the authoritative architecture reference. Each issue is atomic and independently actionable. Apply fixes in the order listed — later issues assume earlier ones may already be resolved.

---

## ISSUE-1: Duplicate/ambiguous module name (`camera` vs `cameras`)

**Location:** Architecture Rules → module list

**Problem:**
The module list contains both `camera` and `cameras` as separate entries with no distinction drawn anywhere in the document. This is either a duplicate/typo or an undocumented split of responsibility.

**Resolution options (pick one):**

| Option | Action |
|---|---|
| A. Typo | Delete `cameras`, keep `camera` as the single per-device abstraction module. |
| B. Intentional split | Rename `cameras` → `camera_manager` (or `camera_registry`) and add a short subsection defining its scope: multi-device lifecycle, registry of active `camera` instances, connection pooling. Explicitly state it does NOT contain detection/tracking/ReID logic (same constraint as `camera`). |

**Decision rule for agent:** If no camera-manager/registry code exists yet in the repo, default to Option A (simpler, matches "Simplicity First" principle in the doc). If such a module already exists in code, use Option B and document it.

**Acceptance criteria:** Module list contains no duplicate or unexplained near-duplicate names.

---

## ISSUE-2: Listed modules with no corresponding rules section

**Location:** Architecture Rules → module list vs. rest of document

**Problem:**
Every module below has dedicated rules; the four modules after it do not:

Has rules: `camera`, `detection`, `tracking`, `target`, `reid`, `identity`, `database`

**Missing rules:**
- `core`
- `pipeline`
- `inference`
- `visualization`

**Resolution — add one short subsection per module, following the existing format (purpose, input/output where applicable, constraints):**

- **`core`** — Define what lives here. Recommended scope: shared typed data structures (`Frame`, `DetectionResult`, `TrackResult`, `Embedding`, etc.) and interface definitions only (`Tracker`, `ReID`, `VectorStore`). No implementation logic, no business logic.
- **`pipeline`** — Define its scope explicitly: orchestrates the flow `camera → detection → tracking → reid → identity`, owns queue/threading boundaries, and is the only module allowed to depend on multiple domain modules at once (per the Dependency Direction rule). State that pipeline must not contain detection/tracking/ReID algorithm logic itself — only orchestration.
- **`inference`** — **Requires a decision, not just documentation** (see ISSUE-3 below before writing this section).
- **`visualization`** — State input (frame + overlay data: boxes, IDs, target lock state) and output (rendered frame/stream). Reiterate existing constraint: must not block the processing pipeline (already stated once elsewhere — move or duplicate reference here). State it must not contain detection/tracking/identity logic — display-only.

**Acceptance criteria:** Every module in the Architecture Rules list has a corresponding rules subsection, following the same structure (purpose / input / output / must-not-contain) already used for `camera`, `detection`, etc.

---

## ISSUE-3: `inference` module responsibility is undefined/possibly redundant

**Location:** Architecture Rules → module list

**Problem:**
`detection` and `reid` each already run model inference internally (per their own rules sections: Frame → DetectionResult, crop → Embedding). It's unclear what `inference` adds.

**Resolution — agent must determine which of these is true and document accordingly (do not guess silently — flag for human confirmation if genuinely ambiguous per "Think Before Coding" principle):**

1. **Shared execution layer** — `inference` is a low-level shared utility (e.g., batching, model loading/caching, device placement, shared GPU memory pool) that `detection` and `reid` call into internally. If so: document `inference` as a `core`-adjacent utility module, not a domain module, and clarify in `detection`/`reid` rules that they depend on it for execution but own their own model-specific pre/post-processing.
2. **Redundant naming** — `inference` was meant to describe what `detection`/`reid` do generically and should not be a separate module at all. If so: remove it from the module list.

**Acceptance criteria:** `inference` either has a clearly scoped, non-overlapping responsibility documented, or is removed from the module list.

---

## ISSUE-4: Multi-Camera Search Strategy section is structurally misplaced

**Location:** End of document, after "Completion Requirements"

**Problem:**
This section defines core architecture (camera graph, search radius, target recovery state machine) but is placed after the document has already concluded with completion/definition-of-done rules. An agent reading top-to-bottom will treat it as an appendix rather than binding architecture.

**Resolution:**
Move the entire "Multi-Camera Search Strategy" section to immediately follow **Identity Rules** and immediately precede **Database Rules** — it sits conceptually between "which identity is this" (Identity Rules) and "where do we look next" (camera graph), and ties into `target` (re-association) and `tracking` (active cameras) as well.

**Acceptance criteria:** Section order is: `... → Target Rules → ReID Rules → Identity Rules → Multi-Camera Search Strategy → Database Rules → Performance Rules → ...` (or equivalent — the requirement is that it precedes Completion Requirements and sits with the other architecture-defining sections, not after them).

---

## ISSUE-5: `database` module name vs. actual scope (`VectorStore` only)

**Location:** Database Rules section, cross-referenced with project stack (Redis + PostgreSQL + FAISS)

**Problem:**
The module is named `database`, implying general persistence, but the rules text only ever discusses vector storage (`VectorStore` / FAISS abstraction). The actual stack includes relational/session storage (PostgreSQL) and caching (Redis) that this section doesn't address.

**Resolution options (pick one):**

| Option | Action |
|---|---|
| A. Narrow the module | Rename `database` → `vectorstore` in the module list. Add separate module(s) for relational storage if/when that code exists (e.g., `records` or `session_store`), each with its own rules section when implemented. |
| B. Broaden the section | Keep `database` as-is but expand Database Rules to explicitly cover all persistence: vector storage via `VectorStore`/FAISS, relational storage via an abstraction (not raw PostgreSQL calls scattered through the app), and caching via Redis abstraction. State the same dependency-inversion pattern applies to all three (no module should import `faiss`, `psycopg2`, or `redis` directly except the storage abstraction implementations). |

**Decision rule for agent:** Option A is preferred under "Simplicity First" and "Minimal unnecessary complexity" — don't rename yet if relational/cache code doesn't exist in the repo. Use Option B if PostgreSQL/Redis integration is already implemented or imminent.

**Acceptance criteria:** Module name and rules text scope match; no persistence technology is referenced without an abstraction layer requirement.

---

## ISSUE-6: Priority ordering — real-time performance ranked below readability/debuggability

**Location:** Purpose → numbered priority list

**Problem:**
Not a bug, but a design decision that should be explicit rather than implicit. Current order:
`1. Correctness → 2. Modularity → 3. Readability → 4. Debuggability → 5. Real-time performance → ...`

For a live forensic tracking system, this means code clarity is prioritized over live performance when the two conflict.

**Resolution:**
No change required by default — **flag this to the human maintainer for explicit confirmation** rather than silently reordering. If confirmed intentional, add one sentence after the list: *"This ordering is intentional: maintainability is prioritized over raw real-time throughput except where real-time performance drops below the minimum viable threshold defined in Performance Rules."* If not intentional, reorder and get sign-off.

**Acceptance criteria:** Priority ordering has an explicit rationale sentence, OR has been reordered per maintainer confirmation. Agent must not reorder this list unilaterally.

---

## ISSUE-7: Search radius expansion has no concrete default

**Location:** Multi-Camera Search Strategy → radius expansion behavior

**Problem:**
"Search radius may progress from 1-hop to 2-hop, then further if necessary" — no default increment, cap, or timeout value is specified. This is acceptable for a rules doc (implementation detail), but should not be left undefined once the search manager is implemented.

**Resolution:**
Not a doc fix — a tracking item. Add a short note: *"Default values (radius increment, per-radius timeout, max radius) are defined in configuration, not hardcoded, and must be set during initial implementation of the search manager — see `target` and `tracking` implementation."*

**Acceptance criteria:** A pointer exists so the value doesn't get silently hardcoded without a config surface.

---

## Summary Table (for quick agent triage)

| ID | Severity | Type | Needs human confirmation? |
|---|---|---|---|
| ISSUE-1 | Medium | Naming/duplication | No — default to Option A unless code says otherwise |
| ISSUE-2 | Medium | Documentation gap | No |
| ISSUE-3 | Medium | Undefined responsibility | **Yes** — ambiguous, do not guess |
| ISSUE-4 | Low | Structural/ordering | No |
| ISSUE-5 | Low-Medium | Naming/scope mismatch | Only if choosing Option B |
| ISSUE-6 | Low | Design intent | **Yes** — do not reorder unilaterally |
| ISSUE-7 | Low | Missing default (tracking item, not a doc error) | No |

**Rule for the resolving agent:** Per the project's own "Think Before Coding" principle, do not silently resolve ISSUE-3 or ISSUE-6 — surface them for confirmation before editing the source document.