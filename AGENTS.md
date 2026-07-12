# AGENTS.md — MINATO (code + tutorials + repo hygiene)

## What this repo is
MINATO is a Python toolkit for the study of massive stars.
Typical tasks:
- improve performance (especially CPU-heavy workflows)
- improve readability/maintainability
- clean up code and docs
- create and maintain tutorial notebooks

## Where I usually work
- Most development happens on my laptop (conda/mamba environment: `mamba activate minato`).
- The institute astro-nodes are used primarily to test large CPU workflows (may use uv there).
When suggesting commands, prefer to *propose* what to run; do not assume you can execute them unless I tell you so.

## Repo boundaries
- Keep changes scoped and reviewable.
- Do NOT add large binary files or big datasets.
- Do NOT do “drive-by” refactors or global style rewrites.
- Do NOT introduce new heavy dependencies unless I explicitly ask.

## Coding expectations
- Prefer clear, readable Python over cleverness.
- Preserve public APIs unless I request a breaking change.
- For performance work:
  - profile/measure first when possible (even lightweight timing)
  - prefer algorithmic wins, vectorization, caching, and reducing I/O
  - avoid premature micro-optimizations
- Add short docstrings/comments when behavior is non-obvious.
- Keep docstrings up to date with changes in the code.
- If you change numerical behavior, call it out clearly.

## Tutorials (important)
Tutorials live in: `minato/tutorials/` (currently notebooks).
Rules:
- Keep notebooks clean and pedagogical (short markdown explanations + runnable code).
- Use a stable dataset or synthetic examples (no large external downloads).
- If a tutorial depends on optional packages or special data, say so at the top.
Future: tutorials may move to a ReadTheDocs-style site; write notebooks so they can be converted later.

## Documentation files that must stay in sync (very important)

## Branch workflow for documentation
- `develop` is the working branch for ongoing development, and development branches should normally be created from `develop`.
- Development records such as `AGENTS.md`, `ROADMAP.md`, `NOTES.md`, `CHANGELOG.md`, and `*_PLAN.md` are tracked on `develop` so they are available on any machine used for development.
- `main` is the release branch. The official release date is the date when the approved release merge from `develop` is committed on `main`.
- Prepare the merge with `git merge --no-commit --no-ff develop`, remove development-only records and unfinished archive recovery from the pending merge tree, validate the resulting release tree, and then create the merge commit.
- Copy the approved user-facing change summary into the GitHub release notes instead of keeping development records on `main`.
- Do not use local-only ignore rules to hide these files during development; they should stay visible and versioned on development branches.

### 1) ROADMAP.md (to-do list)
- ROADMAP items use the status convention: `[][]`.
  - First bracket: `STARTED` (optional) or `Done` when completed
  - Second bracket: completion date (`YYYY-MM-DD`)
  - Example: `- Add smoke-test script [Done][2025-12-15]`
- When we complete and I approve a change, mark the relevant item(s) as `Done` with date.
- Do NOT mark items as `Done` unless the work is actually completed and approved.

### 2) CHANGELOG.md
- Maintain an `Unreleased` section and past versions.
- When a change is completed and approved, add an entry under `Unreleased`:
  - Added / Changed / Fixed (use the appropriate subsection)
- Add new entries at the top of the appropriate subsection so each list is reverse chronological.
- Keep entries short, user-facing, and specific (what changed and why it matters).
- Do NOT bump version numbers unless I explicitly instruct.

### 3) README files
There are multiple READMEs with different purposes:
- `README.md` (repo main; GitHub-facing):
  - Keep this high-level: what MINATO is, installation, quick-start, links.
  - Update this after each release (when I say we are releasing).
- `minato/README.md`:
  - This is the tutorial index / user navigation.
  - Keep the list of tutorials current and well organized.
- `minato/binary_population/README.md`:
  - Module-specific documentation; keep it consistent with the code in that module.

## When to update what (workflow rule)
After each task:
1) Summarize changes (files edited + what changed).
2) If the task produces a stable, approved improvement:
   - Update `CHANGELOG.md` under `Unreleased`.
   - Update `ROADMAP.md` status/date for the completed item(s).
3) If we are doing a release (only when I say so):
   - Update `README.md` (main) for release notes / install / quick-start pointers.
   - Ensure tutorial index (`minato/README.md`) is consistent.

## Output expectations (how to report back)
When done, report:
- What changed (bullet list)
- Any behavior changes (API, numerical results, performance)
- What to run to validate (suggest commands for laptop and/or astro-nodes)
- Which ROADMAP + CHANGELOG entries you updated (or why you didn’t)
