# CLAUDE.md — Project Reference

## Code Quote Verification (MANDATORY)
Before quoting any code from an external file with a specific line number claim:
1. Write the intended quote to `/tmp/quote.txt` via Bash
2. Run: `conda run -n GraduationProject python scripts/verify_quote.py <file> <start> <end> /tmp/quote.txt`
3. The shell output MUST appear in the conversation BEFORE the quote
4. Only proceed if the script prints MATCH OK — never quote if MISMATCH
5. Never use Mode A (no quote file) for making claims — that is not verification

The user can re-run the same command with `! python scripts/verify_quote.py ...` to independently verify.

## Hard Constraints
- `kamtin-fp-model/` — READ ONLY, never modify. All Python must conform to it.
- Log design decisions in `docs/decisions.md` BEFORE implementing non-trivial choices.
- `scripts/` benchmarks (MSD, Bouc-Wen, Cascaded Tanks) are reference examples only — not the target system.
- When the user says "don't touch the code," do NOT edit or run ANY code. Only respond in text. Present the changes you would make and wait for explicit permission before writing or editing any file.

## Python Environment
Bash: `conda run -n GraduationProject python ...`. User manual: `conda activate GraduationProject`.

## Key File Map
| What | Where |
|------|-------|
| FP model — MATLAB ground truth | `kamtin-fp-model/` |
| Augmentation framework | `model_augmentation/fit_systems/` |
| Reference benchmarks | `scripts/` |
| FP model structure | `docs/fp-model-structure.md` |
| FP ↔ augmentation interface | `docs/fp-augmentation-interface.md` |
| Design decisions log | `docs/decisions.md` |
| Session tasks | `tasks/todo.md` |
| Self-improvement ruleset | `tasks/lessons.md` |
| Session handoff | `tasks/handoff.md` |
| Archived docs | `archive/` |
| **Telica data schema** (columns, signals, folder structure) | `docs/kamtin-telica-schema.md` |

## Data Access Policy
- `kamtin-data/Data Telica/` is blocked via `.claudeignore` — **never attempt to read files there**.
- For all column names, signal meanings, unit conversions, and file structure: read `docs/kamtin-telica-schema.md`.
- Do NOT use `Bash` to cat/head/grep files in `kamtin-data/Data Telica/`.

## Multi-Agent Ownership
| File(s) | Claude | Codex |
|---------|--------|-------|
| `tasks/`, `docs/`, `CLAUDE.md`, `.claude/settings.json` | read/write | read + propose only |
| `CODEX.md` | read + propose only | read/write |
| `kamtin-fp-model/` | **never write** | **never write** |

Proposals go in `tasks/handoff.md` under `### Proposed improvements for [Claude/Codex]`.

## Step 0 — Every Session
Read `tasks/lessons.md` before any work. Rules there are active constraints, not suggestions.

## Archival Rules (on read)
- `tasks/handoff.md`: archive full content to `archive/sessions/YYYY-MM-DD-handoff.md`, then trim file to open blockers only.
- `tasks/todo.md`: move any completed sections/tasks to `archive/sessions/YYYY-MM-DD-todo.md`, then remove them from the live file.

## Tracking Additions to `model_augmentation/`
The framework in `model_augmentation/` is Jan's original code. Additions are marked with three mechanisms — use these to instantly see what is ours:

| Situation | Marker | Grep |
|-----------|--------|------|
| New class or function in a modified file | `@added` decorator | `grep -r "@added" model_augmentation/` |
| Entire new file | `__project_origin__ = "added"` at module top | `grep -r "__project_origin__" model_augmentation/` |
| Changed block inside an existing class/function | `# CHANGED: <reason>` inline | `grep -r "# CHANGED:" model_augmentation/` |

**Rule:** whenever new code is added to `model_augmentation/`, apply the appropriate marker before committing.

## Signal Processing & System Identification Code
Any numerical formula, constant, or threshold in signal processing, experiment design, or system identification code **must carry an inline label before it can be written**:
- `# THEORY: <source>` — formula derived directly from literature; source, variable, and context must all match
- `# HEURISTIC: <reason>` — engineering invention with no literature source; flag it explicitly

No label = do not implement yet. Literature validates only if the formula, variable, and context match — not just the numeric constant.

## Workflow
- **Plan mode** for any task with 3+ steps or architectural decisions. Stop and re-plan if something goes sideways.
- **Subagent triggers (non-negotiable):**
  - Codebase search requiring >2 grep/glob calls → Explore subagent
  - Parallel file analysis → separate subagents
  - Heavy research → general-purpose subagent
- **Self-improvement:** after any user correction, apply 3-criteria gate (generalizable + actionable + novel) → update `tasks/lessons.md`. Merge, do not append duplicates.
- **Verification:** never mark a task complete without proving it works.
- **Task flow:** plan to `tasks/todo.md` → check in with user → implement → mark complete → log decisions to `docs/decisions.md`.
