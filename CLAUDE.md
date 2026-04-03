# CLAUDE.md — Project Reference

## Hard Constraints
- `kamtin-fp-model/` — READ ONLY, never modify. All Python must conform to it.
- Log design decisions in `docs/decisions.md` BEFORE implementing non-trivial choices.
- `scripts/` benchmarks (MSD, Bouc-Wen, Cascaded Tanks) are reference examples only — not the target system.

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

## Workflow
- **Plan mode** for any task with 3+ steps or architectural decisions. Stop and re-plan if something goes sideways.
- **Subagent triggers (non-negotiable):**
  - Codebase search requiring >2 grep/glob calls → Explore subagent
  - Parallel file analysis → separate subagents
  - Heavy research → general-purpose subagent
- **Self-improvement:** after any user correction, apply 3-criteria gate (generalizable + actionable + novel) → update `tasks/lessons.md`. Merge, do not append duplicates.
- **Verification:** never mark a task complete without proving it works.
- **Task flow:** plan to `tasks/todo.md` → check in with user → implement → mark complete → log decisions to `docs/decisions.md`.
