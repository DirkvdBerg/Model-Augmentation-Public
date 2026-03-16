# CLAUDE.md — Project Reference

## Project Goal

Convert the ASMPT dual-gantry First Principles (FP) model (García-Herreros et al.) from MATLAB to a discrete-time state-space form compatible with the LFR-based model augmentation framework in this repository. Once in the correct form, the augmentation framework learns data-driven corrections to the FP model from experimental gantry data.

## Python Environment

All packages (deepSI, PyTorch, scipy, etc.) are installed in the **`GraduationProject`** conda environment only.

- When running Python via Bash: always use `conda run -n GraduationProject python ...`
- When the user runs scripts manually: they must activate `conda activate GraduationProject` first

## Hard Constraints

- **`kamtin-fp-model/` is immutable.** These MATLAB files define the FP model structure and are the ground truth. They must never be modified. All Python implementations must conform to the structure they define.
- **Design decisions must be logged in `docs/decisions.md` before implementation begins** on any non-trivial choice.
- **`scripts/` benchmarks (MSD, Bouc-Wen, Cascaded Tanks) are reference-only.** They are working examples of the augmentation framework, not the target system. Do not treat them as the goal.

## Key File Map

| What | Where |
|------|-------|
| FP model — MATLAB ground truth | `kamtin-fp-model/` |
| Augmentation framework — core library | `model_augmentation/fit_systems/` |
| Reference benchmark scripts | `scripts/` |
| Research plan & methods | `Research-Plan/` |
| FP model structure (curated reference) | `docs/fp-model-structure.md` |
| FP ↔ augmentation interface contract | `docs/fp-augmentation-interface.md` |
| Design decisions log | `docs/decisions.md` |
| Session task tracking | `tasks/todo.md` |
| Self-improvement ruleset | `tasks/lessons.md` |
| Session handoff state | `tasks/handoff.md` |
| Archived LPV planning docs | `archive/` |

## Multi-Agent File Ownership

This project may be worked on by Claude and Codex in relay. File ownership rules:

| File(s) | Claude | Codex |
|---------|--------|-------|
| `tasks/lessons.md`, `docs/decisions.md`, `tasks/todo.md`, `tasks/handoff.md`, `docs/` | read/write | read/write |
| `CLAUDE.md`, `.claude/settings.json` | read/write | read + propose only |
| `CODEX.md` | read + propose only | read/write |
| `kamtin-fp-model/` | **never write** | **never write** |

**Proposing changes to the other agent's instruction file:** write the suggestion in `tasks/handoff.md` under `### Proposed improvements for [Claude/Codex]`. The owning agent implements it.

---

## Step 0 — Before Every Response

**Check for corrections first.** If the previous message was a correction, rejection, or redo request:
1. Read `tasks/lessons.md`
2. Apply the 3-criteria gate (generalizable + actionable + novel)
3. Update the ruleset if it passes — merge with existing rules, do not append duplicates
4. Only then proceed with the current request

This is non-optional. Do not defer it until after execution.

---

## Workflow Rules (Hard Constraints)

### Planning
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions).
- If something goes sideways, STOP and re-plan — do not keep pushing.
- Write detailed specs upfront to reduce ambiguity.

### Subagents
- Use subagents to keep the main context window clean.
- Offload research, exploration, and parallel analysis to subagents.
- One focused task per subagent.

### Self-Improvement
- After ANY correction from the user: evaluate whether it meets the lesson criteria (see `tasks/lessons.md`). If yes, update the ruleset — do not just append.
- **At the start of every session**: read `tasks/lessons.md` before doing any work. Rules there are active constraints, not suggestions.

### Verification
- Never mark a task complete without proving it works.
- Ask: "Would a staff engineer approve this?" before presenting.

### Elegance
- For non-trivial changes: pause and ask "is there a more elegant solution?"
- Skip this for simple, obvious fixes — do not over-engineer.

### Autonomous Bug Fixing
- When given a bug report: diagnose and fix it. Point at logs, errors, failing tests — then resolve them.

## Task Management

1. **Plan first** — write plan to `tasks/todo.md` with checkable items.
2. **Check in** — verify plan with user before starting implementation on non-trivial work.
3. **Track progress** — mark items complete as you go; give a high-level summary at each step.
4. **Log design choices** — any architectural or method decision goes to `docs/decisions.md`.
5. **Document results** — add a brief review section to `tasks/todo.md` when work is complete.
6. **Capture lessons** — after any correction, apply the 3-criteria gate and update `tasks/lessons.md`.

## Core Principles

- **Simplicity first** — make every change as simple as possible. Minimal code impact.
- **No laziness** — find root causes. No temporary fixes. Senior developer standards.
- **Minimal impact** — changes touch only what is necessary. Avoid introducing bugs.
