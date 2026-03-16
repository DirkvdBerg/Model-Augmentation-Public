# CODEX.md — Project Reference for Codex

This file is your entry point. Read it first, every session.

## Session Start — Read in This Order
1. This file
2. `tasks/lessons.md` — active ruleset, treat as hard constraints
3. `tasks/handoff.md` — where the previous session left off (if it exists)
4. `tasks/todo.md` — current task state

Confirm your understanding of the current task to the user before acting.

---

## Project Goal

Convert the ASMPT dual-gantry First Principles (FP) model (García-Herreros et al.) from MATLAB to a discrete-time state-space form compatible with the LFR-based model augmentation framework in this repository. Once in the correct form, the augmentation framework learns data-driven corrections to the FP model from experimental gantry data.

## Hard Constraints

- **`kamtin-fp-model/` is immutable.** Never modify any file inside it. All Python implementations must conform to the structure it defines.
- **Log design decisions in `docs/decisions.md` before implementing** any non-trivial choice.
- **`scripts/` benchmarks are reference-only.** Working examples of the augmentation framework — not the target system.
- **Never modify `CLAUDE.md` or `.claude/settings.json`.** Propose improvements via `tasks/handoff.md` instead.

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

## File Ownership

| File(s) | Claude | Codex |
|---------|--------|-------|
| `tasks/lessons.md`, `docs/decisions.md`, `tasks/todo.md`, `tasks/handoff.md`, `docs/` | read/write | read/write |
| `CLAUDE.md`, `.claude/settings.json` | read/write | read + propose only |
| `CODEX.md` | read + propose only | read/write |
| `kamtin-fp-model/` | **never write** | **never write** |

---

## Workflow Rules

### Before Every Response
If the previous message was a correction or rejection of your output:
1. Read `tasks/lessons.md`
2. Apply the 3-criteria gate: generalizable + actionable + novel
3. Update the ruleset if it passes — merge with existing rules, do not duplicate
4. Then proceed with the request

### Planning
- Plan any task with 3+ steps or architectural decisions before starting.
- Write the plan to `tasks/todo.md` with checkable items.
- Check in with the user before starting implementation.
- If something goes wrong, stop and re-plan — do not push through.

### Self-Improvement — Manual (no hooks available)
After any user correction, manually apply the lesson gate:
1. Read `tasks/lessons.md`
2. Is the lesson generalizable, actionable, and novel?
3. If yes: update or merge a rule. If no: do nothing.

### Verification
- Never mark a task complete without demonstrating it works.
- Ask: "Would a staff engineer approve this?" before presenting.

### Elegance
- For non-trivial changes: ask "is there a more elegant solution?"
- Skip for simple, obvious fixes.

### Autonomous Bug Fixing
- When given a bug report: diagnose and fix it directly. No hand-holding needed.

## Task Management

1. **Plan first** — write to `tasks/todo.md`
2. **Check in** — verify plan before implementation
3. **Track progress** — mark items complete; summarise at each step
4. **Log decisions** — append to `docs/decisions.md` using the template below
5. **Document results** — add a review section to `tasks/todo.md` when done
6. **Capture lessons** — update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity first** — minimal code impact
- **No laziness** — find root causes, no temporary fixes
- **Minimal impact** — touch only what is necessary

---

## Manual Skill Equivalents

These replace Claude Code skills that are not available here.

### Logging a lesson (`/lesson` equivalent)
1. Read `tasks/lessons.md`
2. Check 3 criteria: generalizable + actionable + novel
3. If passes: add or merge a rule in this format:
```
### Rule: [Short title]
**Trigger**: When [condition]
**Rule**: [Concrete behavioral change]
**Why**: [Root cause or correction that motivated this]
```

### Logging a decision (`/decide` equivalent)
Append to `docs/decisions.md`:
```
### [D-NNN] Short title
**Date**: YYYY-MM-DD
**What**: What was decided.
**Why**: The reason.
**Ruled out**: Alternatives considered and why rejected.
**Constrains**: What this locks in going forward.
```

### Reading MATLAB files (`/read-matlab` equivalent)
- `.m` files: read directly
- `.slx` files: `unzip -p model.slx simulink/systems/system_root.xml | python3 -c "import sys,xml.etree.ElementTree as ET; root=ET.fromstring(sys.stdin.read()); [print(b.get('BlockType'), b.get('Name')) for b in root.iter('Block')]"`
- See `docs/fp-model-structure.md` for the already-extracted reference — read that first before re-extracting

---

## Session Handoff

**When finishing a session** (context running low or work paused), write `tasks/handoff.md` before stopping. Tell the user it is ready.

**When receiving a handoff**, read `tasks/handoff.md` as step 3 of the session start sequence above.
