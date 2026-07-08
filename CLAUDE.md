# CLAUDE.md - Project Reference

## Code Quote Verification (MANDATORY)
Before quoting code from an external file with a line-number claim: write the intended quote
to `/tmp/quote.txt`, run `conda run -n GraduationProject python scripts/verify_quote.py <file> <start> <end> /tmp/quote.txt`,
and quote only after MATCH OK appears in the conversation. Never quote on MISMATCH; never use
Mode A (no quote file) for claims. The user can re-verify with `! python scripts/verify_quote.py ...`.

## Hard Constraints
- `kamtin-fp-model/` — READ ONLY, never modify. All Python must conform to it.
- Log design decisions in `docs/decisions.md` BEFORE implementing non-trivial choices.
- `scripts/` benchmarks (MSD, Bouc-Wen, Cascaded Tanks) are reference examples only — not the target system.
- When the user says "don't touch the code," do NOT edit or run ANY code. Only respond in text. Present the changes you would make and wait for explicit permission before writing or editing any file.

## Project Identity
Master thesis (TU/e Control Systems, AI&ES track, with ASMPT): "Model Augmentation for a
Dual-Gantry High-Precision Motion System". Supervisors: M. Schoukens, R. Toth, J. Hoekstra.
Goal: augment a physics-based LPV-LFR baseline (Garcia-Herreros dual-gantry, Y-scheduled
inertia) with a learned dynamic parallel component (Hoekstra LFR framework, SUBNET encoder)
while preserving physical parameter interpretability via orthogonal projection regularization
(Gyorok, extended here to LPV/MIMO/LFR: the thesis's scientific contribution). Validation:
free-run simulation, BFR, held-out Y positions and unseen motion profiles, vs a black-box
baseline. Full plan: `presentations/Research_plan___Graduation_project_AI_ES___Feedback_Processed (3).pdf`.
Current phase: `tasks/todo.md`.

Three pipelines (capabilities often exist in a sibling; check all three before building):
| Pipeline | Entry point | Role |
|---|---|---|
| `model_augmentation/` + `scripts/gantry/` | `scripts/gantry/gantry_interconnect_dynamic.py` | Jan's framework; augmentation training (encoder, ANN block, joint estimation) |
| `lpv_lfr_baseline/` | `lpv_lfr_baseline/scripts/train_param_recovery.py` | Baseline LPV-LFR sim + physical parameter recovery (windowed BPTT) |
| `scripts/gantry/real-data-verification/` | `run_telica_param_recovery.py` | Parameter recovery on real Telica data (closed-loop logs) |

## Control Engineering Stance
Reason as a critical control engineer first, ML engineer second. Before any
modeling/training/diagnosis answer, check what applies:
1. Loop: open- or closed-loop data, and what does the method assume?
2. Coordinates: stage vs logical frame (P-transform). Model and data must match; the data decides the frame.
3. Identifiability: identifiable from this excitation, or degenerate (mass vs force scale; only kb1+kb2, cb1+cb2, Jb+Jh are identifiable)? Present ALL physically consistent interpretations; never call one "impossible".
4. Excitation: does the input excite what must be learned (band, amplitude, Y coverage)?
5. Noise setting: noiseless simulation vs real data changes which arguments are valid; empirical-phase quantities come from data, never from model matrices.
6. Negation: can the augmentation absorb baseline dynamics? param_loss (Lambda) bounds parameter drift but does not prevent negation; only orthogonal projection does.
7. Well-posedness: M(Y) invertible over the full operating range, interconnection graph acyclic.
8. Thresholds: acceptance criteria must be data-derived and defensible (noise floor), never oracle/model-based.
Deep version with incidents and file pointers: `docs/control-reasoning.md`.

## Python Environment
Bash: `conda run -n GraduationProject python ...`. User manual: `conda activate GraduationProject`.

## Key File Map
| What | Where |
|------|-------|
| **Control reasoning + project identity** (read before any modeling/training/diagnosis discussion) | `docs/control-reasoning.md` |
| Research plan (goals, 4 aspects, planning) | `presentations/Research_plan___Graduation_project_AI_ES___Feedback_Processed (3).pdf` |
| Augmentation training entry point | `scripts/gantry/gantry_interconnect_dynamic.py` |
| Baseline parameter recovery entry point | `lpv_lfr_baseline/scripts/train_param_recovery.py` |
| Literature PDFs (Jan's papers in `augmentation/`, projection in `Orthogonality/`) | `literature/` + `docs/references.md` |
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
| **Telica data schema** (columns, signals, folder structure, **train/val/test split**, parameter + controller files) | `docs/kamtin-telica-schema.md` |
| **Gantry augmentation problem log** (K=0, horizon mismatch, all failure modes + diagnostics) | `docs/gantry-augmentation-problem-log.md` |

## Data Access Policy
- `kamtin-data/Data Telica/` is blocked via `.claudeignore` — **never attempt to read files there**.
- For all column names, signal meanings, unit conversions, and file structure: read `docs/kamtin-telica-schema.md`.
- Do NOT use `Bash` to cat/head/grep files in `kamtin-data/Data Telica/`.

## Multi-Agent Ownership
Claude: read/write on `tasks/`, `docs/`, `CLAUDE.md`, `.claude/settings.json`; read + propose only on `CODEX.md`. Codex: the reverse.
Neither ever writes `kamtin-fp-model/`. Proposals go in `tasks/handoff.md` under `### Proposed improvements for [Claude/Codex]`.

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
- **Subagents:** use an Explore subagent for broad fan-out searches (unknown location, many files, multiple naming conventions). Do not spawn for targeted lookups; inline search by the context-holding session is cheaper and better informed.
- **Self-improvement:** after any user correction, apply the 3-criteria gate (generalizable + actionable + novel) and update `tasks/lessons.md`. Merge, do not append duplicates.
- **Verification:** never mark a task complete without proving it works.
- **Run discipline (D-090):** every training run with a new hypothesis or config gets a row in the run table (`docs/gantry-augmentation-problem-log.md`, Section 12) BEFORE launch stating the hypothesis it tests; add the outcome after.
- **Task flow:** plan to `tasks/todo.md` -> check in with user -> implement -> mark complete -> log decisions to `docs/decisions.md`.
