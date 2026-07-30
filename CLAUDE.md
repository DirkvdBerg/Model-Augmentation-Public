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

**Running scripts (MANDATORY live-output convention):** any script expected to run more than a few
seconds (training, sims, diagnostics) MUST be launched so its progress streams live and does not block:
- Launch with `run_in_background: true` AND unbuffered streaming:
  `PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject python -u <script>`.
  Plain `conda run` block-buffers stdout (nothing appears until the process exits); `--no-capture-output`
  + `python -u` + `PYTHONUNBUFFERED=1` makes each print appear live.
- Read the streamed output from the job's `.output` file (grep/tail for the signal lines), and tell the
  user that path so they can `! tail -n 30 "<path>"` it themselves.
- `conda run python -c` cannot take a multi-line `-c` arg (newline error): write the snippet to a temp
  file in the scratchpad and run that file instead.
- Quick (<~5 s) checks (compile/import) may run foreground; the convention above is for real runs.

## Key File Map
| What | Where |
|------|-------|
| **Control reasoning + project identity** (read before any modeling/training/diagnosis discussion) | `docs/control-reasoning.md` |
| Research plan (goals, 4 aspects, planning) | `presentations/Research_plan___Graduation_project_AI_ES___Feedback_Processed (3).pdf` |
| Augmentation training entry point | `scripts/gantry/gantry_interconnect_dynamic.py` |
| Baseline parameter recovery entry point | `lpv_lfr_baseline/scripts/train_param_recovery.py` |
| Literature PDFs (Jan's papers in `augmentation/`, projection in `Orthogonality/`) | `literature/` + `docs/references.md` |
| **Literature search procedure** (deep-research skill) | `.claude/skills/deep-research/SKILL.md` |
| FP model — MATLAB ground truth | `kamtin-fp-model/` |
| Augmentation framework | `model_augmentation/fit_systems/` |
| Reference benchmarks | `scripts/` |
| FP model structure | `docs/fp-model-structure.md` |
| FP ↔ augmentation interface | `docs/fp-augmentation-interface.md` |
| **Supervisor write-up + all its figures** (block scheme, training objective, normalisation; shared notation + figure style) | `docs/writeup/` (start at `docs/writeup/README.md`) |
| Design decisions log | `docs/decisions.md` |
| Session tasks | `tasks/todo.md` |
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

## Standing Rules (always active)
These are constraints, not suggestions (D-122).
- **Answer before code.** On a question, shared context, or a problem: respond in text first. No tools, no edits until the user confirms direction.
- **Modify only what was asked.** Only the files, and the parts within a file, explicitly requested. Flag stale siblings in text and ask.
- **Use the user's exact term.** Implement the concept/file/metric they named; never silently substitute a "more correct" sibling. "Don't touch `<file>`" means no reads and no writes on it.
- **Respect session boundaries.** When the user says work happens in a new/other session, prepare the handoff. Do not do that work here.
- **Commit after direction is given.** Once a direction is confirmed, execute. Do not re-request approval you already hold.
- **One recommendation.** On "what next", give exactly one concrete action with rationale, not an option menu.
- **No em-dashes** in any output: not the Unicode character, not `---`, not `--`. Applies to prose, LaTeX, code comments, and text inside figures.
- **Removing a rule needs justification.** An edit that deletes or weakens a user-authored rule must justify it in the same message, check the intent still holds, offer a softened rewrite, and flag the conflict of interest when it constrains your own behavior.

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
- **Subagents (codebase):** use an Explore subagent for broad fan-out searches (unknown location, many files, multiple naming conventions). Do not spawn for targeted lookups; inline search by the context-holding session is cheaper and better informed.
- **Literature / web deep research (D-121):** any request to find, survey, or fetch academic papers ("state of the art on X", "find papers on Y", "who cites Z", related-work sweeps) MUST invoke the `deep-research` skill (`.claude/skills/deep-research/SKILL.md`) — not ad-hoc `WebSearch`. This includes **document-driven** research ("read `docs/<file>.md` and research it", "is this novel", a pasted problem statement), where the skill's step 0 (FRAME) is mandatory before any query. Run it in subagents, one per independent seed paper or sub-question; a single lookup stays inline. Every run returns the skill's mandatory **Research Log**; its *Suggested skill fix* line is how the procedure gets revised. Rationale: keyword search measurably fails on control topics (control publishes in IFAC/CDC/ECC/ACC + Elsevier/IEEE, and authors rename concepts between papers), so the skill enumerates by author ID and citation edge instead of matching keywords.
- **Verification:** never mark a task complete without proving it works.
- **Run discipline (D-090):** every training run with a new hypothesis or config gets a row in the run table (`docs/gantry-augmentation-problem-log.md`, Section 12) BEFORE launch stating the hypothesis it tests; add the outcome after.
- **Task flow:** plan to `tasks/todo.md` -> check in with user -> implement -> mark complete -> log decisions to `docs/decisions.md`.
