# Session context (Dirk, 2026-10-08): what the rules must achieve

Read this before editing or judging anything. It is the brief from Dirk's own prompts; every rule change must serve it and must not contradict it.

## Goal of this run
Improve the writing rules: the thesis README (`rules/README.md`, copy of `Thesis-writeup/Writing/README.md`) and the thesis-section skill (`rules/skill/`, copy of `.claude/skills/thesis-section/`). Section drafts are only the test of the rules. A good draft produced by luck or by hand-fixing is not the goal; rules that make a fresh writer produce a good section are.

## Maarten Schoukens' feedback (supervisor, notes by Dirk, translated)
- The setup section is bad: too much is moved to a later setup section.
- The mathematics is very important to show how it works, but still written concisely. The math is what makes the thesis good. It is also a control thesis.
- Justify choices.
- Write more directly.
- Make clear which models are used on which problem, the cost function, the signals, for each phase: what really is the model. (Dirk explained his PS2 method; Maarten said the model was not clear and he needs to see the math in the thesis.)

## Dirk's direction in this session
- Experiment design replaces Setup. Training-specific content must not drift into Experiment design and make it messy. A compact table of the compared models stays in Experiment design for now.
- Concise: do not show every step; that is not what a thesis looks like.
- Not only formulas: prose carries the reasoning, and the section must be a logical, coherent whole, not a list of formula and reason blocks.
- Generally, throughout the thesis, the math must make things clear for the reader; describing the general process in words is not enough.
- Not model-only: "the model" was one example, not the scope. Do not hyper-focus on one example; find the balance.
- The thesis builds on Jan Hoekstra's framework, but it is a gantry-specific implementation, not a general framework paper. Example: for the added-state initialisation, show the math of how it works, do not derive a proof. Do not overstate either way (not only Jan, not a generic checklist of construction, assumption, condition, guarantee, metric).
- References: Hoekstra's papers are primary (PhD researcher; the thesis builds on his work directly). Drenth et al. 2025 (former master student of the group) is a secondary example of one concrete realisation. Other literature is cited where its component is used.
- Coherence: take inspiration from Hoekstra and Drenth. A (sub)section does not need to leave something open for the next one.
- The README must make sure a writer reads the correct files, code and papers for each (sub)section (the source map), verified against the mainline code, `docs/decisions.md` and references.
- The skill is quite good at concise writing; it is not good at writing a section the way Maarten wants. Do not weaken the concise rules.

## Rejected during the session (do not reintroduce)
- Completeness rules only about models (state, output, cost, parameters) as the whole standard.
- Generic claim-type checklists ("constructions, assumptions, conditions, choices, guarantees and their limits, metrics and acceptance criteria").
- Two-sentence rules too vague to check a draft against.
- A Drenth-only reference.
- "The previous section must leave something open."
- Moving training specifics or method explanations into Experiment design.

## Project rules that always hold
No em-dashes (not the Unicode dash, not ---, not --). Removing or weakening a user-authored rule needs a justification recorded in the changelog. Rules stay general to every section; the rules files must not grow without need (match length to substance, no filler).
