# Self-Improvement Ruleset

This is a **ruleset**, not an error log. A lesson is only added if it meets all three criteria:
1. **Generalizable** — applies to future tasks, not just one specific moment
2. **Actionable** — produces a concrete behavioral change ("when X, do Y instead of Z")
3. **Novel** — not already covered by an existing rule (if similar, strengthen the existing rule instead)

When a similar mistake recurs, merge it into the existing rule — do not append a duplicate.
When a rule fires repeatedly and proves its value, consider promoting it to `CLAUDE.md`.

---

## Rules

### Rule: Reference documents describe what exists — not what to do next
**Trigger**: When writing any reference or summary document about source files (code, models, data)
**Rule**: Document only what is in the source — structure, content, relationships. Do not mix in interpretation, next steps, implications for conversion, or design recommendations. Those belong in `docs/decisions.md` or task plans.
**Why**: User corrected `fp-model-structure.md` which included a "Key Observations for Python Conversion" section. That section was interpretation, not reference.

---

### Rule: Coordinate system choice is driven by data, not model structure
**Trigger**: When reasoning about which representation or coordinate system to use for a model that has multiple equivalent forms (related by a linear transform)
**Rule**: Do not argue that one representation is physically superior when both describe the same system. The deciding factor is which coordinate system the experimental data is in — model and data must match.
**Why**: Argued stage coordinates were better because "the physics is cleaner there." Supervisor correctly pointed out the model is coordinate-independent — only the data determines the choice.

---

### Rule: After any user correction, pause and evaluate for a lesson before continuing
**Trigger**: When the user corrects an output, rejects an approach, or asks for a redo
**Rule**: Before executing the correction, apply the 3-criteria gate and log to `tasks/lessons.md` if it passes. Do not defer it — corrections get forgotten once execution resumes.
**Why**: The fp-model-structure correction was not logged at the time because execution continued immediately. The lesson system only works if the checkpoint is non-optional.

---

### Rule: Never use em-dashes in any writing output
**Trigger**: When writing any text — prose, LaTeX, comments, documentation, or any other output
**Rule**: Do not use em-dashes in any form: not the Unicode character (—), not the LaTeX ligature (---), not double-dash (--) used as an em-dash substitute. Use a comma, colon, parentheses, or rewrite the sentence instead.
**Why**: User explicitly requested this as a hard rule across all writing contexts including LaTeX source.

---

### Rule: Describe proposed changes to documentation before writing them
**Trigger**: When about to write or substantially update `docs/decisions.md`, `tasks/handoff.md`, or any other shared documentation
**Rule**: Before executing the write, summarize the proposed changes and wait for user confirmation. Do not write directly without that checkpoint.
**Why**: User stopped a decisions.md edit because I wrote it without discussing the content first. Documentation changes are hard to review after the fact if the framing is wrong.

---

### Rule: Shape-only checks on outputs do not verify coordinate correctness
**Trigger**: When writing or reviewing a test for any block or function whose output undergoes a coordinate transform (P-transform, rotation, normalization, selection matrix)
**Rule**: Always add a value-correctness check alongside any shape check. The check must compare the actual values against a reference computed via the same transform. A shape check passing is not evidence the coordinate system is correct.
**Why**: `S_y` in `test_jan_compat.py` applied no P-transform for over a session. The y-output check only verified `y_out.shape == (batch, 3)`. The bug was invisible because: (1) shape was correct, (2) the Y-axis component is identical in both coordinate systems so single-channel inspection would also pass, (3) the simulation checks in `lfr_simulate` use a different code path (`simulate()` applies P correctly) and cannot catch Interconnect-level errors.
**How to apply**: For every output that touches a coordinate transform, write a check of the form: call the function, compute the expected result independently (using the known transform), assert values match within tolerance. Do not write a shape check and treat it as sufficient.

---

### Rule: Mathematical implications must be justified for the specific construction, not asserted as general facts
**Trigger**: When claiming "X implies Y" or "X if and only if Y" in a mathematical derivation
**Rule**: Always state explicitly why the implication holds for this specific construction. Do not assert it as if it follows from a general theorem unless it actually does. Show the connecting steps.
**Why**: Claimed "LFR is well-posed if and only if M(Y) is invertible" without justification. User correctly challenged this twice. The claim is true only because D_zw was specifically constructed to encode M(Y)^{-1} through the algebraic loop — so the loop's solvability reduces directly to M(Y) invertibility. For a different D_zw this would not hold. The justification is: substitute the specific D_zw into the algebraic loop, show it collapses to M(Y)·v = f_gen, then cite `LPV/supporting/derivations/M-invertibility.tex`. Without that connecting argument, the claim is unsupported.
**How to apply**: Before writing "X implies Y" in any derivation, ask: is this a general theorem, or does it hold only because of how I constructed the specific objects involved? If the latter, show the construction-specific steps explicitly.
