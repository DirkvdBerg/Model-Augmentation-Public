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
