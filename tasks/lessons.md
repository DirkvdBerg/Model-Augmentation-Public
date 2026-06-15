# Self-Improvement Ruleset

This is a **ruleset**, not an error log. A lesson is only added if it meets all three criteria:
1. **Generalizable** — applies to future tasks, not just one specific moment
2. **Actionable** — produces a concrete behavioral change ("when X, do Y instead of Z")
3. **Novel** — not already covered by an existing rule (if similar, strengthen the existing rule instead)

When a similar mistake recurs, merge it into the existing rule — do not append a duplicate.
When a rule fires repeatedly and proves its value, consider promoting it to `CLAUDE.md`.

---

### Rule: Answer the question before writing any code

**Trigger**: When the user asks a question, shares context, or presents a problem to think through
**Rule**: Respond in text first. Do not call any tools or write any code until the user has acknowledged the explanation and explicitly confirmed the direction. A question is not an implicit "proceed". Sharing context or a problem is not an implicit "proceed". Saying "let me implement" at the end of a text reply and then proceeding in the same turn is also a violation if the user has not confirmed yet.
**Why**: Violated repeatedly. (1) User asked "what are you doing here?" and "what are you using this checkpoint for?" Both times immediately started coding. (2) User shared NotebookLM output and I started coding. (3) User asked "can't we log through SLURM?" and "where are you saving the files?" indicating the design was still being discussed. I said "Let me implement it" and started editing code in the same message without waiting for confirmation. (4) After a plan was approved, immediately launched into implementing ALL steps at once (creating tasks, reading files, writing code) without pausing to discuss Step 1. "Plan approved" means "discuss the first step," not "implement everything now." The rule must fire even when the conversation feels like it is converging on a solution. Implementation is one step at a time, with user confirmation between steps.

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

### Rule: Only modify files the user explicitly asked to modify
**Trigger**: When noticing that a related file is stale, incorrect, or inconsistent with new work
**Rule**: Do not modify any file that was not explicitly requested. Flag the inconsistency to the user in text and ask if they want it updated. This applies to all files: docs, notes, scripts, tests, everything.
**Why**: Updated sysid-experiment-design-notes.md because it was stale relative to the new inject_ref script — but user did not ask for this. Extended from earlier rule about docs: the problem is any unsolicited file edit, not just documentation.

---

### Rule: Shape-only checks on outputs do not verify coordinate correctness
**Trigger**: When writing or reviewing a test for any block or function whose output undergoes a coordinate transform (P-transform, rotation, normalization, selection matrix)
**Rule**: Always add a value-correctness check alongside any shape check. The check must compare the actual values against a reference computed via the same transform. A shape check passing is not evidence the coordinate system is correct.
**Why**: `S_y` in `test_jan_compat.py` applied no P-transform for over a session. The y-output check only verified `y_out.shape == (batch, 3)`. The bug was invisible because: (1) shape was correct, (2) the Y-axis component is identical in both coordinate systems so single-channel inspection would also pass, (3) the simulation checks in `lfr_simulate` use a different code path (`simulate()` applies P correctly) and cannot catch Interconnect-level errors.
**How to apply**: For every output that touches a coordinate transform, write a check of the form: call the function, compute the expected result independently (using the known transform), assert values match within tolerance. Do not write a shape check and treat it as sufficient.

---

### Rule: Diagnose test failures by reading code, not by running more commands
**Trigger**: When a test or smoke test fails during verification of completed implementation work
**Rule**: Read the relevant code and traceback to diagnose the cause. Do not run multiple follow-up test commands that each take minutes. Once the root cause is identified from code reading, report it and stop.
**Why**: User canceled after multiple slow test runs (~5 min each) that were diagnosing a pre-existing race condition. The cause was identifiable from reading lfr_simulate.py and the traceback in under 30 seconds.
**How to apply**: Traceback + 1-2 targeted file reads = sufficient diagnosis. Only run a second test command if it directly confirms/falsifies the hypothesis with a fast result.

---

### Rule: Verify that a theory rule's derivation context matches the application context before implementing it
**Trigger**: When implementing a numerical rule or threshold cited from literature or lecture notes
**Rule**: Check that the context in which the rule was derived matches the current application. A THEORY label is not sufficient if the rule comes from a different identification paradigm (e.g., FRF estimation vs. BPTT training). If contexts differ, flag it explicitly — do not implement the rule as if it applies.
**Why**: Implemented "N ≥ 10 × τ_set,95" segment length rule from Lecture 9, which is derived for non-parametric FRF estimation (transients must settle before corrupting spectral estimates). Applied it to BPTT gradient-based training, where the criterion has no direct equivalent. The result (15722 samples at 1000 Hz = 15.7 s) exceeded available trajectory lengths and was physically meaningless for training.
**How to apply**: Before writing `# THEORY: <source>` and implementing a formula, ask: "Was this rule derived for the same identification method being used here (FRF, PEM, BPTT, etc.)?" If not, flag the mismatch explicitly to the user before implementing.

---

### Rule: Use the user's exact technical term, do not substitute a related concept

**Trigger**: When a user names a specific technical concept (e.g. "additional unmodeled states", "reference injection", "state-space order")
**Rule**: Implement or document exactly that concept. Do not silently map it to a related but different concept (e.g. "unmodeled dynamics", "trust band on frequency"). If you think the user's term maps to something different, flag the mismatch explicitly and ask -- do not proceed with the substitution.
**Why**: User said "additional states not modelled" (state-space augmentation -- extra state variables the model does not have). Output substituted "unmodeled dynamics / trust band" (a frequency-domain concept). These are different: one is about model order, the other is about frequency coverage. The substitution was invisible and wrong.
**How to apply**: Before writing any design entry or code, re-read the user's exact words. If your output uses a different term, stop and verify equivalence first.

---

### Rule: In a nonparametric analysis, derive all outputs from empirical data — never from the parametric model

**Trigger**: When computing any output quantity (time constant, resonance frequency, bandwidth, sampling rate) in a context explicitly framed as nonparametric or empirical
**Rule**: Do not compute that quantity from the parametric model (e.g. eigenvalues of A_c, eig(A_c), model matrices). Derive it from the observed data (e.g. Ŝ(jω) from FFT(u_total)/FFT(f_sim), Ĝ(jω) from FFT(q1)/FFT(u_total), resonance peaks from |Ĝ|). This rule has been violated repeatedly — it is non-negotiable.
**Why**: User has corrected this mistake multiple times across the session. Every time a parametric fallback is suggested (eig(A_c) for fs_new, eigenvalues for f_osc_min, A_c for τ_max), it defeats the entire purpose: the nonparametric approach must work on hardware where the model is unknown or wrong.
**How to apply**: Before writing ANY formula or code in a nonparametric analysis step, check: does this reference a model matrix, eigenvalue, or analytical model quantity? If yes, STOP. Replace with the equivalent quantity read from Ŝ or Ĝ estimated from probe data. No exceptions.

---

### Rule: Do not assert a specific cause for a run failure without evidence from that run

**Trigger**: When diagnosing why a run stopped, crashed, or was killed (early termination, OOM, timeout)
**Rule**: Do not attribute the failure to a specific cause without direct evidence from the failing run itself (its log, its sacct record, its config). In particular, do not assume that an artifact the user shares (a script, config, or output) belongs to the failing run -- confirm which job/config actually produced the failure before building a diagnosis on it.
**Why**: Two incidents. (1) Assumed NaN caused training to stop at 5 epochs; it was an intentional test run. (2) Declared a `--mem=16gb` SLURM limit to be "the missing piece" explaining an OOM, but that sbatch script wrapped the diagnostic job, not the failed training job (which had `--mem=64gb`). Both diagnoses were presented confidently and were wrong.
**How to apply**: Before committing to a cause: ask which exact job failed, request its log tail / `sacct` record, and verify any shared script or config is the one used by that job. Until then, present causes as ranked hypotheses, not conclusions.

---

### Rule: When adapting a reference script, preserve its skeleton and append additions at the end

**Trigger**: When converting or adapting an existing reference script (e.g. Jan's ECC-2025 code) to a new system
**Rule**: Keep the original script's section order, variable naming style, and code structure intact. Make only the minimal changes needed for the new system. Add new sections (evaluation, plotting, extra saves) after the original script's final block (save). Do not restructure or rewrite the file from scratch.
**Why**: User could not verify the gantry adaptation because the new file looked nothing like Jan's original. A complete rewrite makes the diff invisible, removing the ability to catch errors by comparison.
**How to apply**: Before writing any adapted script, open the reference file and write the adapted version section-by-section in the same order. Mark each changed line with a comment if helpful. New additions go at the bottom.

---

### Rule: Mathematical implications must be justified for the specific construction, not asserted as general facts
**Trigger**: When claiming "X implies Y" or "X if and only if Y" in a mathematical derivation
**Rule**: Always state explicitly why the implication holds for this specific construction. Do not assert it as if it follows from a general theorem unless it actually does. Show the connecting steps.
**Why**: Claimed "LFR is well-posed if and only if M(Y) is invertible" without justification. User correctly challenged this twice. The claim is true only because D_zw was specifically constructed to encode M(Y)^{-1} through the algebraic loop — so the loop's solvability reduces directly to M(Y) invertibility. For a different D_zw this would not hold. The justification is: substitute the specific D_zw into the algebraic loop, show it collapses to M(Y)·v = f_gen, then cite `LPV/supporting/derivations/M-invertibility.tex`. Without that connecting argument, the claim is unsupported.
**How to apply**: Before writing "X implies Y" in any derivation, ask: is this a general theorem, or does it hold only because of how I constructed the specific objects involved? If the latter, show the construction-specific steps explicitly.

---

### Rule: Fix bugs where the incorrect assumption lives, not upstream

**Trigger**: When a class or function has a bug caused by a wrong assumption about its input convention
**Rule**: Fix the class/function itself, not the caller or the data pipeline. If a class assumes `input[-1]` means "current time" but the framework convention is "previous time," the bug is the class's wrong assumption, not the framework's convention. Do not work around it by changing how data is passed; fix the assumption at the source.
**Why**: Placed the hybrid encoder off-by-one fix in the training script (changing `na_right` and encoder constructor args) instead of in `HybridGantryEncoder.forward()`. User correctly pointed out: the deepSI convention is well-defined and Jan's encoder works with it; the encoder class was written with a wrong assumption about what `ypast[:, -1]` means. The fix belongs in the class.
**How to apply**: When diagnosing a bug, ask: "which component made the incorrect assumption?" Fix that component. Do not change the interface or callers to accommodate the wrong assumption.

---

### Rule: Verification/diagnostic tools must not require the thing they are verifying
**Trigger**: When building a test, diagnostic, or verification script for a component or pipeline
**Rule**: The diagnostic must be able to run independently of the fully completed pipeline. If the purpose is to verify component X works, the script must construct X from scratch and test it, not require a successful run of the full pipeline that includes X. For example: an encoder diagnostic must build and briefly train its own model, not load a fully trained model from a prior multi-hour run.
**Why**: Built an encoder diagnostic that required loading a trained model, but the whole point was to verify the encoder works before committing to a full training run. The user had to point out this was backwards twice.
**How to apply**: Before designing any diagnostic, ask: "Can I run this in minutes without any prior artifacts?" If the answer is no, restructure so the diagnostic is self-contained.

---

### Rule: Diagnostic results are data plus falsifiable plots, saved to the simulations output tree

**Trigger**: When presenting diagnostic or verification results as figures
**Rule**: (1) Save the underlying measurement data (JSON/CSV) and the figures under `simulations/<system>/diagnostics/`, not under `scripts/`. (2) Design each plot as a hypothesis test the viewer can judge for themselves: show the prediction and the independent measurement side by side with the quantified deviation. Do not assert the conclusion in the title; pose the test and let the data answer.
**Why**: Memory-diagnosis plots were written to `scripts/gantry/figures/` with conclusion-asserting titles ("not a leak"). User rejected this: data must go to `simulations/gantry_subnet/diagnostics` and plots must show whether the claim is correct or not.
**How to apply**: For each figure ask: "If my claim were wrong, would this plot reveal it?" If the plot only illustrates the claim, restructure it into prediction-vs-measurement with the error stated.

---

### Rule: Do not iterate on matplotlib for block diagrams; use text or TikZ instead

**Trigger**: When the user asks for a block diagram, architecture overview, or signal flow figure
**Rule**: Do not use matplotlib. It cannot produce readable block diagrams: text overflows boxes, labels overlap arrows, and each fix creates new collisions. After 5+ iterations the result was still unreadable. Instead, offer (1) a clear text/markdown description, or (2) a TikZ/LaTeX source file. Only use matplotlib for data plots (time series, spectra, loss curves).
**Why**: Spent an entire session iterating on matplotlib box-and-arrow code. Every fix (bigger boxes, moved labels, shaded regions) created new overlap problems. The user correctly called the result unreadable and asked for a text description instead.
**How to apply**: When the user asks for a diagram, respond in text first. If a figure file is needed, suggest TikZ or draw.io. Never start a matplotlib block diagram script.
