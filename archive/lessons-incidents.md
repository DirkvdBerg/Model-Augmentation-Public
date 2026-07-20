# Lessons — Incident Log (archive tier)

This is the **incident-evidence tier** of the self-improvement ruleset. It holds the full
`Trigger` / `Rule` / `Why` / `How to apply` text and all dated scope-extensions for every
lesson. The **active ruleset** (one-line rules, auto-read every session) lives in
`tasks/lessons.md`; each rule there carries a slug that maps to a section here.

Read this file only when you need a rule's backstory (the incident that produced it, the exact
wording, or a prior scope-extension). Do NOT auto-read it. When a rule is added or a scope is
extended, the narrative goes HERE and the one-liner goes in `tasks/lessons.md`.

---

# Self-Improvement Ruleset (full text, preserved 2026-07-16)

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
**Why**: Violated repeatedly. (1) User asked "what are you doing here?" and "what are you using this checkpoint for?" Both times immediately started coding. (2) User shared NotebookLM output and I started coding. (3) User asked "can't we log through SLURM?" and "where are you saving the files?" indicating the design was still being discussed. I said "Let me implement it" and started editing code in the same message without waiting for confirmation. (4) After a plan was approved, immediately launched into implementing ALL steps at once (creating tasks, reading files, writing code) without pausing to discuss Step 1. "Plan approved" means "discuss the first step," not "implement everything now." The rule must fire even when the conversation feels like it is converging on a solution. Implementation is one step at a time, with user confirmation between steps. (5) User shared supervisor meeting notes and asked "can you analyze my notes?"; I wrote and ran a measurement script mid-answer. Analysis requests want structured text (themes, goal, verification steps), not tool runs; even a "cheap, directly relevant" measurement is still unrequested execution. User: "Instead of just you starting to run things maybe better to analyze the current meeting notes, and structure them."
**Scope extension (2026-07-15) -- "I want to do X on the web / in tool Y" means WRITE THE INPUT for that external tool, not run a local equivalent**: when the user says they want to run something in an external tool ("a claude deep research on the web", NotebookLM, an external service), the deliverable is the PROMPT/input text for them to paste there. Do not launch a local skill/workflow that resembles it. Fired: user asked for "a claude deep research on the web"; I invoked the local deep-research workflow and the user had to interrupt ("No i ask for a prompt that ill give claude on the web").

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
**Scope extension (2026-07-15), FIGURE TEXT counts, and existing style is not license**: legend labels, titles, captions, and annotations written into matplotlib figures are writing output; the rule applies there too. Fired: new f03 legend lines were written with " -- " separators ("oracle ... -- discretization floor") and the user flagged them. That the surrounding file already used "--" in its titles/captions does not authorize adding more; when touching such a file, write new text clean and flag the existing occurrences for cleanup.

---

### Rule: Only modify files the user explicitly asked to modify, and only to the extent asked
**Trigger**: When noticing that a related file is stale, incorrect, or inconsistent with new work; or when a small authorized edit tempts a larger cleanup of the same file
**Rule**: Do not modify any file that was not explicitly requested. Flag the inconsistency to the user in text and ask if they want it updated. This applies to all files: docs, notes, scripts, tests, everything. The scope limit applies WITHIN a file too: authorization for a specific addition (e.g. "document the split") is not authorization to rewrite the rest of the file, fold in pending corrections, or add related files. If the message also contains a discussion request, give the discussion first; do not lead with the file work.
**Why**: (1) Updated sysid-experiment-design-notes.md because it was stale relative to the new inject_ref script, but user did not ask for this. (2) User asked to "discuss train/validation/test" plus "document the supervisor's split folder and reference it in CLAUDE.md"; response rewrote the entire kamtin-telica-schema.md (controller tables, timing notes, signal semantics), moved a script into the repo, and put the discussion last. User reaction: "I have no clue wtf youre doing." A previously declined offer ("want me to update the stale doc?") does not become authorized by a narrower later request.

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

### Rule: Verify that a theory rule's derivation context matches the application context before implementing or citing it
**Trigger**: When implementing a numerical rule or threshold cited from literature or lecture notes, or when presenting literature findings as design recommendations in discussion
**Rule**: Check that the context in which the rule was derived matches the current application. A THEORY label is not sufficient if the rule comes from a different identification paradigm (e.g., FRF estimation vs. BPTT training). The noise setting is part of the context: noise-motivated arguments (SNR budgets, averaging over periods or realizations, BLA variance estimation) do not transfer to noiseless simulation data, where repetition adds zero information and realizations matter only for split independence and coverage diversity. If contexts differ, flag it explicitly, do not present the rule as if it applies.
**Why**: (1) Implemented "N ≥ 10 × τ_set,95" segment length rule from Lecture 9, derived for non-parametric FRF estimation, in a BPTT training context; result was physically meaningless. (2) Repeatedly presented noise-based arguments (SNR levels, period averaging, realization variance) as design drivers for the gantry data generation although the current phase is noiseless Simulink simulation; user had to correct the frame explicitly.
**How to apply**: Before writing `# THEORY: <source>`, or citing a literature finding as a recommendation, ask: "Was this derived for the same identification method AND the same noise setting as here?" If not, flag the mismatch or omit the finding.

---

### Rule: Use the user's exact technical term or specified procedure, do not substitute a related one

**Trigger**: When a user names a specific technical concept OR specifies a procedure/metric (e.g. "validation must use the same measure as training"), and a related-but-different one is easier to implement
**Rule**: Implement or document exactly that concept/procedure. Do not silently map it to a related one, and do not substitute an "equivalent" because it already exists in the framework. If substituting, flag it as a RISK in chat at decision time (a note in decisions.md is not enough) and get explicit agreement; when the substitution's assumption later fails, revert to the user's spec rather than patching around it.
**Why**: (1) User said "additional states not modelled" (state-space augmentation); output substituted "unmodeled dynamics / trust band" (frequency-domain). (2) User specified "validation for training should use the same method as training" (windowed loss on held-out data); implementation kept the framework's full-trajectory open-loop RMSE instead (deviation noted only inside D-075). On real data that metric is dominated by friction drift no parameter can reduce, so it degraded monotonically, collapsed the LR scheduler to 1e-5 by epoch 360, and made checkpoint selection meaningless (run 68775). The user's specified selector would not have had this failure mode.
**How to apply**: Before writing any design entry or code, re-read the user's exact words. If your output uses a different term or procedure, stop, state the difference in chat, and get agreement first.
**Scope extension (2026-07-12) — named DATA FILES count too; a fuzzy name is a question, not a license**: when the user names a file even approximately ("the telica1 mat file", "the mat file we read in for the controller"), first locate THAT file by search; if nothing matches the name, SAY SO and ask which file is meant. Do not silently substitute a similarly-named sibling and analyze it. Fired: user said "telica1" and later "the mat file we read in for the controller"; I inspected `Telica.mat` instead across two turns and presented conclusions from it, and the user had to stop me ("wtf i said telica1 not telica.mat dont touch that file"). Also: "don't touch <file>" means stop ALL access (read and write) to that file until told otherwise.
**Scope extension (2026-07-15) — documented FOLDER/PLACEMENT conventions count too; do not carve exceptions out of ambiguous phrasing**: when a written convention exists (here: gantry-zero-mean README, "every script written for this verification lives in THIS folder"), apply it to every new artifact including other-language ones (MATLAB), unless the user explicitly relocates it. A user message that merely MENTIONS another folder ("and the matlab-scripts/augmentation folder") is a request to REFERENCE it, not to move work there; if placement seems ambiguous, follow the documented convention or ask. Fired: placed the v1d MATLAB script in `Matlab-scripts/Augmentation/` and wrote a new "MATLAB-side convention" into the README to justify it; user: "it should be located in this folder not in matlab-scripts".
**Scope extension (2026-07-11) — project-DOCUMENTED concepts count as the user's terms**: when proposing a solution element in a project with a written design corpus (here: drift-diagnosis-status.md §5x, data-silent-regularization-concept.md, all-five-construction-spec.md), use the DOCUMENTED mechanism name and re-read its documented entry (including its demotion/limits status) BEFORE proposing it. Do not improvise a new label ("sim-phase DC pin") for a mechanism the docs already name, and do not resurrect a documented-as-demoted mechanism without explicitly citing and addressing its demotion. Fired: proposed a "sim-phase DC pin" as the next fix; the docs demote the mean-penalty family (§5, fails knowledge-free) and specify Layer 2 as the data-silent projection — user: "are you reading the relevant documents? i feel you dont have the proper research and context of what we've written down."

### Rule: Never assert computational cost ("cheap", "fast", "affordable", "~Nx faster") without a measured basis; the user has the hardware context
**Trigger**: When about to describe a run/config/method as cheap, fast, affordable, lightweight, or quantify a speedup
**Rule**: Do not attach cost/speed adjectives or multipliers to a config unless you have a measured number (a prior run's sec/batch, memory print) or the user gave one. Increasing the BPTT horizon (nf) is EXPENSIVE at every scale that matters here: nf=4000 = 566 MB and infeasible; nf=2000 (0.5 s) is NOT cheap either. When cost matters, ASK the user what is runnable on their hardware, or cite the measured number from a prior run -- never guess. The user runs the jobs and knows the real cost; asserting "cheap" when it is not wastes their time and erodes trust.
**Why**: In the nf-sweep episode I repeatedly labelled heavy runs "cheap/affordable/~10x faster": called stride=100 "~10x faster", called an nf=2000 (0.5 s BPTT, hundreds of MB) run "the cheap/affordable point", and pitched a full nf {2000,3000,4000} sweep as a good server run -- then nf=4000 came back at 566 MB and infeasible, and the user (furious) corrected that nf=2000 is not cheap. Every cost claim was ungrounded.
**How to apply**: State the config change without a cost label; if cost is relevant, quote the measured sec/batch or MB from a prior run, or ask "what nf/batch is runnable on your setup?". Prefer approaches that keep the EXPENSIVE knob (BPTT length/nf) at the already-runnable value and add cheaper per-step terms -- but still do not call those "cheap" without measuring.

**Scope extension (2026-07-11) -- when the user NAMES a specific script/tool, USE IT; do not substitute a "more correct" sibling**: When the user repeatedly points to a specific script to run (here: "the optuna script"), configure and use THAT script. Do not insist on a sibling you judge more appropriate. The user named `gantry_optuna.py` three times; I kept pushing the main entry point `gantry_interconnect_dynamic.py` "because it's a single run" -- but the optuna script is OPTIMIZED FOR SPEED (cropped 8000-sample validation + pruning + large stride), while the entry point runs all 4 FULL validations + baselines + post-training diagnostics (~10 min/validation). For a quick de-confound test, the user's named script was correct and mine was 10x+ slower. A single run is trivially done by the optuna script with N_TRIALS=1 and a fixed (LR_LOW=LR_HIGH) lr. When a user names a runnable artifact, the burden is to make THAT artifact do the job, not to argue for a different one. Check WHY they named it (speed/cropped val here) before overriding.

---

### Rule: When the user wants a next step, commit to ONE recommendation; documents are not progress

**Trigger**: When the user asks "what should we do next" / wants a workable next action, or when tempted to write another analysis/plan/options document during a diagnosis or search phase
**Rule**: Give a single concrete recommendation with its rationale and stop. Do not present menus of options, layered plans, or comparison tables unless the user asks for alternatives; a menu offloads the decision back onto the user and buries the signal. And do not write a new document as a substitute for the next runnable step or the next primary-read: producing volume (a dozen docs) feels like work but is not progress toward the thing needed.
**Why**: In the drift-diagnosis sessions the user asked for one workable step and received walls of options, tables, and layered plans, plus ~a dozen documents; the user's retrospective named "menu-dumping instead of committing" and "volume instead of progress" as core failures that ended the session.
**How to apply**: Before ending a "what next" reply, check: does it contain exactly one recommended action? Before creating any new doc, ask: does this replace or delay the actual next step (a run, a primary-read)? If yes, do the step instead.

---

### Rule: In an empirical/real-data context, derive all verification quantities from the real data, never from the parametric model

**Trigger**: When computing any output quantity (time constant, resonance frequency, bandwidth, sampling rate) in a context framed as nonparametric or empirical; OR when designing a diagnostic/tuning procedure for a real-data pipeline (hyperparameters, window lengths, learning rates, identifiability); OR when defining an acceptance threshold or noise floor that must be defensible in the thesis or transfer to hardware
**Rule**: Do not compute that quantity from the parametric model (eigenvalues, model matrices), and do not ground a diagnostic's verdict in model-generated synthetic data. Derive it from the observed data (empirical FRFs, loss slices on real data, descent behavior on real data). A synthetic self-test answers "is my loss setup sound under a correct model class", which is NOT the question in a real-data verification phase; the model class is known to be wrong there (that is the point). This rule has been violated repeatedly; it is non-negotiable.
**Why**: User has corrected this mistake multiple times. (1) Parametric fallbacks suggested repeatedly in the nonparametric experiment-design phase (eig(A_c) for fs_new, A_c for tau_max). (2) Proposed a window/learning-rate diagnostic centered on synthetic self-test data (model simulating itself) during the Telica real-data verification; user: "we should use the real data, that's the entire point of this verification with real data." (3) Proposed an oracle-model floor (baseline vs oracle NRMS) to set the noise level / acceptance threshold, and re-offered it as a fallback even after noting a model-derived reference is less defensible; user: "i dont think i can defend your oracle method ... determine it from the data." An acceptance threshold or noise floor must come from a data-derived, standard estimate (measured noise), never from the true/oracle model, which does not exist on hardware.
**How to apply**: Before writing ANY formula, code, or diagnostic design in an empirical phase, check: does this reference a model matrix, eigenvalue, analytical model quantity, or model-generated data as ground truth? If yes, STOP and replace with the equivalent measurement on the real data (loss sensitivity/slices, empirical FRF, descent probes). No exceptions.

---

### Rule: A degenerate identification has multiple physical explanations; never call one "impossible"

**Trigger**: When a fit/identification determines a lumped quantity that could arise from more than one physical cause (e.g. F = m*a is degenerate between mass scale and force scale), and one interpretation seems obviously wrong
**Rule**: Present ALL physically-consistent interpretations and state that the data alone cannot distinguish them; name the external reference that would (a datasheet, a mechanical drawing, a units definition, an independent measurement). Do NOT declare a value "impossible" or pin the result on your preferred cause. In particular, do not assume the FP-model parameters equal the real machine's: the model may describe a DIFFERENT or earlier system, so a recovered parameter far from the model's nominal can simply be the real machine's true value (i.e. parameter recovery working, not a bug).
**Why**: The Telica linear identification found the effective moving mass at ~half the FP-model nominal (26 vs 54 kg, consistent across 11 datasets). Output declared the real mass being half "impossible (it's a known machine)" and pinned it on a ~2x force-units bug. User corrected: "this is not impossible" and "different system" -- the kamtin-fp-model masses come from main.m (a simulation of possibly a different gantry) and need not match the real Telica hardware. F = m*a cannot separate "real mass is lighter" from "force under-scaled"; asserting one over-commits and, worse, frames a possibly-correct recovery as a failure.
**How to apply**: When a recovered value deviates strongly from the model nominal, list the interpretations (real parameter differs / input scale wrong / structural), say the fit cannot decide, and ask for the external reference (machine spec, units definition). Treat "recovered != model nominal" as possibly the model being wrong for this system, not automatically a data bug.
**Scope extension (2026-07-16) -- a trajectory/residual PLOT identifies the error CLASS, not the mechanism; separate evidence by source and strength**: a position-domain residual that is low-frequency/drift-like only tells you the error is force/DC-level (rules OUT unmodelled fast dynamics/resonances); it does NOT single out a mechanism, because Coulomb friction, force-scale error, mass error, and open-loop replay-of-closed-loop-data drift ALL produce low-frequency drift. Do not say a plot shows "mostly unmodelled friction." In particular a hysteresis LOOP in residual-vs-velocity is NOT clean friction evidence: any model/measurement phase lag (slightly wrong damping/frequency) also traces a loop. State which evidence is model-based vs plot-based: here the real friction signal is the PARAMETER RECOVERY (viscous damping pinned at 6-7x the datasheet maximum), which is circumstantial, and the discriminating test is either a force-domain friction curve (inverse dynamics) or adding the Coulomb term and checking whether the damping runaway resolves and NRMSE drops. Fired: attributed the 70821 open-loop mismatch to friction leaning on the trajectory overlay and the residual-vs-velocity loop; user: "based on what from the plot can you say its mostly unmodeled friction?"

---

### Rule: Do not assert a specific cause for a run failure without evidence from that run

**Trigger**: When diagnosing why a run stopped, crashed, or was killed (early termination, OOM, timeout)
**Rule**: Do not attribute the failure to a specific cause without direct evidence from the failing run itself (its log, its sacct record, its config). In particular, do not assume that an artifact the user shares (a script, config, or output) belongs to the failing run -- confirm which job/config actually produced the failure before building a diagnosis on it.
**Why**: Three incidents. (1) Assumed NaN caused training to stop at 5 epochs; it was an intentional test run. (2) Declared a `--mem=16gb` SLURM limit to be "the missing piece" explaining an OOM, but that sbatch script wrapped the diagnostic job, not the failed training job (which had `--mem=64gb`). (3) Analyzed run 70799 as the intended lr=1e-3 capability test and drew sweeping conclusions ("gradient starved even at maximal lr; no lr fixes this") -- the log's own Configuration header said `lr: 1e-07`; the deployed entry file lagged the local edit. All diagnoses were presented confidently and were wrong.
**How to apply**: Before committing to a cause: ask which exact job failed, request its log tail / `sacct` record, and verify any shared script or config is the one used by that job. Until then, present causes as ranked hypotheses, not conclusions. MANDATORY first step when reading ANY run log: read its printed Configuration/hyperparameters block (lr, flags, routing) and check it against the intended config -- deployed copies lag local edits; the check costs five seconds.

---

### Rule: Diagnose "not learning" from the loss trajectory shape; lr-overshoot != step-starvation, and optimal lr is routing/architecture-dependent

**Trigger**: When a model "isn't learning" (val not improving) and you are about to prescribe a fix (more data, more epochs, more gradient steps)
**Rule**: First read the TRAINING-loss trajectory to classify the failure before prescribing. If the train loss RISES or bounces after the first step, that is learning-rate OVERSHOOT -> lower lr; adding data/steps will not help and wastes runtime. If the train loss is flat/slowly-decreasing and monotone, that is step/data starvation -> add steps/data. Do not default to "too few steps" when the loss is increasing. Also: the optimal lr is routing/architecture-dependent -- a lr that trains one configuration can overshoot another. In particular, routing an augmentation to free-integrator (K=0) states produces larger gradients, so that configuration needs a smaller lr than a spring-restrained one. When a just-established knob (here: lr, freshly un-bugged in D-101) is in play, re-tune it for the new configuration before blaming the data.
**Why**: Theta+X+Y "wasn't learning" at lr=1e-5 (val bounced ~1e-3, train loss rose 0.0015->0.005). I diagnosed "only ~12 gradient steps, too underpowered" and proposed more data / smaller stride / more epochs. The user instead just lowered lr to 1e-7 and it learned cleanly with the SAME 12 steps: val sim-RMS monotone 1.2e-4 -> 9.1e-5, new low every epoch, train loss flat. The real cause was lr overshoot (X/Y K=0 routing needs ~100x smaller lr than Theta's 1e-5), not step starvation. The rising train loss was the tell and I read past it. Ironic given we had JUST fixed the lr plumbing (D-101) and swept lr for Theta -- I should have re-swept lr for the new routing first.
**How to apply**: On "not learning", look at train loss first: rising/bouncing -> lower lr (and re-tune lr per configuration); flat/slow -> add steps/data. Only prescribe data/epoch changes once overshoot is ruled out.

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

### Rule: Verification/diagnostic tools must not require the thing they are verifying, and must use the minimal evaluation path
**Trigger**: When building a test, diagnostic, or verification script for a component or pipeline
**Rule**: (1) The diagnostic must be able to run independently of the fully completed pipeline. If the purpose is to verify component X works, the script must construct X from scratch and test it, not require a successful run of the full pipeline that includes X. (2) Use the minimal evaluation that tests the component directly. Do not route through the full pipeline when the component can be called in isolation. For example: to test encoder init quality, call the encoder on I/O windows and compare to ground truth states. Do NOT run `apply_experiment` (full sequential model simulation), which is 1000x slower and tests the model+encoder together, not the encoder alone.
**Why**: (1) Built an encoder diagnostic that required loading a trained model, but the whole point was to verify the encoder works before committing to a full training run. (2) Used `apply_experiment` (200k sequential timesteps) to test encoder initialization quality at different sampling rates. The encoder init is a linear map from I/O windows to states, testable with a single matrix multiply in milliseconds. The full simulation made the 20 kHz evaluation hang for minutes.
**How to apply**: Before writing any evaluation, ask: "What is the minimal call that tests THIS component?" If you are importing or calling machinery beyond the component under test, you are doing it wrong.

---

### Rule: Diagnostic results are data plus falsifiable plots, saved to the simulations output tree

**Trigger**: When presenting diagnostic or verification results as figures
**Rule**: (1) Save the underlying measurement data (JSON/CSV) and the figures under `simulations/<system>/diagnostics/`, not under `scripts/`. (2) Design each plot as a hypothesis test the viewer can judge for themselves: show the prediction and the independent measurement side by side with the quantified deviation. Do not assert the conclusion in the title; pose the test and let the data answer.
**Why**: Memory-diagnosis plots were written to `scripts/gantry/figures/` with conclusion-asserting titles ("not a leak"). User rejected this: data must go to `simulations/gantry_subnet/diagnostics` and plots must show whether the claim is correct or not.
**How to apply**: For each figure ask: "If my claim were wrong, would this plot reveal it?" If the plot only illustrates the claim, restructure it into prediction-vs-measurement with the error stated.
**Scope extension (2026-07-14) -- audience-first figure design [fired on the drift-demo F1]**: a supervisor-facing figure must ALSO pass three checks beyond falsifiability. (1) FIRST-THOUGHT TEST: anticipate the naive reading of each line; if a line raises a question the figure does not answer (F1's residual-at-true-x0 line settles at 17x the absorber scale on Y -> first thought "the model or input is bad"), either answer it IN the figure set (a companion showing the model matches the I/O at trajectory scale) or remove the line from the lead figure. (2) LINE BUDGET: <= ~3 curves per panel with SHORT labels; long explanations go to the caption/doc, not the legend -- more lines = less informative. (3) ORIENTATION: show the ABSOLUTE trajectory (what the system actually does) before/alongside error plots, so the viewer knows what signal the errors are relative to. Plan these properties in the figure spec BEFORE building; the user explicitly asked for pre-planned plots and the built version still failed these checks. (4) LAYOUT-COLLISION pass (2026-07-14): critiquing or delivering a figure requires a pixel-level visual inspection, not only a content read. Crop and magnify suspect regions (annotations near curves, insets, titles, panel gaps) and check: no text struck through by a line, no inset occluding data, no clipped/truncated titles, visible separation between stacked panels. Fired: after a critique pass the user still had to ask "can you also visually inspect the figures? some figures just overlap without clear seperation." (5) SELECTION DISCLOSURE (2026-07-14): when a figure shows one channel/row/run out of several that the data contains (e.g. only the Y force row of 8 state rows), that selection is itself a conclusion. Either show all members (small multiples, dot summary) or state the selection ON the figure with the omitted members' behavior quantified from data ("Y shown, largest; other rows: ..."), never only in a footer assertion the viewer cannot check. Fired: user on the zero-mean figures: "we already made a conclusion ourselves/dont give all the information. I dont like that." User policy (2026-07-14): FULL SHOW is the default, not disclosure text; if one figure cannot hold all members, split into more figures.
**Frame check before per-channel claims (2026-07-14)**: before quoting a per-channel number (or labeling a channel "X/yaw/Y"), confirm which FRAME the stored array is in (stage vs logical) and apply the P-transform if needed. Fired: read `f2_dc.npz e_full` (STAGE output, 3 sensor channels) and reported "X drifts -2.5 mm, yaw -2.5 mrad"; the two ~-2.5e-3 columns were the two stage X-encoders (common-mode = logical X), and `stage_pos_to_logical` showed yaw is actually bounded at ~12 urad (sprung, does NOT drift). This is CLAUDE.md control-reasoning #2 (stage vs logical). One convert costs seconds; a wrong per-channel claim to the supervisor is worse.
**Scientific notation is the y-axis default (2026-07-14)**: on every supervisor-facing data figure, format y-axes in scientific notation (use `demo_common.sci_axes` / matplotlib `ticklabel_format(style='sci', scilimits=(0,0))`). These quantities are 1e-6..1e-2 scale; plain-decimal ticks are unreadable. User: "start using scientific notation." Applies to all redesigned drift-demo figures and any new one.
**Full-show applies to the FORCE rows too, not just the drift channels (2026-07-14)**: when showing the learned ANN force, show ALL velocity/rate channels (dX, dTheta, dY, vdelta_a), never only dY because it is the headline. One panel per channel in its OWN physical unit (force mN on dX/dY, torque mN·m on dTheta, absorber force on vdelta_a; the rows share no common physical unit, same reason G3 stays normalized), with the per-channel mass/inertia scaling pulled from the model, not one mass assumed for all. Position rows (X/Theta/Y/delta_a, ~10-100x smaller mean) go to a backup. User: "i dont like only dY then we should create figures for all velocities." This is [[selection-disclosure]] / FULL SHOW (rule above) applied to the force object.
**FULL-SHOW governs the PROPOSAL/description, not just the built PNG (2026-07-14, fired 3x)**: when describing ANY drift-deck figure in a plan/table/proposal, enumerate its COMPLETE member set; never write "X/Theta/Y" as a stand-in for "the channels." The default object here is the FULL STATE: 6 physical states (X, Theta, Y, dX, dTheta, dY) in BOTH frames (logical X/Theta/Y + stage X1/X2/Y), positions AND velocities. `g_correction_channels.png` (6 states x 2 frames) is the accepted standard. Writing "X/Theta/Y" in a figure spec silently drops the velocities and the stage frame and reads as filtering; the user reacted "wtf you keep messing up the figures ... you only show x theta y and not state?" Before proposing any figure, list every channel it will contain (all states, both frames) explicitly; if a figure legitimately shows only positions (e.g. the drift counterfactual, since velocities do not drift), SAY "positions only, because ..." rather than leaving it implicit.
**FULL-SHOW = MORE PANELS, never more curves on one axis; different physical STATES must not share a y-axis (2026-07-14, fired on f03)**: one physical state per panel. If honouring full-show would force X/Theta/Y (different units and scales) onto the same axes, SPLIT into more figures instead of overlaying. Overlaying distinct states on one axis is unreadable and was rejected outright ("we DONT overlay the different channels for the states in the same figure ... just create more figures"). This is DISTINCT from the "<=3 comparison curves per panel" rule: there the overlaid curves are the SAME quantity under different methods (trained-ANN / DC-removed / ANN-off on ONE channel = legit); here they are DIFFERENT quantities (X vs Theta vs Y = never). Small-multiples must fan out over (state x frame) as separate panels, with methods overlaid WITHIN a panel, not states.
**Trajectory reference and baseline-quality are SEPARATE figures; never overlay absolute measured-vs-baseline as the orientation plot (2026-07-14, fired on f02)**: an absolute-position overlay of measured vs baseline reads as a trajectory-scale (~1e-2) ERROR even when the baseline is near-perfect (the eye sees the gap between two large curves, not their near-coincidence). Show ORIENTATION as a pure trajectory reference (the absolute signal alone, both frames X/Theta/Y + X1/X2/Y). Show baseline QUALITY as an ERROR plot (baseline-no-ANN minus truth, near the floor), and include BOTH the true-x0 and the encoder-x0 initialisations so the encoder-IC contribution is visible. Do not conflate "what the system does" with "how well the baseline matches" in one panel.

---

### Rule: Validate with the actual pipeline model and representative data, not a simplified proxy

**Trigger**: When building a diagnostic or validation for a pipeline parameter (sampling rate, window length, integration steps, etc.)
**Rule**: Use the same model the pipeline actually uses (e.g., LPV nonlinear), not a simplified proxy (e.g., LTI linearization). Choose test data that exercises the full operating range, not data tailored to make the proxy valid.
**Why**: Built a downsampling validation using the LTI linearization at Y_op=0, then chose a Y=0 trajectory to match the linearization point. The real pipeline uses the LPV model with Y-sweeping data. The LTI test missed the coupling between Y-scheduling and sampling rate entirely.
**How to apply**: Before writing a diagnostic, ask: "Which model does the pipeline use? Does the test data cover the operating range the pipeline sees?" If the diagnostic uses a different model or narrower data, it is not testing the right thing.
**Also (discretization parity for reference/comparison models)**: A reference model added to the pipeline for comparison (oracle, baseline FP, any "best-case" line) MUST run at the exact same sampling rate and integration substeps (`up_sample`) as the models it is benchmarked against, or the comparison is unfair (discretization error masquerades as a model-quality gap). Only a STANDALONE correctness diagnostic may use finer discretization (native rate, higher `up_sample`) to isolate model error from discretization error. When a standalone check passes at fine settings, re-verify it still holds at the actual pipeline settings before wiring the model in. User: the pipeline oracle's `up_sample` and sampling rate must match `gantry_interconnect_dynamic.py` (up_sample=2, 4 kHz); the diagnostic may crank them.
**Scope extension (2026-07-12) — test inputs must come from the actual producer, not be hand-built to the consumer's assumption**: when verifying a function that consumes pipeline data, obtain the test inputs by CALLING the actual producer (or reading its source for the exact contract), never by fabricating tensors shaped to the consumer's own assumption. Fired: the D-076 windowed validation selector assumed a full-length `state_traj`; the shape test fabricated a (T,6) `state_traj` and passed, but the real producer `load_eval_trajs` stores only `(1,6)` ("only x0 needed"), so the second window sliced an empty tensor and crashed the cluster run at epoch 0. The test verified the assumption against itself.

---

### Rule: Model discretization accuracy and encoder/observer quality have separate sampling rate requirements

**Trigger**: When choosing a sampling rate for training based on a downsampling or discretization validation
**Rule**: A passing downsampling validation (model simulation NRMS < threshold) only confirms the state-transition model is accurate at that rate. It does NOT confirm the encoder initialization, observer, or state reconstruction will work at that rate. Before committing to a sampling rate, verify BOTH: (1) model discretization accuracy (downsampling validation), and (2) encoder initialization quality (evaluate state NRMS at that rate before training). These can have very different cutoff rates. The encoder init quality threshold must be derived from the native-rate (e.g. 20 kHz) encoder init as a reference baseline, not from an arbitrary number. Sweep sampling rates for encoder init, compare each against native rate, and pick the rate where degradation is acceptable.
**Why**: (1) Downsampling validation showed 200 Hz passes (0.82% NRMS). Committed to 200 Hz for the full diagnostic sweep. Encoder init at 200 Hz produced q2 NRMS = 2.078 (unusable), while at 400 Hz it was 0.052. Wasted a diagnostic run and had to backtrack. (2) Then used an arbitrary threshold of 0.5 for the pre-flight gate instead of comparing against the native-rate encoder init quality. The proper reference is always the native-rate result.
**How to apply**: First compute encoder init at native rate (the gold standard). Then sweep downsampled rates and report quality relative to native. Do not invent thresholds.

---

### Rule: Verify code ownership against git history before placing @added markers

**Trigger**: When marking a class or function with `@added` in `model_augmentation/`
**Rule**: Before adding `@added`, check that the class did NOT exist in the "Revert back to Jan's original code" commit (`6d69f6b`). Run `git show 6d69f6b:model_augmentation/...py | grep "class Foo"` to confirm. Do not rely on the current state of the file -- markers can be placed incorrectly.
**Why**: Marked `linear_encoder_init` as `@added` (our code) when it already existed in Jan's original. The error was only caught when the user pointed it out.
**How to apply**: For any file in `model_augmentation/`, check the 6d69f6b commit before placing any ownership marker.

---

### Rule: Do not iterate on matplotlib for block diagrams; use text or TikZ instead

**Trigger**: When the user asks for a block diagram, architecture overview, or signal flow figure
**Rule**: Do not use matplotlib. It cannot produce readable block diagrams: text overflows boxes, labels overlap arrows, and each fix creates new collisions. After 5+ iterations the result was still unreadable. Instead, offer (1) a clear text/markdown description, or (2) a TikZ/LaTeX source file. Only use matplotlib for data plots (time series, spectra, loss curves).
**Why**: Spent an entire session iterating on matplotlib box-and-arrow code. Every fix (bigger boxes, moved labels, shaded regions) created new overlap problems. The user correctly called the result unreadable and asked for a text description instead.
**How to apply**: When the user asks for a diagram, respond in text first. If a figure file is needed, suggest TikZ or draw.io. Never start a matplotlib block diagram script.
**Also (embed-size readability)**: Even a clean TikZ figure can be unreadable once embedded. Before calling a figure done, render it at the exact width it will occupy in the target document and confirm labels are legible. A figure that reads fine standalone can be too small at text width, especially if it carries a wide equation legend or large whitespace margins. Trim in-figure legends (the prose already states the equations) or increase the box/font scale so the diagram fills the embed width. User: "i cant even read the figure" after the block scheme was declared done and embedded at `\linewidth`.

---

### Rule: Keep operational scaffolding out of experiment scripts

**Trigger**: When adding run-management tooling (test hooks, env-var modes, rehearsal switches) to a script whose purpose is a scientific experiment
**Rule**: Do not embed operational hooks in the experiment file, and do not remove or alter existing user-visible behavior (progress bars, log output, prints) based on own judgment that it is "noise". Verbal approval of a goal ("make sure it won't crash") is not approval of scaffolding inside the script. Offer the tooling as a manual procedure or separate mechanism, and if the user hesitates about an implementation ("I don't know if I like this"), treat that as rejection and remove it — do not defend the design.
**Why**: Two incidents in one session on `gantry_interconnect_dynamic.py`. (1) Implemented a SMOKE_TEST env hook after the user approved "smoke test" in a list; on seeing the code the user rejected it twice. (2) Suppressed the tqdm progress bar under SLURM (verbose=1) judging the 7000-line log as noise; the user relied on the bar to monitor long runs. The experiment file should contain only what the experiment needs, and its visible output belongs to the user.
**Scope extension (2026-07-07)**: this applies to standalone operational tooling too, not only in-script scaffolding. A preflight gate script was approved in a value discussion ("please implement both"), written (~400 lines), and rejected on sight ("I'm not sure about this"). Approval of a tool's value in discussion is not commitment to the artifact; when the user hesitates after seeing it, remove it without defending, and for any new tool beyond ~100 lines show the concrete skeleton (checks, thresholds, size) before writing the full implementation.
**Scope extension (2026-07-08) -- carry monitoring prefs into sibling scripts [HARD RULE, fired 3x]**: once the user asks for a per-epoch metric to be PRINTED live during a run (not just stored/plotted), that preference is STICKY and NON-NEGOTIABLE -- every subsequent training/search script that trains the gantry model MUST print `[nf-probe] train nf-RMS=... val nf-RMS=... [m]` each epoch. This is now a mandatory pre-write checklist item: before finishing ANY script that calls train_model / fit on the gantry model, confirm the per-epoch train+val nf-RMS print is wired in. If the script uses plain `train_model` (which does NOT install the probe), install the probe explicitly (reinstall per chunk if chunked, because the end-of-fit `_best` reload clobbers `cal_validation_error`). Do not omit it because of an implementation wrinkle -- solve the wrinkle. Fired three times: (1) built the live print into `diag_xy_routing_blowup.py`; (2) `diag_nf_curriculum.py` stored but didn't print -> "we again dont print train nf-rms and validation nf-rms"; (3) `gantry_optuna.py` used plain `train_model` with no probe -> "why are we not printing train nf and val nf?? how many times do I have to specify this???". The figure only appears at the end; the user monitors runs by these live prints.

---

### Rule: A fix is not delivered until the deployed copy has it — give a verification command

**Trigger**: When fixing a file that the user runs on another machine (cluster, remote repo copy)
**Rule**: State explicitly and prominently that the fix only exists locally until re-synced, and give a one-line command the user can run on the deployment side to verify which version they have (e.g. `grep -n "<removed line>" <file>`). When the same complaint returns, first establish WHICH copy ran before re-diagnosing the code.
**Why**: The tqdm-suppression revert was made locally; the user launched two cluster jobs from a stale copy and concluded twice that the fix was never made ("wtf have you done to the code"). One sync-verification command in the original fix message would have prevented both rounds. Note the changes were also uncommitted, making git-based sync silently miss them.

---

### Rule: End long analytical answers with a compact summary

**Trigger**: When a response is a long analysis, overview, or multi-part explanation
**Rule**: After the detailed overview, close with a short summary block (a few bullets or sentences) stating the bottom line, what is being proposed, and the next action.
**Why**: User explicitly requested this ("can you start giving a small summary after the overview") after several long analytical replies in the gantry augmentation discussion. Long answers without a distilled ending force the user to re-extract the conclusion.

---

### Rule: Scope implementation effort only after checking all pipelines for existing components

**Trigger**: When estimating effort or listing "required changes" for adding a capability
**Rule**: Before declaring any change necessary, search ALL project pipelines (`model_augmentation/`, `lpv_lfr_baseline/`, `scripts/gantry/real-data-verification/`) for an existing implementation of that capability. List a change as needed only after confirming it is absent everywhere.
**Why**: Listed a `# CHANGED` edit to Jan's `interconnect.py` as required for generic `param_loss` support, while `lpv_lfr_baseline/blocks/lfr_fit_system.py` (D-032) already provided exactly that hook as a non-invasive subclass. The user caught it ("joint estimation should already have been implemented in Jan's framework right?"). This project has three parallel pipelines; capabilities are often already built in a sibling pipeline.
**How to apply**: Before writing a change list, grep the whole repo for the key symbol (e.g. `param_loss`), not just the directory being edited.

---

### Rule: When re-implementing a performance-critical baseline path, replicate its compilation, do not drop to eager

**Trigger**: When overriding or re-implementing a baseline hot loop (rk4_step, simulate, a forward used in training BPTT) in a new module
**Rule**: Preserve the baseline's performance machinery. The baseline `rk4_step` is wrapped with `torch.compile(..., backend=COMPILE_BACKEND, fullgraph=True)` (with a `torch.jit.script` fallback); a re-implementation MUST apply the same wrap, not run eager Python. Import `COMPILE_BACKEND`/`_USE_COMPILE` from `lpv_lfr_baseline.core.lfr_simulate` and reuse them. Eager is only acceptable where the baseline itself forces it (eval under `@torch._dynamo.disable`), which happens automatically when the compiled step is called from a disabled eval wrapper.
**Why**: I wrote `coulomb_lfr.rk4_step_coulomb`/`simulate_coulomb` in plain eager Python and was about to run a 40-epoch, 22-trajectory, 2600-step-BPTT recovery on it, then even proposed measuring whether it would OOM. User: "why are you using eager? dont do that please". The baseline compiles the step for exactly this hot path; the override silently threw that away.
**How to apply**: After defining the re-implemented `rk4_step`, add the same `if _USE_COMPILE: torch.compile(...) else: try jit.script` block the baseline uses, and verify the compiled path still matches the reference (the MATLAB/collapsed checks).

---

### Rule: The deliverable is the RECOVERED model in the pipeline, not fixed-parameter diagnostics around it

**Trigger**: When adding a physical term (e.g. Coulomb friction) whose purpose is to be RECOVERED from data by a training pipeline (`run_telica_param_recovery.py`)
**Rule**: Keep the recovery integration (make the term a trainable parameter, wire it into the training loop) as the primary track. Fixed-parameter open-loop what-ifs (does a literature-value cc help on a held checkpoint) are secondary diagnostics; they can mislead (a fixed, different-machine cc is far from the value the data would fit) and must not displace building the integration. When such a diagnostic returns weak/ambiguous, the answer is usually "let the pipeline fit it," not "run more fixed-parameter variants."
**Why**: After the format was verified I ran a fixed-Garcia-cc open-loop comparison (Phase 3) and then offered a further fixed-cc windowed check, when the whole point of the Coulomb model is to be trained inside `run_telica_param_recovery.py`. User: "but the point of this model adjusted with coulomb friction is to use it for run_telica_param_recovery.py". The fixed-cc result (Coulomb barely helps at 16 N) is uninformative precisely because recovery would fit a different cc and rebalance viscous.
**How to apply**: Once the term's format is verified, go straight to making it trainable and wiring it into the recovery entry point; reach for fixed-parameter diagnostics only to debug a specific failure, not as the main progression.

---

### Rule: Confirm a block/config element is in the LOADED model, not just present in the serialized file

**Trigger**: When claiming a Simulink/Simscape (or any packaged binary) model contains a specific block, parameter, or feature, based on reading its serialized contents (e.g. unzipped `.slx` XML)
**Rule**: A `.slx` is an OPC zip that can carry ORPHANED subsystem XML (`system_NN.xml`) left over from deleted/restructured content and NOT wired into the live model graph. Do not assert the model "contains block X (just disabled)" from the raw XML alone. Confirm against the LOADED model: `load_system` then `find_system(mdl,'LookUnderMasks','on','FollowLinks','on', ...)` / a name/type search. Only what `find_system` returns is real.
**Why**: I inspected `gantry_2025a.slx`'s `system_47.xml`, found 3 `Signum` + `cc1/cc2/ccy` `Gain` blocks with `Commented=on`, and asserted across several turns that "the Simscape model contains the Coulomb blocks, just disabled" and planned a Phase-1 step to un-comment them. When actually loaded in MATLAB, `find_system` found ZERO Signum and no cc gains anywhere in the model (only the P-transform gains); a whole-model name search for `[Ss]ign|cc` was empty and `displayReplacedBlocks` reported none removed. The XML was orphaned; the live model has no Coulomb network. The wrong plan step (copy + un-comment) wasted a cycle.
**How to apply**: Before building on a model element read from a serialized file, load the model and query it. For `.slx`, `find_system` is the source of truth, not the unzipped XML.

---

### Rule: A new physical effect must be expressed in the CONSUMING pipeline's representation, checked first

**Trigger**: When asked to add a physical effect (friction, disturbance, extra dynamics) to a model that feeds a specific training/identification pipeline
**Rule**: Before proposing HOW to implement it, read how the target pipeline represents and consumes the model, and design the addition to conform to THAT representation. Do not propose the effect in a different, more convenient form (a standalone MATLAB ode45 wrapper subtracting the force) when the pipeline needs it in another form (here: the Python LPV-LFR simulate path in `train_param_recovery.py` -> `lfr_simulate`/`lfr_param_block`, with M/C/K matrices and the LFR z/w Delta loop). A nonlinearity like Coulomb `sign(v)` does not fit the LTI matrices; the LFR Delta path is the native place for a static nonlinearity, so find that mechanism before proposing. The MATLAB/Simscape model is a cross-check/ground-truth, not the delivery format.
**Why**: For the Telica Coulomb work I laid out a MATLAB ODE wrapper (`u - Fc` via `gantrySystemCoriolisCentripetal`) plus Simscape validation, but the effect must ultimately live in the Python LPV-LFR model that `train_param_recovery.py` trains and recovers parameters for. User: "we will need to add it in LPV-LFR format ... look at how train_param_recovery.py uses the one without coulomb friction, we will have to use it like that."
**How to apply**: For any model addition, first open the consumer's simulate/forward call and identify the exact object it expects; express the new term inside that object (matrix entry, LFR Delta block, added force channel), then validate. Design to the consuming interface before choosing an implementation form.
**Scope extension (2026-07-16) -- Telica/real-data extensions live in `scripts/gantry/real-data-verification/` and OVERRIDE the baseline, they do not edit `lpv_lfr_baseline/` core**: the Telica pipeline extends the LPV-LFR baseline by patching/overriding from its own folder (`run_telica_param_recovery.py` already monkey-patches `precompute._load_trajectory`, `tr._full_traj_eval`, etc.), keeping the shared baseline pristine as the no-augmentation reference. When adding a model term for the Telica work (e.g. Coulomb friction), put the new code (a Coulomb forward/sim module + a `ParameterizedLFRBlock` subclass) in `scripts/gantry/real-data-verification/` and wire it in via the patch mechanism; do NOT edit `lfr_forward.py`/`lfr_simulate.py`/`lfr_param_block.py`/`train_param_recovery.py` in place. Fired: my plan put the Coulomb implementation as edits to the `lpv_lfr_baseline/` core; user: "the python implementation should go in scripts/gantry/real-data-verification". A cc=0 == baseline test then doubles as both the format gate and a fidelity check on the duplicated forward.

---

### Rule: When a model structure is known but one scalar is missing, identify the scalar -- do not approximate the structure away

**Trigger**: When a known model (filter, controller, conversion chain) has one unknown parameter and data is available to identify it
**Rule**: Use least-squares or a similar regression on available calibration data to identify the missing scalar. Do not replace the model with a simpler proxy (static gain, RMS ratio) to avoid the identification step. A static gain is not the same as the original filter and removes the frequency-shaping that may be essential.
**Why**: User corrected a proposal to replace the real IIR feedback controller (Filter1 × Filter2) with a per-axis static gain to work around an unknown input scale. The correct action is to identify the scale from iter0 data (where feedforward=0 and MF30=K(M2) exactly) via LS, then use the full IIR in CLOE.
**How to apply**: When a chain is fully specified except for one scalar, write `scale = <h, MF30> / <h, h>` where `h = model_without_scale(input)`. Do not discard the model structure.

---

### Rule: Quote a parameter value from the script that actually wrote the consumed data, not a sibling/prototype

**Trigger**: When reporting a physical parameter, constant, or config value that feeds a downstream pipeline (e.g. a data-generation parameter used by the Python training data)
**Rule**: Before quoting the value, confirm the script you read is the one that actually produced the data file the pipeline consumes. Repos often have multiple scripts sharing the same variable name (a prototype/analysis script and the real generator) with DIFFERENT values. Trace which script `save`s the `.mat`/data file that the consumer loads, and quote from that one. A script the user names as "the model" may be a standalone prototype, not the generator of the data in use.
**Why**: Asserted the hidden-MSD natural frequency was `fa = 400 Hz`, read from `main_augmentation.m` (which the user pointed to as "the code the Simulink model uses"). That script is a standalone prototype; the actual training `.mat` files were written by `generate_oscillatory_multisine_data.m` and `generate_trajectory_data_without_multisine.m`, both of which use `fa = 150 Hz`. The narrowband multisine (130-180 Hz) in the generator corroborated 150 Hz. Confidently presenting 400 Hz as the truth-system parameter was wrong and the user caught it.
**How to apply**: For any data-derived parameter, grep all candidate scripts for the variable, find which one calls `save`/writes the file the pipeline reads, and quote that. If two scripts disagree, flag the inconsistency rather than picking one silently.

---

### Rule: Design-from-literature research is depth-first recipe extraction, not breadth-first surveying

**Trigger**: When the user asks for a design grounded in existing papers ("don't reinvent the wheel"), or any research task whose deliverable is a copyable method rather than an overview
**Rule**: (1) First name the small set of copy-target artifacts: the experiment/data-generation sections of the closest in-framework papers and any public code that produced their data. Check the local `literature/` folder before fetching anything. (2) Read those end-to-end and extract the full recipe (signal, band, grid, amplitude + stated rationale, duration, samples, splits, validation protocol) into a comparison table BEFORE surveying adjacent domains. (3) A located-but-unread primary artifact (a data-generation script in a public repo, a PDF already on disk) outranks every new search. (4) Present the design as copy-plus-delta: every element names its donor or is explicitly flagged as own synthesis. (5) Negative results ("no precedent found") are provisional, not products: single-pass search negatives have collapsed under one re-search. (6) In discussion answers, state which named source's protocol the answer draws on; if none, label it explicitly as synthesis, never let it read as researched.
**Why**: The gantry data-generation research produced verified bibliographies and negative results while the actual copy stack (Hoekstra EJC 2025 recipe + their public `msd_ndof_data_generation.py`, Bouc-Wen test spec, Bolderman injection design) sat partially unread — two PDFs were already on disk and a located repo script was never opened. A chirp question was answered with textbook synthesis although the Bouc-Wen sweep spec with exact numbers was already in the project's own log. User: "you have not clearly done your research."
**How to apply**: Before any adjacent-domain search, ask: "Have I read the experiment section AND the data-generation code of the closest in-framework work end-to-end?" If no, that is the next action.

---

### Rule: Before flagging a fact as unknown or an assumption, search the already-read sources for it

**Trigger**: When about to write "assumption", "unknown", "to be confirmed", or ask the user for a system property (direction, coupling, convention, parameter)
**Rule**: First grep/re-check the sources already read this session, including code comments and variable-definition lines. Only flag as unknown what is genuinely absent from them.
**Why**: Declared the hidden MSD's direction an open assumption (E1 sweep channel) although `generate_trajectory_data_without_multisine.m`, read in full the same session, comments L0 as "equilibrium offset of ma in +Y direction". User: "i find it concerning you cant find this?"
**How to apply**: Before every ASSUMPTION/OPEN flag, one targeted grep over the session's read files for the relevant symbol (here: L0, ma, delta_a). Cost is seconds; a false "unknown" costs user trust.

---

### Rule: Weigh a proposed acceptance threshold by its defensibility, not only its technical validity

**Trigger**: When a user or supervisor proposes a success criterion / threshold and you are tempted to counter with an alternative reference
**Rule**: Treat the threshold's defensibility as a first-class property: is it a model-free / information-theoretic bound, is it standard in the field, does it transfer to the real-data setting where no oracle exists? A hard, universally-accepted bound (e.g. the noise floor) is more defensible than a technically-equivalent but model-dependent, simulation-only reference (e.g. an oracle-model floor), even when both are numerically valid. Do not present the model-dependent alternative as strictly superior just because it needs no extra setup.
**Why**: Argued the oracle-sim floor was a cleaner target than Jan's noise floor because it needs no noise. The user's and Jan's point was that the noise floor is easier to JUSTIFY as a threshold: it is a model-free information-theoretic limit, universally accepted, and transfers to real data. Defensibility, not just numerical validity, is what makes a threshold usable in a thesis.

---

### Rule: Reference trajectories must be seam-continuous at zero velocity when validated by finite differencing

**Trigger**: When generating or concatenating motion/reference profiles (point-to-point moves, holds, sweeps) that a downstream check validates via finite-difference velocity or acceleration
**Rule**: Every concatenation or truncation seam must fall at a zero-velocity point (the end of a completed jerk-limited move, or inside a constant hold). Never truncate a profile mid-move to hit a target length; instead stop before a segment would overflow and fill the remainder with a hold at the current at-rest position. Finite differencing amplifies any velocity discontinuity by 1/ts (e.g. 0.01 m/s at ts = 5e-5 s becomes 200 m/s^2), so one bad seam trips an acceleration limit even when every profile interior respects amax.
**Why**: `gtd_make_reference`'s `ref_aprbs` truncated the body at `n_active` mid-move, leaving nonzero Y velocity, then appended a constant hold; `diff(vel)/ts` at that single seam exceeded the 50 m/s^2 limit although the moves themselves peaked at 11.25. Caught by `gtd_validate_ref` on T9.
**How to apply**: Build motion as whole move+hold segments, stop when the next would overflow, pad the remainder with an at-rest hold. Any seam must connect rest-to-rest or rest-to-hold.

---

### Rule: Present matrices in full explicit form for review, never block/blkdiag shorthand

**Trigger**: When documenting a system's matrices (mass, damping, stiffness, state-space) for a domain expert to verify
**Rule**: Write every matrix out with all of its entries, including zeros and coupling terms. Do NOT use block-concatenation shorthand (`blkdiag(C, c_a)`, `[[K,0],[0,k_a]]`, "baseline block plus the absorber") that forces the reader to assemble the full matrix from a referenced sub-block. Put the scalar entry definitions in a compact separate list below the matrix, and do not decorate entries with highlighting or mid-matrix parentheticals.
**Why**: In the Jan review writeup, `C_4` and `K_4` were written as `blkdiag` of the baseline `C,K` plus the absorber scalar, and `M` carried inline yellow-highlighted absorber terms with a "here m_h = m_h,rigid" aside inside the equation. User rejected this strongly: "just use the full matrices, not this disgusting concatenation."
**How to apply**: For each matrix, render all N×N entries as clean symbols; define the non-obvious ones in one aligned list underneath. No blkdiag, no highlighting, no comments inside the equation.

---

### Rule: A supervisor review note contains only what is needed to verify the setup, simplest object first

**Trigger**: When writing a short note or document for a supervisor to review a model, derivation, or method
**Rule**: One object per section, and cut every clause that does not help verify an equation or matrix: no "verified in MATLAB" asides, no RK4/implementation parentheticals, no parameter values repeated in prose (they live in the appendix), no mass-conservation-style commentary unless it is the actual point. Order sections simplest-to-most-complex: present the baseline (what we know) before the extended truth system (baseline plus the added physics). If a sentence is not required to check a matrix or an equation, delete it.
**Why**: User called the writeup "dogshit ... so much extra noise," pointing at the `M(Y,δ_a)` commentary and surrounding extra info, and said the baseline should come before the system. Reviewer docs are verified at a glance; volume and out-of-order buildup work against that.
**How to apply**: After drafting, reread each sentence and ask "does the reviewer need this to check an equation?" If no, cut it. Confirm the section order goes from the known/simple object to the extension.

---

### Rule: Removing a user-authored rule requires justification and an intent-preserving alternative in the same proposal

**Trigger**: When a proposed edit to a config or rules file (CLAUDE.md, lessons, settings) deletes or weakens a rule the user put there
**Rule**: Do not list the removal as a mere line item in a diff. In the proposing message itself: state why the rule no longer serves its purpose, check whether the underlying intent is still valid, and if it is, offer a softened rewrite that preserves the intent alongside the deletion option. Default recommendation is soften, not delete. Flag any conflict of interest when the rule constrains your own behavior.
**Why**: The CLAUDE.md optimization diff listed "subagent-trigger block removed" without justification; the user had to ask "why this?" before the reasoning surfaced. The rule's intent (protect main context from sprawling searches) was still valid, and the agreed outcome was a softened one-liner, not deletion.
**How to apply**: For every deleted rule in a proposed diff, add one sentence of justification plus the intent check. If the intent survives, present the softened version as the recommended option.

---

### Rule: Match the user's established house style for supervisor-facing LaTeX notes

**Trigger**: When writing or restyling any formal LaTeX note for the supervisor (derivations, model write-ups, review notes)
**Rule**: Reuse the conventions of `LPV/LFR-derivation-supervisor.tex` (and its siblings, e.g. the loop-matrix rational-rewrite note) rather than inventing formatting. Concretely: `\documentclass[11pt]{article}` with `geometry` ~2.2-2.4cm margins, `fontenc`/`inputenc`/`lmodern`, `amsmath,amssymb,mathtools,bm`, `hyperref[hidelinks]`, macros `\R,\adj,\diag`; every displayed equation numbered with a `\label` and cross-referenced by `\eqref`; full explicit matrices using shorthand scalar constants (`α,β,γ`) that are defined in one compact list, with `\dfrac` and `\\[2mm]` row spacing; bold `Assumption` blocks where a premise is used; a closing `Final Result` summary section. Keep content at the note's intended altitude (a concept note stays concept-level, do not expand into a full derivation) but wear this formatting.
**Why**: User rejected the ad-hoc formatting of the Jan augmentation writeup and pasted a full sibling document as the target, saying "I prefer this." The style is an established, reusable template for this project's supervisor docs.
**How to apply**: Before writing a new formal note, open `LPV/LFR-derivation-supervisor.tex`, copy its preamble and structural conventions, then fill in the content at the appropriate altitude.

---

### Rule: Measurement noise for identification is post-hoc on the output at the working rate; do not escalate to in-loop injection to be "more faithful"

**Trigger**: When adding measurement noise to closed-loop simulation data for identification/training, or moving a pipeline from noiseless to noisy
**Rule**: Add measurement noise post-hoc to the measured output at the working rate (post-decimation), matching the reference convention (Jan's SNR: `y += N(0, sigma_n)`). Do NOT inject it inside the closed loop / regenerate the data so the controller reacts, and do NOT add independent noise to `u`. A measurement-noise benchmark deliberately models only the sensor and deliberately does NOT reproduce closed-loop noise-input correlation, even though that correlation is physically real. Escalating to in-loop injection, errors-in-variables on `u`, or process noise is over-engineering unless the user explicitly asks. Corollary for [[downsampling]]: because the noise is added post-decimation and never passes through the loop, it is band-limited to the working Nyquist by construction and does NOT alias, so it does not reintroduce the anti-alias filter requirement (D-099 holds for the noisy sim too; the filter stays real-data-only).
**Why**: When the user moved toward adding noise, I proposed in-loop injection (reproducing closed-loop `u`/noise correlation) as the "faithful, recommended" option. The supervisor overruled it directly: "only measurement noise. DONT ADD IN THE CLOSED-LOOP. SHOULD NOT GO THROUGH THE CLOSED-LOOP. Same as how jan does it with his SNR." The theoretically-faithful scheme was explicitly not wanted; the simple reference convention was.
**How to apply**: For measurement noise, add it post-hoc to the stored measured output at the working rate, mirror the reference implementation's channels and rate, and stop there. Present the correlation/aliasing/EIV consequences only as caveats, not as a reason to build a heavier scheme.

---

### Rule: Trace the state-reconstruction path in code before claiming how measurement noise reaches unmeasured states

**Trigger**: When reasoning about how measurement noise (or any input perturbation) propagates to states the model does not directly receive (velocities, internal/augmented states) in an encoder/observer-based model
**Rule**: Read the actual encoder/observer forward pass before asserting the effect. A model that "only receives positions" does NOT get velocities for free: an encoder reconstructs them, and a linear reconstructability/deadbeat map (`x0 = W_y @ yhist + W_u @ uhist`, with `W_y = A^n · pinv(O)`) is a generalized differentiation of the position window, so it AMPLIFIES position noise into velocity error. Never claim noise is "avoided by construction" for reconstructed states. The amplification is gentler than 2-point FD (windowed least-squares over `na+1` samples averages noise, ~sqrt(12/(N(N^2-1)))/dt vs sqrt(2)/dt) but non-zero, and worst for low-signal channels (e.g. gantry yaw velocity).
**Why**: Asserted the augmentation model "reconstructs velocity internally from the position window, so the catastrophic differentiate-the-noise scenario is avoided by construction." The user pushed back ("I feel like we're missing something about the velocities"). Reading `pre_encoder.py:281-301` showed the velocity states are exactly a linear map of the noisy position history via `O_inv` — a differentiator that amplifies measurement noise, not an immunity.
**How to apply**: Before stating how noise affects unmeasured states, open the encoder/observer code and identify the map from measured I/O to those states. If it inverts an observability matrix or differences positions, say noise is amplified there and quantify it from the actual weight matrix, do not hand-wave it away.

---

### Rule: Do not kill or discard in-progress work to course-correct on an ambiguous instruction

**Trigger**: When a user message could be read as "change course," and acting on that reading means killing a running job, deleting outputs, or discarding long-running results
**Rule**: Confirm the interpretation before the irreversible step. A running verification/training job is cheap to let finish and expensive to recreate; keep it unless the user explicitly says to stop it. Re-read the message to separate what the user is doing themselves from what they are asking you to do.
**Why**: User said "match the hyperparameters to this run. and I also run this besides your current run on the server." I read it as "make the local run match the server config," ran `Stop-Process` on both in-progress epoch=1 Stage B verification runs, and prepared a multi-hour full-config relaunch. The user actually meant: continue the existing test; they themselves would run the full config on the server. The kill destroyed nearly-complete verification work for nothing.
**How to apply**: Before `Stop-Process`/`rm`/a relaunch that discards running work, state the interpretation and the destructive step in one line and get confirmation, unless the user explicitly asked to stop it.

---

### Rule: Velocity/acceleration-domain loss (fix C) is LAST RESORT — never present it as the solution until the user says so

**Trigger**: When discussing, ranking, planning, or implementing fixes for the X/Y free-integrator drift, and a velocity- or acceleration-domain loss (fitting velocity/acceleration instead of position; DIDIM/force-domain; Ljung differencing/prefilter as the training loss) is on the table
**Rule**: Treat the velocity/acceleration-domain loss as an explicit LAST RESORT. Do NOT recommend it as the primary or first fix, do NOT let literature convergence (DIDIM, Ljung, Tustin-Net all pointing at velocity/force domain) upgrade it to "the answer", and do NOT slot it as the default in a plan. The position-based fixes (DC guardrail / projection / dead-zone; incremental-passivity/NI relaxation; structural integrator factoring that keeps a position loss) are tried first. Only elevate the velocity/accel loss if the user explicitly authorizes it, or the position-based fixes are shown to fail. Always label it "last resort" wherever it appears.
**Why**: Supervisor directive (meeting): velocity fit is a last resort, "not convinced you can't pull it off with just positions," prefers multiple shooting. The user reiterated: "fix C is the last resort ... ignore it as the solution before I say so." Repeatedly the literature keeps converging on the velocity/force domain (Ljung prefilter, DIDIM, Tustin-Net), which tempts presenting it as the natural answer — but that is exactly the framing the user rejected.
**How to apply**: In any fix ranking/plan/doc, list the position-based fixes first and mark the velocity/accel loss "LAST RESORT (per supervisor/user; do not adopt without explicit go-ahead)". Note Tustin-Net separately: it is a STRUCTURAL integrator-factoring architecture that can keep a position-domain loss, so it is NOT the same as a velocity-domain loss and is not gated by this rule — but do not silently conflate the two to sneak the velocity loss back in.

**Trigger**: When the X+Theta+Y (K=0) routing misbehaves (drift, divergence, won't learn) and a Theta-only / Theta+absorber routing would train more cleanly
**Rule**: Do NOT propose, default to, or recommend Theta-only routing as a fix or fallback. Including X and Y in the ANN routing is a hard project constraint (D-103): the augmented system has coupling that cannot be captured without X/Y authority in the LEARNED component. The K=0 free-integrator drift must be solved WITH X/Y kept in the routing (velocity/acceleration-fit, nf-curriculum with per-stage lr, DC-free excitation + drift guardrail) -- never by removing X/Y. Theta-only is a controlled diagnostic baseline only, never the deliverable.
**Why**: I repeatedly offered "revert to Theta+absorber routing (it learns)" as an option when X/Y routing drifted. User: "only theta routing is not acceptable the augmented system has coupling we cant capture it without X and Y." Trading away the requirement to make training converge is not allowed; my earlier argument that the MSD coupling into X/Y "already flows through baseline M(Y) so the ANN need not route there" was rejected as the design basis.
**How to apply**: When X/Y routing fails, keep X/Y and change the loss/data/horizon (structural drift fixes), not the routing set. Do not present Theta-only as a recommended path.

---

### Rule: Do not let a new partial diagnostic override the established metric; match the reduction (endpoint vs RMS) before concluding

**Trigger**: When a fresh or smoke-level diagnostic (few samples/windows, a different reduction like endpoint-|error| vs RMS) suggests a conclusion that conflicts with an existing, more robust metric already computed for the same object (e.g. the training val nf-RMS)
**Rule**: Cross-check the new signal against the established metric before stating a conclusion, and be explicit about the reduction. An endpoint |error| over a few selected windows is NOT the same quantity as the training-loss RMS over all windows; a within-window ramp can be large at the endpoint yet contribute almost nothing to the RMS. If the new diagnostic and the established metric disagree, the established metric wins until the new one is run in full and reduced the same way.
**Why**: I claimed the X+Theta+Y model was "bad in-horizon" from an 8-window endpoint-|error| smoke (Y ~1.2e-4 at 0.1s). The training val nf-RMS for that same checkpoint was ~3.2e-5 and slightly DECREASING -- i.e. fine/improving in-horizon by the metric that actually drives training. The user corrected: Theta-only was good in-horizon and improved sim-RMS, and even some X+Theta+Y runs improved sim-RMS. The truth was subtler and better: the drift ramp IS present within 0.1s but is RMS-invisible, which is why training ignores it. Over-reading the endpoint smoke buried that.
**How to apply**: Before concluding from a new diagnostic, ask "what does the established metric (training loss / nf-RMS) say for this same object, and is my new number the same reduction?" Reconcile them; if they differ, run the new one in full with the matching reduction before asserting anything.

---

### Rule: Attribute an observation to the simplest sufficient cause before a more elaborate secondary one

**Trigger**: When explaining why a run/model behaves well or badly and more than one mechanism could account for it (e.g. "why didn't param recovery drift?" — perfect model vs windowed re-seeding)
**Rule**: Identify the SIMPLEST cause that fully accounts for the observation and lead with it; only invoke a secondary mechanism as a modifier, clearly labelled as secondary. Check the data/structure first (does the model MATCH the data at convergence? then there is no force error and no drift, no further mechanism needed). Do not lead with the more elaborate or more "interesting" mechanism when a plain one already explains the whole effect.
**Why**: I repeatedly attributed simulation param recovery's non-drift primarily to windowed re-seeding (multiple shooting). The user corrected: "with perfect parameters it didn't drift." Verified in code (`precompute.py`): param recovery fits a 6-state baseline to 6-state baseline-derived data, so at perfect params the model matches the data exactly -> zero force error -> no drift on a full free-run, windowing irrelevant (d2's full-truth run confirms: single x0, full trajectory, 1e-7, no re-seed). The windowing only matters while params are still wrong (masks/bounds drift during fitting) and on real data where no perfect params exist (model class wrong -> residual friction/DC force -> integrator drift in open-loop). Drift needs all three: force error AND a free integrator AND a full free-run without re-seed; remove any one and it vanishes.
**How to apply**: When asked "why does/doesn't X drift/diverge", first ask "does the model match the data at this operating point?" If yes, that alone is the answer. Reserve windowing/conditioning/optimizer explanations for cases where a residual force provably remains.

---

### Rule: Do not narrate a physical mechanism as the explanation until it is measured on the actual target object, not a proxy

**Trigger**: When a diagnosis yields an attractive, physically-sound mechanism (e.g. "the ANN injects a nonzero-mean force that ramps on the K=0 axis") inferred from a proxy model or a partial/short run, and you are about to present it as the conclusion or commit to a fix
**Rule**: Keep measured and inferred separate. A mechanism is a HYPOTHESIS until measured on the exact object the deliverable is about (here: the trained ANN), never a stand-in (baseline-only, full-truth, an untrained model). Present proxy results as motivating only, label the mechanism explicitly as unmeasured, and run the direct measurement on the target before asserting it or choosing a fix. Also surface any data that does NOT fit the narrative (e.g. a drift magnitude that depends on start sample when a fixed force-bias ramp would not) instead of smoothing it in.
**Why**: Diagnosing the X+Theta+Y augmentation drift, I reported "structural B / DC force bias ramps" with confidence from baseline-only and full-truth sims, but NO diagnostic used the trained ANN, and the fuller data (drift magnitude depending on start sample; the encoder x0 actually clean, so A2 rejected) did not cleanly support that narrative. The user's skeptical questions surfaced the gap.
**How to apply**: Before writing "the cause is X", ask: did I measure X on the actual object, or infer it from a proxy? If inferred, say so, present it as a hypothesis, and run the direct measurement on the target first (here: the trained ANN's time-mean output on the X/Y force rows).

---

### Rule: A comparison diagnostic proves nothing until its control condition is well-behaved

**Trigger**: When attributing an effect to a treatment variable by comparing two conditions (A vs B, baseline vs modified), and about to state a conclusion or log a decision
**Rule**: First confirm the control/baseline condition behaves as it should (converges, stays bounded, beats its own init, whatever "healthy" means for it). If the control ALSO misbehaves, the experiment does not isolate the variable and you may not attribute the treatment's outcome to the treatment. "B is worse than A" is not "the change from A to B caused a structural failure" when A is itself broken. Fix the setup until the control is clean, then re-run, then conclude. Do not write the confident decision entry off a run where the control failed.
**Why**: Ran a routing diagnostic: Theta-only `[1,4,6,7]` vs Theta+X+Y `[0..7]`. Theta+X+Y diverged x550; I concluded "K=0 X/Y routing is structurally unstable" and drafted a decision entry (D-100). But Theta-only ALSO went x23 above init and never beat epoch 0 in 8 gradient steps -- the control was not healthy (lr likely still too high / far too few gradient steps on 90 samples). User: "I dont agree with this. because now even the theta blows up." With a broken control, all the run shows is "neither routing learns in this degenerate setup and X/Y additionally runs away", not the structural claim I made.
**How to apply**: Before concluding a comparison, check the control's own curve first. If the control is not clean, the only valid next action is to fix the setup (lower lr, more steps/data, gradient clipping) until the control behaves, then re-run. Only then compare and log.

---

### Rule: Expose every tunable in one place in the entry script; do not split the setting surface into a config object plus a separate hand-authored dict

**Trigger**: When designing or refactoring the configuration for an experiment/training script the user edits to launch runs
**Rule**: All user-tunable parameters must be settable from one block at the top of the entry file, and there must be one source of truth. Do not keep a typed config object AND a separately-authored parameter dict (e.g. `RunConfig` + a `default_hp` dict) that the user has to reconcile; fold them into a single object and derive any internal dict view from it. A reason like "the dict is JSON-round-tripped in checkpoints" justifies keeping a dict-shaped *internal view*, not a second place the user must edit.
**Why**: The gantry refactor put ~14 experiment knobs in `RunConfig` but left the model/training hyperparameters (NX_ANN, n_nodes, layers, up_sample, nf, na_nb, batch_size, lr, epochs) in a separate `default_hp(cfg)` dict. The user: "I don't like that we can't set all parameters in gantry_interconnect_dynamic.py ... why is this still a separate dict compared to all the parameters? Config feels really messy."
**How to apply**: One config object holds every field the user sets; the entry file constructs it with all fields visible; derived quantities are properties; any legacy dict interface is generated from the object (`cfg.hp`), not hand-maintained alongside it.

---

### Rule: Verify what the interconnect actually routes into a block before proposing a design constraint on its I/O

**Trigger**: When proposing an architectural invariant on what a learned block may read or output ("feed velocity not position", "route only to K>0 rows", "the passive port reads X/Y velocity")
**Rule**: First read the interconnect wiring (`connect_block_signals` / `connect_signals`, `nz`/`nw`, the selection/expansion matrices) to confirm what signals the block currently receives and how they are produced upstream. Do not assume an idealized clean signal is available. In this pipeline the ANN block reads the FULL propagated state `x` plus `u` (`model.py`: `connect_block_signals(ann_block, ["x","u"], [])`, `nz=nxd+nu`), and the velocity components of that state are encoder-RECONSTRUCTED from a position window (a differentiation map, [[trace-state-reconstruction]]), not directly measured. Phrase the invariant as an explicit SELECTION on the actual routed indices, and separate the property it buys from ones it does not: velocity-only-in-the-force-path buys marginal-mode preservation (no added stiffness on the free position pole); it is distinct from passivity (which would tolerate position input) and from encoder noise amplification (an x0 issue, not a stiffness issue).
**Why**: Proposed "the passive port reads velocity, never position" without first checking that the pipeline feeds the whole state (positions included, idx 0/2) and that velocities (idx 3/5) are derived from positions by the encoder. User corrected: "we only have the real positions and determine velocities from that."
**How to apply**: Before writing a design-constraint proposal on block I/O, grep the model builder for the block's `connect_*` calls, confirm `nz`/`nw` and the selection/expansion matrices, and state the constraint as a selection on those exact indices.
**Scope extension (output-route vs input-read)**: When a hard routing constraint requires a block to OUTPUT to certain rows (D-103: X/Y force rows must be routed), never phrase an INPUT-scoping choice in a way that reads as dropping that output route. State plainly "the force output on X/Y stays (D-103); this is only about which signals feed that force." Conflating the two triggers a false alarm. Fired: said "keep Y-position out of the passive port entirely," which the user read as removing Y routing/coupling; the intent was only to exclude Y-*position* from the *input*, while the Y *force output* and the off-diagonal coupling (via shared internal state) were fully kept.

---

### Rule: Detect DRIFT by position-envelope growth, not by a velocity/slope proxy that a bounded oscillation also trips

**Trigger**: When writing a metric to decide whether a free-integrator (K=0) axis DRIFTS vs stays bounded, especially for an energy-storing/resonant block that can oscillate for many seconds
**Rule**: Drift is UNBOUNDED position growth. Measure it directly as position-ENVELOPE growth across successive time windows (e.g. RMS|q| in the last quarter / third quarter: ~1 bounded, >~1.2 drifting). Do NOT use a late-window linear-SLOPE of position or a velocity ratio as the drift test: a lightly-damped bounded resonator (small dissipation R, skew J) has a nonzero windowed velocity and a slope/v_ratio hovering around 1 while its position stays firmly bounded, so a slope/velocity criterion FALSE-FAILS it as drift. Match the metric to the physical definition (bounded position), not to a proxy confounded by oscillation.
**Why**: The Phase-1 drift probe flagged the stored-energy passive block as FAIL twice: first with a `|slope|<1e-6` position-slope threshold, then with a `v_ratio<1` "settling" criterion. Both mis-read a bounded (max|q|=2.2e-4 m, energy <= H(0)), lightly-damped, slowly-decaying oscillation as drift, because the near-lossless block released only ~0.01% of its stored energy in 12 s and just oscillated. The physically correct test (envelope not growing) passes it and still catches the DC-control (envelope grows ~1.4x = real drift).
**How to apply**: For any K=0 drift check, compute RMS|q| (or max|q|) over successive windows and test envelope growth; reserve slope/velocity only as secondary color. Confirm the falsifiable control (a sustained DC force) yields envelope-ratio clearly >1 while the bounded case yields ~1.

---

### Rule: A structural guarantee gap is fixed structurally, never with an ad-hoc numerical regularizer

**Trigger**: When a required GUARANTEE (bounded position, stability, passivity, no-drift) is found not to hold by construction, and a quick fix by adding a small constant is tempting (`R + eps*I` damping floor, weight decay, a clip, a small penalty)
**Rule**: Do not propose an arbitrary numerical regularizer as the fix for a missing structural/theoretical property. The constant is a HEURISTIC (flag it per CLAUDE.md) and it only masks the failure mode for tuned magnitudes; it provides no guarantee and does not transfer. Identify the correct structural property that yields the guarantee by construction (here: Negative-Imaginary / free-body for bounding a free-integrator POSITION, since plain passivity bounds only velocity) and implement/derive that. Treat the empirical failure as evidence FOR the principled route, not as something to patch.
**Why**: The Phase-1 stored-energy probe showed a growing position envelope (passivity bounds velocity/energy, not position on a free integrator = the A2 gap). I proposed `R = L_R L_R^T + eps*I` (an internal damping floor) as the fix. User rejected it: "I dont like the heuristic you're proposing." The floor would numerically damp the offending internal mode for some eps but gives no bounded-position guarantee and picks an arbitrary constant; the real fix is the NI/free-body structural constraint (the thesis contribution), or a structural zero-net-impulse output.
**How to apply**: When a guarantee fails, ask "what structural property, holding for ALL weights by construction, delivers this?" and build that. Only use a numerical floor as an explicitly-labelled temporary diagnostic, never as the delivered guarantee.

---

### Rule: Achieve a structural property by constraining the ANN's OUTPUT, never by restricting what it READS

**Trigger**: When enforcing marginal-mode preservation / no-drift / no-added-stiffness on the augmentation, and tempted to drop states from the ANN's INPUT (e.g. "feed velocity only, exclude X/Y position") to get it
**Rule**: Do NOT restrict the ANN's input to buy a structural guarantee. Removing states from what the block SEES cripples expressivity (kills LPV scheduling like Y-position -> M(Y), and position-dependent effects like cogging) and is the same information-throwing-away move as the velocity-domain LAST RESORT the supervisor forbade. Keep the FULL-state input (`["x","u"]`); obtain the property from the OUTPUT STRUCTURE instead. The bounded-integral block is the existence proof: it reads the full state INCLUDING X/Y position and still guarantees no drift, because the telescoping OUTPUT (g_k = psi(z_k) - psi(z_{k-1})) bounds the accumulated force regardless of what psi reads. State any marginal/no-drift requirement as a property of F(.) (its net impulse / its q-Jacobian), not as an input selection.
**Why**: In the passive-block design I listed "input is restricted to velocity + xi (not the full state)" as a requirement for marginal-mode preservation. User: "This is exactly the last resort we don't want to do." Velocity-only input echoes fix C (velocity-domain) and needlessly removes Y-scheduling; the property (no stiffness on the free pole) is about the OUTPUT's dependence on q, achievable while the block still reads q, exactly as bounded-integral does.
**How to apply**: Keep `["x","u"]` as the block input. Enforce marginal preservation / no drift through the output parametrization (telescoping/bounded-impulse, or a constraint on dF/dq), and verify it empirically (12 s position probe), not by amputating the input.

---

### Rule: Frame augmentation constraints as STEERING the learning in a targeted (knowable) subspace, not as a blunt output bound

**Trigger**: When designing/discussing how to constrain the augmentation ANN (no-drift, dissipativity, interpretability), and about to describe it only as "bound/cap the output"
**Rule**: Do not reduce the constraint to a blunt magnitude bound on the output. Two complementary mechanism types exist and BOTH are legitimate: (a) HARD architectural constraints (by construction, all weights: passive-PH, bounded-integral) that make bad outputs impossible; (b) SOFT targeted STEERING (a regularizer/projection on training) that shapes WHERE the ANN learns while preserving expressivity elsewhere. Orthogonal projection (Gyorok/Kon; our interpretability contribution) is the canonical (b): it steers the ANN output away from the KNOWN FP-model subspace so the ANN learns only the residual. Dissipativity/no-drift is the SAME kind of steering: steer the output into the energy-removing half-space (known from the SIGN of F.v), away from the drift direction, leaving friction/coupling free. The steering TARGET must be knowable WITHOUT the true dynamics (physics subspace / power sign = OK, knowledge-free; the residual MEAN = NOT knowable -> that is exactly why the DC/mean-force penalty was demoted, it suppresses real friction). These steerings live in the same regularization/projection layer (the LPV/MIMO/LFR extension of orthogonal projection = the thesis contribution) and must COEXIST -- frame the drift constraint as a sibling steering there, not a separate cap.
**Why**: I framed the whole approach as "we bound the output." User: "don't we need to steer the learning we can't just bound the output. Like with the orthogonality for example we bound a specific region for the subspace the fp model captures." The steering frame is richer, matches the in-framework precedent (orthogonality), and unifies the drift constraint with the interpretability contribution.
**How to apply**: When proposing a constraint, say which mechanism it is (hard-architectural vs soft-steering), name the target subspace/region and confirm it is knowable without the true dynamics (else it fails knowledge-free A1), and check it coexists with the orthogonal-projection steering (C5).
**Scope extension (2026-07-10) -- IMPULSE-based no-drift constraints suppress friction exactly like the mean penalty; never elevate one to the friction-permitting deliverable on buildability grounds**: The net-impulse / bounded-integral output constraint (`g_k = psi(z_k)-psi(z_{k-1})`, `Sum g` bounded) is IMPULSE-based, not POWER-based. Like the demoted DC/mean penalty, it forbids ALL net-impulse-carrying force -- which includes DISSIPATIVE friction (Coulomb over asymmetric motion carries net impulse). It CANNOT distinguish energy-injecting drift (forbid) from energy-removing friction (permit); only a POWER-sign (`F.v`) passivity constraint can (drift `F.v>0` forbidden, friction `F.v<=0` permitted, storage/absorber allowed via `int F.v <= V(0)`). So the net-impulse block FAILS criterion 2 (friction-permitting) and forces friction into `f_base`. Do NOT elevate it to "the solution/spec" because it is buildable/validated (D-B/D-C) -- validation status must not promote a requirement-violating mechanism to the deliverable. The net-impulse block is a sim-NO-friction BRIDGE/fallback; the friction-permitting deliverable is the POWER-based passivity (-> NI) constraint (5f). When choosing the deliverable constraint, score EVERY candidate against ALL four requirements first; the one that motivated the whole approach (crit 2, friction-permitting, = why passivity beat the DC/impulse family) is non-negotiable.
**Why (2026-07-10)**: I wrote a one-page constructive spec (`docs/dissipative-block-spec.md`) centered on the net-impulse block; its consequence is "friction forced into `f_base`". User rejected it: "if this is a requirement: forces friction into f_base than its a bad solution" and "we cant accept [the friction casualties]". The net-impulse block's friction limitation is the exact reason passivity (power-based) was chosen over the DC/impulse family (5f); centering the spec on net-impulse silently re-introduced it, biased by the block being the only one already validated.

---

### Rule: For an UNKNOWN-system deliverable, never propose a constraint that restricts the ANN's expressible dynamics class; no-drift must come from the ESTIMATOR (hard-guarantee XOR full-expressivity)

**Trigger**: When proposing any no-drift / boundedness / stability mechanism for the augmentation that will run on REAL data whose true nonlinear dynamics are unknown
**Rule**: Do NOT propose any HARD structural constraint that forbids a class of dynamics -- passivity, pure dissipativity, net-impulse/bounded-integral, contraction, NI-by-construction -- as the DELIVERABLE. On an unknown system, any class the constraint forbids might BE the true residual, and you CANNOT verify it isn't (the physical machine may be passive, but the residual in the chosen coordinates need not be, and it is unverifiable). A "no drift for ALL weights" structural guarantee and FULL expressivity are LOGICALLY INCOMPATIBLE: if the ANN can represent any dynamics, it can represent a drifting one, so a hard guarantee necessarily forbids some representable dynamics. Since full expressivity is REQUIRED when the system is unknown, the no-drift must come from the ESTIMATOR, not the model class: keep the ANN fully expressive; prevent drift by (a) training CONDITIONING (multiple shooting + continuity, makes drift visible without forbidding it) and (b) regularizing ONLY the unexcited/unidentifiable direction (null-space regularization / orthogonal projection / Bayesian minimum-norm prior) -- which acts only where the DATA carries no information (a data property, not a dynamics assumption). Where the true dynamics DO excite the direction, the data overrides the prior and it is learned freely. This preserves the full model class; a structural constraint does not.
**Why**: Across the passivity-augmentation discussion I proposed, in turn, passivity, then net-impulse, then power-based passivity as "the solution/spec". User, emphatic: "I dont want any solution that doesnt allow us to [learn] the full dynamics. we cant have that for when we dont know the actual nonlinear system. DO YOU GET THAT?" Every one of those is a class-restricting structural constraint; on an unknown system each risks excluding the true residual, and none is verifiable. The expressivity-preserving estimation route (5m identifiability reframe; orthogonal projection = a null-space steering = the thesis's own contribution) is the correct deliverable.
**How to apply**: For any unknown-real-data no-drift mechanism, first ask "does this forbid the ANN from representing some dynamics?" If yes, it is NOT the deliverable. Prefer estimator-side mechanisms that act only where the data is silent (null-space regularization, orthogonal projection) or that condition training (multiple shooting); these keep the full model class. A structural constraint (passivity/NI/net-impulse) is admissible only for the SIM phase or as an explicitly-labelled fallback, never as the unknown-real-data deliverable. State plainly that no-drift is then training-CONDITIONAL, not a for-all-weights guarantee -- that is the unavoidable price of full expressivity, not a defect to patch with a structural constraint.

---

### Rule: When several proposed solutions all fail and the user stays unconvinced, PROVE the spec is over-constrained and question a requirement or the metric; do NOT keep multiplying methods

**Trigger**: When you have proposed multiple distinct solutions to the same problem, each has documented fatal limits, and the user is still not convinced
**Rule**: Stop generating new methods. That pattern is the signal to step back and test whether the REQUIREMENT SET is over-constrained or impossible. Try to PROVE the impossibility (here: full expressivity XOR a for-all-weights no-drift guarantee -> exactly two solution families, each dropping one requirement, no third option). Once shown, the productive move is to identify which requirement is BOTH impossible AND unnecessary and relax it, or to question the EVALUATION METRIC, not to invent solution N+1. Check especially whether the metric matches deployment (open-loop free-run vs closed-loop) and whether a mundane, PROVEN, already-blessed tool (multiple shooting, standard conditioning) was buried under exotic novel constructions. Flag requirement/metric relaxations that are the user's or supervisor's call as exactly that, rather than deciding them.
**Why**: Across the passivity-augmentation thread I proposed, in sequence, passivity, net-impulse, power-based passivity, momentum-conservation, cyclo/EID/NI, and data-silent regularization -- each with fatal limits documented. User: "given all the constraints im just still not convinced by the current solutions." The real situation was a proven impossibility (expressivity XOR guarantee), so no method could satisfy the full spec; every candidate necessarily dropped a requirement. The convincing move was to name the impossibility, relax the requirement that is both impossible and unnecessary (the STRUCTURAL for-all-weights no-drift, which no SysID method has), and raise the open-vs-closed-loop METRIC with the supervisor -- not to build solution N+1.
**CORRECTION (2026-07-11, same turn)**: I then proposed multiple shooting as "the proven, supervisor-preferred fallback" as if untried. WRONG -- it WAS tested on the AUGMENTATION and failed: Optuna 69399 swept lr in [1e-8,1e-5] AND nf up to 1600, best checkpoint = epoch 0 (4/5 trials revert; one lr gave 18%). The d4 "multiple shooting prevented drift" result is PARAM RECOVERY with PERFECT params (model matches data -> zero force error -> no drift), which does NOT transfer to the augmentation (the ANN always leaves a residual force). User: "we already tested multiple shooting it cant learn the sim-rms and what about the optuna search result". Do not cite a mechanism as a proven fallback for problem P when its success was on a DIFFERENT problem Q; before recommending any "just try the mundane proven X", grep the run table / Optuna results to confirm X was not already tested on THIS problem and failed. The Optuna lr+nf sweep landing on epoch 0 also means the failure is NOT a tuning problem -- the windowed sim-error loss improves in-window while worsening full-traj sim-RMS on the free integrator (horizon mismatch), which undermines the conditioning route too, not just multiple shooting.
**How to apply**: On the 2nd or 3rd failed solution to the same spec, switch modes: (1) write the requirements as a set and try to prove two of them are incompatible; (2) if so, present the impossibility and the finite set of families each requirement-drop yields; (3) recommend relaxing the requirement that is impossible AND unnecessary, or questioning the metric/deployment match; (4) surface any proven/blessed mundane tool that got buried. Do not propose another novel mechanism until the spec is re-examined.

---

### Rule: Never hand the user a ready-to-run command/prompt/handoff that references artifacts you have not yet created

**Trigger**: When giving the user a command, prompt, or handoff (e.g. a prompt to start another session/model) that names files, directories, downloaded resources, or scripts
**Rule**: Either (a) create ALL referenced artifacts FIRST in the same turn and confirm they exist, or (b) make it unmistakable the command is a DRAFT gated behind explicit creation steps ("do NOT run this until I have created X, Y"). Do not present a prompt/command referencing not-yet-existing files as if it is ready to paste. Assume the user will run it verbatim immediately. When the handoff depends on enabling steps you proposed but have not executed, execute them (or clearly block the handoff) before delivering the prompt.
**Why**: I gave the user a "starter prompt for the Fable session" pointing at `docs/fable-review-brief.md` and `literature/passivity-augmentation/`, which I had only PROPOSED to create ("Want me to go ahead and (a) download and (b) write..."). The user pasted the prompt into a fresh Fable session before I created anything; Fable correctly reported the files/dirs do not exist and the two flagship papers are online-only, wasting the session setup. The prompt looked ready; it was not. [[fix-not-delivered-until-deployed]]
**How to apply**: Before delivering any runnable handoff, list every artifact it references and verify each exists (create it now if not). If you cannot create them this turn, label the command "DRAFT - not runnable until I create <list>".
