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

---

### Rule: Validate with the actual pipeline model and representative data, not a simplified proxy

**Trigger**: When building a diagnostic or validation for a pipeline parameter (sampling rate, window length, integration steps, etc.)
**Rule**: Use the same model the pipeline actually uses (e.g., LPV nonlinear), not a simplified proxy (e.g., LTI linearization). Choose test data that exercises the full operating range, not data tailored to make the proxy valid.
**Why**: Built a downsampling validation using the LTI linearization at Y_op=0, then chose a Y=0 trajectory to match the linearization point. The real pipeline uses the LPV model with Y-sweeping data. The LTI test missed the coupling between Y-scheduling and sampling rate entirely.
**How to apply**: Before writing a diagnostic, ask: "Which model does the pipeline use? Does the test data cover the operating range the pipeline sees?" If the diagnostic uses a different model or narrower data, it is not testing the right thing.

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

---

### Rule: Keep operational scaffolding out of experiment scripts

**Trigger**: When adding run-management tooling (test hooks, env-var modes, rehearsal switches) to a script whose purpose is a scientific experiment
**Rule**: Do not embed operational hooks in the experiment file, and do not remove or alter existing user-visible behavior (progress bars, log output, prints) based on own judgment that it is "noise". Verbal approval of a goal ("make sure it won't crash") is not approval of scaffolding inside the script. Offer the tooling as a manual procedure or separate mechanism, and if the user hesitates about an implementation ("I don't know if I like this"), treat that as rejection and remove it — do not defend the design.
**Why**: Two incidents in one session on `gantry_interconnect_dynamic.py`. (1) Implemented a SMOKE_TEST env hook after the user approved "smoke test" in a list; on seeing the code the user rejected it twice. (2) Suppressed the tqdm progress bar under SLURM (verbose=1) judging the 7000-line log as noise; the user relied on the bar to monitor long runs. The experiment file should contain only what the experiment needs, and its visible output belongs to the user.

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

### Rule: Weigh a proposed acceptance threshold by its defensibility, not only its technical validity

**Trigger**: When a user or supervisor proposes a success criterion / threshold and you are tempted to counter with an alternative reference
**Rule**: Treat the threshold's defensibility as a first-class property: is it a model-free / information-theoretic bound, is it standard in the field, does it transfer to the real-data setting where no oracle exists? A hard, universally-accepted bound (e.g. the noise floor) is more defensible than a technically-equivalent but model-dependent, simulation-only reference (e.g. an oracle-model floor), even when both are numerically valid. Do not present the model-dependent alternative as strictly superior just because it needs no extra setup.
**Why**: Argued the oracle-sim floor was a cleaner target than Jan's noise floor because it needs no noise. The user's and Jan's point was that the noise floor is easier to JUSTIFY as a threshold: it is a model-free information-theoretic limit, universally accepted, and transfers to real data. Defensibility, not just numerical validity, is what makes a threshold usable in a thesis.
