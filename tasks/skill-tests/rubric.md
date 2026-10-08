# thesis-section skill test: rubric and run log

Each run is a fresh agent that sees only the skill, the fixed prompt below, and
the repository files it chooses to read. It never sees this rubric. Output goes
to `tasks/skill-tests/run-N/`. Cap (Dirk, 2026-10-05: keep iterating while the
output is not yet what he wants): iterate until two consecutive runs on
different sections pass A, B and B2, then Dirk judges. (Dirk, 2026-10-05, after
rejecting runs 5 and 6: keep iterating autonomously until the skill is polished.)
Check in with Dirk only if two consecutive iterations show no improvement on
the failing check, or at run 14.

## Fixed test prompt

> Use the thesis-section skill (`.claude/skills/thesis-section/SKILL.md`) to
> restructure Section 3 of the thesis
> (`Thesis-writeup/Writing/sections/03_augmentation.tex`). This is a test run of
> the skill, so three deviations apply. Treat the whole repository as read-only
> and write only inside `tasks/skill-tests/run-N/`. Steps 2a and 2b are
> pre-approved: write both to `tasks/skill-tests/run-N/outline.md` and continue
> without waiting.
> Write the finished section to `tasks/skill-tests/run-N/03_augmentation.tex`
> in the same form as the original file, run the review loop, and give the
> step-5 delivery report as your final message. Do not compile.

## A. Structure (pass/fail each)

| # | Check |
|-|-|
| S1 | The opening paragraph is a roadmap that names the subsections in their actual order. |
| S2 | Every subsection's first paragraph states the problem or need before the construction. |
| S3 | (revised for v6) Every paragraph has one nameable job; no paragraph is a run of isolated cited claims. |
| S4 | Every displayed equation has a lead-in inside a sentence, a "where" clause for new symbols, and at most one sentence of meaning or consequence (v6; equations carry the content). |
| S5 | Standard machinery (RK4 stages, MLP layer formula, ZOH integral) is not displayed; it gets a phrase and a citation (Dirk, 2026-10-05: RK4 as a phrase). |
| S6 | No paragraph is a run of 3 or more attributions; each borrowed component is cited in the subsection that uses it, with a relation verb. |
| S7 | No Setup values in the section (n_xa, widths, encoder lags, window length, epochs, seeds). |
| S8 | (revised for v12, Dirk 2026-10-06: todos and draft bullets stay) No em dash or `--` as a dash; no `because: ?` outside a `\todo`. `\todo`s are allowed in the v12 form (see T1). |
| S9 | Symbols match Section 2 and `util/format.tex`; every symbol defined once. |
| S10 | The section follows its own outline. |
| S11 | Compiles without new `!` lines when swapped into a scratch copy of `Writing/` (compiled by the evaluator, not the agent). The unmodified copy already has one, a bibliography error in `main.bbl`, and starts Section III on p. 5 and Section IV on p. 7. |
| S12 | Known-open reasons are not invented: routing to all six rows, encoding the measured positions, linearisation at Y=0, full-window scoring, W^a init. Each is either supported by a cited source or decision, or listed in the report's open points. |

## B. Coverage (each item present, or moved to Setup/appendix with a reason in the outline)

| # | Content (from the 2026-10-05 draft) |
|-|-|
| C1 | What the baseline omits (Garcia-Herreros list; Coulomb friction left out). |
| C2 | Joint estimation of theta_base, theta_aug and eta from records measured under feedback. |
| C3 | Positioning: dynamic-parallel structure (Hoekstra), encoder (Beintema), closed-loop rollout (Kessels), LPV precedent (Drenth); this work's contribution; OBC as separate contribution. |
| C4 | Why additional states: refitting theta_base or a static correction cannot add model order. |
| C5 | The augmented model equation (f_base + f_aug, g_aug, output map), with normalisation. |
| C6 | f_base is one RK4 step of the CT model, input held, Y moves within the step (self-scheduled), applied in normalised coordinates. |
| C7 | One MLP gives f_aug and g_aug; f_aug writes all six physical rows; discrete-time correction on top of the CT physics. |
| C8 | Output map is kinematic and not learned; well-posedness via acyclic graph (Hoekstra 2026 Thm 9); added states reach the output only through f_aug. |
| C9 | Zero output-layer initialisation reproduces the baseline; the baseline/augmentation split is non-unique, handed to Section 4. |
| C10 | Encoder estimates the window's initial state from I/O history (current sample included), all states including measured positions. |
| C11 | Encoder = linear map + nonlinear correction; W^b from the linearisation (Y=0, ZOH) via the reconstructability map; observability; init only; W^a random, psi-tilde zero. |
| C12 | Why closed loop: data and intended use are under feedback; X and Y free integrators make open-loop position errors persist. |
| C13 | Kessels precedent; controller kept outside the model so it can be replaced. |
| C14 | Recorded and model loop equations; residual form; exactness conditions (same linear controller, not scheduled on model state, same feedforward, no saturation). |
| C15 | x^c_tau = 0 means s-hat_tau = s_tau; differs from Kessels' reconstruction; earlier model error not carried into the window. |
| C16 | Evaluation order within a sample (no feedthrough). |
| C17 | Loss over windows, every sample scored, joint estimation with fixed controller, horizons deferred to Setup. |
| C18 | Both figures (augmentation structure, closed loop) referenced and captioned. |
| C19 | (added after run 2, from the section header) Caution: a good closed-loop fit does not by itself prove an accurate open-loop plant model. |
| C20 | (header) The learned additional states are distinguished from the truth-only absorber states of the simulated data. |
| C21 | (header) The departure from the research plan is documented: open-loop training replaced by closed-loop rollout; earlier state routing superseded (one sentence, not history). |
| C22 | (header) The horizons (encoder history, scored window, validation horizon) are kept separate and related to the dynamics they must cover, or that relation is listed as an open point. |

## B-IV. Coverage for the Section 4 generalisation run (from the 2026-10-05 draft and its header; written before grading run 5)

| # | Content |
|-|-|
| O1 | Opening positions the section: extends Gyorok's OBC (linear I/O, linear-in-parameters) to the nonlinear state-space gantry model; roadmap. |
| O2 | Negation: joint estimation lets the additive network write reproduce a physical parameter change (one-step map not unique). |
| O3 | Transition sensitivity Phi (eq:obc_sens) as the physical directions; first-order cancellation (eq:obc_negation). |
| O4 | Gantry sensitivity (eq:obc_gantry_sens): mass via acc, damping via vel, stiffness via pos; M(Y)^{-1} makes directions Y-dependent; RK4 propagates it. |
| O5 | Raw 14 parameters have 4 null directions; protect the 10 identifiable combinations; recovery needs the later condition. |
| O6 | Gyorok's construction (eq:obc_gyorok) and that it needs no penalty weight (contrast with L4DC regularisation = research-plan departure). |
| O7 | Applied to physical rows of the normalised RK4 transition; regressor = sensitivity because f_base is rational in theta; affine offset not protected, with reason. |
| O8 | Reference tuples (eq:obc_tuples): measured positions, 5-point stencil velocities, x-bar = 0, recorded input includes feedback, no record crossing, normalised; fixed set permitted by Gyorok. Why tuples (open-loop reconstruction drifts on free axes). |
| O9 | Stacked construction (eq:obc_ours) incl. full column rank monitored, Lemma 3, gradient identity. |
| O10 | Coefficient computed once per objective from fixed tuples, differentiated through; sensitivity frozen per epoch, rebuilt between epochs (D-192). |
| O11 | Corrected model (eq:obc_model); correction at rollout point; forward sensitivity through RK4 (appendix); g_aug uncorrected. |
| O12 | Scope (header caution): local in theta, one-step, over tuples, Euclidean metric in normalised coordinates, aggregate not pointwise, x-bar = 0 slice, no trajectory-output orthogonality, Gyorok Thms 16/18 do not transfer. |
| O13 | Recovery condition (eq:obc_delta, eq:obc_bias): assumptions, Phi^T Delta* = 0, one-step surrogate only; no pointwise orthogonal missing force, needs designed excitation. |
| O14 | Overlap measures (eq:obc_overlap) and their use (simulation-only for Delta*, held-out tuples, unprotected baseline-output overlap); pointer to Results. |

## B2. Readability (added after run 3, from Dirk's 2026-10-05 goal: readable, logical order, to-the-point sentences that are not packed)

Measured with `scratchpad/sentstats.py` (prose only: equations, figures, `\todo`s and `\ifdraft` bullet blocks stripped).

| # | Check |
|-|-|
| R1 | (revised, Dirk 2026-10-05: no word limit, it would push out required content) Words per displayed equation is recorded as a diagnostic only, never pass/fail. Conciseness is judged by R6 (deletion audit); completeness by B, which always takes precedence. |
| R2 | No packed sentences: none over 40 words outside an equation's "where" clause. |
| R3 | Reading only the first sentence of every paragraph, in order, tells the section's argument without gaps. |
| R4 | Every term and symbol is introduced before it is used; no forward reference needed to understand a sentence. |
| R5 | (added after run 6, Dirk) No argument rests on a fact the thesis has not yet established (e.g. a vibration mode before the absorber is introduced); earlier sections are referred to, not re-explained. |
| R6 | (added after run 6, Dirk) Deletion audit by the evaluator on two paragraphs per subsection: no sentence or clause the reader would not miss; no decoration ('classical'), hedges ('intended to'), announcements, or explained standard terms. |

Baseline measurements (sentences split at displayed equations): original Section 3 mean 16.3 words, 5% over 30, max 39 (short but staccato); run 1 22.7 / 26%; run 2 23.2 / 18%; run 3 21.5 / 14%; run 4 23.2 / 22%. Every run packs more into each sentence than the original.

## B3. Contributions and sources (added for v10)

| # | Check |
|-|-|
| K1 | The section opening names the Introduction contribution it delivers. |
| K2 | Every adapted component states its difference from the cited method; this work's own parts are stated with "we" and no citation; the gantry-specific realisation is presented as this work's. |
| K3 | Configuration and mechanisms match the THESIS_ARM path (evaluator checks against sources.md and the code), e.g. Section IV describes the tangent OBC, per-epoch refresh, no affine arm; Section III no base-CFG values. |

## B4. Todos and draft bullets (added for v12)

| # | Check |
|-|-|
| T1 | Every `\todo` sits at the end of a paragraph, one per item, as "Missing: X. Candidate: Y (basis)." where a candidate exists; none is a style remark or something the session could have resolved; todos of the original are kept unless resolved, and resolved ones are named in the report. |
| T2 | Every `\section` and `\subsection` has an `\ifdraft` itemize of one-line content bullets (no source lists or figure plans); every bullet is covered by the prose and every paragraph maps to a bullet; open points marked (?). |

## C. Dirk's judgment

After the rubric passes: Dirk reads the passing output (and run-1 for contrast)
and decides whether it is "proper". The rubric is necessary, not sufficient.

## Run log

| Run | Skill version | S failed | C missing | Main failure | Skill change made |
|-|-|-|-|-|-|
| 1 | v1 | S3 partial (III-A P2, P5 and III-B P2 run 9 to 12 sentences with two jobs) | C18 partial (fig:aug_structure never referenced; same in original) | Structure now Jan-like (roadmap, problem-first openings, citations in place, RK4 as phrase, no todos, compiles). But 3 pages vs 2 originally: implementation detail (P stored in code) and optimisation mechanics (gradient flow at init, twice) kept although true. 263k tokens. | v2: reader test with three named exclusions; cut pass when over budget by a third; max 8 sentences, one job per paragraph; every figure referenced; review checklist extended. |
| 2 | v2 | S12 partial (own routing derivation put in prose, flagged only in report) | none (C3 contributions left to the Introduction, acceptable) | Length back to original (III on p. 5 to 7), paragraphs mostly one job, both figures referenced, new III-D for initialisation. But 4 displays dropped (zero init, W^b map, recorded-loop line, MLP layers), against Dirk's rule that formulas stay displayed. 259k tokens. | v3: textbook machinery is the only display that may become a phrase; borrowed constructions applied to this model stay displayed; cut prose, not equations; own derived reasons go to open points, not prose; checklist: removed displays listed, every reason traced. Style profile pattern 4 aligned. |
| 1-2 regraded | | | run 1: C19, C20 missing, C21, C22 partial; run 2: C21, C22 partial | Coverage list C1-C18 checked only 'nothing dropped'; C19-C22 added from the header after run 2. | |
| 3 | v3 | S9 (switched the model input to u_act against Section IV's u = P u_act and asked IV to change) | C20 missing, C21 partial (no routing departure) | All original displays kept, plus a new normalisation display; four subsections (model, encoder, closed loop, criterion); C19 and C22 now stated; III on p. 5 to 7 (agent estimated 2.7 pages, too high). Structural choices differ per run (3 or 4 subsections, init vs criterion split): variance between runs is high. 267k tokens. | v4: header Cautions and Research-plan anchor are requirements; keep notation used by other sections, list conflicts; checklist extended. |
| 4 | v4 | R2 fail (22% over 30 words, mean 23.2); S9 pass (kept u = P u_act, renamed only III-local symbols) | none: C19 to C22 all present | Best coverage so far: header cautions and research-plan departure now stated. Readability not addressed by v1 to v4 at all. Structure: model, closed loop, criterion (encoder inside), initialisation. III on p. 5 to 7. 297k tokens. | v5 (rewrite for generality, Dirk's goal 2026-10-05): three priorities up front (order, paragraphs, to-the-point sentences); one statement per sentence, split over 30 words, packed vs to-the-point example; ordering rules incl. topic sentences and per-section-type order (method, setup, results, discussion); first-sentence reading check and sentence word count in the review loop; Section-3-specific examples removed; style profile warns not to copy Jan's sentence density. Next: run 5 on Section 4 (generalisation). |
| 5 (Sec. IV) | v5 | none hard; minor: two prose claims rest on measurements from pre-thesis data (range changes little with expansion point; rank three), one reason for Thms 16/18 not transferring differs from D-190; u defined as actuator force following the original Section III text (repo itself inconsistent, flagged) | none: O1 to O14 present (RK4 forward sensitivity moved to appendix candidate) | Readability met: mean 15.0 words, 1% over 30, max 31, no semicolons; first sentences of the 15 paragraphs read as the argument; IV on p. 7 to 9 as original; compiles. First run that reads clean. 292k tokens. | none yet: run 6 on Section III with unchanged v5 to test the stop criterion. Candidate v6 tweaks: separate own method derivations (allowed, marked as ours) from own justifications of choices (open points); flag claims whose evidence comes from superseded data. |
| 6 (Sec. III) | v5 | minor: one untraced reason in prose (parallel structure chosen because OBC acts on an additive correction); u_k = u_act kept from original III (conflict with II flagged); encoder subsection is one long paragraph with two displays | none: C1 to C22 present (contributions left to the Introduction) | Readability met: mean 15.2, 2% over 30, max 35, no semicolons; III on p. 5 to 7 as original; compiles; RK4 stages the only removed display. 222k tokens. STOP CRITERION MET: runs 5 and 6 pass A, B and B2 on two different sections with the same skill version. Preview PDF: preview-run6-sec3_run5-sec4.pdf. | Awaiting Dirk's judgment. Candidate v6 tweaks (not applied): own method derivations vs own justifications; flag evidence from superseded data. |
| Dirk on runs 5 and 6 | | | | Rejected as too much text: arguments from unestablished context ("an omitted vibration mode" before the MSD/absorber is introduced), sentence style he would never write, decoration ("classical"), and above all too many words per equation. Measured: words per display 160 to 235 in runs 1 to 6 vs 107 to 117 in the current drafts. The v5 one-statement-per-sentence rule made it worse (split instead of cut). Dirk approved a concise rewrite of the RK4 paragraph (105 to 40 words) as the target style. | v6: Jan for structure only; approved RK4 before/after as the reference example with the rule behind each removal; deletion test; say each fact once; build only on established context; equations carry the content (lead-in, where clause, at most one sentence); expert register without defined standard terms, hedges or announcements; justify only what an examiner would question; words-per-display diagnostic in the review loop; sentence-splitting rule and 4-to-8-sentence paragraph rule removed. |
| 7 (Sec. III) | v6 (with the words-per-display check still in the skill at launch) | R2 borderline: 7 sentences over 30 words, max 42 (the Garcia list), 5 semicolons, a few packed sentences (output-map sentence carries four facts) | C10 partial (does not say the encoder also estimates the measured positions); C21 partial (earlier routing not mentioned; LPV-LFR departure stated against Drenth) | Clear step toward Dirk's style: prose 1222 words vs 1705 in run 6 and 1292 in the original, while adding the header cautions; 136 words per display (diagnostic); RK4 paragraph close to the approved example; displays kept except textbook ones (encoder structure merged into eq:aug_encoder). Open question for Dirk: the 'omitted vibration mode adds model order' opening of III-A now follows the Garcia list in the section opening; whether that counts as established context. III on p. 5 to 7. 246k tokens. | v7: when cutting, do not merge; semicolon or 'and' joining independent facts makes two sentences; review check for sentences with three facts. |
| 8 (Sec. IV) | v7 | R2/R6 fail: packing (14 of 69 sentences over 30 words, 6 semicolons joining independent facts, e.g. the expansion-point sentence with three facts); S12 borderline: several reasons in prose come from the agent's own reading (offset 'state-like direction', fourth-order stencil, Thms 16/18), flagged in its report | O14 partial: held-out overlap measure dropped (not in code; replaced by measures the code computes, defensible) | Register and context are right (expert terms, no decoration, starts from Section III's established non-uniqueness); words 1347 vs 1073 original (135 per display, diagnostic). The v7 one-line rule against merging did not change behaviour. 247k tokens. | v8: a second worked example (packed vs to-the-point, taken from this run) in the Concise section; explicit packing pass in the review loop (list semicolons and sentences over 30 words; cut first, then split). Next: run 9 on Section III with v8. |
| 9 (Sec. III) | v8 | none hard. R2 met: 0 semicolons, 5 of 84 sentences over 30 words (max 40); R6: deletion audit finds little (one weak 'with input u.'); one claim (controller suppresses model error inside its bandwidth) rests on superseded D-139 data, flagged by the agent | C21 partial (earlier routing not mentioned, left out by the reader test and flagged) | PASS. 1316 prose words vs 1292 original while adding all header cautions; 132 per display (diagnostic); RK4 paragraph matches the approved example almost verbatim; u = P u_act kept. Question for Dirk: III-A still opens with 'An omitted vibration mode requires additional model order' right after the Garcia list (base resonance, cross-arm vibration) in the section opening; the agent treats that list as the established context. 288k tokens. | none: run 10 on Section IV with unchanged v8 to test the stop criterion. |
| 10 (Sec. IV) | v8 | none hard. R2 met: 0 semicolons, 1 of 98 sentences over 30 words; two derived reasons in prose (offset exclusion, coordinate Jacobian) are method derivations, flagged; Thms 16/18 reason is the agent's reading of the paper | O8 stencil now a phrase (textbook, allowed); O14 second overlap measure and held-out use moved to Setup with a reason | PASS. 1366 prose words vs 1073 original while adding the header's justify-items; 137 per display (diagnostic); expert register, no decoration; starts from Section III's established non-uniqueness. 274k tokens. STOP CRITERION MET with runs 9 and 10 on v8. Preview: preview-run9-sec3_run10-sec4.pdf. | Awaiting Dirk's judgment. |
| Dirk on runs 9 and 10 | | | | 'An omitted vibration mode requires additional model order' does not fit: it states a specific mechanism as fact; the thesis has only established a rigid-body baseline. The argument must be general (if the omitted effects are dynamic, the six baseline states cannot represent them; refitting adds no states). | v9: rule 'argue from the general case, not an assumed instance' with this example; review check that no specific mechanism is stated as fact before the section that introduces it. Iteration resumes: run 11 Section III, run 12 Section IV, both v9. |
| 11 (Sec. III) | v9 | none hard. R2: 0 semicolons, 2 of 82 sentences over 30 words; R5 fixed: III-A opens from the general case ('If the omitted effects are dynamic, the six baseline states cannot represent them'); the X/Y routing sentence is hedged ('may couple') | C20 handled in general form (added states are a gauge, no physical meaning 'also when the omitted effect is a physical mode') instead of naming the absorber before Setup, consistent with v9; C21 open-loop departure stated against Eq. (22) | PASS. 1291 prose words (original 1292), 129 per display (diagnostic); compiles, III on p. 5 to 7. Output map not learned now has a source (Kessels Remark 5.3). 257k tokens. | none: run 12 on Section IV with unchanged v9. |
| 12 (Sec. IV) | v9 | none hard. R2: 0 semicolons, 0 of 95 sentences over 30 words (max 29); one sentence on rollout-fitted coefficients sits inside the tuple-motivation paragraph (slight ordering drift); offset-exclusion and coordinate-equivalence reasons are derivations, flagged (the offset reason also appears in the original draft) | O8 stencil as a phrase; O14 cos-angle measure replaced by the measure the code computes; forward-sensitivity appendix pointer dropped as implementation detail | PASS. 1306 prose words vs 1073 original, every Choice and Caution item of the header covered; compiles, IV on p. 7 to 9. 279k tokens. STOP CRITERION MET with runs 11 and 12 on v9. Preview: preview-run11-sec3_run12-sec4.pdf. | Awaiting Dirk's judgment. |
| Dirk on runs 11 and 12 | | | | Unsure the sections state our contributions and the gantry-specific realisation; suspected cause: the skill does not point to the right information (implemented pipeline, decisions.md, references.md, correct OBC via THESIS_ARM + _THESIS_FIXED). Each writing session must verify the map itself and read the relevant literature itself. | v10: reference/sources.md (source map built by one agent, key claims spot-checked); read step uses it but verifies every stated value in code and every citation in the PDF; contribution rule (adopted / adapted with difference / ours), section opening names its contribution; outline tags each paragraph; review checks for contribution visibility, code-confirmed configuration, PDF-checked citations; style profile pattern 5 revised. New rubric checks K1 to K3 below. |
| 13 (Sec. III) | v10 | none hard. R2: 0 semicolons, 2 of 93 sentences over 30 words; K1 met (opening: extends dynamic-parallel augmentation to the self-scheduled baseline and to feedback records); K2 met (adapted parts each state their difference: normalisation source, no bypass at zero init, W^b at Y = 0 with nominal parameters, Kaiming vs Xavier, Drenth contrast; residual form marked as ours; RK4 boundary as ours); K3 met (agent confirmed each mechanism in code, citations against PDFs) | none | PASS. 1402 prose words (more than run 11's 1291 because each adaptation now states its difference: required content under v10); 10 displays (RK4 stages removed only). 'Section VI tests separately' promise flagged. | none: run 14 on Section IV with unchanged v10. |
| 14 (Sec. IV) | v10 | none hard. R2: 0 semicolons, 1 of 90 sentences over 30 words; K1 met (extends OBC from linear-in-parameters I/O models to the self-scheduled state transition); K2 met (offset exclusion, per-epoch expansion point vs Gyorok's two options, data tuples answering the obstacle Gyorok names in Sec. 6, recovery condition as state-level counterpart); K3 met (tangent basis, per-epoch refresh, no affine arm, validation re-solve, rebuild at the selected model). Minor: rank tolerance and rank-loss policy are implementation detail that could move to Setup | none (cos-angle measure replaced by the one the code computes) | PASS. 1265 prose words, all 10 displays kept; compiles, IV on p. 7 to 9. 272k tokens. STOP CRITERION MET with runs 13 and 14 on v10 (run 14 is also the agreed check-in point). Preview: preview-run13-sec3_run14-sec4.pdf. | Awaiting Dirk's judgment. |
| Dirk on runs 13 and 14 | | | | Sentences look good; unsure the sections capture what needs to be captured. Agreed: content selection is Dirk's judgment and cannot be tested with pre-approved agent runs (all 14 runs skipped the outline gate). | v11: outline split into 2a (three to six claims the section must establish, each with what needs it; proposed omissions; source conflicts; Dirk answers) and 2b (paragraph outline from the agreed claims); approved claims recorded as a '% MUST ESTABLISH' block in the section header. Next test: a live session with Dirk on Section III, stopping at 2a. |
| Dirk 2026-10-06 | | | | Wants to keep `\todo{}`s and per-(sub)section bullets of what must be stated (proposal from another session, adjusted here). | v12: open points are `\todo`s in the file (paragraph end, 'Missing: X. Candidate: Y (basis)', only open decisions or unverifiable facts, existing todos kept until resolved); `\ifdraft` content bullets under every (sub)section, written after 2b approval and before prose, kept and updated; review checks bullet coverage and todo form; delivery points to the todos. Rubric: S8 revised, T1 and T2 added, metrics exclude todos and bullets, test prompt pre-approves 2a and 2b. |
