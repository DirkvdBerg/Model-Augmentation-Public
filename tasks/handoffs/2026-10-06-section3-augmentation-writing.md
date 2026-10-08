# Handoff: write thesis Section III (Dynamic LPV-LFR Augmentation) with the thesis-section skill
**From**: session of 2026-10-05/06 | **Branch**: Augmentation | **Effort suggested**: high (writing and source checking, no computation)

## 1. Task
Write Section III of the thesis paper, `Thesis-writeup/Writing/sections/03_augmentation.tex`, using the
`thesis-section` skill (`.claude/skills/thesis-section/SKILL.md`). The user's words: the previous
session "has not written a proper section 3", the text was "samen gepropt" (crammed), tried to do too
much per sentence, and did not let equations carry the content. The deliverable is thinking on paper
and LaTeX only: no runs, no numerical scripts. The user requires that open points stay in the `.tex`
as `\todo{}` and that each section and subsection keeps `\ifdraft` bullets of what it must state; the
skill on disk already implements both (SKILL.md "Open points are `\todo{}`s in the file" and "Draft
bullets per (sub)section"). Run the skill's workflow on Section III from step 1, with the user's
approval at 2a and 2b before any prose.

This handoff supplies evidence pointers, prior agreements and boundaries, not a pre-approved
argument. The next session must discuss what Section III needs to establish with the user.
Do not turn the current draft or this handoff's bullets directly into prose. Content and outline
approval may explicitly leave scientific choices unresolved as `\todo{}`s.

### Content structure to discuss at 2a/2b
The retained three-part structure is augmented model, window initialisation, then reference-driven
rollout and objective. Paragraph content and order still require approval.

- **III-A: What model are we identifying?** Open with the dynamic-parallel update equation, then
  define the physical/additional-state partition, the single network's two output partitions and
  the fixed actuator-position output. Explain the CT/RK4 baseline and discrete correction boundary
  without re-deriving Section II or expanding textbook RK4 stages. Attribute the topology to
  Hoekstra and acknowledge Drenth's LPV precedent. Explain baseline-preserving initialisation;
  end with the non-unique physical/network split that motivates Section IV. Additional states
  provide independent memory and increased model order, not identifiable absorber coordinates.
- **III-B: How does each training window obtain its initial state?** Establish the need for an
  encoder, display the history-to-state map and linear-plus-nonlinear construction, and attribute
  the analytical physical-state map to Hoekstra's encoder paper. Distinguish that adopted method
  from the gantry operating point and additional-row initialisation. Keep the physical-state
  encoder choice open; move numerical settings and initialization-distribution details to Setup.
- **III-C: What response are we fitting under feedback?** Start from the same reference,
  feedforward and known controller in recorded/model loops. Derive the residual implementation by
  subtracting the controller equations, explicitly showing reference cancellation. Explain the
  controller-state difference and its initial condition, the avoided absolute-state reconstruction,
  and output/input/next-state timing. Give the joint output-error objective with the controller
  fixed. Closed-loop fit alone does not establish autonomous plant recovery or controller transfer.

Section III's share of the contribution is the implemented gantry realisation and identification
procedure, not a new dynamic-parallel topology or residual identity. Section IV carries the separate
orthogonality construction. Ten structurally identifiable baseline combinations do not by themselves
establish identifiability after joint estimation with an unrestricted network.

### Figure agreements to preserve
- Architecture: `Thesis-writeup/Writing/figures/tex/augmentation_structure.tex` is the author's
  edited included asset. Preserve it; do not overwrite it from the older
  `Thesis-writeup/Code/Figures/augmentation_structure.tex`. Its register exposes state-update timing.
- Closed loop: `Thesis-writeup/Writing/figures/closed_loop_same_reference.pdf`, generated from
  `Thesis-writeup/Code/Figures/closed_loop_same_reference.tikz.tex` with the standalone wrapper
  `closed_loop_same_reference.tex` and publisher `augmentation_diagrams.py` in the same directory.
  Build/provenance notes are in `Thesis-writeup/Code/Figures/README.md`; all active inputs are
  thesis-owned. Read these for context, but do not regenerate the figure in this writing task.
- The closed-loop figure shows only the recorded plant and augmented model sharing one reference
  and one feedforward. Signals are indexed; input-arrow spacing was adjusted with the user.
  Preserve the diagram: no residual panel and no extra `z^{-1}` block. Residual algebra belongs in
  the text. The caption also states that output is evaluated before input advances the state.
- State timing precisely: current output follows from current state, the controller computes the
  current input, then model/controller states advance to the next sample. No plant feedthrough;
  do not imply an additional computational delay or a delayed error signal.
- Do not reintroduce `augmentation_horizons.tex`, the older `augmentation_closed_loop.tex`, or the
  original three-loop PDF crop. Historical meeting figures are provenance, not build dependencies.

## 2. Out of scope
- Sections I, II, IV, V prose. Todos for the needed changes are already placed in I and II (see section 4);
  do not rewrite those sections. If Section III needs something from another section, list it.
- `docs/decisions.md` and `docs/references.md`: report stale entries, do not edit them (D-142 wording, see 5).
- Compiling the thesis (memory rule: the editor rebuilds on save; check `build/main.log` only after the user saves).
- Any run, MATLAB, or numerical check. Items needing a measurement go into a `\todo{}`.
- PS2 (D-236): excluded from the thesis method unless the user says otherwise.

## 3. Where things stand
- Inspect the current worktree and recent history read-only at session start. Do not rely on a
  historical commit, dirty-file list or TODO count; preserve all unrelated edits and running work.
- `03_augmentation.tex` holds an unapproved full draft. Treat prose and equations as candidate
  material requiring independent verification, not as a checked coherent formulation. In particular,
  reconcile input frames, normalisation, baseline-transition notation and loss scaling against the
  other sections and final code before reusing any equation.
- The active task is discussion and writing, not continuation of a diagnostic or training mission.

## 4. Prior corrections and evidence pointers
These are reported findings or derivations from earlier sessions, not blanket verification for the
new session. Some may overlap the skill's current `reference/sources.md`. Re-read primary passages
and code before using them; check the assumptions of project-specific derivations. Numerical
examples and historical configurations do not automatically transfer to the thesis data.

| Claim the draft got wrong | Correct statement | Evidence |
|-|-|-|
| "A static correction has no memory" | A static correction of the state update changes the dynamics of the existing states but cannot add model order; additional states can | Hoekstra EJC 2025 Sec. 2 ("static ... do not add additional states ... such as flexible modes") |
| "Added states are dead at initialisation" | With a zero output layer they initially do not reach the output; the pathway can become active as the output layer trains. This does not guarantee successful learning or justify a gradient-mechanics discussion in the method section | `model_augmentation/utils/torch_nets.py::zero_init_feed_forward_nn`; D-237 |
| "The controller must be biproper" for the step order | The order (y_hat_k from state, then u_hat_k, then advance) follows from the model having no feedthrough; controller feedthrough is only why it matters | `model_augmentation/fit_systems/closed_loop.py::_rollout_segment` |
| Closed loop justified by "drift dominates, integral action removes it" | Primary reason: the model must predict the response to a reference, feedforward and known controller (the machine's operating condition and the use case, incl. a changed controller). Supporting reason: X and Y have zero stiffness, so open-loop position errors from velocity errors persist; the controller acts on them only if the model in feedback with the controller is stable. The D-142 numbers (2.35 vs 77 to 108) come from an older noiseless dataset and belong in Results after re-measuring | D-139, D-142, `01_introduction.tex` use case |
| "Departure from the research plan" in the text | Internal. Contrast with Hoekstra's open-loop truncated simulation instead | `hoekstra2026lfr` Eq. (22) |
| "The baseline is kept as derived (Garcia)" | Wrong: the baseline omits Garcia's Coulomb friction. Reason documented in D-204 (friction kept out of the baseline by construction so the experiment tests whether the augmentation captures it) | `garcia2013model` Sec. 2.2 lists cc1, cc2, ccy; todo inserted after eq. `damping_stiffness_matrices` in `02_system_baseline.tex` |
| "Garcia neglects friction" | False. Garcia Sec. 2.4 (pp. 15 to 16) neglects: base resonance (37.7 Hz support structure), cross-arm vibration on the flexible plates, detent forces, friction variation along the stroke | `literature/gantry/garcia2013_gantry-decoupling-control.pdf`; proposed sentence in a todo in `01_introduction.tex` |
| Garcia "repeated eigenvalues" (p. 6, p. 27) as a source for poles at s = 0 | It supports "X and Y are rigid-body motions" only. The two eigenvalues at the origin of A_c(Y) are this work's derivation: K has rank 1 (only Theta stiff), det(s^2 M + s C + K) has lowest term s^2 k(c_g1+c_g2)c_y, nonzero for every Y | eq. `damping_stiffness_matrices`; own derivation |
| Novelty "augmentation of a self-scheduled LPV-LFR baseline" | Drenth Sec. 5.2 already formulates it, including additional states x_a. State the contribution narrowly: gantry realisation (physical baseline, ten structurally identifiable baseline combinations, CT/RK4 boundary) and closed-loop identification with joint parameter estimation. Orthogonality is Section IV's contribution | `literature/books/drenth2025_lpv-lfr-thesis.pdf` Ch. 5 |
| Normalisation "as Hoekstra Eq. (28)" | Ours is affine (means and stds from the training records), Eq. (28) is scale only. The output needs no offset because mu_y = P^T mu_q (offset cancels) | `scripts/gantry/gantry_dynamic/data.py::compute_normalization` (Cd_norm has no offset) |
| Proposed residual check "rollout with y_hat = y gives u_hat = u" | Vacuous, passes by construction. Candidate independent check: drive the re-discretised controller with recorded r - y and compare its feedback force with the recorded one, using compatible initial states, timing, coordinate frames and signal processing. This needs separate authorization; do not run it in the writing session | D-221, D-141 |
| x^c_tau = 0 "is Kessels' Remark 5.4" (D-142) | Same condition (model controller starts in the recorded controller state, s_hat = s), different method: no reconstruction needed in the residual form. Keep condition and method distinct; D-142 wording is stale | `closed_loop.py::closed_loop_rollout` comment; Kessels thesis Remark 5.4 |
| Finite-difference velocities as initial state | The code comment's sigma_v about 0.79 m/s is an SNR-60 example with sigma_n = 1.4e-4, not evidence for the current thesis noise/filtering. Raw differencing can amplify noise, but its usability requires assessment under the actual sampling, filtering and coordinate transform. This concerns velocity estimation, not a justification for encoding measured positions | `data.py::compute_normalization` comment block, explicitly labelled SNR 60 |

Also reported from code inspection (re-check before drafting): normalisation applied inside the RK4 derivative; Y read per RK4
stage; force enters as P u_stage; output Cd = [P^T 0]; one RK4 substep in production; the controller maps
actuator position error to actuator forces with normalisation folded in; tau_X = m_sum/(c_g1+c_g2) = 1.55 s,
tau_Y = m_h/c_y = 1.01 s from `gantry_ss.py` nominal values (decoupled approximation, matches D-139).

## 5. Assumed but not verified (verify yourself; the user asked for this)
Citation pointers the previous session read. Read each passage yourself in the PDF before citing; the
skill requires it and the user wants independent verification.
- `hoekstra2025lfr` (EJC): Sec. 2 static vs dynamic augmentation, "flexible modes"; Table 1 S-DP.
- `hoekstra2026lfr` (arXiv 2602.17297, `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`):
  Table 1 and Eq. (3) S-DP; Eq. (22) truncated loss driven by recorded u; Eq. (28) normalisation;
  Eq. (29) baseline behaviour at initialisation; Eq. (30) fitted baseline encoder; Eq. (31) Xavier for
  the augmented encoder; Sec. 5.2 non-unique split; Sec. 5.4.3 random linear part allowed; Cond. 6 and 8,
  Thm. 9 well-posedness.
- `hoekstra2026encoder` (arXiv 2602.13108): Eq. (8) linear plus nonlinear encoder; Eqs. (16), (17) W^b;
  Eqs. (30) to (32) linearisation about an equilibrium; its experiment uses n_xa = 0 (no evidence for W^a).
- `beintema2023subnet`: encoder and truncated simulation (not read in the previous session).
- `kessels2025ai` (thesis; PDF page = thesis page + 26): Eq. (5.12) and (5.13a) to (5.13d), thesis
  pp. 156 to 157; Remark 5.1 (no input dependence in the output augmentation under feedback); Remark 5.3
  (positions initialised from measurement); Remark 5.4 (controller-state reconstruction); Remark 5.6
  (open-loop bias from noise plus controller integrator, now possibly relevant since the data are noisy,
  D-225); Sec. 5.3.2.3 (controller C2, gains changed by 20 %).
- `drenth2025thesis` Sec. 5.2: whether its experiments actually use dynamic augmentation is unverified.
- `garcia2013model` Sec. 2.2 and 2.4: verified as quoted in section 4; re-read before citing.

Scientific choices requiring justification (discuss at 2a; they may remain explicit `\todo{}`s):
1. Why the network writes all six physical rows including the measured positions (D-103 covers only "X and Y must be routed").
2. Why the encoder estimates the measured positions instead of taking them from the measurement (Kessels Remark 5.3 does the latter).
3. Why the encoder linearisation is at Y = 0. Verify the actual operating point rather than attributing this choice to the encoder paper.
4. Why the full window is scored (D-220 rules burn-in out by user decision only).
5. W^a Kaiming in the code vs Xavier in Hoekstra Eq. (31): a `\todo{}` only, no discussion in the prose (user, 2026-10-06).

Verification and reporting obligations (not optional disclosure decisions):
- Verify which parameter values initialise W^b, including whether nominal parameters are used in
  detuned runs (skill sources.md trap 1). Disclose the actual construction and available prior
  information; do not silently equate it with the current rollout parameters.
- Reconcile the input frame: Section II uses u for P u_act (generalised force), while Section III's
  draft uses u for u_act. Preserve established notation and obtain the user's decision on any
  cross-section change; define recorded and model inputs unambiguously.
- Verify the code's output-channel averaging (sources.md trap 14). Match the displayed loss to
  the implemented criterion, or explicitly identify a constant scaling convention. Do not leave
  a factor mismatch unreported simply because the minimiser may be unchanged.
- Verify the affine normalisation and output-offset cancellation against the actual code and
  statistics. Reconcile `f_base`/`bar f_base` with Section IV rather than silently introducing
  competing transition definitions.

## 6. Tried and failed
- Drafting before agreeing on claims and paragraph jobs produced repeated rewrites rather than a
  settled argument. Obtain 2a/2b approval before prose; a TODO is preferable to an invented reason.
- Compression displaced load-bearing equations into prose; the opposite extreme repeated Section II
  and textbook RK4 stages. Display model-specific constructions, not routine machinery. Introduce
  the augmented model before its integration and scaling details.
- Sentence-by-sentence fact lists and attribution-heavy introductions lacked connected reasoning.
  Give each paragraph one job and cite the adopted method where it is used.
- Decision-log slogans, older numerical examples and simulation mechanisms became general claims.
  Keep evidence strength and provenance explicit; put experiment settings and measured evidence in
  Setup/Results, not in the general method argument.

## 7. Achieved
- Section I: todo with a verified replacement sentence for the Garcia claim (`01_introduction.tex`, after the Garcia todo).
- Section II: todo with the proposed Coulomb-omission sentence and its D-204 reason (`02_system_baseline.tex`, after eq. `damping_stiffness_matrices`); integration pointer moved to `sec:aug_structure`; header note says Section III owns the RK4 scheme, Section V only its step size.
- Section V: absorber figure moved to the top of "Simulation benchmark and excitation" (`05_setup.tex`).
- `Thesis-writeup/Writing/README.md`: "Order of work" block, claim rules (strength, method vs experiment, checks must be able to fail), softened Phase 1 sentence rule. The skill states which README parts it replaces.
- Section III contains candidate constructions: augmented model, network output split, zero init,
  encoder and analytical initialisation, controller loops, residual form and loss. Inspect the
  current file and verify each construction before reuse. Baseline-linearisation matrices and
  controller matrices must have distinct notation if both appear.

## 8. The open question
What must this section establish, and what is the minimum argument needed to establish it?
Discuss the structure and evidence above with the user at 2a. Approval establishes content and
boundaries, not evidence for unresolved choices. The user may approve an outline containing TODOs;
do not force every scientific justification to be settled before drafting.

## 9. Next action
Invoke the `thesis-section` skill on Section III and complete step 1 (read Section II in full, the
Section III header, the skill references, the code it names, and the PDFs in section 5), then present
step 2a: the claims Section III must establish, starting from the corrections in section 4 and the
open decisions in section 5. Wait for the user's corrections before 2b.

## 10. Acceptance criterion
The user approves 2a (what Section III must establish), then 2b (paragraph outline, written as bullets in the file), then the drafted section. Every check in the skill's review loop passes, every citation was read in its PDF in your session, every open point is a `\todo{}` in the file, and the user marks no paragraph as unacceptable.

## 11. Read these first
1. `.claude/skills/thesis-section/SKILL.md` and its `reference/sources.md`, `reference/style-profile.md`: the method, the thesis configuration, the traps.
2. `Thesis-writeup/Writing/sections/02_system_baseline.tex` in full: what Section III may build on, and its notation.
3. `Thesis-writeup/Writing/sections/03_augmentation.tex`: header WRITING GUIDE and SOURCE MAP, and the current draft's equations.
4. `Thesis-writeup/Writing/README.md` "Writing a section" and "Order of work".
5. The PDFs in section 5 for every citation you use.

## 12. Do not
- Do not write prose before 2a and 2b are approved.
- Do not state the absorber, friction truth, controller K1 design, sample rate or any measured number in Section III.
- Do not present the residual form or the dynamic-parallel topology as new, and do not cite Garcia for poles at the origin.
- Do not discuss Kaiming vs Xavier in the prose; a `\todo{}` only.
- Do not propose a verification that passes by construction (see the residual check in section 4).

## 13. Operational
Read-only inspection is allowed: file/PDF reading and extraction, search, git status/diff/history,
and reading the editor build log. No training, simulations, numerical checks, figure regeneration
or thesis compilation in this writing task.

Content edits are restricted to `Thesis-writeup/Writing/sections/03_augmentation.tex`. Preserve the
other sections, skill files and figure assets. Use the available structured editing tool
(`apply_patch` in Codex, Edit/Write in Claude); do not pipe LaTeX through shell heredocs, which
previously mangled backslashes. OneDrive can briefly lock the `.tex` (Errno 22); retry once. Check
balanced braces/environments and absence of em dashes. After the user saves and the editor rebuilds,
inspect `build/main.log`; without that rebuild, report compilation as unverified rather than claiming
the section passed.

## 14. Delegation
None. All reading is targeted (named files and PDFs); no Explore subagent.
