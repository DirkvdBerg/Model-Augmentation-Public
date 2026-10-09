# Writing

Self-contained LaTeX project for the 12-page IEEE paper. Compile `main.tex`,
nothing else. No `.tex` file here may contain a `..` path: everything LaTeX
needs lives under `Writing/`, so this directory can be zipped and handed to
Overleaf or a supervisor as is.

## Editor build and SyncTeX invariant

The repository VS Code settings at the root and in `Writing/.vscode/`
must preserve the following workflow:

1. Saving any included `.tex` file (Ctrl+S) rebuilds `main.tex`, never the
   section file in isolation. Changes written from outside the editor do not
   start a build (`onSave`): with `onFileChange`, AI edits, OneDrive and git
   started overlapping builds that aborted on a locked `main.aux` and left no
   `main.synctex.gz`.
2. The only editor PDF is `build/main.pdf`, produced together with
   `build/main.synctex.gz` using `-synctex=1`.
3. The PDF is opened with LaTeX Workshop's tab viewer; double-clicking it must
   navigate back to the corresponding source line.

Do not change the root-file pattern, output directory, recipe, SyncTeX flag, or
viewer configuration without testing both an included-section save and reverse
SyncTeX. Do not add per-section root directives: `main.tex` is the sole root.

AI edits of any `.tex` file here must leave this workflow working. Dirk relies
on it for every save.

- Edit content only. Never add `% !TEX root`, `\documentclass`,
  `\begin{document}`, or a `..` path to a section file, and never change
  `main.tex`'s structure, `.latexmkrc`, or either `.vscode/settings.json` unless
  asked.
- Leave the file compiling: braces and environments balanced, and `%`, `&`,
  `_`, `#` escaped outside math. A failed build leaves `build/main.pdf` stale,
  so saving no longer refreshes the PDF and double-clicking lands on old lines.
- Only the editor recipe writes build output, so there is exactly one PDF and
  one SyncTeX file. Test compiles go to a scratch directory, never to
  `Writing/` or `build/`.
- Dirk saves the file before asking for an AI edit. An edit on disk while the
  editor holds unsaved changes causes a save conflict, and that save then
  triggers no rebuild.

Done when: after the automatic rebuild, `build/main.log` has no line starting
with `!`.

| Path | Role |
|-|-|
| `main.tex` | class, inputs, title block. Roughly 70 lines, no content. |
| `util/include.tex` | packages only. Swap templates by touching this file. |
| `util/format.tex` | macros, notation, draft switch. Notation matches `state-level-obc-derivation.tex`. |
| `sections/NN_*.tex` | one file per section, flat, numeric prefix. Each header states its page budget and what it owns. |
| `figures/` | generated PDFs, `\graphicspath` points here and only here. Never hand-edited. |
| `tables/` | generated tabular fragments, `\input` from a section. Numbers are never retyped. |
| `refs.bib` | own entries, cite keys matching `docs/references.md`. |
| `IEEEtran.cls`, `IEEEtran.bst`, `IEEEabrv.bib` | from the shipped IEEE template, kept at project root so the build is standalone. |
| `build/` | aux, log and output PDF. |
| `reference/` | the original untouched IEEE template and BST distribution, for lookup only. Not part of the build. |

## Paper structure and ownership

Model names below are candidates (calligraphic, as plain $M$ clashes with
$M(Y)$); the notation is Dirk's decision, so each use carries the `\todo` of
the open conflicts list.

| Section | Owns |
|-|-|
| II Baseline | coordinates, equations of motion, LPV-LFR, identifiable combinations; models $\mathcal{M}_{\mathrm{LPV}}$, $\mathcal{M}_{\mathrm{LTI}}$ |
| III Augmentation | augmented model, encoder, closed-loop criterion, added-state initialisation (PS2), optimiser and checkpoint selection; models $\mathcal{M}_{\mathrm{U}}$, $\mathcal{M}_{\mathrm{PS2}}$ |
| IV OBC | projected model and the scope of its guarantee; model $\mathcal{M}_{\mathrm{OBC}}$ |
| V Experiment design (`05_setup.tex`) | data-generating system in equations, data and splits, comparator $\mathcal{M}_{\mathrm{BB}}$, table of compared models, metrics, hyperparameter table |
| VI Results | evidence per compared model, against criteria stated first |
| Appendix | expanded matrices, proofs, notation table, implementation checks |

- **Method owns the math of what it introduces, including what is trained.**
  Test: if changing the item changes the model or how it is identified, it
  belongs to a method section.
- **Experiment design never explains training.** It defines what the models are
  tested on, and gives the values of symbols a method section already defined,
  as rows of the hyperparameter table with a basis column (derived, inherited,
  tuned, compute-limited). A training sentence that seems needed there is
  missing from the method section.
- **Observed quantities** (ranks, compatibility checks, convergence) belong to
  Results or the appendix. A method section displays a quantity that checks
  its own guarantee (an overlap, a residual statistic) only if Experiment
  design lists it as a metric or `DOC/THESIS-RESULTS.md` reports it; otherwise
  it states the guarantee and leaves the quantity to a `\todo`.
- **One name per compared model**, defined where its equations are complete,
  with its input and output as the measured signals of Section V (a model
  simulated in closed loop names its external input, the reference or the
  recorded signals its residual form takes, not the actuator force), in a
  paragraph or subsection of its own, never as the tail of a construction's
  paragraph (Section II names $\mathcal{M}_{\mathrm{LPV}}$ and the frozen
  $\mathcal{M}_{\mathrm{LTI}}$ together in such a closing part). A variant
  with another state dimension is another compared model with its own name.
  The naming sentence names the model and cites its displays by number; it
  does not list their ingredients nor restate per-phase changes stated at
  their displays. Results
  use only these names. Each Results subsection opens with one sentence naming
  the models compared, their cost, start, and test records.

## Math standard

**Goal.** For every part of the method (for example our LPV-LFR baseline, the
augmentation from Hoekstra's framework, the OBC from Györök), the reader sees
through concise math what is built on the gantry, which signals go in and out,
and what is trained against which cost. A description of the process in words
is not enough, and formulas alone are not enough either: the math states what a
part is, the prose says why it is there, what was chosen and what follows.
Derivations and proofs appear only where they are this thesis's own
contribution, and only as far as needed. Other literature (for example
Beintema, Kessels) is cited where its component is used.

**References for the level.** Primary: Hoekstra's papers, which this thesis
builds on directly: the first-principles augmentation paper
(`literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`),
the EJC 2025 LFR augmentation paper and the encoder paper; structure in the
thesis-section skill's `reference/style-profile.md`. Secondary: Drenth et al.
2025 (`literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`), a former master
student of the group, as an example of one concrete realisation at this level.
What these papers do, and we follow (equation numbers checked 2026-10-08):

1. Every signal and map is defined with its space at first use, in a "where"
   clause ($x_k\in\mathbb{R}^{n_x}$ is the state, $f:\mathbb{X}\times\mathbb{U}\to\mathbb{X}$).
   A model's input and output are the signals the data hold, in the data's
   frame (actuator, not stage), or the map to them is part of its display.
   A parameter vector whose size depends on a Section V value (a network
   width) gets its space with that size as a symbol
   ($\theta_{\mathrm{aug}}\in\mathbb{R}^{n_{\theta}}$).
2. The data-generating system is stated before the model is fitted to it
   (Hoekstra 2026 Sec. 6.1 with Fig. 7 and Table 3; Drenth Eqs. 1 to 3, 23).
   For us: the simulated truth in Experiment design, the models in Sections
   II to IV.
3. The identification problem is one display: the cost over the trained
   parameters together with the model, encoder and initial-state equations it
   is subject to, each written as its own line of the display with the maps by
   symbol (Hoekstra 2026 Eq. 22a to 22e; Drenth Eq. 19). Constraints cited by
   equation number in the prose after the cost do not satisfy this; a model
   that changes one constraint of an already displayed problem displays the
   cost and the changed lines and cites the unchanged ones inside the
   display ("s.t. (22c) to (22e)"). A component's own subsection displays
   the component (a correction term, an encoder); the model with it
   substituted is displayed once, in the problem display. Repeating a constraint line of an earlier
   display inside the first problem display is required, not a restatement. The trained
   parameter vector and the coordinates it is minimised over are defined
   before the display. A quantity computed from the current model and held
   fixed during an update, or refreshed during training (a target, an
   expansion point), is named as such in the problem display's "where"
   clause, with a pointer to the component subsection that argues its rule. A component display writes the signals it takes inside the
   problem (the closed-loop input, not the recorded one), or its "where"
   clause states the substitution. The optimiser is one prose sentence or
   paragraph.
4. Initialisation and normalisation are displayed: what each block starts at
   and how signals are scaled (Hoekstra 2026 Eqs. 28 to 31; Drenth Eqs. 20,
   21). A standard distribution (Xavier) is named with its source in the
   "where" clause, not written out. The model structure is displayed first,
   in physical units (Hoekstra 2026 Eq. 6); the normalisation follows as the
   coordinates the identification runs in (Eq. 28), never before the model.
   A normalised signal gets its own symbol. The display gives each signal's
   scaling and centring (Hoekstra 2026 Eq. 28), so the reader knows which
   signal (stage or actuator frame) is scaled and whether it is centred; one
   sentence states the rule by which the physical-unit maps become normalised
   maps (each composed with the scalings), not a list of the maps, and no
   identity between statistics is displayed. A physical subsystem the model is
   interconnected with (the controller) enters later displays through the same
   scalings, defined in one sentence.
   Every display writes exactly the map the code computes: an offset the code
   subtracts (the encoder's centring of its physical rows) is a term of that
   display, or the display uses a symbol defined for the uncentred signal; no
   sentence after a display corrects it. Every display after the
   normalisation writes the normalised symbol, and all lines of one display use one
   set of coordinates (a map of $u^{n}$ on the right when $u^{n}$ is on the
   left). When a display cites unchanged lines of an earlier display written
   in the other set, its changed line is written in that set too, or the
   coordinate map is a line of the display.
5. Each subsection opens with the problem it solves, then the construction
   with its equation (Hoekstra 2026 Sec. 5.2: overparameterisation, then the
   remedy). A problem that follows from an already displayed map is stated
   in prose, not given its own display. Only the core own contribution gets a short formal statement with
   a short proof (Drenth Theorem 6), written as `\textit{Proposition N}` with
   `IEEEproof` (no preamble change).
6. The experiment is concise: the truth system with its parameter values, one
   paragraph on the data, the model settings, a hyperparameter table, the
   metric displayed (Hoekstra 2026 Sec. 6, Table 5; Drenth Sec. 6).

**Prose around the math.** A display's lead-in gives one or two sentences of
reasoning that introduce it and, one sentence each with its reason, every
choice that fixes an element of the display (its signals, coordinates, step
order, rate, the rows it acts on) and every condition it needs to exist or to
equal another form (a rank, an invertibility, a recorded input produced by the
same controller); such a choice or condition is never argued in the "where"
clause or after the display. After the display come a "where" clause and at
most one consequence. A property
that follows from a display (an orthogonality, a normal equation) is that
consequence whether written as a display line or a sentence; it is stated
once, and later uses cite it by number. A lead-in that
needs more (the problem, a contrast with a cited method, several reasons)
splits: the problem or contrast becomes the preceding paragraph with its own
topic sentence, and the lead-in keeps the sentences that introduce the
display. The section is
one coherent argument, not a list of formula and reason blocks: it opens with
a short roadmap, each subsection has one job, and each part runs problem,
construction, consequence, building only on what is already defined (as in
Hoekstra's and Drenth's papers). Content a display needs beyond its one
consequence (a choice and its reason, a contrast with a cited method, a
limitation) goes into the lead-in of the display it motivates or into its own
paragraph whose topic sentence introduces the next construction or display; a
contrast, condition or limitation of the display just shown never opens such
a paragraph, so no run of sentences or short paragraphs follows a display. A
limitation that Results or Discussion tests (feedback masking plant error) is
one clause with its pointer. A construction first described in words is the
lead-in of its display. A fact the section has already stated (an
invertibility, the free axes) is cited by equation or subsection number, in a
"where" clause too, never stated again; a later subsection whose problem rests
on it names it in one clause with that pointer, never re-derives it. Only
choices no display shows form one paragraph after the component's last
display; its topic sentence names the one property they jointly fix (when
and from what it is rebuilt, which directions it covers), never that choices
follow, and a choice that does not share that property moves to the lead-in
of the display it concerns or fails the reader test. Each choice is one or
two sentences with its reason. A
fallback for a failing case (keep the previous result, abort after repeated
failures) is implementation detail and stays out.

**Example (PS2, sketch agreed 2026-10-08):** two sentences on the problem (the
zero output layer gives the added states no gradient at the start, and plain
training afterwards learns static corrections or real lags instead of the
missing dynamics, with the evidence pointed to Results; `DOC/PS2-METHOD.md`
Sec. 2.2: the gate alone does not argue the need; the claim is stated no
stronger than `DOC/THESIS-RESULTS.md` reports it: the added states are used,
never "unused", and the error left is stated for the band of the omitted
dynamics, never as "most of the error"); the S1 objective displayed,
its lead-in saying that S1 runs alongside the minimisation of $V$ (one S1
update after every update of $V$, so both criteria are minimised at once),
the parameters S1 trains shown under its minimum, not read back in a
sentence; the S2 regression displayed with
one sentence; one consequence sentence saying why the two stages give the
added states the missing dynamics (S1 makes them carry information that
predicts the current model's closed-loop output residual, S2 makes
$g_{\mathrm{aug}}$ propagate them; `DOC/PS2-METHOD.md` Sec. 1, qualified by
its Sec. 6.2.1: predictive sufficiency is not guaranteed); one sentence each
for S3 and for what stays unchanged (S2 moves every parameter it minimises
over, shared hidden layers included, so the sentence names the maps that
change, $f_{\mathrm{aug}}$ included when they share layers, and what S3
starts from: the S2 weights with the lowest calibration error, which
`_fit_map` restores). About two displays and six sentences, lead-ins and the
attribution clause counted, no derivation. The displays and sentence jobs are
fixed, and each job carries the content given here. The method-defining
switching rules (screen, stopping) follow the
phase they end, one sentence each: the compared quantity as the code computes
it (an inline formula when it is a ratio, its normaliser included:
`_cal_unexplained` divides by the residual's own mean square on the
calibration records), the comparison, and a window symbol introduced here and
listed in the header for Section V, as the code states it (`PS2Spec`); the
maximum S1 update count (`s1_cap`, which ended S1 in the prototype runs) is
part of the stopping rule. The
records the quantity is evaluated on are named once, where it is defined.
These sentences and the model's naming paragraph do not count toward the six. The difference from Hefny et
al. is one clause in a display's lead-in. Each display minimises over the
parameters it trains; a held-fixed target follows Math standard item 3.
Optimiser schedules (which optimiser runs S1, its step size), window
bookkeeping (the minimum update count), how the fixed target is
stored and detached and the decoder's zero output layer stay out.

Derivations that are this work's own follow the Derivation policy below.

## Source map per subsection

The single list of what to read for each part of the paper. Built 2026-10-08
from the code and `docs/decisions.md`.

- **Code**: read on the thesis path (`THESIS_*` block of the entry file) on
  2026-10-08 and confirmed to do what the subsection describes.
- **Decisions**: the current governing entries, checked for later amendments;
  "not" lists superseded entries a writer must not use.
- **Papers**: CANDIDATE. `docs/references.md` marks none of the core keys as
  read, so read the cited passage in the PDF before citing (Reference
  verification gate). After verifying, add the date to the cell.
- Prefixes: `ENT` = `scripts/gantry/gantry_interconnect_dynamic.py`,
  `GD` = `scripts/gantry/gantry_dynamic/`, `FS` = `model_augmentation/fit_systems/`,
  `SYS` = `model_augmentation/systems/`, `TR` = `scripts/gantry/thesis-results/`,
  `DOC` = `Thesis-writeup/Documentation/`.
- Update rule: a session that supersedes a decision or changes a pointer updates
  its row in the same session.
- Rows group sources, not subsections: the skill's Order rules fix the
  subsections and their order, and a label that another section cites is
  kept on the part that now holds its content.

| Part | Papers (candidate) | Decisions | Code | Documentation |
|-|-|-|-|-|
| II-A System, EOM | garcia2013model (Sec. 2.2, 2.3, Eqs. 6, 9 to 11, Table I; accepted-manuscript PDF, cite sections and equations, no page numbers; verified 2026-10-09); the baseline adopts the lumped model of Sec. 2.2 with the small-angle simplification of Sec. 2.3 (Eqs. 9, 10), which already keeps (X, Theta, Y); its one adaptation is dropping the Coulomb terms of Eq. 6 (keeping Y is no adaptation: only the simplified control model (11) drops it); toth2010modeling (Sec. 7.3, Def. 7.2; verified 2026-10-09) | D-001, D-004, D-006, D-022 (baseline = FP model as derived; it argues against, not for, omitting Garcia's Coulomb term); frictionless Python baseline: D-204 keeps friction in the truth only (its Karnopp law superseded by the tanh law of D-224, D-226); its "Why friction in the truth only" (the effect is absent from the baseline by construction, so the learned part is tested on an unmodelled effect) needs the Section V truth, so II-A states the omission with that reason in one clause and the section's one pointer to Experiment design, and the conflict with D-022 goes in the header comment | `SYS/gantry_ss.py` module constants | `docs/fp-model-structure.md` |
| II-B LPV-LFR | drenth2025lpvlfr (Sec. 3.1 Def. 1, Sec. 3.2, Sec. 4.1 Thm. 6, which needs a diagonal Delta and scheduling in the unit infinity-ball, row 62; verified 2026-10-09; its Eq. 6 is discrete time and names the LTI part M, so cite the CT form from drenth2025thesis; Drenth's identification and augmentation method is discrete time (citation-log rows 60, 76), so keeping CT is a contrast with the method, never "as Drenth does"); drenth2025thesis (Sec. 2.1 Eqs. 2.1, 2.2, 2.4, Sec. 2.1.1; verified 2026-10-09); hoekstra2026lfr | D-017 (2026-03-19), D-018, D-020, D-021, D-229; continuous time: D-018 (no pre-discretisation, RK4 at runtime) with D-020 (keep the original parameter structure of $M(Y)$), so the reason is that $\theta$ stays in $M$, $C$, $K$; why the thesis needs the LFR (II-B's first sentence): D-017 (supervisor 2026-03-22: baseline in LPV-LFR form), research-plan Aspect 1 (explicit position dependence) and D-020 (structural analysis): the form separates $Y$ into $\Delta(Y)$ over constant matrices whose entries carry $	heta$, which II-C analyses; II-A's $M(Y) > 0$ already settles solvability, so eq:wellposed is the condition the form needs, never its need; eq:wellposed is cited by III-A in its well-posedness argument for the baseline block of Hoekstra's interconnection (hoekstra2025lfr Sec. 3.2, Thm. 1; verified 2026-10-09); D-020 (derivation and structural analysis), not drenth2025thesis Sec. 5.2 (Drenth's LPV-LFR augmentation, not used here); minimality open (D-017 open question), so the six-channel realisation gets a `\todo`, never a disclaimer | `SYS/gantry_ss.py::build_poly_constants`, `::build_G_matrix_entries`; `FS/blocks.py::Gantry_State_Block._deriv_with`; reference derivation `lpv_lfr_baseline/core/*` (not on the training path) | `docs/Thesis-documentation/LPV_LFR_Stepwise_Derivation/main.tex` |
| II-C Combinations, admissibility | ovchinnikov2021computing (Sec. 2, Def. 7, arXiv 2004.07774v3, not under literature/; verified 2026-10-09; it states the equivalence with Hong et al., so hong2020global is not cited beside it); gautier1990minimum (no local PDF, unverified) | D-191, D-190 item 1 as amended by D-229, D-229; not D-037, D-077, D-190 linear m_Delta | `FS/blocks.py::Reduced_Gantry_State_Block` (`COMBO_NAMES`, `combos_of`, `gauge_section`, `combinations_from_free`, `_m_diff_bound`, `admissibility`) | |
| II Frozen-LTI baseline | none | D-242 (step 1); D-010 (LTI ruled out as baseline) | `Gantry_State_Block` with `Y_op` set (capability only; no runner yet) | `DOC/THESIS-RESULTS.md` step 1 |
| III-A Augmented model | hoekstra2025lfr (Sec. 2, 3.2, 4.3, Thm. 1; verified 2026-10-09); hoekstra2026lfr (Table 1, Eqs. 28, 29, Sec. 5.2, 5.3, 6.3; verified 2026-10-09); gyorok2026obc (Abstract, Sec. 3; verified 2026-10-09); kessels2025ai (Remarks 5.1, 5.3: reason for $h_{\mathrm{aug}}=0$; verified 2026-10-09) | D-003, D-103, D-222, D-180, D-240, D-237, D-233, D-119; not D-068, D-150, D-151, D-158, D-160, D-239 (live ResNet), D-189 | `FS/blocks.py::Gantry_State_Block._rk4` (`up_sample` substeps of $T_s$/`up_sample`; the `ENT` module `CFG` sets `up_sample=1`, overriding the `GD/config.py` default 2, and `_THESIS_FIXED` keeps it, so one RK4 step over $T_s$; checked 2026-10-09), `Static_ANN_Block`, `Linear_Output_Block`; `GD/model.py::build_model`; `FS/interconnect.py::Interconnect.forward`; `model_augmentation/utils/torch_nets.py::zero_init_feed_forward_nn`; `GD/data.py::compute_normalization` | `DOC/AUGMENTATION-INITIALIZATION.md` |
| III-B Encoder | beintema2023subnet (Sec. 2 to 3; verified 2026-10-09); hoekstra2026encoder (Eqs. 8, 10 to 17, 30 to 32, Sec. 4.2, 4.4; verified 2026-10-09; Sec. 3.2 not yet); hoekstra2026lfr (Eqs. 30, 31; verified 2026-10-09); sutanto2020encoding (unverified, no references.md row) | D-233, D-239 (W^a Xavier, gain 1), D-055, D-017 (2026-06-12), D-222 (lag 29); not D-064, D-167 | `FS/pre_encoder.py::linear_encoder_init_aug`, `::forward` (D-055 offsets: W^b and W^a act on scaled, uncentred histories, `x_off` subtracted from the physical rows only); `SYS/gantry_linearization.py::gantry_linearize_and_discretize` (nominal parameters, which equal the simulated truth's, so $W^b$ falls under the SKILL oracle rule); `GD/model.py::get_encoder_dims` | `DOC/AUGMENTATION-INITIALIZATION.md` Sec. 4 |
| III-C Closed-loop identification, optimiser, selection | kessels2025ai (Eqs. 5.12, 5.13, Remarks 5.1, 5.3, 5.4, 5.6, Ch. 5 summary; verified 2026-10-09); forssell1999revisited (verified 2026-10-09 in the LiU report LiTH-ISY-R-2021, Sec. 5.2.1; the Automatica 1999 numbering is unchecked, so cite it without a section number or add a `\todo`); beintema2023subnet (Cond. 1; verified 2026-10-09); hoekstra2026lfr (Eq. 22, Sec. 5.5; verified 2026-10-09; Eq. 22 already co-estimates $\theta_{\mathrm{base}}$, citation-log row 87, so joint estimation is no difference; its Eq. 22e encoder reads k-n to k-1 only, while the thesis encoder includes sample k as in hoekstra2026encoder Eqs. 11 to 17, `GD/model.py::get_encoder_dims` `na_right = 1`, so the criterion is adapted, not adopted); drenth2025lpvlfr (Sec. 5, Adam then L-BFGS-B, bounded, citation-log row 36; verified 2026-10-09; the thesis polish is unbounded L-BFGS, `FS/lbfgs_polish.py`, so cite it as adapted, never "following") | D-140 (its Ruled out (4) bars Kessels Remark 5.6 as an argument because the records were then noiseless; the thesis records are noisy, so a sentence using Remark 5.6 states the noise setting and a `\todo` asks whether the ban still holds), D-141, D-142, D-220, D-222, D-148 (2026-08-19), D-171, D-172, D-174, D-198, D-227, D-229 (m_Delta training coordinates, moved here from II-C); not D-107, D-178, D-146, D-127 | `FS/blocks.py::Reduced_Gantry_State_Block` (`combinations_from_free`); `FS/closed_loop.py::closed_loop_rollout`, `_rollout_segment`, `ClosedLoopSimulator.validation_error`; `GD/controller.py::build_cfb_at`, `build_controller_bank`, `build_closed_loop`; `FS/interconnect.py::SSE_Interconnect_Composed.loss`, `SSE_Interconnect.fit`; `GD/training.py::train_model_with_diagnostics`, `run_lbfgs_polish`; `FS/lbfgs_polish.py::lbfgs_polish`, `decide` with its meters `generic_meters` (`param_loss`) and `GD/training.py::_gantry_meters` (`combo_err` against the true parameters, an oracle under the SKILL oracle rule, $m_\Delta$ scaled by the mean actuator mass; `orth_frac`, `V_orth` only when an orthogonality penalty is attached) | `DOC/TRAINING-DESIGN.md` |
| III-D PS2 | hefny2015supervised (Sec. 2, arXiv 1505.05310v2, citation-log row 86; Sec. 3 is Related Work), downey2017psrnn (Sec. 4.2) (both verified 2026-10-09; refs.bib entries added in cycle 01 with `% CHECK`) | D-234, D-235, D-236, D-240, D-243; not the ps2ctl tier | `FS/ps2_opening.py::PS2Opening` (`after_step`: one S1 update after every update of $V$, `FS/interconnect.py` fit loop; `_sample(pair=True)`: S2 pairs are samples $k$, $k+1$ drawn anywhere in the fit records, not consecutive window starts), `PS2Spec`, `calibration_split`; seam in `GD/training.py::train_model_with_diagnostics` and `FS/interconnect.py::SSE_Interconnect.fit` (`THESIS_PS2=1`) | `DOC/PS2-METHOD.md` |
| IV-A Negation | gyorok2026obc (Ex. 2); gyorok2025l4dc (Ex. 1, Eq. 15); kessels2025ai (Sec. 5.3.1.3) (all verified 2026-10-09); the non-unique split is stated in III-C (hoekstra2026lfr Sec. 5.2), so IV names it by pointer and the Gyorok examples support only what is new here (the first-order directions), never a second statement of non-uniqueness | D-192, D-193; not D-034, D-076 | | |
| IV-B OBC construction | gyorok2026obc (Eqs. 8, 10, 11, 13, Lemma 3); gyorok2025l4dc (Sec. 4, Eqs. 15 to 18: point updated at every evaluation or fixed; Sec. 3, Remark 2, linear case: the projection "can be updated" with current state estimates, the SVD recomputed at the start of each epoch, citation-log row 75; a permitted option of the linear case, not a third scheme beside Sec. 4's, the precedent for the per-epoch refresh, adapted because here the expansion point is refreshed; Sec. 4 after Eq. (19): a fixed point is argued valid, row 81) (all verified 2026-10-09) | D-192, D-194, D-195, D-198, D-201, D-202, D-203, D-229; D-190 items 5, 6 (differentiated coefficient, per-epoch refresh and its sensitivity reason, still governing) and its Why (reduced coordinates: the raw Jacobian has nullity 4, which inflates the pseudoinverse's condition number; a detached coefficient gives exactly the unprojected gradient, measured); D-232 "OBC velocity result" (noisy reference velocities PASS, no change); D-186 of 2026-09-10 (trajectory basis) reasons 1, 2 for the omitted offset column and its caveat (a contribution cancelling the nominal response is charged only through the protected range), as D-192 gives no reason, and its Ruled out (1) (no sentence words the correction as preventing or removing negation); not D-111, D-185, D-186 of the latent-state penalty, D-190 item 3 ([J, c]), D-200, D-183 | `GD/obc_gantry.py::build_reference_set`, `make_basis_builder`, `make_reference_field`, `GantryOBCCorrection`, `attach_obc`; `FS/obc.py::build_stacked_sensitivity`, `OBCBasis`, `OBCLifecycle` (the Jacobian is taken in the free training coordinates through `transition_from_free`, not in $\theta_{\mathrm{base}}$; the trained model uses every stencil tuple of each record, while the D-203 measurement drops the last one, so a statement on the restricted set names it as such; the velocity stencil's order has no decision entry, but its reason is in the listed `obc-gantry-one-step-implementation-plan.tex` after eq. reference-reconstruction: at 4 kHz the fourth-order gain error at 230 Hz is below 0.2 percent against about 2.2 percent for second order, which matters because the damping directions multiply velocity; the code trims two samples per record end instead of the plan's boundary formulas; the tuples hold measured positions and differenced velocities, not recorded states, and neither D-192 nor the plan gives a reason for measured over simulated (open- or closed-loop) reference states, so that choice is a `\todo` candidate) | `scripts/gantry/orthogonal-by-construction/documentation/` (state-level OBC derivations; candidate, check against D-192) |
| IV-C Identification of $\mathcal{M}_{\mathrm{OBC}}$ | hoekstra2026lfr (Eq. 22; verified 2026-10-09) for the reused problem of III-C | D-190 items 5, 6 | `FS/interconnect.py::SSE_Interconnect_Composed.loss` (coefficient solved once per objective, expansion point a detached buffer), `_sync_prediction_state_for_validation` (coefficient re-solved for validation, basis rebuilt after the selected checkpoint is restored) | |
| IV-D Scope, recovery condition | gyorok2026obc (Assumption 1, Cond. 4, Remark 5, Assumption 6, Thm. 7, Eq. 21, Sec. 4.2 Eq. 25, Thms. 16, 18; verified 2026-10-09) | `DOC/THESIS-RESULTS.md` step 5 limit; noise setting: the reference velocities are differenced from noisy positions (`obc-gantry-one-step-implementation-plan.tex` after eq. reference-reconstruction), so the discrepancy also holds reconstruction and noise error, and Gyorok's Assumption 6 is realistic only for noise-free data (citation-log row 45); D-186 (2026-09-10) caveat on the omitted offset (it concerns a network term along the baseline response, so the reason the omission still holds must answer that term, not the true discrepancy; a reason D-186 does not state is a `\todo` candidate); D-213, D-214 design history only (THESIS-RESULTS R5: no T_OA data) | | |
| V Data-generating system | none (lee2020feeddrive belongs to the superseded Karnopp friction) | D-224, D-226, D-225, D-238, D-230, D-231, D-232; D-204 only for "friction in the truth only", not its Karnopp law; not D-209, D-212, D-218 | `Thesis-writeup/Code/Data/gantrySystemExtendedTanh.m`, `generate_thesis_data.m`, `tdg_*.m` | `DOC/DATA-DESIGN.md`, `DOC/NOISE-INJECTION.md` |
| V Data, splits, excitation | none (Geerardyn 2013 has no key) | D-221 (splits 18/6/12), D-219 (band); not D-216, D-217, D-188 | `GD/data.py::load_datasets`, `record_files`, `THESIS_DATASETS`, `_decimate_y`, `_resample_u`; `GD/config.py::y_decimation`, `record_trim` | `DOC/DATA-DESIGN.md` Sec. 7, `DOC/EXCITATION-VALIDATION.md` |
| V Controller, transfer case | kessels2025ai | D-221 (K1 at Y = 0; E5 gain 1.2, evaluation only), D-141 | `GD/controller.py::register_stored_controller` | `DOC/DATA-DESIGN.md` Sec. 5.11 |
| V Compared models, black box | beintema2023subnet | D-242, D-243; not D-134, D-136, D-196 | `scripts/gantry/blackbox-cl-trainable/train_bb_clmap.py`, `bbcl/jan_model.py` | `DOC/BLACKBOX-METHOD.md` |
| V Metrics | none | `DOC/THESIS-RESULTS.md` Sec. 2, D-227, D-228 | `GD/closed_loop_report.py::closed_loop_table`; `GD/evaluation.py::evaluate_and_save`; `TR/band_compare.py::band_rms` and `TR/inspect_checkpoint.py::clamped` (validation records only); noise floor and grey-box NRMS_e: no code yet | `DOC/THESIS-RESULTS.md` Sec. 2 |
| V Hyperparameters | none | D-222, D-220, D-243 | `ENT` `_THESIS_FIXED` and the per-run `replace(...)`; module-level `CFG` for values the block does not set | `DOC/TRAINING-DESIGN.md` Sec. 3 |
| VI Results | none | D-240, D-242, D-243 | as V Metrics | `DOC/THESIS-RESULTS.md` Sec. 3 (supersedes `RESULTS-DESIGN.md` Sec. 4 and 8, `TRAINING-DESIGN.md` Sec. 4) |

Open conflicts found while building the map (resolve before writing the part):

- PS2 is used in the reported results but has no subsection in `03_augmentation.tex`.
- W^a is Xavier in the thesis runs (D-239); the todo in `03_augmentation.tex` says Kaiming.
- Resolved (code wins): x^c = 0 is a definition, not Kessels' Remark 5.4 reconstruction (`FS/closed_loop.py`), although D-142 says otherwise; no `\todo` needed.
- $\rho$ is the $m_\Delta$ margin of III-C (D-229, `_m_diff_bound`). Section II states Drenth's spectral-radius condition (Thm. 6) in words or with another symbol.
- D-017 requires an LPV-structured augmentation; no later decision supersedes it, while the code uses an unrestricted MLP.
- `05_setup.tex` and `06_results.tex` still describe Karnopp friction, disabled noise, the Telica split and the affine OBC arm. Where a consumer section and `DOC/THESIS-RESULTS.md` disagree on what is reported, THESIS-RESULTS wins.
- refs.bib: `gyorok2026obc` and `forssell1999revisited` missing, `gyorok2025l4dc` a placeholder.
- Model names: plain $M_{\mathrm{U}}$, $M_{\mathrm{PS2}}$ and the like clash with the inertia $M(Y)$ and the PS2 horizon, so the ownership table uses the calligraphic candidates. Dirk decides; until then use them with a `\todo`.
- `DOC/PS2-METHOD.md` Sec. 2.1 says W^a Kaiming; the thesis code uses Xavier (D-239). The code wins.

## Writing a section

Write one section at a time, starting from the `WRITING GUIDE` at the top of its
file. Inspect sources in this order:

1. Read the research-plan anchor and identify wording that can be preserved, wording that needs a method update, and explicit departures. Reuse research-plan wording where it is accurate and concise; otherwise write new text, concise and to the point. The research plan is a source, not a constraint.
2. Read `docs/Thesis-documentation/Meeting-audit/candidate-thesis-impact.md` and the relevant theme in `thematic-thesis-audit.md`. Use Quinten's 18 September outline feedback to set priorities: realistic ASMPT motion, settling, controller transfer, justified choices, and final evidence rather than development history.
3. Read the newest relevant entries in `docs/decisions.md` to establish why the final choices were made. If an experimental choice is still open, keep it open rather than converting supervisor advice into a completed decision.
4. Audit the final code path to establish what was actually implemented. The main boundaries are:
   - `lpv_lfr_baseline/`: independent physics and LPV-LFR baseline;
   - `model_augmentation/`: reusable augmentation, rollout, encoder, closed-loop, and OBC framework;
   - `scripts/gantry/gantry_dynamic/`: gantry-specific configuration, data, model assembly, controller, training, diagnostics, OBC adapter, and evaluation;
   - `scripts/gantry/gantry_interconnect_dynamic.py`: thin run entry point and final run knobs.

   The audit informs claims; it is not content. Include only what the reader
   needs to follow the argument.
5. Trace empirical claims to final configurations, saved artefacts, figure data, and evaluation code. Plans and design notes are not evidence.
6. Read the relevant primary papers from the mandatory set below and add literature only where an external claim, method origin, theorem, or comparison needs support.

### Order of work

The reader's argument comes before every style rule below. Work in this
order, and do not start a step before Dirk approves the previous one:

1. **Section brief.** Answer six questions: what must the reader understand;
   which choices need a reason; which equations are load-bearing; which figure
   carries the argument; which claims need code, result, or literature
   verification; and which tempting claims are out of scope. Parameter-recovery
   tests, failed routes, burn-in screens, and similar development steps stay
   out of the main narrative unless they are required to establish a final
   claim.
2. **Claim list.** Per subsection, one governing question and the ordered
   claims that answer it, each with its source and its verified strength.
3. **Equations.** Every load-bearing construction displayed, checked against
   the preceding sections and the code.
4. **Prose,** one paragraph at a time.

Claim rules, applied from step 2 onward:

- **Claim strength.** State the weakest claim the evidence supports.
  Wording in `docs/decisions.md`, plans, or code comments is not evidence for
  a thesis claim; re-derive it or verify it at the source.
- **Method versus experiment.** A method section describes the method and its
  gantry realisation. The simulated truth, controller design and numerical
  values belong to Experiment design, measured numbers to Results (see Paper
  structure and ownership).
- **Checks must be able to fail.** A proposed verification that passes by
  construction is not a check.

Then run the four phases below. Dirk names the phase ("structure", "mark",
"rewrite", "check"); the default is structure. The AI supplies content,
structure, and checks; Dirk writes every sentence of the final text. An AI's
word choice and sentence structure carry its fingerprint and survive
paraphrasing, and rewrite prompts do not remove it (evidence basis below).

### Phase 1: Structure

AI role: co-author of content. Produce the brief, claim list, and equations of
the order of work above in chat. Write prose into the section's `.tex` file only
on request, one paragraph at a time, following the authorial-voice rules below
and the editor build invariant above, until Dirk approves the content.

- Write one claim per sentence, connected by the reasoning between them, so the
  section can be rewritten sentence by sentence without becoming a list of
  disconnected statements. Where a dependency must stay attached, voice rule 9
  takes precedence.
- Display every load-bearing construction as an equation; never compress a
  formula into prose to save space. Prose gives the purpose, assumption, or
  consequence of the equation.
- When a choice or assumption needs a reason that is not in `docs/decisions.md`,
  the code, or a cited source, write `because: ?` in its place. An invented
  reason is either wrong or generic, and reasons are what the examiners test.

Done when: Dirk approves the content.

### Phase 2: Mark

AI role: marker. Wrap each technical term in `\kw{}` and change nothing else,
style issues included. A technical term is a noun phrase that names a quantity,
model object, component, or method defined in this thesis or its cited
literature. Mark every occurrence; a multiword term is one unit. Math, symbols,
and `\cite`/`\ref`/`\eqref` are fixed by default and stay unmarked. `\kw{}`
prints teal in the draft build and plain with `\draftfalse`.

Example (Section II-B):

```latex
The measured \kw{payload position} $Y$ is both a \kw{state} and the \kw{scheduling variable},
making the \kw{baseline} a \kw{self-scheduled LPV model}.
```

Done when: deleting every `\kw{` and its closing brace gives back the approved
text exactly. Dirk then commits the marked file; that commit is the reference
for phase 4.

### Phase 3: Rewrite (Dirk only)

Dirk rewrites each paragraph in place around the `\kw{}` terms: read it, cover
it, write it from memory. Useful moves: split a sentence, start from a
different subject, use a verb instead of a noun, use "we" with an active verb,
move the reason. In this phase the AI does not edit the section files and does
not write or suggest sentences. Asked about a sentence, it answers with a
question or a fix of at most five words.

### Phase 4: Check

1. `git diff --word-diff <phase-2 commit> -- <file>`. Uncoloured
   words were kept from the draft; long uncoloured runs outside `\kw{}` terms
   show where the draft's wording survived.
2. AI role: checker. Compare the phase-2 commit with Dirk's version and report
   every instance as this table, nothing else:

   | # | type | problem | fix (max 5 words) |
   |-|-|-|-|
   | 3 | term | `\kw{scheduling variable}` missing | restore term |
   | 5 | meaning | "only if" became "if" | restore "only if" |

   Types: meaning (a claim changed), term (a `\kw{}` term, symbol, or reference
   changed or missing), grammar.

Done when: the table has no meaning or term rows, and the section meets its
completion criterion and page budget.

## Authorial voice and non-generic prose

The aim is clear, recognisably project-specific technical writing, not imitation
of a detector's idea of human prose. Automated authorship detectors are not a
validation tool: they can misclassify human text and perform poorly on short
passages. Review the argument and evidence instead.

Apply these rules to every AI draft (phase 1) and use them as the checklist for
Dirk's rewrite (phase 3):

1. Preserve Dirk's accepted wording and sentence rhythm where it is accurate.
   Edit the smallest necessary span; do not rewrite an entire paragraph merely
   to make it sound more polished.
2. Give every paragraph one concrete job: introduce a physical fact, make a
   modelling choice, define a construction, verify it, or state a consequence.
   Delete sentences that could be pasted unchanged into an unrelated paper.
3. Name the actual object and action. Prefer “closing the loop recovers
   \(M(Y)\ddot q=f_{\mathrm{net}}\)” over “this highlights the effectiveness of
   the proposed framework.”
4. Use plain technical verbs. Avoid inflated, generic vocabulary such as
   “delve,” “intricate,” “pivotal,” “crucial,” “meticulous,” “seamless,”
   “comprehensive,” “underscore,” “showcase,” “leverage,” “utilize,” “enhance,”
   “subsequently,” or “additionally” unless that word is genuinely the most
   precise choice.
5. Do not manufacture transitions. Use “however,” “therefore,” “because,” and
   “consequently” only when the stated logical relation is real. Avoid routine
   “first/next/finally” scaffolding and concluding summaries that repeat the
   paragraph.
6. Do not paraphrase an equation line by line. Prose around an equation must
   explain its purpose, assumption, interpretation, or consequence.
7. Keep technical terms stable rather than cycling through synonyms. Define a
   symbol or acronym once, then use it consistently.
8. Prefer direct claims with an explicit subject and a finite verb. Give a
   consequence its own clause instead of a trailing participle (“…, making the
   baseline …”), use the verb instead of a noun made from it (“we estimate,” not
   “the estimation of … is performed”), and write “is” rather than “serves as.”
   Avoid throat-clearing phrases such as “it is important to note,” “it should be
   emphasized,” “in order to,” and “it can be seen that.”
9. Vary sentence length only as the reasoning requires. Use a short sentence
   for a decisive result and a longer sentence when a dependency or qualification
   must remain attached. Do not force every paragraph into the same cadence.
10. State uncertainty and limitations where the evidence requires them. Do not
    replace a qualified project claim with smooth but stronger generic prose.
11. Check attribution sentence by sentence. Literature supplies general methods
    and definitions; gantry-specific algebra, implementation choices, and results
    remain explicitly this project's work.
12. In the final voice pass, flag repeated sentence openings, repeated transition
    words, trailing participle clauses, unnecessary three-part lists, “not only …
    but also” constructions, promotional adjectives, and claims with no concrete
    noun, equation, datum, or source. Revise for meaning, not merely for lexical
    variation.

These rules implement IEEE's guidance to use clear, simple sentences without
unnecessary words. They also address measured tendencies of LLM-written and
LLM-modified text: a small set of overused words, noun-heavy grammar with
trailing participle clauses, formulaic structure, and wording that is more
homogeneous across authors. Lexical diversity within one text is not a reliable
marker; studies disagree on its direction. The rules are quality controls, not
evidence about who authored a passage.

Evidence basis: [IEEE technical-English guidance](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/write-in-technical-english/),
[linguistic profiling across human and LLM text](https://aclanthology.org/2025.emnlp-main.1163/),
[changes in LLM-modified scientific prose](https://aclanthology.org/2026.lrec-1.142/),
[excess vocabulary in LLM-era abstracts](https://arxiv.org/abs/2406.07016),
[grammatical and rhetorical style of LLMs](https://doi.org/10.1073/pnas.2422455122),
[cues used by expert human detectors](https://arxiv.org/abs/2501.15654),
[model fingerprints that survive paraphrasing](https://arxiv.org/abs/2502.12150),
[detection of guideline-rewritten text](https://arxiv.org/abs/2607.27183) (detector vendor report),
and [documented limitations of AI-text classification](https://openai.com/index/new-ai-classifier-for-indicating-ai-written-text/).

## Validation story

The application claim is built in layers and must not be reduced to an unseen multisine phase:

1. Interpolation uses held-out phase realizations and operating points inside the training region.
2. Operational validation uses ASMPT-relevant, multisine-free motion profiles and evaluates both tracking and a fixed post-move settling interval.
3. Controller-transfer validation trains with one known controller and evaluates a credible changed controller as the digital-twin use case, while a plant-domain check tests whether feedback merely masks model error.
4. Operating-range extrapolation varies quantities such as stroke, acceleration, amplitude, or scheduling position beyond the represented training region. A controller change is reported separately as a closed-loop distribution shift, not automatically labelled plant extrapolation.

The exact motion profiles, controller variant, seed count, noise level, and primary metric remain open until fixed in `docs/decisions.md`. Current defaults do not settle those scientific choices. When selected, justify them from machine practice, data, or a controlled validation study.

Use physical RMS error as the default primary candidate for tracking and settling because it remains interpretable in metres and is well-defined on low-variation settling segments. Do not mix RMS, NRMS, and BFR as co-equal headline measures. If another primary metric is selected, record the reason before writing. Multisine phase seeds and neural-initialisation seeds answer different questions and must be reported separately.

## Derivation policy

Use derivations to expose the reasoning that is specific to this work, establish
correctness, or connect a cited method to the implemented model. Do not add
algebra merely to make a section appear more mathematical.

For each derivation:

1. State the assumptions and define every signal before manipulating it.
2. Show the load-bearing steps a reader needs to verify the construction.
3. Distinguish an adopted result from this project's own derivation or adaptation.
4. End by connecting the result to the implemented equations, signal routing, or numerical evaluation.
5. Put routine algebra, expanded coefficient matrices, and longer proofs in an appendix or technical supplement. A block partition whose entries a main-text claim relies on (a well-posedness or equivalence statement) is displayed in the main text as a block matrix, at a smaller font size if needed; only the expanded entries go to the appendix.
6. Cite the original source for an established method; do not reproduce its full proof unless the proof is changed or needed for a project-specific claim.
7. Check every own algebraic claim (a count, a determinant, a null direction, an iff condition) symbolically or against the code before stating it, in prose as in displays. State it only in a form that does not depend on an unstated convention (a magnitude when the sign depends on an unstated variable order).

The required depth is section-dependent. The LPV-LFR baseline needs a
project-specific derivation because its exact realization is central to the thesis.
The coordinate transformation, closed-loop residual rollout, RK4 boundary,
orthogonality construction, and loss function need only the short steps required
to remove ambiguity; a loop with two equivalent forms displays the form the
code computes, and the other is one equivalence sentence, never a display
whose lines exist only to be subtracted. Data-design and hyperparameter arguments are methodological
justifications, not derivations.

For every nontrivial design or hyperparameter choice, leave a short choice record
before writing prose:

1. State the selected value or construction.
2. Name the realistic alternatives.
3. Classify the basis as physics-derived, literature-supported, inherited, tuned on validation data, safety-constrained, or compute-constrained.
4. Explain which failure or capability the choice addresses.
5. State the consequence or limitation introduced by the choice.
6. Point to a sensitivity study, diagnostic, decision entry, or source where one exists.

Not every numerical setting needs a theoretical optimum. A transparent tuning or
computational rationale is preferable to a false physics justification. Choices that
change expressivity, identifiability, stability, data coverage, or comparison fairness
must be discussed in the paper; routine software settings can remain in the appendix or
configuration record.

## Mandatory methodological reading

Every thesis-writing session must read the relevant parts of the following local
papers directly. Project summaries, decision records, and code comments do not
replace these papers.

| Paper | Required role in the thesis |
|-|-|
| `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf` | Original encoder-based LFR augmentation framework and interconnection terminology. |
| `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf` | Extended first-principles augmentation formulation, identification procedure, initialization, and evaluation context. |
| `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf` | Model-informed encoder initialization, assumptions, and the boundary between physical and added-state initialization. |
| `literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf` | Affine and rational LPV-LFR identification, scheduling dependence, and well-posedness context. |
| `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf` | Projection-based regularization, protected subspace, negation problem, and limits of the penalty formulation. |
| `literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf` | Orthogonal-by-construction parametrization, uniqueness and recovery assumptions, and the distinction from regularization. |

For each thesis claim based on these papers:

1. Read the actual definition, proposition, assumption, or experiment in the PDF.
2. Record which part is adopted unchanged, which part is extended, and which part does not transfer to the gantry setting.
3. Do not count a later extension and its earlier paper as two independent precedents for the same claim.
4. Check the exact paper version before finalizing metadata or wording.
5. Keep project-specific choices, implementation evidence, and empirical findings attributed to this project rather than to the papers.

The roles of the sources are deliberately different:

| Source | Question it answers |
|-|-|
| Research plan | What was originally proposed and promised? |
| Meeting audit and dated supervisor feedback | What must not be forgotten, which priorities changed, and which historical alternatives must not be mistaken for the final method? |
| `docs/decisions.md` | Why was the final choice made? |
| Final pipeline code | What was actually implemented? |
| Run artefacts and evaluation code | What was actually observed? |
| Verified literature | Which external claims and method origins are supported? |

## Reference verification gate

`docs/references.md` is a candidate map, not evidence that an entry is correct.
Before enabling a citation or bibliography entry:

1. Locate the original paper, thesis, book, or official publication record.
2. Verify title, authors, year, venue, pages, and DOI or stable repository identifier.
3. Read the passage that supports the specific sentence being cited.
4. Record whether the thesis sentence is a direct source claim, an inference, or this project's own derivation.
5. Replace the placeholder in `refs.bib` only after all checks pass.

The bibliography calls in `main.tex` are enabled (user decision 2026-09-22). Entries
copied 1:1 from the research plan's `references.bib` are accepted as-is; a `% CHECK:`
comment above an entry records a known metadata fault. Other entries still pass this gate. A writer may add an entry for a paper whose passage it verified, with a `% CHECK:` comment for any metadata not yet checked. Own implementation details and measured results use
code and artefact provenance; they do not acquire external citations merely to
make the text appear sourced.

## Figure and table pipeline

One stem, three folders:

```
Code/Figures/obc_pareto.py   reads   Results/obc_pareto.csv
                             writes  Writing/figures/obc_pareto.pdf
```

Run `make figures` from `Thesis-writeup/` to regenerate, `make paper` to build
the PDF end to end. Scripts import `_fig_common.py` for the IEEE column widths
and body font size, so no figure ever needs rescaling in LaTeX.

## Draft switch

`util/format.tex` sets `\drafttrue`. Flip it to `\draftfalse` for the
submission build: every `\todo{}` and `\note{}` disappears, and every `\kw{}`
prints as plain text.
