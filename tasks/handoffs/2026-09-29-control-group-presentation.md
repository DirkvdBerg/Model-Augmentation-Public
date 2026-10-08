# Handoff: outline, figures and Claude Design brief for the 15-minute control-group talk
**From**: session of 2026-09-29 | **Branch**: Augmentation | **Effort suggested**: high (design judgment and honest claims matter more than code volume; the code is small)

## 1. Task
Produce everything Claude Design needs to build Dirk's 15-minute thesis talk for the TU/e Control
Systems group (plus 5 minutes of questions), so that Dirk can upload one folder plus the TU/e
template to Claude Design and get proper slides. Concretely: (a) look up what Claude Design can and
cannot do, and decide per slide element who makes it (us or Claude Design); (b) write the
presentation outline and the Claude Design brief in
`scripts/gantry/meeting/control-group-presentation/documentation/claude-design-brief.md`;
(c) make every figure that Claude Design should not make, from the existing run data, into
`control-group-presentation/figures/`, with the code in `control-group-presentation/code/`. The
figures and the outline ARE the deliverable; this is not a planning exercise.

The user's words (verbatim, 2026-09-29):
> "The goal of this session is to outline a presentation for the control group about my thesis.
> I want to make it with claude design, but then we will need to have all the figures etc. the
> presentation should be 15 minuts and we have 5 minutes for questions. I dont want to go in to
> specific, like go in depth of the formula. i think its important to show my model and then with
> and without msd i have figures for that in thesis-writeup."
> "you might also know better whether claude design is capable at making certain figures or we
> should ensure we create all the figures ourselves first"
> "let it look up claude design capabilties, so its knows what it can delegate / what claude
> design is better at."

Supervisor advice (Quinten, Dutch, verbatim): "Geen formules tenzij je echt een formule hebt en die
helemaal uitlegt." (No formulas unless you really have one and explain it completely.)

Dirk's own draft outline (verbatim; this is the confirmed backbone):
> Introduce what ASMPT does / Intro grey box modeling: What is it, and benefit over other methods,
> Why useful for ASMPT / State first principles model: Not going into the formulas (limited to
> equations of motion) / Model augmentation: State the general concept and pictures to illustrate /
> Interpretability: Orthogonality

The results to use, pasted by Dirk (text originally from an earlier session's advice, adopted by
Dirk, who then downloaded exactly these four runs):
> - 84032 (no projection) against 84033 (OBC): joint estimation from a 10 % detuned start. Both
>   went from 23.0 to about 5.9 µm, so OBC is as accurate as no projection, and both parameter
>   errors improved.
> - 81262 (8 states) against 81265 (2 states): the plateau and then the second drop, and 2 states
>   ending within 3 % of 8.
> - Improvement over the baseline: about 3.5 to 3.9x lower error than the physics baseline in the
>   same closed-loop validation.
> - Caveat for the slide: these runs used the earlier configuration: 8 states, 24 × 3, float32, the
>   old band and no friction.

Three of these sentences are imprecise; section 4 gives the corrected facts and they override the
paste.

## 2. Out of scope
- Building the slides themselves (no .pptx, no HTML deck, no Artifact): Claude Design does that.
- New training runs, re-simulations, new datasets, MATLAB: the user chose the four existing runs.
  All figure data is already cached (section 4); nothing needs recomputing.
- The final friction/tanh runs and anything in `tasks/todo.md`: a different thread.
- Black-box comparison results: not among the four runs. The grey-box "benefit over other methods"
  slide is argued conceptually unless Dirk supplies a result.
- Telica real data (out of the thesis main results, user decision 2026-09-24).
- Literature search: this is product documentation, not literature, so the deep-research skill
  does not apply.
- Editing anything under `Thesis-writeup/` (the Writing README forbids build output there),
  `kamtin-fp-model/`, or the earlier meeting folders. Read them; copy and adapt code into
  `control-group-presentation/code/`.

## 3. Where things stand
- Branch `Augmentation`, last commit `235f0af`. Tree dirty in many unrelated directories; none of
  this task's files are touched yet.
- `scripts/gantry/meeting/control-group-presentation/`: `code/`, `documentation/`, `figures/` are
  empty. `server/` (69 MB, downloaded by Dirk) holds `81262/ 81265/ 84032/ 84033/` (each with
  `config.json`, `gantry_<run>`, `gantry_ckpt_<run>.npz/.pt`, `gantry_results_<run>.npz`,
  `gantry_state_recovery_<run>.npz`, `plots/`) plus `obc_arm84032.out`, `obc_arm84033.out`,
  `gantry_interconnect_dynamic81262.out`, `gantry_interconnect_dynamic81265.out`.
- No runs in flight.

## 4. Established and verified
Data and provenance:
- `server/` files are byte-identical (md5) to the copies the earlier figure code used
  (`meeting-07-09-2026/server/augmentation_ma50_b140-230_a6_z03_linear_map/<run>/` and
  `meeting-18-09-2026/server/useable/`). Exception: `server/obc_arm84033.out` is the old log plus
  45 extra lines of post-training evaluation (the old copy is an exact prefix); training history
  identical.
- `server/<run>/gantry_<run>` (no extension) is the pickled full model. For 84032/84033 it holds the
  same histories as the `SSE_Interconnect_Composed_<run>_best.pth` files the old scripts read:
  `Loss_val`, `Loss_train`, `Probe_combo_err`, `epoch_id`, 16 points each, identical (entry 0 of
  `Loss_train` is NaN in both). 84033's model also carries the projection buffers
  `obc_correction.{E, vbar, theta_frozen}`. So `best.pth` is not needed.
- Three caches already hold every number and trace the slides need (read them in place):
  - `meeting-07-09-2026/code/figure_data_<run>.npz` (81262, 81265): closed-loop re-simulations on
    V1 to V4 per arm: `y_data_k`, `full_y_hat_k`, `baseline_y_hat_k`, `zero_add_y_hat_k`
    (k = 0..3), plus `full_agg`, `baseline_agg`, `zero_add_agg`, `*_per_channel`, `fs`, `k0`.
    Record order: `VAL_RECORDS` in `meeting-07-09-2026/code/common.py`. Built by `collect.py` (13 min
    per run, do not rerun).
  - `meeting-18-09-2026/figures/OBC/obc_data.json`: per-validation fit (`val`, metres), combined
    parameter error (`combo_err`, %) and the ten per-parameter errors for 84032/84033 (plus 84037,
    84034, not needed). Built by `extract_obc_data.py` from the `.out` logs and the checkpoints.
  - `meeting-21-09-2026/figures/orthogonality_data.json`: overlap rho per arm, train and val.
- Do not use the time series in `gantry_results_<run>.npz` (`y_hat_enc`, `y_hat_test`) for any
  trace: they are OPEN-LOOP free runs of the augmented model, and they diverge (81262 `nrms_enc`
  about 140 against a baseline `baseline_nrms` of 0.0096, `thesis-results-plan/outputs/npzkeys.log`;
  8403x "NRMS 43 to 1607 against a baseline of 0.03", `extract_obc_data.py` docstring). Their
  `loss_val`/`loss_train` histories ARE usable (fig0 reads them).

Run configurations (`server/<run>/config.json`):
| run | learned states | ANN | encoder na_nb | epochs | joint estimation | projection |
|-|-|-|-|-|-|-|
| 81262 | 8 | 24×3, 10871 params | 29 | 350 | no | no |
| 81265 | 2 | 16×2, 4019 params | 17 | 350 | no | no |
| 84032 | 8 | 24×3 | 29 | 150 | yes | no |
| 84033 | 8 | 24×3 | 29 | 150 | yes | OBC, tangent (10 columns) |
All four: dataset `augmentation_ma50_b140-230_a6_z03`, closed loop, float32, seed 42, lr 1e-5,
`SNR: None` (noiseless simulated data), no friction, single seed per arm.

Numbers (all from `scripts/gantry/thesis-results-plan/outputs/tab.log`, which reads the caches):
- Augmented vs physics baseline, closed loop, V1 to V4 aggregate: 81262 5.77 µm vs 20.62 µm
  (3.57x); 81265 5.95 µm vs 20.62 µm (3.47x). With the 8 added states' corrections zeroed, 81262
  gives 11.67 µm.
- Learned block size: 81265 ends 3.1 % above 81262 (5.9459 vs 5.7679 µm) with 37 % of the
  parameters. Both validation curves start at the baseline (ANN output layer is zero at init), drop
  fast to about 9.8 µm, plateau (8 states until about epoch 20, 2 states until about epoch 50), then
  drop a second time (`meeting-07-09-2026/figures/fig0_block_size.png`).
- OBC vs no projection (joint estimation, every parameter started 10 % off, mixed signs): fit
  22.97 µm at epoch 0 (identical in all arms, i.e. the detuned baseline in the loop) to 5.86 µm (no
  projection) and 5.89 µm (OBC), a 0.5 % difference. Combined parameter error = RMS over the ten
  parameter combinations in % (reproduces 7.20 from the ten final values in tab.log): 9.49 -> 7.20
  (no projection, minimum 7.15) and 9.49 -> 7.69 (OBC, minimum 6.98 at epoch 70, then rising).
  Per parameter, several get worse: no projection cg1 +17 %; OBC kb_sum +15.5 %, d +13.9 %.
- Overlap rho = ||A A⁺ F|| / ||F||, the fraction of the learned correction lying in the span of the
  ten baseline parameter directions (0 = orthogonal, 1 = fully inside; definition in
  `meeting-21-09-2026/code/fig_orthogonality.py` docstring). No projection: 0.865 train, 0.870 val.
  OBC (84033): 8.3e-8 train, 0.334 val. The projection is built on the training points, so it is
  exact there and partial on new trajectories.
- Two different "baselines": 20.62 µm is the nominal baseline (8126x); 22.97 µm is the 10 %
  detuned start (8403x). The pasted "3.5 to 3.9x" mixes them (3.9x = 22.97/5.86).
- OBC = "orthogonal-by-construction", Gyorok 2026 (`docs/decisions.md:1282`). The audience does not
  know the acronym: say "orthogonal projection" on slides.

Model and figures:
- Plant framing (user decision 2026-09-24): X and Y are free integrators, marginally stable, with
  zero stiffness and damping to the ground; only Θ has stiffness (kb_sum). Inertia depends on the
  payload position Y (LPV). Only sums like kb1+kb2, cb1+cb2 are identifiable.
- "With and without MSD" figure: `Thesis-writeup/Writing/figures/gantry_system_with_absorber.tex`
  (TikZ). `\gantryabsorber` 0 draws the baseline only, 1 adds the grey absorber; `\gantrydelta`
  toggles the δa arrow. Both are `\providecommand`, so a standalone wrapper can set them before
  `\input`. Needs tikz with `arrows.meta, calc, decorations.pathmorphing`.
- Detailed augmentation block scheme exists (`docs/writeup/jan-blockscheme-v4.tex`, v5 newer) but is
  far too detailed for this audience. Topology to simplify from: an ANN with its own hidden states
  takes the physical state, its own state and the input; its output corrects the physical state
  update (routed to Θ, X, Y) and updates its own states; the output map stays the physics one
  (h_aug = 0); an encoder estimates the initial state from past inputs and outputs.
- Validation is closed-loop simulation with the controller in the loop,
  u = u_data + Cfb(y_data - y_model) (`docs/writeup/closed-loop-form-v4.tex` explains why).
- The existing meeting figures (`meeting-07-09`, `-18-09`, `-21-09`) are analysis plots (claim text
  inside, dense titles, many panels, metres in scientific notation): reuse their data and logic,
  not the images.
- Result slides follow the supervisor rule: criterion stated first, then result, then verdict and
  why (memory `feedback_thesis_argue_results`). Parameter tables or plots show all ten parameters,
  never a subset.

## 5. Assumed but not verified
- **Claude Design capabilities: all of them.** This session did not look anything up. What it
  accepts (PNG, SVG, PDF, CSV), whether it applies a .pptx/.potx template, whether it renders charts
  faithfully from supplied numbers, whether it renders an equation, what it exports, and whether
  speaker notes survive. Settled by step 1 of section 9.
- Absorber parameters for these runs: the dataset name `ma50_b140-230_a6_z03` encodes them, but
  the meaning of each token was not checked. `meeting-07-09-2026/code/collect.py` states the
  memory value MA_FRAC = 0.10 does NOT match this ma50 dataset. Read the generator config for
  `data/gantry/matlab/trajectory/augmentation_ma50_b140-230_a6_z03/` before putting any absorber
  number on a slide, or show none.
- Single seed per arm: the 0.5 % (OBC vs no projection) and 3.1 % (2 vs 8 states) gaps have no
  seed-spread reference (84034 is a determinism repeat of 84037, not a second seed). "As accurate"
  means "within 0.5 % on one seed"; say so if asked.
- `figure_data_81262.npz` has the same keys as 81265 (listed for 81265 only; tab.log reads the
  `*_agg` keys of both).
- Talk language English; TU/e template format and colours; available PDF-to-PNG/SVG converters
  (pdftocairo, dvisvgm, inkscape) on this machine.

## 6. Tried and failed
This session's own errors, so they are not inherited:
- Proposed building the figures in this session -> the user corrected the split (this session
  writes the prompt only) -> I read "this session should find relevant figures" as "make them
  here".
- Told the user `SSE_Interconnect_Composed_<run>_best.pth` was needed -> wrong -> I read the old
  scripts' input paths instead of checking what `gantry_<run>` contains -> section 4.
- Reported `Loss_train` as differing between the two files -> artefact -> NaN at entry 0 never
  compares equal -> section 4.
- Repeated "3.5 to 3.9x" as one range and "8 states, 24×3" as the caveat for all runs -> the range
  mixes two baselines, and 81265 is 2 states, 16×2, na_nb 17 -> "2 states suffice" is confounded
  with a smaller network and a shorter encoder window: call it "a smaller learned block".
- Proposed "projection costs no accuracy, and parameters still improve" as a slide message -> the
  second half is misleading: OBC's combined error ends worse than no projection (7.69 vs 7.20 %)
  and some single parameters get worse. What the data supports: equal fit, overlap removed on
  training data, partial on validation.
- Claimed Claude Design "cannot see the data, so any curve it draws is made up" and "can draw
  simple concept visuals" -> both unverified -> if it renders charts exactly from supplied numbers,
  simple bar charts can be delegated; the lookup decides.
- My 11-slide outline (section 8) adds result and conclusion slides that Dirk's outline does not
  contain: an interpretation, not his decision.

## 7. Achieved
Nothing built yet. Verified inventory and numbers: section 4.

## 8. The open question
Where do the results go, given that Dirk's outline ends at "Interpretability: Orthogonality" and
has no results section, while he downloaded four result runs? Default, to confirm with Dirk: put each
result right after the concept it supports, and end with one conclusion slide. Proposed 11 slides
(my reading; timings sum to 15:00; open points marked (?)):

1. Title (0:15)
2. ASMPT and the dual-gantry: what the machine does, why nm-level models matter (1:15). Photo and
   facts from Dirk (?); never invent company facts.
3. Grey-box modelling: physics structure plus data; versus white box (misses dynamics) and black
   box (no interpretable parameters, needs more data); why ASMPT cares (1:30). Concept visual.
4. First-principles model: coordinates X, Θ, Y; inertia depends on Y; X and Y free integrators, so
   a controller is always in the loop; the one equation M(Y)q̈ + Cq̇ + Kq = F explained term by
   term (1:45). Schematic without absorber.
5. What the physics model misses: the simulated truth has a hidden mass-spring-damper absorber on
   the payload (1:15). Schematic with absorber; absorber numbers only if verified (section 5).
6. Model augmentation: a learned dynamic block with its own states corrects the physics model's
   state update; encoder gives the initial state (1:45). Simplified block scheme.
7. Result: augmented vs physics baseline in closed loop, 20.6 -> 5.8 µm on held-out records (1:30).
   Error trace plus the aggregate numbers.
8. Result: learned block size, 2 states and a smaller network come within 3.1 % of 8 states, after
   a plateau and a second drop (1:15). Validation curves.
9. Interpretability: the learned block can mimic a change of physical parameters; projecting its
   output orthogonal to the parameter directions prevents that (2:00). Projection sketch, no
   formula.
10. Result: projection costs 0.5 % fit; overlap 0.87 -> 1e-7 on training data, 0.33 on validation;
    parameter error in both arms (1:30). Fit plus overlap bars.
11. Conclusion and next steps; caveat line: noiseless simulation, earlier configuration, no
    friction, one seed (1:00).
Backup slides for questions (?): open-loop drift (the likely control-group question: the augmented
model is validated in closed loop and drifts in open loop, because small force errors integrate
through the X and Y free integrators); all ten parameters for 84032/84033; run configurations.

## 9. Next action
Step 1, before writing anything: look up Claude Design's capabilities inline with WebSearch and
WebFetch, official Anthropic sources first (anthropic.com, support.claude.com, the Claude docs),
recording each URL and date. Answer: accepted input formats (images, vector, PDF, data files), how a
template (.pptx/.potx or brand kit) is applied, whether charts are rendered exactly from supplied
numbers, whether diagrams come from a text spec, equation rendering, export formats, speaker notes,
upload limits, and Anthropic's own prompting advice for it. A capability no official source
confirms counts as unavailable when deciding what to delegate. Then, in order: write the outline
and a delegation table (element | made by | reason) into `claude-design-brief.md` and show the
outline in chat; build the figures; finish the brief with the figure filenames. Keep going through
all steps; Dirk interrupts if the outline needs changing.

Figures, default maker "us" unless the lookup shows Claude Design does it exactly:
| id | content | source |
|-|-|-|
| gantry-baseline | schematic, `\gantryabsorber` 0 | TikZ wrapper in `code/` |
| gantry-absorber | schematic, `\gantryabsorber` 1 (δa arrow: your call) | same |
| augmentation-scheme | simplified topology from section 4 | new TikZ or matplotlib |
| closedloop-trace | one V record, one channel, baseline vs augmented error in µm, with a zoom; crop the encoder start transient at the left edge | `figure_data_81262.npz`; pick the record whose baseline/full ratio is closest to the 3.57x aggregate |
| blocksize-curves | validation sim-RMS in µm vs epoch, 81262 vs 81265, baseline line | `server/8126x/gantry_results_8126x.npz` (`loss_val`, `epoch_id`) |
| projection-sketch | plane of parameter directions, learned correction, its removed in-plane part | new; geometry must be exact |
| obc-fit-params | fit in µm and combined parameter error in % vs epoch, two arms only | `obc_data.json`, or `server/` logs plus `gantry_<run>` histories |
| overlap-bars | rho vs parameter directions (`J`, `applied`), no projection vs OBC, train and validation | `orthogonality_data.json` keys `noproj`, `tangent` |
| equation | M(Y)q̈ + Cq̇ + Kq = F, only if Claude Design cannot render math | matplotlib mathtext or LaTeX |
| grey-box concept | white / grey / black box | Claude Design by default |
Backup figure, if the backup slide is kept: all ten parameters, two arms (the style of
`meeting-18-09-2026/figures/OBC/obc_fig2_per_parameter.png`).

Figure style: one message per figure, no title inside the figure (the slide title carries it),
µm instead of metres in scientific notation, labels readable at slide size, the same colour per
role in every figure (physics baseline, augmented, projected). Take colours from the TU/e template
if Dirk puts it in `control-group-presentation/documentation/`; otherwise one palette constant,
easy to swap. Export in the format the lookup shows Claude Design handles best.

The brief itself is written FOR Claude Design, which knows nothing about the repo. Per slide: title
as a full-sentence message, at most 3 short bullets, figure filename or an exact drawing spec, the
spoken message as speaker notes, and the time. Global rules: 16:9, TU/e template, 15 minutes, no
formulas except the one equation of motion, no em-dashes, no acronyms the audience does not know
(OBC, LPV-LFR, SUBNET: say it in words). Example of the per-slide format:

> **Slide 7 (1:30). The learned block cuts the closed-loop error from 20.6 to 5.8 µm**
> Figure: `closedloop-trace.png`, full width.
> Bullets: held-out records, controller in the loop | criterion: clearly below the physics model
> Notes: "We first ask for a clear improvement over the physics model on records not used in training..."

## 10. Acceptance criterion
- `claude-design-brief.md` can be pasted into Claude Design unchanged: every slide has a message
  title, a figure file that exists in `figures/` or an exact drawing spec, speaker notes and a time;
  the times sum to 14:00 to 15:00; the delegation table cites the lookup's sources.
- Every number on a slide reproduces section 4 (22.97 -> 5.86 / 5.89 µm; 9.49 -> 7.20 / 7.69 %;
  20.62 -> 5.77 µm and 5.95 µm, +3.1 %; overlap 0.865 / 0.870 vs 8.3e-8 / 0.334) and names which
  baseline it refers to.
- Each result slide states criterion, result and verdict; the caveat line (noiseless simulation,
  earlier configuration, no friction, one seed) appears on the result slides or the conclusion.
- Dirk's five outline items are all present and in his order.

## 11. Read these first
1. `scripts/gantry/thesis-results-plan/outputs/tab.log`: every number, as read from the caches.
2. `scripts/gantry/meeting/meeting-07-09-2026/code/common.py` and `fig0_learning.py`: cache layout,
   record order, how the curves were drawn.
3. `scripts/gantry/meeting/meeting-18-09-2026/figures/OBC/extract_obc_data.py` and `plot_obc.py`:
   OBC data structure and parameter naming.
4. `scripts/gantry/meeting/meeting-21-09-2026/code/fig_orthogonality.py` (docstring): what rho
   measures and why OBC is measured on the applied field.
5. `Thesis-writeup/Writing/sections/02_system_baseline.tex`: the model as the thesis states it, for
   the equation-of-motion slide.

## 12. Do not
- Draw traces from `gantry_results_*.npz` `y_hat_*` (open loop, diverged; section 4).
- Present "2 states suffice" without the smaller network and shorter encoder window; present
  "projection gives better parameters"; mix the 20.62 and 22.97 µm baselines; show a subset of the
  ten parameters.
- Rerun `collect.py`, `extract_obc_data.py` in its old location, or the orthogonality computation
  (caches exist; heavy or unnecessary).
- Write into `Thesis-writeup/`, earlier meeting folders, or `kamtin-fp-model/`.
- Invent ASMPT facts or images.

## 13. Operational
- Env `GraduationProject`; figure scripts read caches and run in seconds, so foreground is fine.
- Loading `gantry_<run>`: `torch.load(path, map_location='cpu', weights_only=False)` with
  `scripts/gantry` and the repo root on `sys.path` (otherwise `ModuleNotFoundError: gantry_dynamic`);
  histories are attributes (`vars(obj)['Loss_val']`), projection buffers in `obj.hfn`.
- `Loss_val` and `val` are closed-loop sim-RMS in metres (fig0 docstring); convert to µm.
- TikZ: compile a standalone wrapper in `control-group-presentation/code/` that sets the toggles and
  `\input`s the thesis file by path; output to `figures/`.
- Write only under `scripts/gantry/meeting/control-group-presentation/`.

## 14. Delegation
None. The Claude Design lookup is a handful of inline WebSearch/WebFetch calls; the figures are
small scripts over cached data. No Explore agent, no Workflow.
