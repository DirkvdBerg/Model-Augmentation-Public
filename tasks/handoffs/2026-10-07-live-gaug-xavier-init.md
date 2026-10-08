# Handoff: live (non-zero) g_aug initialisation and Xavier W^a, with and without PS2, after Jan Hoekstra's feedback
**From**: session of 2026-10-07 | **Branch**: Augmentation | **Effort suggested**: high (reading the paper and code, a design discussion, then a small code change and a run plan)

## 1. Task

This is a DISCUSSION session first. Read Jan Hoekstra's paper and public code yourself, then discuss with the user, in
text, the points in Section 8. Only after the user confirms a direction: plan, implement and locally pre-flight the
change, so that the user can run on the server (A) plain S-DP and (B) PS2, both with a non-zero g_aug initialisation and a
Xavier-initialised encoder map W^a.

Your own reading of the paper decides what the paper says. Section 4 gives the previous sessions' readings so that you know
where to look and what to check. They are leads, not facts: confirm, correct or reject each one against the PDF and the code,
and say which you did.

User's words (2026-10-07), after a meeting with Jan:
> "jan hoekstra says that we shouldnt set g_aug to zero, and that we should compare the method (both we dont set it g_aug
> to zero) and maybe now we just have a bad local minimum. so maybe the issue can be fixed by not setting g_aug to zero.
> also what i want to do is set the random weights initialziation to xavier instead of kaiming."
> "so the main thing i want to do is run without the g_aug zero init. and add xavier weights and then see if our approach
> still fixes this. but i also want to check what jan's paper says, because i thought his example code did set it to zero."

Jan's other feedback (the user's paraphrase): the method as a whole can be defended, because it is an initialisation
followed by normal training, which still searches for the global optimum. He is not sure the PS2 screen can be defended:
"at least 10 % of the residual explained" (unexplained fraction <= 0.90) is a heuristic, and a large enough network could
fit the noise.

## 2. Out of scope

- No code edits before the user confirms the direction. No server submission by you: the user moves files and submits.
  Show the exact runs (id, arm, seed, init, PS2 on/off) as a table and get confirmation before giving any submit command.
- No new runs with the OLD initialisation. Zero g_aug plus Kaiming is already covered: thesis core runs 41 to 46 (jobs
  86893, 86894, 86909, 86918, 86945, 86960) and PS2 tier `ps2` (ids 111 to 116). Never resubmit ids 121 to 126 (`ps2ctl`,
  retired, D-236 correction).
- Do not change the PS2 algorithm or its thresholds in this task. The screen question (Section 8, point 5) is for
  discussion only, unless the user decides otherwise.
- Band, pole or absorber-informed initialisations stay excluded (memory feedback_no_pole_init_workaround). Only Jan's
  generic, system-free initialisation choices are in scope.
- `kamtin-fp-model/` is read only. No MATLAB, no data generation.

## 3. Where things stand

Branch `Augmentation`; the tree is dirty in many directories (thesis writing, scripts); nothing for this task has been
started. Server: PS2 U seeds 2 and 3 (job array 87702, ids 113 and 115) may still be running; ask the user for `squeue`
before saying what ran. The slide brief for Jan's meeting is `tasks/handoffs/2026-10-07-ps2-motivation-slides.md`
(background on the PS2 argument, not this task).

## 4. Context to read, and the leads to check

**Read these yourself (primary sources):**
- Jan's paper: `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`, "Learning-based augmentation of
  first-principle models: A linear fractional representation-based approach" (Hoekstra, Gyorok, Toth, Schoukens; cited in
  the repo as arXiv:2602.17297). Read Sec. 3.3 (learning-function structure), Sec. 5.4 in full (initialisation:
  5.4.1 baseline behaviour, 5.4.2 encoder, 5.4.3 learning functions), and Sec. 6.3 (the dynamic-parallel experiments).
- Jan's encoder-initialisation paper: `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf`
  (the source of the analytical W^b map).
- Jan's public code: `Model-Augmentation-Public-main/` (repo root; a ZIP download, no git history). Start at
  `scripts/journal_model_augmentation/msd_ndof_interconnect_fit.py` (the paper's S-DP script),
  `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py`, and `model_augmentation/utils/torch_nets.py`
  (`zero_init_linear_mapping`, `zero_init_feed_forward_nn`, `zero_init_resnet`).

**Leads from earlier sessions (check each one against the sources above):**
1. The paper permits live added states (Eq. 15 ResNet with a linear part; Eq. 29; Sec. 5.4.3 "phi_aug(z_a,k) = 0 + W_a z_a,k";
   p. 10: matrices not needed for the baseline behaviour initialised randomly, m ~ U(-1,1)), but does not state which rows
   of W_a stay random, and the S-DP experiments (Sec. 6.3, p. 11) use feedforward networks with an unstated init (D-233).
2. The public S-DP script zero-initialises the whole learning function with `zero_init_linear_mapping` (a linear layer,
   weight and bias zero), so the added states are dead at the start; no public parallel-augmentation script uses a random
   linear part; that script uses deepSI's default encoder, not the W^b map (D-237).
3. Eq. 31 prescribes Xavier for the augmented encoder; our pipeline uses Kaiming (logged deviation, D-233).
   Previous readings in `docs/decisions.md` D-233 and D-237. Read them, but the PDF and the code take precedence.

**Our pipeline (verify in the code):**
- The augmentation network: `Static_ANN_Block(..., net=zero_init_feed_forward_nn)` in
  `scripts/gantry/gantry_dynamic/model.py:138-144`. One tanh network with random hidden layers and a zero output layer for
  ALL rows, so f_aug = 0 and g_aug = 0 at the start. Just below it in `model.py` there is a ReZero-style gate (D-130): check
  whether the thesis config activates it.
- W^a: `nn.init.kaiming_uniform_` in `model_augmentation/fit_systems/pre_encoder.py:408-409`. W^b: the analytical map.
- Thesis run config: `scripts/gantry/gantry_interconnect_dynamic.py` (thesis block around line 350); `THESIS_*` env
  variables; runner tiers in `scripts/gantry/thesis-results/runs.tsv` and `submit.sh`.
- PS2: `model_augmentation/fit_systems/ps2_opening.py`; method write-up `Thesis-writeup/Documentation/PS2-METHOD.md`.

**Measured facts you may use:**
- Gradient of the normal training loss at the current init (2026-10-07; U arm, seed 1; 4 x 64 windows, nf 400; RMS per
  parameter group): encoder x_a rows 0, g_aug output rows 0, shared hidden layers 0; f_aug output rows 1.9e-5, encoder x_b
  rows 1.4e-7, physical parameters 7.1e-9.
- Random added-state self-maps, NO training (AE1, `scripts/gantry/absorber-learning-diagnosis/DECISIONS.md` line 178):
  - literal U(-1,1): 22 % unstable; an oscillatory pair in 95 to 380 Hz in 3.7 % of draws;
  - torch-default scale: 0 % unstable, fast poles (median |z| 0.24).

  Training from a live g_aug has NOT been tested on the gantry.
- The diagnosis of plain S-DP (same folder, `MECHANISM.md` and `CAUSE-AND-DIRECTION.md`):
  - the network can represent the absorber (K1);
  - the static path learns first and then blocks the mode (M1c);
  - X1: a band pole with re-adapting drive and readout beats the control within 100 updates ("it is the PATH").

  This is consistent with Jan's "bad local minimum" reading.
- PS2 interim results (zero g_aug + Kaiming), validation free-run error:
  - U seed 1: 1.597e-6 m vs 1.027e-5 m;
  - OBC seed 3: 1.808e-6 m vs 1.035e-5 m.

## 5. Assumed but not verified

- That a live g_aug changes the outcome of plain S-DP on the gantry (Jan's hypothesis).
- That a live g_aug matters little for PS2. Reasoning: phase 2 overwrites g_aug by regression, so its init acts only during
  phase 1 (350 updates, x_a not yet read out). Xavier W^a does change phase 1's starting x_a.
- The defence of the screen against noise fitting (Section 8, point 5).

## 6. Tried and failed

- Optimiser-only changes for plain S-DP (U1, U4 learning rates; Levenberg-Marquardt) -> the added poles stay real, 0 %
  complex -> the problem is the gradient direction, not the step -> `DECISIONS.md`, `MECHANISM.md`.
- Static path switched off (N1, O1) -> x_a itself learns a delayed static map -> whatever learns first from zero takes the
  memoryless part -> `MECHANISM.md`, "Update after the phase-1 pre-flight".
- Equation-error fit of g_aug on a random W^a without a target (E1, E2) -> real lags only -> `CAUSE-AND-DIRECTION.md`.
- Not tried: plain S-DP trained from a live g_aug. That is the new experiment.

## 7. Achieved

PS2 is in the production pipeline (`THESIS_PS2=1`) and tested (`scripts/gantry/thesis-results/test_ps2_pipeline.py`,
ALL PASS). Evaluation tools: `inspect_checkpoint.py` (clamp test, quiet records, added-state poles as diagnosis) and
`band_compare.py` (band errors), in `scripts/gantry/thesis-results/`.

## 8. Points to discuss with the user (in this order)

1. **What Jan's paper and code actually say about initialising g_aug and W^a**, with section, equation and page, after
   your own reading. Answer the user's question directly: did the example code set g_aug to zero, and does the paper?
2. **What "g_aug not zero" should mean concretely.** Candidate directions to evaluate against the paper:
   - (a) random x_a output rows of the last layer, with the f_aug rows kept zero. The model still equals the baseline at
     the start (x_a does not reach the output), so the baseline-behaviour condition holds. The scale is a choice (torch
     default, Xavier, or the paper's U(-1,1), which AE1 found 22 % unstable as a self-map).
   - (b) the paper's Sec. 5.4.3 form: a random linear part plus a zero nonlinear part (ResNet). The current network has no
     linear branch.
   - (c) all output rows random. This breaks baseline-at-start; probably not what Jan means.

   Recommend one, and explain why it is the one Jan's paper describes.
3. **The gradient consequence.** With f_aug zero at the start, the loss gradient to x_a, g_aug and the encoder's x_a rows
   is still exactly zero at update 1, because the readout is zero. A live g_aug changes what x_a does once the readout
   moves (live dynamics, a first-order path), not the first step. Explain whether this still tests Jan's local-minimum
   hypothesis, and what to log in the pre-flight to see it (the gradient per group at update 0 and after a few updates;
   the added-state poles at the start).
4. **The run plan.** Arms A (plain, new init) and B (PS2, new init), U and OBC, seeds 1 to 3, thesis settings otherwise,
   compared with the existing references at equal arm and seed. Ask whether the user wants all 12 runs or a subset first.
   Per the run-discipline rule: a run-table row in `docs/gantry-augmentation-problem-log.md` Section 12 and a new D entry
   in `docs/decisions.md`, both before implementing or launching.
5. **Jan's concern about the 0.90 screen.**
   - A proposed defence: the screen is computed on HELD-OUT training records (TR-S5, Y3, P4, T4), whose noise is
     independent of the records the head is fitted on. A head that fits noise cannot lower the held-out error; it would
     show an unexplained fraction of 1 or more there.
   - The value 0.90 itself remains a heuristic. A data-derived alternative: a significance test of the held-out
     improvement over the zero predictor (for example a bootstrap CI over windows).

   Discuss; do not implement unless asked.

## 9. Next action

Read the sources in Section 4, then answer points 1 to 5 of Section 8 in text, with one recommendation each, and wait for
the user's decision. After confirmation, implement:
- a config flag for the g_aug init and a flag for the W^a init (`xavier` / `kaiming`), with defaults that reproduce the
  current pipeline bit-identically;
- both exposed as thesis env variables, plus a new runner tier;
- marked per the `model_augmentation/` tracking rules if framework files change.

Then run a short local pre-flight.

## 10. Acceptance criterion

- **Discussion:** the user agrees on the init option, the scale, the arms and the run ids.
- **Implementation:**
  - with the new flags off, training losses on a fixed batch are identical to the current pipeline;
  - with them on, the model output at the start equals the baseline to machine precision;
  - the g_aug rows (or the linear part) are non-zero, and W^a follows Xavier's bound sqrt(6 / (fan_in + fan_out)).
- **Experiment (after the server runs):** read the validation free-run error and the 230 to 297 Hz band error against the
  existing references, with the clamp test and the band analysis. Compare plain-new-init against PS2-new-init at equal arm
  and seed.
