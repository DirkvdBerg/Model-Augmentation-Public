# Handoff: literature search on encoders for augmentation training, and how encoders survive a de-weighted early window
**From**: session of 2026-08-20 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Run a deep-research literature search (the `deep-research` skill is MANDATORY per D-121, step 0
FRAME first, one subagent per seed) on how the state encoder of an encoder-based identification
method is, or should be, trained when it serves an AUGMENTED model, and what happens to it when
the training loss de-weights the early window samples. The motivating conflict, measured on this
rig: burn-in (not scoring the first ~100 of 400 samples) is the strongest measured fix for the
objective's discrimination (1.13x to 10.1x, MS scale), but the encoder's only gradient path for
the PHYSICAL states runs through exactly those samples (measured 178x gradient collapse at
K = 100, D-148 gate B6), so the user's concern is that burn-in trains the dynamics at the cost of
the encoder. Answer five sub-questions, each with verified citations mapped onto the rig facts of
section 4: (Q1) the SUBNET encoder line itself: what the subspace encoder is for, what trains it,
and its known failure modes (Beintema, Toth, Schoukens: L4DC 2021 and the Automatica 2023 deep
subspace encoder paper, plus any follow-ups on encoder quality); (Q2) encoders specifically for
AUGMENTED models: Hoekstra's encoder paper (reconstructability-map initialisation, Eqs. 16-17,
already the basis of our `linear_map` init) and what it leaves undefined for the augmented block
`W^a`; Kessels 2025 ch. 5 (encoder fed past output, reference, tracking error AND input;
Remark 5.3 direct initialisation of measured states through the inverted output map so the
encoder handles fewer states); whether anyone else constructs or trains the augmented-state
encoder block at all; (Q3) alternatives to training the encoder through the simulation loss:
state-consistency / defect criteria (multiple-shooting style), supervised pre-encoders, observer
constructions (KKL / deadbeat / reconstructability maps) that DERIVE the encoder instead of
learning it, and Forgione and Piga's line on whether a state estimator is needed at all in neural
state-space training; (Q4) what the recurrent-network and reservoir literature knows about
de-weighted early samples: washout periods in echo-state networks, warm-up in truncated BPTT,
teacher forcing, and any result on estimator/encoder degradation under such schemes, since
burn-in is exactly a washout; (Q5) the closed-loop specifics: encoders that use the reference and
tracking error as inputs (Kessels does; ours does not), and controller-state reconstruction at
window starts (Kessels Remark 5.4 replays the controller; our harness sets xc = 0, a documented
HEURISTIC deviation in `model_augmentation/fit_systems/closed_loop.py`). Deliver the skill's
mandatory Research Log per seed, per-question findings with the disqualification filter applied,
and `docs/references.md` rows for everything retained.

## 2. Out of scope

* Any training run, any code change, any edit to the loss or the encoder. The burn-in arm design
  is the integrating session's decision.
* The closed-loop identification questions (CLOE, dual-Youla, informativity) are a SEPARATE
  handoff (`tasks/handoffs/2026-08-20-closed-loop-literature.md`); do not duplicate them. Where a
  paper serves both, report it here only for its encoder content.
* Re-running the 2026-08-19 seeds (lazy vs rich init, latent-state init, encoder co-estimation):
  done, `scripts/gantry/closed-loop-controller/ANN-learning-issue/RESULTS.md` section 6. Note its
  standing conclusion, to be tested rather than assumed: no published work constructs the
  encoder's augmented-state block from anything but a random draw.
* No new `docs/*.md`; findings go in the skill's report and `docs/references.md`.
* Do not modify `scripts/gantry/gantry_dynamic/{config,evaluation,orth_penalty}.py`,
  `kamtin-fp-model/`, or anything under `kamtin-data/`.

## 3. Where things stand

Branch `Augmentation`, last commit `4cdb7c1`, tree dirty (D-150 fix and docs; the commit decision
is the user's). Nothing in flight. Pure literature session; the only code worth opening is listed
in section 11.

## 4. Established and verified (rig facts the findings must map onto)

* The encoder is a SUBNET-style subspace encoder with Hoekstra's reconstructability-map init for
  the physical rows; the augmented rows `W^a` are a random draw (`pre_encoder.py`,
  `linear_encoder_init_aug`; docs/references.md `hoekstra2026encoder` row).
* Under the pre-D-150 model, `W^a` received EXACTLY zero gradient and moved 0 of 108 entries in
  every run (D-130). With live augmented dynamics (D-150, `rho(A_aa) = 0.992`, 125-sample time
  constant), `Wa_psi_y` moved 108 of 108 (Arm F, run table). The augmented-state initial
  condition now influences samples 100-400 directly, so burn-in would NOT starve the encoder's
  augmented rows; the collapse concern is about its PHYSICAL rows, whose influence the loop
  suppresses within ~80 samples (~100 Hz crossover).
* Measured encoder-gradient collapse under burn-in: 178x at K = 100, 4182x at K = 200 (D-148
  gate B6, dead-states model).
* Weighted burn-in (`w_burn = 0.1` on the first 100 samples) keeps discrimination at 2.995x
  against 3.400x for the hard cut and 1.249x for no burn-in (RMS scale, D-148 finding 9), i.e.
  the early window can stay scored at reduced weight.
* A state-consistency (defect) criterion for the encoder exists in this repo, reverted with a
  re-appliable patch (`patches/2026-08-19-interconnect-burnin-consistency.patch`); it measured
  inert on the dead-states model because the defect was 0 - 0, and its mechanism becomes
  meaningful only with live `x_a` (D-148 findings 10 and the ANN-learning-issue folder).
* Kessels 2025 ch. 5 (on disk, `literature/augmentation/kessels2025_ai-control.pdf`, printed
  pp. 154-159): truncated windows; encoder inputs include reference and tracking error
  (footnote 5.4); Remark 5.3 direct init of measured states via the inverted output map, used on
  the ASMPT wire bonder; Remark 5.4 controller-state replay per window; near-zero NN init with
  the loop as an early-training stabiliser.

## 5. Assumed but not verified

* That the encoder's physical rows NEED the early-window gradient at all: with positions directly
  measured, Kessels' Remark 5.3 route (invert the output map for the measured states) might make
  most of the encoder's physical burden unnecessary. Settled by the Q2/Q3 findings plus, later, a
  rig experiment that is not this session's job.
* That the 178x collapse transfers to the live-dynamics model. It was measured with dead states;
  the loss landscape differs now. Flag any literature that quantifies estimator degradation under
  washout rather than assuming the old number.
* That "no published construction of W^a" still holds after this deeper sweep.

## 6. Tried and failed

* Keyword web search for control topics -> fails measurably -> the skill's author-ID and
  citation-edge enumeration is mandatory (D-121).
* The consistency term at 10 % of the MSE on the dead-states model -> indistinguishable from no
  term -> the defect was 0 - 0 with `rho(A_aa) = 0` -> D-148; do not cite it as evidence against
  the criterion class.
* Carrying novelty claims unswept -> refuted three times on 2026-08-19 -> RESULTS.md section 6.

## 7. Achieved

None yet in this search thread. The rig measurements it builds on are in section 4 with artefacts.

## 8. The open question

Does a principled encoder criterion exist that survives, or replaces, a de-weighted early window:
(a) a separate consistency/defect objective, (b) an observer-derived (non-trained) encoder,
(c) direct measured-state initialisation shrinking the encoder's job, or (d) richer encoder
inputs (reference, error) that make the early window less necessary? Candidate answers are
exactly (a)-(d); the found papers' assumptions against section 4 choose between them.

## 9. Next action

Invoke the `deep-research` skill framed by the five sub-questions of section 1, seeded by:
Beintema, Toth, Schoukens (subspace encoder, L4DC 2021 and Automatica 2023); Hoekstra, Toth,
Schoukens (encoder initialisation for augmentation, the `hoekstra2026encoder` entry in
`docs/references.md`); Kessels 2025 chapter 5 (on disk); Forgione and Piga (neural state-space
training, state estimators); echo-state / reservoir washout literature (Jaeger; Lukosevicius and
Jaeger survey) for Q4. One subagent per seed, five total, each returning its Research Log.

## 10. Acceptance criterion

Every sub-question Q1-Q5 answered with at least one verified, accessed source or an explicit
"nothing found" statement naming the enumerations tried; the Q2 novelty statement re-tested and
restated in falsifiable form (searched space named); every retained paper as a
`docs/references.md` row with PDF location or access status; and, because the search exists to
inform a design decision, a closing mapping table: each finding against options (a)-(d) of
section 8, stating which it supports, with the paper's assumptions that bound the transfer.

## 11. Read these first

1. `.claude/skills/deep-research/SKILL.md`: the mandatory procedure.
2. `scripts/gantry/closed-loop-controller/ANN-learning-issue/HYPOTHESES-AND-SOLUTIONS.md`
   sections 2 (H2), 6 (outcome, next arm): why burn-in is on the table at all.
3. `literature/augmentation/kessels2025_ai-control.pdf`, printed pp. 154-159 (PDF offset +27):
   the closest existing encoder design for closed-loop augmentation.
4. `model_augmentation/fit_systems/pre_encoder.py` docstrings around `linear_encoder_init_aug`:
   what our encoder actually is, including the random `W^a`.
5. `docs/references.md`: citation base and format, incl. the `hoekstra2026encoder` row.

## 12. Do not

* No ad-hoc `WebSearch` in place of the skill (D-121).
* Do not write conclusions into `ANN-learning-issue/` or the problem log; report them.
* Do not start any training arm and do not re-apply the reverted patch.
* Do not duplicate the closed-loop-identification handoff's questions.
* Do not commit anything.

## 13. Operational

No runs. PDFs land under `literature/` (existing conventions; `literature/augmentation/` fits
most of this; a subfolder for reservoir/washout papers may be created if needed, consistent with
`deep-ssm-init/` precedent). Artefacts consumed: none beyond the reads in section 11.

## 14. Delegation

Yes: one `deep-research` subagent per seed of section 9 (five). No Explore subagents, no
verification subagents.
