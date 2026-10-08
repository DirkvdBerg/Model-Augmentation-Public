# Handoff: fact-check how Hoekstra's LFR augmentation paper initialises added (dynamic) states, versus Jan's original code and our pipeline
**From**: session of 2026-10-02 | **Branch**: Augmentation | **Effort suggested**: high (careful reading of one paper and a git history; no computation)

## 1. Task
Independently verify or refute the four claims in section 4, using only primary evidence: the paper PDF, Jan's original code as it exists in git history, and our current pipeline code. For each claim, give a verdict (TRUE / FALSE / PARTLY / CANNOT DETERMINE) with direct quotes: paper quotes with page and section, code quotes with `file:line` (or `commit:file:line`), every code quote checked per the code-quote verification rule in `CLAUDE.md`. Look beyond the passages the previous session used: the paper's model-structure definition (Sec. 4 and the ResNet definition, Eq. 15), Appendix A, and the simulation study (Sec. 6 and 7: which augmentation and initialisation produced the reported dynamic-augmentation results). Report in text to the user; do not edit files.

## 2. Out of scope
- Any code change, run, or new experiment; editing `DECISIONS.md`, `docs/decisions.md`, `CLAUDE.md` or any script. The user decides what follows from the verdicts.
- The training-method discussion (innovation opening, E3 to E9): irrelevant here.
- Contacting Jan or proposing an initialisation change.

## 3. Where things stand
Branch Augmentation, tree dirty (unrelated files), no runs in flight.

## 4. Claims to check (from the previous session; treat as unverified)
- **P1 (paper):** In Hoekstra et al. 2026 (arXiv:2602.17297), the added states start "alive" at initialisation: the learning function is a ResNet whose nonlinear part is zero and whose linear part W_a is random, all matrices not needed for baseline behaviour are random U(-1,1), and only the couplings that would change the baseline output are zero. Basis: p.9, Sec. 5.4.1, Eq. (29) and Condition (b); Sec. 5.4.3, "phi_aug(z_a,k) = 0 + W_a z_a,k"; p.10, "All matrices not required to set the baseline model behaviour at initialisation (29) have all elements m of the matrix initialised randomly, according to [32], i.e., m ~ U(-1,1)". Check specifically: is W_a itself random, or could Eq. (33) with B~ = I, C~ = I force W_a or the relevant couplings to zero for the added-state rows? Do the added states really have non-zero dynamics at init under the paper's conditions?
- **P2 (Jan's code):** Jan's example code zero-initialises the whole learning function for dynamic PARALLEL augmentation (added states dead at init): `scripts/journal_model_augmentation/msd_ndof_interconnect_fit.py:175-183` uses `net=zero_init_linear_mapping` (a zero-weight, zero-bias `nn.Linear`, `model_augmentation/utils/torch_nets.py:86-91`); `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py:84-88` uses `net=zero_init_feed_forward_nn` (MLP, final layer zero). Series augmentation uses `identity_init_simple_res_net` (identity linear part, `torch_nets.py:51-56`). Check which script and settings correspond to the paper's reported dynamic-augmentation results.
- **P3 (provenance):** these files are Jan's original code, not ours. Evidence so far: `zero_init_linear_mapping` carries an `@added` marker today, but appears without it in commit 6d69f6b ("Revert back to Jan's original code", 2026-05-29); the marker came in be8ac7c (2026-06-22, marking pass). Earlier commits 40e19d1 (2024-04-02, "Initial submission code"), 1bccce6 (ECC 2025), c9f913a (2025-06-17) do not contain that class. Determine from git history (and from any copy of Jan's original repository you can find on disk; check the sibling folder `../model aug-wafer` only if it is Jan's code) whether `zero_init_linear_mapping` and the journal example script are Jan's or were added in this project, and whether the `@added` marker is correct.
- **P4 (ours):** our gantry pipeline follows Jan's code, not the paper, for the added states: `scripts/gantry/gantry_dynamic/model.py:138-144` builds `Static_ANN_Block(..., net=zero_init_feed_forward_nn, ...)` (one MLP writing routed physical rows and the added rows, final layer zero, no linear part), so x_a(k+1) = 0 at init. Also check whether our encoder initialisation matches the paper (Sec. 5.4.2, Eqs. 30, 31: W^b fitted, W^a random by Xavier; ours: `model_augmentation/fit_systems/pre_encoder.py`, `linear_encoder_init_aug`, kaiming_uniform_).

## 5. Assumed but not verified
- That the arXiv PDF on disk is the version the user cites in the thesis.
- That Jan's paper results were produced with the journal or ECC scripts in this repository.
- That `be8ac7c`'s marking pass compared against Jan's true original code.

## 6. Tried and failed
None (first fact-check).

## 7. Achieved
The previous session read only pp. 9 to 10 of the paper and the files and commits listed in section 4.

## 8. The open question
Do the paper and Jan's own example code agree on how added states are initialised for dynamic parallel augmentation, and which of the two does our pipeline follow?

## 9. Next action
Read the paper in full where it defines the model structure, the learning functions and the initialisation, and where it describes the dynamic-augmentation experiments; then trace the code claims through git (`git log`, `git show <commit>:<file>`, `git log -S`); then write the verdicts.

## 10. Acceptance criterion
Each of P1 to P4 has a verdict backed by at least one verified direct quote; any contradiction between the paper and the code is stated with both quotes side by side; the answer to section 8 is one sentence.

## 11. Read these first
1. `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`: Sec. 4 (LFR structure, Eq. 15), Sec. 5.4 (pp. 9 to 10), Appendix A, Sec. 6 to 7 (experiment settings).
2. `scripts/journal_model_augmentation/msd_ndof_interconnect_fit.py` and `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py`: Jan's examples.
3. `model_augmentation/utils/torch_nets.py`: the learning-function classes and their markers.
4. `scripts/gantry/gantry_dynamic/model.py` around l.138 and `model_augmentation/fit_systems/pre_encoder.py` (`linear_encoder_init_aug`): our pipeline.
5. `CLAUDE.md` section "Tracking Additions to `model_augmentation/`": what the markers mean.

## 12. Do not
- Edit or move any file, including fixing a wrong marker; report it instead.
- Quote code without a MATCH OK from `scripts/verify_quote.py`.
- Infer the paper's method from Jan's code or vice versa; each claim stands on its own source.

## 13. Operational
Quote check: write the quote to a scratchpad file, then `conda run -n GraduationProject python scripts/verify_quote.py <file> <start> <end> <quotefile>`. For files at an old commit, extract them with `git show <commit>:<path>` into the scratchpad first and verify against that copy.

## 14. Delegation
None needed; this is targeted reading.
