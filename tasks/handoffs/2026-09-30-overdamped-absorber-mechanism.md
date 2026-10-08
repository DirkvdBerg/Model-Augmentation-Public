# Handoff: find the mechanism that makes training converge to an over-damped absorber
**From**: session of 2026-09-30 | **Branch**: Augmentation | **Effort suggested**: xhigh (a causal diagnosis that combines literature with decisive local measurements; the method is yours to choose)

## 0. How to read this prompt
- **The user's words (spelling cleaned):** "we have identified where the error is but not what is causing this problem"; "create a prompt for a session that will read Hoekstra his augmentation paper, and tries to find out what is causing this issue"; "make sure it has all the context and it will focus on finding out what the issue is, and that it can think/decide for itself".
- **Task type: a causal diagnosis.** The deliverable is the mechanism, with evidence that separates it from the alternatives. A list of remedies is secondary and only follows from the mechanism.
- **You decide the method.** Sections 5 and 8 give hypotheses; they are a sample, not the list. Add, drop, or reframe them as the evidence tells you. Readings of this prompt's author are marked **(reading)**.

## 1. Task
Establish WHY gradient training of the thesis augmentation model converges, systematically, to a smooth over-damped approximation of the payload absorber instead of its sharp 212 Hz resonance, although the training objective itself prefers the sharp truth by a factor 3.8 or more. Use Hoekstra's augmentation papers and related literature to learn what the framework assumes and what is known about learning lightly damped modes; use local measurements on the existing checkpoints to decide between mechanisms. End with the mechanism stated in one paragraph, the evidence for it and against each alternative, and one decisive experiment (with prediction, criterion and cost) that the user can choose to run.

**First split: representation or optimisation.** The 3.8x is measured with the true plant in the loop, not with a member of the model class (n_a added states written by a 2x16 tanh map). Decide first whether the model class can realise the sharp resonance at all (for example: a supervised fit of the added-state block to the true absorber correction, placed into the trained model; does it reach the truth's loss?). "Cannot represent" and "can represent but training does not get there" lead to different remedies; every later hypothesis sits on one side of this split.

## 2. Out of scope
- Full training runs and server jobs (user decides; propose them, never launch). Allowed as measurements: short local gradient steps on existing checkpoints or on a fitted solution, at most 50 optimizer updates per measurement, with the prediction written in `DECISIONS.md` first.
- Changing the thesis data or the multisine band (DATA-DESIGN, D-219). A band cut to 140 to 230 Hz would hide the defect; it is the user's last resort, not your remedy.
- Editing the main pipeline in place (`scripts/gantry/gantry_dynamic/`, `gantry_interconnect_dynamic.py`, `model_augmentation/`): copy what you change into your own folder.
- The running thesis campaign (array 86892, `scripts/gantry/thesis-results/runs.tsv` ids 41 to 53): do not touch.
- The black-box model (separate session); `Thesis-writeup/` is read only.
- The route-record defect (section 4, last bullet) unless your mechanism explains it too.

## 3. Where things stand
- Branch `Augmentation`, tree dirty in many directories (not yours to clean). Nothing running locally.
- Diagnosis folder: `scripts/gantry/absorber-learning-diagnosis/` holds `DIAGNOSIS.md` (the filled decision table), `DECISIONS.md` (every prediction written before its number, then the result), `RUNS.md`, scripts `b_` to `g_`, and `outputs/`. Continue in this folder; append to its `DECISIONS.md` before each new number.
- Local thesis checkpoints (deepSI whole-system dumps): `scripts/gantry/meeting/meeting-01-10-2026/server/Checkpoints/SSE_Interconnect_Composed_<job>_{best,last}.pth`. Job to run: 86893 = 41 (OBC, n_a 2, seed 1), 86894 = 42 (U, 2, 1), 86909 = 43 (OBC, 2, 2), 86918 = 44 (U, 2, 2), 86945 = 45 (OBC, 2, 3), 86960 = 46 (U, 2, 3), 86961 = 47 (OBC, 0, 1), 87018 = 48 (OBC, 4, 1), 87055 = 49 (OBC, 8, 1, only 1.3k updates: not usable). Loading: `e_thesis_checkpoints.py::cfg_for` + `load_checkpoint(fs, path, True, 'lbfgs')`.
- Logs of those runs: `scripts/gantry/meeting/meeting-01-10-2026/server/thesis_86892_<run>.out`.

## 4. Established and verified
All from 2026-09-30, numbers in `DECISIONS.md` (entry ids in brackets) unless noted.
- **Setup.** Truth = FP gantry + tanh Coulomb rail friction (g 1000) + payload absorber (ma 0.5 mh, fa 150 Hz anti-resonance, coupled pole 212 Hz, zeta_a 0.03, coupled zeta about 0.071); multisine 106 to 297 Hz at A_prod; one controller K1; encoder noise. Model: reduced 10-combination physics block (joint, 10 % detuned) + Static_ANN_Block (2x16 tanh, zero-initialised output layer) writing the 6 physical rows and n_a added-state rows; linear-map encoder over 29 samples; closed-loop residual rollout, nf 400 (0.1 s) at 4 kHz, float64, Adam lr 1e-5, eps 1e-16, batch 512 (D-221, D-222, `Thesis-writeup/Documentation/TRAINING-DESIGN.md`).
- **The absorber is in the data** (B1): Y signature 3.6e-5 m (thesis) vs 3.5e-5 m (old data); friction in 140 to 230 Hz 3e-7 to 7e-7 m, noise floor < 1.2e-7 m.
- **Learned below the pole, not above** (E1, run 42, Y, 4 multisine val records): energy removed 90 % (106 to 140 Hz), 89 % (140 to 230 Hz), 22 % (230 to 297 Hz). 230 to 297 Hz holds the largest share of the thesis absorber error; the old band (140 to 230 Hz) had < 1e-6 m there, which is why old runs (84032, 81265, 82675) looked good.
- **Shape of the learned correction** (F1 to F3, `f_above_pole.py`): H = learned / true correction at Y (Welch 0.5 Hz): gain 1.0 at 150 Hz, 0.60 at 212, 0.45 at 230, 0.30 at 250, 0.26 at 264, 0.28 at 280, 0.42 at 296 Hz; phase error rising smoothly to +125 to +132 deg; coherence 0.98 to 1.00. No notch, no 180 deg jump at the pole: a broad, over-damped bump.
- **Systematic** (F2, F3): runs 42 (seed 1, U, n_a 2), 48 (seed 1, OBC, n_a 4) and 44 (seed 2 = other network init AND other detune direction, U, n_a 2) agree within 0.02 in gain and a few degrees in phase.
- **The objective prefers the truth** (G1, `g_window_loss.py`, exact training loss, own metric = `fs.loss` to 5 digits, 864 windows): model 5.09e-9; truth with its absorber state ZERO at each window start 1.35e-9 (0.27x); truth with true state 5.1e-11 (0.01x); truth without absorber 1.25e-8 (2.5x). The model's Y error is flat over the window (2.16e-5 m steps 0 to 40, 2.17e-5 m steps 40 to 400): its deficit is the steady-state shape, not the initial transient. So: an optimisation problem, not the objective and not the encoder window.
- **Too few epochs: refuted as the main cause** (`DIAGNOSIS.md` table; logs 41 to 49; F2): U curves flat in the last quarter (run 42: 1.031e-5 -> 1.027e-5 m) and the above-pole shape is the same at 22k and 30k updates. Caveat to keep in mind, not to reopen without new evidence: the thesis runs start improving at 13k to 19.5k updates against 2.6k to 7k on the old data, so at lr 1e-5 "stuck" and "very slow" differ only in the landscape; if your mechanism implies "very slow", show why the flat curve and unchanged shape do not contradict it.
- **Refuted as causes** (`DIAGNOSIS.md` table): data size, band signal size, friction in band, noise, record mix, K1 vs controller bank, loss weighting, pipeline change (old checkpoint 84032 reproduced to 5 digits, D1), parameter substitution (tangent span 34 % thesis vs 65 % old, C2), 264 Hz closed-loop peak (|S_YY| smooth 1.28 to 1.56), pure bandwidth limit (gain recovers above 270 Hz), capacity 2 -> 4 states, OBC vs U, seed.
- **Static part and parameters.** Static part alone (added-state rows zeroed) has |H| 1.47 at 110 Hz, 0.96 at 190 Hz: it imitates the mass-like part of the absorber below the pole; the added states correct on top. Masses recover (m_total, m_diff, J_eff within 2 %, mh 5 %); damping and stiffness drift (cg1 +23, cg2 +32, cy +16, cb_sum -13, kb_sum -29 %), also at n_a 0 (run 47): friction's tanh slope (1.7e4 N s/m at v 0) and viscous damping substitute.
- **Dip timing.** On the old data every configuration's dip starts at 2.6k to 7k updates; thesis runs at 13k to 19.5k (inventory in `DECISIONS.md`, 2026-09-30 inventory entry).
- **Routes** (VA-T1/T2): model 4.0e-6 m Y, learned block removed 1.5e-6 to 2.2e-6, static part only 6.0e-6 to 9.6e-6 m.

## 5. Assumed but not verified (candidate mechanisms, a sample)
- **(reading) Local damping preference / narrow basin.** A resonator that is not phase-aligned with the true residual is a nuisance term; damping it lowers the loss locally, so the sharp solution sits in a narrow basin that gradient descent from zero init does not enter. Prior support, different data and setup: problem log section 15.3 (D-150/D-151, old absorber at 158 Hz): once the added mode was driven, `dL/d(nu_log) < 0` on 8 of 8 batches, i.e. descent damped it ("when the augmented mode is finally given a voice in the objective, the objective votes to attenuate it"). Settle: measure the loss along damping and frequency of the learned mode around the trained point, or around a supervised-fitted sharp solution.
- **(reading) Sequential fitting distorts the residual.** Zero-init output rows mean the static part and parameters learn first (mass-like imitation below the pole); the added states then fit a distorted residual whose best low-order approximation is a broad bump. Settle: fit the added-state block with the static part frozen at zero, or on the pure absorber target.
- **(reading) Representation / conditioning.** A 2x16 tanh map used recurrently must realise a discrete pole at |z| = 0.977, angle 0.333 rad (4 kHz); tiny weight errors change the damping a lot, which may flatten the landscape toward damped solutions. Settle: sensitivity of the realised pole to the weights; or supervised fit of the block to the true absorber map (does it reach it?).
- **(reading) Step size against pole sensitivity.** Adam at lr 1e-5 (eps 1e-16) moves each parameter by at most about 1e-5 per update, while the realised pole at |z| = 0.977 may be very sensitive to the weights: the optimiser may be unable to move along the narrow direction that sharpens the mode without leaving it, or may be driven along the flat damping direction instead. Settle: realised pole and loss change per update along the Adam direction versus along the direction to the sharp solution.
- **(reading) Joint-parameter interplay** (damping combos drift with friction) shifting the effective damping the added states must supply.
- Whether Hoekstra's published examples ever contained a lightly damped mode excited ABOVE its resonance: unknown.

## 6. Tried and failed (earlier in the project; do not repeat blindly)
- D-150 live `A_aa` pole pair -> mechanism worked (`rho` to 0.992, loss barrier removed) -> free run -0.665 % only -> the undriven mode carried no plant information -> problem log 15.3.
- D-151 `B_a` input injection -> pole gradient coherent (0.086 -> 0.827) -> but points toward DAMPING the mode -> nuisance-regressor argument -> problem log 15.3. Not run as a full arm.
- Multiple-shooting defect -> validation 2.8x worse at epoch 1 -> defect minimised by collapsing the added states -> problem log 15.2 item 3.
- GRU / J1 and `A_aa + B_u` variants: context in `docs/references.md` (entries `schoukens2021ssnn_init`, the "Literature-level answer to A_aa + B_u versus J1/GRU" note) and `docs/aug-lru-implementation.md`.
- Caution: those were measured on the old absorber (ma 0.1, 158 Hz) and older code; they are hypotheses for today's data, not facts.

## 7. Achieved
- Diagnostic instruments, all validated: vectorised closed-loop oracle (= `common/oracle.py` to 2e-20 m), band budget, checkpoint evaluation (reproduces logged bestfit), per-frequency learned/true ratio, exact training-loss evaluation for model and truth variants. Scripts `b_error_budget.py`, `e_thesis_checkpoints.py`, `f_above_pole.py`, `g_window_loss.py`; oracle val cache `outputs/f_oracle_val_cache.npz`.

## 8. The open question
Which mechanism leads training to the damped solution while a 3.8x better one exists? Candidate answers: section 5, or one you find. Evidence that separates them: a local measurement whose outcome differs between them (for example: the loss landscape along the damping of the learned mode; whether a supervised fit of the added-state block to the true absorber correction, placed into the trained model, lowers the training loss and stays put under a few gradient steps; the gradient direction on the added-state parameters at that point). Literature tells you which mechanisms are known and what the framework assumed.

## 9. Next action
Read Hoekstra's LFR augmentation papers (section 11) for how the added dynamics are parameterised, initialised and trained, and whether their demonstrations cover a lightly damped mode above resonance; in parallel run the literature sub-questions below; then design and run the local measurement that best separates your leading candidates. The literature sub-questions were approved by the user on 2026-09-30 as a starting point:
- Q1: Hoekstra's framework: assumptions and practice for initialising and training the added dynamics (zero-init correction, encoder, horizon, multiple shooting); examples with lightly damped modes.
- Q2: why gradient-based simulation-error training of state-space / recurrent models converges to over-damped approximations of lightly damped resonances; loss-landscape results near sharp poles.
- Q3: initialising or structuring learned dynamics to capture resonances (BLA / subspace initialisation of added states, frequency-domain pretraining, curricula over band or horizon, multiple shooting), and their compatibility with the LFR augmentation framework.
A new sub-question beyond these: state it to the user in one line before launching its agent (standing preference).

## 10. Acceptance criterion
Done when `scripts/gantry/absorber-learning-diagnosis/MECHANISM.md` states: the mechanism in one paragraph; for it and for each alternative you considered, the deciding number (with file) or the source (with page/equation) that supports or refutes it, predictions written in `DECISIONS.md` before each number; and one decisive experiment with prediction, pass/fail threshold and cost (updates, hours, local or server). A mechanism supported only by literature and not by a local measurement on these checkpoints is not done. Errors in metres, scientific notation.

## 11. Read these first
1. `scripts/gantry/absorber-learning-diagnosis/DIAGNOSIS.md` and `DECISIONS.md`: everything established today, with numbers.
2. `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf` (arXiv:2602.17297) and `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf`: the framework (dynamic-parallel added states, Table 1; SUBNET objective Eq. 22a); `docs/references.md` entries `hoekstra2026lfr`, `hoekstra2026lfrfp`, `hoekstra2026encoder`.
3. `docs/gantry-augmentation-problem-log.md` section 15 (added-state learning, D-150/D-151 damping direction).
4. `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf` (encoder init covers the baseline states only).
5. `f_above_pole.py` and `g_window_loss.py`: how the two key measurements were made.

## 12. Do not
- Launch training or server jobs; use `kamtin-data/Telica.mat` or `kamtin-data/Data Telica/`.
- Judge the added states by their internal values (arbitrary gauge); judge at the output.
- Re-establish anything in section 4 unless your mechanism contradicts it.

## 13. Operational
- Local numerical jobs: launch DETACHED so Claude Code's memory reaper cannot kill them (it killed a job on 2026-09-30): PowerShell `Start-Process -FilePath "C:\Users\20203253\AppData\Local\anaconda3\envs\GraduationProject\python.exe" -ArgumentList '-u','<script>','<args>' -WorkingDirectory <diag folder> -WindowStyle Hidden -RedirectStandardOutput outputs\<name>.log -RedirectStandardError outputs\<name>.err -PassThru`, with `PYTHONUNBUFFERED=1` set first; watch the log with a Bash until-loop.
- Check free RAM first (about 3 GB free typical); one numerical job at a time.
- Measured runtimes: U checkpoint, 4 val records full length, 3 rollouts: about 15 min; OBC rollouts several times slower (per-step JVP), over 11 min per rollout; `g_window_loss.py` (864 windows, nf 400) about 5 min.
- Give every runtime with its measured basis.
- If run unattended: run literature agents in the foreground, and never end a turn while a job or agent is still running (two headless sessions died that way on 2026-09-24); poll detached jobs with short foreground checks.

## 14. Delegation
Literature: at most 3 agents, one per sub-question, via the `deep-research` skill. Code or numerical work: at most 1 subagent; prefer doing it inline.
