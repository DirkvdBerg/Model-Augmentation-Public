# Handoff: decide the training (hyper)parameters for all thesis results, starting with a proper method for the joint-estimation window length
**From**: session of 2026-09-27 | **Branch**: Augmentation | **Effort suggested**: xhigh (a design argument across control theory, SUBNET literature and many earlier runs; the window question has resisted several sessions)

## 0. How to read this prompt (read first)
- **The goal, in the user's words (spelling cleaned):**
  - "I want to discuss the (hyper)parameters we should use in training, such as downsampling, number of epochs, quasi-Newton."
  - "For the encoder length we currently use Jan Hoekstra's rule, but when I add additional states, 2 or 8 for example, I'm not sure if I should keep that rule or if it's an unfair comparison. In my previous run I kept that rule, but I also increased the neural network size with added states, and 2 still got a similar result. So I don't know what the right approach is, because I vary multiple things at once, but I kind of think it makes sense to do that. I think we can learn a lot from Jan Hoekstra's papers for this."
  - "The most important thing is to determine the window length for joint estimation. My supervisor Maarten suggested increasing the window and seeing the result of it, so he suggests a training run I think. I don't know the best approach for this."
  - "For the baseline with the correct parameters we also added Coulomb, so I don't know if we can still keep nf = 400 (0.1 s). I would think that Coulomb, since it doesn't have a state or memory, will be enough, but we can discuss that as well."
  - "The most important thing to discuss is that I want a proper approach to determine the window length for the joint estimation; maybe the FRFs we made and our multisine design can also help with that? Because of our new multisine design we might also have to look again at the window length for the baseline with the correct parameters."
  - "Then the next big thing is (hyper)parameter selection, so we are ready to run / generate all of our results. I want a clear (hyper)parameter outline for all of the results we will run."
  - "This should include the window length for detuned and not detuned."
- **Task type: a discussion with the user first, then a design on paper.** You propose, argue and let the user decide; the user decides nothing by default. Small computations that settle a discussion point (for example closed-loop time constants, an impulse-response length from the FRFs, stick durations in the data) are allowed once the user agrees to them, one numerical job at a time. Training runs are proposed with their hypothesis and pre-stated criterion; none is launched in this session unless the user says so.
- **Stance:** control engineer first (CLAUDE.md, Control Engineering Stance). Every setting is argued from the system, the data and the literature, and each criterion is data-derived. When the evidence leaves more than one reading, present all of them.
- **No burn-in (user decision, 2026-09-27).** Every training window is scored from its first sample; nothing is discarded at the window start. The startup transient is handled through the encoder, not by burn-in. Do not propose burn-in, and do not use burn-in numbers as a target; where section 4 quotes one, it is history only.
- Where this prompt gives my reading, it is marked **(reading)**. It is not the user's decision; say so when you use it.
- **An example is a sample, never the scope.** The user's examples name kinds: "downsampling, number of epochs, quasi-Newton" stand for every training setting; "2 or 8 added states" stands for the state-count comparison as a whole; "Maarten suggested a run with a larger window" is one way to settle the window, not the method; "Jan Hoekstra's papers" means the line of work this thesis builds on, with Hoekstra as the entry point, not its boundary. Work out each whole kind; do not reproduce an example or turn it into a rule. The candidate methods in section 8 are candidates too; the class is wider.

## 1. Task
Two parts, in this order:
1. **Window length for joint estimation (most important).** With the user, arrive at a defensible method to choose the training window (nf, truncation length), and a value, for each of two starts:
   - **Not detuned:** the baseline starts at the correct parameters. The window must let the augmentation learn what the baseline misses, and the parameters should stay put.
   - **Detuned:** the baseline starts at the 10 % detuned combinations (D-191, `combo_init_detune`), and the parameters are estimated jointly. This covers the physics-only refit B-refit and the augmented arms of R4. Here the window must also make the parameter error visible to the loss. How well a given window separates true from detuned parameters has not been measured on any dataset (the 1.25x of D-178 is a different quantity, section 4).

   Say whether one window can serve both starts or each needs its own, and why. Everything is now on the thesis truth: Coulomb friction plus payload absorber, the new multisine band 106 to 297 Hz, and the fixed controller K1. The method must say:
   - what sets the lower and upper bounds;
   - how the thesis FRF records and the multisine design enter;
   - whether Coulomb friction changes the answer;
   - what (if any) training runs confirm it, with their pre-stated criterion.

   Decide with the user whether to keep or change nf = 400 (0.1 s), per start.
2. **A (hyper)parameter outline for every result.** For each setting (sample rate, window and stride, encoder lag, network size, added states, epochs, batch, optimiser and the quasi-Newton stage, learning rates, normalisation, parameter prior, starts and detuning, checkpoint selection, black-box search space), give:
   - the value;
   - its class (physics-derived, inherited, validation-tuned, compute-constrained; see `Thesis-writeup/Writing/sections/05_setup.tex:9`);
   - the justification with its source;
   - which results and arms use it (R1 to R8, `RESULTS-DESIGN.md`).

   Include the fairness rule for comparisons that change the state count (encoder lag and network size when going from 0 to 2 to 8 added states, result R2).

## 2. Out of scope
- Generating or changing data: the thesis data set is fixed (`Thesis-writeup/Code/Data/GENERATION-REPORT.md`); the data design and results design documents are not edited here unless the user asks.
- Launching training runs, or editing training code, before the user agrees to the plan; then per the run-discipline rule (run-table row before launch).
- `Thesis-writeup/Writing/`: read only.
- Real Telica data (out of the thesis main results); `kamtin-fp-model/` read only.
- Reopening decided design points (data sets, band, noise model, controller K1, the results R1 to R8 themselves).

## 3. Where things stand
- Branch `Augmentation`, last commit `d641059`; tree dirty with other sessions' work (do not commit or clean).
- Thesis data (`Thesis-writeup/Data/Coulomb-and-MSD/` and `Coulomb-and-MSD-noise-free/`):
  - 12 s records at 20 kHz, each with a 0.5 s hold at the start: windows of several seconds fit inside the existing records, so a longer window never needs new data (it does leave fewer windows per record and costs more per window);
  - training 18, validation 6, test I1 to I6 and E1 to E6, FRF references TF-1 to TF-3 (Y 0.15, 0.225, 0.35) with 4 phase realisations each.
  - All 48 records exist in both versions (48 / 48 files, completed 2026-09-27, checks C3 / C4b / C8 / C14 PASS on all). An earlier note had 38 of 48 with the 10 FRF records pending: confirm the count in `Thesis-writeup/Code/Data/GENERATION-REPORT.md` before relying on the FRF records.
  - Read with `scipy.io.loadmat(..., mat_dtype=True)`; fields include `r_sim`, `f_sim`, `y`, `x_logical`, `u_total`, `d_in`, `meta`.
- The production training script `scripts/gantry/gantry_interconnect_dynamic.py` has not yet been pointed at the thesis data (reading; check its data loader, `scripts/gantry/gantry_dynamic/data.py`).
- No run in flight from this line of work.

## 4. Established (read in the repo; pointers are the evidence)
Current production settings (`scripts/gantry/gantry_interconnect_dynamic.py` = E, `scripts/gantry/gantry_dynamic/config.py` = C):
- Sample rate: 20 kHz data, trained at 4 kHz (E:99 to 100). 4 kHz accepted with a stated bias (D-141); 2 and 1 kHz rejected on phase margin (D-166).
- Window: `nf_seconds = 0.100`, so nf = 400 at 4 kHz (E:213).
  - Justified in C:185 as "5 tau_msd, tau = 1/(zeta wn) = 20 ms", which is an earlier absorber.
  - Stride 10 (E:101); burn-in 0 (E:104, D-178), and it stays 0 (user decision, section 0).
- Encoder lag, "Jan's rule": na = nb = 2 (nx_phys + nx_ann) + 1 (C:516 to 520), so 29 with 8 added states.
  - The rule comes from Hoekstra's ECC-2025 code (`scripts/ecc_2025/msd_ndof_interconnect_dynamic.py:122`), not from a paper; D-064 calls it a margin over the observability minimum.
  - Beintema, Schoukens, Toth 2023 §3.2: n = nx - 1 unbiased, a longer lag lowers the variance (`docs/augmentation-training-status.md:115` to 178).
  - Lags used in the papers: `docs/excitation-recipe-extraction.md:51, 59, 77, 100`.
- Network: 8 added states, 24 nodes x 3 layers, tanh (E:150 to 160, D-189).
- Budget: 200 epochs (E:198, raised from 150 to fit a 24 h wall), batch 512 (E:166).
- Optimiser: Adam, lr 1e-5, eps 1e-16 (E:169 to 170, D-148), then an L-BFGS polish (E:216 to 236, D-171): 100 iterations, history 50, 16384 fixed windows.
- Parameter prior: `param_rmse_baseline` 0.01 (E:82, D-076), off in the OBC arms (D-193); start detuning ±10 % per identifiable combination (D-191).

Window and horizon history:
- Slow closed-loop dynamics measured on earlier setups:
  - two integrators at |z| = 1;
  - X time constants 1.26 to 1.55 s, Y 1.01 s, a 5.1 Hz pair with tau 0.335 s;
  - 0.1 s is 0.065 tau_X.
  - Sources: `tasks/handoffs/2026-08-31-downsampling-and-required-horizon.md:35` to 41; `tasks/handoffs/2026-09-23-joint-estimation-horizon.md:58` to 73.
- Window discrimination (D-178, `closed-loop-controller/cl_burnin_sweep.py`): the window-loss ratio between the planted model (baseline plus the correct absorber) and a trained model, NOT between true and detuned parameters. On the old ma_frac 0.10 dataset: 1.25x at nf 400, 2.54x at nf 3200, 3.40x with burn-in 100. Re-measured on the ma50 production dataset (`scripts/gantry/encoder-transient/REPORT.md`, ET-002): 1.275x at nf 400, 19.23x with burn-in. DC blindness needs about 2 to 3 s (`docs/gantry-augmentation-problem-log.md` = PL:2429 to 2508).
- **The startup transient dominates short windows** (`scripts/gantry/encoder-transient/REPORT.md` G1 to G5): on the ma50 dataset 99.6 % of a correct model's window loss at nf 400 is the initial-state transient. An encoder map built from the model it feeds removes it (transient share 99.6 % to about 0 %, discrimination 1.275x to 177x), and a per-epoch rebuild (`g2refresh`) reproduces that without encoder training. With burn-in ruled out, the encoder fix is the way to remove the transient; settle what the window must buy once the transient is gone, not before.
- Parameters move slowly: by epoch 150 they had removed 24 % of their error, while the network was done by epoch 30 (2026-09-23 handoff:51 to 56).
- `scripts/gantry/joint-estimation-approach/APPROACH.md` section 2 (final version, after its critic), a paper diagnosis, never run:
  - the optimiser sets the parameters' speed: Adam's step is bounded by about the learning rate per coordinate (1e-5, chosen for the network), and below that bound low gradient sign consistency slows them further;
  - the closed-loop weighting does NOT favour the absorber band: `|S_o G|` weights a force residual at 1 Hz and 150 Hz within a factor of about 5;
  - the damping signal is small next to the 35 N Coulomb force on the X rails;
  - in closed loop the window must cover the impulse response of `S_o G`, not `m/c`; that duration is not yet computed;
  - its recommendation: staged estimation (physics-only parameters first, augmentation on the short window with parameters fixed but projected, joint Gauss-Newton refinement as a check).
- Supervisor on the window: the repo records R. Toth (2026-08-28): "the long-term dynamics can only be manifested by increasing the window" (PL:940 to 943). The user now reports Maarten suggesting a run with a larger window; both can hold; ask the user what exactly Maarten proposed.
- Literature rules for the truncation length (`scripts/gantry/joint-estimation-horizon/LITERATURE.md:13` to 17, 25, 54):
  - Beintema 2023 §3.4: "a few times the largest characteristic time scale";
  - Beintema 2021: about 4x;
  - Gyorok 2025 §5.3: about 1.5x;
  - Hoekstra: no rule.
- Coulomb:
  - Karnopp stick in the truth (D-209).
  - Telica TR-014: "friction memory (the controller integrator stops where stiction catches the rail)" (`telica-real/DECISIONS.md:292` to 296).
  - Early Coulomb rig: raising NF from 300 to 900 diverged or got worse (PL:655).
  - `RESULTS-DESIGN.md:105`: Coulomb needs no added state.
- Thesis data, stick fractions (`scripts/gantry/thesis-data-verification/logs/generation_log.csv`, stick columns): about 2 to 10 % per axis on the FRF standstill records; about 75 % on the multisine-free ILC-shape routes (GENERATION-REPORT F1).

Results and budget (`Thesis-writeup/Documentation/RESULTS-DESIGN.md`):
- R1 LPV vs LTI; R2 added states n_a ∈ {0, 1, 2, 4, 8} with the knee at 2; R3 BLA at three Y; R4 parameter spread over 5 starts, OBC vs unconstrained vs penalty; R5 conditional T_OA; R6 generalisation on the I/E pairs incl. black box; R7 controller transfer on E5; R8 training effort.
- 5 starts per arm, about 60 runs, about 300 to 660 GPU-h (:71, :211 to 216).
- Black-box size by hyperparameter search on validation (:69, :196).

## 5. Assumed but not verified (settle before building on them)
- **(reading)** The window justification in C:185 no longer fits the truth. With the thesis absorber (ζ 0.03 at its own 150 Hz, ka = ma (2π 150)^2), the same formula gives tau = 1/(0.03 · 2π · 150) ≈ 35 ms, so 5 tau ≈ 0.18 s. The coupled closed-loop decay can differ: compute it from the linearised thesis truth with K1 (`Thesis-writeup/Code/Data/tdg_truth_lin.m`; or `scripts/gantry/excitation-closed-loop/common.py`, `C.loop`).
- The slow time constants of section 4 were measured with earlier controllers and truths; they must be recomputed for T_AF with K1 designed once at Y = 0.
- **(reading)** The FRFs can bound the window from data. The measured closed-loop FRFs are B6 to B10 at five Y and B11 (`scripts/gantry/excitation-closed-loop/outputs/j2_bla.npz`), plus the thesis TF records. Their inverse transform gives a nonparametric impulse response. The time for it to fall below the noise floor is the memory the loss must span. This is exactly the open check in `APPROACH.md:44, 199`: does the S_oG impulse response last longer than 0.1 s?
- **(reading)** Coulomb is memoryless in the plant, but in closed loop the stick duration depends on the controller's integral state and history. So the effective memory may exceed the linear one on routes with long stick phases. The user's view (memoryless, 0.1 s is enough) is one of the two readings. Stick-episode durations in the thesis records (velocity from `x_logical` against V_BRK 1e-3 m/s) would decide it.
- "The baseline with the correct parameters, we also added Coulomb": does the baseline MODEL now contain a Coulomb term, or does only the truth have it? Ask the user.
- "All the R7 results": `RESULTS-DESIGN.md` R7 is controller transfer. The sentence reads as "all results", so ask once whether the outline covers R1 to R8 or R7 only.
- Where the outline goes: proposed `Thesis-writeup/Documentation/TRAINING-DESIGN.md`, a sibling of DATA-DESIGN and RESULTS-DESIGN. It is a new file, so confirm with the user before writing it.
- Whether the production loader reads the thesis MAT files unchanged.
- Why the parameters move slowly: `APPROACH.md`'s optimiser explanation (step bound plus low gradient signal-to-noise) is argued on paper, not measured. Note that 19,500 updates at lr 1e-5 allow 0.195 in log coordinates against the 0.1 detuning, yet only 24 % was removed, so the bound alone does not explain it; an earlier step-budget argument was withdrawn on 2026-09-18. Treat the cause as open when choosing optimiser settings.
- **Noise in the thesis data**: the noisy version carries a force disturbance on the motor input (`Thesis-writeup/Documentation/NOISE-INJECTION.md`, D-218), so every record has an error floor the model cannot remove (floor = noisy minus noise-free twin). What this means for training settings (normalisation, precision, selection against the floor) is in `NOISE-INJECTION.md` section 8 and `scripts/gantry/noise-training/NOISE-TRAINING.md` (written for encoder injection; section 8 of the noise document says what lapses and what holds). The outline covers both the noisy and the noise-free data.

## 6. Tried and failed (do not repeat blindly)
- Longer-window training: runs 71013 (up to nf 3200) and 74045 (nf 800) failed (`docs/drift-demonstration-plan.md:486` to 507; read the mechanism there before proposing a longer window again).
- Early Coulomb rig: NF 300 to 900 diverged or got worse (PL:655).
- 2 kHz and 1 kHz training rates: rejected on phase margin (D-166). Roland's multi-rate idea was not built beyond a premise test (`tasks/handoffs/2026-08-27-dataset-learnability-and-horizon.md:25` to 30).
- 2 vs 8 added states: every comparison so far changed the lag and the width together with the state count. The results are therefore confounded:
  - D-189: run 80554 (nx 2, 16x2, lag 17) against 80555 (nx 8, 24x3, lag 29);
  - pair 81265 against 81262, marked "NOT SHOWN" (`scripts/gantry/thesis-results-plan/RESULTS-PLAN.md:72` to 84);
  - the 2026-08-21 campaign with the lag pinned at 17 has one arm banned for its band-init recipe (`tasks/handoffs/2026-08-23-augmented-states-did-not-learn.md:29` to 35).

  A matched pair that changes only the state count is still open (`tasks/handoffs/2026-09-23-thesis-results-plan.md:67` to 72).
- Horizon experiments G1 to G3 are pending: g1_sim2 and g1_sim3 were killed by the RAM watchdog (`scripts/gantry/joint-estimation-horizon/REPORT.md:22` to 29, `DECISIONS.md` JH-006, JH-012). The pre-registered 0.1 s vs 1 s arms (PL:938 to 1006) have no outcome.

## 7. Achieved
- Thesis data generated and verified; the current generator reproduces saved records bitwise (C16); see `Thesis-writeup/Code/Data/GENERATION-REPORT.md`.
- Nothing on this task is decided yet. `APPROACH.md` is a paper proposal (staged estimation instead of a longer joint window), never run; `Thesis-writeup/Writing/sections/03_augmentation.tex:9` to 10, 46, 57 lists the horizons to keep apart: encoder history, scored window, joint-estimation horizon, validation horizon.

## 8. The open question
What method sets the joint-estimation window, and does it give 0.1 s, for the not-detuned and for the detuned start? Candidate answers, with what would choose between them:
- (a) **From the dynamics:** the window is a multiple of the slowest time scale the loss must see. This needs the closed-loop poles of T_AF with K1, and the literature multiples 1.5x to 4x.
- (b) **From the data:** the window is the length of the closed-loop impulse response above the noise floor. This needs the FRF records B6 to B11 and TF, and the floors (noisy minus noise-free).
- (c) **By experiment (the supervisor's suggestion):** a few training runs over the window, judged on validation with a pre-stated criterion. This needs the budget (5 to 11 GPU-h per grey-box run) and the history of failed long windows (section 6).
- (d) **Avoid the question:** keep 0.1 s and handle slow parameters by staged estimation (`APPROACH.md`). This needs the user's view and the R4 claim.

These four are candidates, not the class: the literature search (section 14) and the encoder-transient result (section 4) may add or remove methods, for example a window chosen after the encoder fix has removed the transient, or multiple shooting over linked windows. These can combine: (a) and (b) give a bound, and (c) confirms it at few points. They may give different answers per start:
- Not detuned: the window needs to span the dynamics the augmentation learns.
- Detuned: the window must also expose the parameter error. That error can sit in slow modes (the integrators, DC blindness), which is where section 4's time scales and failed long windows matter.

Also: does Coulomb, or the band moving from 140 to 230 Hz to 106 to 297 Hz, change the window of the baseline started at the correct parameters?

## 9. Next action
Open with the user: restate the two parts, ask the short questions of section 5 (Maarten's exact proposal, Coulomb in the baseline model, "R7", outline location), and start the literature search of section 14. Then present the window method as one recommendation, built from section 8, the encoder-transient result and the literature, with the computations it needs, and wait for the user's go before computing.

## 10. Acceptance criterion
- The user has agreed a window-length method and a value (or a small set to test) for each start, not detuned and detuned, including the physics-only baseline in both starts, with the reason when the two differ. Each bound comes from the system or the data, and every confirming run has a pre-stated criterion and budget.
- The outline covers every setting of section 1 for every result arm. Each entry has a value, class, source and the results it serves. There is an explicit fairness rule for the state-count comparison (R2), which says whether the encoder lag and network size scale with the states, and why, with Hoekstra's and Beintema's papers as sources where they apply.
- Format: headers plus short one-line bullets, open points marked (?).

## 11. Read these first
1. `Thesis-writeup/Documentation/RESULTS-DESIGN.md`: R1 to R8, arms, starts, budget, and the setup notes (:69, :71, :188 to 216).
2. `scripts/gantry/joint-estimation-approach/APPROACH.md` and `scripts/gantry/joint-estimation-horizon/` (`LITERATURE.md`, `DECISIONS.md`, `REPORT.md`).
3. `tasks/handoffs/2026-09-23-joint-estimation-horizon.md` and `tasks/handoffs/2026-08-31-downsampling-and-required-horizon.md`.
4. Hoekstra's papers in `literature/augmentation/` (`hoekstra2025_lfr-augmentation-ejc.pdf`, `Encoder initialisation methods in the model augmentation setting.pdf`, `Data-driven augmentation of first-principles models under constraint-free well-posedness and stability guarantees.pdf`); Beintema 2023 via `docs/references.md`.
5. `scripts/gantry/gantry_interconnect_dynamic.py` with `scripts/gantry/gantry_dynamic/config.py` (the settings of section 4).

## 12. Do not
- Launch training or numerical jobs without the user's go; never two numerical jobs at once, never MATLAB beside another session's MATLAB.
- Repeat the longer-window runs of section 6 without first stating what differs from the failed ones.
- Treat "Coulomb has no memory" or "0.1 s is enough" as settled: both are open readings.
- Edit `Thesis-writeup/Documentation/`, `Thesis-writeup/Writing/`, the data, or `kamtin-fp-model/`; read `kamtin-data/Data Telica/`; touch `kamtin-data/Telica.mat`.

## 13. Operational
- Python: `conda run -n GraduationProject python <script>`, live-output convention for anything long.
- Disk on C: fluctuates down to 0 (page file); check before any job that writes files.
- **Resources (user instruction, repeated to several sessions: "ignore the limiter"):** no RAM launch gates and no waiting for free memory; only an emergency watchdog kill at 1.0 GB available RAM. The pending horizon runs G1 were killed by a stricter watchdog (section 6). One numerical job at a time.
- If anything runs unattended: subagents in the foreground, never end a turn while a job or agent runs.
- Thesis data and the FRF evidence: section 3 and section 5 paths.

## 14. Delegation and literature
- **A literature search is part of the plan** (the user: "we can learn a lot from Jan Hoekstra's papers"; Hoekstra is the entry point into the field). Use the `deep-research` skill (D-121, step 0 FRAME mandatory), one subagent in the foreground, and extend `scripts/gantry/joint-estimation-horizon/LITERATURE.md` and `scripts/gantry/joint-estimation-approach/LITERATURE.md` rather than redoing them. Questions: how the field chooses the truncation window and its relation to the slowest closed-loop time scale; how joint estimation of physical parameters with a learned component is set up (windows, optimisers, staging); how comparisons across model orders are made fair (encoder lag, network size); sample rate and epochs or quasi-Newton stages in simulation-error training. Start from Hoekstra's and Beintema's papers in `literature/`, follow who cites them and whom they cite.
- Run the search early, alongside the opening discussion, so the window method rests on it.
- No other subagents; no Workflow tool.
