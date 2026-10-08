# Handoff: a full black-box model that demonstrably trains in closed loop, without a BLA start and without plant information
**From**: session of 2026-09-29 | **Branch**: Augmentation | **Effort suggested**: xhigh (a diagnosis of why training fails, a literature search, implementation, and trainability runs with pre-stated criteria)

## 0. How to read this prompt (read first)
- **The goal, in the user's words (spelling cleaned):** "create the closed-loop black-box implementation that could actually train without BLA or plant info, and refer it to Jan Hoekstra's black-box example and deepSI-master." And earlier: "important that we train in closed loop and that it has actually proven that it could be trained, so that it is worthy of a hyperparameter search."
- **Task type: implementation plus proof.** The deliverable is working code and evidence that it trains, not a plan. A report that says what could work, without a training curve that shows it, does not meet the goal.
- **An example is a sample, never the scope.** Jan Hoekstra's black-box example and the deepSI-master notebooks are entry points into the line of work on neural state-space (SUBNET) models, not the only sources and not templates to copy. The candidate remedies in section 8 are candidates; the class of ways to make a black box trainable inside a closed loop is wider.
- **Diagnose before fixing.** A previous session found why this fails (section 4). Choose remedies because they address that mechanism, and state which part of it each one addresses.
- **Decisions and readings are separate.** Section 4 lists what is fixed. Where this prompt gives its own interpretation it is marked **(reading)**: say so when you use it, and ask the user once where it matters.

## 1. Task
Build a full black-box state-space model (encoder, state map `f`, output map `h`, all learned; Jan's ANN-SS / deepSI SUBNET structure) that is trained and selected in the same closed loop as the grey box, on the thesis data, starting from an initialisation that uses neither a BLA nor any information about the true plant. Then prove that it trains: closed-loop training runs whose validation error falls clearly and reproducibly from its starting value and stays finite, judged by a criterion written down before the runs. End with a model and settings that are ready for a hyperparameter search, and the server runners for it.

## 2. Out of scope
- The hyperparameter search itself (the next session); here only the settings and ranges it should search, with reasons.
- Any change to the grey box, the training pipeline in place, or the data. Import the pipeline read-only; new code lives in `scripts/gantry/blackbox-cl-trainable/` (create it).
- Oracle information of any kind (section 4, fairness).
- The old `scripts/gantry/blackbox-closed-loop/vendor/`: copies of the old pipeline without the thesis modes; do not build on it.
- BB-LTI (the linear black-box row of S5): not built here; one line in the report on whether the remedy carries over.

## 3. Where things stand
- Branch `Augmentation`, tree dirty with other sessions' work (do not commit or clean).
- **Data (use only these):** `Thesis-writeup/Data/Coulomb-tanh-and-MSD/` (noisy, encoder noise, D-225) and `Thesis-writeup/Data/Coulomb-tanh-and-MSD-noise-free/` (D-230). 48 records per version; the split is in each folder's `MANIFEST.csv`, column `block`: 18 training, 6 validation, test I1 to I6 and E1 to E6, plus the FRF references TF-1 to TF-3 (test only). Not these: `Coulomb-and-MSD[-noise-free]` (superseded Karnopp / force-noise set), `Detuned-with-Coulomb-and-MSD`, and anything under `data/gantry/matlab/trajectory/` (legacy; the old black box used one of those).
- **Truth T_AF:** physics baseline + tanh Coulomb rail friction (g = 1000, D-224, D-226) + payload absorber (ma 0.5 mh, 150 Hz, zeta 0.03). One controller K1, designed at Y = 0, for every record; E5 alone uses K2 = 1.2 K1.
- **Loading:** the current pipeline, `scripts/gantry/gantry_dynamic/data.py`, modes `thesis_taf_noisy` and `thesis_taf_noisefree` (D-221): manifest record lists, K1 from each record's meta, 20 to 4 kHz, the D-232 anti-alias decimation of `y` on the noisy set, 5 ms trimmed at both record ends. Where a document still calls the noise a force disturbance (D-218, including a message in `data.py`), D-225 supersedes it.
- **The closed-loop machinery:** `model_augmentation/fit_systems/closed_loop.py` and `scripts/gantry/gantry_dynamic/controller.py`, used by the grey box through `scripts/gantry/gantry_interconnect_dynamic.py`.

## 4. Established
**Fixed decisions (the user, and the design documents):**
- **The arm (RESULTS-DESIGN S5, BB-NL):** "nonlinear black-box state-space model, same encoder, same closed-loop training"; network size and hyperparameters chosen by a search on validation, no size limit (Quinten). **A black-box start is a seed** (user, 2026-09-27): no BLA, no identified start.
- **Closed loop:** the training loss and the model selection both run through the known controller K1 in the grey box's residual form, as for the grey box.
- **Shared training settings** (`TRAINING-DESIGN.md` section 3, user 2026-09-27): window 0.1 s (`nf = 400` at 4 kHz, D-220); encoder lag 29; normalisation zero mean and unit std from the training records; float64; training on the noisy records, each run paired with a noise-free twin of the same seed. The grey box's Adam lr 1e-5 and eps 1e-16 and its 16 x 2 tanh network are grey-box settings, not black-box ones: the black-box search space is deferred (user 2026-09-27), so this session proposes it. Any other deviation from the grey box's training is argued in `DECISIONS.md` and reported.
- **Fairness (R6, R8):** the black box gets exactly the grey box's information. Fit and initialise on the 18 training records only; validation only for selection and the hyperparameter search. Not allowed: plant eigenvalues or the truth model, the TF records, separate experiments (such as the frictionless `joint_lowf_ma50_a5` experiment the old FRF start came from), and the true model order: the old black box's `nx = 8` came from the truth's four degrees of freedom including the absorber, which is plant information. The state dimension is a hyperparameter chosen on validation or by a rule that does not use the true order.
- **R8 measures the effort** (RESULTS-DESIGN R8): GPU-h to convergence, the size of the search needed, the spread over starts and the fraction of diverged or failed runs. A diverged rollout is a failure of that arm on that record (S6). "If the search for the black box is hard, that is stated as a result" (Quinten). So log every configuration tried with its cost and outcome; do not hide failures behind the one setting that works.
- **(reading)** A BLA identified from the 18 training records only (M. Schoukens 2021 ECC; Ramkannan et al. 2023, which another session called the legitimate route) conflicts with "a start is a seed". Use it only as a fallback comparison arm after the seed arms are done and shown untrainable, never as the answer; if the user can be asked, ask once, otherwise build it and label it as outside S5.

**What the previous black-box session found** (`scripts/gantry/blackbox-closed-loop/REPORT.md`, `DECISIONS.md` BB-001 to BB-013; old dataset, so recheck on the thesis data):
- The black box can run in the grey box's closed-loop objective exactly (adapter verified against the grey box to machine precision; `blackbox-closed-loop/adapter/`).
- Random initialisation at 4 kHz: the closed loop diverges on 6 of 6 validation records (NaN loss and gradients), G3.
- Four open-loop-trained checkpoints (random, N4SID BLA, FRF BLA, oracle start; old 800 Hz data, G2) are all unstable in closed loop, although each is stable and reasonable open loop: an open-loop fit says nothing about stability under the controller.
- **Mechanism (G4):** the controller, folded into normalised coordinates, has a feedthrough of about 5.9e3. Around the one stable start, a random change of 1e-5 in the linear weights of `f` and `h` destabilises 76 to 89 % of draws, and one Adam step at Jan's lr 1e-3 destabilised the loop. The grey box avoids this because its physics fixes the plant the controller acts on and its network starts at zero.
- The only stable start was an FRF BLA from a separate experiment, which needed lr 1e-7; it never ran on the server. Its window loss was dominated by the random encoder's initial state (compare the encoder-transient finding in `TRAINING-DESIGN.md` section 3, Encoder).

**The references** (read the code, not summaries):
- Jan's black-box example: `scripts/ecc_2025/msd_ndof_deepSI_encoder.py` (gantry copy `scripts/gantry/msd_ndof_deepSI_encoder.py`): random init, open loop, `nf = 200`, batch 2000, 10 000 epochs, `f` and `h` 2 x 8, encoder 2 x 16, lr 1e-3.
- deepSI-master: `deepSI-master/examples/` (`1. Overview deepSI.ipynb`, `2. pHNN SUBNET demo.ipynb`, `3. LPV SUBNET.ipynb`); these are deepSI v0.4.4, while the installed library is 0.3.29 (`SS_encoder_general_hf`), per `scripts/gantry/full-blackbox/README.md`. Horizons there are 20 to 60 samples, all open loop.
- Neither reference trains in closed loop, so there is no recipe to copy: the closed-loop part is this session's contribution.

## 5. Assumed but not verified
- That the mechanism above (normalised feedthrough, fragility of the random start) holds on the thesis data with K1 at 4 kHz; the previous numbers are on an older dataset and controller bank. Re-measure it first.
- That "trains" can be shown locally at small scale before a server run. Short local runs are allowed (section 13); if the effect only shows at full scale, say so and prepare the server run.

## 6. Tried and failed
- Random start in closed loop -> NaN on 6/6 records -> the loop is unstable around a random plant (section 4).
- Open-loop training first, then closed loop -> every open-loop checkpoint was unstable under the controller -> open-loop quality does not transfer to closed-loop stability.
- Jan's lr 1e-3 in closed loop -> one step destabilised the loop -> the step size was cut to 1e-7, where learning is doubtful.
- N4SID BLA at 4 kHz -> not constructible (every candidate rejected).
- Building on `blackbox-closed-loop/vendor/` -> old pipeline, wrong data; do not repeat.

## 7. Achieved
The closed-loop adapter (exact), the failure mechanism on the old data, and the reference code. No black box has been shown to train in closed loop.

## 8. The open question
What makes a black box trainable inside this closed loop from a start that carries no plant information? Candidates, to be weighed against the mechanism and the literature, not a closed list:
- a stable-by-construction parameterisation of the learned dynamics (the literature on unconstrained parameterisations with stability guarantees, e.g. `literature/Orthogonality/kon2026_unconstrained-parametrizations-stability-dissipativity_IEEE-TAC.pdf`);
- an initialisation that makes the learned plant start inside the controller's stability region without using plant data beyond the training records (for example a near-zero or strongly damped start, or a start scaled to the training data's own statistics);
- a scaling that removes the huge feedthrough of the folded controller (a deviation from the shared normalisation, so argue it);
- a curriculum (short to long windows, or small to full controller gain), gradient clipping, or separate step sizes for the linear part;
- multiple shooting or other objectives that keep gradients finite when a window diverges.
Each kept or rejected with the reason and the source; combinations are allowed.

## 9. Next action: phases, in this order
**Phase A, literature** (`deep-research` skill, step 0 FRAME mandatory; one subagent, in the foreground). How neural state-space and SUBNET-type models are trained in closed loop or on closed-loop data; stable parameterisations and initialisations of learned dynamics; how the stability of a learned plant under a known controller is kept during training. Start from Hoekstra's and Beintema's work and deepSI, follow citations both ways. Write `LITERATURE.md` with the Research Log.

**Phase B, diagnosis on the thesis data.** Re-measure the mechanism of section 4 with K1 at 4 kHz on the thesis data: stability of the loop around the random start, the folded feedthrough, the fraction of small perturbations that destabilise it. Write the prediction first.

**Phase C, implement.** Pick the remedy (or combination) the diagnosis and literature support; implement it in the new folder, importing the pipeline read-only. Record every decision in `DECISIONS.md` before implementing.

**Phase D, prove trainability.** Pre-register the criterion in `DECISIONS.md` before any run, in the S6 metric (validation servo-error RMS per axis, nm), for example: below its starting value by a stated factor, no diverged rollout at any validation, over at least three seeds, on both the noisy and the noise-free records. Run it; if the local scale cannot show it, prepare the server run with the same criterion. Report the curves, and every configuration tried with its cost (R8).

**Phase E, ready for the search.** List the hyperparameters the search should cover (including the state dimension, since the true order is not allowed), with ranges and reasons, and prepare the server runners (not submitted).

**Phase F, critic.** One subagent reads the result cold as a reviewer: does the implementation use any plant information, does the evidence show training rather than luck, is anything in the class of remedies missing. Save `CRITIC.md`, revise, list how each point was handled.

## 10. Acceptance criterion
Done when `scripts/gantry/blackbox-cl-trainable/` holds:
- `REPORT.md`: the diagnosis on the thesis data; the remedy and why it addresses the mechanism; the trainability evidence (curves, seeds, both data versions) against the pre-registered criterion, PASS or FAIL; the hyperparameter list for the search.
- `IMPLEMENTATION.md`: what was built, file by file, how it differs from Jan's example and from the old black box, and how to run it.
- `LITERATURE.md`, `DECISIONS.md`, `RUNS.md`, `CRITIC.md`, and server runners that are ready but not submitted.
- An explicit statement that no plant information entered: initialisation, state dimension, normalisation and data all traced to the 18 training records or to choices independent of the plant.
If trainability without a BLA cannot be shown, say so with the mechanism, and show the fallback arm of section 4 against the same criterion.
Format of the reports: headers plus short one-line bullets, open points marked `(?)`.

## 11. Read these first
1. `scripts/gantry/blackbox-closed-loop/REPORT.md` and `DECISIONS.md`: what failed and why.
2. `scripts/ecc_2025/msd_ndof_deepSI_encoder.py` and `deepSI-master/examples/`: the references.
3. `Thesis-writeup/Documentation/RESULTS-DESIGN.md` (R6, R8, S5), `TRAINING-DESIGN.md`, `DATA-DESIGN.md`; `Thesis-writeup/Code/Data/GENERATION-REPORT.md`.
4. `docs/decisions.md` D-221, D-224, D-225, D-226, D-230, D-232.
5. `scripts/gantry/gantry_dynamic/data.py`, `controller.py`, and `model_augmentation/fit_systems/closed_loop.py`.

## 12. Do not
- Use plant eigenvalues, the truth model, the true model order, the TF records, validation data for fitting, or separate experiments.
- Build on `blackbox-closed-loop/vendor/` or the legacy datasets.
- Edit the main pipeline, the data, or `Thesis-writeup/`.
- Read `kamtin-data/Data Telica/` with file tools; touch `kamtin-data/Telica.mat`.

## 13. Operational
- Env `GraduationProject`, live-output convention.
- **Local runs are allowed** for diagnosis and short trainability runs; the model is small. One job at a time.
- **Resources (user instruction, repeated to several sessions: "ignore the limiter"):** no RAM launch gates, only an emergency watchdog kill at 1.0 GB available RAM (copy `telica-real/tools/watchdog.ps1` and set that level). Check free disk before writing large outputs; C: has run nearly full.
- If run unattended: subagents in the foreground; never end a turn while a job or agent runs; poll long jobs with short foreground checks.

## 14. Delegation
Two subagents at most: phase A literature and phase F critic, both in the foreground. No Workflow tool.
