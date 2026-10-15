# Handoff: write thesis Section V (Experiment design) with Dirk, every implementation choice justified, numbers from scripts on the thesis data
**From**: session of 2026-10-14/15 (Section III-B) | **Branch**: Augmentation | **Effort suggested**: xhigh (many choices to inventory across MATLAB data generation, Python training config and decisions; numbers must come from the data actually used)

## 1. Task
Dirk's words: "This section will need to do a lot, because we will need to justify a lot of our actual implementation.
But i would still like it if we can make good use of concise writing, and use the appendix for a table like the exact
hyperparameters." And: "in this section in general we will now be able to actually use the data etc, to be able to
justify our choices."
Work with Dirk on Section V, "Experiment design" (`Thesis-writeup/Writing/sections/05_setup.tex`), using the
thesis-section skill (`.claude/skills/thesis-section/SKILL.md`) from step 1. Section V must state the data-generating
system in equations, the noise model, the controller, the records (excitation, motion types, downsampling, splits),
the compared models (including the black box) and the metrics, and justify every implementation choice concisely.
Numbers that justify a choice are computed by minimal scripts in a new folder `Thesis-writeup/Code/Setup/` from the
thesis dataset actually used, and reach the text through generated macros, never retyped. The exact hyperparameters go
to an appendix table with a basis column. Interactive: present, wait for Dirk, then act.

## 2. Out of scope
- Other sections, except a label or reference that Section V changes (list it, ask first). Section VI (Results) owns
  measured outcomes; Section V gives settings and how things are measured, not results.
- Training code changes and reruns (current-run differences are `\revnote`s, see section 4).
- Telica real data: out of the main results (Dirk 2026-09-24); never read `kamtin-data/Data Telica/` (CLAUDE.md).
- Compiling `main.tex`; git write operations.

## 3. Where things stand
Branch `Augmentation`, last commit `0246725`; many uncommitted thesis and doc edits from 2026-10-11 to 10-15 (Sections
III and IV prompts, `docs/decisions.md` D-249 to D-251, `FINAL-RUN-FIXES.md` items 7 to 10).
`05_setup.tex` (read 2026-10-15): header (budget 1.50 pages; choices to justify; cautions); visible `outline` blocks in
three subsections:
- V-A System and data: LPV baseline plus payload absorber "50 % of the head mass, 150 Hz, zeta = 0.03" plus tanh
  Coulomb rail friction, closed loop with K1; absorber and friction only in the system; output noise at the measured
  machine level, noise floor = noisy minus noise-free twin; multisine 106 to 297 Hz; motion types (standstill at fixed
  Y, slow Y sweeps, X-Y Lissajous, S-curve point-to-point, sine-shaped); splits 18 / 6 / 12 with interpolation I1 to I6
  and extrapolation E1 to E6 (Y up to 0.39 m, 100 % acceleration, 1.95 m/s, E5 changed controller "not done
  currently", E6 machine setpoint).
- V-B Compared models: LPV baseline true and detuned, LTI at Y = 0; augmented static (n_a = 0), plain, PS (n_a 2, 8),
  U vs OBC; 10 % detuned start; black box = SUBNET on the closed loop (r, f) -> e, size from 11 configurations.
- V-C Training and evaluation: closed-loop training, window 0.1 s, encoder history 29, network 2x16 tanh, one seed,
  RMS per axis in metres over whole test records, validation only for selection.
- A `\todo` defining the two OBC overlaps (moved from IV-C), and a large hidden `wip` block of stale bullets (Karnopp
  friction, "noise disabled", Telica subsection, long protocol list, reproducibility checks).
No `\revnote` in Section V yet. No `Thesis-writeup/Code/Setup/` folder yet.

## 4. Established and verified (decisions by Dirk, 2026-10-15, unless marked)
**Sources: documentation describes intent, code and data decide.** Older designs are still present in all three, so
every stated value is confirmed in the code and the data the reported runs used, never taken from a document alone:
- Documentation (`Thesis-writeup/Documentation/`): `DATA-DESIGN.md` (data design; parts superseded), `NOISE-INJECTION.md`
  (noise model, Sec. 10 current), `EXCITATION-VALIDATION.md` and `EXCITATION-REVIEW.md` (multisine band and checks),
  `BLACKBOX-METHOD.md` (comparator), `THESIS-RESULTS.md` (runs, test records, metrics; wins over `RESULTS-DESIGN.md`),
  `TRAINING-DESIGN.md` (training settings, Sec. 3 hyperparameters).
- Data generation code (`Thesis-writeup/Code/Data/`): `generate_thesis_data.m`, `tdg_config.m` (all generation settings),
  `gantrySystemExtendedTanh.m` (truth system), `tdg_make_model.m`, `tdg_truth_lin.m` (truth FRF), `gtd_make_multisine.m`,
  `gtd_make_reference.m` (motion profiles; `gtd_make_reference_telica.m` is the Telica-based variant), `tdg_noise_psd.m`,
  `tdg_noise_draw.m` (noise model), `gtd_build_plant.m`, `gtd_run_simulation.m`; with `SPEC.md`, `IMPLEMENTATION.md`,
  `REFERENCE.md`, `GENERATION-REPORT.md`.
- Training-side data handling: `scripts/gantry/gantry_dynamic/data.py` (`load_datasets`, `THESIS_DATASETS`,
  `_decimate_y`, `record_files`, trimming), `GD/config.py`, `GD/controller.py` (K1, re-discretisation), and the
  `_THESIS_FIXED` block of `scripts/gantry/gantry_interconnect_dynamic.py`.
- Datasets: `Thesis-writeup/Data/` holds seven versions (`Coulomb-and-MSD`, `Coulomb-and-MSD-noise-free`,
  `Coulomb-tanh-and-MSD`, `Coulomb-tanh-and-MSD-lowpass-noise`, `Coulomb-tanh-and-MSD-noise-free`,
  `Cubic-absorber-test`, `Detuned-with-Coulomb-and-MSD`). Which one the reported and final runs use is NOT settled:
  the loader maps `THESIS_NOISE=noisy` to `Coulomb-tanh-and-MSD` (printed by a check on 2026-10-14), while D-238 and
  `NOISE-INJECTION.md` name `Coulomb-tanh-and-MSD-lowpass-noise` (`THESIS_NOISE=lowpass`). Settle it from the runs'
  `config.json` (`data_dir`), `THESIS-RESULTS.md` and Dirk before any number is computed; all `Code/Setup` scripts read
  that dataset (and its noise-free twin for the noise floor).
**Structural model:** take the section's order from the experiment sections of the papers the thesis builds on, read in
the session: Hoekstra 2026 Sec. 6 (`literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`, with
Fig. 7, Tables 3 and 5: truth system with its parameters, one paragraph on the data, model settings, hyperparameter
table, metric displayed) and Drenth et al. 2025 Sec. 6 (`literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`). The
structure below is a candidate only.
**Splits are justified by the README's validation story** (`Thesis-writeup/Writing/README.md`, "Validation story"):
interpolation (held-out phases and positions inside the training region), operational motion, controller transfer, and
operating-range extrapolation as separate layers. Each test record set answers one of these; that, not the list of
records, is what the text states.
**Section V content agreed:**
- Structure candidate: (1) data-generating system in equations (truth = baseline + absorber + friction), noise model,
  controller; (2) records: multisine design, motion types, downsampling, splits and what each test set tests;
  (3) compared models as a table, one name per model, plus the black box; (4) metrics (RMS per axis, the noise floor as
  reference, combination error, the Section IV overlaps); (5) appendix: hyperparameter table with a basis column
  (derived / inherited / tuned / compute-limited). Dirk said he has not listed everything yet: ask what is missing.
- **Confidentiality:** the noise level and the motion are based on ASMPT's real machine and may be sensitive. State the
  noise model (enters at the position measurement, low-order coloured) and the SNR per axis computed from the records,
  with "a level representative of the industrial machine", not "derived from measured ASMPT data". Describe motions
  generically, limits relative ("75 % of the machine maximum"). Put a `\todo` asking Jasper or Dragan what may be named.
- **Multisine band:** lower edge 106 Hz because the motion profiles already excite up to about there; needs an actual
  number from the records. Placement and upper edge chosen from the FRF of the data-generating system (truth
  formulas): Dirk accepts this oracle use in simulation if stated openly ("available only in simulation"), with one
  clause that on a real machine a measured FRF plays that role. Oracle information is fine for experiment design, never
  for acceptance criteria or stopping rules (CLAUDE.md control stance 8).
- **Controller** (K1 designed at Y = 0, recorded at 20 kHz, re-discretised at 4 kHz in training; the size of that
  mismatch is an open `\todo`) and **downsampling** (20 kHz to 4 kHz, order-8 Butterworth at 1.6 kHz on y - r, D-232)
  are stated concisely with their reasons.
- **Every implementation choice gets one class** (Dirk's example: "we use tanh 1000 for the Coulomb friction, things
  like this can just be said in one sentence with justification"):
  - A: one sentence with its basis (decision, physics, convention), e.g. the tanh(1000 v) smoothing;
  - B: one sentence plus a number from a `Code/Setup` script on the thesis data (multisine lower edge, noise SNR,
    filter leak, window length against the absorber period, controller-rate mismatch, coverage of Y and acceleration);
  - C: setting only, appendix table with its basis (learning rate, batch, epochs, L-BFGS, width, depth, ...).
  In the main text justify only what an examiner would question.
- **`Thesis-writeup/Code/Setup/`** (name chosen by Dirk): one minimal script per class-B number, reading only the thesis
  dataset; writes `Thesis-writeup/Results/<name>.csv` and contributes to a generated
  `Thesis-writeup/Writing/tables/setup_numbers.tex` of `\newcommand` macros used in the text (README: numbers are never
  retyped). A short README in the folder lists per script: the choice it justifies, its input, its output.
**Method that worked in Section III (apply here):**
- Read and quote a source passage in the session before stating it; never from memory.
- Positive reasons for each choice; no argument against an alternative; no measured results in method text (results
  belong to VI), but Section V may and should use data to justify design choices.
- Text describes the FINAL runs; where current runs differ add a `\revnote` ("The current results ... The final runs
  will ... as described here."), each with a FINAL-RUN-FIXES item. Items that may touch Section V: 1 (W^b from true
  parameters), 2 (polish acceptance uses combo_err), 6 (live g_aug), 7 to 10 (encoder: window of 30 samples ending at
  tau-1, data-based physical rows, no separate bias). The encoder history in the outline (29) is outdated: the final
  window is n + 1 = 30 samples (D-250).
- Dirk writes the sentences; fixes of at most five words, or a proposal he asks for; ask "saved?" before every edit
  and tell him to run "File: Revert File" after. Plain wording, his style (see III-A); he flags AI phrasing.
- Measure changed displays against the 252 pt column in a scratch compile; the review build can be compiled to the
  scratchpad with `latexmk -norc -pdf -outdir=<scratch> -auxdir=<scratch> review.tex` from `Thesis-writeup/Writing`
  (never into `build/`).

## 5. Assumed but not verified
- Absorber parameters: the outline says 150 Hz and 50 % of the head mass; D-247 says the truth absorber pole is 212 Hz
  (`meta.truth.f_pole`). Settle from the dataset's `meta` field and the generator (`Thesis-writeup/Code/Data/
  gantrySystemExtendedTanh.m`, `generate_thesis_data.m`, `tdg_*.m`); one of them is stale (pole vs anti-resonance, or
  absorber alone vs coupled, may explain both numbers).
- The dataset actually used by the reported and final runs: see "Sources" in section 4 (loader mapping vs D-238).
- Noise model details: low-order coloured position noise of about 1e-8 m rms per axis
  (`Thesis-writeup/Documentation/NOISE-INJECTION.md` Sec. 10; D-225, D-238); verify against the generator.
- Multisine design and reason: D-219, `Thesis-writeup/Documentation/EXCITATION-VALIDATION.md`, `DATA-DESIGN.md`;
  verify what the band was chosen from, and compute the motion-profile spectrum for the lower edge.
- Black box: `Thesis-writeup/Documentation/BLACKBOX-METHOD.md`, `scripts/gantry/blackbox-cl-trainable/
  train_bb_clmap.py`; D-242, D-243.
- Splits and test records: D-221 (18/6/12), `GD/data.py::load_datasets`, `THESIS_DATASETS`; E5 status (keep as planned
  or drop: ask Dirk).
- README source-map rows "V Data-generating system", "V Data, splits, excitation", "V Controller, transfer case",
  "V Compared models, black box", "V Metrics", "V Hyperparameters" list the decisions, code and documents per part;
  check them, the code wins on conflict.

## 6. Tried and failed
- Text stated from plans or decisions without checking the code or data -> contradictions (150 vs 212 Hz is one) ->
  every stated value comes from the dataset, the generator, the config or a Code/Setup script.
- Arguing against an alternative or putting results in a method section -> Dirk rejected it in Section III.
- Editing while Dirk's buffer was unsaved -> save conflict -> ask "saved?" first.
- Bash here lost `cat`, `awk`, `tr`, `conda` mid-session -> PowerShell plus small Python scripts; keep CRLF; OneDrive
  can briefly lock a file.

## 7. Achieved
Section III-B finished and shown in the review build (2026-10-15). For Section V: the agreed structure, classes and
confidentiality rule above; nothing in `05_setup.tex` changed yet.

## 8. The open question
What else Dirk wants Section V to contain ("I have not stated everything that needs to be in the section"). Ask first,
then build the inventory.

## 9. Next action
Start the thesis-section skill at step 1 for Section V: read the README (ownership, Math standard, validation story,
source-map rows V), `05_setup.tex` in full, Sections II and III (what V gives values for), Hoekstra 2026 Sec. 6 and
Drenth Sec. 6 from their rendered pages (structure), the documentation, the data-generation code and the training-side
data handling listed under "Sources" in section 4, and the decisions named in section 5. Then give Dirk, in chat:
1. the question of section 8;
2. which dataset the reported and final runs use, with the evidence (config.json, loader, D-238), as a question if it
   is not unambiguous;
3. the inventory: every numeric and structural choice in data generation and training, each with its source in the
   CODE (file and line; a document or decision only as the reason) and its proposed class A, B or C, plus for B the
   script that would compute its number; mark where a document describes an older design than the code;
4. a section structure taken from Hoekstra Sec. 6 and Drenth Sec. 6, with the splits organised by the validation story;
5. the conflicts found (absorber values, encoder history 29 vs 30, stale wip block, E5, dataset mapping).
After Dirk confirms the classes: create `Code/Setup/`, write and run the class-B scripts (small, read-only, one at a
time, foreground, on the thesis dataset), generate `tables/setup_numbers.tex`, then continue with skill step 2a.

## 10. Acceptance criterion
Every choice in data generation and training is classified and either justified in Section V (A or B), or listed in the
appendix table with its basis (C); every class-B number in the text is a macro from `tables/setup_numbers.tex`
produced by a `Code/Setup` script on the dataset the reported runs used; nothing in V discloses what Dirk marked
sensitive; every current-run difference has a `\revnote`; the section reads as one argument (the system, the data
that tests the research question through the validation layers, the models compared, how they are judged), with the
choices grouped under it, not as a catalogue of one paragraph per choice; Dirk accepts each paragraph as his own.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/05_setup.tex` (all, including the hidden wip block).
2. `Thesis-writeup/Writing/README.md` (ownership, Math standard, validation story, source-map rows V, table pipeline).
3. `Thesis-writeup/Code/Data/tdg_config.m` and `generate_thesis_data.m` (what the data actually are), then
   `scripts/gantry/gantry_dynamic/data.py` (what training actually loads).
4. Hoekstra 2026 Sec. 6 and Drenth et al. 2025 Sec. 6 (structure of an experiment section).
5. `Thesis-writeup/Documentation/` `DATA-DESIGN.md`, `NOISE-INJECTION.md`, `EXCITATION-VALIDATION.md`,
   `BLACKBOX-METHOD.md`, `THESIS-RESULTS.md` (intent and reasons; checked against items 3).

## 12. Do not
- Retype a number in the text that a script can produce.
- Name ASMPT data, absolute machine limits or the noise spectrum as measured company data before Jasper or Dragan
  confirm; use relative or representative wording.
- Put measured model performance in Section V.
- Touch training code, compile `main.tex`, or write to `build/`.
- Edit without Dirk's request and a "saved?" check.

## 13. Operational
Python: `conda run -n GraduationProject python <script>` from PowerShell. MATLAB data files: read with scipy or h5py in
the script (check the format of the thesis `.mat` files first). Scripts in `Code/Setup` take the dataset folder as an
argument, write to `Results/` and `Writing/tables/`, and print what they wrote. PDF pages: Read tool with `pages`.

## 14. Delegation
None by default. A read-only critique session per finished subsection (Dirk runs it) has caught real errors before;
write its prompt on request, with the rule that every source and number is quoted from the file it comes from.
