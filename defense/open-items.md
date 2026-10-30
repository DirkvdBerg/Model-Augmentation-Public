# Open items

Questions I cannot answer or justify yet, and their resolution. Maintained by `defense-prep`
(section "Open items"); resolved with checked sources or stated as an honest limitation.

### O-01 [resolved] Choice of the training window length
- **Question:** why this window length, and could the encoder reset per window hide poor long-term
  behaviour on X and Y? (cold round 2026-10-29, question 3, Hoekstra)
- **Missing:** a clean justification of the chosen length; I know the benefits (training time,
  convergence), that it was designed around the missing absorber frequency, and that longer windows
  did not improve the result.
- **Possible sources:** Hoekstra 2026 LFR and Beintema 2023 SUBNET (why truncated windows, how T is
  chosen); `docs/decisions.md` and the run table (`docs/gantry-augmentation-problem-log.md`, Section 12)
  for the window-length runs; `Thesis-writeup/Writing/sections/99_appendix.tex` (hyperparameters).
- **Resolution (2026-10-29):** the length is argued from the SUBNET rule and the closed-loop memory;
  the long-term worry is answered by full-record free runs. One part of my own recollection is not
  supported (see limitation).
  - Value: $n_f=400$ samples = 0.1 s (E-030; `Clean/gantry_dynamic/config.py:98`, `nf_seconds`
    0.100, $n_f$ = `nf_seconds / ts_new` at line 127). Records are 12 s, so 120 windows each (E-031).
  - Why short windows at all: smoother loss (thesis, E-029) and serial cost $O(T)$ instead of $O(N)$
    (Beintema Sec. 3.1, E-028).
  - Lower bound: SUBNET overfits when $T$ is below the dominant timescale, and a larger $T$ costs
    compute (Beintema Sec. 5.4, E-027). Because the loop is simulated, the relevant timescale is the
    closed-loop memory, how long a force residual stays visible in the servo error through $PS$: the
    thesis setup code computes a largest 1 % decay of 44 ms and a slowest closed-loop time constant of
    11 ms (E-033, E-034; computed, not printed in the thesis). 0.1 s covers that about twice. Project
    record: D-220 (`docs/decisions.md:6914`), which also gives the absorber mode in the loop
    (tau 4.7 ms).
  - Long-term behaviour: the reset is a training device only. The checkpoint is chosen on a free run
    of the full validation records, encoder used once at the start (E-032), and every reported model
    is simulated over the full 12 s records (E-004). R1 on the test set: NRMS$_e$ from 0.025 (E3, Y)
    to 0.136 (E1, X1), rounded (E-037, E-036).
  - Limitation (honest): no longer window was trained in the final campaign. The planned
    0.1/0.2/0.4 s confirmation sweep (D-220 plan, run table row "TIER 4", outcome "pending") was dropped
    when the run set was cut to 8 runs (D-242, `docs/decisions.md:7712`). "Longer windows did not
    improve the result" has no source I could find: do not say it.
  - My view: in closed loop, $K_1$ also corrects drift of the free X and Y integrators in the model,
    so a closed-loop free run cannot show open-loop drift; that is the masking point of Q-004, not a
    window effect.
- **30-second answer:** "Short windows make the loss smoother and the cost linear in the window, as in
  SUBNET. Beintema shows the window must not be shorter than the dominant timescale. Since the loop is
  simulated, that is the closed-loop memory: a force error fades from the servo error within about
  44 ms, so 0.1 s covers it twice. The reset per window only affects training: I pick the checkpoint on
  free runs of the full validation records, and all results are free runs over full 12 s records,
  started once by the encoder. I did not run a window sweep; that was cut for compute."
- **Practice:** Q-003 (Hoekstra). First challenge to prepare: "then why not 0.05 s?" (answer:
  44 ms is the 1 % decay, 0.05 s leaves no margin and the start of each window is partly blind).

### O-02 [resolved] Sampling rate of training and validation (4 or 20 kHz)
- **Question:** at what sampling rate did you run training, validation and test, and what rate does the
  controller run at? (practical fact; Gerritsen will ask)
- **Missing:** the thesis source has the note "checken of validatie in 4 of 20khz is"
  (`Thesis-writeup/Writing/sections/03_augmentation.tex:887`), and line 930 mentions different controller
  rates. The final values must be settled from the code and data, not from memory.
- **Possible sources:** `Clean/gantry_dynamic/config.py` and the final run configs; `05_setup.tex` and
  `99_appendix.tex` ($f_{\mathrm{sim}}$, $T_s$); `docs/decisions.md`; the final CSVs in `Results/24-10-2026/CSV`.
- **Resolution (2026-10-29):** train, validation and test all run at 4 kHz ($T_s=0.25$ ms). The three
  sources agree.
  - Thesis: data simulated with RK4 at $f_{\mathrm{sim}}=20$ kHz (E-014); all records downsampled to
    $T_s$ (E-015): servo error zero-phase Butterworth at 1.6 kHz, force block-averaged (E-017). The data
    loop ($K_1$ Tustin at 20 kHz) differs from the model loop ($K_{\mathrm{fb}}$ = $K_1$ Tustin at
    $T_s$) (E-016, E-018).
  - Code: `Clean/gantry_dynamic/config.py:95-96` (`fs_orig` 20000, `fs_new` 4000); every record is built
    with `dt=cfg.ts_new` (`data.py:146`) and train, val and test go through the same loader
    (`data.py:183`); the final evaluation calls that loader (`Thesis-writeup/Code/Evaluation/eval_greybox.py:63`)
    and builds the controller at `cfg.ts_new` (`controller.py:224`), Tustin (`controller.py:111`).
  - Results: the final evaluation reproduces the checkpoint validation NRMS of every run (R1 0.077464
    both, E-019), so evaluation and training-time validation use the same 4 kHz data.
  - Rate mismatch size: $K_1$ at 4 kHz vs recorded feedback force, median 0.5 %, max 1.0 % relative RMS
    (E-020, E-021); largest phase-margin loss 4 vs 20 kHz is 3.8 deg (E-022).
  - The revnote at `03_augmentation.tex:887` ("validatie op 20 kHz onlogisch") is answered: validation
    is not at 20 kHz. Stale revnote; removing it is the user's call.
- **30-second answer:** "The data are simulated at 20 kHz, with the controller also at 20 kHz. All
  records, train, validation and test, are reduced to 4 kHz: the servo error is low-pass filtered at
  1.6 kHz forward and backward, the force is averaged per sample. Everything the model sees, trains on
  and is scored on runs at 4 kHz, with the controller re-discretised at 4 kHz. The mismatch to the 20 kHz
  data loop is small: median 0.5 % in feedback force, at most 3.8 degrees phase-margin loss."
- **Practice:** Q-008 (fact).

### O-03 [resolved] Why Tustin and not ZOH for the controller
- **Question:** Q-007. The controller K1 is discretised with the Tustin rule (`05_setup.tex:230`); the
  forces are held with zero-order hold (`03_augmentation.tex:453`).
- **Missing:** a checked justification. Own revnote (`03_augmentation.tex:883-885`): ZOH would be better,
  Tustin suits frequency-domain work, and Jasper thinks ZOH is the right choice. A source comment says the
  choice is argued in the Discussion, but no live text there mentions Tustin.
- **Possible sources:** a discretisation reference in `literature/` (for example
  `literature/sampling-rate/oomen-2018_advanced-motion-control-precision-mechatronics.pdf`,
  `literature/Classes/4CM60 - Advanced motion control/`); `docs/decisions.md`.
- **Resolution (2026-10-29):** the thesis states the choice but does not argue it. The checked
  reasons, and the part that stays a limitation:
  - Where Tustin comes from: the data generator calls Jasper's `ruleOfThumb` unchanged
    (`Clean/data_generation/tdg_config.m:4`, `gtd_build_plant.m:15`), which discretises with
    `c2d(C, ts, 'tustin')` (`Thesis-writeup/Code/Data/vendor/ruleOfThumb.m:14`) at 20 kHz
    (`tdg_config.m:50`). The model's $K_{\mathrm{fb}}$ uses the same rule at $T_s$
    (`Clean/gantry_dynamic/controller.py:111`; thesis E-013, E-018). Data loop and model loop
    therefore use the same rule; only the rate differs (median force error 0.5 %, max 1.0 %; PM loss
    at most 3.8 deg; E-020, E-021, E-022).
  - Literature: turning a continuous-time controller design into a discrete one is usually done with
    the bilinear (Tustin) transform (Skogestad and Postlethwaite, 2nd ed., Sec. 9.4.5, E-024). The
    trapezoidal rule preserves stability of a stable system for any step (Toth 2010, p. 160, E-026).
    ZOH (the complete method) is exact only when the input is held over the sample (Toth 2010,
    p. 152, E-023).
  - Where ZOH is used: the plant side, where the force really is held (RK4 with the force held, E-025).
    For the controller the input is the sampled servo error, not a held signal, so neither rule is
    exact for $K_1$; the discrete controller is what it is, and what matters is that the model uses
    the same one as the data.
  - Limitation (honest): no run compares Tustin with ZOH; whether ZOH would change a result is not
    tested. Jasper's view (revnote `03_augmentation.tex:885`) is that ZOH is the right choice. For the
    real machine, the model must use the discretisation the machine's controller uses (my view).
- **30-second answer:** "I used the controller design function from ASMPT unchanged, and it discretises
  with Tustin, which is the usual way to implement a continuous-time controller design and keeps it
  stable. The model uses the same rule, so the controller in my model matches the one that generated
  the data, only at 4 instead of 20 kHz. ZOH is exact where the input is really held, which is the
  force on the plant, and there I do use it. I did not test ZOH for the controller; on the machine I
  would match whatever its firmware uses."
- **Practice:** Q-007 (considered, Gerritsen). Careful in tone: the Tustin rule is in Jasper's own
  function, and Jasper now favours ZOH; say it as a fact, not as a counter.

### O-04 [open] Black box and a changed controller
- **Question:** could your black-box model evaluate a changed controller, and is it a fair reference for
  the abstract's use case? (practice 2026-10-29, Q-005 answer; possible subquestion)
- **Missing:** how the black box is trained and simulated (in the loop with $K_{\mathrm{fb}}$ or not),
  and whether it could be rerun with another controller; checked in thesis, code and final CSVs.
- **Possible sources:** `05_setup.tex` (black-box model), `Thesis-writeup/Code/Evaluation/`, final CSVs.

### O-05 [open] Tustin versus ZOH error not quantified
- **Question:** how large is the difference between a Tustin and a ZOH discretisation of $K_1$ at
  $T_s$, and does it change the feedback force? (practice 2026-10-29, Q-007; Gerritsen likely asks)
- **Missing:** no computation exists (O-03 limitation). A comparison would be a small numerical check,
  only on the user's go-ahead.
- **Possible sources:** `Clean/gantry_dynamic/controller.py:111`, the setup check behind E-020.

### O-06 [open] Well-posedness of the augmented model only in a \todo
- **Question:** is the augmented model well posed for every state, input and Y? (Q-013, Toth likely asks)
- **Missing:** the argument (acyclic interconnection, smooth blocks, Hoekstra Cond. 6, Cond. 8, Thm. 9) is in a `\todo` at `03_augmentation.tex:459`, not in the final text; Hoekstra's conditions not yet checked on the PDF page.
- **Possible sources:** `hoekstra2026lfr` (Cond. 6, 8, Thm. 9), `02_system_baseline.tex:843` to `856` (baseline admissibility), `Clean/` model code.

### O-07 [open] What a Lur'e structure is, and why not one for friction
- **Question:** why not a Lur'e structure with an explicit friction block in feedback? (Q-021, Shakib; practice 2026-10-30: "I don't know what Lur'e is")
- **Found so far:** LTI block plus static nonlinearity in a feedback loop (Schoukens and Tiels, E-088). Hoekstra 2026 cites Shakib et al. 2022 on Lur'e-type identification with stability guarantees as an LFR for nonlinear models (E-086, E-087): Shakib's own work.
- **Missing:** Shakib 2022 is not in `literature/`; what his method does cannot be stated until it is opened locally.
- **Possible sources:** Shakib et al., Automatica 136:110012 (2022), to download by the user.
