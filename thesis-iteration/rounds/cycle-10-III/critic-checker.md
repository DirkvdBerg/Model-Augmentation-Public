# Critic (checker lens), cycle 10, Section III

Score: 8 / 10

## What was checked
- Compile: `latexmk -pdf main.tex` in `thesis-iteration/Writing` passed. `build/main.log` has no line starting with "!" and no overfull box on Section III lines. The undefined `eq:obc_overlap` and `eq:lfr_chain` belong to Sections IV and the appendix.
- Code, read this session: ENT `_THESIS_FIXED` (route all 6+n_a rows, 16x2 tanh, lag 29, closed loop, L-BFGS, Xavier W^a). Module CFG `up_sample=1`, so one RK4 step per sample. GD/data.py::compute_normalization (training records, q = P^-T y, np.diff velocity, centred and std-scaled; Cd_norm uses no mean). GD/model.py::get_encoder_dims (na_right = 1, so the history includes sample k) and the encoder linearised at Y = 0 with nominal module constants. FS/pre_encoder.py::linear_encoder_init_aug (W^b formula, Xavier on one [W_y W_u] with fan_in 6(n+1), offsets on both W^b and W^a, x_off on the physical rows only, zero psi-tilde output layer). FS/blocks.py (`combinations_from_free`, `_m_diff_bound` with rho = 0.99, eta0 and s, `_rk4`). FS/closed_loop.py (residual form, step order, x^c = 0, record-length-weighted validation RMS). FS/lbfgs_polish.py::decide and meter_regression. GD/training.py::_gantry_meters (combo_err; m_diff scaled by 0.5(m1+m2)). FS/ps2_opening.py (PS2Spec, after_step, the 4- and 5-check plateaus, the screen, S1 head and trained rows, S2 pairs and best-state restore).
- Own algebra rechecked: residual subtraction; folding the normalisation into the controller matrices (centring cancels in r - y); C^n centring cancellation; admissibility of eq:mdelta_map; the zero-start gradient claim; and the W^b block order against the code.
- Citations: every cited location is in citation-log.md. Hoekstra 2026 Sec. 6.3 was reread (p. 11): it uses feedforward tanh learning components and gives no reason that would conflict with the draft.
- Results pointers checked against DOC/THESIS-RESULTS.md: steps 2, 3 and 6 and Sec. 5.

The draft is coherent. It has a roadmap, five one-job subsections, and problem, construction, consequence in each. The identification problem display is complete, normalisation and initialisation are displayed, and no em-dashes appear. Almost every mechanism matches the code. The issues below are what remains.

## Issues

1. MAJOR (F, D). Line 618: "As in [Eq. (22)] hoekstra2026lfr, the criterion therefore scores every sample of such windows". The thesis encoder history ends at tau (eq:aug_encoder, code na_right = 1), so the first scored output yhat_tau|tau comes from a state reconstructed with y_tau itself. Hoekstra's Eq. (22e) reads past samples only (k-n to k-1, recorded in citation-log.md). The criterion is therefore adapted, not adopted, and the difference goes unstated. It surfaces only as a figure-label todo. Rule cause: SKILL "Contributions" (an adapted part says what differs) did not prevent this. The review check "every citation supports the exact sentence" was not applied to index ranges. Missing rule: when two cited sources with different conventions feed one display (encoder paper current sample, Hoekstra 2026 past only), state which convention the display follows and how it differs from the other.

2. minor (F). Lines 678 to 681 argue that a closed-loop fit can mask plant error, "Section VI therefore also evaluates the final model under a changed controller". README "Validation story" item 3 and MUST ESTABLISH item 16 keep the plant-domain check (the masking test) separate from controller transfer. THESIS-RESULTS plans E5 as the controller use case and the plant check as open ("(?)"). "The final model" is also a term this section never defines. Rule cause: SKILL "Reasons and claims" (a Results pointer names only what THESIS-RESULTS plans, in its terms) and "Known before new".

3. minor (B). r, u^ff and the physical yhat (line 687, "Its output is the predicted position yhat") are used without a space or a definition. Only yhat^n is defined. Rule cause: README Math standard item 1 (every signal defined with its space at first use).

4. minor (C). eq:aug_enc_init displays the explicit Glorot bound sqrt(6/(n_xbar+6 n_psi+6)). That is textbook machinery: "W^a Xavier uniform [Eq. (31)]" carries the same content. Rule cause: missing rule. README Math item 4 says initialisation is displayed, and SKILL "Math" says standard schemes are named rather than displayed, but neither rule says whether a standard initialisation distribution is displayed or named.

5. minor (D, E). Two sentences restate a display:
   - Line 451, "The model thus receives the recorded input plus the controller's response to its own output error, without r or u^ff", reads eq:aug_residual_loop back.
   - Line 555, "The encoder starts from the baseline's linearisation and a random added-state map and is trained with the model", repeats eq:aug_enc_init and III-D.

   Line 665 also opens with "From the zero start", which ties the non-uniqueness to the start rather than to the additive structure. Rule cause: SKILL "Equations carry the content" and "Say each fact once" did not prevent these.

6. minor (D). Line 558 gives "we compute W^b directly, since the linearised baseline is known" as a reason, and the todo at line 569 then asks why the model-based rather than the data-based initialisation was chosen. The reason is a non-reason (it names nothing the alternative loses), and it sits beside a todo for the same reason. Rule cause: SKILL "Justify only what an examiner would question" and "Reasons and claims" (a reason that is open is not also stated as a reason clause).

7. minor (F). Line 564 has "W^b uses the nominal parameters also when theta_base starts elsewhere (Section III-D)". III-D only defines theta_base,0 as "the start" and never says it can differ from nominal. The detuned start belongs to Section V. Rule cause: SKILL "Known before new" and the one-pointer rule.

8. minor (E, F). The PS2 text names the withheld (calibration) records three times. It also adds "S2 keeps the network parameters with the lowest evaluated value", which is calibration bookkeeping. That sentence is inexact: if no check beats the initial value, `_fit_map` keeps the final parameters. It also takes the non-rule sentence count past 1.5 times the sketch. Rule cause: conflicting rules. The README PS2 sketch excludes "calibration bookkeeping (withheld records)" and README "Prose around the math" excludes fallbacks. SKILL "Content" requires every compared quantity of a switching rule to be named, and the sketch calls the screen method-defining. Writer note 4 flagged the same conflict.

9. minor (B, F). eq:ps2_s1 minimises over theta^a_enc with e_tau depending on xhat_tau|tau, so as displayed the gradient would pass through the target. The code computes the residual under no_grad from a detached state, so e_tau is a fixed target at the current parameters. The where clause's "current model" only hints at this. Rule cause: conflicting rules. The README PS2 sketch excludes "detached states" as gradient mechanics, while README Math standard requires the display to show what is trained against which cost.

10. minor (E). The fig:aug_structure caption joins clauses with their own subjects by a semicolon and an "and" ("... act in parallel on xhat_k; phi_aug supplies ..., and the output map is not learned"). The last caption sentence is a comma splice. Rule cause: SKILL packing pass (captions included) was not applied.

11. minor (E). Three acronym and naming lapses:
    - "dynamic-parallel (S-DP)" leaves the S (state augmentation, Hoekstra Table 1) unexpanded.
    - RK4 is named without a source.
    - The zero-order hold is named without a source.

    Rule cause: SKILL "Math" ("a standard numerical scheme is named with its order and source") gives no fallback when no verified source exists (writer note 11). The acronym check covered first use but not correct expansion.

12. minor (E). The "VERIFY keys" todo (line 216) states a resolution ("now agree"), and neither it nor the "VERIFY bib" todo uses the Missing / Candidate form. Rule cause: SKILL "Open points are \todo{}s" (form, and only open items) has no rule for kept legacy todos that are already resolved.

13. minor (F). The header's stale-sibling list omits 04_obc.tex eq:obc_loss. Its changed line writes u^n_k inside f_base and f_aug, while III's eq:aug_loss now feeds the closed-loop input uhat^n_k. The cited unchanged "input line" therefore conflicts with the changed line. Rule cause: SKILL "Open points" (name every stale line of another file in the header and the report). The review list has no check that reruns sibling displays citing this section's labels.

14. minor (A, notation). Three gaps:
    - MUST ESTABLISH 1b is only partly covered. The draft never says that the baseline block is itself an LFR inside the LFR augmentation, which is what ties the title "Dynamic LPV-LFR Augmentation" to the content.
    - eq:mdelta_map writes the index condition as the value condition "theta_base,j != m_Delta".
    - The initial state is written xhat^n_tau|tau in eq:aug_encoder and in PS2, but xhat^n_tau in eq:aug_loss.

    Rule cause: SKILL review "every MUST ESTABLISH claim honoured" and "Notation" (one symbol, one form) did not prevent these.

15. minor (C). The section runs about 3.4 pages against a budget of 1.55 (2.2 times). The writer followed SKILL "Length" and proposed 3.3 pages. The required displays justify most of the length, but items 4, 5 and 8 are cuttable. Rule cause: the README ownership table moved PS2 and the training coordinates into III without updating the header budget. SKILL "Length" lets a stale budget persist until a writer proposes a new one.
