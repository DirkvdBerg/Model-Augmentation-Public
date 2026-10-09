# Critic (checker lens), cycle 01, Section III

Score: 6 / 10

Lens: meticulous co-author (E, F, G, with B and C). Checked against the code on the thesis path
(`FS/ps2_opening.py`, `FS/closed_loop.py::closed_loop_rollout`, `::validation_error`,
`::closed_loop_free_run_rms_batch`, `FS/lbfgs_polish.py`, `GD/data.py::compute_normalization`,
`FS/pre_encoder.py::linear_encoder_init_aug`, `ENT` thesis block `wa_encoder_init='xavier_uniform_gain1'`),
`citation-log.md`, and a compile of the sandbox `Writing/` with the draft as Section III (scratch copy).

## What holds

- Code agreement is good. Confirmed: f_aug added after the RK4 step; normalisation from training records
  with q = P^-T y and first differences; W^b reconstructability map; W^a Xavier (gain 1); x^c = 0 per
  window and the step order; record-length-weighted mean of per-record RMS in metres with k0 = n;
  Adam then L-BFGS (strong Wolfe) with acceptance on validation plus meters; PS2 M = na + na_right = n+1,
  batch-RMS normalised residual, calibration split, S2 frozen one-sample pairs normalised by target
  variance, only g_aug rows receive a gradient in S2.
- Identification problem displayed with its constraints (eq:aug_loss), zero init, normalisation and
  W^b displayed (criterion B met for III-A to III-C).
- Citations: all prose citations except one have a citation-log row and match the sentence.
- Compiles: no `!` line, no undefined reference or citation (draft and submission builds).
- No em-dashes.

## Issues

### Major

1. **Notation: x-tilde defined twice (B, F).** Line 189 defines `\tilde x_k=\col(q_k,\dot q_k)` (as
   Sections II and IV do); line 210 redefines `\tilde x=T_x(\col(q,\dot q)-\mu_x)`. Then eq:aug_transition
   is ambiguous: f_base as "one RK4 step of (eom)" and h_base = P^T q hold only in physical coordinates,
   and the network actually sees normalised stage forces, not u = P u_act. The reader cannot tell in
   which coordinates the model runs.
   Rule cause: skill "Notation" (keep the symbols of other sections) and README Math standard item 4
   (normalisation displayed) do not require stating how the displayed model maps between physical
   and normalised coordinates; the review-loop check "No symbol used by another section was renamed"
   does not catch a redefinition inside the same section.

2. **Length and PS2 depth (C, E).** Submission build: Section III runs from p. 4 to p. 7, about 3 pages
   against a 1.55-page budget. III-D alone is about 35 sentences (stopping rules, calibration split,
   screen, gradient mechanics, Hefny/Downey comparison, limits) against the README's agreed PS2 sketch
   of "about two displays and six sentences". III-C also carries the m_Delta coordinates, the optimiser,
   a selection display, model naming and the masking caveat.
   Rule cause: README "Math standard", "Example (PS2, sketch agreed 2026-10-08)" was not enforced; the
   skill "Length" rule ("a target, not a cap") gives no trigger to re-check against the budget or an
   agreed sketch, and the review loop has no "compare each subsection with its agreed sketch" check.

### Minor

3. **Optimisation mechanics in the prose (C, E).** Lines 609 to 611 (gradient reaches W^a only through
   products of zero-start weights) and 655 to 658 (f_aug rows get no gradient, hidden layers move) are
   exactly "which weights receive a gradient first", which the skill's reader test excludes. One
   sentence (606 to 608) states the problem. Rule cause: skill "Content, The reader test" (followed
   only partially; the problem sentence of the README PS2 sketch invites elaborating it).

4. **Restatement (E).** "They have no assigned physical meaning ... Nothing in the method ties them to a
   particular subsystem" (286 to 290) says one fact twice, and 679 ("Neither stage ties x-bar to a
   physical subsystem") repeats it a third time. Line 469 ("The model thus receives the recorded input
   plus the controller's response to its own output error") paraphrases eq:aug_residual_loop.
   Rule cause: skill "Say each fact once" and "Equations carry the content".

5. **Symbol overloading (F).** M is the PS2 horizon (M = n+1) and the inertia M(Y); the draft's own todo
   rejects M_U for exactly this clash. sigma is the std sigma_x, the hidden-layer output sigma_{L-1} and
   the residual RMS sigma_e. phi is the head's parameters and phi_aug the network. xi is the encoder
   regressor xi_tau and the m_Delta coordinate xi_Delta. Rule cause: skill "Notation" only covers symbols
   of other sections; no rule asks for a within-section symbol uniqueness check.

6. **Opening announcement versus roadmap (E).** Line 128 "This section augments the baseline of ..."
   is word for word the announcement the skill's "Expert register" forbids, while skill "Order" and README
   "Prose around the math" require a roadmap paragraph naming the subsections. Rule cause: conflict
   between skill "Expert register" (no announcements, example "This section augments ...") and skill
   "Order" (opening paragraph names subsections); the rules should say the roadmap is the one allowed
   announcement and how to phrase it.

7. **PS2 screen wording inaccurate (F).** Line 661 "If at the end of S1 the head still leaves most of
   ||e||^2 unexplained, S2 is skipped": the code skips S2 when the unexplained fraction exceeds 0.90
   (`screen_max`), checked at update 150 and at the S1 end; "most" (> 0.5) misstates the rule. Rule cause:
   skill "Reasons and claims" (state a claim as the code defines it); review check "Every configuration
   value or mechanism described was confirmed in the code".

8. **Unverified citations (F).** `\cite{sutanto2020encoding}` (line 515) carries a VERIFY todo instead of
   the passage check the task's paper-verification rule requires; `\cite[Remark~5.1]{kessels2025ai}`
   (todo, line 199) and `\cite[Sec.~3.2]{hoekstra2026encoder}` (todo, line 389) have no citation-log row.
   `forssell1999revisited` is the Automatica 1999 article, while the log row uses the LiU report's
   numbering (Sec. 5.2.1); the cited "Sec. 5.2" is unchecked against the Automatica version.
   Rule cause: README "Reference verification gate" step 3 and skill review check "Every citation was
   checked against the passage in the PDF"; neither says that citations inside a `\todo` candidate
   also need the check, nor that the log must record which version's numbering is used.

9. **Contribution claim strength (F).** Lines 138 to 139 "It further contributes the identification in
   closed loop with the known controller": Kessels trains in closed loop with the known controller (the
   draft says so at 415), and sources.md's gap statement says closed-loop training alone is not novel.
   The own part is the residual form with joint estimation of theta_base under feedback. Rule cause:
   skill "Contributions" (adopted, adapted, own) and README "Claim strength".

10. **Packing (E).** Lines 250 to 252 (depends on Y, can represent position-dependent corrections, has no
    LPV structure: three facts), 444 to 446, 486 to 489 (about 35 words), and the figure caption's
    semicolon (264) join independent facts. Rule cause: skill review "Packing pass" (captions are not
    named in the pass).

11. **Repeated pointers to Experiment design (E).** Lines 577 and 664 both point to Section V for values
    (plus two inside todos). Rule cause: skill "Expert register" (one pointer per section suffices).

12. **Stale rule source (rules).** `rules/skill/reference/sources.md` Sec. 4 superseded table says "D-155
    Xavier W^a | code uses Kaiming", contradicting its own Sec. 1 row (Xavier, D-239) and the code
    (`wa_encoder_init='xavier_uniform_gain1'` in the thesis block). The writer chose correctly, but a
    fresh writer can be misled. Rule cause: README "Source map" update rule ("a session that supersedes
    a decision ... updates its row") not applied to sources.md.

## Not issues (checked)

- `x^c_tau = 0` reading versus the code comment: the draft states the algebra correctly (x^c is a
  difference of controller states) and leaves the D-142 conflict as a todo; consistent with the code
  under the stated compatibility assumption.
- L-BFGS "fixed number of iterations": `max_iter` is a maximum with tolerances, but the thesis run sets
  `tol_change = 0.0`, so in practice it runs to the cap; acceptable.
