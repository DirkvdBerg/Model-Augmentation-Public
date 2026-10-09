# Critic (examiner lens, Maarten Schoukens), cycle 07, Section III

Score: 7 / 10

Overall: read once, the section shows what is built, which signals enter and leave, what is trained and fixed, and against which cost. Every part has its display: augmented model with spaces, normalisation in the form of Hoekstra 2026 Eq. 28, zero output layer (Eq. 29), encoder and W^b (Eqs. 30, 31 and encoder paper Eqs. 15 to 17), the residual closed loop, the training coordinates, and one Eq. 22 style criterion with every constraint on its own line. The roadmap and the problem, construction, consequence order hold in III-A, III-B and III-D. It compiles (no "!" lines in build/main.log; the overfull boxes are in Section II). What keeps it from 8 or 9: III-C carries four jobs and reads partly as a list; PS2 hides that plain training runs during S1, so "what is trained when" is wrong for the phase Maarten asked about; four realisation choices in III-A stand without reasons (todos); several smaller rule slips; and the length is roughly twice the 1.55-page budget (about 2600 words with 10 displays and 2 figures, submission build).

## Issues

1. MAJOR (B, F). III-D, S1 and S3 paragraphs (draft lines 612-661). The text runs S1, then S2, then "S3 ... continues from the result, with the model, the criterion V, the optimiser and the selection unchanged". In the code S1 runs while the normal training of V runs (`FS/ps2_opening.py` docstring: "while the normal training runs"; `after_step` is called after every optimiser update, `FS/interconnect.py` line 1046). So during S1, xi, theta_aug and theta_enc are being trained against V at the same time, and "the residual of the current model" moves at every update. An examiner reads a sequential pretraining stage. One clause in the S1 lead-in ("S1 runs alongside the minimisation of V") fixes it.
   Rule cause: README "Math standard", PS2 sketch: it puts "interleaving with the updates of V" under optimiser schedules that "stay out", which removes a fact that changes what is trained in the phase. Missing distinction between optimiser schedule (out) and which problems are solved concurrently (in).

2. MAJOR (A). III-C (lines 425-592) holds four jobs in eleven paragraphs: the closed-loop simulation construction (problem, Kessels and Forssell positioning, residual loop, compatibility, x^c), the training coordinates of theta_base, the criterion with optimiser, and checkpoint selection with the polish rule, the closed-loop masking caveat and the naming of M_U. It is the paragraph run Maarten would read as formula and reason blocks; the residual loop is a component construction and the criterion is the identification problem.
   Rule cause: skill "Order", which gives conflicting guidance: "One job per subsection" and "Constructing a component and stating the identification problem of the model that uses it are two jobs, so two subsections" versus "group them as Hoekstra 2026 Sec. 5 does". The rules do not say whether the closed-loop simulation is a component (own subsection) or a constraint of the problem (grouped).

3. MAJOR (D). III-A, realisation-choices paragraph (lines 248-264): "The learned part rests on five realisation choices" followed by five choices of which four (position rows corrected, correction after the RK4 step, no LPV structure, h_aug = 0) carry no reason, only four todos. The choices are the ones an examiner in this group questions first (LPV structure departs from the research plan and D-017). The todos are honest, but the paragraph also opens with a count announcement.
   Rule cause: skill "Expert register" (no announcements) failed to prevent the count sentence; the missing reasons are a source gap (no decision entry), correctly marked by skill "Reasons and claims".

4. MINOR (F). III-C selection paragraph (lines 572-577): "no monitored quantity of the polish worsens by more than a relative tolerance. One of these quantities is the combination error". The code compares `combo_err`, `orth_frac`, `V_orth` (`GD/training.py::_gantry_meters`) and `param_loss` (`generic_meters`). Only one is named.
   Rule cause: skill "Content", reader test: "every compared quantity named (never 'a monitored quantity')"; failed to prevent (writer slip).

5. MINOR (B). eq:aug_norm, line 234-235: the left side takes u^n, the right side evaluates f_base at the physical u, not at a map of u^n. The line that is meant to show how a physical-unit map acts in normalised coordinates does not close in u. Also the criterion's encoder line writes x_tau where eq:aug_encoder writes x_{tau|tau}.
   Rule cause: README "Math standard" item 4 failed to prevent; skill review "Notation" check does not ask that both sides of a coordinate-change line use the same coordinates.

6. MINOR (A). III-A closing paragraph (lines 293-309): the non-uniqueness of the split under joint estimation and the handover to Section IV sit in the model subsection, before the identification problem that makes theta_base trained is defined (forward pointer "Section III-C"). Known before new is broken; the paragraph belongs after eq:aug_loss.
   Rule cause: MUST ESTABLISH item 7 places it in III-A; skill "2a" says a block fixes claims, not sentences, but gives no rule that a block item's placement yields to "Known before new".

7. MINOR (E). III-C, Kessels and Forssell paragraph (lines 439-450): five sentences of positioning, two for Kessels (trains this way; open-loop attempt failed) and two for Forssell (indirect approach; consistency), plus the scope sentence.
   Rule cause: skill "Order": "one sentence per cited method"; failed to prevent.

8. MINOR (A). III-C x^c paragraph (lines 497-504): the closing "Model error made before tau is therefore not carried into the window" follows the Kessels contrast, but it is a consequence of setting x^c_tau = 0 (and of the encoder), two sentences earlier.
   Rule cause: skill review "every later sentence of a paragraph serves its first; a closing sentence on another point moves"; failed to prevent.

9. MINOR (E). Training-coordinates paragraph (lines 506-529): the lead-in states that the coordinates keep admissibility at every iterate, and the post-display sentence states it again ("every xi satisfies ... so M(Y) > 0 for every Y").
   Rule cause: skill "Say each fact once"; failed to prevent.

10. MINOR (F). Figure fig:aug_structure is first referenced at line 280 ("Training starts from the baseline"), after the model display and the realisation choices it depicts.
    Rule cause: skill "Citations and values": "Refer to every figure in the text where the reader first needs it"; failed to prevent.

11. MINOR (C, E). Length about twice the 1.55-page budget with no delivery note naming the required content that causes it. Ten displays is Hoekstra Sec. 5 density, so the budget looks inconsistent with what the ownership row and the Math standard require (model, encoder, criterion, PS2, selection, two named models).
    Rule cause: missing rule: the header page budget is not reconciled with the README ownership row and Math standard; skill "Length" only asks for a second deletion pass and a report.

12. MINOR (F). Horizon check of the header (relate encoder history, window and validation horizon to the stable modes) is reduced to one sentence "are separate quantities", which fails the deletion test as written and covers neither the relation nor a todo.
    Rule cause: skill "The header": header items other than Cautions and anchors are candidates; no rule says a kept candidate must carry its content or a todo rather than a bare statement.
