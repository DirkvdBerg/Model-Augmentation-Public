# Critic (examiner lens, Maarten Schoukens), cycle 08, Section IV

Score: 7 / 10

Verdict: one argument with a clean roadmap (negation, construction, identification, scope) and four
subsections with one job each. The math shows what is built on the gantry (Phi in the training
coordinates xi, recorded tuples, theta_aux, the corrected transition), what is trained (vartheta) and
what is fixed (Phi per epoch, theta_aux not trained), against V. The proposition and proof are short
and checked (the algebra of the proof is correct under (ii) to (iv)). What keeps it from 8 or higher:
IV-B reads partly as a run of choice-and-reason paragraphs, the corrected transition is displayed twice,
the identification display mixes coordinate sets, and the scope paragraph is a long caveat list.
Compiles (Writing/sections/04_obc.tex is identical to draft.tex; build/main.log has no "!" line).

## Major

1. A Coherence, IV-B (lines 233 to 257). After the tuples and coefficient, three paragraphs each hold
   one realisation choice with its reason (epoch rebuild; rank rule with its fallback; offset
   exclusion), and the rank paragraph's topic sentence ("protects the ten parameter directions and
   nothing else") covers the offset choice while its body is mostly the rank fallback. Read once, IV-B
   becomes construction, then a list of choice blocks, then the rollout model, then the consequence.
   Rule cause: README "Prose around the math" ("the remaining realisation choices of one component
   form one paragraph") is not enforced by the skill's review loop, which checks only "the other
   realisation choices of one component sit in one paragraph" without saying how a writer tells a
   choice paragraph from an argument paragraph; no rule says the rollout model (the construction's
   output) precedes the choices that tune it.

## Minor

2. B/C, eq:obc_model and eq:obc_loss (lines 261 to 324). The three-line corrected transition is
   displayed twice, once in IV-B and again (with theta_aux substituted) in IV-C. Rule cause: skill
   Order, Method ("Constructing a component and stating the identification problem ... two jobs")
   together with README Math standard item 3 ("Repeating a constraint line of an earlier display
   inside the first problem display is required") force the duplicate; no rule lets the component
   subsection state the correction term only (the line that changes) and leave the full transition
   to the problem display.

3. B/F, eq:obc_loss (lines 313 to 321) and eq:obc_model line 268. The replaced line is in normalised
   coordinates (x~^n, u^n), while the cited unchanged lines of eq:aug_loss are in physical units
   (x~_k, u_k = P(...), h_base(x~_k)); eq:obc_model writes g_aug(theta_aug, x^_k, u_k) next to
   f^n_aug(theta_aug, x^^n_k, u^n_k). The reader must convert coordinates to read one problem.
   Rule cause: README Math standard item 4 ("Every later display writes the normalised symbol, and
   both sides of that line use one set of coordinates") is per line, not per problem display; it does
   not say what to do when the substituted problem's unchanged lines (owned by an earlier section) are
   in the other coordinate set.

4. F, rebuild paragraph (lines 233 to 239). The contrast says Gyorok et al. update at every
   evaluation or fix the point at the nominal parameters (2025 Sec. 4), but citation-log line 75
   records gyorok2025l4dc Remark 2: the projection is recomputed at the start of each epoch with current
   state estimates. The per-epoch refresh has a direct precedent the section omits, so the choice is
   positioned as a larger departure than it is. Rule cause: skill "Contributions" (adopted, adapted,
   own) and Workflow step 1 ("Read the method sections of the papers this section adapts in full")
   failed to prevent it; the IV-B source-map row lists gyorok2025l4dc Eq. 18 only, not Remark 2 or
   Sec. 4.

5. E, IV-B opening (lines 165 to 177). "For an input-output baseline that is linear in its
   parameters, Gyorok et al. remove ..." and four sentences later "Its baseline is an input-output
   model linear in the parameters [Eq. (2)]" state the same fact twice. Rule cause: skill "Say each
   fact once" failed to prevent it (within-paragraph restatement separated by intervening
   sentences).

6. D, tuples paragraphs (lines 181 to 192). The todo says no source gives why recorded tuples beat the
   closed-loop simulation of III-C, yet the next sentence gives a reason-like clause ("so the
   controller does not enter the regressor") that is neither sourced nor reconciled with the todo.
   Either it is the reason (then the todo's candidate should use it) or it is not (then delete it).
   Rule cause: skill "Reasons and claims" ("A reason you derive yourself becomes an open point with
   your reason as the candidate") failed to prevent it.

7. A/D, IV-A paragraph 3 (lines 125 to 137). It closes with two sentences on Kessels' broader notion,
   a literature point that does not serve the paragraph's topic (the first-order cancellation).
   Rule cause: skill review check "every later sentence of a paragraph serves its first; a closing
   sentence on another point moves" failed to prevent it; skill Order ("Literature positioning sits
   at the component it contrasts") allows it here, which pulls the other way.

8. C/E, rank paragraph (lines 243 to 249). The fallback mechanics (keep the previous stack, abort after
   two consecutive losses, the 6|I| eps tolerance) are optimisation and implementation detail an
   examiner does not need to understand the method; the full-rank check alone carries the argument
   (counterpart of Assumption 1). Rule cause: skill "Content" reader test lists optimisation mechanics
   as failing, but its exception "a rule that defines the method ... gets one sentence with its
   condition as the code states it" lets a fallback rule pass as method-defining; the boundary is
   unclear.

9. E, scope paragraph (lines 423 to 445). Eleven sentences, seven distinct caveats, in one paragraph;
   "is charged only through its component in the range of Phi" uses an unexplained metaphor. Read
   once, it is a hedge list rather than a consequence. Rule cause: skill Order, Method ("Scope items
   that are not assumptions ... form one scope paragraph after the proof") sets the place but no
   rule ranks or limits scope items (only those an examiner would question, each once).

10. D, IV-C validation sentence (lines 343 to 345). "computed once from the current network and held
    fixed over the rollout" is true in training as well (theta_aux depends on theta_aug only), so the
    sentence distinguishes nothing. Rule cause: skill "Deletion test" failed to prevent it.

## Not issues (checked)

- Proposition 1 proof: delta = Phi(xi - xi*) + (I - Phi Phi^+) f_aug follows from (ii) and (iv); left
  multiplication by Phi^+ under (iii) gives eq:obc_bias. Correct.
- Citations used (gyorok2026obc Sec. 3.2, Eqs. 8, 10, 13, Lemma 3, Assumption 1, Cond. 4, Thm. 7,
  Eq. 21, Assumption 6, Remark 5, Thms. 16, 18, Cond. 10, Sec. 6; gyorok2025l4dc Ex. 1, Eqs. 13, 15,
  18, Remark 1, Secs. 3, 4; kessels2025ai Sec. 5.3.1.3) match citation-log entries.
- Roadmap builds on III-A (non-unique split) and names the third contribution; draft bullets map to
  paragraphs; todos sit at paragraph ends.
