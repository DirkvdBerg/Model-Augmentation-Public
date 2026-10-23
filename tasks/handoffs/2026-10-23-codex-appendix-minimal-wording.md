# Handoff (Codex): reword the appendix subsection "PS details" to minimal, human wording in Dirk's style, applying only on his yes
**From**: Claude session of 2026-10-23 (Section IV walk) | **Branch**: Augmentation | **Effort suggested**: high (judging each sentence against the math next to it and against the main text; no runs)

**Precedence: Dirk's current message always overrides this plan. When he asks a question, answer it directly first, in a few sentences, before anything else.**

## 1. Task
Reword the subsection `\subsection{PS details}\label{app:ps_stop}` of `Thesis-writeup/Writing/sections/99_appendix.tex` (and nothing else in the appendix) with Dirk (the author). The goal is the smallest wording that lets the math speak: every remaining sentence either introduces one of its equations, defines a symbol an equation uses, or states a fact that Section IV-E points to. Strike what repeats IV-E or restates the equation beside it. Where a sentence stays, keep Dirk's words and fix only what is unclear, wrong, or not in his style (4.2). First read the whole subsection in its final reading (4.1) and the IV-E passages that point to it (4.3). Then post the complete list of proposed changes for the subsection in one reply: each item as "now" / "after" in final reading (rendered text, not LaTeX) with a one-line reason, and state that nothing else is open. Then the exact LaTeX with change marks. Apply only after Dirk writes "yes", "apply" or "do it", and only what he approved. When he rewrites a sentence himself, check it only for errors (grammar, LaTeX, a wrong fact) and apply his version.

## 2. Out of scope
- Every other appendix subsection and all main-text sections. If a change in PS details needs a change elsewhere (a pointer in IV-E, a symbol), say so in one line and leave it.
- Equations and numbers: do not change any equation or value. Report a notation error; fix it only on Dirk's yes.
- `\todo`, `\revnote`, `\ifdraft` and `wip` blocks: leave them.
- LaTeX builds, git writes, numerical runs. Do not read `kamtin-data/`.

## 3. Where things stand
Branch `Augmentation`, last commit `4df8faa`; `99_appendix.tex` and `03_augmentation.tex` have uncommitted edits from 2026-10-22/23. Dirk edits in parallel and saves often: re-read the subsection right before editing, and if an edit does not match, re-read instead of guessing. On 2026-10-23 the subsection was already reduced (much prose struck) to four equations: the decoder p, the unexplained fraction nu (`eq:ps2_nu`), the S1 stopping rule (defining i_1), and the S2 model input at the window start. Dirk still finds its wording not human or minimal enough; that is this task. Its final reading today, roughly:
> In (28), sigma_e is the root mean square of all residual values in the minibatch B. The decoder p of (28) is p(theta_p, x-bar) = W_p,2 tanh(W_p,1 x-bar + b_p,1) + b_p,2, where W_p,2 and b_p,2 are initially set to zero. The records are divided into excitation classes, and for classes with at least three records one is left out of S1 and S2. On a fixed set of windows C from these records we compute nu = ... The unexplained fraction nu of (31) is evaluated after every i_c-th update of V, with nu_i its value after update i. S1 ends at update i_1 = min{...}, where J = ..., ... S2 runs if ... If nu_{i_s} > nu_max, PS ends at i_s. The values are given in Table ... At the start of a window x^c = 0 in (19), resulting in the model input u-hat = u + D_fb(y - y-hat), when the encoder reproduces y_{k_i} this equals the recorded input.
Read the source for the exact current text.

## 4. Established and verified

### 4.1 Change marks (`Thesis-writeup/Writing/util/format.tex`)
Final reading: `\chg[tag]{old}{new}` = new; `\del[tag]{old}` and `\delmv[tag]{to}{text}` = gone; `\mv[tag]{from}{text}` and `\add[tag]{new}` = present; `\kw`, `\kwtwo` = plain text (colour only); `\todo`, `\revnote` = not printed.
Rules for your edits: tag `own` for your proposals, `Dirk` for his rewrites; new wording also inside `\kwtwo{...}`. Never nest a mark inside struck text (`\del`, `\delmv`, the old argument of `\chg`): flatten the old text to its current reading and put an existing separate `\del` beside the new one, not inside. Inside struck text write `\mbox{\eqref{...}}`, `\mbox{\ref{...}}`, `\mbox{\cite{...}}`. Never put a mark inside a `\cite[...]` optional argument. Equations take no marks: keep the old version as a `%` comment and add a one-line `\revnote{...}` if an equation changes. Add a `% FEEDBACK (own, 2026-10-2x): <reason>` comment above each change. After editing, check that `{` and `}` in the file balance with the same offset as before (+1, pre-existing; do not fix it).

### 4.2 Dirk's style (follow it; do not describe it to him)
- Subject first, stating a fact; the reason at the end after "because"; purpose with "to" ("... to keep the positions consistent with ...").
- Plain verbs: is, are, uses, gives, is calculated; "we" structure ("We use ...", "We call this step polish.").
- One subordinate clause per sentence; few commas.
- Never: "so" / "so that"; "This makes ..."; participle tails ("..., making ..."); colon lists; semicolons; appositions; opening adverbials; stacked prepositional chains; the words "acts", "operates", "target", "is placed", "intercept", "update V" (say "minimise V").
- Reuse his own wording before inventing any: struck text in the file, `\delmv` originals in `03_augmentation.tex`, `git show 25b5ffb:Thesis-writeup/Writing/sections/03_augmentation.tex` (the original IV-E text this subsection came from).
- When unsure, propose a cut, not a new sentence. He rejects added words and long rewrites.
Example of a good exchange (his sentence kept, errors fixed only):
> Now: "Records are left out of S1 and S2 to evaluate each stage performance."
> After: "A subset of records is left out of S1 and S2 to evaluate the performance of each stage."

### 4.3 What IV-E already says (do not repeat it in PS details)
In `03_augmentation.tex`, subsection `sec:aug_ps2`: the decoder is "a neural network with one hidden layer (Appendix ...)"; "sigma_e scales the residual to unit size (Appendix ...)"; "A subset of records is left out of S1 and S2 to evaluate the performance of each stage."; "We use the left out records to decide when S1 ends and whether S2 runs (Appendix ...)."; "The left out records are still used to minimise V."; "At the start of a window, the recorded force equals the closed-loop input of V when the encoder reproduces y_{k_i} (Appendix ...)."; i_1 is "the last update of S1"; the hand-over uses i_1. PS details only needs to give the math behind these four pointers.

### 4.4 Conventions
D_N is the training set and consists of multiple records ("the records of D_N", never "training data"); decoder p; K_fb; the hyperparameter values are in Table `tab:hyper` and stay there. Maarten (main supervisor) does not read the appendix.

## 5. Assumed but not verified
None.

## 6. Tried and failed
- A first version of this prompt covered the whole appendix and fixed a starting subsection; Codex then reviewed "Nominal physical parameters" while Dirk asked about PS details. Scope is now PS details only, and Dirk's messages take precedence.
- Rewording several sentences at once or adding explanatory clauses -> Dirk: "too many words", "ai", "you're becoming worse". One recommendation per item; when he rejects a wording, offer his own earlier wording next.
- Raising extra points after the list -> he loses track. Give the complete list once, then handle his choices.

## 7. Achieved
PS details reduced to four equations on 2026-10-23; IV-E pointers to it in place.

## 8. The open question
Nothing blocked.

## 9. Next action
Read `\subsection{PS details}` in `99_appendix.tex` in full (final reading) and the four IV-E sentences in 4.3, then post the complete now/after list of proposed changes for PS details.

## 10. Acceptance criterion
Dirk has seen the complete list for PS details and every item is applied with his yes or left by his choice; no equation or value changed without his yes; braces offset unchanged; every change visible through marks.

## 11. Read these first
1. `99_appendix.tex`, `\subsection{PS details}` (the text to reword).
2. `03_augmentation.tex`, subsection `sec:aug_ps2` (what is already said).
3. `util/format.tex` (the marks).

## 12. Do not
- Touch any other subsection or section.
- Edit before Dirk's explicit yes; a pasted phrase or a question gets feedback only.
- Use any construction from the "Never" list in 4.2.
- Run a build or a git write.

## 13. Operational
None: text editing only.

## 14. Delegation
None.
