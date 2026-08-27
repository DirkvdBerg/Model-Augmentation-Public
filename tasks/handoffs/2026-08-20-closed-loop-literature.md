# Handoff: literature search on closed-loop model training and closed-loop augmentation identification
**From**: session of 2026-08-20 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Run a deep-research literature search (the `deep-research` skill is MANDATORY per D-121, step 0
FRAME included, one subagent per seed) on training and identifying models INSIDE a closed loop,
because this project already trains its augmentation through a rolled-out closed loop and the user
believes we are underusing what that field knows. Answer five sub-questions, each with verified
citations and a stated relation to our rig: (Q1) what objectives and criteria exist for training a
model by simulating it in closed loop with a known controller (the classical family is Landau's
closed-loop output error, CLOE; our rollout follows Kessels 2025 Eq. 5.13d), and what bias and
convergence results attach to them; (Q2) what the closed-loop identification of a PERTURBATION of
a known model is called and how it is done (dual-Youla parameterisation, Hansen scheme), and what
our learned dynamic parallel augmentation corresponds to in that language; (Q3) what is known
about objective design in closed loop: input-error and controller-effort criteria (we measured a
feedback-effort loss term this session), sensitivity- or controller-weighted output error, and the
identification-for-control iterative schemes (Schrama, Gevers, Van den Hof); (Q4) what
informativity and discrimination results exist for closed-loop data, i.e. under what conditions
two different models are distinguishable from closed-loop records, because our measured plateau
mechanism is a loss that ranks a correct model only 1.13x above a wrong one (H2); (Q5) the
novelty check: has anyone trained an encoder-based (SUBNET-style) dynamic parallel augmentation
through a rolled-out closed loop, on simulation or hardware. Deliver the skill's mandatory
Research Log, per-question findings with the disqualification filter applied, and the updates to
`docs/references.md` for anything that enters the project's citation base.

## 2. Out of scope

* Any training run, any code change, any edit to the loss. The burn-in and effort-term arms are
  the CURRENT session's ongoing discussion and its decision, not this session's.
* The Telica data and anything under `kamtin-data/` (blocked by policy regardless).
* Re-running the 2026-08-19 literature seeds (lazy vs rich initialisation, latent-state
  initialisation, encoder co-estimation): done, results in
  `scripts/gantry/closed-loop-controller/ANN-learning-issue/RESULTS.md` section 6.
* Writing any new `docs/*.md`: findings go into the skill's report format and
  `docs/references.md` rows; if a dedicated document seems warranted, propose it and stop.
* Do not modify `scripts/gantry/gantry_dynamic/{config,evaluation,orth_penalty}.py` (another
  session's uncommitted work) or `kamtin-fp-model/` (read only).

## 3. Where things stand

Branch `Augmentation`, last commit `4cdb7c1`; the tree is dirty (D-150 implementation in
`gantry_dynamic/model.py`, three `closed-loop-controller` scripts, docs; nothing committed, the
commit decision is the user's). Nothing in flight. This handoff's session is a pure literature
session: it should not need the code at all beyond reading the two rollout references below.

## 4. Established and verified (the rig this search must map onto)

* The augmentation is trained in closed loop: the training rollout replays a verified controller
  around the model and scores the closed-loop tracking error against recorded `y`
  (`model_augmentation/fit_systems/closed_loop.py`, `closed_loop_rollout`; the controller error is
  formed against the model output per Kessels 2025 Eq. 5.13d, cited in that file). Selection is
  the 12 s closed-loop free-run position RMS.
* The plateau mechanism is objective discrimination, not representation and not initialisation
  (D-150 outcome, Arm F run-table row): with live augmented dynamics, no loss barrier, and every
  parameter training, the loss still ranks the known-correct model only `1.131x` above the
  plateau model in its own units (`runs/effort_discrimination.json`, 2026-08-20).
* The loop is the cause of part of that: below its ~100 Hz crossover it suppresses exactly the
  error the loss reads. Scoring the rollout's own feedback effort `u_fb` (target zero,
  oracle-free, equivalently a `|C|^2`-weighted error) lifts discrimination to `2.081x` alone and
  `11.27x` combined with a 100-sample burn-in, against `10.14x` for burn-in alone (same artefact;
  `ANN-learning-issue/RESULTS.md` section 10).
* A prior novelty claim of this project ("dynamic augmentation has never been shown to train")
  was REFUTED by the 2026-08-19 sweep, once by a paper already on disk (`RESULTS.md` section 6).
  Treat every novelty intuition, including this handoff's Q5, as unverified until swept.
* Known nearest neighbours already on disk: Hoekstra et al., EJC 86:101304 (2025),
  `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf` (dynamic parallel augmentation,
  open loop); Kessels et al., Nonlinear Dynamics 113:17335 (2025), DOI
  `10.1007/s11071-025-11092-5` (SUBNET-derived, ASMPT wire bonder, closed-loop data; also the
  thesis chapter our rollout equation cites).

## 5. Assumed but not verified

* "There will not be much literature on closed-loop augmentation" (user's working feeling). The
  search exists to test this; dual-Youla (Q2) is the obvious reason it may be false in spirit even
  if the exact encoder-based combination (Q5) is open.
* That CLOE-family results (bias freedom, stability conditions) transfer to a nonlinear model
  with an encoder and a windowed simulation loss. Settled only by reading the actual assumptions.
* That the effort-term mechanism we measured corresponds to the classical input-error criterion.
  Plausible naming match; verify against the definitions before citing it that way.

## 6. Tried and failed

* Keyword web search for control topics -> measurably fails (authors rename concepts, control
  publishes across IFAC/IEEE/Elsevier) -> use the skill's author-ID and citation-edge enumeration
  -> D-121 rationale, `.claude/skills/deep-research/SKILL.md`.
* Carrying a novelty claim without a sweep -> refuted three times over on 2026-08-19 ->
  `RESULTS.md` section 6.

## 7. Achieved

None yet in this thread; this is the search's first session. The measured rig facts it builds on
are in section 4 with artefacts.

## 8. The open question

Does the closed-loop identification literature already contain (a) an objective for our setting
with better discrimination properties than windowed closed-loop output error, and (b) the
augmentation-in-closed-loop problem under another name (dual-Youla)? Candidate answers: CLOE
variants with filtered errors; input-error criteria; dual-Youla residual identification with a
learned parameter. Evidence that chooses: the actual assumptions and results of the found papers
against the section 4 rig facts.

## 9. Next action

Invoke the `deep-research` skill with the five sub-questions of section 1 as the frame, seeded by:
Landau and Karimi CLOE (closed-loop output error, 1997 onward); Kessels 2025 (thesis chapter 5 and
the Nonlinear Dynamics paper, both partially on disk); Hansen scheme / dual-Youla identification
(Hansen, Franklin; Van den Hof and Schrama survey); identification for control (Gevers; Schrama);
informativity of closed-loop data (Ljung; Gevers, Bazanella). One subagent per seed, each
returning the skill's Research Log.

## 10. Acceptance criterion

Every sub-question Q1-Q5 answered with at least one verified, accessed source or an explicit
"nothing found via author-ID and citation-edge enumeration" statement listing the enumerations
tried; the Q5 novelty answer stated in the falsifiable form the 2026-08-19 sweep used ("no
published work does X", with the searched space named); every retained paper as a
`docs/references.md` row with its PDF location or access status.

## 11. Read these first

1. `.claude/skills/deep-research/SKILL.md`: the mandatory procedure, FRAME first.
2. `scripts/gantry/closed-loop-controller/ANN-learning-issue/RESULTS.md` sections 6, 9, 10: what
   is already measured and already searched.
3. `model_augmentation/fit_systems/closed_loop.py` docstrings around `closed_loop_rollout`: the
   exact rollout the literature must be mapped onto, with its Kessels citation.
4. `docs/aug-lru-implementation.md` section 1: what the current fix is and is not, so findings are
   related to the true state.
5. `docs/references.md`: the citation base and its format.

## 12. Do not

* Do not use ad-hoc `WebSearch` in place of the skill (D-121).
* Do not write conclusions about our rig into `ANN-learning-issue/` or the problem log: report
  them; the integrating session decides what is adopted.
* Do not start the burn-in or effort-term training arms, and do not modify any loss code.
* Do not commit anything.

## 13. Operational

No runs, no conda needed beyond possibly opening PDFs. Paper PDFs land under `literature/`
(existing subfolder conventions: `augmentation/`, `stability-training/`, `deep-ssm-init/`; a new
subfolder like `closed-loop-id/` is consistent with these and is authorised by this handoff).
Artefacts consumed: `runs/effort_discrimination.json` if numbers are needed;
`ANN-learning-issue/RESULTS.md` for every rig fact.

## 14. Delegation

Yes: this is the one task class where multiple subagents are the rule, one `deep-research`
subagent per seed of section 9 (five), each returning its Research Log. No Explore subagents; no
verification subagents.
