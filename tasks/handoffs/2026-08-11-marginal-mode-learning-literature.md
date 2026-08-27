# Handoff: does the literature explain why gradient descent will not learn poles at z = 1, and what fixes it
**From**: session of 2026-08-11 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Run a literature sweep, via the `deep-research` skill, on one question with two halves.
**Why does gradient descent on a subspace-encoder multi-step prediction loss fail to place
discrete-time poles at `|lambda| = 1` from a random initialisation, and what initialisation or
parameterisation makes marginal modes reachable and trainable?** The frame is already built and is
in section 9; use it rather than rebuilding one. The deliverable is the skill's required output
format: findings with full citation metadata and free-copy locations, an access-status line, an
evidence-quality line, and the Research Log. Report every candidate mechanism you find, including
ones that only partly fit and ones that contradict our measurements, tagged with confidence.

## 2. Out of scope

Each of these already has a sweep in this repo. Do not re-run them, and if a hit belongs to one of
them, note it in one line and move on.

- **Drift of the augmented model** (baseline + ANN). Covered by `docs/drift-literature-sweep-2026-07-25.md`, `docs/dc-accumulation-literature-sweep-2026-07-26.md`, `docs/rollout-stability-literature.md`.
- **Adam versus SGD, and optimiser-induced drift.** Covered by `literature/stability-training/claude-research-optimizer-SGD-vs-ADAM{,-v2}.md` and `claude-deep-research-Adam-optimizer-drift.md`.
- **Orthogonal projection regularisation, Györök, parameter-interpretability.** Different thread.
- **ARTBP and truncated-BPTT gradient bias.** Covered; `Unbiasing Truncated Backpropagation Through Time.pdf` is already on disk.
- **Excitation and experiment design.** `docs/excitation-design-literature.md`.

Do not modify anything under `scripts/gantry/ann-blackbox/`, `kamtin-fp-model/`, or
`scripts/gantry/gantry_dynamic/`. This task writes prose, not code.

## 3. Where things stand

Branch `Augmentation`, tree dirty across `docs/`, `scripts/gantry/`, `tasks/`. The session that
produced this handoff wrote `docs/gantry-augmentation-problem-log.md` Section 11b (the write-up)
and five new rows in its Section 12 run log.

One run may still be in flight: corrected MSD morph arms, task `bvdog6zb7`, launched from
`scratchpad/msd_freefloat_train.py --noise abs`, writing `msdff_{grounded,freefloat,freefloat_centered}_abs.json`
to the session scratchpad. If those files exist, read them before writing any novelty claim, since
they decide whether the controlled demonstration stands. If the scratchpad is gone, treat that arm
as unfinished and say so.

A 12 h BLA run may have been launched on the user's server; ask before assuming its result.

## 4. Established and verified

All measured 2026-08-11 on `V2_aprbs_Ylow`, `fs = 800`, unless stated. Detail in Section 12 rows.

- The gantry's X and Y have **no stiffness**: `_K4` diagonal is `[0, kb1+kb2, 0, KA]` in `scripts/gantry/msd-offset/plant.py:50-51`. Two free rigid-body modes, poles at exactly `z = 1`.
- **Random initialisation never produces them.** Learned `max abs(lambda)`: `0.67-0.82` at `nf = 400`, `0.63-0.67` at `nf = 3700`, `0.75` with 14 training records. Zero modes above 0.999 in every case.
- **N4SID does**, from the same input-output data, to within `1.755e-05` of `z = 1` (`bla_init.py` log line, `SS_f = 60`).
- **Started there, the network keeps them.** `--bla dyn --bla-zero-nl`, 5215 updates: `max abs(lambda) = 1.0018`, two modes above 0.999, matching the true system. Score `0.31768 -> 0.03347`, `0.228x` of the mean predictor, `178x` above the FP baseline. Per channel X went `1.03x -> 0.28x` of its own signal std, Y `4.67x -> 1.20x`.
- **Our implementation is Jan's.** Bit-identical parameter count, epoch-0 loss and one-epoch train loss against `scripts/ecc_2025/msd_ndof_deepSI_encoder.py` on his data; identical 150-point loss curves.
- **Jan's ECC black-box script uses no special initialisation**, i.e. deepSI's random default. His `scripts/ecc_2025/msd_ndof_pre_encoder.py` does pre-train an encoder, but supervised against the **true state trajectory**, which a black box cannot use.

## 5. Assumed but not verified

- **The mechanism is unknown.** The natural story, that the encoder re-initialises the state every window so the loss cannot see memory longer than the window, **is refuted by our own data**: `nf = 3700` is 4.6 s, 9x longer, and left the spectrum unchanged. Do not repeat that explanation as if it were established.
- That the `0.65-0.82` outcome is an optimisation-landscape property rather than a normalisation or conditioning artefact. Untested.
- That the corrected MSD morph arms will reproduce the collapse. Pending.
- That N4SID's success is about being a global linear-algebra method rather than about `SS_f` selection. `bla_init.py:50-60` shows `SS_f` choice moves `max abs(eig)` between 0.999995 and 1.41, so the selection is load-bearing.

## 6. Tried and failed

Each of these is a measured dead end for the gantry black box. Do not propose them as fixes.

- More epochs -> lr sweep at `1e-3 / 3e-4 / 1e-4` all flatten at `0.128-0.133` by update 140-245 -> the random-init optimum is reached fast and is useless -> Section 12 lr-sweep row.
- More data -> 14 records spanning Y = -30..+30 made `rms_Y` **worse**, `0.209 -> 0.247` -> the pooled training mean moves further from the validation operating point -> Section 12 14-record row.
- Longer horizon -> `nf = 3700` (Beintema 3-tau at 800 Hz) left `max abs(lambda)` at `0.63-0.67` -> horizon does not move the spectrum at all -> Section 12 LinDY-vs-horizon row.
- Better initial state -> fitting **only** `x0` with weights frozen changed the score by `0.8x` and `1.0x` -> the residual is not an initial-condition offset, so the encoder is not the bottleneck -> `scratchpad/oracle_x0.py`.
- LinENC (`--bla full`, reconstructability-map encoder) -> epoch-0 `0.67728` where it must beat LinDY's `0.31768`; final `0.17202`, worse than a constant -> defect, and training loss `0.018` against validation `0.245` at update 280 points at **time alignment** rather than normalisation -> Section 12 LinENC row.
- Oversampling as an explanation -> gantry arms run at **5.06 samples/period**, fewer than Jan's own 8.67 -> excluded.
- **A methodological failure worth knowing**: the first MSD morph arms scaled sensor noise to the output std, which on the free-floating arm buried the dynamics at **-55 dB**. Sensor noise is absolute. The first `freefloat` and `freefloat_centered` numbers (`0.618`, `0.999`) are **invalid, do not cite them**.

## 7. Achieved

- First working black-box arm: `scripts/gantry/ann-blackbox/results/bla_long/`, `best_sim_rms = 0.03347`, implemented and validated.
- Instrumentation fix in `ann_blackbox.py`: `full_Loss_val` and friends recovered from the `_last` checkpoint, plus `argv`, `n_its`, `timeout` in the metrics JSON. Validated: `loss_val` 5 points, `full_Loss_val` 7, on a run where the old file was structurally blind past the best epoch.
- Paired MSD morph datasets with verified pole locations, grounded `max abs(z) = 0.996232` and free-floating exactly `1.000000` twice.
- Write-up: `docs/gantry-augmentation-problem-log.md` Section 11b.

## 8. The open question

**Why can gradient descent not reach `abs(lambda) = 1`, when a linear subspace method reaches it
to 1e-5 on the same data?** Candidate answers, and the evidence that would choose between them:

1. *Optimisation landscape*: the marginal region is a measure-zero boundary with divergence on one side, so stochastic descent has no incentive to sit on it. Evidence: a paper showing a barrier or an implicit bias toward contractive maps in multi-step losses.
2. *Parameterisation*: an unconstrained MLP state map cannot represent near-unit eigenvalues stably, and the deep-SSM community solved this with explicit eigenvalue parameterisations. Evidence: LRU/S4/S5 initialisation results, which this repo has **never searched**.
3. *Objective*: the windowed multi-step loss is weakly informative about slow modes. Weakened by our own `nf = 3700` result, but not dead, since the encoder resets state regardless of window length.
4. *Conditioning*: normalisation or state scaling makes the marginal direction badly conditioned.

There is a better task hiding here, and it is the user's call, not yours: if the literature says
the fix is a parameterisation rather than an initialisation, the next engineering step changes from
"repair LinENC" to "reparameterise the state map". Say so in one sentence; do not start it.

## 9. Next action

**Run the `deep-research` skill with the frame below.** It is already built, checked against local
holdings, and its gap line is the reason this handoff exists.

```
Sub-questions:
 1. Implicit bias of multi-step / rollout prediction losses toward contractive dynamics:
    does anyone show gradient descent avoids the unit circle?
 2. Parameterisations and initialisations that make near-unit eigenvalues reachable AND
    trainable: deep SSMs (S4, S5, LRU, HiPPO), unitary/orthogonal RNNs, antisymmetric RNNs.
 3. Initialising nonlinear state-space identification from a linear model (BLA, N4SID,
    subspace): state of the art beyond Ramkannan 2023, and when it is reported to help.
 4. Identification of marginally stable / integrating / unit-root systems: is there a
    dedicated SysID literature, and what does it say about pole placement at z = 1?
 5. Does anyone report that encoder / multiple-shooting objectives are biased against slow
    modes, or that free-run and windowed objectives disagree on marginal systems?

Seed DOIs held: Ramkannan et al. 10.1016/j.ifacol.2023.10.010 (see bla_init.py:1-8);
                Beintema et al. Automatica 156:111210
Entry points: IFAC-PapersOnLine OpenAlex source S2898405271; dblp venue:CDC / venue:ECC /
              venue:NeurIPS / venue:ICLR / venue:L4DC; PMLR v283 and v242; arXiv API with
              abs: field prefixes; authors resolved off the two seed papers, not a stored list.
Disqualifies: requires true states or oracle information (must be black-box legal);
              stabilising an UNSTABLE learned model (we need marginal, not stable);
              continuous-time-only results with no discrete-pole statement.
Anti-scope: everything in section 2 of this handoff.
Vocabularies (>=3, state which you searched next to any novelty claim): control
  ("marginally stable", "integrating process", "pole on the unit circle"); machine learning
  ("long-range dependencies", "curse of memory", "spectral radius", "state space model");
  econometrics / time series ("unit root", "I(1)", "non-stationary"); numerical linear
  algebra ("subspace identification", "N4SID").
Already held locally: 10 files in literature/stability-training/, 13 docs/*literature*.md,
  3 scripts/gantry/drift-fix-trials/research/thread-*.md.
  Covered: Beintema 22 files, curse-of-memory 4, antisymmetric RNN 3, unitary 2,
           Ramkannan 2, echo state 1.
  GAP, never searched by this project: orthogonal RNN parameterisations 0,
           LRU / linear recurrent unit 0, S4 / deep state-space models 0.
```

Sub-question 2 is the highest-value one and should get the most budget: the deep-SSM literature
exists precisely to place eigenvalues near `abs(lambda) = 1` and keep them trainable, and this
repo has never looked at it.

## 10. Acceptance criterion

Done when each of the five sub-questions carries either a cited answer with authors, title, venue,
year, DOI and free-copy location, or an explicit coverage gap saying what could not be reached and
why. For sub-question 2 specifically, done means a named parameterisation with the **initialisation
rule stated well enough to implement**, or a stated finding that no such rule exists.

A negative result is a result: per the skill, an arXiv `opensearch:totalResults` of 0 to 2 on a
whole-of-abstract search is the strongest novelty evidence available, and it must be reported with
the exact query that produced it. Any novelty claim needs the vocabularies searched stated beside
it, per the multi-vocabulary rule.

## 11. Read these first

1. `docs/gantry-augmentation-problem-log.md` Section 11b, the measured state of the problem in one page.
2. `.claude/skills/deep-research/SKILL.md`, the procedure, including the Windows preamble and the per-agent scratchpad rule.
3. `scripts/gantry/ann-blackbox/bla_init.py` docstring and lines 50-60, what the BLA does and why `SS_f` selection matters.
4. `docs/rollout-stability-literature.md`, the closest existing sweep, to avoid rediscovering it.
5. `scripts/gantry/drift-fix-trials/research/thread-CD-bias-through-integrator.md`, the only prior thread on integrators specifically.

## 12. Do not

- Do not compress this handoff into a keyword query. That is the documented failure mode of the skill.
- Do not cite the invalid MSD numbers `0.618` and `0.999` (section 6).
- Do not repeat "the window is shorter than the system memory" as the mechanism; `nf = 3700` refutes it.
- Do not fetch publisher URLs; go to `locations[]` and the resolution order.
- Do not run the TU/e browser preflight inside a subagent; do it once in the parent.
- Do not write code or launch training runs. If the literature suggests an experiment, name it in one sentence and stop.

## 13. Operational

No training run, no conda environment needed beyond the skill's own curl and python snippets.
`export PYTHONIOENCODING=utf-8` before every snippet; write helper scripts to a per-agent
subdirectory of the session scratchpad, never `/tmp`, and never name one `enum.py` or `json.py`.

Budgets, because dblp IP-blocks after ~10 queries and OpenAlex has a shared daily spend cap:
with four subagents, allow each **2 dblp queries** and **~12 OpenAlex queries**, and require the
`assert 'error' not in d, d` guard on every OpenAlex parse, since a 429 body renders as zero hits.

Output goes in the reply. A new `docs/*.md` for the sweep needs the user's explicit permission
first, per the no-new-files rule; ask in one line before writing one.

## 14. Delegation

**Four subagents, one per sub-question, with sub-questions 1 and 5 merged into one agent** since
they are the same literature seen from two sides. Give each the frame, its own sub-question, the
budgets above, the per-agent scratchpad instruction, and the requirement to grade its own negative
claims (a "nobody has done this" from an agent whose OpenAlex calls were rate-limited is
provisional and must be reported as such). Merge the logs and deduplicate by DOI.

Run one `search_google_scholar` query per sub-question after enumerating, written as the sentence
you expect the paper to contain rather than as keywords. It is the only route that indexes full
text, and therefore the only one that reaches an in-body scope disclaimer, which is where "nobody
has treated this case" is actually written.
