# Handoff: a literature-grounded BLA initialisation of the augmented states, implemented, trained, and viable under noise and on Telica

**From**: session of 2026-08-22 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Replace the augmented-state initialisation with one built on the **Best Linear Approximation of the
baseline residual** `r = y_data - y_baseline_sim`, **implement it, train it, and show it reaches the
performance the discarded mechanism reached** - with every design choice traceable to a paper or
explicitly derived.

The mechanism being replaced is a residual peak-picking recipe with **no literature source and four
unexplained constants**. The replacement must not reintroduce that: **no heuristics.**

**Order of work: literature, then the residual, then the fit, then the implementation.** Do not open
with a training run (section 6).

### The work lives in three documents, not in this handoff

`scripts/gantry/BLA-Augmentation/` was created for this task. **Read its `README.md` first.**

| file | role | state |
|-|-|-|
| `EVIDENCE.md` | 12 claims to verify, each with its PDF path and page. **Machine-verified quotes only** | skeleton written; claim 4 already CONFIRMED, 11 PENDING |
| `DESIGN.md` | the 10 decisions D1-D10, each with Goal / Why / Held sources / Done when, and an empty `Decision:` block | skeleton written; all not started |
| `RESULTS.md` | measured outcomes, each with artefact path and command | not created yet |

**Those files are both the agenda and the deliverable.** When every `Decision:` block in `DESIGN.md`
is filled and every claim in `EVIDENCE.md` has a verdict, the design exists. This handoff does not
restate their contents; it says what the task is and what "done" means.

### Two phases, and the boundary is a safe stopping point

**Phase A (D1-D8) - the fit. No production file is touched.** Runs on the clean tree with harnesses
that already exist. Output: a reduced realisation with an error bound, validated out of sample and
under noise.

**Phase B (D9-D10) - the implementation.** Restore one file, write the block as NEW code, verify
D-072, train, ablate.

**If context runs out, stop at the A/B seam and ask whether to write Phase B's handoff.** Phase A's
output is self-contained, and a half-filled `DESIGN.md` is self-describing state. D-133:
user-triggered only.

### Quoting is machine-checked, not transcribed

Every quote must pass
`python scripts/gantry/BLA-Augmentation/verify_pdf_quote.py <pdf> <page|any> <quotefile>`.
Only `MATCH OK` may be recorded CONFIRMED. Rationale, failure modes and the verdict vocabulary are
in `EVIDENCE.md`; the checker was verified in both directions on 2026-08-22.

**Two of the twelve claims need their PDF fetched first**: `marconato2014init` (arXiv:1804.08654)
and `relan2017lpm_bla` (arXiv:1805.06237). Check page 1 against the expected title before quoting -
a DOI fetch returned an unrelated paper during the 2026-08-22 sweep.

### A standing consideration, NOT a task item

The thesis contribution is orthogonal-projection regularisation for physical-parameter
interpretability, and `dV_orth/dp` is currently exactly `0.0` for every augmented parameter
(`runs/orth_gauge_probe.json`), i.e. **the penalty is blind to the dynamic augmentation.** Prefer a
parameterisation that does not widen that gap; note in one line if a choice would. **Do not fix or
extend the penalty** (section 2).

## 2. Out of scope

* **Do not modify** `kamtin-fp-model/`.
* **Do not restore the archive wholesale.** Section 3 gives the one file needed.
* **Do not run the wave-1 ablation factorial, or any arm of it.** Designed before the BLA decision;
  its band arms test a mechanism being deleted and its seed arms measured DRAW reliability, of which
  a BLA has none. The runners stay on disk as a record.
* **Do not extend the orthogonal-projection penalty.** `gantry_dynamic/orth_penalty.py` is another
  session's. One paragraph, then stop.
* **Do not touch the full black-box comparison.** The user is building it separately.
* **Do not run a broad BLA literature search.** Mapped in section 11; a narrow sweep already ran and
  its findings are persisted in `tasks/ablation-2026-08-22-what-earned-its-place.md`, section "BLA
  deep-research sweep".
* **Do not implement a Telica arm.** `kamtin-data/Data Telica/` is blocked by policy. Telica is a
  design constraint (`DESIGN.md` D3), not a dataset to touch.
* **Do not overwrite `runs/cl_residual_spectrum.json`** without `CL_RS_OUT`; it was silently
  clobbered once (section 6).

## 3. Where things stand

Branch `Augmentation`, last commit `4cdb7c1`. Nothing committed or pushed, by design. Nothing in
flight.

**The production files were reset on 2026-08-22, deliberately.** `model_augmentation/`,
`gantry_dynamic/` and `gantry_interconnect_dynamic.py` match `4cdb7c1` exactly. 722 insertions of
experimental machinery had accumulated, `model.py` alone at +334 with ten env gates in the
production build path.

**Archived at `tasks/snapshots/2026-08-22-working-implementation/`**: 12 files (11 verified
byte-exact) and 8 patches (all verified to apply against `4cdb7c1`). **Read its `MANIFEST.md` before
assuming anything about the tree. It is the only complete record.** 247 of those lines are ANOTHER
SESSION's P1/P1-e work, discarded from the tree and recoverable only from there.

### Phase B needs exactly ONE file restored, and this was measured

| file | needed? | why |
|-|-|-|
| `model_augmentation/fit_systems/closed_loop.py` | **YES** | 345 lines of real code: the D-147 rollout, `xc = 0` windowing, and `window_starts` / `make_window_tensors`, which **do not exist at `4cdb7c1`** and which `cl_train.py` imports by name |
| `model_augmentation/fit_systems/pre_encoder.py` | **NO** | its +47 lines are **comments only, zero code changed**. `4cdb7c1` already has the full `W^a` block |
| the `AUG_LRU_NA_NB` env gate | **NO** | **`cfg.na_nb_override` already exists at `4cdb7c1`** (`config.py:73`) and flows through `hp['na_nb']`. `dataclasses.replace(CFG, na_nb_override=17)` does the same job |
| everything else in the archive | **NO** | that is what is being replaced |

**So the clean baseline survives Phase B.**

**Live and intact** (`closed-loop-controller/` was not reset): `cl_train.py` with `CL_SEED` and the
`checkpoint.best` record; `cl_residual_spectrum.py` with the noise gates and provenance blocks; all
of `transient-investigation/` including the five BLA harnesses. `cl_train.py` currently fails to
import **only** because of the `closed_loop.py` restore above.

**One reset consequence is silent**: `closed_loop_free_run_rms` and `closed_loop_rollout` still
exist at `4cdb7c1` in their pre-D-147 form, so anything run before the restore returns numbers
**not comparable** with `3.795974e-07` or the D-072 gate. Restore first, measure second.

## 4. Established and verified

Free-run validation RMS on V1-V4, `closed_loop_free_run_rms`, closed-loop rollout, `k0 = 17`.

| fact | value | evidence |
|-|-|-|
| **The residual BLA locates the mode to `+0.001 %`** | `157.8946 Hz` vs truth `157.8937`; `zeta 0.05257` vs `0.05276`; `rho 0.987029` vs `0.986982` | `transient-investigation/runs/residual_bla.json`, MISO ARX, `n_A = 28`, 18 records x 3 channels. **Re-run after the reset and bit-identical** |
| the peak-picker it replaces | band `[149.90234, 164.06250] Hz`, 14.2 Hz wide, bin-quantised at `0.48828 Hz`; fit `158.203 Hz` = `+0.20 %` | `runs/cl_residual_spectrum.json` |
| order selects itself off a stabilisation diagram | median `f` per `n_A`: 4 -> `157.221`, 16 -> `157.848`, 28 -> `157.8946`; `zeta` degrades past 28 | same |
| **frequency is noise-immune; ARX biases DAMPING** | `f` err at 1x/10x/100x Telica sigma: `+0.000/-0.004/-0.008 %`. `zeta` err: `+0.5/+4.8/+8.8 %` | `transient-investigation/runs/residual_bla_noise.json` |
| **IV removes the damping bias** | `zeta` err IV `-0.0/+0.4/+1.9 %`; 12x better at 10x, and 2x more stable across order | same |
| the peak-picker threshold is unsafe on coloured backgrounds | family-wise `alpha=0.05` threshold with a global-median floor: `6.0 dB` white, `25.1 dB` pink, `46.6 dB` at `1/f^2`. It used `10 dB` | `probe_peak_threshold_mc.py` |
| a LOCAL running-median floor makes it background-independent | `6.6-6.9 dB` across `a in [0,2]` | `probe_peak_threshold_localfloor.py` |
| **`AUG_LRU_B` was per-DRAW, not per-dataset** | seeds 0-4 at `nz=17`: spread **20.1 %** | `probe_recal_ba_scale.py` |
| the discarded mechanism's results | seed 0 `3.795974e-07` / ablation `5.2081x`; seed 1 `4.8867311476e-07` / `4.5807x` | `runs/cl_train_s0_seed1.json`, `arm_ablation_s0_seed1.json` |
| pole proximity does NOT predict outcome | seed 1 drew a pole `0.01 Hz` from truth, closer than seed 0's nearest, and landed 29 % WORSE | same |
| blinding the trained model to `x_a` is worse than untrained | `2.238e-06` vs untrained `2.1866011034e-06` | same |
| both ablation surfaces agree to 4 s.f. | `4.5807x` vs `4.5806x`: `x_a` reaches `y` ONLY through the ANN | same |
| **`pre_encoder.py`'s "no literature source" comment was FALSE** | Hoekstra arXiv:2602.17297 p.9 Sec. 5.4.2 Eq. (31) specifies **Xavier**; the code called `kaiming_uniform_` | read at the PDF; **D-152** |
| Orvieto provenance verified line by line | Lemma 3.2 p.7, Sec. 3.3 p.8, Eq. (7) Sec. 3.4 p.10 all MATCH. **`theta_log = log(theta)` is NOT Orvieto**. The 2x2 real Jordan block is **App. E.3 Eq. (26)**, not Forgione Remark 1 | read at the PDFs |

### The current routing, as the comparison baseline

From `build_model` **as it was before the reset**. A BLA design must keep this or replace it
explicitly; `DESIGN.md` D9 owns the decision.

* **ANN input** `z = [x, u]`, `nz = 17` at `nx_aug = 8`. **ANN output** one row per routed state row,
  connected `"additive"`; full routing `0..13`.
* Rows `0..5` are physical corrections, **exactly zero at init** (D-072). Rows `6..` were the
  augmented states.
* **`B_a` acted on the whole `z`**, augmented columns zeroed. **A BLA gives `B_r` acting on `u`
  ALONE.**
* **`Cd_norm` has ZERO columns on the augmented rows.** `x_a` reaches `y` only via the ANN reading
  it and writing into rows `0..5`. **That is the "delay": one state update, ANN-mediated** - a
  consequence of `C = 0`, not a design feature, confirmed by the two ablation surfaces agreeing to
  four significant figures.
* **D-068** originally restricted routing to `K>0` rows because `K=0` position rows integrate a
  correction with no restoring force.

## 5. Assumed but not verified

**The BLA-init literature findings this design rests on are SECOND-HAND**, from a subagent's report;
nobody on this project has opened `schoukens2020lfr` or `schoukens2021ssnn_init` (except claim 4).
**They are listed as claims in `EVIDENCE.md`, not as facts here, because that is what they are.**
Full text: the ablation log's "BLA deep-research sweep" section, which also carries the
do-not-quote list and the sweep's own sympy-verified derivations.

Genuinely open, beyond the literature:

* **Which regressor the prototype used.** `residual_for` passes `sd.u`, the EXTERNAL excitation, not
  the realised loop input - so direct-method bias may be weaker than the sweep assumed. Except under
  `CL_NOISE_CONSISTENT=1`, which correlates `u_data` with the noise by construction. Load-bearing
  for `DESIGN.md` D2 and D3.
* **Why our zeroed readout is a dead zone when the literature's is not.** With `C = 0` the gradient
  to `W^a`, `B_a`, `nu_log`, `theta_log` is exactly `0.000e+00` at init (D-130). Claim 4 (CONFIRMED)
  says a zero readout trains from step one **provided the hidden weights are random and the states
  feeding it are excited**, and that the degenerate case is **both** layers zero - **which was
  exactly our configuration**. Candidate answer: follow the paper. **This makes a random `W^a`
  newly attractive for a literature reason**, not the noise reason it was argued from before.
  Settled by D7, then the first training run.
* **That BLA init composes with closed-loop training.** Every BLA-init paper held re-estimates in
  **open loop**. `DESIGN.md` D10 owns it, with a falsifier.

## 6. Tried and failed

* **The band recipe itself** -> worked in simulation, but no literature source, four unexplained
  constants, a threshold unsafe on coloured backgrounds, and it returns a **zero-width band from a
  single peak at 2000x noise while reporting success**. Replaced, not repaired.
* **Freezing `A_aa` as a design principle** -> proposed on "trained poles move under `0.15 Hz`" ->
  **no literature supports it**: Marconato, `schoukens2020lfr` and Orvieto all RE-ESTIMATE after
  init; reservoir computing freezes only *random* dynamics. -> The defensible version is
  **regularising toward the BLA** (`DESIGN.md` D9).
* **Opening with an F1 training run** (eight plain ANN latent rows, no `A_aa`) -> proposed and
  withdrawn 2026-08-22 -> it is not a BLA experiment, and at `4cdb7c1` it is the **double-zero**
  case (zero-init ANN readout AND zero `W^a`) that claim 4 identifies as untrainable, so a negative
  would say nothing about whether missing dynamics exist. **That question is answered by the
  residual analysis, parametrically and already.** Do not revive it as a warm-up.
* **Running `cl_residual_spectrum.py` with no arguments** -> its default record set is `VAL_FILES`
  (4) but the artefact was made with `CL_RS_FILES=all` (18), so a 4-record band silently replaced
  the 18-record one -> every later arm would have been initialised from a different band with
  nothing in any log saying so -> restored, verified byte-identical, `CL_RS_OUT` added.
* **`CL_PROBE=0` without `CL_CONCURRENT=0`** -> validation froze at the untrained value for 260
  updates -> `cl_train.py:130` makes disabling probes ENABLE concurrent subprocess validation.
* **`na_nb` above 17** -> D-072 fails monotonically -> float32 conditioning of `W^b = A^n O_n^{-1}`.
* **`hash()` for a reproducible per-record seed** -> salted per process -> use `zlib.crc32`.
* Four mechanisms that worked and moved nothing: D-150 live `A_aa` alone (`-0.665 %`), burn-in
  `K=100`, the multiple-shooting defect (**degenerate**: descent SHRINKS `W^a`), the `na_nb` sweep.

## 7. Achieved

**Live on the clean tree**: `probe_residual_bla.py` (re-verified post-reset, `157.8946 Hz`),
`probe_residual_bla_noise.py` (ARX vs IV at 0/1/10/100x sigma), `probe_recal_ba_scale.py`,
`probe_peak_threshold_mc.py`, `probe_peak_threshold_localfloor.py`, and `cl_residual_spectrum.py`
with `CL_RS_NOISE_CONSISTENT`, `CL_RS_SIGMA_SCALE`, `CL_RS_OUT` and provenance blocks. Plus
`BLA-Augmentation/` with `README.md`, a tested `verify_pdf_quote.py`, and the `EVIDENCE.md` /
`DESIGN.md` skeletons.

**Archived, not live**: everything in section 3's table marked NO. `CL_SEED` and `checkpoint.best`
are live in `cl_train.py`, which was not reset.

## 8. The open question

**Does the design survive verification of its own foundations?**

D-072-plus-a-live-input-path, and therefore the whole "`C_r` is discarded, `A_r` and `B_r` are used"
decision, rests on claims 1-3 in `EVIDENCE.md` - **and nobody on this project has opened those
PDFs.** If any comes back `REFUTED`, `DESIGN.md`'s "already decided" block reopens and the wiring
question in D9 changes shape.

Candidate answers: CONFIRMED, in which case Phase A proceeds as written; or REFUTED, in which case
stop and re-scope rather than designing around it. The evidence that chooses is twelve page reads.

## 9. Next action

**Write `EVIDENCE.md`**: verify the twelve claims with `verify_pdf_quote.py`, filling in the verdict,
quote, page and checker output for each. Fetch the two unheld PDFs first.

Then work `DESIGN.md` in its stated order, D1 through D8, filling each `Decision:` block. **D1 comes
first among those**: it asks whether `r` is the residual or partly the controller error, and a
failure there invalidates every BLA number already recorded, including `157.8946 Hz`.

Rationale for evidence-before-everything: the four citations that WERE checked at the PDF on
2026-08-22 yielded three corrections, one a false "no literature source" comment sitting in shipped
code (D-152). **A REFUTED verdict is a success** - it catches a false foundation for twelve page
reads instead of a re-design later.

## 10. Acceptance criterion

### Phase A
1. **Defensibility.** Every filled `Decision:` block cites a claim with verdict `CONFIRMED` in
   `EVIDENCE.md`, or carries a written-out derivation. **A citation that exists only in this
   handoff, in `docs/references.md`, or in a subagent report does not count.**
2. **The fit is validated.** Out-of-sample VAF per order, **clean and at 1x Telica sigma**, order
   chosen by D5's error bound and cross-read against D8's turnover.
3. **Real-system viability is stated, not assumed.** D3's four constraints each marked
   handled-by-design or explicitly deferred.

### Phase B, tiered - a single pass/fail at e-7 would misread a good result

Free-run validation RMS on V1-V4, 520 updates, `na_nb = 17`, serial validation, with the ablation
worst-surface ratio from `probe_arm_ablation.py`:

| | free-run RMS | reading |
|-|-|-|
| **minimum** | `< 1.3933793e-06` | beats the long-standing plateau; the BLA init is doing something |
| **target** | **`3.80e-07` - `4.89e-07`, ablation `>= 2.0x`** | matches the discarded mechanism's measured seed band, with the augmented states load-bearing |
| **stretch** | `< 3.795974e-07` | beats the best draw the band recipe ever found |

The band is data-derived: the observed spread over the only two pole draws ever run.
`# HEURISTIC: 2.0x`, carried forward unchanged from C8 so the comparison is like-for-like.

**Under noise**: the same run with `CL_NOISE_CONSISTENT=1
CL_NOISE_SIGMA="8.544e-9,7.762e-9,6.539e-9"` must keep ablation `>= 2.0x`. That is simultaneously
the first training-level test of the stabilized-PEM rollout.

## 11. Read these first

1. **`scripts/gantry/BLA-Augmentation/README.md`**, then `EVIDENCE.md` and `DESIGN.md` in that
   folder. **They are the agenda and the deliverable.**
2. `docs/references.md` sections "Best Linear Approximation...", "BLA-based initialisation of
   nonlinear models", "Closed-loop identification"; then
   `scripts/gantry/ann-blackbox/BLA-LITERATURE.md`. **The literature is mapped; this is the map.**
3. `tasks/ablation-2026-08-22-what-earned-its-place.md` - sections **"BLA deep-research sweep"**
   (persisted findings, the sweep's derivations, the do-not-quote list), "RESIDUAL BLA",
   "The `over_floor_db > 10` threshold, DERIVED", "DEVIATIONS".
4. `tasks/snapshots/2026-08-22-working-implementation/MANIFEST.md` before assuming anything about
   the tree.
5. `docs/decisions.md` D-152, D-151, D-150, D-147, D-130, D-072.

## 12. Do not

* Do not open with a training run (section 6).
* Do not restore more of the archive than `closed_loop.py` (section 3).
* Do not reintroduce a threshold, a band, or a random pole draw. That is the mechanism being removed.
* Do not apply `gamma` to a fitted `B_r`: `gamma = 0.161` at `rho = 0.987` and would scale a fitted
  gain down 6x. It belongs only on a randomly-initialised nonlinear term (Orvieto Prop. 3.3).
* Do not use plain ARX under noise; use IV. ARX biases `zeta` by `+8.8 %` at 100x sigma.
* Do not use `[min, max]` across record-channels as an aggregation rule; at `n_A = 28` it gives
  `[134.4, 170.9] Hz` while the median is exact.
* Do not take initialisation poles from a dual-Youla `Q_hat`: its poles are the closed-loop
  characteristic polynomial and the missing dynamics are its ZEROS. Use `Delta_hat = P_hat - P0`.
* Do not simulate the baseline inside the loop when forming a residual for a ratio estimator.
* Do not set `na_nb` away from 17. Do not compare arms at different update counts.
* Do not run with `CL_PROBE=0` unless `CL_CONCURRENT=0` is also set.
* Do not commit or push `model_augmentation/` or `gantry_dynamic/`.
* Do not re-derive the Orvieto, Hoekstra or Forgione citations; checked at the PDF, see section 4
  and D-152.

**Citation corrections to apply when next editing those files**: `BLA-LITERATURE.md` Sect. 4.5 has
Sjoberg & Schoukens as Automatica 47(11):2481-2489, 2011 - it is **48(2):353-359, 2012**, DOI
`10.1016/j.automatica.2011.07.007`; its `tiels2015` entry has a wrong title AND DOI (the paper is
"Initial estimates for Wiener-Hammerstein models using phase-coupled multisines", Automatica
60:201-209, DOI `10.1016/j.automatica.2015.07.020`). `docs/references.md` marks
`schoukenstiels2017survey` "not held"; it is now held.

## 13. Operational

Env `GraduationProject`. Live-output convention per `CLAUDE.md` for anything over a few seconds.

**Phase A costs almost nothing**: residual caches regenerate in ~70 s per noise level, and a full
ARX+IV sweep over 8 orders x 54 record-channels takes ~100 s. Residuals from
`cl_residual_spectrum.residual_for(fname, cfg)`; `CL_RS_FILES=all` for 18 records; noise via
`CL_RS_NOISE_SIGMA="8.544e-9,7.762e-9,6.539e-9"` with `CL_RS_NOISE_CONSISTENT=1` and
`CL_RS_SIGMA_SCALE`. **Always pass `CL_RS_OUT`** for anything that is not the canonical artefact.

**Phase B**: ~85 min per training run locally at 6.5 s/it, ~30 min on `kauai` (1.55 s/it, run 76573).
Ablation ~25 min without `PROBE_PERPAIR`. Checkpoints land in
`C:\Users\20203253\AppData\Local\deepSI\checkpoints\`; `cl_train.py` records the exact path in its
result JSON under `checkpoint.best`. Server deployment: `runners/DEPLOY-wave1.md` gives the file
list and the verification commands; **nothing is pushed, the user copies by hand**.

Read the ablation ratio before the RMS: an arm with a good RMS and a ratio near `1.0` is a NEGATIVE.

## 14. Delegation

**One `deep-research` subagent per `DESIGN.md` decision that needs a source not held, ceiling two
concurrent.** Raised from "none": ten decisions each needing literature grounding is more than one
context should carry, and the 2026-08-22 sweep returned findings that changed the design twice.
**Do NOT delegate `EVIDENCE.md`** - verifying a quote is the one thing that must be done by whoever
writes the design. No Explore subagent: every file this task needs is named in section 11.
