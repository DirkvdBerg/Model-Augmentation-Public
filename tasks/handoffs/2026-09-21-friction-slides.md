# Handoff: discuss how the friction implementation is presented and justified on the slides
**From**: session of 2026-09-21 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task
Act as a critical interlocutor on the friction slide deck for the supervisor meeting. The
implementation is finished and verified, and the five panels in
`scripts/gantry/meeting/meeting-21-09-2026/slides/friction-slide-panels.tex` are written and
compile. The job is NOT to build anything: it is to interrogate how the work is stated and whether
each choice is defensible to a supervisor who knows Garcia's paper. Go through the deck claim by
claim and say which are load-bearing, which are over-claimed, which are under-justified, and which
invite an obvious attack the slide does not preempt. Where a claim is weak, propose the sharper
wording. Section 8 names the ones that most need this. Panel edits are in scope; the
implementation is not.

## 2. Out of scope
- **Do not modify the friction implementation.** `gantrySystemExtendedCoulomb.m` is verified
  against Garcia term for term (section 4). No presentation question requires changing it.
- **Do not regenerate any dataset.** All three friction datasets exist and are gated.
- **Do not rewrite `docs/decisions.md` D-204 or `docs/references.md`.** Both are current. Propose
  amendments in the reply if the discussion changes a stated reason; do not edit them unasked.
- **Do not start the training runs.** Server state is section 13; the user launches them.
- **The figure is a separate, lower-priority thread.** `plot_baseline_error_friction.m` produces a
  confounded comparison (section 8b). Do not spend the session on it unless the user redirects.
- **Do not write a new `docs/*.md`.**

## 3. Where things stand
Branch `Augmentation`, last commit `5aa2c89` ("Orthogonal by construction meeting"). Tree is dirty
in many directories from prior work; this session's changes are in
`Matlab-scripts/Augmentation-coulomb/`, `Matlab-scripts/Augmentation-telica-profiles-coulomb/`,
`scripts/gantry/meeting/meeting-21-09-2026/`, `scripts/gantry/gantry_interconnect_dynamic.py`,
`scripts/gantry/GPU/gantry_interconnect_dynamic_gpu.sh`,
`scripts/gantry/orthogonal-by-construction/runners/run_obc_arm.sh`, `docs/`.

**Nothing is committed.** No job is in flight.

## 4. Established and verified
Facts the discussion may treat as settled.

- **The friction matches Garcia term for term.** Law from the Fig. 2 labels; transform
  `X1 = X + sin(Theta)*Lb/2` from Eq. (2); friction lives in the generalised force vector,
  confirmed because Eq. (5) Rayleigh dissipation holds only the viscous coefficients; our
  `u_eff` expands exactly to his `F_Xs` and `F_Theta_s` in Eq. (15); `cc1` and `cc2` come from
  separately measured `F1`, `F2` per Eqs. (18) and (19).
- **The `cos(Theta)` small-angle step is Garcia's own**, not ours: his simplified `M_s` (12) and
  `C_s` (13) carry no `cos(Theta)`, and `C_s` matches our `C4` term for term.
- **Gate 1**: `cc1 = 16.800000` (err `3.6e-15`), `cc2 = 18.350000` (`0`), `ccy = 11.600000` (`0`),
  all affine fit residuals at round-off. Theta torque `Lb/2*(cc1-cc2) = -0.5619 Nm` predicted,
  `-0.561875` measured.
- **Gate 2** at production knobs: census sliding 154451 / held 10163 / broke away 15386 /
  **neither 0**; worst invariant `2.6e-13 N`; slip dissipation exactly `0`.
- **Gate 3** passes at `V_EPS_MARGIN = 9` under a corrected criterion: perturbation gain `1.348`
  friction against `1.332` frictionless, ratio `1.012` on the multisine record; `1.61` to `1.71`
  on three Telica records with zero collocation violations.
- **Absorber output contribution**: preserved on the multisine set (`72.6%` to `73.9%`), cut to a
  quarter on Telica (`2.43%` to `0.62%`) while `rms|y|` is unchanged to four digits.
- **Datasets complete**: 22 + 7 + 29 (merged); limiter never fired; peak force `1327/1385/547 N`
  against `[2000 2000 1420]`.
- **The deck compiles**: 5 pages, one panel per page, no overfull boxes.

## 5. Assumed but not verified
- **Eq. (15) is the only place Coulomb appears in Garcia's force vector.** Inferred from its
  absence in `T`, `V` and `D`, and confirmed by the expansion matching. The full EOM statement was
  never read in the PDF; the OCR of that section is mangled.
- **`E1_resonance_sweep` degeneracy affects only test-time reporting.** Its X rails are frozen at
  `3e-18 m` under a `4e-09 N` drive, which is measured. The consequence for `per_record_nrms` was
  reasoned from the code (`ystd` is global, so no divide-by-zero), not observed in a run.
- **Y is the right channel for the figure**, and `t = 2.0 s` a representative window. Both were
  chosen, not justified.
- **Both figure records sit at `Y_op = 0`** so one `Cfb` serves them. Read from the `Y_OP` map in
  `controller.py`, asserted in the script, not independently confirmed against the generator.

## 6. Tried and failed
- **Open-loop replay for the baseline residual** -> frictionless residual `1.82e-04 m` on T3
  against a signal of `1.47e-05 m`, twelve times the trajectory -> X and Y have `K = 0`, so any
  force error integrates without restoring force over a 6 s free run; it also made the multisine
  invisible because the axis had to span the drift -> superseded figures
  `baseline_error_T3_multisine.png`, `baseline_error_TP1_telica.png`.
- **Locking the absorber by stiffening `ka` by 1e6** to build a no-absorber baseline -> NaN,
  singular mass matrix -> puts a 150 kHz mode into a 20 kHz explicit RK4 -> use the massless limit
  (`ma -> 1e-4*mh_total` at fixed `mh_total` and fixed 150 Hz), which both replay scripts now do.
- **Presenting Garcia's identification as three experiments** (Y, X-symmetric, X-antisymmetric)
  -> wrong -> he runs **two** and reads `F1`, `F2` separately; his antisymmetric motion identifies
  the joint viscous friction `cb`, not Coulomb -> Eqs. (18), (19).
- **Gate 3 passing only on zero collocation violations** -> unsatisfiable by construction
  (`0.03%`, `0.24%`, `0.10%` at margins 1, 3, 9) -> growing the band pulls more rails inside it, so
  the population at the edge scales with the band -> D-204 amendment (b). See 8(a): this is also
  the deck's most attackable point.
- **Comparing `delta_a` against the `0.12%` from job 81088** -> invalid -> that is a trained-model
  ablation and `delta_a` is an internal state; the comparable quantity is the absorber's output
  contribution.
- **Putting `P` and `u_eff` on a slide as Garcia's formulation** -> neither appears in his paper
  -> they are our compact rewriting of Eq. (15); panel 1 now uses his own expressions.
- **Patching `.m` files through Python heredocs** -> `\n` became a real newline three times, each
  costing a failed run -> use the Edit/Write tools.

## 7. Achieved
- `friction-slide-panels.tex` + `.pdf`: 5 panels, house style, Garcia's own equations only.
- Five verification scripts in `Matlab-scripts/Augmentation-coulomb/`: gates 1 to 3,
  `coulomb_v_eps.m`, `measure_absorber_contribution.m`. Implemented **and validated**.
- `plot_baseline_error_friction.m`: implemented, closed loop and windowed, **not validated** as a
  meeting artefact (8b).
- Three friction datasets generated and gated; `cfg.mode` points at the merged one.
- D-204 with three amendments; `references.md` friction literature section.

## 8. The open question
**Which of the deck's claims would not survive a sharp supervisor, and how should they be
restated?** Four candidates, in the order I judge them attackable.

**(a) Gate 3's pass criterion was changed after it failed, and the deck hides this by omission.**
The gate originally passed only on zero collocation points leaving the stick band. It failed at
margins 1, 3 and 9. The criterion was then replaced by worst-excursion, rate, and a
perturbation-gain check, and it passed. The defence is real and recorded in D-204 amendment (b):
the binary form is unsatisfiable by construction, and `leine1998` warns about conditioning, which
the gain ratio of `1.012` measures directly. But "moved the goalposts, then passed" is what it
looks like without that framing, and none of it is on a slide. Decide whether it belongs on the
backup panel, and in what words.

**(b) The figure's comparison is confounded.** Closed loop, windowed at `nf` = 0.1 s, Y channel:
T3 no friction `3.39e-05 m` against friction `4.88e-06 m` (`0.1x`, wrong direction); TP1 no
friction `9.24e-07 m` against friction `3.33e-06 m` (`3.6x`). T3 goes the wrong way because
friction damps that record's motion about 5x so its absolute residual falls with it, and because
the absorber carries about 73% of T3's output while the baseline has none. Normalising by each
dataset's own signal over the same windows would separate those. Not done.

**(c) The absorber asymmetry is a finding the deck does not mention.** Friction preserves the
absorber's output contribution on the multisine set and cuts it to a quarter on Telica while
leaving the trajectory unchanged. That is a statement about identifiability under operational
motion, and arguably more interesting than anything on the current panels. Decide whether it
belongs in this deck or a later one.

**(d) "Truth only, baseline frictionless" is asserted, not defended.** The literature's default is
the opposite: Bolderman keeps `fc*sign(v)` in the physics block, Kon's demo puts viscous damping
there. Panel 5 gives one clause. A supervisor may reasonably ask why the harder configuration was
chosen.

## 9. Next action
Read the deck, D-204 with its three amendments, and the `README.md` gate table, then produce a
claim-by-claim critique of the five panels: for each claim, whether it is verified, over-claimed or
under-justified, and the sharper wording where it is weak. Start with 8(a), since it is the point
an informed supervisor is most likely to press and the deck currently says nothing about it.

Format the reply as one line per claim, panel by panel:
```
P1  "friction enters at the force node only"   VERIFIED (Eq. 5 holds only the viscous terms)
P1  "M, C, K unchanged"                        VERIFIED (gantrySystemExtendedCoulomb.m:249)
P5  "friction in the truth only"               UNDER-JUSTIFIED -> name the alternative and why rejected
```

## 10. Acceptance criterion
Done when every claim on the five panels is labelled verified, over-claimed or under-justified with
its evidence pointer, and each weak one has proposed replacement wording. The criterion is coverage
of the deck, not a numerical target: this is a presentation task and the underlying measurements
are already accepted (section 4).

The test to apply per claim: a claim must trace either to a measurement in section 4 or to a
specific equation in Garcia. A claim that traces to neither is over-claimed by definition.

## 11. Read these first
1. `scripts/gantry/meeting/meeting-21-09-2026/slides/friction-slide-panels.tex` - the subject of
   the discussion; five panels, read in full first.
2. `docs/decisions.md`, D-204 and its three amendments - every choice on the slides and the reason
   given. Amendment (b) is the defence for 8(a).
3. `Matlab-scripts/Augmentation-coulomb/README.md` - gate results table; every number a panel
   quotes is here with its provenance.
4. `literature/gantry/garcia.txt` - OCR, read with `tr -d` of the null byte. Fig. 2 labels and
   Eqs. (1), (2), (5), (12), (13), (15), (18), (19) are what the slides rest on.
5. `Matlab-scripts/Augmentation-coulomb/gantrySystemExtendedCoulomb.m` header - the in-code
   justification for Karnopp, the stick band and the frame.

## 12. Do not
- Do not reintroduce open-loop replay for the residual (section 6).
- Do not lock the absorber by stiffening `ka` (section 6).
- Do not present Garcia's identification as three experiments (section 6).
- Do not put `P` or `u_eff` on a slide as Garcia's formulation; neither is in his paper.
- Do not patch `.m` files through Python heredocs (section 6).
- Do not delete `baseline_error_T3_multisine.png` or `baseline_error_TP1_telica.png` without
  saying so; they are superseded open-loop figures the user has not agreed to remove.

## 13. Operational
MATLAB R2025a is on PATH; the gate and figure work is MATLAB, not Python. Slides compile with
`pdflatex -interaction=nonstopmode -halt-on-error friction-slide-panels.tex` from the slides
directory. Figures land in `scripts/gantry/meeting/meeting-21-09-2026/figures/`. A closed-loop
windowed figure run takes a few minutes.

Server, not needed for this task but currently blocked on it: `cfg.mode` in
`gantry_interconnect_dynamic.py` and the cache-key fixes in `run_obc_arm.sh` and
`gantry_interconnect_dynamic_gpu.sh` are **uncommitted**. Without the push, the three OBC arms fail
with `GLIBC_2.34 not found` (job 85010). `/data/` is gitignored, so the 520 MB merged dataset must
be rsynced separately.

## 14. Delegation
None. This is a reading-and-judgment task over the five files in section 11. An Explore subagent
would add cost and no coverage.
