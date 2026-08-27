# Handoff: explain the MSD-offset result and every figure in its folder

**From**: session of 2026-07-31 | **Branch**: Augmentation | **Effort suggested**: medium

## 1. Task

Act as the explainer for the completed MSD-offset investigation. The user needs to understand,
and then present to a supervisor, **why replaying the MSD plant's recorded input open loop
through the 6-state baseline produces a bounded position offset**, and what each of the 21 files
in `scripts/gantry/msd-offset/figures/` shows. Answer in plain physical language first and
equations second; the user has said repeatedly that matrix-first explanations do not land. Be
ready to rebuild or restyle any figure on request: every figure regenerates from a script in that
folder plus cached artefacts, and no figure needs a training run. The investigation itself is
finished and its conclusions are settled; the work here is exposition, one figure at a time, plus
whatever figure edits the user asks for.

## 2. Out of scope

- **`scripts/gantry/drift-isolation/`**: do not read for this task and do not modify. That thread
  is the *unbounded* drift the augmentation introduces; this one is the *bounded* offset the
  baseline shows with the ANN at zero. `CONCLUSIONS.md` there was read in the previous session
  and the only cross-link that matters is already in section 4.
- **Regenerating the MATLAB records.** A user decision, not a step here.
- **The eight documentation corrections** to `scripts/gantry/msd-offset/ISSUE.md` and
  `docs/msd-offset-mechanism-2026-07-29.md` (listed in section 5). They are real and outstanding,
  but the user asked for an explanation session. Mention they exist if the user opens either
  file; do not start them unprompted.
- **Any augmentation training run.** Nothing here needs one.
- Do not modify `scripts/gantry/gantry_dynamic/*` or anything under `kamtin-fp-model/`.

## 3. Where things stand

Branch `Augmentation`, last commit `8022544`. Nothing is in flight. Tree is dirty:
`scripts/gantry/msd-offset/` is entirely untracked (new this thread), plus
`docs/writeup/offset-mechanism-{equations,derivation,preview}.tex` untracked and
`simulations/gantry_subnet/diagnostics/msd_offset_plant_ablation_traces.npz` untracked.
Nothing has been committed.

Bash threw intermittent `fatal error - add_item` fork failures on this machine late in the
session. PowerShell was unaffected. If Bash misbehaves, switch rather than debug it.

## 4. Established and verified

The offset has two causes, one per axis, and both are measured against the recorded data.

**The complete model difference.** Confirmed by reading both sources, not transcription:
`Matlab-scripts/Augmentation/gantrySystemExtended.m:29-48` against
`kamtin-fp-model/03 Simulink gantry/functions/gantrySystem.m:5-17`. `C` and `K` agree exactly on
the shared three coordinates, so the entire difference is in `M`, and it is two things:

```
dM(1,2) = dM(2,1) = -ma*(L0 + delta_a)              -> owns X
dM(2,2) = ma*[2*Y*(L0+delta_a) + (L0+delta_a)^2]    -> acts on Theta, produces no offset
absorber column M(3,4)=M(4,4)=ma, M(2,4)=-ma*d, K(4,4)=ka, C(4,4)=ca   -> owns Y
```

`dM(1,1)`, `dM(1,3)`, `dM(2,3)`, `dM(3,3)` are identically zero: the mass split conserves total
mass. `M(1,4) = 0`, so the absorber cannot reach X directly.

**Why permanent:** `K(1,1) = K(3,3) = 0` and `K(2,2) = kb1+kb2 > 0`. On an axis with no stiffness
an accumulated impulse becomes a permanent shift, `c*dq(inf) = integral of dF dt`. Theta is
sprung, so the same perturbation decays there. That contrast is the control.

**Two closed forms, nothing fitted** (`msd_offset_x_closed_form.json`, and `F2` in
`msd_offset_figures_V1_standstill_Yp10.json`):

```
X:  (cg1+cg2)*dX(inf) = ma*L0*[Thetadot(T) - Thetadot(t0)]   slope 2.902e-03 s, R2 = 0.999931 (V1)
Y:  cy*dY(inf) = ma*vdelta_a(t0)                             slope 0.101 s,     R2 = 1.000000 (V1)
```

`ma*L0 = 0.101 kg*m`, which over `mh = 10.1 kg` is a **1 cm** error in the assumed payload centre
of mass. That sentence is the one the user found most communicable.

**Four models against the recorded output**, V1 seeded at sample 1863 (`fig_showcase.py` stdout,
and `msd_offset_plant_ablation.json`):

| model | X settled | Y settled |
|-|-|-|
| baseline | `+4.632e-05 m` | `+1.354e-03 m` |
| + centre-of-mass term | `-8.228e-07 m` (1.78 %) | `+1.354e-03 m` (100.00 %) |
| + absorber | `+4.632e-05 m` (100.00 %) | `-1.952e-07 m` (0.01 %) |
| + both (= truth) | `-8.224e-07 m` | `-1.952e-07 m` |

Each fix lands on the truth model's *own* residual against the data, so the last column doubles
as validation of the 8-state model. The same pattern holds on `T6_ysweep_slow` while `Y` sweeps
`-0.30` to `+0.30 m` twice (2.97 % / 100.01 % on X, 100.00 % / 0.08 % on Y).

**X and Y settled values are window-stable** (`46.33, 46.32, 46.31, 46.23, 46.03 um` over
trailing windows of 0.5 to 6 s). **Theta's is not**: it ranges `5.98e-10` to `-2.30e-11 rad` and
changes sign, which is the signature of no offset. Its rms (`26` to `34 nrad`) is
window-independent and is what the figure quotes.

**Bode decomposition, exact and record-independent** (`msd_offset_bode_difference.json`).
Eliminating `delta_a` by Schur complement gives
`Z_eff = Z_baseline + dM*s^2 - s^4*v*v'/(ma*s^2+ca*s+ka)` with `v = [0, -ma*d, ma]'`, so the
A/B split of `Delta(s)` is exact. Consistency checks land at machine precision. On X, B beats A
by 7 to 13 orders; on `Y<-F_Y`, A beats B by `2.4e+04`. Two unanticipated results: `Delta` peaks
on the **sprung Theta mode at 5.1 Hz**, not the absorber, and the absorber's coupled mode is at
**158.114 Hz**, not the 150 Hz design frequency. `Theta<-F_Theta` spreads 21x over `Y` while the
X and Y offset coefficients are exactly `Y`-independent.

**Seed test, pre-registered** (`msd_offset_figures_T10_aprbs_60.json`): on T10 the `L0 = 0` arm
removes 98.2 % of the X offset at **11.5 sigma** against a threshold of 3; Theta reads 0.73 sigma,
below its 2-sigma threshold.

## 5. Assumed but not verified

1. **`ma_frac = 0.10` and `L0 = 0.10 m` are fitted to the data, not measured.** The `.mat` files
   store no absorber parameters. Volunteer this before a supervisor asks. It also means none of
   this transfers to the real Telica system as a physical claim; there is no such absorber there.
2. **The excitation band is unresolved and the user never answered.** The user stated the MSD data
   excites only 130-180 Hz and that a separate joint-estimation set uses 1-200 Hz. It is not known
   whether `T10_aprbs_60` is band-limited or is a motion profile with a 130-180 Hz perturbation on
   top. T10's X carries the 11.5-sigma result, so if it is genuinely band-limited its stable mean
   needs another explanation. Settle it by looking at the low-frequency content of `u_total`.
3. **T10's X residual after `L0 = 0` is 15.8 % of RMS but only 1.7 % of the mean.** The residual
   is seed scatter, not offset, and it is unexplained. The X closed form fails on T10
   (`R2 = -0.03`) because the dropped terms are 200x to 34,000x larger there than on V1.
4. **The eight documentation corrections are unwritten.** `ISSUE.md` and
   `docs/msd-offset-mechanism-2026-07-29.md` still carry: the four corrections listed in the
   2026-07-30 handoff section 6, plus V1's X un-withdrawal, the Theta criterion's missing
   magnitude floor, the Y seed-mean statistic, and the mechanism-C paired result.
5. **`figures/SHOWCASE_offset_mechanism.{png,pdf}` (no record suffix) is stale**, left from before
   the per-record rename. The current files are `..._V1_standstill_Yp10` and `..._T6_ysweep_slow`.
6. **`README.md` in that folder lists figure defects that are partly fixed.** F6's log bar chart
   was replaced by a dot plot with standard-error bars; the F1 defects still stand.
7. **The house figure style is not applied.** `docs/writeup/figure-style.md` section 2 requires
   black plus `black!55` and `black!12`, no colour. The showcase figure uses blue. Fine for a
   screen, wrong for the thesis. Conversion is mechanical: blue becomes black, grey becomes
   `black!55`.

## 6. Tried and failed

The figure went through five constructions before landing. Do not re-propose the dead ones.

- **All four models overlaid on two panels, with the reference drawn as a wide halo** -> user:
  "cluttered", "the overlay is a bit much, I can't say what is happening" -> when two curves are
  bit-identical, any overlay shows one line where the legend promises two, and styling cannot fix
  it -> superseded by one curve per panel.
- **Ablating the *plant* (removing features from reality) with the simulated truth as reference**
  -> user: "based on which data is this? or just comparing the models?" -> grey and blue were
  errors against two *different* truths and the recorded data never appeared -> replaced by four
  models against the one recorded output.
- **Reporting a settled Theta offset of `2e-13 m`** -> user challenged it -> it was the 1 s
  averaging window landing near a zero crossing; the same quantity reads `-1.69e-10`, `-1.42e-11`,
  `-3.95e-11` at other windows -> now reported as a bound plus the window-independent rms, with
  the stability test in code.
- **Theta reported in nanoradians while X and Y were in um and mm** -> user: "the figure is
  inconsistent for theta" -> the row was incommensurable with the others -> all three rows now in
  metres, Theta as `(Lb/2)*Theta = (X1err - X2err)/2`.
- **1.8x y-axis headroom to clear the verdict text** -> user: "I liked the previous axes better"
  -> it squashed the curves into the bottom fifth -> reverted, no headroom.
- **Mechanism C, the `delta_a`-dependent part of `dM(1,2)`, as the offset source** -> freezing
  `delta_a` retains 100.00 % of the Y offset and 99.84 % of T10's X -> `delta_a` has an rms of
  22 micrometres against a 0.4 m arm -> real but worth 0.16 %, detectable at 21.9 sigma paired.
- **Treating V1's X as unusable scatter** (the prior handoff withdrew it) -> wrong -> the scatter
  is a deterministic function of `[Thetadot(T)-Thetadot(t0)]`, predicted at `R2 = 0.999931` by an
  unfitted line -> V1's X is now the *cleanest* evidence in the set. Do not repeat the withdrawal.

## 7. Achieved

**Implemented and validated.** In `scripts/gantry/msd-offset/`:

| script | what it produces |
|-|-|
| `plant.py` | both models, RK4, record loading. Shared by everything else. |
| `make_figures.py` | F1 to F6. F6 rewritten to 4 arms x 40 seeds with both coordinate frames. |
| `diag_bode_difference.py` | B1, B2, B3, the exact A/B split and the mode identification |
| `diag_x_closed_form.py` | B4, the X closed-form test |
| `diag_plant_ablation.py` | B5, the 2x2 plant ablation, and the cached traces `.npz` |
| `fig_showcase.py` | `SHOWCASE_offset_mechanism_<record>.{png,pdf}`, takes a record name as argv |
| `diag_x_discrepancy.py` | the three-way (baseline/truth/data) decomposition, sum checks at 1e-19 |

**Written and compiling clean.** `docs/writeup/offset-mechanism-equations.tex` is the simple
statement: the two fixes as concrete equation edits, ending with the disjoint-rows argument.
`docs/writeup/offset-mechanism-derivation.tex` is the formal counterpart (`Delta M`, `K=0`, the
two closed forms). Both are `\input` fragments using `\mhr` from `jan-augmentation-writeup.tex`;
`offset-mechanism-preview.tex` wraps them and resolves cross-references through the parent's
`.aux` with `xr`. 3 pages, zero errors, zero overfull boxes.

## 8. The open question

**Nothing blocked for the explanation task.** The figures and numbers are all current and
regenerate on demand.

One genuine open item, the user's to decide: the excitation-band question in section 5 item 2. It
does not change the attribution (the Bode result holds in-band, B over A by 7 orders across
130-180 Hz) but it changes how V1's and T10's difference is explained.

## 9. Next action

Write a one-page figure index to `scripts/gantry/msd-offset/README.md`, replacing its current
contents: one row per figure file, giving the question it answers, the record it uses, the script
that regenerates it, and whether it is current or superseded. Then answer the user's questions
about individual figures from that index.

Rationale: the user's request is "explain this and the other figures", there are 21 files in that
folder with no map, and the existing `README.md` documents figure defects that are now partly
fixed. The index is the artefact that makes every later question cheap to answer, and it corrects
a stale file in the same pass.

## 10. Acceptance criterion

The index covers **all 21 files** currently in `figures/`, each mapped to a question, a record and
a regenerating script, with `SHOWCASE_offset_mechanism.{png,pdf}` (no suffix) marked stale and the
two fixed F6 defects struck from the defect list. No figure is left unexplained, including the
ones superseded by the showcase.

There is no numeric threshold here because the deliverable is exposition, not a measurement. Every
number the index quotes must come from the JSON artefacts in section 4, not be recomputed.

## 11. Read these first

1. `scripts/gantry/msd-offset/fig_showcase.py` -- its module docstring records why the figure has
   the construction it has, including the four rejected ones.
2. `docs/writeup/offset-mechanism-equations.tex` -- the simple two-fix statement, and the version
   of the explanation the user accepted.
3. `scripts/gantry/msd-offset/ISSUE.md` -- the issue as originally framed. Still carries the eight
   errors of section 5 item 4; read it for context, not for facts.
4. `simulations/gantry_subnet/diagnostics/msd_offset_plant_ablation.json` -- the retention numbers
   behind the showcase figure, both coordinate frames, all 40 per-seed values.
5. `docs/writeup/figure-style.md` -- binding if any figure is going into the thesis rather than a
   slide. Section 2 on ink, section 7 on legends.

## 12. Do not

- Re-propose any of the five dead figure constructions in section 6.
- Quote `2e-13 m`, or any settled Theta value, as a number. Theta has a bound and an rms only.
- Re-withdraw V1's X, or quote V1's X *seed-mean* as an offset; quote the regression instead.
- Quote the "157.89 versus 164.55 Hz absorber anomaly". It is not an anomaly: the coupled mode is
  at 158.114 Hz and the records span 155.94 to 161.74 Hz.
- Touch `scripts/gantry/drift-isolation/`.
- Rerun `make_figures.py` casually: it takes about 25 minutes. The B-series and showcase scripts
  take seconds to a couple of minutes.

## 13. Operational

Conda env `GraduationProject`. From `scripts/gantry/msd-offset/`:

```
conda run --no-capture-output -n GraduationProject python -u fig_showcase.py V1_standstill_Yp10
conda run --no-capture-output -n GraduationProject python -u fig_showcase.py T6_ysweep_slow
conda run --no-capture-output -n GraduationProject python -u diag_bode_difference.py
conda run --no-capture-output -n GraduationProject python -u diag_plant_ablation.py     # ~20 min
conda run --no-capture-output -n GraduationProject python -u make_figures.py            # ~25 min
```

`fig_showcase.py` takes any record name and rolls out its own traces (about 60 s), so it is not
limited to the two records the 40-seed sweep visited. `diag_bode_difference.py` is seconds and
needs no record at all. Anything over a few seconds goes to the background per the live-output
rule.

LaTeX, from `docs/writeup/`: `pdflatex jan-augmentation-writeup.tex` once so its `.aux` exists,
then `pdflatex offset-mechanism-preview.tex` twice.

Records live in `data/gantry/matlab/trajectory/augmentation/`. Y operating points:
`T1_Ym30`, `T2_Ym15`, `T3_Y000`, `V1_Yp10`, `T4_Yp15`, `T5_Yp30` are standstill at fixed Y;
`T6/T7/T8_ysweep_*` sweep Y across the full range.

## 14. Delegation

None. Everything is in one directory of nine files, and section 4 already holds the numbers. An
Explore subagent would cost more than reading the folder.
