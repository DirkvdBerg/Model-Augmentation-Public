# Handoff: a clean standalone black-box that actually learns, with sampling rate as the primary suspect

**From**: session of 2026-07-31 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Build a clean, standalone pure-ANN (black-box) identification of the 8-state MSD gantry in a
**new folder**, `scripts/gantry/blackbox-clean/`, and get it to learn: the free-run simulation
metric on held-out data must improve over its own initialisation. The existing
`scripts/gantry/full-blackbox/` is of uncertain provenance and the user does not trust it, so this
is a fresh implementation, not a refactor. Build it on the reference implementations that already
work (deepSI's SUBNET examples and Jan's ECC 2025 encoder script, paths in section 13), not on the
existing folder's code. The leading hypothesis for why the current one does not learn is
**oversampling**: the data is loaded at 4000 Hz against dynamics topping out at 158 Hz, which is 20
to 25x, where 2 to 4x is the standard rule. Test that first, as a decimation sweep, because it
simultaneously tests the competing horizon hypothesis (section 8). No augmentation and no physics
baseline is involved anywhere in this task: this is the black-box arm of the thesis comparison, on
its own.

## 2. Out of scope

- **`scripts/gantry/full-blackbox/`**: do **not** modify, extend, refactor or import from it, and do
  not copy its code into the new folder. The user's instruction is a new folder because that one is
  untrusted (section 5 item 1). It may be read late and only to avoid duplicating effort.
- **`scripts/gantry/msd-offset/`**: that investigation is finished and its conclusions are settled.
  Its `plant.py` is a legitimate import for the data loader and the truth model; nothing else there
  needs changing.
- **`scripts/gantry/drift-isolation/`**: a different failure (unbounded drift in the augmentation).
  Do not read or modify.
- **Anything augmentation.** No baseline, no ANN-parallel block, no orthogonal projection, no
  encoder-versus-ANN attribution. This task is the standalone black-box only.
- **Regenerating the MATLAB records**, and **changing `zeta_a`**. Both are user decisions that were
  raised this session and not taken. Do not do either as a side effect.
- Do not modify `kamtin-fp-model/` or `scripts/gantry/gantry_dynamic/*`.

## 3. Where things stand

Branch `Augmentation`, last commit `8022544`. Nothing in flight. Tree is dirty; relevant to this
task, `scripts/gantry/full-blackbox/` and `scripts/gantry/msd-offset/` are both untracked, as are
`docs/writeup/offset-mechanism-{equations,derivation,panels,panels-preview}.tex` and
`docs/blackbox-standalone-audit-2026-07-30.md`. Nothing has been committed.

This session was an explanation session on the MSD offset and produced no runs. The numbers in
section 4 are read from source and from existing JSON artefacts, not from new experiments.

## 4. Established and verified

**Sampling rate and what it does to the poles.** `plant.py:116` loads records with
`fs_new=4000` Hz, so `Ts = 2.5e-4 s`. The system's own timescales, from
`msd_offset_bode_difference.json` and `plant.py:42-43`:

```
free integrators on X and Y (K11 = K33 = 0)   z = 1.000000 exactly
X stage damping,   tau_X = 1.546 s            z = exp(-Ts/tau_X) = 0.999838
Y stage damping,   tau_Y = 1.010 s            z = exp(-Ts/tau_Y) = 0.999753
sprung Theta mode                             5.1 Hz    (where Delta peaks)
absorber coupled mode                       158.114 Hz  (not the 150 Hz design value)
```

Four of the eight poles sit within `2.5e-4` of `z = 1`. The information distinguishing a damped
stage from a pure integrator is in the fourth decimal place of the pole, which is near the float32
floor. The `z` values above are exact arithmetic from the two verified time constants and `Ts`;
they have not been extracted from a trained model.

**The window is defined in samples and is being consumed by the sample rate.** The training horizon
is 400 samples, which at 4000 Hz is 0.1 s (`docs/ann-worse-than-init-diagnosis.md` anti-pattern
example; `msd_offset_figures_*.json` field `F3.window_s = 0.1`). Consequences, measured in the same
artefact:

```
F3.frac_visible_window_X = 0.0626      6.3 % of the X response is inside one window
F3.frac_visible_window_Y = 0.0943      9.4 % on Y
5.1 Hz has a 196 ms period             one window is HALF a period of Delta's peak
158 Hz has a 6.3 ms period             one window is 16 periods
```

So the part of the dynamics the window can resolve is the high-frequency part, which contributes
least to position, and the part that dominates is unresolvable. Decimating to 800 Hz turns the same
400 samples into 0.5 s, which is 2.5 periods of 5.1 Hz, at identical compute. 800 Hz is also
exactly 4x the stated 200 Hz upper excitation limit.

**`y` is decimated without an anti-alias filter.** `plant.py:126` point-samples (`[::D]`) while
`u` gets a block mean (D-087). Any content above the new Nyquist folds into band. This already
applies at 4000 Hz and gets sharper at 800 Hz (Nyquist 400 Hz).

**The residual force the model must produce is zero-mean, by derivation not by measurement.** Both
missing-force terms are exact time derivatives, `f_X = d/dt[ma*L0*dTheta]` and
`f_Y = d/dt[-ma*vDelta_a]`, so their long-run means are zero while their integrals are bounded.
This is what makes a *bounded offset* the correct signature and *linear drift* an incorrect one. A
non-zero-mean force on an axis with `K = 0` gives constant velocity, hence unbounded position.

**Adam damages a near-optimal initialisation, mechanism known.** Its first step is `5.4 x lr` in L2
over ~30 parameters, i.e. `~1.0 x lr` per coordinate, independent of gradient. From an optimal
init that is pure damage, scaling `lr^2` (measured exponents 1.95 and 1.86). SGD at matched `lr`
does nothing and does dip below the init. Full detail and provenance:
`docs/ann-worse-than-init-diagnosis.md` §2 and §6.

## 5. Assumed but not verified

1. **The existing `full-blackbox/` folder is untrusted, and that is the premise of this task.** The
   user does not believe it is clean and has not identified a specific defect; the doubt is
   structural, not a known bug. Treat everything in it, including `README.md`,
   `docs/blackbox-standalone-audit-2026-07-30.md`, `results/` and `figures/`, as **unverified prior
   art**: useful for seeing what was attempted, never a fact to build on. No number, conclusion or
   design decision from that folder may be promoted into section 4 reasoning or into the new
   implementation without being re-derived from source in the new folder. In particular
   `horizon_400_vs_1600.png` and `horizon_law_linear.py` show a horizon experiment was run there;
   read it to avoid duplicating *effort*, not to inherit its *answer*. If its conclusion and this
   task's decimation sweep disagree, the sweep wins.
2. **The excitation band is unresolved.** The user stated the MSD data excites 130 to 180 Hz while a
   separate joint-estimation set uses 1 to 200 Hz. If the data is genuinely band-limited to
   130-180 Hz, then the 5.1 Hz content of the system is **unexcited**, no window length can recover
   it, and the whole horizon argument changes. Settle this by taking the PSD of `u_total` on
   `T10_aprbs_60` and on `V1_standstill_Yp10` before running the sweep. This is the cheapest
   load-bearing check in the file.
3. **The ANN's output is non-zero-mean; the target and the data are zero-mean.** User-reported this
   session, not measured here. If true, the DC is a defect with no counterpart in the target. Not
   yet known whether it is present at initialisation or grows during training; that distinction
   decides whether it is an init bug or the Adam mechanism in section 4.
4. **No pole extraction has been done on any trained black-box.** The claim that a black-box
   mislearns the `z = 1` poles is a prediction from section 4, not a measurement.
5. **The 8-state truth is a synthetic ground truth, not the machine.** The MSD is injected by the
   data generator (`FA` sourced from `gtd_config`); the real Telica system has no such absorber.
   `ma_frac = 0.10` and `L0 = 0.10 m` are fitted from the data, not measured, and the `.mat` files
   store no absorber parameters.

## 6. Tried and failed

- **Adam at `lr = 1e-7` from an optimal init, 30 steps** -> peak 134x `L0`, final 2.05x, never below
  `L0`; extended to 350 steps it reaches 2.7 % above the init -> the optimiser takes a fixed
  `~1.0 x lr` step per coordinate regardless of gradient, then repairs at 0.005 to 0.013 x `lr`
  once the gradient alternates sign, so it damages at full speed and repairs at one hundredth of it
  -> `docs/ann-worse-than-init-diagnosis.md` §2, §3.
- **SGD as the fix for the above** -> does no damage and does dip below `L0`, but the project
  separately measured it learning +0 % on a real residual -> it trades the damage failure for a
  no-learning failure -> same doc §5.
- **Treating "worse than init" as a single failure** -> it is at least two: R2 (the training loss)
  and R4 (free-run drift) were conflated for most of the campaign -> same doc §1.

Nothing in this list is specific to the black-box arm; all of it comes from the augmentation
campaign and transfers only as a hypothesis.

## 7. Achieved

Nothing implemented this session. It was an explanation session, and its output is documentation:

| artefact | state |
|-|-|
| `docs/writeup/offset-mechanism-panels.tex` | written, compiles clean in the preview geometry |
| `docs/writeup/offset-mechanism-panels-preview.tex` + `.pdf` | built, 2 pages, zero overfull/underfull, zero undefined refs |

Both describe the four columns of the MSD-offset showcase figure as edits to the baseline matrices.
Neither is used by this task; they are listed so the successor does not think they are in flight.

## 8. The open question

**Is the black-box failing to learn because of the sample rate, or because of the training horizon,
or because of the optimiser and a DC defect?**

These are not independent, which is what makes the first experiment worth running:

- *Sampling*: at 4000 Hz the poles crowd onto `z = 1` and one-step prediction is nearly trivial
  (`y[k+1] ~ y[k]`), so the loss is dominated by an identity map that carries no dynamics.
- *Horizon*: 400 samples covers 0.1 s, half a period of where `Delta` peaks.
- *Optimiser and DC*: section 4's Adam mechanism plus an unconstrained DC direction the short
  window cannot see.

**Decimation tests the first two at once and holds sample count fixed**, so it separates "too few
seconds" from "too many samples per second". If the sweep produces learning at 800 Hz with an
unchanged 400-sample window, sampling and horizon are jointly confirmed and the optimiser thread
drops in priority. If no decimation arm learns, sampling and horizon are both eliminated and the
optimiser and DC become the whole problem.

The excitation-band check in section 5 item 2 must run first, because a band-limited-to-130-180 Hz
answer invalidates the horizon half of the reasoning before any training run is spent on it.

## 9. Next action

**Create `scripts/gantry/blackbox-clean/` with a standalone SUBNET black-box built on the deepSI
example, and run a decimation sweep at 4000 / 2000 / 800 / 400 Hz with the training horizon held at
400 samples**, so each arm sees the same compute and a different span of seconds.

Take the PSD of `u_total` first (section 5 item 2); it is a few seconds of work and it decides
whether the sweep's premise holds. Add an anti-alias filter to the loader before decimating, since
`plant.py:126` point-samples `y`.

Per the run-discipline rule, each arm gets its row in the run table before launch.

Rationale: this is one experiment that discriminates between the two leading hypotheses and
eliminates one of them either way, it needs no new data and no architecture change, and the user's
supervisor independently raised oversampling, so it is also the answer to a question that will be
asked again.

## 10. Acceptance criterion

**Primary:** on held-out records, the free-run simulation RMS of the trained black-box is **below
its own value at initialisation**, and the best checkpoint is not epoch 0. That is the user's
complaint stated as a number, and it is measured on data.

**Secondary, for interpreting the result:** the free-run RMS of the 6-state baseline on the same
held-out records, computed in the same script, is the reference a black-box should beat if it has
learned anything. The floor is the 8-state truth model's own residual against the recorded data,
about `1e-7 m` (`msd_offset_plant_ablation.json`, the FULL arm), which no model of this system can
go below.

Report the settled value with the trailing-window stability check, not a single-window mean: X and Y
settled values are window-stable to better than 1 % over 0.5 to 6 s, Theta's is not and changes
sign, so Theta gets a bound and an rms only.

## 11. Read these first

The first three are the trusted foundation and the implementation is built from them. The last is
untrusted, per section 5 item 1, and is read last so its conclusions cannot anchor the design.

1. `deepSI-master/examples/1. Overview deepSI.ipynb` -- the reference pure-ANN SUBNET setup this
   implementation should follow. `3. LPV SUBNET.ipynb` is the scheduled variant if it is needed
   later; it is not needed for this task.
2. `scripts/ecc_2025/msd_ndof_deepSI_encoder.py` -- Jan's own pure-encoder example on an MSD, the
   closest working precedent in this repo to what is being built.
3. `scripts/gantry/msd-offset/plant.py` -- `load_record` (line 116) for the loader and the
   decimation, `deriv8` for the truth model. Import it; do not reimplement it.
4. `docs/ann-worse-than-init-diagnosis.md` -- §2 for the Adam mechanism and §5b for the
   pre-registration format the decimation sweep should follow.
5. `scripts/gantry/full-blackbox/README.md` and `docs/blackbox-standalone-audit-2026-07-30.md` --
   read **after** the design is drafted, purely to check whether an experiment being planned has
   already been run there. Unverified prior art, not a source of facts.

## 12. Do not

- Do not modify or refactor `scripts/gantry/full-blackbox/`; read it only.
- Do not touch `scripts/gantry/drift-isolation/` or `scripts/gantry/msd-offset/` (beyond importing
  `plant.py`), or `kamtin-fp-model/`.
- Do not regenerate the MATLAB records or change `zeta_a`. Both were discussed with the user this
  session and left undecided.
- Do not introduce the physics baseline, the augmentation, or the orthogonal projection into this
  task, even as a comparison arm beyond the single reference number in section 10.
- Do not decimate `y` without an anti-alias filter.
- Do not quote a settled Theta value as a number; it has a bound and an rms only.

## 13. Operational

Conda env `GraduationProject`. Long runs go to background with live streaming, per the
running-scripts rule.

```
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject \
    python -u scripts/gantry/blackbox-clean/<script>.py
```

Records live in `data/gantry/matlab/trajectory/augmentation/`. `T1_Ym30`, `T2_Ym15`, `T3_Y000`,
`V1_Yp10`, `T4_Yp15`, `T5_Yp30` are standstill at fixed Y; `T6/T7/T8_ysweep_*` sweep Y across the
range; `T10_aprbs_60` is the APRBS record carrying most of the prior results.

Reference material, verbatim paths:

```
deepSI-master/examples/1. Overview deepSI.ipynb
deepSI-master/examples/2. pHNN SUBNET demo.ipynb
deepSI-master/examples/3. LPV SUBNET.ipynb
deepSI-master/examples/docs/basic-example.py
scripts/ecc_2025/msd_ndof_deepSI_encoder.py
scripts/ecc_2025/msd_ndof_interconnect_dynamic.py
```

Existing black-box arm, for reading only:
`scripts/gantry/full-blackbox/{README.md, blackbox_standalone.py, ref_subnet_v2_example.py,
horizon_law_linear.py, results/, figures/}`.

Citable numbers from the offset work:
`simulations/gantry_subnet/diagnostics/msd_offset_{figures_<record>,plant_ablation,bode_difference,
x_closed_form}.json`.

## 14. Delegation

None. Every path this task needs is named above, the reference implementations are three known
files, and the new code goes in one new directory. An Explore subagent would cost more than reading
them. If the existing `full-blackbox/` audit turns out to need a wide sweep of the results
directories, one Explore subagent is the ceiling.
