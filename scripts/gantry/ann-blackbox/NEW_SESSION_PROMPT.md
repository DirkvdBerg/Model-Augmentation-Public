# New session brief: build a minimal full-ANN black box from the references, then test sampling rate

Written 2026-07-31. This folder is empty by design. Nothing here is inherited from
`scripts/gantry/full-blackbox/`, which the user does not trust.

## The complete task, stated up front

Three parts, in this order.

**Part 1. Reconstruct the reference, line for line.** Read all 37 lines of Jan's
`scripts/ecc_2025/msd_ndof_deepSI_encoder.py` and all 55 lines of Beintema's
`deepSI-master/examples/docs/basic-example.py`, and produce a line-by-line correspondence table:
for every line of Jan's script, what it does, and what the new implementation will do at that
point. Then write `ann_blackbox.py` in this folder as the smallest thing that runs Jan's structure
on the gantry data. Deliverable: the correspondence table plus a file that trains for one epoch
without error.

**Part 2. The sanity check, and it gates everything after it.** Train that model on the
**both-fixed plant**: the 8-state truth with the centre-of-mass correction and the absorber both
present, which is the rightmost column of `SHOWCASE_offset_mechanism_<record>` in
`scripts/gantry/msd-offset/figures/`. That configuration has **no offset** (settled error `-8.2e-07 m`
on X and `-2.0e-07 m` on Y, at the truth model's own floor), therefore no slow curve, therefore
nothing in the data that a short window cannot see. Run it at **`nf = 200`** and at a decimated
sample rate. Deliverable: validation sim-RMS against epoch-0.

**Part 3. Decimation sweep, only if Part 2 is negative or ambiguous.** Train at 4000 / 2000 / 800 /
400 Hz with the horizon fixed at 400 samples, so every arm has identical compute and a different
span of seconds. Deliverable: sim-RMS per arm, each against its own epoch-0.

**Part 4. Read the poles.** Extract the discrete eigenvalues of the trained model and compare
against the truth's, listed in Section 4. Deliverable: where the two integrator poles landed.

Do not start Part 2 before Part 1's file trains. Do not start Part 3 if Part 2 succeeds; go to
Part 4 instead and report.

**The objective, unchanged from the user's own statement:** get a standalone ANN, with no
interconnect and no baseline, to a good validation sim-RMS, to establish that a full ANN can learn
this system at all. Two prior runs failed to do that.

## Why Part 2 gates everything: the chain it tests

From the supervisor meeting of 2026-07-31. This is his reasoning, not a measured result, and Part 2
is the experiment that confirms or kills all of it at once.

1. The baseline drifts away from the data with a very slow time constant, `tau_X = 1.55 s` and
   `tau_Y = 1.01 s`, because X and Y have viscous damping but **no stiffness**.
2. That slow curve cannot be represented inside a 400-sample window, so it corrupts training while
   contributing almost nothing the loss can act on.
3. Remove the slow curve from the data and a short window becomes sufficient. His words:
   `zolang de data niet meer die curve geeft, dan met nf van 200 of black box wel moet kunnen`.
4. If a pure black box cannot train, nothing downstream works:
   `als black box model niet traint gaat niks werken`.

**If Part 2 trains**, the chain holds, the slow curve was the obstacle, and the downstream direction
is his: put the centre-of-mass correction **and an absorber** into the baseline, so the baseline
carries one absorber and the system carries two. There is still a residual to augment, but it is
**fast** rather than slow, and therefore learnable in a short window. That redesign is **not** part
of this task; record the result and stop.

**If Part 2 does not train**, the offset was never the obstacle. Sampling rate (Section 3) and the
optimiser plus DC defect (Section 4, Section 6 item 3) are then the entire remaining problem, and
Part 3 becomes the next step.

**One reframe worth carrying.** He asked why the controller does not pull the offset to zero, and
answered it himself: in closed loop the controller absorbs the low-frequency component, so this
offset would never appear on the real machine. It is visible only because the replay is open loop
with no controller in the path. That makes the offset a property of the evaluation setup, not a
defect of the plant.

## Part 1 is the point of this session

The file you are replacing is **498 lines**. Jan's reference is **37**. That ratio is not itself a
defect, and no specific bug has been found in it, but it is why the user asked for a fresh folder
rather than a fix. So the constraint on `ann_blackbox.py` is this:

**Every line that departs from Jan's 37 must be justified in a comment on that line, naming what
forced it.** If you cannot name the reason, do not write the line. A gantry-specific data loader is
a reason. A normalisation override is a reason only if you can say what breaks without it. "More
capacity seemed sensible" is not.

Aim for something a reader can hold against Jan's script and check in one pass.

## How to report

**Report every deviation and every suspicion, including low-confidence and low-severity ones, and
tag each with confidence and severity.** Do not filter while working. A finding that turns out to be
noise is cheap; a real defect dropped silently is not. Filtering is a separate pass, after the list
exists.

Match the length of what you write to what the finding needs. No restating this brief back.

## 1. The references, and what each is for

| Reference | Lines | Role |
|---|---|---|
| `scripts/ecc_2025/msd_ndof_deepSI_encoder.py` | 37 | **Jan's full black box.** The structure to copy. Identical to `scripts/gantry/msd_ndof_deepSI_encoder.py`, both from his commit `6d69f6b` |
| `scripts/bouc_wen/bouc_wen_ANN_SS.py` | 39 | Jan's same pattern on a second benchmark. Use it to tell what is essential from what is problem-specific |
| `deepSI-master/examples/docs/basic-example.py` | 55 | Beintema's canonical SUBNET example |
| `deepSI-master/examples/1. Overview deepSI.ipynb` | -- | the concepts behind the above |
| `deepSI-master/deepSI/models.py` | 452 | v2 source, for reading semantics only |

**Critical API fact, confirmed 2026-07-31:** the environment has **deepSI 0.3.29**, loaded from
`site-packages`. `deepSI-master` is **deepSI v2 and is not installed**. So v2 material defines the
*structure* and Jan's scripts show that structure *on the installed API*. Where the two disagree,
Jan's script is what will actually run. Confirm this yourself rather than taking it from here.

## 2. Out of scope

- **`scripts/gantry/full-blackbox/`.** Do not import from it, copy code out of it, modify it, or
  refactor it. You may read it late, after your own design is drafted, only to check whether an
  experiment you are about to run has already been run there. Treat everything in it, including its
  `README.md`, `NEW_SESSION_PROMPT.md`, `results/`, `figures/` and
  `docs/blackbox-standalone-audit-2026-07-30.md`, as **unverified prior art**: useful for seeing
  what was attempted, never a fact to build on. If its conclusions and your measurements disagree,
  your measurements win.
- **All augmentation.** No physics baseline, no parallel ANN block, no interconnect, no orthogonal
  projection, no encoder-versus-ANN attribution. This is the black-box arm alone.
- **`scripts/gantry/drift-isolation/`** and **`scripts/gantry/msd-offset/`.** The first is a
  different failure; the second is finished. `msd-offset/plant.py` is a legitimate import for the
  loader and the truth model, and nothing there needs editing.
- **Regenerating the MATLAB records, and changing `zeta_a`.** Both were raised with the user on
  2026-07-31 and left undecided. Not side effects of this task.
- Do not modify `kamtin-fp-model/` or `scripts/gantry/gantry_dynamic/*`.

## 3. Why sampling rate is the lead hypothesis

`scripts/gantry/msd-offset/plant.py:116` loads records at `fs_new=4000` Hz, so `Ts = 2.5e-4 s`. The
fastest dynamics in the system are at 158 Hz. That is 25x, where the standard rule is 2 to 4x. The
user's supervisor raised this independently on 2026-07-31.

Three separate things go wrong at that rate:

1. **The poles crowd onto `z = 1`** (numbers in Section 4). What distinguishes a damped stage from a
   pure integrator lives in the fourth decimal place of the pole, near the float32 floor.
2. **One-step prediction becomes trivial.** At 4 kHz, `y[k+1]` is nearly `y[k]`, so a model that
   learns the identity map scores well on a short-horizon loss having learned no dynamics. Training
   loss looks healthy, free run does not. That is the reported symptom.
3. **It silently eats the window.** The horizon is defined in samples. 400 samples at 4000 Hz is
   0.10 s; at 800 Hz the same 400 samples is 0.50 s. The peak of the model difference is at 5.1 Hz,
   whose period is 196 ms, so the current window is **half a period** of the dominant content and
   the decimated one is 2.5 periods. Same compute.

This is why the sweep holds sample count fixed and varies only the rate: it separates "too few
seconds" from "too many samples per second", which no experiment so far has done.

## 4. Established, with evidence

Read from source or from JSON artefacts on 2026-07-31, not inferred.

**Timescales and modes** (`plant.py:42-43`, `msd_offset_bode_difference.json`):

```
free integrators on X and Y (K11 = K33 = 0)     z = 1.000000 exactly, two of them
X stage damping, tau_X = 1.546 s                z = exp(-Ts/tau_X) = 0.999838  at 4 kHz
Y stage damping, tau_Y = 1.010 s                z = exp(-Ts/tau_Y) = 0.999753  at 4 kHz
sprung Theta mode                                 5.1 Hz    where the model difference peaks
absorber coupled mode                           158.114 Hz  not the 150 Hz design value
```

The `z` values are exact arithmetic from the two time constants and `Ts`; they have not been
extracted from any trained model, which is Part 3's job.

**Window visibility** (`msd_offset_figures_<record>.json`, field `F3`): at 400 samples and 4 kHz,
6.3 % of the X response and 9.4 % of the Y response falls inside one window.

**`y` is decimated with no anti-alias filter.** `plant.py:126` point-samples (`[::D]`) while `u`
gets a block mean (D-087). Fix this before decimating further; at 800 Hz the new Nyquist is 400 Hz.

**Adam damages a near-optimal init.** First step is `~1.0 x lr` per coordinate independent of
gradient; from an optimal init that is pure damage, scaling `lr^2` (measured exponents 1.95, 1.86).
SGD at matched `lr` does not. `docs/ann-worse-than-init-diagnosis.md` §2.

## 5. Carried over from the campaign being replaced, and NOT re-derived

Every number here comes from the untrusted folder or its runs. Use them to size experiments, never
as an established result. Re-derive any that a conclusion rests on.

- **The FP baseline reaches about `1.6e-4 m` sim-RMS with no training.** This is the bar "good"
  means, and it is far below what either run achieved. Re-derive it in this folder; it is a property
  of the baseline model, so it is cheap and it does not depend on the untrusted code.
- **Run 73940**: nf=400, 2x16, lr=1e-3, 500 epochs. epoch-0 `1.0326e-01 m`, best `7.2363e-02`,
  final `9.0927e-02`, max `3.8662e+14`. 69 of 101 validation points above epoch-0.
- **Run 74045**: nf=800, 2x64, killed by the wall clock at ~76 %. epoch-0 `9.5769e-02 m`; val
  drifted `0.062 -> 0.095` while train fell `0.84 -> 0.89`, i.e. train and val moved apart.
- **Two lr screens exist and their rankings disagree with each other.** Do not trust either.
- Both runs used `nx=8`, `na=nb=17`, batch 256, stride 10.

The train-versus-val divergence in 74045 is the single most suggestive carried-over number, because
it is the signature both the sampling and the horizon hypotheses predict.

## 6. Assumed, and what would settle it

1. **The excitation band is unresolved, and it is load-bearing.** The user states the MSD data
   excites 130 to 180 Hz while a separate set uses 1 to 200 Hz. If the data really is band-limited
   to 130-180 Hz, then the 5.1 Hz content is **unexcited**, no window length recovers it, and
   Section 3 item 3 collapses. **Settle this first**: take the PSD of `u_total` on `T10_aprbs_60`
   and `V1_standstill_Yp10`. It is minutes of work and it decides whether the sweep's premise holds.
2. **Nothing specific is known to be wrong with `blackbox_standalone.py`.** The user's doubt is
   structural. Do not write into any document that it is defective.
3. **The ANN's output is reported non-zero-mean while the target and the data are zero-mean**
   (user, 2026-07-31). Not measured here. A DC component on an axis with `K = 0` produces constant
   velocity, hence unbounded position, so if it is real it matters. Whether it is present at
   initialisation or grows during training decides whether it is an init bug or the Adam mechanism.
4. **The 8-state truth is synthetic.** The absorber is injected by the data generator; the real
   Telica machine has none, and `ma_frac = 0.10`, `L0 = 0.10 m` are fitted, not measured.

## 7. Acceptance criterion

**Part 1** is done when `ann_blackbox.py` trains one epoch without error and the correspondence
table covers all 37 lines of Jan's script.

**Part 2** is the gate, and it is a yes or no: does the black box on the both-fixed plant end below
its own epoch-0, with the best checkpoint not at epoch 0. Report the answer plainly either way. A
negative result here is as informative as a positive one and must not be presented as a failure to
be worked around; it eliminates the offset hypothesis, which no experiment so far has done.

**Part 3**, if it runs, is done when every arm has a validation sim-RMS scored against **its own
epoch-0**, never against another arm's, since epoch-0 depends on the net. The result that matters:

- **Minimum bar:** at least one arm ends below its own epoch-0 with the best checkpoint not at
  epoch 0. That is the user's complaint stated as a number.
- **Target:** approaching `1.6e-4 m`, re-derived in this folder, which is what the FP baseline
  reaches untrained. Beating epoch-0 alone is not success.
- **Floor:** the 8-state truth model's own residual against the recorded data, about `1e-7 m`
  (`msd_offset_plant_ablation.json`, FULL arm). Nothing can go below it.

Report settled values with a trailing-window stability check rather than one window: X and Y settle
stably to better than 1 % over 0.5 to 6 s, Theta's settled mean changes sign with the window, so
Theta gets a bound and an rms only, never a single number.

## 8. Do not

- Do not copy code from `scripts/gantry/full-blackbox/` into this folder.
- Do not write a line into `ann_blackbox.py` that departs from Jan's reference without naming the
  reason on that line.
- Do not decimate `y` without an anti-alias filter.
- Do not size a run so its artefacts land only after `fit()` returns. Run 74045 died on a 14 h wall
  clock at 76 % and lost everything except the log, because the npz, the gates and `save_system` all
  ran afterwards. Persist metrics incrementally or run in chunks.
- Do not pass `lr` or a scheduler to `fit` after `init_model` has run; `init_model` creates both and
  `fit`'s kwargs are ignored once `init_model_done` is set (D-101).
- Do not quote a settled Theta value as a number.

## 9. Operational

Conda env `GraduationProject`; installed deepSI is 0.3.29. Long runs go to the background with live
streaming, per the running-scripts rule.

```
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject \
    python -u scripts/gantry/ann-blackbox/ann_blackbox.py
```

Records live in `data/gantry/matlab/trajectory/augmentation/`. `T1_Ym30`, `T2_Ym15`, `T3_Y000`,
`V1_Yp10`, `T4_Yp15`, `T5_Yp30` are standstill at fixed Y; `T6/T7/T8_ysweep_*` sweep Y across the
range; `T10_aprbs_60` is the APRBS record carrying most prior results.

Loader and truth model: `scripts/gantry/msd-offset/plant.py`, `load_record` at line 116 and
`deriv8`. Import them; do not reimplement.

Citable numbers from the offset work:
`simulations/gantry_subnet/diagnostics/msd_offset_{figures_<record>,plant_ablation,bode_difference,
x_closed_form}.json`.

Per the run-discipline rule, each sweep arm gets its row in the run table before launch.

## 10. Delegation

None by default. The references are three files totalling 131 lines and the new code is one file in
one directory; inline reading is cheaper and better informed. If the late read of
`full-blackbox/results/` turns into a wide sweep, one Explore subagent is the ceiling.

## 11. Note on this brief

Written for Claude Opus 5 following Anthropic's prompting guidance for that model: the complete
task is specified up front rather than assembled across turns, scope is constrained explicitly,
subagent use is capped, and the reporting rule asks for full coverage with confidence and severity
rather than self-filtering. Deliberately absent, per the same guidance: any instruction to
double-check or re-verify. Opus 5 does that unprompted and such instructions cost tokens with no
quality gain.

The judgement in this brief is the least reliable thing in it. Sections 3 and 4 are arithmetic and
file reads; Section 5 is inherited from a campaign the user distrusts; Section 6 is unmeasured.
Nothing is established because it is written here.
