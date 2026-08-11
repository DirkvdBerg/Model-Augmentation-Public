# Session brief: why does the black box barely learn, and how does it differ from the references

Written 2026-08-10. The implementation in this folder is already a faithful line-for-line
reconstruction of Jan's reference (`CORRESPONDENCE.md`). It runs, it trains, and it is nowhere near
useful. This session finds out why.

## The complete task, stated up front

Three parts, in this order.

**Part 1. Why do the runs stop after 5 or 6 validation points when `epochs` is 9999?** Every metrics
file in `results/` shows `n_val` of 1, 5 or 6. Jan's reference runs **10000 epochs**. Establish
whether these runs are converged or truncated, and if truncated, by what. Deliverable: the cause,
named, with the evidence.

**Part 2. Given that, is the model undertrained or structurally unable?** Deliverable: a converged
run, or a demonstration that convergence does not help. Score against the bars in Section 2, not
against epoch-0 alone.

**Part 3. What differs from the two references, and does any difference matter?** Compare against
Jan's `scripts/ecc_2025/msd_ndof_deepSI_encoder.py` (37 lines) and Beintema's
`deepSI-master/examples/docs/basic-example.py` (55 lines), reading both in full. `CORRESPONDENCE.md`
already lists the deviations; this part asks which of them are load-bearing. Deliverable: a ranked
list, most-likely-to-matter first, each with the measurement that would settle it.

Do not start Part 2 before Part 1 has an answer. A convergence study on a harness that silently
truncates is wasted.

## 1. Report everything

**Report every difference and every suspicion, including low-confidence and low-severity ones, and
tag each with confidence and severity.** Do not filter while working. Filtering is a separate pass
after the list exists. If Part 1 turns out to be a one-line configuration mistake, say so plainly
rather than looking for something more interesting.

## 2. Where it currently stands, measured

From `results/metrics_*.json`, all at `fs = 800`, train `T10_aprbs_60`, val `V2_aprbs_Ylow`,
`nx = 8`, `na = nb = 17`, width 8, encoder width 16.

| arm | epoch-0 | best | `n_val` | best epoch |
|---|---|---|---|---|
| `nf=400` | 0.16114 | 0.12894 | 5 | 4 |
| `nf=3700` | 0.16114 | 0.11787 | 6 | 5 |
| `nf=3700_blafullz` | 0.67728 | 0.67728 | 1 | 0, `loss_train = NaN` |

Against the bars re-derived in `bars.py` (`results/bars.json`, V2, free run, oracle initial state):

```
floor        4.652e-05 m     best is 2534x above
baseline     1.883e-04 m     best is  626x above       <- the FP baseline, untrained
frozen_lti   4.781e-04 m     best is  247x above
mean_pred    1.470e-01 m     best is 0.80x, i.e. only 20 % better than predicting the mean
```

**That last line is the finding.** After training, the model is barely distinguishable from a
constant predictor. It is not a tuning problem at the margin; the model has learned almost nothing
about the dynamics.

Per channel on the best arm: `rms_x1 = 0.0507`, `rms_x2 = 0.0521`, `rms_Y = 0.1908`. **Y is roughly
4x worse than the X channels.** Y is the axis owned by the absorber, which is an initial-condition
mechanism, so this is worth keeping in view even though a black box with `nx = 8` has the states to
represent it.

`loss_val` decreases **monotonically** in both working arms and never flattens. Nothing in these
runs looks converged.

## 3. The leading hypothesis, and why

`epochs` is 9999 and 5 validation points came out. Jan's reference runs 10000 epochs to get his
result. So the first and cheapest explanation is that **these runs are truncated by three orders of
magnitude and the model is simply undertrained.**

`ann_blackbox.py` added two controls that Jan does not have, either of which stops `fit()` early:

```
--timeout    line 31   DEV: makes fit() return normally instead of dying on a wall clock (run 74045)
--n-its      line 32   DEV: pairs arms on update count rather than epochs
```

Check what was actually passed. If either fired, the entire result set means "5 epochs of training"
and Part 2 is just a long run. Look for the launch commands in `results/` or the shell history, and
if they are not recorded, that is itself a finding worth fixing.

Falsifier: if the runs used neither and genuinely ran 9999 epochs producing 5 validation points,
then validation frequency is the issue and the loss curve is far denser than it appears.

## 4. The reference deltas, for Part 3

Established by reading both files. `CORRESPONDENCE.md` in this folder has the full mapping; this is
the subset most likely to matter.

| | Jan, ECC 2025 | this implementation |
|---|---|---|
| epochs | 10000 | 9999 set, **5 to 6 realised** |
| `nf` | 200 | 400 and 3700 |
| batch size | 2000 | 256 |
| data | 3-dof MSD multisine `.npz` | gantry `.mat`, APRBS, decimated to 800 Hz |
| noise | **added, SNR 20** (`sigma_n = 15e-3`) | none, records are noiseless Simulink |
| dof | 3, so `nx=6`, `na=nb=13` | 4, so `nx=8`, `na=nb=17` |
| widths | f/h 2x8, encoder 2x16 | identical |
| lr | Adam default | identical, `1e-3` |
| `validation_measure` | `sim-RMS` | identical |

Three of these are worth a hypothesis each:

- **Epochs, by a factor of about 2000.** Section 3.
- **`nf = 400` or `3700` against Jan's `200`.** Longer BPTT windows are harder to optimise, and the
  gantry has two poles at `z = 1` so gradients propagate without decay across the whole window.
  Jan's system has no free integrator. Note the paradox in the current results: `nf = 3700` scores
  *better* than `nf = 400`, which is the opposite of an optimisation-difficulty story, and may
  simply be an artefact of both being 5 epochs in.
- **Noiseless data.** Jan deliberately adds noise. The `DEV` comment at lines 43-44 argues noise
  would change the problem, which is reasonable, but noiseless targets with a free integrator can
  make the encoder's job degenerate. Worth listing; do not assume it matters.

The one thing this brief cannot tell you: `deepSI-master/examples/docs/basic-example.py` has not
been read by the session writing this. Read all 55 lines yourself, and note that `deepSI-master` is
**v2 and is not installed**; the environment has **deepSI 0.3.29**, so v2 defines structure and
Jan's script shows the installed API.

## 5. The separate defect

`metrics_fs800_nf3700_s0_blafullz.json` has `loss_train = [NaN]` and an epoch-0 four times worse
than the random init. The BLA arm is broken, not merely unhelpful. `bla_init.py` (215 lines,
Ramkannan et al. IFAC 2023) is where it lives. Fix or disable it, but do not let it contaminate the
main diagnosis. It is not the reason the ordinary arms plateau, because they do not use it.

## 6. Out of scope

- **`scripts/gantry/full-blackbox/`.** The old, distrusted implementation. Do not import from it,
  copy from it, or treat any of its results as fact.
- **All augmentation.** No baseline, no parallel ANN block, no interconnect, no projection. This is
  the black-box arm alone. The one exception is the `baseline` bar in Section 2, already computed.
- **`scripts/gantry/coulomb-offset/`** and **`scripts/gantry/msd-offset/`.** Different threads.
  `msd-offset/plant.py` is a legitimate import; change nothing in either folder.
- **Do not regenerate the MATLAB records, do not change `zeta_a`, do not add friction.** All are
  open user decisions and none is needed here.
- Do not modify `kamtin-fp-model/` or `scripts/gantry/gantry_dynamic/*`.

## 7. Acceptance criterion

**Part 1** is done when the truncation cause is named with evidence, or ruled out.

**Part 2** is done when there is a run whose `loss_val` has visibly flattened, scored against all
four bars in Section 2. The bar that matters is `baseline = 1.883e-04 m`: a black box that cannot
approach what the untrained FP baseline achieves has not learned the system. Beating `mean_pred` is
the floor of usefulness, not success, and the current best only beats it by 20 %.

**Part 3** is done when each deviation from the references carries a verdict and, where it might
matter, the measurement that would settle it.

A result showing the model converges and is still 600x above the baseline is a valid and important
outcome. Report it plainly; it would mean the failure is structural and the next question is the
model class, not the training.

## 8. Do not

- Do not run a long convergence study before Part 1 answers why the short ones stopped.
- Do not pass `lr` or a scheduler to `fit` after `init_model` has run; `init_model` creates both and
  `fit`'s kwargs are ignored once `init_model_done` is set (D-101).
- Do not size a run so its artefacts land only after `fit()` returns without `--timeout` set. Run
  74045 died on a 14 h wall clock at 76 % and lost everything but the log.
- Do not add capacity, change widths, or tune the learning rate as a first move. Jan reaches a
  working result at 2x8 and 2x16 with the Adam default, and matching him is the point of this
  folder.
- Do not report a per-arm comparison without stating the update count; `nf` changes updates per
  epoch by about 9x, so epochs are not comparable across arms.

## 9. Read these first

1. `CORRESPONDENCE.md` in this folder, the existing line-for-line mapping.
2. `ann_blackbox.py`, 97 lines, especially the `DEV` comments at lines 25, 28, 31, 32.
3. `scripts/ecc_2025/msd_ndof_deepSI_encoder.py`, 37 lines, the reference.
4. `deepSI-master/examples/docs/basic-example.py`, 55 lines, not yet read by anyone here.
5. `results/oversampling_diagnostic.json`, `results/encoder_vs_dynamics.json`,
   `results/pretrain_diagnostic.json`, diagnostics added by an earlier session and not summarised
   anywhere. They may already contain half the answer.

## 10. Operational

Conda env `GraduationProject`, deepSI 0.3.29. Stream long runs unbuffered per the running-scripts
rule.

```
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject \
    python -u scripts/gantry/ann-blackbox/ann_blackbox.py --fs 800 --nf 400 --seed 0
```

Per the run-discipline rule, each arm gets a run-table row before launch.

## 11. Delegation

None. Four files totalling under 300 lines, one folder of results.
