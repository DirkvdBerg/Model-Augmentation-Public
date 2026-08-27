# Why the ANN trains to something worse than its initialisation

**Written 2026-07-25.** Diagnosis only, no fix proposed. The complaint this answers is the user's own:
*"the ANN not learning, and it only becomes worse than the encoder initialization"*, which shows up in
the run log as "best checkpoint = epoch 0" and "val loss only increases".

**This is a different failure from the free-run drift** documented in
`docs/drift-conclusions-2026-07-25.md`. That one is R4 (free-run position drift); this one is R2 (the
training loss itself). They were conflated for most of the campaign.

## 1. The measurement

Perfect-match null on rig `e1b0511a4c`, where the zero-output init is **exactly optimal**: the target is
the recorded baseline output, so the correct ANN output is identically zero and
`L0 = 8.847965e-13` is the global minimum for this block.

30 steps per arm from that init, seeds 0 and 1 complete (seed 2 has the Adam arms only; the run was
stopped before its SGD control).

| arm | peak loss / `L0` | final / `L0` | ever below `L0` | first parameter move |
|---|---|---|---|---|
| Adam `lr = 1e-6` | 13565x | 139x | no | 5.4 x lr |
| **Adam `lr = 1e-7`** (the rig's `LR_NULL`) | **134x** | **2.05x** | **no** | 5.4 x lr |
| Adam `lr = 1e-8` | 2.03x | 1.39x | yes | 5.4 x lr |
| Adam `lr = 1e-9` | 1.04x | 1.04x | no | 5.4 x lr |
| **SGD `lr = 1e-7`** | **1.000x** | 1.000x | **yes** | ~0 |
| **SGD `lr = 1e-6`** | **1.000x** | 0.998x | **yes** | ~0 |

Seed 1 shown; seed 0 and seed 2 agree (seed 2: 90x at `lr = 1e-7`, first move `5.24 x lr`).

Script `scripts/gantry/drift-diagnostics/d8_why_worse_than_init.py`, log `logs/d8.output`, units
`data/D8_lr-sweep_30steps_seed{0,1}.json`.

## 2. The mechanism

**Adam's first move is `5.4 x lr` at every learning rate**, i.e. completely independent of the gradient.
That figure is an L2 norm over all ANN parameters; with about 30 effectively-moving parameters it is
`5.4/sqrt(30) = 1.0 x lr` **per coordinate**. Adam takes a full unit step per coordinate on step one
whatever the gradient is worth, because `m_hat/(sqrt(v_hat) + eps) -> 1` for small but consistent
gradients (`eps = 1e-8` against `sqrt(v_hat) ~ 1e-6` contributes nothing). This is the scale-freeness
property that Zhuang et al. (TMLR 2022, `arXiv:2202.00089`, already held in
`scripts/gantry/drift-fix-trials/research/thread-AB-optimizer-mechanics.md` item A3) describe as a
feature; here it is the defect.

From an already-optimal point, a step of fixed size is pure damage, and the cost of an `lr`-sized step
on a locally quadratic minimum is `O(lr^2)`. Measured exponent of peak excess versus `lr`: **1.95 and
1.86** across the two complete seeds, i.e. `lr^2` as expected.

**SGD is the control that settles it.** Its step scales with the gradient, so at the same `lr` it moves
essentially nothing, never exceeds `L0`, and does dip below it. Same data, same init, same `lr`, same
loss: the only difference is the normalisation, and the damage appears only with it.

Note also that the project already observed the other half of this: the SGD null pass was recorded as
"inaction at ~0 gradient", which is the same mechanism seen from the opposite side.

**Correction to the script's own labelling:** `d8_why_worse_than_init.py` prints an `lr^2` scaling as
"H2 loss-geometry". That label is wrong. Both hypotheses predict `lr^2` once the step is
`lr`-proportional, so the exponent does not discriminate; the SGD control does.

## 3. Why recovery is slow

Once damaged, the model does recover, but very slowly, and within this campaign's step budgets it never
gets back below `L0`:

* at `lr = 1e-7`, 30 steps: peak 134x, final 2.05x, never below `L0`;
* extending to 350 steps (D7 seed 0, same protocol): `9.09e-13` against `L0 = 8.85e-13`, still 2.7%
  above the initialisation.

The reason is that the realised step collapses once the gradient starts alternating sign: measured
median per-step move along the protected constant is **`0.005` to `0.013 x lr`** over 9 runs, with the
increment reversing sign on 9 to 22 of 41 tail steps (`docs/drift-conclusions-2026-07-25.md` C5). So the
optimiser damages at full step size and repairs at one hundredth of it.

## 4. What this does and does not establish

**Establishes:** in a regime where the initialisation is optimal, Adam at the campaign's learning rate
makes the loss 90 to 134x worse before recovering, and does not return below the initialisation within
350 steps. That reproduces the user's complaint exactly, in the cleanest possible setting.

**Does not establish:** that this is what breaks training on the MSD data or on the real task. There the
initialisation is *not* optimal and there is a real absorber signal to learn, so a full-size first step
is not automatically damage. The complaint "best checkpoint = epoch 0" could be this mechanism or could
be the train/validate horizon mismatch that `docs/gantry-augmentation-problem-log.md` §3 records as the
original failure mode.

**The one measurement that closes it:** run `d8_why_worse_than_init.py` against the MSD data instead of
the null. The script is rig-agnostic apart from the loss it imports, so this is a small change and no
new theory. If the peak-over-init and the SGD contrast reproduce there, the diagnosis transfers; if they
do not, the MSD failure has a different cause and this document applies only to the null.

## 5. Levers this implies, listed but NOT tested

Diagnosis only. None of these has been run, and each would need its own row in the run table.

* lower `lr` (measured: `1e-8` peaks at 2.0x instead of 134x and does dip below `L0`);
* warmup, which exists precisely because early adaptive steps are wrong-sized;
* RAdam, the principled version of the same idea, rectifying the step until `v_hat` is trustworthy;
* a larger `eps`, which makes Adam behave like SGD wherever gradients are small;
* SGD, which is measured here to do no damage, but which the project already found learns +0% on a real
  residual, so it trades this failure for the other one.

## 5b. Pre-registration: what D8-on-MSD must show

Written BEFORE the run so the reading cannot be fitted afterwards.

### Why the null cannot answer this by itself

On the null the available improvement is **exactly zero**: the init is the global optimum, so any step
is damage and damage wins by construction. The null therefore proves the mechanism EXISTS; it cannot
show whether it MATTERS when there is something to learn. On MSD the available improvement is large
(the absorber signal the augmentation exists to capture), so the question becomes a race:

```
does the useful update the first steps buy   >   the damage a full lr-sized step costs ?
```

That ratio is what D8-on-MSD measures, and nothing measured so far predicts it.

### Hypothesis

**H1.** Adam's per-coordinate first step is `~1.0 x lr` regardless of gradient (measured on the null,
`5.4 x lr` in L2 over ~30 parameters). On MSD this is unchanged, because it is a property of the
optimiser, not of the data. Whether it damages depends on the size of the required parameter update
relative to `lr`:

* **H1a (transfers):** the update needed to fit the absorber is comparable to or smaller than `lr`, so
  the fixed-size step overshoots or misdirects and the loss rises above `L0` before recovering, exactly
  as in the null.
* **H1b (does not transfer):** the required update is much larger than `lr`, the full step lands roughly
  in the right direction, the loss falls from step one, and the null result is an artefact of having
  nothing to gain.

**H2 (the competing explanation, which this run also tests).** The failure is not the optimiser at all
but the train/validate horizon mismatch recorded in `docs/gantry-augmentation-problem-log.md` §3: the
windowed training loss improves while the fixed-eval metric does not.

### What is measured, and why each quantity is there

| quantity | why |
|---|---|
| `L0` = ANN-off windowed loss on MSD | the reference the complaint is stated against ("worse than the init") |
| peak loss / `L0` over the first 30 steps, per arm | does training damage at all |
| ever below `L0`, and the best value reached | can training help at all on this data |
| exponent of peak excess versus `lr` | `lr^2` is the signature of an `lr`-sized step on a quadratic bowl |
| first parameter move / `lr` | tests the mechanism directly: `~1.0 x lr` per coordinate means gradient-independent |
| **SGD at matched `lr`** | the control. Its step scales with the gradient, so it isolates normalisation as the cause |
| **`g.windowed_rms()` on the FIXED eval set, per step** | separates H1 from H2: the complaint is defined on validation, not on the training loss |

Arms unchanged: Adam at `1e-9 / 1e-8 / 1e-7 / 1e-6`, SGD at `1e-7 / 1e-6`, 30 steps, 3 seeds.

### The decision table, pre-committed

| observation | conclusion |
|---|---|
| peak/`L0` > 1 at `lr = 1e-7`, excess scales `~lr^2`, SGD stays at 1.0 | **H1a: the mechanism transfers.** Adam's fixed-size step damages even when there is signal to learn. The lever list in §5 becomes the next work |
| loss falls monotonically from step 1, never above `L0` | **H1b: it does not transfer.** The null damage is an artefact of a zero-gain setting, this document applies to the null only, and the MSD failure has another cause |
| training loss improves but the fixed-eval RMS does not | **H2: horizon mismatch, not the optimiser.** Stop working the optimiser and go to the train/validate horizon |
| SGD also never beats its init | **neither H1 nor H2: the ANN cannot help on this data at all.** An expressivity or routing problem, and the right follow-up is the R2 gate, not a step-size fix |
| best loss reached < `L0` for some `lr`, but not at the campaign's `lr` | training CAN help and the failure is a tuning question, which is a much cheaper problem than either hypothesis |

### What would falsify the hypothesis I am currently defending

If on MSD, Adam at `lr = 1e-7` decreases the loss monotonically from step one and ends below both `L0`
and SGD, then "Adam damages a near-optimal init" is irrelevant to the user's actual failure, and §2 of
this document describes a real but inconsequential effect. I would then drop it and move to H2.

## 6. Provenance

| what | where |
|---|---|
| script, log, units | `scripts/gantry/drift-diagnostics/d8_why_worse_than_init.py`, `logs/d8.output`, `data/D8_lr-sweep_30steps_seed{0,1}.json` |
| `L0` and the loss definition | `docs/drift-conclusions-2026-07-25.md` C1; the same `make_loss` D1 and D7 use |
| the 350-step recovery number | D7 seed 0, `logs/d7.output` (run stopped mid-campaign; one seed) |
| realised step size, 9 runs | `docs/drift-conclusions-2026-07-25.md` C5 |
| scale-freeness prior art | `scripts/gantry/drift-fix-trials/research/thread-AB-optimizer-mechanics.md` A3 |
| the separate drift failure | `docs/drift-conclusions-2026-07-25.md` |

**Seed count:** 2 complete seeds plus a partial third. Below the project's 3-seed floor, because the run
was stopped. Finishing seed 2's SGD arms is a 4-minute job.
