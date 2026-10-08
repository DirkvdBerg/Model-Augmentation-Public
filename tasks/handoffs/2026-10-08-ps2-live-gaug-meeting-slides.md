# Handoff: discuss and revise the PS2 / live-g_aug meeting slides

## User request and workflow

Help the user outline revisions to the existing meeting deck, especially the figures
showing prediction error versus frequency. Preserve the current slide structure and
formatting style as much as possible. Do not replace it with a completely new deck.

**First read the files and discuss a concrete outline and figure plan with the user.**
Do not immediately edit the deck. After the outline is agreed, carry the work through
editing, rendering, visual inspection, and delivery. Keep discussion short and concrete;
the user is frustrated by long, repeatedly changing explanations.

The user explicitly requests this implementation workflow:

1. **Edit the .pptx directly:** change text, layout, images, and speaker notes using
   `python-pptx`, without manually opening PowerPoint to edit.
2. **Render and show the result:** export to PDF and/or slide PNGs. On Windows, use
   installed PowerPoint through background automation for faithful rendering if
   available. Open the PDF in this app's side panel using available preview capabilities.
   If the session lacks a preview tool, provide the artifact and explain the limitation;
   do not claim to have opened a preview that you cannot actually show.
3. **Inspect the rendered slides yourself:** check overflow, alignment, clipping,
   figure readability, equations, and style consistency. Correct problems and rerender.

Work on a clearly named revised copy and retain the original. Preserve fonts, colors,
backgrounds, spacing conventions, aspect ratio, equation style, and incremental builds.
Inspect any features that python-pptx might not preserve before rewriting the file.
Use existing dependencies where possible. Do not install packages, submit jobs, alter
training code, or rerun full local training. Agree on a bounded evaluation plan before
expensive full-record local simulations; prefer existing saved predictions/results.

## Workspace and main files

Repository root:

`C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation`

Main deck:

`scripts/gantry/meeting/meeting-08-10-2026/slides/08-10-2026-additional-states-black-box-noise.pptx`

Results and logs:

- `scripts/gantry/meeting/meeting-08-10-2026/server/`
- `scripts/gantry/meeting/meeting-08-10-2026/server/Checkpoints/`
- `scripts/gantry/meeting/meeting-08-10-2026/server/initial-g/`

Relevant implementation and documentation:

- `model_augmentation/utils/torch_nets.py`
- `model_augmentation/fit_systems/pre_encoder.py`
- `model_augmentation/fit_systems/ps2_opening.py`
- `scripts/gantry/gantry_dynamic/{config,model,data,training}.py`
- `scripts/gantry/thesis-results/{LIVE-COMPARISON.md,live_compare.tsv}`
- `scripts/gantry/thesis-results/{inspect_checkpoint.py,band_compare.py}`
- `scripts/gantry/thesis-results/{test_live_comparison.py,logs/test_live_comparison.log}`
- `docs/ps2-implementation.md`
- `docs/decisions.md`, D-239 including its dated update

Paper, if needed for exact source claims:

- `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`
- `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf`

Use local sources and verify claims independently. The working tree contains many
unrelated changes; do not attribute all changes to this experiment or modify them.

## Current deck structure (physical PPTX slide numbers)

There are 80 physical slides, many of which are successive builds of one logical slide.
Do not interpret that as 80 distinct topics or remove the builds indiscriminately.

- 1: title, data-driven initialization of added states.
- 2: augmented-model diagram.
- 3-4: residual error in the absorber band and weak added-state clamp effect.
- 5-8: capacity and loss diagnostics, then initialization hypothesis.
- 9-12: original zero-output MLP initialization and gradient asymmetry.
- 13-16: hypothesis that early physical correction learning discourages the missing dynamics.
- 17-20: desired properties of the fix, R1-R4.
- 21-24: predictive initialization idea and subspace-identification analogy.
- 25-28: timeline: ordinary updates, phase 1, phase 2, ordinary training.
- 29-37: phase 1 details, residual target, temporary predictor, calibration criteria.
- 38-42: phase 2 details, fitting the added-state transition on encoder pairs.
- 43: same deployed model after the opening.
- 44-48: original experiment evidence, clamp effect and frequency-band result.
- 49-50: original experiment overall RMS and absorber-band RMS, with/without OBC.
- 51: limitations.
- 52: discussion, currently framed as extension versus workaround.
- 53-58: black-box comparison.
- 59: backup ablation.
- 60-80: measurement-noise-model section, also with incremental builds.

Keep the original PS2 explanation/results recognizable. Propose inserting a compact
live-initialization section after the original results and before the final PS2
discussion / black-box section. Add an opening signpost that both zero and live
initialization will be discussed. Condense only where useful and agreed.

## Scientific narrative the user wants

1. Originally: a tanh MLP, with its complete final layer zero-initialized.
2. Original evidence: plain training leaves error near the absorber band; PS2 improves it.
3. Following Jan's feedback, ask whether PS2 still helps when g_aug starts nonzero.
4. Since the previous architecture was MLP, retain a live MLP and compare with a live
   ResNet. The ResNet supplies a direct path for representing linear dynamics.
5. Explain both live initializations clearly, and the numerical problems encountered.
6. Present preliminary evidence without claiming that the bypass caused the later NaN.
7. Show residual error versus frequency again, as Jan explicitly requested, to assess
   whether the frequency-region error is actually reduced.

Do not silently convert the meeting into a defense of one architecture. The comparison
is still incomplete. Do not say zero initialization is proven to be the sole cause.

## Exact initialization details

Normal dimensions: 6 physical states, 2 added states, 3 inputs; learning-function
input z = [x_b, x_a, u] has 11 components. MLP has two hidden tanh layers, width 16.

| Component | Original MLP | Live MLP | Live ResNet |
|---|---|---|---|
| Hidden MLP layers | Random | Random | Random |
| Physical MLP output weights | Zero | Zero | Zero |
| Added-state MLP output weights | Zero | Random, bounds 0.25 | Zero |
| Final-layer biases | Zero | Zero | Zero |
| Physical linear bypass rows | Absent | Absent | Zero |
| Added-state linear bypass rows | Absent | Absent | Random, bounds about 0.302 |

The live weights use PyTorch nn.Linear's default fan-in scale:
`nn.init.kaiming_uniform_(a=sqrt(5))`, equivalent to uniform bounds `1/sqrt(fan_in)`.
MLP final weights receive 16 hidden features: bounds 1/sqrt(16) = 0.25.
ResNet bypass receives 11 inputs: bounds 1/sqrt(11) = 0.301511.
These are one rule applied to different dimensions, not independently tuned bounds.
**Do not call these learning-function initializations Xavier.**

Current added-state encoder: one complete history matrix receives Xavier uniform,
gain 1, bounds sqrt(6/(fan_in + fan_out)), then is split into stored Wy and Wu.
The analytical physical encoder map is unchanged. The user's latest request explicitly
asks to state how current Xavier initialization works, but they do not want a main-slide
digression naming the older encoder's Kaiming initialization or emphasizing the encoder
change. Keep the current explanation clear. Do not imply old/new campaigns differ only
in g initialization: label their datasets/setup separately.

Both live architectures preserve initially baseline-equivalent physical predictions.
Live g does NOT immediately obtain an output-loss gradient at the exact start: the
physical correction is zero. The path becomes trainable after coupling starts learning.
This is verified by the focused preflight, including subsequent nonzero gradients.
Also avoid saying f_aug initially depends only on x_b and u: its network receives x_a
too, and the encoder can produce a nonzero initial x_a even in the original setup.

Jan's reported instruction was g_aug should not be zero; it did not prescribe this
particular scale or conclusively choose our architecture. The general 2026 paper has
NN(z) + W_a z, zero initial nonlinear output, and U(-1,1) unconstrained matrix entries.
Its specific parallel simulation experiments used feedforward networks. Inspected
public parallel examples zeroed the augmentation. Keep those source distinctions.

## Results already inspected; recheck for updates

Original logs are in the server directory, outside initial-g. They use `noise=noisy`
and the older setup. The live campaign uses `Coulomb-tanh-and-MSD-lowpass-noise`.

Useful matched-update original evidence, seed 1, update 7,800:

- U plain, `thesis-ps2ctl_87706_121.out`: validation 1.158e-5 m.
- U PS2, `thesis-ps2_87702_111.out`: validation 1.989e-6 m.
- OBC plain, `thesis-ps2ctl_87701_122.out`: validation 1.162e-5 m.
- OBC PS2, `thesis-ps2_87700_112.out`: validation 1.992e-6 m.

Existing deck uses other best-checkpoint snapshots. Verify checkpoint provenance and
budgets before updating those values; do not mix latest, best, and matched-update
numbers without labeling. Some old logs are incomplete. Do not assume all seeds finished.

Live ResNet logs in initial-g:

- Array task 161/plain: `thesis-tcompare-resnet_88531_161.out`; internal job 88532.
  At update 2,600 validation = 1.016e-5 m. At update 15,600 = 1.00276e-5 m.
  Copied log ends around halfway through 33,400 configured updates; no NaN shown there.
- Array task 162/PS2: `thesis-tcompare-resnet_88531_162.out`; internal job 88531.
  At update 2,600 validation = 3.998e-6 m: about 61% below plain at the same update.
  S1 ended at cap 350, unexplained calibration residual 0.5089.
  S2 calibration residual 0.7930 -> 0.0123 in 2,200 regression steps, opening completed.
  Ordinary training subsequently hit NaN at update 2,889. Best weights were restored.
  L-BFGS polish attempted 16,384 windows in one chunk and failed with CUDA OOM on
  an approximately 11 GB GPU; log says weights unchanged.
  Best-checkpoint evaluation: validation 3.998e-6 m; test 2.375e-6 m.
  Test record E5 was skipped because it used 1.2 times the controller gain.
  The "Training done | 200 ep" banner is configured budget, not achieved training:
  the run stopped around 17.3 epochs. Do not describe it as cleanly completed training.

Initial U(-1,1) live ResNet seed 1 had overflow after one optimizer update in an
earlier 400-sample local trial. The fan-in alternative avoided that immediate failure.
It is conditioning, not a general stability guarantee. rho > 1 alone was not treated
as an implementation bug. Do not infer that later NaN has the same cause without tracing.

Checkpoints observed in initial-g:
`SSE_Interconnect_Composed_88531_best.pth`, `_88531_last.pth`, `_88532_best.pth`.
Checkpoints also exist under server/Checkpoints for several older internal job IDs.
Map array task IDs to internal job IDs from each log header; do not guess by filename.

Live MLP jobs had Slurm launch failures/root-attributed cancellations. One cancellation
occurred during data loading, no Python traceback; PS2 sometimes had no output file.
Cause was not established; controller/node logs were inaccessible. Later the user
reported MLP had reached finite initial validation. Say "launch/cancellation issues",
not "MLP numerical crashes". Recheck for newly copied logs before declaring results pending.

## Figure plan to discuss before generating

Priority: Jan requested **prediction residual versus frequency** again.

- Use residual `e = y - y_hat` from the same closed-loop full-record validation chain.
- Show baseline, plain and PS2; emphasize Y and the 230-297 Hz region, retain enough
  context to see errors elsewhere. Confirm how that band relates to the absorber;
  do not equate a fitted local pole frequency with the entire error band.
- Propose one main figure per setup: original MLP comparison and live ResNet comparison.
  Add live MLP only when results are available. Do not overlay different datasets as
  if they were an isolated initialization comparison.
- Specify spectrum definition, physical units, sampling rate 4 kHz, windowing/segment
  settings, frequency range, checkpoint selection, aggregation and record groups.
- Keep whole-record RMS alongside band RMS. Distinguish RMS reduction from power reduction.
- The existing band helper uses Parseval rFFT energy. If a Welch PSD/ASD is used for
  smooth curves, explain and apply one consistent estimator to every plotted model;
  do not mix spectral magnitudes and integrated RMS without defining the relationship.
- Separate multisine validation records from quiet records where appropriate. Do not
  concatenate record boundaries to form one artificial time series.
- Reuse existing figure styles and artifact scripts where available. Export figures
  with standard plotting tools, not generated imagery.
- Secondary figures: matched-update validation curve and a compact initialization
  diagram/table. Avoid making numerical failure diagnostics dominate the PS2 story.

Checkpoint evaluators now require adjacent config.json to reconstruct architecture and
dataset. Copied checkpoint directories may lack it. Never fabricate a saved configuration
or blindly default to current ResNet when loading an old MLP. Search for original metadata;
if unavailable, document any explicitly reconstructed evaluation config and verify it
against the logs/checkpoint. Use strict state-dict loading. In particular, matching
tensor shapes does not by itself verify normalization or the dataset.

Existing text claims (90% band reduction, 8-9x clamp effect, etc.) may come from earlier
analyses. Verify before carrying them into new captions. A random live initialization
is not proof that the added states encode the missing absorber.

## Deliverables

First reply: a short proposed revision outline mapped onto the current deck, a prioritized
figure list, and any missing evidence. Discuss that with the user.

After agreement: revised .pptx preserving the original style/builds, figure source and
exports, PDF/slide renders, visual QA, and a concise explanation of which conclusions
are supported and which are provisional. Keep source/checkpoint references in notes
or accompanying figure metadata so the main slides stay readable.
