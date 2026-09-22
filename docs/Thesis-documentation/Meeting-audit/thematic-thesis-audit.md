# Thematic thesis audit

This file converts the meeting history into thesis-writing safeguards. It deliberately distinguishes
facts about the present pipeline from useful but abandoned ideas.

## 1. System, coordinates and baseline

What the meetings add:

- Explain why payload position is the scheduling variable: it changes the mass distribution and
  coupling in the physical equations over the operating range.
- State the machine coordinate convention before deriving the equations. Several early discussions
  were caused by mixing rail/stage and logical coordinates.
- Motivate the baseline as an interpretable digital-twin component, not merely as an initialization
  for a neural network.
- Establish well-posedness over the stated operating range, not just at one nominal position.

Do not write:

- that the six-to-four SVD reduction is used;
- that positive parameters alone prove mass-matrix invertibility;
- machine limits from meeting notes without checking the defining model/data source.

## 2. LPV-LFR derivation

Recommended thesis depth:

- Start from the mechanical equation and polynomial mass matrix.
- Show the scheduling normalization/centering and the finite polynomial dependence.
- Show the one key inverse identity or adjugate/determinant step that makes the rational dependence
  explicit.
- Give the resulting LFR interconnection and explain the internal signal dimensions.
- State the well-posedness condition and how it was checked over the operating range.
- Put long coefficient expansions, repeated algebra and implementation-oriented Horner details in an
  appendix or verification note.

The derivation should let a reader reconstruct the method and understand why it is exact. It need not
reproduce every symbolic manipulation performed during development.

## 3. Data design

Required justifications:

- which dynamics and operating regions each dataset is intended to excite;
- why the chosen multisine band contains the relevant pole/anti-resonance or low-frequency behaviour;
- why phases differ between records and partitions;
- why low crest factor matters for actuator/position constraints;
- how force, position, velocity and yaw limits were checked in the closed loop;
- how training, validation and test records differ;
- what region is not covered and therefore outside the claim.

Current-code cross-check:

- The current main configuration points to a merged friction dataset, uses the corrected all-channel
  multisine amplitudes, and also includes multisine-free Telica-derived motion profiles.
- The current output-noise setting is `snr=None`; do not describe measurement-noise training as part
  of the final pipeline unless a different final run configuration is selected.
- The truth contains Garcia-style dry friction in the friction benchmark while the Python baseline
  remains frictionless. This is a truth-model mismatch experiment, not a friction-parameter estimate.

Figures to prioritize:

- 2D/3D coverage of the scheduling/physical state region, with train/validation/test distinguished;
- input spectrum and actual total plant input, not excitation design alone;
- time-domain limit/headroom summary for representative records;
- a small table mapping each record class to its purpose.

Reference need: multisine/random-phase/crest-factor and experiment-design statements require verified
primary sources. Meeting mentions are only search leads.

## 4. Encoder, windows and initialization

Keep four time quantities separate:

- encoder history length;
- simulated/scored training-window length;
- joint-estimation sensitivity horizon;
- free-run or closed-loop validation horizon.

The current main configuration uses a 0.1 s training rollout at 4 kHz and `burn_in=0`. The encoder
history follows the implemented reconstructability convention. The physical justification for 0.1 s
is that it spans multiple cycles/settling constants of the targeted absorber, but it does not cover
the slowest or marginal gantry behaviour. That limitation should be explicit rather than hidden by a
claim that the window captures all dynamics.

For joint estimation, window length should be justified through parameter sensitivity and practical
identifiability, not only the slowest eigenvalue. The meetings and decision record show that a
co-trained encoder can absorb short-window parameter effects and that nominal regularization can pin
parameters once the data objective becomes flat.

Suggested figure: one honest-scale time strip showing encoder history, 0.1 s training window and the
full validation/settling horizon, with no burn-in region.

## 5. Augmentation architecture and routing

The thesis should explain every route in physical terms:

- why the augmentation sees the physical and latent states and input;
- why a dynamic parallel structure is needed to add missing memory/poles;
- why the baseline contribution remains explicit;
- which derivative/state rows receive the learned correction;
- how zero-output initialization preserves the baseline at epoch zero;
- how latent states can affect measured outputs even though their coordinates have no direct physical
  meaning.

Current-code warning: the main entry file currently sets `nx_ann=8` and routes to all 14 interconnected
state rows. Older meetings and comments describe stiff-only, velocity-only and four-row routing.
Those are historical experiments. The final thesis must use the configuration of the actual reported
run and explain why that route was selected, including the marginal-mode risk for X/Y.

Suggested figure: a signal-routing diagram that labels baseline physical-state derivative, learned
correction, latent-state write, fixed output map and controller loop. Put continuous-time/RK4 and
discrete-time/controller boundaries on the same figure or in an adjacent timing figure.

## 6. Closed-loop training

The current method wraps the known controller around the learned model during the training rollout.
The thesis should state:

- the residual-loop equation and sign convention;
- that the same-sample known digital controller is applied with the recorded reference/trajectory
  convention;
- where the continuous-time plant derivative is integrated with fixed-step RK4;
- that no artificial one-sample plant delay is introduced;
- how the controller initial state is handled at each window;
- why the approach matches the available closed-loop data;
- why open-loop/plant-domain evaluation is still needed to rule out controller masking.

Do not introduce burn-in into the method description. It was a diagnostic branch and current runs use
zero burn-in.

Suggested figure: closed-loop training versus ordinary replay in two aligned panels, using the same
symbols for recorded reference, measured output, model output, feedback correction and total input.

## 7. Joint estimation and identifiability

The thesis should parameterize and report the identifiable combinations, not suggest recovery of all
raw component parameters. Explain the rank/flat-direction argument before presenting an optimizer.

Current-code cross-check:

- Joint estimation is off in the ordinary augmentation configuration.
- OBC comparison arms turn it on and use ten reduced identifiable combinations.
- The nominal prior is disabled for those OBC arms because the measured prior gradient would dominate
  recovery away from the detuned start.

Parameter recovery belongs as a verification or ablation unless it is promoted by the final research
question. Its result cannot by itself prove that the augmentation recovered the hidden absorber.

Suggested figure/table: singular values or identifiable directions plus a table mapping raw physical
parameters to the ten combinations used by the model.

## 8. Orthogonal-by-construction

The write-up must name the exact protected object. Based on current decisions, the defensible claim is
an aggregate one-step orthogonality construction in normalized state coordinates relative to a
data-derived reference stack and the range of the baseline sensitivity/regressor.

Do not claim:

- pointwise orthogonality at every sample;
- multistep or closed-loop trajectory orthogonality from the one-step construction;
- uniqueness or physical meaning of the latent state;
- preservation of every baseline behaviour outside the sampled/reference region;
- direct transfer of Györök's theorem without restating the changed assumptions.

The tangent and affine protected-space arms are distinct experiments. Their result and exact
coefficient/basis refresh procedure should come from the final run artefacts and D-190--D-203, not
from meeting slides.

Suggested figure: geometric projection schematic plus a separate architecture panel showing that the
physical correction is projected while the latent-state write has no baseline direction to project
against.

## 9. Validation and claims

Quinten's outline feedback fixes the intended order of evidence:

1. **Interpolation:** held-out phase realizations and operating points inside the represented region.
2. **Operational motion:** multisine-free trajectories that Jasper and Dragan recognize as relevant
   machine motion, including agreed minimum/maximum moves, long stroke and representative acceleration.
3. **Settling:** a fixed 200--400 ms post-move interval shown separately from tracking.
4. **Controller transfer:** train with one known controller and evaluate a credible changed controller
   as the offline digital-twin use case. Report this as closed-loop distribution shift unless the
   physical operating range also changes.
5. **Operating-range extrapolation:** only claims tied to a quantified departure in position, stroke,
   acceleration, excitation amplitude or another defined coverage coordinate.

The present Telica-derived cycloidal records partially address operational motion, but their status as
representative ASMPT trajectories still requires confirmation of the motion parameters and intended
use with Jasper and Dragan. An unseen multisine phase is useful interpolation evidence, not sufficient
application validation.

Minimum comparison set repeatedly supported by the meetings:

- baseline FP model;
- augmented FP model;
- black-box model at a fair capacity/training budget;
- true-state/oracle initialization versus encoder initialization where interpretation requires it;
- open-loop plant-domain and closed-loop operational metrics where both claims are made;
- held-out motion profile and operating-position coverage;
- settling zoom and frequency-domain residual around the missing dynamics;
- multiple seeds for conclusions sensitive to neural initialization.

Neural initialization seeds and multisine phase seeds test different uncertainties and must not be
pooled. Report an aggregate across neural seeds and use a representative rather than best trace where
the compute budget permits repeated runs.

Always separate simulation-truth knowledge from information that would be available on the machine.
Absorber-state correlations and truth-only force decompositions are diagnostic evidence, not deployable
identification signals.

Use one primary predictive metric. Physical RMS is the current default candidate because it is
interpretable during both tracking and low-variation settling; BFR or NRMS should be secondary only if
they answer a distinct question. Similar train and validation performance supports data sufficiency
only when both are good and the validation set covers the intended operating/motion region.

The current simulated output is noiseless. A noise experiment becomes defensible only after its level
and measurement location are estimated from machine, ILC or stationary data. Until then, noiseless
training is a declared scope limitation, not an implicit approximation to real measurements.

## 10. Citation and provenance rules

- A supervisor statement can justify project scope or an engineering requirement, but not a general
  scientific claim that belongs to the literature.
- A meeting deck can identify a useful figure or experiment, but the final figure should be regenerated
  from code/data with provenance.
- A paper title or author in a meeting is a lead only. Read and verify the primary paper before citing.
- Keep the references bibliography disabled until each entry has been verified, as requested.
- The core recurring reading set for writing sessions remains: Hoekstra on encoder initialization,
  LFR and augmentation; Drenth on LPV modelling; Györök on regularization and orthogonal-by-construction.
  The relevant paper must still be read directly for each claim.
