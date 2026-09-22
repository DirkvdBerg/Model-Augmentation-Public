# Meeting context ledger

This ledger groups the scanned material into project phases. It preserves why a topic was discussed
and records whether its conclusion survived. It should prevent an isolated sentence from a meeting
from being mistaken for a final design decision.

## Phase 1: problem definition and industrial scope (February to early March)

Representative packets: ASMPT 5, 9, 13, 16, 19, 23 and 25 February; TU/e 2, 13 and 26 February;
2 and 5 March; research-plan change deck.

- The project began with a broad set of possible directions: inverse/feedforward modelling,
  settling requirements, feature selection, OMNIPUS and model augmentation.
- The scope then moved toward an accurate plant model/digital twin rather than feedforward inversion.
  The industrial motivations repeatedly named are offline controller evaluation, condition or health
  monitoring, and reliable prediction of tracking, settling and cross-coupling behaviour.
- Position dependence was already central. The company baseline was described as FRFs at several
  operating positions with interpolation between them.
- Fixed firmware/controller context and constrained dual-rail motion shaped what could be identified.
  Early notes state that X1/X2 references are constrained together while their forces may differ.
- Numerical limits such as the quoted maximum yaw must be checked against the model/data source
  before publication; a meeting note alone is insufficient.

Status: the plant-modelling scope and industrial motivation survived. OMNIPUS and inverse-first
framing are historical alternatives, not the thesis method.

## Phase 2: exact LPV-LFR baseline (March to April)

Representative packets: 9, 12, 15, 16, 19, 23, 27 and 30 March; 1, 7, 9, 13, 17, 20, 22 and 23 April.

- The baseline was developed as a continuous-time, position-dependent mechanical model with payload
  position `Y` as the scheduling variable.
- The meetings worked through coordinate conventions, the polynomial mass matrix, invertibility over
  the operating range, and the exact rational inverse/LFR representation.
- Roland's notes emphasize centring `Y` for conditioning, checking singular values or determinant
  behaviour over the admissible range, retaining the rational dependence, and resolving the internal
  algebraic loop without discarding the LFR structure.
- A six-to-four-channel SVD reduction was extensively explored. It is not part of the final baseline
  and must not appear as if it were used.
- Several decks begin from the inverse expression. Later supervisor guidance prefers the physical
  polynomial mass equation as the main derivation route, with the inverse form used only to connect
  it to implementation.

Status: exact LPV-LFR and continuous-time modelling are current; SVD reduction and inverse-first
presentation are superseded for the thesis narrative.

## Phase 3: identifiability, data and simulated mismatch (April to June)

Representative packets: 28 April; 11, 13, 18, 19, 26 and 27 May; 1, 4, 8, 11, 15, 17, 22, 24 and
29 June.

- The parameter-recovery work exposed structural non-identifiability of several raw physical
  parameters. The meetings repeatedly returned to identifiable combinations such as total/difference
  mass and summed damping/stiffness terms.
- Regularising a split toward equal or nominal component values does not make the individual
  components identifiable. The defensible thesis object is the identifiable combination.
- Global channel normalization across the dataset was recommended. Per-trajectory normalization was
  rejected because it removes the magnitude differences needed for cross-trajectory identification.
- The synthetic truth gained a physically motivated hidden mass-spring-damper absorber. Its formulas
  were checked against Simscape after an early wiring/offset problem was corrected.
- The extra learned states are latent coordinates. Input-output data alone does not establish that a
  learned state equals the physical absorber displacement or locates the absorber on the machine.
- The hidden mismatch was later deliberately strengthened, including a larger mass split, to make it
  learnable. That is an experimental design choice and must not be presented as a measured machine
  property.
- Excitation evolved from Schroeder/odd-line ideas to random-phase MIMO multisines with candidate
  selection for low crest factor. The frequency band was repeatedly tied to the dynamics to be
  identified, and input/position/velocity/yaw limits had to be checked in closed loop.
- Coverage plots in `X`, `Theta`, `Y` and their combinations were repeatedly proposed to show which
  operating region is actually represented by training and validation data.

Status: identifiable combinations, global normalization, the latent-state caution and the need for
dynamics-based excitation justification survived. Exact final bands, amplitudes, splits and data
partitioning must come from current generators/configuration rather than these meetings.

## Phase 4: augmentation architecture and encoder initialization (June to July)

Representative packets: 15, 17, 22, 24 and 29 June; 6, 7, 13, 15, 20, 23, 27, 28 and 31 July.

- Parallel dynamic augmentation was selected because a static correction cannot add the missing pole
  pair or memory while the physics baseline should remain visible.
- Hoekstra's encoder initialization and augmentation papers became central to the implementation.
  Early separate augmented-state encoder constructions were replaced by a unified/shared state
  architecture.
- Zero output at initialization preserves exact equality with the baseline, but it can also create
  weak or dead gradient paths. The meetings explored direct output routes, velocity routes, full-state
  routes and additional-state writers as this issue became visible.
- Additional-state magnitude is not a valid measure of contribution because latent coordinates can
  be rescaled. Contribution must be measured through the output or physical-state effect.
- Parameter recovery was reframed as machinery verification and a bias/identifiability study rather
  than the main final result.

Status: parallel dynamic augmentation, shared encoder context and latent-state caution are current.
The precise routing changed repeatedly; only current code can define the thesis method.

## Phase 5: marginal modes, horizon mismatch and closed-loop training (July to August)

Representative packets: 13, 15, 20, 23, 27, 28 and 31 July; 12, 14, 17, 24, 26, 27 and 31 August.

- The free X and Y axes contain marginal/integrating modes. Small learned velocity or force biases can
  accumulate into large position drift over a long simulation.
- Short SUBNET windows can contain many cycles of the high-frequency absorber while covering only a
  small fraction of the slow rigid-body/closed-loop behaviour. This is the core physical reason the
  training horizon needs explicit justification.
- The meetings explored longer horizons, curriculum training, multiple shooting, ARTBP, altered
  normalizers and burn-in. These are not interchangeable and several proposed causal explanations
  conflict with one another.
- Open-loop replay of data generated under feedback exposed initial-condition offsets and drift.
  A closed-loop residual rollout was introduced in which the known controller reacts to the model
  error around the recorded trajectory.
- Closed-loop fitting can improve the operational prediction objective but can also allow feedback to
  conceal plant-model errors. Plant-domain/open-loop evaluation therefore remains important if the
  thesis claims recovery of the plant rather than only closed-loop prediction.
- Burn-in was investigated in diagnostics. The current production configuration has `burn_in=0` and
  scores the entire window. It must not be described as part of the final method.

Status: the marginal-mode/horizon issue and closed-loop training are current. Burn-in and many
intermediate remedies are historical diagnostics. Any causal training claim needs a result or
decision record, not a slide narrative.

## Phase 6: orthogonality, final evaluation and thesis structure (September)

Representative packets: 2, 7, 11, 14, 18 and 21 September, including the thesis-outline meeting.

- The focus moved from a soft orthogonality regularizer toward orthogonal-by-construction (OBC).
- Györök's setting differs from this project: the gantry has a nonlinear/quasi-LPV baseline, an
  encoder, closed-loop rollout and additional latent states. The thesis must state exactly what
  subspace, coordinates and data distribution the construction protects.
- The current decision trail protects the range of a one-step baseline sensitivity/regressor in
  normalized state coordinates. It does not prove pointwise, multistep, closed-loop or latent-state
  uniqueness.
- The learned latent write itself is not orthogonalized where the baseline has no corresponding
  additional-state dynamics. Claims must concern the projected physical correction and the measured
  aggregate orthogonality, not a physically unique latent state.
- Evaluation discussions converged on realistic motion profiles, settling zooms, held-out operating
  conditions, multiple seeds where feasible, and fair comparisons among baseline, augmented model
  and black box.
- Presentation feedback repeatedly requested plots rather than prose: reference/input/output/error,
  total plant input, training and validation curves, coverage, settling and contribution ablations.

Status: OBC is an opt-in current experimental arm and requires careful claim boundaries. Exact final
results should be taken from the final run artefacts, not the September presentation decks.

