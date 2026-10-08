# Handoff: slides that motivate the PS2 method for Jan Hoekstra (beyond RMS)
**From**: session of 2026-10-07 | **Branch**: Augmentation | **Effort suggested**: high (argument construction, no code)

This brief is self-contained: the receiving session has no access to the repository. Every number below is measured
and carries its source run; nothing needs to be looked up.

## 1. Task

Build a short slide deck (about 8 slides plus backup) for an online meeting with Jan Hoekstra (author of the model
augmentation framework) that motivates the PS2 method. Jan's request, verbatim (Dutch): "Wat ik mis in de uitleg die je
hebt gestuurd is een motivatie voor de methode. Probeer een analyse te maken van wat er fout gaat in de huidige methode en
wat je dan specifiek probeert op te lossen met de nieuwe methode die je introduceert, verder dan alleen de RMS score."
So the deck must (a) analyse what goes wrong in plain S-DP training, (b) derive from that cause what a remedy must do,
(c) show that PS2 is built from exactly those requirements, with a classical justification, and (d) show evidence that
each requirement is met, before any RMS. The argument is a chain: cause, requirements, method, evidence. It must justify
PS2 positively, not only argue against alternatives.

## 2. Out of scope

- No new experiments or numbers: use only the numbers in Sections 4 and 7.
- No thesis text, no LaTeX.
- Do not present superseded methods as methods (user rule: slides show the current method only). The ingredient
  ablations in Section 6 may appear on ONE backup slide, framed as "remove one ingredient", not as earlier methods.
- No claims of guarantees, of physical identification of the added states, or of generality beyond the tested systems.

## 3. Where things stand

PS2 is implemented in the production pipeline. Server runs on noisy simulated gantry data with Coulomb friction are
partly finished (interim best checkpoints). An email describing the two phases was sent to Jan; he replied asking for the
motivation above. Meeting: online, same day.

**System and setup (context for the slides).**
- Plant: a dual-gantry high-precision motion stage, simulated, 4 kHz, in closed loop with a known controller. Outputs
  X1, X2, Y; inputs three stage forces.
- Baseline: a physics model with 6 physical states x_b (X, Theta, Y and their rates). The truth contains a payload
  absorber the baseline omits (211.9 Hz, damping ratio 0.043), plus Coulomb friction and measurement noise. Training data
  are multisine-excited up to 297 Hz.
- Augmentation (Jan's S-DP, dynamic parallel): 2 added states x_a.
  x_b(k+1) = f_base(x_b, u) + f_aug(x_b, x_a, u); x_a(k+1) = g_aug(x_b, x_a, u); y = h_base(x_b, u).
  f_aug and g_aug are the output rows of ONE tanh network (2 hidden layers of 16) whose output layer is zero at the start,
  so the model starts equal to the baseline (Jan's framework, arXiv:2602.17297, Sec. 5.4).
- Encoder (SUBNET): estimates the initial state from the past 30 samples of inputs and outputs (lag 29 plus the current
  sample). Structure: a linear map plus a nonlinear network, added. The x_b rows of the linear map (W^b) come from the
  baseline's reconstructability map; the x_a rows (W^a) are random.
- Training: closed-loop simulation error over 400-sample windows (0.1 s) with the known controller, Adam, learning rate
  1e-5 for all parameters, about 33,400 updates, best validation checkpoint selected. Two arms: U (unconstrained) and
  OBC (orthogonal projection that protects the physical parameters).

**PS2 in one paragraph.** Phase 1 (first 350 updates, alongside normal training): a temporary network (2 inputs, 16 tanh,
90 outputs; 1578 parameters) maps the encoder's x_a to the model's next 30 closed-loop output residuals (30 samples x 3
channels: measured output minus the current model's closed-loop simulation from the encoder state, recomputed every step).
A separate Adam (1e-3) trains only this network and the encoder weights that produce x_a (the W^a rows of the linear map
and the x_a rows of the encoder network's last layer). Phase 2 (once, directly after): the temporary network is deleted;
8192 pairs are built with the encoder, input [x_b(k), x_a(k), u(k)] from the window ending at k, target x_a(k+1) from
the same encoder on the window shifted one sample; the existing augmentation network's x_a output rows and shared hidden
layers are fitted by plain regression (separate Adam, 1e-3), stopped on held-out training records, best weights kept;
the encoder and the f_aug output rows do not change. Then normal training continues unchanged. The final model is
standard S-DP; no parameters are added. Safeguard: if phase 1 predicts less than 10 % of the residual on held-out
records, phase 2 is skipped (never triggered so far; note that the phase 1 encoder changes then remain).

## 4. Established and verified (use these, with their sources)

**A. The asymmetry in the current structure (the core of the cause).** The output layer is zero for ALL rows (f_aug and
g_aug); the mismatch is in everything around it:

| | Physical states x_b | Added states x_a |
|-|-|-|
| Dynamics at the start | the baseline model f_base | none (g_aug = 0) |
| Encoder initialisation | W^b from the baseline (reconstructability) | W^a random |
| Path to the output | direct, through h_base | only through f_aug, which is zero |
| Gradient at the start | yes | exactly zero |
| Learning rate | 1e-5 | 1e-5, shared |

Measured gradient of the normal training loss at initialisation (RMS over 4 batches of 64 windows, 400 samples):
encoder x_a rows 0, g_aug rows 0, shared hidden layers 0; f_aug rows 1.9e-5, encoder x_b rows 1.4e-7, physical
parameters 7.1e-9. So at the start ONLY the physical correction and the physical side learn.
Key sentence: the physical states start with a model, a meaning and a gradient; the added states with none of the three,
in the same network at the same learning rate. The physical correction learns first and takes the static part of the
error; the memory part never gets its own signal.

**B. What goes wrong, by elimination (plain S-DP, run 42 = U arm, seed 1; each prediction written before its run).**
- Symptom: the reference removes only 9 to 20 % of the Y error in 230 to 297 Hz (the absorber band); zeroing its added
  states changes it by only 17 %; its added states have real eigenvalues only, no oscillatory pair (0 of 672 encoder
  points with a pair in 95 to 380 Hz, runs 42, 44, 48).
- Not capacity (K1): the true absorber written into the same network gives training loss 7.0e-10 against 5.1e-9 for
  the trained model, 7x lower.
- Not the objective (G1): the true system scores 1.35e-9; the loss rewards the absorber.
- Not the loss landscape (L1): scanning the added pole's radius and frequency, the loss falls steadily toward the true
  pole, with a well about +-40 Hz wide around 212 Hz.
- Not the optimiser (U1, U4, Levenberg-Marquardt): other learning rates and 30 Gauss-Newton steps leave the added poles
  real and unchanged, 0 % complex at every step.
- The cause, order: the static path learns first (A above), and afterwards it blocks the mode (M1c): grafting the
  correct absorber onto the trained model lowers its loss only to 0.85x (15 %), against 0.06x on the model without the
  learned block.
- Not only two paths competing (N1, O1): with the static path switched off, the added states themselves first learn a
  delayed static map (negative real eigenvalue). Whatever learns first from zero learns the memoryless part.
- Therefore training alone does not suffice, a random initialisation does not either (it carries no information about
  the missing dynamics), and the initialisation must carry memory, which without system knowledge can only come from data.

**C. Requirements derived from the cause.**
- R1: a training signal for x_a that rewards memory, not static correction, before the model's own loss shapes it.
- R2: no system knowledge (no poles, frequencies, absorber), only data.
- R3: a direct, first-order training signal for the transition g_aug.
- R4: the final model and its objective unchanged (standard S-DP).

**D. How PS2 meets them, and the classical justification.**
- R1, phase 1: predict the FUTURE residual from the PAST window. A static correction acts on the current sample only
  and cannot make the past predict the future error; only dynamics with memory can. The target rewards exactly the
  memory part by construction. The head sees only x_a (2 dimensions): a bottleneck, so x_a must summarise that memory.
- R2: the target is the model's own residual on measured data.
- R3, phase 2: once x_a exists as data, the transition is a regression.
- R4: the head is deleted; normal training continues on the unchanged objective.
- Classical basis, subspace identification (Bauer, "Order estimation for subspace methods", Automatica 2001, Sec. 2):
  all subspace algorithms (1) regress the future outputs on the past (and future inputs), (2) take a rank-n
  approximation, (3) estimate the state as a map of the past, x(t) = K_p Z_past(t), then obtain A, B by least squares of
  x(t+1) on [x(t), u(t)]. PS2 phase 1 is steps 1 to 3 applied to the residual with a nonlinear map and a 2-d bottleneck;
  phase 2 is the least-squares step with the nonlinear network.
- Framework fit: the counterpart of W^b. The baseline supplies the state map for x_b; the data supply it for x_a.
- Supporting literature: Hefny, Downey, Gordon, NeurIPS 2015 (predictive state, the first-stage regression may be
  nonlinear, p. 4); Downey et al., PSRNN, NeurIPS 2017 (two-stage regression initialisation then backpropagation
  refinement, Sec. 4.2; random initialisation followed by backpropagation failed in their experiments, p. 8).
- Differences to state honestly: PS2 predicts residuals, not outputs; its head does not see future inputs or reference
  (Bauer's step 1 does); nonlinear and Adam-trained, so the subspace consistency results do not transfer; closed-loop data
  (subspace methods are known to be biased in closed loop).

**E. Evidence that each requirement is met (production server runs, interim).**
- R1: after phase 1, x_a predicts 32 to 42 % of the residual on held-out training records (unexplained 0.58 to 0.68).
- R3: the phase 2 one-step error drops from 1.0 to 0.004 to 0.011 (3100 to 3300 steps).
- The model uses x_a: zeroing the added-state rows makes the PS2 model 8 to 9 times worse (reference: 17 %).
- The absorber band: PS2 / reference Y error in 230 to 297 Hz is 0.08 to 0.11 (about 90 % removed vs 9 to 20 %).
  X below 106 Hz unchanged (ratio 0.94 to 1.04): the friction correction is not traded away.
- Quiet records (no multisine): no record worse than the model with the learned block off (the reference is worse).
- Diagnosis only: the added-state block has an oscillatory pair at 213 Hz, damping ratio 0.045 (U) and 0.034 (OBC);
  truth 211.9 Hz, 0.043. This describes the learned realisation, not a physical pole (for nonlinear coordinate changes
  the eigenvalues are not invariant; the physical states are held fixed in the computation). Present as supporting.

## 5. Assumed but not verified

- Validation numbers are interim best checkpoints of unfinished runs; U seeds 2 and 3 are still running; no test-set
  result yet. Validation records are also used for checkpoint selection on both sides (fair comparison, not a test claim).
- "Subspace methods are biased in closed loop" is standard knowledge, not checked in a source this session.
- The PS2 settings (350-update phase 1 cap, 10 % screen, phase 2 stopping rule) are heuristics; the cap ended phase 1 in
  every run; no sensitivity study.
- Which initialisation produced the S-DP results in Jan's paper is not determinable from his public code (it zero-inits a
  linear map); open question to Jan.

## 6. Tried and failed (ingredient ablations; local runs of PS2's predecessor, which uses the same two ingredients,
target first then map fit, as temporary loss terms; batch 64, 900 updates, mostly one seed per variant)

| Variant | Missing ingredient | Added-state pair in band | Free-run error |
|-|-|-|-|
| plain (U1) | all | 0 % of points | 1.15e-5 m |
| map fit on random x_a (E1, E2) | the target (phase 1) | real lags only | |
| target, too little map fit (E3) | enough fit (phase 2) | 46 %, damping 0.99 | |
| map fitted before target formed (E4) | the order | 46 %, damping 0.72 | |
| full (E5, E9) | none | 96 to 100 %, 208 to 209 Hz | 3.8e-6 m |

Use on one backup slide only: "each ingredient is necessary".

## 7. Achieved (results, after the mechanism slides)

Validation closed-loop free-run error (best checkpoints so far):

| Arm, seed | PS2 | Plain S-DP |
|-|-|-|
| U, 1 | 1.597e-6 m | 1.027e-5 m |
| OBC, 3 | 1.808e-6 m | 1.035e-5 m |

OBC seeds 1 and 2 also improved (1.992e-6 m and 1.643e-6 m), but their reference checkpoints were copied before their
best epoch, so do not put those reference numbers side by side; mention only as "consistent".
Other tests: local gantry prototype 3 of 3 seeds better (median ratio 0.28); cubic (hardening) absorber spring, small
local test, validation error 45 to 50 % lower, test records 34 to 49 % lower, 3 seeds.
Negative result to show: on a hidden-Duffing mass-spring benchmark in open loop where plain S-DP already learns the
hidden mass, PS2 gives no gain. PS2 targets the case where plain training does not form the state.

## 8. The open question (for Jan, last slide)

Is PS2 an extension of the framework (a data-driven initialisation of the added-state encoder and transition, the
counterpart of W^b for x_b, grounded in subspace identification) or a workaround for this gantry? Supporting points for
"extension": no system information, no change to model or objective, it fills a gap the framework leaves (x_a has no
initialisation principle). Supporting points against: heuristic settings, one plant and one controller, no gain where
plain S-DP already works.

## 9. Next action

Produce the deck in this order, one claim per slide, each slide stating its claim as the title:
1. The problem (symptom beyond RMS: band error, clamp 17 %, no oscillatory pair; the class CAN represent it, 7x).
2. The asymmetry (table A and the measured zero gradient).
3. Why training alone is not enough (order: static first; graft gains only 15 %; static path off still learns static).
4. Requirements R1 to R4.
5. PS2, two phases, each mapped to its requirement (include a simple diagram: past window -> encoder -> x_a -> head ->
   future residual; then pairs (k, k+1) -> regression of g_aug).
6. Why this method: subspace identification steps mapped to the phases; counterpart of W^b; differences stated.
7. Evidence the requirements are met (E).
8. Results and limits (Section 7, Section 5 caveats, Gate D negative).
9. Question to Jan (Section 8).
Backup: ingredient ablations (Section 6); PS2 settings table (phase 1: head 2-16-90, lr 1e-3, 350 updates; phase 2:
8192 pairs, lr 1e-3, stop on held-out records).

## 10. Acceptance criterion

Every slide's claim is supported by a number from Sections 4 or 7 with its source label, and the chain is complete:
a reader who sees only the slide titles gets cause, requirements, method, evidence, result, question. Style: flat short
bullets; errors in metres in scientific notation (e.g. 1.597e-6 m or 1.597 x 10^-6 m), never micrometres; no em-dashes;
RMS appears only on the results slide.
