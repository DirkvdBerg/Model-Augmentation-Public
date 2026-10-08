# Web research prompt: learning lightly damped dynamics in augmented (grey-box) state-space models

Paste the text below into a web research session (no access to the local repo is assumed).

---

I am a control-engineering master student (TU Eindhoven, Control Systems). I need a literature-backed answer to a specific training problem. Please research it in depth, using control-systems sources (IFAC, CDC, ECC, ACC, Automatica, IEEE TAC / TCST / TSP, IJC, Systems & Control Letters) as well as ML sources (ICML, NeurIPS, ICLR, TMLR, arXiv). Control authors often rename concepts between papers, so search under several vocabularies: output-error identification, adaptive IIR filtering, simulation-error minimisation, grey-box / physics-guided model augmentation, neural state-space models, SUBNET, LFR augmentation, recurrent network training.

## The setup
- **Plant.** A dual-gantry high-precision motion system, simulated at 4 kHz. Its physics: rigid-body masses plus a payload "absorber", a mass on a spring that is lightly damped. In the input-output behaviour it shows up as a resonance at 212 Hz with damping ratio 0.043 (discrete pole radius |z| = 0.986 at angle 0.333 rad), next to an anti-resonance at 150 Hz.
- **Data.** The plant is in closed loop with a fixed controller, and the data are closed-loop records excited by a multisine (106 to 297 Hz) plus motion profiles.
- **Model: a physics baseline plus a learned dynamic augmentation in parallel.**
  - The baseline is a discrete-time LPV state-space model of the rigid gantry. It has no absorber, and 10 physical parameter combinations are jointly estimated.
  - The learned part is one MLP (2 hidden layers of 16 tanh units). It takes z = [baseline states, added states x_a, input u] and writes two things:
    - an additive correction into the baseline's state update;
    - the full next value of 2 added states: x_a(k+1) = f(z).
  - The MLP's output layer is zero-initialised, so at the start the model equals the baseline exactly and the added states are dead (all added poles at z = 0).
  - The added states' A, B and readout matrices are therefore the MLP's Jacobians: zero at init, growing as the output layer trains.
  - The model never uses a Jacobian; it is only how the added dynamics are measured. A_aa = ∂f_a/∂x_a is the local linearisation at an operating point. Its eigenvalues (and those of the full one-step Jacobian) are the "added poles" in all results below. Because the MLP is nonlinear (tanh), A_aa varies with the operating point, so the reported values are medians over 24 to 672 encoder states.
- **Training.**
  - An encoder (a learned map from past inputs and outputs) estimates the initial state of each window, added states included.
  - Loss: closed-loop simulation error over 400-step windows (0.1 s).
  - Optimiser: Adam, lr 1e-5, about 33k updates, float64.
  - An orthogonal projection removes from the learned correction the part a physical-parameter change could explain, to keep the physical parameters interpretable.

## The problem (measured)
- **Training never learns the resonance.** After training, the added states realise only real poles: a fast lag at z = +0.6 and one near 0. There is no complex pair near 212 Hz at any of 672 operating points, for several seeds, 2 or 4 added states, and with or without the projection. The learned correction matches the true one below the resonance, misses it above (only 22 % of the error removed in 230 to 297 Hz), and is over-damped.
- **A much better solution exists in the same model class.** Setting the same network to the true linear absorber dynamics gives a 7x lower training loss. The objective clearly rewards the resonance.
- **A pole move alone is uphill.** With the drive and readout fixed at their trained values, moving the added poles toward a 212 Hz pair raises the loss 1.8x to 4.6x. Shorter windows shrink that barrier (about 5x lower at 25 steps than at 400), but the full move stays about 4x uphill at every horizon.
- **The value of a coordinated move fades with training.**
  - Early in training (on a variant whose added states can hold memory), fixing the poles at a band pair and re-fitting the drive and readout for 100 updates lowers the loss 2.5x below the learned real-lag solution.
  - At the end of training, the same move only reaches the same loss (a flat plateau). By then the static correction has absorbed most of the benefit.
- **From the zero start, the gradient asks for drive, not memory.**
  - A 100x learning rate on the added-state outputs, or on the hidden layers too, makes the added states learn faster, but only as a delayed static map (pole radius 0.02 after 150 updates, real).
  - A multi-horizon loss (25/100/400 steps, short first) does not change this.
- **When the added states start near z = 1** (diagnostic only), the gradient builds two real lags with a nearly symmetric 2x2 self-map. It never forms a complex pair.

## Constraints on an acceptable remedy (hard)
The remedy must work for unknown missing dynamics on real data, so it cannot carry knowledge of the system:
- **No initialisation or parameterisation change:** no random or identity linear part, no complex/polar/oscillatory/resonator structure, no mode banks, no initialisation from residuals or a best linear approximation, no pole or frequency priors.
- **No loss burn-in.**
- **No penalties that assume what the static path may learn.**
- **The model must stay as is:** same MLP, routing and zero-initialised output, equal to the baseline at the start. The remedy must remain compatible with joint parameter estimation and the orthogonal projection.

**Allowed families:** the optimiser (update rule, learning-rate allocation, second-order methods), the objective (horizon, weighting, multi-rate / multi-resolution training on the same records at several sampling rates, multiple shooting, prediction-error vs simulation-error, homotopies), the training schedule or order, and the encoder (how the initial added states are estimated or co-estimated).

## Questions
1. **Why this happens.** Why does gradient-based output-error / simulation-error training from a zero (or static) start fit a lightly damped mode with static maps or real lags instead of a resonant pole pair? I want theory with exact locations (theorem, equation, figure, page): local minima and saddles of output-error criteria, realization-dependence (direct vs non-direct form), colored or band-limited excitation, closed-loop effects, and the "coordinated pole and numerator move" problem.
2. **Training methods (not parameterisations) known to reach the resonant solution.** Examples: Steiglitz-McBride / iterative prefiltering, instrumental variables and SRIVC, equation-error to output-error homotopies, Gauss-Newton / Levenberg-Marquardt, variable projection / separable least squares, multiple shooting, multi-rate or multi-resolution curricula, horizon curricula. For each, state whether it applies when the dynamics sit inside a nonlinear neural network rather than a linear transfer function.
3. **Multi-rate training.** Does training one model on the same data at several sampling rates (same window length, so coarser rates cover longer time spans) help learn slow or lightly damped dynamics? Under what conditions on rate transfer of a discrete-time learned correction, aliasing, and the weighting schedule (annealed sum vs hard switch)?
4. **Latent states carrying dynamics.** In SUBNET / neural state-space / LFR model-augmentation work (Beintema, Schoukens, Toth, Hoekstra, Bolderman, Forgione, Ljung, Masti, Bemporad, and others), how are latent added states made to carry dynamics rather than static corrections, without changing the parameterisation or initialisation? Does any paper report a learned resonance (pole, damping or FRF check) in added states?

## Output wanted
- **Ranked sources:** at most 10, each with exact citation (DOI or arXiv ID), the specific claim, its location (page, equation, theorem or figure), whether you read the full text or only an abstract, and one line on how it applies to the setup above.
- **Remedies:** for each remedy that satisfies the constraints, the mechanism by which it would attack the measured cause, and a cheap local test that would show it working (for example, the added-state poles moving toward a complex pair near the resonance during the first few hundred updates).
- **Explicit negatives:** what you searched for and did not find, with the queries used.

Keep the report under 1500 words and do not use em-dashes.
