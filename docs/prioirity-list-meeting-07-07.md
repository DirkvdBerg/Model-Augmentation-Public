# Priority List and Decoded Notes: Supervisor Meeting 07-07-2026

Source: `presentations/07-07-2026.pdf` (Meeting Week 23) plus raw meeting notes.
Central problem discussed: open-loop drift-off in the augmented model (slide 21).

## Priority list (supervisor asked for one)

The critical path is: augmentation reaches the noise floor WITHOUT joint estimation first. Everything else waits.

1. **Turn noise off.** Set `SNR = None` in `gantry_interconnect_dynamic.py`. Make it work noiseless first; noise (60-80 dB, not 50) comes back only when results approach the floor.
2. **Isolation experiment for the drift.** Generate the simplest possible record: reference r = 0, feedback controller killed, no feedforward, only the injected multisine force. Complete open loop. A perfect zero-mean periodic multisine on the true system needs no controller and should cause no drift. If the baseline model still drifts against this data, the error is in the input/mass/integration treatment, not in the closed-loop data path.
3. **Check input and initial state.** Double-check that the total force given to the model matches what the actual (Simulink) system receives, and that simulation starts at the same position/velocity as the data. Hypothesis from slide 21: the two initializations (encoder-init, x_logical-init) drift to two different constant offsets, which points at an initial-velocity error decaying through damping rather than a force-scale error (that would push both the same way).
4. **Visualize where the issue comes from.** State-level error plots (position and velocity error per axis over time), not only loss curves and output overlays. Each run must be preceded by an explicit hypothesis and expected signature in the data; runs are too slow for rapid trial-and-error cycles.
5. **Fix the validation measure.** Double-check train vs validation consistency, and consider multiple validation methods. Training uses nf = 0.1 s windows, so validation should first be judged on 0.1 s horizons; slide 21 already shows divergence within 0.1 s, so the model fails even on its trained horizon. Full-trajectory sim-RMS measures something training never optimized.
6. **Then consider increasing nf.** nf must be long enough to see the dominant dynamics; find the spot between runtime and capturing the dynamics of interest.
7. **Control experiment for optimizer settings.** Baseline parameters set exactly to the true system, ANN active. The ANN should then learn (approximately) nothing; if it learns something and degrades the result, the learning rate / step size is too high (a too-high learning step can blow up the neural network).
8. **Last resort: fit velocity instead of position.** Position output has an integrator (X and Y effectively a double integrator on the force path), so errors accumulate. Fitting velocity removes one integrator; fitting acceleration removes both. Supervisor is not convinced positions cannot work: multiple shooting (short re-anchored segments) is the position-based alternative to try before switching the output.
9. **Reduce the dataset size for now** while debugging (Dutch note: "Data kan ik een stuk minder maken voor nu").
10. **Parked until augmentation works:** joint estimation (needs an orthogonalization mechanism between the augmentation and the baseline parameters; without regularization it is not useful), augmented-state interpretability (too challenging for now; look at baseline-model states instead, which is only possible when parameters are untouched), and the noise-floor acceptance runs.

## Comment-to-slide mapping

### Slide 21: open-loop drift ("Is this an open-loop problem?")

- "Multisine probably fine": the excitation signal is ruled out as the drift cause.
- "Fit the velocity instead. Still has an integrator. Double integrator: need to fit acceleration instead of positions" and "can take the output as velocity, just fit the velocity, last resort. Not convinced can't pull off with just positions, with multiple shooting": see priority item 8.
- "Integration error. Total force: double check the actual input I give matches the actual system" and "seem to start at different position": priority item 3.
- "Try with a reference that is equal to zero, for the controller, then only inject the multisine. Have the feedback force but no multisine feedforward. As simple as possible to detect the issue. KILL the feedback, complete open-loop" and "perfect multisine periodic motion should not have problems with the true system, no disturbances, no need for the feedback controller, just integrator behaviour": priority item 2.
- "Drift off. Didn't have it with parameter recovery. Integrator behaviour is the same. Only thing I add is the additional mass. Wrong treatment or [input error]": the comparison is valid, see the param recovery finding below.
- "Need to visualize a bit better where the issue comes from": priority item 4.
- "What is the hypothesis and what do you expect to see in the data. Can't have rapid cycles": every diagnostic run needs an explicit hypothesis first; runs take too long for blind iteration.

### Slides 9-11, 20: validation loss does not converge

- "Double check train validation" plus Dutch note "Kijken meerdere validatie methodes": priority item 5.
- "Look at training 0.1 seconds so validation should also be good for 0.1 seconds. Already divergence in the 0.1 second in the image": judge on the trained horizon first.
- "Increase nf seconds would better be able to deal with the validation. Need to see the dominant dynamics" and "fine spot in between nf second length runtime and capturing dynamics we want": priority item 6.
- "Learning step too high can blow up neural network" and "baseline same as the system, augmented can be on. If stepsize or learning is too high it shouldn't learn anything, but if it starts to learn something: problematic": priority item 7.

### Slides 24-28: joint estimation (error worse than baseline, slide 26)

- "Was it the data that made only theta trainable": the narrowband 130-180 Hz augmentation data only excites the theta resonance (~157 Hz); the X/Y rigid-body dynamics (~5 Hz) are barely informed, so only theta-related parameters get useful gradients. Broadband 1-200 Hz ('joint' dataset) is the intended fix. (Connects to the slide 29 question.)
- "Without regularization joint estimation not that useful. NEED something that orthogonalizes the augmentation with respect to the baseline parameters or the other way around": explains the slide 26 blowup (-3000%); matches slide 33 "look into adding orthogonality".
- "Look at additional components that resemble the original system, for the physical states. Only possible because I don't touch the parameters. With joint estimation [the roles] can be shared" and "for now just happy to see good behaviour of the augmented model. Look at the states of the baseline model. Augmentation is too challenging to look into now": de-scoped, see priority item 10.
- "Getting the other thing [augmentation] working is the bottleneck": defines the critical path.

### Slide 23: noise floor / when good enough

- "Currently without noise, make it work": priority item 1.
- "For these kind of systems 60-80 dB": replaces the current SNR = 50 choice when noise returns.
- "If I want to compute the minimum: simulate the system without noise and then with noise and compare the two. Just compute the added noise based on the error, signal squared error. Compute wrt the noise that is there": the acceptance floor comes from a with/without-noise comparison of the same simulation, NOT from the error the baseline (no-MSD) model achieves. This directly answers the question on slide 23.
- "Only measurement noise. DON'T ADD IN THE CLOSED-LOOP, SHOULD NOT GO THROUGH THE CLOSED-LOOP": noise goes on y after simulation only, never through the controller. The Python script already adds it post-hoc; this constrains future Simulink data generation.

### Slides 15-17: new data plots

- "Naming convention: reference": the word "reference" means the setpoint r in the MATLAB data plots but the measured output in the Python validation plots (slides 12, 21). Pick one meaning.
- Dutch note "Data kan ik een stuk minder maken voor nu": priority item 9.

### Slides 30-32: real data verification (Telica)

- "Short records. Start increasing segments during training. Error in estimation. Visualize training data sections beyond loss figure": segment-length curriculum for the real-data fitting, plus plots of model behaviour on actual training segments (slide 32's loss curve alone says almost nothing).
- "Use 500 epochs. Do the segment increase. Start to deviate from it immediately. Monitor validation loss: start decreasing, slower divergence": concrete run recipe for the next Telica attempt.

### Slide 33-34: planning

- "Make a priority list": this document.
- "Come to campus to discuss": standing follow-up.

## Param recovery open-loop check (Dutch note: "Kijken of param recovery open loop probleem heeft")

Suspicion was that `lpv_lfr_baseline/scripts/train_param_recovery.py` re-estimates states per segment, so drift would have been invisible there. Verified 07-07-2026:

- **Training**: correct, it re-anchors constantly. Every segment batch takes its initial state from the true state trajectory (`traj['state_traj'][s]`), and segments are 650 samples at 20 kHz (32.5 ms). The model never has to survive longer than that during training.
- **Validation and final eval**: no re-anchoring. `_full_traj_eval` and `_eval_group` simulate the ENTIRE record open-loop from one true initial state (`traj['state_traj'][:1]`), and best-parameter selection was driven by that full-trajectory RMSE.

Conclusion: parameter recovery genuinely did full open-loop simulation with the same K = 0 integrators and did not drift. The drift is specific to the augmentation pipeline. Differences between the two setups (the suspect list):

1. Truth data now contains the hidden absorber (extra mass).
2. Data is closed-loop Simulink with feedback + feedforward + multisine combined into `u_total`.
3. Model block is the `up_sample` discretized `Gantry_State_Block` instead of the RK4 `simulate()`.
4. The current run adds SNR = 50 output noise.

## Ambiguous notes (best-guess readings, to confirm)

- "Was it the data that made only theta trainable": read as a conclusion (narrowband data only informs theta), could also be an open question to investigate.
- "Naming convention: reference": read as the setpoint-vs-measured-output ambiguity; could mean a specific rename.
- "Seem to start at different position": read as an observation about slide 21 (different initializations settle at different offsets); could mean model and data start at different positions.
- "Easiest solution fit velocity should not go up. Can go up but not for the same reason": read as: with velocity fitting the error should not grow unboundedly (no integrator accumulating it); if it still grows, the cause is something other than the integrator.
- "Can't have rapid cycles": read as: hypothesis-driven runs are mandatory because each training run is too slow for trial-and-error iteration.
