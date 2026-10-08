# Meeting 1 October 2026: slide content

Current method only. Lengths and errors in metres, scientific notation.

## 1. Coulomb friction in the simulated truth
- Friction acts on each rail (X1, X2, Y) of the truth only; the baseline model has no friction
- Force per rail: F_i = cc_i tanh(g v_i), subtracted from the rail force
- cc_i from Garcia: 16.8 / 18.35 / 11.6 N (X1 / X2 / Y)
- tanh reaches the full Coulomb level above about 3/g
- g = 1000 s/m, chosen so that:
  - at the rail velocities under the multisine (7e-3 to 8e-3 m/s rms) the friction is at its Coulomb level
  - at standstill the stage settles without a limit cycle (6e-11 / 1e-10 / 1.7e-9 m)
- Below 1e-3 m/s the law is a steep damper (slope cc g, about 1.7e4 N s/m)
- Figure: `figures/F3_friction_tanh.png`

## 2. Measurement noise
- Source: the measured servo error of the Telica machine at standstill, 145 logs, rms X1 9.8e-9, X2 10.4e-9, Y 6.3e-9 m
- Injected on the encoder: the controller sees y = q + v
- One noise generator per axis
- The loop filters the injected noise (recorded error e = -S v), so the injected spectrum is the measured one divided by |S|^2; the recorded error then has the measured spectrum
- Below 50 Hz:
  - the controller's integrator makes |S| very small, so dividing by |S|^2 would demand a large slow noise
  - the controller would follow it and physically move the stage
  - so below 50 Hz the measured spectrum is injected as is; the simulated error there is 0.7 to 0.9 of the measured, under 1 % of the variance
- Result: above 50 Hz simulated / measured 0.97 to 1.02 per band; total rms 9.82e-9 / 10.34e-9 / 6.33e-9 m against measured 9.85e-9 / 10.40e-9 / 6.34e-9 m
- Training at 4 kHz from 20 kHz data: y - r is low-pass filtered (Butterworth, order 8, 1.6 kHz, forward and backward) before taking every 5th sample, so noise above 2 kHz does not fold into the band; the same filtered positions feed training, validation and the velocities for the orthogonal projection
- Figure: `figures/F1_noise_shape.png`

## 3. How the multisine band was chosen
- Measured the closed-loop FRF of the truth with K1 at the five training positions Y = -3.0e-1 to 3.0e-1 m
- Subtracted the FRF of the baseline model: the difference is what the model misses
- Kept only differences larger than the measurement's own spread (2.45 sigma)
- Lower edge 106 Hz: the controller crossover; below it the motion references already excite the differences
- Upper edge 297 Hz: the band above 106 Hz that contains 90 % of the remaining difference
- The band contains the anti-resonance (150 Hz), the absorber pole (212 Hz) and the closed-loop resonance (264 Hz)
- Largest difference: 2.3e-7 m/N at 250 Hz on Y
- Multisine: 383 lines every 0.5 Hz, random phases, lowest crest factor of 30 draws, 240 N / 87 N m / 180 N (sym / anti / Y)
- Figures: `figures/F2_frf_band.png`; cumulative share `coulomb-tanh-gain/outputs/band_tanh_g1000/figures/fig12_band_weight.png`

## 4. The issue: above 230 Hz the absorber is not learned
- Y error removed by the trained model (run 42):
  - 106 to 140 Hz: 90 %
  - 140 to 230 Hz: 89 %
  - 230 to 297 Hz: 22 %
- Learned correction against the true one: gain 1.0 at 150 Hz, 0.60 at 212 Hz, 0.26 at 264 Hz
- The learned correction is a broad over-damped bump, not the sharp absorber resonance
- Same in runs 42, 44 and 48 (with and without orthogonality, 2 and 4 added states, two seeds)
- The added states contain no resonance: only fast real poles, no pole pair near 212 Hz
- What we tested:
  - the model can hold the resonance: a model with the correct 212 Hz pole scores 7x lower training loss
  - the training loss prefers the resonance: the truth scores 3.8x better
  - the learned static correction takes the absorber's place first: a correct absorber added to the trained model gains 15 %, added to the model without its learned static part 94 %
  - with the static part switched off, the added states learn the same static map with one sample delay, and growing a resonance from there raises the loss
- Conclusion: an optimisation problem; whatever learns first fits the static part of the absorber, and from there the resonance is uphill
- Validation free-run rms: start 1.5e-5 to 1.7e-5 m, trained 1.02e-5 to 1.03e-5 m (without orthogonality, done); with orthogonality 0 and 2 added states both near 1.15e-5 m (running)
- Questions:
  - which way to get a resonance into the added states (parameterisation with a linear part, higher learning rate, other)?
  - is a model without a learned resonance acceptable for the thesis if the error above 230 Hz is stated?
