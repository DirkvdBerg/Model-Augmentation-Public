# Black-box method (BB-NL, option B): Jan's SUBNET on the closed-loop servo error

Companion to `DATA-DESIGN.md`, `TRAINING-DESIGN.md` and `RESULTS-DESIGN.md`. This file documents
the black-box comparison arm completely: what it models, from which signals, with which model,
how it is trained, selected and scored, what it achieves, what it does not, and where every file
is. Decision log: `scripts/gantry/blackbox-cl-trainable/DECISIONS.md` (BB2-001 to BB2-027).
Run log: `scripts/gantry/blackbox-cl-trainable/RUNS.md`. Open points are marked (?).

## 0. Summary
- Model: Hoekstra's plain SUBNET (deepSI 0.3.29 `SS_encoder_general_hf`, deepSI's own `fit()`),
  random initialisation, no physics, no controller, no linear (BLA) start.
- Map learned: the closed loop under K1, from its exogenous signals (reference r, injected force
  f) to the servo error e = y - r. The prediction of the stage positions is y_hat = r + e_hat.
- Accuracy on the 6 validation records, free run over the full 12 s, per stage axis X1 / X2 / Y:

| Model | X1 [m] | X2 [m] | Y [m] |
|-|-|-|-|
| Black box, longest run (e8, 9576 updates, 1 seed) | 7.9e-6 | 8.1e-6 | 1.4e-5 |
| Black box, 3 seeds x 2 versions (1596 updates), mean | 1.0e-5 | 1.1e-5 | 1.6e-5 |
| Grey box at epoch 0 (physics start, network zero) | 4.0e-6 | 3.6e-6 | 2.8e-5 |
| No model, y_hat = r | 1.9e-5 | 1.9e-5 | 3.1e-5 |
| Noise floor | 7.6e-9 | 8.2e-9 | 4.6e-9 |

- Reading: the black box sits in the same band as the grey box (1e-6 to 1e-5 m). It is better on
  Y and about 2x worse on X. The X gap sits in the records with large reference moves.
- Limitation by construction: the model is the loop, not the plant, so it cannot be put under a
  different controller (no E5 controller-transfer test for this arm).

## 1. Why this formulation (closed-loop map, not a plant model)
- The data are closed-loop: the plant force is u = K1 (r - y) + f, with K1 a fixed high-gain
  controller (crossover about 100 Hz, integral action), designed at Y = 0, the same in every record.
- The grey box is trained by simulating its plant model inside K1 (residual form). A black-box
  plant model trained the same way diverges from the first steps: K1's normalised feedthrough is
  3.7e3 times a random model's step gain, about 100x growth per 4 kHz step. Five designs failed by
  this one mechanism (random start, window curricula from 10 and from 1 step, controller built into
  the model, BLA start); evidence in `scripts/gantry/blackbox-cl-trainable/REPORT.md`, section
  "Why a black-box plant cannot be trained through K1".
- A BLA start is excluded by the user rule and is also not identifiable from these data (14 to
  26 % off the true plant between 20 and 106 Hz; REPORT.md "BLA check").
- Option B (user decision 2026-09-29, BB2-018): learn the closed-loop map directly. It predicts
  the same quantity the grey box is scored on (the loop's output under K1, given r and f), with
  no controller anywhere in the model or the rollout.
- Deviation from `RESULTS-DESIGN.md` (BB-NL "same closed-loop training"): by user decision.

## 2. Signals
### 2.1 Data sets
- Truth T_AF (FP model + hidden absorber MSD + tanh Coulomb friction), K1, 12 s per record at
  20 kHz, start at rest after a 0.5 s hold (`DATA-DESIGN.md`).
- Two versions, same records: `Thesis-writeup/Data/Coulomb-tanh-and-MSD` (encoder noise) and
  `Thesis-writeup/Data/Coulomb-tanh-and-MSD-noise-free`.
- Split from each folder's `MANIFEST.csv` (`block` column): 18 training records, 6 validation
  records. Test records (`test-I`, `test-E`) and the FRF records (`frf`) are never loaded.

| Split | Records |
|-|-|
| Training (18) | TR-S1 to TR-S5 (standstill, Y -0.30 to +0.30, multisine), TR-Y1 to TR-Y3 (Y sweeps, multisine), TR-P1 to TR-P4 (S-curve moves, multisine), TR-L1, TR-L2 (Lissajous, multisine), TR-T1 to TR-T4 (ILC-shape moves, no multisine) |
| Validation (6) | VA-S1 (standstill Y +0.10), VA-Y1 (Y sweep), VA-P1 (S-curve moves), VA-L1 (Lissajous), VA-T1, VA-T2 (ILC-shape moves at Y +0.04 / -0.04, no multisine) |

### 2.2 Signals used, per record
How the generator builds them (`Thesis-writeup/Code/Data/gtd_run_simulation.m`):

$$ y_{\mathrm{meas}} = q + v, \qquad u_{\mathrm{fb}} = K_1 (r - y_{\mathrm{meas}}), \qquad u_{\mathrm{total}} = u_{\mathrm{fb}} + f $$

| Symbol | File field | Channels | Unit | Meaning |
|-|-|-|-|-|
| r | `r_sim` | X1, X2, Y (stage) | m | reference positions |
| f | `f_sim` | X1, X2, Y (stage) | N | injected force (the multisine; zero in the T records) |
| y | `y` | X1, X2, Y (stage) | m | measured positions (true position + encoder noise v in the noisy set) |
| u | `u_total` | X1, X2, Y | N | plant force; NOT used by the black box |

- r and f are exogenous: neither depends on y. This is what makes the closed-loop map a proper
  input-output map.

### 2.3 Resampling to 4 kHz (the pipeline's own loader)
- All signals go through `_record_at_fs_new` in `scripts/gantry/gantry_dynamic/data.py`, so the
  black box and the grey box use exactly the same samples.
- y: in the noisy set, the D-232 anti-alias rule. With D = 5:

$$ y_{4k}[i] = r_{20k}[5i] + \big(\mathrm{filtfilt}_{\mathrm{Butter8,\,1600\,Hz}}(y_{20k} - r_{20k})\big)[5i] $$

  In the noise-free set y is point-sampled, y_{4k}[i] = y_{20k}[5i] (only the noisy mode is in
  `ANTI_ALIAS_MODES`, config.py).
- r and f: point-sampled, r_{4k}[i] = r_{20k}[5i], f_{4k}[i] = f_{20k}[5i].
- Trim: 20 samples (5 ms) dropped at both record ends (D-232). Every record then has
  N = 48000 - 40 = 47960 samples at Ts = 2.5e-4 s.

### 2.4 Model input and output signals (BB2-019, BB2-020)
- Output (the signal the model learns), the servo error with the sign y - r:

$$ e[k] = y[k] - r[k] \in \mathbb{R}^3 $$

- Input, 9 channels:

$$ u_{bb}[k] = \big[\, r[k]^\top,\ f[k]^\top,\ \Delta r[k]^\top \,\big]^\top \in \mathbb{R}^9, \qquad \Delta r[k] = r[k] - r[k-1],\ \ \Delta r[0] = 0 $$

- Prediction of the positions, and why the score is unchanged:

$$ \hat y[k] = r[k] + \hat e[k] \quad\Rightarrow\quad \hat y[k] - y[k] = \hat e[k] - e[k] $$

- Why e and not y (BB2-019): deepSI normalises every output channel by its standard deviation on
  the training records. For y that is the move range (0.05 to 0.15 m), so the loop's own
  behaviour, |y - r| of about 1e-5 m, is about 1e-4 of the normalised range and invisible to the
  loss. Measured: on raw y the model was 30 to 100x worse than y_hat = r (runs b1 to b6).
- Why Δr (BB2-020): the servo error answers the reference's velocity and acceleration. In the
  normalised r these are differences of about 2.5e-4 and 1e-5 per sample. Δr is the same
  exogenous signal (r is its running sum) at a usable scale. HEURISTIC; its measured gain was 3 to
  5 % on X at the stage it was added (e1 to e2), not ablated at the final settings (?).
- r stays an input so the model can see Y, which schedules the inertia (LPV).
- All three are transforms of exogenous signals only. No plant, controller or truth information
  enters.

### 2.5 Precision
- e and Δr are formed in float64 from float64 r and y, then handed to deepSI as float64 arrays
  (`--in64`).
- deepSI fits and applies its normalisation in float64, then casts each batch to float32 itself
  (`System_data_norm.transform`, then `to_hist_future_data` and the encoder's `init_state_multi`).
  The network therefore runs in float32 on O(1) signals. Resolution of e after normalisation is
  about 1e-7 relative (about 1e-12 m). Without `--in64` the raw cast puts a 3e-8 m step on y near
  0.3 m, above the 7e-9 m encoder noise.
- No override of deepSI is needed; a float64 network would need one and is not used.

## 3. Model
### 3.1 Structure (SUBNET, Beintema, Schoukens, Toth; deepSI 0.3.29)
Discrete-time state space with an encoder for the initial state, no feedthrough. All signals are
normalised (Section 4):

$$ x_{k+1} = f_\theta(x_k, u_k), \qquad \hat e_k = h_\theta(x_k), \qquad x_{k_0} = \psi_\theta\big(u_{k_0-n_b},\dots,u_{k_0-1},\ e_{k_0-n_a},\dots,e_{k_0-1}\big) $$

- nx = 16 states; na = nb = 29 past samples (7.25 ms), na_right = nb_right = 0.
- Every network is deepSI's `simple_res_net`, a linear map plus a tanh MLP:

$$ \mathrm{net}(z) = W_{\mathrm{lin}} z + b_{\mathrm{lin}} + W_3 \tanh\!\big(W_2 \tanh(W_1 z + b_1) + b_2\big) + b_3 $$

### 3.2 Sizes and parameter count (final configuration, nu = 9, ny = 3)
| Network | deepSI class | Input | Output | Hidden | Parameters |
|-|-|-|-|-|-|
| Encoder psi | `default_encoder_net` (as `JanEncoder`) | 29 x 9 + 29 x 3 = 348 | 16 | 2 x 16 tanh | 5584 + 6128 = 11712 |
| State f | `default_state_net` (as `JanStateNet`) | 16 + 9 = 25 | 16 | 2 x 8 tanh | 416 + 424 = 840 |
| Output h | `default_output_net` | 16 | 3 | 2 x 8 tanh | 51 + 235 = 286 |
| Total | | | | | 12838 |

- Hidden sizes are Jan's (`scripts/ecc_2025/msd_ndof_deepSI_encoder.py`: f and h 2 x 8, encoder
  2 x 16). nx = 16 is a free choice (HEURISTIC, not the true order); na = 29 is the grey box's
  encoder lag (inherited from Jan's (nx_phys + nx_ann) 2 + 1 rule, disclosed in
  `scripts/gantry/blackbox-cl-trainable/IMPLEMENTATION.md`).
- `JanEncoder` and `JanStateNet` (`scripts/gantry/blackbox-cl-trainable/bbcl/jan_model.py`) are
  subclasses that replace `.view` with `.reshape` in `forward` (identical values; `.view` fails on
  non-contiguous batches). Nothing else differs from deepSI.
- Initialisation: PyTorch default for every Linear layer, deepSI sets the MLP biases to 0; numpy
  and torch seeded with the run seed (1, 2, 3).

## 4. Normalisation
deepSI `auto_fit_norm`, fitted once on the 18 training records (all samples), per channel:

$$ \bar u = \frac{u - u_0}{\sigma_u}, \qquad \bar e = \frac{e - e_0}{\sigma_e}, \qquad u_0, e_0 = \mathrm{mean},\ \ \sigma_u, \sigma_e = \mathrm{std} + 10^{-15} $$

- The same constants are used for the validation free runs; the output is mapped back with
  e_hat = sigma_e e_bar_hat + e_0 before scoring in metres.

## 5. Training
### 5.1 Windows
deepSI `to_hist_future_data` per record, then concatenated over the 18 records. A window starting
at k holds:

$$ \big(u_{k-29..k-1},\ e_{k-29..k-1}\big) \to \text{encoder}, \qquad \big(u_{k..k+399},\ e_{k..k+399}\big) \to \text{rollout and target} $$

- nf = 400 steps = 0.1 s (shared with the grey box, D-220: the loop forgets a force error within
  about 50 ms).
- Stride 25: (47960 - 29 - 400) / 25 + 1 = 1902 windows per record, 34236 in total.

### 5.2 Loss (deepSI `SS_encoder_general_hf.loss`)
Per batch of B windows, normalised units:

$$ L(\theta) = \frac{1}{n_f} \sum_{j=0}^{n_f-1} \frac{1}{3B} \sum_{b=1}^{B} \big\| \bar e_{k_b+j} - h_\theta(x_{b,j}) \big\|_2^2, \qquad x_{b,0} = \psi_\theta(\cdot), \ \ x_{b,j+1} = f_\theta(x_{b,j}, \bar u_{k_b+j}) $$

- Simulation error over the whole window (no teacher forcing), backpropagated through all 400
  steps, encoder trained jointly. No regularisation, no burn-in.

### 5.3 Optimiser and budget
| Setting | Proof runs (3 seeds x 2 versions) | Longest run e8 | Server runner default |
|-|-|-|-|
| Optimiser | Adam (deepSI default; betas 0.9, 0.999; eps 1e-8) | same | same |
| Learning rate | 1e-2 | 3e-3 | 3e-3 |
| Batch | 256 windows | 256 | 256 |
| Updates | 1596 (12 epochs of 133) | 9576 (72 epochs) | up to 53200 (400 epochs), 22 h timeout |
| Validation | every 399 updates | every 798 updates | every 399 updates |
| Wall time (local CPU, 4 threads) | 13 to 28 min | 112 min | |

- Why these values: Section 8 (the search, one change per run, each argued in DECISIONS.md).
- lr 1e-2 is fast but fragile: the training loss became NaN in 3 of 8 runs at that setting (e6,
  noisy seed 3 at update 749, noise-free seed 2 at update 873). deepSI then stops and keeps the
  best validated model, so every reported model is finite, but those runs trained less. lr 3e-3
  ran 9576 updates without a NaN; the server runner uses it.

### 5.4 Model selection
- After every validation interval deepSI free-runs the 6 validation records (Section 6) and
  computes `sim-RMS`: per record the RMS over all samples and the 3 channels in metres, then the
  sample-weighted mean over records. The best value so far is checkpointed (`_best.pth`); at the
  end the best checkpoint is loaded and reported.
- Selection and reporting use the same 6 records (handoff rule; the test records are reserved).

## 6. Prediction and scoring
### 6.1 Free run (deepSI `apply_experiment`)
- Encoder on the first k0 = 29 samples of the record (measured e and u), then the model runs
  alone for the remaining 47931 samples (about 12 s) on the inputs only. No measured y after k0.

### 6.2 Metric (`RESULTS-DESIGN.md`, primary metric)
Per validation record and stage axis a in {X1, X2, Y}, from k0 on:

$$ \mathrm{RMS}_a = \sqrt{\frac{1}{N - k_0} \sum_{k=k_0}^{N-1} \big(\hat y_a[k] - y_a[k]\big)^2 } = \sqrt{\frac{1}{N - k_0} \sum_{k=k_0}^{N-1} \big(\hat e_a[k] - e_a[k]\big)^2 } $$

- Per axis summary: the mean of RMS_a over the 6 records. Normalised version:

$$ \mathrm{NRMS}_{e,a} = \frac{\overline{\mathrm{RMS}}_a(\hat e - e)}{\overline{\mathrm{RMS}}_a(e)} $$

  NRMS_e = 1 is the zero-parameter predictor y_hat = r.
- Computed in `train_bb_clmap.py` (`per_axis`) and written to `result.json`.

### 6.3 References (`scripts/gantry/blackbox-cl-trainable/ref_scores.py`)
| Reference | Noisy, X1 / X2 / Y [m] | How |
|-|-|-|
| y_hat = r | 1.92e-5 / 1.94e-5 / 3.08e-5 | RMS(r - y) on the same samples |
| Grey box, epoch 0 | 3.99e-6 / 3.64e-6 / 2.83e-5 | `build_model` + `closed_loop_table` row "baseline, start" (detuned physics, network output zero, linear-map encoder), the pipeline's own closed-loop free run from k0 = 29 |
| Physics at true parameters (context) | 2.73e-6 / 2.75e-6 / 2.56e-5 | same table, row "baseline, true" |
| Noise floor | 7.6e-9 / 8.2e-9 / 4.6e-9 | RMS(y_noisy - y_noisefree) |
- Noise-free: the same to 4 digits.
- No trained grey box on the full thesis records existed at the time (the stored `thesis_taf_*`
  runs are 2 s smoke tests). When one exists, compare against it (?).
- The grey-box start is one detuning draw (seed 1); the black box has 3 seeds.

## 7. Results
### 7.1 Proof runs (final configuration of Section 5.3, 1596 updates)
| Version | Seed | X1 / X2 / Y [m] | NRMS_e | / grey box epoch 0 | Wall |
|-|-|-|-|-|-|
| noisy | 1 | 9.8e-6 / 1.05e-5 / 1.43e-5 | 0.51 / 0.54 / 0.46 | 2.45 / 2.88 / 0.51 | 28 min |
| noisy | 2 | 1.00e-5 / 1.14e-5 / 1.52e-5 | 0.52 / 0.59 / 0.49 | 2.51 / 3.14 / 0.54 | 26 min |
| noisy | 3 | 1.09e-5 / 1.23e-5 / 1.81e-5 | 0.57 / 0.63 / 0.59 | 2.74 / 3.38 / 0.64 | 13 min (NaN at 749) |
| noise-free | 1 | 9.96e-6 / 1.07e-5 / 1.49e-5 | 0.52 / 0.55 / 0.48 | 2.49 / 2.94 / 0.53 | 24 min |
| noise-free | 2 | 1.08e-5 / 1.20e-5 / 1.61e-5 | 0.56 / 0.62 / 0.52 | 2.71 / 3.28 / 0.57 | 15 min (NaN at 873) |
| noise-free | 3 | 1.01e-5 / 1.08e-5 / 1.46e-5 | 0.53 / 0.56 / 0.47 | 2.54 / 2.96 / 0.52 | 25 min |
- Seed mean: noisy 1.02e-5 / 1.14e-5 / 1.59e-5 m, noise-free 1.03e-5 / 1.12e-5 / 1.52e-5 m.
  Spread between seeds (max / min) 1.09 to 1.26.
- All 36 validation free runs finite.
- Noisy and noise-free agree within a few percent: the noise (about 1e-8 m) is 1000x below the
  model error.

### 7.2 Longest run (e8: lr 3e-3, 9576 updates, noisy, seed 1)
- 7.88e-6 / 8.13e-6 / 1.39e-5 m; NRMS_e 0.41 / 0.42 / 0.45; 1.97 / 2.23 / 0.49 of the grey box.
- Validation sim-RMS every 798 updates [m]: 1.62, 1.37, 1.27, 1.18, 1.25, 1.55, 1.13, 1.07,
  1.08, 1.05, 1.06, 1.06e-5. Falling to about 6400 updates, flat after; best at 7980.
- One seed only; the 3 seeds x 2 versions at this budget are not run yet (?).

### 7.3 Per record (seed mean of the 6 proof runs)
| Record | Kind | X1 / X2 / Y [m] | / y_hat = r | / grey box |
|-|-|-|-|-|
| VA-S1 | standstill + multisine | 4.8e-6 / 4.6e-6 / 9.8e-6 | 0.35 / 0.42 / 0.34 | 0.95 / 0.91 / 0.25 |
| VA-Y1 | Y sweep + multisine | 4.8e-6 / 5.4e-6 / 1.1e-5 | 0.37 / 0.47 / 0.37 | 1.04 / 1.15 / 0.26 |
| VA-L1 | Lissajous + multisine | 7.5e-6 / 8.7e-6 / 1.2e-5 | 0.54 / 0.60 / 0.41 | 1.72 / 1.92 / 0.30 |
| VA-P1 | S-curve moves + multisine | 2.0e-5 / 2.2e-5 / 2.5e-5 | 0.78 / 0.83 / 0.66 | 4.01 / 4.63 / 0.61 |
| VA-T1 | ILC-shape moves | 1.5e-5 / 1.7e-5 / 2.3e-5 | 0.48 / 0.49 / 0.61 | 4.64 / 9.24 / 5.54 |
| VA-T2 | ILC-shape moves | 1.0e-5 / 1.1e-5 / 1.3e-5 | 0.56 / 0.55 / 0.58 | 5.56 / 9.70 / 4.95 |
- e8 (longer training) on the move records: T1 2.6 / 5.7x, T2 3.1 / 4.8x, P1 4.1 / 4.6x on X;
  S1 and Y1 0.5 to 0.9x.

### 7.4 Interpretation
- Where the servo error is the loop's response to the injected force (standstill and sweep
  records), the black box matches or beats the grey box on X and beats it 3 to 4x on Y. The
  epoch-0 grey box has no absorber and no friction model, so its Y error there is 4e-5 m.
- Where the servo error is the loop's response to large reference moves (S-curves, ILC shapes),
  the grey box is 2 to 10x better. Its physics (masses, K1 in the loop) predicts that tracking
  error by construction; the black box has to learn it from the moves in 18 records.
- Learning order, seen in every run: the force response in the first 3 epochs, the move response
  slowly after (T records from 1.0x of y_hat = r at the start to 0.3x in e8).
- Budget: 6x more updates (e8) took X from 2.5 to 2.0x the grey box and then flattened. Budget
  alone is unlikely to close the move-record gap (?).

### 7.5 Against the handoff's pre-registered bar
- Bar: per axis seed-mean <= 1.25x the epoch-0 grey box (HEURISTIC, from blackbox-closed-loop
  BB-010), all free runs finite, NRMS_e < 1 on every axis and seed.
- Outcome: finite PASS, NRMS_e < 1 PASS (18/18), per-axis bar PASS on Y, FAIL on X1 and X2.
- The Y bar says little: the epoch-0 grey box is no better than y_hat = r on Y (NRMS_e 0.92).
- Judged as "same order of magnitude as the grey box" (user, 2026-10-04: the grey box is between
  1e-6 and 1e-5 m), the black box qualifies.

## 8. How the configuration was found (search on noisy seed 1, one change per run)
| Run | Change | X1 / X2 / Y [m] | NRMS_e | Cost | Decision |
|-|-|-|-|-|-|
| b1 | raw y, float32 (start) | 6e-4 / 5e-4 / 1.3e-3 | 31 / 26 / 42 | 29 min | BB2-018 |
| e1 | target e = y - r, float64 records | 1.40e-5 / 1.53e-5 / 1.89e-5 | 0.73 / 0.79 / 0.61 | 30 min | BB2-019 |
| e2 | + input Δr | 1.35e-5 / 1.45e-5 / 1.89e-5 | 0.71 / 0.75 / 0.62 | 30 min | BB2-020 |
| e3 | + lr 3e-3, validation every 399 | 1.10e-5 / 1.16e-5 / 1.65e-5 | 0.57 / 0.60 / 0.54 | 29 min | BB2-021 |
| e4 | + batch 256, stride 25 | 1.10e-5 / 1.19e-5 / 1.70e-5 | 0.58 / 0.62 / 0.55 | 28 min | BB2-022 |
| e5 | + lr 1e-2 (final, proof) | 9.8e-6 / 1.05e-5 / 1.43e-5 | 0.51 / 0.54 / 0.46 | 28 min | BB2-023 |
| e6 | + f, h 2 x 32, encoder 2 x 64 | 1.26e-5 / 1.31e-5 / 1.66e-5 | 0.66 / 0.67 / 0.54 | NaN at 548 | BB2-024 |
| e7 | + na_right 1 (grey box's window) | not runnable in deepSI 0.3.29 | | 1 min | BB2-025 |
| e8 | final at lr 3e-3, 9576 updates | 7.9e-6 / 8.1e-6 / 1.39e-5 | 0.41 / 0.42 / 0.45 | 112 min | BB2-027 |
- e4 showed that on this CPU an update's cost is set by the 400 sequential steps, not the batch
  (batch 256 at 1.7 updates/s against batch 64 at 2.0), and that per update the progress was the
  same: the step size limited, hence e5.
- e7: deepSI's free run (`init_state` -> `init_state_multi`) builds the encoder window without the
  right extension (encoder expects 360 inputs, gets 348); using na_right would need an override.

## 9. Rules kept, and known differences from the grey box
### 9.1 Rules (user)
- Jan's construction: deepSI model class and `fit()`; changes only to the signals it is trained on
  and to its settings, each argued in DECISIONS.md.
- No controller in the model or the rollout; no BLA, subspace, frequency or pole start; no burn-in;
  no truth information; no monkey patching (the two subclasses in `jan_model.py` are proper
  classes, nothing in deepSI or the pipeline is overridden at runtime).
- Test records never loaded; selection on the 6 validation records only.

### 9.2 Differences from the grey box's scoring
| Item | Black box | Grey box | Effect |
|-|-|-|-|
| What is modelled | the loop (r, f) -> e | the plant, closed with K1 in residual form | same predicted quantity; no E5 for the black box |
| Encoder window | samples k0-29 to k0-1 | k0-29 to k0 (na_right 1) | not available in deepSI 0.3.29 |
| Loop state at k0 | inferred from 29 samples | true controller state inside u_data | decays within about 50 ms of a 12 s record |
| Force input timing | f point-sampled at 4 kHz | u_total block-mean over each 4 kHz interval (D-087) | half-sample timing difference; block-averaging f is allowed and not yet tried (?) |
| Residual form | none | exact only if the 4 kHz Tustin K1 equals the 20 kHz K1 inside u_data | any mismatch leaks a little measured y into the grey box's free run |
| Seeds | 3 | 1 detuning draw | bar is a single draw |
- `result.json`'s `criteria` field is the older BB2-018 mean-model test (all PASS); the section
  7.5 criterion is evaluated in REPORT.md, not in that field.

## 10. How to reproduce
Environment: conda env `GraduationProject` (Python 3, torch 2.5.1, deepSI 0.3.29), CPU.
- References (both versions, about 26 min):
  `python -u scripts/gantry/blackbox-cl-trainable/ref_scores.py`
- One proof run (about 25 min):
  `python -u scripts/gantry/blackbox-cl-trainable/train_bb_clmap.py --version noisy --seed 1 --epochs 12 --target e --in64 --dr --lr 1e-2 --batch 256 --stride 25 --its-per-val 399 --out <dir>`
- The e8 setting: same with `--lr 3e-3 --epochs 72 --its-per-val 798`.
- Detached on Windows (survives the Claude session): `tools/run_one.cmd <same arguments>`,
  started with `Start-Process cmd.exe -ArgumentList '/c ""<path>\tools\run_one.cmd" <args> > "<log>" 2>&1"' -WindowStyle Hidden`.
- Server: `BB_VERSION=noisy BB_SEED=1 sbatch scripts/gantry/blackbox-cl-trainable/runners/run_bb_clmap.sh`
  (CPU, lr 3e-3, up to 53200 updates, deepSI timeout 22 h so `result.json` is always written;
  optional `BB_LR`, `BB_EPOCHS`, `BB_TIMEOUT`, `BB_EXTRA`). Not submitted yet.
- Every default of `train_bb_clmap.py` reproduces the original raw-y runs b1 to b6; the flags
  `--target e --in64 --dr` and the settings above are needed for this method.
- Load a trained model: deepSI `checkpoint_load_system('_best', directory=<out>/localappdata/deepSI/checkpoints)`
  on a model built with the same arguments, or rerun with the same seed.

## 11. File locations
All paths relative to the repository root
(`C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Baseline FP model\Baseline-LPV-Augmentation`).

### 11.1 Code
| File | Role |
|-|-|
| `scripts/gantry/blackbox-cl-trainable/train_bb_clmap.py` | the method: data loading, signal transforms, model, deepSI `fit()`, scoring, `result.json` |
| `scripts/gantry/blackbox-cl-trainable/bbcl/jan_model.py` | `JanEncoder`, `JanStateNet` (Jan's nets, `.reshape` for `.view`) |
| `scripts/gantry/blackbox-cl-trainable/bbcl/setup.py` | the grey box's CFG read from the entry file (data mode, sample rate, trim) |
| `scripts/gantry/blackbox-cl-trainable/ref_scores.py` | references: y_hat = r, grey box epoch 0, noise floor |
| `scripts/gantry/blackbox-cl-trainable/runners/run_bb_clmap.sh` | SLURM runner (server) |
| `scripts/gantry/blackbox-cl-trainable/tools/run_one.cmd` | detached local launcher, one run |
| `scripts/gantry/blackbox-cl-trainable/tools/run_proof.cmd` | detached local launcher, the 5 proof runs |
| `scripts/gantry/blackbox-cl-trainable/tools/run_ref.cmd` | detached local launcher, the references |
| `scripts/gantry/gantry_dynamic/data.py` | pipeline loader used for both arms (`record_files`, `_record_at_fs_new`, `traj_dir`) |
| `scripts/gantry/gantry_dynamic/closed_loop_report.py` | grey-box closed-loop scorer used for the reference |
| `Thesis-writeup/Code/Data/gtd_run_simulation.m` | how r, f, y, u_total are generated |
| deepSI 0.3.29: `<env>/Lib/site-packages/deepSI/fit_systems/encoders.py`, `fit_system.py`, `utils/torch_nets.py` | model class, loss, fit, networks |

### 11.2 Data
| Path | Content |
|-|-|
| `Thesis-writeup/Data/Coulomb-tanh-and-MSD/` | noisy records, `Training/`, `Validation/`, `MANIFEST.csv` |
| `Thesis-writeup/Data/Coulomb-tanh-and-MSD-noise-free/` | noise-free twins |

### 11.3 Results (under `scripts/gantry/blackbox-cl-trainable/outputs/`)
| Path | Content |
|-|-|
| `ref/references.json`, `ref/ref.log` | the references of Section 6.3 |
| `e/e1_noisy_s1/` to `e/e8_noisy_s1/` | search runs e1 to e8 (`result.json`; logs `e/e1.log` to `e/e8.log`) |
| `e/e5_noisy_s1/` | proof row noisy seed 1 |
| `p/noisy_s2/`, `p/noisy_s3/`, `p/noisefree_s1/`, `p/noisefree_s2/`, `p/noisefree_s3/` | proof rows 2 to 6; log `p/proof.log` |
| `e/e8_noisy_s1/localappdata/deepSI/checkpoints/SS_encoder_general_hf_UqizCw_best.pth` | best trained model so far (e8) |
| `<run>/localappdata/deepSI/checkpoints/*_best.pth`, `*_last.pth` | best-validated and last model of every run |
| `b/` | the earlier raw-y runs b1 to b6 |
- `result.json` fields: `per_axis_final` (mean over records, m), `per_record` (6 x 3, m),
  `trivial_rms` and `trivial_per_record` (y_hat = r), `nrms_e`, `nrms_e_per_record`, `loss_val`
  (validation sim-RMS per validation), `loss_train`, `epoch_id`, `args`, `wall_s`.

### 11.4 Documentation
| File | Content |
|-|-|
| `scripts/gantry/blackbox-cl-trainable/REPORT.md` | results report (top section: this method), earlier negative results |
| `scripts/gantry/blackbox-cl-trainable/DECISIONS.md` | BB2-001 to BB2-027 |
| `scripts/gantry/blackbox-cl-trainable/RUNS.md` | every run: hypothesis before, outcome after, cost |
| `scripts/gantry/blackbox-cl-trainable/CRITIC.md` | two cold reviews and how each finding was handled |
| `scripts/gantry/blackbox-cl-trainable/IMPLEMENTATION.md` | implementation notes, flags |
| `scripts/gantry/blackbox-cl-trainable/LITERATURE.md` | literature on training through a controller |

## 12. Open points
- 3 seeds x 2 versions at the e8 budget (lr 3e-3, about 9.6k updates), so the quoted number has
  the same backing as the proof (?).
- Block-averaged f as input, to remove the input timing difference with the grey box (?).
- Comparison against a trained grey box once it exists on the full records (?).
- Δr ablation at the final settings (?).
