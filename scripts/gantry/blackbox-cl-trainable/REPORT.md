# blackbox-cl-trainable: report

## Option B on the servo error, overnight 2026-10-03/04: FAIL (X 2.6 to 3.1x the grey box)
- Answer: Jan's SUBNET on the closed-loop map now learns the loop (NRMS_e 0.47 to 0.63 on every
  axis, seed and version, from 30 to 100 before), but on X1 / X2 it stays 2.6 / 3.1x the epoch-0
  grey box: **FAIL** against section 10 of the handoff. Y passes, but that bar is uninformative:
  the epoch-0 grey box is no better than y_hat = r on Y (NRMS_e 0.92).
- Verdict branch (settled by e8): NRMS_e < 1 and improving with budget, but flattening. 6x the
  updates (e8) closes X from 2.5 to 2.0x the grey box, and validation stopped falling over the
  last 3000 of 9576 updates. Budget alone is unlikely to reach 1.25x; the gap is the loop's
  response to large reference moves (P1, T1, T2 still 2.6 to 5.7x).
- Deviations from RESULTS-DESIGN.md:70 ("same closed-loop training"): option B by user decision;
  the model is trained on the closed-loop map, not inside the K1 loop.

### References (6 validation records, per axis X1 / X2 / Y, metres, from k0 = 29)
| Reference | Noisy | Noise-free | Source |
|-|-|-|-|
| y_hat = r (NRMS_e = 1) | 1.92e-5 / 1.94e-5 / 3.08e-5 | same to 4 digits | `ref_scores.py` |
| Grey box epoch 0 ('baseline, start') | 3.99e-6 / 3.64e-6 / 2.83e-5 | same to 4 digits | pipeline `closed_loop_table` |
| Physics at true combinations (context) | 2.73e-6 / 2.75e-6 / 2.56e-5 | same | same table, row 4 |
| Noise floor RMS(y_noisy - y_noisefree) | 7.6e-9 / 8.2e-9 / 4.6e-9 | | `ref_scores.py` |
- **Provisional**: no trained grey box on the thesis data exists (the stored `thesis_taf_*` runs are
  2 s smoke tests, TRUNCATE_S 2.0, 0.15 epoch), so the bar is the epoch-0 grey box (detuned
  physics, network zero, linear-map encoder). A trained grey box would be better: a lower bar.
  It is one detuning draw (THESIS_SEED 1); the black box gets 3 seeds.
- PASS bar (1.25x epoch 0, HEURISTIC from blackbox-closed-loop BB-010): 5.0e-6 / 4.6e-6 / 3.5e-5.
  On Y, y_hat = r already passes it; X is the real test (grey box 5x better than y_hat = r there).

### Final configuration (DECISIONS.md BB2-019 to BB2-026)
- Jan's deepSI `SS_encoder_general_hf` and `fit()`; nx 16, f and h 2 x 8, encoder 2 x 16, na = nb
  = 29, na_right 0, nf 400 (0.1 s), sim-RMS selection, auto_fit_norm.
- Output signal e = y - r; prediction y_hat = r + e_hat; inputs r, f and dr = r(k) - r(k-1),
  formed in float64 before deepSI's normalisation (deepSI then casts to float32 itself).
- Adam lr 1e-2, batch 256, stride 25, 1596 updates, validation every 399 updates.
- Command: `train_bb_clmap.py --target e --in64 --dr --lr 1e-2 --batch 256 --stride 25
  --its-per-val 399 --epochs 12`.

### Proof: 3 seeds x 2 versions against section 10
| Version | Seed | RMS X1 / X2 / Y [m] | NRMS_e | Grey-box ratio | Floor ratio | Verdict | Wall |
|-|-|-|-|-|-|-|-|
| noisy | 1 | 9.8e-6 / 1.05e-5 / 1.43e-5 | 0.51 / 0.54 / 0.46 | 2.45 / 2.88 / 0.51 | 1.3e3 / 1.3e3 / 3.1e3 | FAIL (X) | 28 min |
| noisy | 2 | 1.00e-5 / 1.14e-5 / 1.52e-5 | 0.52 / 0.59 / 0.49 | 2.51 / 3.14 / 0.54 | 1.3e3 / 1.4e3 / 3.3e3 | FAIL (X) | 26 min |
| noisy | 3 | 1.09e-5 / 1.23e-5 / 1.81e-5 | 0.57 / 0.63 / 0.59 | 2.74 / 3.38 / 0.64 | 1.4e3 / 1.5e3 / 4.0e3 | FAIL (X), NaN at update 749 | 13 min |
| noise-free | 1 | 9.96e-6 / 1.07e-5 / 1.49e-5 | 0.52 / 0.55 / 0.48 | 2.49 / 2.94 / 0.53 | 1.3e3 / 1.3e3 / 3.3e3 | FAIL (X) | 24 min |
| noise-free | 2 | 1.08e-5 / 1.20e-5 / 1.61e-5 | 0.56 / 0.62 / 0.52 | 2.71 / 3.28 / 0.57 | 1.4e3 / 1.5e3 / 3.5e3 | FAIL (X), NaN at update 873 | 15 min |
| noise-free | 3 | 1.01e-5 / 1.08e-5 / 1.46e-5 | 0.53 / 0.56 / 0.47 | 2.54 / 2.96 / 0.52 | 1.3e3 / 1.3e3 / 3.2e3 | FAIL (X) | 25 min |
- Seed mean, noisy: 1.02e-5 / 1.14e-5 / 1.59e-5 m, ratio 2.57 / 3.13 / 0.56; noise-free:
  1.03e-5 / 1.12e-5 / 1.52e-5 m, ratio 2.58 / 3.06 / 0.54. Spread (max / min) 1.09 to 1.26.
- Criteria: finite PASS (6/6); NRMS_e < 1 PASS (18/18); per axis <= 1.25x PASS on Y, FAIL on X1, X2.
- Noisy row 1 is search run e5 (same configuration and 1596-update model); rows 2 to 6 are
  `tools/run_proof.cmd`.
- lr 1e-2 is fragile: NaN training loss in 2 of 5 proof runs (and e6). deepSI stops and keeps the
  best validated model, so every result is finite, but those two runs had only 399 / 798 updates.
- Noisy and noise-free now differ per seed (float64 records keep the noise; 1.6 / 1.9 / 4.3 % at seed 1,
  where the noisy run is e5 with its wall-clock stop),
  but the floor (~1e-8 m) is 1000x below the model error, so the noise is irrelevant at this level.

### Where the X gap sits (seed mean of the 6 runs, per validation record)
| Record | Kind | Black box X1 / X2 / Y [m] | / y_hat = r | / grey box |
|-|-|-|-|-|
| VA-S1 | standstill + multisine | 4.8e-6 / 4.6e-6 / 9.8e-6 | 0.35 / 0.42 / 0.34 | 0.95 / 0.91 / 0.25 |
| VA-Y1 | Y sweep + multisine | 4.8e-6 / 5.4e-6 / 1.1e-5 | 0.37 / 0.47 / 0.37 | 1.04 / 1.15 / 0.26 |
| VA-L1 | Lissajous + multisine | 7.5e-6 / 8.7e-6 / 1.2e-5 | 0.54 / 0.60 / 0.41 | 1.72 / 1.92 / 0.30 |
| VA-P1 | S-curve moves + multisine | 2.0e-5 / 2.2e-5 / 2.5e-5 | 0.78 / 0.83 / 0.66 | 4.01 / 4.63 / 0.61 |
| VA-T1 | ILC-shape moves, no multisine | 1.5e-5 / 1.7e-5 / 2.3e-5 | 0.48 / 0.49 / 0.61 | 4.64 / 9.24 / 5.54 |
| VA-T2 | ILC-shape moves, no multisine | 1.0e-5 / 1.1e-5 / 1.3e-5 | 0.56 / 0.55 / 0.58 | 5.56 / 9.70 / 4.95 |
- Mechanism: the black box already equals the grey box where the error is the response to the
  injected force (standstill, small motion), and beats it 3 to 4x on Y on the four multisine
  records (the epoch-0 grey box has no absorber or friction model; its Y error there is 4e-5 m).
  It loses where the error is the loop's response to large
  reference moves (S-curves, ILC shapes; on T1, T2 also on Y, 5x): the physics carries mass and K1 and predicts that
  tracking error; the black box has to learn it from the moves in 18 records.
- Learning order seen in every run: the force response first (epochs 1 to 3), the move response
  slowly after (T records from 1.0x of y_hat = r in e1 to ~0.5x in the proof).
- Not done: a per-band error PSD against the floor PSD (predictions not stored). The floor is
  1000x below every error in total RMS; that does not prove no band is noise-limited. The
  noisy set is anti-alias decimated and the noise-free set point-sampled (config
  ANTI_ALIAS_MODES), so the 'floor' also holds that decimation difference; both ~1e-8 m.

### Budget (e8, lr 3e-3, 9576 updates, noisy seed 1)
- Result: 7.9e-6 / 8.1e-6 / 1.39e-5 m, NRMS_e 0.41 / 0.42 / 0.45, grey-box ratio 1.97 / 2.23 /
  0.49 (proof seed mean 2.57 / 3.13 / 0.56 at 1596 updates). No NaN at lr 3e-3. 112 min.
- Validation sim-RMS [m] every 798 updates: 1.62, 1.37, 1.27, 1.18, 1.25, 1.55, 1.13, 1.07, 1.08,
  1.05, 1.06, 1.06e-5: falling to ~6400 updates, flat after (best at 7980). Training loss 0.79 to
  0.60 (normalised), still falling slowly.
- Per record (X1 / X2 vs grey box): S1 0.77 / 0.51, Y1 0.86 / 0.81, L1 1.25 / 1.20, P1 4.06 / 4.59,
  T1 2.63 / 5.68, T2 3.14 / 4.81. Budget improves the move records (T1 X2 9.2 -> 5.7x) but
  leaves P1 unchanged.
- Server budget: `runners/run_bb_clmap.sh` (lr 3e-3, up to 53k updates, 22 h deepSI timeout).
  Expectation from this curve: a further gain of order 10 %, not the 1.6x X still needs (?).

### Every configuration tried (noisy seed 1 unless stated; RUNS.md has the curves)
| Run | Change (decision) | RMS X1 / X2 / Y [m] | NRMS_e | Cost |
|-|-|-|-|-|
| b1 | raw y, float32 (BB2-018) | 6e-4 / 5e-4 / 1.3e-3 | 31 / 26 / 42 | 29 min |
| e1 | target e, float64 records (BB2-019) | 1.40e-5 / 1.53e-5 / 1.89e-5 | 0.73 / 0.79 / 0.61 | 30 min |
| e2 | + dr input (BB2-020) | 1.35e-5 / 1.45e-5 / 1.89e-5 | 0.71 / 0.75 / 0.62 | 30 min |
| e3 | + lr 3e-3, val every 399 (BB2-021) | 1.10e-5 / 1.16e-5 / 1.65e-5 | 0.57 / 0.60 / 0.54 | 29 min |
| e4 | + batch 256, stride 25, 1500 s budget (BB2-022) | 1.10e-5 / 1.19e-5 / 1.70e-5 | 0.58 / 0.62 / 0.55 | 28 min |
| e5 | + lr 1e-2 (BB2-023), final | 9.8e-6 / 1.05e-5 / 1.43e-5 | 0.51 / 0.54 / 0.46 | 28 min |
| e6 | + f, h 32 wide, encoder 64 (BB2-024) | 1.26e-5 / 1.31e-5 / 1.66e-5 | 0.66 / 0.67 / 0.54 | NaN at 548, 11 min |
| e7 | + na_right 1 (BB2-025) | not runnable in stock deepSI | | 1 min |
| proof | final, 5 more seed / version runs (BB2-026) | table above | | 2.2 h |
| e8 | final at lr 3e-3, 9576 updates (BB2-027) | 7.9e-6 / 8.1e-6 / 1.39e-5 | 0.41 / 0.42 / 0.45 | 112 min |
| ref | references (`ref_scores.py`) | table above | | 26 min |
- b1 NRMS_e computed from its per-axis RMS and y_hat = r.

### Known asymmetries and limits
- na_right 0 here, 1 in the grey box: deepSI 0.3.29's free run cannot take na_right (e7).
- The grey box's residual-form free run starts with the true controller state at k0 (inside
  u_data); the black box infers the loop state from 29 past samples. Effect decays within ~50 ms
  of a 12 s record (D-220 loop memory).
- Inputs: the black box gets point-sampled r_sim, f_sim; the grey box the block-mean u_total
  (D-087), a half-sample timing difference on the force. Block-averaging f is allowed and open.
- The residual form is exact only if the 4 kHz Tustin controller equals the 20 kHz one inside
  u_data; any mismatch leaks a little measured y into the grey box's free run.
- `result.json` `criteria` is still the BB2-018 mean-model test (all PASS); section 10 is
  evaluated here, not in the file.
- dr (and the target e = y - r) are input / output transforms of exogenous r; dr was not
  ablated at the final settings (e1 to e2 gain 3 to 5 % on X).
- Selection and reporting on the same 6 validation records (handoff rule); the gap to the bar is
  2.5 to 3.4x, far beyond any selection optimism.
- Server runner `runners/run_bb_clmap.sh` updated (CPU, final configuration, lr 3e-3 default,
  deepSI timeout 22 h so result.json is always written); not submitted.

## BLA check (user 2026-09-29): can a proper BLA be estimated from the training data?
- Answer: **no**. Only band-limited: good in 106 to 297 Hz, 14 to 26 % off the real system in
  20 to 106 Hz, where K1's crossovers (66 to 108 Hz) sit. BLA fallback arm dropped (BB2-005).
- Data: 18 training records, both versions, 20 kHz; K1 known; truth used only to score.
- Target: plant BLA, stage force -> stage position, 3 x 3; error e = sigma_max(G_hat - G_true) /
  sigma_max(G_true) per line; criteria pre-registered in BB2-003 and BB2-004.

| Band | Estimator | Noise-free median e | Noisy median e | Criterion |
|-|-|-|-|-|
| 106 to 297 Hz, per standstill Y | LPM, multisine as instrument | 0.073 to 0.083 (p90 <= 0.135) | same | PASS |
| 106 to 297 Hz, pooled 14 records | same | 0.059 (p90 0.105) | same | PASS |
| 20 to 106 Hz, pooled 18 records | LPM full record, reference as instrument | 0.18 | 0.18 | FAIL (<= 0.10) |
| 1 to 10 Hz | same | 0.63 | 0.63 | reported |
| 20 to 106 Hz, control (friction-free LTI, same r and f) | same | 9e-5 | n/a | PASS |

- Noise does not matter here: encoder noise about 10 nm rms against a 13 to 28 um response;
  noisy and noise-free BLAs differ by about 1e-5 (B_tr) and 3e-4 (low band).
- Mechanism: the control run shows the references excite 20 to 106 Hz well enough and the
  estimator is right, so the low-band gap is the system itself (tanh Coulomb friction driven by
  the moves, plus the Y motion), whose BLA is not the linear plant there.
- Direct estimate (y against u) fails when pooled (median e 1.84, both versions): leakage of
  the non-periodic moves, not noise bias.
- Prediction check: B_tr PASS was predicted; "low band not identifiable" was wrong (it is
  excited, kappa median 0.26); the failure is accuracy, not excitation.
- Not tested: whether K1 would stabilise a start built from this BLA (?).
- Files: `bla/bla_check.py` (bla1), `bla/bla_lowband.py` (bla2), `bla/outputs/*.json`.

## Why a black-box plant cannot be trained through K1
Answer: a black-box plant model (u -> y) trained inside the K1 loop diverges from the first
steps and never produces a usable loss; every remedy that keeps the model a plain black box
failed. The black-box arm therefore learns the closed-loop map instead (option B).

### Mechanism
- Closed-loop training simulates the MODEL under the real controller: u = u_data +
  K1(y_data - y_model), grey box's residual form (`closed_loop.py`).
- Open-loop training (Jan's setting, measured u -> y) is possible on 0.1 s windows with an encoder
  start, but its objective differs from the grey box's, and a 12 s open-loop free run drifts on the
  free integrators X and Y. The previous session trained four open-loop checkpoints (old data,
  800 Hz): all four were unstable once closed with the controller (blackbox-closed-loop REPORT
  G2), so "open loop first, then fine-tune under K1" failed there; not rechecked on the thesis
  data (?) (CRITIC finding 5).
- K1 is high gain (crossover about 100 Hz, integral action); in the model's normalised
  coordinates its direct feedthrough is 3.7e3 (B1, thesis data, 4 kHz).
- K1 stabilises only plants close to the gantry. A random black box is not, so the loop is
  unstable at the start, and any error in the model's one-step response is multiplied by
  K1 at every step: measured growth of order 100x per 4 kHz step (loss 0.10 at 1 step, 299 at
  2, 3.7e15 at 5, 7.3e48 at 10).
- The loss and its gradients are useless long before the 400-step window (0.1 s) is reached,
  so gradient descent has nothing to follow.
- Ribeiro, Tiels, Aguirre, Schoukens (Automatica 2020) describe the same class: a system held
  near instability by a linear controller gives a simulation-error cost that is badly
  conditioned and full of local minima.

### Why the grey box is not affected
- Its physics already is a plant that K1 stabilises, and its network starts at zero output, so
  training starts inside the stable region and moves in small steps from there.
- A black box has no such anchor: its start carries no plant information by design (user rule).

### Evidence (all local, thesis data; runs in `RUNS.md`, decisions in `DECISIONS.md`)
| Attempt | What it tests | Result | Ref |
|-|-|-|-|
| Jan's SUBNET, random start, under K1 | Does the plain model start stable? | Free run NaN on 6/6 records, 3/3 seeds | pb3 B2 |
| BLA start | Can a data-based start fix it? | No proper BLA: 14 to 26 % off near crossover, 63 % below 10 Hz | bla1, bla2 |
| Controller built into a stable model (ICI, Boroujeni 2025) + LRU | Stable by construction? | Stable with the leaky Kt (0.997), unstable with K1: slow real pole 1.00002 to 1.00007 from K1's integrator | pb3 B3, pbm |
| Window curriculum from 10 steps (Decuyper 2020) | Do short windows keep training finite? | Loss 7.3e48 in stage 1 | d1 |
| Window curriculum from 1 step | Does one-step training teach a gain K1 can hold? | Loss 0.10, 299, 3.7e15 at 1, 2, 5 steps; free runs NaN throughout | d2b |
| Old data (previous session) | Fragility around a stable start | 1e-5 weight changes destabilise 76 to 89 %; one Adam step at 1e-3 breaks the loop | blackbox-closed-loop BB-013 |

### Literature position (`LITERATURE.md`)
- No published work trains a SUBNET-type black box through a fixed high-gain controller
  (provisional: citation enumeration of two lineages, arXiv abstract searches).
- The only published fix builds the controller into the model structure (Boroujeni et al.,
  CDC 2025); it requires a stable controller, which K1 with its integrator is not.
- Recovering a plant from a learned closed-loop map (indirect method, Forssell and Ljung 1999)
  needs the exogenous plant-input signal f + K1 r, which K1's integrator makes unbounded.
- Model-based RL avoids the problem (one-step models, short rollouts), it does not solve it.

### Consequence for the comparison
- Black-box arm = Jan's SUBNET on the closed-loop map (r, f) -> y: the quantity the grey box
  is scored on, stable, trains like Jan's own example (option B, BB2-018).
- Price: no controller-transfer test (E5) for the black box; it models plant plus K1 as one.
  That is itself a result: the physics-based model can be put under a new controller, a
  black box of the loop cannot.
- Effort (R8): five designs tried, each stopped on a pre-registered rule; local cost under an
  hour of compute in total.

## Trainability of the black-box arm (option B): trains stably, has NOT learned the loop
- Correction after CRITIC finding 1: the pre-registered bar (1/3 of the mean-model RMS) was too
  weak. The zero-parameter predictor y_hat = r (the loop tracks its reference) scores 1.1e-5 to
  3.7e-5 m per axis on the same records; the trained black box (below) is 30 to 100x WORSE than it.
  So the PASS shows stable, finite, decreasing training only, not a useful model.
- Likely cause (CRITIC finding 2): auto_fit_norm scales y by its move std (0.05 to 0.15 m), so the
  loop's own behaviour (y - r, ~1e-5 m) is ~1e-4 of the normalised range. Proposed fix, not run:
  train on the servo error e = r - y (same score) in float64; pending the user (?).

### Results as run (criterion BB2-018)
Jan's SUBNET on the closed-loop map (r, f) -> y, criterion pre-registered in BB2-018 (all
validation free runs finite; final per-axis RMS <= 1/3 of the mean-model RMS; training loss
decreasing). Local: 10 epochs, stride 100, batch 64, float32, CPU.

| Version | Seed | Per-axis RMS X1 / X2 / Y [m] | / mean model | Criteria | Wall |
|-|-|-|-|-|-|
| noisy | 1 | 6e-4 / 5e-4 / 1.3e-3 | 0.011 / 0.010 / 0.009 | PASS | 28.7 min |
| noisy | 2 | 4e-4 / 3e-4 / 1.4e-3 | 0.007 / 0.006 / 0.010 | PASS | 26.5 min |
| noisy | 3 | 3e-4 / 5e-4 / 1.4e-3 | 0.006 / 0.010 / 0.009 | PASS | 30.1 min |
| noise-free | 1 | 6e-4 / 5e-4 / 1.3e-3 | 0.011 / 0.010 / 0.009 | PASS | 25.7 min |
| noise-free | 2 | 4e-4 / 3e-4 / 1.4e-3 | 0.007 / 0.006 / 0.010 | PASS | 28.4 min |
| noise-free | 3 | 3e-4 / 5e-4 / 1.4e-3 | 0.006 / 0.010 / 0.009 | PASS | 29.6 min |

- Curve (noisy, seed 1), validation sim-RMS [m]: 0.108 at start, then 7.0e-3, 4.0e-3, 2.8e-3,
  2.0e-3, 1.7e-3, 1.4e-3, 1.2e-3, 1.1e-3, 8.9e-4 over epochs 1 to 9: finite and falling every
  epoch, not converged after 10 epochs.
- Mean-model reference: 0.053 / 0.053 / 0.146 m; the bar was 1/3, reached about 100x below it.
- Noisy and noise-free agree per seed for two reasons: the data are cast to float32 (resolution
  ~3e-8 m near 0.3 m, above the 7e-9 m noise; 17 to 48 % of noisy samples identical after the
  cast, CRITIC finding 3), and the noise sits far below the model's ~1e-3 m error. So this is 3
  independent seeds, not 6 runs.
- Reading: the arm trains reliably (no divergence, small spread over seeds), but it is worse than
  y_hat = r, so it is not yet a comparison model. The FP epoch-0 score on these same records is
  still missing (?).
- Known scoring asymmetries to resolve before accuracies are compared (CRITIC finding 6):
  na_right 0 here vs 1 in the grey box; the grey box resets its controller state per window; r
  and f point-sampled vs u block-mean; selection and reporting on the same validation records.
- Cost (R8): 1.2 s per update incl. validation, 57 % of wall time is validation (deepSI free run
  of 6 records per epoch); 6 runs x ~28 min on this CPU.
- Not exercised locally: server settings (stride 10, batch 512, 200 epochs), GPU.

## Hyperparameters for the search (Phase E)
| Axis | Default | Range | Why |
|-|-|-|-|
| nx | 16 | 6, 8, 12, 16, 24, 32 | true order is plant information (not used); 6 = two states per output, 32 = generous |
| f, h width x depth | 8 x 2 (Jan) | 8, 16, 32, 64 x 2, 3 | Jan's 8 is sized for a 3-dof MSD; this plant has friction, Y-dependent inertia, 264 Hz peak |
| encoder width x depth | 16 x 2 (Jan) | 16, 32, 64 x 2 | 6 inputs x 29 lags + 3 outputs x 29 lags = 261 inputs |
| na = nb | 29 | 15, 29, 49 | grey box's lag; shorter or longer windows for the initial state |
| nf | 400 (0.1 s) | 200, 400, 800 | shared setting; D-220 loop memory ~50 ms |
| lr | 1e-3 (Jan) | 3e-4, 1e-3, 3e-3 | still improving at 10 epochs, so the rate is not the bottleneck yet |
| batch | 512 (server) | 256, 512, 2000 (Jan) | trades update count against gradient noise |
| epochs | 200 (grey box budget) | until validation stops improving | local runs not converged at 10 |
| seed | 1 to 3 | 3 per configuration | R8 spread over starts |
- Selection on the 6 validation records (sim-RMS), never test; report every configuration's cost
  and outcome (R8).
- Runner NOT ready (CRITIC finding 7): needs early stopping, per-epoch partial results and a
  CPU-or-GPU decision before submission.
- Runner: `runners/run_bb_clmap.sh` (not submitted); deepSI's stock fit runs on the CPU cores
  of the job unless cuda is enabled (a search-phase decision). Estimated CPU cost at server
  settings: 167 updates per epoch; at the local 1.2 s per update (batch 64) a 200-epoch run is
  well over the 24 h wall at batch 512, so either fewer epochs with early stopping or GPU (?).
