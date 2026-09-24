# encoder-transient: decisions (ET-xxx)

Session of 2026-09-24, unattended, per `tasks/handoffs/2026-09-23-encoder-transient.md` section 0.
Each entry is written BEFORE the code or run it governs. Thresholds are pre-registered here and
never edited after the number they judge exists; a revision is a new entry.

### ET-001 Infrastructure: vendor the working tree, patch only paths and record lists (G0)
**What**: `vendor/` holds copies of `scripts/gantry/transient/code/` (D-197 harness),
`scripts/gantry/gantry_dynamic/`, `model_augmentation/` (all `.py`), the entry file
`gantry_interconnect_dynamic.py`, `cl_burnin_sweep.py` with `cl_pipeline.py`, `cl_controller.py`,
`cl_capability.py`, `demo_common.py`, `drift_common.py`, `msd-offset/plant.py`, and
`common/oracle.py`. Source: working tree at HEAD `1ec57c1` (the handoff names `a97a577`; HEAD has
moved, the tree is dirty; provenance in `vendor/VENDORED.md`). The copies are edited only by
`tools/apply_vendor_patches.py`, every edit marked `# VENDORED:`: (1) sys.path points at `vendor/`
instead of the repo, so `model_augmentation` and `gantry_dynamic` are the vendored copies; (2)
`REPO_ROOT` is recomputed so data paths are unchanged; (3) `data.py`'s record lists are reset to the
22-record lists of commit `c82ff9b` (D-197), because D-206 appended Telica-profile records that do
not exist in the frictionless folder; (4) the harness pins `mode='augmentation_ma50_b140-230_a6_z03'`,
because the working-tree entry file names the friction folder (the only other entry-file change since
`c82ff9b` is `epochs`, which R0 does not read); (5) the exact-truth cache is READ from
`scripts/gantry/transient/cache/`, never rebuilt.
**Why**: the handoff forbids editing anything outside this folder, and the D-197 numbers are only a
reproduction target if the normalisation (training-set statistics) and the dataset are the D-197 ones.
**G0 threshold (pre-registered)**: vendored R0 on V1 gives Y RMS error within 1 % of `2.18e-05 m` and
dY within 1 % of `2.79e-02 m/s` (handoff section 9). Additionally the watchdog must end with a
`WATCHDOG SUMMARY` line and exit 0.
**Every run** goes through `tools/run.sh` (watchdog, outputs/<run>/resources.csv), one at a time.

### ET-002 The D-178 comparison numbers belong to another dataset; the planted model is rebuilt exactly
**What**: the 88 % / 1.25x / 3.40x of D-178 were measured on `mode='augmentation'` (`ma_frac 0.10`,
`zeta_a 0.05`, `nx_ann = 2`, `na_nb = 17`; entry file at commit `4553770`, the last before
2026-08-20), with a planted ANN fitted by regression to `msd-offset/plant.py` (`MA_FRAC = 0.10`,
`ZETA_A = 0.05`) and compared against a trained checkpoint (`*XadbYQ_best.pth`). So the handoff's
check "confirm the planted model matches the dataset's absorber" FAILS for the target dataset
`augmentation_ma50_b140-230_a6_z03` (`ma_frac 0.50`, `zeta_a 0.03`). This session therefore
(a) builds an EXACT planted model: the model's own state layout `[X, Th, Y, dX, dTh, dY, da, vda]`,
one step = RK4 of `truth.Plant8.deriv` (the replay-gated 8-state EOM at the dataset's absorber) with
the model's own `Ts` and `up_sample`, input held over the step exactly as `Gantry_State_Block` does;
no fitting, so no approximation error and no training update; (b) re-measures TODAY's numbers
(production encoder, `K = 0` and `K = 100`) on this dataset, same windows, same closed loop, and
states every G2 threshold relative to those re-measured values. The D-178 numbers are quoted only as
context.
**Why**: a threshold measured on a different plant is an oracle, not a data-derived criterion
(CLAUDE.md, control stance 8). An ANN regression fit would also be augmentation-parameter fitting,
which section 0 does not allow beyond 2 updates; the exact planted step needs none.
**Discrimination** is defined as in `cl_burnin_sweep.py`: `RMS_window(untrained) / RMS_window(planted)`
over the same windows, closed loop, residual form, `xc = 0`, each model initialised by its own
encoder policy. "Untrained" = the production baseline with the zero-initialised ANN (the handoff's
G1 wording); no trained checkpoint of this dataset exists on this PC (problem log 19.4).

### ET-003 G1 protocol: one map builder, four map variants, a two-factor attribution, the closed-loop transient re-baseline
**What (state error, no rollout)**: `encoder/recon.py` builds the reconstructability map exactly as
`linear_encoder_init_aug` does (Hoekstra 2026 Eq. 10, 13 to 17: `W_y = A^n O_n^+`,
`W_u = -A^n O_n^+ T_n + r_n`), in float64, in the pipeline's normalised I/O frame (`ystd`,
`std_u` from `normalize_linear_ss_matrices`), and applies it to physical-unit windows. Variants:
- `P0` production: baseline frozen-Y map at `Y = 0` (the `W^b` the pipeline builds);
- `A` operating point: baseline map at the window's own `Y`, read from the MEASURED stage `Y` at the
  window's last sample (deployable), exact frozen-Y maps on a 0.005 m grid over [-0.40, 0.40],
  linear interpolation between nodes;
- `B` model: the planted 8-state model (`truth.Plant8`, dataset absorber, `da = 0`) linearised at
  `Y = 0`, same builder; its 6 physical rows are scored, its 2 absorber rows are scored against the
  exact `[da, vda]` (a known realisation in the planted model, not a gauge);
- `AB` both. Windows `na = nb` in {15, 29, 59, 119} for `P0` and `AB`.
Records T1 to T5 and V1 to V4, window stride 5 (HEURISTIC: memory; still >= 9000 windows per record).
Collapse check: `P0` against the pipeline's own float32 encoder on V1, relative RMS difference
reported (must be far below every effect attributed).
**Attribution (pre-registered)**: per channel, on mean square error pooled over the 9 records,
a two-factor Shapley split of the removed MS: operating point `= ((P0 - A) + (B - AB)) / 2`,
model mismatch `= ((P0 - B) + (A - AB)) / 2`, remainder `= AB`; each as a fraction of `P0`. The
window's share is the further MS removed from `AB` by the best of the four windows, as a fraction
of `P0`. A cause is NAMED for a channel when its share is >= 0.5 (HEURISTIC: majority of the
error). The `|Y|` correlation is Pearson `corr(record RMS, record mean |Y|)` over T1 to T5 (as
D-167) and over all nine, for `P0` and `A`.
**What (closed loop, rollout without gradients)**: V1 to V4, non-overlapping `nf = 400` windows
from `k = na + 1`, production controller bank, residual form, `xc = 0`, float64. Untrained model =
the pipeline `hfn` (zero ANN); planted model = the exact step of ET-002, latent rows normalised by
the training-set std of the exact `[da, vda]`. Encoder policies for the planted latents: `random`
(the first two rows of the pipeline's kaiming `W^a`, i.e. today's literal init) and `zero`.
Reported per arm: window RMS, transient share `= (MS[0:400] - MS[100:400]) / MS[0:400]`
(D-178's definition, `diag_transient_source.py:110`), `grow = RMS(step 399)/RMS(step 0)`
(`interconnect.py:1220`), discrimination `RMS(untrained)/RMS(planted)` at `K = 0` and `K = 100`,
and the planted floor = planted model seeded with the EXACT 8-state truth (must be well below the
encoder transient for the planted test to measure the encoder).
**G1 PASS**: the attribution table exists for all six physical channels with the correlation
before and after `A`, and the planted floor is at most 10 % of the planted `K = 0` window RMS
(HEURISTIC: otherwise the planted test measures discretisation, not the encoder).

### ET-004 G2: the fix, its collapse gate, degree rule and pass thresholds (pre-registered after G1, before G2)
**G1 facts this rests on** (runs R003, R004): Y/dY state error is 100 % model mismatch (the
absorber on the payload; flat across T1-T5 including T3 at `Y = 0`); X, dX 97 % and Theta, dTheta
~100 % operating point, with a model interaction (the planted map alone is worse on Theta, both
together 33x better); window share 0 (longer windows are WORSE once the map is right). On the exact
planted model with today's encoder the window loss is 99.6 % transient, discrimination 1.275x at
`K = 0` and 19.23x at `K = 100` (W^a random, today's literal init; 6.58x / 119.3x with W^a zero).
**The variant** (`encoder/sched_encoder.py`, `ScheduledReconEncoder`): the production
`linear_encoder_init_aug` as parent (its parameters are `W_0`, so degree 0 IS the parent), plus
`W(Y) = W_0 + sum_{k=1..deg} C_k (T_k(s) - T_k(0))`, `s = Y / 0.40`, `T_k` Chebyshev (HEURISTIC:
Chebyshev instead of D-167's monomials, for float32 conditioning; the pin at `Y = 0` is exact by
construction). `Y` = the measured stage `Y` at the window's last sample, de-normalised from the
encoder's own `yhist` (deployable, D-167). `C_k` fitted by least squares to the EXACT frozen-Y maps
on the 0.005 m grid over [-0.36, 0.36] (D-167 method). `source`: `baseline` (6 physical rows from
the production linearisation, W^a kept from the parent) or `planted` (8 rows from the planted
model's linearisation: W^b AND W^a from the model the encoder feeds, so the latent rows are
initialised consistently in the planted realisation). Window: `na = 29` (production), because G1
shows no window gain once the map is right and longer windows degrade Theta (the R1 caveat).
**Collapse gate**: `source='baseline', deg=0` returns the parent's output bitwise (`torch.equal`) on
all V1 windows, float32 and float64.
**Degree rule**: the smallest `deg` in 0..10 such that, pooled over T1-T5 (training records only),
every physical channel's RMS error against the exact truth is within 10 % of the exact grid map's
(HEURISTIC: the polynomial must not give back more than a tenth of what scheduling buys); if none
qualifies, the `deg` minimising the worst ratio, reported as such. Relative map residual per degree
is reported alongside.
**G2 PASS** (planted model, V1-V4, `nf = 400`, `K = 0`, closed loop, float64; each model with the
map built from ITS OWN model: planted with `source='planted'`, untrained with `source='baseline'`):
(a) transient share <= 30 % (HEURISTIC; the literal "no longer dominates" is < 50 %, also reported);
(b) discrimination `RMS(untrained)/RMS(planted)` at `K = 0` >= 0.85 x 19.23 = **16.35x** (the burn-in
value re-measured on this dataset with today's encoder, ET-002; the handoff's 0.85 fraction);
(c) physical-state RMS error against the exact truth (T1-T5 and V1-V4, `na = 29`) lower than `P0`
on all six channels, none worse than `P0` by more than 5 % on any single record (HEURISTIC).
Up to three attempts; each attempt is a new RUNS row with what changed.

### ET-005 G3 protocol: R1 from the G2 encoder, R2 free run on both models
**R1 (structure headroom)**: `diagnose/g3_r1.py`, the vendored R1 recipe (Adam lr 1e-4, batch 4096,
per-channel loss weight = 1 / that channel's MSE at initialisation, epoch 0 a checkpoint candidate,
best validation epoch kept), exact target, train T1-T14 (stride 4), validate V1-V4 (stride 20),
at most 1000 epochs, float64 (the G2 errors are 100x to 300x below `P0`, so float32 rounding, which
is 6e-4 of the `P0` error on Y/dY, would be ~0.1 to 0.2 of the G2 error). Two arms, sequential,
GPU: (i) the G2 planted-source encoder (8 channels, the absorber pair has an exact target in the
planted realisation); (ii) the G2 baseline-source encoder (6 physical channels), the analogue of
D-197's R1 on `P0`. Headroom = init RMS / trained RMS per channel.
**R2 (free run)**: `diagnose/g3_r2.py`, open loop as D-197 R2, recorded input replayed, V1-V4,
20 starts per record, horizons 400 and 4000, float64. Untrained model: `x0` in {exact, `P0`, G2
baseline-source}; planted model: {exact, `P0` W^a random, `P0` W^a zero, G2 planted-source}.
**G3 PASS**: numbers reported, and on the planted model, pooled over V1-V4, on every output channel
and both horizons, RMS(G2) <= RMS(`P0`, W^a random) and <= RMS(`P0`, W^a zero).

### ET-006 Refactor for Telica, and the G4 protocol (pre-registered before any Telica number)
**Refactor**: the pure map algebra (`recon_map`, `pure_map`, `cheb_basis`, `grid_maps`,
`fit_schedule`, `numpy_sched`, `grid_apply`, `ScheduledReconEncoder`) moves to `encoder/core.py`
(numpy/torch only); `recon.py` and `sched_encoder.py` re-export it unchanged. Reason: the Telica
study must run on telica-real's vendored stack (its `gantry_ss` carries the Telica parameters,
`telica-real/vendor/VENDORED.md`, TR-003), so it cannot import the simulation vendor. G2 is re-run
after the refactor as a regression (numbers must reproduce).
**Telica vendor**: `telica/vendor_telica/` = copies of telica-real's `tr_env.py`, `data_index.py`,
`rate.py`, `vendor/`, `params/`, `friction/`, `controller/`, `baseline/records.py`,
`pipeline/{real_data,telica_model}.py`, `pipeline/norm_frozen.json` (HEAD `1ec57c1`); patched only in
`tr_env.py` (repo root for the data path, outputs into `outputs/telica/`). Data only through the
vendored loader; only summaries printed.
**Records and rate**: iter0 (pure feedback, no feedforward) of all 15 operating points, at the
Telica pipeline's own rate and window: 5 kHz by the TR-022 rule (`rate.decimate_record`), cut to
[motion - 100 ms, end] as `real_data._cut`, `na = nb = 29` (Jan's rule, `nx_ann = 8`), frozen
normalisation `norm_frozen.json`.
**Encoders**: `P0-T` = exactly `telica_model.build_model_real`'s encoder (`Y_op = 0` map, Telica
parameters, `_synthetic_sysdata` scaling). `G2-T` = `ScheduledReconEncoder` on that parent,
baseline source (the Telica model has no absorber realisation; its friction is nonlinear and not in
any linear map), W^a untouched. Degree rule, truth-free: smallest `deg` in 0..10 with, per physical
channel on the TRAIN-split windows, `RMS(x_poly - x_grid) <= 0.1 RMS(x_grid - x_P0)` (the ET-004
"do not give back more than a tenth of what scheduling buys", with the exact grid map as reference
instead of the unavailable truth).
**Reference states**: positions `q_ref = P^-T y` from the raw 20 kHz measured positions (`M0 - M2`)
at 20 kHz sample `4k` (= 5 kHz sample `k`, `rate.py`); velocities `P^-T d/dt LP_fc(y)`, zero-phase
Butterworth order 4 (`sosfiltfilt`), central difference (`np.gradient`) at 20 kHz.
**Error bound per channel and record** (# THEORY: Parseval; the reference error of a linear
procedure is that procedure's response to the noise plus its deviation from identity on the signal):
noise part `sigma_n` = std of the same procedure on the pre-motion standstill segment (raw samples
[200, motion - 400], HEURISTIC 10 ms / 20 ms edge margins; it contains the servo's own standstill
micro-motion, so it is an upper bound on the noise); distortion part
`delta^2 = sum_f (1 - G(f))^2 [S_mot(f) - S_st(f)]_+ df`, `G = |H_butter|^2` (zero-phase gain),
`S` = Welch PSD (nperseg 2048) of the raw central-difference velocity in motion and at standstill;
bound = `sqrt(sigma_n^2 + delta^2)`. Positions: bound = standstill std of the linearly detrended
positions. `fc` per velocity channel: the value in {50, 100, 200, 400, 800, 1600} Hz minimising the
pooled bound (chosen before any encoder is scored).
**Scores**: RMS(encoder - reference) per channel, per record, pooled per `ypos` (5 levels) and over
all 15; also the debiased `sqrt(max(MS - bound^2, 0))`, and the encoder's own noise floor (velocity
std of each encoder on standstill windows, where the true velocity is ~0).
**G4 PASS**: (1) resolvable: per channel, pooled over the 15 records, bound < RMS(`P0-T` - ref);
channels that fail this are reported as not resolvable and excluded from (2); (2) on every resolvable
channel `G2-T` has a lower pooled RMS than `P0-T`, and at no `ypos` level is it higher by more than
1 % (HEURISTIC).

### ET-007 G3 extra arm: the closed-form best LINEAR structure against the exact truth (R1-LS)
**Why**: the R1 recipe (Adam, lr 1e-4) cannot answer "how much headroom does the structure have" when
the start is already near-exact: by epoch 200 of arm (i) the weighted validation MSE is 3000x ABOVE
its epoch-0 value, the known scale-free first-step problem of Adam (D-197 R1 notes). A regression that
is linear in its parameters has a closed-form optimum, so the structure's ceiling can be computed.
**What**: `diagnose/g3_r1ls.py`. Structure = the G2 planted-source linear part, `x = sum_{k=0..6}
phi_k(Y) W_k w_pure` (`phi_0 = 1`, `phi_k = T_k(s) - T_k(0)`), 8 outputs, 7 x 180 = 1260 features.
Ridge least squares against the exact state (pure frame), normal equations accumulated in chunks,
float64. Ridge `lambda` = `r * trace(F'F)/1260` with `r` in {1e-14, 1e-12, 1e-10, 1e-8, 1e-6}
(HEURISTIC grid), chosen on held-out TRAINING records T6 and T11 after fitting on the other 12; refit
on all 14 with that `r`; scored on V1-V4 (stride 20, as R1) and T1-T5 (stride 5).
**Reported**: per channel, G2 init RMS / LS RMS on V1-V4 = the linear-structure headroom. No PASS
threshold (G3's PASS is the R2 criterion of ET-005); it is a reported number.

### ET-008 G5: server arms, predictions, and the smoke budget (written before any G5 code runs)
**Pre-check** (`diagnose/g5_modelmap_check.py`, no training): the training-time builder
`encoder/model_map.py` (Jacobians of the model's own step at rest at each `Y`, then the same
reconstructability algebra and Chebyshev fit, deg 6, all rows incl. W^a) must reproduce G2 on the
planted model: planted window RMS <= 1.10 x the exact-state floor 1.1638e-07 m and share <= 30 %.
On the untrained model its latent rows must be numerically zero (unobservable latents: the
minimum-norm, model-consistent W^a).
**(a) simulation arms** (`runners/sim_arm.py`, frictionless dataset, production config otherwise,
same seed): `burnin` = production encoder, `burn_in = 100` (the current burn-in arm); `g2static` =
G2 baseline-source map (deg 2), W^a = 0, `burn_in = 0`; `g2refresh` = model-Jacobian map (deg 6, all
14 rows) rebuilt from the CURRENT model every epoch (HEURISTIC cadence; Adam state of the overwritten
encoder parameters reset), `burn_in = 0`.
Predictions: (P1) `g2static` tracks the production `K = 0` loss within 10 % in early epochs (the
untrained error is model mismatch, G2: 2.0535e-05 vs 2.0548e-05), and does NOT fix the transient once
the ANN uses its latents (W^a is not rebuilt). (P2) `g2refresh`: the `[nf]` line's `grow` moves from
today's 0.46 to 0.81 toward >= 0.9 once the ANN has moved, and its best validation sim-RMS is <= the
`burnin` arm's at equal epochs. (P3) `burnin` behaves as D-178. Falsifier of the fix in training:
`g2refresh` with `grow < 0.8` after the ANN has moved, meaning the per-epoch map does not track the
learned latent realisation (or the affine offset at rest, reported per refresh, is not negligible).
**(b) R2 against a trained checkpoint** (`runners/r2_ckpt.py`): open-loop R2 (V1-V4, 20 starts,
H 400 / 4000) and closed-loop `nf = 400` window metrics for x0 = exact / production encoder /
model-Jacobian map of the TRAINED model. Prediction: the model-Jacobian map gives Y free-run RMS <= the
production encoder's and a lower window transient share; if the exact seed is worse than the
production encoder (D-197 compensation), that is reported as the trained model's residual mismatch.
**(c) Telica arm** (vendored `telica_augment.py`, env `ET_G2_ENCODER`): the telica-real production
arm with the G2-T encoder (G4). No model-Jacobian refresh on Telica: the tanh friction's Jacobian at
rest is a near-infinite viscous term, so a rest-point map would be wrong during motion.
**Smoke budget: 2 optimizer updates in total.** `g2refresh` with `n_its = 1` (1 update, 2 refreshes:
start and after the update) and the Telica arm with `n_its = 1` (1 update). `burnin`, `g2static` and
(b) run DRY (build, swap, one closed-loop forward on V1, zero updates). Nothing is submitted.
**G5 PASS**: the pre-check passes, all five smoke/dry runs exit 0 under the watchdog with the update
counts above, and the predictions are in `runners/PREDICTIONS.md`.

### ET-009 G4 amendment (before any Telica number): the distortion term uses PSDs pooled over records
**What**: in ET-006's bound, `delta^2 = sum_f (1 - G)^2 [S_mot - S_st]_+ df` is computed from the
motion and standstill PSDs AVERAGED over the 15 records, not per record; the noise part `sigma_n`
stays per record. **Why**: a single record has ~7 Welch segments, so at high frequency, where the
raw central-difference velocity is pure noise (its PSD grows as f^2), `S_mot - S_st` fluctuates by
about +-40 % of the noise PSD and the positive part turns that fluctuation into a spurious "signal"
right where `1 - G = 1`. Pooling 15 records cuts that variance ~15x. The bias that remains is
upward, so the bound stays conservative. Everything else in ET-006 is unchanged.

### ET-010 G4 attempt 2: build the Telica map from the model the Telica encoder FEEDS (recovered parameters + its friction)
**Attempt 1 (R011) FAILED** its ET-006 bar: the reference resolves every channel by 2 to 3 decades
(bound dX 4.9e-05 m/s vs encoder error 7.2e-03), the Y schedule lowers the Theta pair 16 % pooled
(37 % at ypos -200) but X/Y are unchanged and X is 2.7 % / 7.6 % WORSE at ypos +40 / +120.
**Mechanism found (read, not measured):** `telica_model.build_model_real` feeds the encoder into a
physical block with the RECOVERED attempt-2 parameters (`recovered_params_a2.json`: viscous cg1/cg2/cy
309/322/308 N s/m against datasheet 136/136/98, mh 20.95 vs 19 kg, re-split m1/m2/mb) and tanh Coulomb
friction cc = 82.6/100.3/80.5 N (TR-015), but builds the encoder's map from
`gantry_linearize_and_discretize`, i.e. the DATASHEET constants of `telica_params` and no friction.
So attempt 1 was not "built from the model the encoder feeds", which is exactly the cause G2 fixed in
simulation. Every Telica record is one positive move (telica-real G4), so friction is a ~80-100 N
constant force during the move, against a force std of ~350-520 N.
**Attempt 2** (`telica/g4_telica_a2.py`), reference and bounds of ET-006/ET-009 unchanged:
the map comes from the Jacobians (`encoder/model_map.py`) of the pipeline's OWN physical block
(`GantryFrictionBlock`, recovered raw14, `rebuild_float64`, RK4 `up_sample = 1` at 5 kHz, frozen
normalisation) with `cc = 0`, at rest at each `Y` on the fit grid; the friction enters as the
model's own input correction `u_eff = u - cc tanh(v_stage / v0)` (cc, v0 of the block) applied to
the window before the map, with `v_stage` the central difference of the MEASURED 5 kHz window
positions (HEURISTIC estimator; its noise, ~2e-05 m/s, is 50x below v0 = 1e-3 m/s). Y schedule by
the ET-006 truth-free degree rule. Variants reported for attribution: `R0` recovered map at Y = 0,
`RS` recovered + schedule, `RF0` recovered + friction at Y = 0, `RFS` all three = the attempt-2 G2-T.
**PASS**: the ET-006 criteria unchanged, applied to `RFS` against `P0-T`.

### ET-011 G4 attempt 3: the window, chosen on the training split; a kinematic estimator as a reported reference
**Attempt 2 (R012) FAILED** only on the Theta pair's per-ypos bar: RFS lowers every channel pooled
(X/dX 25 %, Y/dY 34 %, Theta/dTheta 2 %) but the Theta pair is 11-21 % worse than P0-T at ypos
-40/+40/+120. Mechanism: the measured Theta has std 2.9e-07 rad (the rails track each other), while
every map outputs Theta errors of ~2.7e-05 rad, 100x the signal. The map reconstructs Theta through the
model's Theta row (stiffness kb, damping cb, J_eff), whose parameters telica-real G4 found NOT
identifiable from these one-way moves, so neither the datasheet nor the recovered model has a
Theta row that matches the machine; which one is less wrong depends on Y.
**Attempt 3**: the one G2 design knob left, the window. With a wrong model, the map's error grows
with how far the window propagates it; the encoder's own noise grows as the window shrinks, and on
Telica the noise floor (standstill velocity std 2.6e-05 m/s) is 280x below the error, so shorter
windows should trade little noise for less mismatch. `na` in {7, 11, 15, 21, 29} (1.6 to 6 ms),
RFS rebuilt at each `na` (recovered block Jacobians, friction correction, Y schedule by the ET-006
degree rule), `na` CHOSEN on the 11 TRAIN records by the pooled mean over the six channels of
RMS(RFS)/RMS(P0-T) (HEURISTIC single objective), then judged by the unchanged ET-006 rule on all 15
records, and reported separately on the 4 held-out val/test records (the selection uses 11 of the
15, so the all-15 verdict carries mild selection leakage; the held-out one does not).
**Reference arm, not a candidate**: a kinematic estimator from the same window, positions
`P^-T y(k)` and velocities `P^-T` times the endpoint derivative of a quadratic least-squares fit to
the last 11 samples (HEURISTIC Savitzky-Golay form). It shows how far any model-based map sits from
what the measurement itself supports; it is not the G2 variant (it ignores the model, and D-197 R2
showed a true-state seed can be the WORSE seed for a mismatched model).
If attempt 3 fails, G4 is recorded FAIL after three attempts with this mechanism.

### ET-012 G5 (c) follows G4: the Telica arm uses the attempt-3 encoder (RFS, na = 7)
The encoder that passed G4 is RFS at `na = 7`, so the Telica server arm must run the pipeline with
`na_nb_override = 7` (the SUBNET window is the encoder window) and load that encoder. Two marked
edits to the vendored `telica_augment.py`: `ET_NA_NB` sets `na_nb_override` in `make_cfg`, and
`ET_G2_ENCODER` now loads the saved `FrictionCompEncoder(ScheduledReconEncoder(...))` state into the
same structure built around the pipeline's own encoder (the saved state overwrites W^b, the Y
schedule and the friction constants; W^a keeps the G4 build's seeded draw, documented). The shorter
window also shortens the model's history (`cheat_n`) and changes the window count; the paired
reference is telica-real's own server run. Prediction (PREDICTIONS.md): lower window transient
share and lower initial training loss than telica-real's arm; the final validation free run is not
predicted, because the encoder only sets `x0` of a 40 ms window while the free run is initialised
once. The smoke budget of ET-008 is unchanged (1 update on this arm).

### ET-013 G5 arms freeze the encoder's linear map (evidence from the two smoke updates)
**Evidence** (R020, R025; the two updates of the budget): after ONE Adam update the closed-loop
window error on the validation batch rose 8x in simulation (`[nf val]` 1.91e-05 -> 1.56e-04 m) and
30x on Telica (5.08e-07 -> 1.47e-05 m), and `grow` fell from 1.83 to 0.43 (sim) and 0.41 to 0.01
(Telica). A `grow` below 1 means the error is concentrated at the window START, i.e. it is an
initial-state error; a damaged ANN would make the error grow through the window instead. So the
evidence points at the encoder: Adam's scale-free first step (about lr on every parameter,
whatever its gradient) wrecks a near-exact linear map, the same mechanism R1 showed in encoder-only
training (R007: never recovers the init in 1000 epochs). Not separable further without more updates.
**Decision**: in the `g2static`, `g2refresh` and Telica arms the encoder's linear map (`W^b`, `W^a`,
`C_k`) gets `requires_grad = False` (D-175's meaning of freeze); the correction net stays trainable
(zero-initialised output layer). For `g2refresh` the map is maintained by the per-epoch rebuild
from the model instead of by gradient. `ENC_TRAIN_LINEAR=1` restores the trainable map for a paired
check. The production `burnin` arm is untouched (it is the reference as it runs today).
**Checked without updates**: a dry run per arm must report the frozen parameter count.
