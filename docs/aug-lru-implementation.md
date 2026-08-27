# AUG_LRU: implementation reference for the D-150 augmented-dynamics bypass

Written 2026-08-20. This is the exact, code-level reference for what `AUG_LRU=1` does. The design
rationale lives in `docs/decisions.md` D-150; the measurements live in
`scripts/gantry/closed-loop-controller/ANN-learning-issue/RESULTS.md` section 9; this document
records only what the code IS, so it can be audited, reproduced, and later reimplemented cleanly.

## 1. Status: what this is and is not

**It is a mechanically verified parameterisation change, not a validated method.** Verified
(artefacts in section 9): the augmented states have live, stable, band-initialised dynamics; the
augmented model is bit-identical to the baseline at initialisation; every parameter, including the
encoder's augmented block `W^a`, receives gradient and trains. Not achieved: the free-run
acceptance criterion. Arm F reached `1.3841e-06 m` against a target of `1.215e-06 m`, i.e.
`-0.665 %` relative to the pre-fix plateau. The identified reason is the objective's weak
discrimination (H2), not this implementation. Any write-up that calls this "the working method" is
overstating; the defensible claim is "the initialisation obstruction is removed and measured to be
removed".

**Integration is experiment-grade.** The switch is an environment variable, not a config field,
so that every existing run reproduces bit-identically with the gate off and no frozen file had to
be edited. Section 10 lists every shim this choice created; section 11 is the plan for the clean
version and what blocks it.

## 2. The mathematics implemented

State layout (logical coordinates, `nxd = 8`):
`x = [X, Theta, Y, dX, dTheta, dY, x_a1, x_a2]`, physical rows `0..5`, augmented rows `6..7`.
The ANN input is `z = [x, u]` with `nu = 3`, so `z` has 11 columns and `x_a = z[:, 6:8]`.

Without the gate, the interconnect gives (as read and cross-checked by `cl_aug_spectrum.py`):

```
x_{k+1}[0:6] = phy_block(x_k[0:6], u_k) + ANN(z_k)[0:6]
x_{k+1}[6:8] =                            ANN(z_k)[6:8]        <- rows 6-7 fed by the ANN alone
y_k          = C x_k[0:6] + D u_k                              <- output never reads rows 6-7
```

`zero_init_feed_forward_nn` zeroes the ANN's final Linear, so at initialisation
`x_{k+1}[6:8] = 0`: the augmented states have no dynamics at all (`rho(A_aa) = 0` exactly, the
2026-08-19 finding).

With the gate, the ANN's function `net(z)` is replaced by the wrapped function

```
w        = MLP(z)                                   MLP = zero_init_feed_forward_nn, unchanged
out[0:6] = w[0:6]                                   = 0 at init  ->  D-072 baseline equality
out[6:8] = A_aa z[6:8] + gamma * w[6:8]             = A_aa x_a at init (w = 0)
```

where for `nx_aug = 2` there is one complex-conjugate eigenvalue pair, realised as the 2x2
rotation-scaling block

```
A_aa = r * [[cos w, -sin w],
            [sin w,  cos w]]          rho(A_aa) = r exactly, independent of state
```

with the LRU stable exponential parameterisation and input normalisation
(THEORY: Orvieto et al., ICML 2023, arXiv:2303.06349, Sections 3.3 and 3.4; PDFs in
`literature/deep-ssm-init/`):

```
r = exp(-exp(nu_log))        trainable nu_log,    guarantees 0 < r < 1 for all parameter values
w = exp(theta_log)           trainable theta_log  [rad/sample]
gamma = sqrt(1 - r^2)        scales the NL input into the state
```

For general even `nx_aug` the code builds `nx_aug/2` independent pairs (`nu_log`, `theta_log` are
vectors of length `n_pairs`); odd `nx_aug` is rejected by assertion.

Initialisation of `(r, w)` per pair, uniform on an annulus and a phase arc
(THEORY: Orvieto et al. Lemma 3.2 for the radius; the phase restriction to a band is our variant
of their phase range):

```
r     = sqrt(u * (r_max^2 - r_min^2) + r_min^2),   u ~ U[0,1]
theta = theta_lo + v * (theta_hi - theta_lo),      v ~ U[0,1],   theta = 2*pi*f*Ts
```

`[r_min, r_max]` and `[f_lo, f_hi]` come from data (section 6), never from constants in code.

**Known and accepted limitation** (already in the problem log for the black-box track):
`r = exp(-exp(nu_log))` maps onto the OPEN unit disk, so `|lambda| = 1` is unreachable.
Acceptable for a parallel augmentation because the integrators live in the baseline; not
acceptable for a black-box arm.

## 3. Where the code lives

All in files this project owns (nothing added to Jan's `model_augmentation/` for this feature).

| symbol | file | role |
|-|-|-|
| `class AugLRUBypass(torch.nn.Module)` | `scripts/gantry/gantry_dynamic/model.py` | the wrapper of section 2 |
| `def lru_band_from_artifact(artifact_path, ts)` | same file | the band derivation of section 6 |
| the `if os.environ.get('AUG_LRU') and NX_ANN > 0:` block inside `build_model` | same file | build-time wiring, section 4 |
| burn-in/consistency refuse-to-start guard | `scripts/gantry/closed-loop-controller/cl_train.py`, module level, right after `CONS_FRAC` is parsed | unrelated to the bypass but added the same session; see section 10 |
| `CL_NOISE_SIGMA` noise hook | `cl_train.py`, inside `main()` after the train/val lists are loaded | noise gate, training side |
| `CL_RS_NOISE_SIGMA` noise hook + `_noisy` output redirect | `scripts/gantry/closed-loop-controller/cl_residual_spectrum.py` | noise gate, spectrum side |
| `CL_SPEC_OUT`, `CL_SPEC_SKIP_PLANTED` | `scripts/gantry/closed-loop-controller/cl_aug_spectrum.py` | artefact protection + planted-arm skip |

`AugLRUBypass.__init__(mlp, aug_out_pos, x_aug_in_pos, r_init, theta_init)`:

* `mlp`: the `zero_init_feed_forward_nn` instance the ANN block already built; stored as
  `self.mlp` (a submodule, so its parameters stay in the optimizer).
* `aug_out_pos`: columns of the ANN OUTPUT that write the augmented state rows, i.e. the positions
  of state rows 6 and 7 inside `cfg.ann_route_ix`. For the production routing `(0..7)` these are
  `(6, 7)`; for any other routing they are computed with `np.where`, and an assertion refuses a
  routing that does not contain the augmented rows.
* `x_aug_in_pos`: columns of the ANN INPUT `z` holding `x_a`; `(6, 7)` because `z = [x, u]`.
* Both index tuples are plain Python tuples, not tensors or buffers, so the module pickles under
  deepSI's `torch.save(self.__dict__)` checkpointing without any special handling.
* `r_init`, `theta_init`: numpy vectors of length `n_pairs`; asserted `0 < r < 1`, `theta > 0`;
  converted to the internal parameters as `nu_log = log(-log r)`, `theta_log = log theta`.

`AugLRUBypass.forward(X)` computes exactly section 2, with the output assembled as
`out = w.clone()` followed by indexed assignment on the augmented columns (autograd-safe; no
in-place operation on a leaf).

`AugLRUBypass.net` is a **property** returning `self.mlp.net` (the inner `nn.Sequential`). It
exists purely for compatibility: `cl_aug_spectrum.py` reads `ann.net.net[0].in_features` and
`ann.net.net[-1].out_features`, and `rezero_gate.apply_rezero_gate(net_module)` reads
`net_module.net`. With the property, both see the same Sequential they always saw.

## 4. Build-time wiring and ordering

Inside `build_model` (`gantry_dynamic/model.py`), in this order:

1. `Static_ANN_Block` is constructed with `net=zero_init_feed_forward_nn` exactly as before.
2. The optional `ANN_REZERO_GATE` block runs (unchanged, and NOT used together with `AUG_LRU` in
   any run so far; the combination is untested).
3. **The `AUG_LRU` block**: derive or read the band, draw `(r, theta)`, then
   `ann_block.net = AugLRUBypass(ann_block.net, aug_out_pos, aug_state_ix, r_init, theta_init)`.
4. `fit_sys.init_model(...)` builds the optimizer. Because the wrapper was installed BEFORE this,
   `nu_log` and `theta_log` are ordinary parameters of `hfn` and land in the optimizer's bulk
   group automatically. They receive `lr` and, in `cl_train.py` runs, the `CL_ADAM_EPS` override,
   like every other ANN parameter. There is no separate learning rate for them.
5. `fit_sys.hfn.to(DTYPE_PT)` casts the new parameters to the pipeline dtype (float32 in the
   production config; the parameters are created float64 and cast here).

Randomness: the draw uses a **dedicated** generator,
`torch.Generator().manual_seed(int(cfg.seed) + 150)`, so (a) the draw is reproducible per seed and
(b) the global torch RNG stream is not consumed, which would silently shift the encoder's random
`W^a` initialisation relative to every earlier run at the same seed. With `cfg.seed = 0` (the
closed-loop pipeline's seed) the draw is `r = 0.992040` at `f = 154.52 Hz`, and this exact value
is printed at build time in the `[aug-lru]` line.

A build with the gate on prints one line, e.g.:

```
[aug-lru] D-150 bypass on augmented rows [6, 7]: band f [149.90, 164.06] Hz, rho [0.9794, 0.9956]
(source: <path>); drawn lambda: r 0.9920 at 154.52 Hz; rows 0-5 still exactly zero at init
```

That line is the run's record of which band and which draw it trained from; it appears in every
`.output` log of every gated run.

## 5. Environment-variable contract

Model construction (read inside `build_model`, so they affect ANY entry point that reaches it:
`cl_train.py`, `cl_aug_spectrum.py`, `demo_common.build_pipeline`, scratch scripts):

| variable | default | meaning |
|-|-|-|
| `AUG_LRU` | unset = OFF | any non-empty value enables the bypass. OFF means the build is bit-identical to the pre-D-150 pipeline. |
| `AUG_LRU_ARTIFACT` | `scripts/gantry/closed-loop-controller/runs/cl_residual_spectrum.json` (resolved relative to `model.py`) | path of the residual-spectrum artefact the band is derived from |
| `AUG_LRU_BAND` | unset | `"f_lo,f_hi"` in Hz. Explicit band override; MUST be set together with `AUG_LRU_RHO`. Intended for datasets where no strong residual peak exists (the Telica case): the values then come from band requirements (loop bandwidth, sample rate), stated by the experimenter, not derived here. |
| `AUG_LRU_RHO` | unset | `"r_min,r_max"`, the annulus radii, same rule as above |

Diagnostics and the noise gate (script-local, not read by `build_model`):

| variable | script | meaning |
|-|-|-|
| `CL_NOISE_SIGMA` | `cl_train.py` | `"sx1,sx2,sy"` in metres; white Gaussian noise added to `y` of ALL train and validation records, dedicated `np.random.default_rng(150)`. The Telica-derived values are `8.544e-9,7.762e-9,6.539e-9` (see section 6 of `RESULTS.md` 9e for the derivation); they are launch inputs, not code constants. |
| `CL_RS_NOISE_SIGMA` | `cl_residual_spectrum.py` | same format; noise added to `y` before the residual is formed. Setting it also redirects the output artefact to `runs/cl_residual_spectrum_noisy.json` so the CLEAN artefact, which is the band source, can never be overwritten. |
| `CL_SPEC_CKPT` | `cl_aug_spectrum.py` | (pre-existing) checkpoint to measure |
| `CL_SPEC_OUT` | `cl_aug_spectrum.py` | output JSON path override; use it for any new measurement so `runs/cl_aug_spectrum.json` (the 2026-08-19 record) is preserved |
| `CL_SPEC_SKIP_PLANTED` | `cl_aug_spectrum.py` | skip the planted reference arm. REQUIRED when measuring a gated checkpoint: the planted state_dict predates the wrapper and cannot load into it (see section 8). |
| `CL_BURNIN`, `CL_CONS_FRAC` | `cl_train.py` | now REFUSE TO START if non-zero: the framework support was reverted on 2026-08-19 (`interconnect.py` byte-identical to commit `4cdb7c1`), so a run with either set would print the requested objective and silently train on the full window. Re-apply `patches/2026-08-19-interconnect-burnin-consistency.patch` AND remove this guard together. |

## 6. The band derivation, exactly

`lru_band_from_artifact(artifact_path, ts)` mirrors `cl_residual_spectrum.py`'s READING 1:

1. For every record and every channel in the artefact, keep the peaks with `zeta_ok == true`
   (trustworthy half-power damping estimate) and `over_floor_db > 10`
   (HEURISTIC: the same strong-peak threshold the artefact producer uses; chosen, not derived).
2. Per record-channel, keep only the DOMINANT such peak (largest `over_floor_db`). This is what
   prevents excitation content (the APRBS records carry 20 to 110 Hz peaks) from entering the band.
3. `f_band = [min, max]` of those dominant frequencies over all record-channels. On the 18-record
   simulation artefact this is `[149.90, 164.06] Hz` from 54 peaks; the width IS the estimator
   scatter (about +-5 Hz per estimate), which is exactly the uncertainty the ring should cover.
4. Per peak, `rho = exp(-zeta * wn * Ts)` with `wn = 2*pi*f / sqrt(1 - zeta^2)`
   (THEORY: discrete pole magnitude of a second-order mode; identical formula to the artefact
   producer). `rho_band = [min, max]` = `[0.9794, 0.9956]` on the same artefact.
5. If NO strong peak exists anywhere, the function RAISES with instructions to set
   `AUG_LRU_BAND`/`AUG_LRU_RHO`. It never silently defaults. This is deliberate: on Telica the
   identifiable band is at most 83 Hz (X) / 55 Hz (Y) and no plant resonance is supported
   (`telica_plant_frf.py`), so the band there is an engineering statement, not a measurement, and
   must be visible as one at launch.

Properties that make this the deliverable recipe: it uses only `u`, `y` and the baseline (no
oracle quantities), it collapses toward a mode when one is identifiable (simulation) and stays a
band when not, and it survived Telica-level measurement noise unchanged
(`runs/cl_residual_spectrum_noisy.json`).

## 7. Runtime behaviour and gradients

* At `t = 0` the model IS the baseline: rows 0-5 of the ANN output are exactly zero and the
  output map never reads rows 6-7. Verified bit-identically (`2.1866011034177349e-06 m` untrained
  closed-loop sim-RMS with the gate on and off, difference `0.000e+00`).
* `x_a` is produced by the encoder (`rms(x_a) ~ 1.09` at depth 0 with the random `W^a`) and now
  persists: with the drawn `r = 0.992` the time constant is 125 samples = 31 ms at 4 kHz.
* `dL/dA_aa` (i.e. into `nu_log`, `theta_log`) is zero at step 1 of a rollout and unlocks from
  step 2, because the only path to the loss runs through the physical readout, which starts at
  zero. This is forced by exact baseline equality, not by the implementation. The alternative
  (initialise the readout at a small epsilon, trains from step 1, model no longer exactly the
  baseline) is a deliberate weakening of D-072 and is the user's decision; it is NOT implemented.
* Measured consequence after training (Arm F best checkpoint): `rho(A_aa)` 0.9920 -> 0.9920,
  `f` 154.52 -> 154.56 Hz. Training moved the dynamics parameters by ~4e-05: it neither collapsed
  nor exploited them, which is the H2 finding.
* `Wa_psi_y` and `Wa_psi_u` (encoder augmented block): 108/108 entries moved each (max |dW|
  3.0e-04 / 4.2e-04), against 0/108 in every pre-fix run. The gradient path is restored.

## 8. Checkpoint compatibility rules

* A checkpoint written by a gated run carries extra keys inside the ANN block
  (`...net.mlp.*` instead of `...net.net.*`, plus `nu_log`, `theta_log`). It loads ONLY into a
  pipeline built with `AUG_LRU=1` (same wrapper, matching keys). Loading it into an ungated build,
  or an ungated checkpoint into a gated build, fails on key mismatch, loudly.
* Therefore every diagnostic run on a gated checkpoint must itself be launched with `AUG_LRU=1`
  (the band artefact must be present for the rebuild; the drawn init is then overwritten by
  `load_state_dict`, so the draw does not need to match).
* The planted reference (`runs/cl_capability_planted_ann.pt`) is an ungated ANN state_dict. Do not
  key-remap it into the wrapper: the remapped model would be "planted MLP + our bypass", which is a
  different object from the recorded planted reference and would corrupt any comparison. Use
  `CL_SPEC_SKIP_PLANTED=1`; the planted numbers are on record in `runs/cl_aug_spectrum.json`.
* Unpickling a gated checkpoint requires `gantry_dynamic.model` to be importable (the class is
  resolved by module path), which every script in `closed-loop-controller/` already guarantees via
  its `sys.path` block.

## 9. Verification status, with artefacts

Proven, each against a tool result on disk:

| claim | evidence |
|-|-|
| rows 0-5 exactly zero at init; rows 6-7 equal `A_aa x_a` | scratchpad `verify_aug_lru.py` run log (session 2026-08-20); PASS lines |
| `A_aa` Jacobian state-independent, equals the drawn eigenvalue | same, spread `0.00e+00` over 6 points |
| untrained model bit-identical to baseline, gate on vs off | `verify_aug_lru_equality.py` run log: both `2.1866011034177349e-06 m` |
| gradient reaches `nu_log`/`theta_log` through the `x_a` chain | same session, `|g| = 5.2e-02` on a 2-step chain |
| checkpoints pickle and the training loop runs end to end | `runs/cl_train_smoke_auglru.json` (CL_SMOKE) |
| `rho(A_aa)` live at init and after training | `runs/cl_aug_spectrum_armF_epoch3.json` |
| training result (Arm F) | run-table row in `docs/gantry-augmentation-problem-log.md` section 12; log `cl_train_arm_f_auglru` output (killed at epoch 5, best checkpoint `SSE_Interconnect_MultipleShooting_FFaboQ_best.pth`) |
| band survives Telica-level noise | `runs/cl_residual_spectrum_noisy.json` |

Not verified / not done:

* Epochs 6-10 of Arm F (killed flat at the user's decision); the flat window error says more
  epochs would not help, which is inference.
* The noisy RETRAINING arm (noise-gate step 3): deliberately deferred, there is no gain to qualify.
* `AUG_LRU` combined with `ANN_REZERO_GATE`: untested.
* `nx_aug > 2` (multiple pairs): the code path exists (vectors, asserted even) but has never run.
* The Telica explicit-band route (`AUG_LRU_BAND`/`AUG_LRU_RHO`): mechanism implemented, never
  exercised on real data.

## 10. Known debt: every experiment-grade shim, named

1. **Env-var switching instead of `RunConfig` fields.** `AUG_LRU*` should be config fields with
   the config recorded in the run JSON; today the `[aug-lru]` print line in the log is the only
   in-run record of the band and draw.
2. **The `.net` property on the wrapper.** Exists only so `cl_aug_spectrum.py` and
   `apply_rezero_gate` keep working against `ann.net.net`. A clean net class makes it unnecessary.
3. **`CL_SPEC_SKIP_PLANTED`.** Papering over the planted reference being structurally
   incompatible with gated builds; acceptable while the planted model is a diagnostic-only object.
4. **The refuse-to-start guard in `cl_train.py`** duplicates knowledge of what the framework
   currently supports; it must be removed in the same change that re-applies the burn-in patch, or
   it will block a legitimate run.
5. **Band derivation lives in `model.py`.** Reading a diagnostic artefact inside the model builder
   couples the builder to a script's output format. Clean version: a small module owning the
   recipe, with the builder taking numbers.
6. **`lr` pairing is convention, not code.** The fix needs `lr = 1e-5` (readout growth budget,
   D-150); nothing enforces it, only the run table and this document.

## 11. The clean implementation, deferred, and what blocks it

Target shape, agreed 2026-08-20:

* A net class `lru_bypass_feed_forward_nn` (name to taste) in
  `model_augmentation/utils/torch_nets.py` with the `@added` marker, taking the band/draw as
  constructor arguments, passed to `Static_ANN_Block` via its existing `net=` parameter exactly
  like `zero_init_feed_forward_nn`. No wrapper, no property, no env vars in the builder.
* `RunConfig` fields for enable/band/rho/artifact-path, serialised into `config_json_dict` so
  every run records them.
* The band recipe as its own function next to the config, unit-tested against the recorded
  simulation artefact values (`[149.90, 164.06] Hz`, `[0.9794, 0.9956]`).

Blocked on two things, both user decisions: (1) whether this parameterisation survives the next
arm (burn-in on live dynamics); rewriting before that risks rework; (2) the commit question:
`gantry_dynamic/config.py` is frozen while it carries another session's uncommitted P1/P1-e work,
and the clean version must edit it.
