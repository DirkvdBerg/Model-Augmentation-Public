# Augmentation-no-controller

Copy of `Matlab-scripts/Augmentation/` taken 2026-08-10, to develop an **open-loop** variant of the
trajectory data generation. Everything except `Matlab-output/` was copied verbatim; nothing here has
been modified yet.

**Do not edit `Matlab-scripts/Augmentation/`.** Every number in the MSD-offset investigation and in
`scripts/gantry/msd-offset/` is conditioned on the records that folder produces.

## Why this folder exists

Supervisor meeting 2026-08-10, the note `trainingsdata in open-loop run`.

The current records are generated **closed loop**:

```
u_total = Cfb*(r - q) + f_multisine
```

so `u_total` contains the controller's response to tracking error. That is where the low-frequency
content comes from: measured on `T10_aprbs_60`, **92.5 %** of the input power sits below 20 Hz, and
the input carries a DC component of `[+0.66, -0.46, +0.10] N`.

Those records are then replayed **open loop** through a model with no controller. The loop structure
of the data and of the evaluation do not match, which is the root of the supervisor's repeated
question `waarom trekt de controller het niet naar 0`: in the real loop the controller absorbs the
low-frequency component, and in the replay nothing does.

Generating the data open loop removes that mismatch at the source, and removes the tracking
trajectory with it, so the output stops being dominated by gross motion.

## The problem to solve before this can work

The gantry has `K11 = K33 = 0`: X and Y have viscous damping but **no stiffness**. Drive them open
loop and the position integrates away. Any DC component in the applied force gives a constant
velocity, and even zero-mean noise gives a random walk. You cannot simply apply forces and record
12 s.

This is what the supervisor's other note is about:

```
de hele tijd een force aan het applyen was het oplopen ... zonder demping
zou frictie toe kunnen voegen
```

Three candidate ways to bound it, none chosen yet:

1. **Zero-mean forcing with a bounded integral.** A multisine has this by construction; an APRBS
   does not. Cheapest option and changes no physics.
2. **Coulomb friction**, which gives the axis a resting position. Verified implementation and
   parameter provenance already exist in `scripts/gantry/real-data-verification/coulomb_lfr.py` and
   `COULOMB_HANDOFF.md`. Changes the plant, so it changes the benchmark.
3. **Short records**, accepting the drift and keeping it inside an acceptable band.

Note for the record: the supervisor's `x en y geen damping` is not correct. `cg1+cg2 = 34.8 Ns/m`
and `cy = 10.0 Ns/m` both exist, and they are what set `tau_X = 1.55 s` and `tau_Y = 1.01 s`. What
X and Y lack is stiffness. Damping limits velocity, stiffness holds position, and under a constant
force only the second gives a resting point. So of `frictie` and `demping`, only friction does what
is wanted here.

## Cross-check this folder must pass

Once an open-loop record exists, replay its input through the Python model and compare against the
MATLAB output. The closed-loop equivalent already agrees to about `1e-7 m` (the `FULL` arm of
`simulations/gantry_subnet/diagnostics/msd_offset_plant_ablation.json`). Open loop has no controller
in either path, so it isolates solver and sampling differences: Simulink's variable step against the
Python fixed-step RK4 at 4 kHz, the block mean on `u`, and the point sampling of `y`. If the
open-loop agreement is much tighter than `1e-7 m`, that floor was the loop rather than the solver,
which matters because `1e-7 m` is currently quoted as a noise floor.

## Files

Verbatim copies of `Matlab-scripts/Augmentation/` as of 2026-08-10:

| what | note |
|-|-|
| `data/` | the generation pipeline; `generate_trajectory_data.m` has the `TRACK` switch, `'joint'` = broadband [1,200] Hz, `'augmentation'` = narrowband [130,180] Hz |
| `gantry_additional_state_2025a.slx` | the 8-state Simulink model, contains the closed loop to be removed |
| `gantrySystemExtended.m`, `gantrySystemExtendedMFile.m` | the 8-state EOM |
| `diagnostics/`, `main_augmentation.m`, `additional_state_lagrangian.m`, `generate_frozen_y_mimo_frf_pretest_augmented.m` | carried along, not yet reviewed for relevance |

`Matlab-output/` was deliberately not copied; it is 9.6 MB of generated output, not source.

Output records must go to a **new** folder under `data/gantry/matlab/trajectory/`, never overwriting
`augmentation/` or `joint/`.
