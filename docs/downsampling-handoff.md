# Downsampling: context for the next session

> **RESOLVED 2026-07-08 (D-099).** The scoping diagnostic below was run
> (`scripts/gantry/augmentation-error/diag_downsample_spectra.py`) and settles the topic
> **for the simulated dataset**: `y`, `x_logical`, `delta_a`, AND `u_total` are all band-limited
> far below the 2 kHz Nyquist (worst-case power above 2 kHz: `y`/states 2.5e-8, `u_total` ~4e-14 =
> machine floor; PSDs flatten to the solver noise floor by ~500 Hz). So there is **no aliasing to
> fix** and no `resample_poly` is added to the sim pipeline. "Point sampling is exact" is verified,
> not assumed. The block-mean `u` (D-087) is KEPT, but note its benefit is DC/area-consistency on the
> K=0 integrator axes, **not** anti-aliasing (u has no HF energy). The premise below that "the ZOH
> controller puts step-harmonic energy above 2 kHz in u" is **false for these records** — the
> controller force is smooth. **The anti-alias concern is real only for the real-data (Telica)
> pipeline** (noise/quantization/HF resonances); apply `resample_poly` there, not here.
> The historical scoping notes are retained below for the real-data work.

Purpose: hand off the downsampling topic. Supervisor guidance (2026-07): downsampling
must use a proper anti-aliasing **low-pass filter**, from a **standard library** (not an
ad-hoc scheme). This challenges the current implementation. Below is exactly what the code
does now, what a prior fix (D-087) changed and verified, and the specific tensions to
resolve. No code has been changed for this topic yet.

## The pipeline in one line
Truth data is a Simulink sim logged at **FS_ORIG = 20 kHz**. Training runs at
**FS_NEW = 4 kHz** (decimation factor **D = 5**). So every record is downsampled 20 kHz -> 4 kHz
in Python before training. Nyquist at 4 kHz is 2 kHz. Physical content of interest is < ~200 Hz
(multisine 130-180 Hz, hidden absorber at 150 Hz), but the 20 kHz **ZOH controller** puts
step-harmonic energy far above 2 kHz in the force `u` -- that is what makes anti-aliasing matter.

## What the current code does (`scripts/gantry/gantry_dynamic/data.py`)
- **Force `u` (u_total): block mean** per hold interval, in `_resample_u` (data.py:78):
  `u[:n*D].reshape(n, D, nu).mean(axis=1)`. Not a standard-library call. It IS a low-pass
  filter (a length-D boxcar / moving average), but a weak one (sinc frequency response, first
  null at FS_NEW, heavy passband droop, poor stopband).
- **Output `y`, states `x_logical`, `delta_a`: point-sampled `[::D]`** with NO filter at all
  (data.py:101, 119-121). The inline comment claims "point sampling is exact (D-087)" -- this
  claim is the crux of the tension (see below).
- **`vdelta_a`: backward finite difference** (data.py:122-125).
- Config: `RunConfig.fs_new` / `.d` / `.ts_new` in `config.py`; D-087 note in `docs/decisions.md`.

## What D-087 fixed and verified (do not lose this)
Original code point-sampled u as `u_total[::D]`. That caused a real, diagnosed failure: on the
K=0 (springless) X/Y axes, the point-sample left a **nonzero-mean force error** that integrated
into a permanent open-loop position offset (tau*dv, tau = m/c ~ 1-1.5 s). V1 open-loop settled at
**Y -3.5e-4 m, X +6e-5 m**. Switching u to **block mean collapsed this to ~3e-9 / 3e-8 m**
(verified: `scripts/gantry/augmentation-error/diag_openloop_x0.py`, `diag_onestep_residual.py`;
data in `simulations/gantry_subnet/diagnostics/`). The argument used: `u_total` is a 20 kHz ZOH
signal, so the impulse-equivalent 4 kHz input is the **mean force over each hold interval** =
block mean. This is physically motivated and it demonstrably removed the drift.

So D-087 is correct as far as it goes (drift gone, training runs). The supervisor's point is a
**separate, more general** signal-processing concern that D-087 did not address.

## The tensions to resolve next session
1. **`y` and states have NO anti-aliasing.** They are decimated by bare `[::D]`. Any spectral
   content above 2 kHz folds back into the band of interest. The "point sampling is exact" comment
   is only true if those signals have no energy above the new Nyquist -- UNVERIFIED. First action:
   plot the 20 kHz spectra of `y`, `x_logical`, and `u` for a couple of records and see how much
   sits above 2 kHz. That decides whether this is a real problem or a non-issue for these signals.
   (The plant low-passes the response, so `y` may be clean; `u` almost certainly is not.)
2. **Block mean vs a proper standard-library filter for `u`.** Block mean satisfies "use a
   low-pass filter" only weakly. The supervisor wants a standard library. Candidates:
   - `scipy.signal.decimate(x, D, ftype='fir', zero_phase=True)` -- FIR anti-alias, linear phase.
   - `scipy.signal.resample_poly(x, 1, D)` -- polyphase FIR, linear phase; recommended when u and
     y must stay time-aligned.
   Open question: for a ZOH-held `u` fed to a ZOH model, is the block mean (impulse-exact) actually
   BETTER than a generic anti-alias filter, or should we adopt the standard filter for consistency
   and accept it is near-equivalent? Compare both against the drift diagnostic.
3. **Phase alignment between `u` and `y` after filtering.** Any anti-alias filter has group delay.
   `u` (input) and `y` (target) MUST be filtered with the SAME zero-phase / linear-phase filter so
   they are not shifted relative to each other -- a relative shift would create a fake model error.
   `resample_poly` (linear phase, same filter on both) or `filtfilt`+decimate (zero phase) handle
   this; block-mean-u + point-sample-y currently do NOT treat them the same way.
4. **States as initial conditions.** `x_logical` is used to seed open-loop sims (D-072/D-087
   interior-K0). If states are anti-alias filtered, the velocity components (already FD-derived and
   fragile at the boundaries) change; re-check the interior-K0 seeding still holds.

## What MUST be re-verified after any change
- The **open-loop drift fix** must survive: re-run `diag_openloop_x0.py` and confirm the settled
  offset stays ~1e-8 m (whatever new scheme replaces block mean must not reintroduce the K=0 drift).
- The **oracle verification** (`diag_oracle_vs_data.py`) still reproduces the data (it also resamples
  u via block mean internally -- keep it consistent with whatever data.py adopts).
- A fresh **training run** (numbers shift; prior runs not comparable).

## Files and anchors
- Downsampling code: `scripts/gantry/gantry_dynamic/data.py` (`_resample_u`, `load_traj`, `load_mat_aug`).
- Config: `scripts/gantry/gantry_dynamic/config.py` (`fs_orig=20000`, `fs_new=4000`, `d`, `ts_new`).
- Decision: `docs/decisions.md` D-087 (block-mean u + interior-K0 init).
- Diagnostics: `scripts/gantry/augmentation-error/diag_openloop_x0.py`, `diag_onestep_residual.py`,
  `diag_oracle_vs_data.py`; outputs in `simulations/gantry_subnet/diagnostics/`.
- Also resample in the diagnostics + `gantry_dynamic/oracle.py` -- keep consistent.

## Suggested opening move for the next session
1. Spectra of `u`, `y`, `x_logical` at 20 kHz (energy above 2 kHz?) -> decides scope.
2. Pick a standard resampler (`scipy.signal.resample_poly` is the likely choice: linear phase,
   same filter for u and y, keeps alignment).
3. Decide `u`: keep impulse-exact block mean (justify to supervisor) OR switch to the standard
   filter (consistency). Compare both on the drift diagnostic before committing.
4. Apply the SAME resampler to `y`/states, mind phase alignment and the FD velocities.
5. Re-verify drift + oracle, then retrain.
