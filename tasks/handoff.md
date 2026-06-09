# Session Handoff — Gantry SUBNET Encoder Verification

**Last written**: 2026-06-07

---

## Current status

The gantry dynamic parallel augmentation pipeline is built and runs end-to-end. Training produces results but performance is poor (NRMS X1=0.97, X2=0.90, Y=0.30 on validation). We are currently **verifying the encoder** before investing in long training runs or hyperparameter search.

---

## What to do next

Run the encoder diagnostic script and interpret the results:

```
conda run -n GraduationProject python scripts/gantry/verification/diagnose_encoder.py --epochs 10
```

This script builds a fresh model, checks gradient flow, does a short training, then compares encoder-initialised rollouts against the analytical baseline (positions from measurements + finite-diff velocities, ANN states=0). The encoder must outperform this baseline for the augmentation to add value.

Based on the results:
- If gradients don't reach the encoder: wiring bug
- If encoder weights don't change: optimizer or loss issue
- If ALL rollouts are bad (encoder AND analytical): dynamics model problem, not encoder
- If analytical beats encoder: encoder needs more training or different hyperparameters

---

## Key files

| What | Where |
|------|-------|
| **Training script** | `scripts/gantry/gantry_interconnect_dynamic.py` |
| **Encoder diagnostic** | `scripts/gantry/verification/diagnose_encoder.py` |
| **Physics block** | `model_augmentation/fit_systems/blocks.py` (`Gantry_State_Block`) |
| **Output block** | `model_augmentation/fit_systems/blocks.py` (`Linear_Output_Block`) |
| **SSE_Interconnect** | `model_augmentation/fit_systems/interconnect.py` |
| **Gantry system matrices** | `model_augmentation/systems/gantry_ss.py` |
| **Jan's MSD reference** | `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` |
| **Data statistics** | `scripts/gantry/verification/_check_ystd.py` |

---

## Training script structure (`gantry_interconnect_dynamic.py`)

- `USE_OPTUNA` flag toggles between single run and Optuna Bayesian search
- `DEFAULT_HP`: NX_ANN=3, nodes=128, layers=3, nf=350, batch=4000, lr=7.6e-4, epochs=100
- `build_and_train(hp)`: builds Interconnect + blocks + SSE_Interconnect, sets manual normalization, calls `fit()`
- `evaluate_and_save(fit_sys, hp, rid)`: loads best checkpoint, runs `apply_experiment` on val_data, computes NRMS, plots, saves
- Data: 8 training trajectories (T1-T8), 1 val (V1), 1 test (E1), decimated 20kHz to 1kHz
- Normalization is manual (`auto_fit_norm=False`): `ystd`, `y0`, `std_x`, `x_mean` computed from training data at module level
- Seed reset before `build_and_train` in standalone path (was missing before, now fixed)

---

## Model architecture

- **NX_PHYS=6** physical states: [X1, X2, Y, dX1, dX2, dY] (stage coordinates, not logical)
- **NX_ANN=3** augmentation states (no physical meaning, learned)
- **nxd=9** total state dimension
- **Encoder**: `modified_encoder_net` (simple_res_net), input = flattened [u_past(na x 3), y_past(nb x 3)], output = x0(9)
- **na = nb = 2*nxd+1 = 19** history window
- **Physics block**: `Gantry_State_Block` with LPV-LFR, RK4 integration (10 substeps), Y_OP=None (self-scheduled)
- **ANN block**: `Static_ANN_Block`, zero-initialized, input=[x(9), u(3)], output=xp correction(9)
- **Output block**: `Linear_Output_Block` with Cd_norm (normalized C matrix), Dd=0

---

## Known issues and fixes applied

1. **`Linear_Output_Block` GPU fix**: C, D changed from plain attributes to `register_buffer()` so `.to(device)` propagates
2. **Seed sensitivity**: same hyperparameters give wildly different results with different seeds. Optuna Trial 7 (seed=49) reached bestfit=0.024, standalone (consumed seed) got 0.062
3. **Training instability**: nf=350 + lr=7.6e-4 causes validation loss to oscillate (0.06 to 0.63). May need shorter nf or lower lr.
4. **Normalization convention**: x_mean and std_x are computed from stage-coordinate positions/velocities [X1, X2, Y, dX1, dX2, dY], but `Gantry_State_Block.deriv()` treats the denormalized state as logical coordinates [q1, q2, q3, dq1, dq2, dq3]. This is a potential coordinate mismatch that has not been fully investigated.

---

## Optuna results (partial, 4/40 trials completed before SLURM timeout)

See `throwaway_optuna_output.md` for raw output. Best trial: NX_ANN=3, nodes=128, layers=3, nf=350, batch=4000, lr=7.6e-4 (bestfit 0.024). SQLite DB at `simulations/gantry_subnet/optuna_gantry_subnet_dynamic.db`.

---

## Open blockers

- Encoder verification (in progress)
- Coordinate mismatch between normalization (stage) and physics block (logical) needs investigation
- No GPU training yet (`fit()` doesn't pass `cuda=True`; server has RTX 6000)
