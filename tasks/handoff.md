# Session Handoff

**Last written**: 2026-06-25

---

## Open blocker: Encoder diagnostic (encoder track)

Encoder diagnostic script (`scripts/gantry/encoder/diagnostic_nf_lr.py`) has a Stage 1 bug -- uses `apply_experiment()` instead of direct encoder forward pass. Full context in `archive/sessions/2026-06-23-handoff.md`.

---

## ANN routing: RESOLVED -- output augmentation implemented

### Resolution (D-065, 2026-06-25)

`gantry_interconnect_dynamic.py` now uses output augmentation:
- `y = Cd@x_phys + C_aug@x_aug`
- `out_block = Parameterized_Linear_Output_Block(C=[Cd_norm|C_aug_init], flag_loss_reg=False)`
- C_aug_init: `C_aug_init[2,0] = 1e-2` (Y <- delta_a), C_aug is trainable
- `selection_matrix(np.arange(nxd), nxd)` passes full state to output block

This resolves both constraints:
- Constraint 1 (stability): gradient path never passes through A_phys integrators
- Constraint 2 (gradient): ANN gets nonzero gradient via C_aug -> x_aug -> ANN

Verified by diag15: ANN grad = 3.5e-4 (vs 0), val ratio = 1.03x at nf=400 (vs 800-1634x).

### Diagnostic scripts written this session

| Script | Purpose |
|--------|---------|
| `diag13_routing_isolation.py` | T_vel/T_pos/T_clip -- confirmed all physical row routings blow up |
| `diag14_eigenvalues.py` | Proved gantry |z|=1 exactly; Jan's MSD min(1-|z|)=4.4e-3; amp=400x vs 4.4x |
| `diag15_output_aug_diagnostic.py` | Verified output aug: gradient path + no blowup at nf=400 |

### Ready for full training run

`gantry_interconnect_dynamic.py` is ready. Run with default hyperparameters (NX_ANN=2, nf=400, lr=1e-4, epochs=10). Monitor:
1. Does val loss decrease over epochs?
2. Does C_aug magnitude grow from 1e-2 (indicates ANN learning absorber dynamics)?
3. At end: aug state R2_linmap vs delta_a/vdelta_a (diag function `aug_state_r2` built in)

Note on diag15 T3 (+14% val over 5 steps on 1 trajectory): this is encoder overfitting to a single trajectory, not instability. Full training on 8 trajectories required to assess real convergence.
