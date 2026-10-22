# Comments and naming pass over `Clean/`: plan (phase 1)

2026-10-21. Handoff `tasks/handoffs/2026-10-21-clean-code-comments-naming.md`. Nothing in `Clean/` is edited
yet. Approve, edit or reject each row of (a) and each item of (b); the two samples in (c) set the style for
the other 22 files.

Every proposed edit is a rename, a comment, a docstring or an error message, except two code lines marked as
your call ((b) item 4 and the `gantry_ss.py` docstring). A static check shows it for the samples: with comments and docstrings removed, strings set aside and the renames applied, both have the
syntax tree of `Clean/` (end of (c)). Phase 3 reruns layers 1 to 3c anyway, since H2 refuses outputs of an
older tree.

## (a) Renames

Checked for every row: the name is ours (`@added`, a file with `__project_origin__`, or a local of ours;
Jan's names checked against his last own commit `c9f913a`); no script in `Thesis-writeup/Code/Evaluation/`
imports or reads it (grep of the imports and the attribute reads); no `state_dict` key, `config.json` key or
CLI flag changes (class, method and local names are not in keys, and no attribute that holds a module is
renamed). Pickled class paths: clean dumps are regenerated in phase 3, and the clean code never resumes a
mainline dump (design D3: own `_best.pth` only; mainline runs reach it as component pairs).

| # | Now (file) | New | Why | Edits it forces |
|-|-|-|-|-|
| R1 | `SSE_Interconnect_Composed` (interconnect.py) | `SSE_Interconnect_Weighted` | "Composed" names penalty terms the class no longer has; what it adds to Jan's class is the weighted objective, W_r = diag(sigma_y / sigma_e), with the OBC coefficient per objective | model.py import and call; EP `_dumps` pattern (deepSI names a dump after the class: `SSE_Interconnect_Weighted_<id>_best.pth`); harness `gpu_smoke.py:50-61` (import and loss patch per side), `run_l3bc.py:127-132, 231, 243` (dump names per side), `compare_l3bc.py:109` (`resumed_from` is compared by file name: map the class prefix) |
| R2 | `OBCLifecycle` (obc.py) | `OBCProjector` | says its job: theta_aux of the network on the basis, and the basis refresh | obc_gantry.py import, `attach_obc`, `make_basis_builder` docstring; the local `life` becomes `projector`. The attribute `fit_sys.obc` stays (the harness reads it) |
| R3 | `linear_encoder_init_aug` (pre_encoder.py) | `AugmentedEncoder` | the trained encoder psi = [W^b; W^a] chi + psi_tilde(chi) of the physical and added states; "init" reads as an initialiser | model.py import and call |
| R4 | `live_g_feed_forward_nn` (torch_nets.py) | `AugmentationNet` | the network phi_aug; "live_g" names a removed option. The `config.json` values `G_INIT='live'`, `ANN_ARCH='mlp'` stay | model.py import and call |
| R5 | `ValHistory` (training.py) | `ValidationHistory` | spelled out | `train` |
| R6 | `_sync_prediction_state_for_validation` (method of R1) | `_sync_obc_coefficient` | it pairs theta_frozen with the current network before the encoder start, in PS2 and in the polish too, not only for validation | 6 calls: interconnect.py, model.py, lbfgs_polish.py, ps2_opening.py |
| R7 | `_deriv_both_with` (blocks.py) | `_deriv_with_tangent` | matches `_rk4_with_tangent`, its only caller | 4 calls in blocks.py |
| R8 | `_REGISTRY` (controller.py) | `_STORED_CONTROLLERS` | says what it holds | 3 uses |
| R9 | `_tf_to_ss_batch` (controller.py) | `_tf_to_block_diag_ss` | it builds one block-diagonal system, not a batch | 1 use |
| R10 | `build_model` locals `NX_ANN`, `nxd`, `_names`, `_nominal`, `_combo` | `n_a`, `nx`, `raw_names`, `raw_nominal`, `combo_start` | the config and thesis names (nx = 6 + n_a); `_combo` is the start value of the combinations. The `hp` key `'NX_ANN'` stays | in the function |
| R11 | underscore locals of ours: config `_signs`; training `_opt`, `_opt_sd`, `_dropped`, `_adam`, `_last`, `_polished`, `_no_adam_ran`, `_drop_optimizer`, `_why`; closed_loop `_p`; interconnect `_model_dtype`, `_selected_checkpoint`, `_corr`, `_theta`; pre_encoder `_ny_hist`, `_nu_hist`; EP `_f` | the name without the underscore; `_adam` becomes `adam_rows`, `_last` `last_by_event`, `_no_adam_ran` `polish_only` (it is `start_phase == 'lbfgs'`, while `adam_ran` beside it means something else), `_f` `fh` | an underscore on a local means nothing | in the functions. Kept where the plain name is taken: blocks.py `_alpha` to `_A_combined` (the buffer names), `_C`, `_D` (the arguments) |
| R12 | `ps2_opening.py` `_fit_map` local `ee`; `closed_loop.py` `__call__` local `use_c` | `s2_cost`, `use_compiled` | say what they are | in the functions |

Considered and kept:
- `hp`: the `.npz` field `hp` and `cfg.hp` carry it, and the evaluation calls `build_model(cfg.hp, cfg, data,
  norm)`. Its content duplicates `cfg`; reading from `cfg` instead is a code change, not a rename.
- `RunConfig` names (`nx_ann`, `combo_init_detune`, the ClassVar constants): they are the `config.json` keys
  in lower case, and the evaluation reads `nx_ann`, `ann_route_ix`, `controller_gain` and others.
- Read by the evaluation: `DataBundle.val_ckpt_data` and the other fields, the `Norm` fields, the module
  `closed_loop_report` with `_without_learned`, `_true_free`, the data helpers `_load_u`, `_resample_u`,
  `_record_at_fs_new`, `_load_record`.
- Read or patched by the harness, so a rename costs a harness edit for little clarity: `_ann_ix`, `_mats_ix`,
  `obc_reference`, `epochs_before`, `ps2_opening`, the simulator fields (`record_weights`, `val_rows`,
  `train_ctrl_rows`, `last_per_channel`, `last_sim_rms`, `val_sigma_e`), `least_squares_start`,
  `prefit_physical_encoder`.
- Module attributes (`state_dict` keys) or evaluation reads: `obc_correction`, `Wb_psi_*`, `Wa_psi_*`,
  `free_params`, `theta_frozen`, `vbar`, `out_gate`, `Y_op`, `N_op`, `d_op`.
- Jan's names: `Interconnect`, `SSE_Interconnect`, `Signal_Connection`, `Block`, `Static_ANN_Block`,
  `Discrete_Nonlinear_Function_Block`, `Linear_Output_Block`, `to_tensor`, `expansion_matrix`,
  `selection_matrix` (mainline rewrote `Linear_Output_Block` and `to_tensor` and marks them `@added`; the
  names are Jan's).
- `fixed_System_data`, `fixed_System_data_norm`: named after the deepSI classes they replace.
- `PS2Opening`, `PS2Spec` and its fields: the thesis term; the fields are written into `ps2_summary`.

Strings that stay byte for byte, because they are data in compared outputs or the harness matches them: the
polish `reason` texts (`lbfgs_outcome`), the PS2 status and summary, the validation-history event names, the
`config.json` values (`y_decimation`), "training rollout COMPILED" (`compare_gpu.py:62`), and "PS2" in the
refusal of a dump written inside an unfinished opening (`run_l3bc.py:202`).

## (b) Markers

Facts: of the nine files in `Clean/model_augmentation/`, five are Jan's (`fit_systems/interconnect.py`,
`fit_systems/blocks.py`, `fit_systems/pre_encoder.py`, `utils/torch_nets.py`, `utils/utils.py`); the others
carry `__project_origin__ = "added"`. Today `Clean/` has 26 `# CHANGED (clean): trimmed to the spec path`
lines and 61 other `# CHANGED` lines: 12 in Jan's definitions, 3 on imports in Jan's files, 12 in methods we
added to Jan's classes and 34 in our own files or `@added` definitions.

1. `@added` and `__project_origin__ = "added"` stay as they are.
2. The 26 trim lines go. Each of Jan's five files starts with one line instead, e.g.
   `# CHANGED: trimmed to the thesis runs; the untrimmed file is model_augmentation/fit_systems/interconnect.py
   in the repository root`, followed by a one-line module docstring.
3. `# CHANGED:` stays only inside Jan's definitions where the code behaves differently from his, with a short
   reason about the current code (about 14 lines, all in interconnect.py and blocks.py), e.g.:
   - `self.obc_correction = None   # CHANGED: the OBC correction f_obc, set by attach_obc, or None`
   - forward signature: `# CHANGED: obc_pass and pass_mats, built once per objective, are arguments, so
     torch.compile sees them as graph inputs`
   - `xp = xp - self.obc_correction(...)`: `# CHANGED: subtract f_obc`
   - `int(element)`: `# CHANGED: Python ints, so torch.compile branches on them without a graph break`
   - `output_cone`, `_pass_mats_ix`, the device and dtype of the connection matrices, the forward through
     `_propagate`, the three class attributes of `SSE_Interconnect`, its `norm`; in blocks.py `net_kwargs`,
     `out_gate` (`# CHANGED: a gate per output, 1 in training; _without_learned zeroes it`) and the `mats`
     argument of `Discrete_Nonlinear_Function_Block.forward`.
4. Methods we added to Jan's classes get `@added` instead of a `# CHANGED (...): added` comment or nothing:
   `Interconnect._apply`, `_signal_cone`, `output_only` (`_propagate` has it already) and
   `SSE_Interconnect.simulate`, `make_training_data`, `cal_validation_error`, `fit` (none of the four is in
   `c9f913a`). The decorator returns its argument, so nothing runs differently, but it is a code line: your
   call. The alternative is one `# CHANGED: added` comment per method.
5. In our own files, `@added` definitions and the methods of item 4, `# CHANGED` lines become plain comments
   under the standard, or go. The 3 on imports go.
6. `# THEORY:` / `# HEURISTIC:` labels (23 and 16) all stay, in the plain style: `# THEORY: <formula>
   (<source>)`, `# HEURISTIC: <short reason>`. The source is a paper or a file name (the MATLAB scripts,
   without line numbers); no D-number, job id or thesis label. A label keeps its source even where the paper
   is named nearby (repository rule). Long labels are cut to the formula and its reason, e.g. `_resample_u`.
7. Optional: one row in the CLAUDE.md marker table for the trimmed-file header line of item 2. Your call.

## (c) Style sample

The samples apply the standard of handoff section 3. Choices they fix where the standard leaves one:
- Thesis symbols in plain text: x_tilde, x_bar, n_a, beta (code `free`), beta_lin (`vbar`), theta_aux,
  J = Phi_Gamma, F, psi = [W^b; W^a] chi + psi_tilde(chi), phi_aug, W_r = diag(sigma_y / sigma_e).
- D-numbers, job ids and thesis labels leave prints and error messages too, so the phase-3 grep covers all of
  `Clean/`, not only comments. Error messages become one sentence; the strings listed at the end of (a) stay.
- Paper names: the four of section 3, and Kessels (the known controller closed around the model in the
  rollout) once in the module docstring of `closed_loop.py`, replacing its four equation references: your
  call.
- Docstrings on what is called from outside the module; a private helper gets one line only where the name
  does not say its job (`_prefit`). Shapes only where they are not obvious.

Check (script in the session scratchpad; it can go into the harness folder if you want it kept): with
docstrings and comments removed and the renames applied, both samples have the syntax tree of `Clean/`.
Changed strings: obc.py 2 (the two rank-loss messages), model.py 2 (the CUDA and Adam eps messages), all on
error paths. In phase 2 every file passes the same check before the harness rerun.

## Phases 2 and 3, after approval

- First D-282 in `docs/decisions.md`: the approved renames and marker rule.
- The renames with the edits they force, then the comments file by file; per file pyflakes, the import check
  and the syntax-tree check. At most one general-purpose subagent, for the mechanical rename edits (handoff
  section 14); the comments are written in this session.
- Also fixed in phase 2, the stale texts already listed: the `init_model` docstring, the `fit` docstring
  ("parallel validation"), the encoder's dtype comment (it names the removed reconstructability map), the
  controller.py THEORY label naming `TRAIN_FILES` and the KeyError naming `RECORD_Y_OP`, the CHANGED comment
  naming concurrent validation, the seam called "diagnostics", the R1 docstring ("composed penalty terms"),
  the double hyphen used as a dash in the `_resample_u` docstring (data.py:90).
- `gantry_ss.py` has its docstring after `__project_origin__`, so it is not the module docstring. Putting it
  first is the one statement-order change of the pass (only `__doc__` changes): your call.
- Phase 3 as in the handoff. The current `_clean/` moves to `_clean-D281/` first, so every clean output is
  made from the new tree.

## Your decisions

R1 to R12; (b) items 2, 3, 4 and 7; Kessels in `closed_loop.py`; the docstring order of `gantry_ss.py`.

## The two samples as diffs against `Clean/`

### `Clean/model_augmentation/fit_systems/obc.py`

```diff
@@ -1,4 +1,5 @@
-"""One-step orthogonal-by-construction (OBC) augmentation: the GENERIC part (D-192)."""
+"""Orthogonal projection of the network output on the baseline Jacobian (Gyorok et al.), the part that
+does not depend on the model: the basis, its refresh every epoch and the coefficient theta_aux."""
 __project_origin__ = "added"
 
 from typing import Callable, Optional, Sequence
@@ -8,24 +9,23 @@
 
 
 class OBCRankLossAbort(RuntimeError):
-    """Raised when the reference basis lost rank at two consecutive refreshes (or at the first
-    build, when there is no previous valid basis to retain)."""
+    """The stacked Jacobian lost column rank at two refreshes in a row, or at the first build."""
 
 
 class OBCBasis:
-    """Economy SVD of the stacked baseline sensitivity, frozen for one epoch."""
+    """Economy SVD of the stacked Jacobian J = Phi_Gamma, fixed for one epoch."""
 
     def __init__(self, J: Tensor, vbar: Tensor):
-        """Factorise a float64 `J (n_rows, n_cols)` with its columns scaled to unit norm, `no_grad`
-        throughout. `vbar` is the detached float64 expansion point of `J`."""
+        """J (n_rows, n_cols) in float64, factorised with its columns scaled to unit norm; vbar = beta_lin,
+        the detached float64 point J is taken at."""
         with torch.no_grad():
             n_rows, n_cols = J.shape
             G_raw = J.T @ J
             norms_raw = G_raw.diagonal().clamp_min(0).sqrt()
             column_scale = torch.where(norms_raw > 0, norms_raw, torch.ones_like(norms_raw))
             U, S, Vh = torch.linalg.svd(J / column_scale, full_matrices=False)
-            # THEORY: numpy.linalg.matrix_rank default, S.max() * max(M, N) * eps(dtype)
-            # (Golub and Van Loan, Matrix Computations, 4th ed., sect. 5.4.1 numerical rank).
+            # THEORY: numerical rank, tol = S_max max(M, N) eps (Golub and Van Loan, Matrix Computations,
+            # 4th ed., 5.4.1; the numpy.linalg.matrix_rank default)
             tol = float(S[0]) * (max(n_rows, n_cols) * torch.finfo(S.dtype).eps)
             rank = int((S > tol).sum())
             if rank < n_cols:
@@ -36,10 +36,10 @@
         self.n_cols, self.rank, self.tol = n_cols, rank, tol
 
     def coefficient(self, F: Tensor) -> Tensor:
-        """`theta = J^+ F = V S^-1 U^T F`, float64, differentiable in `F` (any float dtype)."""
+        """theta_aux = (J^T J)^-1 J^T F = V S^-1 U^T F, in float64 and differentiable in F."""
         F64 = F.to(torch.float64)
         theta = self.Vh.T @ ((self.U.T @ F64) / self.S)
-        return theta / self.column_scale
+        return theta / self.column_scale        # undo the column scaling of J
 
     def to(self, device) -> 'OBCBasis':
         self.U, self.S, self.Vh = self.U.to(device), self.S.to(device), self.Vh.to(device)
@@ -55,7 +55,8 @@
 def build_stacked_sensitivity(step: Callable[[Tensor, Tensor], Tensor], vbar: Tensor, Z: Tensor,
                               row_ix: Sequence[int], chunk: int = 16384,
                               device=None) -> Tensor:
-    """`J (N * n_rows_sel, p)` in float64: `d step(v, z_k) / dv` at `vbar`, stacked sample-major."""
+    """Phi = d step(v, z_i) / dv at v = vbar on the rows row_ix, stacked sample-major, float64,
+    shape (N len(row_ix), p)."""
     if vbar.dtype != torch.float64:
         raise TypeError('build in float64: vbar is %s' % vbar.dtype)
     dev = device if device is not None else vbar.device
@@ -73,7 +74,7 @@
 
 def evaluate_step_rows(step: Callable[[Tensor, Tensor], Tensor], vbar: Tensor, Z: Tensor,
                        row_ix: Sequence[int], chunk: int, device) -> Tensor:
-    """`step(vbar, z_k)` restricted to `row_ix`, stacked sample-major (float64, no grad)."""
+    """step(vbar, z_i) on the rows row_ix, stacked sample-major, float64 without gradient."""
     dev = device if device is not None else vbar.device
     row_ix = torch.as_tensor(list(row_ix), dtype=torch.long, device=dev)
     vbar = vbar.detach().to(dev)
@@ -85,12 +86,14 @@
     return torch.cat(out, 0)
 
 
-class OBCLifecycle:
-    """Once-per-objective coefficient and the epoch-refresh policy."""
+class OBCProjector:
+    """theta_aux of the current network on the basis, with the basis rebuilt once per epoch."""
 
     max_consecutive_rank_loss = 2
 
     def __init__(self, reference_field: Callable, basis_builder: Callable):
+        """reference_field: network block -> F, its stacked output on the reference tuples;
+        basis_builder: vbar -> (J, seconds)."""
         self.reference_field = reference_field
         self.basis_builder = basis_builder
 
@@ -99,7 +102,8 @@
         self.consecutive_rank_loss = 0
 
     def refresh(self, vbar: Tensor, epoch: Optional[int] = None) -> None:
-        """Rebuild the basis at a DETACHED copy of `vbar`; apply the rank-loss policy."""
+        """Rebuild the basis at a detached copy of vbar. A candidate without full column rank is dropped
+        and the previous basis kept; the second time in a row, or at the first build, this raises."""
         vbar64 = vbar.detach().to(torch.float64).clone()
         J, _ = self.basis_builder(vbar64)
         cand = OBCBasis(J, vbar64)
@@ -111,13 +115,11 @@
             msg = ('[obc] refresh at epoch %s: numerical rank %d < %d (tol %.3e). '
                    % (epoch, cand.rank, cand.n_cols, cand.tol))
             if self.basis is None:
-                raise OBCRankLossAbort(msg + 'No previous valid basis exists to retain; the '
-                                       'reference set does not separate the protected '
-                                       'directions at this parameter point.')
+                raise OBCRankLossAbort(msg + 'There is no previous basis to keep: the reference tuples do '
+                                       'not separate the training coordinates at this point.')
             if self.consecutive_rank_loss >= self.max_consecutive_rank_loss:
-                raise OBCRankLossAbort(msg + '%d consecutive refreshes lost rank; OBC training '
-                                       'readiness is aborted rather than continued on a '
-                                       'truncated basis.' % self.consecutive_rank_loss)
+                raise OBCRankLossAbort(msg + '%d refreshes in a row lost rank; training stops instead of '
+                                       'continuing on a truncated basis.' % self.consecutive_rank_loss)
         else:
             self.consecutive_rank_loss = 0
             self.basis = cand
@@ -129,8 +131,8 @@
             self.refresh(vbar_fn(), epoch)
 
     def coefficient(self, learning_component) -> Tensor:
-        """`theta_aux` for THIS objective evaluation: the current learning component on the
-        full reference set, solved in float64 with autograd."""
+        """theta_aux for this objective: the network on all reference tuples, solved in float64 with
+        autograd."""
         F = self.reference_field(learning_component)
         if F.device != self.basis.device:
             self.basis.to(F.device)
```

### `Clean/gantry_dynamic/model.py`

```diff
@@ -1,4 +1,4 @@
-"""Model construction (interconnect + encoder) and the least-squares encoder start."""
+"""Model construction (interconnect and encoder) and the least-squares start of the encoder."""
 __project_origin__ = "added"
 
 from contextlib import contextmanager
@@ -7,101 +7,94 @@
 import torch
 
 from model_augmentation.utils.utils import expansion_matrix, selection_matrix
-from model_augmentation.utils.torch_nets import live_g_feed_forward_nn
-from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect_Composed
+from model_augmentation.utils.torch_nets import AugmentationNet
+from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect_Weighted
 from model_augmentation.fit_systems.blocks import Reduced_Gantry_State_Block, Linear_Output_Block, Static_ANN_Block
-from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
+from model_augmentation.fit_systems.pre_encoder import AugmentedEncoder
 from model_augmentation.fit_systems.closed_loop import closed_loop_rollout
 from model_augmentation.systems import gantry_ss
 
 from .config import RunConfig
 from .obc_gantry import attach_obc, P_INV_T64
 
-PHY_IX   = np.arange(6)
+PHY_IX = np.arange(6)      # the physical states x_tilde = (q, qdot) come first in x
 
 
 def build_model(hp, cfg: RunConfig, data, norm):
-    """Build interconnect + SSE_Interconnect from hp dict, return fit_sys (untrained)."""
-    NX_ANN = hp['NX_ANN']
-    nxd = cfg.nx_phys + NX_ANN
+    """The untrained fit system, the augmentation structure of Hoekstra et al.:
+    x_tilde_{k+1} = f_base(x_tilde_k, u_k) + f_aug(x_k, u_k), x_bar_{k+1} = g_aug(x_k, u_k), y_k = C x_tilde_k,
+    with the encoder and, on the OBC arm, the projection."""
+    n_a = hp['NX_ANN']
+    nx = cfg.nx_phys + n_a
     route_ix = np.asarray(cfg.ann_route_ix)
 
-    # THEORY: Hoekstra 2026 Eq. (22e), x_k|k from y and u over k-na .. k-1 (D-250): na = nb
-    # past samples, the current one excluded.
+    # THEORY: x_{k|k} from y and u over k-na .. k-1, the current sample excluded (Hoekstra et al.,
+    # Eq. (22e)): na = nb past samples
     na = nb = hp['na_nb']
 
-    ic = Interconnect(nxd, cfg.nu, cfg.ny)
+    ic = Interconnect(nx, cfg.nu, cfg.ny)
 
-    # CHANGED (D-191): detune and train in the selected coordinate system.
-    _names = Reduced_Gantry_State_Block.PARAM_NAMES
-    _nominal = torch.stack([getattr(gantry_ss, n) for n in _names]).to(cfg.dtype_pt)
-    _combo = Reduced_Gantry_State_Block.combos_of(_nominal, gantry_ss.Lb)
+    # the baseline is trained in the ten identifiable combinations; the detuned arm starts them off nominal
+    raw_names = Reduced_Gantry_State_Block.PARAM_NAMES
+    raw_nominal = torch.stack([getattr(gantry_ss, n) for n in raw_names]).to(cfg.dtype_pt)
+    combo_start = Reduced_Gantry_State_Block.combos_of(raw_nominal, gantry_ss.Lb)
     if cfg.combo_init_detune is not None:
-        _combo = _combo * _combo.new_tensor(cfg.combo_init_detune)
+        combo_start = combo_start * combo_start.new_tensor(cfg.combo_init_detune)
     phy_block = Reduced_Gantry_State_Block(
-        params_init=_nominal, combo_init=_combo, Y_op=None, std_x=norm.std_x, std_u=norm.std_u,
+        params_init=raw_nominal, combo_init=combo_start, Y_op=None, std_x=norm.std_x, std_u=norm.std_u,
         x_mean=norm.x_mean, u_mean=norm.u_mean, Ts=cfg.ts_new, up_sample=hp['up_sample'],
         RMSE_baseline=cfg.param_rmse_baseline).to(cfg.dtype_pt)
-    # CHANGED: C_aug / Parameterized_Linear_Output_Block removed: diag_gradient_routing
-    # confirms near-zero C_aug init (Frobenius ~1e-2) chokes ANN gradient by same factor.
-    out_phys = Linear_Output_Block(C=norm.Cd_norm, D=norm.Dd_np)
+    out_phys = Linear_Output_Block(C=norm.Cd_norm, D=norm.Dd_np)    # fixed output map, not learned
     ic.add_block(phy_block)
     ic.add_block(out_phys)
 
     ann_block = Static_ANN_Block(
-        nz=nxd + cfg.nu, nw=len(route_ix),
+        nz=nx + cfg.nu, nw=len(route_ix),
         n_nodes_per_layer=hp['n_nodes_per_layer'],
         n_hidden_layers=hp['n_hidden_layers'],
-        net=live_g_feed_forward_nn,
+        net=AugmentationNet,
         net_kwargs={'nx_phys': cfg.nx_phys},
         activation=torch.nn.Tanh,
     )
     ic.add_block(ann_block)
 
+    # x_{k+1} = f_base(x_tilde_k, u_k) + phi_aug(x_k, u_k) on the routed states, y_k = C x_tilde_k + D u_k
     ic.connect_block_signals(ann_block, ["x", "u"], [])
-    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(route_ix, nxd))
-    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
+    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(route_ix, nx))
+    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nx))
     ic.connect_block_signals(phy_block, ["u"], [])
-    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
-    ic.connect_signals("x", out_phys, "concat", selection_matrix(PHY_IX, nxd))
+    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nx))
+    ic.connect_signals("x", out_phys, "concat", selection_matrix(PHY_IX, nx))
     ic.connect_block_signals(out_phys, ["u"], ["y"])
 
-    # CHANGED (2026-08-28): one composed class, replacing the
-    # ParamLoss -> OrthLoss -> MultipleShooting chain.
-    fit_sys = SSE_Interconnect_Composed(interconnect=ic, na=na, nb=nb)
+    fit_sys = SSE_Interconnect_Weighted(interconnect=ic, na=na, nb=nb)
 
     fit_sys.norm.u0   = norm.u_mean.flatten()
     fit_sys.norm.ustd = norm.std_u.flatten()
     fit_sys.norm.y0   = norm.y0
     fit_sys.norm.ystd = norm.ystd
 
-    # D-251/D-271: physical rows set to the least-squares map of a closed-loop baseline
-    # simulation by prefit_physical_encoder before training. The class reads n+1 samples for its
-    # argument n, so n = na - 1 gives the na-sample window.
-    fit_sys.encoder = linear_encoder_init_aug(
+    # psi = [W^b; W^a] chi + psi_tilde(chi); prefit_physical_encoder sets the physical rows before training.
+    # The class reads n + 1 samples for its argument n, so na - 1 gives the na-sample window.
+    fit_sys.encoder = AugmentedEncoder(
         nx=cfg.nx_phys, nu=cfg.nu, ny=cfg.ny, na=na - 1, nb=nb - 1,
-        nx_aug=NX_ANN,
+        nx_aug=n_a,
         n_nodes_per_layer=hp['n_nodes_per_layer'],
         n_hidden_layers=hp['n_hidden_layers'],
         dtype=cfg.dtype_pt,
     ).to(cfg.dtype_pt)
 
-    # CHANGED: pass lr and eps at optimizer creation.
-    # CHANGED (D-169, job 80714): the model is BUILT on the CPU even when cfg.device='cuda', and
-    # train moves it for the duration of fit() only.
+    # no CPU fallback: config.json records the device the run asked for
     if cfg.device == 'cuda' and not torch.cuda.is_available():
         raise RuntimeError(
-            'device="cuda" was requested but torch.cuda.is_available() is False. Falling back to '
-            'the CPU is deliberately NOT done: the run would be slower by more than an order of '
-            'magnitude while config.json recorded DEVICE=cuda, and the two would be compared as '
-            'if they were the same experiment. Check the SLURM allocation (--gres=gpu:1) or set '
-            'device="cpu" explicitly.')
+            'device="cuda" was requested but CUDA is not available; check the GPU allocation '
+            '(--gres=gpu:1) or run with --cpu.')
     fit_sys.init_model(sys_data=data.train_data, optimizer_kwargs={'lr': hp['lr'], 'eps': cfg.adam_eps})
     eps_values = {float(group['eps']) for group in fit_sys.optimizer.param_groups}
     if eps_values != {float(cfg.adam_eps)}:
         raise RuntimeError(
-            f'Adam eps mismatch after optimizer construction: config={cfg.adam_eps}, '
-            f'optimizer={sorted(eps_values)}')
+            f'Adam eps after optimizer construction is {sorted(eps_values)}, the config asks '
+            f'{cfg.adam_eps}.')
     fit_sys.hfn.to(cfg.dtype_pt)
 
     if cfg.obc:
@@ -112,7 +105,7 @@
 
 @contextmanager
 def on_device(fit_sys, cfg: RunConfig):
-    """D-169: the model lives on the CPU and visits cfg.device for the work inside."""
+    """The model is built and kept on the CPU and moved to cfg.device for the work inside the block."""
     if cfg.device == 'cuda':
         fit_sys.cuda()
     try:
@@ -123,19 +116,20 @@
 
 
 def prefit_physical_encoder(fit_sys, cfg: RunConfig, data, norm):
-    """D-251, D-271: start the PHYSICAL rows of the encoder at the least-squares map of a closed-loop
-    baseline simulation."""
+    """Start the encoder's physical rows at the least-squares map from the windows of a closed-loop
+    simulation of the start model, where f_aug = 0."""
     with on_device(fit_sys, cfg):
         _prefit(fit_sys, cfg, data, norm)
 
 
 def _prefit(fit_sys, cfg, data, norm):
+    """Simulate every training record in closed loop, then fit the encoder to the simulated states."""
     enc, sim = fit_sys.encoder, fit_sys.simulator
     nx, na, nb, stride = cfg.nx_phys, fit_sys.na, fit_sys.nb, cfg.stride
     dtype, dev = next(enc.parameters()).dtype, next(enc.parameters()).device
     T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=dev)
     rv = lambda a: np.asarray(a, np.float64).ravel()
-    fit_sys._sync_prediction_state_for_validation()
+    fit_sys._sync_obc_coefficient()
     step_kw = {}
     with torch.no_grad():
         step_kw['pass_mats'] = fit_sys.hfn.connected_blocks[fit_sys._mats_ix].pass_mats()
@@ -148,16 +142,18 @@
     uh_all, yh_all, xt_all = [], [], []
     groups = {}
     for i, sd in enumerate(sdl):
-        groups.setdefault(len(sd.u), []).append(i)
+        groups.setdefault(len(sd.u), []).append(i)      # records of equal length share one rollout
     for ixs in groups.values():
         un = np.stack([(sdl[i].u - rv(fit_sys.norm.u0)) / rv(fit_sys.norm.ustd) for i in ixs])
         yn = np.stack([(sdl[i].y - rv(fit_sys.norm.y0)) / rv(fit_sys.norm.ystd) for i in ixs])
+        # start at the measured positions q_0 = P^-T y_0, zero velocity and zero added states, normalised
         x0 = np.zeros((len(ixs), nx + cfg.nx_ann))
         q0 = np.stack([np.asarray(sdl[i].y[0], np.float64) @ P_INV_T64.T for i in ixs])
         x0[:, :nx] = (np.hstack([q0, np.zeros_like(q0)]) - x_mean) / std_x
         xs, us = [], []
 
         def recorder(x, u, *args):
+            # the state and the closed-loop input u + u_fb of every step
             xs.append(x)
             us.append(u)
             return fit_sys.hfn(x, u, *args)
@@ -169,6 +165,7 @@
                 **step_kw)
         X = torch.stack(xs, 1).reshape(len(ixs), len(xs), -1)
         U = torch.stack(us, 1).reshape(len(ixs), len(us), -1)
+        # window s: chi from the samples s .. s+na-1 of the closed-loop input and y_hat, target x_tilde at s+na
         starts = torch.arange(0, X.shape[1] - max(na, nb), stride, device=dev)
         uh_all.append(U[:, starts[:, None] + torch.arange(nb, device=dev)].flatten(0, 1))
         yh_all.append(y_hat[:, starts[:, None] + torch.arange(na, device=dev)].flatten(0, 1))
@@ -178,22 +175,25 @@
     least_squares_start(enc, nx, uh, yh, xt)
 
 
-# HEURISTIC (D-271, Dirk): readiness of the start, every physical state reconstructed with R^2 >= 0.999 on
-# the simulated windows; the accurate start of the ablation (F1) reached about 1e-7, the failing ones 0.04 to 8.8.
+# HEURISTIC: the start is ready when every physical state is reconstructed with R^2 >= 0.999 on the
+# simulated windows; an accurate start reaches about 1e-7 unexplained, the failing starts tried 0.04 to 8.8
 _START_MAX_UNEXPLAINED = 1e-3
 
 
 def least_squares_start(enc, nx, uh, yh, xt):
-    """D-271: set the physical rows of the encoder to the least-squares map x ~ W [yh; uh] + b."""
+    """Set the encoder's physical rows to the least-squares map x_tilde = W^b chi + b; returns (W, b)."""
     final = enc.net[-1]
     wa_before = (enc.Wa_psi_y.detach().clone(), enc.Wa_psi_u.detach().clone())
     cpu = lambda a: a.detach().to('cpu', torch.float64)
     Phi, X = cpu(torch.cat([yh.flatten(1), uh.flatten(1)], 1)), cpu(xt)
     if not (torch.isfinite(Phi).all() and torch.isfinite(X).all()):
         raise RuntimeError('the encoder start windows contain non-finite values')
+    # least squares on standardised regressors, mapped back to chi below
     mu, s, x_mu = Phi.mean(0), Phi.std(0, unbiased=False), X.mean(0)
     s = torch.where(s > 0, s, torch.ones_like(s))
+    # THEORY: rcond = eps max(M, N), the numpy.linalg.lstsq default cutoff
     rcond = torch.finfo(torch.float64).eps * max(Phi.shape)
+    # gelsd: repeatable to the last bit on the CPU, the default gelsy is not
     sol = torch.linalg.lstsq((Phi - mu) / s, X - x_mu, rcond=rcond, driver='gelsd')
     W = sol.solution / s[:, None]
     b = x_mu - mu @ W
@@ -205,11 +205,12 @@
         p = next(enc.parameters())
         enc.Wb_psi_y.copy_(W[:ny_h].T.to(p))
         enc.Wb_psi_u.copy_(W[ny_h:].T.to(p))
+        # physical rows of the output layer of psi_tilde: zero weights, bias b
         final.weight[:nx] = 0.0
         final.bias[:nx] = 0.0
         final.bias[:nx] = b.to(p) - enc(uh[:1] * 0, yh[:1] * 0)[0, :nx]
         got = cpu(enc(uh, yh)[:, :nx])
-    unexplained = ((pred - X) ** 2).mean(0) / X.var(0, unbiased=False)
+    unexplained = ((pred - X) ** 2).mean(0) / X.var(0, unbiased=False)      # 1 - R^2 per state
     print('[encoder start] least squares: %d regressors, rank %d, condition %.3e; unexplained per state %s'
           % (Phi.shape[1], int(sol.rank), float(sol.singular_values[0] / sol.singular_values[-1]),
              ' '.join('%.2e' % v for v in unexplained)), flush=True)
```
