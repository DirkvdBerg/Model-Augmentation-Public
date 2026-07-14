# drift-demo

Clean, purpose-built scripts to demonstrate WHERE the X/Y augmentation drift comes from, for the
supervisor meeting. Distinct from the exploratory `../diagnostics-drift/` tree.

Full plan (issue definition, ladder, numbers, narrative): `docs/drift-demonstration-plan.md`.
Full diagnosis (do not duplicate): `docs/drift-diagnosis-status.md`.

Reuse (verified, not re-derived): `../diagnostics-drift/drift_common.py` (truth EOM, `simulate_baseline`,
`tau_X/tau_Y`, P-transform) and `../gantry_dynamic/` (loader, normalization, `build_model`,
`encoder_init_state`). New code = clean demonstration drivers + plots only.

Outputs: `scripts/gantry/drift-demo/figures/` (user 2026-07-14).

## Scripts (build order)

| Script | Shows | Status |
|---|---|---|
| `demo_common.py` | shared slim-load, pipeline/encoder, baseline free-run, x0_enc/x0_true, spectra | planned |
| `demo1_baseline_encoder_ic.py` | encoder x0 vs true x0 (stage+logical): prong 1 + K=0 stiffness contrast (`E=R+enc_IC`) | planned |
| `demo2_excitation.py` | `u_w` vs `u_n` + spectra + `delta_a`: absorber breaks loop cancellation, data informative | planned |
| `demo3_ann_dc_drift.py` | trained ANN DC on K=0 rows + counterfactual: the DC is the drift | planned |
| `demo4_loss_horizon.py` | windowed loss vs free-run horizon: why the loss is blind | planned |
| `demo5_trained_true_vs_encoder_ic.py` | trained ANN from true x0 vs encoder x0: do measured ICs fix the trained drift (prong 1 vs 2) | planned (NEW, decisive; latent-x0 design first) |
| `demo6_objective_split.py` | window nf-RMS vs sim-RMS split + cross-nf "Roland" figure (log/history parse, no simulation) | planned |

Full per-figure specs (panels, legends, comparators, attack closures): plan doc §12.
Checkpoints: `simulations/gantry_subnet/augmentation_linear_map/trial_ckpts_71013/` (cold, nf 800-3200),
`.../curriculum_70903/rung*_last.pth` (warm, nf 400-2000), `.../diagnostics/checkpoints/gantry_drift_last.pth`.

Run: `conda run -n GraduationProject python scripts/gantry/drift-demo/<script>.py`
(set `PYTHONIOENCODING=utf-8` on PowerShell).
