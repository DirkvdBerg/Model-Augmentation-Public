# Line-by-line correspondence: Jan's reference to `ann_blackbox.py`

Reference: `scripts/ecc_2025/msd_ndof_deepSI_encoder.py`, 37 lines (byte-identical to
`scripts/gantry/msd_ndof_deepSI_encoder.py`, both from Jan's commit `6d69f6b`).
Cross-check: `scripts/bouc_wen/bouc_wen_ANN_SS.py`, 39 lines, same pattern on a second benchmark.
Every line of Jan's script appears below. "Same" means the line is copied with no change of meaning.

| Jan | What it does | `ann_blackbox.py` | Verdict |
|-|-|-|-|
| L1 | `from deepSI.fit_systems.encoders import SS_encoder_general_hf, default_state_net, default_output_net` | same import, verbatim | same |
| L2 | `import deepSI` | same | same |
| L3 | `import os` | same | same |
| L4 | `import numpy as np` | same | same |
| L5 | blank | blank | same |
| L6 | `## ------------- Load data -----------------` | same banner | same |
| L7 | `dof = 3` | `dof = 4` | **DEV**: the 8-state truth is X + Theta + Y + hidden absorber. Moved next to the model construction it feeds |
| L8 | `SNR = 20` | absent | **DEV**: records are noiseless Simulink output, no SNR to set |
| L9 | blank | blank | same |
| L10 | `data_file_path = os.path.join(os.getcwd(), "data", "mass_spring_damper")` | `REPO` + `sys.path.insert` to `scripts/gantry/msd-offset` | **DEV**: path is resolved from `__file__`, not `os.getcwd()`, so the arm runs from any cwd |
| L11 | `train_data = deepSI.load_system_data(...train.npz)` | `train_data = load(args.train, args.fs)` | **DEV**: records are `.mat`; brief section 9 says import `plant.load_record`, do not reimplement. The helper also carries the Part 2 rate change |
| L12 | `val_data = deepSI.load_system_data(...val.npz)` | `val_data = load(args.val, args.fs)` | same role, same reason |
| L13 | blank | blank | same |
| L14 | `## ------------- Add noise -----------------` | banner kept, body empty with a comment saying why | **DEV** |
| L15-L22 | `if SNR == 20: sigma_n = 15e-3` ... `else: raise` | absent | **DEV**: see L8 |
| L23 | `train_data.y += np.random.normal(0, sigma_n, ...)` | absent | **DEV**: see L8 |
| L24 | `val_data.y += np.random.normal(0, sigma_n, ...)` | absent | **DEV**: see L8 |
| L25 | blank | blank | same |
| L26 | `## ------------- Train fit system -----------------` | same banner | same |
| L27 | `h_net_kwargs = f_net_kwargs = {"n_hidden_layers": 2, "n_nodes_per_layer": 8}` | same, width behind `--width` defaulting to 8 | same values |
| L28 | `e_net_kwargs = {"n_hidden_layers": 2, "n_nodes_per_layer": 16}` | same, behind `--ewidth` defaulting to 16 | same values |
| L29 | `hf_net_kwargs = dict(f_net=..., h_net=...)` | verbatim | same |
| L30 | `fit_sys = SS_encoder_general_hf(nx=dof*2, na=dof*4+1, nb=dof*4+1, ...)` | verbatim, `dof=4` gives `nx=8, na=nb=17` | same formulas |
| L31 | blank | `fit_sys.unique_code = f'fs{fs}'` | **DEV**: `name` is a read-only property over `unique_code` (`deepSI/systems/system.py:79-80`) and keys the `_best`/`_last` checkpoints. Without it the sweep arms overwrite each other's checkpoint |
| L32 | `nf = 200; epochs = 10000; batch_size = 2000` | `nf=400; epochs=500; batch_size=256` | **DEV**: `nf=400` is the brief's fixed horizon. `batch_size` reason on the line. `epochs` is a wall-clock choice, not a structural one |
| L33 | `fit_sys.fit(train_sys_data=, val_sys_data=, batch_size=, epochs=, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="sim-RMS")` | verbatim, plus `timeout=` | **DEV**: `timeout` only. `fit` writes `_last` and the metrics arrays after the loop, so a wall-clock kill loses everything (run 74045). `timeout` makes it return normally |
| L34 | blank | blank | same |
| L35 | `# ------------- Save fit system -----------------` | same banner | same |
| L36 | `model_file_name = "msd_{0}dof_ANN_SS_e10000".format(dof)` | `tag = f'fs{fs}_nf{nf}'` | **DEV**: the name must identify the sweep arm |
| L37 | `interconnect_file_path = os.path.join(os.getcwd(), "models", ...)` | `args.out`, default `./results` | **DEV**: see L10 |
| L38 | `fit_sys.save_system(interconnect_file_path)` | same | same |

## Lines in `ann_blackbox.py` with no counterpart in Jan's 37

| Ours | Reason |
|-|-|
| `from scipy.signal import decimate` | brief section 8: do not decimate `y` without an anti-alias filter. `plant.py:126` point-samples it |
| `import sys, json, argparse` | Part 2 needs one arm per rate from the shell, and metrics must reach disk |
| `FS_NATIVE = 4000.0` | every arm decimates from the same 4 kHz source, so the arms differ only in rate |
| `load()` body: block-mean `u`, `decimate` `y` | block mean on `u` keeps D-087 (`u` is ZOH, so the mean is the energy-preserving reduction); `y` is a sampled position and needs the zero-phase FIR |
| `apply_experiment` + per-channel RMS + JSON dump | brief section 7 scores each arm against **its own** epoch-0, and Theta lives in the `x1 - x2` difference of two channels three orders larger, which a pooled RMS cannot show |

## What the cross-reference settles

`bouc_wen_ANN_SS.py` differs from the MSD script in exactly three places: `nx` (3 vs `dof*2`),
`na`/`nb` (`nx*2+1` vs `dof*4+1`), and the net widths (f/h `2x16`, e `2x8`, i.e. the MSD widths
swapped). Everything else, including the whole `fit(...)` call, is identical. So the widths and the
`na`/`nb` rule are problem-specific and the rest of the structure is not. This file keeps the MSD
values, the gantry being the closer analogue.

## deepSI API, confirmed 2026-07-31

`importlib.metadata.version('deepSI')` reports **0.3.29**, loaded from
`.../envs/GraduationProject/Lib/site-packages/deepSI`. `deepSI-master` (v2, with `dsi.SUBNET` and
`dsi.fit`) is **not installed**, so Beintema's `basic-example.py` defines the concept (encoder,
`f`, `h` as MLPs; a horizon `T`; simulate on held-out input) and Jan's script is the API that runs.
The mapping between them: `dsi.SUBNET(nu, ny, norm, nx, nb, na)` is `SS_encoder_general_hf(nx, na,
nb, ...)` with `auto_fit_norm=True` doing what the explicit `norm` argument does in v2; `dsi.fit(...,
T=20)` is `fit(..., loss_kwargs={'nf': 20})`; `model.simulate(test)` is `fit_sys.apply_experiment`.
