# blackbox-cl-trainable: implementation

## The black-box arm (option B, BB2-018)
- `train_bb_clmap.py`: Jan's SUBNET on the closed-loop map (r, f) -> y
  - model: deepSI 0.3.29 `SS_encoder_general_hf`, f and h 2 x 8 tanh, encoder 2 x 16 (Jan's)
  - training: deepSI's own `fit()`, sim-RMS selection, `auto_fit_norm` on the 18 training records
  - data: thesis records through the pipeline's `_record_at_fs_new` (u, y and the fields r_sim,
    f_sim on the same 4 kHz samples, same trim); test records never loaded
  - output: `result.json` with per-axis and per-record RMS [m], losses, BB2-018 criteria, and
    (since BB2-019) y_hat = r and NRMS_e per axis and record
  - flags (BB2-019 to BB2-026, defaults reproduce b1 to b6): `--target e` (train on e = y - r,
    predict r + e_hat), `--in64` (float64 records into deepSI), `--dr` (add r(k) - r(k-1)),
    `--na`, `--na-right`, `--nf`, `--batch`, `--stride`, `--fh-width/-depth`, `--e-width/-depth`,
    `--its-per-val`, `--timeout` (deepSI's own arguments); final configuration in REPORT.md
- `ref_scores.py`: section-10 references (y_hat = r, grey box epoch 0 via the pipeline's
  `closed_loop_table` on train + val records only, noise floor) -> `outputs/ref/references.json`
- `bbcl/jan_model.py`: Jan's nets with `.reshape` for `.view`; `JanHF` / `build` are the
  plant-model variant on the grey box's carrier (used by `train_bb.py`, closed)
- `bbcl/setup.py`: CFG read from `gantry_interconnect_dynamic.py`, overridden in named fields only
- `runners/run_bb_clmap.sh`: SLURM, one seed and version per job, not submitted

## Differences from Jan's example (`scripts/ecc_2025/msd_ndof_deepSI_encoder.py`)
- Same: model class, net sizes, deepSI `fit()`, sim-RMS selection, auto norm, random init, lr 1e-3
- Inputs: the loop's exogenous r (reference) and f (injected force), 6 channels, instead of the
  plant input; output y (3 stage positions)
- na = nb = 29 (grey box's encoder lag) instead of 4 dof + 1; nx 16 instead of 2 dof (plant rule)
- nf 400, local stride 100 and batch 64 (server: stride 10, batch 512)

## Differences from the grey box's training
- No K1 in the rollout: the model IS the loop (plant plus K1), so the same quantity is predicted
  from the same r and f without simulating a controller
- float32 (deepSI default) instead of float64; no L-BFGS polish

## Tried and closed (kept as the record, REPORT.md "Why a black-box plant ...")
- `bla/`: BLA check (BB2-001 to BB2-005)
- `bbcl/ici_model.py`, `phase_b.py`, `phase_b_mech.py`: controller-in-model design (BB2-006 to
  BB2-012); `phase_b.py` contains a temporary instance-method spy used once for B6 (pb3 never
  reached it); not used anywhere else
- `train_bb.py`: Jan's plant model under K1 with window curriculum (BB2-013 to BB2-017)
- `tools/watchdog.ps1`, `tools/run_wd.sh`: RAM-guarded launch, switched off by the user (BB2-011)

## How to run
- Local: `python -u train_bb_clmap.py --version noisy --seed 1 --epochs 10 --out outputs/b/noisy_s1`
  (env python; about 29 min per run on this machine, 57 % of it validation)
- Server: `BB_VERSION=noisy BB_SEED=1 sbatch scripts/gantry/blackbox-cl-trainable/runners/run_bb_clmap.sh`

## Plant information: none from the truth, two disclosed inheritances
- Inherited from the grey box's settings (CRITIC finding 4): na = nb = 29 comes from Jan's rule
  (nx_phys + nx_ann) 2 + 1 (`gantry_dynamic/config.py`), which contains the physical state count;
  nf = 400 rests on D-220, partly computed from model matrices. Kept for identical windows with the
  grey box; both are search axes.
- Initialisation: random seed (PyTorch default), Jan's init
- State dimension: nx 16, a search axis, not the truth's order (Jan's 2 dof rule not used)
- Normalisation: deepSI auto norm on the 18 training records
- Data: training records for fitting, validation records for selection; test records not loaded
- The truth model was used only to score the BLA check and to check Kt in the closed ICI design
