# Server setup for `train_param_recovery.py`

This repo now supports a standalone server setup for
`lpv_lfr_baseline/scripts/train_param_recovery.py` without requiring
`model_augmentation` or `deepSI`.

## What to copy to the server

Copy at least these paths:

- `lpv_lfr_baseline/`
- `Matlab-output/lpv_sim_varying_y.mat`
- `environment.server-train.yml`

Optional but useful:

- `docs/server-setup.md`
- `scripts/server/train_param_recovery.slurm`

The training script writes outputs under:

- `models/gantry/param_recovery/`

You do not need to pre-create that folder. The script creates it.

## Create the conda environment

The PDF in `literature/server/Running Experiments 4.pdf` recommends:

- install Miniconda locally for your user
- let the installer append conda setup to `.bashrc`
- source conda explicitly inside Slurm jobs

For your setup, the clean layout is:

- Miniconda in `home`
- repo in `/dataB1/dirk_van_den_berg/LPV-LFR-Baseline-Augmentation`
- env in `/dataB1/dirk_van_den_berg/conda-envs/GraduationProject`
- conda package cache in `/dataB1/dirk_van_den_berg/conda-pkgs`

From the repo root on the server:

```bash
mkdir -p /dataB1/dirk_van_den_berg/conda-envs
mkdir -p /dataB1/dirk_van_den_berg/conda-pkgs
export CONDA_PKGS_DIRS=/dataB1/dirk_van_den_berg/conda-pkgs
conda env create --prefix /dataB1/dirk_van_den_berg/conda-envs/GraduationProject -f environment.server-train.yml
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
```

If you prefer, you can add the package cache permanently in `~/.condarc` instead:

```bash
cat >> ~/.condarc <<'EOF'
pkgs_dirs:
  - /dataB1/dirk_van_den_berg/conda-pkgs
EOF
```

## Run training

Run from the repo root, for example:

```bash
cd /dataB1/dirk_van_den_berg/LPV-LFR-Baseline-Augmentation
conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
python -m lpv_lfr_baseline.scripts.train_param_recovery
```

Note: the comment inside `train_param_recovery.py` still mentions
`python -m lpv_lfr_baseline.train_param_recovery`, but the actual module path
is `lpv_lfr_baseline.scripts.train_param_recovery`.

## Optional packages

These are not required for `train_param_recovery.py`, but may be useful later.

### `deepSI`

Needed only for `load_gantry_data()` and some Jan-framework workflows.

Local environment reference:

```bash
pip install "deepSI @ git+https://github.com/GerbenBeintema/deepSI@109cf74f49a7b27539232c53981584befc1becc0"
```

### `model_augmentation`

Needed only for Jan-compatible interconnect training (`LFRFitSystem`), not for
standalone parameter recovery.

Local environment reference:

```bash
pip install -e "git+https://github.com/DirkvdBerg/Model-Augmentation-Public.git@f9c51988eb078093736737b0b8dd233ff49aa877#egg=ModelAugmentation"
```

## Versions mirrored from the local `GraduationProject` environment

- Python 3.11.14
- torch 2.5.1
- CUDA runtime 12.1
- torchvision 0.20.1
- torchaudio 2.5.1
- numpy 2.0.1
- scipy 1.17.0
- matplotlib 3.10.8

## Slurm

The PDF in `literature/server/Running Experiments 4.pdf` recommends using Slurm
when possible. A template job file is included at:

- `scripts/server/train_param_recovery.slurm`

Adjust the partition, memory, wall time, and conda activation path for the
server you are using.
