# Snapshot: `model_augmentation/` and `gantry_dynamic/` as of 2026-08-26

**Base commit**: `a0e3f76` (branch `Augmentation`). Taken before a planned revert of both folders.

## COMPLETE copy: `tree/`

`tree/` holds **verbatim copies of both folders in full**, caches excluded. This is the
authoritative record; use it for any restore.

| snapshot path | repo path | files |
|-|-|-|
| `tree/model_augmentation/` | `model_augmentation/` | 20 |
| `tree/scripts_gantry_gantry_dynamic/gantry_dynamic/` | `scripts/gantry/gantry_dynamic/` | 17 |

`patches/model_augmentation.ALL.patch` and `patches/gantry_dynamic.ALL.patch` hold `git diff HEAD`
for every tracked-modified file in each folder.

### Untracked files, which no git operation can restore

These exist only in `tree/` and in no commit:

- `model_augmentation/fit_systems/augmented_dynamics.py` (D-160 recurrence; **imported by
  `model.py`'s `aug_dynamics` branch at HEAD too**, so deleting it breaks a reverted tree as well)
- `model_augmentation/fit_systems/kessels_extension.py` (Kessels block, writer, split encoder,
  `bla_initialize_writer_`, `augmented_state_jacobian`)
- `scripts/gantry/gantry_dynamic/pole_init.py` (**exercised by `test_pole_init.py`, 9 tests**)
- `scripts/gantry/gantry_dynamic/bounded_integral_block.py`
- `scripts/gantry/gantry_dynamic/lipschitz.py`
- `scripts/gantry/gantry_dynamic/passive_ph_block.py`
- `scripts/gantry/gantry_dynamic/patches/2026-08-19-ann-init-scale.patch`

**`git checkout --` leaves all of these alone. `git clean -fd` deletes every one of them.**

## Restore either folder in full

```
cp -r tasks/snapshots/2026-08-26-model-augmentation/tree/model_augmentation/. model_augmentation/
cp -r tasks/snapshots/2026-08-26-model-augmentation/tree/scripts_gantry_gantry_dynamic/gantry_dynamic/. \
      scripts/gantry/gantry_dynamic/
```

---

## Earlier partial record (superseded by `tree/`)

The section below was written when only three `model_augmentation/fit_systems/` files had been
copied into `files/`. Those copies are still present and still valid; `tree/` supersedes them in
scope.

## Contents and provenance

| snapshot file | repo path | git state | sha256 (first 16) |
|-|-|-|-|
| `files/model_augmentation_fit_systems_closed_loop.py` | `model_augmentation/fit_systems/closed_loop.py` | **tracked, modified** | `dc0829586fd1e5e0` |
| `files/model_augmentation_fit_systems_augmented_dynamics.py` | `model_augmentation/fit_systems/augmented_dynamics.py` | **UNTRACKED** | `2bb9aaf47830182c` |
| `files/model_augmentation_fit_systems_kessels_extension.py` | `model_augmentation/fit_systems/kessels_extension.py` | **UNTRACKED** | `5824fe8f355ee8db` |
| `patches/model_augmentation_fit_systems_closed_loop.py.patch` | `git diff HEAD` for the one tracked file | | `0fd7b3b0e3cf2d35` |

`patches/` holds a patch only for the tracked file. The two untracked files have nothing to diff
against, so `files/` is their only record. That is exactly why this snapshot was taken.

## What each file is

- **`closed_loop.py`** (tracked, `+104` against HEAD). The change is **purely additive**: an optional
  `return_error=False` argument on `closed_loop_free_run_rms`, plus three new functions
  `window_starts`, `make_window_tensors`, `closed_loop_window_rms`. The scalar scoring path is
  unchanged, and the file's own docstring states so: "The scalar path is untouched, so the selection
  number cannot move because of this argument." Verified by reading the diff.
- **`augmented_dynamics.py`** (untracked). The D-160 augmented-state recurrence,
  `x_a[k+1] = A_aa x_a[k] + Gamma B z[k] + alpha F(z)[aug]`, which **replaced** the older `AUG_LRU`
  implementation that produced F5. There is no other copy of it in the repository.
- **`kessels_extension.py`** (untracked). The Kessels extension block, writer, split encoder, and
  physical-ANN split, plus `bla_initialize_writer_` and `augmented_state_jacobian` added
  2026-08-25/26. There is no other copy of it in the repository.

## Restore

```
# tracked file, from this snapshot
cp tasks/snapshots/2026-08-26-model-augmentation/files/model_augmentation_fit_systems_closed_loop.py \
   model_augmentation/fit_systems/closed_loop.py

# or re-apply just its diff on top of HEAD
git apply tasks/snapshots/2026-08-26-model-augmentation/patches/model_augmentation_fit_systems_closed_loop.py.patch

# untracked files, which no git operation can bring back
cp tasks/snapshots/2026-08-26-model-augmentation/files/model_augmentation_fit_systems_augmented_dynamics.py \
   model_augmentation/fit_systems/augmented_dynamics.py
cp tasks/snapshots/2026-08-26-model-augmentation/files/model_augmentation_fit_systems_kessels_extension.py \
   model_augmentation/fit_systems/kessels_extension.py
```

## WARNING about resetting this folder

Read before running anything destructive.

1. **`git checkout -- model_augmentation/` does not remove the untracked files**, so it would revert
   only `closed_loop.py`. **`git clean -fd model_augmentation/` would delete both untracked files**,
   which are the entire D-160 recurrence and the entire Kessels implementation. Neither is recoverable
   from git. This snapshot is the only copy.
2. **Reverting `closed_loop.py` will break the test suite.** `scripts/gantry/augmented-states/tests/
   test_window_grid.py` tests `window_starts` and the training-window grid, which exist only in the
   modified version. The 65-test discovery passed 64 with 1 skip on 2026-08-25 with these files in
   place.
3. **The evidence says this folder is not the cause of the 2026-08-25/26 baseline regression.** The
   untrained closed-loop score is `2.5341870076e-06 m` where the recorded scalar is
   `2.1866026634e-06 m`, 15.9 % worse. Measured this session:
   - the numpy harness (`cl_headroom.py`) still reproduces `2.1850 / 2.1975 / 2.1763 / 2.1874e-06`
     exactly, so plant, controller, records and metric are intact;
   - the pure default path and the `aug_dynamics` path give **bit-identical** untrained scores
     (`2.5341870076e-06` both), so the augmented block is not involved;
   - `closed_loop.py`'s scalar scoring path is unchanged against HEAD.

   The remaining suspects are `scripts/gantry/gantry_dynamic/model.py` (dirty `+200`; `W^b` is built
   from `gantry_linearize_and_discretize` and `normalize_linear_ss_matrices(..., norm.x_all)`, and
   D-119 warns that a state scaling from a different array than the one defining `x_mean/std_x` puts
   the encoder in a different state frame from the rollout) and the modified MATLAB generators
   `generate_trajectory_data.m`, `gtd_config.m`, `gtd_run_simulation.m`, since `norm` is fitted on the
   fourteen training records. Neither is covered by this snapshot.

   Symptom that localises it: encoder-init and true-`x0` closed-loop scores used to agree to
   `0.003 to 0.005 %` (D-141) and now differ by 15.9 %, i.e. it is an initial-state or state-frame
   problem, not a plant problem. The decisive test is to compare the encoder's `x_b(k0)` against the
   true `x_logical(k0)` on one record; it needs no historical reference number.

## Not covered by this snapshot

`scripts/gantry/gantry_dynamic/model.py`, `config.py`, and
`scripts/gantry/closed-loop-controller/cl_train.py` are also dirty and carry the Kessels build
integration and the `CL_AUG_*` / `CL_ENC_WA_ZERO` hooks. If a wider reset is intended, snapshot those
first.
