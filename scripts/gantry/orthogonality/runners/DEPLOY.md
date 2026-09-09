# Deploying the trajectory regulariser to the server

Nothing is pushed automatically in this project. Copy by hand to the same relative path under
the server repo root, exactly as `closed-loop-controller/runners/DEPLOY-wave1.md` describes.

    R=/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation

## Files to copy

| Path (same on both sides) | What it is |
|-|-|
| `model_augmentation/fit_systems/trajectory_orth_projection.py` | geometry, projection, exact chunking, `compressed_nuisance_basis` |
| `model_augmentation/fit_systems/trajectory_adapter.py` | the adapter contract |
| `model_augmentation/fit_systems/trajectory_penalty_hook.py` | **new**: the gradient contribution, Eq. (22)/(24) |
| `model_augmentation/fit_systems/interconnect.py` | **modified**: `SSE_Interconnect_Composed.traj_penalty` and its `fit()` override |
| `model_augmentation/fit_systems/blocks.py` | **modified**: the non-persistent ANN `out_gate` |
| `scripts/gantry/gantry_dynamic/config.py` | **modified**: `traj_*` fields, guards, config.json keys |
| `scripts/gantry/gantry_dynamic/model.py` | **modified**: the `traj_orth` notice |
| `scripts/gantry/gantry_dynamic/traj_orth.py` | **new**: geometry artifact build/load and attach |
| `scripts/gantry/orthogonality/` (whole subtree) | verify.py, smoke_train.py, gantry/, testbed/, runners/, documentation/ |
| `environment.server-train.yml` | **modified**: adds `psutil` |

`psutil` is optional. Without it the suite falls back to `resource.getrusage`, which works on
Linux, so a stale env still runs; the per-operation attribution is just coarser. To add it:

    conda install --prefix /dataB1/dirk_van_den_berg/conda-envs/GraduationProject -c conda-forge psutil

## Once, before the first sbatch

SLURM resolves `-o` at schedule time, so the log directory must already exist or the job dies
before the script runs (see `drift-isolation/runners/make_log_dirs.sh`):

    mkdir -p /home/dirk_van_den_berg/logs/augmentation/orthogonality

## Verify the copy landed, before submitting

    cd $R
    python -c "import ast;[ast.parse(open(f).read()) for f in [
      'model_augmentation/fit_systems/trajectory_penalty_hook.py',
      'scripts/gantry/gantry_dynamic/traj_orth.py']];print('parse ok')"
    grep -n "C = U.T @ G" model_augmentation/fit_systems/trajectory_orth_projection.py
    grep -n "traj_penalty is None" model_augmentation/fit_systems/interconnect.py
    python scripts/gantry/orthogonality/testbed/test_trajectory_projection.py   # 32 passed
    python scripts/gantry/orthogonality/testbed/test_penalty_hook.py            # 12 passed
    python scripts/gantry/orthogonality/testbed/test_fit_override.py            # 22 passed
    python scripts/gantry/orthogonality/testbed/test_cache_provenance.py        # 11 passed
    python scripts/gantry/orthogonality/testbed/test_diag.py                    # 11 passed

The first `grep` is not ceremony: `compressed_nuisance_basis` shipped once with `Gamma_w`
omitted, and an old copy of that file on the server would fail V3 in a way the ladder would
report as a resource problem.

## Submit

    sbatch scripts/gantry/orthogonality/runners/run_verify_ladder.sh          # stage A
    LADDER_JOB=<stage A job id> sbatch scripts/gantry/orthogonality/runners/run_smoke_train.sh

Stage B refuses to start unless stage A's JSON reports every requested stage complete. Chain it
with `--dependency=afterok:<A>` if you prefer, but `LADDER_JOB` is still required: the gate reads
that run's results, not merely its exit status.

## Bringing results back

By hand, into `scripts/gantry/orthogonality/server-results/<JOBID>/`, with the `.out` log renamed
to `verify_ladder<JOBID>.out`, matching `ann-blackbox/server-results/` and
`meeting/meeting-07-09-2026/server/`.

Read `summary.stages_incomplete` first. A non-empty list means the headline pass count does not
cover what was requested, and `V3_operations.incomplete` names the operation that never returned.

## Stage C: GPU + torch.compile parity, then the gated longer run (added 2026-09-08)

Everything verified before this ran eager on CPU. Stage C establishes whether the production
path (CUDA + `torch.compile`) executes the same mathematics, and only then starts a longer run.

### Additional files to copy

| File | Why |
|-|-|
| `scripts/gantry/orthogonality/run_gpu_ladder.py` | new: the stage C entry point |
| `scripts/gantry/orthogonality/runners/run_gpu_compile.sh` | new: the sbatch job |
| `scripts/gantry/orthogonality/check_update.py` | changed: `setup` honours `compile_mode`; check 2 rewritten |
| `scripts/gantry/orthogonality/train_orth.py` | changed: `--compile-mode`, corrected `hook_calls` reporting |
| `scripts/gantry/orthogonality/gantry/env.py` | changed: `compile_mode` threaded through `config_from_run` and `load_env` |
| `scripts/gantry/orthogonality/gantry/adapter.py` | changed: `rebind` and `assert_bound` |
| `model_augmentation/fit_systems/interconnect.py` | changed: `checkpoint_load_system` rebinds the adapter |
| `scripts/gantry/orthogonality/testbed/test_fit_override.py` | changed: 2 new rebind tests (31 total) |

`env.py`, `adapter.py` and `interconnect.py` carry the stale-reference fix. An old copy of any of
them on the server resumes training with the penalty bound to the pre-load modules, which does
not raise and is not visible in the loss.

### Submit

    mkdir -p /home/dirk_van_den_berg/logs/augmentation/orthogonality      # once, by hand
    sbatch scripts/gantry/orthogonality/runners/run_gpu_compile.sh

Env-var overrides, all optional: `STAGES` (default `0,1,2,3`), `COMPILE_MODE` (default
`reduce-overhead`), `TRAIN_ITS` (default 200), `WINDOWS`, `BATCH`.

### Expect to iterate

Compilation is expected to fail a few times first. The JSON is written incrementally after every
stage, so a stage 2 crash still leaves stages 0 and 1 on disk.

    STAGES=0,1 sbatch ...          # cheap loop while fixing a codegen error
    ... --no-compile               # confirm the mathematics is fine and the fault is compilation

The likeliest failure is NOT a crash. `ann_gate` mutates `Static_ANN_Block.out_gate`, and dynamo
guards on module attributes, so a gate switch either recompiles (visible as a large stage 2 time
and a rising `dynamo` counter) or is specialised into the graph and IGNORED. The specialised case
does not crash and does not zero the penalty, because the latent-init half of the intervention
still differs. Stage 2's `the gate still changes the compiled rollout` check exists for exactly
that, using the clamped rollout, which differs from full ONLY by the gate.

If the gate is being specialised, the fix is to write the mask IN PLACE (`out_gate.copy_(mask)`)
rather than rebinding the attribute, and to re-run stage 2. Do not "fix" it by loosening the
tolerance.

### Reading the result

    scripts/gantry/orthogonality/results/gpu_ladder_<JOBID>/gpu_ladder.json

`summary.failures` first. Then `stages.2_compiled.out_of_tolerance` (quantities exceeding the
measured floor), `stages.2_compiled.dynamo` (recompilation counters) and
`stages.2_compiled.speedup_vs_eager`. If `stages.2_compiled.skipped` is present, stage 4 ran
eager or not at all: a skipped parity stage is not a passed one.

Tolerances are MEASURED, not chosen: stage 1 runs the identical eager computation twice and takes
the run-to-run difference as the floor. The acceptance margin is 10x that floor, a HEURISTIC
recorded in the JSON as `tolerance_policy`.
