# Snapshot: the augmented-states implementation as of 2026-08-21

**Base commit**: `4cdb7c1` (branch `Augmentation`). Every patch here applies to that commit.

## Why this folder exists

The working implementation that reached `3.795974e-07 m` free-run validation RMS lives in files the
user does not want pushed yet: `model_augmentation/` (Jan's framework) and
`scripts/gantry/gantry_dynamic/` (partly another session's uncommitted work). This folder is a
pushable record of those changes so the result is not lost, without staging either directory.

`files/` holds verbatim copies (path separators flattened to `_`). `patches/` holds one
`git diff` per tracked file, so each change can be applied or rejected **separately**.

## Ownership, so you can delete what is not this track's

| file | owner | contents |
|-|-|-|
| `model_augmentation/fit_systems/closed_loop.py` | **this track** | `closed_loop_rollout` (D-147), the stabilized-PEM rollout; `xc = 0` windowing (D-142) |
| `model_augmentation/fit_systems/pre_encoder.py` | **this track** | `linear_encoder_init_aug`, the `W^b`/`W^a` split (D-130) |
| `scripts/gantry/gantry_dynamic/model.py` | **MIXED, see below** | `AugLRUBypass` + band recipe + `ENC_WA_ZERO` + `AUG_LRU_NA_NB` (this track) AND `split_param_group`/`lr_theta`/`eps_theta` (other session, P1/P1-e) |
| `scripts/gantry/gantry_dynamic/config.py` | **other session** | P1/P1-e: `lr_theta`, `eps_theta` |
| `scripts/gantry/gantry_dynamic/evaluation.py` | **other session** | `orth_frac` meter, `param_init_detune` |
| `scripts/gantry/gantry_dynamic/orth_penalty.py` | **other session** | D-111 penalty basis |
| `scripts/gantry/gantry_dynamic/rezero_gate.py` | **uncertain** | ReZero gate. Referenced by this track's `model.py` as addressing the D-130 `W^a` dead zone, but modified in a session I cannot attribute. Decide before deleting |
| `bounded_integral_block.py`, `lipschitz.py`, `passive_ph_block.py` | **uncertain** | untracked new files, no patch (nothing to diff against). Not used by any run in this snapshot |

### `model.py` cannot be split by file, only by hunk

It is the one file carrying both tracks. Line numbers are into
`patches/scripts_gantry_gantry_dynamic_model.py.patch`:

| patch lines | hunk | owner |
|-|-|-|
| 9-83 | `class AugLRUBypass` | this track (D-150/D-151) |
| 85-126 | `def lru_band_from_artifact` | this track (D-150 band recipe) |
| 128-170 | `get_encoder_dims` + `AUG_LRU_NA_NB` pin | this track (2026-08-21) |
| **172-217** | **`find_log_params`, `split_param_group`** | **other session (P1)** |
| 218-231 | the `[aug-lag]` print in `build_model` | this track |
| 232-322 | the `ANN_REZERO_GATE` block and the whole `AUG_LRU` block | this track |
| 323-351 | `ENC_WA_ZERO` | this track |
| **352-358** | **`if cfg.lr_theta is not None: split_param_group(...)`** | **other session (P1)** |

So: to keep only this track's `model.py`, drop patch lines 172-217 and 352-358. Note that dropping
them also requires dropping `cfg.lr_theta` / `cfg.eps_theta` from `config.py`, or `build_model` will
reference fields that do not exist.

## What does NOT need to be in here, because it is already pushable

These paths are untracked or unfrozen and can be committed directly, without touching
`model_augmentation/` or `gantry_dynamic/`:

* `scripts/gantry/closed-loop-controller/transient-investigation/` - the nine probes and all their
  JSON artefacts. **This is the evidence** for every number in
  `tasks/overnight-2026-08-21-verdicts.md` and should be pushed.
* `scripts/gantry/closed-loop-controller/cl_train.py` - tracked and modified, but not frozen. Carries
  `CL_NOISE_CONSISTENT` (C5), `CL_NX_AUG`, `CL_NODES`.
* `tasks/overnight-2026-08-21-verdicts.md`, `tasks/handoffs/2026-08-21-*.md`,
  `docs/gantry-augmentation-problem-log.md`.

## The result this snapshot corresponds to

All at 520 updates, `na_nb = 17`, serial validation (`CL_CONCURRENT=0`):

| | ANN params | `nx_aug` | free run | ablation |
|-|-|-|-|-|
| untrained (D-072 gate) | | | `2.1866011034177349e-06` | |
| arm 1 | 600 | 2 | `1.379891e-06` | `1.0183x` decoration |
| width-matched control | 828 | 2 | `1.384274e-06` | `1.0169x` decoration |
| **arm 2** | 798 | **8** | **`3.795974e-07`** | **`5.2081x` load-bearing** |

Launch for arm 2, which is the configuration worth preserving:

```
AUG_LRU=1 AUG_LRU_B=0.377 ENC_WA_ZERO=1 CL_NX_AUG=8 AUG_LRU_NA_NB=17 \
CL_EPOCHS=2 CL_LR=1e-5 CL_ADAM_EPS=1e-16 CL_STRIDE=10 CL_ITS_PER_VAL=epoch \
CL_PROBE=0 CL_CONCURRENT=0 CL_FLOOR=0 CL_BURNIN=0 CL_CONS_FRAC=0 python -u cl_train.py
```

**Which of the six mechanisms in that line are actually necessary is NOT yet known.** That is the
task in `tasks/handoffs/2026-08-21-which-change-made-the-augmented-states-train.md`. Do not port any
of this into `model_augmentation/` before that ablation runs.

## Re-applying

```
git checkout 4cdb7c1
git apply tasks/snapshots/2026-08-21-augmented-states/patches/<file>.patch
```

Apply per file, in any order; they touch disjoint files. For a partial `model.py`, edit the patch to
drop the hunks listed above rather than applying it whole.

## Caveat on the copies

`files/` are the working-tree contents at 2026-08-21 11:50, which include the `AUG_LRU_NA_NB` gate
added that morning and verified as a no-op when unset
(`17 2 2.186601103417735e-06 rel dev 0.000e+00 PASS`, `runs/d072_noop_check.json`). They are a
record, not a build: the flattened names mean nothing imports from this folder.
