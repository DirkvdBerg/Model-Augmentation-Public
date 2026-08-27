# Snapshot: the working augmentation implementation, taken immediately before a deliberate reset

**Base commit**: `4cdb7c1` (branch `Augmentation`). Taken 2026-08-22, on the user's instruction, so
that `model_augmentation/`, `scripts/gantry/gantry_dynamic/` and
`scripts/gantry/gantry_interconnect_dynamic.py` could be reset to a clean baseline before the BLA
initialisation work begins.

**This is the ONLY record of that implementation.** The changes were never committed and never
pushed, by design. If you need the `3.795974e-07` result back, it comes from here.

## Why the reset happened

The production files had accumulated 722 insertions of experimental machinery, `model.py` alone
carrying +334 lines and **ten env gates inside the production build path**. The codebase already
contains the argument against that, written into `model.py` when the `ANN_LIPSCHITZ` hook was
removed on 2026-08-13: *"An experiment hook belongs in the experiment script, not in every run's
build path."* The next task is a clean, literature-founded BLA initialisation
(`tasks/handoffs/2026-08-22-bla-initialisation-of-the-augmented-states.md`), and the user's decision
was to build it on a clean baseline rather than on top of the machinery it replaces.

## What is here

`files/` holds verbatim copies, path separators flattened to `_`, **verified byte-identical to the
working tree at the moment of the snapshot** (11 of 11). `patches/` holds one `git diff` per tracked
file against `4cdb7c1`, **all 8 verified to apply clean** against that commit in a scratch tree.

| file | + lines | owner | contents |
|-|-|-|-|
| `model_augmentation/fit_systems/closed_loop.py` | 104 | this track | `closed_loop_rollout` (D-147, the stabilized-PEM rollout), `xc = 0` windowing (D-142), and **`window_starts` / `make_window_tensors`, which do NOT exist at `4cdb7c1`** |
| `model_augmentation/fit_systems/pre_encoder.py` | 47 | this track | `linear_encoder_init_aug`, the `W^b`/`W^a` split (D-130). Carries the false "no literature source" comment at line 422 corrected by **D-152** |
| `scripts/gantry/gantry_dynamic/model.py` | 334 | **MIXED** | `AugLRUBypass`, `lru_band_from_artifact`, the `AUG_LRU` block, `ENC_WA_ZERO`, `AUG_LRU_NA_NB`, `AUG_LRU_FREEZE`, `ENC_WA_FREEZE` (this track) AND `find_log_params` / `split_param_group` / the `cfg.lr_theta` call (**other session, P1/P1-e**) |
| `scripts/gantry/gantry_dynamic/config.py` | 30 | **other session** | P1/P1-e: `lr_theta`, `eps_theta` |
| `scripts/gantry/gantry_dynamic/evaluation.py` | 138 | **other session** | `orth_frac` meter, `param_init_detune` |
| `scripts/gantry/gantry_dynamic/orth_penalty.py` | 16 | **other session** | D-111 penalty basis |
| `scripts/gantry/gantry_dynamic/rezero_gate.py` | 63 | uncertain | the ReZero gate. Note **ReZero is the ML-side named version of the zeroed-readout initialisation** the BLA-init literature uses (arXiv:2003.04887), so this may become relevant again |
| `scripts/gantry/gantry_interconnect_dynamic.py` | 28 | this track | the augmentation training entry point |
| `bounded_integral_block.py`, `lipschitz.py`, `passive_ph_block.py` | untracked | uncertain | copied to `files/` only; no patch, nothing to diff against. **Left in place by the reset**, since `git checkout` does not touch untracked files and other scripts import `lipschitz` and `passive_ph_block` |

## WARNING: 247 of the 722 lines are another session's uncommitted work

`config.py`, `evaluation.py`, `orth_penalty.py` and the `split_param_group` hunks of `model.py`
carry the P1/P1-e work and the `orth_frac` meter from a session this one cannot attribute. The reset
discarded them from the working tree. **They are recoverable only from here.** Tell that session
before it next runs anything, and restore its four files first if it needs them.

## What breaks after the reset, and it does not fail loudly

* **`cl_train.py` will not import.** It does
  `from model_augmentation.fit_systems.closed_loop import window_starts, make_window_tensors`, and
  neither function exists at `4cdb7c1`. Every arm and every probe in
  `closed-loop-controller/transient-investigation/` goes with it.
* **The metric changes value silently.** `closed_loop_free_run_rms` and `closed_loop_rollout` DO
  exist at `4cdb7c1`, but in their pre-D-147 form. So anything that still runs returns numbers that
  are NOT comparable with `3.795974e-07`, the D-072 gate `2.1866011034177349e-06`, or any ablation
  ratio in `tasks/ablation-2026-08-22-what-earned-its-place.md`. **This is the dangerous one**: it
  looks like it works.
* **Existing checkpoints become unloadable.** A gated checkpoint loads only into a gated build
  (`docs/aug-lru-implementation.md` section 8), so
  `SSE_Interconnect_MultipleShooting_Z37aYA_best.pth` (seed 1, `4.8867311476e-07`) needs `model.py`
  restored from here.
* **Wave 1 cannot run.** `runners/run_ablation_wave{1,2}.sh` depend on all of the above.

## How to restore

Whole implementation:

```
git checkout 4cdb7c1 -- <the eight tracked paths>     # if the tree has moved on
for p in tasks/snapshots/2026-08-22-working-implementation/patches/*.patch; do
    git apply --check "$p" && git apply "$p"
done
```

Per file, in any order; they touch disjoint files. For a partial `model.py` the hunk split is in
`../2026-08-21-augmented-states/MANIFEST.md`, which gives patch line ranges for the this-track and
other-session hunks separately.

If a patch will not apply because the base has moved, use the verbatim copy in `files/` instead: the
flattened name maps back by replacing `_` with `/` up to the file extension.

## The result this snapshot corresponds to

All at 520 updates, `na_nb = 17`, serial validation, free-run validation RMS on V1-V4:

| | `nx_aug` | free run | ablation |
|-|-|-|-|
| untrained (D-072 gate) | | `2.1866011034177349e-06` | |
| arm 1 | 2 | `1.379891e-06` | `1.0183x` decoration |
| width-matched control | 2 | `1.384274e-06` | `1.0169x` decoration |
| **arm 2, seed 0** | **8** | **`3.795974e-07`** | **`5.2081x` load-bearing** |
| **arm 2, seed 1** | **8** | **`4.8867311476e-07`** | **`4.5807x` load-bearing** |

Launch line for arm 2, the configuration worth preserving:

```
AUG_LRU=1 AUG_LRU_B=0.377 ENC_WA_ZERO=1 CL_NX_AUG=8 AUG_LRU_NA_NB=17 \
CL_EPOCHS=2 CL_LR=1e-5 CL_ADAM_EPS=1e-16 CL_STRIDE=10 CL_ITS_PER_VAL=epoch \
CL_PROBE=0 CL_CONCURRENT=0 CL_FLOOR=0 CL_BURNIN=0 CL_CONS_FRAC=0 python -u cl_train.py
```

**Which of those mechanisms were necessary was never established.** The 16-arm factorial designed to
answer it (`runners/run_ablation_wave1.sh`) was never submitted. Under the BLA decision most of its
arms are moot; the two questions that survive are F1 (does a plain latent-row set reach e-7 with no
`A_aa` at all) and F3a/b/c under noise (the `W^a` question). Both run with `AUG_LRU` OFF.
