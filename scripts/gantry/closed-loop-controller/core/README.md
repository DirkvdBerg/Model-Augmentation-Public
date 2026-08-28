# `core/` -- SUPERSEDED 2026-08-28. Reference only, not on any live path.

> The training path no longer imports this folder. `gantry_interconnect_dynamic.py` now uses
> `scripts/gantry/gantry_dynamic/controller.py`, one module holding the same five layers
> (FP constants, `ruleOfThumb`, tf-to-ss, `RECORD_Y_OP`, bank + `build_closed_loop`) with no
> `sys.path` insert and no diagnostics on the import chain. Verified bit-identical: the bank
> built over all 22 records at 4 kHz matches this chain's to `max|new - old| = 0` on `M_state`,
> `M_error`, `ystd` and `stdu`, with the same 9 distinct controllers and the same `physical_D`.
> The files below are kept so that equivalence can be re-checked. Do not import them.
>
> Everything after this line describes the pre-2026-08-28 arrangement.

# `core/` -- the closed-loop modules the TRAINING path needs, and nothing else

Created 2026-08-26. This folder is a **copy**, not a move. The originals stay in the parent
directory, where 48 other scripts import them by bare name.

## Why it exists

`scripts/gantry/closed-loop-controller/` holds 49 `.py` files, of which 43 are diagnostics and
probes. Training needs six. `gantry_interconnect_dynamic.py` puts **this** folder on `sys.path`
rather than the parent, so the training entry point does not sit downstream of the diagnostics.

## The dependency chain

```
cl_pipeline.py        build_closed_loop          <- the only entry point used by training
  cl_controller.py    build_controller_bank      Y_op -> which controller row per record
    loss_variants.py  controller_ss
      p2_rate_compare.py  build_cfb_at
      so_filter.py
        verify_controller.py  M_op, P, C_DAMP, K_STIFF, TS, cnorm_*
```

`verify_controller.py` terminates the chain (only numpy/scipy/matplotlib beyond it).
`cl_pipeline.py` additionally imports `ClosedLoopSimulator` from
`model_augmentation/fit_systems/closed_loop.py`, which is framework code and is NOT copied here.

## Provenance, 2026-08-26

| file | original sha256 (16) | copy sha256 (16) | status |
|-|-|-|-|
| `cl_pipeline.py` | `1473f2a58617b94e` | `1473f2a58617b94e` | identical |
| `cl_controller.py` | `90ff140094f5fe0c` | `90ff140094f5fe0c` | identical |
| `loss_variants.py` | `f7aa7d725a466acd` | `f7aa7d725a466acd` | identical |
| `p2_rate_compare.py` | `d5fa1d90f4e06b46` | `d5fa1d90f4e06b46` | identical |
| `so_filter.py` | `23c4cd1f8c24fc28` | `23c4cd1f8c24fc28` | identical |
| `verify_controller.py` | `1ea446d70bba8c79` | `a093d526a37ba16c` | **one line changed, see below** |

### The one deliberate difference

`verify_controller.py` derives the repo root from its own location:

```python
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))     # original
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))  # this copy
```

`core/` is one level deeper than the parent, so three `..` would resolve `REPO` to
`scripts/gantry` and `TRAJ` to a directory that does not exist. Verified after the fix:
`TRAJ` resolves to `data/gantry/matlab/trajectory/augmentation` with 22 `.mat` files present.

Note that `verify_controller.py` also writes figures to `os.path.join(HERE, 'figures')`, which in
this copy means `core/figures/`. That path is only used when the file is run as a script, which is
not what this folder is for.

## Verified self-contained

Imported with **only** `core/` and the repo root on `sys.path`, the parent folder deliberately
absent. All six modules resolved from `core/`, no module was loaded from the parent, and
`build_closed_loop` came from `core/cl_pipeline.py`.

## The cost of a copy, stated plainly

There are now two versions of six files. **They can drift.** If you change any of the originals,
this folder does not follow, and the training path will silently keep using the old behaviour.

Guard: re-run the hash comparison above. Any row that changes from `identical` to differing, other
than `verify_controller.py`, means the copy is stale.

```
cd scripts/gantry/closed-loop-controller
for f in cl_pipeline.py cl_controller.py loss_variants.py p2_rate_compare.py so_filter.py; do
  a=$(sha256sum "$f" | cut -c1-16); b=$(sha256sum "core/$f" | cut -c1-16)
  [ "$a" = "$b" ] || echo "STALE: $f"
done
```

`p2_rate_compare.py` is the only one of the six currently dirty against git HEAD, and its diff is
an env-gated `CL_RATES` default (unchanged when unset) plus a `__main__` reporting loop, so
`build_cfb_at` is HEAD behaviour either way.

## Consumers

Only `scripts/gantry/gantry_interconnect_dynamic.py` points here. `cl_train.py` and the 43
diagnostics continue to use the originals in the parent directory.
