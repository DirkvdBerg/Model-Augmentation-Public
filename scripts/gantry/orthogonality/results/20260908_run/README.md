# What is in this directory, and what each artifact does NOT cover

Written 2026-09-08 after review. Read
[the report](../../documentation/proofs/trajectory-orthogonality-verification-2026-09-08.md)
first; this file only says what each saved file is.

| File | Covers | Does NOT cover |
|-|-|-|
| `gantry_nominal.json` | P0, V0, V1, V2 on checkpoint 81757, 2 windows (records T1, T9), float64, eager CPU. 21 checks, 21 passed | **V3, V4, V5, V6, V7, which were requested and never completed.** This file was written by a build that PREDATES the `stages_requested` / `stages_completed` / `stages_incomplete` fields, so its `summary` says "0 skipped" without saying that three requested stages are missing. Its null `V3`/`V4_V5`/`V7` fields are the only trace. Later runs record the stage status explicitly and the entry point exits non-zero when a requested stage does not finish |
| `gantry_2windows_console.log` | the console of that run, including the point at which V3 stalls | anything after the stall |
| `gantry_4windows_console.log` | an earlier 4-window run that reached V3; the source of the four-window numbers quoted in the report and of the ~13 s per-JVP cost figure | it has no JSON: that run predates incremental snapshotting |
| `small_sympy_console.log` | the SymPy reference suite, 57 checks, all passing, including the `A5b` check that calls the PRODUCTION `compressed_nuisance_basis` | the MATLAB twin does not cover the compressed numerical construction, so the 23 shared checks are not two-engine verification of it |
| `small_matlab_console.log` | the MATLAB twin, 23 checks, all passing | as above |

There is **no perturbed/noisy JSON**: that arm has never been executed.

## Superseded by `../20260908_diag2/`

`20260908_diag2/gantry_nominal.json` is the run that COMPLETED the ladder: 41 checks, 41 passed,
0 failed, exit 0, stages V3/V4/V6/V7 all present, plus a per-operation ledger under
`V3_operations`. Read that one for the gantry geometry. The files here remain as the record of
the attempts that did not complete, and of the P0/V0/V1/V2 rows on a different window pair.

The structured results for the two engines and their comparison live next to the scripts, in
`../../code/trajectory_ownership_geometry/`.
