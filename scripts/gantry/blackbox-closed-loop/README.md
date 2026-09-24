# blackbox-closed-loop

The full black box (Jan's ANN-SS: encoder, f, h all MLPs) trained and selected through the SAME
known controller, in the SAME residual closed-loop form, on the SAME records, split, rate, horizon,
burn-in and selection metric as the grey box, so the two share one table.
Handoff: `tasks/handoffs/2026-09-23-blackbox-closed-loop.md`. Results: `REPORT.md`.
Decisions and pre-registered thresholds: `DECISIONS.md` (BB-xxx). Runs: `RUNS.md`.

## How it is built (BB-003)
The black box is the grey box's own fit-system class, `SSE_Interconnect_Composed`, with
`hfn = BlackBoxHF` (deepSI `hf_net_default` plus `output_only = h(x)`) and a deepSI MLP encoder.
Loss, burn-in, closed-loop simulator, `fit()`, validation and checkpoint selection are therefore
the grey box's code. The grey box's configuration is read at run time from
`scripts/gantry/gantry_interconnect_dynamic.py` (`CFG`), never retyped.

## Layout
| Path | What |
|-|-|
| `adapter/bbcl.py` | bootstrap: vendored imports, `load_cfg()`, G0 checks |
| `adapter/bb_model.py` | `BlackBoxHF`, `build_blackbox(cfg, data, norm, arm)`, BLA/FRF init |
| `score/cl_score.py` | closed-loop scorer (proved equal to the grey box's selection number, G1) |
| `score/g0_infra.py`, `g1c_original.py`, `g1_adapter.py`, `g2_transfer.py` | gates G0 to G2 |
| `init/g3_init.py` | G3: stability and gradients at initialisation, per arm |
| `runners/train_bb.py`, `runners/run_bb_arm.sh` | the server job (G4 smoke uses the same script) |
| `vendor/` | unmodified copies plus marked path patches; provenance in `vendor/VENDORED.md` |
| `tools/run.sh`, `tools/watchdog.ps1` | every local run: RAM gate (4 GB) then watchdog |
| `outputs/<run>/` | logs, `resources.csv`, JSON results |

## Running
Local (always through the watchdog, one process at a time):
`bash scripts/gantry/blackbox-closed-loop/tools/run.sh <run_name> <script> [args]`.
Server: `BB_ARM=<arm> sbatch scripts/gantry/blackbox-closed-loop/runners/run_bb_arm.sh`
(not submitted by the building session; see `REPORT.md` G5 for the arms and the cost).
