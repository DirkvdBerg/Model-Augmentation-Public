# Vendored code (CN-003)

Copied 2026-09-24 from the WORKING TREE of branch `Augmentation`, HEAD `1ec57c1` (the handoff
names `a97a577`; `telica-real/` was committed in `380350c` between the two). Tree dirty: these
copies may differ from any commit. Edits here are marked `# CLOSED-LOOP-NOISE:` inline
(`grep -rn "CLOSED-LOOP-NOISE" scripts/gantry/closed-loop-noise/vendor`); the copy's own earlier
edits keep their `# TELICA-REAL:` markers (see `telica_real/vendor/VENDORED.md`).

| Vendored path | Source path | Commit | State | Date |
|-|-|-|-|-|
| `telica_real/tr_env.py`, `data_index.py`, `rate.py` | `telica-real/` | 1ec57c1 | working tree, dirty | 2026-09-24 |
| `telica_real/params/`, `friction/`, `controller/`, `baseline/` | `telica-real/` (same names) | 1ec57c1 | working tree, dirty | 2026-09-24 |
| `telica_real/vendor/` (framework, gantry_dynamic, common, loader) | `telica-real/vendor/` | 1ec57c1 | working tree, dirty | 2026-09-24 |

What is used from the copy: the identified controller `controller/telica_sos_identified.npz` and
`controller/telica_bank.physical_ss` (G3 of telica-real), the attempt-2 plant
`baseline/recovered_params_a2.json` with `friction/karnopp.GantryFrictionBlock` (tanh, G4), the
simulator `baseline/cl_sim.py`, the split `data_index.py`, and the loader
`vendor/real_data_verification/telica_loader.py`.

Edits log (every `# CLOSED-LOOP-NOISE:` marker):
- `telica_real/tr_env.py`: `REPO` is five levels up; `OUTPUTS` is `closed-loop-noise/outputs`.
- `telica_real/vendor/real_data_verification/telica_loader.py`: `_ROOT` seven levels up;
  `load_telica_log_cl(path, dtype, pre_motion_ms=50.0, engine='python')`, `pre_motion_ms=None`
  = no trimming (default behaviour unchanged, checked in G0), extra keys `motion_idx` and `e`.

MATLAB generators: see `matlab/VENDORED_MATLAB.md` (G5).
