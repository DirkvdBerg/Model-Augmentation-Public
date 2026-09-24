# Vendored files

Copied 2026-09-23 22:12. Repository HEAD `1ec57c1` (the handoff names `a97a577`, an ancestor
of HEAD). Working tree dirty elsewhere; each source below was clean at HEAD (`git status`
reported nothing for it) at copy time. Edits are marked `# CHANGED (vendor, OA-001)`.

| vendored | source | md5 of source at copy | edited |
|-|-|-|-|
| `obc.py` | `model_augmentation/fit_systems/obc.py` | 9b9e12eee1f53012987bf7c38898885a | no |
| `gantry_dynamic/__init__.py` | `scripts/gantry/gantry_dynamic/__init__.py` | 224cc43f77adf7f095495be822f86af2 | no |
| `gantry_dynamic/config.py` | `scripts/gantry/gantry_dynamic/config.py` | 89f3468bdb78bbbd1b3013074aa7ef51 | repo-root depth |
| `gantry_dynamic/data.py` | `scripts/gantry/gantry_dynamic/data.py` | 11032368e27854b8dff74dd2df12380d | file lists pinned to 14/4/4; `delta_a` optional |
| `gantry_dynamic/obc_gantry.py` | `scripts/gantry/gantry_dynamic/obc_gantry.py` | d69388c6573c04c2e4548c59856e7f05 | imports `vendor.obc`; absorber fields optional |
| `gantry_dynamic/baselines.py` | same folder | e38ee0205e4bb82a3dac8eb6f650be08 | no |
| `gantry_dynamic/controller.py` | same folder | cb8bac1f678d130b9d3af81ec0653a1d | no |
| `gantry_dynamic/diagnostics.py` | same folder | b6fba20558c7a28a384c4101d05549e3 | no |
| `gantry_dynamic/evaluation.py` | same folder | 6e82bd6d59bc2d1dd6829acfb33b33a8 | no |
| `gantry_dynamic/model.py` | same folder | 07a9057ee4ed88dc847c86079f4e2444 | no |
| `gantry_dynamic/obc_diagnostics.py` | same folder | 6275f29cfe7127820767a905130f8e1a | no |
| `gantry_dynamic/training.py` | same folder | 1f6619cf1d172d7a5e23bd8fcba8894a | no |
| (config.py, second edit) | | | `save_dir` -> `outputs/sim/` inside this folder |
| `../measure/measure_delta.py` | `scripts/gantry/orthogonal-by-construction/baseline-and-added-dynamics/measure_delta_orthogonality.py` | ff722fd37881fe6ff87e9e25072f40e4 | dataset parameter, outputs path, `m_diff` own scale |
| `../tools/watchdog.ps1` | `telica-real/tools/watchdog.ps1` | f85b4a85327626db83afde9b52a730c5 | no |

Not vendored, imported read only from production: `model_augmentation/fit_systems/blocks.py`
(`Reduced_Gantry_State_Block`, the production Jacobian's step) and
`model_augmentation/systems/gantry_ss.py` (true parameters, `P`).
