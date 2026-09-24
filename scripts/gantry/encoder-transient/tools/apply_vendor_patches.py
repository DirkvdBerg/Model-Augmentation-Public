"""Apply the path/record-list patches to the vendored copies (ET-001). Idempotent: a patch whose
marker is already present is skipped. Every patched line carries `# VENDORED:`.

Run once after copying:  conda run -n GraduationProject python tools/apply_vendor_patches.py
"""
import io
import os

ET = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
V = os.path.join(ET, 'vendor')


def patch(rel, old, new):
    path = os.path.join(V, rel)
    s = io.open(path, encoding='utf-8').read()
    if new in s:
        print(f'  skip (already patched)  {rel}')
        return
    n = s.count(old)
    assert n == 1, f'{rel}: expected exactly one match, found {n}'
    io.open(path, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
    print(f'  patched                 {rel}')


# 1. The package puts the VENDOR dir on sys.path, so `model_augmentation` is the vendored copy.
patch('gantry_dynamic/__init__.py',
      """from .config import REPO_ROOT as _REPO_ROOT

if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)""",
      """# VENDORED: the vendor dir, not the repo root, goes on sys.path, so `model_augmentation`
# resolves to the vendored copy next to this package (vendor/VENDORED.md).
_VENDOR = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
if _VENDOR not in _sys.path:
    _sys.path.insert(0, _VENDOR)""")

# 2. REPO_ROOT (data paths only) is five levels up from vendor/gantry_dynamic.
patch('gantry_dynamic/config.py',
      """REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, '..', '..', '..'))""",
      """# VENDORED: vendor/gantry_dynamic sits two levels deeper than scripts/gantry/gantry_dynamic.
REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, '..', '..', '..', '..', '..'))""")

# 3. The record lists as of D-197 (commit c82ff9b): the frictionless folder has only these 22.
patch('gantry_dynamic/data.py',
      """    'T13_lissajous.mat', 'T14_lissajous_yaw.mat',
    'TP1_telica_y000.mat', 'TP2_telica_y000_rev.mat',
    'TP3_telica_yp06.mat', 'TP4_telica_ym06.mat',
]""",
      """    'T13_lissajous.mat', 'T14_lissajous_yaw.mat',
]   # VENDORED: TP1-TP4 (D-206) removed; not in augmentation_ma50_b140-230_a6_z03 (c82ff9b list)""")
patch('gantry_dynamic/data.py',
      """    'V3_ysweep_Yp10.mat', 'V4_lissajous_Ym10.mat',
    'VP1_telica_ylow.mat', 'VP2_telica_yp12.mat',
]""",
      """    'V3_ysweep_Yp10.mat', 'V4_lissajous_Ym10.mat',
]   # VENDORED: VP1, VP2 (D-206) removed (c82ff9b list)""")
patch('gantry_dynamic/data.py',
      """    'E3_aprbs_above.mat', 'E4_multisine_off.mat',
    'EP1_telica_test.mat',
]""",
      """    'E3_aprbs_above.mat', 'E4_multisine_off.mat',
]   # VENDORED: EP1 (D-206) removed (c82ff9b list)""")

# 4. truth.py: vendor on sys.path, data from the repo, cache READ from the D-197 location.
patch('transient/truth.py',
      """REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))""",
      """# VENDORED: vendor/transient is two levels deeper; the vendor dir replaces REPO on sys.path.
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))""")
patch('transient/truth.py',
      """CACHE = os.path.join(HERE, '..', 'cache')""",
      """CACHE = os.path.join(REPO, 'scripts', 'gantry', 'transient', 'cache')  # VENDORED: read-only reuse""")

# 5. enc_common.py: vendor on sys.path, outputs inside encoder-transient/outputs.
patch('transient/enc_common.py',
      """REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))
sys.path.insert(0, HERE)""",
      """# VENDORED: the vendor dir (gantry_dynamic, model_augmentation, the entry file) replaces REPO.
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, HERE)""")
patch('transient/enc_common.py',
      """FIGDIR = os.path.join(HERE, '..', 'figure')
OUTDIR = os.path.join(HERE, '..', 'results')""",
      """FIGDIR = os.path.join(HERE, '..', '..', 'outputs', 'vendored_r0', 'figure')   # VENDORED
OUTDIR = os.path.join(HERE, '..', '..', 'outputs', 'vendored_r0', 'results')  # VENDORED""")
# The entry CFG now names the friction folder (D-207); D-197 ran on the frictionless one.
patch('transient/enc_common.py',
      """    cfg = replace(cfg, device='cpu', compile_mode=None, save_flag=False, **(overrides or {}))""",
      """    # VENDORED: pin the D-197 dataset; the working-tree entry file names the friction folder.
    cfg = replace(cfg, mode='augmentation_ma50_b140-230_a6_z03')
    cfg = replace(cfg, device='cpu', compile_mode=None, save_flag=False, **(overrides or {}))""")
print('done')

# ==== Telica vendor (ET-006): telica/vendor_telica is two levels deeper than telica-real ====
TV = os.path.join(ET, 'telica', 'vendor_telica')


def patch_tv(rel, old, new):
    global V
    V_saved, V = V, TV
    try:
        patch(rel, old, new)
    finally:
        V = V_saved


patch_tv('tr_env.py',
         """REPO = os.path.dirname(TR)""",
         """REPO = os.path.abspath(os.path.join(TR, '..', '..', '..', '..', '..'))   # VENDORED: repo root""")
patch_tv('tr_env.py',
         """OUTPUTS = os.path.join(TR, 'outputs')""",
         """OUTPUTS = os.path.abspath(os.path.join(TR, '..', '..', 'outputs', 'telica'))   # VENDORED""")
patch_tv('vendor/gantry_dynamic/config.py',
         """REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, '..', '..', '..'))""",
         """# VENDORED: vendor_telica/vendor/gantry_dynamic sits three levels deeper than telica-real/vendor.
REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, '..', '..', '..', '..', '..', '..', '..'))""")
print('telica done')

# ==== Telica arm (ET-008 c): smoke update count and the G2-T encoder hook ====
patch_tv('pipeline/telica_augment.py',
         """                      n_its=2, epochs=1,""",
         """                      n_its=int(os.environ.get('SMOKE_N_ITS', '2')), epochs=1,   # VENDORED-ET: ET-008 budget""")
patch_tv('pipeline/telica_augment.py',
         """    fit_sys.simulator = build_closed_loop_telica(""",
         """    if os.environ.get('ET_G2_ENCODER'):                              # VENDORED-ET: ET-008 (c)
        sys.path.insert(0, os.path.join(TR, '..'))
        import et_hook
        fit_sys = et_hook.swap(fit_sys, cfg, norm, hp, os.environ['ET_G2_ENCODER'])
    fit_sys.simulator = build_closed_loop_telica(""")
print('telica arm done')
patch_tv('vendor/real_data_verification/telica_loader.py',
         """_ROOT     = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))""",
         """_ROOT     = os.path.normpath(os.path.join(_HERE, *(['..'] * 7)))   # VENDORED: repo root from vendor_telica""")
print('telica loader done')
patch_tv('pipeline/telica_augment.py',
         """        param_init_detune=None,
    )
    arm = os.environ.get('OBC_ARM')""",
         """        param_init_detune=None,
    )
    if os.environ.get('ET_NA_NB'):                                   # VENDORED-ET: ET-012 window
        cfg = replace(cfg, na_nb_override=int(os.environ['ET_NA_NB']))
    arm = os.environ.get('OBC_ARM')""")
print('telica window done')
# Correction (R025): the ET-008 marker above was placed mid-line and commented out nf_seconds.
patch_tv('pipeline/telica_augment.py',
         """                      n_its=int(os.environ.get('SMOKE_N_ITS', '2')), epochs=1,   # VENDORED-ET: ET-008 budget nf_seconds=float(os.environ.get('SMOKE_NF_S', '0.005')),""",
         """                      # VENDORED-ET: ET-008 budget (SMOKE_N_ITS); marker moved off the code line (R025)
                      n_its=int(os.environ.get('SMOKE_N_ITS', '2')), epochs=1,
                      nf_seconds=float(os.environ.get('SMOKE_NF_S', '0.005')),""")
print('telica smoke line fixed')
