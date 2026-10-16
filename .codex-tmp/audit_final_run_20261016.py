"""Short audit probes; writes synthetic checkpoints only under .codex-tmp."""
import ast
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / '.codex-tmp' / 'final_run_audit_artifacts'
SCRATCH.mkdir(exist_ok=True)
os.environ['LOCALAPPDATA'] = str(SCRATCH)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/gantry')]

import numpy as np
import torch
import deepSI
from deepSI.fit_systems.encoders import SS_encoder_general_hf
from gantry_dynamic.data import servo_sigma
from gantry_dynamic.config import RunConfig
from model_augmentation.fit_systems.closed_loop import window_controller_index
from model_augmentation.fit_systems.interconnect import SSE_Interconnect_Composed

torch.set_num_threads(2)
paths = [ROOT / 'scripts/gantry/gantry_interconnect_dynamic.py']
paths += list((ROOT / 'scripts/gantry/gantry_dynamic').glob('*.py'))
paths += [ROOT / 'model_augmentation/fit_systems' / n for n in
          ('pre_encoder.py', 'closed_loop.py', 'interconnect.py', 'lbfgs_polish.py')]
paths += list((ROOT / 'Thesis-writeup/Code/Evaluation').glob('*.py'))
paths += [ROOT / 'scripts/gantry/blackbox-cl-trainable/train_bb_clmap.py']
for path in paths:
    compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
print('COMPILE_OK', len(paths), flush=True)

bbpath = ROOT / 'scripts/gantry/blackbox-cl-trainable/train_bb_clmap.py'
tree = ast.parse(bbpath.read_text(encoding='utf-8-sig'))
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
         and n.name in ('per_axis', 'ServoWeightedEncoder')]
ns = dict(np=np, torch=torch, SS_encoder_general_hf=SS_encoder_general_hf,
          servo_sigma=servo_sigma, window_controller_index=window_controller_index,
          args=SimpleNamespace(target='e'), __name__='__main__')
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(bbpath), 'exec'), ns)
Model = ns['ServoWeightedEncoder']
rng = np.random.default_rng(64)
records = [deepSI.System_data(u=rng.normal(size=(16, 2)),
                             y=rng.normal(size=(16, 3))) for _ in range(2)]
sdl = deepSI.System_data_list(records)
ry = [(np.zeros_like(sd.y), sd.y) for sd in records]
history, failure = [], []
model = Model(nx=2, na=3, nb=3)
model.set_objective([sd.y for sd in records], 3, ry, (history, failure))
model.fit(sdl, val_sys_data=sdl, n_its=2, epochs=1, batch_size=4,
          loss_kwargs=dict(nf=3, stride=1), validation_measure='sim-NRMSe',
          its_per_val=1, verbose=0, cuda=False)
print('BB_TINY_FIT_OK', 'history=', len(history), 'failure=', failure, flush=True)
print('BB_ACTUAL_NAME', model.name, flush=True)
print('BB_EVALUATOR_NAME', 'SS_encoder_general_hf_' + model.unique_code, flush=True)
assert model.name != 'SS_encoder_general_hf_' + model.unique_code

ev = ROOT / 'Thesis-writeup/Code/Evaluation/eval_common.py'
oldenv = os.environ.pop('THESIS_EVAL', None)
old = runpy.run_path(str(ev))
os.environ['THESIS_EVAL'] = 'final'
new = runpy.run_path(str(ev))
if oldenv is None:
    os.environ.pop('THESIS_EVAL', None)
else:
    os.environ['THESIS_EVAL'] = oldenv
print('CAMPAIGN_CACHE_COLLISION', old['CACHE'] == new['CACHE'],
      str(new['CACHE'] / 'preds_R1.npz'), flush=True)

import gantry_dynamic.obc_gantry as og
cfg = RunConfig(use_f64=True, obc_ref_successor=True)
norm = SimpleNamespace(x_mean=np.zeros(6), std_x=np.ones(6),
                       u_mean=np.zeros(3), std_u=np.ones(3))
q = rng.normal(size=(20, 3))
fake = (rng.normal(size=(20, 3)), q,
        rng.normal(size=(20, 6)), rng.normal(size=(20, 2)))
with patch.object(og, '_load_record_float64', return_value=fake):
    ref = og.build_reference_set(cfg, norm, nx_ann=2, files=['a.mat', 'b.mat'], verbose=False)
same = ((ref.record_ix[1:] == ref.record_ix[:-1]) &
        (ref.sample_ix[1:] == ref.sample_ix[:-1] + 1))
print('BIAS_STACK_COUNTS', 'coefficient_tuples=', ref.n,
      'evaluation_tuples=', int(same.sum()),
      'stored_successors=', len(ref.x_phys_next_rec), flush=True)

class Life:
    basis = None
    def refresh(self, v, epoch):
        self.basis = SimpleNamespace(vbar=v.clone())
    def validation_coefficient(self, ann):
        return torch.ones(1), {}

class Correction:
    space = 'tangent'
    vbar = torch.tensor([1.0])
    def expansion_point(self):
        return torch.tensor([2.0])
    def set_frozen(self, theta, vbar):
        self.vbar = vbar.clone()

corr = Correction()
fs = SimpleNamespace(obc=Life(), _ann_ix=0,
                     hfn=SimpleNamespace(obc_correction=corr, connected_blocks=[None]))
before = corr.vbar.clone()
SSE_Interconnect_Composed._sync_prediction_state_for_validation(fs, refresh_basis=False)
print('LOADED_BASIS_POINT_CHANGED', before.tolist(), '->', corr.vbar.tolist(), flush=True)

# Inspect native defaults and numerical sigma behavior without a production training run.
assert np.allclose(servo_sigma([np.array([[0., 1.], [2., 5.], [4., 9.]])], 0),
                   np.std([[0., 1.], [2., 5.], [4., 9.]], axis=0)[None])
print('SIGMA_IDENTITY_OK', flush=True)

# Exercise the real preflight helper's plumbing without loading production records.
import gantry_dynamic.data as gd_data
import gantry_dynamic.model as gd_model
import gantry_dynamic.controller as gd_ctrl
helpers = runpy.run_path(str(ROOT / 'scripts/gantry/thesis-results/test_ps2_pipeline.py'))
fake_sd = SimpleNamespace(u=np.zeros((40, 3)), y=np.zeros((40, 3)))
fake_data = SimpleNamespace(train_files=['train'], val_files=['val'], val_ckpt_data=[fake_sd])
fake_norm = SimpleNamespace(ystd=np.ones(3), std_u=np.ones(3))
fake_fs = SimpleNamespace(na=30, nb=30)
with patch.object(gd_data, 'load_datasets', return_value=fake_data), \
     patch.object(gd_data, 'compute_normalization', return_value=fake_norm), \
     patch.object(gd_model, 'build_model', return_value=fake_fs), \
     patch.object(gd_ctrl, 'build_controller_bank',
                  return_value=(SimpleNamespace(), [0, 0], [0., 0.])):
    try:
        helpers['build']('u')
    except ValueError as exc:
        print('EXISTING_PREFLIGHT_PLUMBING_FAILS', str(exc), flush=True)

print('AUDIT_PROBES_COMPLETE', flush=True)
