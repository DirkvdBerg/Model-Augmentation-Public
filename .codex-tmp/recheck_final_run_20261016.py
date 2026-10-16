"""Short audit probes; writes synthetic checkpoints only under .codex-tmp."""
import ast
import json
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
          servo_sigma=servo_sigma, window_controller_index=window_controller_index, json=json,
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
hist_path = SCRATCH / "recheck_bb_history.json"
model.set_objective([sd.y for sd in records], 3, ry, (history, failure), str(hist_path))
model.fit(sdl, val_sys_data=sdl, n_its=2, epochs=1, batch_size=4,
          loss_kwargs=dict(nf=3, stride=1), validation_measure='sim-NRMSe',
          its_per_val=1, verbose=0, cuda=False)
print('BB_TINY_FIT_OK', 'history=', len(history), 'failure=', failure, flush=True)
print('BB_ACTUAL_NAME', model.name, flush=True)
saved_history = json.loads(hist_path.read_text())
assert len(saved_history['history']) == len(history) >= 3
assert not failure and saved_history['failure'] is None
assert (SCRATCH / 'deepSI/checkpoints' / (model.name + '_best.pth')).exists()
print('BB_CHECKPOINT_AND_DURABLE_HISTORY_OK', len(history), flush=True)

ev = ROOT / 'Thesis-writeup/Code/Evaluation/eval_common.py'
oldenv = os.environ.pop('THESIS_EVAL', None)
old = runpy.run_path(str(ev))
os.environ['THESIS_EVAL'] = 'final'
new = runpy.run_path(str(ev))
if oldenv is None:
    os.environ.pop('THESIS_EVAL', None)
else:
    os.environ['THESIS_EVAL'] = oldenv
assert old['CACHE'] != new['CACHE']
old['check_identity']('R1_val', {'job': np.array(old['RUNS']['R1'])})
try:
    old['check_identity']('R1_val', {'job': np.array(-1)})
except ValueError:
    pass
else:
    raise AssertionError('mismatched cache accepted')
print('CAMPAIGN_CACHE_AND_IDENTITY_OK', flush=True)
for name in ('eval_greybox.py', 'eval_overlap.py', 'eval_negation.py'):
    source = (ev.parent / name).read_text(encoding='utf-8-sig')
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == '_sync_prediction_state_for_validation'
                   for n in ast.walk(ast.parse(source)))
print('INFERENCE_NO_BASIS_REBUILD_OK', flush=True)

import gantry_dynamic.obc_gantry as og
cfg = RunConfig(use_f64=True, obc_ref_successor=True)
norm = SimpleNamespace(x_mean=np.zeros(6), std_x=np.ones(6),
                       u_mean=np.zeros(3), std_u=np.ones(3))
q = rng.normal(size=(20, 3))
fake = (rng.normal(size=(20, 3)), q,
        rng.normal(size=(20, 6)), rng.normal(size=(20, 2)))
with patch.object(og, '_load_record_float64', return_value=fake):
    ref = og.build_reference_set(cfg, norm, nx_ann=2, files=['a.mat', 'b.mat'], verbose=False)
source = ast.parse((ev.parent / 'eval_overlap.py').read_text(encoding='utf-8-sig'))
branch = next(n for n in ast.walk(source) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Attribute)
              and n.test.left.attr == 'x_phys_next_rec')
f_true = torch.zeros((ref.n,6), dtype=torch.float64)
Js = torch.arange(ref.n*6*2, dtype=torch.float64).reshape(ref.n*6,2)
local = dict(torch=torch, np=np, ref=ref, norm=norm, row_ix=list(range(6)), f_true=f_true, Js=Js)
exec(compile(ast.Module(body=branch.body, type_ignores=[]), '<bias-stack>', 'exec'), local)
assert local['Phi'].shape[0] == local['xi'].shape[0] == ref.n*6
assert torch.equal(local['xi'], ref.x_phys_next_rec.reshape(-1))
print('COMPLETE_NORMALIZED_BIAS_STACK_OK', ref.n, 'tuples', flush=True)

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

# Exercise actual helper plumbing with mocked data/model, without a production rollout.
import gantry_dynamic.data as gd_data
import gantry_dynamic.model as gd_model
import gantry_dynamic.controller as gd_ctrl
helpers = runpy.run_path(str(ROOT / 'scripts/gantry/thesis-results/test_ps2_pipeline.py'))
fake_data = SimpleNamespace(train_files=['train'], val_files=['val'], val_ckpt_data=[],
                            e_train=[np.ones((40,3))], e_val=[np.ones((40,3))*2])
fake_norm = SimpleNamespace(ystd=np.ones(3), std_u=np.ones(3))
fake_fs = SimpleNamespace(na=30, nb=30)
with patch.object(gd_data, 'load_datasets', return_value=fake_data), \
     patch.object(gd_data, 'compute_normalization', return_value=fake_norm), \
     patch.object(gd_model, 'build_model', return_value=fake_fs), \
     patch.object(gd_ctrl, 'build_closed_loop', return_value=SimpleNamespace()) as closed:
    helpers['build']('u')
    assert closed.call_args.kwargs['e_train'] is fake_data.e_train
    assert closed.call_args.kwargs['e_val'] is fake_data.e_val
print('PREFLIGHT_SERVO_ERROR_PLUMBING_OK', flush=True)
print('RECHECK_PROBES_COMPLETE', flush=True)
# Reproduce the preflight lifecycle failure on the real pre-fit with a tiny synthetic encoder.
class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Wb_psi_y=torch.nn.Parameter(torch.zeros((6,6),dtype=torch.float64))
        self.Wb_psi_u=torch.nn.Parameter(torch.zeros((6,6),dtype=torch.float64))
        self.Wa_psi_y=torch.nn.Parameter(torch.ones((2,6),dtype=torch.float64))
        self.Wa_psi_u=torch.nn.Parameter(torch.ones((2,6),dtype=torch.float64))
        self.net=torch.nn.Sequential(torch.nn.Linear(12,8,dtype=torch.float64))
        with torch.no_grad():
            self.net[-1].weight[6:].fill_(0.1) # state after S1, rather than at initialization
    def forward(self,uh,yh):
        u,y=uh.flatten(1),yh.flatten(1)
        return self.net(torch.cat((u,y),1))+torch.cat((y@self.Wb_psi_y.T+u@self.Wb_psi_u.T,
                                                   y@self.Wa_psi_y.T+u@self.Wa_psi_u.T),1)
enc=Encoder()
pre_cfg=SimpleNamespace(nx_phys=6,nx_ann=2,stride=1,dtype_pt=torch.float64,batch_size=4,
                        encoder_fit_epochs=1,encoder_fit_lr=1e-3,seed=1)
pre_fs=SimpleNamespace(encoder=enc,na=2,nb=2,_mats_ix=None,
    norm=SimpleNamespace(u0=np.zeros(3),ustd=np.ones(3),y0=np.zeros(3),ystd=np.ones(3)),
    simulator=SimpleNamespace(bank=None,train_ctrl_rows=[0]),
    hfn=lambda x,u,*args:(x[:,:3],x),_sync_prediction_state_for_validation=lambda:None)
pre_fs.hfn.output_only=lambda x:x[:,:3]
pre_data=SimpleNamespace(train_data=SimpleNamespace(sdl=[SimpleNamespace(u=np.zeros((5,3)),y=np.zeros((5,3)))]))
pre_norm=SimpleNamespace(x_mean=np.zeros(6),std_x=np.ones(6))
def rollout(recorder,output,un,yn,x0,bank,rows,**kw):
    for i in range(un.shape[1]):
        recorder(x0,un[:,i])
    return yn, None, None
with patch('model_augmentation.fit_systems.closed_loop.closed_loop_rollout', side_effect=rollout):
    try:
        gd_model._prefit(pre_fs,pre_cfg,pre_data,pre_norm)
    except RuntimeError as exc:
        assert 'added-state output row' in str(exc)
        print('REPRODUCED_PREFLIGHT_F3_FAILURE_AFTER_S1',str(exc),flush=True)
    else:
        raise AssertionError('expected initialized-row invariant to reject already trained encoder')
os.environ['THESIS_ARCH']='resnet'
with patch.object(gd_data,'load_datasets',return_value=fake_data), \
     patch.object(gd_data,'compute_normalization',return_value=fake_norm), \
     patch.object(gd_model,'build_model',return_value=fake_fs), \
     patch.object(gd_ctrl,'build_closed_loop',return_value=SimpleNamespace()):
    c,_,_,_=helpers['build']('u')
    assert os.environ['THESIS_ARCH']=='mlp'
    print('REPRODUCED_LIVE_COMPARISON_ARCHITECTURE_OVERRIDE',os.environ['THESIS_ARCH'],flush=True)
print('RECHECK_COMPLETE',flush=True)

