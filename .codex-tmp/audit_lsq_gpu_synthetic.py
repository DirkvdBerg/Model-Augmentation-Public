import ast,contextlib,io,json,os,runpy,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(root/'scripts/gantry')]
torch.set_num_threads(2)
print('GPU',torch.cuda.get_device_name(0),flush=True)
from gantry_dynamic.model import least_squares_start
from gantry_dynamic.config import RunConfig,config_json_dict
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
paths=[root/'scripts/gantry/gantry_dynamic'/s for s in ['model.py','config.py']]+[root/'scripts/gantry/gantry_interconnect_dynamic.py',root/'scripts/gantry/thesis-results/test_encoder_lsq.py']
for p in paths: compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
print('COMPILE_OK',len(paths),flush=True)
g=torch.Generator().manual_seed(71)
for na in (0,2,8):
 for offsets in (False,True):
  kw=dict(u_mean=np.ones(3),std_u=np.ones(3),y0=np.ones(3),ystd=np.ones(3),x_mean=np.ones(6),std_x=np.ones(6)) if offsets else {}
  enc=linear_encoder_init_aug(A=None,B=None,C=None,D=None,nx=6,nu=3,ny=3,na=3,nb=3,nx_aug=na,
   n_nodes_per_layer=16,n_hidden_layers=2,flag_linear_only=False,dtype=torch.float64,
   wa_encoder_init='xavier_uniform_gain1',wb_init='random',**kw).double().cuda()
  u=torch.randn(600,4,3,generator=g,dtype=torch.float64)
  y=torch.randn(600,4,3,generator=g,dtype=torch.float64)
  phi=torch.cat((y.flatten(1),u.flatten(1)),1)
  target=phi@torch.randn(24,6,generator=g,dtype=torch.float64)+torch.arange(6,dtype=torch.float64)
  u,y,target=[v.cuda() for v in (u,y,target)]
  ids={n:id(p) for n,p in enc.named_parameters()}
  before={n:p.detach().clone() for n,p in enc.named_parameters()}
  wa0=enc(u,y)[:,6:].detach().clone()
  W,b=least_squares_start(enc,6,u,y,target)
  assert torch.allclose(enc(u,y)[:,:6],target,atol=1e-10,rtol=0)
  assert torch.equal(enc(u,y)[:,6:],wa0)
  assert ids=={n:id(p) for n,p in enc.named_parameters()}
  for n,p in enc.named_parameters():
   if n.startswith('net.') and n not in ('net.4.weight','net.4.bias'): assert torch.equal(p,before[n])
  state=enc.state_dict(); enc.load_state_dict(state)
  assert torch.allclose(enc(u,y)[:,:6],target,atol=1e-10,rtol=0)
  print('AFFINE_COPY_OFFSETS_ADDED_AND_HIDDEN_OK',na,offsets,flush=True)
try: RunConfig(encoder_init='data_fit')
except ValueError as e: print('OLD_ADAM_CONFIG_REJECTED',str(e),flush=True)
else: raise AssertionError('expected removed configuration')
# Reproduce T1 automation behavior without production data/rollouts.
ns=runpy.run_path(str(root/'scripts/gantry/thesis-results/test_encoder_lsq.py'))
gl=ns['t1'].__globals__
fake=SimpleNamespace(na=30,nb=30,encoder=None)
cfg=SimpleNamespace(nu=3,ny=3,nx_phys=6)
counter=iter(range(10))
gl['config']=lambda *a:cfg
gl['start']=lambda c:(fake,str(next(counter)))
def fail(*a): raise RuntimeError('encoder start not ready')
gl['least_squares_start']=fail
out=io.StringIO()
with contextlib.redirect_stdout(out): result=ns['t1']()
assert '[T1] FAIL' in out.getvalue() and result is None
print('REPRODUCED_T1_FAILURE_RETURNS_SUCCESS_STATUS',flush=True)
print('AUDIT_SYNTHETIC_COMPLETE',flush=True)

