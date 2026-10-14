import os
import sys
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'scripts' / 'gantry'))
os.environ.update(THESIS_ARM='u', THESIS_NOISE='noisy', THESIS_START='correct',
                  THESIS_NA='2', THESIS_SEED='1', THESIS_DEVICE='cpu')
torch.set_num_threads(1)
print('Reading data and normalisation only', flush=True)
from gantry_interconnect_dynamic import CFG
from gantry_dynamic.data import load_datasets, compute_normalization
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
from model_augmentation.systems.gantry_ss import P
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug

data = load_datasets(CFG)
norm = compute_normalization(CFG, data)
print('Data loaded', flush=True)
rawx, rawu, rawy = [a.std(0) for a in (norm.x_all, norm.u_all, norm.y_all)]
sx, su, sy = norm.std_x.ravel(), norm.std_u.ravel(), norm.ystd.ravel()
print('raw std x/u/y', rawx, rawu, rawy, flush=True)
print('max relative scale discrepancy x/u/y', *[np.max(abs(s/r-1)) for s,r in zip((sx,su,sy),(rawx,rawu,rawy))], flush=True)

A, B, C, D = gantry_linearize_and_discretize(CFG.ts_new)
def matrices(x, u, y):
    return A*x[None,:]/x[:,None], B*u[None,:]/x[:,None], C*x[None,:]/y[:,None], D
def wb(x, u, y):
    a,b,c,d = matrices(x,u,y)
    n = 29
    o = np.vstack([c @ np.linalg.matrix_power(a,i) for i in range(n+1)])
    t = np.zeros((3*(n+1),3*(n+1)))
    r = np.zeros((6,3*(n+1)))
    for i in range(n+1):
        for j in range(i):
            t[3*i:3*i+3,3*j:3*j+3] = c @ np.linalg.matrix_power(a,i-j-1) @ b
        if i<n:
            r[:,3*i:3*i+3] = np.linalg.matrix_power(a,n-i-1) @ b
    ay = np.linalg.matrix_power(a,n) @ np.linalg.pinv(o)
    return np.hstack([ay, r-ay@t]), o, t, r, a

wraw, oraw, _, _, _ = wb(rawx,rawu,rawy)
wconsistent, o, t, r, a = wb(sx,su,sy)
print('O rank/condition (consistent frame)', np.linalg.matrix_rank(o), np.linalg.cond(o), flush=True)
print('W relative Frobenius discrepancy', np.linalg.norm(wraw-wconsistent)/np.linalg.norm(wconsistent), flush=True)
means = np.r_[np.tile(norm.y0/sy,30),np.tile(norm.u_mean.ravel()/su,30)]
chi = np.r_[((data.train_list[0].y[:30]-norm.y0)/sy).ravel(),((data.train_list[0].u[:30]-norm.u_mean.ravel())/su).ravel()]
print('Encoder output scale discrepancy on first real window', (wraw-wconsistent)@(chi+means), flush=True)
print('Output offset cancellation', np.max(abs(P.numpy().T @ norm.x_mean.ravel()[:3]-norm.y0)), flush=True)
print('Bias own derivation identity residual', np.max(abs((wconsistent@(chi+means)-norm.x_mean.ravel()/sx)-(wconsistent@chi+wconsistent@means-norm.x_mean.ravel()/sx))), flush=True)

# White equal-variance noise in physical channels becomes anisotropic after signal scaling.
rinv = np.tile(sy**2,30)
weighted_left = np.linalg.solve(o.T@(rinv[:,None]*o), o.T*rinv[None,:])
ordinary_left = np.linalg.pinv(o)
print('Physical isotropic noise: weighted vs ordinary map relative difference', np.linalg.norm(np.linalg.matrix_power(a,29)@(weighted_left-ordinary_left))/np.linalg.norm(np.linalg.matrix_power(a,29)@ordinary_left), flush=True)

enc = linear_encoder_init_aug(*matrices(rawx,rawu,rawy),nx=6,nu=3,ny=3,na=29,nb=29,nx_aug=2,dtype=torch.float64,
    u_mean=norm.u_mean,std_u=norm.std_u,y0=norm.y0,ystd=norm.ystd,x_mean=norm.x_mean,std_x=norm.std_x,
    wa_encoder_init='xavier_uniform_gain1').double()
got = enc(torch.tensor(chi[90:].reshape(1,30,3)),torch.tensor(chi[:90].reshape(1,30,3))).detach().numpy()[0,:6]
expected = wraw@(chi+means)-norm.x_mean.ravel()/sx
print('Code forward vs centred bias formula residual',np.max(abs(got-expected)),flush=True)
