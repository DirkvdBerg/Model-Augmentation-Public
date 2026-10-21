"""Layers 1 and 2 of design 6.3 for one case, on one side, in one process.

    python -u capture_l12.py --side mainline|clean --row 221 --noise lowpass [--full]

Mainline writes the golden (simulations/clean-golden/L1|L2/<row>-<noise>/); the clean side writes
simulations/clean-golden/_clean/L1|L2/<row>-<noise>/ and loads the GOLDEN post-pre-fit state for
the forward-pass and gradient part, as 6.3 prescribes. The call order is EP main's up to the
encoder pre-fit (seed, data, normalisation, hp, seed, build_model, build_closed_loop, pre-fit), so
the global draws are the run's own (H1).
"""
import argparse
import os
import sys
import time

import hcommon as H

ap = argparse.ArgumentParser()
ap.add_argument('--side', required=True, choices=('mainline', 'clean'))
ap.add_argument('--row', type=int, required=True, choices=sorted(H.ROWS))
ap.add_argument('--noise', required=True, choices=('lowpass', 'noisefree'))
ap.add_argument('--full', action='store_true', help='store every array, none as a hash only')
args = ap.parse_args()

H.select_side(args.side)
import numpy as np
import torch

case = '%d-%s' % (args.row, args.noise)
base = H.GOLDEN if args.side == 'mainline' else H.CLEAN_OUT
out1, out2 = os.path.join(base, 'L1', case), os.path.join(base, 'L2', case)
golden_state = os.path.join(H.GOLDEN, 'L1', case, 'prefit_state.pt')
if args.side == 'clean' and not os.path.exists(golden_state):
    sys.exit('no golden post-pre-fit state at %s: capture the mainline side first' % golden_state)

T0 = time.time()
cfg, EP = H.make_cfg(args.row, args.noise, run_id='harness-L1-' + case)
import gantry_dynamic.data as GDD
import gantry_dynamic.model as GDM
from gantry_dynamic.controller import build_closed_loop
H.assert_roots()
c1, c2 = H.Capture(args.full), H.Capture(args.full)
tim = {}

# EP main's order up to the pre-fit (EP:390-445)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
data = GDD.load_datasets(cfg)
norm = GDD.compute_normalization(cfg, data)
hp = cfg.hp
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
t = time.time()
fit_sys = GDM.build_model(hp, cfg, data, norm)
tim['build_model'] = time.time() - t
c1.info['hp'] = hp
c1.info['dims'] = [fit_sys.na, fit_sys.nb, getattr(fit_sys, 'na_right', 0), getattr(fit_sys, 'nb_right', 0)]
c1.add_state('build/hfn', fit_sys.hfn.state_dict())
c1.add_state('build/encoder', fit_sys.encoder.state_dict())

# data per split, normalisation, fit_sys.norm
for split, lst, el in (('train', data.train_list, data.e_train), ('val', data.val_list, data.e_val),
                       ('test', data.test_list, data.e_test)):
    for i, sd in enumerate(lst):
        c1.add('data/%s/%02d/u' % (split, i), sd.u)
        c1.add('data/%s/%02d/y' % (split, i), sd.y)
        c1.add('data/%s/%02d/e' % (split, i), el[i])
c1.info['files'] = [list(data.train_files), list(data.val_files), list(data.test_files)]
for f in ('x_mean', 'std_x', 'std_u', 'u_mean', 'ystd', 'y0', 'Cd_norm', 'Dd_np'):
    c1.add('norm/' + f, getattr(norm, f))
for f in ('u0', 'ustd', 'y0', 'ystd'):
    c1.add('fsnorm/' + f, getattr(fit_sys.norm, f))

t = time.time()
fit_sys.simulator = build_closed_loop(
    fit_sys, norm, cfg, train_files=data.train_files, val_files=data.val_files,
    val_data=data.val_ckpt_data, e_train=data.e_train, e_val=data.e_val)
tim['build_closed_loop'] = time.time() - t
sim = fit_sys.simulator
# D-281 (O8): the clean simulator no longer has bank.ystd, bank.stdu (B41) or the validation record
# names (B51; compare.py skips these three) and holds the validation rows (B51) and the loss
# weights (B5) itself. The mainline capture is unchanged.
MAIN = args.side == 'mainline'
for f in ('M_state', 'M_error') + (('ystd', 'stdu') if MAIN else ()):
    c1.add('bank/' + f, getattr(sim.bank, f))
c1.add('sim/train_ctrl_rows', np.asarray(sim.train_ctrl_rows))
c1.add('sim/val_rows', np.asarray([r for (_, _, r) in sim.val_records] if MAIN else sim.val_rows))
if MAIN:
    c1.info['sim/val_labels'] = [n for (n, _, _) in sim.val_records]
c1.add('sim/val_sigma_e', sim.val_sigma_e)
c1.add('sim/train_weights', np.stack(sim.indexer.record_weights if MAIN else sim.record_weights))
c1.info['sim/extra_array_names'] = list(sim.extra_array_names)

ref = getattr(fit_sys, 'obc_reference', None)
if ref is not None:
    for f in ('Z_step', 'Z_field', 'record_ix', 'sample_ix', 'x_phys_next_rec'):
        c1.add('obcref/' + f, getattr(ref, f))
    c1.info['obcref/record_names'] = list(ref.record_names)
    c1.info['obcref/n'] = ref.n

# the encoder pre-fit, with its least-squares windows captured (harness-side wrapper)
_ls = GDM.least_squares_start
_seen = {}


def _ls_capture(enc, nx, uh, yh, xt):
    _seen.update(uh=uh.detach().clone(), yh=yh.detach().clone(), xt=xt.detach().clone())
    W, b = _ls(enc, nx, uh, yh, xt)
    _seen.update(W=W.detach().clone(), b=b.detach().clone())
    return W, b


GDM.least_squares_start = _ls_capture
t = time.time()
GDM.prefit_physical_encoder(fit_sys, cfg, data, norm)
tim['prefit'] = time.time() - t
GDM.least_squares_start = _ls
for k in ('uh', 'yh', 'xt', 'W', 'b'):
    c1.add('ls/' + k, _seen[k])
Phi = torch.cat([_seen['yh'].flatten(1), _seen['uh'].flatten(1)], 1).to(torch.float64)
X = _seen['xt'].to(torch.float64)
mu, s, x_mu = Phi.mean(0), Phi.std(0, unbiased=False), X.mean(0)
s = torch.where(s > 0, s, torch.ones_like(s))
sol = torch.linalg.lstsq((Phi - mu) / s, X - x_mu,
                         rcond=torch.finfo(torch.float64).eps * max(Phi.shape), driver='gelsd')
c1.add('ls/rank', np.asarray(int(sol.rank)))
c1.add('ls/singular_values', sol.singular_values)
c1.add_state('prefit/hfn', fit_sys.hfn.state_dict())
c1.add_state('prefit/encoder', fit_sys.encoder.state_dict())
if fit_sys.obc is not None:
    B = fit_sys.obc.basis
    for f in ('U', 'S', 'Vh', 'vbar', 'column_scale'):
        c1.add('basis/' + f, getattr(B, f))
    c1.add('basis/rank', np.asarray(B.rank))
    c1.add('basis/tol', np.asarray(B.report.tol if MAIN else B.tol))     # C19b (D-281): no RankReport
    c1.add('basis/theta_frozen', fit_sys.hfn.obc_correction.theta_frozen)
if args.side == 'mainline':
    os.makedirs(out1, exist_ok=True)
    torch.save({'hfn': fit_sys.hfn.state_dict(), 'encoder': fit_sys.encoder.state_dict()}, golden_state)

# with the GOLDEN post-pre-fit state loaded
st = torch.load(golden_state, map_location='cpu')
fit_sys.hfn.load_state_dict(st['hfn'])
fit_sys.encoder.load_state_dict(st['encoder'])
t = time.time()
dtype = next(fit_sys.hfn.parameters()).dtype
tr = fit_sys.make_training_data(fit_sys.norm.transform(data.train_data), nf=hp['nf'], stride=cfg.stride)
c1.info['windows/n'] = len(tr[0])
for k, a in zip(('uh', 'yh', 'uf', 'yf') + tuple(sim.extra_array_names), tr):
    c1.add('windows/' + k, a)
ix = np.sort(np.random.default_rng(0).choice(len(tr[0]), 64, replace=False))
c1.add('batch/ix', ix)
bt = [torch.as_tensor(np.asarray(a)[ix]) for a in tr]
bt = [b.to(dtype) if b.is_floating_point() else b for b in bt]
uh_b, yh_b, uf_b, yf_b = bt[:4]
kw = dict(zip(sim.extra_array_names, bt[4:]))
with torch.no_grad():
    x0 = fit_sys.encoder(uh_b, yh_b)
    c1.add('fwd/x0', x0)
    if fit_sys._mats_ix is not None:
        pm = fit_sys.hfn.connected_blocks[fit_sys._mats_ix].pass_mats()
        for j, m in enumerate(pm):
            c1.add('fwd/pass_mats/%02d' % j, m)
    corr = getattr(fit_sys.hfn, 'obc_correction', None)
    if fit_sys.obc is not None:
        theta = fit_sys.obc.coefficient(fit_sys.hfn.connected_blocks[fit_sys._ann_ix])
        c1.add('fwd/obc_theta', theta)
        for j, m in enumerate(corr.tangent_pair(theta)):
            if isinstance(m, (tuple, list)):
                for q, mm in enumerate(m):
                    c1.add('fwd/tangent/%d/%02d' % (j, q), mm)
            else:
                c1.add('fwd/tangent/%d' % j, m)
    y_pred, x_fin = fit_sys.simulate(x0, uf_b, yf_b, ctrl_ix=kw['ctrl_ix'])
    c1.add('fwd/y_pred', y_pred)
    c1.add('fwd/x_final', x_fin)
val = fit_sys.cal_validation_error(data.val_ckpt_data, cfg.validation_measure)
c1.add('fwd/val_score', np.asarray(val))
c1.add('fwd/last_per_channel', sim.last_per_channel)
c1.add('fwd/last_sim_rms', np.asarray(sim.last_sim_rms))
tim['forward'] = time.time() - t

# layer 2: loss and gradients on the same batch and state
t = time.time()
fit_sys.hfn.zero_grad(set_to_none=True)
fit_sys.encoder.zero_grad(set_to_none=True)
L = fit_sys.loss(uh_b, yh_b, uf_b, yf_b, nf=hp['nf'], stride=cfg.stride, **kw)
L.backward()
c2.add('loss', L.detach())
names = []
for gname, grp in fit_sys.parameters_with_names.items():
    mod = getattr(fit_sys, gname)
    named = list(mod.named_parameters()) if isinstance(mod, torch.nn.Module) else [('', mod)]
    for pname, p in named:
        names.append('%s.%s' % (gname, pname))
        c2.add('grad/%s.%s' % (gname, pname), p.grad if p.grad is not None
               else torch.full(p.shape, float('nan'), dtype=p.dtype))
c2.info['param_order'] = names
tim['loss_backward'] = time.time() - t

H.assert_roots()
tim['total'] = time.time() - T0
meta_extra = dict(case=case, layer='1+2', timings=tim)
H.write_meta(out1, cfg, data, meta_extra)
H.write_meta(out2, cfg, data, meta_extra)
c1.save(out1, 'L1')
c2.save(out2, 'L2')
print('CAPTURE_DONE side=%s case=%s seconds=%.1f %s' % (args.side, case, tim['total'], tim), flush=True)
