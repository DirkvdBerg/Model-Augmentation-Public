"""G1: the adapter is correct (BB-006). (a) zero controller = open loop; (b) residual identity and
units; (c) this folder's scorer reproduces the grey box's own closed-loop selection number.

No optimizer step anywhere. Reads outputs/g1/g1c_original.json (run g1c_original first) for (c)3.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'adapter'))
import bbcl                                                                # noqa: E402

cfg, CFG, entry = bbcl.load_cfg(local=True)
import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402
from torch import nn                                                       # noqa: E402
from gantry_dynamic.data import (load_datasets, compute_normalization, TRAIN_FILES,   # noqa: E402
                                 VAL_FILES)
from gantry_dynamic.model import build_model                               # noqa: E402
from gantry_dynamic.controller import build_closed_loop                    # noqa: E402
from model_augmentation.fit_systems.closed_loop import ControllerBank, closed_loop_rollout  # noqa: E402
import bb_model                                                            # noqa: E402
import cl_score                                                            # noqa: E402

OUT = os.path.join(bbcl.OUT, 'g1')
os.makedirs(OUT, exist_ok=True)
print(bbcl.describe_cfg(cfg, CFG))
res = {}

np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)

# ---------------------------------------------------------------- black box at random init
fs = bb_model.build_blackbox(cfg, data, norm, arm='random', attach=True)
sim = fs.simulator
bank = sim.bank
j = TRAIN_FILES.index(bb_model.BLA_RECORD)          # one record's windows are enough for (a), (b)
row = sim.train_ctrl_rows[j]
sd_n = fs.norm.transform(data.train_list[j])
uh, yh, uf, yf = sd_n.to_hist_future_data(na=fs.na, nb=fs.nb, nf=cfg.nf, na_right=fs.na_right,
                                          nb_right=fs.nb_right, stride=cfg.stride)
rng = np.random.default_rng(cfg.seed)
ix = np.sort(rng.choice(len(uf), size=64, replace=False))
T = lambda a: torch.as_tensor(np.ascontiguousarray(a[ix]), dtype=cfg.dtype_pt)   # noqa: E731
uh, yh, uf, yf = T(uh), T(yh), T(uf), T(yf)
cix = torch.full((64,), int(row), dtype=torch.long)
print('[G1] windows: record %s (controller row %d), 64 of %d, nf=%d, uhist %s'
      % (TRAIN_FILES[j], row, len(sd_n.u), cfg.nf, tuple(uh.shape)))

with torch.no_grad():
    x0 = fs.encoder(uh, yh)
    # (a) zero controller against the open-loop seam of the same fit system
    z = np.zeros
    zb = ControllerBank(z((1, bank.nc, bank.nc)), z((1, bank.nc, bank.ny)), z((1, bank.nu, bank.nc)),
                        z((1, bank.nu, bank.ny)), ystd=norm.ystd, std_u=norm.std_u, dtype=cfg.dtype_pt)
    y_cl0, _, _ = closed_loop_rollout(fs.hfn, fs.hfn.output_only, uf, yf, x0, zb,
                                      torch.zeros(64, dtype=torch.long))
    fs.simulator = None
    y_ol, _ = fs.simulate(x0, uf, yf)
    fs.simulator = sim
    y_cl, _ = fs.simulate(x0, uf, yf, ctrl_ix=cix)          # the real bank, through the seam
    da = float((y_cl0 - y_ol).abs().max())
    scale = float(y_ol.abs().max())
    res['a_max_abs_diff'] = da
    res['a_rel'] = da / scale
    res['a_real_bank_rel_diff'] = float((y_cl - y_ol).abs().max()) / scale
    res['a_pass'] = bool(np.isfinite(da) and da <= 1e-6 * scale)
    print('[G1a] zero controller vs open loop: max|diff| %.3e (rel %.3e, tol 1e-6) -> %s;'
          ' real bank vs open loop rel %.3e (the controller acts)'
          % (da, res['a_rel'], 'PASS' if res['a_pass'] else 'FAIL', res['a_real_bank_rel_diff']))

    # (b) residual identity: a probe whose output IS the data, so e = 0 at every step
    class Probe(nn.Module):
        def __init__(self, y):
            super().__init__()
            self.y, self.k, self.seen = y, 0, []
            self.p = nn.Parameter(torch.zeros(1))

        def output_only(self, x, u=None):
            return self.y[:, self.k]

        def forward(self, x, u):
            self.seen.append(u.clone())
            self.k += 1
            return None, x

    pr = Probe(yf)
    closed_loop_rollout(pr, pr.output_only, uf, yf, x0, bank, cix)
    u_seen = torch.stack(pr.seen, dim=1)
    db = float((u_seen - uf).abs().max())
    res['b_max_abs_diff'] = db
    res['b_rel'] = db / float(uf.abs().max())
    units = []
    for r in range(bank.n_controllers):
        _, _, rel = bank.check_units(ctrl_ix=r)
        units.append(float(np.max(rel)))
    res['b_units_max_rel_per_row'] = units
    res['b_pass'] = bool(db <= 1e-7 * float(uf.abs().max()) and max(units) <= 1e-5)
    print('[G1b] residual identity: max|u_seen - u_data| = %.3e (rel %.3e, tol 1e-7); units '
          'self-check max rel over %d controllers %.2e (tol 1e-5) -> %s'
          % (db, res['b_rel'], bank.n_controllers, max(units), 'PASS' if res['b_pass'] else 'FAIL'))

# ---------------------------------------------------------------- (c) grey box's own number
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
gb = build_model(cfg.hp, cfg, data, norm)
gb.simulator = build_closed_loop(gb, norm, cfg, train_files=TRAIN_FILES, val_files=VAL_FILES,
                                 val_data=data.val_ckpt_data)
t0 = time.time()
c1 = float(gb.cal_validation_error(data.val_ckpt_data, validation_measure='sim-RMS'))
t_c1 = time.time() - t0
per1 = [float(x) for x in gb.simulator.last_per_record]
t0 = time.time()
names = [f[:-4] for f in VAL_FILES]
c2, per2, perch2, _, _ = cl_score.closed_loop_score(gb, names, data.val_list, cfg.ts_new,
                                                    dtype=cfg.dtype_pt)
t_c2 = time.time() - t0
res.update(c1=c1, c2=c2, c1_per_record=per1, c2_per_record=[float(x) for x in per2],
           c2_per_channel=[[float(v) for v in pc] for pc in perch2],
           c1_seconds=t_c1, c2_seconds=t_c2, val_files=VAL_FILES)
res['c21_rel'] = abs(c2 - c1) / abs(c1)
orig = os.path.join(OUT, 'g1c_original.json')
if os.path.exists(orig):
    c3 = json.load(open(orig))['value']
    res['c3'] = c3
    res['c31_rel'] = abs(c3 - c1) / abs(c1)
else:
    res['c3'] = None
    res['c31_rel'] = float('nan')
res['c_pass'] = bool(res['c21_rel'] <= 1e-6 and res['c3'] is not None and res['c31_rel'] <= 1e-6)
print('[G1c] grey box (untrained CFG model) closed-loop val sim-RMS: own path %.10e m (%.0f s), '
      'this scorer %.10e (rel %.2e), original repo packages %s (rel %.2e) -> %s'
      % (c1, t_c1, c2, res['c21_rel'], res['c3'], res['c31_rel'], 'PASS' if res['c_pass'] else 'FAIL'))
for n, a, b in zip(VAL_FILES, per1, per2):
    print('   %-24s own %.6e   scorer %.6e' % (n, a, b))
res['PASS'] = bool(res['a_pass'] and res['b_pass'] and res['c_pass'])
json.dump(res, open(os.path.join(OUT, 'g1.json'), 'w'), indent=1)
print('[G1] %s' % ('PASS' if res['PASS'] else 'FAIL'))
