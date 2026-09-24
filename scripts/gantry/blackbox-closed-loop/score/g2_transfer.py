"""G2: one existing OPEN-loop black-box checkpoint scored in the closed-loop metric (BB-007).

    python score/g2_transfer.py --arm {random,n4sid,frf,oracle}

On the checkpoint's OWN data (old `augmentation` folder), rate (800 Hz) and encoder window, with
the Tustin controller built at 800 Hz and folded with the checkpoint's own normalisation. A
transfer measurement, not a result on the production data. Checkpoints are read in place.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'adapter'))
import bbcl                                                                # noqa: E402

torch = bbcl.bootstrap()
import numpy as np                                                         # noqa: E402
from torch import nn                                                       # noqa: E402
import deepSI                                                              # noqa: E402
from gantry_dynamic.controller import RECORD_Y_OP, build_controller_bank   # noqa: E402
from model_augmentation.fit_systems.closed_loop import ControllerBank      # noqa: E402
import cl_score                                                            # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'data_bb_old', os.path.join(bbcl.VENDOR, 'ann_blackbox', 'data.py'))
data_bb_old = importlib.util.module_from_spec(_spec)
sys.modules['data_bb_old'] = data_bb_old
_spec.loader.exec_module(data_bb_old)
import plant                                                               # noqa: E402

AB = os.path.join(bbcl.REPO, 'scripts', 'gantry', 'ann-blackbox')
CKPTS = {
    'random': ('server-results/paired_random', 'fs800_nf400_s0'),
    'n4sid': ('server-results/paired_bla', 'fs800_nf400_s0_bladynz'),
    'frf': ('results/phase1_frf', 'fs800_nf400_s0_blafrfz'),
    'oracle': ('results/phase1_oracle', 'fs800_nf400_s0_blafrfz_oracle'),
}
FS = 800.0
RECORDS = ('V2_aprbs_Ylow', 'T10_aprbs_60')     # their validation record, their training record

p = argparse.ArgumentParser()
p.add_argument('--arm', choices=sorted(CKPTS), required=True)
args = p.parse_args()
folder, tag = CKPTS[args.arm]
ck = os.path.join(AB, folder, 'ann_blackbox_' + tag)
met = json.load(open(os.path.join(AB, folder, 'metrics_' + tag + '.json')))
print('[G2] arm %s: %s' % (args.arm, ck))
assert met['train'] == 'T10_aprbs_60' and met['val'] == 'V2_aprbs_Ylow' and met['fs'] == FS, met['train']

fs = deepSI.load_system(ck)
assert not getattr(fs.hfn, 'feedthrough', False), 'output_only needs D = 0'
fs.hfn.eval(); fs.encoder.eval()


class HFView(nn.Module):
    """The checkpoint's hf step plus output_only = h(x), for the closed-loop harness."""

    def __init__(self, hfn):
        super().__init__()
        self.inner = hfn

    def forward(self, x, u):
        return self.inner(x, u)

    def output_only(self, x, u=None):
        return self.inner.hn(x)


class View:
    def __init__(self, fs):
        self.norm, self.na, self.nb = fs.norm, fs.na, fs.nb
        self.na_right = getattr(fs, 'na_right', 0)
        self.nb_right = getattr(fs, 'nb_right', 0)
        self.encoder, self.hfn = fs.encoder, HFView(fs.hfn)


view = View(fs)
sdl = [data_bb_old.load(n, FS) for n in RECORDS]
res = dict(arm=args.arm, checkpoint=ck, fs=FS, k0=max(fs.na, fs.nb), records=list(RECORDS),
           metrics_best_sim_rms=met['best_sim_rms'], metrics_epoch0_sim_rms=met['epoch0_sim_rms'])
yops = []
for n in RECORDS:
    y_rec = plant.load_record(n, fs_new=4000)['Y_op']
    yops.append((n, RECORD_Y_OP[n], y_rec))
    assert abs(RECORD_Y_OP[n] - y_rec) <= 1e-6, (n, RECORD_Y_OP[n], y_rec)
res['y_op'] = yops
print('[G2] Y_op (RECORD_Y_OP vs raw record y[0,2]):', yops)

bank, rows, uniq = build_controller_bank(list(RECORDS), 1.0 / FS, ystd=fs.norm.ystd,
                                         std_u=fs.norm.ustd, dtype=torch.float32)
print('[G2] controller at %g Hz, rows %s for Y_op %s, diag(Dc) %s N/m'
      % (FS, rows, uniq, np.array2string(bank.physical_D()[0].diagonal().numpy(), precision=4)))

# Open-loop counterpart 1: deepSI's own validation path on V2, must reproduce the metrics JSON.
t0 = time.time()
ol_deepsi = float(fs.cal_validation_error(sdl[0], validation_measure='sim-RMS'))
res['ol_deepsi_V2'] = ol_deepsi
res['ol_deepsi_rel_to_metrics'] = abs(ol_deepsi - met['best_sim_rms']) / met['best_sim_rms']
print('[G2] open loop, deepSI path, V2: %.6e m (metrics best %.6e, rel %.2e, tol 1e-3)  %.0f s'
      % (ol_deepsi, met['best_sim_rms'], res['ol_deepsi_rel_to_metrics'], time.time() - t0))
load_ok = res['ol_deepsi_rel_to_metrics'] <= 1e-3

# Open-loop counterpart 2: the same harness with a zero controller (same encoder state, samples).
z = np.zeros
zb = ControllerBank(z((1, bank.nc, bank.nc)), z((1, bank.nc, bank.ny)), z((1, bank.nu, bank.nc)),
                    z((1, bank.nu, bank.ny)), ystd=fs.norm.ystd, std_u=fs.norm.ustd)
t0 = time.time()
_, ol_per, _, _, _ = cl_score.closed_loop_score(view, RECORDS, sdl, 1.0 / FS, bank=zb, rows=[0, 0])
cl_s, cl_per, cl_ch, _, _ = cl_score.closed_loop_score(view, RECORDS, sdl, 1.0 / FS, bank=bank,
                                                        rows=rows)
tr_ol = cl_score.closed_loop_traces(view, sdl, zb, [0, 0], per_record_ref=ol_per)
tr_cl = cl_score.closed_loop_traces(view, sdl, bank, rows, per_record_ref=cl_per)
print('[G2] harness runs %.0f s' % (time.time() - t0))
res['per_record'] = {}
for i, n in enumerate(RECORDS):
    d = dict(ol_harness=float(ol_per[i]), cl=float(cl_per[i]),
             cl_per_channel=[float(v) for v in cl_ch[i]],
             ol_stable=tr_ol[i]['stable'], cl_stable=tr_cl[i]['stable'],
             cl_max_abs_err=tr_cl[i]['max_abs_err'], ol_max_abs_err=tr_ol[i]['max_abs_err'],
             max_abs_y=tr_cl[i]['max_abs_y'])
    d['ol_over_cl'] = d['ol_harness'] / d['cl'] if d['cl'] > 0 else float('nan')
    res['per_record'][n] = d
    print('   %-14s open %.4e (%s)  closed %.4e (%s)  open/closed %.1fx  closed per ch %s'
          % (n, d['ol_harness'], 'stable' if d['ol_stable'] else 'UNSTABLE', d['cl'],
             'stable' if d['cl_stable'] else 'UNSTABLE', d['ol_over_cl'],
             np.array2string(np.asarray(d['cl_per_channel']), precision=3)))
res['load_ok'] = bool(load_ok)
os.makedirs(os.path.join(bbcl.OUT, 'g2'), exist_ok=True)
json.dump(res, open(os.path.join(bbcl.OUT, 'g2', args.arm + '.json'), 'w'), indent=1, default=str)
print('[G2] %s: load %s; V2 closed %.4e vs open (harness) %.4e, deepSI open %.4e'
      % (args.arm, 'OK' if load_ok else 'MISMATCH', res['per_record']['V2_aprbs_Ylow']['cl'],
         res['per_record']['V2_aprbs_Ylow']['ol_harness'], ol_deepsi))
