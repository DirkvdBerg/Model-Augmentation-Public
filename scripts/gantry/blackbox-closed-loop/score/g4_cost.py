"""G4 cost projection (BB-009) from the measured JSONs; no model is run.

Inputs: outputs/g3/frf.json and greybox.json (fwd+bwd at batch 512, same CPU), outputs/g1/g1.json
(grey box validation seconds), outputs/g4/smoke_*/result.json (black box update and validation
seconds), CFG (epochs, its_per_val, batch) and the full-data window count from G3.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'adapter'))
import bbcl                                                                # noqa: E402

cfg, CFG, entry = bbcl.load_cfg(local=False)
import numpy as np                                                         # noqa: E402

O = bbcl.OUT
bb = json.load(open(os.path.join(O, 'g3', 'frf.json')))
gb = json.load(open(os.path.join(O, 'g3', 'greybox.json')))
g1 = json.load(open(os.path.join(O, 'g1', 'g1.json')))
sm = json.load(open(sorted(glob.glob(os.path.join(O, 'g4', 'smoke_frf_*', 'result.json')))[-1]))

T_GB_SERVER_UPDATE = 1.236   # s/update, grey box, nf 400, batch 512, compiled, job 83978 (entry-file comment)
T_GB_SERVER_VAL = 619.0      # s/validation, grey box, job 83962 (entry-file comment)

N = int(bb['n_windows'])
upe = N // int(CFG.batch_size)                      # fit(): N_training_samples // batch_size
n_upd = int(upe * CFG.epochs) if CFG.n_its is None else int(CFG.n_its)
n_val = n_upd // int(CFG.its_per_val) + 1          # initial validation + one per its_per_val
t_bb_fb = float(np.mean([d['t_forward_backward'] for d in bb['grads']]))
t_gb_fb = float(np.mean([d['t_forward_backward'] for d in gb['grads']]))
r_upd = t_bb_fb / t_gb_fb
t_bb_val_cpu = float(np.mean(sm['t_validations'] + [bb['val_seconds']]))
t_gb_val_cpu = float(g1['c1_seconds'])
r_val = t_bb_val_cpu / t_gb_val_cpu
# HEURISTIC (BB-009): dispatch-bound on the GPU, so server cost scales with the CPU-measured ratio.
t_upd_srv = T_GB_SERVER_UPDATE * r_upd
t_val_srv = T_GB_SERVER_VAL * r_val
res = dict(n_windows=N, updates_per_epoch=upe, epochs=CFG.epochs, n_updates=n_upd,
           its_per_val=CFG.its_per_val, n_validations=n_val,
           t_bb_fwdbwd_cpu=t_bb_fb, t_gb_fwdbwd_cpu=t_gb_fb, ratio_update=r_upd,
           t_bb_update_cpu_smoke=sm['t_updates'], t_bb_val_cpu=t_bb_val_cpu,
           t_gb_val_cpu=t_gb_val_cpu, ratio_val=r_val,
           server_t_update=t_upd_srv, server_t_val=t_val_srv,
           server_hours_adam=n_upd * t_upd_srv / 3600, server_hours_val=n_val * t_val_srv / 3600,
           cpu_hours_adam=n_upd * float(np.mean(sm['t_updates'])) / 3600,
           cpu_hours_val=n_val * t_bb_val_cpu / 3600)
res['server_hours_total_excl_recording_polish'] = res['server_hours_adam'] + res['server_hours_val']
res['cpu_hours_total_excl_polish'] = res['cpu_hours_adam'] + res['cpu_hours_val']
for k, v in res.items():
    print('  %-42s %s' % (k, v))
json.dump(res, open(os.path.join(O, 'g4', 'cost.json'), 'w'), indent=1)
