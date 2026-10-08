"""Held-out free-run evaluation for PS2 checkpoints (acceptance items 5 to 7 of the handoff).

For each checkpoint: closed-loop free run over the 6 whole validation records (the pipeline's `validation_error`,
sim-RMS in metres, as ad_valfree.py), in three settings: learned block ON; added states CLAMPED (added-row output
layer zeroed, so x_a(k+1) = 0 and the deployed model keeps no added-state memory); learned block OFF (out_gate 0).
Clamp test: per-record increase d_r = clamp_r - on_r; bootstrap over RECORDS (resampling unit = whole record,
10000 draws) of the median of d_r, 95 % percentile interval. Quiet-record check: on vs off per record.
Paired comparison (optional, `--pair cand=ctrl`): per-record ratio cand/ctrl and its bootstrap median interval.
Validation records are used for reporting only: no checkpoint, rule or rate is selected on them.

Usage: python evalc.py CKPT [CKPT ...] [--pair A=B ...]   (paths relative to this folder's outputs/ or the parent's)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
import u_probe as up                                                           # noqa: E402
from ps2 import select_data                                                    # noqa: E402


def path(f):
    for d in (os.path.join(HERE, 'outputs'), os.path.join(PARENT, 'outputs')):
        if os.path.exists(os.path.join(d, f)):
            return os.path.join(d, f)
    raise FileNotFoundError(f)


def boot_median(d, n=10000, seed=0):
    r = np.random.default_rng(seed)
    m = np.median(d[r.integers(len(d), size=(n, len(d)))], axis=1)
    return np.median(d), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main(args):
    files = [a for a in args if not a.startswith('--') and '=' not in a]
    pairs = [a.split('=') for a in args if '=' in a]
    os.environ['U_INIT'] = 'zero'
    select_data()
    cfg, data, norm, fs, ann, others = up.setup()
    net = ann.net
    if os.environ.get('E_SPLIT') == 'test':
        # evaluate on the TEST records instead: a closed-loop simulator built the pipeline's way, with the test
        # records in the validation slot (controller rows looked up per record name); reporting only
        import deepSI
        from gantry_dynamic.controller import build_closed_loop
        tl = deepSI.System_data_list(data.test_list)
        fs.simulator = build_closed_loop(fs, norm, cfg, train_files=data.train_files, val_files=data.test_files,
                                         val_data=tl, verbose=False)
        data.val_ckpt_data = tl
        print('  E_SPLIT=test: records %s' % [os.path.basename(str(f)) for f in data.test_files], flush=True)
    print('\n######## evalc: free run over %d validation records (sim-RMS, m)' % len(fs.simulator.val_records), flush=True)
    res = {}
    for f in files:
        st = torch.load(path(f), weights_only=False)
        fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
        per = {}
        for tag in ('on', 'clamp', 'off'):
            g = ann.out_gate.clone(); wa = net.out_a.weight.detach().clone(); ba = net.out_a.bias.detach().clone()
            try:
                with torch.no_grad():
                    if tag == 'off':
                        ann.out_gate = torch.zeros_like(g)
                    if tag == 'clamp':
                        net.out_a.weight.zero_(); net.out_a.bias.zero_()
                    v = fs.simulator.validation_error(fs, data.val_ckpt_data, 'sim-RMS')
                per[tag] = (v, np.ravel(np.asarray(fs.simulator.last_per_record, dtype=float)))
            finally:
                ann.out_gate = g
                with torch.no_grad():
                    net.out_a.weight.copy_(wa); net.out_a.bias.copy_(ba)
        res[f] = per
        d = per['clamp'][1] - per['on'][1]
        md, lo, hi = boot_median(d)
        fin = all(np.all(np.isfinite(per[t][1])) for t in per)
        above = np.where(per['on'][1] > per['off'][1])[0] + 1
        print('  [%s, %s upd] on %.4e | clamp %.4e | off %.4e | finite %s' % (f, st.get('n_upd'), per['on'][0],
                                                                       per['clamp'][0], per['off'][0], fin), flush=True)
        for t in ('on', 'clamp', 'off'):
            print('      %-5s per record %s' % (t, ' '.join('%.3e' % p for p in per[t][1])), flush=True)
        print('      clamp increase per record %s; median %.3e, 95%% bootstrap CI over records [%.3e, %.3e] -> %s' % (
            ' '.join('%.2e' % x for x in d), md, lo, hi, 'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'), flush=True)
        print('      records with on > off (block hurts): %s' % (list(above) if len(above) else 'none'), flush=True)
    for a, b in pairs:
        r = res[a]['on'][1] / res[b]['on'][1]
        md, lo, hi = boot_median(np.log(r))
        print('  PAIR %s vs %s: per-record on ratio %s; median ratio %.3f, 95%% CI [%.3f, %.3f]; whole %.4e vs %.4e' % (
            a, b, ' '.join('%.3f' % x for x in r), np.exp(md), np.exp(lo), np.exp(hi), res[a]['on'][0], res[b]['on'][0]),
            flush=True)
        dd = (res[a]['clamp'][1] - res[a]['on'][1]) - (res[b]['clamp'][1] - res[b]['on'][1])
        md, lo, hi = boot_median(dd)
        print('      clamp increase cand minus ctrl per record %s; median %.3e, 95%% CI [%.3e, %.3e] -> %s' % (
            ' '.join('%.2e' % x for x in dd), md, lo, hi, 'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'), flush=True)
        qa = res[a]['on'][1] / res[a]['off'][1]; qb = res[b]['on'][1] / res[b]['off'][1]
        print('      on/off per record cand %s | ctrl %s' % (' '.join('%.3f' % x for x in qa),
                                                          ' '.join('%.3f' % x for x in qb)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
