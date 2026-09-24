"""G4 attempt-1 diagnosis: why the replay settles into a post-move oscillation (TR-015).

(1) Linear closed-loop poles: the frictionless block's RK4 step linearised by autograd at rest
    (Y in {-0.2, 0, 0.12} m), closed with the identified and with the zpk controller in the
    exact step order of cl_sim. Unstable or lightly damped poles near 100-200 Hz = a linear
    loop problem; all well damped = the oscillation is friction-induced.
(2) Replay of 4 held-out iter0 records with the attempt-1 parameters under: Karnopp V_BRK
    2.25e-3 (as G4 attempt 1), friction off, Karnopp V_BRK 1e-4, tanh (v0 1e-3). Post-move
    oscillation measured as the rms of y_model - y_data over the last 100 ms of each record.
Prints summaries; writes outputs/g4_diag_lc/diag.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

from params import telica_params as tp                                 # noqa: E402
from baseline.records import iter_records                              # noqa: E402
from baseline.cl_sim import make_block, pack, simulate                 # noqa: E402
from controller.telica_bank import physical_ss                         # noqa: E402
tr_env.check_no_leak()

OUT = tr_env.out_dir('g4_diag_lc')
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else ''
R = json.load(open(os.path.join(HERE, f'recovered_params{TAG}.json')))
P = np.array([[1.0, 1.0, 0.0], [tp.Lb / 2, -tp.Lb / 2, 0.0], [0.0, 0.0, 1.0]])


def lin_poles(raw, source, Y):
    blk = make_block(raw=raw, cc=(0.0, 0.0, 0.0), mode='none')
    x0 = torch.tensor([0.0, 0.0, Y, 0.0, 0.0, 0.0], dtype=torch.float64)
    u0 = torch.zeros(3, dtype=torch.float64)
    f = lambda x, u: blk.nonlinear_function(torch.cat([x, u]).reshape(1, 9, 1)).reshape(6)  # noqa
    Ax, Bu = torch.autograd.functional.jacobian(f, (x0, u0))
    A = Ax.numpy(); B = Bu.numpy()
    C = np.hstack([P.T, np.zeros((3, 3))])
    Ac, Bc, Cc, Dc = physical_ss(source)
    top = np.hstack([A - B @ Dc @ C, B @ Cc])
    bot = np.hstack([-Bc @ C, Ac])
    lam = np.linalg.eigvals(np.vstack([top, bot]))
    f_hz = np.abs(np.angle(lam)) / (2 * np.pi * tp.TS)
    # continuous-equivalent damping of each pole
    s = np.log(lam.astype(complex)) / tp.TS
    zeta = -s.real / np.maximum(np.abs(s), 1e-12)
    order = np.argsort(-np.abs(lam))
    worst = [dict(abs=float(abs(lam[i])), f_hz=float(f_hz[i]), zeta=float(zeta[i]))
             for i in order[:8]]
    osc = [(float(zeta[i]), float(f_hz[i]), float(abs(lam[i]))) for i in range(len(lam))
           if 20 < f_hz[i] < 2000]
    osc.sort()
    return dict(max_abs=float(np.abs(lam).max()), worst=worst, least_damped_osc=osc[:4])


def main():
    out = {'poles': {}, 'replay': {}}
    raw = R['raw14']
    for source in ('identified', 'zpk'):
        for Y in (-0.2, 0.0, 0.12):
            p = lin_poles(raw, source, Y)
            out['poles'][f'{source}_Y{Y}'] = p
            print(f'[poles] {source:10s} Y={Y:+.2f}: max|lambda| {p["max_abs"]:.6f}; least-damped '
                  f'oscillatory (zeta, f Hz, |lambda|): '
                  + ', '.join(f'({z:+.3f}, {f:.0f}, {a:.5f})' for z, f, a in p['least_damped_osc']))
    recs = [d for s, op, it, d in iter_records(splits=('validation', 'test'), iters=('iter0',))]
    data = pack(recs)
    variants = {'karnopp_2.25e-3': dict(mode='karnopp', v_brk=tp.V_BRK),
                'none': dict(mode='none', v_brk=tp.V_BRK),
                'karnopp_1e-4': dict(mode='karnopp', v_brk=1e-4),
                'tanh_1e-3': dict(mode='tanh', v_brk=tp.V_BRK)}
    for name, v in variants.items():
        blk = make_block(raw=raw, cc=tuple(R['cc']) if v['mode'] != 'none' else (0, 0, 0),
                         mode=v['mode'])
        blk.v_brk = v['v_brk']
        yd, yr, _, _ = simulate(blk, data, log_every=0)
        tail, mot = [], []
        for i in range(len(recs)):
            n, km = data['n'][i], data['k_motion'][i]
            tl = slice(n - 2000, n)
            tail.append(np.sqrt(np.mean((yr[i, tl] - data['y'][i, tl]) ** 2, 0)) * 1e9)
            e_sim = data['r'][i, km:n] - yd[i, km:n]
            mot.append(np.sqrt(np.mean((e_sim - data['e'][i, km:n]) ** 2, 0))
                       / np.sqrt(np.mean(data['e'][i, km:n] ** 2, 0)))
        out['replay'][name] = dict(tail_rms_nm=np.median(tail, 0).tolist(),
                                   nrmse_direct=np.median(mot, 0).tolist())
        print(f'[replay] {name:16s}: last-100 ms rms(y_model - y_data) median '
              f'{np.round(np.median(tail, 0), 1).tolist()} nm; NRMSE direct median '
              f'{np.round(np.median(mot, 0), 3).tolist()}', flush=True)
    e_tail = [np.sqrt(np.mean(data['e'][i, data['n'][i] - 2000:data['n'][i]] ** 2, 0)) * 1e9
              for i in range(len(recs))]
    out['measured_tail_rms_nm'] = np.median(e_tail, 0).tolist()
    print(f'[replay] measured servo error over the same last 100 ms: '
          f'{np.round(np.median(e_tail, 0), 1).tolist()} nm')
    json.dump(out, open(os.path.join(OUT, f'diag{TAG}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
