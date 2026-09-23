"""G5 data step: the chosen baseline's closed-loop residual on every record (no metric here).

Residual (training) form of the G4 attempt-2 baseline (tanh friction, held-friction cc), identified
controller, float64, 20 kHz. Per record, from 100 ms before motion to the end, stored
anti-aliased (zero-phase Butterworth 8, 2 kHz) and decimated 4x to 5 kHz, float32:
  res   y_model - y_data              [m]   (3)   the output residual the augmentation must remove
  xm    model state (logical)         [m, m/s] (6)
  ucl   plant input in the loop       [N]   (3)
  ud    logged total force u_data     [N]   (3)
  uff   logged feedforward            [N]   (3)
  r     reference                     [m]   (3)
  e     logged servo error            [m]   (3)
and the pre-motion standstill error (noise floor) at full rate. -> outputs/g5_residuals/<split>.npz
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import json                                                            # noqa: E402
import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402
from scipy.signal import butter, sosfiltfilt                           # noqa: E402

from params import telica_params as tp                                 # noqa: E402
from baseline.records import iter_records                              # noqa: E402
from baseline.cl_sim import make_block, pack                           # noqa: E402
from controller.telica_bank import physical_ss                         # noqa: E402
tr_env.check_no_leak()

OUT = tr_env.out_dir('g5_residuals')
BASE = os.path.join(tr_env.TR, 'baseline', 'recovered_params_a2.json')
DEC = 4
AA = butter(8, 2000.0, fs=tp.FS, output='sos')                         # anti-alias before 4x
P = np.array([[1.0, 1.0, 0.0], [tp.Lb / 2, -tp.Lb / 2, 0.0], [0.0, 0.0, 1.0]])
CHUNK = 40


def residual_rollout(block, data):
    """Residual form only: u = u_data + K(y_data - y_model). Returns y_model, x_model, u_cl."""
    D = torch.float64
    A, Bc, C, Dc = (torch.as_tensor(m, dtype=D) for m in physical_ss('identified'))
    t = lambda a: torch.as_tensor(a, dtype=D)                          # noqa: E731
    y, u = t(data['y']), t(data['u'])
    Pt = t(P)
    Bn, T = y.shape[0], y.shape[1]
    q0 = torch.linalg.solve(Pt.T, y[:, 0, :].T).T
    x = torch.cat([q0, torch.zeros_like(q0)], 1).unsqueeze(-1)
    xc = torch.zeros(Bn, A.shape[0], dtype=D)
    Y = torch.empty(Bn, T, 3, dtype=D); X = torch.empty(Bn, T, 6, dtype=D)
    U = torch.empty(Bn, T, 3, dtype=D)
    with torch.no_grad():
        for k in range(T):
            ym = x[:, :3, 0] @ Pt
            Y[:, k] = ym; X[:, k] = x[:, :, 0]
            e = y[:, k] - ym
            uk = u[:, k] + xc @ C.T + e @ Dc.T
            U[:, k] = uk
            xc = xc @ A.T + e @ Bc.T
            x = block.nonlinear_function(torch.cat([x, uk.unsqueeze(-1)], 1))
    return Y.numpy(), X.numpy(), U.numpy()


def dec(a):
    return sosfiltfilt(AA, a, axis=0)[::DEC].astype(np.float32)


def main():
    R = json.load(open(BASE))
    blk = make_block(raw=R['raw14'], cc=tuple(R['cc']), mode='tanh')
    t0 = time.time()
    store = {s: dict(names=[], res=[], xm=[], ucl=[], ud=[], uff=[], r=[], e=[], floor=[],
                     k_motion=[]) for s in ('train', 'validation', 'test')}
    chunk, meta = [], []

    def flush():
        data = pack(chunk)
        ym, xm, ucl = residual_rollout(blk, data)
        for i, (s, name) in enumerate(meta):
            n = data['n'][i]
            S = store[s]
            S['names'].append(name)
            S['res'].append(dec(ym[i, :n] - data['y'][i, :n]))
            S['xm'].append(dec(xm[i, :n])); S['ucl'].append(dec(ucl[i, :n]))
            S['ud'].append(dec(data['u'][i, :n])); S['uff'].append(dec(data['uff'][i, :n]))
            S['r'].append(dec(data['r'][i, :n])); S['e'].append(dec(data['e'][i, :n]))
            S['k_motion'].append(int(data['k_motion'][i] // DEC))
        print(f'[res] {len(meta)} records done ({time.time() - t0:.0f} s)', flush=True)

    for s, op, it, d in iter_records():
        k0 = d['motion_idx']
        store[s]['floor'].append(d['e'][:max(k0 - 200, 0)].astype(np.float32))
        chunk.append({k: d[k] for k in ('r', 'q1', 'e', 'u', 'u_fb', 'motion_idx')})
        meta.append((s, f'{s}/{op}/{it}'))
        if len(chunk) == CHUNK:
            flush(); chunk, meta = [], []
    if chunk:
        flush()
    for s, S in store.items():
        np.savez_compressed(
            os.path.join(OUT, f'{s}.npz'), names=np.array(S['names']),
            k_motion=np.array(S['k_motion']), fs=tp.FS / DEC,
            **{k: np.array(S[k], dtype=object) for k in ('res', 'xm', 'ucl', 'ud', 'uff', 'r', 'e',
                                                         'floor')})
        print(f'[res] {s}: {len(S["names"])} records saved')
    print(f'[res] done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
