"""Did Jan's MSD dynamic-parallel models learn the missing mode? (read-only, his saved models)

Loads Jan's trained deepSI systems (data/<FP>/SNR<snr>/msd_3dof_*_e3000, from
scripts/ecc_2025/msd_ndof_interconnect_dynamic.py), linearises the one-step map at encoder states of
the training windows (msd_3dof_multisine_train.npz) and compares its eigenvalues with the truth's
(3-DOF linear part, msd_3dof.mat Ad) and the baseline's (2-DOF, msd_2dof[_non_ideal].mat Ad).
Same measurement as h_learned_poles.py (coordinate-free eigenvalues). Nothing is trained or written
except outputs/p_jan_msd_poles.log via stdout.

Usage: python p_jan_msd_poles.py
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
import deepSI                                                                  # noqa: E402
from scipy.io import loadmat                                                    # noqa: E402
from model_augmentation.fit_systems.interconnect import *                      # noqa: E402,F401,F403
from model_augmentation.fit_systems.blocks import *                            # noqa: E402,F401,F403
from model_augmentation.systems.mass_spring_damper import *                    # noqa: E402,F401,F403

MSD = os.path.join(REPO, 'data', 'mass_spring_damper')
NPTS = 300


def modes(ev, ts):
    ev = ev[np.imag(ev) > 1e-9]
    s = np.log(ev) / ts
    return sorted([(abs(s_) / (2 * np.pi), -np.real(s_) / abs(s_), abs(e)) for s_, e in zip(s, ev)])


def fmt(ms):
    return '  '.join('%.2f Hz z%.3f |z|%.3f' % m for m in ms) or 'none'


def main():
    ts = float(loadmat(os.path.join(MSD, 'msd_3dof.mat'))['Ts'][0, 0])
    print('Ts %.3f s' % ts)
    print('truth 3-DOF (linear part):      %s' % fmt(modes(np.linalg.eigvals(loadmat(os.path.join(MSD, 'msd_3dof.mat'))['Ad']), ts)))
    for fp, f in (('ideal', 'msd_2dof.mat'), ('approximate', 'msd_2dof_non_ideal.mat')):
        print('baseline 2-DOF (%-11s):   %s' % (fp, fmt(modes(np.linalg.eigvals(loadmat(os.path.join(MSD, f))['Ad']), ts))))
    train = deepSI.load_system_data(os.path.join(MSD, 'msd_3dof_multisine_train.npz'))
    for fp in ('ideal', 'approximate'):
        for snr in (20, 30, 60):
            for kind in [k + e for k in ('dynamic_parallel', 'linear_dynamic_parallel') for e in ('_e1000', '_e3000')]:
                path = os.path.join(REPO, 'data', fp, 'SNR%d' % snr, 'msd_3dof_%s' % kind)
                if not os.path.exists(path):
                    continue
                try:
                    fs = deepSI.load_system(path)
                except Exception as e:                                     # noqa: BLE001
                    print('%s SNR%d %s: load failed (%s)' % (fp, snr, kind, str(e)[:80])); continue
                for b in fs.hfn.connected_blocks:          # pickles predate the all-ones out_gate buffer
                    if isinstance(b, Static_ANN_Block) and 'out_gate' not in b.__dict__.get('_buffers', {}):
                        b.register_buffer('out_gate', torch.ones(b.nw), persistent=False)
                sdn = fs.norm.transform(train)
                uh, yh, uf, yf = sdn.to_hist_future_data(na=fs.na, nb=fs.nb, nf=1,
                                                         na_right=getattr(fs, 'na_right', 0), nb_right=getattr(fs, 'nb_right', 0))[:4]
                pick = np.linspace(0, len(uf) - 1, NPTS).round().astype(int)
                p = next(fs.hfn.parameters())
                T = lambda a: torch.as_tensor(np.ascontiguousarray(a[pick]), dtype=p.dtype)   # noqa: E731
                with torch.no_grad():
                    x = fs.encoder(T(uh), T(yh))
                u = T(uf)[:, 0].reshape(len(pick), -1)
                f = lambda xx: fs.hfn(xx, u)[1].reshape(len(pick), -1).sum(0)             # noqa: E731
                J = torch.autograd.functional.jacobian(f, x.clone()).permute(1, 0, 2).detach().numpy()
                ms = [modes(np.linalg.eigvals(j), ts) for j in J]
                nmode = np.array([len(m) for m in ms])
                allm = np.array([m for mm in ms for m in mm])
                print('\n%s SNR%d %s (nx %d): complex pairs per point %s' % (
                    fp, snr, kind, J.shape[1], dict(zip(*np.unique(nmode, return_counts=True)))))
                if len(allm):
                    for lo, hi in ((0.5, 2.5), (2.5, 4.7), (4.7, 12.5)):
                        sel = allm[(allm[:, 0] >= lo) & (allm[:, 0] < hi)]
                        if len(sel):
                            print('   %4.1f-%4.1f Hz: %3d pts, median f %.2f Hz, zeta %.3f (p10-p90 %.3f to %.3f), |z| %.3f' % (
                                lo, hi, len(sel), np.median(sel[:, 0]), np.median(sel[:, 1]),
                                np.percentile(sel[:, 1], 10), np.percentile(sel[:, 1], 90), np.median(sel[:, 2])))
                print('   mean-Jacobian modes: %s' % fmt(modes(np.linalg.eigvals(J.mean(0)), ts)), flush=True)


if __name__ == '__main__':
    main()
