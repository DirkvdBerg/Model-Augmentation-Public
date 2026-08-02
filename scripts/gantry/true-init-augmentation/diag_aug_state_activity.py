"""Did training leave the dead-beat corner? Measure rows 6-7 directly.

WHY THIS EXISTS, AND WHY IT CORRECTS AN EARLIER READING.
`model.py:131` wires the ANN's input as `connect_block_signals(ann_block,
["x","u"], [])` with NO selection matrix, so the ANN reads all `nxd = 8` state
rows, rows 6-7 included (`nz = nxd + nu = 11`). `model.py:132` routes its output
into those same rows. The augmented partition is therefore a learnable nonlinear
RECURRENCE,

    x_aug(k+1) = h_ann( x_phys(k), x_aug(k), u(k) ),

not a state "rebuilt from scratch by a static feedforward net every sample" as
`scripts/gantry/coulomb-offset/IMPLEMENTATION-LOG.md` section 12.2 puts it. Gate
G6's `x_aug = 0.000000e+00` is an INITIALISATION result: the zero-initialised
final layer makes the ANN output exactly zero, so the recurrence is dead-beat and
its Jacobian `d x_aug(k+1) / d x_aug(k)` is exactly zero AT INIT. It does not have
to stay that way.

That distinction decides the section-8 attribution, so it is measured rather than
argued. Three quantities, per checkpoint:

  A  the magnitude of x_aug along a real window, against the truth's own
     [delta_a, vdelta_a] scale. Zero means the corner was never left.
  J  the recurrence gain, the largest singular value of d x_aug(k+1)/d x_aug(k),
     averaged over the window. Zero = dead-beat; a damped oscillator needs it
     near, but below, 1.
  C  the correlation of each augmented row with the truth's absorber state, best
     affine map, since rows 6-7 are latent and need not be aligned with anything.

Run:
  ... python -u scripts/gantry/true-init-augmentation/diag_aug_state_activity.py \\
        --ckpt <a.pt> <b.pt> ...
"""
__project_origin__ = "added"

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))

from gantry_dynamic.data import load_datasets, compute_normalization   # noqa: E402
from true_init_train import CFG, build_interconnect, Windows           # noqa: E402
from data_exact import exact_truth                                     # noqa: E402

OUT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')
REC = 'V1_standstill_Yp10'
NF, K0 = 400, 17


def aug_trace(ic, x0, u):
    """Roll a batch of windows, returning x_aug over time. x0 (B,8), u (B,nf,3)."""
    x, out = x0, []
    with torch.no_grad():
        for t in range(u.shape[1]):
            _, x = ic(x, u[:, t])
            out.append(x[:, 6:8].clone())
    return torch.stack(out, dim=1).numpy()          # (B, nf, 2)


def recurrence_gain(ann, x, u):
    """Largest singular value of d x_aug(k+1) / d x_aug(k), per sample."""
    z = torch.cat([x, u], dim=1).clone().requires_grad_(True)
    gains = []
    for i in range(z.shape[0]):
        J = torch.autograd.functional.jacobian(
            lambda zz: ann.net(zz.unsqueeze(0))[0, 6:8], z[i].detach())
        gains.append(float(torch.linalg.svdvals(J[:, 6:8])[0]))
    return float(np.mean(gains)), float(np.max(gains))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', required=True)
    a = ap.parse_args()

    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    data = load_datasets(CFG)
    norm = compute_normalization(CFG, data)
    dtype = torch.float32
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    ic, ann, _ = build_interconnect(CFG, norm, cog=True, dtype=dtype)
    init_state = {k: v.clone() for k, v in ann.state_dict().items()}

    tr = exact_truth(REC)
    rec, x6, x8 = tr['rec'], tr['x6'], tr['x8']
    starts = np.arange(K0, len(x6) - NF + 1, 400)[:64]
    u = torch.stack([torch.as_tensor(
        (rec['u'][s:s + NF] - norm.u_mean.flatten()) / norm.std_u.flatten(),
        dtype=dtype) for s in starts])
    x0 = torch.zeros(len(starts), 8, dtype=dtype)
    x0[:, :6] = torch.as_tensor(
        (x6[starts] - norm.x_mean.flatten()) / norm.std_x.flatten(), dtype=dtype)
    da_true = np.stack([x8[s:s + NF][:, [3, 7]] for s in starts])       # (B, nf, 2)

    print(f'Augmented-state activity, {REC}, {len(starts)} windows of {NF}\n')
    print(f'  truth absorber over these windows: |delta_a| rms {da_true[:,:,0].std():.4e} m, '
          f'|vdelta_a| rms {da_true[:,:,1].std():.4e} m/s')
    print(f'\n  {"checkpoint":<26}{"|x_aug| rms":>14}{"|x_aug| max":>14}'
          f'{"J gain mean":>14}{"J gain max":>13}{"R2 vs [da,vda]":>16}')
    res = {}
    for name in ['(untrained)'] + a.ckpt:
        ann.load_state_dict(init_state if name == '(untrained)'
                            else torch.load(name, map_location='cpu'))
        xa = aug_trace(ic, x0, u)
        jm, jx = recurrence_gain(ann, x0[:16], u[:16, 0])
        # best affine map from x_aug to the truth's absorber state, so a latent
        # rotation or scaling is not counted as failure
        F = xa.reshape(-1, 2)
        T = da_true.reshape(-1, 2) / np.array([da_true[:, :, 0].std(),
                                               da_true[:, :, 1].std()])
        A = np.hstack([F, np.ones((len(F), 1))])
        c, *_ = np.linalg.lstsq(A, T, rcond=None)
        ss = 1.0 - ((T - A @ c) ** 2).sum() / max(((T - T.mean(0)) ** 2).sum(), 1e-300)
        tag = os.path.basename(name).replace('ann_', '').replace('.pt', '')
        res[tag] = dict(rms=float(xa.std()), max=float(np.abs(xa).max()),
                        j_mean=jm, j_max=jx, r2=float(ss))
        print(f'  {tag:<26}{xa.std():>14.4e}{np.abs(xa).max():>14.4e}'
              f'{jm:>14.4e}{jx:>13.4e}{ss:>16.4f}')

    print('\n  READING')
    print('   |x_aug| ~ 0 and J ~ 0  ->  the dead-beat corner was NEVER left. The')
    print('                              augmented partition carried no state, so the')
    print('                              model was a static map of (x_phys, u)')
    print('                              throughout and section 5 applies to it as run.')
    print('   |x_aug| > 0 but R2 low ->  the rows are active but carry something other')
    print('                              than the absorber.')
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'true_init_aug_activity.json')
    with open(p, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'\n  wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
