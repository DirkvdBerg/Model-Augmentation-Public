"""D4 (diagnostic, no training): is x_a a self-propagating state, or only a predictive window summary?

For each checkpoint, on 256 windows of the validation records: the encoder state x(k) starts a rollout of the deployed
one-step map with the RECORDED inputs, x(k+j) = F(x(k+j-1), u(k+j-1)) (the S2 convention; not the closed loop), and the
rolled-out added states are compared with the independently RE-ENCODED x_a(k+j) from the measured window ending at k+j.
Reported per horizon j in (1, 10, 30, 100): RMS(x_a_roll - x_a_enc) / RMS(x_a_enc). Values near 0 at long horizons mean
the encoder's added-state coordinate is propagated by the deployed map; large values mean the encoder carries window
information the map cannot reproduce. Coordinates are a gauge, so this is a consistency diagnostic between the model's
own encoder and its own map, not a check against any physical state. Run with the same P_DATA as the checkpoints.

Usage: python d4_rollout_consistency.py CKPT [CKPT ...]
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import n_phase1 as n1                                                          # noqa: E402
import u_probe as up                                                           # noqa: E402
from ps2 import select_data                                                    # noqa: E402
from evalc import path                                                         # noqa: E402

H = (1, 10, 30, 100)


def main(files):
    os.environ['U_INIT'] = 'zero'
    select_data()
    cfg, data, norm, fs, ann, others = up.setup()
    # validation records as normalised arrays (the n1.windows convention, applied to the val list)
    recs = []
    for sd in data.val_list:
        sdn = fs.norm.transform(sd)
        u, y = np.asarray(sdn.u), np.asarray(sdn.y)
        k0 = max(fs.na, fs.nb)
        recs.append((u, y, np.arange(k0, len(u) - max(H) - 2), 0))
    na_r, nb_r = getattr(fs, 'na_right', 0), getattr(fs, 'nb_right', 0)
    rng = np.random.default_rng(5)
    picks = [(r, recs[r][2][rng.integers(len(recs[r][2]))]) for r in rng.integers(len(recs), size=256)]
    dt = cfg.dtype_pt
    T = lambda a: torch.as_tensor(np.ascontiguousarray(np.stack(a)), dtype=dt)          # noqa: E731

    def hist(j):
        UH = [recs[r][0][k + j - fs.nb:k + j + nb_r] for r, k in picks]
        YH = [recs[r][1][k + j - fs.na:k + j + na_r] for r, k in picks]
        return T(UH), T(YH)
    U = T([recs[r][0][k:k + max(H)] for r, k in picks])                                    # (B, Hmax, nu)
    print('\n######## D4 rollout consistency of x_a (validation windows: %d)' % len(picks), flush=True)
    for f in files:
        st = torch.load(path(f), weights_only=False)
        fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
        out = []
        with torch.no_grad():
            x = fs.encoder(*hist(0))
            enc = {j: fs.encoder(*hist(j))[:, 6:8] for j in H}
            for j in range(1, max(H) + 1):
                x = fs.hfn(x, U[:, j - 1])[1].reshape(len(x), -1)
                if j in H:
                    e = enc[j]
                    out.append('j=%d %.3f' % (j, float((x[:, 6:8] - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt())))
        print('  %-14s %s' % (f, ' | '.join(out)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
