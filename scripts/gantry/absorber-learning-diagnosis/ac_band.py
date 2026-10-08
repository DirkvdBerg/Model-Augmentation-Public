"""AC1: output-side check of probe checkpoints on the VALIDATION records (held out from training).

For each checkpoint (u_probe / f_eeoe / f2_open format, zero-init RowSplitNet model): closed-loop rollouts of
nf 400 from encoder states on non-overlapping windows of every validation record (the simulator's val_records,
each with its own controller row), with the learned block ON and OFF (Static_ANN_Block out_gate 0, i.e. the
baseline at the checkpoint's physical parameters). Reported: mean squared error (the thesis metric) on vs off,
and per channel the error energy per band from the windows' FFT (400 samples, 10 Hz bins) and the fraction
removed by the learned block, 1 - E_on / E_off.
# HEURISTIC: proxy for the thesis criterion "> 22 % of the Y error removed in 230 to 297 Hz" (run 42's value,
# computed on full multisine val records against the TRUE-parameter baseline); here windows of 0.1 s and the
# checkpoint's own baseline, the same for every checkpoint, so it ranks probes, it does not reproduce run 42.

Usage: python ac_band.py CKPT [CKPT ...]   (files in outputs/)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n_phase1 as n1                                                          # noqa: E402
import u_probe as up                                                           # noqa: E402

BANDS = [(20, 106), (106, 140), (140, 230), (230, 297), (297, 2000)]


def windows(fs, cfg):
    na_r, nb_r = getattr(fs, 'na_right', 0), getattr(fs, 'nb_right', 0)
    UH, YH, UF, YF, C = [], [], [], [], []
    for name, sd, row in fs.simulator.val_records:
        sdn = fs.norm.transform(sd)
        u, y = np.asarray(sdn.u), np.asarray(sdn.y)
        for k in range(max(fs.na, fs.nb), len(u) - n1.NF + 1, n1.NF):
            UH.append(u[k - fs.nb:k + nb_r]); YH.append(y[k - fs.na:k + na_r])
            UF.append(u[k:k + n1.NF]); YF.append(y[k:k + n1.NF]); C.append(row)
    T = lambda a: torch.as_tensor(np.ascontiguousarray(np.stack(a)), dtype=cfg.dtype_pt)   # noqa: E731
    return T(UH), T(YH), T(UF), T(YF), torch.as_tensor(C, dtype=torch.long)


def band_energy(e):
    """e (W, nf, ny) -> (n_bands, ny) summed |FFT|^2 per band."""
    E = np.abs(np.fft.rfft(e - e.mean(1, keepdims=True), axis=1)) ** 2
    f = np.fft.rfftfreq(e.shape[1], 1 / up.FS)
    return np.array([E[:, (f >= a) & (f < b)].sum((0, 1)) for a, b in BANDS])


def main(files):
    os.environ['U_INIT'] = 'zero'
    cfg, data, norm, fs, ann, others = up.setup()
    W = None
    for fname in files:
        st = torch.load(os.path.join(HERE, 'outputs', fname), weights_only=False)
        fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
        if W is None:
            W = windows(fs, cfg)
            print('\n######## AC1 validation band check: %d windows of %d samples from %d val records' % (
                len(W[0]), n1.NF, len(fs.simulator.val_records)), flush=True)
        res = {}
        for tag, gate in (('on', None), ('off', 0.0)):
            g = ann.out_gate.clone()
            if gate is not None:
                ann.out_gate = torch.zeros_like(g)
            try:
                with torch.no_grad():
                    x0 = fs.encoder(W[0], W[1])
                    yp = torch.cat([fs.simulate(x0[i:i + 128], W[2][i:i + 128], W[3][i:i + 128],
                                                ctrl_ix=W[4][i:i + 128], nf=n1.NF)[0]
                                    for i in range(0, len(x0), 128)])
            finally:
                ann.out_gate = g
            e = (W[3] - yp).numpy()
            res[tag] = (float((e ** 2).mean()), band_energy(e))
        rem = 1 - res['on'][1] / res['off'][1]
        print('  [%s, %d updates] val MSE on %.4e, off %.4e (ratio %.3f)' % (
            fname, st['n_upd'], res['on'][0], res['off'][0], res['on'][0] / res['off'][0]), flush=True)
        for c in range(rem.shape[1]):
            print('     ch %d removed: %s' % (c, '  '.join('%d-%d Hz %+.2f' % (a, b, r) for (a, b), r in zip(BANDS, rem[:, c]))),
                  flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
