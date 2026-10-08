"""Timing of one training update: local GPU (Quadro P2000, eager, float64) against the CPU, at batch
512 (production) and 64 (probe size). Thesis model of n_phase1.setup (SplitNet, gate 1), loss = fs.loss
on random nf 400 windows. Nothing is saved; no optimizer step is kept beyond the timed ones.

Usage: python t_timing.py
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('N_GATE', '1')
import n_phase1 as n1                                                          # noqa: E402


def time_updates(fs, opt, recs, na_r, nb_r, B, dev, dt, n=3):
    rng = np.random.default_rng(0)
    ts = []
    for _ in range(n):
        b = [t.to(dev) for t in n1.batch(recs, fs, na_r, nb_r, B, rng, dt)]
        if dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad()
        loss = fs.loss(*b[:4], ctrl_ix=b[4], nf=n1.NF)
        loss.backward()
        opt.step()
        if dev == 'cuda':
            torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return ts, float(loss)


def main():
    cfg, data, norm, fs, ann, others = n1.setup()
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    opt = torch.optim.Adam(list(fs.hfn.parameters()) + list(fs.encoder.parameters()), lr=1e-12, eps=cfg.adam_eps)
    print('threads %d' % torch.get_num_threads(), flush=True)
    for B in (64, 512):
        ts, l = time_updates(fs, opt, recs, na_r, nb_r, B, 'cpu', cfg.dtype_pt)
        print('cpu  batch %3d: %s s/upd (loss %.3e)' % (B, ' '.join('%.2f' % t for t in ts), l), flush=True)
    try:
        fs.cuda()
        for B in (64, 512):
            ts, l = time_updates(fs, opt, recs, na_r, nb_r, B, 'cuda', cfg.dtype_pt)
            print('cuda batch %3d: %s s/upd (loss %.3e), peak mem %.0f MB' % (
                B, ' '.join('%.2f' % t for t in ts), l, torch.cuda.max_memory_allocated() / 2 ** 20), flush=True)
    except Exception as e:                                                     # report, do not hide
        print('cuda FAILED: %s: %s' % (type(e).__name__, e), flush=True)


if __name__ == '__main__':
    main()
