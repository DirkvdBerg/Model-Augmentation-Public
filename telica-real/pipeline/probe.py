"""G6 memory probe + G5 M5 gradient SNR + smoke-run ANN-change check. NO optimizer step anywhere.

(1) ANN change of the g6_smoke run: its deepSI `_last` checkpoint (after the 2 updates) against its
    `_best` (= the initial weights, validation epoch 0).
(2) Memory: forward + backward of the closed-loop rollout at batch 8, nf 100 and 200, with the
    process working set before/after (Windows GetProcessMemoryInfo) -> MB per window-step.
(3) M5 (TR-017, descriptive): per-window gradients at initialisation over 32 train windows at the
    G5 horizon (nf = 800, 40 ms), forward + backward only. B_simple = tr(Sigma)/|G|^2
    (mccandlish2018empirical eq. 2.9) and per-parameter GSNR = gbar^2 / var (liu2020gsnr eq. 2).
Writes outputs/g6_probe/probe.json.
"""
import glob
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

os.environ.pop('TELICA_SMOKE', None)
from pipeline import telica_augment as ta                              # noqa: E402
from pipeline.real_data import load_datasets_real, frozen_norm         # noqa: E402
from pipeline.telica_model import build_model_real                     # noqa: E402
from pipeline.memory import rss_mb, peak_rss_mb, mem_probe             # noqa: E402
from controller.telica_bank import build_closed_loop_telica            # noqa: E402
tr_env.check_no_leak()

OUT = tr_env.out_dir('g6_probe')


def ann_of(hfn):
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    return next(m for m in hfn.connected_blocks if isinstance(m, Static_ANN_Block))


def main():
    res = {}
    # (1) smoke-run ANN change
    ck = sorted(glob.glob(os.path.join(tr_env.OUTPUTS, 'g6_smoke_*', 'deepsi_work', 'checkpoints',
                                       '*_last.pth')))
    if ck:
        last = torch.load(ck[-1], weights_only=False)
        best = torch.load(ck[-1].replace('_last.pth', '_best.pth'), weights_only=False)
        a1, a0 = ann_of(last['hfn']).state_dict(), ann_of(best['hfn']).state_dict()
        dmax = float(max((a1[k].double() - a0[k].double()).abs().max() for k in a0))
        res['smoke_ann_max_change'] = dmax
        res['smoke_batch_counter_last'] = int(last.get('batch_counter', -1))
        print(f'[probe] smoke run: ANN max |w_last - w_best| = {dmax:.3e}; batch_counter at _last '
              f'{res["smoke_batch_counter_last"]}', flush=True)
        del last, best
    # model at the server settings, CPU, eager
    cfg = replace(ta.make_cfg(), device='cpu', compile_mode=None, checkpoint_chunk=0, batch_size=8)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    data = load_datasets_real(cfg, max_records=16)
    norm = frozen_norm(cfg)
    hp = cfg.hp
    r0 = rss_mb()
    fit_sys = build_model_real(hp, cfg, data, norm, baseline_tag='_a2')
    fit_sys.simulator = build_closed_loop_telica(fit_sys, norm, cfg, train_files=data.train_names,
                                                 val_files=data.val_names,
                                                 val_data=data.val_ckpt_data, verbose=False)
    print(f'[probe] model built: rss {rss_mb():.0f} MB (before build {r0:.0f}), nf={cfg.nf}', flush=True)
    # (2) memory
    res['mem'] = mem_probe(fit_sys, cfg, data, batch=8, nfs=(100, 200))
    res['rss_after_mem_probe'] = rss_mb(); res['peak_after_mem_probe'] = peak_rss_mb()
    # (3) M5 gradient SNR at init, nf = cfg.nf (G5 horizon)
    t0 = time.time()
    arrs = fit_sys.make_training_data(fit_sys.norm.transform(data.train_data), nf=cfg.nf,
                                      stride=2000)
    n_win = len(arrs[0])
    pick = np.linspace(0, n_win - 1, min(32, n_win)).astype(int)
    params = [(n, p) for n, p in fit_sys.named_parameters() if p.requires_grad] \
        if hasattr(fit_sys, 'named_parameters') else None
    if params is None:
        params = [(f'encoder.{n}', p) for n, p in fit_sys.encoder.named_parameters()] + \
                 [(f'hfn.{n}', p) for n, p in fit_sys.hfn.named_parameters()]
    params = [(n, p) for n, p in params if p.requires_grad]
    G = []
    for i in pick:
        for _, p in params:
            p.grad = None
        batch = [torch.as_tensor(a[i:i + 1]) for a in arrs]
        uh, yh, uf, yf = (b.to(cfg.dtype_pt) for b in batch[:4])
        loss = fit_sys.loss(uh, yh, uf, yf, ctrl_ix=batch[4].long())
        loss.backward()
        G.append(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                            for _, p in params]).double().numpy())
    G = np.stack(G)                                                   # (N, P)
    gbar = G.mean(0); var = G.var(0, ddof=1)
    B_simple = float(var.sum() / (gbar @ gbar))
    nz = var > 0
    gsnr = gbar[nz] ** 2 / var[nz]
    sizes = [p.numel() for _, p in params]
    off = np.cumsum([0] + sizes)
    groups = {}
    for (n, p), a, b in zip(params, off[:-1], off[1:]):
        key = 'ann' if 'connected_blocks.2' in n or 'ann' in n.lower() else n.split('.')[0]
        groups.setdefault(key, []).append((a, b))
    per_group = {}
    for k, spans in groups.items():
        ix = np.concatenate([np.arange(a, b) for a, b in spans])
        v = var[ix]; g = gbar[ix]; m = v > 0
        per_group[k] = dict(n_params=int(len(ix)), n_nonzero=int(m.sum()),
                            B_simple=float(v.sum() / max(g @ g, 1e-300)),
                            gsnr_median=float(np.median(g[m] ** 2 / v[m])) if m.any() else None,
                            frac_gsnr_gt1=float(np.mean(g[m] ** 2 / v[m] > 1)) if m.any() else None)
    res['M5'] = dict(n_windows=int(len(pick)), n_train_windows_available=int(n_win), nf=cfg.nf,
                     B_simple=B_simple, gsnr_median=float(np.median(gsnr)),
                     frac_gsnr_gt1=float(np.mean(gsnr > 1)), per_group=per_group,
                     wall_s=time.time() - t0)
    print(f'[M5] {len(pick)} windows of nf={cfg.nf}: B_simple = {B_simple:.2f} (critical batch scale; '
          f'train windows at stride 2000: {n_win}); GSNR median {np.median(gsnr):.3f}, share > 1 '
          f'{np.mean(gsnr > 1):.3f} ({time.time() - t0:.0f} s)')
    for k, v in per_group.items():
        print(f'[M5]   {k}: {v}')
    res['peak_rss_mb'] = peak_rss_mb()
    json.dump(res, open(os.path.join(OUT, 'probe.json'), 'w'), indent=1)
    print(f'[probe] peak process working set {res["peak_rss_mb"]:.0f} MB')


if __name__ == '__main__':
    main()
