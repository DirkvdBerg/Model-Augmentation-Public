"""Option B (DECISIONS.md BB2-018): Jan's plain SUBNET on the CLOSED-LOOP map (r, f) -> y.

Jan's own construction (`scripts/ecc_2025/msd_ndof_deepSI_encoder.py`): deepSI 0.3.29
`SS_encoder_general_hf`, f and h 2 x 8, encoder 2 x 16, deepSI's own `fit()`, sim-RMS
selection, auto_fit_norm on the training records. The only differences from Jan: the inputs are
the loop's exogenous signals r (reference) and f (injected force) instead of the plant input, the
records are the thesis ones through the pipeline's own resampling, and `.reshape` for `.view`.
The prediction is the same quantity the grey box is scored on: y of the loop under K1, given r
and f. No plant model results, so no controller-transfer test (E5) for this arm.

BB2-019 (`--target e`): the model's output signal is the servo error e = y - r instead of y, so
auto_fit_norm scales the loop's own behaviour to O(1); the prediction is y_hat = r + e_hat and the
score RMS(y_hat - y) = RMS(e_hat - e) is unchanged. `--in64` hands deepSI float64 records so its
normalisation runs before its own float32 cast. BB2-020 (`--dr`): the reference increment
r(k) - r(k-1) is added as three more input channels (r stays). BB2-029: `--f-blockmean` gives f
the grey box's block-mean rule, `--n-its` a fixed update budget. The other flags are Jan's settings (search axes,
REPORT.md); every default reproduces the BB2-018 runs b1 to b6.

Local:  python -u train_bb_clmap.py --version noisy --seed 1 --epochs 10 --out outputs/b/noisy_s1
"""
__project_origin__ = "added"

import argparse
import json
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--version', choices=('noisy', 'noisefree', 'lowpass'), required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--epochs', type=int, required=True)
p.add_argument('--nx', type=int, default=16)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--target', choices=('y', 'e'), default='y')            # BB2-019
p.add_argument('--in64', action='store_true')                           # BB2-019
p.add_argument('--dr', action='store_true')                             # BB2-020: add dr = r(k) - r(k-1)
p.add_argument('--na', type=int, default=29)                            # na = nb
p.add_argument('--na-right', type=int, default=0)                       # = nb_right
p.add_argument('--nf', type=int, default=None)                          # None: cfg.nf (0.1 s)
p.add_argument('--batch', type=int, default=None)                       # None: cfg.batch_size
p.add_argument('--stride', type=int, default=None)                      # None: cfg.stride
p.add_argument('--fh-width', type=int, default=8)                       # Jan: 8
p.add_argument('--fh-depth', type=int, default=2)                       # Jan: 2
p.add_argument('--e-width', type=int, default=16)                       # Jan: 16
p.add_argument('--e-depth', type=int, default=2)                        # Jan: 2
p.add_argument('--its-per-val', type=int, default=None)                 # None: every epoch (deepSI)
p.add_argument('--timeout', type=float, default=None)                   # deepSI's own wall-clock stop [s]
p.add_argument('--n-its', type=int, default=None)                       # BB2-029: fixed update budget (overrides epochs)
p.add_argument('--f-blockmean', action=argparse.BooleanOptionalAction, default=False)                 # BB2-029: f block-averaged like the grey box's u
p.add_argument('--cuda', action='store_true')                           # deepSI's own cuda switch (training only)
p.add_argument('--out', required=True)
p.add_argument('--server', action='store_true')
args = p.parse_args()

OUT = os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True)
os.environ['LOCALAPPDATA'] = os.path.join(OUT, 'localappdata')      # deepSI checkpoints here
os.makedirs(os.path.join(OUT, 'localappdata', 'deepSI', 'checkpoints'), exist_ok=True)

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402
import deepSI                                                          # noqa: E402
from deepSI.fit_systems.encoders import SS_encoder_general_hf, default_output_net  # noqa: E402
from scipy.io import loadmat                                           # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbcl'))
from setup import cfg_for                                              # noqa: E402
from jan_model import JanEncoder, JanStateNet                          # noqa: E402
from gantry_dynamic.data import record_files, traj_dir, _record_at_fs_new, _load_u, _resample_u  # noqa: E402
from gantry_dynamic.config import record_trim                         # noqa: E402

if not args.server:
    torch.set_num_threads(4)
NA = args.na                                                            # 29: the grey box's encoder lag
DT = np.float64 if args.in64 else np.float32


def load(cfg, files):
    """(r, f) -> target records at the pipeline's sample grid (its own resampling and trim).
    Returns the deepSI records and, per record, r and y in float64 (for scoring in metres)."""
    out, ry = [], []
    for fn in files:
        d = loadmat(os.path.join(traj_dir(cfg), fn), squeeze_me=True)
        _, y, r, f = _record_at_fs_new(d, cfg, ('r_sim', 'f_sim'))
        r = np.broadcast_to(r, y.shape) if np.ndim(r) else np.zeros_like(y)
        r, y = np.asarray(r, float), np.asarray(y, float)
        if args.f_blockmean:                                            # BB2-029: the grey box's u rule (D-087)
            n = len(_resample_u(_load_u(d), cfg))                       # the same N as _record_at_fs_new
            f = _resample_u(np.asarray(d['f_sim'], float), cfg)[:n]
            k = record_trim(cfg)
            f = f[k:n - k] if k else f
            if cfg.truncate_s is not None:
                f = f[:int(cfg.truncate_s * cfg.fs_new_hz)]
            assert f.shape == y.shape, (fn, f.shape, y.shape)
        cols = [r, np.asarray(f, float)]
        if args.dr:                                                     # BB2-020, float64 difference
            cols.append(np.diff(r, axis=0, prepend=r[:1]))
        u = np.hstack(cols)
        tgt = y - r if args.target == 'e' else y
        out.append(deepSI.System_data(u=np.ascontiguousarray(u, DT),
                                      y=np.ascontiguousarray(tgt, DT), dt=cfg.ts_new))
        ry.append((r, y))
    return out, ry


def per_axis(model, sdl, ry, k0):
    """Per record and axis RMS(y_hat - y) [m]; y_hat = r + e_hat for target e."""
    rms = []
    for sd, (r, y) in zip(sdl, ry):
        with np.errstate(all='ignore'):
            yp = np.asarray(model.apply_experiment(sd).y, float)
        if args.target == 'e':
            yp = r + yp
        rms.append(np.sqrt(np.mean((yp[k0:] - y[k0:]) ** 2, 0)))
    return np.asarray(rms)


def main():
    t0 = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = cfg_for(args.version, args.seed, local=not args.server)
    nf = args.nf or cfg.nf
    batch = args.batch or cfg.batch_size
    stride = args.stride or cfg.stride
    tr_f, va_f, _ = record_files(cfg)                                   # test records not loaded
    (train, train_ry), (val, val_ry) = load(cfg, tr_f), load(cfg, va_f)
    k0 = NA                                                             # section 10: scored from 29
    y0 = np.mean(np.concatenate([y for _, y in train_ry]), 0)
    mean_rms = np.mean([np.sqrt(np.mean((y[k0:] - y0) ** 2, 0)) for _, y in val_ry], 0)
    triv = np.asarray([np.sqrt(np.mean((r[k0:] - y[k0:]) ** 2, 0)) for r, y in val_ry])
    print('[B] %d train, %d val records; target %s, %s; mean-model per-axis RMS %s m; y_hat = r %s m'
          % (len(train), len(val), args.target, DT.__name__, np.array2string(mean_rms, precision=4),
             np.array2string(triv.mean(0), precision=4)), flush=True)
    fkw = dict(n_hidden_layers=args.fh_depth, n_nodes_per_layer=args.fh_width)
    ekw = dict(n_hidden_layers=args.e_depth, n_nodes_per_layer=args.e_width)
    hf_kw = dict(f_net=JanStateNet, f_net_kwargs=dict(fkw), h_net=default_output_net,
                 h_net_kwargs=dict(fkw))
    model = SS_encoder_general_hf(nx=args.nx, na=NA, nb=NA, hf_net_kwargs=hf_kw,
                                  e_net=JanEncoder, e_net_kwargs=dict(ekw),
                                  na_right=args.na_right, nb_right=args.na_right)
    tr, va = deepSI.System_data_list(train), deepSI.System_data_list(val)
    model.fit(tr, val_sys_data=va, epochs=args.epochs, batch_size=batch,
              loss_kwargs=dict(nf=nf, stride=stride), auto_fit_norm=True,
              validation_measure='sim-RMS', optimizer_kwargs=dict(lr=args.lr),
              its_per_val=args.its_per_val or 'epoch', timeout=args.timeout,
              n_its=args.n_its, cuda=args.cuda)
    pa = per_axis(model, val, val_ry, k0)
    ratio = (pa.mean(0) / mean_rms).tolist()
    nrms_e = (pa.mean(0) / triv.mean(0)).tolist()
    lv = np.asarray(model.Loss_val, float)
    lt = np.asarray(model.Loss_train, float)
    lt_f = lt[np.isfinite(lt)]
    # BB2-029: deepSI reloads '_best' at the end, which cuts the history at the best validation;
    # its own '_last' checkpoint (written just before) still holds the full curve.
    ck = os.path.join(OUT, 'localappdata', 'deepSI', 'checkpoints')
    last = [f for f in os.listdir(ck) if f.endswith('_last.pth')] if os.path.isdir(ck) else []
    full = torch.load(os.path.join(ck, last[0]), weights_only=False) if len(last) == 1 else {}
    lv_full = np.asarray(full.get('Loss_val', lv), float)
    lt_full = np.asarray(full.get('Loss_train', lt), float)
    bid_full = np.asarray(full.get('batch_id', []), float)
    crit = dict(a_finite=bool(np.all(np.isfinite(pa)) and np.all(np.isfinite(lv[1:]))),
                b_each_axis_below_third_of_mean_model=bool(np.all(np.isfinite(pa))
                                                           and max(ratio) <= 1 / 3),
                c_loss_decreases=bool(len(lt_f) >= 2 and lt_f[-1] < lt_f[0]))
    crit['PASS'] = all(crit.values())
    res = dict(args=vars(args), stride=stride, batch=batch, nf=nf, k0=k0,
               mean_model_rms=mean_rms.tolist(), trivial_rms=triv.mean(0).tolist(),
               trivial_per_record=triv.tolist(), per_axis_final=pa.mean(0).tolist(),
               per_record=pa.tolist(), ratio_final_over_mean_model=ratio, nrms_e=nrms_e,
               nrms_e_per_record=(pa / triv).tolist(),
               loss_val=lv.tolist(), loss_train=lt.tolist(), epoch_id=np.asarray(model.epoch_id).tolist(),
               loss_val_full=lv_full.tolist(), loss_train_full=lt_full.tolist(),
               updates_full=bid_full.tolist(), best_update=int(model.batch_counter),
               criteria=crit, wall_s=time.time() - t0)
    with open(os.path.join(OUT, 'result.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('[B] per-axis %s m, NRMS_e %s, /mean-model %s, criteria %s (%.0f s)'
          % (np.array2string(pa.mean(0), precision=4), np.round(nrms_e, 4), np.round(ratio, 4),
             json.dumps(crit), res['wall_s']), flush=True)


if __name__ == '__main__':
    main()
