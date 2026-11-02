"""Training of one black-box configuration: deepSI's `fit()` with the weighted loss
L = mean (w (e_hat - e))^2, w = sigma_y / sigma_e, and checkpoint selection by validation NRMS_e.
Conventions: sigma_y is deepSI's output scale `norm.ystd`; sigma_e is the std of e per record set and axis from
k = n + 1; `encoder` is psi, `hfn` returns (h_bb, f_bb); `yhat` is e_hat.

Local:  python -u Clean/blackbox/train_bb.py --out <folder>       (the selected configuration, index 9)
Server: Clean/blackbox/run_r8.sh (the 11 configurations of thesis_r8_grid.conf)
"""
__project_origin__ = "added"

import argparse
import json
import os
import time

p = argparse.ArgumentParser()
p.add_argument('--nx', type=int, default=16)
p.add_argument('--fh-width', type=int, default=16)                 # f_bb and h_bb: 2 hidden layers of this width
p.add_argument('--e-width', type=int, default=32)                  # psi: 2 hidden layers of this width
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--n-its', type=int, default=80000)
p.add_argument('--its-per-val', type=int, default=1000)
p.add_argument('--truncate-s', type=float, default=None)           # short test runs: first truncate_s s per record
p.add_argument('--out', required=True)
args = p.parse_args()

OUT = os.path.abspath(args.out)
# deepSI writes its checkpoints below LOCALAPPDATA on Windows (~/.deepSI on Linux); set before its import
os.environ['LOCALAPPDATA'] = os.path.join(OUT, 'localappdata')
os.makedirs(os.path.join(OUT, 'localappdata', 'deepSI', 'checkpoints'), exist_ok=True)

import numpy as np                                                                  # noqa: E402
import torch                                                                        # noqa: E402
import deepSI                                                                       # noqa: E402
from deepSI.fit_systems.encoders import SS_encoder_general_hf                       # noqa: E402

from bb import NA, RunConfig, data_cfg, load_records, model_kwargs                  # noqa: E402
from gantry_dynamic.data import record_files, servo_sigma                          # noqa: E402

# One seed, batch size and window stride (Table hyper); 4 threads = the server's cores per task.
SEED, BATCH, STRIDE = 1, 256, 25
torch.set_num_threads(4)


def per_record_rms(model, sds, ry):
    """Free-run RMS error per record.

    RMS_j = RMS(y_hat - y) from k = n + 1, y_hat = r + e_hat.

    Args:
        model: the black box.
        sds: deepSI System_data per record, u = (r, u^ms) and e, physical.
        ry: (r, y) per record, each (N, 3), physical.
    Returns:
        (n_records, 3) RMS_j, physical.
    """
    with np.errstate(all='ignore'):
        return np.array([np.sqrt(np.mean((r[NA:] + np.asarray(model.apply_experiment(sd).y, float)[NA:] - y[NA:])
                                          ** 2, 0)) for sd, (r, y) in zip(sds, ry)])


class ServoWeightedEncoder(SS_encoder_general_hf):
    """SUBNET with the weighted loss and validation by NRMS_e. The history and failure lists are passed in
    by reference because deepSI's `_best` reload replaces `__dict__`."""

    def set_objective(self, sigma_train, sigma_val, val_ry, run_lists, hist_path):
        """Stores the objective's data on the model.

        Args:
            sigma_train: (n_records, 3) sigma_e of the training records.
            sigma_val: (n_records, 3) sigma_e of the validation records.
            val_ry: (r, y) per validation record, physical.
            run_lists: (history, failure), lists owned by the run.
            hist_path: file the history is written to.
        """
        self._sigma_train, self._val_sigma, self._val_ry, self._hist_path = sigma_train, sigma_val, val_ry, hist_path
        self._hist, self._failure = run_lists

    def make_training_data(self, sys_data, **kw):
        """deepSI's training windows plus the weight of each window.

        w = sigma_y / sigma_e of the window's record, per axis.
        The weight makes the normalised loss an error relative to each record's servo error, as NRMS_e.

        Args:
            sys_data: training records (deepSI System_data_list).
            **kw: deepSI's window settings, nf (horizon) and stride.
        Returns:
            deepSI's [uhist, yhist, ufuture, yfuture] plus w, (n_windows, 3).
        """
        data = super().make_training_data(sys_data, **kw)
        # Windows per record, as deepSI counts them: k from na + nf to N in steps of stride.
        counts = [len(range(self.na + kw['nf'], len(sd.u) + 1, kw['stride'])) for sd in sys_data.sdl]
        if sum(counts) != len(data[0]):
            raise RuntimeError('weight rows %d != training windows %d' % (sum(counts), len(data[0])))
        # Weight: w = sigma_y / sigma_e, so w (e_hat^n - e^n) = (e_hat - e) / sigma_e.
        w = np.asarray(self.norm.ystd, float).ravel() / self._sigma_train
        return [*data, np.concatenate([np.tile(wr, (c, 1)) for wr, c in zip(w, counts)])]

    def loss(self, uhist, yhist, ufuture, yfuture, w, **kw):
        """Weighted simulation loss of a batch of windows.

        L = mean over windows, steps k and axes of (w (e_hat_k^n - e_k^n))^2.

        Args:
            uhist: (B, n_b, n_u) past inputs u^n, normalised.
            yhist: (B, n_a, n_y) past servo errors e^n, normalised.
            ufuture: (B, nf, n_u) inputs u^n over the horizon.
            yfuture: (B, nf, n_y) servo errors e^n over the horizon.
            w: (B, n_y) window weights from `make_training_data`.
        Returns:
            Scalar loss L.
        """
        # 1. Encode: x^bb_0 = psi(u_past, e_past).
        x = self.encoder(uhist, yhist)
        # 2. Simulate the horizon.
        errors = []
        for y, u in zip(torch.transpose(yfuture, 0, 1), torch.transpose(ufuture, 0, 1)):
            # Step: e_hat_k = h_bb(x^bb_k), x^bb_k+1 = f_bb(x^bb_k, u_k).
            yhat, x = self.hfn(x, u)
            errors.append(((yhat - y) * w) ** 2)
        L = torch.mean(torch.stack(errors))
        # 3. Record the first non-finite loss.
        if not self._failure and not torch.isfinite(L):
            # deepSI stops on a non-finite loss and reloads `_best` without a record, so the run records it
            self._failure.append({'update': int(self.batch_counter), 'loss': float(L)})
            self._dump()
        return L

    def cal_validation_error(self, val_sys_data, validation_measure=None):
        """Validation score of the black box.

        NRMS_e = mean over records and axes of RMS_j / sigma_e,j.
        deepSI keeps the checkpoint with the lowest score as `_best`; each score is also added to the history.

        Args:
            val_sys_data: validation records (deepSI System_data_list).
            validation_measure: unused; the score is always NRMS_e.
        Returns:
            NRMS_e as a float.
        """
        rms = per_record_rms(self, val_sys_data.sdl, self._val_ry)
        score = float(np.mean(rms / self._val_sigma))
        self._hist.append({'update': int(self.batch_counter), 'nrms_e': score, 'rms': rms.tolist()})
        self._dump()
        return score

    def _dump(self):
        """Writes history and failure at every event, so a killed job keeps them."""
        with open(self._hist_path, 'w') as fh:
            json.dump({'val_sigma_e': self._val_sigma.tolist(), 'history': self._hist,
                       'failure': self._failure[0] if self._failure else None}, fh)


def main():
    """Trains one configuration and writes result.json."""
    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # 1. Data: inputs u = (r, u^ms) and target e = y - r per record; sigma_e from k = n + 1.
    cfg = data_cfg(args.truncate_s)
    tr_f, va_f, _ = record_files(cfg)
    train, val = load_records(tr_f, cfg), load_records(va_f, cfg)
    sds = [[deepSI.System_data(u=np.ascontiguousarray(u), y=np.ascontiguousarray(y - r), dt=RunConfig.ts_new)
            for u, r, y in recs] for recs in (train, val)]
    val_ry = [(r, y) for _, r, y in val]
    sigma_val = servo_sigma([y - r for r, y in val_ry], NA)
    print('[B] %d train, %d val records' % (len(train), len(val)), flush=True)

    # 2. Model and objective.
    model = ServoWeightedEncoder(**model_kwargs(args.nx, args.fh_width, args.e_width))
    val_hist, failure = [], []
    model.set_objective(servo_sigma([y - r for _, r, y in train], NA), sigma_val, val_ry, (val_hist, failure),
                        os.path.join(OUT, 'val_hist.json'))
    # 3. Fit: Adam on L over windows of nf steps, NRMS_e on the validation records every its_per_val updates.
    model.fit(deepSI.System_data_list(sds[0]), val_sys_data=deepSI.System_data_list(sds[1]), n_its=args.n_its,
              batch_size=BATCH, loss_kwargs=dict(nf=RunConfig.nf, stride=STRIDE), validation_measure='sim-NRMSe',
              optimizer_kwargs=dict(lr=args.lr), its_per_val=args.its_per_val)

    # 4. Result of the best checkpoint, which deepSI has reloaded; the loss curves stay in its `_last.pth`.
    pa = per_record_rms(model, sds[1], val_ry)
    res = dict(args=vars(args), k0=NA, checkpoint=model.name + '_best.pth', best_update=int(model.batch_counter),
               status='failed' if failure else 'completed', failure=failure[0] if failure else None,
               per_record=pa.tolist(), nrms_e_per_record=(pa / sigma_val).tolist(), val_rms_hist=val_hist,
               val_sigma_e=sigma_val.tolist(), wall_s=time.time() - t0)
    with open(os.path.join(OUT, 'result.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('[B] best update %d, validation NRMS_e %.4f (%.0f s)'
          % (res['best_update'], np.mean(res['nrms_e_per_record']), res['wall_s']), flush=True)


if __name__ == '__main__':
    main()
