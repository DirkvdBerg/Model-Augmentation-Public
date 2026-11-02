"""Training of one black-box configuration: deepSI's fit() on the loss weighted per record and axis by
sigma_y / sigma_e, keeping the checkpoint with the lowest validation NRMS_e."""
__project_origin__ = "added"

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import deepSI
from deepSI.fit_systems.encoders import SS_encoder_general_hf

from bb import GREY_BOX, data_cfg, load_records, model_kwargs
from gantry_dynamic.config import OUT_ROOT
from gantry_dynamic.data import record_files, servo_sigma

# one configuration of the grid; the defaults are the selected configuration (grid index 9 of run_black_box.sh)
NX = 16              # state dimension n_x
FH_WIDTH = 16        # hidden width of f_bb and h_bb
E_WIDTH = 32         # hidden width of the encoder psi
LR = 1e-3            # Adam learning rate
# run budget and output
N_ITS = 80000        # Adam updates
ITS_PER_VAL = 1000   # updates between validations; the best validated checkpoint is kept
TRUNCATE_S = None    # [s] use only the start of each record, for trial runs
OUT_DIR = None       # run folder; None: <OUT_ROOT>/blackbox/<SLURM job id or time>
# fixed (HEURISTIC: thesis values)
SEED = 1             # initial weights and batch order
BATCH_SIZE = 256     # windows per update
STRIDE = 25          # samples between window starts
N_THREADS = 4        # CPU threads, the server's cores per task


def per_record_rms(model, sys_data, ry):
    """Free-run RMS error per record, RMS_j = RMS(y_hat - y) from k0 = n + 1, with y_hat = r + e_hat.

    Args:
        model: the black box.
        sys_data: deepSI System_data per record, u = (r, u^ms) and e, physical.
        ry: (r, y) per record, each (N, 3), physical.
    Returns:
        (n_records, 3) RMS_j, physical.
    """
    k0 = GREY_BOX.na_nb
    # a diverging configuration gives inf or NaN; it is scored as such and the selection skips it
    with np.errstate(all='ignore'):
        e_hat = [np.asarray(model.apply_experiment(sd).y, float) for sd in sys_data]
        return np.array([np.sqrt(np.mean((r[k0:] + eh[k0:] - y[k0:]) ** 2, 0)) for eh, (r, y) in zip(e_hat, ry)])


class ServoWeightedEncoder(SS_encoder_general_hf):
    """SUBNET with the loss weighted per record and axis and validation by NRMS_e."""

    def set_objective(self, train_sigma_e, val_sigma_e, val_ry, val_hist, failure, hist_path):
        """Stores the objective's data on the model.

        Args:
            train_sigma_e: (n_records, 3) sigma_e of the training records.
            val_sigma_e: (n_records, 3) sigma_e of the validation records.
            val_ry: (r, y) per validation record, physical.
            val_hist: list the validation history is appended to.
            failure: list the first non-finite loss is appended to.
            hist_path: val_hist.json, written at every validation and on a failure.
        """
        self._train_sigma_e, self._val_sigma_e, self._val_ry, self._hist_path = (train_sigma_e, val_sigma_e, val_ry,
                                                                                 hist_path)
        # the lists are owned by main: deepSI's reload of the best checkpoint replaces this model's attributes, and
        # the run's status needs a failure that happened after that checkpoint
        self._hist, self._failure = val_hist, failure

    def make_training_data(self, sys_data, **kw):
        """deepSI's training windows plus the loss weight w = sigma_y / sigma_e of each window's record, per axis, so
        the normalised loss is an error relative to each record's servo error, as NRMS_e.

        Args:
            sys_data: training records (deepSI System_data_list).
            **kw: deepSI's window settings, nf (horizon) and stride.
        Returns:
            deepSI's [uhist, yhist, ufuture, yfuture] plus w, (n_windows, 3).
        """
        data = super().make_training_data(sys_data, **kw)
        # 1. Count the windows per record as deepSI cuts them: window ends k from na + nf to N in steps of stride.
        counts = [len(range(self.na + kw['nf'], len(sd.u) + 1, kw['stride'])) for sd in sys_data.sdl]
        # 2. deepSI cuts the windows itself, so a change of its windowing would silently pair the weights with the
        #    wrong windows; the count must match exactly.
        if sum(counts) != len(data[0]):
            raise RuntimeError('weight rows %d != training windows %d' % (sum(counts), len(data[0])))
        # 3. Weight per record and axis, w = sigma_y / sigma_e, so w (e_hat^n - e^n) = (e_hat - e) / sigma_e.
        w = np.asarray(self.norm.ystd, float).ravel() / self._train_sigma_e
        return [*data, np.concatenate([np.tile(wr, (c, 1)) for wr, c in zip(w, counts)])]

    def loss(self, uhist, yhist, ufuture, yfuture, w, **kw):
        """Weighted simulation loss L = mean over windows, steps k and axes of (w (e_hat_k^n - e_k^n))^2.

        Args:
            uhist: (B, n_b, n_u) past inputs u^n, normalised.
            yhist: (B, n_a, n_y) past servo errors e^n, normalised.
            ufuture: (B, nf, n_u) inputs u^n over the horizon.
            yfuture: (B, nf, n_y) servo errors e^n over the horizon.
            w: (B, n_y) window weights from `make_training_data`.
            **kw: deepSI's loss settings, already used by `make_training_data`.
        Returns:
            Scalar loss L.
        """
        # 1. Encode the start state from the past samples, x^bb_0 = psi(u_past, e_past).
        x = self.encoder(uhist, yhist)
        # 2. Simulate the horizon and weight each step's error.
        errors = []
        for y, u in zip(torch.transpose(yfuture, 0, 1), torch.transpose(ufuture, 0, 1)):
            # e_hat_k = h_bb(x^bb_k), x^bb_k+1 = f_bb(x^bb_k, u_k)
            e_hat, x = self.hfn(x, u)
            errors.append(((e_hat - y) * w) ** 2)
        L = torch.mean(torch.stack(errors))
        # 3. Record the first non-finite loss: deepSI then stops and reloads the best checkpoint without a record.
        if not self._failure and not torch.isfinite(L):
            self._failure.append({'update': int(self.batch_counter), 'loss': float(L)})
            self._write_val_hist()
        return L

    def cal_validation_error(self, val_sys_data, validation_measure):
        """Validation score NRMS_e = mean over records and axes of RMS_j / sigma_e,j; deepSI keeps the checkpoint
        with the lowest score.

        Args:
            val_sys_data: validation records (deepSI System_data_list).
            validation_measure: deepSI's display name; the score is always NRMS_e.
        Returns:
            NRMS_e as a float.
        """
        rms = per_record_rms(self, val_sys_data.sdl, self._val_ry)
        score = float(np.mean(rms / self._val_sigma_e))
        self._hist.append({'update': int(self.batch_counter), 'nrms_e': score})
        self._write_val_hist()
        return score

    def _write_val_hist(self):
        """Writes the validation history and the failure at every event, so a killed job keeps them."""
        with open(self._hist_path, 'w') as fh:
            json.dump({'history': self._hist, 'failure': self._failure[0] if self._failure else None}, fh)


def main(args):
    """Trains one configuration and writes result.json and val_hist.json in the run folder."""
    t0 = time.time()
    out = os.path.abspath(args.out)
    # deepSI writes its checkpoints below LOCALAPPDATA on Windows (~/.deepSI elsewhere); pointing it at the run folder
    # keeps them there; deepSI creates only the last folder levels itself
    os.environ['LOCALAPPDATA'] = os.path.join(out, 'localappdata')
    os.makedirs(os.path.join(out, 'localappdata', 'deepSI', 'checkpoints'), exist_ok=True)
    torch.set_num_threads(N_THREADS)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. Data: input u = (r, u^ms) and target e = y - r per record; sigma_e from k0 = n + 1.
    k0 = GREY_BOX.na_nb
    cfg = data_cfg(args.truncate_s)
    train_files, val_files, _ = record_files(cfg)
    train, val = load_records(train_files, cfg), load_records(val_files, cfg)
    sys_data = [[deepSI.System_data(u=np.ascontiguousarray(u), y=np.ascontiguousarray(y - r), dt=cfg.ts_new)
                 for u, r, y in recs] for recs in (train, val)]
    val_ry = [(r, y) for _, r, y in val]
    val_sigma_e = servo_sigma([y - r for r, y in val_ry], k0)
    print('[BB] %d train, %d val records' % (len(train), len(val)), flush=True)

    # 2. Model and objective: the weights from the training records, the score from the validation records.
    model = ServoWeightedEncoder(**model_kwargs(args.nx, args.fh_width, args.e_width))
    val_hist, failure = [], []
    model.set_objective(servo_sigma([y - r for _, r, y in train], k0), val_sigma_e, val_ry, val_hist, failure,
                        os.path.join(out, 'val_hist.json'))
    # 3. Fit: Adam on L over windows of nf steps, NRMS_e on the validation records every its_per_val updates.
    model.fit(deepSI.System_data_list(sys_data[0]), val_sys_data=deepSI.System_data_list(sys_data[1]),
              n_its=args.n_its, batch_size=BATCH_SIZE, loss_kwargs=dict(nf=cfg.nf, stride=STRIDE),
              validation_measure='sim-NRMSe', optimizer_kwargs=dict(lr=args.lr), its_per_val=args.its_per_val)

    # 4. Result of the best checkpoint, which deepSI has reloaded; the loss curves stay in its _last.pth.
    val_rms = per_record_rms(model, sys_data[1], val_ry)
    res = dict(args=vars(args), k0=k0, checkpoint=model.name + '_best.pth', best_update=int(model.batch_counter),
               status='failed' if failure else 'completed', nrms_e_per_record=(val_rms / val_sigma_e).tolist(),
               val_sigma_e=val_sigma_e.tolist(), wall_s=time.time() - t0)
    with open(os.path.join(out, 'result.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('[BB] best update %d, validation NRMS_e %.4f (%.0f s)'
          % (res['best_update'], np.mean(res['nrms_e_per_record']), res['wall_s']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='no arguments = the settings at the top of this file')
    p.add_argument('--nx', type=int, default=NX)
    p.add_argument('--fh-width', type=int, default=FH_WIDTH)
    p.add_argument('--e-width', type=int, default=E_WIDTH)
    p.add_argument('--lr', type=float, default=LR)
    p.add_argument('--n-its', type=int, default=N_ITS)
    p.add_argument('--its-per-val', type=int, default=ITS_PER_VAL)
    p.add_argument('--truncate-s', type=float, default=TRUNCATE_S)
    p.add_argument('--out', default=OUT_DIR or os.path.join(
        OUT_ROOT, 'blackbox', os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')))
    main(p.parse_args())
