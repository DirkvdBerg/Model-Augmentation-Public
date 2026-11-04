"""Training of one black-box configuration: deepSI's fit() on the loss weighted per record and axis by
sigma_ebar / sigma_e,r (deepSI's pooled servo-error scale over the record's own), keeping the checkpoint with the
lowest validation NRMS_e."""
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

from blackbox.bb import DATA_FIELDS, GREY_BOX, N_HIDDEN_LAYERS, N_THREADS, load_records, model_kwargs, predict, system_data
from gantry_dynamic.config import OUT_ROOT
from gantry_dynamic.data import record_files, servo_sigma
from model_augmentation.fit_systems.closed_loop import window_counts

# one configuration of the grid; the defaults are the selected configuration (grid index 9 of run_black_box.sh)
NX = 16              # state dimension n_x
FH_WIDTH = 16        # hidden width of f_bb and h_bb
E_WIDTH = 32         # hidden width of the encoder psi
LR = 1e-3            # Adam learning rate
# run budget and output
N_ITS = 80000        # Adam updates
OUT_DIR = None       # run folder; None: <OUT_ROOT>/blackbox/<SLURM job id or time>
# fixed, not searched (HEURISTIC)
ITS_PER_VAL = 1000   # updates between validations; the best validated checkpoint is kept
SEED = 1             # initial weights and batch order
BATCH_SIZE = 256     # windows per update; the grey box uses 512, not compared
STRIDE = 25          # samples between window starts; the grey box uses 10, not compared


def per_record_rms(model, recs):
    """RMS error per record and axis of the whole-record simulation (bb.predict),
    RMS_j = ( 1/(N - k0) sum_{k=k0}^{N-1} (y_hat_{j,k} - y_{j,k})^2 )^(1/2), k0 = 30 the first sample after the
    encoder's input. Since y_hat = r + e_hat, y_hat - y = e_hat - e: r cancels.

    Args:
        model: the black box.
        recs: (u, r, y) per record from `load_records`.
    Returns:
        (n_records, 3) RMS_j [m].
    """
    k0 = max(model.na, model.nb)
    # a diverging configuration gives inf or NaN; it is scored as such and the selection skips it
    with np.errstate(all='ignore'):
        return np.array([np.sqrt(np.mean((y_hat[k0:] - y[k0:]) ** 2, 0))
                         for y_hat, (_, _, y) in zip(predict(model, recs), recs)])


class ServoWeightedEncoder(SS_encoder_general_hf):
    """deepSI's SUBNET model with the loss weighted per record and axis, validation by NRMS_e, and checkpoints kept
    in the run folder."""

    def checkpoint_save_system(self, name='_best', directory=None):
        """Keep deepSI's checkpoints in this run folder on every platform."""
        return super().checkpoint_save_system(name=name, directory=directory or self.checkpoint_dir)

    def checkpoint_load_system(self, name='_best', directory=None):
        """Reload from the same folder that fit saves to."""
        return super().checkpoint_load_system(name=name, directory=directory or self.checkpoint_dir)

    def set_objective(self, train_sigma_e, val_sigma_e, val_recs, failure):
        """Store what the weighted loss and NRMS_e validation need: sigma_e of both splits, the validation records.

        Args:
            train_sigma_e: (n_records, 3) sigma_e of the training records.
            val_sigma_e: (n_records, 3) sigma_e of the validation records.
            val_recs: (u, r, y) per validation record, physical.
            failure: list the first non-finite loss is appended to.
        """
        self._train_sigma_e, self._val_sigma_e, self._val_recs = train_sigma_e, val_sigma_e, val_recs
        # deepSI's reload of the best checkpoint replaces the model's attributes, so the failure record lives in main
        self._failure = failure

    def make_training_data(self, sys_data, **kw):
        """Append per-axis weights w_r to each record's deepSI training windows.

        sigma_ebar is deepSI's pooled training servo-error scale. sigma_e,r is each
        record's servo-error standard deviation after the encoder history, per axis.
        For normalised measured and predicted servo errors e^n and e_hat^n,
            w_r = sigma_ebar / sigma_e,r,
            w_r (e_hat^n - e^n) = (e_hat - e) / sigma_e,r.
        Training averages squared scaled errors over windows; validation averages
        whole-record RMS ratios.

        Return [uhist, yhist, ufuture, yfuture, w], with w shaped (windows, 3).
        """
        data = super().make_training_data(sys_data, **kw)
        counts = window_counts(self, sys_data.sdl, kw['nf'], kw['stride'])
        # deepSI cuts the windows itself, so a change of its windowing would silently pair the weights with the wrong
        # windows. Only the total count is checked, so this does not prove each weight sits on its own record's windows
        if sum(counts) != len(data[0]):
            raise RuntimeError('weight rows %d != training windows %d' % (sum(counts), len(data[0])))
        w = np.asarray(self.norm.ystd, float).ravel() / self._train_sigma_e
        return [*data, np.concatenate([np.tile(wr, (c, 1)) for wr, c in zip(w, counts)])]

    def loss(self, uhist, yhist, ufuture, yfuture, w, **kw):
        """Weighted simulation loss L = mean over windows, steps k and axes of (w_r (e_hat_k^n - e_k^n))^2: deepSI's
        SS_encoder_general_hf.loss with the weight added.

        Args:
            uhist: (B, n_b, 6) past inputs u^n.
            yhist: (B, n_a, 3) past servo errors e^n.
            ufuture: (B, nf, 6) inputs u^n over the horizon.
            yfuture: (B, nf, 3) servo errors e^n over the horizon.
            w: (B, 3) window weights from `make_training_data`.
            **kw: deepSI's loss settings, already used by `make_training_data`.
        Returns:
            Scalar loss L.
        """
        # start state from the past samples, x^bb_0 = psi(u_past, e_past)
        x = self.encoder(uhist, yhist)
        errors = []
        for y, u in zip(torch.transpose(yfuture, 0, 1), torch.transpose(ufuture, 0, 1)):
            # e_hat_k = h_bb(x^bb_k), x^bb_k+1 = f_bb(x^bb_k, u_k), with u_k = (r_k, u^ms_k) normalised
            e_hat, x = self.hfn(x, u)
            errors.append(((e_hat - y) * w) ** 2)
        L = torch.mean(torch.stack(errors))
        # record the first non-finite loss: deepSI then stops and reloads the best checkpoint without a record
        if not self._failure and not torch.isfinite(L):
            self._failure.append({'update': int(self.batch_counter), 'loss': float(L)})
        return L

    def cal_validation_error(self, val_sys_data, validation_measure):
        """Validation score NRMS_e = mean over records r and axes j of RMS_j / sigma_e,r; deepSI keeps the checkpoint
        with the lowest score. deepSI's own sim-NRMS divides by the standard deviation of its output instead.

        Args:
            val_sys_data: deepSI's validation records; the score uses the records given to `set_objective`.
            validation_measure: deepSI's display name; the score is always NRMS_e.
        Returns:
            NRMS_e as a float.
        """
        return float(np.mean(per_record_rms(self, self._val_recs) / self._val_sigma_e))


def main(args):
    """Trains one configuration and writes result.json in the run folder."""
    t0 = time.time()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(N_THREADS)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # data: input u = (r, u^ms) and target e = y - r per record; sigma_e from k0 = 30, after the encoder's input
    k0 = GREY_BOX.na_nb
    train_files, val_files, test_files = record_files(GREY_BOX)
    train, val = load_records(train_files), load_records(val_files)
    train_data, val_data = system_data(train), system_data(val)
    val_sigma_e = servo_sigma([y - r for _, r, y in val], k0)
    print('[BB] %d train, %d val records' % (len(train), len(val)), flush=True)

    # the loss weights come from the training records, the score from the validation records
    model = ServoWeightedEncoder(**model_kwargs(args.nx, args.fh_width, args.e_width))
    model.checkpoint_dir = out
    failure = []
    model.set_objective(servo_sigma([y - r for _, r, y in train], k0), val_sigma_e, val, failure)
    # Adam on L over windows of nf steps, NRMS_e on the validation records every its_per_val updates
    model.fit(deepSI.System_data_list(train_data), val_sys_data=deepSI.System_data_list(val_data),
              n_its=args.n_its, batch_size=BATCH_SIZE, loss_kwargs=dict(nf=GREY_BOX.nf, stride=STRIDE),
              validation_measure='sim-NRMSe', optimizer_kwargs=dict(lr=args.lr), its_per_val=ITS_PER_VAL)

    # result of the best checkpoint, which deepSI has reloaded, so batch_counter is the update it was saved at; the
    # loss curves stay in _last.pth. status 'failed' means a non-finite loss occurred at some update; the best
    # checkpoint before it can still have a finite score
    val_rms = per_record_rms(model, val)
    res = dict(args=vars(args), k0=k0, checkpoint=model.name + '_best.pth', best_update=int(model.batch_counter),
               data_settings={key: getattr(GREY_BOX, key) for key in DATA_FIELDS},
               records=dict(train=train_files, val=val_files, test=test_files),
               training_settings=dict(seed=SEED, batch_size=BATCH_SIZE, stride=STRIDE, its_per_val=ITS_PER_VAL,
                                      n_hidden_layers=N_HIDDEN_LAYERS, n_threads=N_THREADS),
               n_params=sum(p.numel() for net in (model.encoder, model.hfn) for p in net.parameters()),
               status='failed' if failure else 'completed', failure=failure[0] if failure else None,
               nrms_e_per_record=(val_rms / val_sigma_e).tolist(),
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
    p.add_argument('--out', default=OUT_DIR or os.path.join(
        OUT_ROOT, 'blackbox', os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')))
    main(p.parse_args())
