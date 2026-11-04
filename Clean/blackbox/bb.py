"""Shared parts of the black box M_BB, a SUBNET model (Beintema et al., deepSI 0.3.29) of the closed loop:

    x^bb_{k+1} = f_bb(x^bb_k, r_k, u^ms_k),   y_hat_k = r_k + e_hat_k,   e_hat_k = h_bb(x^bb_k),

with the state x^bb (n_x numbers), two networks f_bb (state update) and h_bb (output), the reference r, the injected
multisine force u^ms and the servo error e = y - r; the encoder psi gives the start state from past samples.
This file holds the net constructor, the grey box's data settings, the record loader and the free-run prediction.
Run the scripts from Clean/ as python -m blackbox.<script>."""
__project_origin__ = "added"

import os
import warnings
from dataclasses import replace

import numpy as np
import deepSI
from scipy.io import loadmat
from deepSI.fit_systems.encoders import default_encoder_net, default_output_net, default_state_net

from gantry_interconnect_dynamic import SETTINGS
from gantry_dynamic.config import RunConfig
from gantry_dynamic.data import dataset_dir, record_at_fs_new

N_HIDDEN_LAYERS = 2  # hidden tanh layers of psi, f_bb and h_bb (HEURISTIC: deepSI's default, as in the grey box; not
                     # searched)
N_THREADS = 4        # CPU threads, the server's cores per task
# the grey box's settings on the noisy data set: both models see the same samples (fs_new, anti-aliasing, trim),
# the same encoder window (na_nb) and the same training window length (nf)
GREY_BOX = RunConfig(**dict(SETTINGS, noise='noisy'))
# Save the settings that determine the black box's samples and windows; paths belong to the current machine.
DATA_FIELDS = ('noise', 'fs_new', 'aa_order', 'aa_fc_hz', 'aa_trim_s', 'truncate_s', 'na_nb', 'nf_seconds')


def data_config(result):
    """Use a run's saved data settings, with the current machine's data paths."""
    if 'data_settings' not in result:
        warnings.warn('Legacy run has no saved data settings; using the current grey-box settings.', stacklevel=2)
    return replace(GREY_BOX, **result.get('data_settings', {}))


def model_kwargs(nx, fh_width, e_width, cfg=GREY_BOX):
    """The settings that build one black box (encoder psi, f_bb, h_bb); shared by training and selection, so the
    search table counts the parameters of the trained networks.

    Args:
        nx: state dimension n_x.
        fh_width: hidden width of f_bb and h_bb.
        e_width: hidden width of the encoder psi.
    Returns:
        Keyword arguments of `SS_encoder_general_hf` and its subclasses.
    """
    fh_kwargs = dict(n_hidden_layers=N_HIDDEN_LAYERS, n_nodes_per_layer=fh_width)
    # encoder window: the grey box's n + 1 past samples of u and e
    return dict(nx=nx, na=cfg.na_nb, nb=cfg.na_nb, e_net=default_encoder_net,
                e_net_kwargs=dict(n_hidden_layers=N_HIDDEN_LAYERS, n_nodes_per_layer=e_width),
                hf_net_kwargs=dict(f_net=default_state_net, f_net_kwargs=dict(fh_kwargs), h_net=default_output_net,
                                   h_net_kwargs=dict(fh_kwargs)))


def load_records(files, folder=None, cfg=GREY_BOX):
    """Records at the training rate, through the grey box's loader, so both models see the same samples. The input is
    (r, u^ms), not the actuator force: the black box models the whole closed loop from its external inputs, so it
    needs no model of the controller.

    Args:
        files: record file names.
        folder: record folder; None for the noisy data set.
    Returns:
        List of (u, r, y) per record: u (N, 6) = (r, u^ms), r (N, 3) and y (N, 3), physical units.
    """
    out = []
    for fn in files:
        mat = loadmat(os.path.join(folder or dataset_dir(cfg), fn), squeeze_me=True)
        # the grey-box loader reads the plant input from u_total; here it gets the injected force u^ms, which it
        # block-averages over each training sample, while y and r are decimated as for the grey box
        u_ms, y, r = record_at_fs_new({**mat, 'u_total': mat['f_sim']}, cfg, reference=True)
        out.append((np.hstack([r, u_ms]), r, y))
    return out


def system_data(recs, cfg=GREY_BOX):
    """deepSI records with input (r, u^ms) and target e = y - r, the servo error; each array stored as one block of
    memory, which deepSI's reshaping (.view) needs."""
    return [deepSI.System_data(u=np.ascontiguousarray(u), y=np.ascontiguousarray(y - r), dt=cfg.ts_new)
            for u, r, y in recs]


def predict(model, recs, cfg=GREY_BOX):
    """Simulate each whole record from one start state: y_hat = r + e_hat [m], one (N, 3) array per record.

    At k0 = max(model.na, model.nb), the encoder estimates the state from preceding
    input and servo-error histories. The model then runs without state resets.
    deepSI copies measured outputs before k0; predictions start at k0.

    recs contains (u, r, y) from load_records; cfg supplies their sample time.
    """
    # A diverging configuration overflows; its non-finite prediction is returned without warnings
    with np.errstate(all='ignore'):
        return [r + np.asarray(model.apply_experiment(sd).y, float)
                for (_, r, _), sd in zip(recs, system_data(recs, cfg))]
