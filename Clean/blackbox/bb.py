"""Shared parts of the black box M_BB, Jan Hoekstra's SUBNET on the servo error e = y - r (deepSI 0.3.29): the net
constructor, the grey box's data settings, its record loader and the free-run prediction. Run the scripts from
Clean/ as python -m blackbox.<script>."""
__project_origin__ = "added"

import os

import numpy as np
import deepSI
from scipy.io import loadmat
from deepSI.fit_systems.encoders import default_encoder_net, default_output_net, default_state_net

from gantry_interconnect_dynamic import SETTINGS
from gantry_dynamic.config import RunConfig
from gantry_dynamic.data import dataset_dir, record_at_fs_new

N_HIDDEN_LAYERS = 2  # hidden tanh layers of psi, f_bb and h_bb (HEURISTIC: thesis value, Jan's default)
N_THREADS = 4        # CPU threads, the server's cores per task
# the grey box's settings on the noisy data set: both models see the same samples (fs_new, anti-aliasing, trim),
# the same encoder window (na_nb) and the same training window length (nf)
GREY_BOX = RunConfig(**dict(SETTINGS, noise='noisy'))


def model_kwargs(nx, fh_width, e_width):
    """Constructor arguments of the black box; shared by training and selection, so the search table counts the
    parameters of the trained nets.

    Args:
        nx: state dimension n_x.
        fh_width: hidden width of f_bb and h_bb.
        e_width: hidden width of the encoder psi.
    Returns:
        Keyword arguments of `SS_encoder_general_hf` and its subclasses.
    """
    fh_kwargs = dict(n_hidden_layers=N_HIDDEN_LAYERS, n_nodes_per_layer=fh_width)
    # encoder window: the grey box's n + 1 past samples of u and e
    return dict(nx=nx, na=GREY_BOX.na_nb, nb=GREY_BOX.na_nb, e_net=default_encoder_net,
                e_net_kwargs=dict(n_hidden_layers=N_HIDDEN_LAYERS, n_nodes_per_layer=e_width),
                hf_net_kwargs=dict(f_net=default_state_net, f_net_kwargs=dict(fh_kwargs), h_net=default_output_net,
                                   h_net_kwargs=dict(fh_kwargs)))


def load_records(files, folder=None):
    """Records at the training rate, through the grey box's loader, so both models see the same samples.

    Args:
        files: record file names.
        folder: record folder; None for the noisy data set.
    Returns:
        List of (u, r, y) per record: u (N, 6) = (r, u^ms), r (N, 3) and y (N, 3), physical units.
    """
    out = []
    for fn in files:
        mat = loadmat(os.path.join(folder or dataset_dir(GREY_BOX), fn), squeeze_me=True)
        # the grey-box loader reads the plant input from u_total; here it gets the injected force u^ms, which it
        # block-averages over each training sample, while y and r are decimated as for the grey box
        u_ms, y, r = record_at_fs_new({**mat, 'u_total': mat['f_sim']}, GREY_BOX, reference=True)
        out.append((np.hstack([r, u_ms]), r, y))
    return out


def predict(model, recs):
    """Free-run prediction y_hat = r + e_hat over the whole record, from the encoder state x^bb_0 = psi(u_past, e_past).

    Args:
        model: the black box.
        recs: (u, r, y) per record from `load_records`, physical.
    Returns:
        List of (N, 3) predictions y_hat, physical.
    """
    # deepSI's nets flatten their inputs with .view, which needs C-contiguous records; loadmat gives Fortran order
    with np.errstate(all='ignore'):
        return [r + np.asarray(model.apply_experiment(deepSI.System_data(
                    u=np.ascontiguousarray(u), y=np.ascontiguousarray(y - r), dt=GREY_BOX.ts_new)).y, float)
                for u, r, y in recs]
