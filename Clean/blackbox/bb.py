"""Shared parts of the black box M_BB, Jan Hoekstra's SUBNET on the servo error e = y - r (deepSI 0.3.29): the net
constructor, the loader settings and the record loader of the grey box."""
__project_origin__ = "added"

import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat
from deepSI.fit_systems.encoders import default_output_net

from jan_model import JanEncoder, JanStateNet

# the scripts run as python Clean/blackbox/<script>.py, so Clean/ is put on the path for gantry_dynamic
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gantry_dynamic.config import RunConfig, DATA_ROOT                             # noqa: E402
from gantry_dynamic.data import traj_dir, _record_at_fs_new                        # noqa: E402

N_HIDDEN_LAYERS = 2  # hidden tanh layers of psi, f_bb and h_bb (HEURISTIC: thesis value, Jan's default)


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
    return dict(nx=nx, na=RunConfig.na_nb, nb=RunConfig.na_nb, e_net=JanEncoder,
                e_net_kwargs=dict(n_hidden_layers=N_HIDDEN_LAYERS, n_nodes_per_layer=e_width),
                hf_net_kwargs=dict(f_net=JanStateNet, f_net_kwargs=dict(fh_kwargs), h_net=default_output_net,
                                   h_net_kwargs=dict(fh_kwargs)))


def data_cfg(truncate_s=None):
    """Loader settings: the lowpass data set at the training rate, the RunConfig fields the grey-box loader reads.

    Args:
        truncate_s: [s] use only the start of each record; None for whole records.
    """
    return SimpleNamespace(noise='lowpass', data_root=DATA_ROOT, d=RunConfig.d, fs_orig=RunConfig.fs_orig,
                           fs_new=RunConfig.fs_new, truncate_s=truncate_s)


def load_records(files, cfg, folder=None):
    """Records at the training rate, through the grey box's loader, so both models see the same samples.

    Args:
        files: record file names.
        cfg: loader settings from `data_cfg`.
        folder: record folder; None for the lowpass data set.
    Returns:
        List of (u, r, y) per record: u (N, 6) = (r, u^ms), r (N, 3) and y (N, 3), physical units.
    """
    out = []
    for fn in files:
        mat = loadmat(os.path.join(folder or traj_dir(cfg), fn), squeeze_me=True)
        # the grey-box loader reads the plant input from u_total; here it gets the injected force u^ms, which it
        # block-averages over each training sample, while y and r are decimated as for the grey box
        u_ms, y, r = _record_at_fs_new({**mat, 'u_total': mat['f_sim']}, cfg, ('r_sim',))
        out.append((np.hstack([r, u_ms]), r, y))
    return out
