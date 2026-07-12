import numpy as np
import torch
from torch import nn
import warnings


class state_measure_encoder:
    """
    for known y[k]=x[k] cases
    """
    def __init__(self, nb, nu, na, ny, nx, **kwargs):
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu, int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny, int) else ny)

    def __call__(self, upast, ypast):
        # in:                               | out:
        #  - u_past (Nd, nb+nb_right, Nu)   |  - x0 (Nd, Nx=Ny)
        #  - y_past (Nd, na+na_right, Ny)   |
        return ypast[:, -1, :]
