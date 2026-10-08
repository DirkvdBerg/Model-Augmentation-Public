from deepSI.fit_systems.encoders import (
    SS_encoder_general,
    default_encoder_net,
    hf_net_default,
    default_state_net,
    default_output_net,
)
from deepSI.system_data import System_data
from torch import nn
import torch
import numpy as np

from model_augmentation.utils.deepSI_corrections import fixed_System_data_norm


## deepSI corrections
# from model_augmentation.utils.deepSI_corrections import fixed_System_data

# deepSI.System_data = fixed_System_data


def convert_to_same_type(base, target):
    # converts target to the same type as base
    import torch

    if isinstance(base, (float, int)):
        return target
    elif isinstance(base, np.ndarray):
        return np.array(target, dtype=base.dtype)
    elif isinstance(base, torch.Tensor):
        return torch.as_tensor(target, dtype=base.dtype)
    else:
        raise NotImplementedError(f"cannot convert to type {base}")


class System_data_with_x(System_data):
    def __init__(self, u=None, y=None, x=None, cheat_n=0, normed=False, dt=None):
        super(System_data_with_x, self).__init__(
            u=u, y=y, x=x, cheat_n=cheat_n, normed=normed, dt=dt
        )

    def to_hist_future_data(
        self,
        na=10,
        nb=10,
        nf=5,
        na_right=0,
        nb_right=0,
        stride=1,
        force_multi_u=False,
        force_multi_y=False,
        online_construct=False,
    ):
        # assert stride==1, "stride > 1 not implemented yet"
        # u, y, x = self.u.astype(np.float32, copy=False), self.y.astype(np.float32, copy=False), self.x.astype(np.float32, copy=False) # type: ignore
        # def window(x,window_shape=nf):
        #     x = np.lib.stride_tricks.sliding_window_view(x, window_shape=window_shape,axis=0, writeable=True)
        #     s = (0,len(x.shape)-1) + tuple(range(1,len(x.shape)-1))
        #     return x.transpose(s)
        # npast = max(na, nb)
        # ufuture = window(u[npast:len(u)], window_shape=nf)
        # yfuture = window(y[npast:len(y)], window_shape=nf)
        # xfuture = window(x[npast:len(y)], window_shape=nf)
        # uhist = window(u[npast-nb:len(u)-nf], window_shape=nb)
        # yhist = window(y[npast-na:len(y)-nf], window_shape=na)

        u, y, x = np.copy(self.u), np.copy(self.y), np.copy(self.x)  # type: ignore
        yhist = []
        uhist = []
        ufuture = []
        yfuture = []
        xfuture = []
        k0 = max(nb, na)
        k0_right = max(nf, na_right, nb_right)
        for k in range(k0 + k0_right, len(u) + 1, stride):
            kmid = k - k0_right
            yhist.append(y[kmid - na : kmid + na_right])
            uhist.append(u[kmid - nb : kmid + nb_right])
            yfuture.append(y[kmid : kmid + nf])
            ufuture.append(u[kmid : kmid + nf])
            xfuture.append(x[kmid : kmid + nf])
        uhist, yhist, ufuture, yfuture, xfuture = (
            np.array(uhist),
            np.array(yhist),
            np.array(ufuture),
            np.array(yfuture),
            np.array(xfuture),
        )

        if force_multi_u and uhist.ndim == 2:  # (N, time_seq, nu)
            uhist = uhist[:, :, None]
            ufuture = ufuture[:, :, None]
        if force_multi_y and yhist.ndim == 2:  # (N, time_seq, ny)
            yhist = yhist[:, :, None]
            yfuture = yfuture[:, :, None]

        # print(uhist.shape)

        return uhist, yhist, ufuture, yfuture, xfuture


class System_data_norm_with_x(fixed_System_data_norm):
    """A utility to normalize system_data before fitting or usage

    Attributes
    ----------
    u0 : float or array
        average u to be subtracted
    ustd : float or array
        standard divination of u to be divided by
    y0 : float or array
        average y to be subtracted
    ystd : float or array
        standard divination of y to be divided by
    """

    def __init__(self, u0=0, ustd=1, y0=0, ystd=1):
        super(System_data_norm_with_x, self).__init__(
            u0=u0, ustd=ustd, y0=y0, ystd=ystd
        )

    def transform(self, sys_data):
        if self.is_id:
            return System_data_with_x(
                u=sys_data.u,
                x=sys_data.x,
                y=sys_data.y,
                cheat_n=sys_data.cheat_n,
                normed=True,
                dt=sys_data.dt,
            )

        if isinstance(sys_data, System_data) or isinstance(
            sys_data, System_data_with_x
        ):
            assert sys_data.normed is False, "System_data is already normalized"
            u_transformed = (
                (sys_data.u - self.u0) / self.ustd if sys_data.u is not None else None
            )
            y_transformed = (
                (sys_data.y - self.y0) / self.ystd if sys_data.y is not None else None
            )
            return System_data_with_x(
                u=u_transformed,
                x=sys_data.x,
                y=y_transformed,
                cheat_n=sys_data.cheat_n,
                normed=True,
                dt=sys_data.dt,
            )
        else:
            raise NotImplementedError(
                f"type={type(sys_data)} cannot yet be transformed by norm"
            )

    def inverse_transform(self, sys_data):
        if self.is_id:
            return System_data_with_x(
                u=sys_data.u,
                x=sys_data.x,
                y=sys_data.y,
                cheat_n=sys_data.cheat_n,
                normed=False,
                dt=sys_data.dt,
            )

        if isinstance(sys_data, System_data) or isinstance(
            sys_data, System_data_with_x
        ):
            assert sys_data.normed is True, "System_data is already un-normalized"
            u_inv_transformed = (
                sys_data.u * self.ustd + self.u0 if sys_data.u is not None else None
            )
            y_inv_transformed = (
                sys_data.y * self.ystd + self.y0 if sys_data.y is not None else None
            )
            return System_data_with_x(
                u=u_inv_transformed,
                x=sys_data.x,
                y=y_inv_transformed,
                cheat_n=sys_data.cheat_n,
                normed=False,
                dt=sys_data.dt,
            )
        else:
            raise NotImplementedError(
                f"type={type(sys_data)} cannot yet be inverse_transform by norm"
            )


class linear_encoder_init(nn.Module):
    def __init__(
        self,
        A,
        B,
        C,
        D,
        nx,
        nu,
        ny,
        na,
        nb,
        n_nodes_per_layer=64,
        n_hidden_layers=2,
        activation=nn.Tanh,
        flag_linear_only = False
    ):
        super(linear_encoder_init, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.nu = nu
        self.ny = ny
        self.nx = nx

        self.na = na
        self.nb = nb

        assert na == nb, "only na=nb is implemented for linear encoder init"
        n = na
        
        self.flag_linear_only = flag_linear_only

        Gamma_n = np.zeros(((n + 1) * ny, (n + 1) * nu))

        for i in range(n + 1):
            for j in range(i, n + 1):
                # flip row and column indices
                flipped_i = n - i
                flipped_j = n - j

                if i != j:
                    Gamma_n[
                        flipped_i * ny : (flipped_i + 1) * ny,
                        flipped_j * nu : (flipped_j + 1) * nu,
                    ] = C @ np.linalg.matrix_power(A, j - i - 1) @ B
                else:
                    Gamma_n[
                        flipped_i * ny : (flipped_i + 1) * ny,
                        flipped_j * nu : (flipped_j + 1) * nu,
                    ] = D

        O_n = np.zeros(((n + 1) * ny, nx))  # preallocate
        for i in range(n + 1):
            O_n[i * ny : (i + 1) * ny, :] = C @ np.linalg.matrix_power(A, i)
        O_inv = np.linalg.pinv(O_n)

        A_n = np.linalg.matrix_power(A, n)

        gamma_n = np.zeros((nx, (n + 1) * nu))
        for i in range(0, n):
            # Place A^{i-1} B into the corresponding block
            gamma_n[:, i * nu : (i + 1) * nu] = np.linalg.matrix_power(A, n - i - 1) @ B

        self.Wb_psi_y = nn.Parameter(torch.tensor(A_n @ O_inv, dtype=torch.float32))
        self.Wb_psi_u = nn.Parameter(
            torch.tensor(-A_n @ O_inv @ Gamma_n + gamma_n, dtype=torch.float32)
        )

        if not self.flag_linear_only:
            self.n_in = nu * (nb + 1) + ny * (na + 1)
            self.n_out = nx
            seq = [nn.Linear(self.n_in, n_nodes_per_layer), activation()]
            assert n_hidden_layers > 0
            for i in range(n_hidden_layers - 1):
                seq.append(nn.Linear(n_nodes_per_layer, n_nodes_per_layer))
                seq.append(activation())

            final_layer = nn.Linear(n_nodes_per_layer, self.n_out)
            seq.append(final_layer)

            self.net = nn.Sequential(*seq)

            nn.init.constant_(final_layer.bias, val=0.0)
            nn.init.constant_(final_layer.weight, val=0.0)

    def forward(self, uhist, yhist):
        # print(uhist.shape, yhist.shape)

        if len(uhist.size()) <= 2:
            uhist_mod = uhist.view(uhist.size(0), self.nu * (self.nb + 1), 1)
            state_has_correct_dimension = False
        else:
            state_has_correct_dimension = True

        if len(yhist.size()) <= 2:
            yhist_mod = yhist.view(yhist.size(0), self.ny * (self.na + 1), 1)

        x = self.Wb_psi_u @ uhist_mod + self.Wb_psi_y @ yhist_mod

        if not state_has_correct_dimension:
            x = x.view(-1, self.nx)
        
        if self.flag_linear_only:
            return x
        else:
            return x + self.net(torch.cat((uhist.view(uhist.size(0), -1), yhist.view(yhist.size(0), -1)), dim=1))
        


class SS_pre_encoder(SS_encoder_general):
    """The encoder function with combined h and f functions

    the hf_net_default has the arguments
       hf_net_default(nx, nu, ny, feedthrough=False, **hf_net_kwargs)
    and is used as
       ynow, xnext = hfn(x,u)
    """

    def __init__(
        self,
        nx=10,
        na=20,
        nb=20,
        feedthrough=False,
        hf_net=hf_net_default,
        hf_net_kwargs=dict(
            f_net=default_state_net,
            f_net_kwargs={},
            h_net_kwargs={},
            h_net=default_output_net,
        ),
        e_net=default_encoder_net,
        e_net_kwargs={},
        na_right=0,
        nb_right=0,
    ):
        super(SS_pre_encoder, self).__init__(
            nx=nx, nb=nb, na=na, na_right=na_right, nb_right=nb_right
        )

        self.e_net = e_net
        self.e_net_kwargs = e_net_kwargs

        self.hf_net = hf_net
        hf_net_kwargs["feedthrough"] = feedthrough
        self.hf_net_kwargs = hf_net_kwargs

        self.feedthrough = feedthrough

        self.norm = System_data_norm_with_x()
        
        self.encoder = None

    def init_nets(self, nu, ny):  # a bit weird
        na_right = self.na_right if hasattr(self, "na_right") else 0
        nb_right = self.nb_right if hasattr(self, "nb_right") else 0

        if self.encoder is None:
            print("Initializing encoder network...")
            self.encoder = self.e_net(
                nb=self.nb + nb_right,
                nu=nu,
                na=self.na + na_right,
                ny=ny,
                nx=self.nx,
                **self.e_net_kwargs,
            )
        self.hfn = self.hf_net(nx=self.nx, nu=self.nu, ny=self.ny, **self.hf_net_kwargs)

    def loss(self, uhist, yhist, ufuture, yfuture, xfuture, **Loss_kwargs):
        xhat = self.encoder(uhist, yhist)  # initialize Nbatch number of states # type: ignore
        errors = []
        i = 0
        for x in torch.transpose(xfuture, 0, 1):  # iterate over time
            # yhat, x = self.hfn(x, u)
            errors.append(
                nn.functional.mse_loss(x, xhat)
            )  # calculate error after taking n-steps
            i = i + 1
            assert i == 1, "This should not be used for nf>1"
        return torch.mean(torch.stack(errors))

    def n_step_error(self, sys_data, nf=100, stride=1, mode="NRMS", mean_channels=True):
        norm = System_data_norm_with_x()
        if isinstance(mode, tuple):
            norm, error_mode = mode
        elif isinstance(mode, System_data_norm_with_x):
            norm, error_mode = mode, "RMS"
        else:
            # figure out the error mode
            if "RMS" in mode:
                error_mode = "RMS"
            elif "MSE" in mode:
                error_mode = "MSE"
            elif "MAE" in mode:
                error_mode = "MAE"
            else:
                raise NotImplementedError(f"mode {mode} should has one of RMS MSE MAE")

            if "_sys_norm" in mode:
                norm = self.norm
            elif mode[0] == "N":
                norm.fit(sys_data)

        sys_data = self.norm.transform(sys_data)
        k0 = self.init_state_multi(sys_data, nf=nf, stride=stride)

        _, _, ufuture, yfuture, xfuture = sys_data.to_hist_future_data(
            na=k0, nb=k0, nf=nf, stride=stride
        )

        Losses = []
        for xnow in np.swapaxes(xfuture, 0, 1):
            res = convert_to_same_type(xnow, self.state) - xnow

            Losses = np.mean(res**2, axis=0) ** 0.5

            # print(Losses.shape)

            # if callable(error_mode):
            #     Losses.append(error_mode(res))
            # elif error_mode=='RMS':
            #     Losses.append(np.mean(res**2,axis=0)**0.5)
            # elif error_mode=='MSE':
            #     Losses.append(np.mean(res**2,axis=0))
            # elif error_mode=='MAE':
            #     Losses.append(np.mean(np.abs(res),axis=0))
            # else:
            #     raise NotImplementedError('error_mode should be one of ["RMS","MSE","MAE"]')

        return np.array(Losses)


if __name__ == "__main__":
    # Example usage
    A = np.array([[2, 0], [0, 2]])
    B = np.array([[1], [1]])
    C = np.array([[2.0, 0.0]])
    D = np.array([[1.0]])

    nx = A.shape[0]
    nu = B.shape[1]
    ny = C.shape[0]
    na = 4
    nb = 4

    encoder_init = linear_encoder_init(A, B, C, D, nx, nu, ny, na, nb)

    u = np.arange(0, 10).reshape(-1, 1)[:, 0]
    y = np.arange(0, 10).reshape(-1, 1)[:, 0]
    x = np.arange(0, 10).reshape(-1, 1)[:, 0]

    # data = System_data_with_x(u=u, y=y, x=x)
    # uhist, yhist, _, _, _ = data.to_hist_future_data(na=na, nb=nb, nf=3, na_right=1, nb_right=1)

    data = System_data(u=u, y=y)
    uhist, yhist, _, _ = data.to_hist_future_data(
        na=na, nb=nb, nf=3, na_right=1, nb_right=1
    )

    # Dummy input data
    # uhist = torch.randn(10, nb * nu)
    # yhist = torch.randn(10, na * ny)

    x0 = encoder_init(
        torch.tensor(uhist, dtype=torch.float32),
        torch.tensor(yhist, dtype=torch.float32),
    )
    print(x0)
