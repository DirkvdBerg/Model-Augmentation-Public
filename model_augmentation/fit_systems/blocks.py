from typing import Callable, Type, Union

import numpy as np
import torch
from deepSI.system_data import System_data
from torch import Tensor, nn

from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.utils.utils import to_tensor, added


class Block(nn.Module):
    """Basic block implementation with variables that need to be defined for interconnect."""

    def __init__(self, nz, nw, name=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.nw = nw
        self.nz = nz

        self.name = name
        self.block_ix = None

    def init_block(self, z: Tensor):
        return

    def forward(self, z: Tensor):
        raise NotImplementedError("This function should return w computed from z.")


class Static_ANN_Block(Block):
    def __init__(
        self,
        net: Union[
            Type[nn.Module], Callable[..., nn.Module]
        ] = zero_init_feed_forward_nn,
        n_nodes_per_layer=64,
        n_hidden_layers=2,
        activation=nn.Tanh,
        *args,
        **kwargs,
    ) -> None:
        super(Static_ANN_Block, self).__init__(*args, **kwargs)

        self.net = net(
            n_in=self.nz,
            n_out=self.nw,
            n_nodes_per_layer=n_nodes_per_layer,
            n_hidden_layers=n_hidden_layers,
            activation=activation,
        )

    def forward(self, z: Tensor):
        # print(self.name + " forward called for w" + str(self.block_ix))
        assert z.size(1) == self.nz

        # z_scaled = torch.matmul(self.D_z, z)

        w = self.net(z.view(-1, self.nz))
        return w.view(-1, self.nw, 1)
        # return torch.matmul(self.D_yw, w.view(-1,self.nw,1))


class Cubic_Block(Block):
    def __init__(self, aL, *args, **kwargs) -> None:
        super(Cubic_Block, self).__init__(*args, **kwargs)
        self.aL = aL

    def forward(self, z: Tensor):
        assert z.size(1) == self.nz

        return torch.mul(torch.pow(z, 3), self.aL)


class Linear_State_Block(Block):
    def __init__(
        self, A=torch.empty((0, 0)), B=torch.empty((0, 0)), *args, **kwargs
    ) -> None:
        # Matrices defining the known linear ss model
        self.A = to_tensor(A)
        self.B = to_tensor(B)

        self.nx = self.A.size(0) if self.A.numel() else 0  # type: ignore
        self.nu = self.B.size(1) if self.B.numel() else 0  # type: ignore
        if self.nx == 0:
            self.nx = self.B.size(0) if self.B.numel() else 0  # type: ignore

        super().__init__(nw=self.nx, nz=self.nx + self.nu, *args, **kwargs)

    def forward(self, z: Tensor):
        # print(self.name + " forward called for w" + str(self.block_ix))
        # print(z.shape)
        assert z.size(1) == self.nx + self.nu
        x = z[:, : self.nx, :]
        u = z[:, self.nx :, :]
        w = torch.matmul(self.A, x) + torch.matmul(self.B, u)  # type: ignore
        return w


@added
class Linear_Output_Block(Block):
    def __init__(
        self, C=torch.empty((0, 0)), D=torch.empty((0, 0)), *args, **kwargs
    ) -> None:
        _C = to_tensor(C)
        _D = to_tensor(D)

        self.ny = _C.size(0) if _C.numel() else 0
        self.nx = _C.size(1) if _C.numel() else 0
        self.nu = _D.size(1) if _D.numel() else 0
        if self.ny == 0:
            self.ny = _D.size(0) if _D.numel() else 0

        super().__init__(nw=self.ny, nz=self.nx + self.nu, *args, **kwargs)

        self.register_buffer("C", _C)
        self.register_buffer("D", _D)

    def forward(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu
        x = z[:, : self.nx, :]
        u = z[:, self.nx :, :]
        w = torch.matmul(self.C, x) + torch.matmul(self.D, u)  # type: ignore
        return w


class Parameterized_Linear_State_Block(Block):
    def __init__(
        self,
        A=torch.empty((0, 0)),
        B=torch.empty((0, 0)),
        RMSE_baseline=1.0,
        flag_loss_reg=True,
        *args,
        **kwargs,
    ) -> None:
        # Matrices defining the known linear ss model
        self.A_init = to_tensor(A)
        self.B_init = to_tensor(B)

        self.nx = self.A_init.size(0) if self.A_init.numel() else 0  # type: ignore
        self.nu = self.B_init.size(1) if self.B_init.numel() else 0  # type: ignore
        if self.nx == 0:
            self.nx = self.B_init.size(0) if self.B_init.numel() else 0  # type: ignore

        super().__init__(nw=self.nx, nz=self.nx + self.nu, *args, **kwargs)

        self.A = nn.Parameter(to_tensor(A).clone())  # type: ignore
        self.B = nn.Parameter(to_tensor(B).clone())  # type: ignore

        # These lambda matrices are used to scale the loss function in interconnect.py
        self.Lambda_A = (torch.ones(self.A.shape) / self.A_init) * RMSE_baseline
        self.Lambda_A[torch.isinf(self.Lambda_A)] = 0.0
        self.Lambda_B = (torch.ones(self.B.shape) / self.B_init) * RMSE_baseline
        self.Lambda_B[torch.isinf(self.Lambda_B)] = 0.0

        self.flag_loss_reg = flag_loss_reg

    def forward(self, z: Tensor):
        # print(self.name + " forward called for w" + str(self.block_ix))
        # print(z.shape)
        assert z.size(1) == self.nx + self.nu
        x = z[:, : self.nx, :]
        u = z[:, self.nx :, :]
        w = torch.matmul(self.A, x) + torch.matmul(self.B, u)
        return w

    def param_loss(self):
        if self.flag_loss_reg:
            loss_theta = nn.functional.mse_loss(
                self.Lambda_A * self.A, self.Lambda_A * self.A_init, reduction="sum"
            ) + nn.functional.mse_loss(
                self.Lambda_B * self.B, self.Lambda_B * self.B_init, reduction="sum"
            )
            return loss_theta
        else:
            return 0.0


class Parameterized_Linear_Output_Block(Block):
    def __init__(
        self,
        C=torch.empty((0, 0)),
        D=torch.empty((0, 0)),
        RMSE_baseline=1.0,
        flag_loss_reg=True,
        *args,
        **kwargs,
    ) -> None:
        # Matrices defining the known linear ss model
        self.C_init = to_tensor(C)
        self.D_init = to_tensor(D)

        self.ny = self.C_init.size(0) if self.C_init.numel() else 0  # type: ignore
        self.nx = self.C_init.size(1) if self.C_init.numel() else 0  # type: ignore
        self.nu = self.D_init.size(1) if self.D_init.numel() else 0  # type: ignore
        if self.ny == 0:
            self.ny = self.D_init.size(0) if self.D_init.numel() else 0  # type: ignore

        super().__init__(nw=self.ny, nz=self.nx + self.nu, *args, **kwargs)

        self.C = nn.Parameter(to_tensor(C).clone())  # type: ignore
        self.D = nn.Parameter(to_tensor(D).clone())  # type: ignore

        # These lambda matrices are used to scale the loss function in interconnect.py
        self.Lambda_C = (torch.ones(self.C.shape) / self.C_init) * RMSE_baseline
        self.Lambda_C[torch.isinf(self.Lambda_C)] = 0.0
        self.Lambda_D = (torch.ones(self.D.shape) / self.D_init) * RMSE_baseline
        self.Lambda_D[torch.isinf(self.Lambda_D)] = 0.0

        self.flag_loss_reg = flag_loss_reg

    def forward(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu
        x = z[:, : self.nx, :]
        u = z[:, self.nx :, :]
        w = torch.matmul(self.C, x) + torch.matmul(self.D, u)
        return w

    def param_loss(self):
        if self.flag_loss_reg:
            loss_theta = nn.functional.mse_loss(
                self.Lambda_C * self.C, self.Lambda_C * self.C_init, reduction="sum"
            ) + nn.functional.mse_loss(
                self.Lambda_D * self.D, self.Lambda_D * self.D_init, reduction="sum"
            )
            return loss_theta
        else:
            return 0.0


class Discrete_Nonlinear_Function_Block(Block):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def init_norm(self, sys_data: System_data):
        raise NotImplementedError(
            "This function should determine normalization matrices Tw and Tiz for the given input data"
        )

    def forward(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu
        w = self.nonlinear_function(z)
        return w

    def nonlinear_function(self, z: Tensor):
        raise NotImplementedError(
            "This function should describe the discrete nonlinear function of the block"
        )


class Cascaded_Tanks_State_Block(Discrete_Nonlinear_Function_Block):
    def __init__(self, params, sys_data: System_data, Ts=4.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.Ts = Ts

        self.nu = 1
        self.ny = 1
        self.nx = 2

        self.k1 = params[0]
        self.k2 = params[1]
        self.k3 = params[2]
        self.k4 = params[3]
        self.k5 = params[4]
        self.k6 = params[5]

        self.x1max = 10
        self.x2max = 10  # params[7]
        self.ymax = 10
        self.yoffset = params[8]

        self.ystd = 2.165135419802158
        self.y0 = 5.582729231040664
        self.ustd = 0.9995115994824683
        self.u0 = 2.8

    def nonlinear_function(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu
        x = z[:, : self.nx, :]
        x1 = x[:, 0, :]
        x2 = x[:, 1, :]
        u = z[:, self.nx :, :]
        u = u[:, 0, :]

        # denormalize
        x1 = (x1 * self.ystd) + self.y0
        x2 = (x2 * self.ystd) + self.y0
        u = (u * self.ustd) + self.u0

        x1 = torch.clamp(x1, min=0.00001)
        x1k = torch.clamp(
            x1 + self.Ts * (-self.k1 * torch.sqrt(x1) + self.k2 * x1 + self.k3 * u),
            max=self.x1max,
        )

        x2 = torch.clamp(x2, min=0.00001)
        mask = torch.le(torch.ones(x1.size()) * self.x1max, x1)
        x2k = torch.clamp(
            x2
            + self.Ts
            * (
                self.k1 * torch.sqrt(x1)
                - self.k2 * x1
                - self.k4 * torch.sqrt(x2)
                + self.k5 * x2
            ),
            max=self.x2max,
        )
        x2k_overflow = torch.clamp(
            x2
            + self.Ts
            * (
                self.k1 * torch.sqrt(x1)
                - self.k2 * x1
                - self.k4 * torch.sqrt(x2)
                + self.k5 * x2
                + self.k6 * u
            ),
            max=self.x2max,
        )

        x2k[mask] = x2k_overflow[mask]

        yk = torch.clamp(x2 + self.yoffset, max=self.ymax)

        # normalize
        x1k = (x1k - self.y0) / self.ystd
        x2k = (x2k - self.y0) / self.ystd
        # x1 = (x1 - self.y0)/self.ystd
        # x2 = (x2 - self.y0)/self.ystd
        yk = (yk - self.y0) / self.ystd

        w = torch.hstack((x1k, x2k, yk)).unsqueeze(-1)
        return w


class Parameterized_Cascaded_Tanks_State_Block(Discrete_Nonlinear_Function_Block):
    def __init__(self, params, sys_data: System_data, Ts=4.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.Ts = Ts

        self.nu = 1
        self.ny = 1
        self.nx = 2

        self.k1 = nn.Parameter(torch.tensor(params[0]))
        self.k2 = nn.Parameter(torch.tensor(params[1]))
        self.k3 = nn.Parameter(torch.tensor(params[2]))
        self.k4 = nn.Parameter(torch.tensor(params[3]))
        self.k5 = nn.Parameter(torch.tensor(params[4]))
        self.k6 = nn.Parameter(torch.tensor(params[5]))

        self.x1max = 10
        self.x2max = 10  # params[7]
        self.ymax = 10
        self.yoffset = nn.Parameter(torch.tensor(params[8]))  # params[8]

        self.ystd = 2.165135419802158
        self.y0 = 5.582729231040664
        self.ustd = 0.9995115994824683
        self.u0 = 2.8

    def nonlinear_function(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu
        x = z[:, : self.nx, :]
        x1 = x[:, 0, :]
        x2 = x[:, 1, :]
        u = z[:, self.nx :, :]
        u = u[:, 0, :]

        # denormalize
        x1 = (x1 * self.ystd) + self.y0
        x2 = (x2 * self.ystd) + self.y0
        u = (u * self.ustd) + self.u0

        x1 = torch.clamp(x1, min=0.00001)
        x1k = torch.clamp(
            x1 + self.Ts * (-self.k1 * torch.sqrt(x1) + self.k2 * x1 + self.k3 * u),
            max=self.x1max,
        )

        x2 = torch.clamp(x2, min=0.00001)
        mask = torch.le(torch.ones(x1.size()) * self.x1max, x1)
        x2k = torch.clamp(
            x2
            + self.Ts
            * (
                self.k1 * torch.sqrt(x1)
                - self.k2 * x1
                - self.k4 * torch.sqrt(x2)
                + self.k5 * x2
            ),
            max=self.x2max,
        )
        x2k_overflow = torch.clamp(
            x2
            + self.Ts
            * (
                self.k1 * torch.sqrt(x1)
                - self.k2 * x1
                - self.k4 * torch.sqrt(x2)
                + self.k5 * x2
                + self.k6 * u
            ),
            max=self.x2max,
        )

        x2k[mask] = x2k_overflow[mask]

        yk = torch.clamp(x2 + self.yoffset, max=self.ymax)

        # normalize
        x1k = (x1k - self.y0) / self.ystd
        x2k = (x2k - self.y0) / self.ystd
        # x1 = (x1 - self.y0)/self.ystd
        # x2 = (x2 - self.y0)/self.ystd
        yk = (yk - self.y0) / self.ystd

        w = torch.hstack((x1k, x2k, yk)).unsqueeze(-1)
        return w


class Parameterized_MSD_State_Block(Discrete_Nonlinear_Function_Block):
    def __init__(self, Ts=0.02, FP_type="ideal", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.Ts = Ts

        self.nu = 1
        self.ny = 1
        self.nx = 4

        if FP_type == "ideal":
            self.init_params = to_tensor(
                np.array([0.5, 0.4, 100, 100, 0.5, 0.5])
            )  # m1, m2, k1, k2, c1, c2
        elif FP_type == "approximate":
            self.init_params = to_tensor(
                np.array([0.5, 0.4, 95, 95, 0.45, 0.45])
            )  # non ideal: m1, m2, k1, k2, c1, c2
        else:
            raise ValueError("FP_type must be either 'ideal' or 'approximate'")

        self.params = nn.Parameter(self.init_params.clone())  # type: ignore

        self.Tx = to_tensor(
            np.array(
                [
                    [8.9022888, 0.0, 0.0, 0.0],
                    [0.0, 0.77655197, 0.0, 0.0],
                    [0.0, 0.0, 5.86104999, 0.0],
                    [0.0, 0.0, 0.0, 0.59214659],
                ]
            )
        )
        self.Tix = to_tensor(
            np.array(
                [
                    [0.11233066, 0.0, 0.0, 0.0],
                    [0.0, 1.28774381, 0.0, 0.0],
                    [0.0, 0.0, 0.17061789, 0.0],
                    [0.0, 0.0, 0.0, 1.68877101],
                ]
            )
        )
        self.Tiu = to_tensor(np.array([[10.0]]))
        self.Ty = to_tensor(np.array([[5.85660401]]))

        # RMSE_baseline = 0.2 # ideal
        RMSE_baseline = 0.2  # non ideal
        self.epsilon = 1
        self.Lambda = (
            np.sqrt(1 / self.epsilon)
            * RMSE_baseline
            * torch.linalg.inv(torch.diag(self.init_params))  # type: ignore
        )

    def nonlinear_function(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu

        x_n = z[:, : self.nx, :]
        u_n = z[:, self.nx :, :]

        A1 = torch.tensor([0, 1, 0, 0])
        A2 = torch.stack(
            (
                -(self.params[2] + self.params[3]) / self.params[0],
                -(self.params[4] + self.params[5]) / self.params[0],
                (self.params[3]) / self.params[0],
                (self.params[5]) / self.params[0],
            )
        )
        A3 = torch.tensor([0, 0, 0, 1])
        A4 = torch.stack(
            (
                (self.params[3]) / self.params[1],
                (self.params[5]) / self.params[1],
                -(self.params[3]) / self.params[1],
                -(self.params[5]) / self.params[1],
            )
        )

        A = torch.stack((A1, A2, A3, A4))
        # print(A)

        # A = torch.stack([[0, 1, 0, 0],
        #                 [-(self.params[2]+self.params[3])/self.params[0], -(self.params[4]+self.params[5])/self.params[0], (self.params[3])/self.params[0], (self.params[5])/self.params[0]],
        #                 [0, 0, 0, 1],
        #                 [(self.params[3])/self.params[1], (self.params[5])/self.params[1], -(self.params[3])/self.params[1], -(self.params[5])/self.params[1]]])

        B = torch.tensor([[0], [1], [0], [0]]) / self.params[0]

        Ad = torch.linalg.matrix_exp(self.Ts * A)
        Bd = torch.matmul(torch.linalg.inv(A), torch.matmul((Ad - torch.eye(4)), B))

        An = self.Tx @ Ad @ self.Tix
        Bn = self.Tx @ Bd @ self.Tiu

        w = torch.matmul(An, x_n) + torch.matmul(Bn, u_n)

        # xk_n = torch.matmul(self.Tx, xk)
        # yk_n = torch.matmul(self.Ty, yk)

        # print(xk.size())
        # print(xk_n.size())

        # w = xk_n
        return w


@added
class Nonlinear_MSD_State_Block(Discrete_Nonlinear_Function_Block):
    def __init__(
        self, Ts=0.02, std_x=np.ones((4, 1)), std_u=1, up_sample=10, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.Ts = Ts
        self.up_sample = up_sample

        self.nu = 1
        self.ny = 1
        self.nx = 4
        
        self.n = 2

        self.m = to_tensor([0.5, 0.4])
        self.k = to_tensor([100, 100])
        self.c = to_tensor([0.5, 0.5])
        self.a = to_tensor([0, 1000])
        self.d = to_tensor([0.05, 0])

        self.std_x = to_tensor(std_x)
        self.std_u = std_u
        
    def nonlinear_function(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu
        x = z[:, : self.nx, :]
        u = z[:, self.nx :, :]
        
        for i in range(self.up_sample):
            k1 = (self.Ts/self.up_sample)*self.deriv(x,u)
            k2 = (self.Ts/self.up_sample)*self.deriv(x+k1/2,u)
            k3 = (self.Ts/self.up_sample)*self.deriv(x+k2/2,u)
            k4 = (self.Ts/self.up_sample)*self.deriv(x+k3,u)
        
            x = (x + (k1+2*k2+2*k3+k4)/6)
        return x

    def deriv(self, x, u: Tensor):
        x1 = x[:, 0, :]
        x2 = x[:, 1, :]
        x3 = x[:, 2, :]
        x4 = x[:, 3, :]
        u = u[:, 0, :]

        # denormalize
        x1 = x1 * self.std_x[0]
        x2 = x2 * self.std_x[1]
        x3 = x3 * self.std_x[2]
        x4 = x4 * self.std_x[3]
        u = u * self.std_u

        # simulation continuous
        # print(x1.shape)
        x1k = x2
        x2k = (
            -(self.k[0] + self.k[1]) / self.m[0] * x1
            + self.k[1] / self.m[0] * x3
            - (self.c[0] + self.c[1]) / self.m[0] * x2
            + self.c[1] / self.m[0] * x4
            - self.a[0] / self.m[0] * torch.pow((x1), 3)
            + self.a[1] / self.m[0] * torch.pow((x3 -x1), 3)
            - self.d[0] / self.m[0] * torch.pow((x2), 3)
            + self.d[1] / self.m[0] * torch.pow((x4 - x2), 3)
            + 1 / self.m[0] * u
        )

        # x2k = (
        #     -(self.k[0]) / self.m[0] * x1
        #     # + self.k[1] / self.m[0] * x[2]
        #     - (self.c[0]) / self.m[0] * x2
        #     # + self.c[1] / self.m[0] * x[3]
        #     - self.a[0] / self.m[0] * torch.pow((x1), 3)
        #     # + self.a[1] / self.m[0] * np.power((x[2] -x1), 3)
        #     # - self.d[0] / self.m[0] * torch.pow((x2), 3)
        #     # + self.d[1] / self.m[0] * np.power((x[3] - x2), 3)
        #     + 1 / self.m[0] * u[0]
        # )

        xn = x3
        dxn = x4
        xn_ = x1
        dxn_ = x2
        x3k = dxn
        x4k = (
            -self.k[self.n - 1] / self.m[self.n - 1] * (xn - xn_)
            - self.c[self.n - 1] / self.m[self.n - 1] * (dxn - dxn_)
            - self.a[self.n - 1] / self.m[self.n - 1] * torch.pow((xn - xn_), 3)
            - self.d[self.n - 1] / self.m[self.n - 1] * torch.pow((dxn - dxn_), 3)
        )

        # normalize
        x1k = (x1k) / self.std_x[0]
        x2k = (x2k) / self.std_x[1]
        x3k = (x3k) / self.std_x[2]
        x4k = (x4k) / self.std_x[3]

        # xk = torch.hstack((x1k, x2k)).unsqueeze(-1)
        xk = torch.hstack((x1k, x2k, x3k, x4k)).unsqueeze(-1)
        return xk


# TODO: Not certain we need the output block
# physical_output_model_block = Nonlinear_MSD_Output_Block()


@added
class Gantry_State_Block(Discrete_Nonlinear_Function_Block):
    """
    Gantry continuous-time ODE integrated with RK4, using the LFR rational structure.

    State convention: x = [q1, q2, q3, q1_dot, q2_dot, q3_dot]  (logical coordinates)

    LFR signal flow inside deriv() — matches lfr_forward.py signal ordering:
      fnet = -K@q - C@qdot + P@u_stage          net logical force
      a    = N(Y)/d(Y) @ fnet                   rational M(Y)^{-1} (no matrix solve)
      z    = [a;  Y*a]                           LFR latent z (6-vec, not yet routed)
      w    = Y * z                               LFR latent w = Delta(Y)*z
      xdot = Ax@x + Bw@w + Bu@u_log             through G — NOT directly from a

    G is built from M0_inv = N0/d0 (constant, Y=0). Y-variation only enters via z, w.

    Frozen Y (Phase 1/2): Y_op is a float; N(Y_op), d(Y_op) precomputed at init.
                          deriv() is pure matmul — no dynamic solve.
    LPV     (Phase 3):    Y_op=None; Y = x[2] per step; Horner form for N(Y)/d(Y).

    Parameters
    ----------
    Y_op : float or None
        Frozen operating-point Y [m]. None enables LPV self-scheduling (Phase 3).
    std_x : array (6,1)
        State normalisation std, precomputed from training data.
    std_u : array (3,1)
        Input normalisation std, precomputed from training data.
    u_mean : array (3,1)
        Input mean offset, precomputed from training data. Must match
        fit_sys.norm.u0 so the block can recover physical forces.
        Default zeros (backward compatible with u0=0 normalisation).
    Ts : float
        Sample period [s]. Default 1/20000.
    """

    def __init__(
        self,
        Y_op: float = 0.3,
        std_x=np.ones((6, 1)),
        std_u=np.ones((3, 1)),
        x_mean=np.zeros((6, 1)),
        u_mean=np.zeros((3, 1)),
        Ts: float = 1 / 20000,
        up_sample: int = 10,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(nz=9, nw=6, *args, **kwargs)

        self.Ts   = Ts
        self.up_sample = up_sample
        self.nu   = 3
        self.ny   = 3
        self.nx   = 6
        self.Y_op = Y_op

        from model_augmentation.systems.gantry_ss import (
            mh as _mh, m1 as _m1, m2 as _m2, mb as _mb,
            Jb as _Jb, Jh as _Jh, Lb as _Lb, d as _d,
            M1 as _M1, M2 as _M2, K as _K, C as _C, P as _P,
            build_poly_constants, build_G_matrix_entries,
        )

        # Derive LFR polynomial constants from physical parameters.
        # This mirrors the pattern needed for joint estimation: when params become
        # nn.Parameter, call these same functions with the Parameter tensors.
        _alpha, _beta, _gamma, _N0, _N1, _N2 = build_poly_constants(
            _m1, _m2, _mb, _mh, _Jb, _Jh, _Lb, _d
        )
        _d0 = _mh * (_alpha * _gamma - _beta ** 2)   # det(M0)

        # Derive G matrix entries from M0_inv = N0/d0 (no solve — purely polynomial).
        _Ax, _Bw, _Bu, _A_combined = build_G_matrix_entries(_N0, _d0, _M1, _M2, _K, _C)

        # Register everything as buffers — moves with .to(device), not trainable.
        self.register_buffer("mh",     _mh.clone())
        self.register_buffer("alpha",  _alpha.clone())
        self.register_buffer("beta",   _beta.clone())
        self.register_buffer("gamma_", _gamma.clone())  # 'gamma' shadows Python builtin
        self.register_buffer("N0",     _N0.clone())
        self.register_buffer("N1",     _N1.clone())
        self.register_buffer("N2",     _N2.clone())

        self.register_buffer("Ax",         _Ax.clone())
        self.register_buffer("Bw",         _Bw.clone())
        self.register_buffer("Bu",         _Bu.clone())
        self.register_buffer("A_combined", _A_combined.clone())

        self.register_buffer("K_mat", _K.clone())
        self.register_buffer("C_mat", _C.clone())
        self.register_buffer("P_mat", _P.clone())

        # Precompute frozen-Y rational constants — avoids recomputing each RK4 substep.
        if Y_op is not None:
            Y_t = torch.tensor(Y_op, dtype=_N0.dtype)
            self.register_buffer("N_op", _N0 + _N1 * Y_t + _N2 * Y_t ** 2)  # (3,3)
            _d_op = _mh * (_alpha * _gamma - _beta ** 2
                           + 2 * _beta * _mh * Y_t
                           + _mh * (_alpha - _mh) * Y_t ** 2)
            self.register_buffer("d_op", _d_op)

        # Register as buffers so .to(device) moves them with the model.
        # std_x stored in both shapes to avoid reshape on every deriv() call:
        #   std_x    (6,1) — broadcast against (batch,6,1) for denorm
        #   std_x_1d (6,)  — broadcast against (batch,6)   for renorm
        # x_mean (6,1): state mean offset, subtracted before normalisation.
        #   For Y (index 2): x_mean[2] = Y_op (~0.3 m). All others zero.
        #   xdot renorm is unaffected: d(x_norm)/dt = d(x_phys)/dt / std_x
        #   because x_mean is constant.
        self.register_buffer("std_x",    to_tensor(std_x).reshape(6, 1))
        self.register_buffer("std_x_1d", to_tensor(std_x).reshape(6))
        self.register_buffer("std_u",    to_tensor(std_u).reshape(3, 1))
        self.register_buffer("x_mean",   to_tensor(x_mean).reshape(6, 1))
        self.register_buffer("u_mean",   to_tensor(u_mean).reshape(3, 1))

    def nonlinear_function(self, z: Tensor):
        # Copied verbatim from Nonlinear_MSD_State_Block — only self.nx/self.nu differ.
        assert z.size(1) == self.nx + self.nu
        x = z[:, : self.nx, :]
        u = z[:, self.nx :, :]

        for i in range(self.up_sample):
            k1 = (self.Ts / self.up_sample) * self.deriv(x, u)
            k2 = (self.Ts / self.up_sample) * self.deriv(x + k1 / 2, u)
            k3 = (self.Ts / self.up_sample) * self.deriv(x + k2 / 2, u)
            k4 = (self.Ts / self.up_sample) * self.deriv(x + k3, u)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x

    def _mats(self):
        # Hook: parameter-dependent quantities used by deriv(). The joint-estimation
        # subclass overrides this to supply per-forward rebuilt tensors (D-076).
        # The LPV branch also consumes the M(Y)-rational structure (mh, alpha, beta,
        # gamma_, N0, N1, N2); these become trainable once masses are estimated, so
        # they flow through the hook too (D-077) instead of being read as buffers.
        return (self.K_mat, self.C_mat, self.A_combined,
                self.mh, self.alpha, self.beta, self.gamma_,
                self.N0, self.N1, self.N2)

    def deriv(self, x: Tensor, u: Tensor) -> Tensor:
        # x: (batch, 6, 1) normalised   u: (batch, 3, 1) normalised
        (K_mat, C_mat, A_combined,
         mh, alpha, beta, gamma_, N0, N1, N2) = self._mats()

        # --- denormalise -------------------------------------------------
        x_phys = x * self.std_x + self.x_mean   # (batch, 6, 1)
        u_phys = u * self.std_u + self.u_mean      # (batch, 3, 1)

        # Work in 2D (batch, n) to match LFR signal-flow convention.
        x2 = x_phys.squeeze(-1)          # (batch, 6)
        u2 = u_phys.squeeze(-1)          # (batch, 3)

        # --- P transform: stage forces -> logical forces -----------------
        u_log = u2 @ self.P_mat.T        # (batch, 3)

        # --- net logical force -------------------------------------------
        # fnet = -K@q - C@qdot + u_logical
        fnet = (-(x2[:, :3] @ K_mat.T)
                - (x2[:, 3:] @ C_mat.T)
                + u_log)                 # (batch, 3)

        # --- LFR loop solve: a = N(Y)/d(Y) @ fnet -----------------------
        # Frozen (Phase 1/2): N_op, d_op precomputed at init → pure matmul.
        # LPV   (Phase 3):    Y = x[2]; Horner form for N(Y), d(Y) per step.
        if self.Y_op is not None:
            Y_val = self.Y_op                                  # Python scalar
            a = (self.N_op @ fnet.T).T / self.d_op            # (batch, 3)
        else:
            Y   = x2[:, 2]                                     # (batch,)
            dY  = mh * (alpha * gamma_ - beta ** 2
                        + 2 * beta * mh * Y
                        + mh * (alpha - mh) * Y ** 2)          # (batch,)
            Y_r = Y.unsqueeze(0)                               # (1, batch)
            n0f = N0 @ fnet.T                                  # (3, batch)
            n1f = N1 @ fnet.T                                  # (3, batch)
            n2f = N2 @ fnet.T                                  # (3, batch)
            a   = (n0f + Y_r * (n1f + Y_r * n2f)).T / dY[:, None]  # (batch, 3)  Horner
            Y_val = Y[:, None]                                 # (batch, 1)

        # --- LFR latent signals -----------------------------------------
        # z and w are computed explicitly — structurally present for future
        # routing through the Interconnect (Phase future: split into G-block + Δ-block).
        z = torch.cat([a, Y_val * a], dim=-1)   # (batch, 6)
        w = Y_val * z                            # (batch, 6)  w = Δ(Y)·z

        # --- state update through G (NOT directly from a) ---------------
        # xdot = Ax@x + Bw@w + Bu@u_log   (fused via A_combined)
        combined  = torch.cat([x2, w, u_log], dim=-1)   # (batch, 15)
        xdot_phys = combined @ A_combined.T              # (batch, 6)

        # --- renormalise and restore trailing dim -----------------------
        xdot = (xdot_phys / self.std_x_1d).unsqueeze(-1)  # (batch, 6, 1)
        return xdot


@added
class Parameterized_Gantry_State_Block(Gantry_State_Block):
    """
    Gantry_State_Block with ALL physical parameters trainable (joint estimation,
    D-076/D-077). Mirrors lpv_lfr_baseline/train_param_recovery exactly:

      * 14 RAW physical scalars are log-reparameterized (D-035) and trained
        individually: kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d.
        Positivity by construction guarantees M(Y) > 0 for all Y (invertibility
        proof, D-077), zero-init in log space -> start at params_init, uniform
        Adam gradient scaling.
      * Only Lb stays frozen -- it defines the coordinate frame (enters P), not
        the M(Y) rational structure.

    Because masses are trainable, the ENTIRE M(Y)-rational structure
    (alpha/beta/gamma, N0/N1/N2, d0, M1, M2) and K, C are rebuilt from the
    parameters once per timestep in nonlinear_function() via the gantry_ss
    builders (kept autograd-safe for exactly this call) and handed to the parent
    RK4 loop through the widened _mats() hook.

    Reporting exposes only the 10 data-identifiable quantities (kb_sum, cg1,
    cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d) -- see identifiable_
    combinations()/param_table(). The raw splits kb1/kb2, cb1/cb2, Jb/Jh and
    the individual mass components are trained but NOT separately trusted (flat
    directions, held near params_init by the param_loss anchor). m_diff is
    signed and is a DERIVED readout of the individually-logged m1, m2 -- it is
    never itself a parameter. "Train raw, trust combinations" (D-077).

    param_loss(): Lambda-weighted L2 toward params_init in physical space,
    Lambda = RMSE_baseline / params_init (D-034). Picked up by
    SSE_Interconnect_ParamLoss via its hasattr sweep.

    Parameters
    ----------
    RMSE_baseline : float
        Loss-scale reference for Lambda (D-034). Pipeline value is measured,
        see PARAM_RMSE_BASELINE in gantry_interconnect_dynamic.py.
    flag_loss_reg : bool
        Disable to drop the regularization term (param_loss() returns 0.0).
    params_init : array-like (14,), optional
        Override of the nominal initial values -- used by the detuned recovery
        diagnostic. Regularization anchors to THESE values. Order = PARAM_NAMES.
    """

    PARAM_NAMES = ["kb1", "kb2", "cg1", "cg2", "cy", "cb1", "cb2",
                   "mh", "m1", "m2", "mb", "Jb", "Jh", "d"]

    def __init__(self, RMSE_baseline: float = 1.0, flag_loss_reg: bool = True,
                 params_init=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        from model_augmentation.systems.gantry_ss import (
            kb1 as _kb1, kb2 as _kb2, cg1 as _cg1, cg2 as _cg2, cy as _cy,
            cb1 as _cb1, cb2 as _cb2, mh as _mh, m1 as _m1, m2 as _m2,
            mb as _mb, Jb as _Jb, Jh as _Jh, d as _d, Lb as _Lb,
            build_poly_constants as _build_poly,
            build_G_matrix_entries as _build_G,
        )
        # Plain attributes (module-level fns, picklable) — the autograd-safe
        # builders called with the trainable parameters each forward.
        self._build_poly = _build_poly
        self._build_G = _build_G

        if params_init is None:
            params_init = torch.stack([_kb1, _kb2, _cg1, _cg2, _cy, _cb1, _cb2,
                                       _mh, _m1, _m2, _mb, _Jb, _Jh, _d])
        params_init = to_tensor(params_init).reshape(14).clone()

        # log(theta/params_init): zeros -> start exactly at params_init (D-035)
        self.log_params = nn.Parameter(torch.zeros(14, dtype=params_init.dtype))
        self.register_buffer("params_init", params_init)
        # Lambda[i] = RMSE_baseline / params_init[i]  (D-034)
        self.register_buffer(
            "Lambda",
            torch.as_tensor(RMSE_baseline, dtype=params_init.dtype) / params_init)
        self.flag_loss_reg = flag_loss_reg

        self.register_buffer("Lb", _Lb.clone())   # frozen: coordinate frame only

        # Per-forward rebuilt structure, set in nonlinear_function(); the parent's
        # nominal mh/alpha/.../N0 buffers are shadowed by these and go unused.
        self._cur = None

    def _recover_params(self) -> Tensor:
        """Physical parameters: params_init * exp(log_params), clamped > 0 (D-035)."""
        return (self.params_init * torch.exp(self.log_params)).clamp(min=1e-6)

    def _build_KC(self, p: Tensor):
        """Differentiable K(theta), C(theta) -- same structure as gantry_ss K, C.
        Uses only the identifiable sums kb1+kb2, cb1+cb2 (matrices never depend
        on the individual splits)."""
        (kb1, kb2, cg1, cg2, cy, cb1, cb2,
         mh, m1, m2, mb, Jb, Jh, d) = p
        z = p.new_zeros(())
        Lb = self.Lb
        kb_sum, cb_sum = kb1 + kb2, cb1 + cb2
        K = torch.stack([
            torch.stack([z, z,      z]),
            torch.stack([z, kb_sum, z]),
            torch.stack([z, z,      z]),
        ])
        C = torch.stack([
            torch.stack([cg1 + cg2,            (cg1 - cg2) * Lb / 2,                z]),
            torch.stack([(cg1 - cg2) * Lb / 2, cb_sum + (cg1 + cg2) * Lb ** 2 / 4,  z]),
            torch.stack([z,                    z,                                   cy]),
        ])
        return K, C

    @staticmethod
    def _build_M1M2(mh: Tensor):
        """M1, M2 coefficients of M(Y) -- depend on mh only (gantry_ss M1, M2)."""
        z = mh.new_zeros(())
        M1 = torch.stack([
            torch.stack([z,   -mh,  z]),
            torch.stack([-mh,  z,   z]),
            torch.stack([z,    z,   z]),
        ])
        M2 = torch.stack([
            torch.stack([z,   z,   z]),
            torch.stack([z,   mh,  z]),
            torch.stack([z,   z,   z]),
        ])
        return M1, M2

    def nonlinear_function(self, z: Tensor):
        # Rebuild the full M(Y) rational structure once per timestep from the
        # current parameters; every RK4 substep reuses it via the _mats() hook.
        p = self._recover_params()
        (kb1, kb2, cg1, cg2, cy, cb1, cb2,
         mh, m1, m2, mb, Jb, Jh, d) = p
        alpha, beta, gamma_, N0, N1, N2 = self._build_poly(
            m1, m2, mb, mh, Jb, Jh, self.Lb, d)
        d0 = mh * (alpha * gamma_ - beta ** 2)          # det(M0)
        M1, M2 = self._build_M1M2(mh)
        K, C = self._build_KC(p)
        _, _, _, A_combined = self._build_G(N0, d0, M1, M2, K, C)
        self._cur = (K, C, A_combined, mh, alpha, beta, gamma_, N0, N1, N2)
        return super().nonlinear_function(z)

    def _mats(self):
        if self._cur is None:
            raise RuntimeError(
                "Parameterized_Gantry_State_Block.deriv() must be reached via "
                "nonlinear_function(), which rebuilds the M(Y) structure first.")
        return self._cur

    def param_loss(self):
        """Lambda-weighted L2 toward params_init in physical space (D-034)."""
        if not self.flag_loss_reg:
            return 0.0
        p = self._recover_params()
        return nn.functional.mse_loss(
            self.Lambda * p, self.Lambda * self.params_init, reduction="sum")

    def physical_params(self) -> dict:
        """Current RAW physical parameter values as {name: float} (14 scalars)."""
        vals = self._recover_params().detach()
        return {n: vals[i].item() for i, n in enumerate(self.PARAM_NAMES)}

    # ------------------------------------------------------------------
    # Reporting: the 10 data-identifiable combinations (D-077). Raw params are
    # trained; only these are trusted. m_diff is signed (derived from m1, m2).
    # ------------------------------------------------------------------
    _IDENT_ORDER = ["kb_sum", "cg1", "cg2", "cy", "cb_sum", "mh",
                    "m_total", "m_diff", "J_eff", "d"]

    @staticmethod
    def _combos_from_raw(p: dict, Lb: float) -> dict:
        return {
            "kb_sum":  p["kb1"] + p["kb2"],
            "cg1":     p["cg1"],
            "cg2":     p["cg2"],
            "cy":      p["cy"],
            "cb_sum":  p["cb1"] + p["cb2"],
            "mh":      p["mh"],
            "m_total": p["m1"] + p["m2"] + p["mb"],
            "m_diff":  p["m1"] - p["m2"],                       # signed
            "J_eff":   p["Jb"] + p["Jh"] + (p["m1"] + p["m2"]) * Lb ** 2 / 4,
            "d":       p["d"],
        }

    def identifiable_combinations(self) -> dict:
        """The 10 trusted combinations from the current raw parameters."""
        return self._combos_from_raw(self.physical_params(), self.Lb.item())

    def param_table(self) -> str:
        """True / init / learned comparison of the 10 identifiable combinations."""
        from model_augmentation.systems import gantry_ss as _gss
        Lb = self.Lb.item()
        true_raw = {n: getattr(_gss, n).item() for n in self.PARAM_NAMES}
        det_raw = {n: self.params_init[i].item()
                   for i, n in enumerate(self.PARAM_NAMES)}
        true_c = self._combos_from_raw(true_raw, Lb)
        det_c = self._combos_from_raw(det_raw, Lb)
        lrn_c = self.identifiable_combinations()
        hdr = (f"{'Quantity':<10}{'True':>12}{'Init':>12}"
               f"{'Learned':>12}{'delta%':>10}")
        lines = ["  Identifiable combinations (raw trained, combos trusted):",
                 hdr, "-" * 56]
        for n in self._IDENT_ORDER:
            t, i, l = true_c[n], det_c[n], lrn_c[n]
            dp = (l - t) / t * 100 if abs(t) > 1e-12 else float("nan")
            lines.append(f"{n:<10}{t:>12.4f}{i:>12.4f}{l:>12.4f}{dp:>+9.2f}%")
        return "\n".join(lines)
