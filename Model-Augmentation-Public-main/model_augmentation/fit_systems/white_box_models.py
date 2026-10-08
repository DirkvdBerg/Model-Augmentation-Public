from deepSI import system_data

import torch
from torch import nn, Tensor
import numpy as np

from model_augmentation.utils.utils import determine_std_T_sys_data

class Discrete_White_Box_Model(nn.Module):
    def __init__(self, nx, nu, ny, dt, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dt = dt
        self.nx = nx
        self.nu = nu
        self.ny = ny

    def forward(self, x: Tensor, u: Tensor):
        if len(x.size()) <= 2:
            x = x.view(x.size(0), self.nx, 1)
        if len(u.size()) <= 2:
            u = u.view(u.size(0), self.nu, 1)
        self.nb = x.size(0)
        
        xp = self.f(x,u)
        y = self.h(x,u)

        xp = xp.view(self.nb, self.nx)
        if self.ny == 1: y = y.view(self.nb)
        if self.ny >= 2: y = y.view(self.nb, self.ny)
        return y, xp
    
    def f(self, x: Tensor, u: Tensor):
        assert NotImplementedError("Should return xp")
    
    def h(self, x: Tensor, u: Tensor):
        assert NotImplementedError("Should return y")

class Discrete_Cascaded_Tanks(Discrete_White_Box_Model):
    def __init__(self, dt=4.0, *args, **kwargs) -> None:
        super().__init__(dt=dt, *args, **kwargs)

        self.k1 = torch.nn.Parameter(torch.ones((1,1))/5)
        self.k2 = torch.nn.Parameter(torch.ones((1,1))/5)
        self.k3 = torch.nn.Parameter(torch.ones((1,1))/5)
        self.k4 = torch.nn.Parameter(torch.ones((1,1)))

        self.Tu = torch.eye(self.nu)
        self.Tiu = torch.eye(self.nu)
        self.Ty = torch.eye(self.ny)
        self.Tiy = torch.eye(self.ny)
        self.Tx = torch.eye(self.nx)
        self.Tix = torch.eye(self.nx)

        self.m = nn.ReLU()

    def init_model(self, sys_data: system_data):
        Tu, Tiu, Ty, Tiy = determine_std_T_sys_data(sys_data)
        Tx = np.diag(np.concatenate((Ty, Ty)).reshape(2*self.ny,))
        Tix = np.linalg.inv(Tx)

        self.Tu = Tensor(Tu); self.Tiu = Tensor(Tiu)
        self.Ty = Tensor(Ty); self.Tiy = Tensor(Tiy)
        self.Tx = Tensor(Tx); self.Tix = Tensor(Tix)

    def f(self, xn: Tensor, un: Tensor):
        x = torch.matmul(self.Tix, xn)
        u = torch.matmul(self.Tiu, un)
        x1 = x[:,0,:].view(-1,1,1)
        x2 = x[:,1,:].view(-1,1,1)

        xp1 = x1 + self.dt*(-self.k1*torch.sqrt(x1) + self.k4*u)
        xp2 = x2 + self.dt*(self.k2*torch.sqrt(x1) - self.k3*torch.sqrt(x2))

        xp = torch.cat((xp1, xp2), dim=1)
        xpn = torch.matmul(self.Tx, xp)

        return self.m(xpn)

    def h(self, xn: Tensor, un: Tensor):
        x = torch.matmul(self.Tix, xn)
        u = torch.matmul(self.Tiu, un)


        y = x[:,1,:].view(-1, self.ny, 1)
        yn = torch.matmul(self.Ty, y)

        return yn