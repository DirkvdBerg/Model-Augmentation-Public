from deepSI.utils.torch_nets import feed_forward_nn, simple_res_net
from deepSI.system_data import System_data

from torch import nn, Tensor
import torch

from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.utils.utils import *

class Block(nn.Module):
    '''Basic block implementation with variables that need to be defined for interconnect.'''
    def __init__(self, nz, nw, name=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.nw = nw
        self.nz = nz

        self.name = name
        self.block_ix = None

    def init_block(self, z: Tensor):
        return

    def forward(self, z: Tensor):
        raise NotImplementedError('This function should return w computed from z.')
    
class Static_ANN_Block(Block):
    def __init__(self, net=zero_init_feed_forward_nn, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh, *args, **kwargs) -> None:
        super(Static_ANN_Block, self).__init__(*args, **kwargs)

        self.net = net(n_in=self.nz, n_out=self.nw, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, z: Tensor):
        # print(self.name + " forward called for w" + str(self.block_ix))
        assert z.size(1) == self.nz

        # z_scaled = torch.matmul(self.D_z, z)

        w = self.net(z.view(-1,self.nz))
        return w.view(-1,self.nw,1)
        # return torch.matmul(self.D_yw, w.view(-1,self.nw,1))

class Cubic_Block(Block):
    def __init__(self, aL, *args, **kwargs) -> None:
        super(Cubic_Block, self).__init__(*args, **kwargs)
        self.aL = aL
        
    def forward(self, z: Tensor):
        assert z.size(1) == self.nz

        return torch.mul(torch.pow(z, 3),self.aL)

class Linear_State_Block(Block):
    def __init__(self, A=torch.empty((0,0)), B=torch.empty((0,0)), *args, **kwargs) -> None:
        # Matrices defining the known linear ss model
        self.A = to_tensor(A)
        self.B = to_tensor(B)
        
        self.nx = self.A.size(0) if self.A.numel() else 0
        self.nu = self.B.size(1) if self.B.numel() else 0
        if self.nx == 0:
            self.nx = self.B.size(0) if self.B.numel() else 0

        super().__init__(nw=self.nx, nz=self.nx+self.nu, *args, **kwargs)
    
    def forward(self, z: Tensor):
        # print(self.name + " forward called for w" + str(self.block_ix))
        # print(z.shape)
        assert z.size(1) == self.nx + self.nu
        x = z[:,:self.nx,:]
        u = z[:,self.nx:,:]
        w = torch.matmul(self.A, x) + torch.matmul(self.B, u)
        return w
    
class Linear_Output_Block(Block):
    def __init__(self, C=torch.empty((0,0)), D=torch.empty((0,0)), *args, **kwargs) -> None:

        self.C = to_tensor(C)
        self.D = to_tensor(D)

        self.ny = self.C.size(0) if self.C.numel() else 0
        self.nx = self.C.size(1) if self.C.numel() else 0
        self.nu = self.D.size(1) if self.D.numel() else 0
        if self.ny == 0:
            self.ny = self.D.size(0) if self.D.numel() else 0

        super().__init__(nw=self.ny, nz=self.nx+self.nu, *args, **kwargs)

    def forward(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu
        x = z[:,:self.nx,:]
        u = z[:,self.nx:,:]
        w = torch.matmul(self.C, x) + torch.matmul(self.D, u)
        return w

class Parameterized_Linear_State_Block(Block):
    def __init__(self, A=torch.empty((0,0)), B=torch.empty((0,0)), RMSE_baseline=1.0, flag_loss_reg = True,*args, **kwargs) -> None:
        # Matrices defining the known linear ss model
        self.A_init = to_tensor(A)
        self.B_init = to_tensor(B)
        
        self.nx = self.A_init.size(0) if self.A_init.numel() else 0
        self.nu = self.B_init.size(1) if self.B_init.numel() else 0
        if self.nx == 0:
            self.nx = self.B_init.size(0) if self.B_init.numel() else 0

        super().__init__(nw=self.nx, nz=self.nx+self.nu, *args, **kwargs)

        self.A = nn.Parameter(to_tensor(A).clone())
        self.B = nn.Parameter(to_tensor(B).clone())

        # These lambda matrices are used to scale the loss function in interconnect.py
        self.Lambda_A = (torch.ones(self.A.shape) / self.A_init)*RMSE_baseline
        self.Lambda_A[torch.isinf(self.Lambda_A)] = 0.0
        self.Lambda_B = (torch.ones(self.B.shape) / self.B_init)*RMSE_baseline
        self.Lambda_B[torch.isinf(self.Lambda_B)] = 0.0

        self.flag_loss_reg = flag_loss_reg
    
    def forward(self, z: Tensor):
        # print(self.name + " forward called for w" + str(self.block_ix))
        # print(z.shape)
        assert z.size(1) == self.nx + self.nu
        x = z[:,:self.nx,:]
        u = z[:,self.nx:,:]
        w = torch.matmul(self.A, x) + torch.matmul(self.B, u)
        return w
    
    def param_loss(self):
        if self.flag_loss_reg:
            loss_theta = nn.functional.mse_loss(self.Lambda_A * self.A, self.Lambda_A * self.A_init, reduction="sum") \
                    + nn.functional.mse_loss(self.Lambda_B * self.B, self.Lambda_B * self.B_init, reduction="sum")
            return loss_theta
        else:
            return 0.0
    
class Parameterized_Linear_Output_Block(Block):
    def __init__(self, C=torch.empty((0,0)), D=torch.empty((0,0)), RMSE_baseline=1.0, flag_loss_reg = True, *args, **kwargs) -> None:
        # Matrices defining the known linear ss model
        self.C_init = to_tensor(C)
        self.D_init = to_tensor(D)
        
        self.ny = self.C_init.size(0) if self.C_init.numel() else 0
        self.nx = self.C_init.size(1) if self.C_init.numel() else 0
        self.nu = self.D_init.size(1) if self.D_init.numel() else 0
        if self.ny == 0:
            self.ny = self.D_init.size(0) if self.D_init.numel() else 0

        super().__init__(nw=self.ny, nz=self.nx+self.nu, *args, **kwargs)

        self.C = nn.Parameter(to_tensor(C).clone())
        self.D = nn.Parameter(to_tensor(D).clone())

        # These lambda matrices are used to scale the loss function in interconnect.py
        self.Lambda_C = (torch.ones(self.C.shape) / self.C_init)*RMSE_baseline
        self.Lambda_C[torch.isinf(self.Lambda_C)] = 0.0
        self.Lambda_D = (torch.ones(self.D.shape) / self.D_init)*RMSE_baseline
        self.Lambda_D[torch.isinf(self.Lambda_D)] = 0.0

        self.flag_loss_reg = flag_loss_reg
    
    def forward(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu
        x = z[:,:self.nx,:]
        u = z[:,self.nx:,:]
        w = torch.matmul(self.C, x) + torch.matmul(self.D, u)
        return w
    
    def param_loss(self):
        if self.flag_loss_reg:
            loss_theta = nn.functional.mse_loss(self.Lambda_C * self.C, self.Lambda_C * self.C_init, reduction="sum") \
                    + nn.functional.mse_loss(self.Lambda_D * self.D, self.Lambda_D * self.D_init, reduction="sum")
            return loss_theta
        else:
            return 0.0
    
class Discrete_Nonlinear_Function_Block(Block):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def init_norm(self, sys_data: System_data):
        raise NotImplementedError("This function should determine normalization matrices Tw and Tiz for the given input data")

    def forward(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu
        w = self.nonlinear_function(z)
        return w
    
    def nonlinear_function(self, z: Tensor):
        raise NotImplementedError("This function should describe the discrete nonlinear function of the block")
    
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
        self.x2max = 10 #params[7]
        self.ymax = 10
        self.yoffset = params[8]

        self.ystd = 2.165135419802158
        self.y0 = 5.582729231040664
        self.ustd = 0.9995115994824683
        self.u0 = 2.8

    
    def nonlinear_function(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu
        x = z[:,:self.nx,:]; x1 = x[:,0,:]; x2 = x[:,1,:]
        u = z[:,self.nx:,:]; u = u[:,0,:]

        #denormalize
        x1 = (x1*self.ystd) + self.y0
        x2 = (x2*self.ystd) + self.y0
        u = (u*self.ustd) + self.u0

        x1 = torch.clamp(x1, min=0.00001)
        x1k = torch.clamp(x1 + self.Ts*(-self.k1*torch.sqrt(x1) + self.k2*x1 + self.k3*u), max=self.x1max)
        
        x2 = torch.clamp(x2, min=0.00001)
        mask = torch.le(torch.ones(x1.size())*self.x1max, x1)
        x2k = torch.clamp(x2 + self.Ts*(self.k1*torch.sqrt(x1) - self.k2*x1 - self.k4*torch.sqrt(x2) + self.k5*x2), max=self.x2max)
        x2k_overflow = torch.clamp(x2 + self.Ts*(self.k1*torch.sqrt(x1) - self.k2*x1 - self.k4*torch.sqrt(x2) + self.k5*x2 + self.k6*u), max=self.x2max)
        
        x2k[mask] = x2k_overflow[mask]

        yk = torch.clamp(x2 + self.yoffset, max=self.ymax)

        #normalize
        x1k = (x1k - self.y0)/self.ystd
        x2k = (x2k - self.y0)/self.ystd 
        # x1 = (x1 - self.y0)/self.ystd
        # x2 = (x2 - self.y0)/self.ystd
        yk = (yk - self.y0)/self.ystd 

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
        self.x2max = 10 #params[7]
        self.ymax = 10
        self.yoffset = nn.Parameter(torch.tensor(params[8]))#params[8]

        self.ystd = 2.165135419802158
        self.y0 = 5.582729231040664
        self.ustd = 0.9995115994824683
        self.u0 = 2.8

    
    def nonlinear_function(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu
        x = z[:,:self.nx,:]; x1 = x[:,0,:]; x2 = x[:,1,:]
        u = z[:,self.nx:,:]; u = u[:,0,:]

        #denormalize
        x1 = (x1*self.ystd) + self.y0
        x2 = (x2*self.ystd) + self.y0
        u = (u*self.ustd) + self.u0

        x1 = torch.clamp(x1, min=0.00001)
        x1k = torch.clamp(x1 + self.Ts*(-self.k1*torch.sqrt(x1) + self.k2*x1 + self.k3*u), max=self.x1max)
        
        x2 = torch.clamp(x2, min=0.00001)
        mask = torch.le(torch.ones(x1.size())*self.x1max, x1)
        x2k = torch.clamp(x2 + self.Ts*(self.k1*torch.sqrt(x1) - self.k2*x1 - self.k4*torch.sqrt(x2) + self.k5*x2), max=self.x2max)
        x2k_overflow = torch.clamp(x2 + self.Ts*(self.k1*torch.sqrt(x1) - self.k2*x1 - self.k4*torch.sqrt(x2) + self.k5*x2 + self.k6*u), max=self.x2max)
        
        x2k[mask] = x2k_overflow[mask]

        yk = torch.clamp(x2 + self.yoffset, max=self.ymax)

        #normalize
        x1k = (x1k - self.y0)/self.ystd
        x2k = (x2k - self.y0)/self.ystd 
        # x1 = (x1 - self.y0)/self.ystd
        # x2 = (x2 - self.y0)/self.ystd
        yk = (yk - self.y0)/self.ystd 

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
            self.init_params = to_tensor(np.array([0.5, 0.4, 100, 100, 0.5, 0.5])) # m1, m2, k1, k2, c1, c2
        elif FP_type == "approximate":
            self.init_params = to_tensor(np.array([0.5, 0.4, 95, 95, 0.45, 0.45])) # non ideal: m1, m2, k1, k2, c1, c2
        else:
            raise ValueError("FP_type must be either 'ideal' or 'approximate'")

        self.params = nn.Parameter(self.init_params.clone())

        self.Tx = to_tensor(np.array([[8.9022888, 0., 0., 0.],
                            [0., 0.77655197, 0., 0.],
                            [0., 0., 5.86104999, 0.],
                            [0., 0., 0., 0.59214659]]))
        self.Tix = to_tensor(np.array([[0.11233066, 0., 0., 0.],
                            [0., 1.28774381, 0., 0.],
                            [0., 0., 0.17061789, 0.],
                            [0., 0., 0., 1.68877101]]))
        self.Tiu = to_tensor(np.array([[10.]]))
        self.Ty = to_tensor(np.array([[5.85660401]]))

        # RMSE_baseline = 0.2 # ideal
        RMSE_baseline = 0.2 # non ideal
        self.epsilon = 1
        self.Lambda = np.sqrt(1/self.epsilon)*RMSE_baseline*torch.linalg.inv(torch.diag(self.init_params))
    
    def nonlinear_function(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu

        x_n = z[:,:self.nx,:]
        u_n = z[:,self.nx:,:]


        A1 = torch.tensor([0,1,0,0])
        A2 = torch.stack((-(self.params[2]+self.params[3])/self.params[0], -(self.params[4]+self.params[5])/self.params[0], (self.params[3])/self.params[0], (self.params[5])/self.params[0]))
        A3 = torch.tensor([0,0,0,1])
        A4 = torch.stack(((self.params[3])/self.params[1], (self.params[5])/self.params[1], -(self.params[3])/self.params[1], -(self.params[5])/self.params[1]))

        A = torch.stack((A1,A2,A3,A4))
        # print(A)

        # A = torch.stack([[0, 1, 0, 0],
        #                 [-(self.params[2]+self.params[3])/self.params[0], -(self.params[4]+self.params[5])/self.params[0], (self.params[3])/self.params[0], (self.params[5])/self.params[0]],
        #                 [0, 0, 0, 1],
        #                 [(self.params[3])/self.params[1], (self.params[5])/self.params[1], -(self.params[3])/self.params[1], -(self.params[5])/self.params[1]]])
        
        B = torch.tensor([[0], [1], [0], [0]])/self.params[0]

        Ad = torch.linalg.matrix_exp(self.Ts*A)
        Bd = torch.matmul(torch.linalg.inv(A),torch.matmul((Ad - torch.eye(4)),B))

        An = self.Tx @ Ad @ self.Tix
        Bn = self.Tx @ Bd @ self.Tiu

        w = torch.matmul(An, x_n) + torch.matmul(Bn, u_n)

        # xk_n = torch.matmul(self.Tx, xk)
        # yk_n = torch.matmul(self.Ty, yk)

        # print(xk.size())
        # print(xk_n.size())

        # w = xk_n
        return w
