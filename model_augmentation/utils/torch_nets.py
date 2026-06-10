from torch import nn, Tensor
import torch
import numpy as np

class linear_encoder_net(nn.Module):
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(linear_encoder_net, self).__init__()
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = nn.Linear(nb*np.prod(self.nu,dtype=int) + na*np.prod(self.ny,dtype=int), nx, bias=False)
        # self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + na*np.prod(self.ny,dtype=int), \
        #     n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, upast, ypast):
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1) # type: ignore
        return self.net(net_in)

class feed_forward_nn(nn.Module): #a simple MLP
    def __init__(self,n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(feed_forward_nn,self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        seq = [nn.Linear(n_in,n_nodes_per_layer),activation()]
        assert n_hidden_layers>0
        for i in range(n_hidden_layers-1):
            seq.append(nn.Linear(n_nodes_per_layer,n_nodes_per_layer))
            seq.append(activation())
        seq.append(nn.Linear(n_nodes_per_layer,n_out))
        self.net = nn.Sequential(*seq)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, val=0) #bias
    def forward(self,X):
        return self.net(X)

class identity_init_simple_res_net(nn.Module):
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        #linear + non-linear part 
        super(identity_init_simple_res_net,self).__init__()
        self.net_lin = nn.Linear(n_in,n_out)
        self.n_in = n_in
        self.n_out = n_out
        if n_hidden_layers>0:
            self.net_non_lin = zero_init_feed_forward_nn(n_in,n_out,n_nodes_per_layer=n_nodes_per_layer,n_hidden_layers=n_hidden_layers,activation=activation)
        else:
            self.net_non_lin = None
        
        for m in self.net_lin.modules():
            if isinstance(m, nn.Linear):
                nn.init.eye_(m.weight)
                nn.init.constant_(m.bias, val=0)


    def forward(self,x):
        if self.net_non_lin is not None:
            return self.net_lin(x) + self.net_non_lin(x)
        else: #linear
            return self.net_lin(x)
        
class linear_mapping(nn.Module):
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        #linear + non-linear part 
        super(linear_mapping,self).__init__()
        self.net_lin = nn.Linear(n_in,n_out)
        self.n_in = n_in
        self.n_out = n_out
        
        for m in self.net_lin.modules():
            if isinstance(m, nn.Linear):
                nn.init.eye_(m.weight)
                nn.init.constant_(m.bias, val=0)


    def forward(self,x):
            return self.net_lin(x)
        
class zero_init_linear_mapping(nn.Module):
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        #linear + non-linear part 
        super(zero_init_linear_mapping,self).__init__()
        self.net_lin = nn.Linear(n_in,n_out)
        self.n_in = n_in
        self.n_out = n_out
        
        nn.init.constant_(self.net_lin.bias, val=0.0)
        nn.init.constant_(self.net_lin.weight, val=0.0)


    def forward(self,x):
            return self.net_lin(x)

class zero_init_feed_forward_nn(nn.Module): #a simple MLP
    def __init__(self,n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(zero_init_feed_forward_nn,self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        seq = [nn.Linear(n_in,n_nodes_per_layer),activation()]
        assert n_hidden_layers>0
        for i in range(n_hidden_layers-1):
            seq.append(nn.Linear(n_nodes_per_layer,n_nodes_per_layer))
            seq.append(activation())

        final_layer = nn.Linear(n_nodes_per_layer,n_out)
        seq.append(final_layer)

        self.net = nn.Sequential(*seq)

        nn.init.constant_(final_layer.bias, val=0.0)
        nn.init.constant_(final_layer.weight, val=0.0)


    def forward(self,X):
        return self.net(X)
    
class zero_init_resnet(nn.Module): #a simple MLP
    def __init__(self,n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(zero_init_resnet,self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        seq = [nn.Linear(n_in,n_nodes_per_layer),activation()]
        assert n_hidden_layers>0
        for i in range(n_hidden_layers-1):
            seq.append(nn.Linear(n_nodes_per_layer,n_nodes_per_layer))
            seq.append(activation())

        final_layer = nn.Linear(n_nodes_per_layer,n_out)
        seq.append(final_layer)

        self.net = nn.Sequential(*seq)

        self.net_lin = nn.Linear(n_in,n_out)

        nn.init.constant_(final_layer.bias, val=0.0)
        nn.init.constant_(final_layer.weight, val=0.0)        
        
        nn.init.constant_(self.net_lin.bias, val=0.0)
        nn.init.constant_(self.net_lin.weight, val=0.0)

    def forward(self,X):
        return self.net(X) + self.net_lin(X)
    
## unit variance should be removed, but was still used to train current models
class unit_variance_feed_forward_nn(nn.Module): #a simple MLP
    def __init__(self,n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(unit_variance_feed_forward_nn,self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        seq = [nn.Linear(n_in,n_nodes_per_layer),activation()]
        assert n_hidden_layers>0
        for i in range(n_hidden_layers-1):
            seq.append(nn.Linear(n_nodes_per_layer,n_nodes_per_layer))
            seq.append(activation())

        final_layer = nn.Linear(n_nodes_per_layer,n_out)
        seq.append(final_layer)

        self.net = nn.Sequential(*seq)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # nn.init.uniform_(m.weight, a=-1, b=1)
                nn.init.constant_(m.bias, val=0)
                # nn.init.zeros_(m.bias)

        # nn.init.zeros_(final_layer.bias)
        # nn.init.zeros_(final_layer.weight)

        nn.init.constant_(final_layer.bias, val=0.0)
        nn.init.constant_(final_layer.weight, val=0.0)

    def forward(self,X):
        return self.net(X)
    
class HybridGantryEncoder(nn.Module):
    """Hybrid encoder: analytical physical states + learned augmented states.

    Physical states (positions + velocities) are computed analytically from the
    output measurements using the measurement equation q = P_inv @ y and backward
    finite differences. Only the augmented states are learned by a zero-init ANN.
    """
    def __init__(self, nb, nu, na, ny, nx,
                 P_inv_T, y0, ystd, x_mean, std_x, fs, NX_PHYS=6,
                 n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(HybridGantryEncoder, self).__init__()
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu, int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny, int) else ny)
        self.NX_PHYS = NX_PHYS
        self.NX_ANN = nx - NX_PHYS
        self.fs = fs

        # Physics constants (non-learnable, move with .to(device))
        self.register_buffer('P_inv', torch.tensor(np.asarray(P_inv_T).T, dtype=torch.float32))  # inv(P), (3,3)
        self.register_buffer('y0', torch.tensor(np.asarray(y0), dtype=torch.float32))             # (3,)
        self.register_buffer('ystd', torch.tensor(np.asarray(ystd), dtype=torch.float32))         # (3,)
        self.register_buffer('x_mean', torch.tensor(np.asarray(x_mean), dtype=torch.float32))     # (6,)
        self.register_buffer('std_x', torch.tensor(np.asarray(std_x), dtype=torch.float32))       # (6,)

        # ANN for augmented states (zero-init → starts at zero)
        if self.NX_ANN > 0:
            n_in = nb * np.prod(self.nu, dtype=int) + na * np.prod(self.ny, dtype=int)
            self.ann = zero_init_feed_forward_nn(
                n_in=n_in, n_out=self.NX_ANN,
                n_nodes_per_layer=n_nodes_per_layer,
                n_hidden_layers=n_hidden_layers,
                activation=activation,
            )
        else:
            self.ann = None

    def forward(self, upast, ypast):
        # ── Physical states (analytical, no gradients) ──
        # Un-normalize y: ypast is (batch, na, ny) in normalized coords
        y_denorm = ypast * self.ystd + self.y0                  # (batch, na, ny)

        # deepSI convention (na_right=0): ypast ends at y[k-1], not y[k].
        # The encoder must produce x(k), so we extrapolate one step forward.
        # THEORY: measurement equation q = P^{-1} y (row-vector form: q = y @ inv(P))
        pos_km1 = y_denorm[:, -1, :] @ self.P_inv              # q(k-1)
        pos_km2 = y_denorm[:, -2, :] @ self.P_inv              # q(k-2)
        pos_km3 = y_denorm[:, -3, :] @ self.P_inv              # q(k-3)

        # HEURISTIC: linear extrapolation to q(k) and v(k), both O(Ts^2) accurate.
        # v(k-1) = (q(k-1) - q(k-2)) * fs, v(k-2) = (q(k-2) - q(k-3)) * fs
        # q(k) = 2*q(k-1) - q(k-2)         (linear extrapolation of position)
        # v(k) = 2*v(k-1) - v(k-2)         (linear extrapolation of velocity)
        vel_km1 = (pos_km1 - pos_km2) * self.fs
        vel_km2 = (pos_km2 - pos_km3) * self.fs
        pos = 2 * pos_km1 - pos_km2                            # q(k) extrapolated
        vel = 2 * vel_km1 - vel_km2                            # v(k) extrapolated

        x_phys = torch.cat([pos, vel], dim=1)                  # (batch, 6)
        x_phys_norm = ((x_phys - self.x_mean) / self.std_x).detach()  # (batch, 6), no grad

        # ── Augmented states (learned) ──
        if self.ann is not None:
            net_in = torch.cat([upast.view(upast.shape[0], -1),
                                ypast.view(ypast.shape[0], -1)], dim=1)
            x_ann = self.ann(net_in)                            # (batch, NX_ANN)
            return torch.cat([x_phys_norm, x_ann], dim=1)       # (batch, nx)
        else:
            return x_phys_norm


class positive_default_encoder_net(nn.Module):
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(positive_default_encoder_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + na*np.prod(self.ny,dtype=int), \
            n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)
        self.m = nn.ReLU()

    def forward(self, upast, ypast):
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1) # type: ignore
        return self.m(self.net(net_in))