from torch import nn, Tensor
import torch
import numpy as np

from model_augmentation.utils.utils import added


@added
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
        
@added
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
    
@added
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
    
@added
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


@added
class LinearInitEncoderWrapper(nn.Module):
    """Wraps linear_encoder_init (physical states) + zero-init ANN (augmented states).

    linear_encoder_init outputs nx_phys states initialized from the baseline
    model's reconstructability map (Hoekstra 2026, Eq. 16-17). All weights are
    trainable (nn.Parameter). The ANN adds NX_ANN zero-initialized augmented
    states, matching the HybridGantryEncoder structure but with full trainability
    on the physical states.

    THEORY: Encoder architecture is the ResNet from Hoekstra 2026 Eq. 8:
        x_k = W_psi_y @ y_hist + W_psi_u @ u_hist + psi_tilde(y_hist, u_hist)
    where W matrices are initialized from reconstructability theory and
    psi_tilde is a zero-init ANN for nonlinear corrections.

    Convention fix: normalize_linear_ss_matrices produces Wb_psi_y/Wb_psi_u
    for pure-scaled data (x/std_x, u/std_u, y/std_y). But the pipeline feeds
    mean-subtracted data ((u-u_mean)/std_u, (y-y0)/std_y). This wrapper
    converts pipeline -> pure-scaled before the linear map, and pure-scaled ->
    pipeline after, using registered buffers (constant, no gradients).
    """
    def __init__(self, phys_encoder, nx_ann, nb, nu, na, ny,
                 n_nodes_per_layer=16, n_hidden_layers=2, activation=nn.Tanh,
                 u_mean=None, std_u=None, y0=None, ystd=None,
                 x_mean=None, std_x=None):
        super().__init__()
        self.phys_encoder = phys_encoder  # linear_encoder_init instance
        self.nx_ann = nx_ann

        # --- Convention-fix buffers ---
        # When provided, forward() undoes mean subtraction before the linear map
        # and converts the output to pipeline convention.
        if u_mean is not None and std_u is not None:
            # Offset tiled over history: (nb * nu,)
            u_off = torch.tensor(
                (np.asarray(u_mean).flatten() / np.asarray(std_u).flatten()),
                dtype=torch.float32).repeat(nb)
            self.register_buffer('u_offset', u_off)
        else:
            self.u_offset = None

        if y0 is not None and ystd is not None:
            y_off = torch.tensor(
                (np.asarray(y0).flatten() / np.asarray(ystd).flatten()),
                dtype=torch.float32).repeat(na)
            self.register_buffer('y_offset', y_off)
        else:
            self.y_offset = None

        if x_mean is not None and std_x is not None:
            x_off = torch.tensor(
                (np.asarray(x_mean).flatten() / np.asarray(std_x).flatten()),
                dtype=torch.float32)
            self.register_buffer('x_offset', x_off)
        else:
            self.x_offset = None

        if nx_ann > 0:
            nu_tup = (nu,) if isinstance(nu, int) else nu
            ny_tup = (ny,) if isinstance(ny, int) else ny
            n_in = nb * np.prod(nu_tup, dtype=int) + na * np.prod(ny_tup, dtype=int)
            self.ann = zero_init_feed_forward_nn(
                n_in=n_in, n_out=nx_ann,
                n_nodes_per_layer=n_nodes_per_layer,
                n_hidden_layers=n_hidden_layers,
                activation=activation,
            )
        else:
            self.ann = None

    def forward(self, upast, ypast):
        # Flatten to 2D: linear_encoder_init expects (batch, flat) not (batch, T, ch)
        u_flat = upast.reshape(upast.shape[0], -1)
        y_flat = ypast.reshape(ypast.shape[0], -1)

        # Convention fix: pipeline (mean-sub) -> pure-scaled (what Wb matrices expect)
        if self.u_offset is not None:
            u_flat = u_flat + self.u_offset
        if self.y_offset is not None:
            y_flat = y_flat + self.y_offset

        x_phys = self.phys_encoder(u_flat, y_flat)  # (batch, nx_phys), pure-scaled

        # Convention fix: pure-scaled -> pipeline (what the rest of the model expects)
        if self.x_offset is not None:
            x_phys = x_phys - self.x_offset

        if self.ann is not None:
            # ANN sees original pipeline-convention data for nonlinear corrections
            u_orig = upast.reshape(upast.shape[0], -1)
            y_orig = ypast.reshape(ypast.shape[0], -1)
            net_in = torch.cat([u_orig, y_orig], dim=1)
            x_ann = self.ann(net_in)  # (batch, nx_ann)
            return torch.cat([x_phys, x_ann], dim=1)
        return x_phys


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