from torch import nn, Tensor
import torch
import numpy as np

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
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1)
        return self.m(self.net(net_in))