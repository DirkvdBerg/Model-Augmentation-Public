from deepSI.fit_systems.encoders import SS_encoder_general, default_encoder_net, hf_net_default, default_state_net, default_output_net
from deepSI.system_data import System_data, System_data_norm
from torch import nn, Tensor
import torch
import numpy as np
import warnings

def convert_to_same_type(base, target):
    #converts target to the same type as base
    import torch
    if isinstance(base, (float,int)):
        return target
    elif isinstance(base, np.ndarray):
        return np.array(target,dtype=base.dtype)
    elif isinstance(base, torch.Tensor):
        return torch.as_tensor(target, dtype=base.dtype)
    else:
        raise NotImplementedError(f'cannot convert to type {base}')

class System_data_with_x(System_data):
    def __init__(self, u=None, y=None, x=None, cheat_n=0, normed=False, dt=None):
        super(System_data_with_x, self).__init__(u=u, y=y, x=x, cheat_n=cheat_n, normed=normed, dt=dt)

    def to_hist_future_data(self, na=10, nb=10, nf=5, na_right = 0, nb_right = 0, stride=1, \
                            force_multi_u=False, force_multi_y=False, online_construct=False):


        assert stride==1, "stride > 1 not implemented yet"
        u, y, x = self.u.astype(np.float32, copy=False), self.y.astype(np.float32, copy=False), self.x.astype(np.float32, copy=False)
        def window(x,window_shape=nf): 
            x = np.lib.stride_tricks.sliding_window_view(x, window_shape=window_shape,axis=0, writeable=True)
            s = (0,len(x.shape)-1) + tuple(range(1,len(x.shape)-1))
            return x.transpose(s)
        npast = max(na, nb)
        ufuture = window(u[npast:len(u)], window_shape=nf)
        yfuture = window(y[npast:len(y)], window_shape=nf)
        xfuture = window(x[npast:len(y)], window_shape=nf)
        uhist = window(u[npast-nb:len(u)-nf], window_shape=nb)
        yhist = window(y[npast-na:len(y)-nf], window_shape=na)

        return uhist, yhist, ufuture, yfuture, xfuture

class System_data_norm_with_x(System_data_norm):
    '''A utility to normalize system_data before fitting or usage

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
    '''
    def __init__(self, u0=0, ustd=1, y0=0, ystd=1):
        super(System_data_norm_with_x, self).__init__(u0=0, ustd=1, y0=0, ystd=1)
        
    def transform(self,sys_data):
        '''Transform the data by 
           u <- (sys_data.u-self.u0)/self.ustd
           y <- (sys_data.y-self.y0)/self.ystd

        Parameters
        ----------
        sys_data : System_data
            sys_data to be transformed

        Returns
        -------
        System_data or System_data_list if a list was given
        '''
        if isinstance(sys_data,System_data_with_x):
            assert sys_data.normed==False, 'System_data is already normalized'
            u_transformed = (sys_data.u-self.u0)/self.ustd if sys_data.u is not None else None
            y_transformed = (sys_data.y-self.y0)/self.ystd if sys_data.y is not None else None
            return System_data_with_x(u=u_transformed,x=sys_data.x,y=y_transformed, \
                cheat_n=sys_data.cheat_n,normed=True,dt=sys_data.dt)
        else:
            raise NotImplementedError(f'type={type(sys_data)} cannot yet be transformed by norm')

    def inverse_transform(self,sys_data):
        '''Inverse Transform the data by 
           u <- sys_data.u*self.ustd+self.u0
           y <- sys_data.y*self.ystd+self.y0

        Parameters
        ----------
        sys_data : System_data

        Returns
        -------
        System_data or System_data_list if a list was given
        '''

        if isinstance(sys_data,System_data_with_x):
            assert sys_data.normed==True, 'System_data is already un-normalized'
            u_inv_transformed = sys_data.u*self.ustd + self.u0 if sys_data.u is not None else None
            y_inv_transformed = sys_data.y*self.ystd + self.y0 if sys_data.y is not None else None
            return System_data_with_x(u=u_inv_transformed,x=sys_data.x,y=y_inv_transformed,
                               cheat_n=sys_data.cheat_n,normed=False,dt=sys_data.dt)
        else:
            raise NotImplementedError(f'type={type(sys_data)} cannot yet be inverse_transform by norm')

class SS_pre_encoder(SS_encoder_general):
    """The encoder function with combined h and f functions
    
    the hf_net_default has the arguments
       hf_net_default(nx, nu, ny, feedthrough=False, **hf_net_kwargs)
    and is used as 
       ynow, xnext = hfn(x,u)
    """
    def __init__(self, nx=10, na=20, nb=20, feedthrough=False, \
                 hf_net=hf_net_default, \
                 hf_net_kwargs = dict(f_net=default_state_net, f_net_kwargs={}, h_net_kwargs={}, h_net=default_output_net), \
                 e_net=default_encoder_net,   e_net_kwargs={}, na_right=0, nb_right=0):

        super(SS_pre_encoder, self).__init__(nx=nx, nb=nb, na=na, na_right=na_right, nb_right=nb_right)
        
        self.e_net = e_net
        self.e_net_kwargs = e_net_kwargs
        
        self.hf_net = hf_net
        hf_net_kwargs['feedthrough'] = feedthrough
        self.hf_net_kwargs = hf_net_kwargs

        self.feedthrough = feedthrough

        self.norm = System_data_norm_with_x()

    def init_nets(self, nu, ny): # a bit weird
        na_right = self.na_right if hasattr(self,'na_right') else 0
        nb_right = self.nb_right if hasattr(self,'nb_right') else 0
        self.encoder = self.e_net(nb=self.nb+nb_right, nu=nu, na=self.na+na_right, ny=ny, nx=self.nx, **self.e_net_kwargs)
        self.hfn = self.hf_net(nx=self.nx, nu=self.nu, ny=self.ny, **self.hf_net_kwargs)

    def loss(self, uhist, yhist, ufuture, yfuture, xfuture, **Loss_kwargs):
        xhat = self.encoder(uhist, yhist) #initialize Nbatch number of states
        errors = []
        i = 0
        for x in torch.transpose(xfuture,0,1): #iterate over time
            # yhat, x = self.hfn(x, u)
            errors.append(nn.functional.mse_loss(x, xhat)) #calculate error after taking n-steps
            i = i+1
            assert i == 1, "This should not be used for nf>1"
        return torch.mean(torch.stack(errors))
    
    def n_step_error(self,sys_data,nf=100,stride=1,mode='NRMS',mean_channels=True):
        norm = System_data_norm_with_x()
        if isinstance(mode, tuple):
            norm, error_mode = mode
        elif isinstance(mode,System_data_norm_with_x):
            norm, error_mode = mode, 'RMS'
        else:
            #figure out the error mode
            if 'RMS' in mode:
                error_mode = 'RMS'
            elif 'MSE' in mode:
                error_mode = 'MSE'
            elif 'MAE' in mode:
                error_mode = 'MAE'
            else:
                raise NotImplementedError(f'mode {mode} should has one of RMS MSE MAE')
            
            if '_sys_norm' in mode:
                norm = self.norm
            elif mode[0]=='N':
                norm.fit(sys_data)

        sys_data = self.norm.transform(sys_data)
        k0 = self.init_state_multi(sys_data, nf=nf, stride=stride)

        _, _, ufuture, yfuture, xfuture = sys_data.to_hist_future_data(na=k0, nb=k0, nf=nf, stride=stride)

        Losses = []
        for xnow in np.swapaxes(xfuture,0,1):
            res = (convert_to_same_type(xnow,self.state) -  xnow)

            Losses = np.mean(res**2, axis=0)**0.5

            print(Losses.shape)

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