from deepSI.system_data.system_data import hist_future_dataset, System_data
import numpy as np

class fixed_System_data(System_data):
    def __init__(self, u=None, y=None, x=None, cheat_n=0, normed=False, dt=None):
        super().__init__(u=u, y=y, cheat_n=cheat_n, normed=normed, dt=dt)
        
    def to_hist_future_data(self, na=10, nb=10, nf=5, na_right = 0, nb_right = 0, stride=1, \
                            force_multi_u=False, force_multi_y=False, online_construct=False):

        u, y = np.copy(self.u), np.copy(self.y) # type: ignore
        yhist = []
        uhist = []
        ufuture = []
        yfuture = []
        k0 = max(nb, na)
        k0_right = max(nf, na_right, nb_right)
        for k in range(k0+k0_right,len(u)+1,stride):
            kmid = k - k0_right
            yhist.append(y[kmid-na:kmid+na_right])
            uhist.append(u[kmid-nb:kmid+nb_right])
            yfuture.append(y[kmid:kmid+nf])
            ufuture.append(u[kmid:kmid+nf])
        uhist, yhist, ufuture, yfuture = np.array(uhist), np.array(yhist), np.array(ufuture), np.array(yfuture)
        
        return uhist, yhist, ufuture, yfuture
    
class fixed_System_data_norm(object):
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
        self.u0 = u0
        self.ustd = ustd
        self.y0 = y0
        self.ystd = ystd

    @property
    def is_id(self):
        return np.all(self.u0==0) and np.all(self.ustd==1) and np.all(self.y0==0) and np.all(self.ystd==1)


    def make_training_data(self,sys_data):
        if isinstance(sys_data,(list,tuple)):
            out = [self.make_training_data(s) for s in sys_data]
            return [np.concatenate(a,axis=0) for a in zip(*out)] #transpose + concatenate
        return sys_data.u, sys_data.y

    def fit(self,sys_data):
        '''set the values of u0, ustd, y0 and ystd using sys_data (can be a list) given'''
        u, y = self.make_training_data(sys_data)
        self.u0 = np.mean(u,axis=0)
        self.ustd = np.std(u,axis=0) + 1e-15 #does this work with is_id?
        self.y0 = np.mean(y,axis=0)
        self.ystd = np.std(y,axis=0) + 1e-15
        
    def transform(self,sys_data):

        if self.is_id:
            return fixed_System_data(u=sys_data.u,x=sys_data.x,y=sys_data.y, \
                               cheat_n=sys_data.cheat_n,normed=True,dt=sys_data.dt)

        if isinstance(sys_data,System_data) or isinstance(sys_data,fixed_System_data):
            assert sys_data.normed==False, 'System_data is already normalized'
            u_transformed = (sys_data.u-self.u0)/self.ustd if sys_data.u is not None else None
            y_transformed = (sys_data.y-self.y0)/self.ystd if sys_data.y is not None else None
            return fixed_System_data(u=u_transformed,x=sys_data.x,y=y_transformed, \
                cheat_n=sys_data.cheat_n,normed=True,dt=sys_data.dt)
        else:
            raise NotImplementedError(f'type={type(sys_data)} cannot yet be transformed by norm')
    
    def inverse_transform(self,sys_data):
        if self.is_id:
            return fixed_System_data(u=sys_data.u,x=sys_data.x,y=sys_data.y, \
                               cheat_n=sys_data.cheat_n,normed=False,dt=sys_data.dt)

        if isinstance(sys_data,fixed_System_data) or isinstance(sys_data,System_data):
            assert sys_data.normed==True, 'System_data is already un-normalized'
            u_inv_transformed = sys_data.u*self.ustd + self.u0 if sys_data.u is not None else None
            y_inv_transformed = sys_data.y*self.ystd + self.y0 if sys_data.y is not None else None
            return fixed_System_data(u=u_inv_transformed,x=sys_data.x,y=y_inv_transformed,
                               cheat_n=sys_data.cheat_n,normed=False,dt=sys_data.dt)
        else:
            raise NotImplementedError(f'type={type(sys_data)} cannot yet be inverse_transform by norm')
        
    def u_transform(self, u):
        return (u - convert_to_same_type(u,self.u0))/convert_to_same_type(u,self.ustd)
    def y_transform(self, y):
        return (y - convert_to_same_type(y,self.y0))/convert_to_same_type(y,self.ystd)
    def u_inv_transform(self, u):
        return u*convert_to_same_type(u,self.ustd) + convert_to_same_type(u,self.u0)
    def y_inv_transform(self, y):
        return y*convert_to_same_type(y,self.ystd) + convert_to_same_type(y,self.y0)

    def __repr__(self):
        return f'System_data_norm: (u0={self.u0}, ustd={self.ustd}, y0={self.y0}, ystd={self.ystd})'
    

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
