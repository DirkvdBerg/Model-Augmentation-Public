__project_origin__ = "added"

from deepSI.system_data.system_data import System_data
import numpy as np

class fixed_System_data(System_data):
    def __init__(self, u=None, y=None, x=None, cheat_n=0, normed=False, dt=None):
        super().__init__(u=u, y=y, cheat_n=cheat_n, normed=normed, dt=dt)

    def to_hist_future_data(self, na=10, nb=10, nf=5, na_right = 0, nb_right = 0, stride=1, \
                            force_multi_u=False, force_multi_y=False, online_construct=False):

        u, y = np.copy(self.u), np.copy(self.y)
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
    '''A utility to normalize system_data before fitting or usage'''
    def __init__(self, u0=0, ustd=1, y0=0, ystd=1):
        self.u0 = u0
        self.ustd = ustd
        self.y0 = y0
        self.ystd = ystd

    @property
    def is_id(self):
        return np.all(self.u0==0) and np.all(self.ustd==1) and np.all(self.y0==0) and np.all(self.ystd==1)


    def transform(self,sys_data):
        from deepSI.system_data.system_data import System_data_list
        if isinstance(sys_data, (list, tuple, System_data_list)):
            return System_data_list([self.transform(s) for s in sys_data])

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
