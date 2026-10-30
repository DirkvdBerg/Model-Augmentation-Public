"""deepSI data classes whose training windows keep the data's dtype.

Conventions: superscript n = normalised; u0, ustd, y0, ystd are mu_u, sigma_u, mu_y, sigma_y.
"""
# CHANGED: Jan's fixed_System_data and fixed_System_data_norm, renamed WindowedSystemData and SystemDataNormalizer;
# restricted to the thesis training path; full implementation in model_augmentation/utils/deepSI_corrections.py at
# the repository root

from deepSI.system_data.system_data import System_data, System_data_list
import numpy as np

class WindowedSystemData(System_data):
    """System_data whose training windows keep the data's dtype: deepSI's stride-1 path casts them to float32."""
    def __init__(self, u=None, y=None, x=None, cheat_n=0, normed=False, dt=None):
        """deepSI's System_data."""
        super().__init__(u=u, y=y, cheat_n=cheat_n, normed=normed, dt=dt)

    def to_hist_future_data(self, na=10, nb=10, nf=5, na_right = 0, nb_right = 0, stride=1, \
                            force_multi_u=False, force_multi_y=False, online_construct=False):
        """Training windows of a record.

        chi_tau = (y[tau - na:tau], u[tau - nb:tau]) and the futures [tau:tau + nf], for window ends tau + nf at the
        stride (deepSI's loop, for every stride).

        Args:
            na, nb: history lengths (n + 1); nf: n_f; stride: window stride; other arguments as deepSI's.
        Returns:
            uhist, yhist, ufuture, yfuture, each (n_windows, length, 3).
        """

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

class SystemDataNormalizer(object):
    """deepSI's normalisation u^n = (u - u0) / ustd, y^n = (y - y0) / ystd, returning WindowedSystemData."""
    def __init__(self, u0=0, ustd=1, y0=0, ystd=1):
        """Store mu_u, sigma_u, mu_y, sigma_y as u0, ustd, y0, ystd."""
        self.u0 = u0
        self.ustd = ustd
        self.y0 = y0
        self.ystd = ystd

    def transform(self,sys_data):
        """Normalisation of the records.

        u^n = (u - mu_u) / sigma_u, y^n = (y - mu_y) / sigma_y, record lengths and signal shapes unchanged.

        Args:
            sys_data: unnormalised System_data, or a list of them.
        Returns:
            WindowedSystemData, or a System_data_list of them, normalised.
        """
        if isinstance(sys_data, (list, tuple, System_data_list)):
            return System_data_list([self.transform(s) for s in sys_data])
        # CHANGED: lists handled record by record; errors name the expected input
        if not isinstance(sys_data, System_data):
            raise NotImplementedError(f'type={type(sys_data)} cannot be transformed by norm; pass System_data '
                                      f'or a System_data_list')

        assert sys_data.normed==False, 'System_data is already normalized; pass the original unnormalised data'
        # Normalise: u^n = (u - u0) / ustd, y^n = (y - y0) / ystd.
        u_transformed = (sys_data.u-self.u0)/self.ustd if sys_data.u is not None else None
        y_transformed = (sys_data.y-self.y0)/self.ystd if sys_data.y is not None else None
        return WindowedSystemData(u=u_transformed,x=sys_data.x,y=y_transformed, \
            cheat_n=sys_data.cheat_n,normed=True,dt=sys_data.dt)
