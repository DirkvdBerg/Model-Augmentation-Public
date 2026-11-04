"""Jan's dtype-preserving training windows and record normalisation, restricted to the thesis path."""
# CHANGED: renamed fixed_System_data and fixed_System_data_norm to WindowedSystemData and SystemDataNormalizer.
# Original: Model-Augmentation-Public-main/model_augmentation/utils/deepSI_corrections.py

from deepSI.system_data.system_data import System_data, System_data_list
import numpy as np

class WindowedSystemData(System_data):
    """Jan's window builder preserves array dtype; deepSI's stride-1 windowing casts to float32."""
    def __init__(self, u=None, y=None, x=None, cheat_n=0, normed=False, dt=None):
        """Store inputs and outputs; x is accepted for deepSI compatibility but not stored.

        normed marks normalised data; dt is the sample time [s].
        """
        super().__init__(u=u, y=y, cheat_n=cheat_n, normed=normed, dt=dt)

    def to_hist_future_data(self, na=10, nb=10, nf=5, na_right = 0, nb_right = 0, stride=1, \
                            force_multi_u=False, force_multi_y=False, online_construct=False):
        """Split a record into histories and prediction windows, preserving array dtype.

        kmid starts each prediction window [kmid:kmid + nf]. Histories contain samples before kmid;
        na_right and nb_right optionally include samples from kmid onward.

        Args:
            na, nb: past output and input samples per window.
            nf: samples per prediction window.
            stride: samples between window starts.
            na_right, nb_right: output and input samples from kmid onward included in the histories (0 here).
            force_multi_u, force_multi_y, online_construct: unused; retained for deepSI compatibility.
        Returns:
            For multi-channel records with N > 0 windows:
            uhist (N, nb + nb_right, nu), yhist (N, na + na_right, ny),
            ufuture (N, nf, nu), yfuture (N, nf, ny).
            For a single signal stored as a 1D array, there is no separate channel dimension.
            If no window fits, all four arrays have shape (0,).
        """

        u, y = np.copy(self.u), np.copy(self.y)
        yhist = []
        uhist = []
        ufuture = []
        yfuture = []
        # Required samples before the window start and from that start onward.
        k0 = max(nb, na)
        k0_right = max(nf, na_right, nb_right)
        for k in range(k0+k0_right,len(u)+1,stride):
            # k is one past the last sample needed; kmid starts the prediction window.
            kmid = k - k0_right
            yhist.append(y[kmid-na:kmid+na_right])
            uhist.append(u[kmid-nb:kmid+nb_right])
            yfuture.append(y[kmid:kmid+nf])
            ufuture.append(u[kmid:kmid+nf])
        uhist, yhist, ufuture, yfuture = np.array(uhist), np.array(yhist), np.array(ufuture), np.array(yfuture)

        return uhist, yhist, ufuture, yfuture

class SystemDataNormalizer(object):
    """Normalise records using supplied means and scales; no fitting or inverse transform.

    u^n = (u - u0) / ustd, y^n = (y - y0) / ystd; model.py sets the training means and standard deviations.
    """
    # CHANGED: removed fit, make_training_data, inverse and per-signal transforms, __repr__ and their type helper;
    # these are unused by the thesis training and simulation code.
    def __init__(self, u0=0, ustd=1, y0=0, ystd=1):
        """Store input/output means and scales; defaults give identity normalisation."""
        self.u0 = u0
        self.ustd = ustd
        self.y0 = y0
        self.ystd = ystd

    def transform(self,sys_data):
        """Normalise each record with the stored means and scales, keeping record lengths and signal shapes.

        Args:
            sys_data: unnormalised System_data, or a list, tuple or System_data_list of records.
        Returns:
            normalised WindowedSystemData, or a System_data_list for multiple records.
        """
        # CHANGED: also accept multiple records; normalise each separately
        if isinstance(sys_data, (list, tuple, System_data_list)):
            return System_data_list([self.transform(s) for s in sys_data])
        # CHANGED: name the accepted record types in the error.
        if not isinstance(sys_data, System_data):
            raise NotImplementedError(f'type={type(sys_data)} cannot be transformed by norm; pass System_data '
                                      f'or a System_data_list')

        # CHANGED: removed is_id; always check that the record is unnormalised, even with zero means and unit scales.
        assert sys_data.normed==False, 'System_data is already normalized; pass the original unnormalised data'
        u_transformed = (sys_data.u-self.u0)/self.ustd if sys_data.u is not None else None
        y_transformed = (sys_data.y-self.y0)/self.ystd if sys_data.y is not None else None
        return WindowedSystemData(u=u_transformed,x=sys_data.x,y=y_transformed, \
            cheat_n=sys_data.cheat_n,normed=True,dt=sys_data.dt)
