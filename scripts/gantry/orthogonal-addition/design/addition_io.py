"""Convert a designed member into the gantrySystemOA parameters (oa_Ma0 ... oa_b1) and back into
the force field on reference tuples, so the MATLAB truth and the Python screen use ONE definition.
"""
__project_origin__ = "added"

import numpy as np

NS = 4


def empty():
    Z = np.zeros((3, 3))
    return dict(oa_Ma0=Z.copy(), oa_Ma1=Z.copy(), oa_Ma2=Z.copy(), oa_Ca0=Z.copy(), oa_Ka0=Z.copy(),
                oa_Ka1=Z.copy(), oa_Ka2=Z.copy(), oa_m=np.zeros((1, NS)), oa_w=np.ones((1, NS)),
                oa_b0=np.zeros((3, NS)), oa_b1=np.zeros((3, NS)))


def add_slot(oa, slot, c):
    """Slot (mat, i, j, p) with coefficient c: matrix entry (i, j) and (j, i) of Y^p term += c."""
    m, i, j, p = slot
    key = {('M', 0): 'oa_Ma0', ('M', 1): 'oa_Ma1', ('M', 2): 'oa_Ma2', ('C', 0): 'oa_Ca0', ('K', 0): 'oa_Ka0',
           ('K', 1): 'oa_Ka1', ('K', 2): 'oa_Ka2'}[(m, p)]
    oa[key][i, j] += c
    if i != j:
        oa[key][j, i] += c


def add_physical(oa, name, c, mh, d):
    if name == 'k_x':
        oa['oa_Ka0'][0, 0] += c
    elif name == 'k_y':
        oa['oa_Ka0'][2, 2] += c
    elif name == 'k_xp':
        oa['oa_Ka0'][0, 0] += c
        oa['oa_Ka1'][0, 1] += -c
        oa['oa_Ka1'][1, 0] += -c
        oa['oa_Ka2'][1, 1] += c
    elif name == 'gamma':
        oa['oa_Ma0'][0, 2] += mh * c
        oa['oa_Ma0'][2, 0] += mh * c
        oa['oa_Ma1'][1, 1] += -2.0 * mh * d * c
    else:
        raise KeyError(name)


def add_absorber(oa, slot, mass, f_hz, alpha, lever):
    oa['oa_m'][0, slot] = mass
    oa['oa_w'][0, slot] = 2.0 * np.pi * f_hz
    oa['oa_b0'][:, slot] = [np.cos(alpha), lever, np.sin(alpha)]
    oa['oa_b1'][:, slot] = [0.0, -np.cos(alpha), 0.0]


def force_field(pb, oa):
    """The addition's generalised force on the tuples of `pb`, from the oa_* arrays (screen model):
    f = -(M_a(Y) qdd + C_a qd + K_a(Y) q) - SUM m_i b_i(Y) d_i''."""
    Y = pb.Y[:, None, None]
    Ma = oa['oa_Ma0'][None] + Y * oa['oa_Ma1'][None] + Y ** 2 * oa['oa_Ma2'][None]
    Ka = oa['oa_Ka0'][None] + Y * oa['oa_Ka1'][None] + Y ** 2 * oa['oa_Ka2'][None]
    f = -(np.einsum('kij,kj->ki', Ma, pb.qdd) + pb.qd @ oa['oa_Ca0'].T
          + np.einsum('kij,kj->ki', Ka, pb.q))
    for i in range(NS):
        m = oa['oa_m'][0, i]
        if m != 0.0:
            f = f + m * pb.absorber_force(oa['oa_w'][0, i] / (2 * np.pi), oa['oa_b0'][:, i],
                                          oa['oa_b1'][:, i])
    return f


def save_mat(path, oa, meta):
    from scipy.io import savemat
    d = {k: np.asarray(v, dtype=np.float64) for k, v in oa.items()}
    d['meta'] = meta
    savemat(path, d)


def load_mat(path):
    from scipy.io import loadmat
    L = loadmat(path)
    oa = empty()
    for k in oa:
        if k in L:                       # files written before OA-018 carry no oa_Ma2: zero
            oa[k] = np.asarray(L[k], dtype=np.float64).reshape(oa[k].shape)
    return oa
