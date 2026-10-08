"""Linear vs tanh dynamic-parallel on Jan's 3-DOF MSD example: does the added mode get learned,
and how reliably? (follow-up of P1: tanh 0 of 6, linear 1 of 4 of his saved models)

Model and training copied from scripts/ecc_2025/msd_ndof_interconnect_dynamic.py (his file is not
touched): 2-DOF Parameterized_MSD_State_Block baseline, Static_ANN_Block (2 x 8, zero-init output
layer) writing all 6 rows from [x, u], identity (linear) or tanh activation, SSE_Interconnect,
nf 200, batch 2000, Adam at deepSI's default lr (optimizer_kwargs not passed), auto_fit_norm, noise
at the chosen SNR. Seeds fixed (Jan's script had none). Trained in chunks; after each chunk the LAST
state is reloaded (fit() restores the best one at its end) and the one-step map is linearised at
100 encoder states: the third mode (4.7 to 12.5 Hz; truth 5.79 Hz, zeta 0.091) is reported.

Usage: python q_msd_linear_vs_tanh.py KIND SEED EPOCHS CHUNK [SNR] [FP]   (KIND linear | tanh)
"""
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)                                                   # Jan's paths are cwd-relative
import deepSI                                                    # noqa: E402
from scipy.io import loadmat                                     # noqa: E402
from model_augmentation.utils.utils import *                     # noqa: E402,F401,F403
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn   # noqa: E402
from model_augmentation.fit_systems.interconnect import *        # noqa: E402,F401,F403
from model_augmentation.fit_systems.blocks import *              # noqa: E402,F401,F403
from model_augmentation.systems.mass_spring_damper import *      # noqa: E402,F401,F403

MSD = os.path.join(REPO, 'data', 'mass_spring_damper')
SIGMA = {20: 15e-3, 30: 52e-4, 60: 15e-5}                        # Jan's values
# HEURISTIC: validate once per chunk instead of every epoch (speed); fit's end-of-call validation
# still runs, so best-model selection works at chunk granularity only. Env Q_VAL_EPOCH=1 restores Jan's.
VAL_PER_CHUNK = os.environ.get('Q_VAL_EPOCH') != '1'
CUDA = os.environ.get('Q_CUDA') == '1' and torch.cuda.is_available()


def build(kind, snr, fp, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    train = deepSI.load_system_data(os.path.join(MSD, 'msd_3dof_multisine_train.npz'))
    val = deepSI.load_system_data(os.path.join(MSD, 'msd_3dof_multisine_val.npz'))
    train.y = train.y + np.random.normal(0, SIGMA[snr], train.y.shape)
    val.y = val.y + np.random.normal(0, SIGMA[snr], val.y.shape)
    mat = loadmat(os.path.join(MSD, 'msd_2dof.mat' if fp == 'ideal' else 'msd_2dof_non_ideal.mat'), squeeze_me=False)
    nu, ny = mat['nu'][0, 0], mat['ny'][0, 0]
    _, _, C, D = normalize_linear_ss_matrices(mat['Ad'], mat['Bd'], mat['Cd'], mat['Dd'], train, state_ix=np.array([0, 1, 2, 3]))
    nxd = 6
    ic = Interconnect(nxd, nu, ny, debugging=False)
    phy = Parameterized_MSD_State_Block(nz=5, nw=4, FP_type=fp)
    out = Linear_Output_Block(C=C, D=D)
    ic.add_block(phy); ic.add_block(out)
    ann = Static_ANN_Block(nz=nxd + nu, nw=nxd, n_nodes_per_layer=8, net=zero_init_feed_forward_nn,
                           activation=torch.nn.Identity if kind == 'linear' else torch.nn.Tanh)
    ic.add_block(ann)
    ic.connect_block_signals(ann, ["x", "u"], ["xp"])
    ic.connect_signals("x", phy, "concat", selection_matrix(np.array([0, 1, 2, 3]), nxd))
    ic.connect_block_signals(phy, ["u"], [])
    ic.connect_signals(phy, "xp", "additive", expansion_matrix(np.array([0, 1, 2, 3]), nxd))
    ic.connect_signals("x", out, "concat", selection_matrix(np.array([0, 1, 2, 3]), nxd))
    ic.connect_block_signals(out, ["u"], ["y"])
    fs = SSE_Interconnect(interconnect=ic, na=nxd * 2 + 1, nb=nxd * 2 + 1, e_net_kwargs={"n_nodes_per_layer": 16})
    return fs, train, val


def third_mode(fs, train, ts, npts=100):
    sdn = fs.norm.transform(train)
    uh, yh, uf, _ = sdn.to_hist_future_data(na=fs.na, nb=fs.nb, nf=1)[:4]
    pick = np.linspace(0, len(uf) - 1, npts).round().astype(int)
    p = next(fs.hfn.parameters())
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a[pick]), dtype=p.dtype)      # noqa: E731
    with torch.no_grad():
        x = fs.encoder(T(uh), T(yh))
    u = T(uf)[:, 0].reshape(npts, -1)
    f = lambda xx: fs.hfn(xx, u)[1].reshape(npts, -1).sum(0)                         # noqa: E731
    J = torch.autograd.functional.jacobian(f, x.clone()).permute(1, 0, 2).detach().numpy()
    res = []
    for j in J:
        ev = np.linalg.eigvals(j); ev = ev[np.imag(ev) > 1e-9]
        s = np.log(ev) / ts
        fz = [(abs(q) / (2 * np.pi), -np.real(q) / abs(q)) for q in s]
        m = [v for v in fz if 4.7 <= v[0] < 12.5]
        res.append(m[0] if m else (np.nan, np.nan))
    res = np.array(res)
    ok = np.isfinite(res[:, 0])
    ev = np.linalg.eigvals(J.mean(0)); ev = ev[np.imag(ev) > 1e-9]
    mj = sorted((abs(q) / (2 * np.pi), -np.real(q) / abs(q)) for q in np.log(ev) / ts)
    return ok.mean(), (np.median(res[ok, 0]) if ok.any() else np.nan), (np.median(res[ok, 1]) if ok.any() else np.nan), mj


def main(kind, seed, epochs, chunk, snr, fp):
    ts = float(loadmat(os.path.join(MSD, 'msd_3dof.mat'))['Ts'][0, 0])
    fs, train, val = build(kind, snr, fp, seed)
    # unique checkpoint file per run: parallel array tasks must not share '<name>_last.pth'
    fs.unique_code = 'msd_%s_s%d_snr%d_%s_%d' % (kind, seed, snr, fp, os.getpid())   # name = class + unique_code
    print('######## MSD %s seed %d, %d epochs in chunks of %d, SNR%d %s (truth 3rd mode 5.79 Hz zeta 0.091)' % (
        kind, seed, epochs, chunk, snr, fp), flush=True)
    print('  device: %s' % ('cuda' if CUDA else 'cpu'), flush=True)
    t0 = time.time(); done = 0
    while done < epochs:
        n = min(chunk, epochs - done)
        fs.fit(train_sys_data=train, val_sys_data=val, batch_size=2000, epochs=n, auto_fit_norm=True,
               loss_kwargs={'nf': 200}, validation_measure="sim-RMS", verbose=0,
               its_per_val=10 ** 9 if VAL_PER_CHUNK else 'epoch', cuda=CUDA)
        done += n
        best_val = float(np.min(fs.Loss_val)) if len(fs.Loss_val) else np.nan
        fs.checkpoint_load_system(name='_last')                  # continue the trajectory, not the best
        fs.cpu()
        frac, f3, z3, mj = third_mode(fs, train, ts)
        print('  epoch %5d (%.0f s): best val sim-RMS %.4e | 3rd pair at %3.0f %% of points, median %.2f Hz zeta %.3f | mean-J modes %s'
              % (done, time.time() - t0, best_val, 100 * frac, f3, z3,
                 ' '.join('%.2fHz/z%.3f' % m for m in mj)), flush=True)


if __name__ == '__main__':
    a = sys.argv
    main(a[1], int(a[2]), int(a[3]), int(a[4]), int(a[5]) if len(a) > 5 else 20, a[6] if len(a) > 6 else 'approximate')
