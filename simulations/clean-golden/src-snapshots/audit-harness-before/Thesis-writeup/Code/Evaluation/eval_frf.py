"""Open-loop frequency response of the final model against the simulated truth and the baseline.

Result B (THESIS-RESULTS step 3; Dirk 2026-10-15, after Hoekstra et al. 2026 Fig. 10, which compares a
learned component with the true missing element in a Bode magnitude plot).
The trained model is linearised at physical rest (q = (0, 0, Y), qdot = 0, added states 0, zero stage
force) in its normalised coordinates: A = d x+/d x and B = d x+/d u of the full transition (physical
and added states, OBC correction applied as in prediction), output map Cd_norm on the physical rows.
Its response from stage force to stage position is compared with the simulated system of
eq:setup_truth linearised at rest without friction (tdg_truth_lin.m with g = 0) and the rigid baseline
at the true parameters (both from meta.truth, Code/Setup/_setup_common.py).
The network is nonlinear: the response holds at this operating point only.

Writes Results/res_frf_full.csv (frequency, Y, all nine |G_jl| for model, truth, baseline; appendix),
Results/res_frf.csv (the Y -> Y entry only; results figure) and
Results/res_frf_modes.csv (complex modes of the linearised model and of the truth).
Usage: python Thesis-writeup/Code/Evaluation/eval_frf.py [run] [Y ...]   (default R1, Y = -0.39 to 0.39)
"""
import csv
import os
import sys

import numpy as np
import torch

from eval_common import RUNS, RUNS_DIR, TW, checkpoint_status
from eval_greybox import run_config, build

sys.path.insert(0, str(TW / 'Code' / 'Setup'))
import _setup_common as sc                                   # noqa: E402

FS = 4000.0
F_HZ = np.linspace(20.0, 600.0, 1161)                       # HEURISTIC: display band, 0.5 Hz grid


def linearise(fs, norm, cfg, Y):
    """A, B, C of the trained model at physical rest at Y (normalised coordinates)."""
    nxp, nxa = cfg.nx_phys, cfg.nx_ann
    xm, sx = np.asarray(norm.x_mean, float).ravel(), np.asarray(norm.std_x, float).ravel()
    um, su = np.asarray(norm.u_mean, float).ravel(), np.asarray(norm.std_u, float).ravel()
    x0 = np.concatenate([(np.array([0.0, 0.0, Y, 0.0, 0.0, 0.0]) - xm) / sx, np.zeros(nxa)])
    u0 = -um / su                                            # zero physical stage force
    dt = next(fs.hfn.parameters()).dtype
    X0, U0 = torch.tensor(x0, dtype=dt)[None], torch.tensor(u0, dtype=dt)[None]
    # keep the SAVED OBC correction (vbar, theta_frozen are checkpoint buffers); a re-sync would rebuild the basis
    step = lambda x, u: fs.hfn(x, u)[1].reshape(-1)          # noqa: E731  (next state)
    with torch.no_grad():
        drift = float(np.abs(step(X0, U0).double().numpy() - x0).max())
    A, B = torch.autograd.functional.jacobian(step, (X0, U0))
    n = nxp + nxa
    A = A.reshape(n, n).double().numpy()
    B = B.reshape(n, 3).double().numpy()
    C = np.hstack([np.asarray(norm.Cd_norm, float), np.zeros((3, nxa))])
    return A, B, C, drift


def frf_model(A, B, C, ystd, su):
    """G(e^{j w T}) in m/N: ystd C (zI - A)^-1 B / su."""
    # THEORY: frequency response of a discrete-time state-space model, z = exp(j 2 pi f / fs)
    out = np.empty((len(F_HZ), 3, 3), complex)
    I = np.eye(A.shape[0])
    for i, f in enumerate(F_HZ):
        z = np.exp(2j * np.pi * f / FS)
        out[i] = np.diag(ystd) @ (C @ np.linalg.solve(z * I - A, B)) @ np.diag(1.0 / su)
    return out


def modes(eigs, discrete):
    s = np.log(eigs.astype(complex)) * FS if discrete else eigs
    return sorted((abs(x.imag) / (2 * np.pi), -x.real / abs(x)) for x in s if x.imag > 1e-6)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'R1'
    # Dirk 2026-10-15: the model is LPV, so the response is averaged over the full Y range, training
    # positions (+-0.30, +-0.15, 0) and the extrapolation positions of E1, E2 (+-0.39)
    Ys = [float(a) for a in sys.argv[2:]] or [-0.39, -0.30, -0.15, 0.0, 0.15, 0.30, 0.39]
    from gantry_dynamic.training import load_checkpoint
    sys.path.insert(0, str(TW / 'Code' / 'Setup'))
    from loop_memory import plant_ss
    rid = RUNS[run]
    st = checkpoint_status(rid)                    # final, or the best so far of a running run
    cfg = run_config(RUNS_DIR / str(rid))
    data, norm, fs = build(cfg)
    load_checkpoint(fs, st['path'], joint_estimation=True, start_phase='lbfgs')
    print('[%s %d] %s (%s)' % (run, rid, st['label'], st['path']), flush=True)
    T = sc.truth_params(sc.DEFAULT_DATASET)
    ystd = np.asarray(norm.ystd, float).ravel()
    su = np.asarray(norm.std_u, float).ravel()
    rows, mrows = [], []
    for Y in Ys:
        A, B, C, drift = linearise(fs, norm, cfg, Y)
        Gm = frf_model(A, B, C, ystd, su)
        Gt = sc.frf_stage(T, Y, F_HZ, truth=True)
        Gb = sc.frf_stage(T, Y, F_HZ, truth=False)
        for i, f in enumerate(F_HZ):
            rows.append([run, Y, f] + ['%.6e' % abs(G[i, j, l]) for G in (Gm, Gt, Gb)
                                       for j in range(3) for l in range(3)] + [st['label']])
        mm = modes(np.linalg.eigvals(A), True)
        mt = modes(np.linalg.eigvals(plant_ss(T, Y)[0]), False)
        for name, ms in (('model', mm), ('truth', mt)):
            for fr, z in ms:
                mrows.append([run, Y, name, '%.3f' % fr, '%.4f' % z, st['label'] if name == 'model' else ''])
        print('[%s] Y %+.2f  model modes %s  truth modes %s  rest drift %.1e'
              % (run, Y, [(round(a, 1), round(b, 3)) for a, b in mm],
                 [(round(a, 1), round(b, 3)) for a, b in mt], drift))
    # |G_jl|: output axis j (stage position), input axis l (stage force); the 3 x 3 goes to the appendix
    # figure (res_frf_full), the Y -> Y entry to the results figure (res_frf)
    head = ['%s_%s_%s' % (m, a, b) for m in ('model', 'truth', 'base') for a in sc.AXES for b in sc.AXES]
    out = TW / 'Results' / 'res_frf_full.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'Y', 'f_Hz'] + head + ['status'])
        w.writerows(rows)
    print('wrote', out)
    keep = [3 + head.index('%s_Y_Y' % m) for m in ('model', 'truth', 'base')]
    out = TW / 'Results' / 'res_frf.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'Y', 'f_Hz', 'model_Y_Y', 'truth_Y_Y', 'base_Y_Y', 'status'])
        w.writerows([r[:3] + [r[k] for k in keep] + [r[-1]] for r in rows])
    print('wrote', out)
    out = TW / 'Results' / 'res_frf_modes.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'Y', 'system', 'f_Hz', 'zeta', 'status'])
        w.writerows(mrows)
    print('wrote', out)


if __name__ == '__main__':
    main()
