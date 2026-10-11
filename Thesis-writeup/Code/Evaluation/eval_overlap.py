"""The two inner products of D-246 (THESIS-RESULTS step 5), on held-out one-step tuples.

Tuples: `build_reference_set` (the OBC construction, D-192) over the test records I1 to I6, so
every tuple is held out; x_a = 0 (the reference slice); normalised coordinates of Section IV;
the rows the network writes (Euclidean inner product, eq. aug_norm). Pi is the orthogonal
projection onto ran Phi, solved by least squares.

  network overlap  ||Pi f_aug|| / ||f_aug||, Phi at the trained physical coordinate. f_aug is the
                   learned correction on the physical rows as the model applies it: the network
                   field, minus the OBC correction Phi(vbar) theta_frozen in an OBC arm.
  truth overlap    ||Pi* eps|| / ||eps||, eps = x~+ - f_base(xi*)(x~, u) (eq. obc_eps), Phi* at the
                   true combinations. Run independent up to the normalisation; computed per run.

Usage: python Thesis-writeup/Code/Evaluation/eval_overlap.py R4 R1 R5 R2
"""
import json
import sys
import time

import numpy as np
import torch

from eval_common import RUNS, RUNS_DIR, TW, record_name
from eval_greybox import run_config, build

INTERP = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']


def overlap(J, v):
    """||Pi v|| / ||v|| with Pi the projection onto ran J (least squares, float64)."""
    c = torch.linalg.lstsq(J, v.unsqueeze(1)).solution
    return float(torch.linalg.norm(J @ c) / torch.linalg.norm(v))


def evaluate(run):
    from gantry_dynamic.training import load_checkpoint
    from gantry_dynamic.closed_loop_report import _true_free
    from gantry_dynamic.obc_gantry import build_reference_set, make_float64_block
    from model_augmentation.fit_systems.obc import build_stacked_sensitivity
    from model_augmentation.fit_systems.blocks import Static_ANN_Block

    rid = RUNS[run]
    t0 = time.time()
    cfg = run_config(RUNS_DIR / str(rid))
    data, norm, fs = build(cfg)
    load_checkpoint(fs, str(RUNS_DIR / str(rid) / ('gantry_ckpt_%d' % rid)), joint_estimation=True,
                    start_phase='lbfgs')
    fs._sync_prediction_state_for_validation(refresh_basis=False)
    blocks = list(fs.hfn.connected_blocks)
    phy = next(b for b in blocks if hasattr(b, 'free_params'))
    ann = next(b for b in blocks if isinstance(b, Static_ANN_Block))
    files = [f for f in data.test_files if record_name(f) in INTERP]
    ref = build_reference_set(cfg, norm, nx_ann=cfg.nx_ann, files=files, verbose=False)

    route = [int(i) for i in cfg.ann_route_ix]
    row_ix = [i for i in route if i < cfg.nx_phys]
    cols = [route.index(i) for i in row_ix]
    blk = make_float64_block(phy)
    step = lambda v, z: blk.transition_from_free(v, z)               # noqa: E731

    corr = getattr(fs.hfn, 'obc_correction', None)
    v_hat = (corr.vbar if corr is not None else phy.free_params).detach().double()
    J = build_stacked_sensitivity(step, v_hat, ref.Z_step, row_ix)
    with torch.no_grad():
        F = ann(ref.Z_field.to(next(ann.parameters()).dtype))[:, cols, 0].double().reshape(-1)
        if corr is not None:
            F = F - J @ corr.theta_frozen.double()
    net = overlap(J, F)
    if corr is not None:
        # check: on the TRAINING reference tuples OBC makes the correction orthogonal by
        # construction, so this overlap must be near zero (validates sign and coordinates)
        tr = fs.obc_reference
        Jt = build_stacked_sensitivity(step, v_hat, tr.Z_step, row_ix)
        with torch.no_grad():
            Ft = ann(tr.Z_field.to(next(ann.parameters()).dtype))[:, cols, 0].double().reshape(-1)
            Ft = Ft - Jt @ corr.theta_frozen.double()
        print('  check: network overlap on the training tuples %.2e (OBC: ~0 by construction)'
              % overlap(Jt, Ft), flush=True)
        del Jt, Ft

    # truth: eps = x~+ - f_base(xi*)(x~, u) on consecutive tuples of the same record
    v_true = _true_free(phy).detach().double()
    Js = build_stacked_sensitivity(step, v_true, ref.Z_step, row_ix)
    same = (ref.record_ix[1:] == ref.record_ix[:-1]) & (ref.sample_ix[1:] == ref.sample_ix[:-1] + 1)
    k = torch.nonzero(same).squeeze(1)
    with torch.no_grad():
        f_true = blk.transition_from_free(v_true, ref.Z_step.double())[:, :, 0]       # normalised
    x_next = ref.Z_step[:, :cfg.nx_phys, 0].double()                                   # x~ normalised
    eps = (x_next[k + 1] - f_true[k])[:, row_ix].reshape(-1)
    rows = torch.stack([k * len(row_ix) + j for j in range(len(row_ix))], 1).reshape(-1)
    tru = overlap(Js[rows], eps)
    print('[%s %d] %s | %d held-out tuples | network overlap %.4f | truth overlap %.4f | %.0f s'
          % (run, rid, 'OBC' if cfg.obc else 'U', ref.n, net, tru, time.time() - t0), flush=True)

    # learned added-state mode: eigenvalues of d x_a+ / d x_a at 24 encoder states of validation
    # windows (inspect_checkpoint.added_poles; eigenvalues do not depend on the x_a coordinates)
    from inspect_checkpoint import added_poles
    dom, share, fr, zeta = added_poles(fs, data, cfg.nx_phys, cfg.nx_ann)
    print('  added-state mode: %.1f Hz, zeta %.4f, complex at %.0f %% of points, dominant |z| %.4f'
          % (fr, zeta, 100 * share, dom), flush=True)
    return dict(run=run, job=rid, arm='obc' if cfg.obc else 'u', n_tuples=ref.n,
                network_overlap=net, truth_overlap=tru, mode_hz=fr, mode_zeta=zeta,
                complex_share=share)


if __name__ == '__main__':
    torch.set_num_threads(4)
    out = [evaluate(r) for r in sys.argv[1:]]
    path = TW / 'Results' / 'res_obc_overlap.csv'
    with open(path, 'w') as fh:
        fh.write('run,job,arm,n_tuples,network_overlap,truth_overlap,mode_hz,mode_zeta,complex_share\n')
        for o in out:
            fh.write('%s,%d,%s,%d,%.6e,%.6e,%.3f,%.5f,%.3f\n' % (
                o['run'], o['job'], o['arm'], o['n_tuples'], o['network_overlap'],
                o['truth_overlap'], o['mode_hz'], o['mode_zeta'], o['complex_share']))
    print('wrote', path)
