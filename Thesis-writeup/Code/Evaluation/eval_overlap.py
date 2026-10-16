"""The two inner products of D-246 and the predicted first-order bias (THESIS-RESULTS step 5).

Tuples: `build_reference_set` (the OBC construction, D-192) over three stacks (D-268): the training
records (where theta_aux is formed), the validation records and the test records I1 to I6; x_a = 0
(the reference slice); normalised coordinates of Section IV; the rows the network writes (Euclidean
inner product, eq. aug_norm). Pi is the orthogonal projection onto ran Phi, solved by least squares.

  network overlap  ||Pi f_aug|| / ||f_aug||, Phi at the trained physical coordinate. f_aug is the
                   learned correction on the physical rows as the model applies it: the network
                   field minus the deployed OBC correction, [Phi] theta (tangent) or
                   [Phi F_base] theta (affine, D-257), in an OBC arm. Zero by construction on the
                   training stack of an OBC run (printed as a check).
  truth overlap    ||Pi* Xi|| / ||Xi||, Xi = x~+ - f_base(xi*)(x~, u) (eq. obc_eps), Phi* at the
                   true combinations, x~+ the stored reconstructed successor of every tuple (D-252;
                   runs before it: the next tuple of the record), so Xi stacks the coefficient's tuples.
  predicted bias   eq:obc_recovery, first line: dbeta = (Phi*^T Phi*)^-1 Phi*^T Xi in the free
                   coordinates, mapped to the ten combinations by the block's own map (first order)
                   and scaled per combination as the combination error (D-259: m_Delta by the mean
                   actuator mass). The TRAINING stack is the prediction (the one-step least-squares
                   minimiser at x_a = 0 on the stack where theta_aux is formed); validation and test
                   show how it carries over. Written beside the observed combination error.

Usage: python Thesis-writeup/Code/Evaluation/eval_overlap.py R4 R1 R5 R2
Output: Results/res_obc_overlap.csv (per run and stack) and Results/res_obc_bias.csv (per run,
stack and combination).
"""
import sys
import time

import numpy as np
import torch

from eval_common import RUNS, RUNS_DIR, TW, record_name
from eval_greybox import run_config, build

INTERP = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']
STACKS = ('train', 'val', 'test')


def overlap(J, v):
    """||Pi v|| / ||v|| with Pi the projection onto ran J (least squares, float64)."""
    c = torch.linalg.lstsq(J, v.unsqueeze(1)).solution
    return float(torch.linalg.norm(J @ c) / torch.linalg.norm(v))


def deployed_basis(blk, ref, row_ix, cfg, corr, v_hat):
    """The columns the OBC correction is built from, exactly as in training (make_basis_builder):
    [J] for the tangent space, [J F_base(vbar)] for the affine one (D-257)."""
    from gantry_dynamic.obc_gantry import make_basis_builder
    space = getattr(corr, 'space', 'tangent')
    B, _ = make_basis_builder(blk, ref, row_ix, cfg.obc_ref_chunk, device=torch.device('cpu'),
                              space=space)(v_hat)
    return B


def combination_scale(phy, c_true):
    """Per-combination error scale (D-259): |true value|, m_Delta by the mean actuator mass."""
    scale = c_true.abs().clone()
    names = list(phy.PARAM_NAMES)
    raw = phy.params_init.double()
    scale[list(phy.COMBO_NAMES).index('m_diff')] = 0.5 * (raw[names.index('m1')] + raw[names.index('m2')])
    return scale


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
    # keep the SAVED OBC correction (vbar, theta_frozen are checkpoint buffers); a re-sync would rebuild the basis
    blocks = list(fs.hfn.connected_blocks)
    phy = next(b for b in blocks if hasattr(b, 'free_params'))
    ann = next(b for b in blocks if isinstance(b, Static_ANN_Block))
    route = [int(i) for i in cfg.ann_route_ix]
    row_ix = [i for i in route if i < cfg.nx_phys]
    cols = [route.index(i) for i in row_ix]
    blk = make_float64_block(phy)
    step = lambda v, z: blk.transition_from_free(v, z)               # noqa: E731
    corr = getattr(fs.hfn, 'obc_correction', None)
    v_hat = (corr.vbar if corr is not None else phy.free_params).detach().double()
    v_true = _true_free(phy).detach().double()
    c_true = phy.combinations_from_free(v_true).detach().double()
    Jc = torch.autograd.functional.jacobian(lambda v: phy.combinations_from_free(v).double(), v_true)
    scale = combination_scale(phy, c_true)
    observed = (phy.combinations_from_free(phy.free_params.detach().double()).detach() - c_true) / scale

    files = {'train': data.train_files, 'val': data.val_files,
             'test': [f for f in data.test_files if record_name(f) in INTERP]}
    over, bias = [], []
    for stack in STACKS:
        ref = build_reference_set(cfg, norm, nx_ann=cfg.nx_ann, files=files[stack], verbose=False)
        J = build_stacked_sensitivity(step, v_hat, ref.Z_step, row_ix)
        with torch.no_grad():
            F = ann(ref.Z_field.to(next(ann.parameters()).dtype))[:, cols, 0].double().reshape(-1)
            if corr is not None:
                F = F - deployed_basis(blk, ref, row_ix, cfg, corr, v_hat) @ corr.theta_frozen.double()
        net = overlap(J, F)
        # truth: Xi = x~+ - f_base(xi*)(x~, u) (eq:obc_eps), normalised like the tuples
        Js = build_stacked_sensitivity(step, v_true, ref.Z_step, row_ix)
        with torch.no_grad():
            f_true = blk.transition_from_free(v_true, ref.Z_step.double())[:, :, 0]
        if ref.x_phys_next_rec is not None:
            # D-252: the stored reconstructed successor of EVERY tuple, so Xi and Phi* stack the
            # same tuples as the OBC coefficient (D-268, audit 2026-10-16)
            x_mean = torch.as_tensor(np.ravel(norm.x_mean), dtype=torch.float64)
            std_x = torch.as_tensor(np.ravel(norm.std_x), dtype=torch.float64)
            xi = ((ref.x_phys_next_rec - x_mean) / std_x - f_true)[:, row_ix].reshape(-1)
            Phi = Js
        else:
            # earlier reference sets store no successor: consecutive tuples of the same record
            same = (ref.record_ix[1:] == ref.record_ix[:-1]) & (ref.sample_ix[1:] == ref.sample_ix[:-1] + 1)
            k = torch.nonzero(same).squeeze(1)
            x_next = ref.Z_step[:, :cfg.nx_phys, 0].double()
            xi = (x_next[k + 1] - f_true[k])[:, row_ix].reshape(-1)
            Phi = Js[torch.stack([k * len(row_ix) + j for j in range(len(row_ix))], 1).reshape(-1)]
        tru = overlap(Phi, xi)
        # THEORY: thesis eq:obc_recovery, first line, solved on this stack
        dbeta = torch.linalg.lstsq(Phi, xi.unsqueeze(1)).solution.squeeze(1)
        predicted = (Jc @ dbeta) / scale
        print('[%s %d] %-5s %s | %d tuples | network overlap %.4f | truth overlap %.4f | %.0f s'
              % (run, rid, stack, 'OBC' if cfg.obc else 'U', ref.n, net, tru, time.time() - t0),
              flush=True)
        over.append(dict(run=run, job=rid, arm='obc' if cfg.obc else 'u', stack=stack,
                         n_tuples=ref.n, network_overlap=net, truth_overlap=tru))
        for n, p_, o in zip(phy.COMBO_NAMES, predicted.tolist(), observed.tolist()):
            bias.append((run, rid, stack, n, p_, o))
        del J, Js, Phi

    # learned added-state mode: eigenvalues of d x_a+ / d x_a at 24 encoder states of validation
    # windows (inspect_checkpoint.added_poles; eigenvalues do not depend on the x_a coordinates)
    from inspect_checkpoint import added_poles
    dom, share, fr, zeta = added_poles(fs, data, cfg.nx_phys, cfg.nx_ann)
    print('  added-state mode: %.1f Hz, zeta %.4f, complex at %.0f %% of points, dominant |z| %.4f'
          % (fr, zeta, 100 * share, dom), flush=True)
    for o in over:
        o.update(mode_hz=fr, mode_zeta=zeta, complex_share=share)
    return over, bias


if __name__ == '__main__':
    torch.set_num_threads(4)
    over, bias = [], []
    for r in sys.argv[1:]:
        o, b = evaluate(r)
        over += o
        bias += b
    path = TW / 'Results' / 'res_obc_overlap.csv'
    with open(path, 'w') as fh:
        fh.write('run,job,arm,stack,n_tuples,network_overlap,truth_overlap,mode_hz,mode_zeta,complex_share\n')
        for o in over:
            fh.write('%s,%d,%s,%s,%d,%.6e,%.6e,%.3f,%.5f,%.3f\n' % (
                o['run'], o['job'], o['arm'], o['stack'], o['n_tuples'], o['network_overlap'],
                o['truth_overlap'], o['mode_hz'], o['mode_zeta'], o['complex_share']))
    print('wrote', path)
    path = TW / 'Results' / 'res_obc_bias.csv'
    with open(path, 'w') as fh:
        fh.write('run,job,stack,combination,predicted_rel,observed_rel\n')
        for row in bias:
            fh.write('%s,%d,%s,%s,%.6e,%.6e\n' % row)
    print('wrote', path)
