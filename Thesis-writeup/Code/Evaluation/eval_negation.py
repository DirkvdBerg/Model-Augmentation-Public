"""Negation cosine of the learned correction against the baseline response (Kessels' sense).

Tuples, coordinates, rows and Phi exactly as eval_overlap.py (D-246): `build_reference_set` over
the training, validation or test (I1 to I6) records (D-269), x_a = 0, normalised coordinates (eq. aug_norm), the physical rows the
network writes, Phi at the trained physical coordinate (vbar of the correction in an OBC arm).

  F_base  stacked baseline next-state map f_base(theta(beta), x_i, u_i) at that same coordinate
          (the transition rows, i.e. the offset column Gamma of Gyorok 2025 Eq. 18)
  F_aug   network physical-row output as the model applies it (minus the deployed OBC correction,
          [Phi] or [Phi F_base] times theta_frozen, D-257)
  c_neg   cosine of F_aug with the baseline one-step INCREMENT F_base - x~ (the definition of
          04_obc.tex, FINAL-RUN-FIXES item 19); near 0 no negation, near -1 cancellation
  c_next  cosine of F_aug with the full next state F_base (the earlier printout, kept beside it)
  c_perp  cosine of F_aug with (I - Pi) F_base, Pi the projection onto ran Phi (pooled stack)

Per record and pooled, on the training, validation and test (I1 to I6) stacks.
Output: Results/res_negation.csv (run, stack, record, the cosines, ||F_aug||/||F_base||, checkpoint label).
Evaluation only, nothing is trained or modified.

Usage: python Thesis-writeup/Code/Evaluation/eval_negation.py R1 R4   (all three stacks)
"""
import sys
import time

import torch

from eval_common import RUNS, RUNS_DIR, TW, record_name, checkpoint_status
from eval_greybox import run_config, build
from eval_overlap import deployed_basis, STACKS

INTERP = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']


def cos(a, b):
    return float(a @ b / (torch.linalg.norm(a) * torch.linalg.norm(b)))


def evaluate(run):
    from gantry_dynamic.training import load_checkpoint
    from gantry_dynamic.obc_gantry import build_reference_set, make_float64_block
    from model_augmentation.fit_systems.obc import build_stacked_sensitivity
    from model_augmentation.fit_systems.blocks import Static_ANN_Block

    rid = RUNS[run]
    st = checkpoint_status(rid)                    # final, or the best so far of a running run
    t0 = time.time()
    cfg = run_config(RUNS_DIR / str(rid))
    data, norm, fs = build(cfg)
    load_checkpoint(fs, st['path'], joint_estimation=True, start_phase='lbfgs')
    print('[%s %d] %s (%s)' % (run, rid, st['label'], st['path']), flush=True)
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
    if corr is not None:
        print('  ||vbar - free_params|| = %.3e' % float(torch.linalg.norm(
            v_hat - phy.free_params.detach().double())), flush=True)
    files = {'train': data.train_files, 'val': data.val_files,
             'test': [f for f in data.test_files if record_name(f) in INTERP]}
    out = {}
    for stack in STACKS:
        ref = build_reference_set(cfg, norm, nx_ann=cfg.nx_ann, files=files[stack], verbose=False)
        J = build_stacked_sensitivity(step, v_hat, ref.Z_step, row_ix)
        with torch.no_grad():
            F = ann(ref.Z_field.to(next(ann.parameters()).dtype))[:, cols, 0].double().reshape(-1)
            if corr is not None:
                # the deployed correction: [J] or [J F_base] (affine, D-257) times the coefficient
                F = F - deployed_basis(blk, ref, row_ix, cfg, corr, v_hat) @ corr.theta_frozen.double()
            Fb = step(v_hat, ref.Z_step.double())[:, row_ix, 0].reshape(-1)
            # FIX-19 evaluation: the baseline one-step INCREMENT F_base - x~ on the same rows
            Fi = Fb - ref.Z_step[:, :cfg.nx_phys, 0].double()[:, row_ix].reshape(-1)
        c = torch.linalg.lstsq(J, Fb.unsqueeze(1)).solution
        Fp = Fb - (J @ c).squeeze(1)                                        # (I - Pi) F_base
        rec = ref.record_ix.repeat_interleave(len(row_ix))
        names = [record_name(n) for n in ref.record_names]
        rows = []
        for r, nm in enumerate(names):
            m = rec == r
            rows.append((nm, cos(F[m], Fi[m]), cos(F[m], Fb[m]), cos(F[m], Fp[m]),
                         float(torch.linalg.norm(F[m]) / torch.linalg.norm(Fb[m]))))
        rows.append(('pooled', cos(F, Fi), cos(F, Fb), cos(F, Fp), float(torch.linalg.norm(F) / torch.linalg.norm(Fb))))
        print('[%s %d] %s %s | %d tuples | ||(I-Pi)F_base||/||F_base|| %.3e | %.0f s'
              % (run, rid, stack, 'OBC' if corr is not None else 'U', ref.n,
                 float(torch.linalg.norm(Fp) / torch.linalg.norm(Fb)), time.time() - t0), flush=True)
        for nm, a, b, c, q in rows:
            print('  %-6s c_neg %+.3e | cos(F_aug,F_base) %+.3e | cos(F_aug,(I-Pi)F_base) %+.3e | '
                  '||F_aug||/||F_base|| %.3e' % (nm, a, b, c, q), flush=True)
        out[stack] = [(run, rid, 'obc' if corr is not None else 'u', stack) + r + (st['label'],) for r in rows]
    return out


if __name__ == '__main__':
    torch.set_num_threads(4)
    rows = []
    for r in sys.argv[1:]:
        rows += [row for stack_rows in evaluate(r).values() for row in stack_rows]
    path = TW / 'Results' / 'res_negation.csv'
    with open(path, 'w') as fh:
        fh.write('run,job,arm,stack,record,c_neg,c_next,c_perp,norm_ratio,status\n')
        for row in rows:
            fh.write('%s,%d,%s,%s,%s,%.6e,%.6e,%.6e,%.6e,%s\n' % row)
    print('wrote', path)
