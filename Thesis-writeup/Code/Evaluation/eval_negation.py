"""Negation cosine of the learned correction against the baseline response (Kessels' sense).

Tuples, coordinates, rows and Phi exactly as eval_overlap.py (D-246): `build_reference_set` over
the test records I1 to I6, x_a = 0, normalised coordinates (eq. aug_norm), the physical rows the
network writes, Phi at the trained physical coordinate (vbar of the correction in an OBC arm).

  F_base  stacked baseline next-state map f_base(theta(beta), x_i, u_i) at that same coordinate
          (the transition rows, i.e. the offset column Gamma of Gyorok 2025 Eq. 18)
  F_aug   network physical-row output as the model applies it (minus Phi theta_frozen under OBC)
  c_neg   <F_aug, F_base> / (||F_aug|| ||F_base||); near 0 no negation, near -1 cancellation
  c_perp  cosine of F_aug with (I - Pi) F_base, Pi the projection onto ran Phi (pooled stack)

Per record and pooled. Evaluation only, nothing is trained or modified.

Usage: python Thesis-writeup/Code/Evaluation/eval_negation.py R1 R4
"""
import sys
import time

import torch

from eval_common import RUNS, RUNS_DIR, record_name
from eval_greybox import run_config, build

INTERP = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']


def cos(a, b):
    return float(a @ b / (torch.linalg.norm(a) * torch.linalg.norm(b)))


def evaluate(run):
    from gantry_dynamic.training import load_checkpoint
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
    if corr is not None:
        print('  ||vbar - free_params|| = %.3e' % float(torch.linalg.norm(
            v_hat - phy.free_params.detach().double())), flush=True)
    J = build_stacked_sensitivity(step, v_hat, ref.Z_step, row_ix)
    with torch.no_grad():
        F = ann(ref.Z_field.to(next(ann.parameters()).dtype))[:, cols, 0].double().reshape(-1)
        if corr is not None:
            F = F - J @ corr.theta_frozen.double()
        Fb = step(v_hat, ref.Z_step.double())[:, row_ix, 0].reshape(-1)
    c = torch.linalg.lstsq(J, Fb.unsqueeze(1)).solution
    Fp = Fb - (J @ c).squeeze(1)                                        # (I - Pi) F_base
    rec = ref.record_ix.repeat_interleave(len(row_ix))
    names = [record_name(n) for n in ref.record_names]
    rows = []
    for r, nm in enumerate(names):
        m = rec == r
        rows.append((nm, cos(F[m], Fb[m]), cos(F[m], Fp[m]),
                     float(torch.linalg.norm(F[m]) / torch.linalg.norm(Fb[m]))))
    rows.append(('pooled', cos(F, Fb), cos(F, Fp), float(torch.linalg.norm(F) / torch.linalg.norm(Fb))))
    print('[%s %d] %s | %d held-out tuples | ||(I-Pi)F_base||/||F_base|| %.3e | %.0f s'
          % (run, rid, 'OBC' if corr is not None else 'U', ref.n,
             float(torch.linalg.norm(Fp) / torch.linalg.norm(Fb)), time.time() - t0), flush=True)
    for nm, a, b, q in rows:
        print('  %-6s c_neg %+.3e | cos(F_aug,(I-Pi)F_base) %+.3e | ||F_aug||/||F_base|| %.3e'
              % (nm, a, b, q), flush=True)
    return rows


if __name__ == '__main__':
    torch.set_num_threads(4)
    for r in sys.argv[1:]:
        evaluate(r)
