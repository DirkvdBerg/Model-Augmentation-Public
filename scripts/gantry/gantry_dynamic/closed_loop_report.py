"""The post-run report in closed loop (D-227): one table per split, five rows, per record.

Every row is scored the way training and checkpoint selection score: a closed-loop free run
through the known controller in the residual form, the full record from k0, the error in metres.

  1 model                 the trained augmented model
  2 physics part          the same model with the learned block's output zeroed
  3 baseline, start       the physics block at its START coordinate (free = 0: the detuned or
                          correct start of the arm), learned block zeroed
  4 baseline, true        the physics block at the TRUE combinations (those of the truth's nominal
                          parameters), learned block zeroed
  5 oracle                the truth itself (FP + absorber + tanh friction, parameters from the
                          record's meta.truth), seeded from the true state at k0 (common/oracle.py)

Rows 1 to 4 share one scoring chain (`closed_loop_free_run_rms_batch`) and one encoder x0. Rows 2
to 4 score a deep COPY of the model (`_without_learned`); the trained model is never modified
(user 2026-09-28, todo 7b).
"""
__project_origin__ = "added"

import copy
import os

import numpy as np
import torch
from scipy.io import loadmat

from model_augmentation.fit_systems.closed_loop import closed_loop_free_run_rms_batch
from model_augmentation.fit_systems.blocks import Static_ANN_Block

from .controller import build_controller_bank, controller_ss, gain_for, y_op_for
from .data import load_mat_aug, traj_dir
from common import oracle

ROWS = ('model', 'physics part', 'baseline, start', 'baseline, true', 'oracle')


def _without_learned(fit_sys, free=None):
    """A deep copy of the model with the learned component removed and, if `free` is given, the
    physics block's free coordinate set to it. The model passed in is not touched (todo 7b).

    Removing the learned component is two things. The learned block's complete output is disabled
    with its non-persistent intervention gate. In an OBC arm the frozen coefficient `theta_frozen`
    is zeroed as well: the
    correction -E J theta is the learned field's projection onto the physics span, so it belongs
    to the learned component and vanishes with it (exactly, being linear in theta). D-227 zeroed
    only the output layer in place, so its OBC rows 2 to 4 still carried the trained -E J theta.

    `copy.deepcopy` goes through the fit system's `__getstate__`, which drops the OBC lifecycle and
    reference set (training-time state, D-192); the physics block's per-forward cache `_cur` is
    rebuilt from the copy's parameters on every forward.
    """
    m = copy.deepcopy(fit_sys)
    blocks = m.hfn.connected_blocks
    ann = next((b for b in blocks if isinstance(b, Static_ANN_Block)), None)
    corr = getattr(m.hfn, 'obc_correction', None)
    with torch.no_grad():
        if ann is not None:
            ann.out_gate.zero_()
        if corr is not None:
            corr.theta_frozen.zero_()
        if free is not None:
            next(b for b in blocks if hasattr(b, 'free_params')).free_params.copy_(free)
    return m


def _true_free(phy):
    """The free coordinate at which the reduced block holds the TRUE combinations.

    `params_init` is the truth's nominal raw vector (model.py); the start `combo_init` may be
    detuned. The block owns both directions of its coordinate map so reporting cannot retain an
    obsolete inverse when the training coordinate changes."""
    dt = phy.free_params.dtype
    c_true = type(phy).combos_of(phy.params_init.double(), float(phy.Lb)).to(dt)
    free = phy.free_from_combinations(c_true)
    tol = 1e-10 if dt == torch.float64 else 1e-5
    assert torch.allclose(phy.combinations_from_free(free), c_true, rtol=tol, atol=0)
    return free


def _score(fit_sys, sdl, bank, rows, want_pred=False):
    out = closed_loop_free_run_rms_batch(fit_sys, sdl, bank, rows, return_pred=want_pred)
    return (out[0], out[2]) if want_pred else (out[0], None)


def closed_loop_table(fit_sys, cfg, data, norm, up_sample, split):
    """Score the five rows on one split ('val' or 'test'); print the table; return a dict.

    Returns {'files', 'rms' (5, n_records, 3) in metres, NaN where a row is not available,
    'pred_model' (the row-1 closed-loop output of each record, from k0), 'k0'}.
    """
    files = list(data.val_files if split == 'val' else data.test_files)
    sdl = list(data.val_list if split == 'val' else data.test_list)
    ok = []
    for i, f in enumerate(files):
        try:
            y_op_for(f, cfg.controller_gain)
            ok.append(i)
        except KeyError as e:
            print('  [closed-loop report] %s skipped: %s' % (f, e))
    files_ok = [files[i] for i in ok]
    sdl_ok = [sdl[i] for i in ok]
    rms = np.full((len(ROWS), len(files), 3), np.nan)
    preds = [None] * len(files)
    if not sdl_ok:
        return {'files': files, 'rms': rms, 'pred_model': preds, 'k0': None}
    bank, rows, _ = build_controller_bank(files_ok, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u,
                                          dtype=cfg.dtype_pt, use_gain=cfg.controller_gain)
    k0 = max(fit_sys.na, fit_sys.nb)
    phy = next((b for b in fit_sys.hfn.connected_blocks if hasattr(b, 'free_params')), None)

    r1, p1 = _score(fit_sys, sdl_ok, bank, rows, want_pred=True)
    r2, _ = _score(_without_learned(fit_sys), sdl_ok, bank, rows)
    r3 = r4 = None
    if phy is not None and hasattr(phy, 'combo_init'):
        r3, _ = _score(_without_learned(fit_sys, torch.zeros_like(phy.free_params)), sdl_ok, bank, rows)
        r4, _ = _score(_without_learned(fit_sys, _true_free(phy)), sdl_ok, bank, rows)
    for j, i in enumerate(ok):
        rms[0, i], rms[1, i] = r1[j], r2[j]
        if r3 is not None:
            rms[2, i], rms[3, i] = r3[j], r4[j]
        preds[i] = p1[j]

    # row 5: the truth in closed loop, from the true state at k0, same controller and residual form
    for j, i in enumerate(ok):
        f = files[i]
        meta = loadmat(os.path.join(traj_dir(cfg), f), variable_names=['meta'], squeeze_me=True,
                       struct_as_record=False).get('meta')
        tp = oracle.truth_from_meta(meta.truth) if meta is not None and hasattr(meta, 'truth') else None
        if tp is None:
            continue
        _, _, xl, xa = load_mat_aug(f, cfg)
        x0 = np.array([xl[k0, 0], xl[k0, 1], xl[k0, 2], xa[k0, 0],
                       xl[k0, 3], xl[k0, 4], xl[k0, 5], xa[k0, 1]])
        ctrl = controller_ss(y_op_for(f, cfg.controller_gain), cfg.ts_new, gain_for(f, cfg.controller_gain))
        y_or = oracle.oracle_closed_loop(np.asarray(sdl[i].u)[k0:], np.asarray(sdl[i].y)[k0:], x0, tp,
                                         ctrl, cfg.ts_new, up_sample)
        rms[4, i] = np.sqrt(np.mean((y_or - np.asarray(sdl[i].y)[k0:]) ** 2, axis=0))

    print('\n== A. CLOSED-LOOP FREE RUN: %s, RMS [m] per record, X1 / X2 / Y ==' %
          ('validation (the selection measure)' if split == 'val' else 'test (scored as the selection measure)'))
    for i, f in enumerate(files):
        print('  %s' % os.path.basename(str(f)))
        for r, name in enumerate(ROWS):
            v = rms[r, i]
            print('    %-16s %s' % (name, 'n/a' if np.all(np.isnan(v)) else
                                     '  '.join('%.3e' % x for x in v)))
    return {'files': files, 'rms': rms, 'pred_model': preds, 'k0': k0}
