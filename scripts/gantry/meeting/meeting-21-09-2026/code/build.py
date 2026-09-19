"""Checkpoint -> closed-loop system, and the closed-loop simulation that keeps y_hat.

Kept separate from `collect.py` so a diagnostic can import `build_arm` without triggering the
simulation, and separate from the figure script so redrawing never touches torch at all.

EVERY SWITCH IS PINNED, NOT INHERITED. `gantry_interconnect_dynamic.CFG` is edited between runs,
so a figure that reads whatever happens to be checked out is measuring the file rather than the
run. Each arm's `config.json` was written next to its checkpoint, and `_cfg_from_json` replays it
field by field. Only the knobs that cannot affect a forward simulation are forced: device to cpu,
compilation off, saving off, and the two training-time regularisers (`orth`, `obc`) off, since
both act on the GRADIENT and building their bases at construction time would cost minutes for a
projection that no forward pass ever sees.

ARCHITECTURE STILL COMES OFF THE WEIGHTS. `config_for_checkpoint` reads nx_ann, the width and the
depth out of the state dict, on top of the replayed config. That is what lets one script serve
three runs, and it is the check that catches a config.json that no longer matches its own .pth.
"""
__project_origin__ = "added"

import json
from dataclasses import replace, fields

import numpy as np
import torch

from fig_common import (ARMS, CTRL_DONOR, RECORD_MAT, SETTLE_MODE, TRAIN_MODE, add_repo_to_path,
                    ckpt_path, config_path)

add_repo_to_path()


# Knobs that must NOT be replayed from config.json, and why.
#   device/compile_mode/save_flag : environment, not experiment
#   burn_in                       : we keep every sample of the simulation
#   orth/obc                      : gradient-side regularisers; building their bases costs
#                                   minutes and changes no forward value
#   lbfgs/start_phase/epochs/...  : training schedule, never reached here
_FORCE = dict(device='cpu', compile_mode=None, save_flag=False, burn_in=0,
              orth=False, obc=False, lbfgs=False, param_prior=False)


def _cfg_from_json(base_cfg, run):
    """Replay a run's config.json onto a RunConfig. Keys are the UPPERCASE field names."""
    with open(config_path(run), encoding='utf-8') as fh:
        raw = json.load(fh)
    kw = {}
    for f in fields(base_cfg):
        key = f.name.upper()
        if key in raw and f.name not in _FORCE:
            v = raw[key]
            if isinstance(v, list) and f.name in ('ann_route_ix',):
                v = tuple(v)
            kw[f.name] = v
    kw.update(_FORCE)
    return replace(base_cfg, **kw), raw


def build_arm(arm, verbose=True):
    """The arm's closed-loop system on the CPU, plus its data, norm and cfg.

    `ARMS[arm]['weights'] = False` returns the same architecture at its INITIALISATION: linear-map
    encoder, ANN output exactly zero. With the ANN silent the augmented states are inert, because
    nothing but the ANN writes those rows and the output block reads only the physical ones, so it
    is the baseline FP model running inside the same loop. For these joint-estimation runs the
    physical parameters are then at COMBO_INIT_DETUNE, i.e. detuned by +/-10%.
    """
    import gantry_interconnect_dynamic as G
    from gantry_dynamic.data import load_datasets, compute_normalization, VAL_FILES, TRAIN_FILES
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop
    from gantry_dynamic.training import config_for_checkpoint, resolve_checkpoint

    spec = ARMS[arm]
    run = spec['run']
    ckpt = ckpt_path(run)
    hfn_sd, enc_sd, _opt, meta, fmt = resolve_checkpoint(str(ckpt))
    if verbose:
        print('[ckpt] %s  format=%s  done_epochs=%s  bestfit=%.15e'
              % (ckpt.name, fmt, meta['done_epochs'], float(meta['bestfit'])))

    cfg, raw = _cfg_from_json(G.CFG, run)
    if raw['MODE'] != TRAIN_MODE:
        raise RuntimeError('run %s was trained on %r, not %r; the normalisation this figure '
                           'uses would be the wrong one' % (run, raw['MODE'], TRAIN_MODE))
    cfg = config_for_checkpoint(cfg, hfn_sd)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(cfg.hp, cfg, data, norm)
    fit_sys.simulator = build_closed_loop(fit_sys, norm, cfg, train_files=TRAIN_FILES,
                                          val_files=VAL_FILES, val_data=data.val_ckpt_data,
                                          verbose=verbose)
    if spec['weights']:
        fit_sys.hfn.load_state_dict(hfn_sd)
        fit_sys.encoder.load_state_dict(enc_sd)
    else:
        _assert_ann_silent(fit_sys)
    fit_sys.hfn.eval()
    fit_sys.encoder.eval()
    return fit_sys, data, norm, cfg


def _assert_ann_silent(fit_sys):
    """The baseline arm is only a baseline if the ANN really does output zero."""
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    ann = next((b for b in fit_sys.hfn.connected_blocks if isinstance(b, Static_ANN_Block)), None)
    if ann is None:
        raise RuntimeError('no Static_ANN_Block in this interconnect')
    last = [m for m in ann.net.net if isinstance(m, torch.nn.Linear)][-1]
    w = float(last.weight.abs().max())
    b = float(last.bias.abs().max()) if last.bias is not None else 0.0
    if max(w, b) != 0.0:
        raise RuntimeError('baseline arm: ANN output layer is not zero (max |w| = %.3e, '
                           'max |b| = %.3e), so this is not the FP model alone' % (w, b))
    print('[baseline] ANN output layer is exactly zero: the FP model alone in the loop')


def controller_row(fit_sys, name=CTRL_DONOR):
    """The controller-bank row for `name`.

    S1 was generated at Y_op = -0.22, the SAME operating point as V2_aprbs_Ylow, and the bank is
    keyed by Y_op (`build_controller_bank`: one (A,B,C,D) per distinct Y_op, rows index into it).
    Borrowing V2's row is therefore exact, not an approximation, and it means the bank is never
    rebuilt and `gantry_dynamic/controller.py` needs no new entry.
    """
    rows = [row for (n, _sd, row) in fit_sys.simulator.val_records if n == name]
    if len(rows) != 1:
        raise RuntimeError('expected exactly one %r in the validation records, found %d'
                           % (name, len(rows)))
    return int(rows[0])


def load_record(cfg, filename, mode):
    """One record as a deepSI System_data, from `mode`'s folder rather than the training one."""
    from gantry_dynamic.data import load_traj
    return load_traj(filename, replace(cfg, mode=mode))


def closed_loop_sim(fit_sys, sd, ctrl_row):
    """Closed-loop simulation of ONE record. Returns (y_hat, y_true, k0).

    NOT a free run, whatever `closed_loop_free_run_rms` is called: `y_data` enters through the
    controller at every timestep. What is free is the model STATE, propagated by the model and
    never teacher-forced, from an x_0 the encoder window alone produces.

    The chain is `closed_loop_free_run_rms_batch`'s, verbatim in order and in dtype, with the
    trajectory kept instead of thrown away: normalise, encoder window at k0 = max(na, nb), one
    `closed_loop_rollout`, denormalise, back to metres.
    """
    from model_augmentation.fit_systems.closed_loop import closed_loop_rollout

    norm = fit_sys.norm
    na, nb = fit_sys.na, fit_sys.nb
    na_r, nb_r = getattr(fit_sys, 'na_right', 0), getattr(fit_sys, 'nb_right', 0)
    k0 = max(na, nb)
    p = next(fit_sys.hfn.parameters())
    dtype, device = p.dtype, p.device
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    rv = lambda a: np.asarray(a).ravel()                                           # noqa: E731
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)  # noqa: E731

    un = ((sd.u - rv(norm.u0)) / rv(norm.ustd)).astype(np_dtype)
    yn = ((sd.y - rv(norm.y0)) / rv(norm.ystd)).astype(np_dtype)

    with torch.no_grad():
        x0 = fit_sys.encoder(T(un[None, k0 - nb:k0 + nb_r]), T(yn[None, k0 - na:k0 + na_r]))
        y_pred, _, _ = closed_loop_rollout(
            fit_sys.hfn, fit_sys.hfn.output_only, T(un[None, k0:]), T(yn[None, k0:]), x0,
            fit_sys.simulator.bank,
            torch.tensor([ctrl_row], dtype=torch.long, device=device))
    y_hat = y_pred[0].cpu().numpy() * rv(norm.ystd) + rv(norm.y0)
    return y_hat.astype(np.float64), np.asarray(sd.y)[k0:].astype(np.float64), k0


def nrms(y_hat, y_true, norm):
    """Per-channel NRMS exactly as `evaluation.py:665` prints it: rms(error) / ystd."""
    e = y_hat - y_true
    return np.sqrt((e ** 2).mean(axis=0)) / np.asarray(norm.ystd).ravel()


__all__ = ['build_arm', 'controller_row', 'load_record', 'closed_loop_sim', 'nrms',
           'RECORD_MAT', 'SETTLE_MODE']
