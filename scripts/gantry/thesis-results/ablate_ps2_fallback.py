"""PS2 fallback ablation (D-270): which change of the final method keeps the PS2 screen above 0.90.

The R4 setting (Aug-U, low-pass noise, detuned, n_a 2, PS2) with ONE factor of the 2026-10-11 run 204 restored per arm,
375 Adam updates (S1 to its cap and S2), no polish, no evaluation. Read in the log: "[nf val  ]" at update 0,
"[ps2] update ..." and the S2 line. The main pipeline is unchanged: this script sets configuration values and calls the
pipeline's own functions in the order of gantry_interconnect_dynamic.main().

  ABL_ARM  A    final configuration (negative control; R4 gave 1.04)
           B    all factors as run 204: encoder linear_map with na_nb 29, g_aug zero, no weighting (positive control)
           C    final, encoder as run 204 (linear_map, window up to tau, na_nb 29; one switch in get_encoder_dims)
           D    final, g_aug zero
           E    final, no loss weighting
           F    final, the physical encoder rows at the least-squares map of the closed-loop baseline simulation at the
                detuned theta^0 (no true parameters), no Adam pre-fit (D-270 amendment)
           LSQ  no training: the encoder initialisation on the pre-fit's own windows, per state:
                least squares (encoder paper Eq. 33, with a bias column), the 20-epoch Adam fit, Adam continued to 200
  THESIS_SEED  1 | 2
Usage (server): Thesis-writeup/Code/Server/ablation/ps2_fallback.sbatch
"""
import os
import sys
from dataclasses import replace

import numpy as np
import torch

ARMS = {
    'A': {},
    'B': dict(encoder_init='linear_map', na_nb_override=29, g_init='zero', loss_weighting='none',
              validation_measure='sim-RMS'),
    'C': dict(encoder_init='linear_map', na_nb_override=29),
    'D': dict(g_init='zero'),
    'E': dict(loss_weighting='none'),
    'F': {},
    'LSQ': {},
}
N_ITS = 375          # HEURISTIC: S1 cap 350 (PS2Spec) plus the S2 fit and a few updates after it
LSQ_EPOCHS = (50, 100, 200)   # Adam epochs at which the continued fit is reported (Table 3 of the encoder paper: 50)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))


def prefit_windows(fit_sys, cfg, data, norm):
    """The pre-fit's windows (y_hat, u_cl) and targets x at tau. Copy of model.py::_prefit's dataset construction
    (the closed-loop baseline simulation from the measured positions); the fit itself is not copied."""
    from model_augmentation.fit_systems.closed_loop import closed_loop_rollout
    from model_augmentation.systems.gantry_ss import P
    enc, sim = fit_sys.encoder, fit_sys.simulator
    nx, na, nb, stride = cfg.nx_phys, fit_sys.na, fit_sys.nb, cfg.stride
    dtype, dev = next(enc.parameters()).dtype, next(enc.parameters()).device
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=dev)   # noqa: E731
    rv = lambda a: np.asarray(a, np.float64).ravel()                      # noqa: E731
    fit_sys._sync_prediction_state_for_validation()
    step_kw = {}
    with torch.no_grad():
        if fit_sys._mats_ix is not None:
            step_kw['pass_mats'] = fit_sys.hfn.connected_blocks[fit_sys._mats_ix].pass_mats()
    sdl = data.train_data.sdl
    PinvT = np.linalg.inv(P.detach().cpu().numpy().astype(np.float64).T)
    x_mean, std_x = rv(norm.x_mean), rv(norm.std_x)
    uh_all, yh_all, xt_all = [], [], []
    groups = {}
    for i, sd in enumerate(sdl):
        groups.setdefault(len(sd.u), []).append(i)
    for ixs in groups.values():
        un = np.stack([(sdl[i].u - rv(fit_sys.norm.u0)) / rv(fit_sys.norm.ustd) for i in ixs])
        yn = np.stack([(sdl[i].y - rv(fit_sys.norm.y0)) / rv(fit_sys.norm.ystd) for i in ixs])
        x0 = np.zeros((len(ixs), nx + cfg.nx_ann))
        q0 = np.stack([np.asarray(sdl[i].y[0], np.float64) @ PinvT.T for i in ixs])
        x0[:, :nx] = (np.hstack([q0, np.zeros_like(q0)]) - x_mean) / std_x
        xs, us = [], []

        def recorder(x, u, *args):
            xs.append(x)
            us.append(u)
            return fit_sys.hfn(x, u, *args)

        with torch.no_grad():
            y_hat, _, _ = closed_loop_rollout(
                recorder, fit_sys.hfn.output_only, T(un), T(yn), T(x0), sim.bank,
                torch.tensor([sim.train_ctrl_rows[i] for i in ixs], dtype=torch.long, device=dev), **step_kw)
        X = torch.stack(xs, 1).reshape(len(ixs), len(xs), -1)
        U = torch.stack(us, 1).reshape(len(ixs), len(us), -1)
        starts = torch.arange(0, X.shape[1] - max(na, nb), stride, device=dev)
        uh_all.append(U[:, starts[:, None] + torch.arange(nb, device=dev)].flatten(0, 1))
        yh_all.append(y_hat[:, starts[:, None] + torch.arange(na, device=dev)].flatten(0, 1))
        xt_all.append(X[:, starts + max(na, nb), :nx].flatten(0, 1))
    return torch.cat(uh_all), torch.cat(yh_all), torch.cat(xt_all)


def per_state(err):
    return ' '.join('%.3e' % v for v in err.pow(2).mean(0).tolist()) + '  | mean %.4e' % float(err.pow(2).mean())


def least_squares(uh, yh, xt):
    """THEORY: encoder paper Eq. (33), the least-squares data-based map, here with a bias column as in D-251.
    Returns (W_y, W_u, b, error) with x ~ W_y yh.flatten + W_u uh.flatten + b, the encoder's own window order."""
    Phi = torch.cat([yh.flatten(1), uh.flatten(1), torch.ones(len(xt), 1, dtype=xt.dtype, device=xt.device)], 1)
    W = torch.linalg.lstsq(Phi.cpu(), xt.cpu()).solution.to(xt.device)
    ny_h = yh[0].numel()
    return W[:ny_h].T, W[ny_h:-1].T, W[-1], Phi @ W - xt


def set_least_squares_start(enc, nx, W_y, W_u, b):
    """Physical rows of the encoder = the least-squares map: W^b set, the physical output rows of psi_tilde zero,
    and their bias absorbing the encoder's constant (D-055 offsets and the hidden layers' zero-input output)."""
    final = enc.net[-1]
    with torch.no_grad():
        enc.Wb_psi_y.copy_(W_y)
        enc.Wb_psi_u.copy_(W_u)
        final.weight[:nx] = 0.0
        final.bias[:nx] = 0.0
        p = next(enc.parameters())
        zu = torch.zeros((1,) + tuple(enc_window_shape(enc, 'u')), dtype=p.dtype, device=p.device)
        zy = torch.zeros((1,) + tuple(enc_window_shape(enc, 'y')), dtype=p.dtype, device=p.device)
        final.bias[:nx] = b - enc(zu, zy)[0, :nx]


def enc_window_shape(enc, which):
    return (enc.nb + 1, enc.nu) if which == 'u' else (enc.na + 1, enc.ny)


def run_f_prefit(fit_sys, cfg, data, norm):
    """Arm F: the physical encoder rows start at the least-squares map (encoder paper Eq. 33, a linear restriction of
    the data-based initialisation), the physical output rows of psi_tilde at zero; no Adam pre-fit (D-270 amendment)."""
    if cfg.device == 'cuda':
        fit_sys.cuda()
    uh, yh, xt = prefit_windows(fit_sys, cfg, data, norm)
    nx, enc = cfg.nx_phys, fit_sys.encoder
    W_y, W_u, b, err = least_squares(uh, yh, xt)
    set_least_squares_start(enc, nx, W_y, W_u, b)
    with torch.no_grad():
        ls_mse = float(torch.mean((enc(uh, yh)[:, :nx] - xt) ** 2))
    print('[arm F] least-squares start: MSE %.4e (solve %.4e)' % (ls_mse, float(err.pow(2).mean())), flush=True)
    if not np.isclose(ls_mse, float(err.pow(2).mean()), rtol=1e-6, atol=1e-12):
        raise RuntimeError('the encoder does not reproduce the least-squares map')
    print('[arm F] per state: ' + per_state(err), flush=True)
    if cfg.device == 'cuda':
        fit_sys.cpu()


def run_lsq(fit_sys, cfg, data, norm):
    from gantry_dynamic.model import prefit_physical_encoder
    if cfg.device == 'cuda':
        fit_sys.cuda()
    uh, yh, xt = prefit_windows(fit_sys, cfg, data, norm)
    nx, enc = cfg.nx_phys, fit_sys.encoder
    print('[lsq] %d windows, per-state MSE in normalised states (order of the model state vector)' % len(xt))
    print('[lsq] least squares (Eq. 33):  ' + per_state(least_squares(uh, yh, xt)[3]), flush=True)
    with torch.no_grad():
        print('[lsq] encoder before the fit:  ' + per_state(enc(uh, yh)[:, :nx] - xt), flush=True)
    prefit_physical_encoder(fit_sys, cfg, data, norm)       # the production fit, 20 epochs (moves back to CPU)
    if cfg.device == 'cuda':
        fit_sys.cuda()
    with torch.no_grad():
        print('[lsq] Adam %3d epochs (prod): ' % cfg.encoder_fit_epochs + per_state(enc(uh, yh)[:, :nx] - xt),
              flush=True)
    # Adam continued with the production loop's settings (model.py::_prefit); a fresh optimizer state.
    final = enc.net[-1]
    params = [enc.Wb_psi_y, enc.Wb_psi_u] + [p for layer in enc.net for p in layer.parameters()]
    opt = torch.optim.Adam(params, lr=cfg.encoder_fit_lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    for ep in range(cfg.encoder_fit_epochs, max(LSQ_EPOCHS)):
        for ix in torch.randperm(len(xt), generator=gen).to(xt.device).split(cfg.batch_size):
            opt.zero_grad()
            torch.mean((enc(uh[ix], yh[ix])[:, :nx] - xt[ix]) ** 2).backward()
            final.weight.grad[nx:] = 0.0
            final.bias.grad[nx:] = 0.0
            opt.step()
        if ep + 1 in LSQ_EPOCHS:
            with torch.no_grad():
                print('[lsq] Adam %3d epochs:        ' % (ep + 1) + per_state(enc(uh, yh)[:, :nx] - xt), flush=True)


def main():
    # RETIRED (D-271): this ablation ran against the code before D-271 (jobs 89778, 89789, logs in
    # scripts/gantry/meeting/meeting-19-10-2026/ps2-ablation). Its arms A, D, E and LSQ call the production pre-fit,
    # which is now the least-squares start, and the Adam settings it reads no longer exist; a rerun would not
    # reproduce the logged experiment.
    raise SystemExit('ablate_ps2_fallback.py is retired (D-271); the D-270 logs are the record')
    arm = os.environ.get('ABL_ARM')
    if arm not in ARMS:
        raise ValueError('ABL_ARM must be one of %s, got %r' % (sorted(ARMS), arm))
    os.environ.update(THESIS_ARM='u', THESIS_NOISE='lowpass', THESIS_START='detuned', THESIS_NA='2',
                      THESIS_PS2='1', THESIS_INIT='live')
    os.environ.setdefault('THESIS_SEED', '1')
    import gantry_interconnect_dynamic as gid
    from gantry_dynamic.data import load_datasets, compute_normalization
    from gantry_dynamic.model import build_model, prefit_physical_encoder
    from gantry_dynamic.controller import build_closed_loop
    from gantry_dynamic.training import train_model_with_diagnostics
    cfg = replace(gid.CFG, **ARMS[arm], n_its=N_ITS, its_per_val=N_ITS, lbfgs=False, save_flag=False)
    print('[ablation] arm %s seed %d: encoder=%s na_nb=%d g_init=%s loss_weighting=%s validation=%s n_its=%d'
          % (arm, cfg.seed, cfg.encoder_init, cfg.na_nb, cfg.g_init, cfg.loss_weighting, cfg.validation_measure,
             N_ITS), flush=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(cfg.hp, cfg, data, norm)
    fit_sys.simulator = build_closed_loop(
        fit_sys, norm, cfg, train_files=data.train_files, val_files=data.val_files, val_data=data.val_ckpt_data,
        e_train=data.e_train, e_val=data.e_val)
    if os.environ.get('ABL_BUILD_ONLY') == '1':          # local check: the arm builds with its configuration
        print('[ablation] built: na %d nb %d na_right %d' % (fit_sys.na, fit_sys.nb, getattr(fit_sys, 'na_right', 0)))
        return
    if arm == 'LSQ':
        run_lsq(fit_sys, cfg, data, norm)
        return
    if arm == 'F':
        run_f_prefit(fit_sys, cfg, data, norm)
    elif cfg.encoder_init == 'data_fit':
        prefit_physical_encoder(fit_sys, cfg, data, norm)
    if os.environ.get('ABL_INIT_ONLY') == '1':           # local check: stop after the encoder start
        return
    train_model_with_diagnostics(fit_sys, cfg.hp, cfg, data, norm, checkpoint_dir=None, run_id='ablation_' + arm)


if __name__ == '__main__':
    main()
