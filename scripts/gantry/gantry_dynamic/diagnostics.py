"""Diagnostics: R2 helpers, encoder-window state recovery, gradient norms.

The R2 formula, the encoder-window builder, and the affine-map R2 are shared
here (they appeared 2-3 times pre-refactor with identical bodies). All numerics
are verbatim; the extractions are the only structural change.
"""
__project_origin__ = "added"

import os

import numpy as np
import torch
import torch.nn as nn

from .config import RunConfig
from .model import get_encoder_dims


def r2_per_channel(ref, est):
    """Per-channel coefficient of determination between reference and estimate."""
    # THEORY: R^2 = 1 - SS_res/SS_tot (standard OLS); 1e-12 guards constant channels
    ss_res = ((ref - est) ** 2).sum(axis=0)
    ss_tot = ((ref - ref.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def best_affine_r2(X, target, dtype):
    """Best affine map target ~ [X, 1] @ W and its per-channel R2.

    # THEORY: ordinary least squares -- uses all X channels jointly.

    Width-agnostic in X: W is (X.shape[1]+1, target.shape[1]), so this works for any latent
    width against the fixed 2-channel absorber ground truth.
    """
    A = np.hstack([X, np.ones((len(X), 1), dtype=dtype)])
    W, *_ = np.linalg.lstsq(A, target, rcond=None)
    return W, r2_per_channel(target, A @ W)


def heldout_affine_r2(X, target, dtype):
    """@added (2026-09-01). best_affine_r2 fitted on the first half, SCORED on the second.

    WHY. The in-sample R2 of `best_affine_r2` costs X.shape[1]+1 free parameters per target
    channel, so a wider latent state scores higher for free: 3 parameters at NX_ANN=2 against 9
    at NX_ANN=8. On the nominal sample count that inflation looks like ~p/n and negligible
    (9/2086 = 0.004), but these are smooth 4 kHz trajectories with heavy autocorrelation, so the
    EFFECTIVE sample size is far below the nominal one.

    MEASURED (scratchpad/aug_r2_check.py, Nk = 2086 = job 80557's window count, latent = smooth
    autocorrelated noise UNRELATED to the ground truth):

        NX_ANN     R2_linmap in-sample     R2 held-out
             2                   0.018          -0.039
             8                   0.175          -3.06
            32                   0.610          -8.84

    So an unrelated 8-wide latent scores R2_linmap = 0.175 for free, two orders of magnitude
    above the p/n estimate. Reading an NX_ANN=8 R2_linmap against the NX_ANN=2 numbers already on
    record compares a statistic with a 0.175 floor against one with a 0.018 floor.

    Negative values are expected and carry no magnitude information: below zero simply means the
    fitted map does not transfer, i.e. no evidence the absorber is in the span. On real records a
    negative can also reflect an operating-point shift between halves rather than overfitting,
    since the split is first-half / second-half; treat the SIGN as the result, not the size.
    """
    h = len(X) // 2
    A = lambda Z: np.hstack([Z, np.ones((len(Z), 1), dtype=dtype)])   # noqa: E731
    W, *_ = np.linalg.lstsq(A(X[:h]), target[:h], rcond=None)
    return r2_per_channel(target[h:], A(X[h:]) @ W)


def best_single_channel_r2(X, target, dtype):
    """@added (2026-09-01). Per target channel: the best R2 from ANY ONE column of X, affinely
    scaled.

    REPLACES the old `R2_raw`, which compared latent channel i directly against ground-truth
    channel i. That pairing has no justification at any width: the latent basis is not
    identifiable, so channel 0 has no reason to be delta_a. It merely happened not to crash while
    NX_ANN == 2, where it reported ~0 throughout -- consistent with the statistic being
    meaningless rather than with the absorber being absent.

    This asks the question the old one was reaching for, "is any single latent state the
    absorber", and is defined for every width. Together with R2_linmap (is the absorber anywhere
    in the span) it separates one-state identification from distributed representation.
    """
    r2 = np.empty(target.shape[1], dtype=float)
    for c in range(target.shape[1]):
        r2[c] = max(best_affine_r2(X[:, [j]], target[:, [c]], dtype)[1][0]
                    for j in range(X.shape[1]))
    return r2


def encoder_state_estimates(fit_sys, sysdata, na, nb, na_right, nb_right, cfg: RunConfig,
                            max_windows=2000):
    """Run the encoder on strided validation windows -> (k_ix, stride, x_hat).

    deepSI hist convention: ypast = y[k-na : k+na_right], upast = u[k-nb : k+nb_right];
    the encoder output initializes the state at time k.
    """
    DTYPE_NP, DTYPE_PT = cfg.dtype_np, cfg.dtype_pt
    val_norm = fit_sys.norm.transform(sysdata)
    yn = np.ascontiguousarray(val_norm.y, dtype=DTYPE_NP)
    un = np.ascontiguousarray(val_norm.u, dtype=DTYPE_NP)
    N = len(yn)
    k0 = max(na, nb) + 1                                      # +1 so k-1 exists for the lag column
    stride = max(1, (N - k0) // max_windows)                  # HEURISTIC: cap window count to bound memory
    k_ix = np.arange(k0, N, stride)
    ypast = np.stack([yn[k - na : k + na_right] for k in k_ix])  # (Nk, na+na_right, ny)
    upast = np.stack([un[k - nb : k + nb_right] for k in k_ix])  # (Nk, nb+nb_right, nu)
    with torch.no_grad():
        x_hat = fit_sys.encoder(
            torch.tensor(upast, dtype=DTYPE_PT),
            torch.tensor(ypast, dtype=DTYPE_PT),
        ).numpy()                                             # (Nk, nxd)
    return k_ix, stride, x_hat


def aug_state_r2(fit_sys, hp, cfg: RunConfig, data, norm):
    """Encoder augmented state R2 vs delta_a / vdelta_a from mat file.

    Runs the encoder on strided validation windows and computes, PER GROUND-TRUTH CHANNEL:
      R2_best1  -- best single latent channel, affinely scaled (one state IS the absorber?)
      R2_linmap -- best affine map from ALL latent channels  (absorber anywhere in the span?)
      R2_ho     -- R2_linmap fitted on the first half, scored on the second

    CHANGED (2026-09-01): all three are shape (2,), the ABSORBER dimension, not (NX_ANN,).
    NX_ANN is a hyperparameter (the model's latent width, 2 or 8 in this project); the ground
    truth is [delta_a, vdelta_a] and is 2-wide always, fixed by the hidden MSD in the
    data-generating system. Treating them as equal is what produced
        ValueError: operands could not be broadcast together with shapes (2086,2) (2086,8)
    at every NX_ANN != 2, killing this diagnostic and everything after it (job 80557).
    """
    NX_PHYS = cfg.nx_phys
    na, nb, na_right, nb_right = get_encoder_dims(hp, cfg)
    fit_sys.eval()

    k_ix, stride, x_hat = encoder_state_estimates(
        fit_sys, data.val_data, na, nb, na_right, nb_right, cfg)   # (Nk, NX_PHYS + NX_ANN)

    x_ann = x_hat[:, NX_PHYS:]                    # (Nk, NX_ANN)

    # Normalize GT so encoder (dimensionless) and GT are on comparable axes
    gt_raw  = data.val_x_aug[k_ix]                # (Nk, 2) physical units: delta_a, vdelta_a
    gt_mean = gt_raw.mean(axis=0)
    gt_std  = gt_raw.std(axis=0) + 1e-8
    gt_norm = (gt_raw - gt_mean) / gt_std          # (Nk, 2) normalized

    r2_best = best_single_channel_r2(x_ann, gt_norm, cfg.dtype_np)
    W_aug, r2_lin = best_affine_r2(x_ann, gt_norm, cfg.dtype_np)
    r2_ho = heldout_affine_r2(x_ann, gt_norm, cfg.dtype_np)

    return r2_best, r2_lin, r2_ho


def state_recovery_diagnostic(fit_sys, hp, rid, cfg: RunConfig, data, norm, save_dir,
                              max_windows=2000):
    """Linear-map state recovery test: basis rotation vs lost information.

    Compares encoder state estimates x_hat(k) on the validation set against
    physical states reconstructed from measurements:
      R2_raw      x_hat[:, :6] read directly as normalized physical states
      R2_linmap   best least-squares linear map x_true ~ x_hat @ W + b
      R2_raw_lag1 raw comparison against x_true(k-1) (detects one-sample lag)
    Interpretation:
      R2_linmap ~ 1, R2_raw low  -> information present, basis rotated
      R2_linmap low              -> information absent from encoder state
      R2_raw_lag1 > R2_raw       -> encoder aligned to k-1 (history off-by-one)
    """
    NX_PHYS = cfg.nx_phys
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    na, nb, na_right, nb_right = get_encoder_dims(hp, cfg)
    fit_sys.eval()
    DTYPE_NP = cfg.dtype_np
    x_mean, std_x = norm.x_mean, norm.std_x

    # True physical states from augmented simulation mat file (more accurate than P_inv+FD)
    x_true = data.val_x_logical.astype(DTYPE_NP)                             # (N,6) x_logical from mat
    x_true_norm = (x_true - x_mean.flatten()) / std_x.flatten()             # (N,6)

    # Encoder estimates. deepSI hist convention: the encoder output initializes
    # the state at time k. k0 = max(na,nb)+1 so k-1 exists for the lag column.
    k_ix, stride, x_hat = encoder_state_estimates(
        fit_sys, data.val_data, na, nb, na_right, nb_right, cfg, max_windows)

    xt   = x_true_norm[k_ix]                                  # x_true(k)
    xt_l = x_true_norm[k_ix - 1]                              # x_true(k-1)

    r2_raw = r2_per_channel(xt,   x_hat[:, :NX_PHYS])
    r2_lag = r2_per_channel(xt_l, x_hat[:, :NX_PHYS])

    # Best affine map x_true ~ [x_hat, 1] @ W
    W, r2_lin = best_affine_r2(x_hat, xt, cfg.dtype_np)

    labels = ['q1 ', 'q2 ', 'q3 ', 'dq1', 'dq2', 'dq3']
    print('\n=== State recovery diagnostic (D-053) ===')
    print(f'  {len(k_ix)} windows (stride {stride}), na=nb={na}, '
          f'encoder={cfg.encoder_init}')
    print('  channel   R2_raw      R2_linmap   R2_raw_lag1')
    for ch in range(NX_PHYS):
        print(f'  {labels[ch]}     {r2_raw[ch]:+10.4f}  {r2_lin[ch]:+10.4f}  {r2_lag[ch]:+10.4f}')
    print('  R2_linmap ~ 1 & R2_raw low -> basis rotation;')
    print('  R2_linmap low              -> information absent from encoder state;')
    print('  R2_raw_lag1 > R2_raw       -> encoder aligned to k-1 (one-sample lag)')

    # --- Augmented state R2 vs delta_a / vdelta_a ---
    # CHANGED (2026-09-01): guarded. This used to be an unguarded call, so any failure here took
    # down Section D (gradient norms) and the npz save below with it -- which is exactly what
    # happened on every NX_ANN=8 run. The sibling call in evaluation.py was already guarded, so
    # the same fault was a warning there and fatal here.
    r2_aug_best = r2_aug_lin = r2_aug_ho = None
    try:
        r2_aug_best, r2_aug_lin, r2_aug_ho = aug_state_r2(fit_sys, hp, cfg, data, norm)
    except Exception as e:
        print(f'\nWarning: aug_state_r2 failed ({type(e).__name__}: {e}); section skipped')
    if r2_aug_lin is not None:
        aug_labels = ['delta_a  ', 'vdelta_a ']
        aug_notes  = ['(mat file)', '(FD estimate)']
        print('\n=== Augmented state R2 vs saved GT (delta_a/vdelta_a from mat file) ===')
        print(f'  {"state":<12s}  {"R2_best1":>10s}  {"R2_linmap":>10s}  {"R2_ho":>10s}  note')
        # Over the GT channels (2), NOT over NX_ANN: see aug_state_r2.
        for ch in range(len(r2_aug_lin)):
            lbl  = aug_labels[ch] if ch < len(aug_labels) else f'gt[{ch}]'
            note = aug_notes[ch]  if ch < len(aug_notes)  else ''
            print(f'  {lbl}  {r2_aug_best[ch]:+10.4f}  {r2_aug_lin[ch]:+10.4f}  '
                  f'{r2_aug_ho[ch]:+10.4f}  {note}')
        print('  R2_best1  ~ 1 -> ONE latent state is the absorber')
        print('  R2_linmap ~ 1 -> absorber lies in the span of the latent states')
        print('  R2_ho: held-out; the only column comparable ACROSS NX_ANN (see heldout_affine_r2)')

    if cfg.save_flag:
        # r2_aug_raw is NOT reused as a key: its meaning changed (see best_single_channel_r2),
        # and silently reusing it would make old and new npz files look comparable.
        np.savez(os.path.join(save_dir, f'gantry_state_recovery_{rid}.npz'),
                 r2_raw=r2_raw, r2_lin=r2_lin, r2_lag=r2_lag,
                 W=W, k_ix=k_ix, x_hat=x_hat, x_true_norm=xt,
                 r2_aug_best=r2_aug_best, r2_aug_lin=r2_aug_lin, r2_aug_ho=r2_aug_ho)
        print(f'Saved state recovery diagnostic: gantry_state_recovery_{rid}.npz')


def compute_gradient_norms(fit_sys, hp, cfg: RunConfig, data):
    """Single forward+backward pass on training data, return gradient norms per parameter group."""
    DTYPE_PT = cfg.dtype_pt
    fit_sys.train()
    # CHANGED (2026-07-29): pass `stride`, which was missing. deepSI's
    # make_training_data does `stride = Loss_kwargs.get('stride', 1)`, so omitting
    # it built EVERY window instead of every cfg.stride-th, and then kept 256.
    # Two consequences, both real:
    #   MEMORY. Training at nf=6400/stride=100 builds 5824 windows (855 MB). At
    #   stride=1 that is ~582,000 windows, ~86 GB, against a 32 GB job. This
    #   SIGKILLed the T1 rungs nf=1600, 3200 and 6400 at the very end of the run,
    #   after training and the per-record NRMS had completed, so all three lost
    #   their results.json. 400 and 800 were under the limit and survived.
    #   MEANING. The slice below takes the FIRST batch_size windows. At stride=1
    #   those are offsets 0,1,2,... of record 1, i.e. 256 near-identical windows
    #   overlapping by nf-1 samples, so the gradient norm was measured on an
    #   almost degenerate sample. With the training stride they span 100x more
    #   data. Numbers reported by this function before this date, on long-horizon
    #   runs, describe that degenerate sample.
    # Verified by preflight/check_grad_norm_stride.py.
    data_train = fit_sys.make_training_data(fit_sys.norm.transform(data.train_data),
                                            nf=hp['nf'], stride=cfg.stride)
    batch_size = min(hp['batch_size'], len(data_train[0]))
    # CHANGED (2026-09-01): integer arrays keep their dtype. `ctrl_ix` is a per-window row index
    # into the controller bank; casting it to float and back is the exact hazard
    # Dtype_DataLoader documents and avoids (interconnect.py:495-502).
    batch = []
    for d in data_train:
        t = torch.as_tensor(d[:batch_size])
        batch.append(t.to(DTYPE_PT) if t.is_floating_point() else t)

    # CHANGED (2026-09-01): arrays beyond deepSI's four go in BY NAME, mirroring the training
    # loop (interconnect.py:833-851). With a simulator attached, make_training_data appends its
    # `extra_array_names` arrays, so `loss(*batch, ...)` handed 5 positionals to a 4-positional
    # signature and raised. main() swallows that in a try/except, so Section D has silently
    # reported nothing for every closed-loop run -- and the per-block gradient split is precisely
    # the diagnostic the open-loop collapse calls for. In open loop `names` is empty and the
    # split is an exact no-op.
    names = getattr(getattr(fit_sys, 'simulator', None), 'extra_array_names', ())
    if len(batch) - 4 != len(names):
        raise RuntimeError(
            'training data carries %d array(s) beyond deepSI\'s four but the simulator names '
            '%d of them (%s); one would reach loss() unnamed'
            % (len(batch) - 4, len(names), names))
    loss_kwargs = dict(zip(names, batch[4:]))
    loss_kwargs['nf'] = hp['nf']

    fit_sys.optimizer.zero_grad()
    loss = fit_sys.loss(*batch[:4], **loss_kwargs)
    loss.backward()

    grad_norms = {}
    for attr_name, item in fit_sys.parameters_with_names.items():
        params = item['params']
        if isinstance(params, nn.Parameter):
            params = [params]
        else:
            params = list(params)
        for i, p in enumerate(params):
            pname = f'{attr_name}.{i}'
            if p.grad is not None:
                grad_norms[pname] = float(torch.norm(p.grad).item())
            else:
                grad_norms[pname] = 0.0

    # Aggregate by parameter group (encoder, hfn.blocks.0=physics, hfn.blocks.2=ANN)
    group_norms = {}
    for name, norm_val in grad_norms.items():
        if name.startswith('encoder'):
            group = 'encoder'
        elif name.startswith('hfn'):
            group = 'hfn'
        else:
            group = 'other'
        group_norms[group] = group_norms.get(group, 0.0) + norm_val ** 2
    group_norms = {k: float(np.sqrt(v)) for k, v in group_norms.items()}

    print('\n=== Gradient norms (single batch, post-training) ===')
    for group, norm_val in sorted(group_norms.items()):
        print(f'  {group:20s}: {norm_val:.4e}')

    return grad_norms, group_norms


def encoder_init_state(fit_sys, sysdata, K0, na, nb, na_right, nb_right, cfg: RunConfig):
    """Untrained linear-map encoder estimate of x(K0) from the first I/O window (D-072).

    Runs BEFORE training, so fit_sys.encoder is still the pure reconstructability
    map built from the baseline linearization -- a baseline-only quantity.
    """
    DTYPE_NP, DTYPE_PT = cfg.dtype_np, cfg.dtype_pt
    dn = fit_sys.norm.transform(sysdata)
    yp = np.ascontiguousarray(dn.y, dtype=DTYPE_NP)[K0 - na: K0 + na_right][None]
    up = np.ascontiguousarray(dn.u, dtype=DTYPE_NP)[K0 - nb: K0 + nb_right][None]
    with torch.no_grad():
        x0 = fit_sys.encoder(torch.tensor(up, dtype=DTYPE_PT),
                             torch.tensor(yp, dtype=DTYPE_PT)).numpy()[0]
    return x0[:cfg.nx_phys]
