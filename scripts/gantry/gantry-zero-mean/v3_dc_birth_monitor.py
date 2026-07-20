"""v3_dc_birth_monitor.py -- per-update-step monitor of the ANN DC birth (Theme C / V3).

Question (Jan, README section 7 G-C; supervisor 2026-07-17): during training, does the
ANN's constant / non-zero-mean ("DC") output appear in the FIRST update steps with a
consistent sign (a systematic gradient the loss geometry pushes), or does it wander in
with seed-dependent sign (unconstrained diffusion)? G-A is closed (the physics carries no
DC the baseline lacks, v1f), so the DC is an estimator/training artifact; this locates it.

What it logs, per optimizer step (the two objects a proper diagnosis needs):
  A. WHERE THE DC IS  -- per-row mean/std of the ANN output on a FIXED probe set (Z_pts).
                         Starts exactly at 0 (zero-init). mean = the DC; std = the
                         legitimate dynamic part.
  B. THE FORCE ON THE DC -- dLoss/d(bias_r): the gradient of the windowed training loss
                         w.r.t. a CONSTANT per-row bias added to the ANN output at every
                         rollout step. Sign is the discriminator: consistently negative on
                         a K=0 row = the loss REWARDS a DC there (systematic); ~0 = the loss
                         is indifferent and any DC is diffusion. This is the online analogue
                         of a profile-loss slice along the mean-force direction
                         (# THEORY: directional first-order loss sensitivity, Goodfellow
                         et al. 2015 "Qualitatively Characterizing NN Optimization"; the
                         geometric test is the right one here because variance-based tests
                         degenerate under near-deterministic gradients).
  + training loss per step, and a post-run multi-horizon free-run gap (the earliest
    train/deploy-mismatch warning, SUBNET / Farina-Piroddi multi-step).

Run over several SEEDS (do NOT fix one; Jan): sign agreement across seeds = systematic.

Mechanism (contract-preserving, no edit to the frozen deepSI/training path):
  - build the model through the existing gantry_dynamic package;
  - a forward hook on the Static_ANN_Block adds a zero leaf `probe_bias`; its .grad after
    each backward IS dLoss/d(bias) (B);
  - patch torch.optim.Adam.step for the run's duration (deepSI creates fit_sys.optimizer =
    Adam inside fit() and calls self.optimizer.step(closure) once per update), read A and B
    there, restore afterwards;
  - the standard [nf-probe] per-epoch train/val nf-RMS stays installed (STICKY rule).

Convention: lives in scripts/gantry/gantry-zero-mean/; data -> ./data/v3_*.npz,
figures -> ./figures/v3_*.png. Run by the user (D-090 run-table row first). This script
DOES train (short); it does not touch kamtin-fp-model or any other run.
"""
import os
import sys
import json
from dataclasses import replace

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))     # .../scripts/gantry/gantry-zero-mean
GANTRY = os.path.dirname(HERE)                          # .../scripts/gantry
ROOT   = os.path.dirname(os.path.dirname(GANTRY))       # repo root
for p in (ROOT, GANTRY):
    if p not in sys.path:
        sys.path.insert(0, p)

from gantry_dynamic.config import RunConfig
from gantry_dynamic.data import load_datasets, compute_normalization
from gantry_dynamic.model import build_model, train_model
from gantry_dynamic.training import _install_nf_val_probe
from model_augmentation.fit_systems.blocks import Static_ANN_Block

# ─────────────────────────────────────────────────────────────────────────────
# Config knobs (one surface). Base config MIRRORS run 71167 (the drift checkpoint)
# so this is the SAME phenomenon, only shorter and instrumented per step.
# ─────────────────────────────────────────────────────────────────────────────
SEEDS       = tuple(int(s) for s in os.environ.get('V3_SEEDS', '0,1,2').split(','))  # HEURISTIC:
                   # Monte Carlo over Xavier draws; do NOT fix one (Jan). V3_SEEDS=0 for a fast signal.
EPOCHS      = 1             # Jan: the birth is in epoch 1 (per-step); raise to see it settle
LR          = 1e-7          # user 2026-07-17: keep 1e-7 (the K=0 routing rate, D-101/D-102)
PROFILE_HORIZONS_S = (0.1, 0.5, 2.0)  # HEURISTIC: free-run gap horizons [s] (post-run snapshot)
PRINT_EVERY = 50           # console: brief per-step line cadence (full data is saved)
PREFIX      = os.environ.get('V3_PREFIX', 'v3')   # output filename prefix (V3_PREFIX=v3b for a parallel run)
MODE        = os.environ.get('V3_MODE', 'augmentation')  # 'augmentation' = 130-180 Hz; 'joint' = 1-200 Hz
                   # broadband (identifiability test): V3_MODE=joint V3_PREFIX=v3joint
ENC         = os.environ.get('V3_ENC', 'linear_map')     # 'linear_map' (Hoekstra init, our default) |
                   # 'default' (deepSI learned encoder = Jan's ECC setup). Implementation-check:
                   # V3_ENC=default V3_PREFIX=v3enc  (auto_fit_norm is NOT toggled: the nonlinear
                   # Gantry_State_Block requires manual norm, model.py comment -- a forced deviation.)
NORM        = os.environ.get('V3_NORM', 'finite_diff')   # 'finite_diff' (data.py:205, our default) |
                   # 'true' = renormalize velocity std_x from the TRUE sim velocities (x_logical) instead
                   # of finite-diff of measured y. Tests the normalization mismatch: V3_NORM=true V3_PREFIX=v3norm
NF          = int(os.environ.get('V3_NF', '0'))          # 0 = default nf (400 = 0.1s at 4kHz); >0 pins
                   # nf directly (truncation-length sweep to test the truncated-BPTT-bias source):
                   # V3_NF=800 V3_PREFIX=v3nf800 . Longer nf = less truncation bias -> DC should shrink.

# Python augmentation state layout (model.py line 31; NOT the MATLAB [X,Th,Y,da,...] order):
STATE_NAMES = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a']
K0_DRIFT_ROWS = (0, 2, 3, 5)   # X, Y (positions) + dX, dY (velocities): the K=0 drift channels

figDir = os.path.join(HERE, 'figures')
datDir = os.path.join(HERE, 'data')
os.makedirs(figDir, exist_ok=True)
os.makedirs(datDir, exist_ok=True)


def base_config(seed):
    """RunConfig identical to 71167 except epochs/lr/seed (per-step drift-birth probe)."""
    return RunConfig(
        mode=MODE, encoder_init=ENC, ann_activation='tanh',
        joint_estimation=False, param_rmse_baseline=0.01,
        orth_beta=0.0, orth_observe=True,          # penalty OFF; Z_pts probe ATTACHED
        param_init_detune=None, snr=None, seed=seed,
        fs_orig=20000, fs_new=4000, stride=10, use_f64=False,
        save_flag=False, nf_probe_print=True,
        nx_ann=2, ann_route_ix=(0, 1, 2, 3, 4, 5, 6, 7),
        n_nodes_per_layer=16, n_hidden_layers=2, up_sample=1,
        batch_size=256, lr=LR, epochs=EPOCHS, nf_seconds=0.100,
        nf_override=(NF if NF > 0 else None),     # V3_NF>0 pins nf (truncation-length sweep)
    )


def get_ann_and_probe(fit_sys):
    """The Static_ANN_Block and the fixed probe points Z_pts (attached by orth_observe)."""
    ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    pen = getattr(fit_sys, 'orth_penalty', None)
    if pen is None:
        raise RuntimeError('orth_penalty is None; set orth_observe=True so Z_pts is attached.')
    return ann, pen.Z_pts


def run_seed(seed, data, norm):
    """One instrumented short training run; returns the per-step record dict."""
    cfg = base_config(seed)
    hp  = cfg.hp
    np.random.seed(seed)
    torch.manual_seed(seed)
    fit_sys = build_model(hp, cfg, data, norm)

    ann, Zpts = get_ann_and_probe(fit_sys)
    nw = ann.nw
    # ANN output column j is the correction on state route_ix[j] (cfg.ann_route_ix).
    route_ix = [int(i) for i in np.asarray(cfg.ann_route_ix).ravel()]
    if nw != len(route_ix):
        raise RuntimeError(f'ANN nw={nw} != len(ann_route_ix)={len(route_ix)}')
    labels = [STATE_NAMES[s] if s < len(STATE_NAMES) else f'row{s}' for s in route_ix]
    dtype = torch.float64 if cfg.use_f64 else torch.float32

    # B: the interconnect calls block.forward(z) DIRECTLY (interconnect.py:92), bypassing module
    # hooks, so patch ann.forward to add a zero per-row bias; its .grad after backward =
    # dLoss/d(constant correction). No-op deepSI checkpoint_save for the run so this local closure
    # never needs to pickle (end-of-fit load('_best') then raises FileNotFoundError, which deepSI
    # catches at interconnect.py:717).
    probe_bias = torch.zeros(nw, dtype=dtype, requires_grad=True)
    orig_forward = ann.forward
    def forward_with_bias(z):
        return orig_forward(z) + probe_bias.view(1, -1, 1)
    ann.forward = forward_with_bias
    orig_ckpt_save = fit_sys.checkpoint_save_system
    fit_sys.checkpoint_save_system = lambda *a, **k: None

    rec = {'mean': [], 'std': [], 'bias_grad': [], 'loss': []}

    orig_step = torch.optim.Adam.step

    def patched_step(opt_self, closure=None):
        loss = orig_step(opt_self, closure)          # runs closure (backward) + Adam update
        with torch.no_grad():
            wr = ann(Zpts)[..., 0]                    # (Npts, nw); hook adds 0
            rec['mean'].append(wr.mean(0).cpu().numpy().copy())     # A: the DC
            rec['std'].append(wr.std(0).cpu().numpy().copy())       # A: dynamic part
        if probe_bias.grad is not None:
            rec['bias_grad'].append(probe_bias.grad.detach().cpu().numpy().copy())  # B: the force
            probe_bias.grad = None                    # not an optimizer param: zero it ourselves
        else:
            rec['bias_grad'].append(np.full(nw, np.nan))
        lv = float(loss.item()) if hasattr(loss, 'item') else float(loss)
        rec['loss'].append(lv)
        k = len(rec['loss'])
        if k <= 10 or k % PRINT_EVERY == 0:
            m = rec['mean'][-1]
            g = rec['bias_grad'][-1]
            def _byname(a, nm):
                return a[labels.index(nm)] if nm in labels else float('nan')
            print(f'    [step {k:5d}] loss={lv:.4e} | DC dX={_byname(m,"dX"):+.3e} '
                  f'dY={_byname(m,"dY"):+.3e} | dLoss/dbias dX={_byname(g,"dX"):+.3e} '
                  f'dY={_byname(g,"dY"):+.3e}')
        return loss

    # keep the STICKY per-epoch [nf-probe] train/val nf-RMS print installed
    orig_cve = _install_nf_val_probe(fit_sys, hp, cfg, data.train_list[0], data.val_ckpt_data)
    torch.optim.Adam.step = patched_step
    print(f'\n=== seed {seed} | lr={LR:.0e} | epochs={EPOCHS} | nf={hp["nf"]} | routing={route_ix} ===')
    try:
        train_model(fit_sys, hp, cfg, data, epochs=EPOCHS, nf=hp['nf'], validation_measure='sim-RMS')
    finally:
        torch.optim.Adam.step = orig_step
        ann.forward = orig_forward
        fit_sys.checkpoint_save_system = orig_ckpt_save
        fit_sys.cal_validation_error = orig_cve

    out = {k: np.asarray(v) for k, v in rec.items()}
    out['labels'] = np.array(labels)
    out['route_ix'] = np.array(route_ix)
    out['nf'] = hp['nf']
    out['lr'] = LR
    out['seed'] = seed

    # post-run multi-horizon free-run gap (train/deploy mismatch snapshot; non-fatal)
    try:
        horizons = []
        for hs in PROFILE_HORIZONS_S:
            nf_h = int(round(hs / cfg.ts_new))
            with torch.no_grad():
                e = fit_sys.n_step_error(data.val_ckpt_data, nf=nf_h, stride=nf_h,
                                         mode='RMS', mean_channels=True)
            horizons.append((hs, nf_h, float(np.mean(e))))
        out['horizon_gap'] = np.array([[h[0], h[2]] for h in horizons])
        gstr = '  '.join(f'{h[0]:g}s={h[2]:.3e}' for h in horizons)
        print(f'  [horizon gap] free-run RMS [m]: {gstr}')
    except Exception as e:
        print(f'  [horizon gap] skipped (non-fatal): {e}')
        out['horizon_gap'] = np.empty((0, 2))

    np.savez(os.path.join(datDir, f'{PREFIX}_perstep_seed{seed}.npz'), **out)
    return out


def plot_seed(out):
    """Per-seed 2x2: DC birth (A), the force (B), loss+DC-norm, dynamic std (A)."""
    seed = int(out['seed']); labels = list(out['labels'])
    mean, std, bg, loss = out['mean'], out['std'], out['bias_grad'], out['loss']
    steps = np.arange(1, len(loss) + 1)
    k0 = [labels.index(STATE_NAMES[i]) for i in K0_DRIFT_ROWS if STATE_NAMES[i] in labels]

    fh, ax = plt.subplots(2, 2, figsize=(15, 9))
    for j, lab in enumerate(labels):
        lw = 1.4 if j in k0 else 0.6
        ax[0, 0].plot(steps, mean[:, j], lw=lw, label=lab)
        ax[0, 1].plot(steps, bg[:, j], lw=lw, label=lab)
        ax[1, 1].plot(steps, std[:, j], lw=lw, label=lab)
    ax[0, 0].axhline(0, color='k', lw=0.6)
    ax[0, 0].set_title('A: ANN output per-row MEAN vs step (the DC; K=0 rows bold)')
    ax[0, 0].set_ylabel('mean of ann(Z_pts)'); ax[0, 0].legend(fontsize=6, ncol=2); ax[0, 0].grid(True)
    ax[0, 1].axhline(0, color='k', lw=0.6)
    ax[0, 1].set_title('B: dLoss/d(bias) per row vs step (sign<0 on K=0 = loss rewards a DC)')
    ax[0, 1].set_ylabel('dLoss/d(constant correction)'); ax[0, 1].grid(True)

    dc_norm = np.linalg.norm(mean, axis=1)
    axl = ax[1, 0]; axr = axl.twinx()
    axl.plot(steps, loss, color='tab:blue', lw=0.8, label='train loss')
    axr.plot(steps, dc_norm, color='tab:red', lw=0.8, label='||DC|| (row means)')
    axl.set_title('mismatch: does loss fall while the DC grows?')
    axl.set_xlabel('update step'); axl.set_ylabel('train loss', color='tab:blue')
    axr.set_ylabel('||DC||', color='tab:red'); axl.grid(True)
    ax[1, 1].set_title('A: ANN output per-row STD vs step (the dynamic part)')
    ax[1, 1].set_xlabel('update step'); ax[1, 1].set_ylabel('std of ann(Z_pts)'); ax[1, 1].grid(True)
    fh.suptitle(f'v3 DC-birth monitor | seed {seed} | lr={out["lr"]:.0e} | nf={int(out["nf"])}')
    fh.tight_layout()
    fh.savefig(os.path.join(figDir, f'{PREFIX}_perstep_seed{seed}.png'), dpi=150)
    plt.close(fh)


def plot_multiseed(outs):
    """Systematic-vs-diffusion: DC of each K=0 drift row vs step, all seeds overlaid."""
    rows = [STATE_NAMES[i] for i in K0_DRIFT_ROWS]
    fh, ax = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4.2), squeeze=False)
    for c, rn in enumerate(rows):
        for out in outs:
            labels = list(out['labels'])
            if rn not in labels:
                continue
            j = labels.index(rn)
            steps = np.arange(1, out['mean'].shape[0] + 1)
            ax[0, c].plot(steps, out['mean'][:, j], lw=0.9, label=f'seed {int(out["seed"])}')
        ax[0, c].axhline(0, color='k', lw=0.6)
        ax[0, c].set_title(f'DC on {rn}: sign agreement=systematic, scatter=diffusion')
        ax[0, c].set_xlabel('update step'); ax[0, c].grid(True)
        if c == 0:
            ax[0, c].set_ylabel('mean of ann(Z_pts)'); ax[0, c].legend(fontsize=7)
    fh.suptitle('v3 DC-birth across seeds (do-not-fix-seed Monte Carlo)')
    fh.tight_layout()
    fh.savefig(os.path.join(figDir, f'{PREFIX}_multiseed_dc.png'), dpi=150)
    plt.close(fh)


def override_norm_true_velocities(norm, cfg):
    """V3_NORM=true: rebuild the state normalization from the TRUE simulation velocities (x_logical of
    the train records) instead of the finite-diff-of-measured-y velocities (data.py:205). Overrides
    x_mean/std_x/x_all/Cd_norm, which feed BOTH the model's state norm AND the encoder linear-map norm
    (build_model uses norm.x_all for normalize_linear_ss_matrices when baseline_states.npz is missing)."""
    import dataclasses
    from gantry_dynamic.data import load_mat_aug, TRAIN_FILES
    from model_augmentation.systems.gantry_ss import Cd as _Cd
    xls = [load_mat_aug(f, cfg)[2] for f in TRAIN_FILES]          # (N,6) TRUE states (true velocities)
    x_all  = np.concatenate(xls).astype(cfg.dtype_np)
    x_mean = x_all.mean(0).reshape(cfg.nx_phys, 1).astype(cfg.dtype_np)
    std_x  = (x_all.std(0).reshape(cfg.nx_phys, 1) + 1e-8).astype(cfg.dtype_np)
    Cd_np  = _Cd.numpy() if hasattr(_Cd, 'numpy') else np.asarray(_Cd)
    Cd_norm = (Cd_np * std_x.flatten()[None, :] / np.asarray(norm.ystd)[:, None]).astype(cfg.dtype_np)
    o, n = norm.std_x.flatten(), std_x.flatten()
    print(f'  [V3_NORM=true] velocity std_x finite-diff -> true:  '
          f'dX {o[3]:.3e}->{n[3]:.3e}  dTheta {o[4]:.3e}->{n[4]:.3e}  dY {o[5]:.3e}->{n[5]:.3e}')
    return dataclasses.replace(norm, x_mean=x_mean, std_x=std_x, x_all=x_all, Cd_norm=Cd_norm)


def main():
    print('v3_dc_birth_monitor | per-update-step ANN DC birth + gradient force')
    print(f'  seeds={SEEDS}  epochs={EPOCHS}  lr={LR:.0e}  mode={MODE}  encoder={ENC}  norm={NORM}  '
          f'prefix={PREFIX}  (G-A closed; locating the DC in G-C)')
    cfg0 = base_config(SEEDS[0])
    np.random.seed(cfg0.seed); torch.manual_seed(cfg0.seed)
    data = load_datasets(cfg0)
    norm = compute_normalization(cfg0, data)
    if NORM == 'true':
        norm = override_norm_true_velocities(norm, cfg0)

    outs = []
    for seed in SEEDS:
        out = run_seed(seed, data, norm)
        plot_seed(out)
        outs.append(out)
    plot_multiseed(outs)
    print(f'\ndone | per-seed data -> {datDir}\\{PREFIX}_perstep_seed*.npz | figures -> {figDir}')


if __name__ == '__main__':
    main()
