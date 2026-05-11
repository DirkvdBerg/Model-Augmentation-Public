"""
parameter_evaluation.py
-----------------------
Diagnostic: is cg1 recoverable from base vs identification trajectories?

All other parameters are fixed at true values. cg1 is the only trainable
parameter, log-parameterized identically to the main training script, tested
at +10% and +2% detuning.

Phases
------
0  qdot PSD         — PSD of q1 and qdot per trajectory, per dataset
1  gradient check   — distribution of ∂L/∂cg1 over all FULL_COVERAGE positions
                      per dataset × segment_len × detuning (no optimizer step)
2  training         — 300 epochs FULL_COVERAGE, single cg1 parameter
                      cg1 convergence per dataset × segment_len × detuning

Run as:
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.parameter_evaluation
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import welch

from lpv_lfr_baseline.blocks.lfr_param_block import _TRUE_PARAMS, _build_matrices
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.lfr_simulate import simulate
from lpv_lfr_baseline.core.physics import P as _P, build_poly_constants, ts as _ts
from lpv_lfr_baseline.scripts.precompute import _build_segment_pools, precompute

# ── Config ───────────────────────────────────────────────────────────────────
DTYPE        = torch.float64
_ROOT        = os.path.join(os.path.dirname(__file__), '..', '..')
SAVE_DIR     = os.path.join(_ROOT, 'simulations', 'parameter_evaluation')

SEGMENT_LENS = [600, 1000, 2000, 3000, 4000]
DETUNES      = [0.10, 0.02]
EPOCHS       = 300
LR           = 1e-3
FS_NEW       = 20000   # no decimation (matches main training script)

_TRAJ_BASE = (
    {'id': 'T1', 'file': 'T1_Y_sweep_conservative.mat'},
    {'id': 'T2', 'file': 'T2_X_sym_Y030.mat'},
    {'id': 'T3', 'file': 'T3_X_sym_Y000.mat'},
    {'id': 'T4', 'file': 'T4_X_antisym_Y020.mat'},
    {'id': 'T5', 'file': 'T5_X_sym_Y_sweep.mat'},
    {'id': 'T6', 'file': 'T6_Y_sweep_aggressive.mat'},
)
_TRAJ_EXTENDED = _TRAJ_BASE + (
    {'id': 'T7', 'file': 'T7_X_antisym_Y_sweep.mat'},
    {'id': 'T8', 'file': 'T8_X_sym_anti_Y_sweep.mat'},
)
DATASETS = {
    'base': dict(
        traj_dir   = os.path.join(_ROOT, 'Matlab-output', 'parameter-recovery'),
        traj_specs = _TRAJ_BASE,
    ),
    'identification': dict(
        traj_dir   = os.path.join(_ROOT, 'Matlab-output', 'identification-trajectories'),
        traj_specs = _TRAJ_EXTENDED,
    ),
}

# ── Fixed true-param tensors (all except cg1) ────────────────────────────────
_T      = _TRUE_PARAMS
_kb_sum = torch.tensor(_T['kb1'] + _T['kb2'], dtype=DTYPE)
_cg2    = torch.tensor(_T['cg2'],              dtype=DTYPE)
_cy     = torch.tensor(_T['cy'],               dtype=DTYPE)
_cb_sum = torch.tensor(_T['cb1'] + _T['cb2'], dtype=DTYPE)
_mh     = torch.tensor(_T['mh'],               dtype=DTYPE)
_m1     = torch.tensor(_T['m1'],               dtype=DTYPE)
_m2     = torch.tensor(_T['m2'],               dtype=DTYPE)
_mb     = torch.tensor(_T['mb'],               dtype=DTYPE)
_J_sum  = torch.tensor(_T['Jb'] + _T['Jh'],   dtype=DTYPE)
_Jb     = torch.tensor(_T['Jb'],               dtype=DTYPE)
_Jh     = torch.tensor(_T['Jh'],               dtype=DTYPE)
_Lb     = torch.tensor(0.725,                  dtype=DTYPE)
_d      = torch.tensor(_T['d'],                dtype=DTYPE)
_P_t    = _P.to(DTYPE)
_ts_t   = _ts.to(DTYPE)

CG1_TRUE = float(_T['cg1'])   # 14.5

# Poly constants do not depend on cg1 — build once
_alpha, _beta, _gamma, _N0, _N1, _N2 = build_poly_constants(
    _m1, _m2, _mb, _mh, _Jb, _Jh, _Lb, _d
)
_d0 = _mh * (_alpha * _gamma - _beta ** 2)


# ── Core helpers ─────────────────────────────────────────────────────────────

def _make_G_KC(cg1: torch.Tensor):
    """Build G, K, C from true params + trainable cg1."""
    params_10 = torch.stack([_kb_sum, cg1, _cg2, _cy, _cb_sum, _mh, _m1, _m2, _mb, _J_sum])
    _, M1, M2, K, C = _build_matrices(params_10, _Lb, _d)
    return build_G_matrix(_N0, _d0, M1, M2, K, C), K, C


def _get_batch(trajs, pools, seg_len, pos):
    """One segment batch at pool position pos across all trajectories."""
    x0, u_segs, q1_segs = [], [], []
    for t in trajs:
        s = pools[t['id']][pos]
        x0.append(t['state_traj'][s])
        u_segs.append(t['u'][0, s:s + seg_len])
        q1_segs.append(t['q1'][s:s + seg_len])
    return torch.stack(x0), torch.stack(u_segs), torch.stack(q1_segs)
    # (B,6)  (B,seg_len,3)  (B,seg_len,3)


def _compute_loss(cg1, x0, u, q1, sigma_batch, ts_tensor):
    """Sigma-normalised MSE — identical to main training script."""
    G, K, C = _make_G_KC(cg1)
    result  = simulate(
        x0, u, G, K, C, _mh, _alpha, _beta, _gamma, _N0, _N1, _N2,
        _P_t, ts_tensor, bptt_mode='full', return_latents=False,
    )
    err = (result.Y - q1) / sigma_batch.unsqueeze(1)   # (B, seg_len, 3)
    return err.pow(2).mean()


# ── Phase 0: qdot PSD ────────────────────────────────────────────────────────

def phase0_qdot(ds_name, trajs, ts_eff):
    """One figure per dataset. Rows = trajectories, cols = 3 q1 PSDs + 3 qdot PSDs."""
    fs       = 1.0 / ts_eff
    n_trajs  = len(trajs)
    ch_lbl   = ['X', 'θ', 'Y']
    fig, axes = plt.subplots(n_trajs, 6, figsize=(22, 2.8 * n_trajs), sharey='col')
    if n_trajs == 1:
        axes = axes[None, :]

    for row, traj in enumerate(trajs):
        q1   = traj['q1'].numpy()             # (T, 3)
        qdot = traj['state_traj'][:, 3:].numpy()  # (T, 3)
        for ch in range(3):
            for offset, sig, lbl in [(0, q1, 'q1'), (3, qdot, 'qdot')]:
                f, P = welch(sig[:, ch], fs=fs, nperseg=4096)
                ax   = axes[row, ch + offset]
                ax.semilogy(f, P, lw=0.7)
                ax.set_title(f'{traj["id"]} {lbl} {ch_lbl[ch]}', fontsize=7)
                ax.set_xlabel('Hz', fontsize=7)
                ax.grid(True, alpha=0.3)
                if ch + offset == 0:
                    ax.set_ylabel('PSD', fontsize=7)

    fig.suptitle(f'Phase 0 — PSD of q1 (left) and qdot (right): {ds_name}')
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f'phase0_psd_{ds_name}.png')
    plt.savefig(path, dpi=120)
    print(f'  Saved: {path}')


# ── Phase 1: gradient distribution ───────────────────────────────────────────

def phase1_gradient(all_data):
    """
    Boxplot of ∂L/∂cg1 over all FULL_COVERAGE positions.
    Figure: rows = datasets, cols = segment lengths.
    Two boxes per subplot: 10% and 2% detuning.
    """
    n_ds, n_seg = len(all_data), len(SEGMENT_LENS)
    fig, axes   = plt.subplots(n_ds, n_seg, figsize=(4 * n_seg, 4 * n_ds), sharey='row')
    if n_ds == 1:
        axes = axes[None, :]

    print(f'\n{"=" * 70}\nPhase 1: gradient check  (CG1_TRUE={CG1_TRUE})\n{"=" * 70}')
    print(f'  {"Dataset":<16} {"seg_len":>8} {"detune":>8} {"mean|g|":>12} {"correct%":>10} {"std":>12}')
    print(f'  {"-" * 68}')

    for row, (ds_name, (trajs, sigma, ts_eff)) in enumerate(all_data.items()):
        ts_tensor   = torch.tensor(ts_eff,  dtype=DTYPE)
        sigma_batch = torch.stack([sigma[t['id']] for t in trajs])

        for col, seg_len in enumerate(SEGMENT_LENS):
            ax    = axes[row, col]
            pools = _build_segment_pools(trajs, seg_len)
            n_pos = min(len(pools[t['id']]) for t in trajs)

            box_data, box_labels = [], []
            for detune in DETUNES:
                cg1_init = CG1_TRUE * (1 + detune)
                grads    = []
                for pos in range(n_pos):
                    x0, u, q1 = _get_batch(trajs, pools, seg_len, pos)
                    cg1  = torch.tensor(cg1_init, dtype=DTYPE, requires_grad=True)
                    loss = _compute_loss(cg1, x0, u, q1, sigma_batch, ts_tensor)
                    loss.backward()
                    grads.append(cg1.grad.item())

                g        = torch.tensor(grads)
                mean_abs = g.abs().mean().item()
                correct  = (g > 0).float().mean().item()
                print(
                    f'  {ds_name:<16} {seg_len:>8} {detune:>+7.0%} '
                    f'{mean_abs:>12.4e} {correct:>9.0%} {g.std().item():>12.4e}'
                )
                box_data.append(grads)
                box_labels.append(f'{detune:+.0%}')

            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['steelblue', 'darkorange']):
                patch.set_facecolor(color)
            ax.axhline(0, color='k', lw=0.8, ls='--')
            ax.set_title(f'{ds_name}  seg={seg_len}', fontsize=8)
            ax.set_xlabel('Detuning')
            if col == 0:
                ax.set_ylabel('∂L/∂cg1')
            ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Phase 1 — gradient distribution (FULL_COVERAGE, all other params = true)')
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'phase1_gradient.png')
    plt.savefig(path, dpi=120)
    print(f'  Saved: {path}')


# ── Phase 2: training convergence ────────────────────────────────────────────

def phase2_train(all_data):
    """
    FULL_COVERAGE training, single cg1 nn.Parameter.
    Figure: rows = datasets, cols = detunings. Lines = segment lengths.
    """
    n_ds, n_det = len(all_data), len(DETUNES)
    fig, axes   = plt.subplots(n_ds, n_det, figsize=(6 * n_det, 5 * n_ds), sharey='row')
    if n_ds == 1:
        axes = axes[None, :]

    print(f'\n{"=" * 70}\nPhase 2: training  ({EPOCHS} epochs, lr={LR}, FULL_COVERAGE)\n{"=" * 70}')
    print(f'  {"Dataset":<16} {"seg_len":>8} {"detune":>8} {"init":>8} {"final":>8} {"true":>8} {"Δ%":>8}')
    print(f'  {"-" * 68}')

    for row, (ds_name, (trajs, sigma, ts_eff)) in enumerate(all_data.items()):
        ts_tensor   = torch.tensor(ts_eff, dtype=DTYPE)
        sigma_batch = torch.stack([sigma[t['id']] for t in trajs])

        for col, detune in enumerate(DETUNES):
            ax       = axes[row, col]
            cg1_init = CG1_TRUE * (1 + detune)
            ax.axhline(CG1_TRUE, color='k',    lw=1.5, ls='--', label='true', zorder=5)
            ax.axhline(cg1_init, color='gray', lw=1.0, ls=':',  label='init', zorder=4)

            for seg_len in SEGMENT_LENS:
                pools = _build_segment_pools(trajs, seg_len)
                n_pos = min(len(pools[t['id']]) for t in trajs)

                # Log-parameterized: cg1 = cg1_init * exp(log_cg1), init at 0
                log_cg1 = nn.Parameter(torch.zeros(1, dtype=DTYPE))
                opt     = torch.optim.Adam([log_cg1], lr=LR)
                history = []

                for _ in range(EPOCHS):
                    for pos in range(n_pos):
                        x0, u, q1 = _get_batch(trajs, pools, seg_len, pos)
                        cg1_phys  = cg1_init * torch.exp(log_cg1[0])
                        loss      = _compute_loss(cg1_phys, x0, u, q1, sigma_batch, ts_tensor)
                        loss.backward()
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    history.append((cg1_init * torch.exp(log_cg1[0])).item())

                final = history[-1]
                delta = (final - CG1_TRUE) / CG1_TRUE * 100
                print(
                    f'  {ds_name:<16} {seg_len:>8} {detune:>+7.0%} '
                    f'{cg1_init:>8.3f} {final:>8.4f} {CG1_TRUE:>8.3f} {delta:>+7.2f}%'
                )
                ax.plot(history, label=f'seg={seg_len}')

            ax.set_title(f'{ds_name}  detune={detune:+.0%}', fontsize=9)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('cg1 [N/(m/s)]')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle('Phase 2 — cg1 convergence (all other params = true)')
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'phase2_convergence.png')
    plt.savefig(path, dpi=120)
    print(f'  Saved: {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(SAVE_DIR, exist_ok=True)

    print('Loading data via precompute...')
    all_data = {}
    for ds_name, cfg in DATASETS.items():
        pre = precompute(
            cfg['traj_specs'], cfg['traj_dir'], SAVE_DIR,
            dtype=DTYPE, norm_mode='global', fs_new=FS_NEW,
            segment_len=max(SEGMENT_LENS),
        )
        all_data[ds_name] = (pre['trajs'], pre['sigma'], float(pre['ts_eff']))
        print(f'  {ds_name}: {len(pre["trajs"])} trajectories, ts_eff={pre["ts_eff"]:.2e} s')

    print('\nPhase 0: PSD...')
    for ds_name, (trajs, _, ts_eff) in all_data.items():
        phase0_qdot(ds_name, trajs, ts_eff)

    print('\nPhase 1: Gradient check...')
    phase1_gradient(all_data)

    print('\nPhase 2: Training...')
    phase2_train(all_data)

    plt.show()
    print(f'\nAll outputs saved to: {SAVE_DIR}')
